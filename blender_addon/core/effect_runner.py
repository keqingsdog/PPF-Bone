# Fork modification: changed from the upstream baseline.
# File: effect_runner.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The impure shell that executes ``Effect`` objects produced by the pure
# ``transition()`` function.
#
# This is the ONLY module that performs real I/O (network, file system,
# ``bpy`` calls).  Background I/O runs on daemon threads; when it completes,
# the thread dispatches a result ``Event`` back into the ``Engine`` queue.
#
# The separation means:
#   - ``transitions.py`` decides what to do (pure, testable)
#   - ``effect_runner.py`` does it (impure, but mechanical)

from __future__ import annotations

import contextlib
from dataclasses import replace
import os
import pickle
import posixpath
import threading
import time
from typing import TYPE_CHECKING

import numpy

from ..models.console import console
from .backends import ConnectionBackend, create_backend
from .connection import NATIVE_BACKENDS, remote_target_dir
from .derived import is_sim_running_from_response
from .state import Activity, Phase, Server
from .gpu_devices import AUTOMATIC, describe_launch, shell_prefix
from .effects import (
    DoClearAnimation,
    DoClearInterrupt,
    DoConnect,
    DoDisconnect,
    DoExec,
    DoFetchFrames,
    DoFetchMap,
    DoLaunchServer,
    DoLog,
    DoQuery,
    DoReceiveData,
    DoRedrawUI,
    DoResetAnimationBuffer,
    DoSaveAndQuit,
    DoSendData,
    DoSetInterrupt,
    DoStopServer,
    DoTerminate,
    DoUploadAtomic,
    DoValidateRemotePath,
    Effect,
)
from .events import (
    Connected,
    ConnectionFailed,
    ErrorOccurred,
    ExecComplete,
    FetchComplete,
    FetchFailed,
    FetchMapComplete,
    ProgressUpdated,
    ReceiveDataComplete,
    SendDataComplete,
    ServerLaunched,
    ServerLost,
    ServerPolled,
    StatusQueryResult,
    StatusQueryCancelled,
    ServerRefreshRequested,
    ServerStopped,
    UploadPipelineComplete,
)
from .protocol import DEFAULT_CHUNK_SIZE
from .server_kill import KillReport, kill_remote_server
from .session import new_session_id
from . import collision_receipt
from . import body_control_receipt

_SCENE_QUERY_COMMANDS = frozenset(("build", "start", "resume"))

if TYPE_CHECKING:
    from .engine import Engine


# Live polls may outpace a paused artist for minutes. Leave accepted solver
# frames on the backend and download them in order as the main thread drains.
_LIVE_PENDING_FRAME_LIMIT = 16


def _server_join(backend, *parts: str) -> str:
    # win_native talks to a Windows server (Windows-style paths).
    # Every other backend talks to a POSIX server, so the path must use
    # forward slashes regardless of the client OS, otherwise a Windows
    # client mixes in backslashes that break both shell quoting on the
    # remote and the server's open().
    if backend.backend_type == "win_native":
        return os.path.join(*parts)
    return posixpath.join(*parts)


def _producer_hash_drift(expected_data, expected_param, data_hash, param_hash):
    """Which producer fingerprints no longer match a frozen expectation.

    Returns a diagnostic naming the families that disagree, or "" when none
    does. A family whose expected hash is empty is not compared: that is the
    shape a partially bound expectation has, and comparing against "" would
    refuse a scene for a reason nobody can act on.
    """
    drifted = []
    if expected_data and data_hash != expected_data:
        drifted.append("geometry data hash")
    if expected_param and param_hash != expected_param:
        drifted.append("parameter hash")
    return " and ".join(drifted)


def _producer_hash_mismatch(data, param, data_hash, param_hash):
    """Diagnose a producer self-check that disagrees with its own reported hashes.

    Returns "" when the encoder's two fingerprints describe the two byte
    strings it just produced. Names each failing family, with both the
    expected and the computed digest: the caller turns this into a refusal, and
    a refusal that says only "Encode failed." sends the artist looking in the
    wrong place. The digests it prints are the scene's own encrypted-once
    fingerprints -- the same values the server echoes and the panel shows -- so
    naming them exposes nothing a status response has not already carried.
    """
    import hashlib

    failed = []
    if hashlib.sha256(data).hexdigest() != data_hash:
        failed.append(f"data hash expected {data_hash or '<empty>'} "
                      f"but produced bytes hash to "
                      f"{hashlib.sha256(data).hexdigest()}")
    if hashlib.sha256(param).hexdigest() != param_hash:
        failed.append(f"param hash expected {param_hash or '<empty>'} "
                      f"but produced bytes hash to "
                      f"{hashlib.sha256(param).hexdigest()}")
    return "; ".join(failed)


def _decode_vertex_map_cbor(blob: bytes) -> dict:
    """Decode a ``map.pickle`` CBOR envelope into ``dict[str, ndarray]``.

    Producer is ``frontend/_cbor_bridge_.dumps_envelope("VertexMap", ...)``.
    Numpy arrays cross the wire as nested Python lists; we cast back to int64.
    """
    from .module import get_cbor2

    cbor2 = get_cbor2()
    env = cbor2.loads(blob)
    if not isinstance(env, dict) or env.get("kind") != "VertexMap":
        raise ValueError(
            f"map.pickle envelope kind mismatch: got {env.get('kind') if isinstance(env, dict) else type(env)!r}"
        )
    payload = env.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("VertexMap payload must be a map")
    return {k: numpy.asarray(v, dtype=numpy.int64) for k, v in payload.items()}


def _decode_surface_map_cbor(blob: bytes) -> dict:
    """Decode a ``surface_map.pickle`` CBOR envelope.

    Producer is ``frontend/_cbor_bridge_.dumps_envelope("SurfaceMap", ...)``.
    The inner shape stays ``{"version": 2, "maps": {uuid: (tri, coefs, surf_tri)}}``;
    we just rehydrate the inner numpy arrays.
    """
    from .module import get_cbor2

    cbor2 = get_cbor2()
    env = cbor2.loads(blob)
    if not isinstance(env, dict) or env.get("kind") != "SurfaceMap":
        raise ValueError(
            f"surface_map.pickle envelope kind mismatch: "
            f"got {env.get('kind') if isinstance(env, dict) else type(env)!r}"
        )
    payload = env.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("SurfaceMap payload must be a map")
    maps_in = payload.get("maps", {}) or {}
    rehydrated: dict = {}
    for name, entry in maps_in.items():
        tri_indices, coefs, surf_tri = entry
        rehydrated[name] = (
            numpy.asarray(tri_indices, dtype=numpy.int64),
            numpy.asarray(coefs, dtype=numpy.float64),
            numpy.asarray(surf_tri, dtype=numpy.int64),
        )
    return {"version": payload.get("version"), "maps": rehydrated}


def _decode_display_pin_map_cbor(blob: bytes) -> dict:
    """Decode a ``display_pin_map.pickle`` CBOR envelope.

    Producer is ``frontend/_scene_.py:export_fixed``. The payload is
    ``{"version": 1, "n_total": int, "blocks": [{"uuid", "blender_index",
    "offset"}, ...]}``, with blocks in the order the solver writes them to
    ``display_pin_<N>.bin``. Returns ``{"n_total": int, "blocks": [(uuid,
    blender_index, offset), ...]}``. Anything malformed is refused: placing
    vertices from a misread map would move the wrong ones.
    """
    from .module import get_cbor2

    cbor2 = get_cbor2()
    env = cbor2.loads(blob)
    if not isinstance(env, dict) or env.get("kind") != "DisplayPinMap":
        raise ValueError(
            f"display_pin_map.pickle envelope kind mismatch: "
            f"got {env.get('kind') if isinstance(env, dict) else type(env)!r}"
        )
    payload = env.get("payload")
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError(
            "display_pin_map.pickle has an unsupported payload (expected "
            "version 1); the session needs to be re-baked."
        )
    n_total = int(payload["n_total"])
    blocks = []
    offset = 0
    for block in payload["blocks"]:
        index = numpy.asarray(block["blender_index"], dtype=numpy.int64)
        if int(block["offset"]) != offset:
            raise ValueError(
                f"display_pin_map.pickle block for {block['uuid']!r} starts "
                f"at {block['offset']}, expected {offset}"
            )
        blocks.append((str(block["uuid"]), index, offset))
        offset += int(index.size)
    if offset != n_total:
        raise ValueError(
            f"display_pin_map.pickle blocks cover {offset} positions, "
            f"but n_total is {n_total}"
        )
    return {"n_total": n_total, "blocks": blocks}


class ConnectionOriginChanged(RuntimeError):
    """A frozen local request no longer owns the connected endpoint."""


class _LifecycleRetired(BaseException):
    """Internal control flow, never a user-visible transport failure."""


class EffectRunner:
    """Execute ``Effect`` objects, dispatching result events to the engine.

    Instantiate one ``EffectRunner`` per ``Engine`` and pass it to
    ``engine.tick(runner)`` on every main-thread timer tick.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._resume_fence = None
        self._backend: ConnectionBackend | None = None
        self._interrupt = threading.Event()
        self._project_name: str | None = None
        self._chunk_size: int = DEFAULT_CHUNK_SIZE
        # What the most recent ``_do_stop_server`` found and killed. Written
        # on the worker thread, read on the main thread after the stop has
        # settled to ``server=UNKNOWN``, so the two never overlap.
        self.last_kill_report: KillReport | None = None

        # Raw last-seen server response for UI display.  Kept here —
        # NOT in AppState — because it's a cache, not state.  See
        # core/cache.py for the rationale.
        from .cache import ResponseCache
        self._response_cache = ResponseCache()

        # Only the main-thread effect boundary reads current authorship.
        # Background jobs receive immutable expectations and a generation
        # ticket; retired connections/uploads can never publish validation.
        self._collision_lock = threading.RLock()
        self._collision_epoch = 0
        self._collision_expected = None
        self._collision_seal = None
        self._collision_verified = False
        self._collision_required = False
        self._collision_retransfer = False
        self._collision_identity = None
        self._collision_build_started = False
        # COLD REBUILD LATCH. A cold session has authored controls but no
        # expectation, because nothing was uploaded HERE. Reconstructing that
        # expectation means encoding the whole local scene, and a rebuild that
        # cannot succeed (the local producer requests no controls, or its bytes
        # cannot be produced) would otherwise be attempted again on every
        # status poll, in a ``suspend_playback`` window on Blender's main
        # thread. One entry per (connection ticket, scene, project, author) is
        # enough: a repeat of the SAME identity cannot reach a different
        # answer, and every way of reaching a different one -- a reconnect, a
        # Transfer, an author edit, a project change, a new .blend -- carries a
        # new key. What is remembered is only the OUTCOME, so a rebuild that
        # succeeded leaves nothing here to consult. Reads and writes happen
        # under ``_collision_lock``.
        self._cold_key: tuple | None = None
        self._cold_refused: dict[tuple, str] = {}
        self._cold_refused_order: list[tuple] = []
        self._cold_connection: tuple | None = None
        # Preflight encode latch: the exact manual-fetch jobs whose local
        # producer bytes have already been verified once.
        self._preflight_verified: list[tuple] = []
        self._body_expected = None
        self._body_seal = None
        self._body_verified = False
        self._body_required = False
        self._body_retransfer = False
        self._body_intent_hash = None

        # Stacked task I/O: one worker thread, two slots.
        # Commands (data send, build, exec, etc.) queue in order.
        # Polls (status queries) stack — only the latest matters.
        self._cmd_queue: list[tuple] = []      # [(fn, args), ...]
        self._poll_slot: tuple | None = None   # (fn, args) or None
        self._io_lock = threading.Lock()       # Protects _cmd_queue + _poll_slot
        self._work_event = threading.Event()   # Wakes the worker
        self._lifecycle_lock = threading.Lock()
        self._desired_running = True
        self._retirement_complete = True
        self._stop_event = threading.Event()  # Never cleared or reused.
        self._worker_stop_event = self._stop_event
        self._worker = None
        self._restart_waiter = None
        # Controlled module reload is a permanent retirement, unlike ordinary
        # stop/enable. Retain every admitted thread until it has really exited.
        self._reload_retiring = False
        self._reload_teardown_error = None
        self._reload_threads = set()
        self._reload_cleanup_thread = None
        self._reload_cleanup_backend = None
        self._reload_cleanup_done = False
        self._reload_close_failures = []

        # Connection attempts: a counter, and the lock that makes taking a
        # ticket and adopting a backend one step. A connect runs on a thread
        # of its own rather than on the worker above (see ``_do_connect``), so
        # more than one attempt can be in flight and exactly one of them may
        # land. The holder of the current ticket is that one.
        self._connect_lock = threading.Lock()
        self._connection_event_ticket = None
        self._connection_guard_owner = None
        self._connect_epoch: int = 0

        # Animation state (thread-safe buffer)
        self._anim_lock = threading.Lock()
        self._anim_map: dict[str, numpy.ndarray] = {}
        self._anim_surface_map: dict = {}
        # ``display_pin_map.pickle`` decoded by _decode_display_pin_map_cbor,
        # or empty when the session writes no display pins.
        self._anim_display_pin_map: dict = {}
        self._anim_bone_bindings: dict | None = None
        self._anim_statistics_manifest: bytes | None = None
        self._anim_statistics_zero_fetched = False
        # The upload the three session artifacts above were downloaded
        # for. They are fetched only while unset, so without an identity
        # they outlive the dataset they describe; see
        # ``_drop_stale_session_artifacts``.
        self._anim_upload_id: str | None = None
        # (solver frame, vertices, statistics blobs, display pins). The last
        # is ``(active, positions)`` parsed from ``display_pin_<N>.bin``, or
        # None when the session writes no display pins.
        self._anim_frames: list[
            tuple[int, numpy.ndarray, list[bytes], tuple | None]
        ] = []
        self._anim_total: int = 0
        self._anim_applied: int = 0
        self._fetched: list[int] = []
        self._anim_map_generation: int = 0
        self._anim_map_context = None
        self._anim_playback_origin = None
        self._anim_legacy_context = None
        self._anim_legacy_approved = False
        self._anim_fetch_origin = None
        # Last live status target. The worker downloads at most one bounded
        # queue window per job; FramePump schedules later windows after a
        # paused artist resumes, even if the solver has already finished.
        self._live_backlog = None

        # Data result store
        self._data_lock = threading.Lock()
        self._received_data: bytes | None = None
        self._exec_output: dict | None = None
        with self._lifecycle_lock:
            self._start_worker_locked()

    # -- public API --

    def _fetch_job_current(self, ticket, generation):
        if not self._execution_current():
            return False
        with self._collision_lock:
            return ((ticket is None or self._collision_current(ticket))
                    and (generation < 0 or generation == self._anim_map_generation))

    def _dispatch_fetch_event(self, event, ticket, generation):
        """Tag and enqueue atomically; stale progress/errors/completions are silent."""
        with self._collision_lock:
            if not self._fetch_job_current(ticket, generation):
                return
            self._engine.dispatch(replace(event, fetch_generation=generation))

    def _reject_fetch_current(self, exc, ticket, generation):
        with self._collision_lock:
            if self._fetch_job_current(ticket, generation):
                self._collision_reject(exc, ticket)

    def _producer_data_param_hashes(self):
        """Encode the local scene once and return ``(data_hash, param_hash)``.

        MAIN THREAD ONLY, through the same producer entry and the same
        authored-playback suspension the Transfer path uses, so the
        fingerprints agree with what an upload of this scene would carry.
        """
        import bpy
        from .encoder import prepare_upload
        from .rigid_pose import suspend_playback
        with suspend_playback():
            _data, _param, data_hash, param_hash = prepare_upload(bpy.context)
        return data_hash, param_hash

    def _preflight_key(self, event):
        """Identity of ONE manual fetch job, or None when it has none."""
        tag = getattr(event, "fetch_generation", -1)
        if type(tag) is not int or tag < 0:
            return None
        with self._collision_lock:
            return (self._collision_epoch, id(self._backend), self._project_name,
                    tag, self._scene_pointer())

    def _preflight_producer_hashes(self, event):
        """Verify the local producer bytes at most ONCE per manual fetch job.

        The Fetch All / Fetch Latest buttons reach the frames through
        ``FetchRequested -> DoFetchMap -> FetchMapComplete``, so this preflight
        runs before any frame worker is queued. ``only_latest`` live-sim
        fetches are untagged (-1) and never come through here, which is what
        keeps a streaming solve from re-encoding the scene per frame.

        Returns the two fingerprints, or ``None`` when this job was already
        verified, when it carries no identity, or when the scene holds no
        expectation to compare them against.
        """
        key = self._preflight_key(event)
        if key is None:
            return None
        with self._collision_lock:
            if key in self._preflight_verified:
                return None
            # Claim the job BEFORE deciding whether it needs an encode, so a
            # repeat of the same job can never encode twice however the first
            # answer is reached.
            self._preflight_verified.append(key)
            del self._preflight_verified[:-8]
            if self._body_expected is None and self._collision_expected is None:
                # Nothing was ever acknowledged on this connection, so there is
                # no fingerprint to drift from: no encode here at all.
                return None
            expected_data = getattr(
                self._body_expected or self._collision_expected, "data_hash", "")
            expected_param = getattr(
                self._body_expected or self._collision_expected, "param_hash", "")
            if not expected_data and not expected_param:
                return None
        return self._producer_data_param_hashes(), (expected_data, expected_param)

    def prepare_fetch_event(self, event):
        """Main-thread author snapshot before acquiring the event transition guard."""
        if not isinstance(event, FetchMapComplete) or event.fetch_generation < 0:
            return
        if not self.fetch_event_is_current(event):
            return
        try:
            self._collision_main_snapshot()
            with self._collision_lock:
                drifted = self._body_retransfer or self._collision_retransfer
            if drifted:
                raise ValueError("Physical authoring changed; Transfer the scene again before fetching frames")
            # The frame-producing inputs are compared HERE, on the manual fetch
            # path and before any frame worker is queued, because the author
            # gate above covers the collision-filter and body-control contracts
            # only: an edit to the mesh or to the parameters leaves those
            # contracts untouched, and the frames about to be downloaded were
            # produced by the solver from the OLD bytes. Frames from a build the
            # scene no longer describes are refused by name, and the artist
            # Transfers again.
            preflight = self._preflight_producer_hashes(event)
            if preflight is not None:
                (data_hash, param_hash), (expected_data, expected_param) = preflight
                drift = _producer_hash_drift(
                    expected_data, expected_param, data_hash, param_hash)
                if drift:
                    raise ValueError(
                        f"the local {drift} changed since the acknowledged "
                        "Transfer; the fetched frames would not describe this "
                        "scene. Transfer the scene again before fetching frames"
                    )
            preflight_key = self._preflight_key(event)
            if preflight_key is not None and self._anim_legacy_context == preflight_key[:4]:
                import bpy
                from ..models.groups import iterate_active_object_groups
                expected_hash = self._engine.state.server_data_hash
                actual_hash, _ = self._producer_data_param_hashes()
                groups = list(iterate_active_object_groups(bpy.context.scene))
                origin = self._anim_fetch_origin
                current = self._engine.state
                if (origin != (self._scene_pointer(), current.remote_root,
                               current.server_upload_id, current.server_data_hash)
                        or not expected_hash or actual_hash != expected_hash or not groups
                        or any(group.object_type not in {"SHELL", "STATIC", "ROD", "PDRD", "SAND"} for group in groups)):
                    raise ValueError("The legacy result mapping cannot be verified; transfer and run again before fetching")
            # The map and its output-generation identity are validated by the
            # worker before this main-thread event. Prepare targets only now,
            # before frame downloads or any animation writes are scheduled.
            import bpy
            from .animation import prepare_animation_targets, validate_fetch_scene_membership
            validate_fetch_scene_membership(bpy.context.scene, event.map_data)
            prepare_animation_targets(bpy.context, clear_existing=False)
            if preflight_key is not None and self._anim_legacy_context == preflight_key[:4]:
                with self._collision_lock:
                    self._anim_legacy_approved = True
        except Exception as exc:
            self._collision_reject(exc, self._collision_ticket())

    def fetch_effect_requires_atomic_schedule(self, effect):
        if getattr(effect, 'expected_connection_ticket', None) is not None:
            return True
        return isinstance(effect, (DoFetchFrames, DoClearAnimation, DoSetInterrupt,
                                   DoClearInterrupt, DoResetAnimationBuffer))

    def fetch_event_is_current(self, event) -> bool:
        """True when a tagged internal fetch event may be transitioned.

        Untagged (-1) events are always allowed, preserving legacy callers and
        every non-fetch event. Internal lock/read work only: no bpy, no network.
        """
        expected = getattr(event, 'expected_connection_ticket', None)
        if expected is not None and not self.connection_ticket_is_current(expected):
            return False
        tag = getattr(event, "fetch_generation", -1)
        if not isinstance(tag, int) or tag < 0:
            return True
        with self._collision_lock:
            return tag == self._anim_map_generation

    @contextlib.contextmanager
    def fetch_event_guard(self, event):
        expected = getattr(event, 'expected_connection_ticket', None)
        if expected is not None:
            # Reuse the engine's existing pre-transition/scheduling gate. All
            # bound pipeline effects are local preflight/queue work; transport
            # calls run on the I/O worker after this lock has been released.
            if self._connection_guard_owner == threading.get_ident():
                raise ConnectionOriginChanged('Nested connection-bound submission is not supported.')
            with self._connect_lock:
                self._connection_event_ticket = expected
                self._connection_guard_owner = threading.get_ident()
                try:
                    yield
                finally:
                    self._connection_event_ticket = None
                    self._connection_guard_owner = None
            return
        tag = getattr(event, "fetch_generation", -1)
        if type(tag) is not int or tag < 0:
            yield
            return
        with self._collision_lock:
            yield

    def _retire_animation_state_locked(self):
        """Retire published frames atomically. Caller MUST hold _collision_lock.

        Takes _anim_lock SECOND. That is the only order used anywhere: collision
        then animation. Nothing takes _anim_lock and then _collision_lock, so no
        inversion is possible. Clearing the queue here, rather than in a later
        ErrorOccurred handler, is what stops a frame published before retirement
        from reaching the consumer. The external animation packet tuple itself is
        never rewritten.
        """
        self._anim_map_generation += 1
        self._resume_fence = None
        with self._anim_lock:
            self._anim_frames.clear()
            self._live_backlog = None

    def _collision_invalidate(self, discard=False):
        with self._collision_lock:
            self._collision_epoch += 1
            self._collision_verified = False
            self._body_verified = False
            self._retire_animation_state_locked()
            if discard:
                self._collision_expected = None
                self._collision_seal = None
                self._collision_build_started = False
                self._body_expected = None
                self._body_seal = None

    def _scene_pointer(self):
        import bpy
        scene = bpy.context.scene
        return 0 if scene is None else scene.as_pointer()

    def _cold_rebuild_key(self, author, identity):
        """Identity of one cold-rebuild question, or None when it has none.

        The CONNECTION is named by the backend object itself, never by
        ``_collision_epoch``. An epoch is bumped by every rejection as well as
        by every reconnect, and a key that moved on each rejection could never
        remember one: a refusal that goes through ``_collision_reject`` would
        mint a new key on the next poll and encode the scene again. What has to
        separate two connection ATTEMPTS is the transport that carries them,
        and a new attempt publishes a new backend object; the runner holds the
        reference for exactly as long as the connection is up.
        """
        with self._collision_lock:
            connection = (id(self._backend), self._project_name)
            if connection != self._cold_connection:
                self._cold_connection = connection
                self._cold_key = None
                self._cold_refused.clear()
                self._cold_refused_order.clear()
            key = (connection, identity, author)
        return key if identity[1] else None

    def _cold_refused_reason(self, key):
        with self._collision_lock:
            return self._cold_refused.get(key)

    def _record_cold_refusal(self, key, reason):
        """Remember that THIS identity cannot rebuild an expectation, and why."""
        if key is None:
            return
        with self._collision_lock:
            if key not in self._cold_refused:
                self._cold_refused_order.append(key)
                del self._cold_refused_order[:-16]
                for stale in list(self._cold_refused):
                    if stale not in self._cold_refused_order:
                        del self._cold_refused[stale]
            self._cold_refused[key] = reason

    def cold_rebuild_status(self):
        """Diagnosis of the last cold rebuild attempt: ``(key, reason)``.

        Read-only, no bpy and no network, so a panel may call it on any
        thread. ``reason`` is empty while the current identity's attempt either
        succeeded or was never refused. NOTHING here resets the latch, and no
        caller may: reaching a different answer takes a new connection, a new
        Transfer, an author edit, a project change or a new .blend, each of
        which carries a new key by itself.
        """
        with self._collision_lock:
            key = self._cold_key
            if key is None:
                return None, ""
            return key, self._cold_refused.get(key, "")

    def _collision_main_snapshot(self, *, rebuild_cold=True):
        """Called only by execute(), never by the I/O worker."""
        import bpy
        from ..models.bone_collision import collect_collision_filters
        identity = (self._scene_pointer(), self._project_name)
        author = collision_receipt.required_contract(collect_collision_filters(bpy.context))
        with self._collision_lock:
            if identity != self._collision_identity:
                self._collision_invalidate(discard=True)
                self._collision_identity = identity
                self._collision_retransfer = False
                self._body_retransfer = False
            if self._collision_expected is not None and author != self._collision_expected.author_hash:
                self._collision_retransfer = True
                self._collision_invalidate(discard=True)
            self._collision_required = author is not None or self._collision_retransfer
            cold = (author is not None and self._collision_expected is None
                    and not self._collision_retransfer)
        if cold and rebuild_cold:
            self._cold_adopt_expectation("collision", author, identity)
        self._body_main_snapshot(rebuild_cold=rebuild_cold)
        return author

    def _body_main_snapshot(self, *, rebuild_cold=True):
        """Immutable author intent captured only on Blender's main thread."""
        import bpy
        from ..models.bone_bindings import collect_body_controls_contract
        identity = (self._scene_pointer(), self._project_name)
        author = body_control_receipt.required_contract(collect_body_controls_contract(bpy.context))
        with self._collision_lock:
            if self._body_expected is not None and author != self._body_expected.intent_hash:
                self._body_retransfer = True
                self._collision_invalidate(discard=True)
            self._body_intent_hash = author
            self._body_required = author is not None or self._body_retransfer
            cold = (author is not None and self._body_expected is None
                    and not self._body_retransfer)
        if cold and rebuild_cold:
            self._cold_adopt_expectation("body", author, identity)
        return author

    def _cold_expectation_pending(self):
        """A status-only reconnect has not verified the current scene bytes."""
        with self._collision_lock:
            return (
                self._collision_required and self._collision_expected is None
                and not self._collision_retransfer
            ) or (
                self._body_required and self._body_expected is None
                and not self._body_retransfer
            )

    def _cold_adopt_expectation(self, family, author, identity):
        """ONE cold rebuild attempt per identity, with a remembered outcome.

        ``family`` is ``"body"`` or ``"collision"``; both read the same local
        producer bytes, so only the family whose expectation is missing is
        attempted. Gating here rather than inside each adoption is what keeps
        the whole-scene encode off the poll loop: the answer for a given
        identity cannot change until one of the key's components does, and the
        key names all of them. A refusal is RECORDED and reported, never
        re-encoded -- and never reset by a generic invalidation, because
        invalidating state does not invalidate the identities that would ask
        the identical question again.
        """
        key = self._cold_rebuild_key(author, identity)
        if key is None:
            # No project to bind against: nothing can be asked, so nothing is
            # remembered either.
            if family == "body":
                self._body_adopt_cold_expectation(author)
            else:
                self._collision_adopt_cold_expectation(author)
            return
        with self._collision_lock:
            self._cold_key = key
            if key in self._cold_refused:
                return
            if family == "body":
                already = self._body_expected is not None
            else:
                already = self._collision_expected is not None
        if already:
            return
        if family == "body":
            self._body_adopt_cold_expectation(author)
            bound = self._body_expected is not None
        else:
            self._collision_adopt_cold_expectation(author)
            bound = self._collision_expected is not None
        if not bound:
            self._record_cold_refusal(key, self._response_cache.get(
                "error", "") or
                "the local scene asks for nothing this session can acknowledge, "
                "so no cold expectation could be rebuilt")

    def _cold_local_upload(self):
        """Encode the CURRENT LOCAL scene exactly as a Transfer would.

        MAIN THREAD ONLY. Reuses the single producer entry ``prepare_upload``
        under the same authored-playback suspension the encoder path already
        establishes, so the bytes and hashes are byte-identical to what
        ``DoUploadAtomic`` carries. Local bytes only: nothing here reads the
        remote pickle, a receipt, or any server evidence.
        """
        import bpy
        from .encoder import prepare_upload
        from .rigid_pose import suspend_playback
        with suspend_playback():
            return prepare_upload(bpy.context)

    def _body_adopt_cold_expectation(self, author):
        """Rebuild the body expectation from local bytes on a COLD session.

        A fresh process (reconnect, or a new .blend) has authored controls but
        no ``_body_expected``, because nothing was uploaded here; the ordinary
        status poll then refuses before it can consume an otherwise valid
        completed session. The producer-side checks below are the same ones
        ``DoUploadAtomic`` applies, so an existing byte-identical upload can be
        acknowledged instead of refused.

        The result starts UNBOUND: ``root`` and ``upload_id`` stay empty until
        an alive READY/RESUMABLE echo proves identity in
        ``_bind_cold_expectations``. Nothing is verified here.

        Returns True when an expectation now exists, False when this attempt
        rebuilt nothing -- the caller latches that outcome so a refusal is not
        re-encoded on the next poll.
        """
        try:
            data, param, data_hash, param_hash = self._cold_local_upload()
            uploaded = body_control_receipt.uploaded_contract(param)
        except Exception as exc:
            self._collision_reject(exc, self._collision_ticket())
            return False
        if uploaded is None:
            # Local authorship no longer requests controls: nothing to bind.
            return False
        try:
            if uploaded[2] != author:
                body_control_receipt.fail(
                    "encoded body author intent differs from current mode/events/source")
            if not data_hash or not param_hash:
                body_control_receipt.fail("cold-session local producer bytes carry no hash")
            # The BODY family's producer check must refuse as a BODY refusal:
            # reporting it through the collision-filter family names a subsystem
            # that is not involved and drops the fingerprint detail, so the
            # artist is told "Encode failed." about a scene whose collision
            # filters were never in question.
            mismatch = _producer_hash_mismatch(data, param, data_hash, param_hash)
            if mismatch:
                body_control_receipt.fail(
                    "encoded upload hashes do not match producer bytes (" + mismatch + ")")
        except Exception as exc:
            self._collision_reject(exc, self._collision_ticket())
            return False
        with self._collision_lock:
            if self._body_expected is not None or self._body_retransfer:
                return self._body_expected is not None
            self._body_expected = body_control_receipt.Expectation(
                uploaded[0], uploaded[2], uploaded[1], data_hash, param_hash)
            return True

    def _collision_adopt_cold_expectation(self, current):
        """Collision-filter analogue of ``_body_adopt_cold_expectation``.

        Only reached when the CURRENT local scene requires enabled filters, so
        an ordinary scene without required controls keeps the original fetch
        path. A body proof never marks the collision proof valid; each family
        is reconstructed and validated on its own. Returns True when an
        expectation now exists; see ``_body_adopt_cold_expectation``.
        """
        try:
            data, param, data_hash, param_hash = self._cold_local_upload()
            uploaded = collision_receipt.uploaded_contract(param)
        except Exception as exc:
            self._collision_reject(exc, self._collision_ticket())
            return False
        if uploaded is None:
            return False
        try:
            if uploaded != current:
                collision_receipt.fail("encoded filter contract differs from current authorship")
            if not data_hash or not param_hash:
                collision_receipt.fail("cold-session local producer bytes carry no hash")
            mismatch = _producer_hash_mismatch(data, param, data_hash, param_hash)
            if mismatch:
                collision_receipt.fail(
                    "encoded upload hashes do not match producer bytes (" + mismatch + ")")
        except Exception as exc:
            self._collision_reject(exc, self._collision_ticket())
            return False
        with self._collision_lock:
            if self._collision_expected is not None or self._collision_retransfer:
                return self._collision_expected is not None
            self._collision_expected = collision_receipt.Expectation(
                uploaded, data_hash, param_hash)
            return True

    def _bind_cold_expectations(self, response, ticket, query_id=None):
        """Bind an UNBOUND cold expectation to an alive READY/RESUMABLE echo.

        I/O worker. Adopts ``root`` and ``upload_id`` from the response only
        when the echoed ``param_hash`` AND ``data_hash`` both equal the frozen
        local snapshot, the echoed root still names the CURRENT project, the
        response carries no error, and the ticket is current. That identifies a
        presently acknowledged byte-identical upload; it mints no token and
        performs no upload, build or Run.
        """
        if not self._collision_current(ticket):
            return False
        with self._collision_lock:
            body, collision = self._body_expected, self._collision_expected
        pending_body = body is not None and not body.upload_id
        pending_collision = collision is not None and not collision.upload_id
        if not (pending_body or pending_collision):
            return False
        if response.get("error"):
            return False
        if response.get("status") not in ("READY", "RESUMABLE"):
            return False
        root, upload_id = response.get("root"), response.get("upload_id")
        if not root or not isinstance(root, str) or not upload_id:
            return False
        # The server reports ``data_root/project``; a root whose last segment is
        # not the project this runner asked about is an answer for something
        # else, whatever its hashes say.
        project = ticket[2]
        if not project or root.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] != project:
            return False
        try:
            body_control_receipt.token(upload_id)
        except Exception:
            return False
        if pending_body:
            if (response.get("param_hash") != body.param_hash
                    or response.get("data_hash") != body.data_hash):
                return False
            body = body_control_receipt.Expectation(
                body.author_hash, body.intent_hash, body.controls,
                body.data_hash, body.param_hash, upload_id, root)
        if pending_collision:
            if (response.get("param_hash") != collision.param_hash
                    or response.get("data_hash") != collision.data_hash):
                return False
            collision = collision_receipt.Expectation(
                collision.author_hash, collision.data_hash, collision.param_hash,
                upload_id, root)
        with self._collision_lock, self._query_state_guard(query_id) as current:
            if not current:
                return False
            if not self._collision_current(ticket):
                return False
            if pending_body:
                if self._body_expected is not None and self._body_expected.upload_id:
                    return False
                self._body_expected = body
            if pending_collision:
                if self._collision_expected is not None and self._collision_expected.upload_id:
                    return False
                self._collision_expected = collision
        return True

    def _body_validate(self, response, ticket, force=False, query_id=None):
        backend, project = ticket[1:]
        self._bind_cold_expectations(response, ticket, query_id)
        with self._collision_lock:
            if not self._collision_current(ticket):
                return False
            expected, previous = self._body_expected, self._body_seal
            required, verified = self._body_required, self._body_verified
            retransfer = self._body_retransfer
        if retransfer:
            body_control_receipt.fail(
                "Physical body changed; Transfer the scene again before Run, Resume, or Fetch."
            )
        if expected is None:
            if required:
                body_control_receipt.fail("requested body controls have no acknowledged current Transfer")
            return True
        self._collision_response_identity(response, expected)
        if verified and not force:
            return True
        try:
            raw = {path: self._call_owned(backend.receive_data, _server_join(backend, expected.root, "session", path),
                    project, chunk_size=self._chunk_size) for path in body_control_receipt.required_files(expected)}
            from .fem_body_control_receipt import extra_files
            for path in extra_files(raw,expected):
                raw[path]=self._call_owned(backend.receive_data, _server_join(backend,expected.root,'session',path),project,chunk_size=self._chunk_size)
        except Exception as exc:
            body_control_receipt.fail("cannot download required body receipt/seal (" + str(exc) + ")")
        seal = body_control_receipt.verify_downloads(raw, expected, previous)
        after, alive = self._call_owned(backend.query, {}, project, self._chunk_size)
        if not alive:
            body_control_receipt.fail("server disconnected during body-control validation")
        self._collision_response_identity(after, expected)
        with self._collision_lock, self._query_state_guard(query_id) as current:
            if not current:
                return False
            if not self._collision_current(ticket) or self._body_expected != expected:
                return False
            self._body_seal, self._body_verified = seal, True
        return True

    def _collision_ticket(self):
        with self._collision_lock:
            return (self._collision_epoch, self._backend, self._project_name)

    def _collision_current(self, ticket):
        if not self._execution_current():
            return False
        return ticket == self._collision_ticket()

    def _collision_reject(self, exc, ticket, response=None):
        with self._collision_lock:
            if not self._collision_current(ticket):
                return
            self._collision_invalidate(discard=True)
            # Preserve the real reported phase and add the client refusal. The
            # original terminal-with-error transition releases the async operator.
            rejected = dict(response or self._response_cache.last_response)
            rejected["error"] = str(exc)
            if rejected.get("status"):
                self._response_cache.record(rejected)
                self._engine.dispatch(ServerPolled(response=rejected))
            self._engine.dispatch(ErrorOccurred(error=str(exc), source="collision_filter_receipt"))

    def _collision_response_identity(self, response, expected):
        if (response.get("root") != expected.root
                or response.get("upload_id") != expected.upload_id
                or response.get("param_hash") != expected.param_hash
                or (expected.data_hash and response.get("data_hash") != expected.data_hash)):
            collision_receipt.fail("server upload/session identity changed")

    def _collision_validate(self, response, ticket, force=False, query_id=None):
        backend, project = ticket[1:]
        self._bind_cold_expectations(response, ticket, query_id)
        with self._collision_lock:
            if not self._collision_current(ticket):
                return False
            expected = self._collision_expected
            previous = self._collision_seal
            required = self._collision_required
            verified = self._collision_verified
        if expected is None:
            if required:
                collision_receipt.fail("enabled filters have no acknowledged current Transfer")
            return self._body_validate(response, ticket, force, query_id)
        self._collision_response_identity(response, expected)
        if verified and not force:
            return self._body_validate(response, ticket, force, query_id)
        root = expected.root
        try:
            receipt_raw = self._call_owned(backend.receive_data, 
                _server_join(backend, root, "session", collision_receipt.RECEIPT_FILE),
                project, chunk_size=self._chunk_size)
            manifest_raw = self._call_owned(backend.receive_data, 
                _server_join(backend, root, "session", collision_receipt.MANIFEST_FILE),
                project, chunk_size=self._chunk_size)
        except Exception as exc:
            collision_receipt.fail("cannot download required collision receipt/manifest (" + str(exc) + ")")
        seal = collision_receipt.verify(receipt_raw, manifest_raw, expected, previous)
        after, alive = self._call_owned(backend.query, {}, project, self._chunk_size)
        if not alive:
            collision_receipt.fail("server disconnected during receipt validation")
        self._collision_response_identity(after, expected)
        with self._collision_lock, self._query_state_guard(query_id) as current:
            if not current:
                return False
            if not self._collision_current(ticket) or self._collision_expected != expected:
                return False
            self._collision_seal = seal
            self._collision_verified = True
        return self._body_validate(response, ticket, force, query_id)

    def stop(self) -> None:
        """Retire this lifecycle without joining an in-flight I/O operation.

        Lifecycle changes, like Engine.tick and Blender registration, belong
        to the main thread. Publication commits alone use the short lock;
        collision/connect retirement never runs while that lock is held.
        """
        self._require_lifecycle_main_thread()
        with self._lifecycle_lock:
            self._desired_running = False
            self._retirement_complete = False
            self._stop_event.set()
            self._worker_stop_event.set()
        self._work_event.set()
        try:
            state = self._engine.state
            if self._status_query_matches(state, state.status_query_id):
                self._engine.dispatch(StatusQueryCancelled(
                    query_id=state.status_query_id,
                    reason="Query wait cancelled because its worker stopped; result remains unconfirmed; no stop or retry was sent."))
            self._collision_invalidate(discard=True)
            self._take_connect_ticket()
            with self._io_lock:
                self._cmd_queue.clear()
                self._poll_slot = None
        finally:
            with self._lifecycle_lock:
                self._retirement_complete = True

    def restart(self) -> None:
        """Request one successor, after the retired worker has actually exited.

        No main-thread join and no second I/O worker. A blocked backend can
        delay handoff, but cannot make an early enable silently lose it.
        """
        self._require_lifecycle_main_thread()
        with self._lifecycle_lock:
            self.assert_reload_available()
            self._desired_running = True
            if self._stop_event.is_set():
                self._stop_event = threading.Event()
            # A waiter owns the handoff even after its captured worker exits.
            # Do not overtake it: a second stop/restart could otherwise lose
            # its successor while the first waiter still owns this slot.
            if self._restart_waiter is not None:
                return
            if self._worker is not None and self._worker.is_alive():
                if self._worker_stop_event is self._stop_event:
                    return
                if self._restart_waiter is None:
                    waiter, ready, admitted = self._prepare_thread_start(
                        self._finish_worker_handoff, (self._worker,),
                        "ppf-worker-handoff")
                    self._restart_waiter = waiter
                    try:
                        waiter.start()
                    except BaseException:
                        # Even a bootstrap that arrives late must not perform
                        # a handoff after this attempt has relinquished its slot.
                        self._restart_waiter = None
                        self._reload_threads.discard(waiter)
                        ready.set()  # admitted remains false: target exits.
                        raise
                    admitted.set()
                    ready.set()
                return
            if self._retirement_complete:
                self._start_worker_locked()

    @staticmethod
    def _require_lifecycle_main_thread():
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("EffectRunner stop/restart must run on the main thread")

    def _prepare_thread_start(self, target, args, name):
        """Keep target inert until Thread.start returns successfully.

        A start exception is not proof that no native thread was created.
        The private gate lets such a late bootstrap exit without executing
        runner work, while the owner safely rolls its published slot back.
        """
        ready = threading.Event()
        admitted = threading.Event()
        def start_guard():
            ready.wait()
            if admitted.is_set():
                target(*args)
        thread = threading.Thread(target=start_guard, name=name, daemon=True)
        self._live_reload_threads_locked()
        self._reload_threads.add(thread)  # caller holds _lifecycle_lock
        return thread, ready, admitted

    def _start_worker_locked(self):
        previous, previous_origin = self._worker, self._worker_stop_event
        worker, ready, admitted = self._prepare_thread_start(
            self._worker_loop, (self._stop_event,), "ppf-io-worker")
        self._worker_stop_event = self._stop_event
        self._worker = worker
        try:
            worker.start()
        except BaseException:
            # Restore the real prior Thread, never expose an unstarted one to
            # cleanup.join. Keep desired intent, current token and queued work
            # so an explicit main-thread restart can retry after recovery.
            self._worker, self._worker_stop_event = previous, previous_origin
            self._reload_threads.discard(worker)
            ready.set()  # A possibly-created thread has no permission to run.
            raise
        admitted.set()
        ready.set()

    def _finish_worker_handoff(self, previous):
        previous.join()  # Only this daemon waits; no lock is held.
        failure = None
        with self._lifecycle_lock:
            if self._restart_waiter is threading.current_thread():
                self._restart_waiter = None
            if (self._worker is previous and self._desired_running
                    and self._retirement_complete and not self._stop_event.is_set()):
                try:
                    self._start_worker_locked()
                except Exception as exc:
                    failure = exc
        if failure is not None:
            # No busy retry and no Engine business error from a retired waiter.
            # The old worker reference and accepted queue are retained for retry.
            console.write(f"I/O worker restart could not start its worker: {failure}. "
                          "Retry enabling the add-on after thread resources recover.")

    def _execution_current(self):
        if self._reload_retiring:
            return False
        origin = self._engine.current_event_origin()
        if origin is None:
            return threading.current_thread() is threading.main_thread()
        return origin is self._stop_event and not origin.is_set()

    @contextlib.contextmanager
    def _lifecycle_commit(self):
        # Short in-memory commits only. Never call I/O, Blender or arbitrary
        # callbacks here. stop releases this lock before collision/connect.
        with self._lifecycle_lock:
            yield self._execution_current()

    def _owned_callback(self, callback):
        origin = self._engine.current_event_origin()
        def owned(*args, **kwargs):
            if origin is not None and origin.is_set():
                return
            with self._engine.event_origin(origin):
                if not self._execution_current():
                    return
                return callback(*args, **kwargs)
        return owned

    def _run_connect_owned(self, origin, *args):
        with self._engine.event_origin(origin):
            try:
                if self._execution_current():
                    self._do_connect(*args)
            except _LifecycleRetired:
                pass

    def _call_owned(self, fn, *args, **kwargs):
        # Claim one operation atomically against retirement, then release the
        # lock before I/O. An operation already claimed cannot be revoked.
        with self._lifecycle_commit() as current:
            if not current:
                raise _LifecycleRetired()
        result = fn(*args, **kwargs)
        if not self._execution_current():
            raise _LifecycleRetired()
        return result

    def _owned_interrupt(self, callback):
        origin = self._engine.current_event_origin()
        def interrupt():
            if origin is not None and origin.is_set():
                return True
            with self._engine.event_origin(origin):
                return not self._execution_current() or callback()
        return interrupt

    @property
    def backend(self) -> ConnectionBackend | None:
        # Legacy Jupyter/UI consumers use this property outside the I/O queue.
        # They must not borrow a transport while retirement disposes it.
        return None if self._reload_retiring else self._backend

    @property
    def reload_retiring(self) -> bool:
        return self._reload_retiring

    @property
    def reload_retirement_message(self):
        if self._reload_teardown_error is not None or self._reload_close_failures:
            return ("PPF is retired after uncertain addon teardown or cleanup. "
                    "Restart Blender; Reload cannot retry this safely. "
                    "Remote results remain unconfirmed.")
        return ("PPF is retired for reload. Retry Reload after existing I/O "
                "finishes, or restart Blender. Remote results remain unconfirmed.")

    def assert_reload_available(self):
        if self._reload_retiring:
            raise RuntimeError(self.reload_retirement_message)

    def begin_reload_retirement(self):
        """Main-thread, irreversible admission closure; never joins or closes I/O."""
        self._require_lifecycle_main_thread()
        with self._lifecycle_lock:
            first = not self._reload_retiring
            self._reload_retiring = True
        if first:
            self.stop()

    def record_reload_teardown_failure(self, error):
        """Latch uncertain unregister/register outcome until a fresh process.

        The receipt belongs to the owner, not a ReloadServer instance. Blender
        can mark an addon disabled before unregister raises; another retry must
        not treat a subsequent no-op disable as successful cleanup.
        """
        self._require_lifecycle_main_thread()
        with self._lifecycle_lock:
            if self._reload_teardown_error is None:
                self._reload_teardown_error = str(error)
        self.begin_reload_retirement()

    def _live_reload_threads_locked(self):
        # Failed-start bootstraps are inert and removed by their start owner.
        # All remaining records are admitted, including a start still pending.
        self._reload_threads = {
            t for t in self._reload_threads
            if t.ident is None or t.is_alive()
        }
        return sorted(t.name for t in self._reload_threads)

    def reload_retirement_status(self, allow_cleanup=True):
        """Poll a retired owner without waiting; disposal runs on a tracked daemon.

        A close error is uncertain and is NOT retried. The captured backend is
        retained for diagnostics; only a clean close permits module eviction.
        """
        self._require_lifecycle_main_thread()
        with self._lifecycle_lock:
            if self._reload_teardown_error is not None:
                return {"ready": False, "reason": "addon teardown failed; restart Blender",
                        "error": self._reload_teardown_error}
            if not self._reload_retiring or not self._retirement_complete:
                return {"ready": False, "reason": "retirement not complete"}
            live = self._live_reload_threads_locked()
            if live:
                return {"ready": False, "reason": "I/O still running", "threads": live}
        with self._connect_lock:
            if self._reload_close_failures:
                return {"ready": False, "reason": "backend close failed; restart Blender",
                        "errors": [message for _, message in self._reload_close_failures]}
            if self._reload_cleanup_done:
                return {"ready": self._backend is None,
                        "reason": "disposed" if self._backend is None else "unexpected backend"}
            if self._reload_cleanup_backend is not None:
                return {"ready": False, "reason": "backend disposal incomplete"}
            if self._backend is None:
                self._reload_cleanup_done = True
                return {"ready": True, "reason": "no backend"}
            if not allow_cleanup:
                return {"ready": False, "reason": "other participant still running"}
            backend = self._backend
            self._backend = None
            self._reload_cleanup_backend = backend
        def dispose():
            try:
                backend.disconnect()
            except BaseException as exc:
                with self._connect_lock:
                    self._reload_close_failures.append((backend, str(exc)))
            else:
                with self._connect_lock:
                    self._reload_cleanup_backend = None
                    self._reload_cleanup_done = True
        try:
            with self._lifecycle_lock:
                thread, ready, admitted = self._prepare_thread_start(
                    dispose, (), "ppf-reload-dispose")
                self._reload_cleanup_thread = thread
                try:
                    thread.start()
                except BaseException:
                    self._reload_threads.discard(thread)
                    ready.set()  # late bootstrap is inert; no close was admitted
                    raise
                admitted.set()
                ready.set()
        except BaseException:
            # Includes Thread construction failure. Never reverse the normal
            # connect -> lifecycle lock order while restoring ownership.
            self._reload_cleanup_thread = None
            with self._connect_lock:
                self._backend = backend
                self._reload_cleanup_backend = None
            raise
        return {"ready": False, "reason": "backend disposal started"}

    @property
    def project_name(self) -> str | None:
        return self._project_name

    @project_name.setter
    def project_name(self, value: str) -> None:
        if self._connection_guard_owner == threading.get_ident():
            if value != self._project_name:
                raise ConnectionOriginChanged('Connection project changed during submission.')
            return
        with self._connect_lock:
            if value != self._project_name:
                self._collision_invalidate(discard=True)
                state = self._engine.state
                if self._status_query_matches(state, state.status_query_id):
                    self._engine.dispatch(StatusQueryCancelled(
                        query_id=state.status_query_id,
                        reason="Query wait cancelled because the project changed; result remains unconfirmed; no stop or retry was sent."))
            self._project_name = value

    def capture_connection_ticket(self):
        """Freeze backend object, connect generation and selected project."""
        if self._connection_guard_owner == threading.get_ident():
            return (self._backend, self._connect_epoch, self._project_name)
        with self._connect_lock:
            return (self._backend, self._connect_epoch, self._project_name)

    def connection_ticket_is_current(self, expected):
        return expected == self.capture_connection_ticket()

    def execute(self, effect: Effect) -> None:
        """Execute one effect.  Called from ``Engine.tick()`` on the main thread.

        I/O effects go to the worker thread via cmd queue or poll slot.
        UI effects (log, redraw, interrupt, animation) run immediately.
        """

        self.assert_reload_available()
        expected = getattr(effect, 'expected_connection_ticket', None)
        if expected is not None and (self._connection_guard_owner != threading.get_ident()
                                     or self._connection_event_ticket != expected
                                     or not self.connection_ticket_is_current(expected)):
            raise ConnectionOriginChanged('Connection changed before submission; no remote request was submitted.')
        match effect:
            # -- Connection (commands) --
            case DoConnect(backend_type=bt, config=cfg, server_port=sp):
                self._collision_invalidate(discard=True)
                cfg_copy = dict(cfg)
                if sp:
                    cfg_copy["server_port"] = sp
                # Mint the fresh session id and read the last-saved id here,
                # on the main thread, so the bpy read happens where it is
                # safe (execute() runs inside Engine.tick() on the main
                # thread) and the Connected transition arm stays pure. The
                # values ride on the Connected event the attempt dispatches.
                sid = new_session_id()
                saved = self._last_saved_session_id()
                # ON ITS OWN THREAD, NOT ON THE I/O WORKER. See ``_do_connect``
                # for why: a handshake against a host that does not answer
                # would hold the one thread every later operation queues onto.
                epoch = self._take_connect_ticket()
                with self._lifecycle_lock:
                    self.assert_reload_available()
                    if not self._desired_running or not self._execution_current():
                        return
                    thread, ready, admitted = self._prepare_thread_start(
                        self._run_connect_owned,
                        (self._stop_event, bt, cfg_copy, sid, saved, epoch),
                        "ppf-connect-attempt")
                    try:
                        thread.start()
                    except BaseException:
                        self._reload_threads.discard(thread)
                        ready.set()
                        raise
                    admitted.set()
                    ready.set()

            case DoDisconnect():
                self._do_disconnect()

            case DoValidateRemotePath():
                if self._backend:
                    self._submit_cmd(self._do_validate_path)

            # -- Server lifecycle (commands) --
            case DoLaunchServer(
                cuda_device=device,
                cuda_device_uuid=device_uuid,
                device=compute_device,
                gpu_backend=gpu_backend,
            ):
                self._submit_cmd(
                    self._do_launch_server,
                    device,
                    device_uuid,
                    compute_device,
                    gpu_backend,
                )

            case DoStopServer():
                self._collision_invalidate(discard=True)
                self._submit_cmd(self._do_stop_server)

            # -- Queries --
            case DoQuery(request=req, expected_connection_ticket=expected_connection,
                         status_query_id=status_query_id):
                if status_query_id is not None:
                    from .query_contract import validate_query
                    validate_query(req)
                try:
                    command = req.get("request") if req else None
                    self._collision_main_snapshot(rebuild_cold=command in _SCENE_QUERY_COMMANDS)
                    if req and req.get("request") == "build":
                        with self._collision_lock:
                            if self._collision_required and (self._collision_expected is None or self._collision_build_started):
                                collision_receipt.fail("filtered rebuild requires a new atomic Transfer")
                            if self._body_required and (self._body_expected is None or self._collision_build_started):
                                body_control_receipt.fail("body-control rebuild requires a new atomic Transfer")
                            self._collision_build_started = True
                        self._collision_invalidate()
                    ticket = self._collision_ticket()
                    if req and req.get("request") == "resume":
                        self._begin_resume_generation(req, ticket)
                except Exception as exc:
                    self._collision_reject(exc, self._collision_ticket())
                    return
                if req:
                    # Command query (build, start, etc.) — never drop
                    if status_query_id is not None:
                        self._submit_cmd(self._do_query, req, ticket,
                                         self.capture_connection_ticket(), status_query_id)
                    elif expected_connection is None:
                        self._submit_cmd(self._do_query, req, ticket)
                    else:
                        self._submit_cmd(self._do_query, req, ticket, expected_connection)
                else:
                    # Status poll — latest wins
                    if status_query_id is None:
                        self._submit_poll(self._do_query, req, ticket)
                    else:
                        # A user probe is not expendable background polling.
                        # Queue it durably; _worker_loop still classifies the
                        # empty request as a poll and never auto-replays it.
                        self._submit_cmd(self._do_query, req, ticket,
                                         self.capture_connection_ticket(), status_query_id)

            # -- Data transfer (commands) --
            case DoSendData(remote_path=p, data=d):
                self._submit_cmd(self._do_send_data, p, d)

            case DoUploadAtomic(project_root=pr, data=d, param=pm,
                                 data_hash=dh, param_hash=ph, expected_connection_ticket=expected_connection):
                self._collision_invalidate(discard=True)
                try:
                    current = self._collision_main_snapshot(rebuild_cold=False)
                    uploaded = collision_receipt.uploaded_contract(pm)
                    body_upload = body_control_receipt.uploaded_contract(pm)
                    if (body_upload is not None) != (self._body_intent_hash is not None):
                        body_control_receipt.fail("encoded body request differs from current authorship")
                    if body_upload and body_upload[2] != self._body_intent_hash:
                        body_control_receipt.fail("encoded body author intent differs from current mode/events/source")
                    if uploaded != current:
                        collision_receipt.fail("encoded filter contract differs from current authorship")
                    if uploaded or body_upload:
                        import hashlib
                        if hashlib.sha256(pm).hexdigest() != ph or (d and hashlib.sha256(d).hexdigest() != dh):
                            collision_receipt.fail("encoded upload hashes do not match producer bytes")
                        if uploaded:
                            self._collision_expected = collision_receipt.Expectation(uploaded, dh, ph)
                        if body_upload:
                            self._body_expected = body_control_receipt.Expectation(
                                body_upload[0], body_upload[2], body_upload[1], dh, ph)
                    self._collision_retransfer = False
                    self._collision_required = uploaded is not None
                    self._body_retransfer = False
                    self._body_required = body_upload is not None
                except Exception as exc:
                    self._collision_reject(exc, self._collision_ticket())
                    return
                ticket = self._collision_ticket()
                if expected_connection is None:
                    self._submit_cmd(self._do_upload_atomic, pr, d, pm, dh, ph, ticket)
                else:
                    self._submit_cmd(self._do_upload_atomic, pr, d, pm, dh, ph, ticket, expected_connection)

            case DoReceiveData(remote_path=p):
                self._submit_cmd(self._do_receive_data, p)

            # -- Fetch (commands) --
            case DoFetchMap(root=r):
                try:
                    self._collision_main_snapshot()
                except Exception as exc:
                    self._collision_reject(exc, self._collision_ticket())
                    return
                # Capture the immutable identity ONCE, at submission, under the
                # collision lock, after Reset/snapshot have finished: ticket and
                # generation travel together and are never re-read live later.
                with self._collision_lock:
                    _map_ticket = self._collision_ticket()
                    _map_generation = self._anim_map_generation
                self._submit_cmd(self._do_fetch_map, r, _map_ticket, _map_generation)

            case DoFetchFrames(root=r, frame_count=fc, already_fetched=_af, only_latest=ol,
                               fetch_generation=me):
                # A map completion may be enqueued BEFORE retirement and consumed
                # AFTER it; dispatching under a lock cannot prevent that. Reject a
                # stale TAGGED effect here, at consumption and before scheduling, so
                # a retired completion can never derive a fresh frame job for the
                # replacement context. ``me < 0`` means untagged (existing callers),
                # which keeps ordinary wire behavior unchanged.
                if me >= 0 and me != self._anim_map_generation:
                    return
                # Re-read current local authorship before any frame is handed to
                # the animation. Automatic live preview waits for an explicit
                # operation to rebuild a cold scene's receipt expectation.
                if me < 0:
                    try:
                        self._collision_main_snapshot(rebuild_cold=not ol)
                    except Exception as exc:
                        self._collision_reject(exc, self._collision_ticket())
                        return
                    if ol and self._cold_expectation_pending():
                        return
                with self._collision_lock:
                    drifted = self._body_retransfer or self._collision_retransfer
                if drifted:
                    # A PROVED current-author edit. Refuse synchronously, BEFORE any
                    # frame worker is queued: the later auto-query that used to raise
                    # this arrives after _do_fetch_frames has already published. Cold
                    # reconstruction stays enabled only when no drift was proved.
                    try:
                        body_control_receipt.fail(
                            "the local author contract changed since the acknowledged Transfer")
                    except Exception as exc:
                        self._collision_reject(exc, self._collision_ticket())
                        return
                # The immutable ticket rides with the job so the worker can validate
                # real server identity before it publishes anything.
                # Always use self._fetched — set by facade.fetch() from
                # Blender-side applied frames. This matches the old
                # Communicator._fetch() which used self._fetched directly.
                # The immutable ticket AND its generation ride with the job, captured
                # together here at submission so the worker validates real server
                # identity before publishing and never re-reads a live generation.
                with self._collision_lock:
                    _frame_ticket = self._collision_ticket()
                    _frame_generation = self._anim_map_generation
                if ol:
                    with self._anim_lock:
                        record = self._live_backlog
                        if (record is None or record["root"] != r
                                or record["ticket"] != _frame_ticket
                                or record["generation"] != _frame_generation
                                or record["fetched"] is not self._fetched):
                            record = {"root": r, "target": fc,
                                      "ticket": _frame_ticket,
                                      "generation": _frame_generation,
                                      "fetched": self._fetched, "queued": False,
                                      "blocked": False, "status_serial": 1,
                                      "first": None, "retries": 0, "retry_at": 0.0,
                                      "validation_failures": 0, "validation_exhausted": False,
                                      "origin_scene_pointer": self._scene_pointer()}
                            self._live_backlog = record
                        else:
                            record["target"] = max(record["target"], fc)
                            # A fresh server status permits one new attempt
                            # after an absent or not-yet-published file.
                            record["status_serial"] += 1
                            record["blocked"] = False
                        if (record["validation_exhausted"] or record["queued"] or len(self._anim_frames)
                                >= _LIVE_PENDING_FRAME_LIMIT
                                or not self._live_missing_locked(record)):
                            return
                        record["queued"] = True
                    self._submit_cmd(self._do_live_fetch_window, record)
                else:
                    self._submit_cmd(self._do_fetch_frames, r, fc, self._fetched,
                                     False, _frame_ticket, _frame_generation)

            # -- Exec (command) --
            case DoExec(command=c, shell=s):
                self._submit_cmd(self._do_exec, c, s)

            # -- Immediate (main thread) --
            case DoSetInterrupt():
                self._interrupt.set()

            case DoClearInterrupt():
                self._interrupt.clear()

            case DoClearAnimation():
                self._collision_invalidate()
                with self._anim_lock:
                    self._anim_frames.clear()
                    self._anim_total = 0
                    self._anim_applied = 0
                    # Replace the identity in-flight jobs captured, so a stale
                    # worker cannot append into the newly cleared context.
                    self._fetched = []

            case DoResetAnimationBuffer():
                fence = self._resume_fence
                if fence is not None and fence["active"] is None:
                    self._engine.dispatch(FetchFailed(reason="RESUME_GENERATION_PENDING: wait for the resumed native generation"))
                    return
                self._collision_invalidate()
                self._anim_legacy_context = None
                self._anim_legacy_approved = False
                snapshot = self._engine.state
                self._anim_fetch_origin = (self._scene_pointer(), snapshot.remote_root,
                                           snapshot.server_upload_id, snapshot.server_data_hash)
                if fence is not None:
                    fence["ticket"] = self._collision_ticket()
                    fence["generation"] = self._anim_map_generation
                    self._resume_fence = fence
                with self._anim_lock:
                    self._anim_frames.clear()
                    self._anim_playback_origin = None
                    self._anim_total = 0
                    self._anim_applied = 0
                    self._anim_map = {}
                    self._anim_surface_map = {}
                    self._anim_display_pin_map = {}
                    self._anim_bone_bindings = None
                    self._anim_statistics_manifest = None
                    self._anim_statistics_zero_fetched = False
                    self._anim_upload_id = None
                    self._fetched = []

            # -- Terminate / Save-and-quit (commands) --
            case DoTerminate():
                self._submit_cmd(self._do_terminate)

            case DoSaveAndQuit():
                self._submit_cmd(self._do_save_and_quit)

            # -- UI (immediate) --
            case DoLog(message=m):
                console.write(m)

            case DoRedrawUI():
                self._redraw()

    # -- animation buffer access (called from main thread) --

    def has_pending_animation_frames(self) -> bool:
        """True while frames wait to be applied (main thread).

        The frame-pump modal reads this to decide whether it still has
        work: a solve can leave its last frames queued after the state
        machine has already returned to idle, and those frames need a
        modal-operator context to be applied.
        """
        with self._anim_lock:
            return bool(self._anim_frames)

    def animation_epoch(self):
        """Identity of the current fetch/upload, read without touching bpy."""
        with self._collision_lock:
            with self._anim_lock:
                return (self._anim_map_generation, self._anim_upload_id)

    def _live_missing_locked(self, record):
        first = record.get("first")
        if first is None:
            return True
        downloaded = set(record["fetched"])
        return any(i not in downloaded for i in range(first, record["target"] + 1))

    def live_fetch_paused(self, scene=None) -> bool:
        """Read-only UI status for exhausted live validation retries."""
        pointer = scene.as_pointer() if scene is not None else None
        with self._anim_lock:
            record = self._live_backlog
            return bool(record is not None and record["fetched"] is self._fetched
                        and record["validation_exhausted"]
                        and (pointer is None or record["origin_scene_pointer"] == pointer))

    def has_live_backlog_work(self) -> bool:
        """An in-flight window is work even when the consumer queue is empty."""
        with self._anim_lock:
            record = self._live_backlog
            if (record is None or record["fetched"] is not self._fetched
                    or record["validation_exhausted"]):
                return False
            return (record["queued"] or (self._live_missing_locked(record)
                    and (not record["blocked"] or record["retries"] < 3)))

    def continue_live_backlog(self) -> bool:
        """Queue one remote window once the main thread has freed capacity."""
        with self._anim_lock:
            record = self._live_backlog
            if (record is None or record["fetched"] is not self._fetched
                    or record["validation_exhausted"] or record["queued"]
                    or len(self._anim_frames) >= _LIVE_PENDING_FRAME_LIMIT
                    or not self._live_missing_locked(record)):
                return False
            if record["blocked"]:
                if record["retries"] >= 3 or time.monotonic() < record["retry_at"]:
                    return False
                record["retries"] += 1
                record["blocked"] = False
            record["queued"] = True
        self._submit_cmd(self._do_live_fetch_window, record)
        return True

    def retry_live_backlog(self) -> None:
        """Explicit Resume permits one retry of a stalled live window."""
        with self._anim_lock:
            if self._live_backlog is not None:
                self._live_backlog["status_serial"] += 1
                self._live_backlog["blocked"] = False
                self._live_backlog["validation_failures"] = 0
                self._live_backlog["validation_exhausted"] = False
                self._live_backlog["retries"] = 0
                self._live_backlog["retry_at"] = 0.0

    def _do_live_fetch_window(self, record):
        with self._anim_lock:
            before = len(record["fetched"])
            status_serial = record["status_serial"]
        try:
            self._do_fetch_frames(record["root"], record["target"],
                                  record["fetched"], True, record["ticket"],
                                  record["generation"])
        finally:
            with self._anim_lock:
                if self._live_backlog is record:
                    if len(record["fetched"]) == before:
                        # Suppress 10 Hz polling of a missing frame. A new
                        # status or explicit Resume changes status_serial.
                        record["blocked"] = (record["status_serial"]
                                             == status_serial)
                        record["retry_at"] = time.monotonic() + 0.5
                        if record["retries"] >= 3:
                            console.write("Live cache still has unavailable frames; manual Fetch can retry")
                    record["queued"] = False

    def take_one_animation_frame(
        self,
    ) -> tuple[tuple | None, dict, dict, dict, bytes | None, int, int]:
        """Pop one pending frame (main thread)."""
        with self._collision_lock:
            self._retire_resume_local_tail()
            with self._anim_lock:
                frame = self._anim_frames.pop(0) if self._anim_frames else None
                if frame is not None:
                    self._anim_applied += 1
                return (
                    frame,
                    self._anim_map,
                    self._anim_surface_map,
                    self._anim_display_pin_map,
                    self._anim_statistics_manifest,
                    self._anim_applied,
                    self._anim_total,
                )

    @property
    def received_data(self) -> bytes | None:
        with self._data_lock:
            return self._received_data

    @property
    def exec_output(self) -> dict | None:
        with self._data_lock:
            return self._exec_output

    def set_fetched_frames(self, frames: list[int]) -> None:
        """Set the list of already-fetched frame numbers."""
        # Hold ``_anim_lock`` so a concurrent ``_do_fetch_frames`` running
        # on the I/O worker can detect the reassignment via its
        # ``self._fetched is not fetched`` guard and discard a stale
        # live-fetch instead of leaking frames into the new context.
        with self._anim_lock:
            self._fetched = list(frames)

    def clear_fetched_frames(self) -> None:
        """Clear the fetched frame tracking."""
        with self._anim_lock:
            self._fetched = []

    # -- background thread helpers --

    def _submit_cmd(self, fn, *args) -> bool:
        """Queue a command for the currently requested worker lifecycle."""
        with self._lifecycle_lock:
            if not self._desired_running or not self._execution_current():
                return False
            with self._io_lock:
                self._cmd_queue.append((fn, args, self._stop_event))
        self._work_event.set()
        return True

    def _submit_poll(self, fn, *args) -> bool:
        """Keep only the latest poll in the current worker lifecycle."""
        with self._lifecycle_lock:
            if not self._desired_running or not self._execution_current():
                return False
            with self._io_lock:
                self._poll_slot = (fn, args, self._stop_event)
        self._work_event.set()
        return True

    def _pick_job(self, origin) -> tuple | None:
        with self._io_lock:
            while self._cmd_queue:
                fn, args, owner = self._cmd_queue[0]
                if owner is not origin:
                    # A retiring worker must not consume successor work.
                    return None
                self._cmd_queue.pop(0)
                return fn, args
            if self._poll_slot is not None:
                fn, args, owner = self._poll_slot
                if owner is origin:
                    self._poll_slot = None
                    return fn, args
            return None

    def _worker_loop(self, origin) -> None:
        """One serial worker, with immutable event and queue ownership."""
        with self._engine.event_origin(origin):
            while not origin.is_set():
                job = self._pick_job(origin)
                if job is None:
                    self._work_event.wait(timeout=0.25)
                    self._work_event.clear()
                    continue
                if origin.is_set():
                    break
                fn, args = job
                backend = self._backend
                is_poll = (fn == self._do_query and (not args or not args[0]))
                is_stop = fn == self._do_stop_server
                is_tagged_query = (fn == self._do_query and len(args) > 3 and args[3] is not None)
                try:
                    fn(*args)
                    if origin.is_set():
                        break
                    if not is_poll and not is_stop and not is_tagged_query and backend and self._project_name:
                        with self._io_lock:
                            has_pending_cmd = bool(self._cmd_queue)
                        if not has_pending_cmd:
                            self._engine.dispatch(ServerRefreshRequested())
                except _LifecycleRetired:
                    break
                except Exception as exc:
                    if origin.is_set():
                        break
                    if fn == self._do_upload_atomic and len(args) > 5 and isinstance(args[5], tuple):
                        if not self._collision_current(args[5]):
                            continue
                        if isinstance(exc, collision_receipt.CollisionReceiptError):
                            self._collision_reject(exc, args[5])
                            continue
                    # stop retires collision after its origin mark, so this
                    # guarded mutation cannot run after retirement completed.
                    with self._collision_lock:
                        if origin.is_set():
                            break
                        self._collision_invalidate()
                    alive = backend is None or backend.is_alive()
                    if origin.is_set():
                        break
                    if not alive:
                        from .events import ConnectionLost
                        console.write(f"Connection lost: {exc}")
                        self._engine.dispatch(ConnectionLost(cause=str(exc)))
                        with self._lifecycle_commit() as current:
                            if current:
                                with self._io_lock:
                                    self._cmd_queue.clear()
                                    self._poll_slot = None
                    else:
                        self._engine.dispatch(ErrorOccurred(error=str(exc), source=fn.__name__))

    # -- I/O implementations --

    def _last_saved_session_id(self) -> str:
        """Session id stored in the active scene at last save, or "".

        Reads ``bpy`` state, so it must run on the main thread (it is
        called from ``execute()`` inside ``Engine.tick()``). Mirrors
        ``facade.last_saved_session_id`` but logs a genuine read failure to
        the console instead of swallowing it silently, so the reconcile
        branch is not skipped without a trace.
        """
        try:
            import bpy  # pyright: ignore
            from ..models.groups import get_addon_data, has_addon_data
            scene = bpy.context.scene
            if scene is None or not has_addon_data(scene):
                return ""
            return get_addon_data(scene).state.last_session_id or ""
        except Exception as e:
            console.write(f"last-saved session id read failed: {e}")
            return ""

    # -- connection attempts --

    def _take_connect_ticket(self) -> int:
        """Invalidate every attempt in flight and return the new ticket.

        Every path that starts, cancels or supersedes a connection attempt
        comes through here, so the newest caller always holds the only current
        ticket and an older attempt can tell that it is no longer wanted.
        """
        with self._connect_lock:
            self._connect_epoch += 1
            return self._connect_epoch

    def _connect_attempt_is_current(self, epoch: int) -> bool:
        """True while *epoch* is still the attempt the add-on is waiting for."""
        with self._connect_lock:
            return epoch == self._connect_epoch

    def _adopt_backend(self, backend, epoch: int) -> bool:
        """Publish *backend* as the connection, unless the attempt was dropped.

        The test and the assignment are ONE step under the lock, because the
        two orderings against a disconnect have to differ. A disconnect that
        takes the ticket first must find nothing published, so this attempt
        closes what it opened; one that arrives after must find the reference,
        so it closes it. Either way the backend has exactly one owner, and no
        live transport is ever left in a runner the state machine believes is
        offline.
        """
        with self._connect_lock:
            if epoch != self._connect_epoch:
                return False
            self._backend = backend
            return True

    def _close_abandoned(self, backend) -> None:
        """Close a connection that landed after the user gave up on it."""
        try:
            backend.disconnect()
        except BaseException as e:
            with self._connect_lock:
                self._reload_close_failures.append((backend, str(e)))
            console.write(f"Canceled connection attempt: close failed: {e}")
            return
        console.write("Closed a connection attempt that was canceled.")

    def _do_connect(
        self,
        backend_type: str,
        config: dict,
        session_id: str,
        saved_session_id: str,
        epoch: int,
    ) -> None:
        """Connect on an explicitly origin-bound thread, retaining its ticket.

        The caller installs this attempt's Engine.event_origin token. Both the
        lifecycle token and connection epoch must survive every publication;
        transport calls, closing, logging and dispatch stay outside the locks.
        """
        # Match Engine's connection-event admission order: connect then
        # lifecycle. stop never holds lifecycle while waiting for connect.
        with self._connect_lock:
            with self._lifecycle_commit() as current:
                if not current:
                    return
                if epoch != self._connect_epoch:
                    return
                project_name = self._project_name or ""
        backend = None
        try:
            # Native handshakes select the same project as the first poll.
            config = dict(config, project_name=project_name)
            backend = create_backend(backend_type, config)
            remote_root = ""
            if hasattr(backend, "current_directory"):
                remote_root = backend.current_directory
        except Exception as e:
            if backend is not None:
                self._close_abandoned(backend)
            else:
                # A factory can allocate transport resources and then fail
                # before returning their owner. Do not certify that unknown
                # cleanup merely because this connect thread has exited.
                with self._connect_lock:
                    self._reload_close_failures.append((None,
                        "Connection factory failed before returning ownership; cleanup unconfirmed: " + str(e)))
            if (not self._execution_current()
                    or not self._connect_attempt_is_current(epoch)):
                console.write(f"Canceled connection attempt failed: {e}")
                return
            self._engine.dispatch(ConnectionFailed(error=str(e)))
            return

        # Adoption is one local commit against lifecycle retirement AND the
        # existing connect ticket, using the Engine's connect->lifecycle order.
        adopted = False
        with self._connect_lock:
            with self._lifecycle_commit() as current:
                if current and epoch == self._connect_epoch:
                    self._backend = backend
                    adopted = True
        if not adopted:
            self._close_abandoned(backend)
            return

        def still_current():
            if not self._execution_current():
                return False
            with self._connect_lock:
                return (epoch == self._connect_epoch
                        and self._backend is backend)

        def retire_attempt():
            # Keep the established ownership rule: disconnect owns a backend
            # still published here; a replaced reference belongs to this
            # attempt to close. Always close the captured object, never the
            # replacement that self._backend might now name.
            with self._connect_lock:
                replaced = self._backend is not backend
            if replaced:
                self._close_abandoned(backend)
            console.write("Connection canceled while the solver host was probed.")

        try:
            if not still_current():
                return
            self._probe_solver_host_gpus(backend, expected_connection_epoch=epoch)
            # A stop/cancel during the first round trip must not start a second.
            if not still_current():
                return
            self._probe_solver_host_builds(backend, expected_connection_epoch=epoch)
            if not still_current():
                return
            self._engine.dispatch(Connected(
                remote_root=remote_root,
                session_id=session_id,
                saved_session_id=saved_session_id,
            ))
        finally:
            # An owned-I/O guard may signal retirement with BaseException.
            # Preserve abandoned-object cleanup even on that control path;
            # the outer origin-bound thread wrapper handles the signal.
            if not still_current():
                retire_attempt()

    def probe_solver_host_gpus(self) -> None:
        """Re-enumerate the connected solver host's GPUs on the worker thread.

        The Refresh button calls this. It goes straight to the command queue
        rather than through an event, because it changes no application state:
        it refills a cache the panel reads.
        """
        backend = self._backend
        if backend is not None:
            self._submit_cmd(self._probe_solver_host_gpus, backend)

    def probe_solver_host_builds(self) -> None:
        """Re-list the connected solver host's solver builds on the worker thread.

        The same Refresh button, for the same reason as the GPU list: it refills
        a cache the panel reads and changes no application state.
        """
        backend = self._backend
        if backend is not None:
            self._submit_cmd(self._probe_solver_host_builds, backend)

    def _probe_solver_host_builds(
        self, backend, expected_connection_epoch=None,
    ) -> None:
        """Probe remotely, committing only for the current lifecycle and host.

        Explicit connect calls carry their adoption epoch. Worker Refresh jobs
        capture it on entry, while also proving that their backend still owns
        the connection. Cache helpers perform only local cache work; transport
        and result preparation are outside the lifecycle/connection locks.
        """
        from . import remote_builds

        # Match Engine's connection-event admission order: connect then
        # lifecycle. stop never holds lifecycle while waiting for connect.
        with self._connect_lock:
            with self._lifecycle_commit() as current:
                if not current:
                    return
                if backend is not self._backend:
                    return
                if expected_connection_epoch is None:
                    expected_connection_epoch = self._connect_epoch
                elif expected_connection_epoch != self._connect_epoch:
                    return
                remote_builds.forget_builds()
        # Native builds are resolved directly on this machine by the panel.
        if backend.backend_type in NATIVE_BACKENDS:
            return
        root = remote_builds.normalize_root(backend.current_directory)
        error = ""
        lines = ()
        if not root:
            error = (
                "The connection names no directory on the solver host, so its "
                "solver builds could not be listed."
            )
        else:
            command = remote_builds.probe_command(root)
            if (not self._execution_current()
                    or not self._connect_attempt_is_current(expected_connection_epoch)):
                return
            try:
                result = self._call_owned(backend.exec_command, 
                    command,
                    shell=True,
                    timeout=remote_builds.PROBE_TIMEOUT_SECONDS,
                )
            except Exception as e:  # noqa: BLE001 - recorded, not swallowed
                error = f"Could not list the solver builds on the solver host: {e}"
            else:
                if result.get("exit_code") != 0:
                    detail = " ".join(result.get("stderr") or []).strip() or "no output"
                    error = f"Listing the solver builds on the solver host failed: {detail}"
                else:
                    lines = tuple(result.get("stdout") or ())

        # Parse potentially large output before taking either publication lock.
        parsed_builds = remote_builds.parse_listing(lines) if not error else None
        # Revalidate after parsing as well as after the blocking boundary.
        # Match Engine's connection-event admission order: connect then
        # lifecycle. stop never holds lifecycle while waiting for connect.
        with self._connect_lock:
            with self._lifecycle_commit() as current:
                if not current:
                    return
                if (backend is not self._backend
                        or expected_connection_epoch != self._connect_epoch):
                    return
                if error:
                    remote_builds.record_probe_failure(error)
                else:
                    # Same cache fields as load_builds, with parsing staged
                    # outside the lifecycle lock. Do not call I/O here.
                    remote_builds._builds = parsed_builds
                    remote_builds._root = root
                    remote_builds._probe_error = ""
                    remote_builds._probed = True

    def _probe_solver_host_gpus(
        self, backend, expected_connection_epoch=None,
    ) -> None:
        """Enumerate locally/remotely, publishing only for this current host."""
        from . import gpu_devices

        # Match Engine's connection-event admission order: connect then
        # lifecycle. stop never holds lifecycle while waiting for connect.
        with self._connect_lock:
            with self._lifecycle_commit() as current:
                if not current:
                    return
                if backend is not self._backend:
                    return
                if expected_connection_epoch is None:
                    expected_connection_epoch = self._connect_epoch
                elif expected_connection_epoch != self._connect_epoch:
                    return
                gpu_devices.forget_devices()
        # Metal offers no selectable device and has no nvidia-smi. Keep the
        # empty, unprobed cache rather than inventing a failure for its panel.
        if backend.backend_type == "mac_native":
            return
        direct = backend.backend_type in ("win_native", "linux_native")
        command = (
            gpu_devices.NVIDIA_SMI_ARGS
            if direct
            else gpu_devices.NVIDIA_SMI_COMMAND
        )
        if (not self._execution_current()
                or not self._connect_attempt_is_current(expected_connection_epoch)):
            return
        error = ""
        text = ""
        try:
            result = self._call_owned(backend.exec_command, 
                command,
                shell=not direct,
                timeout=gpu_devices.PROBE_TIMEOUT_SECONDS,
            )
        except Exception as e:  # noqa: BLE001 - recorded, not swallowed
            error = f"Could not run nvidia-smi on the solver host: {e}"
        else:
            if result.get("exit_code") != 0:
                detail = " ".join(result.get("stderr") or []).strip() or "no output"
                error = f"nvidia-smi failed on the solver host: {detail}"
            else:
                text = "\n".join(result.get("stdout") or [])

        parsed_devices = []
        if not error:
            try:
                parsed_devices = gpu_devices.parse_nvidia_smi_devices(text)
            except gpu_devices.GpuProbeError as exc:
                error = str(exc)
        # Match Engine's connection-event admission order: connect then life.
        with self._connect_lock:
            with self._lifecycle_commit() as current:
                if not current:
                    return
                if (backend is not self._backend
                        or expected_connection_epoch != self._connect_epoch):
                    return
                # Same cache fields as load_devices; parsing is already done.
                gpu_devices._devices = parsed_devices
                gpu_devices._probe_error = error
                gpu_devices._probed = True

    def _do_disconnect(self) -> None:
        self._collision_invalidate(discard=True)
        from . import gpu_devices, remote_builds

        # ABANDON ANY ATTEMPT STILL HANDSHAKING, which is what Cancel is. The
        # attempt runs on its own thread and cannot be interrupted, so taking
        # the ticket is the whole of the cancellation: whatever that thread
        # opens it will close itself, and it reports nothing. This comes FIRST
        # so an attempt that adopts a backend a moment later finds the ticket
        # already gone; see ``_adopt_backend``.
        self._take_connect_ticket()
        # The next connection may reach a different machine, where a list left
        # over from this one would name GPUs, or build directories, that are
        # not there.
        gpu_devices.forget_devices()
        remote_builds.forget_builds()
        if self._backend:
            self._backend.disconnect()
            self._backend = None
        with self._anim_lock:
            self._anim_map = {}
            self._anim_surface_map = {}
            self._anim_display_pin_map = {}
            self._anim_bone_bindings = None
            self._anim_statistics_manifest = None
            self._anim_statistics_zero_fetched = False
            self._anim_upload_id = None
            self._anim_frames.clear()
            self._anim_total = 0
            self._anim_applied = 0
            # Inside the lock so a live-fetch in flight on the I/O worker
            # observes the reassigned reference via its stale-context
            # guard and bails instead of appending to the new list.
            self._fetched = []
        self._response_cache.clear()
        # The project_name is bound to the connection, not to the
        # Blender scene; clearing it here makes load-time disconnect
        # truly idempotent so a new scene's first ``set_project_name``
        # is always the one that sticks.
        self._project_name = None

    def _do_validate_path(self) -> None:
        if not self._backend:
            return
        directory = self._backend.current_directory
        # For the three native backends the local filesystem is probed
        # directly, each with the resolver its own connect path uses; for SSH /
        # Docker the check goes through ``exec_command``.
        # SSH/Docker targets are always Linux, so use POSIX joins and no .exe
        # regardless of the client OS.
        #
        # THE REFUSAL HERE IS NOT A MESSAGE BUT A CANCELLATION, which is why
        # each branch probes exactly what its spawn path probes. The
        # `Connected` transition schedules this check, and a caller that starts
        # an upload right after connecting (every `bl_*` scenario does) has its
        # pipeline in flight when an `ErrorOccurred` lands: that transition
        # resets the activity and clears the pending build, the upload then
        # completes into nothing, and the solver sits at NO_BUILD until the
        # caller's wait expires. Measured on the Windows leg of Blender CI,
        # where `build.bat` puts the CUDA server under target/cuda/release and
        # the check looked only under target/release: 23 scenarios failed by
        # timeout with the upload landed and no build request ever sent (runs
        # 35159051178 to 35193244970, and a Windows reproduction whose event
        # trail named `validate_path`).
        if self._backend.backend_type in NATIVE_BACKENDS:
            from .connection import native_path_check
            device = getattr(self._backend, "_device", "GPU")
            gpu_backend = getattr(self._backend, "_gpu_backend", "AUTO")
            error = native_path_check(
                self._backend.backend_type, directory, device, gpu_backend
            )
            if error is not None:
                # Same sentence the launch path raises, so the user is not told
                # two different things about one missing binary.
                self._engine.dispatch(ErrorOccurred(
                    error=error,
                    source="validate_path",
                ))
            return
        # THE REMOTE ROOT IS JUDGED BY THE LISTING CONNECT ALREADY TOOK, not by
        # a second command naming one path. Asking about
        # `target/release/ppf-cts-server` alone refuses a solver host holding a
        # DISTRIBUTION, which ships `target/<backend>/release` and no
        # `target/release`: the message is "Remote path not found" naming a
        # path that host is right not to have. That refusal is a cancellation
        # rather than a message (see the note above), so the artist's first
        # upload lands into nothing.
        #
        # ASKED ONLY WHETHER THE ROOT HOLDS A SOLVER AT ALL, never whether it
        # holds the device currently selected. The device is applied at Start
        # Server, which is where a selection the host cannot serve is refused
        # by name; refusing it here would cancel a pipeline over a choice the
        # artist can still change.
        from . import remote_builds
        from .connection import remote_holds_any_server

        if remote_builds.probe_error():
            self._engine.dispatch(ErrorOccurred(
                error=remote_builds.probe_error(),
                source="validate_path",
            ))
            return
        probed = remote_builds.normalize_root(directory)
        if not remote_holds_any_server(probed, remote_builds.cached_builds()):
            from .connection import remote_not_found_message
            self._engine.dispatch(ErrorOccurred(
                error=remote_not_found_message(
                    probed, remote_builds.cached_builds()
                ),
                source="validate_path",
            ))

    @contextlib.contextmanager
    def _query_state_guard(self, query_id):
        """Serialize local response mutations with query retirement, never I/O.

        Caller holds the collision lock, then this takes the engine lock, the
        same order as result publication. Never call engine.state inside it.
        """
        if query_id is None:
            yield True
            return
        with self._engine._lock:
            yield self._status_query_matches(self._engine._state, query_id)

    @staticmethod
    def _status_query_matches(state, query_id):
        return (state.status_query_id == query_id
                and (state.activity == Activity.STATUS_QUERY
                     or (state.activity == Activity.EXECUTING and state.query_is_command))
                and state.phase == Phase.ONLINE and state.server != Server.STOPPING)

    def _status_query_current(self, query_id):
        return query_id is None or self._status_query_matches(self._engine.state, query_id)

    def _publish_status_query_result(self, query_id, response, connection_ticket,
                                     failure_reason="", invalidate=False):
        # The caller holds _collision_lock. Pair the final query-id check with
        # Engine.tick's state-commit lock so a completed local cancellation
        # cannot be followed by a late cache write or result enqueue. Do not
        # call engine.state under this non-reentrant lock. Cache.record and
        # dispatch only copy Python data / enqueue; no I/O or bpy runs here.
        with self._engine._lock:
            if not self._status_query_matches(self._engine._state, query_id):
                return False
            if invalidate:
                self._collision_invalidate(discard=True)
            if response is not None:
                self._response_cache.record(response)
            self._engine.dispatch(StatusQueryResult(
                query_id=query_id, response=response,
                expected_connection_ticket=connection_ticket,
                failure_reason=failure_reason))
            return True

    def _cancel_status_query(self, query_id):
        if query_id is not None:
            self._engine.dispatch(StatusQueryCancelled(
                query_id=query_id,
                reason="Query wait cancelled because its connection or project changed; result remains unconfirmed; no stop or retry was sent."))

    def _do_query(self, request: dict | None = None, collision_ticket=None,
                  expected_connection_ticket=None, status_query_id=None) -> None:
        if not self._status_query_current(status_query_id):
            return
        if expected_connection_ticket is not None and not self.connection_ticket_is_current(expected_connection_ticket):
            self._cancel_status_query(status_query_id)
            return
        ticket = collision_ticket or self._collision_ticket()
        backend, project = ticket[1:]
        if not backend or not project or not self._collision_current(ticket):
            self._cancel_status_query(status_query_id)
            return
        response = None
        try:
            with self._collision_lock:
                needs_collision = (self._collision_expected is not None or self._collision_required
                                   or self._body_expected is not None or self._body_required)
            if needs_collision and request and request.get("request") in ("start", "resume"):
                # Download fresh on the ordinary I/O worker immediately before
                # the real start query. No network or bpy work in a modal tick.
                response, alive = self._call_owned(backend.query, {}, project, self._chunk_size)
                if not alive:
                    collision_receipt.fail("server unavailable before Run")
                if not self._collision_validate(response, ticket, force=True, query_id=status_query_id):
                    return
            if not self._collision_current(ticket):
                self._cancel_status_query(status_query_id)
                return
            if request and request.get("request") == "resume":
                with self._collision_lock:
                    fence = self._resume_fence
                if fence is None or fence["ticket"] != ticket:
                    raise RuntimeError("RESUME_GENERATION_UNAVAILABLE: request context retired")
                baseline = self._read_resume_generation(backend, fence["root"], project)
                if baseline is None:
                    # Only the server can prove status is genuinely absent;
                    # download exceptions also cover permissions and bad data.
                    reply, alive = self._call_owned(backend.query, {"request":"resume_generation_baseline"}, project, self._chunk_size)
                    if not alive or not self._collision_current(ticket):
                        raise RuntimeError("RESUME_GENERATION_UNAVAILABLE: baseline query retired")
                    from .resume_generation_baseline import prior_generation
                    current = self._engine.state
                    identity = fence["identity"]
                    if (current.remote_root != identity["root"] or current.server_upload_id != identity["upload_id"]
                            or current.server_param_hash != identity["param_hash"] or current.server_data_hash != identity["data_hash"]):
                        raise RuntimeError("RESUME_GENERATION_UNAVAILABLE: project generation changed")
                    baseline = prior_generation(reply, identity, project)
                if baseline is None:
                    raise RuntimeError("RESUME_GENERATION_UNAVAILABLE: cannot establish prior native generation")
                with self._collision_lock:
                    if self._resume_fence is not fence:
                        return
                    fence["baseline"] = baseline["launch_id"]
            if expected_connection_ticket is not None and not self.connection_ticket_is_current(expected_connection_ticket):
                self._cancel_status_query(status_query_id)
                return
            if not self._status_query_current(status_query_id):
                return
            response, alive = self._call_owned(backend.query, request or {}, project, self._chunk_size)
            if not self._status_query_current(status_query_id):
                return
            if not self._collision_current(ticket):
                self._cancel_status_query(status_query_id)
                return
            if alive and (response.get("error") or response.get("status") in ("NO_DATA", "NO_BUILD", "BUILDING")):
                with self._collision_lock, self._query_state_guard(status_query_id) as current:
                    if not current:
                        return
                    if not self._collision_current(ticket):
                        self._cancel_status_query(status_query_id)
                        return
                    if response.get("error"):
                        self._collision_invalidate()
                        ticket = self._collision_ticket()
                    elif self._collision_verified or self._body_verified:
                        self._collision_invalidate(discard=True)
                        ticket = self._collision_ticket()
            if alive and response.get("status") in ("READY", "RESUMABLE") and not response.get("error"):
                command = request.get("request") if request else None
                # Show project status on a cold connection. Actions that use
                # the result still rebuild and validate the local expectation.
                inspect_only = command not in _SCENE_QUERY_COMMANDS and self._cold_expectation_pending()
                if not inspect_only and not self._collision_validate(response, ticket, query_id=status_query_id):
                    self._cancel_status_query(status_query_id)
                    return
        except Exception as exc:
            if status_query_id is not None:
                if self._status_query_current(status_query_id):
                    with self._collision_lock:
                        if self._collision_current(ticket):
                            self._publish_status_query_result(
                                status_query_id, None, expected_connection_ticket,
                                failure_reason=str(exc), invalidate=True)
                        else:
                            self._cancel_status_query(status_query_id)
                return
            self._collision_reject(exc, ticket, response)
            return
        if alive:
            # Mirror into cache before the transition runs so any parallel
            # UI read sees the latest.
            with self._collision_lock:
                if not self._collision_current(ticket):
                    self._cancel_status_query(status_query_id)
                    return
                if status_query_id is None:
                    self._response_cache.record(response)
                    self._engine.dispatch(ServerPolled(response=response))
                else:
                    self._publish_status_query_result(
                        status_query_id, response, expected_connection_ticket)
        elif status_query_id is not None:
            with self._collision_lock:
                if self._collision_current(ticket):
                    self._publish_status_query_result(
                        status_query_id, None, expected_connection_ticket)
                else:
                    self._cancel_status_query(status_query_id)
        elif request:
            with self._collision_lock:
                self._resume_fence = None
            # User-initiated query (a non-empty request payload) couldn't
            # reach the server. Without this branch ``Activity.EXECUTING``
            # never clears and the modal hangs on its status message
            # ("Deleting Remote Data...", etc.) until its timeout. Empty
            # background polls stay silent so a server that hasn't booted
            # yet doesn't trip a spurious server-lost reset.
            self._engine.dispatch(ServerLost())

    def _launch_device(self, device: str = "") -> str:
        """The Compute Device this launch runs on.

        What Start Server was given, else what the connection was made with.
        One reader, so the binary a launch resolves and the GPU token it writes
        cannot disagree about which device is in play.
        """
        from .connection import DEVICE_GPU

        return device or getattr(self._backend, "_device", DEVICE_GPU)

    def _remote_server_binary(
        self, directory: str, device: str = "", gpu_backend: str = ""
    ) -> str:
        """The server on the solver host this connection's selection names.

        Raises rather than falling back, which is the whole point: a selection
        that cannot be satisfied has to say so at Start Server, naming what the
        host holds, instead of launching another build and reporting success.
        The three refusals `remote_not_found_message` distinguishes are each
        something the artist can act on without changing the path.

        A HOST THAT COULD NOT BE LISTED IS NOT A HOST WITH NO BUILDS. Where the
        probe failed, its reason is what the artist needs, so it is raised
        instead of a refusal derived from an empty listing.
        """
        from . import remote_builds
        from .connection import (
            DEVICE_GPU,
            GPU_BACKEND_AUTO,
            remote_not_found_message,
            remote_server_binary,
        )

        # THE SELECTION THE LAUNCH WAS GIVEN, falling back to the one the
        # connection was made with. The panel's remote rows are drawn only once
        # a connection is up, so the artist's answer arrives with Start Server;
        # reading the backend's connect-time copy alone would leave those rows
        # movable and inert, which is the silent substitution the whole device
        # mechanism exists to prevent.
        device = self._launch_device(device)
        gpu_backend = gpu_backend or getattr(
            self._backend, "_gpu_backend", GPU_BACKEND_AUTO
        )
        # THE ROOT THE LISTING WAS TAKEN UNDER, which is what its keys are
        # joined from. See `remote_builds.normalize_root`.
        directory = remote_builds.normalize_root(directory)
        listing = remote_builds.cached_builds()
        if not listing and remote_builds.probe_error():
            raise FileNotFoundError(remote_builds.probe_error())
        found = remote_server_binary(directory, listing, device, gpu_backend)
        if found is None:
            raise FileNotFoundError(
                remote_not_found_message(directory, listing, device, gpu_backend)
            )
        return found

    def _do_launch_server(
        self,
        cuda_device: int = -1,
        cuda_device_uuid: str = "",
        device: str = "",
        gpu_backend: str = "",
    ) -> None:
        # BOUND ONCE, for the reason `_do_stop_server` states: a disconnect
        # runs on the main thread and can clear `self._backend` while this
        # method is inside its startup wait loop.
        backend = self._backend
        if not backend:
            return

        # THE THREE NATIVES SPAWN A CHILD PROCESS instead of writing a launch
        # script, because there is no shell on the other side of them. Their
        # launch is one path rather than three: the differences are the name in
        # the message and whether a GPU can be named at all.
        if backend.backend_type in NATIVE_BACKENDS:
            # A NATIVE LAUNCH TAKES NO DEVICE FROM HERE. Its device was settled
            # when the connection was made, because connecting itself refuses a
            # root that holds no build for it, and the backend keeps that answer
            # so a Stop/Start cycle makes the same choice.
            self._launch_native_server(backend, cuda_device, cuda_device_uuid)
            return

        port = backend.server_port
        directory = backend.current_directory

        # Check Docker port exposure on the SSH host (not inside container).
        # Only meaningful for SSH+container backends, where Blender runs on
        # the user's machine and reaches the container through an SSH tunnel
        # to the docker host. For plain DOCKER backend, Blender talks to the
        # local docker daemon directly via the Container API and the
        # _instance object has no exec_command.
        if (backend.backend_type == "ssh"
                and hasattr(backend, '_container')
                and backend._container):
            container = backend._container
            if hasattr(backend, '_instance') and backend._instance:
                # Run docker port directly on the SSH host
                _stdin, stdout, stderr = self._call_owned(backend._instance.exec_command, 
                    f"docker port {container} {port}"
                )
                exit_code = stdout.channel.recv_exit_status()
                if exit_code != 0:
                    raise ConnectionError(
                        f"Docker port {port} is not exposed on container '{container}'. "
                        f"Please expose the port with '-p {port}:{port}' when starting the container."
                    )

        # Paths constructed here are sent to a Linux remote over SSH/Docker
        # (this branch never runs for a native backend, see the early
        # return above).
        # Use posixpath so a Windows client doesn't emit backslashes into
        # the shell commands.
        server_log = posixpath.join(directory, "server.log")
        progress_file = posixpath.join(directory, "progress.log")
        script_path = "/tmp/start_server.sh"

        # Clear progress.log
        self._call_owned(backend.exec_command, f"rm -f {progress_file}", shell=True)

        # ppf-cts-server defaults to binding 127.0.0.1. That is what we want
        # for SSH (Direct): paramiko's direct-tcpip channel terminates at the
        # remote's loopback. Inside a container, however,
        # docker -p HOST:CONTAINER forwards traffic to the container's
        # external interface (eth0), not loopback, so we must bind
        # 0.0.0.0 there.
        in_container = bool(getattr(backend, "_container", ""))
        host_flag = "--host 0.0.0.0 " if in_container else ""

        # The Rust ppf-cts-server binary writes ``progress.log`` markers
        # (SERVER_STARTING / SERVER_READY) and serves the same wire
        # protocol the addon speaks; build it with
        # ``cargo build --release -p ppf-cts-server``.
        # An environment assignment in front of the binary is how the GPU
        # choice is delivered to a server started through a shell. It is
        # applied on the solver host, so the index is that host's, which is
        # also where the panel's device list was enumerated.
        #
        # WHICH BUILD DIRECTORY, ASKED OF THE LISTING THE SOLVER HOST GAVE,
        # rather than spelled here. Naming one path would be two defects at
        # once: the artist's Compute Device would reach nothing on a remote
        # connection, and a solver host holding a DISTRIBUTION could not be
        # launched at all, since `build-linux-native/bundle.sh` ships
        # `target/<backend>/release` and no `target/release`.
        rust_bin = self._remote_server_binary(directory, device, gpu_backend)
        # A CPU RUN NAMES NO GPU, and dropping the selection here is not tidiness.
        # `CUDA_VISIBLE_DEVICES` in front of a binary with no CUDA in it changes
        # nothing about the run, and `describe_launch` would then write a console
        # line naming a GPU the solve is not on, which is a record of something
        # that did not happen. The panel already hides the picker for a CPU
        # device; this is the same answer where it is applied.
        from .connection import DEVICE_CPU

        on_cpu = self._launch_device(device) == DEVICE_CPU
        gpu_index = AUTOMATIC if on_cpu else cuda_device
        gpu_uuid = "" if on_cpu else cuda_device_uuid
        server_cmd = (
            f"{shell_prefix(gpu_index, gpu_uuid)}"
            f"{rust_bin} {host_flag}--port {port}"
        )

        # Activate the project venv so the build worker subprocess
        # (spawned by ppf-cts-server) resolves `python3` to one with the
        # runtime deps (cbor2 / psutil). Without this, a non-login SSH
        # shell falls through to the system python, which lacks them and
        # the build fails with a ModuleNotFoundError. The `_ppf_cts_py`
        # cdylib itself is not installed into the venv; `frontend` loads
        # it directly from `target/<profile>/`.
        # Convention: $HOME/.local/share/ppf-cts/venv.
        # Falls through silently when the venv is absent (Docker /
        # win-native paths use embedded Python set up differently).
        venv_activate = '$HOME/.local/share/ppf-cts/venv/bin/activate'
        activate_clause = (
            f'[ -f {venv_activate} ] && source {venv_activate}; '
        )
        # THE BUILD DIRECTORY IS NAMED, NOT ONLY THE BINARY, and leaving it out
        # is the split `_apply_target_dir` prevents on the native path. A run
        # takes three things out of a build directory, and the server binary is
        # one: the build worker's Python loads the cdylib from whichever target
        # directory `frontend._target_dirs` finds first, and `frontend`
        # then writes THAT directory into the session's `command.sh` as
        # `SOLVER_PATH`. Launching `target/cpu/release/ppf-cts-server` without
        # naming its target directory therefore starts the CPU server and runs
        # the solve on whatever `<root>/target` holds, with nothing anywhere
        # reporting the split, because each binary answers `--backend` honestly
        # about itself and neither is asked about the other.
        #
        # It is exported on its own line rather than inlined into the
        # `bash -c "..."` below, where a `$` or a backtick would be expanded by
        # the remote shell before the server ever saw it.
        target_dir = remote_target_dir(rust_bin)
        target_clause = (
            f'export CARGO_TARGET_DIR="{target_dir}"\n' if target_dir else ""
        )
        script = (
            f'#!/bin/bash\ncd "{directory}"\n'
            f'{target_clause}'
            f'nohup bash -c "{activate_clause}{server_cmd}" > "{server_log}" 2>&1 &\n'
        )
        self._call_owned(backend.exec_command, f"cat <<'EOF' > {script_path}\n{script}EOF\n", shell=True)
        self._call_owned(backend.exec_command, f"chmod +x {script_path}", shell=True)
        result = self._call_owned(backend.exec_command, script_path, shell=True)
        if result["exit_code"] != 0:
            raise FileNotFoundError("Failed to launch server")
        console.write(describe_launch(gpu_index, True, gpu_uuid))

        # Monitor startup. We keep looping until the client can actually
        # reach the server through whatever transport the backend uses
        # (direct-tcpip over SSH, docker-proxy, etc.). ``SERVER_READY`` in
        # progress.log only confirms that the server's own ``bind()`` call
        # returned; it says nothing about host-side port forwarding. In
        # setups where the docker port binding is misconfigured, the old
        # break-on-SERVER_READY path would dispatch ServerLaunched even
        # though no subsequent client query could reach the server — the
        # Start Server button would silently flip from "Server Starting..."
        # to a greyed-out "Start Server on Remote" (because is_server_running
        # is True) while the connection was actually broken.
        max_wait = 16
        start = time.time()
        last_lines = 0
        ready_marker_seen = False
        while True:
            elapsed = time.time() - start
            result = self._call_owned(backend.exec_command, 
                f"cat {progress_file} 2>/dev/null", shell=True,
            )
            lines = result.get("stdout", [])
            if len(lines) > last_lines:
                last_lines = len(lines)
                for line in lines:
                    if "ERROR" in line.upper() or "FAILED" in line.upper():
                        raise RuntimeError(f"Server startup failed: {line}")
                if lines and "SERVER_READY" in lines[-1]:
                    ready_marker_seen = True

            # A failed bind (port already taken) is written to server.log,
            # not progress.log, so SERVER_READY never appears and the loop
            # would otherwise burn the full max_wait and report a generic
            # timeout that leaves the user guessing. Detect the "port in
            # use" case and fail fast, naming the port: on a shared host the
            # holder is usually a stale ppf-cts-server or a Docker container
            # publishing the same port (`-p PORT:PORT`, whose docker-proxy
            # binds the host port even with nothing live inside).
            if not ready_marker_seen:
                log_tail = self._call_owned(backend.exec_command, 
                    f"tail -20 {server_log} 2>/dev/null", shell=True,
                )
                tail_text = "\n".join(log_tail.get("stdout", []))
                if "Address already in use" in tail_text or "os error 98" in tail_text:
                    raise ConnectionError(
                        f"Server port {port} is already in use on the remote "
                        "host, so ppf-cts-server could not bind it. Press Kill "
                        "Process to end a stale ppf-cts-server there, stop "
                        "whatever else holds the port (a Docker container "
                        f"publishing it with `-p {port}:{port}`), or choose a "
                        f"different server port.\n{tail_text}"
                    )

            if elapsed > max_wait:
                log_result = self._call_owned(backend.exec_command, 
                    f"tail -20 {server_log}", shell=True,
                )
                details = "\n".join(log_result.get("stdout", []))
                if ready_marker_seen:
                    raise ConnectionError(
                        "Server reached SERVER_READY but the client cannot "
                        "reach it. Check that the server port is forwarded "
                        f"to the client (e.g. `docker run -p {port}:{port}`).\n"
                        f"{details}"
                    )
                raise TimeoutError(f"Server startup timed out.\n{details}")

            # Only try querying after SERVER_READY — before that the
            # listener isn't up yet and every query is guaranteed to fail.
            if ready_marker_seen and self._project_name:
                response, alive = self._call_owned(backend.query, 
                    {}, self._project_name, self._chunk_size,
                )
                if alive:
                    break

            time.sleep(1)

        self._engine.dispatch(ServerLaunched())

    def _launch_native_server(
        self,
        # TAKEN FROM THE CALLER, never re-read. `_do_launch_server` bound it
        # once for the whole launch, and re-reading `self._backend` here would
        # reopen the window that binding closed: a disconnect runs on the main
        # thread and can clear the attribute while this helper is inside its
        # readiness wait.
        backend,
        cuda_device: int = -1,
        cuda_device_uuid: str = "",
    ) -> None:
        """Start the server a native backend owns, and wait for it to answer.

        A FAILURE AFTER A SUCCESSFUL SPAWN STOPS WHAT IT STARTED. Otherwise a
        server that came up and then failed its readiness wait would be left
        holding the port, and the next Connect would meet a squatter it did not
        start rather than the error that actually happened.
        """
        kind = backend.backend_type
        label = NATIVE_BACKENDS[kind]
        launched = False
        try:
            # Acquisition must retain its ownership result even if retirement
            # lands during start_server. Do not use the generic post-I/O throw.
            with self._lifecycle_commit() as current:
                if not current:
                    raise _LifecycleRetired()
            if kind == "mac_native":
                # No device selection accompanies it: the Metal backend opens
                # the system default device and offers no way to name another,
                # so there is nothing for a device argument to carry.
                launched = backend.start_server()
            else:
                launched = backend.start_server(cuda_device, cuda_device_uuid)
            if not self._execution_current():
                raise _LifecycleRetired()
            self._wait_for_native_server(backend, label)
        except (Exception, _LifecycleRetired) as e:
            cleanup_error = None
            if launched:
                try:
                    backend.stop_server()
                except Exception as stop_error:
                    cleanup_error = stop_error
            if isinstance(e, _LifecycleRetired):
                if cleanup_error is not None:
                    console.write(f"Retired launch cleanup failed: {cleanup_error}")
                # Only an owned successful spawn is stopped above. An attach
                # (launched=False) never gains permission to stop that process.
                raise
            detail = str(e)
            if cleanup_error is not None:
                detail += f" Cleanup also failed: {cleanup_error}"
            self._engine.dispatch(ErrorOccurred(
                error=f"Failed to start server: {detail}",
                source="launch_server",
            ))
            return
        if kind == "mac_native":
            console.write(
                "Solver server started on the system default Metal device."
                if launched else
                "Solver server: attached to one that was already running, so "
                "it keeps the Metal device it started with."
            )
        else:
            console.write(describe_launch(cuda_device, launched, cuda_device_uuid))
        self._engine.dispatch(ServerLaunched())

    def _wait_for_native_server(
        self,
        # TAKEN FROM THE CALLER, for the reason the launch path states: this
        # is the LONGEST window of the three, a readiness loop that polls
        # until a deadline, and it re-read `self._backend` on every pass.
        backend,
        label: str,
        timeout: float = 16.0,
    ) -> None:
        """Wait until a native child on this machine answers the solver protocol.

        ONE WAITER FOR THE THREE NATIVES, because what is waited for is
        identical: a child process the add-on spawned, on a loopback port, that
        answers the protocol once it is ready and whose exit is the one thing
        that can end the wait early. Only *label* differs, and it appears in the
        two messages the artist reads.

        A CHILD THAT EXITS IS REPORTED AS AN EXIT rather than waited out, so a
        server that cannot start says so in a second instead of at the timeout,
        with the code it exited with and where to read why.
        """
        from .connection import _probe_ppf_cts_server

        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if self._call_owned(_probe_ppf_cts_server, 
                backend.server_port, timeout=min(0.5, remaining)
            ):
                return
            process = getattr(backend, "_process", None)
            if process is not None:
                returncode = process.poll()
                if returncode is not None:
                    raise RuntimeError(
                        f"{label} server exited with code {returncode} before "
                        "becoming ready. Check server.log."
                    )
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        raise TimeoutError(
            f"{label} server did not become ready within {timeout:g} seconds. "
            "Check server.log."
        )

    def _do_stop_server(self) -> None:
        # Capture the original target once; never follow a new connection.
        backend, project, chunk_size = self._backend, self._project_name or "", self._chunk_size
        if not backend:
            return
        if backend.backend_type in NATIVE_BACKENDS:
            report = self._call_owned(backend.stop_server)
            with self._lifecycle_commit() as current:
                if not current:
                    return
                self.last_kill_report = report
                self._response_cache.clear()
            self._engine.dispatch(ServerStopped())
            return
        container = getattr(backend, "container", "") or ""
        report = self._call_owned(
            kill_remote_server, lambda *a, **k: self._call_owned(backend.exec_command, *a, **k),
            where=f"container {container}" if container else "the solver host",
            port=backend.server_port)
        alive = True
        for _ in range(5):
            _response, alive = self._call_owned(backend.query, {}, project, chunk_size)
            if not alive:
                break
            time.sleep(0.25)
        with self._lifecycle_commit() as current:
            if not current:
                return
            self.last_kill_report = report
            self._response_cache.clear()
        if alive:
            self._engine.dispatch(ErrorOccurred(
                error=("Stop Server: the solver is still answering on port "
                       f"{backend.server_port}. Check that the process is "
                       "reachable from the container or host the add-on is driving."),
                source="stop_server"))
        self._engine.dispatch(ServerStopped())

    def _do_send_data(self, remote_path: str, data: bytes) -> None:
        if not self._backend or not self._project_name:
            return

        def progress_cb(p, t):
            self._engine.dispatch(ProgressUpdated(progress=p, traffic=t))

        def interrupt_cb():
            return self._interrupt.is_set()

        self._call_owned(self._backend.send_data, 
            remote_path, data, self._project_name,
            chunk_size=self._chunk_size,
            progress_cb=self._owned_callback(progress_cb), interrupt_cb=self._owned_interrupt(interrupt_cb),
        )
        self._engine.dispatch(SendDataComplete())

    def _do_upload_atomic(self, project_root: str, data: bytes, param: bytes,
                          data_hash: str, param_hash: str, collision_ticket=None,
                          expected_connection_ticket=None) -> None:
        """Atomic combined upload of (data.pickle, param.pickle).

        On success dispatches ``UploadPipelineComplete`` so the transition
        can move from SENDING straight into BUILDING. Errors are routed
        via ``ErrorOccurred`` the same way ``_safe_run`` handles
        transport-level failures. ``param_hash`` rides along on the
        wire so the server can store it next to the pickles and echo
        it on every status response.
        """
        if expected_connection_ticket is not None and not self.connection_ticket_is_current(expected_connection_ticket):
            return
        ticket = collision_ticket or self._collision_ticket()
        backend, project = ticket[1:]
        if not backend or not project or not self._collision_current(ticket):
            return
        with self._collision_lock:
            expected = self._collision_expected
            body_expected = self._body_expected
        previous_upload = ""
        if expected is not None or body_expected is not None:
            before, alive = self._call_owned(backend.query, {}, project, self._chunk_size)
            if not alive or "upload_id" not in before or not isinstance(before["upload_id"], str):
                collision_receipt.fail("server cannot identify the pre-upload session")
            previous_upload = before["upload_id"]
        if not self._collision_current(ticket):
            return

        def progress_cb(p, t):
            self._engine.dispatch(ProgressUpdated(progress=p, traffic=t))

        def interrupt_cb():
            return self._interrupt.is_set()

        if expected_connection_ticket is not None and not self.connection_ticket_is_current(expected_connection_ticket):
            return
        self._call_owned(backend.upload_atomic, 
            project_root, data, param, project,
            data_hash=data_hash, param_hash=param_hash,
            chunk_size=self._chunk_size,
            progress_cb=self._owned_callback(progress_cb), interrupt_cb=self._owned_interrupt(interrupt_cb),
        )
        if not self._collision_current(ticket):
            return
        if expected is not None or body_expected is not None:
            from dataclasses import replace
            response, alive = self._call_owned(backend.query, {}, project, self._chunk_size)
            if not alive:
                collision_receipt.fail("server did not acknowledge the complete atomic upload")
            upload = collision_receipt.token(response.get("upload_id"))
            root = response.get("root")
            if upload == previous_upload or not isinstance(root, str) or not root:
                collision_receipt.fail("atomic upload did not mint a fresh session identity")
            bound = replace(expected, upload_id=upload, root=root) if expected is not None else None
            body_bound = replace(body_expected, upload_id=upload, root=root) if body_expected is not None else None
            self._collision_response_identity(response, bound or body_bound)
            with self._collision_lock:
                if (not self._collision_current(ticket) or self._collision_expected != expected
                        or self._body_expected != body_expected):
                    return
                self._collision_expected = bound
                self._body_expected = body_bound
        with self._collision_lock:
            if self._collision_current(ticket):
                self._engine.dispatch(UploadPipelineComplete(expected_connection_ticket=expected_connection_ticket))

    def _do_receive_data(self, remote_path: str) -> None:
        backend, project, chunk_size = self._backend, self._project_name, self._chunk_size
        if not backend or not project:
            return
        def progress_cb(p, t):
            self._engine.dispatch(ProgressUpdated(progress=p, traffic=t))
        def interrupt_cb():
            return self._interrupt.is_set()
        data = self._call_owned(backend.receive_data, 
            remote_path, project, chunk_size=chunk_size,
            progress_cb=self._owned_callback(progress_cb),
            interrupt_cb=self._owned_interrupt(interrupt_cb))
        with self._lifecycle_commit() as current:
            if not current:
                return
            with self._data_lock:
                self._received_data = data
        self._engine.dispatch(ReceiveDataComplete(data=data))

    def _count_remote_frames(self, root: str, backend=None, project=None, chunk_size=None) -> int:
        """Discover the max frame index from vert_*.bin files on the remote.

        Returns the highest N found in vert_N.bin, which matches the
        server's ``frame`` field semantics (max frame index, not file count).
        vert_0.bin is the rest pose and is excluded from the count.
        """
        backend = backend if backend is not None else self._backend
        project = project if project is not None else self._project_name
        chunk_size = chunk_size if chunk_size is not None else self._chunk_size
        if not backend or not project:
            return 0
        try:
            # The native backends read the output directory off THIS machine,
            # so glob it rather than shelling out. On Windows the shell-out
            # below is not merely slower but wrong: cmd.exe has no `ls`, so it
            # exits non-zero and this returns 0 frames, which the caller cannot
            # tell apart from a solve that produced none. The fetch then
            # applies nothing, no mesh cache is attached, and the failure
            # surfaces far from its cause. Globbing is also the cheaper of the
            # two on POSIX, where it is what the shell would have done anyway,
            # which is why the other two natives take this branch as well.
            if backend.backend_type in NATIVE_BACKENDS:
                import glob
                output_dir = os.path.join(root, "session", "output")
                names = [
                    os.path.basename(p)
                    for p in glob.glob(os.path.join(output_dir, "vert_*.bin"))
                ]
            else:
                output_dir = posixpath.join(root, "session", "output")
                result = self._call_owned(backend.exec_command, 
                    f"ls -1 {output_dir}/vert_*.bin 2>/dev/null",
                    shell=True,
                )
                if result.get("exit_code", 1) != 0 or not result.get("stdout"):
                    return 0
                names = [line.strip().rsplit("/", 1)[-1] for line in result["stdout"]]
            max_frame = 0
            for name in names:
                if name.startswith("vert_") and name.endswith(".bin"):
                    try:
                        idx = int(name[5:-4])
                        if idx > max_frame:
                            max_frame = idx
                    except ValueError:
                        continue
            return max_frame
        except Exception as e:
            console.write(f"[count_remote_frames] {e}")
        return 0

    def _drop_stale_session_artifacts(self) -> None:
        """Drop the cached session artifacts when they describe an earlier upload.

        ``_anim_map``, ``_anim_surface_map`` and
        ``_anim_statistics_manifest`` are downloaded once and reused for
        every frame of a fetch, which is what keeps a live run from
        re-reading them on each poll. They describe ONE uploaded dataset,
        though: a scene whose group set changed produces frames with a
        different object set, and ``decode_frame`` rejects those against
        the previous manifest ("statistics frame object count does not
        match manifest"), so every frame of the new run is dropped.

        Clearing them at the transitions that rebuild a session is what
        the fetch path already does (``DoResetAnimationBuffer``), but
        Transfer and Run emit ``DoClearAnimation``, which drops the queued
        frames and the counters and leaves these three in place. Rather
        than adding this to the list of places that must remember, bind
        the artifacts to the upload they came from: the server mints a
        fresh ``upload_id`` for each upload and echoes it on every status
        response, so any path that lands a new dataset invalidates them,
        including one another client initiated.

        Called at both fetch entry points, before anything reads them.

        The queued frames go with them, under the one lock, because a frame
        was fetched against these artifacts and cannot be applied against
        the next set: ``take_one_animation_frame`` hands the map to the
        main thread, and a frame drawn while the map is empty applies to
        no object at all and writes no PC2, silently. Every other place
        that drops the artifacts (``DoResetAnimationBuffer``,
        ``_do_disconnect``) clears the queue in the same breath for the
        same reason. Both callers set the counters after this returns, so
        zeroing them here cannot race a fetch already in progress.
        """
        upload_id = str(self._response_cache.get("upload_id", "") or "")
        with self._anim_lock:
            if self._anim_upload_id == upload_id:
                return
            if self._anim_upload_id is not None:
                self._live_backlog = None
            self._anim_upload_id = upload_id
            self._anim_map = {}
            self._anim_surface_map = {}
            self._anim_display_pin_map = {}
            self._anim_bone_bindings = None
            self._anim_statistics_manifest = None
            self._anim_statistics_zero_fetched = False
            self._anim_frames.clear()
            self._anim_total = 0
            self._anim_applied = 0

    def _ensure_anim_map(self, root, collision_ticket=None, fetch_generation=-1, *, allow_legacy=False):
        """Download to locals, then commit the complete current session atomically."""
        if collision_ticket is None:
            collision_ticket = self._collision_ticket()
        backend, project = collision_ticket[1:]
        if backend is None or not project:
            raise RuntimeError("fetch map: no backend or project")
        with self._collision_lock:
            if fetch_generation < 0:
                fetch_generation = self._anim_map_generation
            if not self._fetch_job_current(collision_ticket, fetch_generation):
                return None
            chunk_size = self._chunk_size
            upload_id = str(self._response_cache.get("upload_id", "") or "")
            context = (fetch_generation, backend, project, root, upload_id)
            with self._anim_lock:
                if self._anim_map_context == context and self._anim_map:
                    if self._anim_legacy_context is not None and not self._anim_legacy_approved:
                        raise RuntimeError("The legacy result mapping is awaiting main-thread verification")
                    return (self._anim_map, self._anim_surface_map,
                            self._anim_display_pin_map, self._anim_bone_bindings)
        # No shared map/binding writes, bpy or network under either lock.
        map_path = _server_join(backend, root, "session", "map.pickle")
        map_data = self._call_owned(backend.receive_data, map_path, project, chunk_size=chunk_size)
        # Format-sniff between pickle (first byte 0x80) and CBOR map
        # (first byte 0xa0-0xb7). Producer: ``frontend/_scene_.py:export_fixed``.
        if map_data and map_data[0] != 0x80:
            anim_map = _decode_vertex_map_cbor(map_data)
        else:
            anim_map = pickle.loads(map_data)

        surface_map = {}
        smap_data = None
        try:
            smap_path = _server_join(backend, root, "session", "surface_map.pickle")
            smap_data = self._call_owned(backend.receive_data, smap_path, project, chunk_size=chunk_size)
            if smap_data and smap_data[0] != 0x80:
                payload = _decode_surface_map_cbor(smap_data)
            else:
                payload = pickle.loads(smap_data)
            # Require wire format v2: {"version": 2, "maps": {uuid: (tri_indices, coefs, surf_tri)}}.
            # Anything else is rejected so the client cannot apply the wrong reconstruction math.
            if (
                isinstance(payload, dict)
                and payload.get("version") == 2
                and isinstance(payload.get("maps"), dict)
            ):
                surface_map = payload["maps"]
            else:
                raise ValueError("surface_map.pickle has unsupported format")
        except Exception as exc:
            if smap_data is not None:
                raise RuntimeError(f"surface_map.pickle is invalid: {exc}") from exc
            if not isinstance(exc, FileNotFoundError) and not any(
                text in str(exc).lower() for text in ("not found", "no such file", "os error 2")
            ):
                raise

        # The snapshot is written for the output generation by the solver.
        # Comparing exact mapping bytes is required: reordered surface rows
        # can have the same dimensions yet reconstruct entirely different data.
        import hashlib
        import json
        identity_path = _server_join(backend, root, "session", "output", "fetch_mapping_identity.json")
        try:
            identity_data = self._call_owned(backend.receive_data, identity_path, project, chunk_size=chunk_size)
        except Exception as exc:
            missing = isinstance(exc, FileNotFoundError) or any(
                text in str(exc).lower() for text in ("not found", "no such file", "os error 2")
            )
            if not missing or smap_data is not None or not allow_legacy:
                raise RuntimeError("The result has no verifiable mapping identity; transfer and run again before fetching.") from exc
            status = self._read_resume_generation(backend, root, project)
            snapshot = self._engine.state
            if (status is None or status["frame"] <= 0 or snapshot.remote_root != root
                    or not snapshot.server_upload_id or not snapshot.server_data_hash
                    or self._anim_fetch_origin is None
                    or self._anim_fetch_origin[1:] != (snapshot.remote_root,
                        snapshot.server_upload_id, snapshot.server_data_hash)):
                raise RuntimeError("The legacy result generation cannot be verified; transfer and run again before fetching.") from exc
            with self._collision_lock:
                self._anim_legacy_context = (self._collision_epoch, id(backend), project,
                                             fetch_generation)
                self._anim_legacy_approved = False
            identity_data = None
        expected = {
            "version": 1,
            "map_sha256": hashlib.sha256(map_data).hexdigest(),
            "surface_sha256": hashlib.sha256(smap_data).hexdigest() if smap_data is not None else None,
        }
        if identity_data is not None:
            with self._collision_lock:
                self._anim_legacy_context = None
                self._anim_legacy_approved = False
            try:
                identity = json.loads(identity_data.decode("utf-8"))
            except (UnicodeError, ValueError, TypeError) as exc:
                raise RuntimeError("The result mapping identity is invalid") from exc
            if not isinstance(identity, dict) or type(identity.get("version")) is not int or identity["version"] not in (1, 2):
                raise RuntimeError("The result mapping identity is invalid")
            version = identity["version"]
            origin = identity.get("source_playback_origin")
            keys = {"version", "map_sha256", "surface_sha256"}
            if version == 2:
                keys.add("source_playback_origin")
            if set(identity) != keys or (version == 2 and type(origin) is not int):
                raise RuntimeError("The result playback origin is invalid")
            if any(identity[k] != expected[k] for k in ("map_sha256", "surface_sha256")):
                raise RuntimeError("The result mapping differs from the current build; transfer and run again before fetching.")
        else:
            origin = None
        if origin is None:
            raise RuntimeError("The legacy result has no verifiable playback origin; transfer and run again before fetching.")

        # Every session this frontend exports carries a display-pin map, empty
        # when it has no exact SOLID pin, so a map that cannot be read is an
        # error to report rather than a feature the session lacks.
        dpin_path = _server_join(
            backend, root, "session", "display_pin_map.pickle",
        )
        try:
            dpin_data = self._call_owned(backend.receive_data, 
                dpin_path, project, chunk_size=chunk_size,
            )
        except Exception as e:
            raise RuntimeError(
                "display_pin_map.pickle could not be read from the session; a "
                f"session built before display pins must be transferred again ({e})"
            ) from e
        display_pins = _decode_display_pin_map_cbor(dpin_data)
        display_pin_map = display_pins if display_pins["blocks"] else {}

        # Ordinary sessions have no bone receipt. A present receipt is strict:
        # malformed bytes must fail instead of falling back to mesh-only fetch.
        bone_path = _server_join(backend, root, "session", "blender_bone_bindings.cbor")
        try:
            bone_data = self._call_owned(backend.receive_data, 
                bone_path, project, chunk_size=chunk_size,
            )
        except FileNotFoundError:
            bone_data = None
        except Exception as exc:
            # Backends wrap remote ENOENT differently; only an explicit missing
            # file is optional. Other connection/read failures stay failures.
            if "not found" not in str(exc).lower() and "no such file" not in str(exc).lower() and "os error 2" not in str(exc).lower():
                raise
            bone_data = None
        from .rigid_pose import decode_bindings
        bone_bindings = decode_bindings(bone_data) if bone_data is not None else None

        with self._collision_lock:
            if not self._fetch_job_current(collision_ticket, fetch_generation):
                return None
            if self._body_expected is not None and any(isinstance(row,tuple) and len(row)==8 for row in self._body_expected.controls):
                if not self._body_verified or self._body_seal is None or bone_bindings is None:
                    body_control_receipt.fail('FEM event program must be verified before animation mapping')
                from .fem_body_control_events_receipt import runtime_context
                bone_bindings=dict(bone_bindings)
                bone_bindings['fem_control_events_v2']=runtime_context(self._body_seal)
            with self._anim_lock:
                if self._anim_upload_id != upload_id:
                    if self._anim_upload_id is not None:
                        self._live_backlog = None
                    self._anim_frames.clear()
                    self._anim_total = 0
                    self._anim_applied = 0
                if self._anim_map_context != context:
                    self._anim_statistics_manifest = None
                    self._anim_statistics_zero_fetched = False
                self._anim_playback_origin = origin
                self._anim_map = anim_map
                self._anim_surface_map = surface_map
                self._anim_display_pin_map = display_pin_map
                self._anim_bone_bindings = bone_bindings
                self._anim_upload_id = upload_id
                self._anim_map_context = context
                return (anim_map, surface_map, display_pin_map, bone_bindings)

    def _do_fetch_map(self, root, map_ticket=None, map_generation=-1):
        if map_ticket is None:
            with self._collision_lock:
                map_ticket = self._collision_ticket()
                if map_generation < 0:
                    map_generation = self._anim_map_generation
        try:
            if not self._fetch_job_current(map_ticket, map_generation):
                return
            # ResetAnimationBuffer retires receipt verification as well as
            # animation maps. Revalidate on this I/O worker before map download,
            # using the same barrier as Fetch Frames.
            backend, project = map_ticket[1:]
            try:
                response, alive = self._call_owned(backend.query, {}, project, self._chunk_size)
                if not alive:
                    self._dispatch_fetch_event(FetchFailed(
                        reason="fetch map: server unavailable before validation"),
                        map_ticket, map_generation)
                    return
                if not self._collision_validate(response, map_ticket, force=True):
                    return
            except Exception as exc:
                self._reject_fetch_current(exc, map_ticket, map_generation)
                return
            bundle = self._ensure_anim_map(root, map_ticket, map_generation, allow_legacy=True)
            if bundle is None:
                return
            self._dispatch_fetch_event(FetchMapComplete(
                map_data=bundle[0], surface_map=bundle[1]), map_ticket, map_generation)
        except Exception as exc:
            self._dispatch_fetch_event(FetchFailed(reason=f"fetch map: {exc}"),
                                       map_ticket, map_generation)

    def _read_resume_generation(self, backend, root, project):
        import cbor2
        path = _server_join(backend, root, "session", "output", "status.cbor")
        try:
            raw = self._call_owned(backend.receive_data, path, project, chunk_size=self._chunk_size)
        except _LifecycleRetired:
            raise
        except Exception:
            # This alone never proves absence; pre-send must obtain the
            # server's NotFound-only, generation-bound baseline proof.
            return None
        try:
            document = cbor2.loads(raw)
        except (ValueError, TypeError, cbor2.CBORDecodeError):
            return None
        if not isinstance(document, dict) or document.get("version") != 2 or document.get("kind") != "RunStatus":
            return None
        value = document.get("payload")
        if (not isinstance(value, dict) or type(value.get("frame")) is not int
                or value["frame"] < 0 or not isinstance(value.get("launch_id"), str)
                or not value["launch_id"]):
            return None
        return value

    def _begin_resume_generation(self, request, ticket):
        local_plan = self._prepare_resume_local_tail(request.get("resume_from"))
        snapshot = self._engine.state
        identity = {"root":snapshot.remote_root, "upload_id":snapshot.server_upload_id,
                    "param_hash":snapshot.server_param_hash, "data_hash":snapshot.server_data_hash}
        with self._collision_lock:
            self._retire_animation_state_locked()
            self._resume_fence = dict(ticket=ticket, generation=self._anim_map_generation, identity=identity,
                root=snapshot.remote_root, scene=self._scene_pointer(),
                from_frame=request.get("resume_from"), baseline=None, active=None,
                local_retired=False, local_plan=local_plan)

    def _resume_bound(self, backend, project, root, ticket, generation):
        with self._collision_lock:
            fence = self._resume_fence
            if fence is None:
                return -1
            if (fence["ticket"] != ticket or fence["generation"] != generation
                    or fence["root"] != root):
                return None
            baseline, active = fence["baseline"], fence["active"]
        if baseline is None:
            return None
        record = self._read_resume_generation(backend, root, project)
        if record is None or record["launch_id"] == baseline:
            return None
        if active is not None and record["launch_id"] != active:
            return None
        phase = record.get("phase")
        if phase == "ended":
            if record.get("outcome", {}).get("kind") not in ("finished", "saved_and_quit"):
                raise RuntimeError("RESUME_GENERATION_FAILED: native resume ended without a valid cache generation")
        elif phase not in ("initialized", "running", "saving"):
            return None
        with self._collision_lock:
            if self._resume_fence is not fence or not self._fetch_job_current(ticket, generation):
                return None
            fence["active"] = record["launch_id"]
        return record["frame"]

    def _prepare_resume_local_tail(self, frame, scene=None):
        """Read-only whole-batch proof before the Resume HTTP command."""
        if type(frame) is not int or frame < 0:
            return None
        import bpy, copy, os
        from pathlib import Path
        from . import rigid_pose, pc2
        from ..models.groups import get_addon_data
        scene = bpy.context.scene if scene is None else scene
        if bpy.context.scene != scene:
            raise RuntimeError("RESUME_CACHE_CONTEXT_CHANGED: return to the resumed scene")
        sid = get_addon_data(scene).state.last_session_id
        plan = dict(scene=scene.as_pointer(), blend_path=bpy.data.filepath, session_id=sid, frame=frame, bone=None, meshes=[])
        key = scene.get(rigid_pose.CACHE_KEY)
        index = (rigid_pose._cache.get(key) or rigid_pose._read_index_file(key)) if key else None
        if key and index is None:
            raise RuntimeError("RESUME_CACHE_UNOWNED: malformed bone cache")
        found = (key, index) if key else None
        if found:
            key, index = found
            expected = self._anim_bone_bindings
            if (not expected or any(index["receipt"].get(k) != expected.get(k)
                                    for k in ("scene_uuid", "scene_inputs_sha256"))):
                raise RuntimeError("RESUME_CACHE_UNOWNED: bone cache differs from the active native binding")
            rigid_pose._bound_targets(scene, index)
            source = Path(key).resolve()
            destination = (Path(pc2.get_pc2_dir()) / "bone_poses" /
                           index["receipt"]["scene_inputs_sha256"] / "index.json").resolve()
            shared = any(other != scene and other.get(rigid_pose.CACHE_KEY)
                         and Path(other[rigid_pose.CACHE_KEY]).resolve() == destination
                         for other in bpy.data.scenes)
            if shared or (source != destination and destination.parent.exists()
                          and any(destination.parent.iterdir())):
                raise RuntimeError("RESUME_CACHE_SHARED: bone working cache is referenced by another scene or already occupied")
            prefix = copy.deepcopy(index)
            prefix["frames"] = {n:v for n,v in index["frames"].items() if int(n) <= frame}
            blobs = {n:rigid_pose._io_path(source.parent / ("pose_"+n+".cbor")).read_bytes()
                     for n in prefix["frames"]}
            plan["bone"] = dict(source=source, destination=destination, index=prefix,
                                prefix=blobs, old_frames=list(index["frames"]),
                                key=key, receipt=copy.deepcopy(index["receipt"]))
        for obj in scene.objects:
            if obj.get("_solver_uuid") not in self._anim_map:
                continue
            mod = obj.modifiers.get(pc2.MODIFIER_NAME)
            if obj.type not in ("MESH", "CURVE") or (obj.type == "MESH" and mod is None):
                continue
            key = pc2.object_pc2_key_readonly(obj)
            shared_object = any(other != scene and other.objects.get(obj.name) == obj
                                for other in bpy.data.scenes)
            if obj.type == "CURVE":
                if (not sid or obj.library or obj.data.library or not obj.is_editable
                        or shared_object or key not in pc2.assigned_uuids_in_active_groups(scene)
                        or key not in pc2._curve_cache
                        or any(other != obj and pc2.object_pc2_key_readonly(other) == key
                               for other in bpy.data.objects)):
                    raise RuntimeError("RESUME_CACHE_UNOWNED: unproven or shared Curve producer cache")
                source = Path(pc2.get_pc2_path(key)).resolve()
                from . import curve_rod
                from .client import _get_curve_cv_count
                params = curve_rod.compute_params(obj)
                expected_samples = sum(len(sp["params"]) for sp in params["splines"])
                mapping = numpy.asarray(self._anim_map[key])
                disk = pc2._load_pc2_array(str(source))
                memory = pc2._curve_cache[key]
                if (mapping.ndim != 1 or len(mapping) != expected_samples
                        or len(numpy.unique(mapping)) != len(mapping)
                        or not numpy.issubdtype(mapping.dtype, numpy.integer)
                        or numpy.any(mapping < 0) or disk is None
                        or disk.shape[1] != _get_curve_cv_count(obj)
                        or not numpy.array_equal(disk, memory)):
                    raise RuntimeError("RESUME_CACHE_UNOWNED: Curve PC2/memory/map producer proof differs")
            else:
                if (not pc2.mesh_output_matches_session(scene, obj, sid, key=key, mod=mod)
                        or shared_object):
                    raise RuntimeError("RESUME_CACHE_UNOWNED: linked/shared object or unproven producer cache")
                source = Path(bpy.path.abspath(mod.filepath)).resolve()
            destination = Path(pc2.get_pc2_path(key)).resolve()
            if (pc2._mesh_output_path_shared(os.path.normcase(os.path.realpath(destination)), obj, mod)
                    or (destination.exists() and source != destination)):
                raise RuntimeError("RESUME_CACHE_SHARED: working PC2 is shared or already occupied")
            count = pc2.read_pc2_frame_count(str(source))
            vertices = pc2.read_pc2_n_verts(str(source))
            plan["meshes"].append(dict(obj=obj, mod=mod, source=source,
                destination=destination, count=min(count,frame+1), vertices=vertices,
                key=key, obj_pointer=obj.as_pointer(), mod_pointer=mod.as_pointer() if mod else None,
                data_pointer=obj.data.as_pointer(), object_type=obj.type))
        return plan

    def _retire_resume_local_tail(self):
        """Commit proven private working caches after native post-prune status."""
        fence = self._resume_fence
        if not fence or not fence["active"] or fence["local_retired"]:
            return
        plan = fence["local_plan"]
        if plan is None:
            fence["local_retired"] = True
            return
        import bpy, json, os, struct
        from pathlib import Path
        from . import rigid_pose, pc2
        from ..models.groups import get_addon_data
        scene = next((s for s in bpy.data.scenes if s.as_pointer()==plan["scene"]),None)
        if scene is None:
            raise RuntimeError("RESUME_CACHE_UNOWNED: author scene disappeared")
        fresh = self._prepare_resume_local_tail(plan["frame"], scene=scene)
        def identity(candidate):
            bone = candidate["bone"]
            return (candidate["scene"], candidate["blend_path"], candidate["session_id"],
                    [(r["source"],r["destination"],r["key"],r["obj_pointer"],
                      r["mod_pointer"],r["data_pointer"],r["object_type"])
                     for r in candidate["meshes"]],
                    None if bone is None else (bone["key"],bone["source"],
                        bone["destination"],bone["receipt"]))
        if identity(fresh) != identity(plan):
            raise RuntimeError("RESUME_CACHE_CONTEXT_CHANGED: cache ownership changed during resume")
        # Stage all COW files before changing any bound modifier/index. Source
        # files outside the current canonical cache are never modified/deleted.
        staged=[]
        try:
            for row in plan["meshes"]:
                destination=row["destination"]
                destination.parent.mkdir(parents=True,exist_ok=True)
                temporary=destination.with_name(destination.name+".resume-"+fence["active"]+".tmp")
                with open(row["source"],"rb") as source,open(temporary,"xb") as output:
                    staged.append((temporary,destination,row))
                    header=bytearray(source.read(32));struct.pack_into("<i",header,28,row["count"]);output.write(header)
                    remaining=row["count"]*row["vertices"]*12
                    while remaining:
                        chunk=source.read(min(remaining,1024*1024))
                        if not chunk:raise RuntimeError("RESUME_CACHE_INVALID: incomplete PC2 prefix")
                        output.write(chunk);remaining-=len(chunk)
                    output.flush();os.fsync(output.fileno())
            bone=plan["bone"]
            if bone:
                destination=bone["destination"]
                destination.parent.mkdir(parents=True,exist_ok=True)
                for n,blob in bone["prefix"].items():
                    target=destination.parent/("pose_"+n+".cbor")
                    if bone["source"]!=destination:rigid_pose._write(target,blob)
                # Source index is replaced only if it is this scene's proven,
                # unshared canonical index; external source remains untouched.
                rigid_pose._write(destination,json.dumps(bone["index"],ensure_ascii=False).encode("utf8"))
                rigid_pose._cache[str(destination)]=bone["index"]
                scene[rigid_pose.CACHE_KEY]=str(destination)
                if bone["source"]==destination:
                    for n in bone["old_frames"]:
                        if int(n)>plan["frame"]:
                            rigid_pose._io_path(destination.parent/("pose_"+n+".cbor")).unlink(missing_ok=True)
            for temporary,destination,row in staged:
                if row["object_type"] == "CURVE":
                    from .curve_output import commit_new_output, legacy_new_output
                    with legacy_new_output(row['obj'],row['key'],destination):
                        os.replace(temporary,destination)
                        pc2.load_curve_cache(row["key"])
                        pc2.ensure_curve_handler()
                        commit_new_output(row['obj'], row['key'], destination)
                else:
                    os.replace(temporary,destination)
                    row["mod"].filepath=str(destination)
            state=get_addon_data(scene).state
            prefix=[n for n in state.convert_fetched_frames_to_list() if n<=plan["frame"]]
            state.fetched_frame.clear()
            for n in prefix:state.add_fetched_frame(n)
            fence["local_retired"]=True
        finally:
            for temporary,_destination,_row in staged:
                temporary.unlink(missing_ok=True)

    def _do_fetch_frames(
        self, root: str, frame_count: int, fetched: list[int], only_latest: bool,
        collision_ticket=None, fetch_generation: int = -1,
    ) -> None:
        """Download the frame range. Guarantees a terminal event on every
        path (FetchComplete on success, FetchFailed on precondition miss
        or exception). Counter reset (_anim_total/_applied/_frames) now
        happens via DoResetAnimationBuffer, not here."""
        if collision_ticket is None:
            with self._collision_lock:
                collision_ticket = self._collision_ticket()
                if fetch_generation < 0:
                    fetch_generation = self._anim_map_generation
        backend, project = collision_ticket[1:]
        producer_sid = self._engine.state.session_id
        chunk_size = self._chunk_size
        def dispatch(event):
            self._dispatch_fetch_event(event, collision_ticket, fetch_generation)
        try:
            # AUTHORIZATION BARRIER: before any download or publication.
            if not self._fetch_job_current(collision_ticket, fetch_generation):
                return
            resume_bound = self._resume_bound(backend, project, root, collision_ticket, fetch_generation)
            if resume_bound is None:
                if not only_latest:
                    dispatch(FetchFailed(reason="RESUME_GENERATION_PENDING: native checkpoint restore is not initialized", fetch_generation=fetch_generation))
                return
            if collision_ticket is not None:
                if not self._collision_current(collision_ticket):
                    return
                # fetch_generation arrived WITH the ticket at submission; never
                # re-derive it from live state here.
                try:
                    response, alive = self._call_owned(backend.query, 
                        {}, project, chunk_size)
                    if not alive:
                        if only_latest:
                            # No download/publication without current validation.
                            # Keep the accepted cache and the same bounded backlog.
                            exhausted = False
                            with self._anim_lock:
                                record = self._live_backlog
                                if (record is not None and record["root"] == root
                                        and record["ticket"] == collision_ticket
                                        and record["generation"] == fetch_generation
                                        and record["fetched"] is fetched):
                                    record["validation_failures"] += 1
                                    record["validation_exhausted"] = (
                                        record["validation_failures"] >= 3)
                                    exhausted = record["validation_exhausted"]
                            if exhausted:
                                console.write(
                                    "自动取帧已暂停：服务器验证连续失败，请点击恢复实时跟随重试。")
                            return
                        # Stale job: silent. Current manual job: terminal failure.
                        if self._collision_current(collision_ticket):
                            dispatch(FetchFailed(
                                reason="fetch frames: server unavailable before validation",
                                fetch_generation=fetch_generation))
                        return
                    if not self._collision_validate(response, collision_ticket, force=True):
                        return          # retired: silent, no state change
                    if only_latest:
                        with self._anim_lock:
                            record = self._live_backlog
                            if (record is not None and record["root"] == root
                                    and record["ticket"] == collision_ticket
                                    and record["generation"] == fetch_generation
                                    and record["fetched"] is fetched):
                                record["validation_failures"] = 0
                                record["validation_exhausted"] = False
                except Exception as exc:
                    self._reject_fetch_current(exc, collision_ticket, fetch_generation)
                    return
            if not backend:
                if not only_latest:
                    dispatch(FetchFailed(reason="fetch frames: no backend"))
                return
            if not project:
                if not only_latest:
                    dispatch(FetchFailed(reason="fetch frames: no project"))
                return
            if not only_latest:
                # Always discover actual file count for Fetch All — the
                # status response "frame" field may lag behind the
                # actual output files.
                discovered = self._count_remote_frames(root, backend, project, chunk_size)
                if discovered > 0:
                    frame_count = discovered
            elif frame_count < 1:
                frame_count = self._count_remote_frames(root, backend, project, chunk_size)
            if resume_bound >= 0:
                frame_count = min(frame_count, resume_bound)
            if only_latest and frame_count > 0:
                # Status may report frame=0 while files have already landed.
                # Publish the discovered high-water to this SAME live job,
                # otherwise a bounded first window would hide later frames.
                with self._anim_lock:
                    record = self._live_backlog
                    if (record is not None and record["root"] == root
                            and record["ticket"] == collision_ticket
                            and record["generation"] == fetch_generation
                            and record["fetched"] is fetched):
                        record["target"] = max(record["target"], frame_count)
            if frame_count < 1:
                if not only_latest:
                    if collision_ticket is None or self._collision_current(collision_ticket):
                        dispatch(FetchComplete(
                            total_frames=0, fetch_generation=fetch_generation))
                return

            bundle = self._ensure_anim_map(root, collision_ticket, fetch_generation)
            if bundle is None:
                return
            display_pin_map, bone_bindings = bundle[2:]
            with self._collision_lock:
                if not self._fetch_job_current(collision_ticket, fetch_generation):
                    return
                with self._anim_lock:
                    statistics_manifest = self._anim_statistics_manifest
                    statistics_zero_fetched = self._anim_statistics_zero_fetched
            if statistics_manifest is None:
                manifest_path = _server_join(
                    backend,
                    root,
                    "session",
                    "output",
                    "statistics_manifest.cbor",
                )
                try:
                    statistics_manifest = self._call_owned(backend.receive_data, 
                        manifest_path,
                        project,
                        chunk_size=chunk_size,
                    )
                except Exception as exc:
                    # Sessions produced before timeline statistics remain
                    # fetchable. Their panel reports that a rerun is required.
                    console.write(
                        "statistics manifest unavailable; this session must be "
                        f"rerun for timeline statistics ({exc})"
                    )
                    statistics_manifest = b""
                with self._collision_lock:
                    if (not self._fetch_job_current(collision_ticket, fetch_generation)
                            or self._engine.state.session_id != producer_sid):
                        return
                    with self._anim_lock:
                        if self._fetched is not fetched:
                            if not only_latest:
                                if collision_ticket is None or self._collision_current(
                                        collision_ticket):
                                    dispatch(FetchFailed(
                                        reason="fetch frames: context reset while "
                                        "fetching statistics manifest",
                                        fetch_generation=fetch_generation))
                            return
                        self._anim_statistics_manifest = statistics_manifest

            if only_latest:
                # Do not discard intermediate solver frames after a scrub.
                # The queue is bounded in RAM; frames beyond capacity remain
                # on the backend and are fetched on later polls, in order.
                with self._collision_lock:
                    if (not self._fetch_job_current(collision_ticket, fetch_generation)
                            or self._engine.state.session_id != producer_sid):
                        return
                    with self._anim_lock:
                        if self._fetched is not fetched:
                            return
                        capacity = max(0, _LIVE_PENDING_FRAME_LIMIT - len(self._anim_frames))
                        first = 0 if bone_bindings else 1
                        record = self._live_backlog
                        if record is not None and record["fetched"] is fetched:
                            record["first"] = first
                        downloaded = set(fetched)
                from itertools import islice
                frames = list(islice((i for i in range(first, frame_count + 1)
                                      if i not in downloaded), capacity))
            else:
                frames = list(range(0 if bone_bindings else 1, frame_count + 1))

            def interrupt_cb():
                return self._interrupt.is_set()

            # Stale-context guard. ``self._fetched`` is reassigned (not
            # mutated) by ``clear_fetched_frames``, ``set_fetched_frames``,
            # ``_do_disconnect``, and ``DoResetAnimationBuffer``-adjacent
            # flows (the FetchRequested transition runs ``clear_fetched_frames``
            # on the caller side just before dispatching). Live-fetches
            # queued under the previous reference must not append into the
            # new context — otherwise frames they fetched leak into the
            # newly-cleared ``_anim_frames`` and into the new ``_fetched``,
            # making a subsequent full fetch under-count ``_anim_total``
            # while ``_anim_applied`` over-counts (the bl_chain_reconnect
            # macOS regression: ``(applied=9, total=7)``).
            if self._fetched is not fetched:
                if not only_latest:
                    if collision_ticket is None or self._collision_current(collision_ticket):
                        dispatch(FetchFailed(
                            reason="fetch frames: context reset before start",
                            fetch_generation=fetch_generation))
                return

            to_fetch = [i for i in frames if i not in fetched]
            if not only_latest:
                # The list check above and this counter write must be one atomic
                # step, otherwise a retirement can reset between check and write.
                with self._collision_lock:
                    if not self._fetch_job_current(collision_ticket, fetch_generation):
                        return
                    with self._anim_lock:
                        if self._fetched is not fetched:
                            return
                        self._anim_total = len(to_fetch)
                        self._anim_applied = 0

            # The fetch pipeline has two phases, each shown as its own
            # 0→100% bar in the UI (phase is disambiguated by the status
            # label): the download half runs while activity=FETCHING and
            # the apply half while activity=APPLYING. Progress is reset
            # to 0 at the FETCHING→APPLYING transition.
            to_fetch_total = max(1, len(to_fetch))

            fetched_count = 0
            for idx, i in enumerate(to_fetch):
                if interrupt_cb():
                    break

                def inner_progress(p, t, idx=idx):
                    frac = min(1.0, (idx + p) / to_fetch_total)
                    dispatch(
                        ProgressUpdated(progress=frac, traffic=t)
                    )

                filename = f"vert_{i}.bin"
                path = _server_join(backend, root, "session", "output", filename)
                data = self._call_owned(backend.receive_data, 
                    path, project, chunk_size=chunk_size,
                    progress_cb=self._owned_callback(inner_progress) if not only_latest else None,
                    interrupt_cb=self._owned_interrupt(interrupt_cb),
                )
                vert = numpy.frombuffer(data, dtype=numpy.float32).reshape(-1, 3)
                bone_packet = None
                if bone_bindings:
                    # Receive exactly the ORIGINAL native sidecars the sealed
                    # receipt requires, then publish one packet. An ordinary
                    # rigid scene fetches only the PDRD measurement and keeps the
                    # original rigid packet; a FEM or mixed scene additionally
                    # requires the FEM material measurement. A required sidecar
                    # that is absent fails the fetch here rather than falling
                    # back to a rigid pose or a guessed runtime body index, and
                    # a sidecar the receipt does not require is never fetched.
                    from .fem_pose import FEM, binding_kind, decode_native_frames
                    rigid_data = fem_data = None
                    kinds = {binding_kind(row) for row in bone_bindings["bindings"]}
                    for kind, prefix in (("PDRD", "rigid_pose"), (FEM, "fem_pose")):
                        if kind not in kinds:
                            continue
                        pose_path = _server_join(backend, root, "session", "output", f"{prefix}_{i}.cbor")
                        pose_data = self._call_owned(backend.receive_data, 
                            pose_path, project, chunk_size=chunk_size,
                            interrupt_cb=self._owned_interrupt(interrupt_cb),
                        )
                        if kind == "PDRD":
                            rigid_data = pose_data
                        else:
                            fem_data = pose_data
                    bone_packet = decode_native_frames(rigid_data, fem_data, data, i, bone_bindings)
                # The frame's display pins (see _decode_display_pin_map_cbor):
                # one byte per block, set while the block is still pinned,
                # then every scripted position as three float32s.
                display_pin_frame = None
                if display_pin_map:
                    n_blocks = len(display_pin_map["blocks"])
                    n_total = display_pin_map["n_total"]
                    dpin_path = _server_join(
                        backend, root, "session", "output",
                        f"display_pin_{i}.bin",
                    )
                    dpin_data = self._call_owned(backend.receive_data, 
                        dpin_path, project,
                        chunk_size=chunk_size,
                        interrupt_cb=self._owned_interrupt(interrupt_cb),
                    )
                    if len(dpin_data) != n_blocks + 12 * n_total:
                        raise ValueError(
                            f"display_pin_{i}.bin holds {len(dpin_data)} bytes, "
                            f"expected {n_blocks + 12 * n_total} for {n_blocks} "
                            f"blocks and {n_total} positions"
                        )
                    active = numpy.frombuffer(
                        dpin_data, dtype=numpy.uint8, count=n_blocks,
                    ).astype(bool)
                    positions = numpy.frombuffer(
                        dpin_data, dtype="<f4", offset=n_blocks,
                    ).reshape(-1, 3)
                    display_pin_frame = (active, positions)
                statistics_data = []
                includes_statistics_zero = False
                if statistics_manifest:
                    if not statistics_zero_fetched:
                        zero_path = _server_join(
                            backend,
                            root,
                            "session",
                            "output",
                            "statistics_0.cbor",
                        )
                        statistics_data.append(self._call_owned(backend.receive_data, 
                            zero_path,
                            project,
                            chunk_size=chunk_size,
                            interrupt_cb=self._owned_interrupt(interrupt_cb),
                        ))
                        includes_statistics_zero = True
                    statistics_path = _server_join(
                        backend,
                        root,
                        "session",
                        "output",
                        f"statistics_{i}.cbor",
                    )
                    statistics_data.append(self._call_owned(backend.receive_data, 
                        statistics_path,
                        project,
                        chunk_size=chunk_size,
                        interrupt_cb=self._owned_interrupt(interrupt_cb),
                    ))
                current_bound = self._resume_bound(backend, project, root, collision_ticket, fetch_generation)
                if current_bound is None or (current_bound >= 0 and i > current_bound):
                    if not only_latest:
                        dispatch(FetchFailed(reason="RESUME_GENERATION_PENDING: native frame generation changed", fetch_generation=fetch_generation))
                    return
                # Re-check the context inside the lock that gates every
                # ``_anim_frames`` mutation. ``clear_fetched_frames`` /
                # ``set_fetched_frames`` take the same lock, so once we
                # observe the reference still matches we know no reset
                # can land before the append completes.
                with self._collision_lock:
                    if (not self._fetch_job_current(collision_ticket, fetch_generation)
                            or self._engine.state.session_id != producer_sid):
                        return
                    with self._anim_lock:
                        if self._fetched is not fetched:
                            if only_latest:
                                return
                            dispatch(FetchFailed(
                                reason="fetch frames: context reset mid-fetch"))
                            return
                        self._anim_frames.append(
                            (i, vert, statistics_data, display_pin_frame, bone_packet,
                             producer_sid, self._anim_map_generation)
                        )
                        if includes_statistics_zero:
                            self._anim_statistics_zero_fetched = True
                            statistics_zero_fetched = True
                        self._fetched.append(i)
                fetched_count += 1
                if not only_latest:
                    dispatch(ProgressUpdated(
                        progress=(idx + 1) / to_fetch_total, traffic="",
                    ))

            if not only_latest:
                if collision_ticket is not None and not self._collision_current(
                        collision_ticket):
                    return
                dispatch(FetchComplete(
                    total_frames=fetched_count, fetch_generation=fetch_generation))
        except Exception as e:
            console.write(f"_do_fetch_frames failed: {e}")
            if not only_latest:
                dispatch(FetchFailed(reason=f"fetch frames: {e}"))
            else:
                # only_latest is a background poll during live sim — don't
                # tear down state, just log. A stuck state here isn't
                # possible because live sim doesn't transition to FETCHING.
                pass

    def _do_exec(self, command: str, shell: bool) -> None:
        backend = self._backend
        if not backend:
            return
        output = self._call_owned(backend.exec_command, command, shell=shell)
        with self._lifecycle_commit() as current:
            if not current:
                return
            with self._data_lock:
                self._exec_output = output
        # Logging is diagnostic, outside lifecycle locks. Retired backend logs
        # may still arrive; they cannot update Engine or the result cache.
        if not self._execution_current():
            return
        if output:
            for line in output.get("stdout", []):
                console.write(line)
            for line in output.get("stderr", []):
                console.write(line)
            if not output.get("stdout") and not output.get("stderr"):
                console.write(f"Command completed with exit code: {output['exit_code']}")
        self._engine.dispatch(ExecComplete(output=output))

    def _do_terminate(self) -> None:
        backend, project, chunk_size = self._backend, self._project_name, self._chunk_size
        if not backend or not project:
            return
        self._call_owned(backend.query, {"request": "terminate"}, project, chunk_size)
        for _ in range(20):
            response, alive = self._call_owned(backend.query, {}, project, chunk_size)
            if not is_sim_running_from_response(response):
                break
            time.sleep(0.25)
        with self._lifecycle_commit() as current:
            if not current:
                return
            self._response_cache.record(response)
        self._engine.dispatch(ServerPolled(response=response))

    def _do_save_and_quit(self) -> None:
        if not self._backend or not self._project_name:
            return
        self._call_owned(self._backend.query, {"request": "save_and_quit"}, self._project_name, self._chunk_size)

    # -- UI helpers --

    @staticmethod
    def _redraw() -> None:
        """Tag Blender viewport areas for redraw."""
        try:
            import bpy  # pyright: ignore
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == "VIEW_3D":
                        area.tag_redraw()
        except Exception:
            pass
