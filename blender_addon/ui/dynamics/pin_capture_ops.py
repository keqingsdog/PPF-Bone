from ...core.pin_reference import decode_stored_pin_identifier, PinReferenceError
# Fork modification: changed from the upstream baseline.
# File: pin_capture_ops.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Capture Deformation for SHELL/SOLID/ROD pins. Walks the scene frame
# range on a modal timer, samples the depsgraph-evaluated mesh per
# frame, extracts only the pinned vertices, and stores them as a
# ``_pindeform.pc2`` cache. The encoder's pin path consumes this cache
# in preference to the manual Make-Keyframe vertex-co fcurves
# (implicit PC2-wins).
#
# Structurally mirrors ``static_deform_ops.py``; the differences are:
#   - per-pin (not per-object) granularity, so the entry carries the
#     vg_name alongside the object UUID and the pin item index;
#   - vertex storage is the pinned subset, not the full mesh;
#   - finalize sets ``pin_item.has_captured_anim = True`` and ensures
#     the EMBEDDED_MOVE sentinel is on the pin's operations list so
#     the encoder treats this pin as animated.

from ...i18n import props as _ppf_props
from ... import i18n as _ppf_i18n
import time

import bpy  # pyright: ignore
import numpy as np

from bpy.types import Operator  # pyright: ignore
from ...i18n import pgettext_iface as iface_, pgettext_tip as tip_

from ...core.pc2 import (
    display_only_modifier_names,
    has_pin_anim_pc2,
    remove_pin_anim_pc2,
    write_pin_anim_pc2,
)
from ...core.transform import zup_to_yup
from ...core.utils import (
    eval_deform_local_positions,
    has_deforming_modifier_stack,
    redraw_all_areas,
)
from .utils import get_group_from_index


# ---------------------------------------------------------------------------
# Job state (module-level singleton, mirrors the static-deform pattern)
# ---------------------------------------------------------------------------

_capture_job: dict = {
    "active": False,
        "token": None,
        "owner": None,
    "aborted": False,
    # One entry per pin captured in this run. Each carries:
    # obj_uuid, obj_name, vg_name, group_index, pin_index,
    # pin_indices (snapshot at job start), frame_start, frame_end,
    # range_source, frames (ndarray), frames_done.
    "entries": [],
    "entry_cursor": 0,
    "total_frames_processed": 0,
    "total_frames": 0,
    "saved_frame": 1,
    "origin_scene": None,
    "origin_pointer": None,
    "origin_filepath": None,
    "saved_subframe": 0.0,
    "status": "",
    "error": "",
}


def _reset_job() -> None:
    _capture_job.update({
        "active": False,
        "token": None,
        "owner": None,
        "aborted": False,
        "entries": [],
        "entry_cursor": 0,
        "total_frames_processed": 0,
        "total_frames": 0,
        "saved_frame": 1,
        "origin_sampling_targets": None,
        "origin_scene": None,
        "origin_pointer": None,
        "origin_filepath": None,
        "saved_subframe": 0.0,
        "status": "",
        "error": "",
    })


def is_pin_capture_running() -> bool:
    """True while a Capture Pin Deformation job is in flight."""
    return bool(_capture_job.get("active"))


def pin_capture_progress_snapshot() -> tuple[int, int, str, int]:
    """(done_frames, total_frames, status_line, pin_count) for the UI."""
    return (
        int(_capture_job.get("total_frames_processed", 0)),
        int(_capture_job.get("total_frames", 0)),
        str(_capture_job.get("status", "")),
        len(_capture_job.get("entries", [])),
    )


# ---------------------------------------------------------------------------
# Per-frame sampling: extract pinned-vertex positions in solver world space
# ---------------------------------------------------------------------------


def _sample_pin_frame_world(eval_obj, co) -> np.ndarray:
    """Sample one frame's world-space positions for the pin's vertices.

    Returns ``(n_pin_verts, 3)`` float32 in solver world space
    (``zup_to_yup @ eval_obj.matrix_world @ co_local``). Translation is
    included so the captured frame-0 row matches the cloth mesh's
    initial-upload position exactly; the encoder builds MoveBy deltas
    between consecutive frames, so the absolute frame is irrelevant
    beyond that alignment.
    """
    co = np.asarray(co, dtype=np.float64)
    mw = np.array(eval_obj.matrix_world, dtype=np.float64).reshape(4, 4)
    z2y = np.array(zup_to_yup(), dtype=np.float64).reshape(4, 4)
    m = z2y @ mw
    homog = np.concatenate(
        [co, np.ones((co.shape[0], 1), dtype=np.float64)], axis=1,
    )
    world = (homog @ m.T)[:, :3]
    return world.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Frame range discovery: reuse the static-deform helper unchanged, since
# the question "what frames could change the evaluated mesh of *this*
# object" is the same for STATIC colliders and for cloth meshes whose
# pins ride along with bone deformation.
# ---------------------------------------------------------------------------


def _effective_frame_range(scene, obj) -> tuple[int, int, str]:
    from .static_deform_ops import _effective_frame_range as _sd_range
    return _sd_range(scene, obj)


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------


def _build_entry(scene, group_index, pin_index, error_collector: list, *, allocate=True):
    """Validate a single pin and return a ready-to-capture entry."""
    from ...core.encoder.pin import _get_pin_indices
    from ...core.uuid_registry import plan_pin_reference, get_object_uuid

    group = get_group_from_index(scene, group_index)
    if group is None or group.object_type == "STATIC":
        error_collector.append(iface_("Pin capture: invalid or STATIC group"))
        return None
    if not (0 <= pin_index < len(group.pin_vertex_groups)):
        error_collector.append(iface_("Pin capture: pin index out of range"))
        return None
    pin_item = group.pin_vertex_groups[pin_index]
    try:
        plan = plan_pin_reference(pin_item)
    except ValueError as exc:
        error_collector.append(str(exc))
        return None
    if plan is None or plan[0].type != "MESH":
        error_collector.append(iface_("Pin capture: only mesh pins can be captured"))
        return None
    obj, _, vg_name, _ = plan
    if not vg_name:
        error_collector.append(iface_("Pin capture: pin identifier is missing the vertex group"))
        return None
    if any(op.op_type == "TORQUE" for op in pin_item.operations):
        error_collector.append(
            iface_(
                "Pin '{pin}' on '{object}': torque cannot be combined with "
                "captured embedded animation"
            ).format(pin=vg_name, object=obj.name)
        )
        return None
    pin_indices = _get_pin_indices(obj, vg_name)
    if not pin_indices:
        error_collector.append(
            iface_(
                "Pin '{pin}' on '{object}': no pinned vertices to capture"
            ).format(pin=vg_name, object=obj.name)
        )
        return None
    uid = get_object_uuid(obj)
    if not uid:
        error_collector.append(
            iface_(
                "'{name}' has no UUID (library-linked?); skipped"
            ).format(name=obj.name)
        )
        return None
    if sum(1 for row in group.assigned_objects if row.uuid == uid) != 1:
        error_collector.append("Pin capture owner membership is missing or ambiguous")
        return None
    frame_start, frame_end, range_source = _effective_frame_range(scene, obj)
    n_frames = frame_end - frame_start + 1
    if n_frames <= 0:
        error_collector.append(
            iface_(
                "Pin '{pin}' on '{object}': empty frame range "
                "(start={start}, end={end})"
            ).format(pin=vg_name, object=obj.name, start=frame_start, end=frame_end)
        )
        return None
    return {
        "obj_uuid": uid,
        "obj_name": obj.name,
        "obj_pointer": obj.as_pointer(),
        "vg_name": vg_name,
        "pin_raw": pin_item.name,
        "group_index": group_index,
        "pin_index": pin_index,
        "group_pointer": group.as_pointer(),
        "pin_pointer": pin_item.as_pointer(),
        "pin_indices": np.asarray(pin_indices, dtype=np.int64),
        "object_type": group.object_type,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "range_source": range_source,
        "frames": np.empty((n_frames, len(pin_indices), 3), dtype=np.float32) if allocate else None,
        "frames_done": 0,
    }


def _resolve_capture_pin(scene, entry):
    """Resolve the captured author item, never whichever item occupies its index."""
    from ...core.uuid_registry import plan_pin_reference
    from ...core.encoder.pin import _get_pin_indices
    group = get_group_from_index(scene, entry["group_index"])
    if group is None or group.as_pointer() != entry["group_pointer"]:
        raise RuntimeError("Captured Pin group changed; capture cancelled")
    for item in group.pin_vertex_groups:
        if item.as_pointer() != entry["pin_pointer"]:
            continue
        if item.name != entry["pin_raw"] or item.object_uuid != entry["obj_uuid"]:
            break
        plan = plan_pin_reference(item)
        if (plan is not None and plan[0].as_pointer() == entry["obj_pointer"]
                and plan[0].name == entry["obj_name"] and plan[2] == entry["vg_name"]
                and sum(1 for row in group.assigned_objects if row.uuid == entry["obj_uuid"]) == 1
                and np.array_equal(entry["pin_indices"], _get_pin_indices(plan[0], plan[2]))):
            return item
        break
    raise RuntimeError("Captured Pin was removed or retargeted; capture cancelled")


def _process_one_frame(scene, depsgraph, entry, frame: int) -> tuple[bool, str]:
    """Capture one frame of one pin's deformed positions into ``entry``."""
    from ...core.uuid_registry import get_object_by_uuid

    obj = get_object_by_uuid(entry["obj_uuid"])
    if obj is None:
        return False, iface_("'{name}' disappeared during capture").format(name=entry['obj_name'])
    n_verts_total = len(obj.data.vertices)
    pin_idx = entry["pin_indices"]
    if int(pin_idx.max(initial=-1)) >= n_verts_total:
        return False, (
            iface_(
                "Pin '{pin}' on '{object}': vertex count dropped below "
                "pin's max index at frame {frame} (have {count} verts, "
                "need >{max_index}). A topology-changing modifier moved "
                "above the deformer mid-capture, or the mesh was edited; "
                "re-bind the pin and retry."
            ).format(
                pin=entry['vg_name'],
                object=entry['obj_name'],
                frame=frame,
                count=n_verts_total,
                max_index=int(pin_idx.max()),
            )
        )
    # The SAME evaluation the encoder ships the object at (display-only
    # modifiers hidden, a count-changing stack cut where the output cache
    # sits), so row 0 of the cache is the shipped pose by construction.
    # Indexing a mesh whose count a modifier changed would read other
    # vertices, or none at all.
    local = eval_deform_local_positions(
        obj, bpy.context,
        exclude_modifier_names=display_only_modifier_names(
            obj, entry["object_type"],
        ),
    )
    if local is None:
        return False, (
            iface_(
                "Pin '{pin}' on '{object}': a modifier changes the vertex "
                "count at frame {frame} in front of the cache, where only a "
                "modifier the add-on does not count as topology-changing can "
                "sit (a Fluid domain, an Ocean in Generate mode, or a Mesh "
                "Sequence Cache of different topology), so the pinned "
                "vertices cannot be read. Remove or disable it and capture "
                "again."
            ).format(pin=entry['vg_name'], object=entry['obj_name'], frame=frame)
        )
    world = _sample_pin_frame_world(
        obj.evaluated_get(depsgraph), local[pin_idx],
    )
    if world.shape != (len(pin_idx), 3):
        return False, (
            iface_(
                "Pin '{pin}' on '{object}': unexpected sample shape "
                "{shape} at frame {frame}"
            ).format(pin=entry['vg_name'], object=entry['obj_name'], shape=world.shape, frame=frame)
        )
    entry["frames"][entry["frames_done"]] = world
    entry["frames_done"] += 1
    return True, ""


def _start_job(context, entries: list) -> tuple[bool, str]:
    if _capture_job["active"]:
        return False, iface_("Another Capture Deformation job is in progress")
    # Refuse to start if a STATIC capture is in flight: the two share
    # the depsgraph + frame_set side effects and would corrupt each
    # other's running frame cursor.
    from .static_deform_ops import is_capture_running as _static_running
    if _static_running():
        return False, iface_("Static-collider capture is in progress; wait for it to finish")
    if not entries:
        return False, iface_("No eligible pins to capture")
    scene = context.scene
    total = sum(e["frames"].shape[0] for e in entries)
    _capture_job.update({
        "active": True,
        "token": object(),
        "owner": None,
        "aborted": False,
        "entries": entries,
        "entry_cursor": 0,
        "total_frames_processed": 0,
        "total_frames": total,
        "saved_frame": int(scene.frame_current),
        "status": iface_("Capturing pin..."),
        "error": "",
    })
    from ...core.capture_origin import remember_origin
    try:
        remember_origin(_capture_job, scene)
    except (ReferenceError, RuntimeError, ValueError) as exc:
        _reset_job()
        return False, str(exc)
    return True, ""


def _tick_job(context, *, budget_ms: int = 40) -> bool:
    from ...core.capture_origin import require_origin, sampling_cache_scope, sampling_targets
    require_origin(_capture_job, context)
    from ...core.capture_origin import require_capture_ranges
    require_capture_ranges(context.scene, _capture_job["entries"])
    token = _capture_job.get("token")
    with sampling_cache_scope(sampling_targets(_capture_job, _capture_job["entries"])):
        more = _tick_sampling(context, budget_ms=budget_ms, token=token)
    require_capture_ranges(context.scene, _capture_job["entries"])
    return more


def _tick_sampling(context, *, budget_ms: int = 40, token=None) -> bool:
    """Capture frames until the per-tick budget expires."""
    from ...core.capture_origin import require_origin
    require_origin(_capture_job, context)
    from ...core.capture_origin import require_capture_token
    require_capture_token(_capture_job, token)
    deadline = time.perf_counter() + budget_ms / 1000.0
    entries = _capture_job["entries"]
    scene = context.scene
    while time.perf_counter() < deadline:
        if _capture_job["aborted"]:
            return False
        idx = _capture_job["entry_cursor"]
        if idx >= len(entries):
            return False
        entry = entries[idx]
        _resolve_capture_pin(scene, entry)
        if entry["frames_done"] >= entry["frames"].shape[0]:
            _capture_job["entry_cursor"] += 1
            continue
        frame = entry["frame_start"] + entry["frames_done"]
        scene.frame_set(frame)
        require_capture_token(_capture_job, token)
        # Frame handlers can change the scene, any row or its object owner.
        require_origin(_capture_job, context)
        from ...core.capture_origin import require_capture_ranges
        require_capture_ranges(scene, entries)
        for candidate in entries:
            _resolve_capture_pin(scene, candidate)
        # frame_set re-evaluates cache visibility F-curves, including those
        # on dependency objects. Hide the whole batch again after that update,
        # without muting or rewriting any author's animation channels.
        from ...core.capture_origin import sampling_cache_scope, sampling_targets
        with sampling_cache_scope(sampling_targets(_capture_job, entries)):
            depsgraph = context.evaluated_depsgraph_get()
            require_capture_token(_capture_job, token)
            require_origin(_capture_job, context)
            for candidate in entries:
                _resolve_capture_pin(scene, candidate)
            ok, reason = _process_one_frame(scene, depsgraph, entry, frame)
            require_capture_token(_capture_job, token)
        if not ok:
            _capture_job["error"] = reason
            _capture_job["aborted"] = True
            return False
        _capture_job["total_frames_processed"] += 1
        _capture_job["status"] = (
            iface_(
                "Capturing pin '{pin}' on '{object}' "
                "[{done}/{total}] ({source})"
            ).format(
                pin=entry['vg_name'],
                object=entry['obj_name'],
                done=entry['frames_done'],
                total=entry['frames'].shape[0],
                source=entry['range_source'],
            )
        )
    return _capture_job["entry_cursor"] < len(entries)


def _finalize_job(context) -> tuple[int, int]:
    """Write completed entries to PC2 and restore the saved frame."""
    from ...core.capture_origin import require_origin
    require_origin(_capture_job, context)
    # A depsgraph callback in the final sample may replace a participant
    # after the per-frame guard. Recheck the whole frozen batch before writes.
    from ...core.capture_origin import sampling_targets
    sampling_targets(_capture_job, _capture_job["entries"])
    from ...core.capture_origin import require_capture_ranges
    require_capture_ranges(context.scene, _capture_job["entries"])
    from ...core.uuid_registry import get_object_by_uuid
    from .pin_ops import _ensure_embedded_move_op

    # Validate every target before writing any completed cache.
    targets = [_resolve_capture_pin(context.scene, entry)
               for entry in _capture_job["entries"]]
    # A PC2 column is bound to the original vertex index, not just the VG
    # name/count. Validate every completed target before any cache is written.
    # Do this once at commit, rather than scanning full meshes each frame.
    from ...core.encoder.pin import _get_pin_indices
    for entry, pin_item in zip(_capture_job["entries"], targets):
        if entry["frames_done"] < entry["frames"].shape[0]:
            continue
        # Manual author keys may have been added while the modal passed UI
        # events through. Keep those keys; never silently override them by PC2.
        if any(op.op_type == "TORQUE" for op in pin_item.operations):
            raise RuntimeError("Pin acquired torque during capture; captured embedded animation is incompatible")
        if _pin_has_vertex_co_fcurves(pin_item):
            raise RuntimeError("Pin acquired manual keyframes during capture; capture cancelled")
        obj = get_object_by_uuid(entry["obj_uuid"])
        live_indices = _get_pin_indices(obj, entry["vg_name"]) if obj else []
        if not np.array_equal(entry["pin_indices"], live_indices):
            raise RuntimeError("Captured Pin vertex membership changed; capture cancelled")
    pins_written = 0
    frames_written = 0
    for entry, pin_item in zip(_capture_job["entries"], targets):
        if entry["frames_done"] < entry["frames"].shape[0]:
            continue
        require_origin(_capture_job, context)
        for candidate in _capture_job["entries"]:
            _resolve_capture_pin(context.scene, candidate)
        obj = get_object_by_uuid(entry["obj_uuid"])
        if obj is None:
            continue
        from ...core.capture_provenance import make_proof
        write_pin_anim_pc2(obj, entry["vg_name"], entry["frames"], proof=make_proof(obj, "PIN", entry), scene=context.scene)
        pin_item.has_captured_anim = True
        _ensure_embedded_move_op(pin_item)
        pins_written += 1
        frames_written += entry["frames"].shape[0]
    return pins_written, frames_written


def _restore_frame(context) -> None:
    from ...core.capture_origin import restore_frame
    restore_frame(_capture_job)


def _cleanup_after_job(context) -> None:
    """Restore the saved frame and reset the job singleton (the non-timer half of the modal
    teardown, shared with the all-pins coordinator)."""
    from ...core.uuid_registry import get_object_by_uuid

    token = _capture_job.get("token")
    try:
        _restore_frame(context)
    finally:
        if _capture_job.get("token") is token:
            _reset_job()


# ---------------------------------------------------------------------------
# Modal capture operator
# ---------------------------------------------------------------------------


class OBJECT_OT_CapturePinDeformation(Operator):
    """Sample the pin's vertices from the depsgraph each frame and
    store them as a captured-animation cache the solver consumes in
    place of manual keyframes"""

    bl_idname = "object.capture_pin_deformation"
    bl_label = "Capture Deformation"
    bl_options = {"UNDO"}

    group_index: _ppf_props.IntProperty(options={'HIDDEN'})  # pyright: ignore
    pin_index: _ppf_props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    @classmethod
    def poll(cls, context):
        return not _capture_job["active"]

    def execute(self, context):
        # Refuse if the pin already carries manual vertex-co fcurves.
        # The user should clear them first so the captured cache owns
        # the EMBEDDED_MOVE sentinel without ambiguity, and so the
        # encoder doesn't have to choose between two animation sources
        # at transfer time.
        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is not None and 0 <= self.pin_index < len(group.pin_vertex_groups):
            pin_item = group.pin_vertex_groups[self.pin_index]
            try:
                manual_has = _pin_has_vertex_co_fcurves(pin_item)
            except PinReferenceError as exc:
                self.report({"ERROR"}, str(exc))
                return {"CANCELLED"}
            if manual_has:
                self.report(
                    {"WARNING"},
                    iface_(
                        "Pin has manual keyframes (vertex-co fcurves); press "
                        "Delete All Keyframes first, then re-run Capture "
                        "Deformation"
                    ),
                )
                return {"CANCELLED"}
            # Refuse on non-deformable objects up front so EXEC_DEFAULT
            # (panel button, MCP, Python API) never starts a modal job
            # that would just bake the rest pose. The panel greys the
            # button out via the same predicate; this is the same check
            # for callers that bypass the panel.
            from ...core.uuid_registry import resolve_pin as _rp
            try:
                _pin_obj = _rp(pin_item)
            except ValueError:
                _pin_obj = None
            if _pin_obj is None or _pin_obj.type != "MESH" or not pin_object_supports_capture(_pin_obj):
                self.report(
                    {"WARNING"},
                    iface_(
                        "Pin object has no enabled deformer or animated shape keys "
                        "(Armature, Lattice, Mesh Deform, Shape Keys, ...); "
                        "Capture Deformation has nothing to record"
                    ),
                )
                return {"CANCELLED"}

        errors: list = []
        entry = _build_entry(scene, self.group_index, self.pin_index, errors)
        if entry is None:
            self.report({"WARNING"}, errors[0] if errors else iface_("Nothing to capture"))
            return {"CANCELLED"}
        ok, err = _start_job(context, [entry])
        if not ok:
            self.report({"WARNING"}, err)
            return {"CANCELLED"}
        self._capture_token = _capture_job["token"]
        _capture_job["owner"] = self
        wm = context.window_manager
        self._capture_wm = wm
        try:
            self._timer = wm.event_timer_add(0.01, window=context.window)
            wm.modal_handler_add(self)
        except Exception as exc:
            self._teardown(context)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        redraw_all_areas(context)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        from ...core.capture_origin import owns_capture_job, detach_capture_timer
        if not owns_capture_job(_capture_job, getattr(self, '_capture_token', None)):
            detach_capture_timer(self, context)
            return {"CANCELLED"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        # Any exception inside the tick must still run _teardown, or the
        # temporary capture state leaks and the job flag
        # stays set forever (worse than a visible failure).
        try:
            if _capture_job["aborted"]:
                err = _capture_job.get("error", "")
                self._teardown(context)
                redraw_all_areas(context)
                if err:
                    self.report({"ERROR"}, err)
                else:
                    self.report({"INFO"}, iface_("Pin capture aborted"))
                return {"CANCELLED"}
            more = _tick_job(context)
            redraw_all_areas(context)
            if not more and not _capture_job["aborted"]:
                n_pins, n_frames = _finalize_job(context)
                self._teardown(context)
                redraw_all_areas(context)
                from ...models.groups import invalidate_overlays
                invalidate_overlays()
                self.report(
                    {"INFO"},
                    iface_("Captured {frames} frame(s) across {pins} pin(s)").format(
                        frames=n_frames, pins=n_pins
                    ),
                )
                return {"FINISHED"}
            return {"RUNNING_MODAL"}
        except Exception as exc:  # noqa: BLE001 — must restore state
            self._teardown(context)
            redraw_all_areas(context)
            self.report({"ERROR"}, iface_("Pin capture failed: {error}").format(error=exc))
            return {"CANCELLED"}

    def cancel(self, context):
        self._teardown(context)
        redraw_all_areas(context)

    def _teardown(self, context):
        from ...core.capture_origin import owns_capture_job, detach_capture_timer
        detach_capture_timer(self, context)
        if owns_capture_job(_capture_job, getattr(self, '_capture_token', None)):
            _cleanup_after_job(context)


# ---------------------------------------------------------------------------
# Clear operator: drop the cache and (when no manual fcurves exist) the
# EMBEDDED_MOVE sentinel.
# ---------------------------------------------------------------------------


def _pin_has_vertex_co_fcurves(pin_item) -> bool:
    """True if the pin's object has any vertex-co fcurves on its mesh action.

    Used to decide whether Clear should also drop the EMBEDDED_MOVE
    sentinel. If the manual Make-Keyframe path authored fcurves, the
    sentinel still belongs to it.
    """
    from ...core.encoder.pin import _collect_pin_vertex_fcurve_frames
    from ...core.uuid_registry import plan_pin_reference
    plan = plan_pin_reference(pin_item)
    if plan is None:
        raise PinReferenceError("Pin owner is unavailable; manual animation state is unproven")
    obj, _, vg_name, _ = plan
    if obj.type != "MESH":
        return False
    if not vg_name:
        return False
    frames, _ = _collect_pin_vertex_fcurve_frames(obj, vg_name)
    return bool(frames)


class OBJECT_OT_ClearPinDeformation(Operator):
    """Delete this pin's captured-deformation cache. If the pin has no
    manual keyframes, also remove the embedded-move sentinel."""

    bl_idname = "object.clear_pin_deformation"
    bl_label = "Clear Deformation Cache"
    bl_options = {"UNDO"}

    group_index: _ppf_props.IntProperty(options={'HIDDEN'})  # pyright: ignore
    pin_index: _ppf_props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    @classmethod
    def poll(cls, context):
        return not _capture_job["active"]

    def execute(self, context):
        from ...core.uuid_registry import resolve_pin
        from ...models.groups import decode_vertex_group_identifier
        from .pin_ops import _remove_embedded_move_ops

        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            self.report({"ERROR"}, iface_("Group not found"))
            return {"CANCELLED"}
        if not (0 <= self.pin_index < len(group.pin_vertex_groups)):
            self.report({"ERROR"}, iface_("Pin index out of range"))
            return {"CANCELLED"}
        pin_item = group.pin_vertex_groups[self.pin_index]
        try:
            obj = resolve_pin(pin_item)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, iface_("Pin object not found"))
            return {"CANCELLED"}
        _, vg_name = decode_stored_pin_identifier(pin_item.name)
        if not vg_name:
            self.report({"ERROR"}, iface_("Invalid pin identifier"))
            return {"CANCELLED"}
        manual_has = _pin_has_vertex_co_fcurves(pin_item)
        if not remove_pin_anim_pc2(obj, vg_name, scene, allow_missing=True):
            self.report({"WARNING"}, iface_("Captured deformation was not cleared; cache ownership or file removal refused"))
            return {"CANCELLED"}
        pin_item.has_captured_anim = False
        # Drop the sentinel only if the manual Make Keyframe path
        # didn't author its own fcurves. Otherwise it still owns the
        # EMBEDDED_MOVE slot.
        if not manual_has:
            _remove_embedded_move_ops(pin_item)
        redraw_all_areas(context)
        from ...models.groups import invalidate_overlays
        invalidate_overlays()
        self.report(
            {"INFO"},
            iface_("Cleared captured deformation for pin '{pin}' on '{object}'").format(
                pin=vg_name, object=obj.name
            ),
        )
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Abort button (mirrors the static-deform abort)
# ---------------------------------------------------------------------------


class SOLVER_OT_PinCaptureAbort(Operator):
    """Abort the running Capture Pin Deformation job"""

    bl_idname = "solver.pin_capture_abort"
    bl_label = "Abort"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return bool(_capture_job.get("active")) and not _capture_job.get("aborted")

    def execute(self, context):
        _capture_job["aborted"] = True
        _capture_job["status"] = iface_("Aborting...")
        redraw_all_areas(context)
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Public predicates the panel uses to gate the row
# ---------------------------------------------------------------------------


def pin_object_supports_capture(obj) -> bool:
    """Cheap capture gate: enabled deformers or animated/driven shape keys.

    Never sample the depsgraph or move the timeline from a panel predicate.
    Keep this pin-specific instead of changing engine modifier classification.
    """
    from ...core.utils import static_mesh_deforms
    return static_mesh_deforms(obj, None, allow_eval=False)


def pin_has_captured_anim(pin_item) -> bool:
    """True if this pin's PropertyGroup flag is set OR a PC2 cache
    exists on disk. The flag is the cheap source of truth; the disk
    fallback catches cases where the flag desynced (rare; the
    load_post reconciler handles the common case)."""
    if pin_item is None:
        return False
    if bool(getattr(pin_item, "has_captured_anim", False)):
        return True
    try:
        from ...core.uuid_registry import resolve_pin
        from ...models.groups import decode_vertex_group_identifier
        obj = resolve_pin(pin_item)
        if obj is None or obj.type != "MESH":
            return False
        _, vg_name = decode_stored_pin_identifier(pin_item.name)
        if not vg_name:
            return False
        return has_pin_anim_pc2(obj, vg_name)
    except Exception:
        return False


def pin_captured_frame_count(pin_item) -> int:
    """Frame count of this pin's captured cache, or 0 if absent."""
    from ...core.pc2 import get_pin_anim_cache
    from ...core.uuid_registry import resolve_pin
    from ...models.groups import decode_vertex_group_identifier

    if pin_item is None:
        return 0
    try:
        obj = resolve_pin(pin_item)
    except ValueError:
        return 0
    if obj is None or obj.type != "MESH":
        return 0
    _, vg_name = decode_stored_pin_identifier(pin_item.name)
    if not vg_name:
        return 0
    cache = get_pin_anim_cache(obj, vg_name)
    if cache is None:
        return 0
    return int(cache.shape[0])


# ---------------------------------------------------------------------------
# Refresh full-pin coverage (gates the rest-pose tracking toggle)
# ---------------------------------------------------------------------------


class OBJECT_OT_RefreshFullPinState(Operator):
    """Check whether this pin's vertex group covers every vertex of the mesh
    (a full pin) and cache the result on the pin. "Track Rest-Pose
    Deformation" needs a full pin, so its toggle stays disabled until this
    confirms one. The check is an O(N) vertex scan, so it runs only on click,
    not on every panel redraw. Run it again after editing the vertex group to
    update the cached state."""

    bl_idname = "object.refresh_full_pin_state"
    bl_label = "Refresh"
    bl_options = {"REGISTER", "UNDO"}

    group_index: _ppf_props.IntProperty(options={'HIDDEN'})  # pyright: ignore
    pin_index: _ppf_props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    def execute(self, context):
        from ...core.utils import pin_covers_all_vertices
        from ...core.uuid_registry import resolve_pin
        from ...models.groups import decode_vertex_group_identifier

        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            self.report({"ERROR"}, iface_("Group not found"))
            return {"CANCELLED"}
        if not (0 <= self.pin_index < len(group.pin_vertex_groups)):
            self.report({"ERROR"}, iface_("Pin index out of range"))
            return {"CANCELLED"}
        pin_item = group.pin_vertex_groups[self.pin_index]
        try:
            obj = resolve_pin(pin_item)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        _, vg_name = decode_stored_pin_identifier(pin_item.name)
        is_full = bool(
            obj is not None
            and obj.type == "MESH"
            and pin_covers_all_vertices(obj, vg_name)
        )
        pin_item.full_pin_checked = True
        pin_item.full_pin_cached = is_full
        redraw_all_areas(context)
        self.report(
            {"INFO"},
            iface_("Full pin: rest-pose tracking available")
            if is_full
            else iface_("Partial pin: rest-pose tracking unavailable"),
        )
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# "Re-capture all" support: enumerate every capturable pin across the active
# non-STATIC groups and drive the shared job from the coordinator in
# solver.py (which owns its own timer). The job machinery already processes
# a list of entries, so "all pins" is just a larger entry set.
# ---------------------------------------------------------------------------


def collect_capturable_pins(context, *, cheap: bool = False) -> list:
    """``(group_index, pin_index)`` for every pin in active non-STATIC groups
    whose object can be captured.

    Mirrors the per-pin button gate: a mesh pin whose object has a deforming
    modifier stack, no TORQUE op, and (when ``cheap`` is False) no manual
    vertex-co fcurves. ``cheap=True`` skips the fcurve scan so the result is
    light enough for a panel poll; ``cheap=False`` is the accurate set used
    when actually starting the job.
    """
    from ...core.uuid_registry import plan_pin_reference
    from ...models.groups import N_MAX_GROUPS, get_addon_data

    data = get_addon_data(context.scene)
    specs: list = []
    for gi in range(N_MAX_GROUPS):
        group = getattr(data, f"object_group_{gi}", None)
        if not group or not group.active or group.object_type == "STATIC":
            continue
        for pi, pin_item in enumerate(group.pin_vertex_groups):
            try:
                plan = plan_pin_reference(pin_item)
                if plan is None:
                    raise PinReferenceError("Pin capture owner is unavailable")
                obj = plan[0]
                if sum(1 for row in group.assigned_objects if row.uuid == pin_item.object_uuid) != 1:
                    raise PinReferenceError("Pin capture owner membership is missing or ambiguous")
            except ValueError:
                if cheap:
                    continue
                raise
            if obj.type != "MESH":
                continue
            if not pin_object_supports_capture(obj):
                continue
            if any(op.op_type == "TORQUE" for op in pin_item.operations):
                continue
            if not cheap and _pin_has_vertex_co_fcurves(pin_item):
                continue
            specs.append((gi, pi))
    return specs


def validate_pin_capture_specs(context, specs):
    """Prove all queued rows without allocating samples or author writes."""
    errors = []
    for gi, pi in specs:
        if _build_entry(context.scene, gi, pi, errors, allocate=False) is None:
            raise PinReferenceError(errors[0] if errors else "Pin capture row is unproven")


def start_capture_for_pins(context, specs: list, *, strict=False) -> tuple[bool, str]:
    """Build entries for each ``(group_index, pin_index)`` in *specs* and
    start the shared pin-capture job. Returns ``(ok, error_message)``."""
    errors: list = []
    entries = []
    for gi, pi in specs:
        entry = _build_entry(context.scene, gi, pi, errors)
        if entry is not None:
            entries.append(entry)
        elif strict:
            return False, (errors[0] if errors else iface_("Queued Pin could not be captured"))
    if not entries:
        return False, (errors[0] if errors else iface_("No eligible pins to capture"))
    return _start_job(context, entries)


def advance_pin_capture(context) -> tuple[bool, bool, str]:
    """Tick the running pin capture once; returns ``(more, aborted, error)``."""
    if _capture_job.get("aborted"):
        return False, True, str(_capture_job.get("error", ""))
    more = _tick_job(context)
    if _capture_job.get("aborted"):
        return False, True, str(_capture_job.get("error", ""))
    return more, False, ""


def finalize_pin_capture(context) -> tuple[int, int]:
    """Write completed pin entries to PC2; returns ``(pins, frames)`` written."""
    return _finalize_job(context)


def cleanup_pin_capture(context) -> None:
    """Restore frame and clear the job state."""
    _cleanup_after_job(context)


def collect_pins_with_captured_anim(context) -> list:
    """``(group_index, pin_index)`` for every pin in active non-STATIC groups
    that currently holds a captured-deformation cache. Used by "Clear All
    Deformations" (and its poll)."""
    from ...models.groups import N_MAX_GROUPS, get_addon_data

    data = get_addon_data(context.scene)
    specs: list = []
    for gi in range(N_MAX_GROUPS):
        group = getattr(data, f"object_group_{gi}", None)
        if not group or not group.active or group.object_type == "STATIC":
            continue
        for pi, pin_item in enumerate(group.pin_vertex_groups):
            if pin_has_captured_anim(pin_item):
                specs.append((gi, pi))
    return specs


def clear_captured_pin(context, group_index: int, pin_index: int, *, attempted_keys=None) -> bool:
    """Drop one pin's captured-deformation cache, clear its
    ``has_captured_anim`` flag, and (when the pin carries no manual vertex-co
    fcurves) remove the EMBEDDED_MOVE sentinel. Returns True if a pin was
    cleared. Mirrors ``OBJECT_OT_ClearPinDeformation.execute`` for the
    batch "Clear All Deformations" path."""
    from ...core.uuid_registry import resolve_pin
    from ...models.groups import decode_vertex_group_identifier
    from .pin_ops import _remove_embedded_move_ops

    group = get_group_from_index(context.scene, group_index)
    if group is None or not (0 <= pin_index < len(group.pin_vertex_groups)):
        return False
    pin_item = group.pin_vertex_groups[pin_index]
    if attempted_keys is not None and pin_item.object_uuid:
        # Even a deleted/ambiguous target leaves an author Pin entry. Protect
        # its recorded caches before resolution can refuse, so the orphan pass
        # cannot bypass this wrapper and leave captured=True with missing data.
        from ...core.pc2 import _capture_records
        attempted_keys.update(key for key, uid in _capture_records(context.scene).items()
                              if uid == pin_item.object_uuid)
    try:
        decode_stored_pin_identifier(pin_item.name)
        obj = resolve_pin(pin_item)
    except ValueError:
        return False
    if obj is None or obj.type != "MESH":
        return False
    _, vg_name = decode_stored_pin_identifier(pin_item.name)
    if not vg_name:
        return False
    if attempted_keys is not None:
        from ...core.pc2 import pin_anim_pc2_key
        attempted_keys.add(pin_anim_pc2_key(obj, vg_name))
    manual_has = _pin_has_vertex_co_fcurves(pin_item)
    if not remove_pin_anim_pc2(obj, vg_name, context.scene, allow_missing=True):
        return False
    pin_item.has_captured_anim = False
    if not manual_has:
        _remove_embedded_move_ops(pin_item)
    return True


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


classes = (
    OBJECT_OT_CapturePinDeformation,
    OBJECT_OT_ClearPinDeformation,
    OBJECT_OT_RefreshFullPinState,
    SOLVER_OT_PinCaptureAbort,
)


def register():
    for cls in classes:
        _ppf_i18n.register_class(cls)


def unregister():
    from ...core.capture_origin import cancel_capture_jobs
    cancel_capture_jobs()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
