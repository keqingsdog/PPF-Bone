# Fork modification: changed from the upstream baseline.
# File: async_op.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import time

from bpy.types import Operator  # pyright: ignore
from ..i18n import pgettext_iface as iface_, pgettext_tip as tip_

from . import encode_progress
from .utils import get_timer_wait_time, redraw_all_areas


_INVALID_CONNECTION_ORIGIN = object()


def _connection_origin(context, communicator, *, include_remote_root=True):
    """In-memory comparison only; never report endpoint/SSH/authentication values.

    The active fields mirror connection_ops.execute; inactive profile fields
    cannot retarget the selected connection. Backend+epoch additionally catches
    reconnects to the same address, before a new Connected event is delivered.
    """
    from ..models.groups import get_addon_data, has_addon_data
    selected = None
    if has_addon_data(context.scene):
        props = get_addon_data(context.scene).ssh_state
        kind = props.server_type
        fields = ['docker_port', 'native_device']
        native_paths = {'WIN_NATIVE': 'win_native_path', 'MAC_NATIVE': 'mac_native_path',
                        'LINUX_NATIVE': 'linux_native_path'}
        if kind in native_paths:
            from .utils import resolve_local_path
            import os
            root = resolve_local_path(getattr(props, native_paths[kind]))
            selected = (kind, os.path.normcase(os.path.normpath(root)) if root else root)
            if kind != 'MAC_NATIVE':
                fields.append('native_gpu_backend')
        else:
            selected = (kind,)
            fields.append('native_gpu_backend')
            if kind in ('COMMAND', 'DOCKER_SSH_COMMAND'):
                # Compare opaque command text, without parsing/logging tokens.
                fields.append('command')
            elif kind in ('CUSTOM', 'DOCKER_SSH'):
                fields.extend(('host', 'port', 'username', 'key_path', 'proxy_jump'))
            if 'DOCKER' in kind:
                fields.extend(('container', 'docker_path'))
            else:
                fields.append('ssh_remote_path')
        selected += tuple(getattr(props, name) for name in fields)
    runner = getattr(communicator, '_runner', None)
    if runner is not None:
        with runner._connect_lock:
            backend, epoch = runner.backend, runner._connect_epoch
            project = runner.project_name
            endpoint = None if backend is None else (
                backend, backend.backend_type, backend.current_directory,
                backend.server_port, getattr(backend, 'container', ''),
                getattr(backend, '_device', None), getattr(backend, '_gpu_backend', None))
    else:
        # Lightweight communicator adapters retain their object identity too.
        info = getattr(communicator, 'connection', None)
        endpoint = None if info is None else (
            getattr(info, 'instance', None), getattr(info, 'type', None),
            getattr(info, 'current_directory', None), getattr(info, 'server_port', None),
            getattr(info, 'container', None))
        epoch = None
        project = None
    remote_root = (getattr(getattr(communicator, 'connection', None), 'remote_root', None)
                   if include_remote_root else None)
    return (communicator, selected, endpoint, epoch, project, remote_root,
            getattr(communicator, 'session_id', None))


def _origin_cancel_message(operator):
    requested = getattr(operator, '_remote_request_started', None)
    if requested is True:
        return "Scene, project or connection changed; local waiting ended. A remote request was initiated; no stop request was sent. Return to the original connection and scene to check status or stop it."
    if requested is False:
        return "Scene, project or connection changed before submission; local operation cancelled. No remote request was submitted."
    return "Scene, project or connection changed; local operation ended."


def _cancel_origin(operator, context):
    operator._end_stages()
    operator.cleanup_modal(context)
    operator.report({'WARNING'}, iface_(_origin_cancel_message(operator)))
    redraw_all_areas(context)
    return {'CANCELLED'}


class StageAbort(Exception):
    """Raised by a staged-work callback to cancel the run with a user-facing
    message. :meth:`AsyncOperator.modal` catches it, reports it through
    ``self.report`` at ``level`` (default ``ERROR``), and ends the modal."""

    def __init__(self, message: str, level: str = "ERROR"):
        super().__init__(message)
        self.level = level


STAGE_PENDING = object()


class AsyncOperator(Operator):
    """Base class for modal operators with timer-based polling.

    Subclasses must implement:
        is_complete() -> bool: Return True when the operation is done.

    Optionally override:
        on_complete(context): Called when is_complete() returns True.
        on_timeout(context): Called when timeout is exceeded.
        timeout: float = 60.0: Maximum seconds before timeout.

    Optionally drive blocking pre-work with :meth:`start_stages`: a list of
    ``(label, fn)`` stages is run one per timer tick (so the panel paints a
    labeled progress bar between them) before the modal falls through to the
    ``is_complete()`` wait. Used by Transfer / Run to surface the otherwise
    silent scene-encode and drift-check phases.
    """

    _timer = None
    _start_time: float = 0.0
    timeout: float = 60.0
    auto_redraw: bool = False
    _stages = None
    _stage_index: int = 0

    def is_complete(self) -> bool:
        raise NotImplementedError

    def is_cancelled(self) -> bool:
        return False

    def on_complete(self, context):
        pass

    def on_timeout(self, context):
        self.report({"ERROR"}, iface_("Operation timed out"))

    def setup_modal(self, context):
        if getattr(self, '_stage_origin_scene', None) is None:
            self._capture_origin(context)
        self._start_time = time.time()
        self._timer = context.window_manager.event_timer_add(
            getattr(self, '_stage_timer_interval', None) or get_timer_wait_time(),
            window=context.window
        )
        context.window_manager.modal_handler_add(self)

    def start_stages(self, context, stages):
        """Begin a staged modal: run each ``(label, fn)`` in *stages* one per
        timer tick, publishing an ``encode_progress`` snapshot between them so
        the panel paints a labeled bar from the moment the operator returns
        ``RUNNING_MODAL``. ``fn(context)`` does one chunk of blocking work and
        may raise :class:`StageAbort` to cancel with a message. The final
        stage typically dispatches the async request; once every stage has
        run, the modal proceeds to the normal ``is_complete()`` wait.

        Call this from ``execute`` instead of ``setup_modal`` and return
        ``{"RUNNING_MODAL"}``.
        """
        # A later TIMER receives the *current* window context. Never encode,
        # upload, clear animation, or complete against another scene/project.
        self._capture_origin(context)
        self._remote_request_started = False
        self._stages = list(stages)
        self._stage_index = 0
        first_label = self._stages[0][0] if self._stages else ""
        encode_progress.begin(len(self._stages), first_label)
        # Paint the bar with the first stage's label now, during the ~one-tick
        # gap before the first timer fires, so that label is on screen while
        # the first (blocking) stage runs rather than appearing only after it.
        redraw_all_areas(context)
        self.setup_modal(context)

    def _capture_origin(self, context):
        import bpy
        self._stage_origin_scene = context.scene
        self._stage_origin_path = bpy.data.filepath
        try:
            from ..models.groups import get_addon_data, has_addon_data
            self._stage_origin_project = (get_addon_data(context.scene).state.project_name
                                          if has_addon_data(context.scene) else None)
        except (AttributeError, ReferenceError):
            self._stage_origin_project = None
        source = getattr(self, '_stage_communicator', None)
        if source is not None:
            try:
                self._stage_origin_connection = _connection_origin(
                    context, source, include_remote_root=getattr(self, '_origin_tracks_remote_root', True))
                runner = getattr(source, '_runner', None)
                self._stage_connection_ticket = (runner.capture_connection_ticket()
                                                 if runner is not None else None)
            except (AttributeError, ReferenceError, RuntimeError, ValueError):
                self._stage_origin_connection = _INVALID_CONNECTION_ORIGIN
                self._stage_connection_ticket = None

    def mark_remote_requested(self):
        """The communicator accepted local queuing; this is not a server ack."""
        self._remote_request_started = True

    def _origin_changed(self, context):
        origin = getattr(self, "_stage_origin_scene", None)
        if origin is None:
            return False
        import bpy
        try:
            if context.scene != origin or bpy.data.filepath != self._stage_origin_path:
                return True
            project = getattr(self, "_stage_origin_project", None)
            if project is not None:
                from ..models.groups import get_addon_data, has_addon_data
                if not has_addon_data(origin) or get_addon_data(origin).state.project_name != project:
                    return True
            if getattr(self, '_stage_communicator', None) is not None:
                frozen = getattr(self, '_stage_origin_connection', _INVALID_CONNECTION_ORIGIN)
                if frozen is _INVALID_CONNECTION_ORIGIN or _connection_origin(context, self._stage_communicator,
                        include_remote_root=getattr(self, '_origin_tracks_remote_root', True)) != frozen:
                    return True
            return False
        except (AttributeError, ReferenceError, RuntimeError, ValueError):
            return True

    def _end_stages(self):
        # Only clear the shared snapshot when THIS operator owns an active
        # staged run, so a non-staging operator's teardown can't wipe a bar
        # another operator is publishing.
        if self._stages is not None:
            encode_progress.end()
        self._stages = None
        self._stage_index = 0

    def _run_stage_tick(self, context):
        """Run the current stage. Returns a modal result dict to return now,
        or ``None`` when every stage has finished (fall through to the
        completion poll)."""
        if self._origin_changed(context):
            return _cancel_origin(self, context)
        _label, fn = self._stages[self._stage_index]
        try:
            pending = fn(context)
        except StageAbort as exc:
            self._end_stages()
            self.cleanup_modal(context)
            self.report({exc.level}, str(exc))
            redraw_all_areas(context)
            return {"CANCELLED"}
        except Exception as exc:  # noqa: BLE001 - surface any encode failure
            self._end_stages()
            self.cleanup_modal(context)
            # Record it on the connection as well, not only in the status bar.
            # ``self.report`` writes a line that the next redraw clears, so an
            # unclassified encode failure would otherwise leave the panel
            # looking healthy while nothing had been uploaded. The stages that
            # recognize their own refusals already do this; doing it here too
            # means every encode failure is visible in the same place.
            from .client import communicator
            communicator.set_error(str(exc))
            self.report({"ERROR"}, iface_("Encoding failed: {error}").format(error=exc))
            redraw_all_areas(context)
            return {"CANCELLED"}
        if self._origin_changed(context):
            return _cancel_origin(self, context)
        if pending is STAGE_PENDING:
            redraw_all_areas(context)
            return {"PASS_THROUGH"}
        self._stage_index += 1
        if self._stage_index >= len(self._stages):
            # All stages done; the dispatching stage (if any) has set the
            # server status, so let the completion poll take over this tick.
            self._end_stages()
            return None
        # Publish the next (now-current) stage so the upcoming redraw shows
        # its label and an advanced bar before it blocks the main thread.
        encode_progress.update(
            self._stage_index, self._stages[self._stage_index][0]
        )
        redraw_all_areas(context)
        return {"PASS_THROUGH"}

    def cleanup_modal(self, context):
        if self._timer:
            try:
                context.window_manager.event_timer_remove(self._timer)
            except (RuntimeError, ReferenceError):
                pass
            self._timer = None

    def cancel(self, context):
        """Called by Blender when the operator class is torn down (e.g.
        addon reload). Drop the event timer so it can't fire against a
        freed handler, and clear any in-flight encode-progress bar."""
        self._end_stages()
        self.cleanup_modal(context)

    def _is_stale_class(self):
        """Detect an instance whose class has been replaced by a module
        reload: ``bpy.types`` now points at a different class object
        bearing the same ``bl_idname``. Touching ``self`` properties on
        that instance risks reading a freed operator type."""
        import bpy  # pyright: ignore
        bl_idname = getattr(self.__class__, "bl_idname", "")
        if not bl_idname:
            return False
        # bl_idname like "ssh.run_command" → bpy.types.SSH_OT_run_command
        try:
            area, name = bl_idname.split(".", 1)
        except ValueError:
            return False
        registry_name = f"{area.upper()}_OT_{name}"
        current = getattr(bpy.types, registry_name, None)
        return current is not None and current is not self.__class__

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        # If the addon was reloaded while this modal was alive, our
        # class object is no longer the one registered under our
        # bl_idname. Bail out before touching attributes that would
        # read from a freed operator type.
        if self._is_stale_class():
            self._end_stages()
            self.cleanup_modal(context)
            redraw_all_areas(context)
            return {"CANCELLED"}
        if self._origin_changed(context):
            return _cancel_origin(self, context)
        if self.auto_redraw:
            redraw_all_areas(context)
        if self.is_cancelled():
            self._end_stages()
            self.cleanup_modal(context)
            redraw_all_areas(context)
            return {"CANCELLED"}
        # Run any staged pre-work one stage per tick, before the completion
        # poll, so the panel paints the labeled encode bar between stages.
        if self._stages is not None:
            staged = self._run_stage_tick(context)
            if staged is not None:
                return staged
        if time.time() - self._start_time > self.timeout:
            self.cleanup_modal(context)
            self.on_timeout(context)
            redraw_all_areas(context)
            return {"CANCELLED"}
        if self.is_complete():
            if self._origin_changed(context):
                return _cancel_origin(self, context)
            self.cleanup_modal(context)
            self.on_complete(context)
            return {"FINISHED"}
        return {"PASS_THROUGH"}
