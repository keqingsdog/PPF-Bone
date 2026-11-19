"""Main-thread ownership of the live-simulation playhead.

External frame changes (including a timeline scrub) suspend automatic follow.
The streamed frames still enter the cache; F3 -> 恢复实时跟随
explicitly gives the playhead back to the live preview.
"""

from .. import i18n as _ppf_i18n
from contextlib import contextmanager
import threading

import bpy  # pyright: ignore
from bpy.types import Operator  # pyright: ignore

_scene_pointer = None
_epoch = None
_active = False
_following = True
_own_frame_change = 0


def assert_main_thread():
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Blender scene/cache application must run on the main thread")


def reset():
    global _scene_pointer, _epoch, _active, _following, _own_frame_change
    _scene_pointer = None
    _epoch = None
    _active = False
    _following = True
    _own_frame_change = 0


def sync_epoch(epoch):
    """Discard a paused playhead when a fetch or upload is replaced."""
    if _active and epoch != _epoch:
        reset()


def begin(scene, epoch=None):
    """Activate follow for this scene's current fetch/run session."""
    global _scene_pointer, _epoch, _active, _following
    pointer = scene.as_pointer()
    if not _active or pointer != _scene_pointer or epoch != _epoch:
        _scene_pointer = pointer
        _epoch = epoch
        _active = True
        _following = True


def is_following(scene):
    return not _active or scene.as_pointer() != _scene_pointer or _following


def is_paused(scene):
    return _active and scene.as_pointer() == _scene_pointer and not _following


def pause(scene, epoch):
    """Give the playhead to the artist until explicit Resume or a new run."""
    global _following
    begin(scene, epoch)
    _following = False


@contextmanager
def own_frame_change():
    global _own_frame_change
    assert_main_thread()
    _own_frame_change += 1
    try:
        yield
    finally:
        _own_frame_change -= 1


@bpy.app.handlers.persistent
def _observe_frame_change(scene, _depsgraph=None):
    global _following
    if _active and scene.as_pointer() == _scene_pointer and not _own_frame_change:
        _following = False


@bpy.app.handlers.persistent
def _reset_on_load(*_args):
    reset()


class PPF_OT_ResumeLiveFollow(Operator):
    bl_idname = "ppf.resume_live_follow"
    bl_label = "恢复实时跟随"
    bl_description = "重新跟随最新回传的模拟帧"

    @classmethod
    def poll(cls, context):
        from .facade import runner
        return context.scene is not None and (
            is_paused(context.scene) or runner.live_fetch_paused(context.scene))

    def execute(self, context):
        global _following
        assert_main_thread()
        from .facade import runner
        from .playback import stop_for_edit, TimelineBusyError
        try:
            stop_for_edit(context)
        except TimelineBusyError as exc:
            self.report({'WARNING'}, str(exc))
            return {"CANCELLED"}
        begin(context.scene, runner.animation_epoch())
        _following = True
        # Once the queue has drained there may be no next packet to move the
        # playhead. Resume must also display the latest already cached frame.
        from ..models.groups import get_addon_data
        from .encoder import resolve_start_frame
        state = get_addon_data(context.scene).state
        frames = state.convert_fetched_frames_to_list()
        if frames:
            with own_frame_change():
                context.scene.frame_set(resolve_start_frame(state) + max(frames))
        runner.retry_live_backlog()
        from .frame_pump import ensure_modal_running
        ensure_modal_running()
        return {"FINISHED"}


def register():
    reset()
    for handlers, handler in (
        (bpy.app.handlers.frame_change_post, _observe_frame_change),
        (bpy.app.handlers.load_post, _reset_on_load),
    ):
        for old in list(handlers):
            if getattr(old, "__name__", "") == handler.__name__:
                handlers.remove(old)
        handlers.append(handler)
    _ppf_i18n.register_class(PPF_OT_ResumeLiveFollow)


def unregister():
    reset()
    for handlers, name in (
        (bpy.app.handlers.frame_change_post, "_observe_frame_change"),
        (bpy.app.handlers.load_post, "_reset_on_load"),
    ):
        for handler in list(handlers):
            if getattr(handler, "__name__", "") == name:
                handlers.remove(handler)
    try:
        bpy.utils.unregister_class(PPF_OT_ResumeLiveFollow)
    except RuntimeError:
        pass
