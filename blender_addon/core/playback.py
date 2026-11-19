"""Playback controls must distinguish a native timeline drag from playback.

Blender 5.1's is_animation_playing includes scrubbing, across all windows.
Calling animation_cancel with only a scrub active dereferences a missing
playback timer in Blender. Inspect every window before invoking that operator.
"""

import bpy


class TimelineBusyError(RuntimeError):
    pass


def is_scrubbing(context):
    manager = getattr(context, 'window_manager', None)
    if manager is None:
        return False
    for window in manager.windows:
        try:
            if window.screen and window.screen.is_scrubbing:
                return True
        except ReferenceError:
            # A window is being closed. Do not invoke screen operators against it.
            return True
    return False


def is_playing(context):
    screen = getattr(context, 'screen', None)
    return bool(screen and screen.is_animation_playing and not is_scrubbing(context))


def stop_for_edit(context):
    """Reject a drag before callers clear caches or change the playhead."""
    if is_scrubbing(context):
        raise TimelineBusyError('请先松开时间轴，结束拖帧后再执行此操作。')
    if is_playing(context):
        bpy.ops.screen.animation_cancel(restore_frame=False)
