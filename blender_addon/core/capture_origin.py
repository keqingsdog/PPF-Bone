"""Origin guards for frame-sampling jobs; never redirect a job to a new Scene."""
import bpy

from .frame_time import set_scene_frame


def remember_origin(job, scene):
    from .uuid_registry import get_object_by_uuid
    entries = job.get('objects', job.get('entries', ()))
    targets = freeze_sampling_targets(
        scene, [get_object_by_uuid(entry['obj_uuid']) for entry in entries])
    job.update(origin_sampling_targets=targets,
               origin_scene=scene, origin_pointer=scene.as_pointer(),
               origin_filepath=bpy.data.filepath,
               saved_subframe=float(scene.frame_subframe))


def require_origin(job, context):
    try:
        scene = job.get('origin_scene')
        valid = (scene is not None and scene.as_pointer() == job.get('origin_pointer')
                 and context.scene == scene
                 and bpy.data.filepath == job.get('origin_filepath'))
    except (ReferenceError, RuntimeError):
        valid = False
    if not valid:
        raise RuntimeError('Capture source Scene or project changed; capture cancelled')


def freeze_capture_ranges(scene, objects):
    """Freeze effective sampling bounds, not unrelated timeline/profile fields."""
    from .uuid_registry import get_object_uuid
    from ..ui.dynamics.static_deform_ops import _effective_frame_range
    result = []
    for obj in objects:
        start, end, _ = _effective_frame_range(scene, obj)
        result.append({'obj_uuid': get_object_uuid(obj),
                       'frame_start': start, 'frame_end': end})
    return tuple(result)


def require_capture_ranges(scene, entries):
    """Refuse stale buffers before publication; never undo authored settings."""
    from .uuid_registry import get_object_by_uuid
    from ..ui.dynamics.static_deform_ops import _effective_frame_range
    ranges = {}
    for entry in entries:
        uid = entry['obj_uuid']
        if uid not in ranges:
            obj = get_object_by_uuid(uid)
            if obj is None:
                raise RuntimeError('Capture range target is no longer available; capture cancelled')
            ranges[uid] = _effective_frame_range(scene, obj)
        current = ranges[uid][:2]
        if 'pin_pointer' in entry:
            from ..ui.dynamics.pin_capture_ops import _resolve_capture_pin, _effective_pin_frame_range
            pin = _resolve_capture_pin(scene, entry, identity_only=True)[0]
            obj = get_object_by_uuid(uid)
            current = _effective_pin_frame_range(scene, obj, pin, base_range=ranges[uid])[:2]
        if current != (entry['frame_start'], entry['frame_end']):
            raise RuntimeError('Effective capture frame range changed; capture cancelled')


def restore_frame(job):
    """Restore the original surviving Scene, never the newly active Scene."""
    scene = job.get('origin_scene')
    try:
        if (scene is not None and scene.as_pointer() == job.get('origin_pointer')
                and any(item == scene for item in bpy.data.scenes)):
            set_scene_frame(scene, int(job['saved_frame']),
                            float(job.get('saved_subframe', 0.0)))
    except (ReferenceError, RuntimeError):
        pass  # The originating Scene may have been removed or the file reloaded.


def abort_before_save(job):
    """Do not serialize transient frame/cache visibility during modal capture.

    Leave the active job marked aborted so its owning modal/coordinator removes
    its own timer on the next event rather than continuing to a queued phase.
    """
    if not job.get('active'):
        return
    owner = job.get('owner')
    if getattr(owner, '_combined_capture_owner', False):
        owner.abort_for_save()
        return
    if owner is not None:
        owner._capture_cancelled = True
        if hasattr(owner, '_queue'):
            owner._queue = []
    job['aborted'] = True
    job['error'] = 'Capture cancelled before saving the project'
    restore_frame(job)


from contextlib import contextmanager


def freeze_sampling_targets(scene, objects):
    """Remember only this RecaptureAll queue's object/data identities."""
    from .uuid_registry import get_object_uuid
    targets = []
    seen = {}
    for obj in objects:
        uid = get_object_uuid(obj) if obj is not None else None
        if not uid or obj.name not in scene.objects:
            raise RuntimeError('Queued capture object no longer belongs to the source Scene')
        if uid in seen:
            if seen[uid] != obj.as_pointer():
                raise RuntimeError('Queued capture objects share an ambiguous UUID')
            continue
        seen[uid] = obj.as_pointer()
        targets.append({'obj_uuid': uid, 'obj_pointer': obj.as_pointer(),
                        'data_pointer': obj.data.as_pointer() if obj.data else 0})
    return tuple(targets)


def sampling_targets(job, fallback):
    """Keep both phases hidden, including targets of an already finished phase."""
    owner = job.get('owner')
    targets = getattr(owner, '_capture_sampling_targets', None)
    if targets is None:
        targets = job.get('origin_sampling_targets')
    if targets is None:
        return fallback
    from .uuid_registry import get_object_by_uuid
    scene = job.get('origin_scene')
    for entry in targets:
        obj = get_object_by_uuid(entry['obj_uuid'])
        if (obj is None or obj.as_pointer() != entry['obj_pointer']
                or (obj.data.as_pointer() if obj.data else 0) != entry['data_pointer']
                or scene is None or obj.name not in scene.objects):
            raise RuntimeError('Queued capture object identity changed; capture cancelled')
    return targets


@contextmanager
def sampling_cache_scope(entries):
    """Hide all captured caches only while synchronously evaluating a tick.

    Restore before yielding to author UI events. Keeping the whole target set
    hidden preserves dependencies between captured objects during evaluation.
    """
    from .pc2 import suspend_mesh_cache_display, resume_mesh_cache_display
    from .uuid_registry import get_object_by_uuid
    saved = []
    seen = set()
    try:
        for entry in entries:
            uid = entry['obj_uuid']
            if uid in seen:
                continue
            seen.add(uid)
            obj = get_object_by_uuid(uid)
            saved.append((obj, suspend_mesh_cache_display(obj)))
        yield
    finally:
        errors = []
        for obj, prior in reversed(saved):
            try:
                resume_mesh_cache_display(obj, prior)
            except ReferenceError:
                pass  # The original object was removed by a callback.
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]


def owns_capture_job(job, token):
    return token is not None and job.get('token') is token


def detach_capture_timer(operator, context=None):
    timer = getattr(operator, '_timer', None)
    operator._timer = None
    if timer is None:
        return
    wm = getattr(operator, '_capture_wm', None)
    if wm is None and context is not None:
        wm = context.window_manager
    try:
        if wm is not None:
            wm.event_timer_remove(timer)
    except (ReferenceError, RuntimeError):
        pass  # The old file's window manager may already be disposed.


def cancel_capture_jobs(*, restore=True):
    """Invalidate modal owners before teardown; Undo must not re-evaluate RNA."""
    from ..ui.dynamics import static_deform_ops, pin_capture_ops
    from ..ui.dynamics.bone_capture import cancel_active
    cancel_active(restore=restore)
    from .encoder.captured_bone_motion import clear
    clear()
    errors = []
    combined_seen = set()
    for module in (static_deform_ops, pin_capture_ops):
        job = module._capture_job
        token = job.get('token')
        owner = job.get('owner')
        if owner is not None:
            owner._capture_cancelled = True
            if hasattr(owner, '_queue'):
                owner._queue = []
        try:
            if getattr(owner, '_combined_capture_owner', False):
                if id(owner) not in combined_seen:
                    combined_seen.add(id(owner))
                    if restore:
                        owner._restore_once()
                    else:
                        owner.restored = True
                    operator = getattr(owner, '_timer_owner', None)
                    if operator is not None:
                        operator._end_timer(None, {"CANCELLED"})
                        operator._cleanup_active(None)
            elif job.get('active') and restore:
                module._restore_frame(None)
        except Exception as exc:
            errors.append(exc)
        finally:
            if job.get('token') is token:
                module._reset_job()
            if owner is not None:
                owner._capture_sampling_targets = None
                detach_capture_timer(owner)
    if errors:
        raise errors[0]


def require_capture_token(job, token):
    if not owns_capture_job(job, token):
        raise RuntimeError('Capture job was cancelled or replaced')
