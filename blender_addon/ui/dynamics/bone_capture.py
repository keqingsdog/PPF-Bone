"""Modal authored bone survey. No persistent or Transfer-consumable cache."""
import time

import bpy

from ...core import capture_origin as origin
from ...core.encoder import resolve_solver_fps, resolve_start_frame
from ...core.encoder import bones
from ...core.rigid_pose import has_bone_pose_cache, _detached_display_pending
from ...models.bone_bindings import bone_frame_cache
from ...models.groups import get_addon_data
from . import capture_frame_batch as guard


def has_bones(context):
    return any(o.type == 'ARMATURE' and any(
        b.ppf_bone_physics.enabled for b in o.data.bones
    ) for o in context.scene.objects)


def _stamp(context):
    s = context.scene
    state = get_addon_data(s).state
    return (resolve_start_frame(state), int(state.frame_count), resolve_solver_fps(state),
            float(state.world_scaling), float(s.unit_settings.scale_length))


def _require_idle(context):
    if ((context.screen is not None and context.screen.is_animation_playing)
            or has_bone_pose_cache(context.scene)
            or _detached_display_pending(context.scene)):
        raise RuntimeError("Stop playback and clear displayed solver bone animation before bone capture")


class BoneSampling:
    """Explicit per-frame demand from potentially different author and solver windows."""
    def __init__(self, context):
        _require_idle(context)
        self.job = {'saved_frame': int(context.scene.frame_current)}
        origin.remember_origin(self.job, context.scene)
        self.stamp = _stamp(context)
        self.start, self.count, self.fps, *_ = self.stamp
        if self.count <= 0:
            raise ValueError("Bone capture requires at least one solver frame")
        # Freeze the author scale reference at the original playhead/subframe,
        # separately from the document's reference at the solver start frame.
        self.author_start = int(context.scene.frame_current)
        with bone_frame_cache():
            self.author_sources = tuple(bones._pose_scale_sources(context))
        self.author_next = self.author_start if self.author_sources else None
        self.solver_next = self.start
        self.solver_plan = None
        self.actor_sources = tuple((o, o.data, o.as_pointer()) for o in context.scene.objects
                                   if o.type == 'ARMATURE')
        self.processed = self.elapsed_sampling = 0
        self.total = self.count
        if self.author_sources:
            self.total += min(self.count, abs(self.author_start - self.start))
        self.document = None

    @property
    def pending(self):
        return min((i for i in (self.author_next, self.solver_next) if i is not None), default=None)

    def require_valid(self, context):
        origin.require_origin(self.job, context)
        _require_idle(context)
        if _stamp(context) != self.stamp:
            raise RuntimeError("Bone capture timing or unit settings changed")
        for arm, data, pointer in self.actor_sources:
            if (arm.as_pointer() != pointer or arm.data != data
                    or context.scene.objects.get(arm.name) != arm):
                raise RuntimeError("Bone capture armature identity changed")

    def sample(self, context, f):
        if self.pending != f:
            return False
        self.require_valid(context)
        begin = time.perf_counter()
        if self.author_next == f:
            with bone_frame_cache():
                bones._validate_pose_scales_at_frame(self.author_sources, f)
            self.author_next = f + 1 if f + 1 < self.author_start + self.count else None
        if self.solver_next == f:
            if f == self.start:
                self.solver_plan = bones.prepare_transport_bones(
                    context, get_addon_data(context.scene).state, self.fps, self.start)
                if self.solver_plan is None:
                    raise RuntimeError("No enabled bone transport bindings found")
                self._has_track = any(self.solver_plan[2:])
            if self._has_track:
                _, _, scale_sources, anchors, roots = self.solver_plan
                bones.sample_transport_bone_frame(
                    scale_sources, anchors.values(), roots, f - self.start, self.fps, f)
            self.solver_next = f + 1 if f + 1 < self.start + self.count else None
            if self.solver_next is None:
                self.document = bones.finish_transport_bones(self.solver_plan)
        self.processed += 1
        self.elapsed_sampling += time.perf_counter() - begin
        self.require_valid(context)
        return True


class StandaloneBoneCapture:
    def __init__(self, context):
        from . import pin_capture_ops as pin, static_deform_ops as static
        if is_active() or pin.is_pin_capture_running() or static.is_capture_running():
            raise RuntimeError("Another deformation capture is in progress")
        self.bones = BoneSampling(context)
        self.context = context
        self.scene = context.scene
        self.layer = context.view_layer.as_pointer()
        self.world = guard._world_stamp(context.scene)
        self.callbacks = guard._callback_stamp()
        self.seeded = None
        self.closed = False
        self.frame_sets = 0
        self.started = time.perf_counter()

    def _require(self, context):
        if self.closed or self.layer != context.view_layer.as_pointer():
            raise RuntimeError("Bone capture view layer changed")
        self.bones.require_valid(context)
        if self.world != guard._world_stamp(context.scene) or not guard._same_callbacks(self.callbacks):
            raise RuntimeError("Bone capture frame callbacks or rigid body world changed")
        if self.seeded is not None and (context.scene.frame_current != self.seeded
                                       or context.scene.frame_subframe):
            raise RuntimeError("Timeline changed during bone capture")

    def tick(self, context, *, budget_ms=35):
        try:
            deadline = time.perf_counter() + budget_ms / 1000
            self._require(context)
            while (frame := self.bones.pending) is not None:
                self.scene.frame_set(frame)
                self.frame_sets += 1
                self.seeded = frame
                context.view_layer.update()
                self._require(context)
                self.bones.sample(context, frame)
                if time.perf_counter() >= deadline:
                    break
            return self.bones.pending is not None
        except BaseException:
            self.close()
            raise

    def close(self, *, restore=True):
        if self.closed:
            return
        self.closed = True
        if restore:
            origin.restore_frame(self.bones.job)


_active = None


def is_active():
    return _active is not None and not _active._job.closed


def standalone_progress(context):
    if not is_active() or _active._job.scene != context.scene:
        return None
    job = _active._job
    return job.bones.processed, job.bones.total, job.seeded


def cancel_active(*, restore=True):
    if _active is not None:
        _active._finish(None, {'CANCELLED'}, restore=restore)


class SOLVER_OT_CaptureBoneMotion(bpy.types.Operator):
    """Sample and validate authored bone inputs without writing solver caches."""
    bl_idname = "solver.capture_bone_motion"
    bl_label = "Capture Bone Motion (Preview)"

    @classmethod
    def poll(cls, context):
        from . import static_deform_ops as static, pin_capture_ops as pin
        if not context.window or context.mode != "OBJECT" or is_active() or static.is_capture_running() or pin.is_pin_capture_running():
            return False
        from ..solver import com, is_bake_running, is_running
        return (not is_bake_running() and not com.busy() and not is_running(com.info.response)
                and has_bones(context))

    def execute(self, context):
        global _active
        self._capture_wm = context.window_manager
        self._timer = None
        try:
            self._job = StandaloneBoneCapture(context)
            _active = self
            self._timer = self._capture_wm.event_timer_add(0.01, window=context.window)
            self._capture_wm.modal_handler_add(self)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return self._finish(context, {'CANCELLED'})
        return {'RUNNING_MODAL'}

    def _finish(self, context, outcome, *, restore=True):
        global _active
        try:
            job = getattr(self, '_job', None)
            if job is not None:
                job.close(restore=restore)
                self._job = None
        finally:
            origin.detach_capture_timer(self, context)
            if _active is self:
                _active = None
        if context is not None:
            from ..solver import redraw_all_areas
            redraw_all_areas(context)
        return outcome

    def cancel(self, context):
        self._finish(context, {'CANCELLED'})

    def modal(self, context, event):
        if _active is not self:
            return {'CANCELLED'}
        if event.type == 'ESC' and event.value == 'PRESS':
            return self._finish(context, {'CANCELLED'})
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        try:
            job = self._job
            if job.tick(context):
                from ..solver import redraw_all_areas
                redraw_all_areas(context)
                return {'RUNNING_MODAL'}
            self.report({'INFO'}, f"Bone preview: {len(job.bones.document['bindings'])} bindings, "
                        f"{job.frame_sets} frame evaluations, {time.perf_counter() - job.started:.2f}s. "
                        "Transfer still evaluates live inputs.")
            return self._finish(context, {'FINISHED'})
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return self._finish(context, {'CANCELLED'})


class SOLVER_OT_AbortBoneCapture(bpy.types.Operator):
    bl_idname = "solver.abort_bone_capture"
    bl_label = "Abort Bone Capture"

    def execute(self, context):
        cancel_active()
        return {'FINISHED'}
