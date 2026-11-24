"""Capture authored bone motion for reuse by Transfer."""
import time

import bpy

from ...core import capture_origin as origin
from ...core.encoder import resolve_solver_fps, resolve_start_frame
from ...core.encoder import bones
from ...core.rigid_pose import has_bone_pose_cache, _detached_display_pending
from ...i18n import pgettext_iface as iface_
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
        raise RuntimeError(iface_("Stop playback and clear displayed solver bone animation before bone capture"))


def prepare_capture_authoring(context):
    """Synchronize edited proxy geometry before recording the motion snapshot."""
    from ...core.bone_author_follow import transfer_author_transaction, sync_for_transfer
    from ...core.bone_proxy_outliner import flush_pending
    from ...core.rigid_pose import suspend_playback
    with transfer_author_transaction(context) as commit_notification:
        with suspend_playback():
            sync_for_transfer(context, scan_scale_frames=False)
        commit_notification()
    flush_pending()
    context.view_layer.update()


def restore_sampled_author_pose(context, scene, saved_channels):
    from ...core.rigid_pose import _capture_native_channels, _restore_native_channels
    changed = False
    for arm_name, pointer, data, name, channels in saved_channels:
        arm = scene.objects.get(arm_name)
        if arm is None or arm.as_pointer() != pointer or arm.data != data:
            continue
        pose = arm.pose.bones.get(name)
        if pose is not None and _capture_native_channels(pose) != channels:
            _restore_native_channels(pose, channels)
            changed = True
    if changed:
        context.view_layer.update()


class BoneSampling:
    """Read bone motion at the solver's requested sample times."""
    def __init__(self, context):
        _require_idle(context)
        self.job = {'saved_frame': int(context.scene.frame_current)}
        origin.remember_origin(self.job, context.scene)
        self.stamp = _stamp(context)
        self.start, self.count, self.fps, *_ = self.stamp
        if self.count <= 0:
            raise ValueError(iface_("Bone capture requires at least one solver frame"))
        self.solver_next = self.start
        self.solver_plan = None
        self.actor_sources = tuple((o, o.data, o.as_pointer()) for o in context.scene.objects
                                   if o.type == 'ARMATURE')
        self.processed = 0
        self.total = self.count
        self.document = None
        self.proof = None

    @property
    def pending(self):
        return self.solver_next

    def require_valid(self, context):
        origin.require_origin(self.job, context)
        _require_idle(context)
        if _stamp(context) != self.stamp:
            raise RuntimeError(iface_("Bone capture timing or unit settings changed"))
        for arm, data, pointer in self.actor_sources:
            if (arm.as_pointer() != pointer or arm.data != data
                    or context.scene.objects.get(arm.name) != arm):
                raise RuntimeError(iface_("Bone capture armature identity changed"))

    def sample(self, context, f):
        if self.pending != f:
            return False
        self.require_valid(context)
        if f == self.start:
            state = get_addon_data(context.scene).state
            self.solver_plan = bones.prepare_transport_bones(context, state, self.fps, self.start)
            if self.solver_plan is None:
                raise RuntimeError(iface_("No enabled bone transport bindings found"))
            from ...core.encoder.transfer_bone_stage import _signature
            self.proof = _signature(self.solver_plan, state, include_scale_sources=False)
            self._has_track = bones.has_sampled_bone_motion(self.solver_plan[3], self.solver_plan[4])
            self.total = self.count if self._has_track else 1
        _, _, scale_sources, anchors, roots = self.solver_plan
        if self._has_track or roots:
            bones.sample_transport_bone_frame(
                scale_sources, anchors.values(), roots, f - self.start, self.fps, f)
        self.solver_next = f + 1 if self._has_track and f + 1 < self.start + self.count else None
        if self.solver_next is None:
            self.document = bones.finish_transport_bones(self.solver_plan)
        self.processed += 1
        self.require_valid(context)
        return True


class StandaloneBoneCapture:
    def __init__(self, context):
        from . import pin_capture_ops as pin, static_deform_ops as static
        if is_active() or pin.is_pin_capture_running() or static.is_capture_running():
            raise RuntimeError(iface_("Another deformation capture is in progress"))
        prepare_capture_authoring(context)
        from ...core.rigid_pose import _capture_native_channels
        self.pose_channels = [(arm.name, arm.as_pointer(), arm.data, pb.name, _capture_native_channels(pb))
                              for arm in context.scene.objects if arm.type == 'ARMATURE'
                              for pb in arm.pose.bones] if context.mode == 'POSE' else []
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
        context.view_layer.update()
        from ...core.encoder.captured_bone_motion import CaptureWatch
        self.watch = CaptureWatch()

    def _require(self, context):
        if self.closed or self.layer != context.view_layer.as_pointer():
            raise RuntimeError(iface_("Bone capture view layer changed"))
        self.bones.require_valid(context)
        if self.world != guard._world_stamp(context.scene) or not guard._same_callbacks(self.callbacks):
            raise RuntimeError(iface_("Bone capture frame callbacks or rigid body world changed"))
        if self.seeded is not None and (context.scene.frame_current != self.seeded
                                       or context.scene.frame_subframe):
            raise RuntimeError(iface_("Timeline changed during bone capture"))

    def tick(self, context, *, budget_ms=35):
        try:
            deadline = time.perf_counter() + budget_ms / 1000
            self.watch.require_current()
            from ...core.encoder.captured_bone_motion import evaluating
            with evaluating(self.watch):
                return self._tick_evaluated(context, deadline)
        except BaseException:
            self.close()
            raise

    def _tick_evaluated(self, context, deadline):
        self._require(context)
        while (frame := self.bones.pending) is not None:
            self.scene.frame_set(frame)
            self.frame_sets += 1
            self.seeded = frame
            context.view_layer.update()
            if (self.pose_channels and frame == self.bones.start
                    and frame == self.bones.job['saved_frame']
                    and not self.bones.job['saved_subframe']):
                restore_sampled_author_pose(context, self.scene, self.pose_channels)
            self._require(context)
            self.bones.sample(context, frame)
            if time.perf_counter() >= deadline:
                break
        return self.bones.pending is not None

    def close(self, *, restore=True):
        if self.closed:
            return
        self.closed = True
        self.watch.close()
        if restore:
            origin.restore_frame(self.bones.job)
            restore_sampled_author_pose(self.context, self.scene, self.pose_channels)


_active = None


def is_active():
    return _active is not None and not _active._job.closed


def standalone_progress(context):
    if not is_active() or _active._job.scene != context.scene:
        return None
    job = _active._job
    return job.bones.processed, job.bones.total, job.seeded


def draw_capture_controls(layout, context):
    progress = standalone_progress(context)
    if progress is None:
        row = layout.row(align=True)
        row.operator(SOLVER_OT_CaptureBoneMotion.bl_idname, icon="ARMATURE_DATA")
        row.operator(SOLVER_OT_ClearBoneCapture.bl_idname, icon="TRASH")
        return
    done, total, _frame = progress
    layout.progress(
        factor=done / max(total, 1), type="BAR",
        text=iface_("Sampling bone motion: {done}/{total} frames").format(done=done, total=total),
    )
    layout.operator(SOLVER_OT_AbortBoneCapture.bl_idname, icon="CANCEL")


def cancel_active(*, restore=True):
    if _active is not None:
        _active._finish(None, {'CANCELLED'}, restore=restore)


class SOLVER_OT_CaptureBoneMotion(bpy.types.Operator):
    """Capture bone motion for Transfer in the current Blender session."""
    bl_idname = "solver.capture_bone_motion"
    bl_label = "Capture Bone Motion"

    @classmethod
    def poll(cls, context):
        from . import static_deform_ops as static, pin_capture_ops as pin
        if not context.window or context.mode not in {"OBJECT", "POSE"} or is_active() or static.is_capture_running() or pin.is_pin_capture_running():
            return False
        from ..solver import com, is_bake_running, is_running, _staged_encode_active
        return (not is_bake_running() and not _staged_encode_active() and not com.busy() and not is_running(com.info.response)
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
            job.close()
            context.view_layer.update()
            from ...core.encoder import captured_bone_motion
            captured_bone_motion.publish(context, job.bones.proof, job.bones.document)
            self.report({'INFO'}, iface_("Bone capture complete: {bindings} bindings, {frames} frame evaluations, {seconds:.2f} s").format(
                bindings=len(job.bones.document['bindings']),
                frames=job.frame_sets,
                seconds=time.perf_counter() - job.started,
            ))
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


class SOLVER_OT_ClearBoneCapture(bpy.types.Operator):
    bl_idname = "solver.clear_bone_capture"
    bl_label = "Clear Bone Capture"

    @classmethod
    def poll(cls, context):
        from ...core.encoder.captured_bone_motion import has_capture
        from ..solver import _staged_encode_active, com, is_bake_running, is_running
        from . import static_deform_ops as static, pin_capture_ops as pin
        return (has_capture(context.scene) and not is_active()
                and not static.is_capture_running() and not pin.is_pin_capture_running()
                and not _staged_encode_active() and not is_bake_running()
                and not com.busy() and not is_running(com.info.response))

    def execute(self, context):
        from ...core.encoder.captured_bone_motion import clear
        clear()
        return {'FINISHED'}
