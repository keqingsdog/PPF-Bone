"""Invocation-scoped bone motion preparation for Transfer and Update Params."""
from contextlib import contextmanager
import hashlib
import time

import bpy

from . import bones, resolve_solver_fps, resolve_start_frame
from .cbor_encode import dumps_envelope
from .. import capture_origin
from ..frame_time import set_scene_frame
from ..rigid_pose import has_bone_pose_cache, _detached_display_pending, suspend_playback
from ...models.groups import get_addon_data

_active = None


def active():
    return _active is not None


def cancel(*, restore=True):
    if _active is not None:
        _active.close(restore=restore)


def _invalidate(_scene, graph):
    job = _active
    if job is not None and not job.internal and any(graph.updates):
        job.invalid = True


def _callback_token():
    return tuple(tuple(getattr(bpy.app.handlers, n)) for n in
                 ('frame_change_pre', 'frame_change_post',
                  'depsgraph_update_pre', 'depsgraph_update_post'))


def _check_scene(context):
    if context.mode != 'OBJECT' or (context.screen and context.screen.is_animation_playing):
        raise ValueError('Sliced Transfer needs an idle timeline in Object Mode')
    if any(has_bone_pose_cache(sc) or _detached_display_pending(sc) for sc in bpy.data.scenes):
        raise ValueError('Sliced Transfer cannot run while a solver pose cache is displayed')
    world = context.scene.rigidbody_world
    if world and world.enabled:
        raise ValueError('Sliced Transfer cannot reorder a running rigid body world')
    if any(mod.type in {'CLOTH','SOFT_BODY','FLUID','DYNAMIC_PAINT','PARTICLE_SYSTEM'}
           for ob in context.scene.objects for mod in ob.modifiers if mod.show_viewport):
        raise ValueError('Sliced Transfer cannot reorder native simulations')
    root = __package__.split('.core.', 1)[0]
    external = [f"{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', type(fn).__name__)}"
                for bucket in _callback_token() for fn in bucket
                if not getattr(fn, '__module__', '').startswith(root + '.')]
    if external:
        registered = ', '.join(external)
        raise ValueError(
            f'Sliced Transfer cannot reorder frame or graph handlers ({registered}). Use Transfer.')


def _signature(plan, state):
    result, _, scales, anchors, roots = plan
    flags = [(entry[2]['body_uuid'], entry[4]) for entry in roots]
    refs = sorted((a.name, b.name, s) for a,b,s in scales)
    rows = [(key, request['parent'].name,
             request['ancestor'].name if request['ancestor'] else None,
             [pb.name for pb in request['chain']]) for key,request in anchors.items()]
    return hashlib.sha256(dumps_envelope('Param', {
        'raw_initial_bones':result, 'source_scales':refs, 'anchor_tracks':rows,
        'root_sources':flags, 'frame_count':int(state.frame_count),
    })).digest()


class BoneSamplingRequiresLegacy(ValueError):
    """Supported authoring that needs the original sampling order."""


def can_slice_bone_motion(context):
    try:
        _check_scene(context)
    except ValueError:
        return False
    return not active()


class SlicedBoneStage:
    def __init__(self, context):
        global _active
        if _active is not None:
            raise RuntimeError('A sliced Transfer is already active')
        _check_scene(context)
        scene = context.scene
        self.origin = {'saved_frame': int(scene.frame_current)}
        capture_origin.remember_origin(self.origin,scene)
        self.layer = context.view_layer.as_pointer()
        self.scene = scene
        self.state = get_addon_data(scene).state
        self.start = resolve_start_frame(self.state)
        self.fps = resolve_solver_fps(self.state)
        self.count = int(self.state.frame_count)
        self.next = 0
        self.seeded = None
        self.plan = self.proof = self.document = None
        self.prepared = self.closed = self.invalid = self.internal = self.used = False
        bpy.app.handlers.depsgraph_update_post.append(_invalidate)
        _active = self
        self.callbacks = _callback_token()

    @contextmanager
    def _evaluation(self):
        self.internal = True
        try:
            with suspend_playback():
                yield
        finally:
            self.internal = False

    def _guard(self, context, *, check_frame=True):
        capture_origin.require_origin(self.origin,context)
        if context.mode != 'OBJECT' or (context.screen and context.screen.is_animation_playing):
            raise ValueError('Sliced Transfer needs an idle timeline in Object Mode')
        if self.closed or self.invalid or context.view_layer.as_pointer() != self.layer:
            raise ValueError('Sliced Transfer cancelled because the source graph changed')
        if _callback_token() != self.callbacks:
            raise ValueError('Sliced Transfer cancelled because frame or graph handlers changed')
        now = (int(self.scene.frame_current),float(self.scene.frame_subframe))
        expected = self.seeded or (self.origin['saved_frame'],self.origin['saved_subframe'])
        if check_frame and now != expected:
            raise ValueError('Sliced Transfer cancelled because the timeline moved')
        if (resolve_start_frame(self.state),resolve_solver_fps(self.state),int(self.state.frame_count)) != (self.start,self.fps,self.count):
            raise ValueError('Sliced Transfer cancelled because timing changed')

    def tick(self, context, *, budget_ms=25):
        self._guard(context)
        # No mutable transfer or Blender playback context crosses this boundary.
        deadline = time.perf_counter()+budget_ms/1000
        if not self.prepared:
            with self._evaluation():
                self.scene.frame_set(self.start)
                self.plan = bones.prepare_transport_bones(context,self.state,self.fps,self.start)
                self.prepared = True
                self.seeded = (self.start,0.)
                if self.plan is not None:
                    self.proof = _signature(self.plan,self.state)
                    if any(row[4] == 'OBJECT' for row in self.plan[4]):
                        raise BoneSamplingRequiresLegacy('Sliced Transfer does not support proxy-driven bone roots')
            if self.plan is not None and any(self.plan[2:]):
                return True  # allow redraw after preparation as well
        with self._evaluation():
            if self.plan is not None and any(self.plan[2:]):
                _,_,scales,anchors,roots = self.plan
                while self.next < self.count:
                    frame = self.start+self.next
                    self.scene.frame_set(frame)
                    context.view_layer.update()
                    if (self.scene.frame_current,self.scene.frame_subframe) != (frame,0.):
                        raise ValueError('A callback moved the timeline during Sliced Transfer')
                    bones.sample_transport_bone_frame(scales, anchors.values(), roots,
                                                      self.next,self.fps,frame)
                    self.next += 1
                    self.seeded = (frame,0.)
                    if time.perf_counter() >= deadline:
                        break
                if self.next < self.count:
                    return True
                # Finish exactly where the original sampler finishes: solver start.
                self.scene.frame_set(self.start)
                context.view_layer.update()
            if self.plan is not None:
                self.document = bones.finish_transport_bones(self.plan)
            self.plan = None
            self._restore()
        return False

    def _restore(self):
        set_scene_frame(self.scene,self.origin['saved_frame'],self.origin['saved_subframe'])
        self.seeded = (self.origin['saved_frame'],self.origin['saved_subframe'])

    @contextmanager
    def encoding(self, context):
        self._guard(context)
        # Revalidate immediately before the final synchronous transaction.
        _check_scene(context)
        if not self.prepared or self.plan is not None:
            raise ValueError('Bone motion is incomplete')
        self.internal = True
        try:
            yield
        finally:
            self.internal = False

    def consume(self, context, state, plan):
        if not self.internal or self.closed or self.invalid or self.used or context.scene != self.scene:
            raise ValueError('Bone motion proof expired; restart Transfer')
        from ..bone_author_follow import transfer_author_geometry_changed
        if transfer_author_geometry_changed():
            raise BoneSamplingRequiresLegacy('Sliced Transfer needs stable author proxies. Use Transfer to synchronize them.')
        if ((plan is not None and _signature(plan,state) != self.proof)
            or (plan is None and self.proof is not None)):
            raise ValueError('Bone inputs changed during Transfer; restart Transfer')
        self.used = True
        return self.document

    def require_upload_ready(self, context):
        self._guard(context)
        _check_scene(context)
        if not self.used:
            raise ValueError('Bone motion was not validated before upload')

    def close(self, *, restore=True):
        global _active
        if self.closed:
            return
        self.closed = True
        if _active is self:
            _active = None
        if _invalidate in bpy.app.handlers.depsgraph_update_post:
            bpy.app.handlers.depsgraph_update_post.remove(_invalidate)
        try:
            if restore and (self.scene.frame_current,self.scene.frame_subframe) != (
                    self.origin['saved_frame'],self.origin['saved_subframe']):
                capture_origin.restore_frame(self.origin)
        except ReferenceError:
            pass  # Source scene already removed.
        finally:
            self.document = self.plan = self.proof = None
