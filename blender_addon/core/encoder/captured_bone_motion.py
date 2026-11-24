"""Session-scoped motion captured explicitly in the Blender authoring scene."""

from contextlib import contextmanager

import bpy
from bpy.app.handlers import persistent

from ...i18n import pgettext_iface as iface_

_captured = None
_evaluating = None
_watch = None


def _motion_dependencies(scene):
    """Collect RNA IDs reachable from participating rigs and their controllers."""
    pending = [obj for obj in scene.objects if obj.type == 'ARMATURE'
               and any(b.ppf_bone_physics.enabled for b in obj.data.bones)]
    seen = set()
    unknown = False
    def references(owner, *, collections=True):
        for prop in owner.bl_rna.properties:
            if prop.identifier == 'rna_type':
                continue
            if prop.type == 'POINTER':
                target = getattr(owner, prop.identifier, None)
                if isinstance(target, bpy.types.ID):
                    pending.append(target)
            elif collections and prop.type == 'COLLECTION':
                for entry in getattr(owner, prop.identifier, ()):
                    for field in entry.bl_rna.properties:
                        if field.type == 'POINTER' and field.identifier != 'rna_type':
                            target = getattr(entry, field.identifier, None)
                            if isinstance(target, bpy.types.ID):
                                pending.append(target)

    def strips(items):
        for strip in items:
            pending.append(strip.action)
            strips(strip.strips)

    while pending:
        item = pending.pop()
        if item is None:
            continue
        try:
            pointer = item.as_pointer()
            if pointer in seen:
                continue
            seen.add(pointer)
            if isinstance(item, bpy.types.Collection):
                pending.extend(item.objects)
                pending.extend(item.children)
            if isinstance(item, bpy.types.Object):
                pending.extend((item.data, item.parent))
                constraints = list(item.constraints)
                if item.type == 'ARMATURE':
                    constraints.extend(c for bone in item.pose.bones for c in bone.constraints)
                    pending.extend(b.ppf_bone_physics.proxy for b in item.data.bones
                                   if b.ppf_bone_physics.enabled)
                for constraint in constraints:
                    references(constraint)
                for modifier in item.modifiers:
                    references(modifier)
                    if modifier.type == 'NODES':
                        unknown = True
            if isinstance(item, (bpy.types.Mesh, bpy.types.Curve, bpy.types.Texture)):
                # Direct IDs include a Curve's taper/bevel objects and a
                # displacement Texture's Image. Do not enumerate vertices,
                # spline points or other large geometry collections.
                references(item, collections=False)
                if isinstance(item, bpy.types.Texture) and item.use_nodes:
                    unknown = True
            shape_keys = getattr(item, 'shape_keys', None)
            if shape_keys is not None:
                pending.append(shape_keys)
            animation = getattr(item, 'animation_data', None)
            if animation is not None:
                pending.append(animation.action)
                for track in animation.nla_tracks:
                    strips(track.strips)
                for curve in animation.drivers:
                    driver = curve.driver
                    if driver.type == 'SCRIPTED' and (
                            not getattr(driver, 'is_simple_expression', False)
                            or getattr(driver, 'use_self', False)):
                        unknown = True
                    for variable in curve.driver.variables:
                        if variable.type == 'CONTEXT_PROP':
                            unknown = True
                        for target in variable.targets:
                            pending.append(target.id)
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            unknown = True
    return seen, unknown


def _timing(scene):
    from ...models.groups import get_addon_data
    from . import resolve_start_frame, resolve_solver_fps
    state = get_addon_data(scene).state
    return resolve_start_frame(state), int(state.frame_count), resolve_solver_fps(state)


class CaptureWatch:
    def __init__(self):
        global _watch
        clear()
        self.valid = True
        self.scene = bpy.context.scene
        self.dependencies, self.unknown = _motion_dependencies(self.scene)
        self.timing = _timing(self.scene)
        _watch = self

    def require_current(self):
        if not self.valid:
            raise ValueError(iface_("Bone motion source changed during capture; start capture again"))

    def close(self):
        global _watch
        if _watch is self:
            _watch = None


class CapturedBoneMotion:
    def __init__(self, context, proof, document):
        from .transfer_bone_stage import _callback_token

        self.scene = context.scene
        self.layer = context.view_layer.as_pointer()
        self.filepath = bpy.data.filepath
        self.mode = context.mode
        self.playhead = (self.scene.frame_current, self.scene.frame_subframe)
        self.callbacks = _callback_token()
        self.proof = proof
        self.document = document
        self.valid = True
        self.used = False
        self.dependencies, self.unknown = _motion_dependencies(self.scene)
        self.timing = _timing(self.scene)

    def matches_origin(self, context):
        from .transfer_bone_stage import _callback_token

        return (self.valid and context.scene == self.scene and _timing(self.scene) == self.timing
                and context.view_layer.as_pointer() == self.layer
                and bpy.data.filepath == self.filepath and context.mode == self.mode
                and (context.scene.frame_current, context.scene.frame_subframe) == self.playhead
                and _callback_token() == self.callbacks)

    def consume(self, context, state, plan):
        from ..bone_author_follow import transfer_author_geometry_changed
        from .transfer_bone_stage import _signature

        if (self.valid and not self.used and _evaluating is self and plan is not None
                and not transfer_author_geometry_changed()
                and _signature(plan, state, include_scale_sources=False) == self.proof):
            self.used = True
            return self.document
        return None

    def require_upload_ready(self, context):
        if self.used and not self.matches_origin(context):
            raise ValueError(iface_("Captured bone motion changed before upload; start Transfer again"))

    def after_display_cleanup(self, context):
        if self.valid and context.scene == self.scene:
            self.playhead = (context.scene.frame_current, context.scene.frame_subframe)
            self.used = False


def publish(context, proof, document):
    global _captured
    clear()
    _captured = CapturedBoneMotion(context, proof, document)


def for_transfer(context):
    if _captured is not None and _captured.matches_origin(context):
        _captured.used = False
        return _captured
    return None


def has_capture(scene):
    return _captured is not None and _captured.valid and _captured.scene == scene


def for_display_cleanup(context):
    if _captured is not None and _captured.matches_origin(context):
        return _captured
    return None


@contextmanager
def preserve_during_output(context):
    """Keep a source snapshot across a synchronous output-only operation."""
    if not has_capture(context.scene):
        yield
        return
    # Flush any author updates queued before this operation takes ownership.
    context.view_layer.update()
    motion = for_display_cleanup(context)
    with evaluating(motion):
        yield
        if motion is not None:
            context.view_layer.update()
    if motion is not None:
        motion.after_display_cleanup(context)


def clear():
    global _captured, _watch
    if _captured is not None:
        _captured.valid = False
        _captured = None
    if _watch is not None:
        _watch.valid = False
        _watch = None


@contextmanager
def evaluating(motion):
    """Ignore evaluations caused by one synchronous Transfer callback."""
    global _evaluating
    previous = _evaluating
    _evaluating = motion
    try:
        yield
    finally:
        _evaluating = previous


@persistent
def author_graph_updated(_scene, graph):
    owner = _watch or _captured
    if owner is None or _evaluating is owner:
        return
    if _timing(owner.scene) != owner.timing:
        clear()
        return
    for update in graph.updates:
        original = getattr(update.id, 'original', update.id)
        if owner.unknown or original.as_pointer() in owner.dependencies:
            clear()
            return


@persistent
def author_frame_changed(_scene, _graph=None):
    owner = _watch or _captured
    if owner is not None and _evaluating is not owner:
        clear()


def register():
    for handlers, callback in ((bpy.app.handlers.depsgraph_update_post, author_graph_updated),
                               (bpy.app.handlers.frame_change_post, author_frame_changed)):
        if callback not in handlers:
            handlers.append(callback)


def unregister():
    clear()
    for handlers, callback in ((bpy.app.handlers.depsgraph_update_post, author_graph_updated),
                               (bpy.app.handlers.frame_change_post, author_frame_changed)):
        if callback in handlers:
            handlers.remove(callback)
