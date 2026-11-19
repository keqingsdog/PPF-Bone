"""Object collision layers, shared by bone proxies and ordinary dynamic objects."""
from .. import i18n as _ppf_i18n
import bpy
import contextlib
import sys
from types import SimpleNamespace
from ..i18n.props import BoolProperty, BoolVectorProperty, PointerProperty
from bpy.types import PropertyGroup

_scene_contracts = {}
_batch_depth = 0
_invalidating = False


def _effective_contract(scene):
    """Only collision author values that can affect this scene's solver pairs."""
    # Testing hasattr/getattr on the registered PointerProperty initializes it.
    # Without persisted root data this scene cannot have active PPF groups.
    if not scene.is_property_set('zozo_contact_solver'):
        return None
    records = collect_collision_filters(SimpleNamespace(scene=scene))['objects']
    filtered = tuple(sorted((uid, row['groups'], row['mask'])
                            for uid, row in records.items() if row['enabled']))
    adjacent = []
    for arm in scene.objects:
        if arm.type != 'ARMATURE' or not hasattr(bpy.types.Bone, 'ppf_bone_physics'):
            continue
        for bone in arm.data.bones:
            settings = bone.ppf_bone_physics
            parent = bone.parent
            if (settings.enabled and settings.proxy and settings.body_uuid and parent
                    and parent.ppf_bone_physics.enabled and parent.ppf_bone_physics.proxy
                    and parent.ppf_bone_physics.body_uuid and not settings.collide_connected):
                adjacent.append((parent.ppf_bone_physics.body_uuid, settings.body_uuid))
    return filtered, tuple(sorted(adjacent))


def remember_collision_contracts(*_args):
    _scene_contracts.clear()
    _scene_contracts.update((int(scene.session_uid), _effective_contract(scene)) for scene in bpy.data.scenes)


def _initial_contract_timer():
    if not hasattr(bpy.data, 'scenes'):
        return .01
    remember_collision_contracts()
    return None


def schedule_collision_baseline():
    """register() runs under RestrictBlend; defer all scene reads."""
    if not bpy.app.timers.is_registered(_initial_contract_timer):
        bpy.app.timers.register(_initial_contract_timer, first_interval=0.)


def _suspend_failed_mesh_playback(scene):
    """Keep failed Clear references, but stop only verified PPF mesh output."""
    from .groups import iterate_active_object_groups
    from ..core import pc2
    from ..core.uuid_registry import resolve_assigned
    for group in iterate_active_object_groups(scene):
        for assigned in group.assigned_objects:
            obj = resolve_assigned(assigned)
            if obj is None or obj.type != 'MESH':
                continue
            mod = obj.modifiers.get(pc2.MODIFIER_NAME)
            if mod is None or pc2._owned_mesh_output_path(obj, pc2.object_pc2_key_readonly(obj), mod) is None:
                continue
            # Visibility F-curves would otherwise re-enable old output on the
            # next frame. Preserve filepath/UUID for the normal Clear retry.
            pc2.remove_cache_visibility_keys(obj)
            mod.show_viewport = False
            mod.show_render = False


def collision_settings_changed(_settings=None, _context=None):
    """Retire builds and PPF playback only after an effective author change."""
    global _invalidating
    if _batch_depth or _invalidating:
        return
    # A real edit has established the live baseline. A late register timer
    # must not overwrite a pending failed-Clear signature with current intent.
    if bpy.app.timers.is_registered(_initial_contract_timer):
        bpy.app.timers.unregister(_initial_contract_timer)
    changed = []
    for scene in bpy.data.scenes:
        key = int(scene.session_uid)
        current = _effective_contract(scene)
        previous = _scene_contracts.get(key)
        if previous != current:
            changed.append((scene, current))
    if not changed:
        return
    _invalidating = True
    try:
        # Existing singleton only: author property writes never start a runner.
        facade = sys.modules.get(__package__.rsplit('.', 1)[0] + '.core.facade')
        communicator = getattr(facade, 'communicator', None)
        if communicator is not None:
            runner = communicator._runner
            pointers = {scene.as_pointer() for scene, _current in changed}
            identity = runner._collision_identity
            if (identity and identity[0] in pointers) or (bpy.context.scene and bpy.context.scene.as_pointer() in pointers):
                communicator.bone_physical_body_changed()
        from ..core.animation import clear_animation_data
        for scene, current in changed:
            # Production Clear touches PPF-owned output, preserving author
            # Actions, pin caches and unrelated/user-named MESH_CACHE modifiers.
            try:
                clear_animation_data(SimpleNamespace(scene=scene), move_to_frame=False)
            except Exception as error:
                # Keep the previous signature so the same-value RNA path and
                # explicit recovery call can retry after the file is released.
                try:
                    _suspend_failed_mesh_playback(scene)
                except Exception as suspend_error:
                    print(f'PPF collision playback suspension failed ({scene.name}): {suspend_error}')
                print(f'PPF collision Clear pending retry ({scene.name}): {error}')
                continue
            _scene_contracts[int(scene.session_uid)] = current
    finally:
        _invalidating = False


@contextlib.contextmanager
def collision_edit_batch():
    """RNA writes and their rollback finish before any playback is retired."""
    global _batch_depth
    _batch_depth += 1
    try:
        yield
    finally:
        _batch_depth -= 1
        if not _batch_depth:
            collision_settings_changed()


def layers_to_bits(layers):
    return sum(1 << i for i, enabled in enumerate(layers) if enabled)


def settings_record(settings):
    return {"enabled": bool(settings.collision_filter_enabled),
            "groups": layers_to_bits(settings.collision_group_layers),
            "mask": layers_to_bits(settings.collision_mask_layers)}


def collect_collision_filters(context):
    from .groups import iterate_active_object_groups
    from ..core.uuid_registry import get_object_uuid
    records = {}
    for arm in context.scene.objects:
        if arm.type == "ARMATURE":
            for bone in arm.data.bones:
                settings = bone.ppf_bone_physics
                if settings.enabled and settings.proxy and settings.body_uuid:
                    records[settings.body_uuid] = settings_record(settings)
    objects = {get_object_uuid(obj): obj for obj in context.scene.objects if obj.type == "MESH"}
    for group in iterate_active_object_groups(context.scene):
        for assigned in group.assigned_objects:
            if not assigned.included or assigned.uuid in records:
                continue
            obj = objects.get(assigned.uuid)
            if obj is not None and hasattr(obj, "ppf_collision_filter"):
                record = settings_record(obj.ppf_collision_filter)
                if record["enabled"]:
                    records[assigned.uuid] = record
    return {"rule": "symmetric_membership_mask_and_v1", "objects": records}


class PPF_ObjectCollisionFilter(PropertyGroup):
    collision_filter_enabled: BoolProperty(name="Enable Collision Group Filter", default=False, update=collision_settings_changed, options=set())
    collision_group_layers: BoolVectorProperty(name="Collision Membership", size=16, default=(True,) + (False,) * 15, update=collision_settings_changed, options=set())
    collision_mask_layers: BoolVectorProperty(name="Allowed Collision Groups", size=16, default=(True,) * 16, update=collision_settings_changed, options=set())


def register():
    _ppf_i18n.register_class(PPF_ObjectCollisionFilter)
    bpy.types.Object.ppf_collision_filter = PointerProperty(type=PPF_ObjectCollisionFilter)
    bpy.app.handlers.load_post.append(bpy.app.handlers.persistent(remember_collision_contracts))
    bpy.app.handlers.undo_post.append(bpy.app.handlers.persistent(collision_settings_changed))
    bpy.app.handlers.redo_post.append(bpy.app.handlers.persistent(collision_settings_changed))


def unregister():
    if bpy.app.timers.is_registered(_initial_contract_timer):
        bpy.app.timers.unregister(_initial_contract_timer)
    for handlers, callback in ((bpy.app.handlers.load_post, remember_collision_contracts),
                               (bpy.app.handlers.undo_post, collision_settings_changed),
                               (bpy.app.handlers.redo_post, collision_settings_changed)):
        if callback in handlers:
            handlers.remove(callback)
    _scene_contracts.clear()
    del bpy.types.Object.ppf_collision_filter
    bpy.utils.unregister_class(PPF_ObjectCollisionFilter)
