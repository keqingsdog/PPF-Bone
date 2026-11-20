"""Follow authored pose bones with their dedicated proxies outside playback.

Detection is read-only and may run in the persistent poll. Author writes occur
from the FramePump TIMER or explicit transfer preflight, before mass/geometry
sampling. Native frames and their caches remain authoritative while present.
"""
import math
import json
from contextlib import contextmanager
from copy import deepcopy

import bpy
from bpy.app.handlers import persistent


_seen = {}
_rest_seen = {}
_author_signatures = {}  # Primitive author inputs, independent of evaluated pose.
_frame_object_observations = {}
_motion_dirty_scenes = set()
_EPS = 2e-6
_initialized = False
_active = False
_identity_rebuilding = False
_baseline_generation = 0
_pose_preview = {}  # Primitive row keys -> frozen local author-offset matrices.
_edit_metadata = {}  # UUID tuple -> primitive-only binding display metadata.
_unproven = set()  # Legacy baselines lack an authoritative persisted offset.
_transfer_deferred = None  # None outside an encoder transaction; bool inside.
_AUTHOR_RECORD = 'ppf_bone_author_follow_v1'
_TIMER_KEY = 'ppf_bone_author_follow_initial_snapshot'


def _close(a, b):
    return max(abs(float(a[i][j])-float(b[i][j])) for i in range(4) for j in range(4)) <= _EPS


def _assigned_close(actual, target):
    """Float32 matrix assignment can round Euler frames near a singularity.

    This bound applies only after assignment, not motion detection or shear
    admission. Per-element bounds keep a large translation from relaxing a
    rotation check. Both matrices must retain an affine bottom row.
    """
    for matrix in (actual, target):
        if any(abs(float(matrix[3][j]) - (1. if j == 3 else 0.)) > _EPS for j in range(4)):
            return False
    return all(abs(float(actual[i][j])-float(target[i][j]))
               <= 32.*2.**-23 * max(1., abs(float(actual[i][j])), abs(float(target[i][j])))
               for i in range(3) for j in range(4))


def _finite_invertible(matrix):
    return (all(math.isfinite(float(v)) for row in matrix for v in row)
            and math.isfinite(float(matrix.determinant()))
            and abs(float(matrix.determinant())) > 1e-12)


def _author_scene(scene):
    from .facade import communicator
    from .state import Solver
    from .pc2 import scene_has_solver_cache
    from .rigid_pose import has_bone_pose_cache
    from ..models.groups import get_addon_data, has_addon_data
    if not has_addon_data(scene) or has_bone_pose_cache(scene) or scene_has_solver_cache(scene):
        return False
    if communicator.busy() or communicator._engine.state.solver in (
            Solver.RUNNING, Solver.STARTING, Solver.SAVING, Solver.BUILDING):
        return False
    if communicator._runner.has_pending_animation_frames():
        return False
    state = get_addon_data(scene).state
    if state.convert_fetched_frames_to_list():
        return False
    return True


def time_driven_bones(arm):
    """Names whose evaluated pose may be driven by animation, including ancestors."""
    from .utils import get_id_fcurves
    ad = arm.animation_data
    curves = list(get_id_fcurves(arm))
    if ad:
        curves.extend(ad.drivers)
    paths = tuple(fc.data_path for fc in curves if not fc.mute)
    def strip_paths(strip):
        if strip.mute:
            return ()
        result = tuple(path for child in getattr(strip, 'strips', ()) for path in strip_paths(child))
        action = strip.action
        if action is None:
            return result
        slot = getattr(strip, 'action_slot', None)
        for layer in getattr(action, 'layers', ()):
            for action_strip in layer.strips:
                bag = action_strip.channelbag(slot) if slot else None
                if bag:
                    result += tuple(fc.data_path for fc in bag.fcurves if not fc.mute)
        return result + tuple(fc.data_path for fc in getattr(action, 'fcurves', ()) if not fc.mute)
    if ad and ad.use_nla:
        tracks = [track for track in ad.nla_tracks if not track.mute]
        solos = [track for track in tracks if track.is_solo]
        for track in solos or tracks:
            for strip in track.strips:
                paths += strip_paths(strip)
    # Object animation changes every bone's world placement even when no
    # pose channel is keyed. Its frame-stamped observation distinguishes
    # ordinary playback from an independent author transform.
    if any(path in {'location', 'rotation_euler', 'rotation_quaternion',
                    'rotation_axis_angle', 'scale', 'delta_location',
                    'delta_rotation_euler', 'delta_rotation_quaternion',
                    'delta_scale'} for path in paths):
        return {bone.name for bone in arm.data.bones}
    prefixes = {bone.name: arm.pose.bones[bone.name].path_from_id() + '.' for bone in arm.data.bones}
    direct = {name for name, prefix in prefixes.items() if any(path.startswith(prefix) for path in paths)}
    names = set(direct)
    for bone in arm.data.bones:
        if any(parent.name in direct for parent in bone.parent_recursive):
            names.add(bone.name)
    return names

def _author_signature(arm, bone, proxy):
    def matrix(m):
        return tuple(float(v) for row in m for v in row)
    return (matrix(arm.matrix_basis), matrix(bone.matrix_local),
            tuple(bone.head_local), tuple(bone.tail_local),
            bone.parent.name if bone.parent else None, matrix(proxy.matrix_basis))

def _record_row_signature(scene, key, proxy):
    arm = next((o for o in scene.objects if o.as_pointer() == key[1]), None)
    bone = arm.data.bones.get(key[2]) if arm else None
    if bone is not None:
        _author_signatures[key] = _author_signature(arm, bone, proxy)

@persistent
def _observe_frame_object_basis(scene, depsgraph=None):
    from ..ui.dynamics.group_ops import cleanup_identity_rebuilding
    if not _active or _identity_rebuilding or cleanup_identity_rebuilding():
        return
    for arm in scene.objects:
        if arm.type == 'ARMATURE':
            _frame_object_observations[(scene.as_pointer(), arm.as_pointer())] = (
                scene.session_uid, arm.session_uid, scene.frame_current, scene.frame_subframe,
                tuple(float(v) for row in arm.matrix_basis for v in row))

def _authored_signature_changed(scene, arm, old, current):
    if old[1:] != current[1:]:
        return True
    observed = _frame_object_observations.get((scene.as_pointer(), arm.as_pointer()))
    return bool(old[0] != current[0] and observed is not None
                and observed[:4] == (scene.session_uid, arm.session_uid, scene.frame_current, scene.frame_subframe)
                and observed[4] != current[0])

def _rows(scene, *, validate=True, skip_animated=False):
    from ..models.bone_bindings import validate_proxy_owner, armature_data_owner_counts
    data_owners = armature_data_owner_counts() if validate else None
    from ..ops.bone_proxy import owned_bone_material
    from ..models.groups import iterate_active_object_groups
    uuid_counts = {}
    if validate:
        for group in iterate_active_object_groups(scene):
            uuid_counts[group.uuid] = uuid_counts.get(group.uuid, 0) + 1
    for arm in scene.objects:
        if arm.type != 'ARMATURE' or arm.mode == 'EDIT':
            continue
        driven = time_driven_bones(arm) if skip_animated else ()
        for bone in arm.data.bones:
            settings = bone.ppf_bone_physics
            proxy = settings.proxy
            if bone.name in driven:
                if proxy is None:
                    continue
                key = (scene.as_pointer(), arm.as_pointer(), bone.name, proxy.as_pointer())
                previous_signature = _author_signatures.get(key)
                if previous_signature is None or not _authored_signature_changed(scene, arm, previous_signature, _author_signature(arm, bone, proxy)):
                    continue
            if (proxy is not None and scene.objects.get(proxy.name) == proxy
                    and proxy.get('ppf_armature_uuid') == arm.ppf_armature_uuid
                    and proxy.get('ppf_bone_uuid') == settings.bone_uuid
                    and proxy.get('_solver_uuid') == settings.body_uuid):
                _edit_metadata[(arm.ppf_armature_uuid, settings.bone_uuid, settings.body_uuid)] = {
                    'bone_name': bone.name,
                    'display': {'enabled': bool(settings.enabled), 'physical_model': settings.physical_model,
                                'bone_uuid': settings.bone_uuid, 'body_uuid': settings.body_uuid,
                                'group_uuid': settings.group_uuid,
                                'preview_color_custom': bool(settings.preview_color_custom),
                                'preview_color': tuple(settings.preview_color)}}
            if not settings.enabled or proxy is None or proxy.mode == 'EDIT':
                continue
            if scene.objects.get(proxy.name) != proxy or len(proxy.users_scene) != 1:
                continue
            try:
                if validate:
                    validate_proxy_owner(arm, bone, data_owner_counts=data_owners)
                    group = owned_bone_material(scene, arm, bone)
                    if group is None or uuid_counts.get(group.uuid) != 1:
                        continue
                pose = arm.pose.bones.get(bone.name)
                if pose is None:
                    continue
                bone_world = arm.matrix_world @ pose.matrix
                proxy_world = proxy.matrix_world.copy()
                if not (_finite_invertible(bone_world) and _finite_invertible(proxy_world)):
                    continue
            except (AttributeError, RuntimeError, TypeError, ValueError):
                continue
            key = (scene.as_pointer(), arm.as_pointer(), bone.name, proxy.as_pointer())
            rest = (arm.matrix_world.copy(), arm.matrix_world @ bone.head_local,
                    arm.matrix_world @ bone.tail_local)
            yield key, settings, proxy, bone_world, proxy_world, rest


def _transform_active(scene):
    """A transform preview belongs to Blender until confirm/cancel, not authoring."""
    manager = getattr(bpy.context, 'window_manager', None)
    for window in getattr(manager, 'windows', ()):
        if window.scene != scene:
            continue
        for operator in getattr(window, 'modal_operators', ()):
            identifiers = (getattr(operator, 'bl_idname', ''),
                           getattr(getattr(operator, 'bl_rna', None), 'identifier', ''))
            if any(name.startswith('TRANSFORM_OT_') for name in identifiers):
                return True
    return False


def _save_author_record(settings, proxy, bone_world, proxy_world, rest, bone_name=None):
    """Store only primitive author data; Blender Undo restores it with the proxy."""
    def rows(matrix):
        return [float(v) for row in matrix for v in row]
    value = json.dumps({
        'bone_name': bone_name, 'display': {'enabled': bool(settings.enabled), 'physical_model': settings.physical_model, 'bone_uuid': settings.bone_uuid, 'body_uuid': settings.body_uuid, 'group_uuid': settings.group_uuid, 'preview_color_custom': bool(settings.preview_color_custom), 'preview_color': list(settings.preview_color)},
        'version': 1, 'armature_uuid': proxy.get('ppf_armature_uuid', ''),
        'bone_uuid': settings.bone_uuid, 'body_uuid': settings.body_uuid,
        'bone_world': rows(bone_world), 'proxy_world': rows(proxy_world),
        'offset': rows(bone_world.inverted() @ proxy_world),
        'arm_world': rows(rest[0]), 'rest_head': list(rest[1]), 'rest_tail': list(rest[2]),
    }, separators=(',', ':'))
    if proxy.get(_AUTHOR_RECORD) != value:
        proxy[_AUTHOR_RECORD] = value


def _load_author_record(settings, proxy):
    from mathutils import Matrix, Vector
    try:
        record = json.loads(proxy.get(_AUTHOR_RECORD, ''))
        if (record.get('version') != 1 or not settings.bone_uuid or not settings.body_uuid
                or record.get('armature_uuid') != proxy.get('ppf_armature_uuid')
                or record.get('bone_uuid') != settings.bone_uuid
                or record.get('body_uuid') != settings.body_uuid):
            return None
        def matrix(name):
            values = record[name]
            if len(values) != 16 or any(type(v) not in (int, float) for v in values):
                raise ValueError('Invalid author matrix')
            result = Matrix([values[i:i+4] for i in range(0, 16, 4)])
            if not _finite_invertible(result):
                raise ValueError('Invalid author matrix')
            return result
        bone, proxy_matrix, offset, arm = [matrix(name) for name in ('bone_world', 'proxy_world', 'offset', 'arm_world')]
        head, tail = Vector(record['rest_head']), Vector(record['rest_tail'])
        if len(head) != 3 or len(tail) != 3 or any(not math.isfinite(v) for point in (head, tail) for v in point):
            return None
        return (bone, proxy_matrix, offset), (arm, head, tail)
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
        return None


def read_author_offset(scene, arm, bone):
    """Read-only raw bone-local proxy offset and its authority for Edit preview.

    A legacy fallback describes current placement only; it is never persisted
    or presented as recovery of an author's unknown historical placement.
    """
    from ..ui.dynamics.group_ops import cleanup_identity_rebuilding
    if _identity_rebuilding or cleanup_identity_rebuilding():
        return None, 'rebuilding'
    try:
        from ..models.bone_bindings import validate_proxy_owner
        validate_proxy_owner(arm, bone)
        settings = bone.ppf_bone_physics
        proxy = settings.proxy
        if proxy is None or scene.objects.get(proxy.name) != proxy:
            return None, 'unavailable'
        record = _load_author_record(settings, proxy)
        if record is not None:
            return record[0][2].copy(), 'durable'
        key = (scene.as_pointer(), arm.as_pointer(), bone.name, proxy.as_pointer())
        if key in _seen:
            return _seen[key][2].copy(), 'legacy_baseline' if key in _unproven else 'baseline'
        raw = arm.matrix_world @ arm.pose.bones[bone.name].matrix
        if _finite_invertible(raw) and _finite_invertible(proxy.matrix_world):
            return raw.inverted() @ proxy.matrix_world, 'legacy_current_pose'
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        pass
    return None, 'unavailable'


def _reference_length(rest):
    """Rest endpoints are world values; remove object scale exactly once."""
    inverse = rest[0].inverted()
    length = ((inverse @ rest[2]) - (inverse @ rest[1])).length
    if not math.isfinite(length) or length <= 1.e-8:
        raise ValueError('Invalid author reference bone length')
    return length


def read_author_reference_length(scene, arm, bone):
    """Read the rest length represented by the current proxy mesh, never Edit L."""
    from ..ui.dynamics.group_ops import cleanup_identity_rebuilding
    if _identity_rebuilding or cleanup_identity_rebuilding():
        return None, 'rebuilding'
    try:
        from ..models.bone_bindings import validate_proxy_owner
        validate_proxy_owner(arm, bone)
        settings = bone.ppf_bone_physics
        proxy = settings.proxy
        if proxy is None or scene.objects.get(proxy.name) != proxy:
            return None, 'unavailable'
        record = _load_author_record(settings, proxy)
        if record is not None:
            return _reference_length(record[1]), 'durable'
        key = (scene.as_pointer(), arm.as_pointer(), bone.name, proxy.as_pointer())
        if key in _rest_seen:
            return _reference_length(_rest_seen[key]), 'legacy_baseline' if key in _unproven else 'baseline'
        length = float(bone.length)
        if math.isfinite(length) and length > 1.e-8:
            return length, 'legacy_current_pose'
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        pass
    return None, 'unavailable'


def author_length_stretch(proxy, ratio):
    """Read-only X stretch about source mesh bounds, retaining mesh-edit offset."""
    from ..ui.dynamics.group_ops import cleanup_identity_rebuilding
    if _identity_rebuilding or cleanup_identity_rebuilding():
        raise RuntimeError('Bone length preview waits for Undo/Redo reconstruction')
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError('Bone length ratio must be finite and positive')
    from mathutils import Matrix
    values = [float(v.co.x) for v in proxy.data.vertices]
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError('Bone length follow requires finite source mesh vertices')
    pivot = (min(values) + max(values)) / 2.
    return (Matrix.Translation((pivot, 0., 0.))
            @ Matrix.Diagonal((ratio, 1., 1., 1.))
            @ Matrix.Translation((-pivot, 0., 0.)))


def _snapshot_all():
    global _initialized
    from ..ui.dynamics.group_ops import cleanup_identity_rebuilding
    if _identity_rebuilding or cleanup_identity_rebuilding():
        raise RuntimeError('Bone author baseline waits for Undo/Redo reconstruction')
    # Even enumerating Scene RNA must wait until the identity barrier is released.
    scenes = tuple(bpy.data.scenes)  # Restricted addon registration cannot read this.
    if any(_transform_active(scene) for scene in scenes):
        raise RuntimeError('Bone author baseline waits for the active transform')
    if any(arm.type == 'ARMATURE' and arm.mode == 'EDIT'
           
           for scene in scenes for arm in scene.objects):
        # Do not erase a proven baseline merely because Edit mode hides its
        # pose rows. A fresh lifecycle waits; it must not adopt a later drift.
        raise RuntimeError('Bone author baseline waits for Edit mode to finish')
    layer = getattr(bpy.context, 'view_layer', None)
    if layer is not None:
        layer.update()
    staged_seen, staged_rest, staged_unproven = {}, {}, set()
    for scene in scenes:
        try:
            for key, settings, proxy, bone_world, proxy_world, rest in _rows(scene):
                durable = _load_author_record(settings, proxy)
                if key in _seen and key not in _unproven:
                    staged_seen[key], staged_rest[key] = _seen[key], _rest_seen[key]
                elif durable is not None:
                    staged_seen[key], staged_rest[key] = durable
                elif key in _seen:
                    staged_seen[key], staged_rest[key] = _seen[key], _rest_seen[key]
                    if key in _unproven:staged_unproven.add(key)
                else:
                    # Compatibility only: this is not proof of an old proxy's
                    # intended placement and is never promoted to durable data.
                    staged_seen[key] = (bone_world.copy(), proxy_world.copy(), bone_world.inverted() @ proxy_world)
                    staged_rest[key] = rest
                    staged_unproven.add(key)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            continue
    _seen.clear(); _seen.update(staged_seen)
    _rest_seen.clear(); _rest_seen.update(staged_rest)
    _unproven.clear(); _unproven.update(staged_unproven)
    _author_signatures.clear()
    for scene in scenes:
        for key, _settings, proxy, *_ in _rows(scene, validate=False):
            _record_row_signature(scene, key, proxy)
    _pose_preview.clear()
    _motion_dirty_scenes.clear()
    _motion_dirty_scenes.update(scene.session_uid for scene in scenes)
    _initialized = True


def _seed_current_frame_observations():
    # _snapshot_all already evaluated the current view layer. Other scenes
    # may not have been evaluated and must wait for their own frame event.
    scene = getattr(bpy.context, 'scene', None)
    if scene is not None:
        for arm in scene.objects:
            if arm.type != 'ARMATURE':
                continue
            key = (scene.as_pointer(), arm.as_pointer())
            stamp = (scene.session_uid, arm.session_uid, scene.frame_current, scene.frame_subframe)
            observed = _frame_object_observations.get(key)
            if observed is None or observed[:4] != stamp:
                _frame_object_observations[key] = stamp + (
                    tuple(float(v) for row in arm.matrix_basis for v in row),)


def _initialize_baseline(generation=None):
    """One-shot read-only initialization after Blender lifts RestrictBlend."""
    if not _active or generation is not None and generation != _baseline_generation:
        return None
    try:
        _snapshot_all()
        _seed_current_frame_observations()
    except (AttributeError, RuntimeError, TypeError):
        return .1
    return None


def _schedule_initial_snapshot():
    _cancel_initial_snapshot()
    generation = _baseline_generation
    def tick():
        return _initialize_baseline(generation)
    bpy.app.driver_namespace[_TIMER_KEY] = tick
    bpy.app.timers.register(tick, first_interval=0.0)


def _cancel_initial_snapshot():
    previous = bpy.app.driver_namespace.pop(_TIMER_KEY, None)
    if previous is not None and bpy.app.timers.is_registered(previous):
        bpy.app.timers.unregister(previous)


@persistent
def _mark_author_motion(scene, depsgraph):
    if not _initialized or _identity_rebuilding:
        return
    for update in depsgraph.updates:
        item = update.id.original
        if (isinstance(item, bpy.types.Object) and item.mode == 'EDIT'
                and item.get('ppf_bone_proxy_visibility_editing')):
            continue
        if (isinstance(item, bpy.types.Armature) and update.is_updated_geometry) or (
            isinstance(item, bpy.types.Object)
            and (update.is_updated_transform or update.is_updated_geometry)
            and (item.type == 'ARMATURE' or item.get('ppf_bone_proxy_owned'))
        ):
            _motion_dirty_scenes.add(scene.session_uid)
            return


def has_pending():
    """Read-only poll for modal work; no object or PropertyGroup writes."""
    if not _initialized or _identity_rebuilding:
        return False
    scene = bpy.context.scene
    if scene is None:
        return False
    if any(key[0] == scene.as_pointer() for key in _pose_preview):
        return True
    if scene.session_uid not in _motion_dirty_scenes:
        return False
    for scene in (scene,):
        try:
            if not _author_scene(scene):
                continue
            changed = set()
            # Poll only motion. Ownership/material admission remains mandatory
            # in _sync before any preview, author matrix or mesh is written.
            for key, _settings, _proxy, bone_world, proxy_world, _rest in _rows(scene, validate=False, skip_animated=True):
                previous = _seen.get(key)
                if previous is None or not _close(bone_world, previous[0]) or not _close(proxy_world, previous[1]):
                    changed.add(key)
                    continue
                old_rest = _rest_seen.get(key)
                if old_rest is not None and abs(_reference_length(_rest) / _reference_length(old_rest) - 1.) > _EPS:
                    changed.add(key)
            # A changed row is work only if ownership and the enabled group
            # still admit it. Rebindings/disabled groups cannot keep a modal
            # alive or authorize writes. The idle path never enters this scan.
            if changed and any(row[0] in changed for row in _rows(scene, skip_animated=True)):
                return True
            _motion_dirty_scenes.discard(scene.session_uid)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            continue
    return False


def _same_rest_world(previous, rest):
    return (previous is not None and not _close(previous[0], rest[0])
            and all((old-new).length <= _EPS * max(1., old.length, new.length)
                    for old, new in zip(previous[1:], rest[1:])))


def _proxy_only_inherited_arm_motion(previous, previous_rest, rest, proxy_world):
    """A parented proxy can move before a pending pose edit is settled.

    Compare in armature space so an Object transform does not look like an
    independent proxy edit and replace the proven bone-local author offset.
    """
    if previous is None or previous_rest is None:
        return False
    old_arm, new_arm = previous_rest[0], rest[0]
    if not (_finite_invertible(old_arm) and _finite_invertible(new_arm)):
        return False
    return _close(new_arm.inverted() @ proxy_world, old_arm.inverted() @ previous[1])


def _rigid_scale(matrix):
    """Split only a positive orthogonal scale, never shear or reflection."""
    from mathutils import Matrix
    columns = [matrix.to_3x3().col[i] for i in range(3)]
    scales = [column.length for column in columns]
    if (not _finite_invertible(matrix) or min(scales) <= 1e-8
            or matrix.to_3x3().determinant() <= 0
            or any(abs(columns[i].dot(columns[j]) / (scales[i]*scales[j])) > 1e-5
                   for i in range(3) for j in range(i))):
        return None
    rigid = Matrix.Translation(matrix.translation) @ matrix.to_quaternion().normalized().to_matrix().to_4x4()
    return rigid, rigid.inverted() @ matrix


def _rigid_frame(matrix):
    columns = [matrix.to_3x3().col[i] for i in range(3)]
    return (_finite_invertible(matrix)
            and abs(matrix.to_3x3().determinant()-1.) <= 1e-6
            and all(abs(columns[i].dot(columns[j])-(1. if i == j else 0.)) <= 1e-6
                    for i in range(3) for j in range(i+1)))


def _representation_snapshot(proxy):
    return (proxy.rotation_mode, proxy.location.copy(), proxy.rotation_euler.copy(),
            proxy.rotation_quaternion.copy(), proxy.rotation_axis_angle[:], proxy.scale.copy())


def _restore_representation(proxy, state):
    mode, location, euler, quaternion, axis_angle, scale = state
    proxy.rotation_mode = mode
    proxy.location = location
    proxy.rotation_euler = euler
    proxy.rotation_quaternion = quaternion
    proxy.rotation_axis_angle = axis_angle
    proxy.scale = scale


def _assign_follow_targets(context, entries, message):
    """Assign all targets; retry a rigid managed proxy using quaternion channels."""
    for proxy, target in entries:
        proxy.matrix_world = target
    context.view_layer.update()
    for proxy, target in entries:
        if _assigned_close(proxy.matrix_world, target):
            continue
        eligible = (proxy.get('ppf_armature_uuid') and proxy.get('ppf_bone_uuid')
                    and proxy.get('_solver_uuid') and not proxy.library and not proxy.override_library
                    and not proxy.animation_data and not proxy.constraints
                    and all(abs(float(v)) < 1e-12 for v in proxy.delta_location)
                    and all(abs(float(v)) < 1e-12 for v in proxy.delta_rotation_euler)
                    and all(abs(float(v) - (1. if i == 0 else 0.)) < 1e-12
                            for i, v in enumerate(proxy.delta_rotation_quaternion))
                    and all(abs(float(v) - 1.) < 1e-12 for v in proxy.delta_scale)
                    and _rigid_frame(target) and _rigid_frame(proxy.matrix_world))
        if not eligible:
            raise ValueError(message)
        proxy.rotation_mode = 'QUATERNION'
        proxy.matrix_world = target
        context.view_layer.update()
        if not _assigned_close(proxy.matrix_world, target):
            raise ValueError(message)
    if any(not _assigned_close(proxy.matrix_world, target) for proxy, target in entries):
        raise ValueError(message)


def _preview_pose(context):
    """Derived display only: freeze author offsets and never bake preview mesh."""
    scene = context.scene
    pose_arms = {obj.as_pointer() for obj in scene.objects
                 if obj.type == 'ARMATURE' and obj.mode == 'POSE'}
    rows = list(_rows(scene))  # Ownership is resolved afresh on every tick.
    valid = {key for key, *_rest in rows}
    for key in list(_pose_preview):
        if key[0] == scene.as_pointer() and key not in valid:
            _pose_preview.pop(key, None)
    changes = []
    for key, _settings, proxy, bone_world, proxy_world, _rest in rows:
        if key[1] not in pose_arms or proxy.animation_data or proxy.constraints:
            continue
        previous = _seen.get(key)
        if previous is None:
            continue  # No proven author offset: never infer one from a preview.
        offset = _pose_preview.setdefault(key, previous[2].copy())
        target = bone_world @ offset
        if _rigid_scale(target) is None:
            continue  # True shear/reflection is not repaired or baked.
        if not _close(proxy_world, target):
            changes.append((proxy, _representation_snapshot(proxy), target))
    try:
        _assign_follow_targets(context, [(proxy, target) for proxy, _old, target in changes],
                               'Bone proxy pose preview cannot represent its rigid author frame')
    except Exception:
        for proxy, old, _target in changes:
            _restore_representation(proxy, old)
        context.view_layer.update()
        raise
    return len(changes)


def _author_scale_checks_enabled(scene):
    """Whether transfer's author-scale validator can run for this scene."""
    return (_author_scene(scene)
            and not any(arm.type == 'ARMATURE' and arm.mode == 'EDIT'
                        for arm in scene.objects))


def _transfer_scale_preflight_applicable(context):
    """Read-only subset of the existing transfer gates before frame changes."""
    if context is None or context.scene is None:
        return False
    from ..ui.dynamics.group_ops import cleanup_identity_rebuilding
    if _identity_rebuilding or cleanup_identity_rebuilding():
        return False
    if _transform_active(context.scene):
        return False
    return _author_scale_checks_enabled(context.scene)


def _sync(context, *, transfer=False):
    """Stage a whole author batch before writes; Apply Transform preserves shape."""
    global _transfer_deferred
    if context is None or context.scene is None:
        return 0
    if not transfer and getattr(getattr(context, 'screen', None), 'is_animation_playing', False):
        return 0
    from ..ui.dynamics.group_ops import cleanup_identity_rebuilding
    if _identity_rebuilding or cleanup_identity_rebuilding():
        if transfer:
            raise RuntimeError('Wait for Undo/Redo reconstruction before Transfer')
        return 0
    if _transform_active(context.scene):
        if transfer:
            raise ValueError('Finish or cancel the active transform before Transfer')
        if (_initialized and getattr(context, 'mode', '') == 'POSE'
                and _author_scene(context.scene)):
            from .bone_geometry_edit import active
            if not active():
                context.view_layer.update()
                return _preview_pose(context)
        return 0
    # An armature property can change before its parented proxy's world matrix
    # is evaluated. Reading both without this flush invents a local offset.
    context.view_layer.update()
    if not _initialized:
        if not transfer:
            return 0
        _snapshot_all()
    scene = context.scene
    if not _author_scale_checks_enabled(scene):
        return 0  # Edit uses a read-only overlay; retain all proven pose baselines.
    if transfer:
        # Cached physical display is not authored pose. Keep the existing cache
        # and busy gates ahead of this validator and ahead of any mesh bake.
        from ..models.groups import get_addon_data
        from .encoder.bones import _validate_static_pose_scales
        _validate_static_pose_scales(context, get_addon_data(scene).state, scene.frame_current)
    from .bone_geometry_edit import active
    if active():
        return 0
    from .facade import communicator
    rows = list(_rows(scene, skip_animated=not transfer))
    valid = {key for key, *_rest in rows}
    for key in list(_pose_preview):
        if key[0] == scene.as_pointer() and key not in valid:
            _pose_preview.pop(key, None)
    plans = []
    frozen_offsets = {}
    for key, settings, proxy, bone_world, proxy_world, rest in rows:
        previous = _seen.get(key)
        if previous is None:
            durable = _load_author_record(settings, proxy)
            if durable is not None:
                previous, _rest_seen[key] = durable
                _seen[key] = previous
                _unproven.discard(key)
            else:
                _unproven.add(key)
        moved = previous is not None and (not _close(proxy_world, previous[1]) or not _close(bone_world, previous[0]))
        target = proxy_world.copy()
        if key in _pose_preview:
            frozen_offsets[key] = _pose_preview[key].copy()
            target = bone_world @ frozen_offsets[key]
        elif (previous is not None and (_close(proxy_world, previous[1])
                or _proxy_only_inherited_arm_motion(previous, _rest_seen.get(key), rest, proxy_world))
                and not _close(bone_world, previous[0])
                and not _same_rest_world(_rest_seen.get(key), rest)):
            frozen_offsets[key] = previous[2].copy()
            target = bone_world @ previous[2]
        length_bake = None
        old_rest = _rest_seen.get(key)
        if previous is not None and old_rest is not None:
            old_length, new_length = _reference_length(old_rest), _reference_length(rest)
            ratio = new_length / old_length
            if abs(ratio - 1.) > _EPS and not _same_rest_world(old_rest, rest):
                from mathutils import Matrix
                # Bone matrices encode the head frame, not rest bone length.
                # Move the baseline midpoint, retaining any additive hand offset.
                offset = bone_world.inverted() @ target
                offset.translation.y += (new_length - old_length) / 2.
                target = bone_world @ offset
                length_bake = author_length_stretch(proxy, ratio)
        split = _rigid_scale(target)
        # An unapplied armature or animated pose scale remains subject to the
        # original frame validator. Do not partially repair an invalid batch.
        # A frozen positive uniform author scale is baked into proxy geometry;
        # raw bone_world still owns the follow offset and is never overwritten.
        split_bone = _rigid_scale(bone_world)
        bone_rigid = split_bone is not None and max(bone_world.to_scale()) - min(bone_world.to_scale()) <= 1.e-5 * max(bone_world.to_scale())
        if transfer and (split is None or not bone_rigid):
            return 0
        bake = None
        if split is not None and bone_rigid and not _rigid_frame(target):
            if (proxy.library or proxy.override_library or proxy.parent is None
                    or not (proxy.parent.type == 'ARMATURE'
                            or proxy.parent.get('ppf_bone_proxy_object_container') == proxy.get('ppf_armature_uuid')) or proxy.animation_data
                    or proxy.constraints or proxy.modifiers or proxy.data.shape_keys):
                if transfer:
                    return 0
            else:
                target, bake = split
        if length_bake is not None:
            if (proxy.library or proxy.override_library or proxy.animation_data
                    or proxy.constraints or proxy.modifiers or proxy.data.shape_keys):
                raise ValueError('Bone length follow requires an editable proxy mesh without modifiers, constraints, animation or shape keys')
            bake = length_bake if bake is None else bake @ length_bake
        if not _finite_invertible(target):
            return 0
        if moved or not _close(target, proxy_world) or bake is not None:
            plans.append((key, proxy, bone_world, proxy_world, target, bake))
    if not plans:
        for key, _settings, _proxy, bone_world, proxy_world, rest in rows:
            offset = frozen_offsets.get(key, bone_world.inverted() @ proxy_world)
            _seen[key] = (bone_world.copy(), proxy_world.copy(), offset)
            _rest_seen[key] = rest
            _record_row_signature(scene, key, _proxy)
            _pose_preview.pop(key, None)
            if key not in _unproven:
                _save_author_record(_settings, _proxy, bone_world, proxy_world, rest, key[2])
        return 0
    communicator.assert_bone_body_switch_idle()
    staged = []
    applied = []
    try:
        # Independent copies preserve other mesh users and permit whole-batch
        # rollback. No shape/data or transform changes before staging succeeds.
        for key, proxy, bone_world, proxy_world, target, bake in plans:
            mesh = None
            if bake is not None:
                mesh = proxy.data.copy()
                staged.append(mesh)
                mesh.transform(bake)
                mesh.update()
            applied.append((proxy, proxy.data, _representation_snapshot(proxy), target, mesh))
        for proxy, old_mesh, old_world, target, mesh in applied:
            if mesh is not None:
                proxy.data = mesh
        _assign_follow_targets(context, [(proxy, target) for proxy, _old_mesh, _old, target, _mesh in applied],
                               'Bone proxy transform cannot preserve its authored geometry; remove unsupported parent shear')
        if _transfer_deferred is None:
            communicator.bone_physical_body_changed()
        else:
            _transfer_deferred = True
    except Exception:
        for proxy, old_mesh, old_world, _target, _mesh in applied:
            proxy.data = old_mesh
            _restore_representation(proxy, old_world)
        context.view_layer.update()
        raise
    finally:
        for mesh in staged:
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
    if staged:
        from .bone_mass import invalidate_all
        invalidate_all()
        for _proxy, old_mesh, _old_world, _target, mesh in applied:
            if mesh is not None and old_mesh.users == 0:
                bpy.data.meshes.remove(old_mesh)
    for key, _settings, proxy, bone_world, _proxy_world, rest in rows:
        actual = proxy.matrix_world.copy()
        # Once any scale is baked, the new unit-scale mesh owns that scale.
        # Persist the committed raw offset, never a transient preview matrix.
        offset = bone_world.inverted() @ actual
        _seen[key] = (bone_world.copy(), actual, offset)
        _rest_seen[key] = rest
        _record_row_signature(scene, key, proxy)
        _pose_preview.pop(key, None)
        if key not in _unproven:
            _save_author_record(_settings, proxy, bone_world, actual, rest, key[2])
    return len(plans)


def flush_author_before_update(context):
    """Settle ordinary author edits before Update can replace their baseline.

    This is deliberately not transfer synchronization. Cached animation,
    identity reconstruction, Edit and active transforms retain their gates.
    A caller must hold the existing author rollback transaction around this
    call and its subsequent proxy update.
    """
    from ..ui.dynamics.group_ops import cleanup_identity_rebuilding
    from .bone_geometry_edit import active
    if (context is None or context.scene is None or _identity_rebuilding
            or cleanup_identity_rebuilding()):
        raise ValueError('Wait for Undo/Redo reconstruction before Update')
    scene = context.scene
    if (not _author_scene(scene) or _transform_active(scene) or active()
            or any(arm.type == 'ARMATURE' and arm.mode == 'EDIT' for arm in scene.objects)
            or getattr(getattr(context, 'screen', None), 'is_animation_playing', False)):
        raise ValueError('Finish author editing or cached playback before Update')
    if not _initialized:
        # Current geometry cannot prove the historic dimensions of a legacy
        # proxy after a rest edit. Only durable authority permits this seed.
        if any(_load_author_record(settings, proxy) is None
               for _key, settings, proxy, *_rest in _rows(scene)):
            raise ValueError('Author follow baseline unavailable; cannot safely update pending edits')
        _snapshot_all()
    changed = _sync(context)
    for key, _settings, _proxy, _bone_world, _proxy_world, rest in _rows(scene):
        previous = _rest_seen.get(key)
        if previous is None:
            raise ValueError('Author follow baseline unavailable; Update left the proxy unchanged')
        old_inverse, new_inverse = previous[0].inverted(), rest[0].inverted()
        if any(((old_inverse @ old) - (new_inverse @ new)).length > _EPS
               for old, new in zip(previous[1:], rest[1:])):
            raise ValueError('Pending rest edit could not be followed; Update must preserve its baseline')
    return changed


def apply_pending(context):
    """Apply from the FramePump modal TIMER, preserving authored world geometry."""
    if context is None or context.window is None:
        return 0
    if not has_pending():
        return 0
    return _sync(context)


@contextmanager
def transfer_author_transaction(context):
    """Hold proxy meshes and roll back synchronous encoder failures."""
    global _initialized, _transfer_deferred
    if _transfer_deferred is not None:
        raise RuntimeError('Nested author transfer transaction is unsupported')
    rows = list(_rows(context.scene, validate=False))
    saved = []
    meshes = {}
    # The cleanup scope starts before the first fake-user write.  If even a
    # later snapshot or a second mesh fails, every acquired hold is released.
    try:
        for _key, _settings, proxy, _bone, _world, _rest in rows:
            mesh = proxy.data
            if mesh not in meshes and not mesh.library:
                meshes[mesh] = mesh.use_fake_user
                mesh.use_fake_user = True
            saved.append((proxy, mesh, _representation_snapshot(proxy),
                          _AUTHOR_RECORD in proxy, proxy.get(_AUTHOR_RECORD)))
        states = (_seen.copy(), _rest_seen.copy(), _pose_preview.copy(),
                  _unproven.copy(), _initialized, _author_signatures.copy())
        from . import bone_mass
        mass_state = (deepcopy(bone_mass.CACHE.__dict__), deepcopy(bone_mass._indexes),
                      deepcopy(bone_mass._snapshots), bone_mass._inventory_dirty)
        _transfer_deferred = False

        def commit_notification():
            global _transfer_deferred
            if _transfer_deferred:
                from .facade import communicator
                communicator.bone_physical_body_changed()
                _transfer_deferred = False

        try:
            yield commit_notification
            if _transfer_deferred:
                raise RuntimeError('Author transfer notification was not committed')
        except BaseException:
            replaced = []
            for proxy, mesh, representation, had_record, record in saved:
                if proxy.data != mesh:
                    replaced.append(proxy.data)
                    proxy.data = mesh
                _restore_representation(proxy, representation)
                if had_record:
                    proxy[_AUTHOR_RECORD] = record
                elif _AUTHOR_RECORD in proxy:
                    del proxy[_AUTHOR_RECORD]
            context.view_layer.update()
            _seen.clear(); _seen.update(states[0])
            _rest_seen.clear(); _rest_seen.update(states[1])
            _pose_preview.clear(); _pose_preview.update(states[2])
            _unproven.clear(); _unproven.update(states[3])
            _initialized = states[4]
            _author_signatures.clear(); _author_signatures.update(states[5])
            bone_mass.CACHE.__dict__.clear()
            bone_mass.CACHE.__dict__.update(mass_state[0])
            bone_mass._indexes = mass_state[1]
            bone_mass._snapshots = mass_state[2]
            bone_mass._inventory_dirty = mass_state[3]
            for mesh in replaced:
                if mesh.users == 0:
                    bpy.data.meshes.remove(mesh)
            raise
        finally:
            _transfer_deferred = None
    finally:
        for mesh, fake_user in meshes.items():
            mesh.use_fake_user = fake_user
            if not fake_user and mesh.users == 0:
                bpy.data.meshes.remove(mesh)


def sync_for_transfer(context):
    """Flush before geometry/mass sampling; usable without a Window or timer.

    Busy, cached or actively edited scenes remain untouched. Unsupported frames
    remain untouched for the original strict encoder to refuse.
    """
    return _sync(context, transfer=True)


def _copy_preset_record(value):
    """Detach IDProperty containers while retaining any Blender ID references."""
    from array import array
    if isinstance(value,bpy.types.ID):return value
    if hasattr(value,'typecode') and hasattr(value,'to_list'):
        return array(value.typecode,value.to_list())
    if hasattr(value,'items'):return {key:_copy_preset_record(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [_copy_preset_record(item) for item in value]
    return value


def snapshot_preset_baselines(scene,arm,bones):
    """Checkpoint only baselines owned by one synchronous preset batch."""
    prefixes={(scene.as_pointer(),arm.as_pointer(),bone.name) for bone in bones}
    records=[]
    for bone in bones:
        proxy=bone.ppf_bone_physics.proxy
        if proxy is not None:
            records.append((proxy,_AUTHOR_RECORD in proxy,_copy_preset_record(proxy.get(_AUTHOR_RECORD))))
    caches=tuple({key:value for key,value in cache.items() if key[:3] in prefixes}
                 for cache in (_seen,_rest_seen,_author_signatures,_pose_preview))
    return prefixes,records,caches,{key for key in _unproven if key[:3] in prefixes}


def restore_preset_baselines(snapshot):
    """Restore exact old records/caches; do not infer or rebuild authority."""
    prefixes,records,caches,unproven=snapshot
    for proxy,had_record,record in records:
        if had_record:proxy[_AUTHOR_RECORD]=record
        elif _AUTHOR_RECORD in proxy:del proxy[_AUTHOR_RECORD]
    for cache,saved in zip((_seen,_rest_seen,_author_signatures,_pose_preview),caches):
        for key in tuple(cache):
            if key[:3] in prefixes:del cache[key]
        cache.update(saved)
    _unproven.difference_update({key for key in _unproven if key[:3] in prefixes})
    _unproven.update(unproven)


def baseline_proxy(scene, arm, bone):
    """Refresh the stored pose/offset after explicit Add / Update authoring."""
    settings = bone.ppf_bone_physics
    proxy = settings.proxy
    if proxy is None:
        return
    pose = arm.pose.bones.get(bone.name)
    if pose is None:
        return
    bone_world = arm.matrix_world @ pose.matrix
    proxy_world = proxy.matrix_world.copy()
    if _finite_invertible(bone_world) and _finite_invertible(proxy_world):
        key = (scene.as_pointer(), arm.as_pointer(), bone.name, proxy.as_pointer())
        _pose_preview.pop(key, None)
        _seen[key] = (bone_world.copy(), proxy_world.copy(), bone_world.inverted() @ proxy_world)
        _rest_seen[key] = (arm.matrix_world.copy(), arm.matrix_world @ bone.head_local,
                           arm.matrix_world @ bone.tail_local)
        _record_row_signature(scene, key, proxy)
        _unproven.discard(key)
        _save_author_record(settings, proxy, bone_world, proxy_world, _rest_seen[key], bone.name)


@persistent
def _reset_after_load(*_args):
    _reset_baseline()


@persistent
def _reset_after_undo(*_args):
    global _identity_rebuilding
    _identity_rebuilding = False
    _reset_baseline(defer=True)


@persistent
def _author_before_identity_rebuild(*_args):
    _edit_metadata.clear()
    _frame_object_observations.clear()
    _motion_dirty_scenes.clear()
    global _identity_rebuilding, _initialized, _baseline_generation
    _identity_rebuilding = True
    _initialized = False
    _baseline_generation += 1
    _cancel_initial_snapshot()
    _seen.clear()
    _unproven.clear()
    _rest_seen.clear()
    _author_signatures.clear()
    _pose_preview.clear()


def _reset_baseline(*, defer=False):
    _edit_metadata.clear()
    _frame_object_observations.clear()
    _motion_dirty_scenes.clear()
    global _initialized, _baseline_generation
    _baseline_generation += 1
    _initialized = False
    _seen.clear()
    _unproven.clear()
    _rest_seen.clear()
    _pose_preview.clear()
    if defer:
        _schedule_initial_snapshot()
        return
    try:
        _snapshot_all()
        _seed_current_frame_observations()
        _cancel_initial_snapshot()
    except (AttributeError, RuntimeError, TypeError):
        _schedule_initial_snapshot()


def register():
    global _active, _initialized, _identity_rebuilding, _baseline_generation
    _active = True
    _initialized = False
    _identity_rebuilding = False
    _baseline_generation += 1
    _seen.clear()
    _author_signatures.clear()
    _frame_object_observations.clear()
    _motion_dirty_scenes.clear()
    _unproven.clear()
    _rest_seen.clear()
    _pose_preview.clear()
    for collection, fn in ((bpy.app.handlers.load_post, _reset_after_load),
                           (bpy.app.handlers.depsgraph_update_post, _mark_author_motion),
                           (bpy.app.handlers.frame_change_post, _observe_frame_object_basis),
                           (bpy.app.handlers.undo_pre, _author_before_identity_rebuild),
                           (bpy.app.handlers.redo_pre, _author_before_identity_rebuild),
                           (bpy.app.handlers.undo_post, _reset_after_undo),
                           (bpy.app.handlers.redo_post, _reset_after_undo)):
        for old in list(collection):
            if old.__name__ == fn.__name__:
                collection.remove(old)
        collection.append(fn)
    _schedule_initial_snapshot()


def unregister():
    global _active, _initialized
    _active = False
    _initialized = False
    _cancel_initial_snapshot()
    for collection in (bpy.app.handlers.load_post, bpy.app.handlers.frame_change_post,
                       bpy.app.handlers.depsgraph_update_post, bpy.app.handlers.undo_pre,
                       bpy.app.handlers.redo_pre, bpy.app.handlers.undo_post,
                       bpy.app.handlers.redo_post):
        for fn in list(collection):
            if fn.__name__ in ('_reset_after_load', '_reset_after_undo', '_author_before_identity_rebuild',
                               '_observe_frame_object_basis', '_mark_author_motion'):
                collection.remove(fn)
    _seen.clear()
    _unproven.clear()
    _rest_seen.clear()
    _author_signatures.clear()
    _frame_object_observations.clear()
    _motion_dirty_scenes.clear()
    _pose_preview.clear()



