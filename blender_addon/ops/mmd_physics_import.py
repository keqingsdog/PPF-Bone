"""Import selected MMD bodies as PPF bone proxies, with explicit approximations."""
import math
import json
import struct
import uuid
from contextlib import contextmanager
import bpy
from bpy.props import FloatProperty, StringProperty
from mathutils import Matrix, Vector
from . import bone_proxy
from ..models.bone_bindings import bone_world_matrix
from .. import i18n as _i18n


def _root(obj):
    while obj:
        if getattr(obj, 'mmd_type', None) == 'ROOT':
            return obj
        obj = obj.parent
    return None


def _frame(matrix):
    if abs(matrix.to_3x3().determinant()) < 1.e-12:
        raise ValueError('MMD transform is singular')
    return Matrix.Translation(matrix.translation) @ matrix.to_quaternion().to_matrix().to_4x4()


def _mmd_joint_limits(joint):
    """Read MMD Tools' already converted Blender-local bounds (metres/radians)."""
    c = joint.rigid_body_constraint
    result = {}
    for kind in ('lin', 'ang'):
        enabled, lower, upper = [], [], []
        for axis in 'xyz':
            prefix = f'{kind}_{axis}'
            lo = float(getattr(c, f'limit_{prefix}_lower'))
            hi = float(getattr(c, f'limit_{prefix}_upper'))
            use = bool(getattr(c, f'use_limit_{prefix}'))
            if not math.isfinite(lo) or not math.isfinite(hi):
                raise ValueError('MMD 关节限制无效：' + joint.name)
            enabled.append(use and lo <= hi)
            lower.append(lo)
            upper.append(hi)
        result[kind] = (tuple(enabled), tuple(lower), tuple(upper))
    return result


def _mmd_joint_record(joint, bone_a, bone_b, frames, joint_id):
    """Build an ordered independent joint row after both proxies exist."""
    limits = _mmd_joint_limits(joint)

    def axes(values):
        enabled, lower, upper = values
        result = {}
        for i, axis in enumerate('xyz'):
            if not enabled[i]:
                result[axis] = {'mode': 'free'}
            elif lower[i] == upper[i]:
                result[axis] = {'mode': 'locked', 'value': lower[i]}
            else:
                result[axis] = {'mode': 'limited', 'lower': lower[i], 'upper': upper[i]}
        return result

    def local_frame(bone, world):
        local = _frame(bone.ppf_bone_physics.proxy.matrix_world).inverted() @ world
        return {'translation_m': list(local.translation),
                'quaternion_wxyz': list(local.to_quaternion())}

    return {'joint_uuid': str(joint_id),
            'body_a_uuid': bone_a.ppf_bone_physics.body_uuid,
            'body_b_uuid': bone_b.ppf_bone_physics.body_uuid,
            'frame_a': local_frame(bone_a, frames[0]),
            'frame_b': local_frame(bone_b, frames[1]),
            'translation': axes(limits['lin']), 'rotation': axes(limits['ang']),
            'linear_spring_N_per_m': list(joint.mmd_joint.spring_linear),
            'angular_spring_Nm_per_rad': list(joint.mmd_joint.spring_angular)}


def _pose_delta(arm, bone_name, deps):
    evaluated = arm.evaluated_get(deps)
    pose = evaluated.pose.bones.get(bone_name)
    if pose is None:
        raise ValueError('MMD 骨骼不存在：' + bone_name)
    world = evaluated.matrix_world
    return world @ pose.matrix @ pose.bone.matrix_local.inverted() @ world.inverted()


def _body_follows_bone(obj, arm, bone_name):
    parent_follow = obj.parent_type == 'BONE'
    if parent_follow and (obj.parent != arm or obj.parent_bone != bone_name):
        raise ValueError('MMD 刚体有不支持的骨骼父级：' + obj.name)
    follows = parent_follow
    ancestor = obj.parent
    while ancestor and ancestor != arm:
        if ancestor.parent_type == 'BONE' and ancestor.parent == arm:
            raise ValueError('MMD 刚体经间接骨骼父级跟随，无法安全导入：' + obj.name)
        if any(not c.mute and c.influence != 0 and getattr(c, 'target', None) == arm for c in ancestor.constraints):
            raise ValueError('MMD 刚体经上级约束跟随骨架，无法安全导入：' + obj.name)
        ancestor = ancestor.parent
    for constraint in obj.constraints:
        if constraint.mute or constraint.influence == 0:
            continue
        if (constraint.type == 'CHILD_OF' and constraint.name == 'mmd_tools_rigid_parent'
                and constraint.target == arm and constraint.subtarget == bone_name
                and constraint.influence == 1. and not follows and constraint.is_valid
                and constraint.owner_space == 'WORLD' and constraint.target_space == 'WORLD'
                and all(getattr(constraint, 'use_' + component + '_' + axis)
                        for component in ('location', 'rotation', 'scale') for axis in 'xyz')):
            follows = True
        else:
            raise ValueError('MMD 刚体有无法安全解释的活动约束：' + obj.name)
    return follows


def _geometry(deps, obj, world=None):
    evaluated = obj.evaluated_get(deps)
    mesh = evaluated.to_mesh()
    try:
        world = evaluated.matrix_world.copy() if world is None else world
        frame = _frame(world)
        transform = frame.inverted() @ world
        vertices = [tuple(transform @ v.co) for v in mesh.vertices]
        faces = [tuple(p.vertices) for p in mesh.polygons]
        if not vertices or not faces:
            raise ValueError('MMD body has no mesh geometry: ' + obj.name)
        return vertices, faces, frame
    finally:
        evaluated.to_mesh_clear()


@contextmanager
def _helper_scope(context, keep=False):
    from . import mmd_helper_bones
    arm, bones = bone_proxy.selected_bones(context)
    from ..core.bone_geometry_edit import preflight
    preflight(context)
    root = _root(arm)
    if not root:
        raise ValueError('Active armature is not inside an MMD model')
    if root.mmd_root.is_built:
        raise ValueError('MMD 物理已构建；请先在 MMD Tools 中清理物理，再导入 PPF')
    grouped = {}
    names = {b.name for b in bones}
    for obj in context.scene.objects:
        if _root(obj) == root and getattr(obj, 'mmd_type', '') == 'RIGID_BODY' and obj.mmd_rigid.bone in names:
            grouped.setdefault(obj.mmd_rigid.bone, []).append(obj)
    extras = []
    for name, objects in grouped.items():
        if len(objects) < 2:
            continue
        if any(str(obj.mmd_rigid.type) != '0' for obj in objects):
            raise ValueError('同骨骼多个动态或混合类型 MMD 刚体暂不支持：' + name)
        objects.sort(key=lambda obj: obj.name)
        original = arm.data.bones[name].ppf_bone_physics.custom_object
        primary = original if original in objects else objects[0]
        extras.extend((name, obj) for obj in objects if obj != primary)
    mapping = {}
    pending = []
    for name, obj in extras:
        existing = mmd_helper_bones.find(arm, name, obj)
        if existing:
            mapping[obj] = existing.name
        else:
            pending.append((name, obj))
    created = mmd_helper_bones.create(context, arm, pending) if pending else {}
    mapping.update(created)
    success = False
    try:
        yield mapping
        success = True
    finally:
        if created and (not keep or not success):
            mmd_helper_bones.remove(context, arm, list(created.values()))


def _plan(context, helpers=None):
    arm, bones = bone_proxy.selected_bones(context)
    root = _root(arm)
    if not root:
        raise ValueError('Active armature is not inside an MMD model')
    if root.mmd_root.is_built:
        raise ValueError('MMD 物理已构建；请先在 MMD Tools 中清理物理，再导入 PPF')
    names = {b.name for b in bones}
    helpers = helpers or {}
    original_names = set(names)
    bones.extend(arm.data.bones[n] for n in helpers.values())
    bodies = {}
    joints = []
    for obj in context.scene.objects:
        if _root(obj) != root:
            continue
        typ = getattr(obj, 'mmd_type', '')
        if typ == 'RIGID_BODY' and obj.mmd_rigid.bone in names:
            name = helpers.get(obj, obj.mmd_rigid.bone)
            if name in bodies:
                raise ValueError('多个刚体缺少独立辅助骨骼：' + name)
            bodies[name] = obj
        elif typ == 'JOINT' and obj.rigid_body_constraint:
            joints.append(obj)
    missing = original_names - bodies.keys()
    bones = [b for b in bones if b.name in bodies]
    if not bones:
        return arm, [], {}, [], 0, len(missing)
    existing = []
    for bone in bones:
        s = bone.ppf_bone_physics
        if s.proxy or s.enabled:
            if not (s.proxy and s.proxy.get('ppf_mmd_source') and s.custom_object == bodies[bone.name]):
                raise ValueError('Existing PPF proxy conflicts with MMD body: ' + bone.name)
            existing.append(bone.name)
    bones = [b for b in bones if b.name not in existing]
    by_obj = {obj: name for name, obj in bodies.items()}
    rows = {}
    deps = context.evaluated_depsgraph_get()
    for bone in bones:
        obj = bodies[bone.name]; rigid = obj.mmd_rigid
        source_bone = rigid.bone
        group = int(rigid.collision_group_number)
        if not 0 <= group < 16:
            raise ValueError('Invalid MMD collision group: ' + obj.name)
        typ = str(rigid.type)
        if typ not in {'0', '1', '2'}:
            raise ValueError('Unknown MMD body type: ' + obj.name)
        # Imported PMX parent connections are enabled through pmx_parent_joint_enabled.
        # Keep the generated PPF fallback restricted to isolated dynamic bodies.
        follow_parent = typ in {'1', '2'} and bone.parent is not None
        settings = dict(shape='CUSTOM', custom_object=obj, physical_model='RIGID', new_binding_physical_model='RIGID', auto_follow_parent=follow_parent, use_hierarchy_joint=True,
            root_mode='KINEMATIC' if typ == '0' else 'FREE', root_motion_source='POSE',
            collision_filter_enabled=True,
            collision_group_layers=tuple(i == group for i in range(16)),
            collision_mask_layers=tuple(not v for v in rigid.collision_group_mask))
        if obj.rigid_body and obj.rigid_body.mass > 0:
            settings.update(mass_mode='MASS', target_mass_kg=obj.rigid_body.mass)
        world = obj.evaluated_get(deps).matrix_world.copy()
        if not _body_follows_bone(obj, arm, source_bone):
            world = _pose_delta(arm, source_bone, deps) @ world
        rows[bone.name] = {'settings': settings, 'geometry': _geometry(deps, obj, world),
                           'source': obj.name, 'hybrid': typ == '2', 'joints': joints}
    plans = []
    for joint in joints:
        c = joint.rigid_body_constraint
        if c.object1 not in by_obj or c.object2 not in by_obj:
            continue
        a, b = by_obj[c.object1], by_obj[c.object2]
        if a == b:
            continue
        if any(not math.isfinite(float(x)) or x < 0 for x in (*joint.mmd_joint.spring_linear, *joint.mmd_joint.spring_angular)):
            raise ValueError('MMD 关节弹簧刚度必须为有限非负值：' + joint.name)
        ba, bb = arm.data.bones[a], arm.data.bones[b]
        if any(not con.mute and con.influence != 0 for con in joint.constraints):
            raise ValueError('MMD 关节有无法安全解释的活动约束：' + joint.name)
        node = joint
        while node:
            if node.parent_type == 'BONE':
                raise ValueError('MMD 关节经骨骼父级跟随，无法安全导入：' + joint.name)
            if node != joint and any(not con.mute and con.influence != 0 and getattr(con, 'target', None) == arm for con in node.constraints):
                raise ValueError('MMD 关节经上级约束跟随骨架，无法安全导入：' + joint.name)
            node = node.parent
        joint_world = joint.evaluated_get(deps).matrix_world.copy()
        frames = (_frame(_pose_delta(arm, c.object1.mmd_rigid.bone, deps) @ joint_world),
                  _frame(_pose_delta(arm, c.object2.mmd_rigid.bone, deps) @ joint_world))
        child = bb if bb.parent == ba else ba if ba.parent == bb else None
        plans.append(('native', joint, a, b, child.name if child else None, frames))
    return arm, bones, rows, plans, len(existing), len(missing)


def _parent_default_snapshot(context, settings):
    from ..core.bone_preset_apply import _read, _capture_curves
    from .restoring_curve_operators import snapshot_rna
    fields = (
        'anchor_stiffness', 'angular_stiffness', 'angular_damping', 'soft_limits_enabled',
        'soft_lower', 'soft_upper', 'soft_stiffness', 'soft_limit_mode', 'swing_cone_half_angle',
        'swing_cone_stiffness', 'soft_limit_policy', 'soft_limit_frequency_hz', 'soft_limit_strength',
        'collide_connected', 'auto_follow_parent', 'joint_mode', 'hinge_twist_mode', 'hinge_lower',
        'hinge_upper', 'use_auto_frames', 'joint_axis_euler', 'angular_restoring_mode',
        'swing_restoring_stiffness', 'swing_restoring_damping')
    fields += ('parent_frame_translation', 'parent_frame_quaternion', 'child_frame_translation', 'child_frame_quaternion')
    groups = ('drive_twist_x', 'drive_swing_y', 'drive_swing_z', 'd6_translation_x',
              'd6_translation_y', 'd6_translation_z', 'd6_rotation_x', 'd6_rotation_y', 'd6_rotation_z')
    curves = _capture_curves(context.scene, settings.joint_uuid)
    if not any(curves['channels'].values()):
        curves = None
    return json.dumps({'fields': _read(settings, fields),
                       'groups': {k: snapshot_rna(getattr(settings, k)) for k in groups},
                       'curves': curves}, sort_keys=True)


def _mark_parent_defaults(context, arm, bones, rows):
    for bone in bones:
        if bone.parent is None:
            continue
        source = rows[bone.name]['settings']['custom_object']
        direct = []
        for joint in rows[bone.name]['joints']:
            c = getattr(joint, 'rigid_body_constraint', None)
            if getattr(joint, 'mmd_type', '') != 'JOINT' or c is None:
                continue
            other = c.object2 if c.object1 == source else c.object1 if c.object2 == source else None
            if other and getattr(other, 'mmd_type', '') == 'RIGID_BODY' and other.mmd_rigid.bone == bone.parent.name:
                direct.append(joint)
        if direct:
            proxy = bone.ppf_bone_physics.proxy
            proxy['ppf_mmd_pending_parent_sources'] = {str(i): j for i, j in enumerate(direct)}
            proxy['ppf_mmd_pending_parent_snapshot'] = _parent_default_snapshot(context, bone.ppf_bone_physics)


def _pending_parent_source(bone, joint):
    proxy = bone.ppf_bone_physics.proxy
    if not proxy:
        return False
    sources = proxy.get('ppf_mmd_pending_parent_sources')
    return bool(sources and joint in sources.values())


def _apply(context, arm, bones, rows, plans):
    from ..core import cross_links
    stored = json.loads(arm.get('ppf_mmd_joints_v1', '[]'))
    sources = dict(arm.get('ppf_mmd_joint_sources_v1', {}).items())
    source_ids = {obj: key for key, obj in sources.items() if obj is not None}
    indices = {r['joint_uuid']: i for i, r in enumerate(stored)}
    additions = []
    disable_parent = []
    migrated_pairs = set()
    old_links = cross_links._records(arm)
    for _kind, joint, a, b, child_name, frames in plans:
        ba, bb = arm.data.bones[a], arm.data.bones[b]
        pair = (ba.ppf_bone_physics.bone_uuid, bb.ppf_bone_physics.bone_uuid)
        joint_id = source_ids.get(joint)
        record = _mmd_joint_record(joint, ba, bb, frames, joint_id or str(uuid.uuid4()))
        if joint_id in indices:
            stored[indices[joint_id]] = record
            continue
        child = arm.data.bones[child_name] if child_name else None
        pending = bool(child and _pending_parent_source(child, joint))
        unchanged = bool(pending and child.ppf_bone_physics.proxy.get('ppf_mmd_pending_parent_snapshot') == _parent_default_snapshot(context, child.ppf_bone_physics))
        if child and child_name not in {bone.name for bone in bones} and child.ppf_bone_physics.use_hierarchy_joint and not unchanged:
            raise ValueError('骨骼“' + child_name + '”已有层级关节，无法判断是否包含手工修改。请先备份场景；如需保留修改，请保留旧设置并暂不导入该关节；如需按 MMD 原关节重建，请在 PPF 骨骼物理中关闭该骨骼的层级关节后重新导入。')
        if pending and not child.ppf_bone_physics.use_hierarchy_joint:
            disable_parent.append(child.ppf_bone_physics)
        migrated_pairs.add(frozenset(pair))
        additions.append(record)
        sources[record['joint_uuid']] = joint
        source_ids[joint] = record['joint_uuid']
    for bone in bones:
        proxy = bone.ppf_bone_physics.proxy
        proxy['ppf_mmd_source'] = rows[bone.name]['source']
        proxy['ppf_mmd_source_object'] = rows[bone.name]['settings']['custom_object']
    retained = [r for r in old_links if not (r.get('source') == 'MMD_APPROX' and
                frozenset((r['bone_a_uuid'], r['bone_b_uuid'])) in migrated_pairs)]
    old_joint_json = arm.get('ppf_mmd_joints_v1')
    old_sources = dict(arm.get('ppf_mmd_joint_sources_v1', {}).items())
    old_parent = [(s, s.pmx_parent_joint_enabled, s.is_property_set('pmx_parent_joint_enabled')) for s in disable_parent]
    try:
        for settings, _, _ in old_parent:
            settings.pmx_parent_joint_enabled = False
        if len(retained) != len(old_links):
            cross_links._replace(arm, retained)
        arm['ppf_mmd_joints_v1'] = json.dumps(stored + additions, ensure_ascii=False)
        arm['ppf_mmd_joint_sources_v1'] = sources
    except Exception:
        for settings, value, was_set in old_parent:
            if was_set: settings.pmx_parent_joint_enabled = value
            else: settings.property_unset('pmx_parent_joint_enabled')
        if old_joint_json is None:
            if 'ppf_mmd_joints_v1' in arm: del arm['ppf_mmd_joints_v1']
        else:
            arm['ppf_mmd_joints_v1'] = old_joint_json
        if old_sources:
            arm['ppf_mmd_joint_sources_v1'] = old_sources
        elif 'ppf_mmd_joint_sources_v1' in arm:
            del arm['ppf_mmd_joint_sources_v1']
        if len(retained) != len(old_links):
            cross_links._replace(arm, old_links)
        raise
    return {'hierarchy': 0, 'cross': len(additions), 'skipped': len(plans) - len(additions)}

def _mass_analysis(context, rows):
    """Use the same strict closed-shell measurement as the proxy transaction."""
    from ..core.bone_mass import measure_volume
    from ..core.bone_mass_math import DENSITY_MIN, DENSITY_MAX
    if float(context.scene.unit_settings.scale_length) != 1.:
        raise ValueError('骨骼质量使用 SI 单位；场景单位缩放必须为 1')
    lower, upper = 0., float('inf')
    details = []
    for name, row in rows.items():
        settings = row['settings']
        if settings.get('mass_mode') != 'MASS':
            continue
        vertices, faces, frame = row['geometry']
        mesh = bpy.data.meshes.new('PPF_MMD_mass_preflight')
        try:
            mesh.from_pydata(vertices, [], faces)
            mesh.update()
            volume = measure_volume(mesh, frame, edit_mesh=False)['volume_m3']
        finally:
            bpy.data.meshes.remove(mesh)
        mass = float(settings['target_mass_kg'])
        density = mass / volume
        if not math.isfinite(density) or density <= 0:
            raise ValueError(f'{name}：质量或体积无效')
        lower = max(lower, DENSITY_MIN / density)
        upper = min(upper, DENSITY_MAX / density)
        details.append((name, mass, volume, density))
    return lower, upper, details


def _mass_message(details, multiplier=1.):
    invalid = []
    for name, mass, volume, density in details:
        try:
            stored = struct.unpack('<f', struct.pack('<f', mass * multiplier))[0]
        except OverflowError:
            stored = float('inf')
        actual = stored / volume
        if not math.isfinite(actual) or not .01 <= actual <= 10000.:
            invalid.append(f'{name}: 质量 {mass:.6g} kg，体积 {volume:.6g} m³，密度 {actual:.6g} kg/m³')
    return '\n'.join(invalid)


def _suggest_mass_multiplier(arm, details, lower, upper):
    from statistics import median

    lo, hi = max(lower, 1.e-20), min(upper, 1.e20)
    root = _root(arm)
    if root is not None and 'mmd_tools_version' in root:
        # MMD Tools initializes the root display size to import_scale / 0.2.
        # It is editable, so only use it as a hint and validate actual volumes.
        import_scale = float(root.empty_display_size) * .2
        volume_scale = abs(root.matrix_world.to_3x3().determinant())
        candidate = import_scale ** 3 * volume_scale
        if lo <= candidate <= hi:
            stored = struct.unpack('<f', struct.pack('<f', candidate))[0]
            if not _mass_message(details, stored):
                return stored, f'根据 MMD 模型尺度 {import_scale:.6g} 估算质量倍率'

    # Choose a central density for the batch while preserving mass ratios.
    reference_density = 1000.
    central_density = math.exp(median(math.log(row[3]) for row in details))
    preferred = reference_density / central_density
    multiplier = min(max(preferred, lo * 1.001), hi / 1.001)
    if not lo <= multiplier <= hi:
        multiplier = math.sqrt(lo * hi)
    return multiplier, '根据代理体积估算质量倍率（参考密度 1000 kg/m³）'


def _selection_identity(context, arm, rows):
    _arm, selected = bone_proxy.selected_bones(context)
    return (getattr(context.scene, 'session_uid', context.scene.as_pointer()), arm.as_pointer(), arm.data.as_pointer(),
            tuple(sorted(b.name for b in selected)), tuple(sorted((name, row['settings']['custom_object'].as_pointer())
                                           for name, row in rows.items())))


class PPF_OT_ImportMMDPhysics(bpy.types.Operator):
    bl_idname = 'ppf.import_mmd_physics'
    bl_label = '导入选中骨骼的 MMD 物理'
    bl_options = {'REGISTER', 'UNDO'}

    mass_multiplier: FloatProperty(name='统一质量倍率', default=1., min=1.e-20, max=1.e20, precision=8, options={'SKIP_SAVE'},
        description='按同一倍率调整本次导入的 MMD 质量；源刚体和代理几何不变')
    mass_warning: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def invoke(self, context, event):
        from ..core.bone_mass_math import DensityOutOfRange
        try:
            return self._run(context, preflight=False)
        except DensityOutOfRange as error:
            failure = error
        except (ValueError, RuntimeError, TypeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        try:
            with _helper_scope(context) as helpers:
                arm, _bones, rows, _plans, _existing, _missing = _plan(context, helpers)
                if failure.armature_name != arm.name or failure.bone_name not in rows:
                    raise ValueError(str(failure))
                lower, upper, details = _mass_analysis(context, rows)
                invalid = _mass_message(details, self.mass_multiplier)
                if not invalid:
                    raise ValueError(str(failure))
                if lower > upper or upper < 1.e-20 or lower > 1.e20:
                    raise ValueError('选中刚体不存在合规的统一质量倍率；请分别调整质量或体积。\n' + invalid)
                suggested, basis = _suggest_mass_multiplier(arm, details, lower, upper)
                self.mass_multiplier = suggested
                if _mass_message(details, self.mass_multiplier):
                    raise ValueError('统一质量倍率在数值精度限制下不可行；请分别调整质量或体积。\n' + invalid)
                self.mass_warning = invalid + f'\n{basis}\n合规倍率范围：{lower:.8g}–{upper:.8g}'
                self._mmd_selection_identity = _selection_identity(context, arm, rows)
            return context.window_manager.invoke_props_dialog(self, width=640)
        except (ValueError, RuntimeError, TypeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

    def draw(self, context):
        layout = self.layout
        layout.label(text='以下刚体密度超出 0.01–10000 kg/m³：', icon='ERROR')
        for line in self.mass_warning.splitlines():
            layout.label(text=line)
        layout.label(text='确认后统一调整质量，保留相对质量；重量和运动将改变。')
        layout.label(text='源 MMD 刚体与代理几何不变。')
        layout.prop(self, 'mass_multiplier')

    def _run(self, context, preflight):
        with _helper_scope(context, keep=True) as helpers:
            return self._run_impl(context, preflight, helpers)

    def _run_impl(self, context, preflight, helpers):
        arm, bones, rows, plans, existing, missing = _plan(context, helpers)
        expected = getattr(self, '_mmd_selection_identity', None)
        if expected is not None and _selection_identity(context, arm, rows) != expected:
            raise ValueError('确认期间骨架、选中骨骼或 MMD 刚体已变化；请重新导入')
        if not rows:
            result = _apply(context, arm, [], {}, plans)
            self.report({'INFO'}, f"{existing} existing bodies; {result['cross']} joints added, {result['skipped']} already present")
            return {'FINISHED'}
        if preflight:
            lower, upper, details = _mass_analysis(context, rows)
            invalid = _mass_message(details, self.mass_multiplier)
            if invalid:
                if lower > upper:
                    raise ValueError('选中刚体不存在合规的统一质量倍率；请分别调整质量或体积。\n' + invalid)
                raise ValueError('密度超出 0.01–10000 kg/m³：\n' + invalid + f'\n合规统一质量倍率：{lower:.8g}–{upper:.8g}')
        for row in rows.values():
            settings = row['settings']
            if settings.get('mass_mode') == 'MASS':
                settings['target_mass_kg'] *= self.mass_multiplier
        result = {}
        def after_apply(a, bs):
            _mark_parent_defaults(context, a, bs, rows)
            result.update(_apply(context, a, bs, rows, plans))
        authoring = {name: {'settings': row['settings'], 'geometry': row['geometry']} for name, row in rows.items()}
        from ..core import cross_links
        old_link_count = len(arm.ppf_cross_links)
        try:
            bone_proxy.update_proxies(context, all_bones=bool(helpers), rebuild=True, authoring_rows=authoring, after_apply=after_apply)
        except Exception:
            while len(arm.ppf_cross_links) > old_link_count:
                arm.ppf_cross_links.remove(len(arm.ppf_cross_links) - 1)
            raise
        hybrid = sum(row['hybrid'] for row in rows.values())
        self.report({'INFO'}, f"MMD: {len(bones)} bodies, {result['cross']} joints, {result['skipped']} existing joints, {hybrid} hybrid, {existing} existing bodies, {missing} unmatched bones; mass multiplier {self.mass_multiplier:.8g}")
        return {'FINISHED'}
    def execute(self, context):
        from ..core.bone_mass_math import DensityOutOfRange
        try:
            return self._run(context, preflight=getattr(self, '_mmd_selection_identity', None) is not None)
        except DensityOutOfRange as exc:
            try:
                with _helper_scope(context) as helpers:
                    arm, _bones, rows, _plans, _existing, _missing = _plan(context, helpers)
                    if exc.armature_name != arm.name or exc.bone_name not in rows:
                        raise ValueError(str(exc))
                    lower, upper, details = _mass_analysis(context, rows)
                    invalid = _mass_message(details, self.mass_multiplier)
                    if lower > upper:
                        raise ValueError('选中刚体不存在合规的统一质量倍率；请分别调整质量或体积。\n' + invalid)
                    raise ValueError('密度超出 0.01–10000 kg/m³：\n' + invalid + f'\n合规统一质量倍率：{lower:.8g}–{upper:.8g}')
            except (ValueError, RuntimeError, TypeError) as diagnostic:
                self.report({'ERROR'}, str(diagnostic))
                return {'CANCELLED'}
        except (ValueError, RuntimeError, TypeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


def register():
    _i18n.register_class(PPF_OT_ImportMMDPhysics)


def unregister():
    bpy.utils.unregister_class(PPF_OT_ImportMMDPhysics)

