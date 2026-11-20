"""Undoable chain material/frequency presets and direct restoring controls."""
from .. import i18n as _ppf_i18n
import math

import bpy
from ..i18n.props import EnumProperty,FloatProperty

from ..models import restoring_curves as curves
from ..models.bone_chain_presets import PRESETS as CHAIN_PRESETS
from ..models.bone_bindings import active_armature, active_bone, validate_proxy_owner, sync_owned_material
from ..models.bone_material_presets import PRESETS as MATERIAL_PRESETS, preset_values
from ..ops import restoring_curves as curve_author
from .bone_proxy import SCOPES, selected_bones, _owner_animation_paths
from .restoring_curve_operators import read_state, write_state
from . import restoring_curve_presets


_CHANNELS = ('twist_x', 'swing_y', 'swing_z', 'anchor')
_DIRECT_CHANNELS = ('twist_x', 'swing_y', 'swing_z')
_LINEAR_NODES = {
    'twist_x': {'domain': 'ODD', 'nodes': [[0.0, 0.0], [1.0, 1.0]]},
    'swing_y': {'domain': 'ODD', 'nodes': [[0.0, 0.0], [1.0, 1.0]]},
    'swing_z': {'domain': 'ODD', 'nodes': [[0.0, 0.0], [1.0, 1.0]]},
    'anchor': {'domain': 'RADIAL', 'nodes': [[0.0, 0.0], [1.0, 1.0]]},
}


def frequency_to_strength(frequency_hz):
    """Invert f=10*sqrt(strength); zero frequency denotes disabled."""
    value = float(frequency_hz)
    if not math.isfinite(value) or value < 0:
        raise ValueError('Invalid restoring frequency')
    return (value / 10.0) ** 2


def strength_to_frequency(strength):
    """Convert dimensionless restoring strength to authored natural frequency."""
    value = float(strength)
    if not math.isfinite(value) or value < 0:
        raise ValueError('Restoring strength must be finite and nonnegative')
    return 10.0 * math.sqrt(value)


def _joint_parent_status(bone, arm=None):
    """Whether this author bone exports a real compliant parent joint."""
    parent = bone.parent
    if parent is None:
        return False, 'bone has no parent'
    parent_settings = parent.ppf_bone_physics
    settings = bone.ppf_bone_physics
    if arm is not None:
        from ..core.encoder.pmx_joints import parent_joint_pair, parent_joint_pairs
        if parent_joint_pair(bone, parent_joint_pairs(arm)):
            return False, 'parent uses an imported MMD joint'
    if not settings.use_hierarchy_joint:
        return False, 'parent joint is disabled'
    if parent_settings.enabled:
        if not parent_settings.proxy or not parent_settings.body_uuid:
            return False, 'enabled parent has no physical proxy/body'
        return True, ''
    if not getattr(settings, 'auto_follow_parent', False):
        return False, 'nonphysical parent is not followed'
    if settings.root_mode != 'FREE' or settings.root_control_events:
        return False, 'nonphysical-parent joints require a free root without control events'
    return True, ''


def _targets_for_bone(arm, bone):
    settings = bone.ppf_bone_physics
    valid_parent, _reason = _joint_parent_status(bone, arm)
    if not valid_parent:
        return []
    if settings.joint_mode in {'HINGE', 'D6'}:
        raise ValueError(f'{bone.name}: chain restoring presets require a compliant joint; '
                         f'{settings.joint_mode} is a hard joint')
    if not settings.joint_uuid:
        raise ValueError(f'{bone.name}: a real parent joint is available but has no joint UUID')
    if bone.parent.ppf_bone_physics.enabled:
        validate_proxy_owner(arm, bone.parent)
    curves.canonical_uuid(settings.joint_uuid, 'joint_uuid')
    return [{'kind': 'JOINT', 'joint_uuid': settings.joint_uuid, 'channel': channel}
            for channel in (_CHANNELS[:3] if settings.joint_mode == 'BALL_FIXED' else _CHANNELS)]


def _linear_frequency_document(targets, settings_by_channel):
    tables, bindings = [], []
    for target in targets:
        channel = target['channel']
        template = _LINEAR_NODES[channel]
        table_id = f"linear_{channel}_{target['joint_uuid'].replace('-', '')[:16]}"
        table = {'table_id': table_id, 'domain': template['domain'],
                 'tail': 'LINEAR', 'nodes': [list(node) for node in template['nodes']]}
        spec = settings_by_channel[channel]
        if spec.get('zero_effort', False):
            table['nodes'] = [[x, 0.0] for x, _f in table['nodes']]
        tables.append(table)
        frequency = max(float(spec['frequency_hz']), 1.0e-6)
        bindings.append({
            'target': dict(target), 'enabled': bool(spec['enabled']),
            'table_id': table_id, 'mode': 'AUTO_FREQUENCY',
            'frequency_hz': frequency, 'damping_ratio': float(spec['damping_ratio']),
            'coordinate_scale': .001 if channel == 'anchor' else 1.0,
            'mass_policy': 'LOCAL_REST_V1',
        })
    return curves.validate_author_map({'version': 1, 'tables': tables,
                                       'bindings': bindings, 'stitch_pairs': []})


def _apply_targets(scene, targets, channel_settings):
    if not targets:
        return None
    author_map = _linear_frequency_document(targets, channel_settings)
    state = read_state(scene)
    if state['mode'] == 'LEGACY' and state['document'] is not None:
        raise ValueError('Dormant legacy curves exist; explicitly convert the restoring mode before applying controls to avoid activating unrelated curves')
    preset = restoring_curve_presets.named_preset('Bone restoring controls', author_map)
    remap = [{'source': target, 'destination': target} for target in targets]
    curve_author.apply_preset(state, preset, remap, targets, [])
    write_state(scene, state)
    return state


def _active_physical_bone(context):
    arm, bone = active_armature(context), active_bone(context)
    if arm is None or bone is None or context.mode not in {'OBJECT', 'POSE'}:
        raise ValueError('Select an active physical bone in Object or Pose mode')
    settings = bone.ppf_bone_physics
    if not settings.enabled or not settings.proxy:
        raise ValueError(f'{bone.name}: create and enable its physical body first')
    validate_proxy_owner(arm, bone)
    valid_parent, reason = _joint_parent_status(bone, arm)
    if not valid_parent:
        raise ValueError(f'{bone.name}: no real parent joint is available ({reason})')
    if bone.parent.ppf_bone_physics.enabled:
        validate_proxy_owner(arm, bone.parent)
    if settings.joint_mode in {'HINGE', 'D6'}:
        raise ValueError(f'{bone.name}: direct restoring controls require a compliant parent joint')
    curves.canonical_uuid(settings.joint_uuid, 'joint_uuid')
    curves.canonical_uuid(settings.bone_uuid, 'bone_uuid')
    return arm, bone


def _read_direct_values(context, bone):
    """Return authored normalized controls only for compatible linear AUTO curves."""
    state = read_state(context.scene)
    if state['mode'] == 'LEGACY':
        return {'bend_strength': frequency_to_strength(curve_author.COMMON_ANGULAR_FREQUENCY_HZ),
                'twist_strength': frequency_to_strength(curve_author.COMMON_ANGULAR_FREQUENCY_HZ),
                'damping_ratio': .7, 'defaults': True}
    document = state['document']
    if document is None:
        raise ValueError('No curve document is applied; defaults are drafts until Apply')
    by_target = {curves.target_key(row['target']): row for row in document['bindings']}
    tables = {row['table_id']: row for row in document['tables']}
    values = {}
    for channel in _DIRECT_CHANNELS:
        key = curves.target_key({'kind': 'JOINT',
                                 'joint_uuid': bone.ppf_bone_physics.joint_uuid,
                                 'channel': channel})
        binding = by_target.get(key)
        if binding is None:
            raise ValueError(f'{channel}: no saved restoring curve; use the professional editor to explicitly define missing channels')
        table = tables[binding['table_id']]
        template = _LINEAR_NODES[channel]
        if not binding['enabled']:
            raise ValueError(f'{channel}: disabled override retains legacy angular stiffness/damping; use the professional editor')
        zero_effort = table['nodes'] == [[0.0, 0.0], [1.0, 0.0]] and binding.get('damping_ratio') == 0
        if (binding['mode'] != 'AUTO_FREQUENCY'
                or table['domain'] != template['domain'] or table['tail'] != 'LINEAR'
                or (table['nodes'] != template['nodes'] and not zero_effort)):
            raise ValueError(f'{channel}: custom/SI curves cannot be represented by the simple controls; Apply explicitly converts them')
        values[channel] = dict(binding, _zero_effort=zero_effort)
    bend = [values.get(axis) for axis in ('swing_y', 'swing_z')]
    present_bend = [row for row in bend if row is not None]
    if len(present_bend) == 2 and (present_bend[0]['_zero_effort'] != present_bend[1]['_zero_effort']
                                   or (not present_bend[0]['_zero_effort'] and present_bend[0]['frequency_hz'] != present_bend[1]['frequency_hz'])
                                   or present_bend[0]['damping_ratio'] != present_bend[1]['damping_ratio']
                                   or present_bend[0]['enabled'] != present_bend[1]['enabled']):
        raise ValueError('Swing Y/Z currently differ; use the advanced editor before loading linked bend controls')
    anchor_bend = present_bend[0] if present_bend else None
    twist = values.get('twist_x')
    ratios = [row['damping_ratio'] for row in (anchor_bend, twist) if row is not None and not row['_zero_effort']]
    if ratios and any(item != ratios[0] for item in ratios[1:]):
        raise ValueError('Bend and twist damping ratios differ; use the advanced editor before loading linked controls')
    return {
        'bend_strength': frequency_to_strength(anchor_bend['frequency_hz']) if anchor_bend and not anchor_bend['_zero_effort'] else 0.0,
        'twist_strength': frequency_to_strength(twist['frequency_hz']) if twist and not twist['_zero_effort'] else 0.0,
        'damping_ratio': ratios[0] if ratios else .7,
        'defaults': False,
    }


def _read_anchor_values(context, bone):
    """A disabled/missing override retains legacy N/m; never infer its Hz."""
    defaults = {'anchor_strength': 1.0, 'anchor_damping_ratio': .7,
                'anchor_status': '保留旧 N/m 连接；数值仅为替换草稿'}
    state = read_state(context.scene)
    if state['mode'] == 'LEGACY' or state['document'] is None:
        return defaults
    key = curves.target_key({'kind': 'JOINT', 'joint_uuid': bone.ppf_bone_physics.joint_uuid, 'channel': 'anchor'})
    binding = next((b for b in state['document']['bindings'] if curves.target_key(b['target']) == key), None)
    if binding is None or not binding['enabled']:
        return defaults
    table = next(t for t in state['document']['tables'] if t['table_id'] == binding['table_id'])
    if (binding['mode'] == 'AUTO_FREQUENCY' and table['domain'] == 'RADIAL'
            and table['tail'] == 'LINEAR'):
        if table['nodes'] == [[0.0, 0.0], [1.0, 1.0]]:
            return {'anchor_strength': frequency_to_strength(binding['frequency_hz']),
                    'anchor_damping_ratio': binding['damping_ratio'],
                    'anchor_status': '已载入线性自动连接；按构建有效质量适配'}
        if table['nodes'] == [[0.0, 0.0], [1.0, 0.0]] and binding['damping_ratio'] == 0:
            return {'anchor_strength': 0.0, 'anchor_damping_ratio': 0.0,
                    'anchor_status': '已载入零力、零阻尼连接曲线'}
    return dict(defaults, anchor_status='当前为 SI 或自定义连接；数值仅为替换草稿')


class PPF_OT_BoneRestoringControlsLoad(bpy.types.Operator):
    bl_idname = 'ppf.bone_restoring_controls_load'
    bl_label = '载入当前恢复参数'
    bl_description = '读取固定SI参数或上次构建实际K/C；自动值属于上次构建，不冒充材质或形状改变后的当前值。载入不改变仿真数据'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            _arm,bone=_active_physical_bone(context)
            from .bone_physical_controls import load
            load(context,bone)
        except (ValueError,TypeError,KeyError,RuntimeError,OSError) as error:
            self.report({'ERROR'},str(error));return {'CANCELLED'}
        return {'FINISHED'}


class PPF_OT_BoneRestoringControlsApply(bpy.types.Operator):
    bl_idname = 'ppf.bone_restoring_controls_apply'
    bl_label = '应用当前恢复参数'
    bl_description = '只把改动的轴写成固定物理刚度/阻尼；未修改通道保留自动质量适配或原曲线。位置连接须显式勾选，零值写入启用的零曲线'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            _arm,bone=_active_physical_bone(context)
            from .bone_physical_controls import apply
            changed=apply(context,bone)
        except (ValueError,TypeError,KeyError,RuntimeError,OSError) as error:
            self.report({'ERROR'},str(error));return {'CANCELLED'}
        self.report({'INFO'},'已应用改动的物理通道' if changed else '参数未改变，保留原设置')
        return {'FINISHED'}


class PPF_OT_BoneRestoringPhysicalNew(bpy.types.Operator):
    bl_idname='ppf.bone_restoring_physical_new'
    bl_label='设置固定刚度'
    bl_description='显式输入新的固定刚度和阻尼，确认后立即替换此通道；零初值是新输入，不代表预设实际值。其他通道保留'
    bl_options={'REGISTER','UNDO'}
    channel: EnumProperty(items=[('twist_x','X扭转',''),('swing_y','Y弯曲',''),('swing_z','Z弯曲',''),('anchor','位置连接','')])
    stiffness: FloatProperty(name='新固定刚度',min=0.,default=0.,precision=6)
    damping: FloatProperty(name='新固定阻尼',min=0.,default=0.,precision=6)

    def invoke(self,context,event):
        self.stiffness=0.;self.damping=0.
        return context.window_manager.invoke_props_dialog(self)

    def draw(self,context):
        self.layout.label(text='新固定值；确认后立即生效')
        self.layout.label(text='N/cm · N·s/cm' if self.channel=='anchor' else 'N·mm/° · N·mm·s/°')
        self.layout.prop(self,'stiffness');self.layout.prop(self,'damping')

    def execute(self,context):
        try:
            _arm,bone=_active_physical_bone(context)
            from .bone_physical_controls import direct_commit,refresh
            count,skipped=direct_commit(context,bone,self.channel,self.stiffness,self.damping,replace=True)
        except (ValueError,TypeError,KeyError,RuntimeError,OSError) as error:
            self.report({'ERROR'},str(error));return {'CANCELLED'}
        try:refresh(context,bone)
        except Exception as error:
            self.report({'WARNING'},'作者参数已提交；显示刷新失败：'+str(error))
            return {'FINISHED'}
        self.report({'INFO'},f'已修改 {count} 个父关节'+('；跳过无父关节：'+', '.join(skipped) if skipped else ''))
        return {'FINISHED'}


def _material_plan(context, arm, bone, material_id, mask):
    validate_proxy_owner(arm, bone)
    settings = bone.ppf_bone_physics
    paths = [path for block, label in ((context.scene, 'scene'), (arm.data, 'armature data'), (arm, 'armature object'))
             for path in _owner_animation_paths(block, label)]
    prefix = bone.path_from_id() + '.ppf_bone_physics.'
    fields = set()
    if mask.get('material'): fields.update(('material_preset','solid_model','solid_young_modulus','solid_poisson_ratio','solid_deformation_damping'))
    if mask.get('mass'): fields.update(('mass_mode','density'))
    if mask.get('mass') and not mask.get('material') and settings.physical_model=='SOFT':
        # Density-only maintenance synchronizes effective material authors.
        fields.update(('solid_model','solid_young_modulus','solid_poisson_ratio','solid_deformation_damping'))
    if any(any(path==prefix+field or path.startswith(prefix+field+'[') for field in fields) for path in paths):
        raise ValueError(f'{bone.name}: animated material/mass properties cannot be changed by a preset')
    preset = MATERIAL_PRESETS[material_id]
    # A combined chain preset explicitly chooses density-derived body mass.
    # This makes a light preset change actual mass instead of leaving an old
    # fixed target in force; AUTO_FREQUENCY then resolves K/C from current
    # effective mass/inertia on the next Build.
    values = preset_values(material_id, 'DENSITY', settings.target_mass_kg)
    return settings, values


def apply_chain_preset(context, preset_id, scope='SELECTED', mask=None):
    mask = mask if mask is not None else dict(restoring=True, material=True, mass=True)
    if not any(mask.get(k, False) for k in ('restoring','material','mass')):
        return 0, 0, 0
    if preset_id not in CHAIN_PRESETS:
        raise ValueError('Unknown bone-chain preset')
    arm, bones = selected_bones(context, scope=scope)
    from . import bone_proxy as ops
    ops._facade_communicator().assert_bone_body_switch_idle()
    if ops.stale_simulation_output(context): raise ValueError('请先Clear已有模拟帧，再应用预设')
    if mask.get('restoring'):
        for bone in bones:
            paths=[path for block,label in ((context.scene,'scene'),(arm.data,'armature data'),(arm,'armature object')) for path in _owner_animation_paths(block,label)]
            prefix=bone.path_from_id()+'.ppf_bone_physics.'
            fields=('soft_limit_policy','soft_limit_frequency_hz','soft_limit_strength')
            cp=bone.path_from_id()+'.ppf_bone_restoring_controls.'
            if any(any(path==prefix+f or path.startswith(prefix+f+'[') for f in fields) or path.startswith(cp) for path in paths): raise ValueError(bone.name+': selected restoring fields are animated')
    preset = CHAIN_PRESETS[preset_id]
    plans = [_material_plan(context, arm, bone, preset.material_id, mask) for bone in bones] if mask.get('material') or mask.get('mass') else []
    targets = []
    target_bones = []
    material_only_bones = []
    for bone in bones:
        validate_proxy_owner(arm, bone)
        # Material applies to every scoped bone, but only an actual joint gets
        # restoring rows (the root and unfollowed-parent bones are skipped).
        target_rows = _targets_for_bone(arm, bone) if mask.get('restoring') else []
        if not target_rows:
            material_only_bones.append(bone)
            continue
        targets.extend(target_rows)
        target_bones.append(bone)
    frequency_by_channel = {
        'twist_x': preset.twist_hz,
        'swing_y': preset.bend_hz,
        'swing_z': preset.bend_hz,
        'anchor': preset.anchor_hz,
    }
    channel_settings = {channel: {'frequency_hz': frequency_by_channel[channel],
                                  'enabled': True, 'damping_ratio': preset.damping_ratio}
                        for channel in _CHANNELS}
    # Validate/prepare the complete curve candidate before the first RNA write.
    next_state = None
    if targets:
        author_map = _linear_frequency_document(targets, channel_settings)
        state = read_state(context.scene)
        if state['mode'] == 'LEGACY' and state['document'] is not None:
            raise ValueError('Dormant legacy curves exist; explicitly convert the restoring mode before applying a chain preset')
        source = restoring_curve_presets.named_preset(preset.label, author_map)
        curve_author.apply_preset(state, source,
            [{'source': target, 'destination': target} for target in targets], targets, [])
        next_state = state
    from ..core.bone_mass import material_write_guard, mark_settings, _density_plan, _apply_density
    from ..models.bone_bindings import _owned_group
    from ..models.groups import iterate_active_object_groups
    from .restoring_curve_operators import snapshot_rna, restore_rna
    # Validate ownership and animation before any writes.
    groups = []
    for bone in bones if plans else []:
        settings=bone.ppf_bone_physics
        group=_owned_group(settings,'SOLID' if settings.physical_model=='SOFT' else 'PDRD')
        if settings.proxy is None or context.scene.objects.get(settings.proxy.name)!=settings.proxy or group is None or sum(g.uuid==settings.group_uuid for g in iterate_active_object_groups(context.scene))!=1:
            raise ValueError(bone.name+': proxy or material ownership invalid')
        fields=set()
        if mask.get('material'): fields.update(('solid_model','solid_young_modulus','solid_poisson_ratio','deformation_damping','young_mod_density_normalized'))
        if mask.get('mass'): fields.update(('solid_density','pdrd_density'))
        if mask.get('mass') and not mask.get('material') and group.object_type=='SOLID':
            fields.update(('solid_young_modulus','young_mod_density_normalized'))
        gp=group.path_from_id()+'.'
        paths=[path for block,label in ((context.scene,'scene'),(arm.data,'armature data'),(arm,'armature object')) for path in _owner_animation_paths(block,label)]
        if any(any(path==gp+field or path.startswith(gp+field+'[') for field in fields) for path in paths): raise ValueError(bone.name+': selected material fields are animated')
        groups.append(group)
    if mask.get('mass') and not mask.get('material'):
        # Plan every target before writes; never clamp or change storage convention.
        for (settings, values), group in zip(plans, groups):
            _density_plan(settings,group,values['density'])
    snapshots = [(settings, snapshot_rna(settings), group, snapshot_rna(group))
                 for (settings, _), group in zip(plans, groups)]
    curve_snapshot = snapshot_rna(context.scene.ppf_restoring_curves)
    control_snapshots = [(bone, snapshot_rna(bone.ppf_bone_physics), snapshot_rna(bone.ppf_bone_restoring_controls)) for bone in target_bones]
    try:
        with material_write_guard():
            for ((settings, values), group) in zip(plans, groups):
                if mask.get('material'):
                    settings.material_preset = preset.material_id
                    if 'solid_model' in values:
                        for field in ('solid_model', 'solid_young_modulus', 'solid_poisson_ratio', 'solid_deformation_damping'):
                            setattr(settings, field, values[field])
                        if group.object_type == 'SOLID':
                            group.solid_model = settings.solid_model
                            group.solid_young_modulus = settings.solid_young_modulus
                            group.solid_poisson_ratio = settings.solid_poisson_ratio
                            group.deformation_damping = settings.solid_deformation_damping
                            group.young_mod_density_normalized = False
                if mask.get('mass'):
                    settings.mass_mode = 'DENSITY'
                    if not mask.get('material'):
                        _apply_density(settings,group,values['density'])
                    else:
                        settings.density = values['density']
                        if group.object_type == 'SOLID': group.solid_density = values['density']
                        else: group.pdrd_density = values['density']
        for settings, _values in plans: mark_settings(settings, context)
        if next_state is not None: write_state(context.scene, next_state)
        strength_bend = frequency_to_strength(preset.bend_hz)
        strength_twist = frequency_to_strength(preset.twist_hz)
        for bone in target_bones:
            bone.ppf_bone_physics.soft_limit_policy='AUTO_INERTIA'
            bone.ppf_bone_physics.soft_limit_frequency_hz=max(10.,2.*max(preset.bend_hz,preset.twist_hz))
            bone.ppf_bone_physics.soft_limit_strength=1.
            controls = bone.ppf_bone_restoring_controls
            controls.bend_strength = strength_bend
            controls.twist_strength = strength_twist
            controls.damping_ratio = preset.damping_ratio
            controls.anchor_strength = frequency_to_strength(preset.anchor_hz)
            controls.anchor_damping_ratio = preset.damping_ratio
            controls.anchor_status = '已应用链预设自动连接；按构建有效质量适配'
            controls.apply_anchor = False
            controls.loaded_joint_uuid = bone.ppf_bone_physics.joint_uuid
            controls.loaded_bone_uuid = bone.ppf_bone_physics.bone_uuid
            controls.draft_loaded = True
            controls.physical_draft_loaded = False
            controls.physical_available = (False,)*4
    except BaseException:
        with material_write_guard():
            for settings, snapshot, group, group_snapshot in snapshots:
                restore_rna(settings, snapshot); restore_rna(group, group_snapshot)
            for bone, settings_snapshot, controls_snapshot in control_snapshots:
                restore_rna(bone.ppf_bone_physics, settings_snapshot)
                restore_rna(bone.ppf_bone_restoring_controls, controls_snapshot)
            restore_rna(context.scene.ppf_restoring_curves, curve_snapshot)
        raise
    return len(bones), len(target_bones), len(material_only_bones)


class PPF_OT_BoneChainPresetApply(bpy.types.Operator):
    bl_idname = 'ppf.bone_chain_preset_apply'
    bl_label = '应用骨骼链预设'
    bl_description = '显式应用材质/质量起点和弯曲、扭转、锚点质量/惯量自适应恢复曲线'
    bl_options = {'REGISTER', 'UNDO'}
    preset: EnumProperty(name='链条预设', items=[(key, value.label, value.evidence)
                            for key, value in CHAIN_PRESETS.items()], default='LIGHT_HAIR')
    scope: EnumProperty(name='骨骼范围', items=SCOPES, default='SELECTED')

    def execute(self, context):
        try:
            state = context.window_manager.ppf_bone_preset_library
            mask = {key: getattr(state, 'apply_'+key) for key in ('restoring','material','mass')}
            if not any(mask.values()):
                self.report({'INFO'}, '未选择应用内容，未做修改'); return {'CANCELLED'}
            count, joint_count, material_only_count = apply_chain_preset(context, self.preset, self.scope, mask)
        except (ValueError, RuntimeError, OSError, KeyError) as exc:
            self.report({'ERROR'}, str(exc)); return {'CANCELLED'}
        preset = CHAIN_PRESETS[self.preset]
        suffix = (f'；{material_only_count} 根无有效父关节' 
                  if material_only_count else '')
        self.report({'INFO'}, f'已应用 {preset.label}：{count} 根骨骼，{joint_count} 个关节曲线组{suffix}；数值未标定')
        return {'FINISHED'}


CLASSES = (PPF_OT_BoneRestoringControlsLoad,
           PPF_OT_BoneRestoringControlsApply,
           PPF_OT_BoneRestoringPhysicalNew,
           PPF_OT_BoneChainPresetApply)


def register():
    for cls in CLASSES:
        _ppf_i18n.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
