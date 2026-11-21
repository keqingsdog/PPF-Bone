"""Mass-aware bone-chain authoring presets and direct frequency drafts.

Preset values are deliberately engineering starting points, not calibrated
material or modal data. Frequencies are author intent; fresh Build resolves
the actual spring and damping from current effective mass/inertia.
"""
from ..i18n import props as _ppf_props
from .. import i18n as _ppf_i18n
from dataclasses import dataclass

import bpy
from ..i18n.props import BoolProperty, BoolVectorProperty, FloatProperty, FloatVectorProperty, StringProperty
from bpy.app.handlers import persistent


@dataclass(frozen=True)
class BoneChainPreset:
    label: str
    material_id: str
    bend_hz: float
    twist_hz: float
    anchor_hz: float
    damping_ratio: float
    evidence: str


PRESETS = {
    'LIGHT_HAIR': BoneChainPreset(
        '轻质发束', 'LIGHT_HAIR', 5.0, 3.0, 18.0, .85,
        '未标定起点；采用轻质代理密度与质量/惯量自适应恢复力'),
    'FLEXIBLE_STRAP': BoneChainPreset(
        '柔软织带', 'SOFT_CLOTH', 8.0, 4.0, 25.0, .9,
        '未标定起点；采用柔软代理材质与质量/惯量自适应恢复力'),
    'STRAP': BoneChainPreset(
        '织带', 'STIFF_STRAP', 12.0, 8.0, 35.0, .78,
        '未标定起点；采用织带代理材质与质量/惯量自适应恢复力'),
    'STIFF_RIBBON': BoneChainPreset(
        '挺括飘带', 'STIFF_RIGID', 25.0, 15.0, 50.0, .72,
        '未标定起点；采用较高密度代理材质与质量/惯量自适应恢复力'),
    'HARD_WEBBING': BoneChainPreset(
        '硬质织带', 'HARD_WEBBING', 20.0, 12.0, 45.0, .8,
        '未标定起点；采用硬挺织带代理材质与质量/惯量自适应恢复力'),
}

PRESET_ITEMS = [(key, value.label, value.evidence) for key, value in PRESETS.items()]


def _physical_changed(self,context):
    from ..ops.bone_physical_controls import field_changed
    field_changed(self,context)


def _custom_changed(self,context):
    from ..ops.bone_physical_controls import custom_changed
    custom_changed(self,context)


class PPF_PG_BoneRestoringControls(bpy.types.PropertyGroup):
    """Explicit simple-control draft, loaded from and applied to curve intent."""
    angular_stiffness_degree: FloatVectorProperty(
        name='角向刚度', description='每偏离1度产生的回正力矩，单位N·mm/°；1000N·mm=1N·m，实际写回乘180/(1000π)为N·m/rad。固定数值修改后立即写入当前范围的父关节',
        size=3,min=0.,soft_max=1.,precision=6,options={'SKIP_SAVE'},update=_physical_changed)
    angular_damping_degree: FloatVectorProperty(
        name='角向阻尼', description='每1度/秒相对角速度产生的阻力矩，单位N·mm·s/°；1000N·mm=1N·m，实际写回乘180/(1000π)',
        size=3,min=0.,soft_max=1.,precision=6,options={'SKIP_SAVE'},update=_physical_changed)
    linear_stiffness_cm: FloatProperty(
        name='位置刚度',description='每拉开1厘米产生的回正力，单位N/cm；实际写回乘100为N/m',
        min=0.,soft_max=100.,precision=6,options={'SKIP_SAVE'},update=_physical_changed)
    linear_damping_cm: FloatProperty(
        name='位置阻尼',description='每1厘米/秒相对速度产生的阻力，单位N·s/cm；实际写回乘100为N·s/m',
        min=0.,soft_max=10.,precision=6,options={'SKIP_SAVE'},update=_physical_changed)
    physical_custom: BoolVectorProperty(name='自定义',size=3,options={'SKIP_SAVE'},update=_custom_changed)
    physical_auto_backup: StringProperty(options={'HIDDEN'})
    physical_channel_labels: StringProperty(options={'HIDDEN','SKIP_SAVE'})
    physical_live: BoolProperty(options={'HIDDEN','SKIP_SAVE'})
    physical_available: BoolVectorProperty(size=4,options={'HIDDEN','SKIP_SAVE'})
    physical_force_replace: BoolVectorProperty(size=4,options={'HIDDEN','SKIP_SAVE'})
    physical_draft_loaded: BoolProperty(options={'HIDDEN','SKIP_SAVE'})
    physical_loaded_values: StringProperty(options={'HIDDEN','SKIP_SAVE'})
    physical_channel_snapshot: StringProperty(options={'HIDDEN','SKIP_SAVE'})
    physical_status: StringProperty(options={'HIDDEN','SKIP_SAVE'})
    bend_strength: FloatProperty(
        name='弯曲恢复强度', description='无量纲强度；频率=10×√强度 Hz，因此恢复刚度与强度成正比；0 关闭 Swing Y/Z',
        default=.01, min=0.0, soft_max=10.0, max=1000000.0, precision=3, options=set())
    twist_strength: FloatProperty(
        name='扭转恢复强度', description='无量纲强度；频率=10×√强度 Hz，因此恢复刚度与强度成正比；0 关闭 Twist X',
        default=.01, min=0.0, soft_max=10.0, max=1000000.0, precision=3, options=set())
    damping_ratio: FloatProperty(
        name='回弹阻尼', description='三个弯曲/扭转通道共用的阻尼比 ζ；0 自由回弹，1 临界阻尼',
        default=.7, min=0.0, max=5.0, precision=3, options=set())
    anchor_strength: FloatProperty(
        name='连接恢复强度', description='显式替换连接曲线时使用；频率=10×√强度 Hz，按当前构建有效质量适配；0 使用零力零阻尼曲线',
        default=1.0, min=0.0, soft_max=100.0, max=1000000.0, precision=3, options=set())
    anchor_damping_ratio: FloatProperty(
        name='连接阻尼', description='连接通道独立的阻尼比 ζ；强度为0时同时使用零阻尼',
        default=.7, min=0.0, max=5.0, precision=3, options=set())
    apply_anchor: BoolProperty(
        name='替换连接恢复设置', description='显式勾选后，应用按钮才会用连接草稿替换旧N/m、SI或自定义锚点曲线；每次载入重置为不替换',
        default=False, options={'SKIP_SAVE'})
    anchor_status: StringProperty(default='连接草稿尚未载入', options={'HIDDEN'})
    loaded_joint_uuid: StringProperty(default='', options={'HIDDEN'})
    loaded_bone_uuid: StringProperty(default='', options={'HIDDEN'})
    draft_loaded: BoolProperty(default=False, options={'HIDDEN'})


def _clear_anchor_optins():
    """Clear only existing opt-in drafts, without creating linked/missing data."""
    for armature in bpy.data.armatures:
        if not armature.is_editable:
            continue
        for bone in armature.bones:
            if not bone.is_property_set('ppf_bone_restoring_controls'):
                continue
            owner = bone.ppf_bone_restoring_controls
            if owner.apply_anchor:
                owner.apply_anchor = False
    return None


def _cancel_anchor_startup():
    if bpy.app.timers.is_registered(_clear_anchor_optins):
        bpy.app.timers.unregister(_clear_anchor_optins)


@persistent
def _clear_anchor_optins_on_load(_unused):
    _cancel_anchor_startup()
    _clear_anchor_optins()


def register():
    _ppf_i18n.register_class(PPF_PG_BoneRestoringControls)
    bpy.types.Bone.ppf_bone_restoring_controls = _ppf_props.PointerProperty(
        type=PPF_PG_BoneRestoringControls)
    handlers = bpy.app.handlers.load_post
    for handler in list(handlers):
        if (getattr(handler, '__name__', '') == '_clear_anchor_optins_on_load'
                and getattr(handler, '__module__', '') == __name__):
            handlers.remove(handler)
    handlers.append(_clear_anchor_optins_on_load)
    # Registration may run with RestrictBlend. Defer data access once only.
    _cancel_anchor_startup()
    bpy.app.timers.register(_clear_anchor_optins, first_interval=.1)
    from ..ops.bone_physical_controls import refresh_active
    if not bpy.app.timers.is_registered(refresh_active):
        bpy.app.timers.register(refresh_active,first_interval=.3,persistent=True)


def unregister():
    from ..ops.bone_physical_controls import refresh_active
    if bpy.app.timers.is_registered(refresh_active):bpy.app.timers.unregister(refresh_active)
    _cancel_anchor_startup()
    handlers = bpy.app.handlers.load_post
    for handler in list(handlers):
        if (getattr(handler, '__name__', '') == '_clear_anchor_optins_on_load'
                and getattr(handler, '__module__', '') == __name__):
            handlers.remove(handler)
    del bpy.types.Bone.ppf_bone_restoring_controls
    bpy.utils.unregister_class(PPF_PG_BoneRestoringControls)
