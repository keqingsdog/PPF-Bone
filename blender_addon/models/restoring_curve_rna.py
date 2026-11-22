"""Persistent curve intent and editor drafts; registration is explicit."""
from .. import i18n as _ppf_i18n
import bpy
from ..i18n.props import BoolProperty, CollectionProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy.app.handlers import persistent
from .bone_chain_presets import PRESET_ITEMS, get_scene_preset, set_scene_preset

def _tier_changed(owner, context):
    """Keep the pre-tier display flag readable by older saved scenes."""
    owner.advanced_ui = owner.ui_tier != 'SIMPLE'

class PPF_PG_CurveNode(bpy.types.PropertyGroup):
    coordinate: FloatProperty(name='Coordinate',default=0,precision=6,options=set())
    effort: FloatProperty(name='Positive Energy Gradient',default=0,precision=6,options=set())

class PPF_PG_RestoringCurves(bpy.types.PropertyGroup):
    advanced_ui: BoolProperty(name='Advanced Parameters',description='Show custom curves and detailed physics controls without changing saved simulation settings',default=False,options=set())
    ui_tier: EnumProperty(name='显示层级',items=[('SIMPLE','简易','常用恢复参数与预设'),('ADVANCED','进阶','物理体、材质、根运动与约束设置'),('PRO','专业','恢复力函数、单位与独立通道')],default='SIMPLE',update=_tier_changed,options=set())
    bone_chain_preset: EnumProperty(name='骨链预设', items=PRESET_ITEMS, default='LIGHT_HAIR',
                                    get=get_scene_preset, set=set_scene_preset, options=set())
    bone_chain_scope: EnumProperty(name='预设应用范围',items=[('SELECTED','仅选中','只处理当前选中的物理骨骼'),('CHAIN','当前骨链','处理活动骨骼所属的物理链'),('ALL','整个骨架','处理当前活动骨架中的物理骨骼')],default='CHAIN',options=set())
    mode: EnumProperty(name='Restoring Mode',items=[('LEGACY','Legacy','Keep existing restoring parameters'),('COMMON','Common Restoring','Frequency and damping ratio'),('ADVANCED','Advanced','Edit individual restoring curves')],default='LEGACY',options=set())
    document_json: StringProperty(default='',options={'HIDDEN'})
    requested_mode: EnumProperty(name='Restoring Mode',items=[('LEGACY','Legacy','Keep existing restoring parameters'),('COMMON','Common Restoring','Frequency and damping ratio'),('ADVANCED','Advanced','Edit individual restoring curves')],default='LEGACY',options=set())
    channel: EnumProperty(name='Channel',items=[('twist_x','Twist X',''),('swing_y','Swing Y',''),('swing_z','Swing Z',''),('anchor','Bone Anchor',''),('stitch','Soft Stitch','')],default='twist_x',options=set())
    joint_channel: EnumProperty(name='Joint Channel',items=[('twist_x','Twist X',''),('swing_y','Swing Y',''),('swing_z','Swing Z',''),('anchor','Bone Anchor','')],default='twist_x',options=set())
    editor_target_json: StringProperty(default='',options={'HIDDEN'})
    enabled: BoolProperty(name='Enable Curve',default=True,options=set())
    policy: EnumProperty(name='Units / Mass Policy',items=[('AUTO_FREQUENCY','Automatic Mass Scaling','Compute from current mass and rest shape'),('ABSOLUTE_SI','Absolute SI','Explicit effort and damping in SI units')],default='AUTO_FREQUENCY',options=set())
    frequency_hz: FloatProperty(name='Natural Frequency (Hz)',default=1,min=.000001,options=set())
    damping_ratio: FloatProperty(name='Damping Ratio',default=.7,min=0,options=set())
    coordinate_scale: FloatProperty(name='Coordinate Scale (rad or m)',default=1,min=1e-9,precision=6,options=set())
    damping: FloatProperty(name='Damping (SI)',default=0,min=0,precision=6,options=set())
    domain: EnumProperty(name='Curve Domain',items=[('ODD','Odd Mirror','Mirror positive nodes on the negative side'),('SIGNED','Signed','Independent negative and positive nodes'),('RADIAL','Radial','Nonnegative anchor distance')],default='ODD',options=set())
    tail: EnumProperty(name='Outside Last Node',items=[('LINEAR','Linear Tail','Continue the endpoint slope'),('HOLD_EFFORT','Hold Effort','Constant endpoint effort')],default='LINEAR',options=set())
    nodes: CollectionProperty(type=PPF_PG_CurveNode)
    node_index: IntProperty(default=0,options=set())
    preset_name: StringProperty(name='Preset Name',default='恢复预设',options=set())
    preset_path: StringProperty(name='Preset File',subtype='FILE_PATH',default='',options=set())
    preset_json: StringProperty(default='',options={'HIDDEN'})
    preview_coordinate: FloatProperty(name='Preview Coordinate',default=0,precision=6,options=set())

CLASSES=(PPF_PG_CurveNode,PPF_PG_RestoringCurves)

def _legacy_tier_for(owner):
    try:
        if owner.is_property_set('ui_tier'):
            return None
    except (AttributeError, TypeError):
        return None
    return 'PRO' if bool(owner.advanced_ui) else 'SIMPLE'

def migrate_legacy_tiers(scenes=None):
    """Migrate old display flags outside the UI draw context."""
    if scenes is None:
        scenes = bpy.data.scenes
    for scene in scenes:
        owner = getattr(scene, 'ppf_restoring_curves', None)
        if owner is None:
            continue
        tier = _legacy_tier_for(owner)
        if tier is not None:
            try:
                owner.ui_tier = tier
            except (AttributeError, RuntimeError, TypeError):
                # Linked/read-only scenes remain readable through the legacy
                # fallback in current_tier; do not break file loading.
                continue

@persistent
def _migrate_legacy_tiers_on_load(_unused):
    migrate_legacy_tiers()

def _migrate_legacy_tiers_after_register():
    """Run after Blender leaves the restricted add-on registration context."""
    migrate_legacy_tiers()
    return None

def register():
    for cls in CLASSES:_ppf_i18n.register_class(cls)
    bpy.types.Scene.ppf_restoring_curves=PointerProperty(type=PPF_PG_RestoringCurves)
    handlers = bpy.app.handlers.load_post
    for handler in list(handlers):
        if getattr(handler, '__name__', '') == '_migrate_legacy_tiers_on_load':
            handlers.remove(handler)
    handlers.append(_migrate_legacy_tiers_on_load)
    if not bpy.app.timers.is_registered(_migrate_legacy_tiers_after_register):
        bpy.app.timers.register(_migrate_legacy_tiers_after_register, first_interval=0.1)

def unregister():
    if bpy.app.timers.is_registered(_migrate_legacy_tiers_after_register):
        bpy.app.timers.unregister(_migrate_legacy_tiers_after_register)
    handlers = bpy.app.handlers.load_post
    for handler in list(handlers):
        if getattr(handler, '__name__', '') == '_migrate_legacy_tiers_on_load':
            handlers.remove(handler)
    del bpy.types.Scene.ppf_restoring_curves
    for cls in reversed(CLASSES):bpy.utils.unregister_class(cls)
