"""Persistent bone authoring data; no solver or pose playback in this module.

Candidate (jointed-material-addon-staging): adds one physical-body material
choice to the original Bone Physics authoring data.

* ``physical_model == 'RIGID'`` is the DEFAULT and keeps the original behaviour
  exactly: the proxy stays a PDRD body and every existing field, key and joint
  record is unchanged.
* ``physical_model == 'SOFT'`` makes the *real* proxy an official SOLID body
  (TetGen + Stable NeoHookean) carrying the solid material fields below. The
  bone graph, joint UUIDs, anchor stiffness, angular control/damping/soft limits
  and root control are all preserved; nothing is replaced by a pull force and no
  second material group is created.
"""
from .. import i18n as _ppf_i18n
import math
import uuid

import bpy
from ..i18n import pgettext_iface as iface
from ..i18n.props import BoolProperty, BoolVectorProperty, CollectionProperty, EnumProperty, FloatProperty, FloatVectorProperty, PointerProperty, StringProperty
from bpy.types import PropertyGroup
from mathutils import Matrix
from .bone_material_presets import PRESET_ITEMS
from .bone_collision import collision_settings_changed
from .proxy_generation import PROXY_QUALITY_ITEMS, PROXY_FACE_ITEMS

AXES = ("twist_x", "swing_y", "swing_z")
MODEL = "parent_local_swing_twist_x_v1"
NOT_ANIMATED = set()

# The physical body a bone's proxy is simulated as.
PHYSICAL_MODELS = [
    ('RIGID', 'Rigid Body', 'Simulate the proxy as a rigid PDRD body (original behaviour)', 0),
    ('SOFT', 'Soft Body', 'Simulate the real proxy as an official SOLID tet body', 1),
]
# The official SOLID constitutive models, spelled as the group RNA does.
SOFT_MODELS = [
    ('STABLE_NEOHOOKEAN', 'Stable NeoHookean', 'Stable NeoHookean model', 0),
    ('ARAP', 'ARAP', 'As-Rigid-As-Possible model', 1),
]
# Defaults for the SOFT physical body. The Young's modulus is a TRUE value in
# pascals (10 MPa) and is paired with ``young_mod_density_normalized = False`` so
# the encoder applies the solver's density-normalized convention.
SOFT_DEFAULT_YOUNG_PA = 1.0e7
SOFT_DEFAULT_POISSON = 0.3
SOFT_DEFAULT_MODEL = 'STABLE_NEOHOOKEAN'


PREVIEW_SOFT_COLOR = (0.25, 0.75, 1.0)
PREVIEW_RIGID_COLOR = (0.55, 0.8, 0.35)


def effective_preview_color(settings, physical_model):
    """Resolve display-only RGB, including defensive non-RNA callers."""
    automatic = PREVIEW_SOFT_COLOR if physical_model == 'SOFT' else PREVIEW_RIGID_COLOR
    if not getattr(settings, 'preview_color_custom', False):
        return automatic
    try:
        color = tuple(float(v) for v in settings.preview_color)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return automatic
    if len(color) != 3 or not all(math.isfinite(v) for v in color):
        return automatic
    return tuple(min(1.0, max(0.0, v)) for v in color)


def _update_preview_color(_self, context):
    """Display-only redraw; never synchronize material or retire a session."""
    manager = getattr(context, 'window_manager', None)
    if manager is not None:
        for window in manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()


def _error(message, **values):
    return ValueError(iface(message).format(**values))


def _refuse(field, bone_name, detail):
    """Author-stage refusal naming the offending field and bone."""
    return _error("Refused for {name}: {field}: {detail}", name=bone_name, field=field, detail=iface(detail))


def _mesh_only(_self, obj):
    return obj.type == "MESH"


def _material_groups_of(self):
    """Every active group this bone's proxy is actually a member of.

    Used for read-only queries only. Material WRITES go through
    :func:`_owned_group`, which additionally requires sole ownership, because a
    shared or unrelated group must never be steered by a bone pointer.
    """
    from .groups import get_group_by_uuid, has_addon_data
    proxy=self.proxy
    if not (self.group_uuid and self.body_uuid and proxy):
        return []
    found = []
    for scene in bpy.data.scenes:
        if scene.objects.get(proxy.name)!=proxy:continue
        if not has_addon_data(scene):
            continue
        group = get_group_by_uuid(scene, self.group_uuid)
        if group and group.active and any(a.uuid == self.body_uuid for a in group.assigned_objects):
            found.append(group)
    return found


def armature_data_owner_counts():
    """Count structural Object owners, excluding retained recovery ID references.

    Callers may share this snapshot within one synchronous batch only. Undo
    rebuilds identities, so neither RNA wrappers nor these keys are cached.
    """
    from collections import Counter
    return Counter(obj.data.as_pointer() for obj in bpy.data.objects
                   if obj.type == 'ARMATURE')


def _owned_group(self, object_type):
    """Only the unique bone/armature may synchronize its dedicated material."""
    if not self.body_uuid or self.proxy is None:return None
    proxy=self.proxy;owner=proxy_parent_armature(proxy)
    if not proxy.get('ppf_bone_proxy_owned') or len(proxy.users_scene)!=1:return None
    if owner is not None and owner.type=='ARMATURE' and owner.data!=self.id_data:return None
    if proxy.get('ppf_bone_uuid')!=self.bone_uuid or proxy.get('_solver_uuid')!=self.body_uuid:return None
    if isinstance(self.id_data,bpy.types.Armature):
        if sum(b.ppf_bone_physics.proxy==proxy for b in self.id_data.bones)!=1:return None
    arms=[o for o in bpy.data.objects if o.type=='ARMATURE' and o.data==self.id_data]
    if len(arms)!=1 or len(arms[0].users_scene)!=1:return None
    scene=proxy.users_scene[0];arm=arms[0]
    if scene.objects.get(arm.name)!=arm or proxy.get('ppf_armature_uuid')!=arm.ppf_armature_uuid:return None
    if sum(o.get('_solver_uuid')==self.body_uuid for o in scene.objects)!=1:return None
    if owner is None or owner.type!='ARMATURE':
        if any(o.type=='ARMATURE' and o.data!=self.id_data and
               o.ppf_armature_uuid==proxy.get('ppf_armature_uuid') for o in bpy.data.objects):return None
    for group in _material_groups_of(self):
        if group.object_type==object_type and len(group.assigned_objects)==1 and group.assigned_objects[0].uuid==self.body_uuid:
            return group
    return None



def proxy_parent_armature(proxy):
    """Resolve only the actual parent or our UUID-matched object container."""
    parent = proxy.parent
    if parent is None:
        return None
    if parent.type == 'ARMATURE':
        return parent
    if parent.get('ppf_bone_proxy_object_container'):
        arm = parent.parent
        if (parent.type != 'EMPTY' or arm is None or arm.type != 'ARMATURE'
                or parent.get('ppf_bone_proxy_object_container') != proxy.get('ppf_armature_uuid')
                or getattr(arm, 'ppf_armature_uuid', '') != proxy.get('ppf_armature_uuid')):
            raise ValueError('Bone proxy container owner identity is invalid')
        return arm
    return None


def validate_proxy_owner(armature,bone, *, data_owner_counts=None):
    """Refuse copied/stale pointers before any authoring mutation."""
    if armature.library or armature.data.library:raise _error('Make the armature local before authoring bone physics')
    if len(armature.users_scene)>1:raise _error('骨架被多个场景共享；请使用独立的场景副本来编辑骨骼物理')
    if data_owner_counts is None:data_owner_counts=armature_data_owner_counts()
    if data_owner_counts[armature.data.as_pointer()]>1:raise _error('Duplicate armature identity; duplicated rigs need independent bindings')
    s=bone.ppf_bone_physics;proxy=s.proxy
    if proxy is None:
        if s.body_uuid and any(o.get('_solver_uuid')==s.body_uuid for o in bpy.data.objects):
            raise _error('Duplicate bone or body identity; do not share a proxy between bones')
        return
    if len(proxy.users_scene)>1:raise _error('物理代理被多个场景共享；请先创建独立副本')
    if (proxy.type!='MESH' or not proxy.get('ppf_bone_proxy_owned') or
        proxy.get('ppf_bone_uuid')!=s.bone_uuid or proxy.get('ppf_armature_uuid')!=armature.ppf_armature_uuid or
        proxy.get('_solver_uuid')!=s.body_uuid):raise _error('{name}: proxy UUID changed; update its binding',name=bone.name)
    owner = proxy_parent_armature(proxy)
    if owner is not None and owner!=armature:
        raise _error('Duplicate armature identity; duplicated rigs need independent bindings')
    if owner is None and any(
            o!=armature and o.type=='ARMATURE' and o.ppf_armature_uuid==armature.ppf_armature_uuid for o in bpy.data.objects):
        raise _error('Duplicate armature identity; duplicated rigs need independent bindings')
    if any(other!=bone and other.ppf_bone_physics.proxy==proxy for other in armature.data.bones):
        raise _error('Duplicate bone or body identity; do not share a proxy between bones')


def validate_final_binding_ids(scene,selected_armature,selected_bones):
    """Check the complete final producer set before assigning IDs or meshes."""
    selected={b.name for b in selected_bones};seen={kind:{} for kind in ('armature','bone','body','joint','group')}
    for arm in scene.objects:
        if arm.type!='ARMATURE':continue
        producing=[b for b in arm.data.bones if b.ppf_bone_physics.enabled or
                   arm==selected_armature and b.name in selected]
        if not producing:continue
        aid=arm.ppf_armature_uuid
        if aid:
            if aid in seen['armature'] and seen['armature'][aid]!=arm:
                raise _error('Duplicate armature identity; duplicated rigs need independent bindings')
            seen['armature'][aid]=arm
        # Playback resolves every Bone in an active producer rig by UUID.
        # A disabled copied Bone must not make an enabled binding ambiguous.
        for bone in arm.data.bones:
            if bone in producing:validate_proxy_owner(arm,bone)
            settings=bone.ppf_bone_physics
            for kind,prop in (('bone','bone_uuid'),('body','body_uuid'),('joint','joint_uuid'),('group','group_uuid')):
                value=getattr(settings,prop)
                if not value:continue
                if value in seen[kind]:raise _error('Duplicate bone or body identity; do not share a proxy between bones')
                seen[kind][value]=(arm,bone)


def validate_proxy_participation(scene):
    """Do not silently export a disabled bone-owned proxy as an ordinary mesh."""
    from .groups import get_group_by_uuid,has_addon_data,iterate_active_object_groups
    if not has_addon_data(scene):return
    included={a.uuid for g in iterate_active_object_groups(scene) for a in g.assigned_objects if a.included}
    for proxy in scene.objects:
        if not proxy.get('ppf_bone_proxy_owned') or proxy.get('_solver_uuid') not in included:continue
        body_id=proxy.get('_solver_uuid');bone_id=proxy.get('ppf_bone_uuid');arm_id=proxy.get('ppf_armature_uuid')
        owners=[]
        for arm in scene.objects:
            if arm.type!='ARMATURE' or arm.ppf_armature_uuid!=arm_id:continue
            for bone in arm.data.bones:
                settings=bone.ppf_bone_physics
                if settings.bone_uuid!=bone_id or settings.body_uuid!=body_id or settings.proxy!=proxy:continue
                try:validate_proxy_owner(arm,bone)
                except ValueError:continue
                owners.append(settings)
        if len(owners)!=1 or not owners[0].enabled:
            raise _error('{name}: included bone-owned proxy needs one enabled owning bone; remove stale assignments before transfer',name=proxy.name)

def _sync_density(self, _context):
    from ..core.bone_mass import mark_settings, writing_material
    from ..core.bone_mass_math import valid_authored_density
    if writing_material(): return
    mark_settings(self, _context)
    if self.mass_mode == 'MASS': return
    try: valid_authored_density(float(self.density))
    except ValueError: return
    group = _owned_group(self, 'SOLID') or _owned_group(self, 'PDRD')
    if group is None: return
    from ..core.bone_mass import _apply_density, material_write_guard
    old_density = group.solid_density if group.object_type == 'SOLID' else group.pdrd_density
    try:
        _apply_density(self, group, float(self.density), authored=True)
    except Exception:
        with material_write_guard(): self.density = old_density
        raise


def effective_solid_material(settings, group=None):
    group = group or _owned_group(settings, 'SOLID')
    if group is None:
        return {key: getattr(settings, key) for key in ('solid_model','solid_young_modulus','solid_poisson_ratio','solid_deformation_damping')}
    young = float(group.solid_young_modulus)
    if group.young_mod_density_normalized:
        young *= float(group.solid_density)
    return dict(solid_model=group.solid_model, solid_young_modulus=young,
                solid_poisson_ratio=group.solid_poisson_ratio,
                solid_deformation_damping=group.deformation_damping)


def apply_solid_material_patch(settings, patch):
    from ..core.bone_mass import material_write_guard
    from ..core.bone_preset_apply import _validate_rna
    allowed = {'solid_model', 'solid_young_modulus', 'solid_poisson_ratio', 'solid_deformation_damping'}
    if not patch or not set(patch) <= allowed:
        raise ValueError('不支持的软体材质字段')
    group = _owned_group(settings, 'SOLID')
    if group is None:
        return
    previous_effective = effective_solid_material(settings, group)
    values = dict(previous_effective)
    values.update(patch)
    for key, value in values.items(): _validate_rna(settings, key, value)
    writes = {}
    for key, value in patch.items():
        if key == 'solid_young_modulus':
            density = float(group.solid_density)
            if group.young_mod_density_normalized and (not math.isfinite(density) or density <= 0):
                raise ValueError('软体材质密度必须大于零')
            value = value / density if group.young_mod_density_normalized else value
        name = 'deformation_damping' if key == 'solid_deformation_damping' else key
        _validate_rna(group, name, value)
        writes[name] = value
    old_group = {key: getattr(group, key) for key in writes}
    old_author = {key: (settings.is_property_set(key), getattr(settings, key)) for key in allowed}
    try:
        for key, value in writes.items():
            setattr(group, key, value)
            actual = getattr(group, key)
            if isinstance(value, float):
                if not math.isclose(actual, value, rel_tol=2e-6, abs_tol=1e-9): raise ValueError('材质值被限制：'+key)
            elif actual != value: raise ValueError('材质值被拒绝：'+key)
        with material_write_guard():
            for key, value in values.items(): setattr(settings, key, value)
    except Exception:
        for key, value in old_group.items(): setattr(group, key, value)
        with material_write_guard():
            for key, (was_set, value) in old_author.items():
                setattr(settings, key, previous_effective[key] if key in patch else value)
                if not was_set and key not in patch: settings.property_unset(key)
        raise


def _solid_update(field):
    def update(self, _context):
        from ..core.bone_mass import writing_material, material_write_guard
        if writing_material(): return
        previous = effective_solid_material(self)
        try:
            apply_solid_material_patch(self, {field: getattr(self, field)})
        except Exception:
            with material_write_guard(): setattr(self, field, previous[field])
            raise
    return update


def _get_young_mpa(self):
    return effective_solid_material(self)['solid_young_modulus'] / 1e6


def _set_young_mpa(self, value):
    self.solid_young_modulus = value * 1e6


def _sync_enabled(self, _context):
    """Mirror the enable flag onto the bone's own group membership.

    BOTH physical bodies are handled. Handling only PDRD (as the first cut did)
    left a disabled SOLID bone still assigned and ``included`` in its material
    group, so the encoder kept simulating a proxy the author had switched off.
    """
    for object_type in ("PDRD", "SOLID"):
        group = _owned_group(self, object_type)
        if group is not None:
            group.assigned_objects[0].included = self.enabled


def sync_owned_material(settings):
    """Push this bone's material fields onto the group it owns, if it owns one.

    The authoring fields and the group must agree in BOTH directions, because
    ``_group`` (during a proxy geometry update) writes the group's material from
    these very fields. If a field were only ever shown from the group, a later
    rebuild would overwrite the group with a stale value. So every write to a
    material field goes through here as well as through its RNA callback, and
    the pair can never drift.

    Returns the group that was synchronized, or None when this bone owns no
    compatible group (a shared, unrelated, wrong-type or absent group is left
    strictly alone).
    """
    settings = settings if isinstance(settings, PPF_BonePhysics) else settings
    object_type = 'SOLID' if settings.physical_model == 'SOFT' else 'PDRD'
    group = _owned_group(settings, object_type)
    if group is None:
        return None
    from ..core.bone_mass import _apply_density
    _apply_density(settings, group, float(settings.density), authored=True)
    if object_type == 'SOLID':
        apply_solid_material_patch(settings, {
            key: getattr(settings, key) for key in
            ('solid_model', 'solid_young_modulus', 'solid_poisson_ratio', 'solid_deformation_damping')})
    return group


def soft_body_groups(self):
    """Groups whose proxy is a SOLID body for this bone (empty in RIGID)."""
    return [g for g in _material_groups_of(self) if g.object_type == "SOLID"]


CONTROL_MODES = [('FREE', 'Dynamic', 'Physics controls this body', 0), ('KINEMATIC', 'Animated / Kinematic', 'Follow authored transforms', 1), ('STATIC', 'Pinned / Static', 'Hold the initial authored transform', 2)]


class PPF_BoneControlEvent(PropertyGroup):
    frame: FloatProperty(name='Frame', default=1., options=NOT_ANIMATED)
    mode: EnumProperty(name='Control Mode', items=CONTROL_MODES, default='FREE', options=NOT_ANIMATED)
    pose_binding: EnumProperty(name='Takeover Pose', items=[('CURRENT','Current Accepted Pose','Pin current pose or follow authored motion increments from it'),('ABSOLUTE','Absolute Author Pose','ABS02绝对接管仅在当前与作者姿态匹配时接受；当前位置模式固定当前姿态或连续跟随动画')],default='CURRENT',options=NOT_ANIMATED)


class PPF_BoneDriveTarget(PropertyGroup):
    frame: FloatProperty(name='Frame', default=1., options=NOT_ANIMATED)
    value: FloatProperty(name='Target', default=0., options=NOT_ANIMATED)


class PPF_BoneDriveChannel(PropertyGroup):
    enabled: BoolProperty(name='Enable Drive', default=False, options=NOT_ANIMATED)
    gain: FloatProperty(name='Gain', default=1., min=0., options=NOT_ANIMATED)
    max_effort: FloatProperty(name='Maximum Torque (Nm)', default=1., min=0., options=NOT_ANIMATED)
    targets: CollectionProperty(type=PPF_BoneDriveTarget)


class PPF_BoneDriveAxis(PropertyGroup):
    position: PointerProperty(type=PPF_BoneDriveChannel)
    velocity: PointerProperty(type=PPF_BoneDriveChannel)
    torque: PointerProperty(type=PPF_BoneDriveChannel)


class PPF_BoneD6Axis(PropertyGroup):
    mode: EnumProperty(name='Axis Constraint', items=[('free','Free','No hard constraint'),('limited','Limited','Hard lower and upper bounds'),('locked','Locked','Hard coordinate value')], default='locked', options=NOT_ANIMATED)
    lower: FloatProperty(name='Lower', default=-.1, options=NOT_ANIMATED)
    upper: FloatProperty(name='Upper', default=.1, options=NOT_ANIMATED)
    value: FloatProperty(name='Locked Value', default=0., options=NOT_ANIMATED)


def _mark_mass_dirty(self, context):
    from ..core.bone_mass import mark_settings
    mark_settings(self, context)


def _sync_fem_scale_display(self,context):
    from ..core.rigid_pose import request_cached_refresh
    request_cached_refresh(getattr(context,'scene',None))


def _get_soft_stiffness_degree(self):
    return tuple(float(v)*(1000.*math.pi/180.) for v in self.soft_stiffness)


def _set_soft_stiffness_degree(self,values):
    values=tuple(float(v)/(1000.*math.pi/180.) for v in values)
    if any(not math.isfinite(v) or v<0 for v in values):
        raise ValueError('软限位刚度必须为有限非负值')
    self.soft_stiffness=values


def _get_cone_stiffness_degree(self):
    return float(self.swing_cone_stiffness)*(1000.*math.pi/180.)


def _set_cone_stiffness_degree(self,value):
    value=float(value)/(1000.*math.pi/180.)
    if not math.isfinite(value) or value<0.:raise ValueError('圆锥刚度必须为有限非负值')
    self.swing_cone_stiffness=value


def _cancel_recreate_geometry_intent(self, _context):
    # A deliberate edit after Delete cancels the one-shot reconstruction
    # request immediately, even if the author later enters the old value.
    from ..core.bone_mass import writing_material
    from .new_binding_defaults import recreate_geometry_intent, preserve_recreate_after_edit
    if (self.recreate_geometry_intent_json and not writing_material()
            and recreate_geometry_intent(self) is None):
        preserve_recreate_after_edit(self)


class PPF_BonePhysics(PropertyGroup):
    enabled: BoolProperty(name="Enable Bone Physics", description="启用此骨骼的物理代理；同面数场景的计算时间会随关节数量显著增长", default=False, update=_sync_enabled, options=NOT_ANIMATED)
    bone_uuid: StringProperty(options=NOT_ANIMATED)
    body_uuid: StringProperty(options=NOT_ANIMATED)
    joint_uuid: StringProperty(options=NOT_ANIMATED)
    auto_follow_parent: BoolProperty(name="Follow Nonphysical Parent", description="通过内部运动学锚点跟随父骨骼，保留作者偏移，不改变父骨骼物理状态；跟随时仅支持自由动态，不支持独立接管事件", default=True, options=NOT_ANIMATED)
    group_uuid: StringProperty(options=NOT_ANIMATED)
    proxy: PointerProperty(type=bpy.types.Object, poll=_mesh_only, update=_mark_mass_dirty, options=NOT_ANIMATED)
    auto_geometry_snapshot_json: StringProperty(default='', options={'HIDDEN'})
    recreate_geometry_intent_json: StringProperty(default='', options={'HIDDEN'})
    shape: EnumProperty(name="Proxy Shape", items=[("CAPSULE", "Capsule", "Closed bone segment", 0), ("SPHERE", "Sphere", "Closed sphere", 3), ("BOX", "Box", "Closed rectangular segment", 1), ("CUSTOM", "Custom Mesh", "Independent copy of a closed rigid mesh", 2)], default="CAPSULE", update=_cancel_recreate_geometry_intent, options=NOT_ANIMATED)
    proxy_quality: EnumProperty(name='代理分辨率',items=PROXY_QUALITY_ITEMS,default='AUTO',options=NOT_ANIMATED)
    new_proxy_quality: EnumProperty(name='新建代理分辨率',items=PROXY_QUALITY_ITEMS,default='AUTO',options=NOT_ANIMATED)
    proxy_face_mode: EnumProperty(name='代理面类型',items=PROXY_FACE_ITEMS,default='QUAD',options=NOT_ANIMATED)
    custom_object: PointerProperty(name="Closed Mesh", type=bpy.types.Object, poll=_mesh_only, options=NOT_ANIMATED)
    radius: FloatProperty(name="Radius", default=.0175, min=.000001, subtype="DISTANCE", update=_cancel_recreate_geometry_intent, options=NOT_ANIMATED)
    length_ratio: FloatProperty(name="Bone Length Fraction", default=.85, min=.05, max=1., update=_cancel_recreate_geometry_intent, options=NOT_ANIMATED)
    box_width: FloatProperty(name="Box Width", default=.035, min=.000001, subtype="DISTANCE", update=_cancel_recreate_geometry_intent, options=NOT_ANIMATED)
    box_depth: FloatProperty(name="Box Depth", default=.035, min=.000001, subtype="DISTANCE", update=_cancel_recreate_geometry_intent, options=NOT_ANIMATED)
    density: FloatProperty(name="Density (kg/m³)", default=1000., min=.01, max=10000., update=_sync_density, options=NOT_ANIMATED)
    mass_mode: EnumProperty(name='质量输入模式', items=[('DENSITY','固定密度','保持密度；显示作者体积估计，最终以构建为准'),('MASS','固定目标质量','刚体采用离散点总质量与对应惯量；软体按体积推导密度')], default='DENSITY', update=_mark_mass_dirty, options=NOT_ANIMATED)
    target_mass_kg: FloatProperty(name='目标质量（kg）', default=.01, precision=6, update=_mark_mass_dirty, options=NOT_ANIMATED)
    mass_volume_m3: FloatProperty(name='作者体积（m³）', default=0., precision=6, options=NOT_ANIMATED)
    mass_measured_kg: FloatProperty(name='体积估计质量（kg，最终以构建为准）', default=0., precision=6, options=NOT_ANIMATED)
    mass_valid: BoolProperty(default=False, options=NOT_ANIMATED)
    mass_status: StringProperty(default='等待作者体积测量', options=NOT_ANIMATED)
    material_preset: EnumProperty(name='材质预设', items=PRESET_ITEMS, default='STIFF_RIGID', options=NOT_ANIMATED)
    solid_deformation_damping: FloatProperty(name='材料变形阻尼（秒）', default=.002, min=0., precision=5, update=_solid_update('solid_deformation_damping'), options=NOT_ANIMATED)
    # Physical body selection. RIGID is the default so an authored scene that
    # never touches this field reproduces the original behaviour exactly.
    fem_scale_enabled: BoolProperty(name='随软体形变缩放骨骼',default=True,update=_sync_fem_scale_display,options=NOT_ANIMATED)
    fem_scale_last: FloatVectorProperty(name='骨骼方向缩放',size=3,default=(1.,1.,1.),precision=6,options=NOT_ANIMATED)
    fem_scale_shear: FloatProperty(name='非正交程度',default=0.,options=NOT_ANIMATED)
    fem_scale_variation: FloatProperty(name='局部伸缩离散度',default=0.,options=NOT_ANIMATED)
    fem_scale_orientation_error: FloatProperty(name='继承轴向近似误差',default=0.,options=NOT_ANIMATED)
    fem_scale_status: StringProperty(name='形变映射状态',default='尚未收到软体帧',options=NOT_ANIMATED)
    preview_color_custom: BoolProperty(name='自定义预览色', description='仅改变GPU代理预览，不改变材质或模拟参数', default=False, update=_update_preview_color, options=NOT_ANIMATED)
    preview_color: FloatVectorProperty(name='预览颜色', description='GPU代理预览颜色；关闭自定义时自动按物理材质显示', subtype='COLOR', size=3, default=PREVIEW_SOFT_COLOR, min=0., max=1., update=_update_preview_color, options=NOT_ANIMATED)
    physical_model: EnumProperty(name="Physical Body", items=PHYSICAL_MODELS, default='RIGID', options=NOT_ANIMATED)
    new_binding_physical_model: EnumProperty(name='新建物理体', items=PHYSICAL_MODELS, default='RIGID', options=NOT_ANIMATED)
    solid_model: EnumProperty(name="Solid Model", items=SOFT_MODELS, default=SOFT_DEFAULT_MODEL, update=_solid_update('solid_model'), options=NOT_ANIMATED)
    solid_young_modulus: FloatProperty(name="Young's Modulus (Pa)", description="True Young's modulus in pascals for the soft body material; the encoder divides it by the group density for the solver's density-normalized convention", default=SOFT_DEFAULT_YOUNG_PA, min=1.0, max=1.0e12, precision=1, update=_solid_update('solid_young_modulus'), options=NOT_ANIMATED)
    solid_young_mpa: FloatProperty(name='材料硬度（MPa）', min=0.000001, max=1000000., precision=6, get=_get_young_mpa, set=_set_young_mpa, options={'SKIP_SAVE'})
    solid_poisson_ratio: FloatProperty(name="Poisson's Ratio", description="Poisson's ratio for the soft body material", default=SOFT_DEFAULT_POISSON, min=0., max=0.4999, precision=4, update=_solid_update('solid_poisson_ratio'), options=NOT_ANIMATED)
    anchor_stiffness: FloatProperty(name="Anchor Stiffness (N/m)", default=200000., min=0., options=NOT_ANIMATED)
    angular_stiffness: FloatVectorProperty(name="Angular Stiffness (Nm/rad)", size=3, default=(.02, .04, .04), min=0., precision=5, options=NOT_ANIMATED)
    angular_damping: FloatVectorProperty(name="Angular Damping (Nms/rad)", size=3, default=(.001, .001, .001), min=0., precision=5, options=NOT_ANIMATED)
    soft_limit_mode: EnumProperty(name='摆动范围', items=[('AXES','分轴','分别设置Y/Z软角度范围'),('CONE','圆锥','各个摆动方向共用圆形软范围，X扭转独立')], default='AXES', options=NOT_ANIMATED)
    swing_cone_half_angle: FloatProperty(name='圆锥半角', description='相对父关节轴的圆形软范围；超过后额外回弹', subtype='ANGLE', unit='ROTATION', default=math.pi/4, min=1.e-6, max=3.1405, options=NOT_ANIMATED)
    swing_cone_stiffness: FloatProperty(name='圆锥刚度（Nm/rad）', default=.04, min=0., options=NOT_ANIMATED)
    swing_cone_stiffness_degree: FloatProperty(name='圆锥手动刚度 N·mm/°', min=0., precision=6, get=_get_cone_stiffness_degree, set=_set_cone_stiffness_degree, options={'SKIP_SAVE'})
    soft_limits_enabled: BoolVectorProperty(name="启用软角度范围", description="角度范围是开始额外回弹的位置，超过后仍可运动", size=3, default=(True, True, True), options=NOT_ANIMATED)
    soft_lower: FloatVectorProperty(name="下限", description="转角下限，内部保存弧度；按Blender场景角度单位显示", subtype='EULER', unit='ROTATION', size=3, default=(-.8, -.8, -.8), min=-3.14, max=3.14, options=NOT_ANIMATED)
    soft_upper: FloatVectorProperty(name="上限", description="转角上限，内部保存弧度；按Blender场景角度单位显示", subtype='EULER', unit='ROTATION', size=3, default=(.8, .8, .8), min=-3.14, max=3.14, options=NOT_ANIMATED)
    soft_limit_policy: EnumProperty(name='限位力度', items=[('ABSOLUTE_SI','手动','使用已保存的绝对刚度；切换不会猜测或载入上次构建值'),('AUTO_INERTIA','自动','按真实构建有效惯量与尺寸适配；软范围仍允许越界')], default='ABSOLUTE_SI', options=NOT_ANIMATED)
    soft_limit_strength: FloatProperty(name='力度倍率', description='乘以自动惯量刚度；越大越难越界，不是硬停止', default=1., min=0., options=NOT_ANIMATED)
    soft_limit_frequency_hz: FloatProperty(name='限位内部频率', default=10., min=1.e-6, options=NOT_ANIMATED)
    soft_stiffness_degree: FloatVectorProperty(name='手动限位刚度 N·mm/°', description='已保存手动绝对值；1000N·mm=1N·m，不会自动读取构建结果', size=3, min=0., precision=6, get=_get_soft_stiffness_degree, set=_set_soft_stiffness_degree, options={'SKIP_SAVE'})
    soft_stiffness: FloatVectorProperty(name="软限位刚度（Nm/rad）", description="力度越大越难越界，不同质量和尺寸需要匹配", size=3, default=(.04, .04, .04), min=0., options=NOT_ANIMATED)
    collide_connected: BoolProperty(name="Collide with Adjacent Bone", default=False, update=collision_settings_changed, options=NOT_ANIMATED)
    collision_filter_enabled: BoolProperty(name="Enable Collision Group Filter", default=False, update=collision_settings_changed, options=NOT_ANIMATED)
    collision_group_layers: BoolVectorProperty(name="Collision Membership", size=16, default=(True,) + (False,) * 15, update=collision_settings_changed, options=NOT_ANIMATED)
    collision_mask_layers: BoolVectorProperty(name="Allowed Collision Groups", size=16, default=(True,) * 16, update=collision_settings_changed, options=NOT_ANIMATED)
    root_mode: EnumProperty(name="Root Mode", description="自由动态由物理控制；静态固定当前姿态；运动学跟随作者动画。SOFT独立控制仅支持根骨骼", items=CONTROL_MODES, default="FREE", options=NOT_ANIMATED)
    root_motion_source: EnumProperty(name='Animation Source', items=[('POSE','Bone Pose','Sample the authored bone world transform'),('OBJECT','Proxy Object','Sample the authored proxy world transform')], default='POSE', options=NOT_ANIMATED)
    root_control_events: CollectionProperty(type=PPF_BoneControlEvent)
    joint_mode: EnumProperty(name='Joint Type', items=[('SOFT','Compliant Joint','Directional springs and soft limits'),('BALL_FIXED','硬球关节（连接点固定）','固定连接点，保留角向柔性'),('HINGE','Hard Hinge','Lock swing and constrain twist'),('D6','Hard D6','Independent translation and rotation constraints')], default='SOFT', options=NOT_ANIMATED)
    hinge_twist_mode: EnumProperty(name='Twist Constraint', items=[('free','Free','Unbounded twist'),('bounded','Limited','Hard twist bounds')], default='free', options=NOT_ANIMATED)
    hinge_lower: FloatProperty(name='Lower (rad)', default=-.8, options=NOT_ANIMATED)
    hinge_upper: FloatProperty(name='Upper (rad)', default=.8, options=NOT_ANIMATED)
    drive_twist_x: PointerProperty(type=PPF_BoneDriveAxis)
    drive_swing_y: PointerProperty(type=PPF_BoneDriveAxis)
    drive_swing_z: PointerProperty(type=PPF_BoneDriveAxis)
    d6_translation_x: PointerProperty(type=PPF_BoneD6Axis)
    d6_translation_y: PointerProperty(type=PPF_BoneD6Axis)
    d6_translation_z: PointerProperty(type=PPF_BoneD6Axis)
    d6_rotation_x: PointerProperty(type=PPF_BoneD6Axis)
    d6_rotation_y: PointerProperty(type=PPF_BoneD6Axis)
    d6_rotation_z: PointerProperty(type=PPF_BoneD6Axis)
    use_auto_frames: BoolProperty(name="Automatic Joint Frames", default=True, options=NOT_ANIMATED)
    joint_axis_euler: FloatVectorProperty(name="Joint Axis Rotation", size=3, subtype="EULER", default=(0., 0., 0.), options=NOT_ANIMATED)
    parent_frame_translation: FloatVectorProperty(name="Parent Anchor", size=3, subtype="TRANSLATION", options=NOT_ANIMATED)
    parent_frame_quaternion: FloatVectorProperty(name="Parent Frame (wxyz)", size=4, default=(1., 0., 0., 0.), options=NOT_ANIMATED)
    child_frame_translation: FloatVectorProperty(name="Child Anchor", size=3, subtype="TRANSLATION", options=NOT_ANIMATED)
    child_frame_quaternion: FloatVectorProperty(name="Child Frame (wxyz)", size=4, default=(1., 0., 0., 0.), options=NOT_ANIMATED)


def active_armature(context):
    obj = context.object
    return obj if obj and obj.type == "ARMATURE" else None


def active_bone(context):
    arm = active_armature(context)
    if not arm:
        return None
    return arm.data.bones.active


def ensure_binding_ids(armature, bone):
    if armature.library or armature.data.library:
        raise _error("Make the armature local before authoring bone physics")
    if not armature.ppf_armature_uuid:
        armature.ppf_armature_uuid = str(uuid.uuid4())
    settings = bone.ppf_bone_physics
    if not settings.bone_uuid:
        settings.bone_uuid = str(uuid.uuid4())
    if not settings.joint_uuid:
        settings.joint_uuid = str(uuid.uuid4())
    return settings


def rigid_matrix(matrix, name):
    """Refuse scale/shear so manual endpoint frames have unambiguous meaning."""
    columns = [matrix.to_3x3().col[i] for i in range(3)]
    if any(not math.isfinite(value) for row in matrix for value in row) or any(abs(c.length - 1.) > 1e-5 for c in columns) or any(abs(columns[i].dot(columns[j])) > 1e-5 for i in range(3) for j in range(i)) or matrix.to_3x3().determinant() <= 0:
        raise _error("{name}: apply object scale and remove pose scale/shear before authoring", name=name)
    # Blender evaluates parent/inverse-parent products in float32.  A frame
    # accepted above must be emitted as a rigid frame, rather than forwarding
    # those accumulated rounding errors to the stricter transport validator.
    # This does not admit additional scale or shear: the authoring gate above
    # still validates the original matrix before projecting its rotation.
    frame = matrix.to_quaternion().normalized().to_matrix().to_4x4()
    frame.translation = matrix.translation
    return frame


def similarity_frame_scale(matrix, name):
    """Split a finite positive uniform object/pose transform, never shear."""
    # A positive similarity has an unambiguous rigid head frame and a real
    # length multiplier. Keep that multiplier separate from native joint frames.
    columns = [matrix.to_3x3().col[i] for i in range(3)]
    scales = [c.length for c in columns]
    scale = sum(scales) / 3.
    determinant = matrix.to_3x3().determinant()
    if (not all(math.isfinite(v) for row in matrix for v in row)
            or any(abs(matrix[3][j] - (1. if j == 3 else 0.)) > 1.e-7 for j in range(4))
            or not math.isfinite(scale) or scale <= 1.e-8
            or not math.isfinite(determinant) or determinant <= 0
            or any(abs(s / scale - 1.) > 1.e-5 for s in scales)
            or any(abs(columns[i].dot(columns[j]) / (scales[i] * scales[j])) > 1.e-5
                   for i in range(3) for j in range(i))):
        raise _error('{name}: only static positive uniform scale is supported; remove nonuniform scale, shear or reflection', name=name)
    frame = Matrix.Translation(matrix.translation) @ matrix.to_quaternion().normalized().to_matrix().to_4x4()
    return rigid_matrix(frame, name), scale


def _rest_frame_for_drift(rest):
    """Admit only the existing narrow finite, near-rigid rest envelope."""
    if (not all(math.isfinite(v) for row in rest for v in row)
            or any(abs(rest[3][j] - (1. if j == 3 else 0.)) > 1.e-7 for j in range(4))):
        return None
    columns = [rest.to_3x3().col[i] for i in range(3)]
    gram = max(abs(columns[i].dot(columns[j]) - (1. if i == j else 0.))
               for i in range(3) for j in range(i + 1))
    determinant = rest.to_3x3().determinant()
    if (not math.isfinite(gram) or gram > 1.e-4 or not math.isfinite(determinant)
            or abs(determinant - 1.) > 1.e-4):
        return None
    corrected = rest.to_quaternion().normalized().to_matrix().to_4x4()
    corrected.translation = rest.translation
    return corrected, gram


def _animated_rest_drift_frame(armature, bone):
    """Prove local similarity before correcting a narrow rest-only drift.

    Blender handles inheritance and local-location semantics during both
    conversions. Nothing in the live armature is written or cached.
    """
    chain, seen = [], set()
    current = bone
    while current is not None:
        if current.name in seen:
            return None
        seen.add(current.name)
        info = _rest_frame_for_drift(current.matrix_local)
        if info is None:
            return None
        chain.append((current, info[0], info[1]))
        current = getattr(current, 'parent', None)
    if not any(error > 5.e-7 for _, _, error in chain):
        return None
    chain.reverse()
    try:
        corrected_parent = corrected_parent_rest = None
        for item, corrected_rest, _ in chain:
            pose = armature.pose.bones[item.name].matrix
            parent = getattr(item, 'parent', None)
            old_parent = {} if parent is None else {
                'parent_matrix': armature.pose.bones[parent.name].matrix,
                'parent_matrix_local': parent.matrix_local,
            }
            local = item.convert_local_to_pose(pose, item.matrix_local, invert=True, **old_parent)
            local_frame, local_scale = similarity_frame_scale(local, item.name)
            local_frame = local_frame @ Matrix.Diagonal((local_scale, local_scale, local_scale, 1.))
            corrected_parent_args = {} if parent is None else {
                'parent_matrix': corrected_parent, 'parent_matrix_local': corrected_parent_rest,
            }
            corrected_parent = item.convert_local_to_pose(local_frame, corrected_rest, **corrected_parent_args)
            corrected_parent_rest = corrected_rest
        return similarity_frame_scale(corrected_parent, bone.name)
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as failure:
        # Constraints may make the recovered basis non-similar even if the
        # authored channels are valid. Do not let a valid delta conceal an
        # independently invalid channel; missing channel evidence is a failure.
        missing_conversion_api = any(not callable(getattr(item, 'convert_local_to_pose', None))
                                     for item, _, _ in chain)
        if not missing_conversion_api:
            try:
                for item, _, _ in chain:
                    similarity_frame_scale(armature.pose.bones[item.name].matrix_basis, item.name)
            except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
                raise failure
        # Some external/fake armatures cannot supply the conversion API.
        # A left-multiplicative similarity is otherwise an independent proof.
        try:
            pose = armature.pose.bones[bone.name].matrix
            delta = pose @ bone.matrix_local.inverted()
            frame, scale = similarity_frame_scale(delta, bone.name)
            frame = frame @ chain[-1][1]
            frame.translation = pose.translation
            return rigid_matrix(frame, bone.name), scale
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
            return None


def bone_world_frame_scale(armature, bone):
    """Return a rigid world head frame and frozen total object/pose scale."""
    world, object_scale = similarity_frame_scale(armature.matrix_world, armature.name)
    pose = armature.pose.bones[bone.name].matrix
    rest = bone.matrix_local
    same_rest_orientation = all(abs(pose[i][j] - rest[i][j]) <= 1.e-7
                                for i in range(3) for j in range(3))
    columns = [rest.to_3x3().col[i] for i in range(3)]
    gram_error = max(abs(columns[i].dot(columns[j]) - (1. if i == j else 0.))
                     for i in range(3) for j in range(i + 1))
    determinant = rest.to_3x3().determinant()
    repaired = _animated_rest_drift_frame(armature, bone)
    if repaired is not None:
        pose_frame, pose_scale = repaired
    elif (same_rest_orientation and _rest_frame_for_drift(rest) is not None
            and all(math.isfinite(v) for row in pose for v in row)
            and all(abs(pose[3][j] - (1. if j == 3 else 0.)) <= 1.e-7 for j in range(4))
            and math.isfinite(gram_error) and gram_error <= 1.e-4
            and math.isfinite(determinant) and abs(determinant - 1.) <= 1.e-4):
        # Preserve the existing narrowly bounded Blender-rest drift correction.
        pose_frame = Matrix.Translation(pose.translation) @ rest.to_quaternion().normalized().to_matrix().to_4x4()
        pose_scale = 1.
    else:
        pose_frame, pose_scale = similarity_frame_scale(pose, bone.name)
    rotation = (world.to_quaternion() @ pose_frame.to_quaternion()).normalized()
    frame = rotation.to_matrix().to_4x4()
    frame.translation = armature.matrix_world @ pose.translation
    return rigid_matrix(frame, bone.name), object_scale * pose_scale


def bone_world_matrix(armature, bone):
    return bone_world_frame_scale(armature, bone)[0]


def bone_world_length(armature, bone):
    return bone.length * bone_world_frame_scale(armature, bone)[1]


def _matrix_rows(matrix):
    return [[float(x) for x in row] for row in matrix]


def _rigid_matrix_rows(matrix, name):
    """Serialize an admitted rigid frame without another float32 round trip.

    Mathutils quaternion normalization still returns float32 coefficients. Its
    determinant can round beyond the stricter wire tolerance; project in double
    only after the unchanged authoring gate has accepted the original frame.
    Armature metadata must keep using _matrix_rows to preserve authored scale.
    """
    import numpy as np
    rigid_matrix(matrix, name)
    result = np.asarray(_matrix_rows(matrix), dtype=np.float64)
    if not np.allclose(result[3], [0., 0., 0., 1.], atol=1.e-8, rtol=0.):
        raise _error("{name}: expected an affine rigid frame", name=name)
    u, _, vt = np.linalg.svd(result[:3, :3])
    rotation = u @ vt
    if np.linalg.det(rotation) <= 0.:
        raise _error("{name}: reflected rigid frame", name=name)
    result[:3, :3] = rotation
    result[3] = [0., 0., 0., 1.]
    return result.tolist()


def _frame(settings, prefix):
    q = tuple(getattr(settings, prefix + "_frame_quaternion"))
    if abs(sum(v*v for v in q) - 1.) > 1e-5:
        raise _error("{prefix} frame quaternion must be normalized (wxyz)", prefix=iface("Parent") if prefix == "parent" else iface("Child"))
    return {"translation_m": list(getattr(settings, prefix + "_frame_translation")), "quaternion_wxyz": list(q)}


def has_physical_joint_parent(bone, enabled_names=None):
    """Physical or automatic parent endpoint; a Blender hierarchy alone is not a joint."""
    if bone is None or bone.parent is None:
        return False
    parent_enabled = (bone.parent.name in enabled_names if enabled_names is not None
                      else bool(bone.parent.ppf_bone_physics.enabled))
    return bool(parent_enabled or getattr(bone.ppf_bone_physics, 'auto_follow_parent', True))


def soft_material_refusal(settings, bone_name, has_parent, assumed_model=None):
    """Every reason SOFT cannot be authored, as (field, detail) or None.

    These are the FEM paths this engine does NOT implement yet. They are refused
    here, by name, so the panel can never claim support and the author intent is
    never silently dropped.

    *assumed_model* evaluates the refusal for a physical body these settings do not
    declare yet. The explicit Change Physical Body operator needs exactly that: it
    must refuse an unsupported switch BEFORE it writes anything, and at that moment
    ``settings.physical_model`` still names the OLD body. Passing None (the
    default) keeps every existing caller on the declared model, unchanged.
    """
    if (settings.physical_model if assumed_model is None else assumed_model) != 'SOFT':
        return None
    if has_parent and (settings.root_mode != 'FREE' or settings.root_control_events):
        return ('root_mode', '当前版本只支持软体根骨骼的静态或运动学控制；此骨骼已有物理父级或自动父锚点，请保持动态模式')
    if has_parent and settings.joint_mode != 'SOFT':
        return ('joint_mode', '软体端点不支持硬球关节、硬铰链或硬 D6 关节；请选择柔性关节')
    return None


def check_soft_supported(settings, bone_name, has_parent, assumed_model=None):
    refusal = soft_material_refusal(settings, bone_name, has_parent, assumed_model=assumed_model)
    if refusal is not None:
        raise _refuse(refusal[0], bone_name, refusal[1])
    return True


def collect_body_controls_contract(context):
    """Pure author intent snapshot for the encoder and main-thread producer gate.

    Never samples animation, changes frames, generates identities or touches caches.
    Old boundary KINEMATIC/POSE input retains its legacy optional-field vocabulary.
    """
    controls=[]
    extended=False
    for arm in context.scene.objects:
        if arm.type!='ARMATURE':continue
        for bone in arm.data.bones:
            settings=bone.ppf_bone_physics
            if not settings.enabled or (settings.root_mode=='FREE' and not settings.root_control_events):continue
            events=[{'frame':float(event.frame),'mode':event.mode,'pose_binding':event.pose_binding if event.mode!='FREE' else 'ABSOLUTE'} for event in settings.root_control_events]
            controls.append({'body_uuid':settings.body_uuid,'mode':settings.root_mode,'source':settings.root_motion_source,'events':events})
            incoming=bool(bone.parent and (bone.parent.ppf_bone_physics.enabled
                                            or getattr(settings, 'auto_follow_parent', True)))
            outgoing=any(child.ppf_bone_physics.enabled for child in bone.children)
            if (settings.physical_model=='SOFT' or incoming or not outgoing or settings.root_mode=='STATIC'
                    or settings.root_mode=='FREE' and events
                    or any(event['mode']=='STATIC' or event['mode']!='FREE' and event['pose_binding']=='CURRENT' for event in events)):
                extended=True
    controls.sort(key=lambda row:row['body_uuid'])
    return {'body_controls_v1':extended,'controls':controls}


def hard_settings(settings):
    if settings.joint_mode == 'HINGE':
        twist = {'mode': settings.hinge_twist_mode}
        if twist['mode'] == 'bounded':
            twist.update(lower_rad=float(settings.hinge_lower), upper_rad=float(settings.hinge_upper))
        return 'angular_hard_hinge_v1', {'coordinate_model': MODEL, 'twist_x': twist}
    if settings.joint_mode == 'D6':
        result = {'coordinate_model': 'parent_local_d6_swing_twist_x_v1'}
        for kind in ('translation', 'rotation'):
            result[kind] = {}
            for axis in 'xyz':
                settings_axis = getattr(settings, 'd6_' + kind + '_' + axis)
                item = {'mode': settings_axis.mode}
                if item['mode'] == 'limited':item.update(lower=float(settings_axis.lower), upper=float(settings_axis.upper))
                elif item['mode'] == 'locked':item['value'] = float(settings_axis.value)
                result[kind][axis] = item
        return 'joint_d6_hard_v1', result
    return None


def collect_bindings(context):
    """JSON-safe authoring contract for the original transfer/build encoder.

    Matrices/positions use Blender Z-up scene units. The encoder owns the single
    unit/axis conversion and authoritative native body-local frame construction.
    Missing proxies/groups or duplicate identities refuse transfer.

    The document schema is unchanged (``PPF_BLENDER_BONES_V1``) and so is the
    binding record: the physical body choice lives ONLY in the bone authoring
    (and in the actual material group it drives), never in this document. The
    consumer infers the real body kind from native/statistics membership and
    receives the SOLID material through the normal group transport, so adding a
    redundant marker here would only have to be ignored -- or, as the strict
    mixed-joint installer actually does, refused as an unsupported field.
    """
    from .groups import get_group_by_uuid, has_addon_data
    from ..core.uuid_registry import get_object_uuid
    from mathutils import Euler
    from .bone_collision import settings_record
    out = {"schema": "PPF_BLENDER_BONES_V1", "coordinate_space": "BLENDER_Z_UP", "scene_unit_scale_m": float(context.scene.unit_settings.scale_length), "frame_start": int(context.scene.frame_current), "armatures": [], "bindings": []}
    # Preserve legacy enum values for old files, but refuse their transfer.
    for arm in context.scene.objects:
        if arm.type == 'ARMATURE':
            for bone in arm.data.bones:
                settings = bone.ppf_bone_physics
                if settings.enabled and settings.joint_mode != 'SOFT':
                    raise _error('{name}: 已保存的硬关节不再支持；请显式选择柔性关节', name=bone.name)
    has_hard = any(b.ppf_bone_physics.enabled and b.parent and (b.parent.ppf_bone_physics.enabled or getattr(b.ppf_bone_physics, 'auto_follow_parent', True)) and b.ppf_bone_physics.joint_mode in {'HINGE', 'D6'} for arm in context.scene.objects if arm.type == 'ARMATURE' for b in arm.data.bones)
    validate_final_binding_ids(context.scene,None,[])
    validate_proxy_participation(context.scene)
    seen_armatures, seen_bones, seen_bodies = set(), set(), set()
    automatic_parent_sources = {}
    mass_directives=[]
    for arm in sorted((o for o in context.scene.objects if o.type == "ARMATURE"), key=lambda o: o.name):
        enabled = [b for b in arm.data.bones if b.ppf_bone_physics.enabled]
        if not enabled:
            continue
        for bone in enabled:
            validate_proxy_owner(arm,bone)
            settings=bone.ppf_bone_physics
            if not arm.ppf_armature_uuid or any(not getattr(settings,key) for key in ('bone_uuid','body_uuid','joint_uuid','group_uuid')):
                raise _error("{name}: create/update a physics proxy before transfer",name=bone.name)
        if arm.ppf_armature_uuid in seen_armatures:
            raise _error("Duplicate armature identity; duplicated rigs need independent bindings")
        seen_armatures.add(arm.ppf_armature_uuid)
        out["armatures"].append({"armature_uuid": arm.ppf_armature_uuid, "object_name": arm.name, "matrix_world_blender": _matrix_rows(arm.matrix_world)})
        for bone in sorted(enabled, key=lambda b: b.name):
            s = bone.ppf_bone_physics; proxy = s.proxy
            if not proxy or proxy.type != "MESH" or context.scene.objects.get(proxy.name) != proxy:
                raise _error("{name}: create/update a physics proxy before transfer", name=bone.name)
            if get_object_uuid(proxy) != s.body_uuid or not s.body_uuid:
                raise _error("{name}: proxy UUID changed; update its binding", name=bone.name)
            if s.bone_uuid in seen_bones or s.body_uuid in seen_bodies:
                raise _error("Duplicate bone or body identity; do not share a proxy between bones")
            seen_bones.add(s.bone_uuid);seen_bodies.add(s.body_uuid)
            parent = bone.parent if bone.parent and bone.parent.ppf_bone_physics.enabled else None
            nonphysical_parent = bone.parent if bone.parent and parent is None else None
            auto_parent = bool(nonphysical_parent and getattr(s, 'auto_follow_parent', True))
            has_joint_parent = bool(parent or auto_parent)
            if auto_parent and (s.root_mode != 'FREE' or len(s.root_control_events)):
                raise _error("{name}: disable Follow Nonphysical Parent to use root control or release events", name=bone.name)
            # The physical body choice is validated before any material field is
            # read, and it refuses rather than dropping an unsupported control.
            check_soft_supported(s, bone.name, has_joint_parent)
            group = get_group_by_uuid(context.scene, s.group_uuid) if has_addon_data(context.scene) else None
            expected_type = "SOLID" if s.physical_model == 'SOFT' else "PDRD"
            if not group or not group.active or group.object_type != expected_type or not any(a.uuid == s.body_uuid and a.included for a in group.assigned_objects):
                raise _error("{name}: proxy must be included in its actual {kind} group", name=bone.name, kind="SOLID" if s.physical_model == 'SOFT' else "PDRD")
            if s.is_property_set("density") and abs((group.solid_density if expected_type == "SOLID" else group.pdrd_density) - s.density) > 1e-4:
                raise _error("{name}: update proxy to synchronize density", name=bone.name)
            if expected_type == "SOLID":
                material = effective_solid_material(s, group)
                from ..core.bone_preset_apply import _validate_rna
                for field, value in material.items():
                    _validate_rna(s, field, value)
            if not math.isfinite(group.friction) or group.friction < 0:
                raise _error("{name}: proxy friction must be finite and nonnegative", name=bone.name)
            if any(lo > hi for lo, hi in zip(s.soft_lower, s.soft_upper)):
                raise _error("{name}: soft lower bound exceeds upper bound", name=bone.name)
            if parent:
                ps = parent.ppf_bone_physics
                if not ps.proxy or not ps.body_uuid:
                    raise _error("{name}: enabled parent needs its physics proxy", name=bone.name)
            if has_joint_parent and s.joint_mode == 'BALL_FIXED' and parent and parent.ppf_bone_physics.physical_model == 'SOFT':
                raise _error('{name}: 硬球关节不支持软体父级', name=bone.name)
            bone_world, author_scale = bone_world_frame_scale(arm, bone)
            proxy_world = rigid_matrix(proxy.matrix_world.copy(), proxy.name)
            axis = Matrix.Rotation(math.pi/2, 4, 'Z') @ Euler(tuple(s.joint_axis_euler)).to_matrix().to_4x4()
            physics = {"density_kg_per_m3": float(s.density if s.is_property_set("density") else (group.solid_density if expected_type == "SOLID" else group.pdrd_density)), "anchor_stiffness_N_per_m": float(s.anchor_stiffness), "collide_connected": bool(s.collide_connected), "angular_control_v1": {"coordinate_model": MODEL, "stiffness_Nm_per_rad": dict(zip(AXES, s.angular_stiffness)), "damping_Nms_per_rad": dict(zip(AXES, s.angular_damping))}, "angular_soft_limits_v1": {"coordinate_model": MODEL, **{a: {"enabled": bool(s.soft_limits_enabled[i]), "lower_rad": float(s.soft_lower[i]), "upper_rad": float(s.soft_upper[i]), "stiffness_Nm_per_rad": float(s.soft_stiffness[i])} for i, a in enumerate(AXES)}}}
            if s.soft_limit_mode=='CONE':
                limits=physics['angular_soft_limits_v1']
                for a in AXES[1:]:limits[a]['enabled']=False
                limits['swing_cone']=dict(enabled=True,half_angle_rad=float(s.swing_cone_half_angle),stiffness_Nm_per_rad=float(s.swing_cone_stiffness))
            if has_joint_parent and s.joint_mode == 'BALL_FIXED':
                physics['translation_constraint_v1'] = {'mode': 'FIXED'}
            if has_hard and has_joint_parent and s.joint_mode == 'BALL_FIXED':
                raise _error('{name}: 硬球关节不能与旧硬铰链或硬 D6 同场景使用', name=bone.name)
            if has_hard:
                physics.pop('angular_soft_limits_v1', None)
            hard = hard_settings(s)
            if has_joint_parent and hard:
                physics[hard[0]] = hard[1]
                if s.joint_mode == 'HINGE':
                    for key in ('stiffness_Nm_per_rad','damping_Nms_per_rad'):
                        physics['angular_control_v1'][key]['swing_y'] = 0.
                        physics['angular_control_v1'][key]['swing_z'] = 0.
                if s.joint_mode == 'D6':
                    physics['angular_control_v1']['stiffness_Nm_per_rad'] = dict.fromkeys(AXES, 0.)
                    physics['angular_control_v1']['damping_Nms_per_rad'] = dict.fromkeys(AXES, 0.)
            row = {"armature_uuid": arm.ppf_armature_uuid, "bone_uuid": s.bone_uuid, "bone_name": bone.name, "body_uuid": s.body_uuid, "proxy_object_name": proxy.name, "group_uuid": s.group_uuid, "parent_bone_uuid": parent.ppf_bone_physics.bone_uuid if parent else None, "parent_body_uuid": parent.ppf_bone_physics.body_uuid if parent else None, "joint_uuid": s.joint_uuid, "enabled": True, "root_mode": s.root_mode, "root_boundary": {"is_root": parent is None, "nonphysical_parent_bone": bone.parent.name if bone.parent and parent is None else None}, "bone_world_matrix_blender": _rigid_matrix_rows(bone_world, bone.name), "proxy_world_matrix_blender": _rigid_matrix_rows(proxy_world, proxy.name), "body_to_bone_rest_blender": _rigid_matrix_rows(proxy_world.inverted() @ bone_world, bone.name), "joint_world_matrix_blender": _rigid_matrix_rows(bone_world @ axis, bone.name), "use_auto_frames": bool(s.use_auto_frames), "physics": physics}
            if author_scale != 1.:
                row['author_uniform_scale'] = author_scale
            if auto_parent:
                from ..core.encoder.automatic_parent_anchors import parent_anchor_identity
                parent_bone_uuid, anchor_body_uuid = parent_anchor_identity(
                    arm.ppf_armature_uuid,
                    getattr(nonphysical_parent.ppf_bone_physics, 'bone_uuid', '') or None,
                    nonphysical_parent.name,
                )
                source = (arm.ppf_armature_uuid, nonphysical_parent.name)
                previous = automatic_parent_sources.setdefault(anchor_body_uuid, source)
                if previous != source:
                    raise _error('Automatic parent identity conflict between {first} and {second}; rename the parent without a bone UUID before transfer', first=previous[1], second=nonphysical_parent.name)
                parent_world = bone_world_matrix(arm, nonphysical_parent)
                row['automatic_parent_anchor_v1'] = {
                    'version': 1,
                    'anchor_body_uuid': anchor_body_uuid,
                    'parent_bone_uuid': parent_bone_uuid,
                    'parent_bone_name': nonphysical_parent.name,
                    'parent_world_matrix_blender': _rigid_matrix_rows(parent_world, nonphysical_parent.name),
                    'anchor_proxy_world_matrix_blender': _rigid_matrix_rows(parent_world, nonphysical_parent.name),
                }
            if not s.use_auto_frames:
                row.update(parent_local_frame=_frame(s, "parent"), child_local_frame=_frame(s, "child"))
            out["bindings"].append(row)
            if s.mass_mode=='MASS' and s.physical_model=='RIGID':
                target=float(s.target_mass_kg)
                if not math.isfinite(target) or target<=0:
                    raise _error('{name}: target mass must be finite and positive',name=bone.name)
                mass_directives.append({'armature_uuid':arm.ppf_armature_uuid,'bone_uuid':s.bone_uuid,
                    'body_uuid':s.body_uuid,'proxy_object_uuid':s.body_uuid,'target_mass_kg':target,
                    'physical_model':'PDRD_UNIFORM_POINTS_V1','source':'AUTHOR_TARGET_KG'})
    from .bone_collision import collect_collision_filters
    collision = collect_collision_filters(context)["objects"]
    if collision:
        out["collision_filter_v1"] = {"rule": "symmetric_membership_mask_and_v1", "objects": collision}
    if mass_directives:
        out['bone_mass_intent_v1']={'kind':'ppf.bone_mass_intent','version':1,'directives':mass_directives}
    return out


def register():
    from . import bone_collision
    bone_collision.register()
    for cls in (PPF_BoneControlEvent, PPF_BoneDriveTarget, PPF_BoneDriveChannel, PPF_BoneDriveAxis, PPF_BoneD6Axis, PPF_BonePhysics):_ppf_i18n.register_class(cls)
    bpy.types.Bone.ppf_bone_physics = PointerProperty(type=PPF_BonePhysics)
    bpy.types.Object.ppf_armature_uuid = StringProperty(options=NOT_ANIMATED)
    from ..core import bone_mass
    bone_mass.register()
    bone_collision.schedule_collision_baseline()


def unregister():
    from ..core import bone_mass
    bone_mass.unregister()
    del bpy.types.Object.ppf_armature_uuid
    del bpy.types.Bone.ppf_bone_physics
    for cls in reversed((PPF_BoneControlEvent, PPF_BoneDriveTarget, PPF_BoneDriveChannel, PPF_BoneDriveAxis, PPF_BoneD6Axis, PPF_BonePhysics)):bpy.utils.unregister_class(cls)
    from . import bone_collision
    bone_collision.unregister()
