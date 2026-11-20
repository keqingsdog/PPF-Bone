"""UI-only selected-author transactions; internal RNA writers stay single-target."""
from ..i18n import props as _ppf_props
from .. import i18n as _ppf_i18n
import bpy
import math
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from types import SimpleNamespace
from ..i18n.props import FloatProperty,PointerProperty,StringProperty
from ..models.bone_bindings import PPF_BonePhysics

# Explicit author whitelist: no identity, proxy pointer, material-body switch,
# participation, nested events, display diagnostics or geometry provenance.
FIELDS=('collide_connected','collision_filter_enabled','collision_group_layers','collision_mask_layers',
    'shape','proxy_quality','new_proxy_quality','proxy_face_mode','radius','length_ratio','box_width','box_depth',
    'new_binding_physical_model','mass_mode','target_mass_kg','density','material_preset',
    'solid_model','solid_young_modulus','solid_young_mpa','solid_poisson_ratio','solid_deformation_damping',
    'auto_follow_parent','root_mode','root_motion_source','anchor_stiffness','angular_stiffness','angular_damping',
    'soft_limits_enabled','soft_lower','soft_upper','soft_stiffness','soft_stiffness_degree',
    'soft_limit_mode','swing_cone_half_angle','swing_cone_stiffness','swing_cone_stiffness_degree',
    'soft_limit_policy','soft_limit_strength','preview_color_custom','preview_color')
COLLISION={'collide_connected','collision_filter_enabled','collision_group_layers','collision_mask_layers'}
DIMENSIONS={'shape','radius','length_ratio','box_width','box_depth'}
MATERIAL={'solid_model','solid_young_modulus','solid_young_mpa','solid_poisson_ratio','solid_deformation_damping'}
AUTHOR_FIELDS={'solid_young_mpa':'solid_young_modulus',
               'soft_stiffness_degree':'soft_stiffness',
               'swing_cone_stiffness_degree':'swing_cone_stiffness'}
_busy=False
_draw_reads=ContextVar('ppf_bone_parameter_draw_reads',default=None)


def _draw_context_key(context):
    def pointer(value):return value.as_pointer() if value is not None else 0
    obj=context.object
    bone=obj.data.bones.active if obj is not None and obj.type=='ARMATURE' else None
    return (pointer(context.scene),pointer(getattr(context,'view_layer',None)),
            pointer(getattr(context,'window',None)),pointer(getattr(context,'region',None)),
            context.mode,pointer(obj),bone.name if bone is not None else '')


def _draw_cache(context):
    state=_draw_reads.get()
    return state['values'] if state is not None and state['key']==_draw_context_key(context) else None


@contextmanager
def draw_read_scope(context):
    """Share read-only RNA resolution for exactly one panel draw."""
    existing=_draw_reads.get()
    key=_draw_context_key(context)
    if existing is not None and existing['key']==key:
        yield
        return
    state={'key':key,'values':{}}
    token=_draw_reads.set(state)
    try:yield
    finally:
        state['values'].clear()  # A retained scope state must not retain owner RNA.
        _draw_reads.reset(token)


@contextmanager
def _live_read_scope():
    previous=_draw_reads.get()
    token=_draw_reads.set(None)
    try:yield
    finally:
        if previous is not None:previous['values'].clear()
        _draw_reads.reset(token)


def _fresh_reads(function):
    @wraps(function)
    def call(*args,**kwargs):
        with _live_read_scope():return function(*args,**kwargs)
    return call


def _proxy_owner(scene,proxy):
    pairs=[(arm,bone) for arm in scene.objects if arm.type=='ARMATURE' for bone in arm.data.bones
           if bone.ppf_bone_physics.proxy==proxy]
    if len(pairs)!=1:raise ValueError(proxy.name+'：代理绑定不唯一')
    return pairs[0]


def _resolve_selected_authors_live(context):
    """Actual current selection only; never infer other rigs from Bone.select."""
    active=context.object
    selected=tuple(context.selected_objects)
    if active is None or active not in selected:return []
    rows=[]
    if context.mode=='POSE':
        for pb in context.selected_pose_bones or ():
            arm=pb.id_data
            if arm in selected and context.scene.objects.get(arm.name)==arm:
                rows.append((arm,pb.bone))
    elif context.mode=='OBJECT':
        for obj in selected:
            if obj.type=='MESH' and obj.get('ppf_bone_proxy_owned'):
                rows.append(_proxy_owner(context.scene,obj))
        if not rows and active.type=='ARMATURE' and active.data.bones.active is not None:
            rows=[(active,active.data.bones.active)]
    unique=[];seen=set()
    for arm,bone in rows:
        key=(arm.as_pointer(),bone.name)
        if key not in seen:unique.append((arm,bone));seen.add(key)
    return unique


def resolve_selected_authors(context):
    cache=_draw_cache(context)
    if cache is None:return _resolve_selected_authors_live(context)
    if 'selected' not in cache:cache['selected']=_resolve_selected_authors_live(context)
    return list(cache['selected'])


def _active_author_live(context):
    obj=context.object
    if obj is None:return None
    if obj.type=='ARMATURE' and obj.data.bones.active is not None:
        return obj,obj.data.bones.active
    if obj.type=='MESH' and obj.get('ppf_bone_proxy_owned'):
        try:return _proxy_owner(context.scene,obj)
        except ValueError:return None
    return None


def active_author(context):
    cache=_draw_cache(context)
    if cache is None:return _active_author_live(context)
    if 'active' not in cache:cache['active']=_active_author_live(context)
    return cache['active']


def _group_live(context,arm,bone):
    from ..ops.bone_proxy import owned_bone_material
    return owned_bone_material(context.scene,arm,bone) if bone.ppf_bone_physics.proxy else None


def _group(context,arm,bone):
    cache=_draw_cache(context)
    if cache is None:return _group_live(context,arm,bone)
    key=('group',arm.as_pointer(),bone.name)
    if key not in cache:cache[key]=_group_live(context,arm,bone)
    return cache[key]


def _value_owner(context,field):
    source=active_author(context)
    if source is None:return None
    return _group(context,*source) if field=='friction' else source[1].ppf_bone_physics


def _plain(value):
    return value if isinstance(value,(str,bool,int,float)) else list(value)


def _equal(a,b):
    if isinstance(a,(list,tuple)) or isinstance(b,(list,tuple)):
        return len(a)==len(b) and all(_equal(x,y) for x,y in zip(a,b))
    if isinstance(a,(int,float)) and isinstance(b,(int,float)) and not isinstance(a,bool) and not isinstance(b,bool):
        return math.isclose(a,b,rel_tol=2.e-7,abs_tol=1.e-10)
    return a==b


def _animated(owner,key):
    """Never redirect an author animation/driver property to WindowManager RNA."""
    prop=owner.bl_rna.properties.get(key)
    block=owner.id_data
    from .utils import get_id_fcurves
    anim=getattr(block,'animation_data',None)
    curves=list(get_id_fcurves(block))+list(getattr(anim,'drivers',()))
    if getattr(anim,'use_nla',True):
        tracks=list(getattr(anim,'nla_tracks',()))
        solo=any(getattr(track,'is_solo',False) and not track.mute for track in tracks)
        def strip_curves(strip):
            action=getattr(strip,'action',None)
            if action is not None:
                # Reuse the slot-aware reader; never take another ID's first bag.
                view=SimpleNamespace(animation_data=SimpleNamespace(action=action,action_slot=getattr(strip,'action_slot',None)))
                curves.extend(get_id_fcurves(view))
            for child in getattr(strip,'strips',()):strip_curves(child)
        for track in tracks:
            if track.mute or solo and not getattr(track,'is_solo',False):continue
            for strip in track.strips:strip_curves(strip)
    if not curves:return False
    try:path=owner.path_from_id(key)
    except ValueError:
        if isinstance(owner,PPF_BonePhysics):
            bones=[b for b in block.bones if b.ppf_bone_physics==owner]
            if len(bones)!=1:return True
            path=bones[0].path_from_id()+'.ppf_bone_physics.'+key
        else:return True
    return any(f.data_path==path for f in curves)


@_fresh_reads
def set_selected_parameter(context,field,value,*,assign=None,_indices=None):
    """Preflight and commit a single common field; rollback precedes one Clear."""
    global _busy
    if _busy:raise ValueError('参数批量修改正在进行')
    if field not in FIELDS and field!='friction':raise ValueError('不支持的批量作者字段：'+field)
    if field=='friction' and context.scene.tool_settings.use_keyframe_insert_auto:
        raise ValueError('自动关键帧开启：摩擦请沿原材质RNA单项编辑')
    rows=resolve_selected_authors(context)
    source=active_author(context)
    if not rows or source is None or not any(a==source[0] and b==source[1] for a,b in rows):
        raise ValueError('请选中要统一修改的骨骼或已绑定代理')
    from ..models.bone_bindings import validate_proxy_owner,check_soft_supported,has_physical_joint_parent
    from .bone_proxy_visibility import owned_proxy
    from ..models.bone_collision import collision_edit_batch
    from ..ops.restoring_curve_operators import snapshot_rna,restore_rna
    from .bone_preset_apply import _validate_rna
    from .bone_mass import material_write_guard,mark_settings
    from .facade import communicator
    communicator.assert_bone_body_switch_idle()
    value=_plain(value);entries=[];group_snapshots=[];seen_groups=set()
    for arm,bone in rows:
        validate_proxy_owner(arm,bone)
        if context.scene.objects.get(arm.name)!=arm:raise ValueError('目标不在当前场景')
        settings=bone.ppf_bone_physics;group=_group(context,arm,bone)
        if settings.proxy:
            if not owned_proxy(context.scene,arm,bone,settings.proxy):
                raise ValueError(bone.name+'：代理不在当前场景或绑定身份不唯一，请先修复绑定')
            if group is None:raise ValueError(bone.name+'：代理没有独占材质')
            if group.object_type!={'RIGID':'PDRD','SOFT':'SOLID'}.get(settings.physical_model):
                raise ValueError(bone.name+'：代理材质类型与骨骼物理类型不一致，请先修复绑定')
        owner=group if field=='friction' else settings
        if owner is None:raise ValueError(bone.name+'：尚无可编辑的独占材质')
        if _animated(owner,field):raise ValueError(bone.name+'：动画或驱动字段请沿原作者RNA编辑')
        # UI unit aliases write the underlying author property as well.
        author_field=AUTHOR_FIELDS.get(field,field)
        if author_field!=field and _animated(owner,author_field):
            raise ValueError(bone.name+'：动画或驱动字段请沿原作者RNA编辑')
        # The actual SOLID material may carry supported animation independently
        # of its non-animatable BonePhysics editor. Preflight only this write's
        # destination, before any selected author or material value is changed.
        if field in MATERIAL and group is not None and group.object_type=='SOLID':
            material_field='deformation_damping' if field=='solid_deformation_damping' else author_field
            if _animated(group,material_field):
                raise ValueError(bone.name+'：材质动画或驱动字段请沿原材质RNA编辑')
        target_value=value
        if _indices is not None:
            prop=owner.bl_rna.properties.get(field)
            if prop is None or not prop.is_array or any(type(i) is not int or not 0<=i<prop.array_length for i in _indices):
                raise ValueError('批量向量分量无效')
            target_value=_plain(getattr(owner,field))
            for index in _indices:target_value[index]=value[index]
        _validate_rna(owner,field,target_value)
        keys={field,'auto_geometry_snapshot_json','recreate_geometry_intent_json'} if owner==settings else {field}
        if field in MATERIAL: keys.update(MATERIAL - {'solid_young_mpa'})
        if field=='soft_stiffness_degree':keys.add('soft_stiffness')
        if field=='swing_cone_stiffness_degree':keys.add('swing_cone_stiffness')
        old={k:(owner.is_property_set(k),_plain(getattr(owner,k))) for k in keys}
        entries.append((arm,bone,owner,old,target_value))
        if group is not None and group.as_pointer() not in seen_groups:
            seen_groups.add(group.as_pointer());group_snapshots.append((group,snapshot_rna(group)))
    def current(entry):
        if field in MATERIAL:
            from ..models.bone_bindings import effective_solid_material
            values=effective_solid_material(entry[2])
            return values['solid_young_modulus']/1e6 if field=='solid_young_mpa' else values[field]
        return _plain(getattr(entry[2],field))
    # Equal RNA fallback values still need an explicit author write. First-creation
    # planning deliberately distinguishes unset values from authored defaults.
    # Unit aliases store that intent on their backing property, not the getter.
    authored_field=AUTHOR_FIELDS.get(field,field)
    changed=[entry for entry in entries
             if not entry[3][authored_field][0] or not _equal(current(entry),entry[4])]
    if not changed:return 0
    assign=assign or setattr;_busy=True
    try:
        with collision_edit_batch():
            try:
                with material_write_guard():
                    for _arm,_bone,owner,_old,target_value in changed:
                        assign(owner,field,target_value)
                        if field not in MATERIAL and not _equal(_plain(getattr(owner,field)),target_value):raise ValueError('参数赋值被拒绝')
                from ..models import bone_bindings as bindings
                for _arm,bone,owner,_old,_target_value in changed:
                    settings=bone.ppf_bone_physics
                    if field=='density':bindings._sync_density(settings,context)
                    elif field in MATERIAL:
                        key='solid_young_modulus' if field=='solid_young_mpa' else field
                        bindings.apply_solid_material_patch(settings,{key: _target_value * 1e6 if field=='solid_young_mpa' else _target_value})
                    elif field in DIMENSIONS:bindings._cancel_recreate_geometry_intent(settings,context)
                    elif field in {'mass_mode','target_mass_kg'}:mark_settings(settings,context)
                    if field in {'root_mode','root_motion_source','auto_follow_parent'}:
                        check_soft_supported(settings,bone.name,has_physical_joint_parent(bone))
            except Exception:
                with material_write_guard():
                    for _arm,_bone,owner,old,_target_value in reversed(entries):
                        for key,(was_set,previous) in old.items():
                            if key in AUTHOR_FIELDS: continue  # Restore backing RNA only, never computed aliases.
                            setattr(owner,key,previous)
                            if not was_set:owner.property_unset(key)
                    for group,snapshot in group_snapshots:restore_rna(group,snapshot)
                raise
    finally:_busy=False
    return len(changed)


# DeferredProperty.function is the original native factory even when created
# through i18n.props. Identity checks intentionally use original_factory; this is not
# a property-construction bypass.
def _get(field,definition):
    def getter(_self):
        owner=_value_owner(bpy.context,field)
        if owner is None:
            value=definition.keywords.get('default',tuple(0. for _ in range(definition.keywords['size'])) if 'size' in definition.keywords else 0.)
        else:
            value=getattr(owner,field)
            if field in MATERIAL:
                from ..models.bone_bindings import effective_solid_material
                values=effective_solid_material(owner)
                value=values['solid_young_modulus']/1e6 if field=='solid_young_mpa' else values[field]
            if field=='density' and not owner.is_property_set(field):
                from ..models.new_binding_defaults import new_binding_plan
                value=new_binding_plan(owner).get(field,value)
        if definition.function==_ppf_props.original_factory('EnumProperty'):
            items=definition.keywords['items']
            return next((item[-1] if len(item)>3 and isinstance(item[-1],int) else i for i,item in enumerate(items) if item[0]==value),0)
        return tuple(value) if not isinstance(value,(str,int,float,bool)) else value
    return getter


def _set(field,definition):
    def apply_value(self,value):
        if definition.function==_ppf_props.original_factory('EnumProperty'):
            value=next(item[0] for i,item in enumerate(definition.keywords['items'])
                       if (item[-1] if len(item)>3 and isinstance(item[-1],int) else i)==value)
        try:
            indices=None
            if definition.function in tuple(_ppf_props.original_factory(name) for name in ('FloatVectorProperty','BoolVectorProperty','IntVectorProperty')):
                owner=_value_owner(bpy.context,field)
                previous=_plain(getattr(owner,field)) if owner is not None else []
                indices=[i for i,(old,new) in enumerate(zip(previous,value)) if not _equal(old,new)]
                if not indices:self.error='';return
            set_selected_parameter(bpy.context,field,value,_indices=indices);self.error=''
        except Exception as exc:
            self.error=str(exc)
            from ..models.console import console
            console.write('批量参数未应用：'+str(exc))
    def setter(self,value):
        # Blender RNA requires exactly two positional callback arguments.
        with _live_read_scope():return apply_value(self,value)
    return setter


class PPF_PT_BoneFrictionAnimation(bpy.types.Panel):
    bl_idname='PPF_PT_bone_friction_animation'
    bl_label='摩擦关键帧'
    bl_space_type='VIEW_3D'
    bl_region_type='UI'
    bl_options={'INSTANCED'}
    bl_ui_units_x=15

    def draw(self,context):
        owner=_value_owner(context,'friction')
        if owner is None:
            self.layout.label(text='活动骨骼没有独占材质',icon='INFO')
            return
        self.layout.label(text='活动骨骼关键帧（不批量）',icon='KEY_HLT')
        self.layout.use_property_decorate=True
        self.layout.prop(owner,'friction',text='摩擦系数')


class PPF_SelectionParameters(bpy.types.PropertyGroup):
    error:StringProperty(default='',options={'SKIP_SAVE'})


def register():
    definitions={key:PPF_BonePhysics.__annotations__[key] for key in FIELDS}
    definitions['friction']=FloatProperty(name='摩擦系数',default=.3,min=0.,max=1.,description='修改所选骨骼的接触摩擦；不同值显示活动骨骼值，输入后统一。关键帧请展开右侧按钮，使用活动骨骼真实材质字段',options=set())
    for key,definition in definitions.items():
        kwargs={k:v for k,v in definition.keywords.items() if k not in {'get','set','update','options'}}
        kwargs.update(get=_get(key,definition),set=_set(key,definition),options={'SKIP_SAVE'})
        PPF_SelectionParameters.__annotations__[key]=definition.function(**kwargs)
    _ppf_i18n.register_class(PPF_PT_BoneFrictionAnimation)
    _ppf_i18n.register_class(PPF_SelectionParameters)
    bpy.types.WindowManager.ppf_selection_parameters=PointerProperty(type=PPF_SelectionParameters,options={'SKIP_SAVE'})


def unregister():
    del bpy.types.WindowManager.ppf_selection_parameters
    bpy.utils.unregister_class(PPF_SelectionParameters)
    bpy.utils.unregister_class(PPF_PT_BoneFrictionAnimation)


def parameter_owner(context,owner,field):
    if field not in FIELDS and field!='friction':return owner
    if field!='friction' and not isinstance(owner,PPF_BonePhysics):return owner
    if context.mode not in {'POSE','OBJECT'}:return owner
    source=active_author(context)
    rows=resolve_selected_authors(context)
    if source is None or not any(a==source[0] and b==source[1] for a,b in rows):return owner
    expected=_value_owner(context,field)
    prop=owner.bl_rna.properties.get(field)
    if field=='friction' and context.scene.tool_settings.use_keyframe_insert_auto:return owner
    if field!='friction' and prop is not None and prop.is_animatable:return owner
    if owner!=expected or _animated(owner,field):return owner
    return context.window_manager.ppf_selection_parameters


class SelectionLayout:
    """Route only whitelisted active author props; keep all other UI RNA intact."""
    def __init__(self,layout,context):object.__setattr__(self,'layout',layout);object.__setattr__(self,'context',context)
    def __getattr__(self,key):
        member=getattr(self.layout,key)
        if key in {'row','column','box','split','grid_flow'}:
            return lambda *a,**kw:SelectionLayout(member(*a,**kw),self.context)
        if key=='panel':
            def panel(*a,**kw):return tuple(SelectionLayout(x,self.context) if x is not None else None for x in member(*a,**kw))
            return panel
        if key in {'prop','prop_enum'}:
            def prop(owner,field,*a,**kw):
                target=parameter_owner(self.context,owner,field)
                if key=='prop' and field=='friction' and owner==_value_owner(self.context,field):
                    row=self.layout.row(align=True);row.use_property_decorate=False
                    friction_kw=dict(kw)
                    friction_kw['text']='摩擦系数（所选）' if target!=owner else '活动骨骼摩擦（动画）'
                    row.prop(target,field,*a,**friction_kw)
                    if target==owner:
                        row.prop_decorator(owner,field)
                    else:
                        row.popover(panel='PPF_PT_bone_friction_animation',text='',icon='KEY_HLT')
                    rows=resolve_selected_authors(self.context)
                    if target==owner and len(rows)>1:
                        self.layout.label(text='动画或自动关键帧：仅修改活动骨骼，不批量',icon='KEY_HLT')
                    elif len(rows)>1:
                        from ..ops.bone_proxy import selected_bone_materials
                        values = [group.friction for group in selected_bone_materials(self.context.scene, rows)]
                        if values and any(not _equal(values[0],value) for value in values[1:]):
                            self.layout.label(text='所选摩擦值不同；输入将统一',icon='INFO')
                else:return member(target,field,*a,**kw)
            return prop
        return member
    def __setattr__(self,key,value):setattr(self.layout,key,value)
