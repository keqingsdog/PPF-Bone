"""Portable author capture and one prevalidated, rollback-safe scene transaction."""
from copy import deepcopy
from types import SimpleNamespace
import math,uuid
from . import bone_preset_library as lib


def _plain(value):
    if isinstance(value,(str,int,float,bool)) or value is None:return value
    return [_plain(v) for v in value]

def _read(owner,fields):return {key:_plain(getattr(owner,key)) for key in fields}

def _matrix(value):return [[float(v) for v in row] for row in value]

def _frame(translation,quaternion):
    from mathutils import Matrix,Quaternion
    result=Quaternion(quaternion).to_matrix().to_4x4();result.translation=translation;return result

def _author_frame(arm,bone):
    from ..models.bone_bindings import rigid_matrix
    world=arm.matrix_world@arm.pose.bones[bone.name].matrix
    try:
        return rigid_matrix(world,bone.name+'作者骨架')
    except ValueError as exc:
        raise ValueError(bone.name+'：请先在对象模式按 Ctrl+A 应用骨架对象缩放，并清除姿态缩放或剪切，再保存或应用用户预设') from exc

def _parent_frame(arm,bone):
    parent=bone.parent
    if parent is None:raise ValueError('手动关节帧需要目标父骨骼')
    settings=parent.ppf_bone_physics
    return settings.proxy.matrix_world.copy() if settings.enabled and settings.proxy else _author_frame(arm,parent)

def _capture_curves(scene,joint):
    from ..ops.restoring_curve_operators import read_state
    state=read_state(scene);document=state.get('document') or {'bindings':[],'tables':[]}
    tables={table['table_id']:table for table in document['tables']};channels={channel:None for channel in lib.CHANNELS}
    for binding in document['bindings']:
        target=binding['target']
        if target.get('kind')!='JOINT' or target.get('joint_uuid')!=joint:continue
        channel=target['channel']
        if channel not in channels or channels[channel] is not None:raise ValueError('恢复曲线通道不明确')
        channels[channel]={'binding':{k:deepcopy(v) for k,v in binding.items() if k not in ('target','table_id')},'table':{k:deepcopy(v) for k,v in tables[binding['table_id']].items() if k!='table_id'}}
    return {'mode':'LEGACY' if state['mode']=='LEGACY' else 'ACTIVE','channels':channels}

def capture(context,name):
    from ..models.bone_bindings import active_armature,active_bone,validate_proxy_owner
    from ..ops import bone_proxy as ops
    arm=active_armature(context);bone=active_bone(context)
    if arm is None or bone is None:raise ValueError('请选中要保存的物理骨骼')
    settings=bone.ppf_bone_physics
    if settings.proxy is None:raise ValueError('请先为当前骨骼添加物理代理')
    validate_proxy_owner(arm,bone);group=ops._owned_bone_material(context.scene,arm,bone)
    if group is None or group.object_type!=ops.OBJECT_TYPES[settings.physical_model]:raise ValueError('当前代理或独占材质失配')
    ops._facade_communicator().assert_bone_body_switch_idle()
    if ops.stale_simulation_output(context):raise ValueError('请先Clear模拟帧，再保存作者预设')
    proxy=settings.proxy
    if proxy.mode!='OBJECT' or any(mod.type=='MESH_CACHE' and mod.show_viewport for mod in proxy.modifiers):raise ValueError('请退出网格编辑并移除活动PC2播放后保存预设')
    from .bone_mass import _measure
    _measure(context.scene,settings) # production closed, outward, nonintersecting author geometry gate
    B=_author_frame(arm,bone);inverse=B.inverted()
    values=_read(settings,lib.BONE_FIELDS)
    from ..models.proxy_generation import effective_proxy_quality,effective_proxy_face_mode
    values['proxy_quality']=effective_proxy_quality(settings)
    values['proxy_face_mode']=effective_proxy_face_mode(settings)
    # Capture the actually owned material, not a stale fallback RNA value.
    values['density']=float(group.solid_density if group.object_type=='SOLID' else group.pdrd_density)
    if group.object_type=='SOLID':
        values['solid_model']=group.solid_model
        values['solid_young_modulus']=float(group.solid_young_modulus)*(values['density'] if group.young_mod_density_normalized else 1.)
        values['solid_poisson_ratio']=float(group.solid_poisson_ratio);values['solid_deformation_damping']=float(group.deformation_damping)
    drives={}
    for axis in lib.CHANNELS[:3]:
        drives[axis]={}
        for channel in ('position','velocity','torque'):
            owner=getattr(getattr(settings,'drive_'+axis),channel)
            drives[axis][channel]={**_read(owner,lib.DRIVE_FIELDS),'targets':[_read(item,('frame','value')) for item in owner.targets]}
    d6={key:_read(getattr(settings,'d6_'+key),('mode','lower','upper','value')) for key in (kind+'_'+axis for kind in ('translation','rotation') for axis in 'xyz')}
    # Manual frames are portable world joint frames expressed in the source bone,
    # then rebound to the target's actual parent/child local frames during Apply.
    if settings.use_auto_frames:
        Jp=Jc=B
    else:
        Jp=_parent_frame(arm,bone)@_frame(settings.parent_frame_translation,settings.parent_frame_quaternion)
        Jc=proxy.matrix_world@_frame(settings.child_frame_translation,settings.child_frame_quaternion)
    display={key:_plain(getattr(settings,key)) for key in lib.DISPLAY_FIELDS}
    return dict(kind=lib.KIND,version=lib.VERSION,id=None,name=name,source_bone_length=float(bone.length),settings=dict(bone=values,events=[_read(event,lib.EVENT_FIELDS) for event in settings.root_control_events],drives=drives,d6=d6),material=_read(group,lib.GROUP_FIELDS),display=display,geometry=dict(vertices=[list(v.co) for v in proxy.data.vertices],faces=[list(poly.vertices) for poly in proxy.data.polygons],bone_local_from_mesh=_matrix(inverse@proxy.matrix_world)),frames=dict(parent_joint_in_bone=_matrix(inverse@Jp),child_joint_in_bone=_matrix(inverse@Jc)),curves=_capture_curves(context.scene,settings.joint_uuid))

def _validate_rna(owner,key,value):
    prop=owner.bl_rna.properties.get(key)
    _validate_property(prop,key,value)

def _validate_property(prop,key,value):
    if prop is None or prop.is_readonly or prop.type in ('POINTER','COLLECTION'):raise ValueError('不支持的作者字段：'+key)
    array=getattr(prop,'is_array',False)
    values=value if array else [value]
    if array and (not isinstance(value,list) or len(value)!=prop.array_length):raise ValueError('作者数组长度无效：'+key)
    for item in values:
        if prop.type=='BOOLEAN' and type(item) is not bool:raise ValueError('作者开关无效：'+key)
        if prop.type=='ENUM' and item not in prop.enum_items.keys():raise ValueError('作者选项不支持：'+key)
        if prop.type=='INT' and type(item) is not int:raise ValueError('作者整数无效：'+key)
        if prop.type in ('INT','FLOAT') and (type(item) not in (int,float) or not math.isfinite(item) or item<prop.hard_min or item>prop.hard_max):raise ValueError('作者数值超出范围：'+key)

def _validate_restoring_properties(settings,document):
    """Read RNA types without materializing unset author PropertyGroups."""
    properties=settings.bl_rna.properties
    event_type=properties['root_control_events'].fixed_type
    for event in document['settings']['events']:
        for key,value in event.items():_validate_property(event_type.properties.get(key),key,value)
    for axis,channels in document['settings']['drives'].items():
        axis_type=properties['drive_'+axis].fixed_type
        for channel,row in channels.items():
            channel_type=axis_type.properties[channel].fixed_type
            for key in lib.DRIVE_FIELDS:_validate_property(channel_type.properties.get(key),key,row[key])
            target_type=channel_type.properties['targets'].fixed_type
            for target in row['targets']:
                for key,value in target.items():_validate_property(target_type.properties.get(key),key,value)
    for axis,row in document['settings']['d6'].items():
        axis_type=properties['d6_'+axis].fixed_type
        for key,value in row.items():_validate_property(axis_type.properties.get(key),key,value)


class _SettingsView:
    def __init__(self,original,values,events,drives):self.original=original;self.values=values;self.events=events;self.drives=drives
    def __getattr__(self,key):
        if key in self.values:return self.values[key]
        if key=='root_control_events':return [SimpleNamespace(**v) for v in self.events]
        if key.startswith('drive_') and key[6:] in self.drives:
            return SimpleNamespace(**{name:SimpleNamespace(**{k:v for k,v in row.items() if k!='targets'},targets=[SimpleNamespace(**t) for t in row['targets']]) for name,row in self.drives[key[6:]].items()})
        return getattr(self.original,key)

def _curve_candidate(scene,document,bones,settings_views=None):
    from ..ops.restoring_curve_operators import read_state
    from ..models import restoring_curves as model
    state=read_state(scene);incoming=document['curves'];ids={b.ppf_bone_physics.joint_uuid for b in bones}
    if incoming['mode']=='ACTIVE' and any(incoming['channels'].values()) and state['mode']=='LEGACY' and state.get('document'):
        raise ValueError('目标含未启用的旧曲线；请先明确处理旧曲线，预设不会自动激活它们')
    if incoming['mode']=='LEGACY' and any(incoming['channels'].values()) and state['mode']!='LEGACY':
        raise ValueError('预设含未启用的旧曲线，不能隐式激活；请先明确选择旧恢复模式')
    doc=deepcopy(state.get('document')) if state.get('document') is not None else dict(version=1,tables=[],bindings=[],stitch_pairs=[])
    removed={binding['table_id'] for binding in doc['bindings'] if binding['target'].get('kind')=='JOINT' and binding['target'].get('joint_uuid') in ids}
    doc['bindings']=[binding for binding in doc['bindings'] if not (binding['target'].get('kind')=='JOINT' and binding['target'].get('joint_uuid') in ids)]
    used={binding['table_id'] for binding in doc['bindings']};doc['tables']=[table for table in doc['tables'] if table['table_id'] not in removed or table['table_id'] in used]
    if any(incoming['channels'].values()):
        for bone in bones:
            settings=(settings_views or {}).get(bone.name,bone.ppf_bone_physics)
            channels=[(key,value) for key,value in incoming['channels'].items() if value is not None]
            parent_settings=(settings_views or {}).get(bone.parent.name,bone.parent.ppf_bone_physics) if bone.parent else None
            if channels and not (parent_settings and (parent_settings.enabled or settings.auto_follow_parent and settings.root_mode=='FREE')):
                raise ValueError(bone.name+'：恢复曲线需要真实父关节，预设不会制造父链接')
            for channel,record in channels:
                table_id='preset_'+uuid.uuid4().hex
                doc['tables'].append(dict(table_id=table_id,**deepcopy(record['table'])))
                doc['bindings'].append(dict(target=dict(kind='JOINT',joint_uuid=settings.joint_uuid,channel=channel),table_id=table_id,**deepcopy(record['binding'])))
    if doc['bindings']:
        state['document']=model.validate_author_map(doc)
        if incoming['mode']=='ACTIVE' and any(incoming['channels'].values()):state['mode']='ADVANCED' # internal author contract, never a visible tier
    elif state.get('document') is not None:state['document']=None
    return state

def _soft_material_plan(scene,settings,group,material,density):
    """Validate existing and not-yet-created groups without mutating scene RNA."""
    from ..ops import bone_proxy as ops
    if group is None:
        group_type=ops.get_addon_data(scene).bl_rna.properties['bone_material_groups'].fixed_type
        target=SimpleNamespace(bl_rna=group_type)
    else:target=group
    return ops._plan_copied_soft_material(settings,target,material,density,
        bool(group and group.object_type=='SOLID' and group.young_mod_density_normalized))


def _unchanged_soft_material(settings,group):
    """Mass-only edits preserve current material; they never repair prior drift."""
    from ..models.bone_bindings import effective_solid_material
    material=effective_solid_material(settings,group)
    for key,value in material.items():
        actual=getattr(settings,key)
        equal=math.isclose(actual,value,rel_tol=2e-6,abs_tol=1e-9) if type(value) in (int,float) else actual==value
        if not equal:raise ValueError('作者材质与当前真实材质不一致；请先明确处理后再仅应用质量')
    return material


def _validate_shape_pin_targets(scene,bones):
    """Do not replace geometry whose Pin vertex mapping cannot be preserved."""
    from ..models.groups import get_addon_data,N_MAX_GROUPS
    owners={b.ppf_bone_physics.body_uuid:b.name for b in bones
            if b.ppf_bone_physics.proxy is not None and b.ppf_bone_physics.body_uuid}
    if not owners:return
    root=get_addon_data(scene)
    groups=list(root.bone_material_groups)
    # Unused fixed slots have no stored Pins; do not materialize their RNA.
    groups.extend(getattr(root,'object_group_'+str(i)) for i in range(N_MAX_GROUPS)
                  if root.is_property_set('object_group_'+str(i)))
    for group in groups:
        for pin in group.pin_vertex_groups:
            if pin.object_uuid in owners:
                raise ValueError(owners[pin.object_uuid]+'：代理被Pin引用，预设不能自动重映射顶点；请取消形状与尺寸，或先明确处理Pin')


def _restore_preset_pin_group_flags(group, saved):
    """Restore only Pin collection flags after preset-specific row recreation."""
    entry = saved.get('pin_vertex_groups')
    if entry is None:
        return
    was_set, (_kind, values) = entry
    if not was_set:
        group.property_unset('pin_vertex_groups')
        return
    if not values and not group.is_property_set('pin_vertex_groups'):
        # RNA has no setter for an explicit empty collection.
        group.pin_vertex_groups.add()
        group.pin_vertex_groups.clear()
    for pin, pin_saved in zip(group.pin_vertex_groups, values):
        operations = pin_saved.get('operations')
        if operations is None:
            continue
        op_was_set, (_op_kind, op_values) = operations
        if not op_was_set:
            pin.property_unset('operations')
        elif not op_values and not pin.is_property_set('operations'):
            pin.operations.add()
            pin.operations.clear()


def _restore_preset_pin_flags(root, snapshot):
    """Preserve top-level and nested Pin flags after full-batch recreation."""
    for key, (_was_set, (kind, value)) in snapshot.items():
        if key == 'bone_material_groups' and kind == 'COLLECTION':
            for group, saved in zip(root.bone_material_groups, value):
                _restore_preset_pin_group_flags(group, saved)
        elif key.startswith('object_group_') and kind == 'GROUP':
            _restore_preset_pin_group_flags(getattr(root, key), value)


def _snapshot_bone_settings(owner):
    """Include mutable contents behind Blender's read-only group pointers."""
    import bpy
    from ..ops.restoring_curve_operators import snapshot_rna
    # Reading an unset Object pointer can materialize an explicit None.
    pointer_flags={prop.identifier:owner.is_property_set(prop.identifier)
                   for prop in owner.bl_rna.properties if prop.type=='POINTER' and not prop.is_readonly}
    result=snapshot_rna(owner)
    for key,was_set in pointer_flags.items():
        if key in result:result[key]=(was_set,result[key][1])
    for prop in owner.bl_rna.properties:
        if prop.identifier=='rna_type' or prop.type!='POINTER':continue
        group_type=bpy.types.PropertyGroup.bl_rna_get_subclass_py(prop.fixed_type.identifier)
        if group_type is None or not issubclass(group_type,bpy.types.PropertyGroup):continue
        was_set=owner.is_property_set(prop.identifier)
        saved=_snapshot_bone_settings(getattr(owner,prop.identifier)) if was_set else {}
        result[prop.identifier]=(was_set,('GROUP',saved))
    return result


def _restore_bone_settings(owner,snapshot):
    from ..ops.restoring_curve_operators import restore_rna
    restore_rna(owner,snapshot)
    def restore_collection_flags(current,saved):
        for key,(was_set,(kind,value)) in saved.items():
            if kind=='GROUP':
                if was_set:restore_collection_flags(getattr(current,key),value)
                else:current.property_unset(key)
            elif kind=='COLLECTION':
                collection=getattr(current,key)
                for item,item_snapshot in zip(collection,value):restore_collection_flags(item,item_snapshot)
                if not was_set and not collection:current.property_unset(key)
    restore_collection_flags(owner,snapshot)


def _commit_target(context,arm,bone,plan):
    """One target commit; split for a controlled later-target rollback test."""
    import bpy
    from ..ops import bone_proxy as ops
    from ..models.bone_bindings import ensure_binding_ids,sync_owned_material
    settings=ensure_binding_ids(arm,bone)
    if not settings.body_uuid:settings.body_uuid=str(uuid.uuid4())
    for key,value in plan['values'].items():
        if key!='enabled':setattr(settings,key,value)
    for key,value in plan['display'].items():setattr(settings,key,value)
    settings.custom_object=None # portable actual mesh, never a copied external pointer
    settings.root_control_events.clear()
    for value in plan['events']:
        owner=settings.root_control_events.add()
        for key,item in value.items():setattr(owner,key,item)
    for axis,channels in plan['drives'].items():
        for channel,value in channels.items():
            owner=getattr(getattr(settings,'drive_'+axis),channel)
            for key in lib.DRIVE_FIELDS:setattr(owner,key,value[key])
            owner.targets.clear()
            for target in value['targets']:
                key=owner.targets.add();key.frame=target['frame'];key.value=target['value']
    for key,value in plan['d6'].items():
        for field,item in value.items():setattr(getattr(settings,'d6_'+key),field,item)
    proxy=settings.proxy
    if proxy is None:
        proxy=bpy.data.objects.new('PPF_Bone_'+bone.name,plan['mesh']);context.scene.collection.objects.link(proxy);settings.proxy=proxy
        proxy['_solver_uuid']=settings.body_uuid;proxy['ppf_bone_proxy_owned']=True;proxy['ppf_bone_uuid']=settings.bone_uuid;proxy['ppf_armature_uuid']=arm.ppf_armature_uuid
        proxy.parent=arm;proxy.display_type='WIRE';proxy.show_in_front=True;proxy.hide_render=True
    else:proxy.data=plan['mesh']
    proxy.matrix_world=plan['world']
    from .bone_proxy_visibility import manage_proxy
    manage_proxy(context.scene,arm,bone,proxy)
    group=ops.get_group_by_uuid(context.scene,settings.group_uuid) if settings.group_uuid else None
    if group is not None:group.object_type=ops.object_type_for(settings)
    group=ops._group(context.scene,bone,{},initialize_material=False)
    if not group.assigned_objects:
        assigned=group.assigned_objects.add();assigned.uuid=settings.body_uuid;assigned.name=proxy.name
    for key,value in plan['material'].items():setattr(group,key,value)
    from .bone_mass import _apply_density,_diagnostics
    if plan['soft_material'] is not None:
        # The explicit material transaction replaces E and density together.
        # MASS derives density from the target mesh, not the source preset.
        settings.density=plan['density']
        ops._apply_copied_soft_material(settings,group,plan['soft_material'])
    else:
        _apply_density(settings,group,plan['density'])
        sync_owned_material(settings)
    settings.enabled=plan['values']['enabled'];group.assigned_objects[0].included=settings.enabled
    _diagnostics(settings,dict(plan['mass_measure'],mass_kg=plan['mass']))


def apply(context,document,scope,size_mode,mask=None):
    if mask is not None:
        if not any(mask.values()): return dict(applied=0, skipped=[])
        if not all(mask.get(k,False) for k in ('restoring','material','mass','shape','collision','display')):
            return _apply_selected(context,document,scope,size_mode,mask)
    import bpy
    from mathutils import Matrix,Vector
    from ..ops import bone_proxy as ops
    from ..models.bone_bindings import validate_final_binding_ids,validate_proxy_owner,check_soft_supported,has_physical_joint_parent
    from ..models.groups import get_addon_data,assign_display_indices,invalidate_overlays
    from ..ops.restoring_curve_operators import snapshot_rna,restore_rna,write_state
    from .bone_mass import material_write_guard,_measure,mark_settings
    from .bone_mass_math import resolve_mass
    from ..models.bone_collision import collision_edit_batch
    if size_mode not in ('FIT_BONE','KEEP_SIZE'):raise ValueError('未知尺寸模式')
    arm,bones=ops.selected_bones(context,scope=scope)
    if not bones:raise ValueError('请选择目标骨骼')
    com=ops._facade_communicator();com.assert_bone_body_switch_idle()
    if ops.stale_simulation_output(context):raise ValueError('请先Clear已有模拟帧，再应用完整预设')
    validate_final_binding_ids(context.scene,arm,bones)
    _validate_shape_pin_targets(context.scene,bones)
    before_objects={obj.as_pointer() for obj in bpy.data.objects};before_meshes={mesh.as_pointer() for mesh in bpy.data.meshes}
    before_collections={c.as_pointer() for c in bpy.data.collections}
    plans={};created_meshes=[]
    try:
        for bone in bones:
            validate_proxy_owner(arm,bone);settings=bone.ppf_bone_physics
            group=ops._owned_bone_material(context.scene,arm,bone) if settings.proxy else None
            if settings.proxy and (group is None or group.object_type!=ops.OBJECT_TYPES[settings.physical_model]):raise ValueError(bone.name+'：目标代理/独占材质失配')
            if settings.proxy and (settings.proxy.mode!='OBJECT' or any(m.type=='MESH_CACHE' and m.show_viewport for m in settings.proxy.modifiers)):raise ValueError('目标处于编辑或PC2播放，先结束后再应用')
            values=deepcopy(document['settings']['bone'])
            if values['joint_mode'] != 'SOFT':raise ValueError('HARD已封存；完整预设不能启用硬关节')
            factor=float(bone.length)/document['source_bone_length'] if size_mode=='FIT_BONE' else 1.
            if not math.isfinite(factor) or factor<=0:raise ValueError('目标骨长无效')
            for key in ('radius','box_width','box_depth'):values[key]*=factor
            for key,value in values.items():_validate_rna(settings,key,value)
            if not values['use_auto_frames'] and bone.parent is None:raise ValueError(bone.name+'：手动关节帧需要目标父骨骼')
            for key,value in document['display'].items():_validate_rna(settings,key,value)
            root=ops.get_addon_data(context.scene)
            group_type=root.bl_rna.properties['bone_material_groups'].fixed_type
            for key,value in document['material'].items():_validate_property(group_type.properties.get(key),key,value)
            _validate_restoring_properties(settings,document)
            # Reject animated author channels through the same owned-material gate.
            if settings.proxy:
                from .bone_mass import _owned
                _owned(context.scene,arm,bone)
            B=_author_frame(arm,bone);local=Matrix(document['geometry']['bone_local_from_mesh']);local.translation*=factor
            world=B@local;ops.rigid_matrix(world,bone.name+'预设代理')
            mesh=bpy.data.meshes.new('PPF_Preset_'+bone.name);created_meshes.append(mesh)
            mesh.from_pydata([[v*factor for v in vertex] for vertex in document['geometry']['vertices']],[],document['geometry']['faces']);mesh.update()
            checked=mesh.copy()
            try:ops.validate_closed_mesh(checked)
            finally:bpy.data.meshes.remove(checked)
            measured=_measure(context.scene,settings,mesh,world);density,mass=resolve_mass(values['mass_mode'],values['density'],values['target_mass_kg'],measured['volume_m3'])
            soft_material=_soft_material_plan(context.scene,settings,group,values,density) if values['physical_model']=='SOFT' else None
            plans[bone.name]=dict(values=values,display=document['display'],events=document['settings']['events'],drives=document['settings']['drives'],d6=document['settings']['d6'],material=document['material'],mesh=mesh,world=world,B=B,factor=factor,density=density,mass=mass,mass_measure=measured,soft_material=soft_material)
        enabled_names={b.name for b in arm.data.bones if (plans[b.name]['values']['enabled'] if b.name in plans else b.ppf_bone_physics.enabled)}
        views={b.name:_SettingsView(b.ppf_bone_physics,plans[b.name]['values'],plans[b.name]['events'],plans[b.name]['drives']) if b.name in plans else b.ppf_bone_physics for b in arm.data.bones}
        pseudo={b.name:SimpleNamespace(name=b.name,ppf_bone_physics=views[b.name]) for b in arm.data.bones}
        for bone in arm.data.bones:pseudo[bone.name].parent=pseudo.get(bone.parent.name) if bone.parent else None;pseudo[bone.name].children=[pseudo[c.name] for c in bone.children]
        models={name:view.physical_model for name,view in views.items()}
        if document['curves']['mode']=='ACTIVE' and any(document['curves']['channels'].values()):
            from ..ops.restoring_curve_operators import read_state
            state=read_state(context.scene)
            if state['mode']=='LEGACY' and state.get('document'):raise ValueError('目标含未启用的旧曲线；请先明确处理旧曲线')
            for bone in bones:
                if not has_physical_joint_parent(pseudo[bone.name],enabled_names):raise ValueError(bone.name+'：恢复曲线需要真实父关节')
        for bone in arm.data.bones:
            if bone.name not in enabled_names:continue
            current=pseudo[bone.name]
            check_soft_supported(views[bone.name],bone.name,has_physical_joint_parent(current,enabled_names),assumed_model=models[bone.name])
            for child,kind,mode in ops.incident_edge_refusals(current,models,enabled_names):
                if bone.name in plans or child in plans:raise ValueError(bone.name+'：预设与目标父子物理能力不兼容：'+kind+' '+mode)
        # Capture all affected author data, retaining every old mesh until success.
        root=get_addon_data(context.scene);root_snapshot=snapshot_rna(root)
        bone_snapshots=[(b,_snapshot_bone_settings(b.ppf_bone_physics)) for b in arm.data.bones]
        curve_snapshot=snapshot_rna(context.scene.ppf_restoring_curves)
        old_arm_id=arm.ppf_armature_uuid
        old_arm_id_was_set=arm.is_property_set('ppf_armature_uuid')
        objects=[(b.ppf_bone_physics.proxy,b.ppf_bone_physics.proxy.data,b.ppf_bone_physics.proxy.matrix_world.copy()) for b in bones if b.ppf_bone_physics.proxy]
        from .bone_proxy_visibility import snapshot_proxy,restore_proxy,cleanup_new_collections
        visibility=[(obj,snapshot_proxy(obj,context.scene)) for obj,_mesh,_matrix in objects]
        from .bone_author_follow import snapshot_preset_baselines,restore_preset_baselines
        follow_snapshot=snapshot_preset_baselines(context.scene,arm,bones)
        with collision_edit_batch():
            try:
                with material_write_guard():
                    for bone in bones:_commit_target(context,arm,bone,plans[bone.name])
                    # Rebind after every target mesh/frame is final, including parents.
                    if not document['settings']['bone']['use_auto_frames']:
                        for bone in bones:
                            settings=bone.ppf_bone_physics;plan=plans[bone.name]
                            for side,parent in (('parent',True),('child',False)):
                                joint=Matrix(document['frames'][side+'_joint_in_bone']);joint.translation*=plan['factor']
                                endpoint=_parent_frame(arm,bone) if parent else settings.proxy.matrix_world
                                frame=endpoint.inverted()@plan['B']@joint
                                frame=ops.rigid_matrix(frame,side+'目标关节帧')
                                setattr(settings,side+'_frame_translation',frame.translation);setattr(settings,side+'_frame_quaternion',frame.to_quaternion())
                    candidate=_curve_candidate(context.scene,document,bones);write_state(context.scene,candidate)
                    assign_display_indices(context.scene)
                for bone in bones:mark_settings(bone.ppf_bone_physics,context)
                from .bone_author_follow import baseline_proxy
                for bone in bones:baseline_proxy(context.scene,arm,bone)
                context.view_layer.update()
                com.bone_physical_body_changed()
            except BaseException:
                with material_write_guard():
                    for obj,mesh,matrix in objects:obj.data=mesh;obj.matrix_world=matrix
                    restore_rna(root,root_snapshot)
                    _restore_preset_pin_flags(root,root_snapshot)
                    if not root_snapshot['bone_material_groups'][0] and not root.bone_material_groups:
                        root.property_unset('bone_material_groups')
                    for bone,snapshot in bone_snapshots:_restore_bone_settings(bone.ppf_bone_physics,snapshot)
                    restore_rna(context.scene.ppf_restoring_curves,curve_snapshot);arm.ppf_armature_uuid=old_arm_id
                    if not old_arm_id_was_set:arm.property_unset('ppf_armature_uuid')
                for obj,snapshot in reversed(visibility):restore_proxy(obj,snapshot)
                for obj in list(bpy.data.objects):
                    if obj.as_pointer() not in before_objects:bpy.data.objects.remove(obj,do_unlink=True)
                for collection in list(bpy.data.collections):
                    if collection.as_pointer() not in before_collections and not collection.objects and not collection.children:bpy.data.collections.remove(collection)
                cleanup_new_collections(before_collections)
                restore_preset_baselines(follow_snapshot)
                raise
        # Release only meshes replaced by this successful commit, never user data.
        # Collect unique live IDs before any release invalidates shared RNA.
        replaced_meshes={mesh.as_pointer():mesh for _obj,mesh,_matrix in objects}
        for pointer,mesh in replaced_meshes.items():
            if mesh.users==0 and pointer in before_meshes:bpy.data.meshes.remove(mesh)
        invalidate_overlays()
        return dict(applied=len(bones),skipped=[])
    finally:
        for mesh in created_meshes:
            if mesh.name in bpy.data.meshes and mesh.users==0:bpy.data.meshes.remove(mesh)


# Selective application is planned against current author data. It never creates
# an unselected mesh, curve or material; full application retains its old path.
def _apply_selected(context, document, scope, size_mode, mask):
    import bpy
    from mathutils import Matrix
    from ..ops import bone_proxy as ops
    from ..models.bone_bindings import validate_final_binding_ids
    from ..models.groups import invalidate_overlays
    from ..ops.restoring_curve_operators import snapshot_rna, restore_rna, write_state
    from .bone_mass import material_write_guard, mark_settings, _measure, _apply_density, _diagnostics
    from .bone_mass_math import resolve_mass
    from ..models.bone_collision import collision_edit_batch
    if size_mode not in ('FIT_BONE','KEEP_SIZE'): raise ValueError('未知尺寸模式')
    arm,bones=ops.selected_bones(context,scope=scope)
    if not bones: raise ValueError('请选择目标骨骼')
    com=ops._facade_communicator(); com.assert_bone_body_switch_idle()
    if ops.stale_simulation_output(context): raise ValueError('请先Clear已有模拟帧，再应用预设')
    validate_final_binding_ids(context.scene,arm,bones)
    if mask.get('shape'):_validate_shape_pin_targets(context.scene,bones)
    groups=[]; plans=[]; meshes=[]
    material_fields={'enabled','physical_model','material_preset','solid_model','solid_young_modulus','solid_poisson_ratio','solid_deformation_damping','fem_scale_enabled'}
    mass_fields={'density','mass_mode','target_mass_kg'}
    shape_fields={'shape','proxy_quality','proxy_face_mode','radius','length_ratio','box_width','box_depth'}
    collision_fields={'collide_connected','collision_filter_enabled','collision_group_layers','collision_mask_layers'}
    restoring_fields=set(lib.BONE_FIELDS)-material_fields-mass_fields-shape_fields-collision_fields
    selected=set()
    for k,fields in (('material',material_fields),('mass',mass_fields),('shape',shape_fields),('collision',collision_fields),('restoring',restoring_fields)):
        if mask.get(k): selected.update(fields)
    if 'joint_mode' in selected and document['settings']['bone']['joint_mode'] != 'SOFT':
        raise ValueError('硬关节不再支持；预设不能启用硬关节')
    try:
        for bone in bones:
            from ..models.bone_bindings import _owned_group, validate_proxy_owner
            from ..models.groups import iterate_active_object_groups
            settings=bone.ppf_bone_physics; validate_proxy_owner(arm,bone)
            if settings.proxy is None or context.scene.objects.get(settings.proxy.name)!=settings.proxy: raise ValueError(bone.name+'：请先创建物理代理')
            group=_owned_group(settings,'SOLID' if settings.physical_model=='SOFT' else 'PDRD')
            if group is None or sum(g.uuid==settings.group_uuid for g in iterate_active_object_groups(context.scene))!=1: raise ValueError(bone.name+'：代理或材质被共享、失配')
            groups.append(group)
            paths=[path for block,label in ((context.scene,'scene'),(arm.data,'armature data'),(arm,'armature object')) for path in ops._owner_animation_paths(block,label)]
            prefix=bone.path_from_id()+'.ppf_bone_physics.'
            written=set(selected)
            if mask.get('display'): written.update(lib.DISPLAY_FIELDS)
            if mask.get('shape'): written.add('custom_object')
            if mask.get('restoring'): written.update(('root_control_events','drive_twist_x','drive_swing_y','drive_swing_z','parent_frame_translation','parent_frame_quaternion','child_frame_translation','child_frame_quaternion')); written.update('d6_'+kind+'_'+axis for kind in ('translation','rotation') for axis in 'xyz')
            if any(any(path==prefix+k or path.startswith(prefix+k+'.') or path.startswith(prefix+k+'[') for k in written) for path in paths): raise ValueError(bone.name+'：选中预设字段带动画或驱动')
            group_written=set()
            if mask.get('material'): group_written.update(('object_type','friction','shrink','enable_plasticity','plasticity','plasticity_threshold','solid_model','solid_young_modulus','solid_poisson_ratio','deformation_damping','young_mod_density_normalized','solid_density','pdrd_density','assigned_objects'))
            if mask.get('mass'):
                group_written.update(('solid_density','pdrd_density'))
                if group.object_type=='SOLID':group_written.update(('solid_young_modulus','young_mod_density_normalized'))
            if mask.get('collision'): group_written.update(('use_group_bounding_box_diagonal','contact_gap_rat','contact_offset_rat','contact_gap','contact_offset'))
            gp=group.path_from_id()+'.'
            if any(any(path==gp+k or path.startswith(gp+k+'.') or path.startswith(gp+k+'[') for k in group_written) for path in paths): raise ValueError(bone.name+'：选中材质字段带动画或驱动')
            if settings.proxy.mode!='OBJECT' or any(m.type=='MESH_CACHE' and m.show_viewport for m in settings.proxy.modifiers): raise ValueError('目标处于编辑或PC2播放')
            values={k:deepcopy(document['settings']['bone'][k]) for k in selected if k in document['settings']['bone']}
            if mask.get('material') and not mask.get('mass'):
                actual=group.solid_density if group.object_type=='SOLID' else group.pdrd_density
                if not math.isclose(float(settings.density),float(actual),rel_tol=1e-6): raise ValueError(bone.name+'：作者密度与当前材质密度不一致；请先明确处理后再仅应用材质')
            factor=float(bone.length)/document['source_bone_length'] if size_mode=='FIT_BONE' else 1.
            if not math.isfinite(factor) or factor<=0: raise ValueError('目标骨长无效')
            for k in ('radius','box_width','box_depth'):
                if k in values: values[k]*=factor
            for k,v in values.items(): _validate_rna(settings,k,v)
            if mask.get('restoring') and not document['settings']['bone']['use_auto_frames'] and bone.parent is None:raise ValueError(bone.name+'：手动关节帧需要目标父骨骼')
            for k,v in (document['display'].items() if mask.get('display') else ()): _validate_rna(settings,k,v)
            # Selective commits must use the same RNA bounds as full Apply.
            # Validate only categories that will be written, before allocation.
            for key,value in document['material'].items():
                if key in group_written:_validate_rna(group,key,value)
            if mask.get('restoring'):_validate_restoring_properties(settings,document)
            mesh=None; world=None; B=None
            if mask.get('shape'):
                B=_author_frame(arm,bone); local=Matrix(document['geometry']['bone_local_from_mesh']); local.translation*=factor; world=B@local; ops.rigid_matrix(world,bone.name+'预设代理')
                mesh=bpy.data.meshes.new('PPF_Preset_'+bone.name); meshes.append(mesh)
                mesh.from_pydata([[v*factor for v in vertex] for vertex in document['geometry']['vertices']],[],document['geometry']['faces']); mesh.update()
                checked=mesh.copy()
                try: ops.validate_closed_mesh(checked)
                finally: bpy.data.meshes.remove(checked)
            mass=None
            if mask.get('mass'):
                measured=_measure(context.scene,settings,mesh,world) if mesh else _measure(context.scene,settings)
                density,weight=resolve_mass(values['mass_mode'],values['density'],values['target_mass_kg'],measured['volume_m3']); mass=(density,weight,measured)
            if mask.get('material') and not mask.get('mass') and settings.mass_mode=='MASS':
                measured=_measure(context.scene,settings,mesh,world) if mesh else _measure(context.scene,settings)
                final_density,_weight=resolve_mass('MASS',settings.density,settings.target_mass_kg,measured['volume_m3'])
                if not math.isclose(final_density,float(settings.density),rel_tol=2e-6):
                    raise ValueError(bone.name+'：固定质量的密度尚待刷新；请先刷新质量后再仅应用材质')
            soft_material=None
            if values.get('physical_model',settings.physical_model)=='SOFT' and (mask.get('material') or mask.get('mass')):
                density=mass[0] if mass else float(group.solid_density if group.object_type=='SOLID' else group.pdrd_density)
                if mask.get('material'):
                    soft_material=_soft_material_plan(context.scene,settings,group,values,density)
                else:
                    # A mass-only edit preserves true E and the legacy storage
                    # convention. Copy/material fallback must not migrate it.
                    from .bone_mass import _density_plan
                    _density_plan(settings,group,density)
            plans.append(dict(values=values,mesh=mesh,world=world,B=B,factor=factor,mass=mass,soft_material=soft_material))
        from ..models.bone_bindings import check_soft_supported, has_physical_joint_parent
        by_name={b.name:p for b,p in zip(bones,plans)}
        views={b.name: _SettingsView(b.ppf_bone_physics,by_name[b.name]['values'],document['settings']['events'] if mask.get('restoring') else [_read(e,lib.EVENT_FIELDS) for e in b.ppf_bone_physics.root_control_events],document['settings']['drives'] if mask.get('restoring') else {}) if b.name in by_name else b.ppf_bone_physics for b in arm.data.bones}
        pseudo={b.name:SimpleNamespace(name=b.name,ppf_bone_physics=views[b.name]) for b in arm.data.bones}
        for b in arm.data.bones:
            pseudo[b.name].parent=pseudo.get(b.parent.name) if b.parent else None
            pseudo[b.name].children=[pseudo[c.name] for c in b.children]
        enabled_names={b.name for b in arm.data.bones if views[b.name].enabled}
        models={name:view.physical_model for name,view in views.items()}
        for b in arm.data.bones:
            if b.name not in enabled_names: continue
            current=pseudo[b.name]
            check_soft_supported(views[b.name],b.name,has_physical_joint_parent(current,enabled_names),assumed_model=models[b.name])
            for child,kind,mode in ops.incident_edge_refusals(current,models,enabled_names):
                if b.name in by_name or child in by_name: raise ValueError(b.name+'：预设与目标父子物理能力不兼容：'+kind+' '+mode)
        candidate=_curve_candidate(context.scene,document,bones,views) if mask.get('restoring') else None
        snapshots=[((_snapshot_bone_settings(b.ppf_bone_physics) if mask.get('restoring') else snapshot_rna(b.ppf_bone_physics)),snapshot_rna(g),b.ppf_bone_physics.proxy.data,b.ppf_bone_physics.proxy.matrix_world.copy()) for b,g in zip(bones,groups)]
        curve_snapshot=snapshot_rna(context.scene.ppf_restoring_curves)
        follow_snapshot=None
        if mask.get('shape'):
            from .bone_author_follow import snapshot_preset_baselines,restore_preset_baselines,baseline_proxy
            follow_snapshot=snapshot_preset_baselines(context.scene,arm,bones)
        try:
            with collision_edit_batch(), material_write_guard():
                for bone,group,plan in zip(bones,groups,plans):
                    settings=bone.ppf_bone_physics
                    old_density = group.solid_density if group.object_type=='SOLID' else group.pdrd_density
                    for key,value in plan['values'].items(): setattr(settings,key,value)
                    if mask.get('material'):
                        group.object_type=ops.OBJECT_TYPES[settings.physical_model]
                        if not mask.get('mass'):
                            if group.object_type=='SOLID': group.solid_density=old_density
                            else: group.pdrd_density=old_density
                        if group.assigned_objects: group.assigned_objects[0].included=settings.enabled
                    if mask.get('display'):
                        for key,value in document['display'].items(): setattr(settings,key,value)
                    if mask.get('restoring'):
                        settings.root_control_events.clear()
                        for value in document['settings']['events']:
                            owner=settings.root_control_events.add()
                            for k,v in value.items(): setattr(owner,k,v)
                        for axis,channels in document['settings']['drives'].items():
                            for channel,value in channels.items():
                                owner=getattr(getattr(settings,'drive_'+axis),channel)
                                for key in lib.DRIVE_FIELDS: setattr(owner,key,value[key])
                                owner.targets.clear()
                                for target in value['targets']:
                                    t=owner.targets.add(); t.frame=target['frame']; t.value=target['value']
                        for axis,row in document['settings']['d6'].items():
                            for k,v in row.items(): setattr(getattr(settings,'d6_'+axis),k,v)
                    if plan['mesh']:
                        settings.custom_object=None; settings.proxy.data=plan['mesh']; settings.proxy.matrix_world=plan['world']
                    if mask.get('material'):
                        for k,v in document['material'].items():
                            if k in ('friction','shrink','enable_plasticity','plasticity','plasticity_threshold'): setattr(group,k,v)
                    if mask.get('collision'):
                        for k,v in document['material'].items():
                            if k in ('use_group_bounding_box_diagonal','contact_gap_rat','contact_offset_rat','contact_gap','contact_offset'): setattr(group,k,v)
                    if plan['mass']:
                        density,weight,measured=plan['mass']
                        if plan['soft_material'] is None: _apply_density(settings,group,density)
                        else: settings.density=density
                        _diagnostics(settings,dict(measured,mass_kg=weight))
                    if plan['soft_material'] is not None:ops._apply_copied_soft_material(settings,group,plan['soft_material'])
                if mask.get('restoring') and not document['settings']['bone']['use_auto_frames']:
                    for bone,plan in zip(bones,plans):
                        settings=bone.ppf_bone_physics
                        B=_author_frame(arm,bone)
                        for side,parent in (('parent',True),('child',False)):
                            joint=Matrix(document['frames'][side+'_joint_in_bone']); joint.translation*=plan['factor']
                            endpoint=_parent_frame(arm,bone) if parent else settings.proxy.matrix_world
                            frame=ops.rigid_matrix(endpoint.inverted()@B@joint,side+'目标关节帧')
                            setattr(settings,side+'_frame_translation',frame.translation); setattr(settings,side+'_frame_quaternion',frame.to_quaternion())
                if candidate is not None: write_state(context.scene,candidate)
            for bone in bones: mark_settings(bone.ppf_bone_physics,context)
            context.view_layer.update()
            if follow_snapshot is not None:
                for bone in bones:baseline_proxy(context.scene,arm,bone)
            com.bone_physical_body_changed(); invalidate_overlays()
        except BaseException:
            try:
                with material_write_guard():
                    for bone,group,(ss,gs,mesh,matrix) in zip(bones,groups,snapshots):
                        bone.ppf_bone_physics.proxy.data=mesh; bone.ppf_bone_physics.proxy.matrix_world=matrix
                        _restore_bone_settings(bone.ppf_bone_physics,ss); restore_rna(group,gs)
                        _restore_preset_pin_group_flags(group,gs)
                    restore_rna(context.scene.ppf_restoring_curves,curve_snapshot)
            finally:
                if follow_snapshot is not None:restore_preset_baselines(follow_snapshot)
            raise
        if mask.get('shape'):
            # Two distinct proxies may have shared the same replaced Mesh.
            replaced_meshes={mesh.as_pointer():mesh for _ss,_gs,mesh,_matrix in snapshots}
            for mesh in replaced_meshes.values():
                if mesh.users==0: bpy.data.meshes.remove(mesh)
        return dict(applied=len(bones),skipped=[])
    finally:
        for mesh in meshes:
            if mesh.name in bpy.data.meshes and mesh.users==0: bpy.data.meshes.remove(mesh)
