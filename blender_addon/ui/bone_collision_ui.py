"""Simple collision grouping edits the existing 16-bit author fields only."""
from .. import i18n as _ppf_i18n
import bpy
from ..i18n.props import BoolProperty,CollectionProperty,EnumProperty,PointerProperty,StringProperty

_enum_history=[]

class PPF_PG_CollisionLabel(bpy.types.PropertyGroup):
    label:StringProperty(default='',options=set())

def slot_items(self,context):
    labels=self.labels
    items=[(str(i),f'{labels[i].label if i<len(labels) and labels[i].label else "组"+str(i+1).zfill(2)}（{i+1:02d}）','') for i in range(16)]
    # Blender retains references to dynamic enum strings. Keep returned item
    # strings alive until this module's unregister rather than dangling them.
    if not _enum_history or _enum_history[-1]!=items:_enum_history.append(items)
    return _enum_history[-1]

def _slot_changed(self,context):
    index=int(self.slot)
    self.label=self.labels[index].label if index<len(self.labels) else ''

class PPF_PG_CollisionSimple(bpy.types.PropertyGroup):
    labels:CollectionProperty(type=PPF_PG_CollisionLabel)
    slot:EnumProperty(name='Collision Slot',items=slot_items,update=_slot_changed,options=set())
    label:StringProperty(name='Collision Slot Name',default='',options=set())
    enabled:BoolProperty(name='Enable Collision Group Filter',default=True,options=set())
    within:BoolProperty(name='Within Collision Group',default=True,options=set())
    between:BoolProperty(name='Between Collision Groups',default=True,options=set())
    scope:EnumProperty(name='Apply Scope',items=[('SELECTED','Selected',''),('CHAIN','Active Chain',''),('ALL','All','')],default='SELECTED',options=set())

def simple_bits(slot,within,between):
    if type(slot) is not int or not 0<=slot<16:raise ValueError('Invalid collision slot')
    group=1<<slot
    return group,(group if within else 0)|((65535^group) if between else 0)

def simple_summary(settings):
    if not settings.collision_filter_enabled:return '过滤关闭：允许所有网格碰撞组'
    groups=sum((1<<i) for i,b in enumerate(settings.collision_group_layers) if b)
    mask=sum((1<<i) for i,b in enumerate(settings.collision_mask_layers) if b)
    if groups==0 or groups&(groups-1):return '高级组合：多层所属 / 自定义允许层'
    slot=groups.bit_length()-1
    for within in (False,True):
        for between in (False,True):
            if simple_bits(slot,within,between)==(groups,mask):
                return f'当前组{slot+1:02d}：组内{"允许" if within else "禁止"}，组间{"允许" if between else "禁止"}'
    return '高级组合：多层所属 / 自定义允许层'

def targets(context,scope=None):
    from ..models.bone_bindings import active_armature,validate_proxy_owner
    from ..ops.bone_proxy import selected_bones
    from ..models.groups import iterate_active_object_groups
    if active_armature(context):
        arm,bones=selected_bones(context,scope=context.scene.ppf_collision_simple.scope if scope is None else scope)
        for bone in bones:
            validate_proxy_owner(arm,bone)
            uid=bone.ppf_bone_physics.body_uuid
            if uid and sum(obj.get('_solver_uuid')==uid for obj in context.scene.objects)>1:
                raise ValueError(f'{bone.name}: duplicate body identity')
        return [(bone.name,bone.ppf_bone_physics) for bone in bones]
    mesh_objects=[obj for obj in context.selected_objects if obj.type=='MESH']
    if not mesh_objects:raise ValueError('Select physical bones or mesh objects')
    active_uuids={a.uuid for group in iterate_active_object_groups(context.scene) for a in group.assigned_objects if a.included}
    result=[]
    for obj in mesh_objects:
        if obj.get('ppf_bone_proxy_owned'):raise ValueError('Return to the bound bone before applying collision settings')
        if not obj.get('_solver_uuid') or obj.get('_solver_uuid') not in active_uuids:
            raise ValueError(f'{obj.name}: add the mesh to an active PPF group first')
        if sum(other.get('_solver_uuid')==obj.get('_solver_uuid') for other in context.scene.objects)>1:
            raise ValueError(f'{obj.name}: duplicate object identity')
        result.append((obj.name,obj.ppf_collision_filter))
    return result

_COLLISION_FIELDS=('collision_filter_enabled','collision_group_layers','collision_mask_layers')

def _preflight_editable(name,settings,owner):
    """Reject linked or RNA-read-only collision settings before a batch writes."""
    datablocks=[owner]
    # Bone settings are stored on Armature data; ordinary mesh collision
    # settings are Object-owned and remain writable with linked mesh data.
    if getattr(owner,'type',None)=='ARMATURE':datablocks.append(getattr(owner,'data',None))
    for datablock in datablocks:
        if datablock is not None and getattr(datablock,'library',None) is not None and getattr(datablock,'override_library',None) is None:
            raise ValueError(f'{name}: linked collision settings are read-only')
    for field in _COLLISION_FIELDS:
        try:read_only=settings.is_property_readonly(field)
        except (AttributeError,TypeError):read_only=False
        if read_only:raise ValueError(f'{name}: {field} is read-only')

def _apply_collision_snapshot(entries,snapshot,assign=None,fields=_COLLISION_FIELDS):
    """Apply a value snapshot atomically; restore every entry after any write error."""
    if assign is None:assign=setattr
    previous=[(settings,bool(settings.collision_filter_enabled),tuple(settings.collision_group_layers),tuple(settings.collision_mask_layers)) for _name,settings,_owner in entries]
    values=dict(zip(_COLLISION_FIELDS,snapshot))
    from ..models.bone_collision import collision_edit_batch
    with collision_edit_batch():
        try:
            for _name,settings,_owner in entries:
                for field in fields:assign(settings,field,values[field])
        except Exception as primary:
            failures=[]
            for settings,enabled,membership,allowed in previous:
                for field,value in zip(_COLLISION_FIELDS,(enabled,membership,allowed)):
                    # Roll back through the native RNA assignment path, independent
                    # of the writer that just failed.
                    try:setattr(settings,field,value)
                    except Exception as rollback_error:failures.append(f'{field}: {rollback_error}')
            if failures:raise RuntimeError(f'Collision copy failed ({primary}); rollback errors: {"; ".join(failures)}') from primary
            raise

class PPF_OT_CollisionSimpleApply(bpy.types.Operator):
    bl_idname='ppf.collision_simple_apply';bl_label='Apply Collision Group';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        try:
            from ..models.bone_bindings import active_armature
            pairs=targets(context)  # Whole-batch ownership preflight.
            arm=active_armature(context)
            entries=[(name,settings,arm if arm else context.scene.objects[name]) for name,settings in pairs]
            for name,settings,owner in entries:_preflight_editable(name,settings,owner)
            draft=context.scene.ppf_collision_simple
            groups,mask=simple_bits(int(draft.slot),draft.within,draft.between)
            snapshot=(bool(draft.enabled),tuple(bool(groups&(1<<i)) for i in range(16)),tuple(bool(mask&(1<<i)) for i in range(16)))
            # The compatibility operator keeps its historical disable-only
            # behaviour: switching off filtering preserves each target's masks.
            fields=_COLLISION_FIELDS if draft.enabled else ('collision_filter_enabled',)
            _apply_collision_snapshot(entries,snapshot,fields=fields)
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
        self.report({'INFO'},f'已应用 {len(entries)} 个碰撞设置；请重新传输和构建')
        return {'FINISHED'}

class PPF_OT_CollisionSlotRename(bpy.types.Operator):
    bl_idname='ppf.collision_slot_rename';bl_label='保存碰撞组名称';bl_options={'REGISTER','UNDO'}
    bl_description='保存所选编号的碰撞组名称，不改变碰撞组归属和允许关系'
    def execute(self,context):
        owner=context.scene.ppf_collision_simple
        while len(owner.labels)<16:owner.labels.add()
        owner.labels[int(owner.slot)].label=owner.label.strip()
        return {'FINISHED'}

class PPF_OT_CollisionCopyActual(bpy.types.Operator):
    """Copy the active target's effective author fields as one atomic batch."""
    bl_idname='ppf.collision_copy_actual';bl_label='复制活动目标碰撞设置';bl_options={'REGISTER','UNDO'}
    bl_description='将活动目标的碰撞组、允许组和过滤开关复制到指定范围；可撤销'
    scope:EnumProperty(name='复制范围',items=[('LEGACY','使用原碰撞范围','兼容原面板与调用方'),('SELECTED','选中骨骼',''),('CHAIN','当前骨链',''),('ALL','整个骨架','')],default='LEGACY')
    def execute(self,context):
        from ..models.bone_bindings import active_armature,active_bone
        try:
            arm=active_armature(context)
            if arm:
                source_bone=active_bone(context)
                if source_bone is None:raise ValueError('Select an active source bone')
                source=source_bone.ppf_bone_physics
                target_pairs=targets(context,scope=None if self.scope=='LEGACY' else self.scope)
                if source_bone.name not in {name for name,_settings in target_pairs}:
                    raise ValueError('The active source bone must be included in the selected copy scope')
                if not source.proxy or not source.body_uuid:
                    raise ValueError('The active source bone has no valid physical proxy binding')
                from ..models.bone_bindings import validate_proxy_owner
                validate_proxy_owner(arm,source_bone)
                entries=[(name,settings,arm) for name,settings in target_pairs]
            else:
                source_obj=context.active_object
                if source_obj is None or source_obj.type!='MESH' or source_obj.get('ppf_bone_proxy_owned'):
                    raise ValueError('Make a supported mesh the active source')
                source=source_obj.ppf_collision_filter
                target_pairs=targets(context)
                if source_obj not in context.selected_objects or source_obj.name not in {name for name,_settings in target_pairs}:
                    raise ValueError('The active source mesh must be selected and belong to an active PPF group')
                entries=[(name,settings,next(obj for obj in context.selected_objects if obj.name==name)) for name,settings in target_pairs]
            source_owner=arm if arm else source_obj
            _preflight_editable('活动源',source,source_owner)
            for name,settings,owner in entries:_preflight_editable(name,settings,owner)
            snapshot=(bool(source.collision_filter_enabled),tuple(source.collision_group_layers),tuple(source.collision_mask_layers))
            _apply_collision_snapshot(entries,snapshot)
        except Exception as exc:
            self.report({'ERROR'},str(exc));return {'CANCELLED'}
        self.report({'INFO'},f'已从活动目标实际设置复制到 {len(entries)} 个目标；请重新传输和构建')
        return {'FINISHED'}

def draw_filter(layout,context,settings):
    owner=context.scene.ppf_collision_simple
    layout.prop(settings,'collision_filter_enabled',text='启用碰撞组过滤')
    for key,label in (('collision_group_layers','所属组（可多选）'),('collision_mask_layers','允许碰撞的组')):
        layout.label(text=label)
        for first in (0,8):
            row=layout.row(align=True)
            for index in range(first,first+8):
                # Fixed numeric identifiers keep every layer recognizable in a
                # narrow panel. Names are shown in the selector and legend below.
                row.prop(settings,key,index=index,text=f'{index+1:02d}',toggle=True)
    header,body=layout.panel('ppf_collision_group_names',default_closed=True)
    header.label(text='碰撞组名称与编号')
    if body:
        body.prop(owner,'slot',text='要命名的组')
        row=body.row(align=True);row.prop(owner,'label',text='名称');row.operator('ppf.collision_slot_rename',text='保存名称')
        legend,legend_body=body.panel('ppf_collision_group_legend',default_closed=True)
        legend.label(text='编号对应名称')
        if legend_body:
            for index in range(16):
                name=owner.labels[index].label.strip() if index<len(owner.labels) else ''
                legend_body.label(text=f'{index+1:02d}　{name or "未命名"}')
    from ..models.bone_bindings import active_armature
    from ..ops.bone_proxy_editor import editor_active,editor_scope
    unified=bool(active_armature(context) and editor_active(context))
    if active_armature(context) and not unified:layout.prop(owner,'scope',text='复制范围')
    row=layout.row();op=row.operator('ppf.collision_copy_actual',text='从活动目标复制当前设置')
    op.scope=editor_scope(context) if unified else 'LEGACY'
    layout.label(text='双方允许 · 相邻另判',icon='INFO')
    header,body=layout.panel('ppf_collision_rule_scope',default_closed=True)
    header.label(text='规则说明与重新计算')
    if body:
        body.label(text='16层过滤与邻骨排除仅作用于网格')
        body.label(text='解析墙/球按各自面板设置参与碰撞')
        body.label(text='组过滤通过后，再按相邻设置追加排除',icon='INFO')
        body.label(text='相邻设置不能恢复组过滤已拒绝的碰撞')
        body.label(text='改动后重新传输、构建并从头计算')
        body.label(text='网格共用；不可见墙与解析球保留原设置')
        body.label(text='动力学材质组与这16个碰撞组相互独立')
        body.label(text='原生允许相交、时间窗仍可排除碰撞')
        body.label(text='原生入口：动力学组 → 材质参数 → 允许相交')

CLASSES=(PPF_PG_CollisionLabel,PPF_PG_CollisionSimple,PPF_OT_CollisionSimpleApply,PPF_OT_CollisionSlotRename,PPF_OT_CollisionCopyActual)
def register():
    for cls in CLASSES:_ppf_i18n.register_class(cls)
    bpy.types.Scene.ppf_collision_simple=PointerProperty(type=PPF_PG_CollisionSimple)
def unregister():
    del bpy.types.Scene.ppf_collision_simple
    for cls in reversed(CLASSES):bpy.utils.unregister_class(cls)
    _enum_history.clear()
