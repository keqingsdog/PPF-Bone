"""Undo operators and compact author UI over core.cross_links transactions."""
from .. import i18n as _ppf_i18n
import json
from contextlib import contextmanager
import bpy
from ..i18n.props import EnumProperty, StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper
from ..core import cross_links as author
from ..models.bone_bindings import active_armature
from ..core.compat import pose_bone_selected
from bpy.app.handlers import persistent


_focus_selection = None


def _selected_bone_uuids(context, arm):
    """Current authors only; bone selection is not list batch selection."""
    from ..core.bone_selection_parameters import resolve_selected_authors
    try:
        return frozenset(b.ppf_bone_physics.bone_uuid for a,b in resolve_selected_authors(context)
                         if a == arm and b.ppf_bone_physics.bone_uuid)
    except (ValueError, ReferenceError):
        return frozenset()


def _related(row, selected):
    return bool(selected and (row.bone_a_uuid in selected or row.bone_b_uuid in selected))


def _focus_for_selection(context):
    """Follow changed bone selection once; never alter checked batch targets."""
    global _focus_selection
    arm=active_armature(context)
    if arm is None or context.mode not in {'OBJECT','POSE'}:
        _focus_selection=None
        return
    selected=_selected_bone_uuids(context,arm)
    draft=arm.ppf_cross_link_draft
    query=draft.search_text.strip().casefold()
    signature=(arm.session_uid,tuple(sorted(selected)),query)
    if signature==_focus_selection:return
    _focus_selection=signature
    if not selected:return
    names={b.ppf_bone_physics.bone_uuid:b.name for b in arm.data.bones if b.ppf_bone_physics.bone_uuid}
    for index,row in enumerate(arm.ppf_cross_links):
        label=names.get(row.bone_a_uuid,row.bone_a_name)+' '+names.get(row.bone_b_uuid,row.bone_b_name)
        if _related(row,selected) and (not query or query in label.casefold()):
            if draft.active_index!=index:draft.active_index=index
            break


def _focus_timer():
    try:
        _focus_for_selection(bpy.context)
    except (AttributeError, ReferenceError, RuntimeError):
        pass
    return 0.25


@persistent
def _clear_focus(*_args):
    global _focus_selection
    _focus_selection=None


def _arm(context):
    arm=active_armature(context)
    if arm is None:raise ValueError('请选择骨架或它的物理代理')
    return arm


class PPF_OT_CrossLinksCreate(bpy.types.Operator):
    bl_idname='ppf.cross_links_create'
    bl_label='创建链间弹簧'
    bl_options={'REGISTER','UNDO'}

    def execute(self,context):
        try:
            arm=_arm(context);draft=arm.ppf_cross_link_draft
            names=[p.name for p in arm.pose.bones if pose_bone_selected(p)]
            ids=author.create(context,arm,names,mode=draft.mode,diagonals=draft.diagonals,use_mass_defaults=draft.use_mass_defaults,stiffness_N_per_m=draft.stiffness_N_per_m,damping_Ns_per_m=draft.damping_Ns_per_m,tension_enabled=draft.tension_enabled,compression_enabled=draft.compression_enabled,tension_stiffness_N_per_m=draft.tension_stiffness_N_per_m,compression_stiffness_N_per_m=draft.compression_stiffness_N_per_m,exclude_collision=draft.exclude_collision,exclude_direct_only=draft.exclude_direct_only,preview_color_custom=draft.preview_color_custom,preview_color=tuple(draft.preview_color))
            self.report({'INFO'},f'新增 {len(ids)} 条链间弹簧；已有骨对保持原参数')
            return {'FINISHED'}
        except (ValueError,RuntimeError) as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}


class PPF_OT_CrossLinksEdit(bpy.types.Operator):
    bl_idname='ppf.cross_links_edit'
    bl_label='批量修改链间弹簧'
    bl_options={'REGISTER','UNDO'}
    action: EnumProperty(items=[('SET','应用参数',''),('SET_TENSION','应用拉伸',''),('SET_COMPRESSION','应用压缩',''),('SET_DAMPING','应用阻尼',''),('SET_COLLISION','应用碰撞排除',''),('DELETE','删除',''),('RECAPTURE','重记作者参考',''),('DEFAULTS','按质量重置默认',''),('ENABLE','启用',''),('DISABLE','停用','')])
    delete_arm_uuid: StringProperty(options={'HIDDEN'})
    delete_arm_name: StringProperty(options={'HIDDEN'})
    delete_link_uuids: StringProperty(options={'HIDDEN'})

    def invoke(self,context,event):
        # A fresh UI click is a new operation. Blender Repeat Last re-executes
        # the registered operation and retains the locked properties below.
        if self.action=='DELETE':
            self.delete_arm_uuid='';self.delete_arm_name='';self.delete_link_uuids=''
        return self.execute(context)

    def execute(self,context):
        try:
            arm=_arm(context);draft=arm.ppf_cross_link_draft
            if self.action=='DELETE' and self.delete_link_uuids:
                try:ids=json.loads(self.delete_link_uuids)
                except (ValueError,TypeError):raise ValueError('删除目标记录无效，操作已取消')
                matching=[o for o in context.scene.objects if o.type=='ARMATURE' and o.ppf_armature_uuid==self.delete_arm_uuid]
                if (not self.delete_arm_uuid or len(matching)!=1 or matching[0]!=arm
                        or arm.name!=self.delete_arm_name or not isinstance(ids,list)
                        or not ids or any(not isinstance(uid,str) or not uid for uid in ids)
                        or len(set(ids))!=len(ids)):
                    raise ValueError('原删除骨架或目标不再有效，操作已取消')
                counts={uid:sum(row.link_uuid==uid for row in arm.ppf_cross_links) for uid in ids}
                if any(count!=1 for count in counts.values()):
                    raise ValueError('原删除连接已不存在或不唯一，操作已取消')
            else:
                ids=author.selected_link_ids(arm)
                if self.action=='DELETE':
                    if not ids or not arm.ppf_armature_uuid:raise ValueError('请明确选择连接 UUID')
                    matching=[o for o in context.scene.objects if o.type=='ARMATURE' and o.ppf_armature_uuid==arm.ppf_armature_uuid]
                    if len(matching)!=1 or matching[0]!=arm:raise ValueError('骨架身份不唯一，操作已取消')
                    self.delete_arm_uuid=arm.ppf_armature_uuid;self.delete_arm_name=arm.name
                    self.delete_link_uuids=json.dumps(ids,separators=(',',':'))
            action=self.action;params={}
            if action=='SET':params=dict(tension_enabled=draft.tension_enabled,compression_enabled=draft.compression_enabled,tension_stiffness_N_per_m=draft.tension_stiffness_N_per_m,compression_stiffness_N_per_m=draft.compression_stiffness_N_per_m,damping_Ns_per_m=draft.damping_Ns_per_m)
            elif action=='SET_TENSION':params=dict(tension_enabled=draft.tension_enabled,tension_stiffness_N_per_m=draft.tension_stiffness_N_per_m);action='SET'
            elif action=='SET_COMPRESSION':params=dict(compression_enabled=draft.compression_enabled,compression_stiffness_N_per_m=draft.compression_stiffness_N_per_m);action='SET'
            elif action=='SET_COLLISION':params=dict(exclude_collision=draft.exclude_collision);action='SET'
            elif action=='SET_DAMPING':params=dict(damping_Ns_per_m=draft.damping_Ns_per_m);action='SET'
            elif action in {'ENABLE','DISABLE'}:params={'enabled':action=='ENABLE'};action='SET'
            author.edit(context,arm,ids,action,**params)
            return {'FINISHED'}
        except (ValueError,RuntimeError) as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}


class PPF_OT_CrossLinksSelect(bpy.types.Operator):
    bl_idname='ppf.cross_links_select'
    bl_label='选择链间弹簧'
    action: EnumProperty(items=[('ALL','全选',''),('NONE','取消选择','')])

    def execute(self,context):
        arm=_arm(context)
        for row in arm.ppf_cross_links:row.selected=self.action=='ALL'
        return {'FINISHED'}


class PPF_OT_CrossLinksExport(bpy.types.Operator,ExportHelper):
    bl_idname='ppf.cross_links_export'
    bl_label='保存链间弹簧网络预设'
    filename_ext='.json'
    filter_glob:StringProperty(default='*.json',options={'HIDDEN'})

    def execute(self,context):
        try:author.save_preset(context,_arm(context),self.filepath);return {'FINISHED'}
        except (ValueError,RuntimeError,OSError) as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}


class PPF_OT_CrossLinksImport(bpy.types.Operator,ImportHelper):
    bl_idname='ppf.cross_links_import'
    bl_label='载入链间弹簧网络预设'
    bl_options={'REGISTER','UNDO'}
    filename_ext='.json'
    filter_glob:StringProperty(default='*.json',options={'HIDDEN'})
    bone_map_json:StringProperty(name='跨骨架骨名映射',default='{}',description='源骨名到当前骨名的JSON字典；同骨架按UUID恢复')

    def execute(self,context):
        try:author.load_preset(context,_arm(context),self.filepath,json.loads(self.bone_map_json));return {'FINISHED'}
        except (ValueError,RuntimeError,OSError) as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}


class PPF_OT_CrossLinksExcludePair(bpy.types.Operator):
    bl_idname='ppf.cross_links_exclude_pair'
    bl_label='保存选中骨对碰撞排除'
    bl_description='记录两个选中物理骨骼的碰撞排除，不修改已有弹簧参数；重新传输并构建后用于模拟'
    bl_options={'REGISTER','UNDO'}

    def execute(self,context):
        try:
            arm=_arm(context)
            names=[p.name for p in arm.pose.bones if pose_bone_selected(p)]
            if len(names)!=2:raise ValueError('请选中恰好两个不同物理骨骼')
            author.require_ready(context,arm)
            uids={arm.data.bones[n].ppf_bone_physics.bone_uuid for n in names}
            for uid in uids:author.endpoint(context,arm,uid)
            existing=[r.link_uuid for r in arm.ppf_cross_links if {r.bone_a_uuid,r.bone_b_uuid}==uids]
            if existing:author.edit(context,arm,existing,'SET',exclude_collision=True)
            else:author.create(context,arm,names,use_mass_defaults=False,tension_enabled=False,compression_enabled=False,exclude_collision=True)
            self.report({'INFO'},'已记录碰撞排除；请重新传输并构建')
            return {'FINISHED'}
        except (ValueError,RuntimeError) as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}


class PPF_OT_CrossLinkToggleCollision(bpy.types.Operator):
    bl_idname='ppf.cross_link_toggle_collision'
    bl_label='排除连接骨对碰撞'
    bl_description='切换此连接的碰撞排除；重新传输后生效'
    bl_options={'REGISTER','UNDO'}
    link_uuid: StringProperty()

    def execute(self,context):
        try:
            arm=_arm(context)
            matches=[r for r in arm.ppf_cross_links if r.link_uuid==self.link_uuid]
            if len(matches)!=1:raise ValueError('连接不存在或标识不唯一')
            author.edit(context,arm,[self.link_uuid],'SET',exclude_collision=not matches[0].exclude_collision)
            return {'FINISHED'}
        except (ValueError,RuntimeError) as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}


class PPF_UL_CrossLinks(bpy.types.UIList):
    def filter_items(self,context,data,propname):
        links=getattr(data,propname)
        index={b.ppf_bone_physics.bone_uuid:b.name for b in data.data.bones if b.ppf_bone_physics.bone_uuid}
        selected=_selected_bone_uuids(context,data)
        query=data.ppf_cross_link_draft.search_text.strip().casefold()
        flags=[self.bitflag_filter_item if not query or query in (index.get(r.bone_a_uuid,r.bone_a_name)+' '+index.get(r.bone_b_uuid,r.bone_b_name)).casefold() else 0 for r in links]
        indices=sorted(range(len(links)),key=lambda i:not (links[i].bone_a_uuid in selected or links[i].bone_b_uuid in selected))
        order=[0]*len(indices)
        for position,original in enumerate(indices):order[original]=position
        return flags,order


    def draw_item(self,context,layout,data,item,icon,active_data,active_propname,index=0):
        arm=data;label,error=author.status(context,arm,item,getattr(PPF_UL_CrossLinks,'_bone_index',None),getattr(PPF_UL_CrossLinks,'_proxy_counts',None))
        selected=_selected_bone_uuids(context,arm)
        related=_related(item,selected)
        any_related=any(_related(row,selected) for row in arm.ppf_cross_links) if selected else False
        row=layout.row(align=True);row.alert=bool(error)
        row.prop(item,'selected',text='')
        row.label(text='',icon='BONE_DATA' if related else 'BLANK1')
        label_row=row.row(align=True)
        label_row.active=bool(error or related or not any_related)
        label_row.label(text=label,icon='ERROR' if error else 'LINKED' if item.enabled else 'UNLINKED')
        row.operator('ppf.cross_link_toggle_collision',text='',icon='CHECKBOX_HLT' if item.exclude_collision else 'CHECKBOX_DEHLT',emboss=False).link_uuid=item.link_uuid


@contextmanager
def _list_index_scope(bone_index, proxy_counts):
    cls=PPF_UL_CrossLinks
    old_index=getattr(cls,'_bone_index',None);old_counts=getattr(cls,'_proxy_counts',None)
    cls._bone_index=bone_index;cls._proxy_counts=proxy_counts
    try:yield
    finally:
        if old_index is None:del cls._bone_index
        else:cls._bone_index=old_index
        if old_counts is None:del cls._proxy_counts
        else:cls._proxy_counts=old_counts


class PPF_PT_CrossLinks(bpy.types.Panel):
    bl_idname='PPF_PT_CrossLinks'
    bl_label='链间弹簧'
    bl_parent_id='PPF_PT_BonePhysics'
    bl_space_type='VIEW_3D'
    bl_region_type='UI'
    bl_options={'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls,context):return hasattr(context.scene, 'ppf_bone_display')

    def draw(self,context):
        layout=self.layout
        from ..ui import bone_display_context
        bone_display_context.remember(context)
        display=context.scene.ppf_bone_display
        gpu=layout.box();gpu.label(text='链间弹簧 GPU 显示')
        for key,title in (('enabled','GPU 总闸'),('cross_links','显示连接'),('cross_link_color','默认颜色'),('cross_link_width','线宽'),('cross_link_in_front','连接透视')):
            gpu.prop(display,key,text=title)
        if not display.enabled:gpu.label(text='GPU 总开关关闭，连接当前不可见',icon='INFO')
        gpu.label(text='连接示意；软体锚点位置为近似',icon='INFO')
        arm=active_armature(context)
        frozen = False
        if arm is None:
            display_context = bone_display_context.resolve(context) or bone_display_context.resolve_arm(context)
            if display_context is not None:
                context = display_context
                arm = active_armature(context)
                frozen = True
        if arm is None:
            disabled=layout.column();disabled.enabled=False
            disabled.label(text='列表显示结构绑定状态；传输时另行检查当前姿态',icon='INFO')
            disabled.label(text='以模拟起始姿态为零力参考')
            create=disabled.box();create.label(text='创建草稿（仅创建时使用）')
            create.label(text='方式  —')
            create.label(text='骨对仅一条边，无相邻层双斜向',icon='INFO')
            create.label(text='同链及父子骨对可用；自己不可连接',icon='INFO')
            create.label(text='使用质量默认值  —')
            create.label(text='同时排除骨对碰撞  —')
            create.label(text='自定义颜色  —')
            create.operator('ppf.cross_links_create',icon='ADD',text='创建链间弹簧')
            create.operator('ppf.cross_links_exclude_pair',text='仅保存选中骨对碰撞排除')
            disabled.label(text='搜索  —',icon='VIEWZOOM')
            empty_list=disabled.box()
            for _ in range(3):empty_list.label(text=' ')
            row=disabled.row(align=True)
            row.operator('ppf.cross_links_select',text='全选').action='ALL'
            row.operator('ppf.cross_links_select',text='取消').action='NONE'
            from .cross_link_presets import draw_presets
            draw_presets(disabled,context,None)
            row=disabled.row(align=True)
            row.operator('ppf.cross_links_export',text='导出网络')
            row.operator('ppf.cross_links_import',text='载入网络')
            return
        from ..core.bone_selection_parameters import resolve_selected_authors
        content=layout.column()
        content.enabled=bool(not frozen and context.mode in {'OBJECT','POSE'} and resolve_selected_authors(context))
        layout=content
        draft=arm.ppf_cross_link_draft
        layout.label(text='列表显示结构绑定状态；传输时另行检查当前姿态',icon='INFO')
        layout.label(text='以模拟起始姿态为零力参考')
        if arm.mode=='EDIT':layout.label(text='请退出骨骼编辑模式后修改',icon='INFO')
        create=layout.box();create.label(text='创建草稿（仅创建时使用）')
        create.prop(draft,'mode',text='方式')
        if draft.mode!='PAIR':create.prop(draft,'diagonals',text='添加双斜向')
        else:create.label(text='骨对仅一条边，无相邻层双斜向',icon='INFO')
        create.label(text='同链及父子骨对可用；自己不可连接',icon='INFO')
        create.prop(draft,'use_mass_defaults')
        if not draft.use_mass_defaults:
            row=create.row(align=True);row.prop(draft,'tension_enabled',text='拉伸');row.prop(draft,'tension_stiffness_N_per_m',text='N/m')
            row=create.row(align=True);row.prop(draft,'compression_enabled',text='压缩');row.prop(draft,'compression_stiffness_N_per_m',text='N/m')
            create.prop(draft,'damping_Ns_per_m',text='阻尼 N·s/m')
        create.prop(draft,'exclude_collision',text='同时排除骨对碰撞')
        if draft.mode!='PAIR' and draft.diagonals and draft.exclude_collision:
            create.prop(draft,'exclude_direct_only',text='仅排除直接相邻')
        create.prop(draft,'preview_color_custom',text='自定义颜色')
        if draft.preview_color_custom:create.prop(draft,'preview_color',text='颜色')
        create.operator('ppf.cross_links_create',icon='ADD',text='创建链间弹簧')
        create.operator('ppf.cross_links_exclude_pair',text='仅保存选中骨对碰撞排除')
        names=[p.name for p in arm.pose.bones if pose_bone_selected(p)]
        if len(names)==2:
            uids={arm.data.bones[n].ppf_bone_physics.bone_uuid for n in names}
            matched=next((r for r in arm.ppf_cross_links if {r.bone_a_uuid,r.bone_b_uuid}==uids),None)
            state='已保存排除（作者）；需重新传输' if matched and matched.exclude_collision else '作者未排除碰撞'
            if matched and matched.exclude_collision and not matched.enabled:state='已保存排除但连接停用，不会传输'
            create.label(text=' ↔ '.join(names)+'：'+state,icon='INFO')
        bone_index={};proxy_counts={}
        for bone in arm.data.bones:
            uid=bone.ppf_bone_physics.bone_uuid
            if uid:bone_index.setdefault(uid,[]).append(bone)
            proxy=bone.ppf_bone_physics.proxy
            if proxy:proxy_counts[proxy.as_pointer()]=proxy_counts.get(proxy.as_pointer(),0)+1
        layout.label(text='左侧勾选用于批量编辑；右侧为碰撞排除',icon='INFO')
        layout.prop(draft,'search_text',text='',icon='VIEWZOOM')
        with _list_index_scope(bone_index,proxy_counts):
            layout.template_list('PPF_UL_CrossLinks','',arm,'ppf_cross_links',draft,'active_index',rows=3,maxrows=6)
        row=layout.row(align=True)
        row.operator('ppf.cross_links_select',text='全选').action='ALL';row.operator('ppf.cross_links_select',text='取消').action='NONE'
        target_ids=author.selected_link_ids(arm)
        link=author.display_link(arm)
        if link is not None and not target_ids:
            layout.label(text='当前连接被搜索隐藏；清空搜索后编辑',icon='INFO')
        if link is not None:
            actual=arm.ppf_cross_link_actual;box=layout.box()
            box.enabled=bool(target_ids)
            label,error=author.status(context,arm,link,bone_index,proxy_counts)
            box.label(text=label)
            count=len(author.selected_link_ids(arm))
            box.label(text=f'实际参数 · 修改即提交 {count} 条连接')
            mixed=author.mixed_fields(arm,('tension_enabled','compression_enabled','tension_stiffness_N_per_m','compression_stiffness_N_per_m','damping_Ns_per_m','exclude_collision','preview_color_custom','preview_color'))
            if mixed:box.label(text='多选参数混合；显示首个勾选连接，仅修改编辑项',icon='INFO')
            box.prop(actual,'enabled',text='启用连接')
            row=box.row(align=True);row.prop(actual,'tension_enabled',text='拉伸');row.prop(actual,'tension_stiffness_N_per_m',text='N/m')
            row=box.row(align=True);row.prop(actual,'compression_enabled',text='压缩');row.prop(actual,'compression_stiffness_N_per_m',text='N/m')
            box.prop(actual,'damping_Ns_per_m',text='共享阻尼 N·s/m')
            box.prop(actual,'exclude_collision',text='排除连接骨对碰撞')
            box.label(text='已记录碰撞排除；需重新传输' if link.exclude_collision else '此骨对碰撞未排除',icon='INFO')
            if link.exclude_collision and not link.enabled:box.label(text='连接停用：排除不会传输',icon='ERROR')
            box.prop(actual,'preview_color_custom',text='自定义此连接颜色')
            if link.preview_color_custom:box.prop(actual,'preview_color',text='颜色')
            if actual.last_error:box.label(text=actual.last_error,icon='ERROR')
            if error:box.label(text=error,icon='ERROR')
            row=box.row(align=True)
            row.operator_context='INVOKE_DEFAULT'
            row.operator('ppf.cross_links_edit',text='删除',icon='TRASH').action='DELETE'
            row.operator_context='EXEC_DEFAULT'
            row.operator('ppf.cross_links_edit',text='质量默认').action='DEFAULTS'
            row.operator('ppf.cross_links_edit',text='重记参考').action='RECAPTURE'
        from .cross_link_presets import draw_presets
        draw_presets(layout,context,arm)
        row=layout.row(align=True)
        row.operator('ppf.cross_links_export',text='导出网络');row.operator('ppf.cross_links_import',text='载入网络')


CLASSES=(PPF_OT_CrossLinksExcludePair,PPF_OT_CrossLinksCreate,PPF_OT_CrossLinksEdit,PPF_OT_CrossLinksSelect,PPF_OT_CrossLinksExport,PPF_OT_CrossLinksImport,PPF_OT_CrossLinkToggleCollision,PPF_UL_CrossLinks,PPF_PT_CrossLinks)
def register():
    for cls in CLASSES:_ppf_i18n.register_class(cls)
    if _clear_focus not in bpy.app.handlers.load_pre:bpy.app.handlers.load_pre.append(_clear_focus)
    if not bpy.app.timers.is_registered(_focus_timer):bpy.app.timers.register(_focus_timer,first_interval=0.25,persistent=True)
def unregister():
    if bpy.app.timers.is_registered(_focus_timer):bpy.app.timers.unregister(_focus_timer)
    if _clear_focus in bpy.app.handlers.load_pre:bpy.app.handlers.load_pre.remove(_clear_focus)
    _clear_focus()
    for cls in reversed(CLASSES):bpy.utils.unregister_class(cls)
