"""One collapsed bone section in the original PPF sidebar.

Candidate (jointed-material-addon-staging): the Rigid Body / Shape section now
starts with a Physical Body choice (Rigid Body / Soft Body). Rigid Body is the
default and shows exactly the original fields; Soft Body shows the official
SOLID material fields for the real proxy (TetGen + Stable NeoHookean) and hides
the PDRD-only ones. The joint, angular control/limits, drives, frames, root
control and batch sections are shared by both bodies, so nothing an author set
disappears when the material changes.
"""
from .. import i18n as _ppf_i18n
import bpy
from ..i18n import pgettext_iface as iface
from ..core.utils import get_category_name
from ..core.compat import pose_bone_selected
from ..models.bone_bindings import active_armature, active_bone, collect_body_controls_contract, has_physical_joint_parent, soft_material_refusal
from ..models.groups import has_addon_data, get_group_by_uuid, iterate_active_object_groups


def _draw_collision_filter(layout, settings, context):
    from .bone_collision_ui import draw_filter
    draw_filter(layout,context,settings)

def actual_physical_model(context,arm,bone):
    """Only a validated owned group may identify an existing physical body."""
    from ..ops.bone_proxy import owned_bone_material
    from ..core.bone_proxy_visibility import owned_proxy
    s=bone.ppf_bone_physics
    if s.proxy is None:return None,None,'尚未创建'
    try:
        if not owned_proxy(context.scene,arm,bone,s.proxy):return None,None,'绑定待修复'
        group=owned_bone_material(context.scene,arm,bone)
    except (ValueError,RuntimeError,ReferenceError):return None,None,'绑定待修复'
    if group is None:return None,None,'绑定待修复'
    actual={'PDRD':'RIGID','SOLID':'SOFT'}.get(group.object_type)
    if actual is None or actual!=s.physical_model:return None,None,'绑定类型不一致'
    return actual,group,'刚体' if actual=='RIGID' else '软体'


def _selected_physical_model_counts(context, rows):
    """One read-only draw-local inventory for the multi-selection summary.

    Match the strict single-bone presentation gates without resolving the
    same scene, sibling bones and group collection once per selected bone.
    No RNA or result survives this draw; author/write gates are unchanged.
    """
    from collections import Counter
    from ..models.bone_bindings import proxy_parent_armature, armature_data_owner_counts
    from ..models.groups import iterate_object_groups
    scene = context.scene
    body_counts = Counter(obj.get('_solver_uuid') for obj in scene.objects)
    scene_users = Counter(obj.as_pointer() for candidate in bpy.data.scenes for obj in candidate.objects)
    data_owners = armature_data_owner_counts()
    arm_counts = Counter(getattr(obj, 'ppf_armature_uuid', '')
                         for obj in scene.objects if obj.type == 'ARMATURE')
    global_arm_counts = Counter(getattr(obj, 'ppf_armature_uuid', '')
                                for obj in bpy.data.objects if obj.type == 'ARMATURE')
    proxy_counts = Counter(b.ppf_bone_physics.proxy.as_pointer()
                           for obj in scene.objects if obj.type == 'ARMATURE'
                           for b in obj.data.bones if b.ppf_bone_physics.proxy is not None)
    groups = {}
    for group in iterate_object_groups(scene):
        groups.setdefault(group.uuid, group)  # Same first match as get_group_by_uuid.
    counts = {'RIGID': 0, 'SOFT': 0, None: 0}
    for arm, bone in rows:
        s = bone.ppf_bone_physics
        proxy = s.proxy
        actual = None
        try:
            group = groups.get(s.group_uuid) if s.group_uuid else None
            parent_armature = proxy_parent_armature(proxy) if proxy is not None else None
            valid = (proxy is not None and proxy.type == 'MESH'
                     and scene.objects.get(proxy.name) == proxy
                     and scene.objects.get(arm.name) == arm
                     and scene_users[proxy.as_pointer()] == 1 and scene_users[arm.as_pointer()] == 1
                     and not arm.library and not arm.data.library and data_owners[arm.data.as_pointer()] == 1
                     and bool(s.body_uuid) and bool(s.bone_uuid) and bool(arm.ppf_armature_uuid)
                     and proxy.get('ppf_bone_proxy_owned')
                     and proxy.get('_solver_uuid') == s.body_uuid
                     and proxy.get('ppf_bone_uuid') == s.bone_uuid
                     and proxy.get('ppf_armature_uuid') == arm.ppf_armature_uuid
                     and body_counts[s.body_uuid] == 1 and proxy_counts[proxy.as_pointer()] == 1
                     and arm_counts[arm.ppf_armature_uuid] == 1
                     and parent_armature in (None, arm)
                     and (parent_armature is not None or global_arm_counts[arm.ppf_armature_uuid] == 1)
                     and group is not None and group.active
                     and len(group.assigned_objects) == 1
                     and group.assigned_objects[0].uuid == s.body_uuid)
            if valid:
                model = {'PDRD': 'RIGID', 'SOLID': 'SOFT'}.get(group.object_type)
                if model == s.physical_model:
                    actual = model
        except (AttributeError, ValueError, RuntimeError, ReferenceError):
            pass
        counts[actual] += 1
    return counts


def _draw_actual_physics(layout,context,arm,bone):
    from ..core.bone_selection_parameters import resolve_selected_authors
    model,group,label=actual_physical_model(context,arm,bone)
    layout.label(text='实际物理：'+label,icon='ERROR' if model is None and label!='尚未创建' else 'MOD_PHYSICS')
    rows=resolve_selected_authors(context)
    if len(rows)>1:
        counts=_selected_physical_model_counts(context,rows)
        text=f'选中：刚体 {counts["RIGID"]} · 软体 {counts["SOFT"]}'
        if counts[None]:text+=f' · 未确认 {counts[None]}'
        layout.label(text=text,translate=False)
    return model,group


def _draw_soft_material_choice(layout,context,settings,group):
    from ..models.bone_material_presets import material_preset_matches
    layout.label(text='材质设置 · 柔体代理', icon='MATERIAL')
    from ..core.bone_selection_parameters import resolve_selected_authors
    from ..models.bone_bindings import effective_solid_material
    rows = resolve_selected_authors(context)
    if len(rows) > 1:
        values = [effective_solid_material(bone.ppf_bone_physics) for _arm, bone in rows
                  if bone.ppf_bone_physics.physical_model == 'SOFT']
        if values and any(value != values[0] for value in values[1:]):
            layout.label(text='所选材质值不同；修改的字段将统一', icon='INFO')
    layout.prop(settings, 'solid_model', text='材料模型')
    layout.prop(settings, 'solid_young_mpa', text='硬度（MPa）')
    layout.prop(settings, 'solid_poisson_ratio', text='泊松比')
    layout.prop(settings, 'solid_deformation_damping', text='变形阻尼（秒）')
    layout.prop(settings,'material_preset',text='待应用软体材质预设')
    matched=material_preset_matches(settings,group)
    text=('本构参数匹配，质量独立设置' if settings.mass_mode=='MASS' else '材质参数匹配所选方案') if matched else '当前参数自定义'
    layout.label(text=text,icon='INFO')
    if active_armature(context) is not None:
        from ..ops.bone_proxy_editor import editor_scope,editor_active
        op=layout.operator('ppf.bone_material_preset_apply',text='应用所选软体材质')
        op.scope=editor_scope(context) if editor_active(context) else 'SELECTED'


def _selected_anchor_curve_active(context, fallback_bone):
    from ..core.bone_selection_parameters import resolve_selected_authors
    from ..ops.bone_physical_controls import _channel_state
    from ..ops.restoring_curve_operators import read_state
    state = read_state(context.scene)
    if state['mode'] == 'LEGACY': return False
    bones = [bone for _arm, bone in resolve_selected_authors(context)] or [fallback_bone]
    for bone in bones:
        saved = _channel_state(state, bone.ppf_bone_physics.joint_uuid, 'anchor')
        if saved and saved['binding']['enabled']: return True
    return False


def _draw_edit_display(layout,context):
    """Keep the GPU surface controls beside E; drawing never changes display RNA."""
    settings=context.scene.ppf_bone_display
    row=layout.row(align=True);row.prop(settings,'enabled',text='GPU 总闸')
    row=layout.row(align=True);row.prop(settings,'proxy_enabled',text='显示代理')
    controls=layout.column(align=True);controls.enabled=settings.enabled and settings.proxy_enabled
    row=controls.row(align=True);row.prop(settings,'faces',text='表面');row.prop(settings,'edges',text='线框')
    controls.prop(settings,'opacity',text='透明度')
    controls.prop(settings,'refresh_fps',text='共享刷新 FPS')
    controls.label(text='多场景取最高值')
    row=controls.row(align=True);row.prop(settings,'in_front',text='透视');row.prop(settings,'selected_only',text='仅选中')


def _draw_generation(layout,context,settings):
    """Read-only UI: effective defaults come from the mesh producer's resolver."""
    from ..models.proxy_generation import effective_proxy_quality,PROXY_QUALITY_ITEMS
    from ..ops.bone_proxy_editor import editor_scope
    layout.prop(settings,'shape',text='形状')
    if settings.shape=='CUSTOM':layout.prop(settings,'custom_object',text='源网格')
    else:
        row=layout.row(align=True);row.prop(settings,'proxy_face_mode',text='面');row.prop(settings,'proxy_quality',text='面数')
        quality=effective_proxy_quality(settings)
        labels={item[0]:item[1] for item in PROXY_QUALITY_ITEMS}
        layout.label(text=iface('实际面数：{quality}').format(quality=iface(labels.get(quality,quality))),translate=False)
        if settings.shape in {'CAPSULE','SPHERE'}:layout.prop(settings,'radius',text='半径')
        if settings.shape!='SPHERE':layout.prop(settings,'length_ratio',text='骨长比例')
        if settings.shape=='BOX':
            row=layout.row(align=True);row.prop(settings,'box_width',text='宽');row.prop(settings,'box_depth',text='深')
    op=layout.operator('ppf.bone_proxy_generate',text='按当前参数更新范围',icon='FILE_REFRESH')
    op.scope=editor_scope(context)
    layout.label(text='替换范围内已有网格；可撤销',icon='INFO')
    if settings.shape=='CUSTOM':
        layout.label(text='自定义网格不适用重算默认尺寸',icon='INFO')
    else:
        op=layout.operator('ppf.bone_proxy_generate',text='重算默认尺寸',icon='LOOP_BACK')
        op.scope=editor_scope(context);op.reset_default_size=True;op.copy_settings=False


def _has_joint_parent(bone):
    from ..models.bone_bindings import has_physical_joint_parent
    return has_physical_joint_parent(bone)


def _draw_empty_bone_sections(layout):
    """Keep fold identities available without binding an absent author."""
    empty = layout.column(); empty.enabled = False
    empty.label(text='实际物理：尚未创建')
    empty.label(text='参与骨骼物理  —')
    empty.label(text='自定颜色  —')
    for key, title, closed in (
        ('ppf_bone_essentials', '常用物理参数', False),
        ('ppf_bone_all_presets', '骨骼预设', True),
        ('ppf_bone_collision', '碰撞组', True),
        ('ppf_bone_simple_limits', '软角度范围', True),
    ):
        header, body = layout.panel(key, default_closed=closed)
        header.label(text=title)
        if body:
            if key == 'ppf_bone_essentials':
                empty = body.column(); empty.enabled = False
                empty.label(text='回正力矩（每度）', icon='CONSTRAINT_BONE')
                empty.label(text='无父关节；使用下方根控制', icon='INFO')
                empty.label(text='物理体  —')
                section, generation = body.panel('ppf_bone_simple_generation', default_closed=False)
                section.label(text='代理形状与面数')
                if generation:
                    empty = generation.column(); empty.enabled = False
                    for label in ('范围  —', '形状  —', '面 / 面数  —', '半径  —', '骨长比例  —'):
                        empty.label(text=label)
                    empty.operator('ppf.bone_proxy_generate', text='按当前参数更新范围', icon='FILE_REFRESH')
                    empty.label(text='替换范围内已有网格；可撤销', icon='INFO')
                    empty.operator('ppf.bone_proxy_generate', text='重算默认尺寸', icon='LOOP_BACK')
                empty = body.column(); empty.enabled = False
                empty.label(text='质量  —')
                empty.label(text='初始运动方式  —')
            elif key == 'ppf_bone_all_presets':
                empty = body.column(); empty.enabled = False
                for label in ('来源  —', '骨链  —', '恢复力按质量和惯量自动适配', '应用内容', '关节/恢复 · 材质 · 质量 / 密度'):
                    empty.label(text=label)
                empty.operator('ppf.bone_chain_preset_apply', text='应用预设')
            elif key == 'ppf_bone_collision':
                empty = body.column(); empty.enabled = False
                empty.label(text='启用碰撞组过滤  —')
                empty.label(text='所属组 / 允许组  —')
            else:
                empty = body.column(); empty.enabled = False
                empty.label(text='当前骨骼无适用软关节  —')


class _FrozenPropertyLayout:
    """Use live read-only owners while preserving SelectionLayout's presentation."""
    def __init__(self, layout, context):
        object.__setattr__(self, '_layout', layout)
        object.__setattr__(self, '_context', context)

    def __setattr__(self, name, value):
        setattr(self._layout, name, value)

    def __getattr__(self, name):
        member = getattr(self._layout, name)
        if name in {'row', 'column', 'box', 'split', 'grid_flow'}:
            return lambda *args, **kwargs: _FrozenPropertyLayout(member(*args, **kwargs), self._context)
        if name == 'panel':
            return lambda *args, **kwargs: tuple(
                _FrozenPropertyLayout(item, self._context) if item is not None else None
                for item in member(*args, **kwargs))
        if name in {'prop', 'prop_enum'}:
            def prop(owner, field, *args, **kwargs):
                if owner == self._context.window_manager.ppf_selection_parameters:
                    from ..core.bone_selection_parameters import _value_owner
                    owner = _value_owner(self._context, field)
                    if owner is None:
                        return None
                return member(owner, field, *args, **kwargs)
            return prop
        return member


class PPF_PT_BonePhysics(bpy.types.Panel):
    @classmethod
    def poll(cls, context):
        from .layout import classic_visible
        return classic_visible(context) and cls.content_poll(context)

    bl_idname = "PPF_PT_BonePhysics"
    bl_label = "骨骼物理"
    bl_order = 11
    bl_ui_units_x = 28
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = get_category_name()
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def content_poll(cls, context):
        return hasattr(context.scene, 'ppf_bone_display')

    def draw(self, context):
        from ..core.bone_selection_parameters import draw_read_scope
        from . import bone_display_context
        bone_display_context.remember(context)
        frozen = active_armature(context) is None and not (context.object and context.object.get('ppf_bone_proxy_owned'))
        display = bone_display_context.resolve(context) if frozen else None
        with draw_read_scope(display or context):
            self._draw_read_contents(display or context, frozen=display is not None)

    def _draw_read_contents(self, context, frozen=False):
        from ..core.bone_selection_parameters import SelectionLayout,active_author,resolve_selected_authors
        base = _FrozenPropertyLayout(self.layout, context) if frozen else self.layout
        layout = SelectionLayout(base, context)
        bone = active_bone(context)
        if active_armature(context) is None and not (context.object and context.object.get('ppf_bone_proxy_owned')):
            header, editor = layout.panel('ppf_bone_editor_workspace', default_closed=False)
            header.label(text='代理编辑模式', icon='EDITMODE_HLT')
            if editor:
                _draw_edit_display(editor, context)
                disabled = editor.column(); disabled.enabled = False
                disabled.label(text='未选骨骼 · 选中 0 · 范围 0', icon='BONE_DATA')
                disabled.operator('ppf.bone_proxy_editor', text='进入代理编辑', icon='EDITMODE_HLT')
                op = disabled.operator('ppf.bone_editor_action', text='添加 / 更新整个骨架', icon='FILE_REFRESH')
                op.action = 'UPDATE'; op.scope = 'ALL'
                disabled.operator('ppf.bone_rig_duplicate', text='复制骨架与代理', icon='DUPLICATE')
            _draw_empty_bone_sections(layout)
            return
        try:
            selected = resolve_selected_authors(context)
        except ValueError as exc:
            # Reject ambiguous/missing proxy ownership before exposing writers.
            # Drawing must never repair a binding or pick an arbitrary owner.
            warning = layout.box(); warning.alert = True
            warning.label(text='骨骼绑定异常，已暂停参数编辑', icon='ERROR')
            warning.label(text=str(exc), translate=False)
            warning.label(text='请取消问题代理的选择，并核对其骨骼绑定', icon='INFO')
            _draw_empty_bone_sections(layout)
            return
        if context.mode not in {'POSE','OBJECT'} or not selected:bone=None
        if active_armature(context) is None:
            layout.label(text=context.object.name, icon='MESH_DATA', translate=False)
            source=active_author(context)
            actual,owned_group=_draw_actual_physics(layout,context,*source) if source is not None else (None,None)
            if context.mode=='OBJECT':
                count=len(resolve_selected_authors(context))
                if source is not None and count:
                    s=source[1].ppf_bone_physics
                    if actual=='SOFT':_draw_soft_material_choice(layout,context,s,owned_group)
                    layout.label(text=f'多选 {count} 项 · 显示活动项；修改即统一',icon='BONE_DATA')
                    layout.prop(s,'collide_connected',text='允许相邻骨骼碰撞')
                    layout.prop(s,'mass_mode',text='质量输入')
                    layout.prop(s,'target_mass_kg' if s.mass_mode=='MASS' else 'density')
                    group=get_group_by_uuid(context.scene,s.group_uuid) if s.group_uuid else None
                    if group is not None:layout.prop(group,'friction',text='摩擦系数')
                    layout.prop(s,'preview_color_custom',text='自定义预览色')
                    if s.preview_color_custom:layout.prop(s,'preview_color',text='颜色')
                    layout.prop(s,'collision_filter_enabled',text='启用碰撞组过滤')
                    header,masks=layout.panel('ppf_proxy_selection_masks',default_closed=True)
                    header.label(text='碰撞层')
                    if masks:
                        for key,label in (('collision_group_layers','所属组'),('collision_mask_layers','允许组')):
                            masks.label(text=label)
                            for first in (0,8):
                                row=masks.row(align=True)
                                for index in range(first,first+8):row.prop(s,key,index=index,text=f'{index+1:02d}',toggle=True)
                    error=context.window_manager.ppf_selection_parameters.error
                    if error:layout.label(text=error,icon='ERROR')
            else:layout.label(text='正在编辑真实代理网格 · Tab/E 返回',icon='EDITMODE_HLT')
            _draw_edit_display(layout,context)
            layout.operator('ppf.bone_proxy_return', text='返回绑定骨骼', icon='BONE_DATA')
            return
        self._draw_editor(layout,context,bone,frozen=frozen)
        content = layout.column(); content.enabled = not frozen
        layout = content
        if not bone:
            layout.label(text="选中骨骼，或点击进入代理编辑", icon='BONE_DATA')
            _draw_empty_bone_sections(layout)
            return
        s = bone.ppf_bone_physics
        count=len(resolve_selected_authors(context))
        if count>1:layout.label(text=f'多选 {count} 项 · 显示活动项；修改即统一',icon='INFO')
        error=context.window_manager.ppf_selection_parameters.error
        if error:layout.label(text=error,icon='ERROR')
        layout.label(text=bone.name, icon='BONE_DATA', translate=False)
        _draw_actual_physics(layout,context,active_armature(context),bone)
        layout.prop(s, "enabled", text='参与骨骼物理')
        color = layout.row(align=True)
        color.prop(s, 'preview_color_custom', text='自定颜色')
        if s.preview_color_custom:
            color.prop(s, 'preview_color', text='')
        else:
            color.label(text='自动')
        self._draw_simple(layout,context,s,bone)

    def _draw_editor(self,layout,context,bone,frozen=False):
        from ..ops.bone_proxy_editor import editor_active,editor_scope
        from ..ops.bone_proxy import selected_bones
        header,editor=layout.panel('ppf_bone_editor_workspace',default_closed=False)
        header.label(text='代理编辑模式',icon='EDITMODE_HLT')
        if not editor:return
        _draw_edit_display(editor,context)
        content = editor.column(); content.enabled = not frozen
        editor = content
        wm=context.window_manager;scope=editor_scope(context)
        arm=active_armature(context)
        selected=sum(pose_bone_selected(p) for p in arm.pose.bones) if arm else 0
        try:_arm,bones=selected_bones(context,scope=scope);count=len(bones)
        except ValueError:count=0
        editor.label(text=f'{bone.name if bone else "未选骨骼"} · 选中 {selected} · 范围 {count}',icon='BONE_DATA',translate=False)
        editor.operator('ppf.bone_proxy_editor',text='退出代理编辑' if editor_active(context) else '进入代理编辑',icon='EDITMODE_HLT')
        selection = editor.row(align=True)
        selection.operator('ppf.select_physics_bones', text='选物理骨骼', icon='BONE_DATA')
        selection.operator('ppf.select_collision_group_bones', text='按碰撞组选骨', icon='RESTRICT_SELECT_OFF')
        authoring = editor.column(align=True)
        authoring.enabled = context.mode in {'POSE', 'OBJECT'} and bool(selected)
        authoring.operator_context = 'INVOKE_DEFAULT'
        authoring.operator('ppf.import_mmd_physics', text='导入选中骨骼的 MMD 物理', icon='IMPORT')
        authoring.operator('ppf.fit_bone_proxies_to_weights', text='按蒙皮权重拟合代理', icon='MOD_ARMATURE')
        if not editor_active(context):
            op=editor.operator('ppf.bone_editor_action',text='添加 / 更新代理' if count else '添加 / 更新整个骨架',icon='FILE_REFRESH')
            op.action='UPDATE';op.scope=scope if count else 'ALL'
            editor.operator('ppf.bone_rig_duplicate',text='复制骨架与代理',icon='DUPLICATE')
            return
        editor.operator('ppf.bone_rig_duplicate',text='复制骨架与代理',icon='DUPLICATE')
        editor.operator_context='INVOKE_DEFAULT'
        editor.prop(wm,'ppf_bone_editor_scope',text='范围')
        row=editor.row(align=True)
        for action,label,icon in (('UPDATE','添加 / 更新','ADD'),('DELETE','删除','TRASH')):
            op=row.operator('ppf.bone_editor_action',text=label,icon=icon);op.action=action;op.scope=scope
        row=editor.row(align=True)
        for action,label in (('ENABLE','启用物理'),('DISABLE','停用物理')):
            op=row.operator('ppf.bone_editor_action',text=label);op.action=action;op.scope=scope
        if not bone:return
        s=bone.ppf_bone_physics
        header,shape=editor.panel('ppf_bone_editor_shape',default_closed=False)
        header.label(text='形状与面数',icon='MESH_DATA')
        if shape:_draw_generation(shape,context,s)
        editor.label(text='T 循环形状 · 自定义源齐全时加入',translate=False)
        editor.prop(wm,'ppf_bone_editor_category',text='复制活动骨骼参数')
        op=editor.operator('ppf.bone_settings_copy',text=f'复制到范围（{count} 根）',icon='CHECKMARK')
        op.scope=scope;op.category=wm.ppf_bone_editor_category

    def _draw_simple(self,layout,context,s,bone):
        """Only the common bone authoring controls; saved detailed data is untouched."""
        from ..models.new_binding_defaults import uses_new_binding_preset,new_binding_plan
        first=uses_new_binding_preset(s);plan=new_binding_plan(s)
        model=plan.get('physical_model',s.physical_model)
        has_joint_parent=_has_joint_parent(bone)
        header,body=layout.panel('ppf_bone_essentials',default_closed=False)
        header.label(text='常用物理参数')
        if body:
            restoring=bone.ppf_bone_restoring_controls
            direct=body.column(align=True)
            from ..ops.bone_chain_presets import _joint_parent_status
            valid_parent,_parent_reason=_joint_parent_status(bone)
            direct.enabled=bool(context.mode in {'OBJECT','POSE'} and s.enabled and s.proxy and valid_parent and s.joint_mode == 'SOFT')
            target_loaded=bool(getattr(restoring,'draft_loaded',False) and getattr(restoring,'loaded_bone_uuid','')==getattr(s,'bone_uuid','') and getattr(restoring,'loaded_joint_uuid','')==getattr(s,'joint_uuid',''))
            direct.label(text='回正力矩（每度）',icon='CONSTRAINT_BONE')
            if not valid_parent:direct.label(text='无父关节；使用下方根控制',icon='INFO')
            physical_loaded=bool(restoring.physical_draft_loaded and target_loaded)
            if physical_loaded and restoring.physical_status:
                direct.label(text=restoring.physical_status,icon='INFO')
            from ..ops.bone_physical_controls import draw_inline
            draw_inline(direct,context,bone,restoring,physical_loaded)
            from ..ops.bone_proxy_editor import editor_active,editor_scope
            unified=editor_active(context)

            self._draw_physical_model_choice(body, s, first, context)
            self._draw_body_switch(body,context,s)
            actual,owned_group,_label=actual_physical_model(context,active_armature(context),bone)
            if actual=='SOFT':_draw_soft_material_choice(body,context,s,owned_group)
            if not unified:
                header,generation=body.panel('ppf_bone_simple_generation',default_closed=False)
                header.label(text='代理形状与面数')
                if generation:
                    generation.prop(context.window_manager,'ppf_bone_editor_scope',text='范围')
                    _draw_generation(generation,context,s)
            body.prop(s,'mass_mode',text='质量')
            body.prop(s,'target_mass_kg' if s.mass_mode=='MASS' else 'density',text='质量 kg' if s.mass_mode=='MASS' else '密度 kg/m³')
            group=get_group_by_uuid(context.scene,s.group_uuid) if s.group_uuid else None
            if has_joint_parent and s.joint_mode == 'SOFT':
                body.label(text='柔性关节')
                try:
                    controlled = _selected_anchor_curve_active(context, bone)
                except (ValueError, KeyError, TypeError):
                    controlled = True
                    body.label(text='恢复曲线状态无效，请检查曲线', icon='ERROR')
                anchor = body.row()
                anchor.enabled = not controlled
                anchor.prop(s, 'anchor_stiffness', text='连接刚度（N/m）')
                if controlled: body.label(text='连接由恢复曲线控制，请使用曲线编辑入口', icon='INFO')
            if group and group.active:body.prop(group,'friction',text='摩擦系数')
            self._draw_basic_control(body,s,bone,model)
            if has_joint_parent:
                self._draw_joint_choice(body,s,model)
        header,presets=layout.panel('ppf_bone_all_presets',default_closed=True)
        header.label(text='骨骼预设',icon='PRESET')
        if presets:
            state=context.window_manager.ppf_bone_preset_library
            presets.prop(state,'source',text='来源',expand=True)
            builtin=state.source=='BUILTIN'
            restoring_owner=context.scene.ppf_restoring_curves
            if builtin:
                presets.prop(restoring_owner,'bone_chain_preset',text='骨链')
                presets.label(text='恢复力按质量和惯量自动适配',icon='INFO')
                from ..models.bone_chain_presets import PRESETS as chain_presets
                from ..models.bone_material_presets import PRESETS as material_presets
                material=material_presets[chain_presets[restoring_owner.bone_chain_preset].material_id]
                presets.label(text=iface('预设密度：{density} kg/m³').format(density=f'{material.density_kg_m3:g}'),translate=False)
                actual,_,_=actual_physical_model(context,active_armature(context),bone)
                if actual=='SOFT':presets.label(text=iface('预设软体硬度：{young} MPa').format(young=f'{material.young_pa/1e6:g}'),translate=False)
            else:
                from ..ops.bone_presets import draw_user_presets
                draw_user_presets(presets,context,embedded=True,controls=False)
            presets.label(text='应用内容')
            row=presets.row(align=True)
            row.prop(state,'apply_restoring',text='关节/恢复')
            row.prop(state,'apply_material',text='材质')
            row.prop(state,'apply_mass',text='质量 / 密度')
            if not builtin:
                row=presets.row(align=True)
                row.prop(state,'apply_shape',text='形状')
                row.prop(state,'apply_collision',text='碰撞')
                row.prop(state,'apply_display',text='显示')
            from ..ops.bone_proxy_editor import editor_active,editor_scope
            unified=editor_active(context)
            row=presets.row(align=True)
            if not unified:row.prop(state,'scope',text='范围')
            if not builtin:row.prop(state,'size_mode',text='')
            apply_row=presets.row();apply_row.enabled=builtin or bool(state.entries)
            if builtin:
                op=apply_row.operator('ppf.bone_chain_preset_apply',text='应用预设',icon='CHECKMARK')
                op.preset=restoring_owner.bone_chain_preset
                op.scope=editor_scope(context) if unified else state.scope
            else:
                op=apply_row.operator('ppf.bone_user_preset_apply',text='应用预设',icon='CHECKMARK')
                op.scope_override=editor_scope(context) if unified else 'STATE'
        follows_parent=bool(bone.parent and not bone.parent.ppf_bone_physics.enabled and getattr(s,'auto_follow_parent',False))
        has_joint_parent=bool((bone.parent and bone.parent.ppf_bone_physics.enabled) or follows_parent)
        header,collision=layout.panel('ppf_bone_collision',default_closed=True)
        header.label(text='碰撞组')
        if collision:
            _draw_collision_filter(collision,s,context)
            if has_joint_parent and s.joint_mode == 'SOFT':
                if follows_parent:collision.label(text='内部锚点不参与碰撞',icon='INFO')
                else:collision.prop(s,'collide_connected',text='允许相邻骨骼碰撞')
        if has_joint_parent and s.joint_mode == 'SOFT':
            header,limits=layout.panel('ppf_bone_simple_limits',default_closed=True)
            header.label(text='软角度范围')
            if limits:
                limits.prop(s,'soft_limit_mode',text='摆动范围')
                limits.prop(s,'soft_limit_policy',text='力度')
                if s.soft_limit_mode=='CONE':
                    limits.prop(s,'swing_cone_half_angle',text='圆锥半角')
                    if s.soft_limit_policy=='ABSOLUTE_SI':limits.prop(s,'swing_cone_stiffness_degree',text='圆锥 N·mm/°')
                if s.soft_limit_policy=='AUTO_INERTIA':limits.prop(s,'soft_limit_strength',text='强度倍率')
                for index,label in enumerate(('X 扭转','Y 弯曲','Z 弯曲')):
                    if s.soft_limit_mode=='CONE' and index>0:continue
                    row=limits.row(align=True);row.prop(s,'soft_limits_enabled',index=index,text=label)
                    row=limits.row(align=True);row.enabled=s.soft_limits_enabled[index]
                    row.prop(s,'soft_lower',index=index,text='下限');row.prop(s,'soft_upper',index=index,text='上限')
                    if s.soft_limit_policy=='ABSOLUTE_SI':limits.prop(s,'soft_stiffness_degree',index=index,text=label+' N·mm/°')
        else:
            header,limits=layout.panel('ppf_bone_simple_limits',default_closed=True)
            header.label(text='软角度范围')
            if limits:
                empty=limits.column();empty.enabled=False
                empty.label(text='当前骨骼无适用软关节  —')

    def _draw_body_switch(self, layout, context, s):
        """Explicit RIGID <-> SOFT switch for the bone's EXISTING proxy.

        The enum above stays the author's declared body; this row is the ONE action
        that moves an existing owned proxy to the other body without deleting it, so
        edited geometry, transforms, local frames and identities survive.
        """
        from ..ops.bone_proxy import owned_bone_material
        from ..ops.bone_proxy_editor import editor_active,editor_scope
        arm = active_armature(context)
        bone = active_bone(context)
        group = owned_bone_material(context.scene, arm, bone) if (arm and bone) else None
        declared = 'SOFT' if s.physical_model == 'SOFT' else 'RIGID'
        if group is not None:
            actual = 'SOFT' if group.object_type == 'SOLID' else 'RIGID'
            if actual != declared:
                layout.label(text="物理类型声明与专属材质不一致", icon='ERROR')
        else:
            return
        if declared != actual:
            op = layout.operator('ppf.bone_physical_body_change', text='按专属材质修复声明', icon='FILE_REFRESH')
            op.scope = editor_scope(context) if editor_active(context) else 'ALL'
            op.repair_from_material = True
        target = 'RIGID' if actual == 'SOFT' else 'SOFT'
        label = '切换为刚体' if target=='RIGID' else '切换为软体'
        scopes=[(editor_scope(context),'')] if editor_active(context) else [('SELECTED','选中骨骼'),('CHAIN','当前骨链')]
        for scope, scope_label in scopes:
            row = layout.row(align=True)
            if scope_label:row.label(text=scope_label)
            op = row.operator("ppf.bone_physical_body_change", text=label)
            op.scope = scope
            op.physical_body = target

    def _draw_physical_model_choice(self, layout, s, first, context=None):
        if first:
            layout.prop(s, 'new_binding_physical_model')
        else:
            from ..ops.bone_proxy_editor import editor_active
            label='物理体：' + ('软体' if s.physical_model == 'SOFT' else '刚体')
            if context is not None and editor_active(context):layout.label(text=label)
            else:layout.operator_menu_enum('ppf.bone_physical_body_change', 'physical_body', text=label)

    def _draw_shape(self, layout, context, s):
        # The physical body comes first: it is what decides whether the proxy is
        # a rigid PDRD body or a real SOLID tet body, and therefore which
        # material fields below are meaningful.
        from ..models.new_binding_defaults import uses_new_binding_preset,new_binding_plan
        first_creation=uses_new_binding_preset(s)
        plan=new_binding_plan(s)
        effective_model=plan.get('physical_model',s.physical_model)
        self._draw_physical_model_choice(layout, s, first_creation, context)
        self._draw_body_switch(layout, context, s)
        layout.prop(s, "shape")
        if s.shape == 'CUSTOM':
            layout.prop(s, "custom_object")
        else:
            layout.prop(s, "length_ratio")
            layout.label(text='新建连接代理按接触余量自动缩短；已有网格不自动改变',icon='INFO')
            if s.shape == 'CAPSULE':
                layout.prop(s, "radius")
                layout.prop(s,'new_proxy_quality' if first_creation and not s.is_property_set('proxy_quality') else 'proxy_quality')
                layout.label(text='物理网格精度会影响碰撞与质量惯量，并非仅显示',icon='INFO')
                if not first_creation:layout.label(text='修改精度后须显式重建；更新保留已编辑网格')
            else:
                row = layout.row(align=True);row.prop(s, "box_width");row.prop(s, "box_depth")
        self._draw_mass(layout, context, s)
        group = get_group_by_uuid(context.scene, s.group_uuid) if s.group_uuid else None
        if effective_model == 'SOFT':
            if first_creation:
                from ..models.new_binding_defaults import NEW_SOFT_DAMPING_SECONDS
                layout.label(text=f"新建软体：杨氏模量 {plan.get('solid_young_modulus',s.solid_young_modulus)/1e6:g} MPa")
                layout.label(text=f"材料阻尼 {NEW_SOFT_DAMPING_SECONDS:g} 秒；关节锚点 {plan.get('anchor_stiffness',s.anchor_stiffness):g} 牛/米")
                layout.label(text='仅新建初始化；材料刚度与关节恢复力分别调节',icon='INFO')
            layout.label(text="软体代理使用实体四面体与稳定弹性材料", icon='MOD_PHYSICS')
            # The BONE's own fields are the editable surface. Each one carries an
            # ownership-guarded callback that mirrors it onto the group this bone
            # owns, and `_group` reads these same fields on every geometry
            # update, so the two directions cannot drift: an update cannot
            # overwrite the group with a stale value, and a copy carries the
            # material between two owned soft bodies.
            if not first_creation:
                layout.prop(s, "solid_model")
                layout.prop(s, "solid_young_modulus", text="材料硬度（帕）")
                layout.prop(s, "solid_poisson_ratio")
            if group and group.active and group.object_type == 'SOLID':
                layout.prop(group, "friction")
                if s.is_property_set('solid_deformation_damping'):
                    layout.prop(s, 'solid_deformation_damping', text='材料变形阻尼（秒）')
                else:
                    layout.prop(group, 'deformation_damping', text='材料变形阻尼（秒）')
            layout.prop(s, 'material_preset')
            layout.operator('ppf.bone_material_preset_apply', text='应用所选材质预设')
            from ..models.bone_material_presets import PRESETS
            layout.label(text=PRESETS[s.material_preset].evidence, icon='INFO')
            layout.label(text='低密度为厚代理的等效质量；并非材料体密度标定', icon='INFO')
            layout.label(text='材质刚度和阻尼与关节恢复力独立设置', icon='INFO')
            layout.prop(s, 'fem_scale_enabled')
            measured = layout.column(align=True)
            measured.enabled = False
            measured.label(text='初始骨骼局部轴缩放')
            axes = measured.row(align=True)
            for index, axis in enumerate(('X', 'Y', 'Z')):
                axes.prop(s, 'fem_scale_last', index=index, text=axis)
            layout.label(text=s.fem_scale_status, icon='INFO')
            layout.label(text='局部 Y 为骨长方向；形变缩放每帧从初始形状测量')
            layout.label(text='单根骨骼只能近似表示剪切和弯曲', icon='INFO')
            if s.fem_scale_shear > 1e-6 or s.fem_scale_variation > 1e-6 or s.fem_scale_orientation_error > 1e-6:
                diagnostics = layout.column(align=True)
                diagnostics.enabled = False
                diagnostics.prop(s, 'fem_scale_shear')
                diagnostics.prop(s, 'fem_scale_variation')
                diagnostics.prop(s, 'fem_scale_orientation_error')
            layout.label(text="软体连接支持柔性关节；根可固定、跟随或释放", icon='INFO')
        else:
            if group and group.active:
                layout.prop(group, "friction")
            layout.label(text="刚体保留原代理、关节与角向控制", icon='INFO')
        op = layout.operator("ppf.bone_editor_action", text="按参数重建形状", icon='FILE_REFRESH');op.action='REBUILD'
        from ..ops.bone_proxy_editor import editor_scope
        op.scope=editor_scope(context)
        layout.label(text="重建会替换手工网格和变换", icon='INFO')
        # A refusal is surfaced where the offending setting is authored, with the
        # same field name the transfer-time gate uses, so the panel never claims
        # a support this engine does not have.
        bone = active_bone(context)
        refusal = soft_material_refusal(s, bone.name if bone else '', has_physical_joint_parent(bone),assumed_model=effective_model)
        if refusal is not None:
            layout.label(text=iface("Unsupported: {field}: {detail}").format(field=refusal[0], detail=iface(refusal[1])), icon='ERROR')

    def _draw_mass(self, layout, context, s):
        layout.prop(s, 'mass_mode')
        if s.mass_mode == 'MASS': layout.prop(s, 'target_mass_kg')
        rho = layout.row(); rho.enabled = s.mass_mode == 'DENSITY'
        rho.prop(s, 'density', text='密度（kg/m³）')
        from ..core.bone_mass import cached_status
        valid, status = cached_status(context.scene, active_armature(context), active_bone(context))
        if valid:
            display = layout.column(); display.enabled = False
            display.prop(s, 'mass_volume_m3'); display.prop(s, 'mass_measured_kg')
        layout.label(text=status, icon='INFO' if valid else 'ERROR')
        layout.label(text='体积来自作者基础网格；含对象缩放；提交前须应用缩放', icon='INFO')
        layout.label(text='单一外向闭合连通壳；自交、镜像、开口会拒绝换算')

    @staticmethod
    def _effective_model(s):
        from ..models.new_binding_defaults import new_binding_plan
        return new_binding_plan(s).get('physical_model', s.physical_model)

    def _draw_basic_control(self,control,s,bone,model):
        """Basic movement controls plus explicit removal of saved invalid events."""
        physical_parent=bool(s.use_hierarchy_joint and bone.parent and bone.parent.ppf_bone_physics.enabled)
        follows_parent=bool(s.use_hierarchy_joint and bone.parent and not physical_parent and s.auto_follow_parent)
        from ..core.bone_selection_parameters import resolve_selected_authors
        if bone.parent or any(selected.parent for _arm,selected in resolve_selected_authors(bpy.context)):
            control.prop(s, 'use_hierarchy_joint', text='连接父骨物理')
        if s.use_hierarchy_joint and bone.parent and not physical_parent:
            control.prop(s,'auto_follow_parent',text='跟随父骨骼')
        restricted=follows_parent or (model=='SOFT' and physical_parent)
        if restricted:
            if s.root_mode!='FREE':
                saved=control.row();saved.enabled=False
                saved.prop(s,'root_mode',text='已保存的运动方式（不支持）')
            control.prop_enum(s,'root_mode','FREE',text='改为自由动态' if s.root_mode!='FREE' else '自由动态')
        else:
            control.prop(s,'root_mode',text='初始运动方式')
        if restricted and s.root_control_events:
            control.label(text='移除不支持的旧事件',icon='ERROR')
            for index,event in enumerate(s.root_control_events):
                row=control.row(align=True)
                saved=row.row();saved.enabled=False
                saved.label(text=f'帧 {event.frame} · {event.mode}',translate=False)
                op=row.operator('ppf.bone_control_event_edit',text='移除',icon='REMOVE')
                op.action='REMOVE';op.index=index

    def _draw_control(self, control, s, bone, model=None):
        """Read-only capability surface; saved invalid values require explicit edits."""
        model = self._effective_model(s) if model is None else model
        physical_parent = bool(s.use_hierarchy_joint and bone.parent and bone.parent.ppf_bone_physics.enabled)
        follows_parent = bool(s.use_hierarchy_joint and bone.parent and not physical_parent and s.auto_follow_parent)
        connected_soft = model == 'SOFT' and (physical_parent or follows_parent)
        if s.use_hierarchy_joint and bone.parent and not physical_parent:
            control.prop(s, 'auto_follow_parent', text='跟随非物理父骨骼')
        restricted = connected_soft or follows_parent
        if restricted:
            if s.root_mode != 'FREE':
                saved = control.row(); saved.enabled = False
                saved.prop(s, 'root_mode', text='已保存的初始运动方式（不支持）')
            control.prop_enum(s, 'root_mode', 'FREE', text='自由动态（显式修正）' if s.root_mode != 'FREE' else '初始运动方式：自由动态')
            if connected_soft:
                control.label(text='软体连接子骨仅支持自由动态', icon='INFO')
            else:
                control.label(text='内部运动学锚点跟随父骨骼', icon='INFO')
        else:
            control.prop(s, 'root_mode', text='初始运动方式')
        if s.root_mode == 'KINEMATIC' or s.root_control_events:
            source = control.row(); source.enabled = not restricted
            source.prop(s, 'root_motion_source')
        # Keep every saved event visible, including unsupported legacy events.
        for index, event in enumerate(s.root_control_events):
            row = control.row(align=True)
            fields = row.row(align=True); fields.enabled = not restricted
            fields.prop(event, 'frame'); fields.prop(event, 'mode', text='')
            op = row.operator('ppf.bone_control_event_edit', text='', icon='REMOVE')
            op.action = 'REMOVE'; op.index = index
            if event.mode != 'FREE':
                binding = control.row(); binding.enabled = not restricted
                binding.prop(event, 'pose_binding')
        add = control.row(); add.enabled = not restricted
        add.operator('ppf.bone_control_event_edit', text='添加释放 / 接管事件', icon='ADD').action = 'ADD'
        if restricted and s.root_control_events:
            control.label(text='已保存事件不受支持；请逐条移除以修正', icon='ERROR')
    def _draw_joint_choice(self, joint, s, model):
        # Legacy enum identifiers remain readable so old files are never
        # silently converted; only the compliant option can be selected.
        if s.joint_mode != 'SOFT':
            joint.label(text='已保存的硬关节不再支持；请显式选择柔性关节', icon='ERROR')
        joint.prop_enum(s, 'joint_mode', 'SOFT', text='柔性关节（显式修正）' if s.joint_mode != 'SOFT' else '关节类型：柔性关节')

    def _draw_joint(self, joint, s, model=None):
        model = s.physical_model if model is None else model
        self._draw_joint_choice(joint, s, model)
        if s.joint_mode != 'SOFT':
            return
        joint.prop(s, "collide_connected")
        joint.prop(s, 'anchor_stiffness')
        box = joint.box();box.label(text="角向恢复与柔性限位")
        box.label(text="下方角刚度与阻尼仅为无启用恢复曲线时的常数", icon="INFO")
        header = box.row(align=True);header.label(text="轴向");header.label(text="刚度");header.label(text="阻尼")
        for i, axis in enumerate(("X 扭转", "Y 弯曲", "Z 弯曲")):
            row = box.row(align=True);row.label(text=axis);row.prop(s, "angular_stiffness", index=i, text="");row.prop(s, "angular_damping", index=i, text="")
            row = box.row(align=True);row.prop(s, "soft_limits_enabled", index=i, text="启用柔性限位")
            row.prop(s, "soft_lower", index=i, text="下限（弧度）");row.prop(s, "soft_upper", index=i, text="上限（弧度）")
            box.prop(s, "soft_stiffness", index=i, text="限位刚度（牛·米/弧度）")
        box.label(text="柔性限位允许少量越界，不会瞬间截断运动", icon='INFO')


    def _draw_drives(self, layout, s):
        for axis,label in (('twist_x','X 扭转'),('swing_y','Y 弯曲'),('swing_z','Z 弯曲')):
            layout.label(text=label)
            settings_axis=getattr(s,'drive_'+axis)
            for mode,label,gain,unit in (('position','目标角度驱动','增益（牛·米/弧度）','rad'),('velocity','角速度驱动','增益（牛·米·秒/弧度）','rad/s'),('torque','力矩驱动','增益','Nm')):
                channel=getattr(settings_axis,mode)
                layout.prop(channel,'enabled',text=label)
                if not channel.enabled:continue
                row=layout.row(align=True);row.prop(channel,'gain',text=gain);row.prop(channel,'max_effort')
                for index,key in enumerate(channel.targets):
                    row=layout.row(align=True);row.prop(key,'frame');row.prop(key,'value',text=unit)
                    op=row.operator('ppf.bone_drive_target_edit',text='',icon='REMOVE');op.action='REMOVE';op.axis=axis;op.channel=mode;op.index=index
                op=layout.operator('ppf.bone_drive_target_edit',text='添加目标关键帧',icon='ADD');op.action='ADD';op.axis=axis;op.channel=mode


class PPF_PT_ObjectCollisionGroups(bpy.types.Panel):
    @classmethod
    def poll(cls, context):
        # Object collision data remains available through per-dynamics-group tables.
        return False

    bl_idname = 'PPF_PT_ObjectCollisionGroups'
    bl_label = '物体碰撞组'
    bl_order = 12
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = get_category_name()
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def content_poll(cls, context):
        return bool(context.mode == 'OBJECT' and context.object and context.object.type == 'MESH' and has_addon_data(context.scene))

    def draw(self, context):
        layout = self.layout
        obj = context.object
        if obj.get('ppf_bone_proxy_owned'):
            layout.label(text='骨骼代理使用其绑定骨骼的碰撞设置', icon='BONE_DATA')
            layout.operator('ppf.bone_proxy_return', text='返回绑定骨骼', icon='BONE_DATA')
            return
        uuid = obj.get('_solver_uuid', '')
        groups = [group for group in iterate_active_object_groups(context.scene)
                  if any(a.uuid == uuid and a.included for a in group.assigned_objects)] if uuid else []
        if not groups:
            layout.label(text='请先将此网格加入已启用的 PPF 物理组', icon='INFO')
            return
        _draw_collision_filter(layout, obj.ppf_collision_filter,context)
        layout.label(text='双方所属组必须被对方允许，才参与碰撞', icon='INFO')
        layout.label(text='此处用于网格；解析墙和球使用原生设置', icon='INFO')


classes = (PPF_PT_BonePhysics, PPF_PT_ObjectCollisionGroups)


def register():
    from . import bone_overlay
    from . import bone_display_context
    bone_display_context.register()
    bone_overlay.register()
    for cls in classes:_ppf_i18n.register_class(cls)


def unregister():
    for cls in reversed(classes):bpy.utils.unregister_class(cls)
    from . import bone_overlay
    bone_overlay.unregister()
    from . import bone_display_context
    bone_display_context.unregister()
