"""User bone physics library operators; disk actions are deliberately not Undo."""
from .. import i18n as _ppf_i18n
import bpy
from ..i18n.props import (BoolProperty, CollectionProperty, EnumProperty, IntProperty,
                       PointerProperty, StringProperty)


def _library():
    from ..core import bone_preset_library
    return bone_preset_library


def _state(context):
    return context.window_manager.ppf_bone_preset_library


def _selected(state):
    if 0 <= state.active_index < len(state.entries):
        return state.entries[state.active_index]
    raise ValueError('请先刷新并选择一个用户预设')


def _target_id(operator, state):
    # INVOKE freezes the confirmed identity. Direct EXEC uses current selection.
    return getattr(operator, 'target_preset_id', '') or _selected(state).preset_id


def _refresh(context, selected_id=''):
    state = _state(context)
    selected_id = selected_id or (state.entries[state.active_index].preset_id
        if 0 <= state.active_index < len(state.entries) else '')
    rows = _library().list_presets()
    state.entries.clear()
    state.active_index = 0
    for index, row in enumerate(rows):
        item = state.entries.add()
        item.name = row['name']
        item.preset_id = row['id']
        item.valid = row.get('valid', True)
        item.error = row.get('error', '')
        if item.preset_id == selected_id:
            state.active_index = index
    state.status = f'用户目录 · {len(rows)} 个预设' if rows else '用户目录暂无预设；可保存当前骨骼'


def _fail(operator, context, exc):
    message = str(exc) or type(exc).__name__
    _state(context).status = message
    operator.report({'ERROR'}, message)
    return {'CANCELLED'}


def _refresh_after_commit(operator, context, identifier, action):
    """A committed file is not cancelled by a later display-only failure."""
    try:
        _refresh(context, identifier)
    except Exception as exc:
        message = action + '已提交；显示刷新失败，请刷新用户预设列表：' + (str(exc) or type(exc).__name__)
        _state(context).status = message
        operator.report({'WARNING'}, message)
        return False
    return True


class PPF_BonePresetEntry(bpy.types.PropertyGroup):
    preset_id: StringProperty()
    valid: BoolProperty(default=True)
    error: StringProperty()


class PPF_BonePresetLibraryState(bpy.types.PropertyGroup):
    source: EnumProperty(name='预设来源', items=(('BUILTIN','内置预设',''),('USER','我的预设','')), default='BUILTIN')
    apply_restoring: BoolProperty(name='关节与恢复力', description='包括根控制、驱动、限位与恢复曲线', default=True)
    apply_material: BoolProperty(name='材质与物理类型', description='用户预设包括摩擦与物理类型；内置预设未定义摩擦，保持当前摩擦与物理类型', default=True)
    apply_mass: BoolProperty(name='质量与密度', description='应用质量模式、目标质量与密度', default=True)
    apply_shape: BoolProperty(name='形状与尺寸', description='应用代理形状与网格', default=True)
    apply_collision: BoolProperty(name='碰撞', description='应用碰撞分组、排除与接触间距；摩擦由材质类别应用', default=True)
    apply_display: BoolProperty(name='显示颜色', description='应用预览颜色', default=True)
    entries: CollectionProperty(type=PPF_BonePresetEntry)
    active_index: IntProperty(default=0, min=0)
    preset_name: StringProperty(name='名称', default='我的骨骼预设', maxlen=80)
    scope: EnumProperty(name='范围', items=(
        ('SELECTED', '选中骨骼', ''), ('CHAIN', '当前骨链', ''), ('ALL', '整个骨架', '')))
    size_mode: EnumProperty(name='尺寸', items=(
        ('FIT_BONE', '按骨长缩放', ''), ('KEEP_SIZE', '保持原尺寸', '')))
    status: StringProperty(default='点击刷新载入跨工程用户预设')


class PPF_UL_BoneUserPresets(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layout.label(text=item.name, icon='PRESET' if item.valid else 'ERROR', translate=False)


class PPF_OT_BonePresetRefresh(bpy.types.Operator):
    bl_idname = 'ppf.bone_user_preset_refresh'
    bl_label = '刷新用户预设'
    bl_description = '从 Blender 用户目录读取跨工程预设列表'

    def execute(self, context):
        try:
            _refresh(context)
            return {'FINISHED'}
        except Exception as exc:
            return _fail(self, context, exc)


class PPF_OT_BonePresetSave(bpy.types.Operator):
    bl_idname = 'ppf.bone_user_preset_save'
    bl_label = '保存用户预设'
    bl_description = '将当前骨骼的完整物理设置及代理网格保存到用户目录；文件保存不参与撤销'

    def execute(self, context):
        try:
            state = _state(context)
            document = _library().capture_preset(context, state.preset_name)
            path = _library().save_preset(state.preset_name, document)
        except Exception as exc:
            return _fail(self, context, exc)
        if _refresh_after_commit(self, context, path.stem, '保存'):
            self.report({'INFO'}, '已保存用户预设；可在其他工程刷新后使用')
        return {'FINISHED'}


class PPF_OT_BonePresetOverwrite(bpy.types.Operator):
    target_preset_id: StringProperty(options={'HIDDEN'})
    bl_idname = 'ppf.bone_user_preset_overwrite'
    bl_label = '覆盖所选用户预设'
    bl_description = '用当前骨骼覆盖列表中所选的用户预设文件；此文件操作不能撤销'

    def invoke(self, context, event):
        try:
            item = _selected(_state(context))
            self.target_preset_id = item.preset_id
            return context.window_manager.invoke_confirm(self, event,
                message=f'覆盖用户预设“{item.name}”？此文件操作不能撤销。')
        except Exception as exc:
            return _fail(self, context, exc)

    def execute(self, context):
        try:
            identifier = _target_id(self, _state(context))
            # Resolve the confirmed file before capturing any scene data.
            old = _library().load_preset(identifier)
            name = old['name']
            document = _library().capture_preset(context, name)
            path = _library().save_preset(name, document, overwrite=True, preset_id=identifier)
        except Exception as exc:
            return _fail(self, context, exc)
        if _refresh_after_commit(self, context, path.stem, '覆盖'):
            self.report({'INFO'}, '已覆盖所选用户预设')
        return {'FINISHED'}


class PPF_OT_BonePresetApply(bpy.types.Operator):
    bl_idname = 'ppf.bone_user_preset_apply'
    bl_label = '应用用户预设'
    bl_description = '将所选用户预设应用到指定范围；场景更改支持撤销'
    bl_options = {'REGISTER', 'UNDO'}
    scope_override: EnumProperty(name='应用范围',items=(('STATE','预设面板范围','保持已有调用语义'),('SELECTED','选中骨骼',''),('CHAIN','当前骨链',''),('ALL','整个骨架','')),default='STATE',options={'HIDDEN'})

    def execute(self, context):
        try:
            state = _state(context)
            mask = {key: getattr(state, 'apply_'+key) for key in ('restoring','material','mass','shape','collision','display')}
            if not any(mask.values()):
                state.status = '未选择应用内容，未做修改'
                self.report({'INFO'}, state.status); return {'CANCELLED'}
            document = _library().load_preset(_selected(state).preset_id)
            scope=state.scope if self.scope_override=='STATE' else self.scope_override
            result = _library().apply_preset(context, document, scope=scope, size_mode=state.size_mode, mask=mask)
            message = f"已应用到 {result['applied']} 根骨骼"
            skipped = result.get('skipped', [])
            if skipped:
                message += f'；跳过 {len(skipped)} 根：' + ', '.join(skipped)
            state.status = message
            self.report({'INFO'}, message)
            return {'FINISHED'}
        except Exception as exc:
            return _fail(self, context, exc)


class PPF_OT_BonePresetRename(bpy.types.Operator):
    target_preset_id: StringProperty(options={'HIDDEN'})
    bl_idname = 'ppf.bone_user_preset_rename'
    bl_label = '重命名用户预设'
    bl_description = '重命名所选用户预设文件；此文件操作不能撤销'
    new_name: StringProperty(name='新名称', maxlen=80)

    def invoke(self, context, event):
        try:
            item = _selected(_state(context))
            self.target_preset_id = item.preset_id
            self.new_name = item.name
            return context.window_manager.invoke_props_dialog(self)
        except Exception as exc:
            return _fail(self, context, exc)

    def execute(self, context):
        try:
            identifier = _target_id(self, _state(context))
            _library().rename_preset(identifier, self.new_name)
        except Exception as exc:
            return _fail(self, context, exc)
        # The atomic file replacement is committed. A display failure must not
        # report cancellation or overwrite a file another process may edit.
        try:
            _refresh(context, identifier)
        except Exception as exc:
            message = '重命名已提交；显示刷新失败，请刷新用户预设列表：' + (str(exc) or type(exc).__name__)
            _state(context).status = message
            self.report({'WARNING'}, message)
        return {'FINISHED'}


class PPF_OT_BonePresetDelete(bpy.types.Operator):
    target_preset_id: StringProperty(options={'HIDDEN'})
    bl_idname = 'ppf.bone_user_preset_delete'
    bl_label = '删除用户预设'
    bl_description = '永久删除所选用户预设文件；此文件操作不能撤销'

    def invoke(self, context, event):
        try:
            item = _selected(_state(context))
            self.target_preset_id = item.preset_id
            return context.window_manager.invoke_confirm(self, event,
                message=f'删除用户预设“{item.name}”？此文件操作不能撤销。')
        except Exception as exc:
            return _fail(self, context, exc)

    def execute(self, context):
        try:
            _library().delete_preset(_target_id(self, _state(context)))
            _refresh(context)
            return {'FINISHED'}
        except Exception as exc:
            return _fail(self, context, exc)


class PPF_MT_BoneUserPresetManage(bpy.types.Menu):
    bl_idname = 'PPF_MT_BoneUserPresetManage'
    bl_label = '管理用户预设'

    def draw(self, context):
        self.layout.operator('ppf.bone_user_preset_rename', icon='GREASEPENCIL')
        self.layout.operator('ppf.bone_user_preset_delete', icon='TRASH')


def draw_user_presets(layout, context, embedded=False, controls=True):
    """Read cached RNA only; directory reads happen in explicit operators."""
    if embedded:
        body = layout
    else:
        header, body = layout.panel('ppf_bone_user_presets', default_closed=True)
        header.label(text='用户预设', icon='PRESET')
        if body is None:
            return
    state = _state(context)
    row = body.row(align=True)
    row.template_list('PPF_UL_BoneUserPresets', '', state, 'entries', state, 'active_index', rows=2, maxrows=2)
    buttons = row.column(align=True)
    buttons.operator('ppf.bone_user_preset_refresh', text='', icon='FILE_REFRESH')
    buttons.menu('PPF_MT_BoneUserPresetManage', text='', icon='DOWNARROW_HLT')
    row = body.row(align=True)
    row.prop(state, 'preset_name', text='')
    row.operator('ppf.bone_user_preset_save', text='保存')
    overwrite = row.row(align=True); overwrite.enabled = bool(state.entries)
    overwrite.operator('ppf.bone_user_preset_overwrite', text='覆盖')
    from .bone_proxy_editor import editor_active,editor_scope
    editing=editor_active(context)
    if controls:
        row = body.row(align=True)
        if not editing:row.prop(state, 'scope', text='')
        row.prop(state, 'size_mode', text='')
        apply = body.row(align=True); apply.enabled = bool(state.entries)
        operator=apply.operator('ppf.bone_user_preset_apply', text='应用用户预设', icon='CHECKMARK')
        operator.scope_override=editor_scope(context) if editing else 'STATE'
    body.label(text=state.status, icon='INFO', translate=False)
    if 0 <= state.active_index < len(state.entries) and not state.entries[state.active_index].valid:
        body.label(text=state.entries[state.active_index].error, icon='ERROR', translate=False)


classes = (PPF_BonePresetEntry, PPF_BonePresetLibraryState, PPF_UL_BoneUserPresets,
           PPF_OT_BonePresetRefresh, PPF_OT_BonePresetSave, PPF_OT_BonePresetOverwrite,
           PPF_OT_BonePresetApply, PPF_OT_BonePresetRename, PPF_OT_BonePresetDelete,
           PPF_MT_BoneUserPresetManage)


def register():
    for cls in classes:
        _ppf_i18n.register_class(cls)
    bpy.types.WindowManager.ppf_bone_preset_library = PointerProperty(type=PPF_BonePresetLibraryState)


def unregister():
    del bpy.types.WindowManager.ppf_bone_preset_library
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


