"""Explicit user-library IO and category-masked spring setting operators."""
import bpy
from .. import i18n as _ppf_i18n
from ..i18n.props import BoolProperty, CollectionProperty, EnumProperty, IntProperty, PointerProperty, StringProperty
from ..core import cross_link_preset_library as library
from ..models.bone_bindings import active_armature


def _state(context):
    return context.window_manager.ppf_cross_link_preset_library


def _arm(context):
    arm = active_armature(context)
    if arm is None:
        raise ValueError('请选择骨架或它的物理代理')
    return arm


def _selected(state):
    if 0 <= state.active_index < len(state.entries):
        return state.entries[state.active_index]
    raise ValueError('请先刷新并选择一个用户预设')


def _refresh(context, selected_id=''):
    state = _state(context)
    if not selected_id and 0 <= state.active_index < len(state.entries):
        selected_id = state.entries[state.active_index].preset_id
    rows = library.list_presets()
    state.entries.clear()
    state.active_index = 0
    for index, row in enumerate(rows):
        item = state.entries.add()
        item.name, item.preset_id = row['name'], row['id']
        item.valid, item.error = row['valid'], row.get('error', '')
        if item.preset_id == selected_id:
            state.active_index = index
    state.status = f'我的预设 · {len(rows)} 个预设'


def _fail(operator, context, error):
    message = str(error) or type(error).__name__
    _state(context).status = message
    operator.report({'ERROR'}, message)
    return {'CANCELLED'}


def _refresh_after_commit(operator, context, identifier, action):
    """Report display failure without misclassifying a committed file."""
    try:
        _refresh(context, identifier)
    except Exception as error:
        message = action + '已提交；显示刷新失败，请刷新用户预设列表：' + (str(error) or type(error).__name__)
        _state(context).status = message
        operator.report({'WARNING'}, message)
    return {'FINISHED'}


class PPF_CrossLinkPresetEntry(bpy.types.PropertyGroup):
    preset_id: StringProperty()
    valid: BoolProperty(default=True)
    error: StringProperty()


class PPF_CrossLinkPresetLibraryState(bpy.types.PropertyGroup):
    entries: CollectionProperty(type=PPF_CrossLinkPresetEntry)
    active_index: IntProperty(min=0)
    preset_name: StringProperty(name='名称', default='我的链间弹簧预设', maxlen=80)
    source: EnumProperty(name='保存来源', items=(('ACTIVE', '当前连接', '读取列表当前连接的实际参数'), ('DRAFT', '创建参数', '读取手动创建参数')))
    target: EnumProperty(name='应用目标', items=(('SELECTED', '选中连接', '当前连接勾选时应用到勾选连接；否则只应用到当前连接'), ('DRAFT', '创建参数', '用于之后创建连接')))
    apply_physics: BoolProperty(name='物理参数', default=True)
    apply_collision: BoolProperty(name='碰撞排除', default=True)
    apply_display: BoolProperty(name='显示颜色', default=True)
    status: StringProperty(default='点击刷新载入跨工程用户预设')


class PPF_UL_CrossLinkUserPresets(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layout.label(text=item.name, icon='PRESET' if item.valid else 'ERROR', translate=False)


class PPF_OT_CrossLinkPresetRefresh(bpy.types.Operator):
    bl_idname = 'ppf.cross_link_preset_refresh'
    bl_label = '刷新用户预设'
    def execute(self, context):
        try:
            _refresh(context)
            return {'FINISHED'}
        except Exception as error:
            return _fail(self, context, error)


class PPF_OT_CrossLinkPresetSave(bpy.types.Operator):
    bl_idname = 'ppf.cross_link_preset_save'
    bl_label = '保存链间弹簧预设'
    def execute(self, context):
        try:
            state = _state(context)
            document = library.capture_preset(context, _arm(context), state.preset_name, state.source)
            path = library.save_preset(state.preset_name, document)
            return _refresh_after_commit(self, context, path.stem, '保存')
        except Exception as error:
            return _fail(self, context, error)


class PPF_OT_CrossLinkPresetManage(bpy.types.Operator):
    bl_idname = 'ppf.cross_link_preset_manage'
    bl_label = '管理链间弹簧预设'
    action: EnumProperty(items=(('OVERWRITE', '覆盖', ''), ('RENAME', '重命名', ''), ('DELETE', '删除', '')))
    target_preset_id: StringProperty(options={'HIDDEN'})
    new_name: StringProperty(name='新名称', maxlen=80)
    def invoke(self, context, event):
        try:
            item = _selected(_state(context))
            self.target_preset_id, self.new_name = item.preset_id, item.name
            if self.action == 'RENAME':
                return context.window_manager.invoke_props_dialog(self)
            return context.window_manager.invoke_confirm(self, event,
                message=f'对“{item.name}”执行文件操作？此操作不能撤销。')
        except Exception as error:
            return _fail(self, context, error)
    def draw(self, context):
        self.layout.prop(self, 'new_name')
    def execute(self, context):
        try:
            state = _state(context)
            identifier = self.target_preset_id or _selected(state).preset_id
            if self.action == 'DELETE':
                library.delete_preset(identifier)
            elif self.action == 'RENAME':
                library.rename_preset(identifier, self.new_name)
                return _refresh_after_commit(self, context, identifier, '重命名')
            else:
                old = library.load_preset(identifier)
                document = library.capture_preset(context, _arm(context), old['name'], state.source)
                library.save_preset(old['name'], document, True, preset_id=identifier)
                return _refresh_after_commit(self, context, identifier, '覆盖')
            _refresh(context, identifier)
            return {'FINISHED'}
        except Exception as error:
            return _fail(self, context, error)


class PPF_OT_CrossLinkPresetApply(bpy.types.Operator):
    bl_idname = 'ppf.cross_link_preset_apply'
    bl_label = '应用链间弹簧预设'
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        try:
            state = _state(context)
            document = library.load_preset(_selected(state).preset_id)
            count = library.apply_preset(context, _arm(context), document, target=state.target,
                mask={group: getattr(state, 'apply_' + group) for group in library.GROUPS})
            state.status = f'已应用到 {count} 项'
            return {'FINISHED'}
        except Exception as error:
            return _fail(self, context, error)


class PPF_MT_CrossLinkPresetManage(bpy.types.Menu):
    bl_label = '管理我的预设'
    def draw(self, context):
        for action, label in (('OVERWRITE', '覆盖'), ('RENAME', '重命名'), ('DELETE', '删除')):
            operator = self.layout.operator('ppf.cross_link_preset_manage', text=label)
            operator.action = action


def draw_presets(layout, context, arm):
    state = _state(context)
    item = state.entries[state.active_index] if 0 <= state.active_index < len(state.entries) else None
    box = layout.box()
    box.label(text='链间弹簧预设 · 我的预设', icon='PRESET')
    row = box.row(align=True)
    row.template_list('PPF_UL_CrossLinkUserPresets', '', state, 'entries', state, 'active_index', rows=2, maxrows=2)
    buttons = row.column(align=True)
    buttons.operator('ppf.cross_link_preset_refresh', text='', icon='FILE_REFRESH')
    buttons.menu('PPF_MT_CrossLinkPresetManage', text='', icon='DOWNARROW_HLT')
    row = box.row(align=True)
    row.prop(state, 'source', text='')
    row.prop(state, 'preset_name', text='')
    row.operator('ppf.cross_link_preset_save', text='保存')
    row = box.row(align=True)
    for group in library.GROUPS:
        row.prop(state, 'apply_' + group)
    row = box.row(align=True)
    row.prop(state, 'target', text='')
    apply = row.row(align=True)
    apply.enabled = item is not None and item.valid
    apply.operator('ppf.cross_link_preset_apply', text='应用预设')
    box.label(text=state.status, icon='INFO', translate=False)
    if item is not None and not item.valid:
        box.label(text=item.error, icon='ERROR', translate=False)


classes = (PPF_CrossLinkPresetEntry, PPF_CrossLinkPresetLibraryState, PPF_UL_CrossLinkUserPresets,
    PPF_OT_CrossLinkPresetRefresh, PPF_OT_CrossLinkPresetSave, PPF_OT_CrossLinkPresetManage,
    PPF_OT_CrossLinkPresetApply, PPF_MT_CrossLinkPresetManage)


def register():
    for cls in classes:
        _ppf_i18n.register_class(cls)
    bpy.types.WindowManager.ppf_cross_link_preset_library = PointerProperty(type=PPF_CrossLinkPresetLibraryState)


def unregister():
    del bpy.types.WindowManager.ppf_cross_link_preset_library
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
