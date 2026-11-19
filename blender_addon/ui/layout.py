"""Optional sidebar workbench; existing panel drawers remain the content source."""
from .. import i18n as _ppf_i18n
import inspect

import bpy
from ..i18n.props import BoolProperty, EnumProperty, StringProperty

from .. import i18n
from ..core.utils import get_category_name

ADDON_KEY = __package__.rsplit('.', 1)[0]
MIN_COLUMN_WIDTH = 680.0


def preferences(context):
    addon = context.preferences.addons.get(ADDON_KEY)
    return addon.preferences if addon is not None else None


def _redraw(_self, context):
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _native_solver_path_changed(self, context):
    from ..core import user_defaults
    user_defaults.preference_changed(self, context)


class PPF_LayoutPreferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_KEY

    ui_language: i18n.language_property()

    native_solver_path: StringProperty(
        name='本机求解器目录', subtype='DIR_PATH', default='',
        description='本机求解器根目录，供场景中的本机连接使用',
        update=_native_solver_path_changed)

    layout_mode: EnumProperty(name='界面布局', items=(
        ('CLASSIC', '经典', '保留原有纵向面板'),
        ('TWO_COLUMN', '双栏', '宽侧栏并排显示模块；窄侧栏自动单列')),
        default='CLASSIC', update=_redraw)
    show_connection: BoolProperty(name='连接', default=True)
    show_solver: BoolProperty(name='求解', default=True)
    show_scene: BoolProperty(name='场景', default=False)
    show_groups: BoolProperty(name='动力学组', default=True)
    show_bones: BoolProperty(name='骨骼物理', default=True)
    show_cross_links: BoolProperty(name='跨链弹簧', default=False)
    show_object_collision: BoolProperty(name='物体碰撞组', default=False)
    show_snap: BoolProperty(name='吸附与合并', default=False)
    show_tools: BoolProperty(name='工具', default=False)
    show_visualization: BoolProperty(name='显示', default=False)
    show_statistics: BoolProperty(name='物体统计', default=False)

    def draw(self, _context):
        i18n.draw_language(self.layout, self)
        self.layout.prop(self, 'native_solver_path')
        self.layout.prop(self, 'layout_mode', expand=True)


def classic_visible(context):
    prefs = preferences(context)
    return prefs is None or prefs.layout_mode == 'CLASSIC'


def logical_width(context):
    region = context.region
    scale = max(float(context.preferences.system.ui_scale), .1)
    return float(region.width) / scale if region is not None else 0.0


# Native inline panels assume a full-width panel root. Workbench drawers use
# columns, so give their nested sections explicit, transient UI-only folds.
_section_open = {}


class PPF_OT_WorkbenchSection(bpy.types.Operator):
    bl_idname = 'ppf.workbench_section'
    bl_label = '展开 / 折叠工作台分区'
    bl_options = {'INTERNAL'}
    section: StringProperty(options={'HIDDEN'})
    opened: BoolProperty(options={'HIDDEN'})

    def execute(self, context):
        _section_open[self.section] = self.opened
        _redraw(self, context)
        return {'FINISHED'}


def draw_inline_section(layout, idname, title='', *, default_closed=False):
    """UI-only fold that stays inside its caller's box, including classic UI."""
    opened = _section_open.get(idname, not default_closed)
    section = layout.column(align=True)
    header = section.row(align=True)
    op = header.operator(PPF_OT_WorkbenchSection.bl_idname, text=title,
                         icon='TRIA_DOWN' if opened else 'TRIA_RIGHT',
                         emboss=False)
    op.section = idname
    op.opened = not opened
    return header, section.column() if opened else None


class _WorkbenchLayout:
    """Forward RNA UI layout operations, keeping nested folds in their column."""
    def __init__(self, layout):
        object.__setattr__(self, '_layout', layout)

    def __setattr__(self, name, value):
        setattr(self._layout, name, value)

    def __getattr__(self, name):
        member = getattr(self._layout, name)
        if name in {'row', 'column', 'box', 'split', 'grid_flow'}:
            return lambda *args, **kwargs: _WorkbenchLayout(member(*args, **kwargs))
        return member

    def panel(self, idname, *, default_closed=False):
        header, body = draw_inline_section(self._layout, idname, default_closed=default_closed)
        return _WorkbenchLayout(header), _WorkbenchLayout(body) if body is not None else None


class _LegacyDrawer:
    """Bind only the chosen panel's Python descriptors to a supplied layout.

    In particular Bone's ordinary helper methods and static methods keep their
    original binding. No registered Panel instance or global class is patched.
    """
    def __init__(self, panel_type, layout):
        self._panel_type = panel_type
        self.layout = layout

    def __getattr__(self, name):
        descriptor = inspect.getattr_static(self._panel_type, name)
        binder = getattr(descriptor, '__get__', None)
        return binder(self, self._panel_type) if binder is not None else descriptor


def _modules():
    # Lazy imports avoid a registration/import cycle with the classic roots.
    from .main_panel import MAIN_PT_RemotePanel
    from .solver import SOLVER_PT_SolverPanel
    from .bone_physics import PPF_PT_BonePhysics
    from .dynamics.panels import (MAIN_PT_SceneConfiguration, DYNAMICS_PT_Groups,
        SNAPMERGE_PT_SnapAndMerge, UTILITY_PT_UtilityTools, VISUALIZATION_PT_Visualization)
    from .dynamics.statistics import STATISTICS_PT_Statistics
    from ..ops.cross_links import PPF_PT_CrossLinks
    return (
        (0, 'show_connection', MAIN_PT_RemotePanel, None),
        (0, 'show_solver', SOLVER_PT_SolverPanel, None),
        (0, 'show_scene', MAIN_PT_SceneConfiguration, None),
        (0, 'show_snap', SNAPMERGE_PT_SnapAndMerge, None),
        (0, 'show_tools', UTILITY_PT_UtilityTools, None),
        (0, 'show_groups', DYNAMICS_PT_Groups, None),
        (1, 'show_bones', PPF_PT_BonePhysics, None),
        (1, 'show_cross_links', PPF_PT_CrossLinks, PPF_PT_BonePhysics),
        (1, 'show_visualization', VISUALIZATION_PT_Visualization, None),
        (1, 'show_statistics', STATISTICS_PT_Statistics, None),
    )


def content_available(panel_type, context):
    poll = getattr(panel_type, 'content_poll', None)
    if poll is None:
        poll = getattr(panel_type, 'poll', None)
    return bool(poll(context)) if poll is not None else True


def draw_content(layout, context):
    prefs = preferences(context)
    if prefs is None:
        layout.label(text='请先启用插件以保存界面偏好', icon='INFO')
        return
    wide = logical_width(context) >= MIN_COLUMN_WIDTH
    if wide:
        split = layout.split(factor=0.5)
        columns = (split.column(), split.column())
    else:
        layout.label(text='拉宽侧栏可并排显示；当前为单列', icon='INFO')
        columns = (layout.column(),) * 2
    for side, key, panel_type, parent in _modules():
        if not content_available(panel_type, context):
            continue
        if parent is not None and not content_available(parent, context):
            continue
        column = columns[side]
        module = column.box()
        header = module.row(align=True)
        header.prop(prefs, key, text=panel_type.bl_label,
                    icon='TRIA_DOWN' if getattr(prefs, key) else 'TRIA_RIGHT',
                    emboss=False)
        if getattr(prefs, key):
            panel_type.draw(_LegacyDrawer(panel_type, _WorkbenchLayout(module.column())), context)
        column.separator()


class PPF_PT_LayoutSelector(bpy.types.Panel):
    bl_idname = 'PPF_PT_LayoutSelector'
    bl_label = '界面布局'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = get_category_name()
    bl_order = -10000
    bl_options = set()

    def draw(self, context):
        prefs = preferences(context)
        if prefs is None:
            self.layout.label(text='经典布局', icon='INFO')
        else:
            self.layout.prop(prefs, 'layout_mode', expand=True)


class PPF_PT_TwoColumnWorkbench(bpy.types.Panel):
    bl_idname = 'PPF_PT_TwoColumnWorkbench'
    bl_label = '工作台'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = get_category_name()
    bl_order = -9999
    bl_options = set()

    @classmethod
    def poll(cls, context):
        return not classic_visible(context)

    def draw(self, context):
        draw_content(self.layout, context)


classes = (PPF_LayoutPreferences, PPF_OT_WorkbenchSection, PPF_PT_LayoutSelector, PPF_PT_TwoColumnWorkbench)


def register():
    for cls in classes:
        _ppf_i18n.register_class(cls)
    prefs = preferences(bpy.context)
    if prefs is not None:
        i18n.sync_preferences(prefs)


def unregister():
    _section_open.clear()
    for cls in reversed(classes):
        if cls.is_registered:
            bpy.utils.unregister_class(cls)
