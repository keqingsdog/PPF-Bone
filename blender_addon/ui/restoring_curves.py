"""Compact author/editor surface; actual native support comes from Build gate."""
from .. import i18n as _ppf_i18n
import bpy
from ..models.restoring_curves import preview_table, validate_table, target_key
from ..ops.restoring_curve_operators import editor_table,draft_matches,draft_label,live_target,read_state
from ..i18n import pgettext_iface as iface
from . import restoring_curve_preview

class PPF_UL_RestoringNodes(bpy.types.UIList):
    def draw_item(self,context,layout,data,item,icon,active_data,active_property,index):
        row=layout.row(align=True)
        row.prop(item,'coordinate',text='x');row.prop(item,'effort',text='f')

def current_tier(owner):
    """Read the stored tier; draw callbacks must not migrate or write IDs."""
    try:
        is_set=owner.is_property_set('ui_tier')
    except (AttributeError, TypeError):
        is_set=True
    if not is_set:
        return 'PRO' if bool(owner.advanced_ui) else 'SIMPLE'
    return owner.ui_tier

def draw_mode(layout,context,*,stitch=False):
    owner=context.scene.ppf_restoring_curves
    current_tier(owner)
    layout.prop(owner,'ui_tier',text='参数显示')

def draw_editor(layout,context,*,stitch=False):
    owner=context.scene.ppf_restoring_curves
    tier=current_tier(owner)
    if tier!='PRO':return
    advanced=True
    scope='STITCH' if stitch else 'BONE'
    header,body=layout.panel('ppf_restoring_curves_'+('stitch' if stitch else 'bone'),default_closed=advanced)
    header.label(text='自定义恢复力曲线' if advanced else '恢复力',icon='FCURVE')
    if not body:return
    if stitch:draw_mode(body,context,stitch=True)
    try:
        target=live_target(context,scope=scope)
    except ValueError as exc:
        body.label(text=iface(str(exc)),icon='INFO')
        return
    if owner.mode=='LEGACY':body.label(text='现有恢复参数保持不变；载入并应用后才启用新设置',icon='INFO')
    elif not advanced:
        state=read_state(context.scene)
        applied=next((b for b in (state['document'] or {}).get('bindings',[]) if target_key(b['target'])==target_key(target)),None)
        if applied and applied['mode']=='AUTO_FREQUENCY':
            body.label(text=iface('Current: {frequency:g} Hz / damping ratio {damping:g}').format(frequency=applied['frequency_hz'],damping=applied['damping_ratio']),translate=False)
        elif applied:body.label(text='当前通道使用自定义单位；打开高级参数编辑',icon='INFO')
    if stitch:
        body.label(text='当前接缝：独立伸长曲线（不等同骨骼锚点）')
        body.operator('ppf.restoring_edit_begin',text='载入当前接缝曲线').scope=scope
    else:
        body.prop(owner,'joint_channel',text='独立通道')
        body.operator('ppf.restoring_edit_begin',text='编辑当前通道参数').scope=scope
    if not owner.editor_target_json:
        body.label(text='先载入目标，再编辑并应用',icon='INFO');return
    body.label(text=draft_label(context),icon='FCURVE')
    if not draft_matches(context,scope):
        body.label(text=iface('Draft belongs to another target; load the current curve'),icon='INFO')
        row=body.row();row.enabled=False
        row.operator('ppf.restoring_edit_apply',text='应用当前曲线').scope=scope
        return
    body.prop(owner,'enabled',text='启用此恢复曲线')
    if not advanced and owner.policy!='AUTO_FREQUENCY':
        body.label(text='当前通道使用自定义单位；打开高级参数编辑',icon='INFO')
        return
    if advanced:body.prop(owner,'policy',text='单位 / 质量依据')
    if owner.policy=='AUTO_FREQUENCY':
        row=body.row(align=True);row.prop(owner,'frequency_hz',text='频率 Hz');row.prop(owner,'damping_ratio',text='阻尼比 ζ')
        if advanced:body.prop(owner,'coordinate_scale',text='坐标尺度（角 rad / 位移 m）')
        body.label(text='按当前质量与尺寸自动计算恢复力',icon='INFO')
        body.label(text='软体使用局部质量估计，响应受材质与网格影响')
        if advanced:
            body.label(text='此频率不代表软体整体模态频率')
            body.label(text='锚点采用平均方向；非线性响应随变形幅度变化')
    else:
        angular=owner.channel not in ('anchor','stitch')
        body.prop(owner,'damping',text='角阻尼 N·m·s/rad' if angular else '位移阻尼 N·s/m')
        body.label(text='x 为 rad、f 为 N·m' if angular else 'x 为 m、f 为 N')
    if advanced:
        row=body.row(align=True);row.prop(owner,'domain',text='定义域');row.prop(owner,'tail',text='尾端')
        body.template_list('PPF_UL_RestoringNodes','',owner,'nodes',owner,'node_index',rows=4)
        row=body.row(align=True)
        op=row.operator('ppf.restoring_node',text='加点',icon='ADD');op.action='ADD';op.scope=scope
        op=row.operator('ppf.restoring_node',text='删点',icon='REMOVE');op.action='REMOVE';op.scope=scope
        body.prop(owner,'preview_coordinate',text='预览坐标 x')
        try:
            table=editor_table(owner)
            icon=restoring_curve_preview.preview_icon(table,validate_table,preview_table)
            body.template_icon(icon_value=icon,scale=6)
            if owner.policy=='AUTO_FREQUENCY':
                body.label(text='横轴：归一化坐标；纵轴：归一化梯度 f')
            elif owner.channel in ('anchor','stitch'):
                body.label(text='横轴：位移 m；纵轴：梯度 f（N）')
            else:
                body.label(text='横轴：角度 rad；纵轴：梯度 f（N·m）')
            points=table['nodes']
            body.label(text=f'节点坐标：{points[0][0]:.4g} ~ {points[-1][0]:.4g}')
            sample=preview_table(table,owner.preview_coordinate)
            body.label(text=f'E={sample["energy"]:.6g}  恢复力/矩={-sample["gradient"]:.6g}')
            body.label(text=f'局部斜率={sample["curvature"]:.6g}（表单位）')
        except ValueError as exc:body.label(text=str(exc),icon='ERROR')
        except OSError:body.label(text='曲线图暂不可用；可继续使用节点和数值预览',icon='ERROR')
        body.label(text='f 是正能量梯度；恢复方向为 -f')
    row=body.row(align=True)
    row.operator('ppf.restoring_edit_apply',text='应用当前曲线').scope=scope
    row.operator('ppf.restoring_reset',text='恢复线性常用默认').scope=scope
    body.label(text='应用后请重新传输、构建与运行',icon='INFO')
    if not advanced:return
    header,preset=body.panel('ppf_restoring_preset_'+('stitch' if stitch else 'bone'),default_closed=True)
    header.label(text='当前通道预设')
    if preset:
        preset.prop(owner,'preset_name',text='名称');preset.prop(owner,'preset_path',text='文件')
        row=preset.row(align=True)
        op=row.operator('ppf.restoring_preset',text='载入 / 选择');op.action='LOAD';op.scope=scope
        op=row.operator('ppf.restoring_preset',text='保存');op.action='SAVE';op.scope=scope
        row=preset.row();row.enabled=bool(owner.preset_json)
        op=row.operator('ppf.restoring_preset',text='应用所选预设到当前通道');op.action='APPLY';op.scope=scope
        preset.label(text='载入只选择；应用才改变当前目标',icon='INFO')
    if not stitch:
        header,chain=body.panel('ppf_restoring_chain_preset',default_closed=True)
        header.label(text='骨骼关节组合预设',icon='CONSTRAINT_BONE')
        if chain:
            row=chain.row(align=True)
            row.prop(owner,'preset_name',text='组合名')
            row.prop(owner,'preset_path',text='文件')
            chain.label(text='未包含通道保留目标设置')
            row=chain.row(align=True)
            row.prop(owner,'bone_chain_scope',text='范围')
            op=row.operator('ppf.restoring_chain_preset',text='保存');op.action='SAVE'
            op=row.operator('ppf.restoring_chain_preset',text='载入');op.action='LOAD'
            op=row.operator('ppf.restoring_chain_preset',text='应用');op.action='APPLY'
            chain.label(text='载入只选择；应用按当前范围重映射到关节 UUID',icon='INFO')

def register():_ppf_i18n.register_class(PPF_UL_RestoringNodes)
def unregister():
    restoring_curve_preview.clear()
    bpy.utils.unregister_class(PPF_UL_RestoringNodes)
