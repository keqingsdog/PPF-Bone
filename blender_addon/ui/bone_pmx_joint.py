"""Edit the spring stiffness of an imported PMX parent joint."""
import json
import math
import struct

import bpy

from ..core.encoder.pmx_joints import parent_joint_pair, parent_joint_pairs
from ..i18n.props import FloatVectorProperty, StringProperty


_STORAGE = 'ppf_mmd_joints_v1'
_SPRINGS = (
    ('linear', 'linear_spring_N_per_m', '平移刚度（N/m）'),
    ('angular', 'angular_spring_Nm_per_rad', '旋转刚度（N·m/rad）'),
)
_F32 = struct.Struct('=f')


def _read_rows(arm):
    rows = json.loads(arm.get(_STORAGE, '[]'))
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError('PMX 关节记录无效')
    return rows


def _joint_label(arm, row, a, b):
    endpoints = f'{a.name if a else "未找到骨骼"} ↔ {b.name if b else "未找到骨骼"}'
    source = arm.get('ppf_mmd_joint_sources_v1', {}).get(row['joint_uuid'])
    return f'{source.name} · {endpoints}' if source else endpoints


def _pair(row):
    return frozenset((row.get('body_a_uuid'), row.get('body_b_uuid')))


def draw_parent_joint(layout, arm, pair):
    """Show controls for the selected bone's connection to its own parent."""
    try:
        rows = [row for row in _read_rows(arm) if _pair(row) == pair]
        sources = arm.get('ppf_mmd_joint_sources_v1', {})
        for index, row in enumerate(rows, start=1):
            controls = layout.row()
            controls.operator_context = 'INVOKE_DEFAULT'
            label = '编辑 PMX 连接刚度'
            if len(rows) > 1:
                source = sources.get(row['joint_uuid'])
                label += f' · {source.name if source else index}'
            edit = controls.operator(PPF_OT_BonePMXJointEdit.bl_idname, text=label)
            edit.joint_uuid = row['joint_uuid']
    except (KeyError, TypeError, ValueError) as exc:
        layout.label(text=f'PMX 关节记录无效 · {exc}', icon='ERROR', translate=False)


def _edit_target(context, joint_uuid):
    from ..core.bone_selection_parameters import active_author, resolve_selected_authors
    from ..core.facade import communicator
    from ..models.bone_bindings import validate_proxy_owner
    from ..ops.bone_proxy import stale_simulation_output

    communicator.assert_bone_body_switch_idle()
    if stale_simulation_output(context):
        raise ValueError('请先清除已有模拟帧，再编辑 PMX 关节')
    author = active_author(context)
    if author is None or author not in resolve_selected_authors(context):
        raise ValueError('请选中关节所属的骨骼或代理')
    arm, bone = author
    if arm.library or arm.data.library:
        raise ValueError('请使用本地骨架编辑 PMX 关节')
    validate_proxy_owner(arm, bone)
    rows = _read_rows(arm)
    matches = [index for index, row in enumerate(rows) if row.get('joint_uuid') == joint_uuid]
    if not joint_uuid or len(matches) != 1:
        raise ValueError('PMX 关节不存在或身份不唯一')
    index = matches[0]
    row = rows[index]
    if parent_joint_pair(bone, parent_joint_pairs(arm)) != _pair(row):
        raise ValueError('请选择该 PMX 父关节的子骨骼')
    return arm, rows, index


def _parameters(draft, row):
    updated = dict(row)
    for field, spring_key, label in _SPRINGS:
        springs = []
        for index, value in enumerate(getattr(draft, field)):
            # Preserve the source's precision when its displayed RNA value is unchanged.
            spring = row[spring_key][index] if value == draft._initial_values[field][index] else value
            axis = 'XYZ'[index]
            if not math.isfinite(spring) or spring < 0:
                raise ValueError(f'{label} {axis} · 请输入有限的非负值')
            try:
                _F32.pack(spring)
            except OverflowError as exc:
                raise ValueError(f'{label} {axis} · 数值超出求解器范围') from exc
            springs.append(spring)
        updated[spring_key] = springs
    return updated


class PPF_OT_BonePMXJointEdit(bpy.types.Operator):
    bl_idname = 'ppf.bone_pmx_joint_edit'
    bl_label = '编辑 PMX 连接刚度'
    bl_options = {'UNDO'}

    joint_uuid: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    linear: FloatVectorProperty(size=3, precision=6, options={'SKIP_SAVE'})
    angular: FloatVectorProperty(size=3, precision=6, options={'SKIP_SAVE'})

    def invoke(self, context, event):
        try:
            arm, rows, index = _edit_target(context, self.joint_uuid)
            row = rows[index]
            self._target_arm = arm
            self._target_row = row
            self._initial_values = {}
            by_body = {b.ppf_bone_physics.body_uuid: b for b in arm.data.bones}
            self._label = _joint_label(arm, row, by_body.get(row['body_a_uuid']), by_body.get(row['body_b_uuid']))
            for field, spring_key, _ in _SPRINGS:
                setattr(self, field, row[spring_key])
                self._initial_values[field] = tuple(getattr(self, field))
            return context.window_manager.invoke_props_dialog(self, width=400)
        except (IndexError, KeyError, ReferenceError, RuntimeError, TypeError, ValueError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

    def draw(self, context):
        self.layout.label(text=self._label, translate=False)
        for field, _, label in _SPRINGS:
            self.layout.label(text=label)
            row = self.layout.row(align=True)
            for index, axis in enumerate('XYZ'):
                row.prop(self, field, index=index, text=axis)

    def execute(self, context):
        from ..core.facade import communicator

        try:
            arm, rows, index = _edit_target(context, self.joint_uuid)
            row = rows[index]
            if arm != getattr(self, '_target_arm', None) or row != self._target_row:
                raise ValueError('PMX 关节已变化，请重新打开编辑窗口')
            updated = _parameters(self, row)
            if updated == row:
                return {'CANCELLED'}
            rows[index] = updated
            previous = arm[_STORAGE]
            arm[_STORAGE] = json.dumps(rows, ensure_ascii=False, allow_nan=False)
            try:
                communicator.bone_physical_body_changed()
            except Exception:
                arm[_STORAGE] = previous
                raise
            return {'FINISHED'}
        except (KeyError, ReferenceError, RuntimeError, TypeError, ValueError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
