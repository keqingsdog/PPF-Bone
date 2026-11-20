"""Inspect and edit one imported PMX joint on the active bone."""
import json
import math
import struct

import bpy

from ..i18n.props import EnumProperty, FloatProperty, StringProperty


_STORAGE = 'ppf_mmd_joints_v1'
_KINDS = (
    ('translation', '平移（米）', 'linear_spring_N_per_m', '弹簧刚度（N/m）'),
    ('rotation', '旋转（弧度）', 'angular_spring_Nm_per_rad', '弹簧刚度（N·m/rad）'),
)
_MODES = (
    ('free', '自由', ''),
    ('locked', '锁定', ''),
    ('limited', '限位', ''),
)
_F32 = struct.Struct('=f')
_PMX_ROTATION_BOUND = _F32.unpack(_F32.pack(math.pi))[0]


def _read_rows(arm):
    rows = json.loads(arm.get(_STORAGE, '[]'))
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError('PMX 关节记录无效')
    return rows


def _joint_label(arm, row, a, b):
    endpoints = f'{a.name if a else "未找到骨骼"} ↔ {b.name if b else "未找到骨骼"}'
    source = arm.get('ppf_mmd_joint_sources_v1', {}).get(row['joint_uuid'])
    return f'{source.name} · {endpoints}' if source else endpoints


def draw_joints(layout, arm, bone):
    """Return whether the active bone has any incident native joints."""
    from ..core.bone_selection_parameters import hierarchy_joint_enabled

    body = bone.ppf_bone_physics.body_uuid
    if not body:
        return False
    try:
        rows = [row for row in _read_rows(arm)
                if body in (row.get('body_a_uuid'), row.get('body_b_uuid'))]
        if not rows:
            return False
        by_body = {b.ppf_bone_physics.body_uuid: b for b in arm.data.bones}
        for row in rows:
            a = by_body.get(row['body_a_uuid'])
            b = by_body.get(row['body_b_uuid'])
            child = None
            if a and b:
                child = b if b.parent == a else a if a.parent == b else None
            kind = 'PMX 父子关节' if child else 'PMX 跨骨关节'
            if child:
                kind += ' · 已连接' if hierarchy_joint_enabled(arm, child) else ' · 已关闭'
            box = layout.box()
            box.label(text=kind, icon='CONSTRAINT_BONE')
            box.label(text=_joint_label(arm, row, a, b), translate=False)
            box.operator_context = 'INVOKE_DEFAULT'
            op = box.operator(PPF_OT_BonePMXJointEdit.bl_idname, text='编辑 PMX 轴与弹簧')
            op.joint_uuid = row['joint_uuid']
        return True
    except (KeyError, TypeError, ValueError) as exc:
        layout.label(text=f'PMX 关节记录无效 · {exc}', icon='ERROR', translate=False)
        return False


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
    body = bone.ppf_bone_physics.body_uuid
    if not body or body not in (row.get('body_a_uuid'), row.get('body_b_uuid')):
        raise ValueError('活动骨骼未连接到该 PMX 关节')
    return arm, rows, index


def _parameters(draft, row):
    updated = dict(row)
    initial = draft._initial_values

    def number(key, saved):
        value = getattr(draft, key)
        # Preserve the source's precision when its displayed RNA value is unchanged.
        return saved if saved is not None and value == initial[key] else value

    for kind, kind_label, spring_key, _ in _KINDS:
        axes, springs = {}, []
        for index, axis in enumerate('xyz'):
            prefix = f'{kind}_{axis}'
            mode = getattr(draft, prefix + '_mode')
            values = {'mode': mode}
            old_axis = row[kind][axis]
            fields = {'free': (), 'locked': ('value',), 'limited': ('lower', 'upper')}[mode]
            for key in fields:
                values[key] = number(prefix + '_' + key, old_axis.get(key))
            spring = number(prefix + '_spring', row[spring_key][index])
            coordinates = [values[key] for key in fields]
            if not all(math.isfinite(value) for value in (*coordinates, spring)):
                raise ValueError(f'{kind_label} {axis.upper()} · 请输入有限数值')
            if kind == 'rotation' and any(abs(value) > _PMX_ROTATION_BOUND for value in coordinates):
                raise ValueError(f'{kind_label} {axis.upper()} · 必须在 −π 至 π 之间')
            try:
                _F32.pack(spring)
                bounds = [_F32.unpack(_F32.pack(value))[0] for value in coordinates]
            except OverflowError as exc:
                raise ValueError(f'{kind_label} {axis.upper()} · 数值超出求解器范围') from exc
            if mode == 'limited' and (values['lower'] >= values['upper'] or bounds[0] >= bounds[1]):
                raise ValueError(f'{kind_label} {axis.upper()} · 下限必须小于上限')
            if spring < 0:
                raise ValueError(f'{kind_label} {axis.upper()} · 弹簧刚度不能为负值')
            axes[axis] = values
            springs.append(spring)
        updated[kind] = axes
        updated[spring_key] = springs
    return updated


class PPF_OT_BonePMXJointEdit(bpy.types.Operator):
    bl_idname = 'ppf.bone_pmx_joint_edit'
    bl_label = '编辑 PMX 关节'
    bl_options = {'UNDO'}

    joint_uuid: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def invoke(self, context, event):
        try:
            arm, rows, index = _edit_target(context, self.joint_uuid)
            row = rows[index]
            self._target_arm = arm
            self._target_row = row
            self._initial_values = {}
            by_body = {b.ppf_bone_physics.body_uuid: b for b in arm.data.bones}
            self._label = _joint_label(arm, row, by_body.get(row['body_a_uuid']), by_body.get(row['body_b_uuid']))
            for kind, _, spring_key, _ in _KINDS:
                for axis_index, axis in enumerate('xyz'):
                    prefix = f'{kind}_{axis}'
                    data = row[kind][axis]
                    setattr(self, prefix + '_mode', data['mode'])
                    for field, value in (('value', data.get('value', 0.)),
                                         ('lower', data.get('lower', -.1)),
                                         ('upper', data.get('upper', .1)),
                                         ('spring', row[spring_key][axis_index])):
                        key = prefix + '_' + field
                        setattr(self, key, value)
                        self._initial_values[key] = getattr(self, key)
            return context.window_manager.invoke_props_dialog(self, width=560)
        except (IndexError, KeyError, ReferenceError, RuntimeError, TypeError, ValueError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

    def draw(self, context):
        self.layout.label(text=self._label, translate=False)
        for kind, title, _, spring_title in _KINDS:
            box = self.layout.box()
            box.label(text=title)
            for axis in 'xyz':
                prefix = f'{kind}_{axis}'
                mode = getattr(self, prefix + '_mode')
                row = box.row(align=True)
                row.prop(self, prefix + '_mode', text=axis.upper())
                if mode == 'locked':
                    row.prop(self, prefix + '_value', text='锁定值')
                elif mode == 'limited':
                    row.prop(self, prefix + '_lower', text='下限')
                    row.prop(self, prefix + '_upper', text='上限')
                box.prop(self, prefix + '_spring', text=f'{axis.upper()} {spring_title}')

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


for _kind, _, _, _ in _KINDS:
    for _axis in 'xyz':
        _prefix = f'{_kind}_{_axis}'
        PPF_OT_BonePMXJointEdit.__annotations__[_prefix + '_mode'] = EnumProperty(
            items=_MODES, options={'SKIP_SAVE'})
        for _field in ('value', 'lower', 'upper', 'spring'):
            PPF_OT_BonePMXJointEdit.__annotations__[_prefix + '_' + _field] = FloatProperty(
                precision=6, options={'SKIP_SAVE'})
