"""Group-scoped controls for the existing object/bone 16 collision layers."""
from .. import i18n as _ppf_i18n
import bpy
from ..i18n import pgettext_iface as iface_
from ..i18n.props import BoolProperty, EnumProperty, IntProperty, StringProperty

_FIELDS = ('collision_filter_enabled', 'collision_group_layers', 'collision_mask_layers')
_editing = False


def _group(scene, uid):
    from ..models.groups import iterate_object_groups
    rows = [group for group in iterate_object_groups(scene) if group.uuid == uid]
    if not uid or len(rows) != 1 or not rows[0].active:
        raise ValueError('动力学组身份失效或冲突，请重新展开有效组')
    return rows[0]


def _members(context, uid):
    """Resolve current included members without creating/reassigning identities."""
    from ..core.uuid_registry import get_object_by_uuid
    from ..core.bone_selection_parameters import _proxy_owner
    from ..models.bone_bindings import validate_proxy_owner
    group = _group(context.scene, uid)
    entries = []
    seen = set()
    for assigned in group.assigned_objects:
        if not assigned.included:
            continue
        body_uid = assigned.uuid
        if not body_uid or body_uid in seen:
            raise ValueError('组内成员UUID缺失或重复')
        seen.add(body_uid)
        # Even a hot registry owner cannot make a duplicated author ID safe
        # for a group-wide edit. Do not repair duplicate IDs during drawing.
        matches = [obj for obj in bpy.data.objects if obj.get('_solver_uuid') == body_uid]
        if len(matches) != 1:
            raise ValueError(assigned.name + '：对象UUID失效或冲突')
        obj = get_object_by_uuid(body_uid)
        if obj is None or context.scene.objects.get(obj.name) != obj or obj.type != 'MESH':
            raise ValueError(assigned.name + '：参与成员必须是当前场景的有效网格')
        owners = [(arm, bone) for arm in context.scene.objects if arm.type == 'ARMATURE'
                  for bone in arm.data.bones if bone.ppf_bone_physics.proxy == obj]
        if obj.get('ppf_bone_proxy_owned') or owners:
            arm, bone = _proxy_owner(context.scene, obj)
            validate_proxy_owner(arm, bone)
            settings = bone.ppf_bone_physics
            if not settings.enabled or settings.proxy != obj or settings.body_uuid != body_uid:
                raise ValueError(obj.name + '：骨骼代理绑定未参与解算或身份失效')
            entries.append((bone.name, settings, arm))
        else:
            entries.append((obj.name, obj.ppf_collision_filter, obj))
    if not entries:
        raise ValueError('此组没有参与解算的成员')
    return group, entries


def _editable(context, group, entries):
    from .bone_collision_ui import _preflight_editable
    for block in (context.scene, group.id_data):
        if not block.is_editable:
            raise ValueError('当前场景或动力学组只读')
    for name, settings, owner in entries:
        _preflight_editable(name, settings, owner)
        if not owner.is_editable or not settings.id_data.is_editable:
            raise ValueError(name + '：碰撞设置只读')


def _idle():
    from ..core.facade import communicator
    from .dynamics.bake_ops import is_bake_running
    from .dynamics.pin_capture_ops import is_pin_capture_running
    from .dynamics.static_deform_ops import is_capture_running
    communicator.assert_bone_body_switch_idle()
    if is_bake_running() or is_pin_capture_running() or is_capture_running():
        raise ValueError('请先结束当前烘焙或捕获作业，再修改碰撞层')


def read_group_state(context, group_uuid):
    """Read-only tri-state summary: None denotes a mixed field/bit."""
    _group_row, entries = _members(context, group_uuid)
    def common(values):
        values = tuple(values)
        return values[0] if all(value == values[0] for value in values) else None
    return dict(count=len(entries),
                enabled=common(bool(settings.collision_filter_enabled) for _, settings, _ in entries),
                groups=[common(bool(settings.collision_group_layers[i]) for _, settings, _ in entries) for i in range(16)],
                mask=[common(bool(settings.collision_mask_layers[i]) for _, settings, _ in entries) for i in range(16)])


def edit_group_collision(context, group_uuid, field, value, index=-1, *, assign=None):
    """One bit/enable, whole-current-group preflight, rollback before one Clear."""
    global _editing
    if _editing:
        raise ValueError('组碰撞层批量修改正在进行')
    if field not in _FIELDS or type(value) is not bool:
        raise ValueError('碰撞层操作参数无效')
    if field != _FIELDS[0] and (type(index) is not int or not 0 <= index < 16):
        raise ValueError('碰撞层编号应为1至16')
    _idle()
    group, entries = _members(context, group_uuid)
    _editable(context, group, entries)
    from ..core.bone_preset_apply import _validate_rna
    from ..models.bone_collision import collision_edit_batch
    changes = []
    for name, settings, _owner in entries:
        previous = getattr(settings, field)
        previous = bool(previous) if field == _FIELDS[0] else list(previous)
        target = value if field == _FIELDS[0] else list(previous)
        if field != _FIELDS[0]:
            target[index] = value
        _validate_rna(settings, field, target)
        if previous != target:
            changes.append((name, settings, previous, target, settings.is_property_set(field)))
    if not changes:
        return 0
    assign = assign or setattr
    _editing = True
    try:
        with collision_edit_batch():
            try:
                for name, settings, _previous, target, _was_set in changes:
                    assign(settings, field, target)
                    actual = getattr(settings, field)
                    actual = bool(actual) if field == _FIELDS[0] else list(actual)
                    if actual != target:
                        raise ValueError(name + '：碰撞层赋值未生效')
            except Exception as primary:
                failures = []
                for name, settings, previous, _target, was_set in reversed(changes):
                    try:
                        setattr(settings, field, previous)
                        if not was_set:
                            settings.property_unset(field)
                    except Exception as error:
                        failures.append(name + '：' + str(error))
                if failures:
                    raise RuntimeError('碰撞层修改失败：' + str(primary) + '；恢复错误：' + '；'.join(failures)) from primary
                raise
    finally:
        _editing = False
    return len(changes)


class PPF_OT_GroupCollisionEdit(bpy.types.Operator):
    bl_idname = 'ppf.group_collision_edit'
    bl_label = '统一组成员碰撞层'
    bl_description = '应用到本动力学组所有参与成员；只改此开关或此位，其他位保持；混合位点击统一开启'
    bl_options = {'REGISTER', 'UNDO'}
    group_uuid: StringProperty(options={'HIDDEN'})
    field: EnumProperty(items=[(key, key, '') for key in _FIELDS], options={'HIDDEN'})
    index: IntProperty(default=-1, min=-1, max=15, options={'HIDDEN'})
    value: BoolProperty(default=True, options={'HIDDEN'})

    def execute(self, context):
        try:
            count = edit_group_collision(context, self.group_uuid, self.field, self.value, self.index)
        except (ValueError, RuntimeError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        self.report({'INFO'}, iface_('已修改 {count} 个组成员；请重新传输和构建').format(count=count))
        return {'FINISHED'}


def draw_group_collision(layout, context, group, actual_index):
    """No author RNA writes; actual_index is display context, never identity."""
    from .layout import draw_inline_section
    _header, body = draw_inline_section(layout.box(), 'ppf_group_collision_' + group.uuid,
                                        '碰撞层', default_closed=True)
    if body is None:
        return
    try:
        state = read_group_state(context, group.uuid)
        row, entries = _members(context, group.uuid)
        _editable(context, row, entries)
        _idle()
    except (ValueError, RuntimeError) as error:
        body.label(text=str(error), icon='ERROR')
        return
    body.label(text=iface_('本组全部参与成员：{count} 项').format(count=state['count']), translate=False)
    def button(row, field, value, index, text):
        op = row.operator(PPF_OT_GroupCollisionEdit.bl_idname, text=text,
                          depress=value is True, icon='REMOVE' if value is None else 'NONE')
        op.group_uuid = group.uuid
        op.field = field
        op.index = index
        op.value = value is not True
    button(body.row(), _FIELDS[0], state['enabled'], -1,
           '启用过滤（混合）' if state['enabled'] is None else '启用碰撞层过滤')
    for field, key, title in ((_FIELDS[1], 'groups', '所属层（可多选）'),
                              (_FIELDS[2], 'mask', '允许碰撞的层')):
        body.label(text=iface_(title) + (iface_(' · 混合') if None in state[key] else ''), translate=False)
        for start in (0, 8):
            row = body.row(align=True)
            for index in range(start, start + 8):
                button(row, field, state[key][index], index, f'{index + 1:02d}')
    body.label(text='混合位显示横线；点击仅统一这一位', icon='INFO')
    body.label(text='双方允许才碰撞；相邻骨骼另行排除')
    body.label(text='不同动力学组若包含同一对象，共享实际设置')


def register():
    _ppf_i18n.register_class(PPF_OT_GroupCollisionEdit)


def unregister():
    bpy.utils.unregister_class(PPF_OT_GroupCollisionEdit)
