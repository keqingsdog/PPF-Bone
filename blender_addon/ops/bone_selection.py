"""User-invoked selection of physical bones and collision memberships."""
import bpy
from ..core.compat import set_pose_bone_selected
from ..i18n import iface, register_class
from ..i18n.props import BoolVectorProperty
from ..models.bone_bindings import active_armature


def _selectable(bone):
    if bone.hide or bone.hide_select:
        return False
    collections = bone.collections
    return not collections or any(c.is_visible_effectively for c in collections)


def _select(context, operator, slots=None):
    arm = active_armature(context)
    if arm is None or context.mode not in {'OBJECT', 'POSE'}:
        operator.report({'WARNING'}, iface('请在物体或姿态模式选择骨架'))
        return {'CANCELLED'}
    matches = []
    for pb in arm.pose.bones:
        settings = pb.bone.ppf_bone_physics
        if not settings.enabled or not _selectable(pb.bone):
            continue
        if slots is not None and (not settings.collision_filter_enabled or
                not any(wanted and member for wanted, member in zip(slots, settings.collision_group_layers))):
            continue
        matches.append(pb.name)
    if not matches:
        operator.report({'WARNING'}, iface('没有匹配的可选物理骨骼'))
        return {'CANCELLED'}
    if context.mode == 'OBJECT':
        try:
            from .bone_proxy import enter_proxy_pose
            enter_proxy_pose(context, 'ALL')
        except (ValueError, RuntimeError) as exc:
            operator.report({'WARNING'}, str(exc))
            return {'CANCELLED'}
    names = set(matches)
    for pb in arm.pose.bones:
        if _selectable(pb.bone):
            set_pose_bone_selected(pb, pb.name in names)
    arm.data.bones.active = arm.data.bones.get(matches[0])
    operator.report({'INFO'}, iface('已选择 {count} 根物理骨骼').format(count=len(matches)))
    return {'FINISHED'}


class PPF_OT_SelectPhysicsBones(bpy.types.Operator):
    bl_idname = 'ppf.select_physics_bones'
    bl_label = '选择物理骨骼'
    bl_description = '选择在当前骨架中参与物理模拟的可选骨骼'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        return _select(context, self)


class PPF_OT_SelectCollisionGroupBones(bpy.types.Operator):
    bl_idname = 'ppf.select_collision_group_bones'
    bl_label = '选择碰撞组骨骼'
    bl_description = '选择属于指定碰撞组的可选物理骨骼'
    bl_options = {'REGISTER', 'UNDO'}

    groups: BoolVectorProperty(name='碰撞组', size=16, default=(True,) + (False,) * 15)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        for start in (0, 8):
            row = self.layout.row(align=True)
            for index in range(start, start + 8):
                row.prop(self, 'groups', index=index, text=f'{index+1:02d}', toggle=True)

    def execute(self, context):
        return _select(context, self, self.groups)


classes = (PPF_OT_SelectPhysicsBones, PPF_OT_SelectCollisionGroupBones)


def register():
    for cls in classes:
        register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
