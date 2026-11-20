"""Give each extra MMD body its own non-deforming PPF binding."""

from contextlib import contextmanager

import bpy

from ..core.compat import pose_bone_selected, set_pose_bone_selected


_SOURCE = "ppf_mmd_helper_source"


def find(arm, original_bone_name, source):
    parent = arm.data.bones.get(original_bone_name)
    if parent is None:
        return None
    matches = [bone for bone in parent.children if bone.get(_SOURCE) == source]
    if len(matches) > 1:
        raise ValueError("Multiple PPF helper bones refer to MMD rigid body: " + source.name)
    return matches[0] if matches else None


def helper_name(original_bone_name, source):
    return f"PPF MMD {source.name}"


def _mode_set(mode):
    if bpy.ops.object.mode_set(mode=mode) != {'FINISHED'}:
        raise RuntimeError("Could not switch armature mode for MMD physics import")


@contextmanager
def _edit(context, arm):
    mode = context.mode
    if mode not in {'OBJECT', 'POSE'} or context.view_layer.objects.active != arm:
        raise ValueError("Select the MMD armature in Object or Pose mode")
    if arm.data.users - int(arm.data.use_fake_user) > 1 or arm.data.library or arm.data.override_library:
        raise ValueError("MMD helper bones require a local, single-user armature")
    selected = [obj for obj in context.selected_objects if obj != arm]
    pose_selection = {bone.name for bone in arm.pose.bones if pose_bone_selected(bone)}
    active = arm.data.bones.active
    active_name = active.name if active else None
    try:
        if mode == 'POSE':
            _mode_set('OBJECT')
        for obj in selected:
            obj.select_set(False)
        _mode_set('EDIT')
        yield arm.data.edit_bones
    finally:
        if context.mode == 'EDIT_ARMATURE':
            _mode_set('OBJECT')
        for obj in selected:
            obj.select_set(True)
        if mode == 'POSE' and context.mode == 'OBJECT':
            _mode_set('POSE')
        for bone in arm.pose.bones:
            set_pose_bone_selected(bone, bone.name in pose_selection)
        arm.data.bones.active = arm.data.bones.get(active_name) if active_name else None


def create(context, arm, sources):
    sources = tuple(sources)
    created = {}
    if not sources:
        return created
    try:
        with _edit(context, arm) as bones:
            for original, source in sources:
                parent = bones[original]
                bone = bones.new(helper_name(original, source))
                created[source] = bone.name
                bone.parent = parent
                bone.head = parent.head
                bone.tail = parent.tail
                bone.roll = parent.roll
                bone.use_deform = False
        for source, name in created.items():
            bone = arm.data.bones[name]
            bone[_SOURCE] = source
            bone.hide = True
        context.view_layer.update()
    except Exception:
        remove(context, arm, created.values())
        raise
    return created


def remove(context, arm, names):
    names = tuple(names)
    if names:
        with _edit(context, arm) as bones:
            for name in names:
                bone = bones.get(name)
                if bone is not None:
                    bones.remove(bone)
        context.view_layer.update()
