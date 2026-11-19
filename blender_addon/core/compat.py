"""Small capability adapters shared by the supported Blender API generations."""


def pose_bone_selected(pose_bone):
    target=pose_bone if hasattr(pose_bone,'select') else pose_bone.bone
    return bool(target.select)


def set_pose_bone_selected(pose_bone,selected):
    target=pose_bone if hasattr(pose_bone,'select') else pose_bone.bone
    target.select=bool(selected)
