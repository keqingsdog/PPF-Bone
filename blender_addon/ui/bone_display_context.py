"""Short-lived display-only selection; cached entries contain only scalar IDs."""
import bpy
from bpy.app.handlers import persistent

_views = {}
_arms = {}

def _key(context):
    return (context.scene.session_uid, context.view_layer.as_pointer(),
            context.window.as_pointer() if context.window else 0,
            context.area.as_pointer() if context.area else 0)

def remember(context):
    from ..models.bone_bindings import active_armature, active_bone
    from ..core.bone_selection_parameters import resolve_selected_authors
    arm, bone = active_armature(context), active_bone(context)
    if arm is None or context.mode not in {'OBJECT', 'POSE'}:
        return
    _arms[_key(context)] = (arm.name, arm.session_uid, arm.data.session_uid, context.mode)
    if bone is None:
        _views.pop(_key(context), None)
        return
    try:
        rows = resolve_selected_authors(context)
    except ValueError:
        _views.pop(_key(context), None)
        return
    if not rows:
        _views.pop(_key(context), None)
        return
    _views[_key(context)] = (arm.name, arm.session_uid, arm.data.session_uid,
        bone.name, bone.ppf_bone_physics.bone_uuid,
        context.mode, tuple((a.name, a.session_uid, b.name,
                             b.ppf_bone_physics.bone_uuid) for a,b in rows))

def resolve(context):
    key = _key(context)
    data = _views.get(key)
    if data is None:
        return None
    name, uid, data_uid, bone_name, uuid, mode, rows = data
    arm = context.scene.objects.get(name)
    if (arm is None or arm.type != 'ARMATURE' or arm.session_uid != uid or
            arm.data.session_uid != data_uid or context.view_layer.objects.get(name) != arm):
        _views.pop(key, None); return None
    def find(a, n, u):
        b = a.data.bones.get(n)
        if b is None or b.ppf_bone_physics.bone_uuid != u:
            return None
        if u and sum(x.ppf_bone_physics.bone_uuid == u for x in a.data.bones) != 1:
            return None
        return b
    bone = find(arm, bone_name, uuid)
    authors = []
    for an, au, bn, bu in rows:
        a = context.scene.objects.get(an)
        b = find(a, bn, bu) if (a and a.type == 'ARMATURE' and a.session_uid == au
                               and context.view_layer.objects.get(an) == a) else None
        if b is None:
            _views.pop(key, None); return None
        authors.append((a, b))
    if bone is None or arm.data.bones.active != bone:
        _views.pop(key, None); return None
    return DisplayContext(context, arm, bone, mode, authors)

def resolve_arm(context):
    key = _key(context)
    data = _arms.get(key)
    if data is None:
        return None
    name, uid, data_uid, mode = data
    arm = context.scene.objects.get(name)
    if (arm is None or arm.type != 'ARMATURE' or arm.session_uid != uid or
            arm.data.session_uid != data_uid or context.view_layer.objects.get(name) != arm):
        _arms.pop(key, None)
        return None
    bone = arm.data.bones.active
    return DisplayContext(context, arm, bone, mode, [(arm, bone)] if bone else [])

class DisplayContext:
    def __init__(self, context, arm, bone, mode, authors):
        self._context = context
        self.object = self.active_object = arm
        self.mode = mode
        self.selected_objects = tuple(dict.fromkeys(a for a, _ in authors))
        self.selected_pose_bones = tuple(a.pose.bones[b.name] for a,b in authors) if mode == 'POSE' else ()
        self.display_bone = bone
    def __getattr__(self, name):
        return getattr(self._context, name)

@persistent
def _clear(*_args):
    _views.clear()
    _arms.clear()

def register():
    for handlers in (bpy.app.handlers.load_pre, bpy.app.handlers.undo_pre, bpy.app.handlers.redo_pre):
        if _clear not in handlers:
            handlers.append(_clear)

def unregister():
    _views.clear()
    _arms.clear()
    for handlers in (bpy.app.handlers.load_pre, bpy.app.handlers.undo_pre, bpy.app.handlers.redo_pre):
        if _clear in handlers:
            handlers.remove(_clear)
