"""Read-only placement of owned proxies at the evaluated solver start frame."""

def projected(arm, bone, proxy, driven):
    """Project a proven authoring offset without touching transform channels."""
    from ..bone_author_follow import _load_author_record, _rigid_frame
    if bone.name not in driven:
        return None
    settings = bone.ppf_bone_physics
    # The proxy animation/constraint graph, not the source bone, owns placement.
    ad = proxy.animation_data
    if ((settings.root_motion_source == 'OBJECT' and
         (settings.root_mode != 'FREE' or settings.root_control_events))
            or (ad and (ad.action or ad.drivers or ad.nla_tracks))
            or any(not c.mute for c in proxy.constraints)):
        return None
    record = _load_author_record(settings, proxy)
    if record is None:
        return None
    pose = arm.pose.bones.get(bone.name)
    if pose is None:
        return None
    target = arm.matrix_world @ pose.matrix @ record[0][2]
    # Do not mask a scale error in the stored record or bypass native guards.
    if not _rigid_frame(target):
        return None
    return target


def projection_lookup(scene):
    """Build a lightweight per-armature map only for owned proxies encountered."""
    from ...models.bone_bindings import proxy_parent_armature
    from ..bone_author_follow import time_driven_bones
    armatures = {}
    def lookup(proxy):
        if not proxy.get('ppf_bone_proxy_owned'):
            return None
        arm = proxy_parent_armature(proxy)
        if arm is None:
            return None
        key = arm.as_pointer()
        if key not in armatures:
            armatures[key] = (
                {bone.ppf_bone_physics.proxy.as_pointer(): bone
                 for bone in arm.data.bones
                 if bone.ppf_bone_physics.enabled and bone.ppf_bone_physics.proxy is not None},
                time_driven_bones(arm))
        bones, driven = armatures[key]
        bone = bones.get(proxy.as_pointer())
        if bone is None or scene.objects.get(arm.name) != arm:
            return None
        return projected(arm, bone, proxy, driven)
    return lookup
