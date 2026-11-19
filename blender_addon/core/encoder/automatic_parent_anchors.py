"""Stable identities and sampled pose tracks for hidden kinematic parent anchors."""
from __future__ import annotations

from uuid import UUID, uuid5


def parent_anchor_identity(armature_uuid: str, parent_bone_uuid: str | None,
                           parent_bone_name: str) -> tuple[str, str]:
    """Return deterministic UUIDs for an unbound Blender parent and its body.

    Enabled bones retain their author-assigned UUID. A nonphysical bone has no
    binding row, so its name is the stable local key until it is explicitly
    given an author UUID in a future schema. The armature UUID scopes names and
    prevents two rigs with identically named bones from sharing an endpoint.
    """
    try:
        namespace = UUID(armature_uuid)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("BLENDER_BONE_INVALID: armature UUID is required for an automatic parent anchor") from exc
    parent_key = parent_bone_uuid or parent_bone_name
    if not isinstance(parent_key, str) or not parent_key:
        raise ValueError("BLENDER_BONE_INVALID: nonphysical parent needs a stable bone identity")
    parent_uuid = str(uuid5(namespace, "ppf:auto-parent-bone:v1:" + parent_key))
    body_uuid = str(uuid5(namespace, "ppf:auto-parent-body:v1:" + parent_uuid))
    return parent_uuid, body_uuid


def sample_parent_motion(context, armature, parent_bone_name: str,
                         start_frame: int, frame_count: int, fps: float,
                         initial_world_matrix) -> dict:
    """Sample a Blender pose as canonical solver-world kinematic keys.

    Caller is responsible for choosing the parent only when an automatic
    anchor is enabled. The timeline and depsgraph are restored even if a pose
    sample is invalid.
    """
    import math
    from .bones import _frame
    from ...models.bone_bindings import bone_world_matrix

    if type(start_frame) is not int or type(frame_count) is not int or frame_count < 2:
        raise ValueError("BLENDER_BONE_INVALID: automatic parent follow requires at least two sampled frames")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("BLENDER_BONE_INVALID: automatic parent follow FPS must be finite and positive")
    initial = _frame(initial_world_matrix, "automatic parent initial")
    scene = context.scene
    previous, subframe = scene.frame_current, scene.frame_subframe
    keys = []
    try:
        for offset in range(frame_count):
            scene.frame_set(start_frame + offset)
            context.view_layer.update()
            pose_bone = armature.pose.bones.get(parent_bone_name)
            if pose_bone is None:
                raise ValueError(f"BLENDER_BONE_INVALID: missing nonphysical parent bone {parent_bone_name!r}")
            world = bone_world_matrix(armature, pose_bone.bone)
            sampled = initial if offset == 0 else _frame(world, f"automatic parent {parent_bone_name} frame {start_frame + offset}")
            keys.append({
                "time_s": offset / float(fps),
                "center_world_m": sampled["translation_m"],
                "rotation_world_wxyz": sampled["quaternion_wxyz"],
            })
    finally:
        scene.frame_set(previous, subframe=subframe)
        context.view_layer.update()
    return {
        "mode": "kinematic",
        "authoring": "object_world_trs",
        "initial_frame": initial,
        "keys": keys,
    }



def prepare_parent_track(armature, parent_bone_name):
    """Plan one shared parent sample in the existing root-motion frame pass."""
    parent = armature.pose.bones[parent_bone_name]
    chain = []
    current = parent
    while current is not None and not current.bone.ppf_bone_physics.enabled:
        chain.append(current)
        current = current.parent
    request = dict(arm=armature, parent=parent, ancestor=current, chain=chain,
                   reference=None, records=None, keys=[])
    if current is None:
        return request  # Preserve ordinary world-kinematic root behavior.
    _validate_bridge_chain(request)
    return request


def _validate_bridge_chain(request):
    """Match the frozen FK consumer contract at every existing sample time."""
    import numpy as np
    armature = request['arm']
    ad = armature.animation_data
    for pb in request['chain']:
        if any(not constraint.mute for constraint in pb.constraints):
            _bridge_fail(request, f'骨骼 {pb.name} 的约束')
        if not pb.bone.use_inherit_rotation or pb.bone.inherit_scale != 'FULL':
            _bridge_fail(request, f'骨骼 {pb.name} 的特殊继承设置')
        if ad is not None and any(not curve.mute and curve.data_path.startswith(pb.path_from_id()+'.') for curve in ad.drivers):
            _bridge_fail(request, f'骨骼 {pb.name} 的驱动器')
    for item in request['records'] or ():
        pb = armature.pose.bones[item['name']]
        if (not np.allclose(np.asarray(pb.matrix_basis), item['matrix_basis'], atol=1.e-6, rtol=0.)
                or not np.allclose(np.asarray(pb.bone.matrix_local), item['matrix_local'], atol=1.e-6, rtol=0.)
                or pb.bone.inherit_scale != item['inherit_scale']
                or bool(pb.bone.use_inherit_rotation) != item['use_inherit_rotation']):
            _bridge_fail(request, f'相对祖先随时间变化的作者动画（中段 {pb.name}）；当前接口不能表达时变局部桥接')


def _bridge_fail(request, detail):
    raise ValueError(f'骨骼 {request["parent"].name}：非物理中段到参与物理祖先 {request["ancestor"].name} 的桥接不支持{detail}；请保留中段参与物理，或关闭下段“跟随非物理父骨骼”后明确作为独立根。')


def sample_parent_track(request, offset, fps, initial_world):
    """Read a single frame; caller owns one timeline pass for all requests."""
    import numpy as np
    from ...models.bone_bindings import bone_world_matrix
    from .bones import _frame
    arm, parent, ancestor = request['arm'], request['parent'], request['ancestor']
    world = bone_world_matrix(arm, parent.bone)
    if ancestor is None:
        frame = _frame(initial_world if offset == 0 else world, 'automatic parent pose')
        request['keys'].append(dict(time_s=offset/float(fps), center_world_m=frame['translation_m'],
                                    rotation_world_wxyz=frame['quaternion_wxyz']))
        return
    _validate_bridge_chain(request)
    relative = np.asarray(bone_world_matrix(arm, ancestor.bone).inverted() @ world, dtype=float)
    if not np.isfinite(relative).all():
        _bridge_fail(request, '非有限变换')
    if request['reference'] is None:
        request['reference'] = relative.copy()
        request['records'] = [dict(name=pb.name, parent_name=pb.parent.name if pb.parent else None,
            matrix_basis=[list(row) for row in pb.matrix_basis],
            matrix_local=[list(row) for row in pb.bone.matrix_local],
            inherit_scale=pb.bone.inherit_scale,
            use_inherit_rotation=bool(pb.bone.use_inherit_rotation)) for pb in reversed(request['chain'])]
    elif not np.allclose(relative, request['reference'], atol=1.e-6, rtol=0.):
        _bridge_fail(request, '相对祖先随时间变化的作者动画；当前接口不能表达随动态祖先移动的时变局部锚点')


def finish_parent_track(request, initial_world):
    from .bones import _frame
    ancestor = request['ancestor']
    if ancestor is None:
        return 'root_motion', dict(mode='kinematic', authoring='object_world_trs',
            initial_frame=_frame(initial_world, 'automatic parent initial'), keys=request['keys'])
    return 'physical_ancestor_bridge_v1', dict(version=1,
        ancestor_bone_uuid=ancestor.bone.ppf_bone_physics.bone_uuid,
        ancestor_body_uuid=ancestor.bone.ppf_bone_physics.body_uuid,
        ancestor_bone_name=ancestor.name, ancestor_to_parent_blender=request['reference'].tolist(),
        chain=request['records'])
