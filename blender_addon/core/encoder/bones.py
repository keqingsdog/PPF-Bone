"""Transport bone intent through the original parameter upload and hash."""
from __future__ import annotations

from copy import deepcopy
from itertools import pairwise
import json
import math

import numpy as np

SCHEMA = "PPF_BLENDER_BONES_V1"
_S = np.array([[1., 0., 0., 0.], [0., 0., 1., 0.],
               [0., -1., 0., 0.], [0., 0., 0., 1.]])


def _frame(value, label):
    """Convert a Blender world rigid frame, retaining bone-local coordinates."""
    from mathutils import Matrix
    a = np.asarray(value, dtype=np.float64)
    if (a.shape != (4, 4) or not np.isfinite(a).all()
            or not np.allclose(a[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or abs(np.linalg.det(a[:3, :3]) - 1) > 1e-6):
        raise ValueError(f"BLENDER_BONE_INVALID: {label}: scale/shear/reflection unsupported")
    solver = _S @ a
    q = Matrix(solver[:3, :3].tolist()).to_quaternion().normalized()
    quat = np.asarray(tuple(q), dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return {"translation_m": solver[:3, 3].tolist(),
            "quaternion_wxyz": quat.tolist()}


def _root_events(settings, start_frame, fps):
    events = [{'frame':float(event.frame),'mode':event.mode,'pose_binding':event.pose_binding} for event in settings.root_control_events]
    result = []
    modes = {"FREE": "dynamic", "KINEMATIC": "kinematic", "STATIC": "static"}
    for event in events:
        if not isinstance(event, dict) or set(event) != {"frame", "mode", "pose_binding"} or event["mode"] not in modes:
            raise ValueError("BLENDER_BONE_INVALID: root_control_events_json: expected frame/mode event")
        frame = event["frame"]
        if isinstance(frame, bool) or not isinstance(frame, (int, float)) or not math.isfinite(frame) or frame <= start_frame:
            raise ValueError("BLENDER_BONE_INVALID: root event must follow simulation start")
        item={'time_s':(frame-start_frame)/fps,'mode':modes[event['mode']]}
        if event['mode']!='FREE':
            item['pose_binding'] = ('current_hold' if event['mode']=='STATIC' else 'rigid_delta_v1') if event['pose_binding']=='CURRENT' else 'absolute'
        result.append(item)
    if any(a['time_s'] >= b['time_s'] for a, b in zip(result, result[1:])):
        raise ValueError('Bone control events must be in strictly increasing frame order')
    return result


def _drives(settings, start_frame, fps):
    result = {'coordinate_model': 'parent_local_swing_twist_x_v1'}
    for axis in ('twist_x', 'swing_y', 'swing_z'):
        channels = {}
        for mode in ('position', 'velocity', 'torque'):
            channel = getattr(getattr(settings, 'drive_' + axis), mode)
            if not channel.enabled:continue
            gain_key = {'position':'gain_Nm_per_rad','velocity':'gain_Nms_per_rad','torque':'gain'}[mode]
            value_key = {'position':'value_rad','velocity':'value_rad_per_s','torque':'value_Nm'}[mode]
            keys = [{'time_s':(float(key.frame)-start_frame)/fps, value_key:float(key.value)} for key in channel.targets]
            if not keys or keys[0]['time_s'] != 0. or any(a['time_s'] >= b['time_s'] for a,b in zip(keys,keys[1:])):
                raise ValueError('Drive targets must start at the simulation start frame and increase strictly')
            channels[mode] = {'enabled':True,gain_key:float(channel.gain),'max_effort_Nm':float(channel.max_effort),'interpolation':'linear_hold','target_keys':keys}
        if channels:result[axis] = channels
    return result if len(result)>1 else None



def _select_joint_coordinate_model(row, physical_ids, rigid_ids=None):
    """Mark real rigid/FEM joints and generated physical parent anchors."""
    parent = row.get('parent_body_uuid')
    anchor = row.get('automatic_parent_anchor_v1')
    anchored = isinstance(anchor, dict) and bool(anchor.get('anchor_body_uuid'))
    if row.get('body_uuid') in physical_ids and (parent in physical_ids or anchored):
        control = row['physics'].get('angular_control_v1')
        hard = any(key in row['physics'] for key in ('angular_hard_hinge_v1', 'joint_d6_hard_v1'))
        if hard and (row.get('body_uuid') not in (rigid_ids if rigid_ids is not None else physical_ids)
                     or parent not in (rigid_ids if rigid_ids is not None else physical_ids)):
            return
        if control is not None:
            control['coordinate_model'] = ('parent_local_swing_twist_x_continuous_v2'
                                           if hard else 'material_spin_path_v3')
        for field in ('angular_soft_limits_v1', 'angular_drives_v1'):
            if field in row['physics']:
                row['physics'][field]['coordinate_model'] = ('parent_local_swing_twist_x_continuous_v2'
                                                           if hard else 'material_spin_path_v3')


def _action_curves(action, slot=None):
    if action is None:
        return
    if getattr(action, 'is_action_layered', False):
        if slot is None:
            return
    else:
        yield from getattr(action, 'fcurves', ())
        return
    for layer in getattr(action, 'layers', ()):
        for strip in layer.strips:
            for bag in getattr(strip, 'channelbags', ()):
                if bag.slot_handle == slot.handle:
                    yield from bag.fcurves


def _has_unsupported_scale_animation(ad, paths):
    def affects_scale(curve):
        return not curve.mute and curve.data_path in paths

    if any(affects_scale(curve) for curve in ad.drivers):
        return True

    def strip_keys_scale(strip):
        return (any(affects_scale(curve) for curve in
                    _action_curves(strip.action, getattr(strip, 'action_slot', None)))
                or any(strip_keys_scale(child) for child in strip.strips))

    if any(strip_keys_scale(strip) for track in ad.nla_tracks for strip in track.strips):
        return True

    action = ad.action
    curves = [curve for curve in _action_curves(action, getattr(ad, 'action_slot', None))
              if affects_scale(curve)]
    if not curves:
        return False
    if (ad.use_tweak_mode or ad.action_blend_type != 'REPLACE' or ad.action_influence != 1.
            or ad.action_extrapolation != 'HOLD'
            or (ad.use_nla and any(not track.mute and track.strips for track in ad.nla_tracks))):
        return True
    layers = getattr(action, 'layers', ())
    if len(layers) > 1 or any(len(layer.strips) != 1 for layer in layers):
        return True
    return any(not _is_constant_positive_scale_curve(curve) for curve in curves)


def _is_constant_positive_scale_curve(curve):
    """Prove constancy between keys as well as outside the keyed range."""
    keys = curve.keyframe_points
    if not keys or curve.sampled_points or curve.modifiers or curve.extrapolation != 'CONSTANT':
        return False
    value = keys[0].co.y
    if not math.isfinite(value) or value <= 0. or any(key.co.y != value for key in keys):
        return False
    for left, right in pairwise(keys):
        if left.interpolation == 'BEZIER':
            if left.handle_right.y != value or right.handle_left.y != value:
                return False
        elif left.interpolation not in {'CONSTANT', 'LINEAR'}:
            return False
    return True


def _known_scale_validation_handlers(scene):
    """Only allow the plugin's own reviewed callbacks, with pose playback off."""
    import bpy
    from .. import bone_author_follow, bone_mass, live_timeline, pc2, rigid_pose, uuid_registry
    from ...ui import bone_overlay, joint_axes_overlay
    from ...ui.dynamics import group_ops, statistics

    if (scene.get(rigid_pose.CACHE_KEY) or rigid_pose._detached_display_pending(scene)
            or getattr(getattr(scene, 'rigidbody_world', None), 'enabled', False)):
        return False
    allowed = {
        'frame_change_pre': (bone_overlay._frame_placement_pre,),
        'frame_change_post': (
            bone_overlay._frame_placement_post, bone_author_follow._observe_frame_object_basis,
            live_timeline._observe_frame_change, pc2.curve_frame_change_handler,
            rigid_pose.bone_pose_frame_change, statistics.statistics_frame_change_handler,
        ),
        'depsgraph_update_pre': (),
        'depsgraph_update_post': (
            bone_overlay._depsgraph_changed, bone_author_follow._mark_author_motion,
            joint_axes_overlay._invalidate_armature_index,
            bone_mass.mass_depsgraph_updated, uuid_registry._on_depsgraph_update,
            group_ops._cleanup_deleted_objects,
        ),
    }
    return all(all(any(handler is own for own in known)
                   for handler in getattr(bpy.app.handlers, name))
               for name, known in allowed.items())


def _provably_fixed_fk_scale(arm, names, paths):
    """Prove bounded FK and its object-parent chain have fixed uniform scale.

    Unknown actions, inheritance, controllers or curve interpolation retain
    evaluation. This is not a general dependency graph proof.
    """
    from ...models.bone_bindings import _rest_frame_for_drift

    if arm.constraints or any(
            pose.constraints for pose in arm.pose.bones):
        return False  # Sibling/descendant IK can affect relevant bones.
    def action_valid(owner, scale_paths, movement_paths):
        ad = owner.animation_data
        if ad is None:
            return True
        if (ad.drivers or ad.nla_tracks or ad.use_tweak_mode
                or ad.action_blend_type != 'REPLACE' or ad.action_influence != 1.
                or ad.action_extrapolation != 'HOLD' or ad.action is None):
            return False
        action = ad.action
        if (getattr(action, 'is_action_layered', False)
                and getattr(ad, 'action_slot', None) is None):
            return False
        layers = getattr(action, 'layers', ())
        if len(layers) > 1 or any(len(layer.strips) != 1 for layer in layers):
            return False
        for curve in _action_curves(action, getattr(ad, 'action_slot', None)):
            if curve.mute:
                continue
            if curve.data_path in scale_paths:
                if (not 0 <= curve.array_index < 3
                        or not _is_constant_positive_scale_curve(curve)
                        or abs(curve.keyframe_points[0].co.y
                               - owner.path_resolve(curve.data_path)[curve.array_index]) > 1.e-7):
                    return False
                continue
            if (curve.data_path not in movement_paths or not 0 <= curve.array_index < 3
                    or curve.extrapolation != 'CONSTANT' or curve.modifiers
                    or curve.sampled_points or not curve.keyframe_points):
                return False
            for key in curve.keyframe_points:
                if (key.interpolation not in {'BEZIER', 'LINEAR', 'CONSTANT'}
                        or not all(math.isfinite(float(v)) and abs(v) <= 1.e3
                                   for v in (key.co.y, key.handle_left.y, key.handle_right.y))
                        or not all(math.isfinite(float(v)) for v in
                                   (key.co.x, key.handle_left.x, key.handle_right.x))):
                    return False
        return True

    def constant_basis(owner):
        return (all(math.isfinite(float(value)) and abs(float(value)) <= 1.e3
                    for field in ('location', 'rotation_euler', 'rotation_quaternion',
                                  'rotation_axis_angle')
                    for value in getattr(owner, field))
                and 0.25 <= owner.scale[0] <= 4.
                and all(value == owner.scale[0] for value in owner.scale))

    if (not constant_basis(arm) or not all(math.isfinite(float(value)) and abs(value) <= 1.e3
            for value in (*arm.delta_location, *arm.delta_rotation_euler,
                          *arm.delta_rotation_quaternion))
            or any(value != 1. for value in arm.delta_scale)):
        return False
    # Prove every object link is a positive similarity. A fixed uniform
    # inverse and fixed uniform scale stay uniform under animated rotations.
    from ...models.bone_bindings import similarity_frame_scale
    object_scale = 1.0
    accumulated_error = 0.0
    depth = 0
    current = arm
    while current.parent is not None:
        depth += 1
        if depth > 12 or current.parent_type != 'OBJECT':
            return False
        matrix = current.matrix_parent_inverse
        if (any(not math.isfinite(float(v)) or abs(float(v)) > 1.e3
                for v in matrix.translation)
                or any(abs(matrix[3][j] - (1. if j == 3 else 0.)) > 1.e-7
                       for j in range(4))):
            return False
        basis = matrix.to_3x3()
        if basis.determinant() <= 0.:
            return False
        columns = [basis.col[i] for i in range(3)]
        scale_sq = sum(column.dot(column) for column in columns) / 3.
        if not math.isfinite(scale_sq) or scale_sq <= 0.:
            return False
        gram_error = max(abs(columns[i].dot(columns[j]) / scale_sq
                             - (1. if i == j else 0.))
                         for i in range(3) for j in range(3))
        accumulated_error += gram_error
        if accumulated_error > 5.e-7:
            return False
        try:
            _, inverse_scale = similarity_frame_scale(matrix, current.name)
        except (ValueError, RuntimeError):
            return False
        parent = current.parent
        if parent.constraints or not constant_basis(parent):
            return False
        if (any(value != 1. for value in parent.delta_scale)
                or not all(math.isfinite(float(value)) and abs(value) <= 1.e3
                           for value in (*parent.delta_location, *parent.delta_rotation_euler,
                                         *parent.delta_rotation_quaternion))):
            return False
        movement_paths = {'location'}
        if parent.rotation_mode not in {'QUATERNION', 'AXIS_ANGLE'}:
            movement_paths.update(('rotation_euler', 'delta_rotation_euler'))
        if not action_valid(parent, {'scale', 'delta_scale'}, movement_paths):
            return False
        object_scale *= inverse_scale * parent.scale[0]
        if not 0.25 <= object_scale <= 4.:
            return False
        current = parent
    movement = {'location'}
    if arm.rotation_mode not in {'QUATERNION', 'AXIS_ANGLE'}:
        movement.add('rotation_euler')
        movement.add('delta_rotation_euler')
    for name in names:
        pose = arm.pose.bones[name]
        bone = pose.bone
        rest = _rest_frame_for_drift(bone.matrix_local)
        if (not constant_basis(pose) or bone.inherit_scale != 'FULL'
                or rest is None or rest[1] > 5.e-7
                or len(pose.parent_recursive) > 12):
            return False
        # Uniform factors can be composed without shear; bound the aggregate
        # to avoid overflow/near-singularity along an otherwise simple chain.
        effective = object_scale * arm.scale[0] * pose.scale[0] * math.prod(
            parent.scale[0] for parent in pose.parent_recursive)
        if not 0.25 <= effective <= 4.:
            return False
        if pose.rotation_mode not in {'QUATERNION', 'AXIS_ANGLE'}:
            movement.add(pose.path_from_id('rotation_euler'))
        movement.add(pose.path_from_id('location'))
    return action_valid(arm, paths, movement)



def _pose_scale_sources(context, *, explicit_only=False):
    """Collect the frozen scale references needed by the frame sampler.

    Check FREE and STATIC bodies too, including ancestors that are not physical.
    Reject curves that can change scale between samples, including at subframes.
    """
    from ...models.bone_bindings import bone_world_frame_scale
    sources = {}
    sample_needed = False
    fast = None
    for arm in context.scene.objects:
        if arm.type != 'ARMATURE':
            continue
        names = set()
        for bone in arm.data.bones:
            if bone.ppf_bone_physics.enabled:
                names.add(bone.name)
                names.update(p.name for p in arm.pose.bones[bone.name].parent_recursive)
        if not names:
            continue
        ad = arm.animation_data
        arm_ad = ad
        paths = {arm.pose.bones[name].path_from_id('scale') for name in names} | {'scale', 'delta_scale'}
        if ad is not None:
            if _has_unsupported_scale_animation(ad, paths):
                raise ValueError(f'{arm.name}: animated pose/object scale is unsupported; use a static positive uniform pose scale')
            sample_needed = True
        sample_needed |= bool(arm.constraints) or any(arm.pose.bones[name].constraints for name in names)
        ancestor = arm.parent
        while ancestor is not None:
            # An animated object ancestor can change the armature world frame
            # even when the armature has no action of its own.
            sample_needed = True
            ad = ancestor.animation_data
            if ad is not None and _has_unsupported_scale_animation(ad, {'scale', 'delta_scale'}):
                raise ValueError(f'{ancestor.name}: animated ancestor object scale is unsupported')
            ancestor = ancestor.parent
        if explicit_only:
            continue
        if arm_ad is not None or arm.parent is not None:
            if fast is None:
                fast = _known_scale_validation_handlers(context.scene)
            if fast and _provably_fixed_fk_scale(arm, names, paths):
                # Keep the entry-frame validation even when no frame walk is needed.
                for name in names:
                    bone_world_frame_scale(arm, arm.data.bones[name])
                continue
        for name in names:
            sources[(arm.name, name)] = (arm, arm.data.bones[name], bone_world_frame_scale(arm, arm.data.bones[name])[1])
    return sources.values() if sample_needed and not explicit_only else ()


def _validate_pose_scales_at_frame(sources, frame):
    from ...models.bone_bindings import bone_world_frame_scale
    for arm, bone, initial in sources:
        scale = bone_world_frame_scale(arm, bone)[1]
        if abs(scale / initial - 1.) > 1.e-5:
            raise ValueError(f'{bone.name}: animated pose scale is unsupported; frozen author scale changed at frame {frame}')


def _validate_static_pose_scales_steps(context, state, start_frame, *, explicit_only=False):
    from ...models.bone_bindings import bone_frame_cache
    sources = _pose_scale_sources(context, explicit_only=explicit_only)
    if not sources:
        return
    from ..frame_time import set_scene_frame
    scene = context.scene
    previous, subframe = scene.frame_current, scene.frame_subframe
    try:
        for offset in range(int(state.frame_count)):
            frame = start_frame + offset
            scene.frame_set(frame)
            context.view_layer.update()
            with bone_frame_cache():
                _validate_pose_scales_at_frame(sources, frame)
            yield ("Validating bone scales: {done}/{total} frames", offset + 1, int(state.frame_count))
    finally:
        set_scene_frame(scene, previous, subframe)
        context.view_layer.update()


def _validate_static_pose_scales(context, state, start_frame, *, explicit_only=False):
    from ..utils import finish_encode_steps
    return finish_encode_steps(_validate_static_pose_scales_steps(context, state, start_frame, explicit_only=explicit_only))


def read_evaluated_bone_frame(armature, bone_name):
    """Read a rigid world matrix at the caller's current evaluated frame.

    This is a read only; no frame set, graph evaluation, persistent cache, or
    playback mode change. The caller owns author-pose evaluation and may wrap
    a batch of reads in bone_frame_cache(), never across timeline steps.
    """
    from ...models.bone_bindings import bone_world_matrix
    return bone_world_matrix(armature, armature.data.bones[bone_name])


def sample_transport_bone_frame(scale_sources, anchors, roots, offset, fps, sample_frame):
    """Consume one already-evaluated frame inside the existing Transfer preflight.

    Preserve validation order and the exact initial-frame overrides. The
    request containers are transaction-owned and their Blender references
    must not survive a graph replacement, scene reload, or separate capture.
    """
    from ...models.bone_bindings import bone_frame_cache
    from .automatic_parent_anchors import sample_parent_track
    with bone_frame_cache():
        _validate_pose_scales_at_frame(scale_sources, sample_frame)
        for request in anchors:
            sample_parent_track(request, offset, fps,
                                request['anchors'][0]['parent_world_matrix_blender'],
                                sample_frame)
        for arm, bone_name, row, root, source_mode in roots:
            if root["mode"] == "static" and offset > 0 and not root.get("control_events"):
                continue
            if source_mode == "POSE":
                world = read_evaluated_bone_frame(arm, bone_name)
            else:
                proxy = arm.data.bones[bone_name].ppf_bone_physics.proxy
                if proxy is None:
                    raise ValueError("BLENDER_BONE_INVALID: missing root proxy motion source")
                world = proxy.matrix_world
            frame = _frame(world, f"root {bone_name} frame {sample_frame}")
            if offset == 0:
                frame = root["initial_frame"]
            root["keys"].append({"time_s": offset / float(fps),
                                  "center_world_m": frame["translation_m"],
                                  "rotation_world_wxyz": frame["quaternion_wxyz"]})


def prepare_transport_bones(context, state, fps, start_frame, *, cone_restoring_joint_ids=None,
                 mmd_automatic_channels=None):
    """Prepare a transaction-local plan at the evaluated solver start frame."""
    from ...models.bone_bindings import (
        collect_bindings, collect_body_controls_contract,
    )
    from .automatic_parent_anchors import prepare_parent_track
    from ...models.restoring_curves import mmd_automatic_angular_channels
    source = collect_bindings(context, project_author_pose=True)
    from ..cross_links import encode as encode_cross_links, collision_pairs
    cross_links = encode_cross_links(context, source['bindings'])
    cross_collision_pairs = collision_pairs(context, source['bindings'])
    if not source["bindings"]:
        return None
    scale_sources = _pose_scale_sources(context)
    if source.get("schema") != SCHEMA or source.get("coordinate_space") != "BLENDER_Z_UP":
        raise ValueError("BLENDER_BONE_INVALID: unsupported authoring schema/coordinates")
    if source.get("scene_unit_scale_m", 1.) != 1.:
        raise ValueError("BLENDER_BONE_INVALID: original mesh transport requires scene unit scale 1 for bones")
    if float(state.world_scaling) != 1.:
        raise ValueError("BLENDER_BONE_INVALID: native joint scenes require world scaling exactly 1")
    result = {"schema": SCHEMA, "armatures": deepcopy(source["armatures"]),
              "bindings": deepcopy(source["bindings"]), "start_frame": int(start_frame),
              "fps": float(fps), "world_scaling": 1.}
    if cross_links is not None:
        result['cross_links_v1'] = cross_links
    from .pmx_joints import encode as encode_pmx_joints
    pmx_joints = encode_pmx_joints(context, source['bindings'])
    if pmx_joints:
        result['pmx_joints_v1'] = pmx_joints
    if cross_collision_pairs:
        result['cross_link_collision_pairs_v1'] = cross_collision_pairs
    if 'bone_mass_intent_v1' in source:
        result['bone_mass_intent_v1']=deepcopy(source['bone_mass_intent_v1'])
    author_intent = collect_body_controls_contract(context)
    if author_intent['body_controls_v1']:
        result['body_controls_v1']=True
        result['body_controls_author_intent_v1'] = deepcopy(author_intent)
    if "collision_filter_v1" in source:
        result["collision_filter_v1"] = deepcopy(source["collision_filter_v1"])
    armatures = {a["armature_uuid"]: context.scene.objects.get(a["object_name"]) for a in source["armatures"]}
    # Blender RNA iteration is expensive in a deep rig. Preserve first
    # match semantics of the old UUID search, but index each armature once.
    # Plans remain invocation-local; no evaluated pose or index survives it.
    bone_index = {}
    physical_ids, rigid_ids, fem_control_ids = set(), set(), set()
    for arm_uuid, arm in armatures.items():
        if arm is None:
            continue
        arm_index = bone_index[arm_uuid] = {}
        for bone in arm.data.bones:
            settings = bone.ppf_bone_physics
            arm_index.setdefault(settings.bone_uuid, bone)
            kind = settings.physical_model
            body_id = settings.body_uuid
            if kind == 'SOFT' and settings.root_control_events:
                fem_control_ids.add(body_id)
            if settings.enabled and kind in ('RIGID', 'SOFT'):
                physical_ids.add(body_id)
                if kind == 'RIGID':
                    rigid_ids.add(body_id)
    fem_event_scene = any(r.get('enabled') and r['body_uuid'] in fem_control_ids
                          for r in result['bindings'])
    roots = []
    sampled_auto_anchors = {}
    for row in result["bindings"]:
        _frame(row["bone_world_matrix_blender"], row["bone_name"])
        _frame(row["proxy_world_matrix_blender"], row["bone_name"] + " proxy")
        arm = armatures[row["armature_uuid"]]
        if arm is None or arm.ppf_armature_uuid != row["armature_uuid"]:
            raise ValueError("BLENDER_BONE_INVALID: armature UUID cannot be resolved")
        bone = bone_index[row["armature_uuid"]].get(row["bone_uuid"])
        if bone is None:
            raise ValueError("BLENDER_BONE_INVALID: root bone UUID cannot be resolved")
        has_joint = bool(row.get('parent_body_uuid') or row.get('automatic_parent_anchor_v1'))
        settings = bone.ppf_bone_physics
        if (cone_restoring_joint_ids is not None and has_joint
                and settings.angular_restoring_mode == 'CONE'):
            cone_restoring_joint_ids.add(row['joint_uuid'])
        if mmd_automatic_channels is not None and has_joint:
            mmd_automatic_channels.extend((row['joint_uuid'], channel)
                                          for channel in mmd_automatic_angular_channels(settings))
        drives = _drives(settings, start_frame, fps) if has_joint else None
        if drives:
            if any('angular_hard_hinge_v1' in r['physics'] or 'joint_d6_hard_v1' in r['physics'] for r in result['bindings']):
                raise ValueError('Hard joints cannot be combined with angular drive channels')
            row['physics']['angular_drives_v1'] = drives
        _select_joint_coordinate_model(row, physical_ids, rigid_ids)
        anchor = row.get('automatic_parent_anchor_v1')
        if anchor is not None:
            cache_key = anchor['anchor_body_uuid']
            if cache_key not in sampled_auto_anchors:
                sampled_auto_anchors[cache_key] = prepare_parent_track(arm, anchor['parent_bone_name'])
            sampled_auto_anchors[cache_key].setdefault('anchors', []).append(anchor)
        mode = row["root_mode"]
        events = _root_events(bone.ppf_bone_physics, start_frame, fps)
        if mode == "FREE" and not events:
            continue
        if mode not in ("KINEMATIC", "STATIC", "FREE"):
            raise ValueError(f"BLENDER_BONE_INVALID: unsupported root mode {mode}")
        source_mode = getattr(bone.ppf_bone_physics, "root_motion_source", "POSE")
        if source_mode not in ("POSE", "OBJECT"):
            raise ValueError("BLENDER_BONE_INVALID: unsupported root motion source")
        initial = _frame(row["bone_world_matrix_blender"] if source_mode == "POSE" else row["proxy_world_matrix_blender"], "root initial")
        root = {"body_uuid": row["body_uuid"], "mode": "dynamic" if mode == "FREE" else mode.lower(), "authoring": "object_world_trs",
                "initial_frame": initial, "keys": []}
        if events:
            root["control_events"] = events
        row["root_motion"] = root
        if bone.ppf_bone_physics.physical_model == "SOFT":
            row['fem_root_control_events_v1' if fem_event_scene else 'fem_root_control_v1'] = True
        if result.get('body_controls_v1'):
            row['root_motion_source'] = source_mode
        roots.append((arm, bone.name, row, root, source_mode))
    if sampled_auto_anchors and int(state.frame_count) < 2:
        raise ValueError("BLENDER_BONE_INVALID: automatic parent follow requires at least two sampled frames")
    return result, bone_index, tuple(scale_sources), sampled_auto_anchors, roots


def finish_transport_bones(plan):
    from .automatic_parent_anchors import finish_parent_track
    result, bone_index, _, sampled_auto_anchors, _ = plan
    for request in sampled_auto_anchors.values():
        key, value = finish_parent_track(request, request['anchors'][0]['parent_world_matrix_blender'])
        for anchor in request['anchors']:
            anchor[key] = deepcopy(value)
    # Author-only intent; installers remove it before the unchanged native wire.
    limit_intent=[]
    for row in result['bindings']:
        if not (row.get('parent_body_uuid') or row.get('automatic_parent_anchor_v1')):continue
        bone=bone_index[row['armature_uuid']][row['bone_uuid']]
        settings=bone.ppf_bone_physics
        if settings.soft_limit_policy!='AUTO_INERTIA' or not (any(settings.soft_limits_enabled) or settings.soft_limit_mode=='CONE'):continue
        if settings.joint_mode not in {'SOFT','BALL_FIXED'}:raise ValueError('SOFT_LIMIT_AUTO_UNSUPPORTED: compliant joint required')
        frequency=float(settings.soft_limit_frequency_hz);strength=float(settings.soft_limit_strength)
        if not math.isfinite(frequency) or frequency<=0 or not math.isfinite(strength) or strength<0:
            raise ValueError('SOFT_LIMIT_AUTO_INVALID: finite positive frequency/nonnegative strength required')
        cone=settings.soft_limit_mode=='CONE'
        flags=[bool(v) for v in settings.soft_limits_enabled]
        if cone:flags[1:]=[False,False]
        spec=dict(joint_uuid=row['joint_uuid'],frequency_hz=frequency,strength=strength,enabled_axes=flags)
        if cone:spec['swing_cone']=True
        limit_intent.append(spec)
    if limit_intent:result['soft_limit_auto_v1']={'version':1,'joints':limit_intent}
    return result


def encode_bones_steps(context, state, fps, start_frame, *, cone_restoring_joint_ids=None,
                 mmd_automatic_channels=None):
    """Capture authoring at start and root motion at the original frame times."""
    plan = prepare_transport_bones(
        context, state, fps, start_frame,
        cone_restoring_joint_ids=cone_restoring_joint_ids,
        mmd_automatic_channels=mmd_automatic_channels,
    )
    if plan is None:
        return None
    _, _, scale_sources, anchors, roots = plan
    if scale_sources or roots or anchors:
        from ..frame_time import set_scene_frame
        scene = context.scene
        previous, subframe = scene.frame_current, scene.frame_subframe
        try:
            for offset in range(int(state.frame_count)):
                sample_frame = start_frame + offset
                scene.frame_set(sample_frame)
                context.view_layer.update()
                sample_transport_bone_frame(
                    scale_sources, anchors.values(), roots, offset, fps, sample_frame)
                yield ("Sampling bone motion: {done}/{total} frames", offset + 1, int(state.frame_count))
        finally:
            set_scene_frame(scene, previous, subframe)
            context.view_layer.update()
    return finish_transport_bones(plan)


def encode_bones(context, state, fps, start_frame, **options):
    from ..utils import finish_encode_steps
    return finish_encode_steps(encode_bones_steps(context, state, fps, start_frame, **options))
