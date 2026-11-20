"""Transport bone intent through the original parameter upload and hash."""
from __future__ import annotations

from copy import deepcopy
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
    yield from getattr(action, 'fcurves', ())
    for layer in getattr(action, 'layers', ()):
        for strip in layer.strips:
            for bag in getattr(strip, 'channelbags', ()):
                if slot is None or bag.slot_handle == slot.handle:
                    yield from bag.fcurves


def _pose_scale_sources(context, *, explicit_only=False):
    """Collect the frozen scale references needed by the frame sampler.

    Check FREE and STATIC bodies too, including ancestors that are not physical.
    Explicit scale curves/drivers fail before sampling, including subframe curves.
    """
    from ...models.bone_bindings import bone_world_frame_scale
    sources = {}
    sample_needed = False
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
        paths = {arm.pose.bones[name].path_from_id('scale') for name in names} | {'scale', 'delta_scale'}
        if ad is not None:
            actions = [(ad.action, getattr(ad, 'action_slot', None))]
            actions.extend((strip.action, getattr(strip, 'action_slot', None)) for track in ad.nla_tracks for strip in track.strips if strip.action)
            curves = [curve for action, slot in actions for curve in _action_curves(action, slot)]
            curves.extend(ad.drivers)
            if any(not curve.mute and curve.data_path in paths for curve in curves):
                raise ValueError(f'{arm.name}: animated pose/object scale is unsupported; use a static positive uniform pose scale')
            sample_needed = True
        sample_needed |= bool(arm.constraints) or any(arm.pose.bones[name].constraints for name in names)
        ancestor = arm.parent
        while ancestor is not None:
            # An animated object ancestor can change the armature world frame
            # even when the armature has no action of its own.
            sample_needed = True
            ad = ancestor.animation_data
            if ad is not None:
                actions = [(ad.action, getattr(ad, 'action_slot', None))]
                actions.extend((strip.action, getattr(strip, 'action_slot', None)) for track in ad.nla_tracks for strip in track.strips if strip.action)
                curves = [curve for action, slot in actions for curve in _action_curves(action, slot)] + list(ad.drivers)
                if any(not curve.mute and curve.data_path in {'scale', 'delta_scale'} for curve in curves):
                    raise ValueError(f'{ancestor.name}: animated ancestor object scale is unsupported')
            ancestor = ancestor.parent
        if explicit_only:
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


def _validate_static_pose_scales(context, state, start_frame, *, explicit_only=False):
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
    finally:
        set_scene_frame(scene, previous, subframe)
        context.view_layer.update()


def encode_bones(context, state, fps, start_frame):
    """Capture authoring at start and root motion at the original frame times."""
    from ...models.bone_bindings import (
        bone_frame_cache, bone_world_matrix, collect_bindings, collect_body_controls_contract,
    )
    from .automatic_parent_anchors import prepare_parent_track, sample_parent_track, finish_parent_track
    source = collect_bindings(context)
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
    fem_event_scene = any(r.get('enabled') and any(b.ppf_bone_physics.body_uuid == r['body_uuid'] and b.ppf_bone_physics.physical_model == 'SOFT' and b.ppf_bone_physics.root_control_events for arm in armatures.values() if arm is not None for b in arm.data.bones) for r in result['bindings'])
    # Physical joints use the v3 restoring path; generated parent anchors are
    # real endpoints.  An unparented root alone does not create a joint.
    physical_ids = {bone.ppf_bone_physics.body_uuid for arm in armatures.values() if arm is not None
                 for bone in arm.data.bones if bone.ppf_bone_physics.enabled
                 and bone.ppf_bone_physics.physical_model in ('RIGID', 'SOFT')}
    rigid_ids = {bone.ppf_bone_physics.body_uuid for arm in armatures.values() if arm is not None
                 for bone in arm.data.bones if bone.ppf_bone_physics.enabled
                 and bone.ppf_bone_physics.physical_model == 'RIGID'}
    roots = []
    sampled_auto_anchors = {}
    for row in result["bindings"]:
        _frame(row["bone_world_matrix_blender"], row["bone_name"])
        _frame(row["proxy_world_matrix_blender"], row["bone_name"] + " proxy")
        arm = armatures[row["armature_uuid"]]
        if arm is None or arm.ppf_armature_uuid != row["armature_uuid"]:
            raise ValueError("BLENDER_BONE_INVALID: armature UUID cannot be resolved")
        bone = next((b for b in arm.data.bones if b.ppf_bone_physics.bone_uuid == row["bone_uuid"]), None)
        if bone is None:
            raise ValueError("BLENDER_BONE_INVALID: root bone UUID cannot be resolved")
        drives = _drives(bone.ppf_bone_physics, start_frame, fps) if (row.get("parent_body_uuid") or row.get("automatic_parent_anchor_v1")) else None
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
    # Frozen scale and motion both use the evaluated pose at the same frame.
    if scale_sources or roots or sampled_auto_anchors:
        if sampled_auto_anchors and int(state.frame_count) < 2:
            raise ValueError("BLENDER_BONE_INVALID: automatic parent follow requires at least two sampled frames")
        from ..frame_time import set_scene_frame
        scene = context.scene
        previous, subframe = scene.frame_current, scene.frame_subframe
        try:
            for offset in range(int(state.frame_count)):
                sample_frame = start_frame + offset
                scene.frame_set(sample_frame)
                context.view_layer.update()
                with bone_frame_cache():
                    _validate_pose_scales_at_frame(scale_sources, sample_frame)
                    for request in sampled_auto_anchors.values():
                        sample_parent_track(request, offset, fps,
                                            request['anchors'][0]['parent_world_matrix_blender'])
                    for arm, bone_name, row, root, source_mode in roots:
                        if root["mode"] == "static" and offset > 0 and not root.get("control_events"):
                            continue
                        if source_mode == "POSE":
                            world = bone_world_matrix(arm, arm.data.bones[bone_name])
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
        finally:
            set_scene_frame(scene, previous, subframe)
            context.view_layer.update()
    for request in sampled_auto_anchors.values():
        key, value = finish_parent_track(request, request['anchors'][0]['parent_world_matrix_blender'])
        for anchor in request['anchors']:
            anchor[key] = deepcopy(value)
    # Author-only intent; installers remove it before the unchanged native wire.
    limit_intent=[]
    for row in result['bindings']:
        if not (row.get('parent_body_uuid') or row.get('automatic_parent_anchor_v1')):continue
        arm=armatures[row['armature_uuid']]
        bone=next(b for b in arm.data.bones if b.ppf_bone_physics.bone_uuid==row['bone_uuid'])
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
