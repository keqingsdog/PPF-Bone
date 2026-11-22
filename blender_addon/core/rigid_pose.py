"""Validate accepted native frame poses and display them on bound bones.

The native sidecar measures the published vertex frame. This module does not
fit geometry, solve constraints, change source animation, or invent frames.
"""
from contextlib import contextmanager
import hashlib
import io
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np

_suspended = 0
_busy = False
_cache = {}
_scene_cache_keys = {}
_completion = {}
_verified_playback = {}
_capture_for_bake = None
_completion_serial = 0


def clear_pose_completion(scene=None):
    if scene is None:
        _completion.clear()
        _verified_playback.clear()
    else:
        _completion.pop(scene.as_pointer(), None)
        _verified_playback.pop(scene.as_pointer(), None)


def pose_completion(scene):
    """Return only a scene-local immutable completed native pose packet."""
    packet = _completion.get(scene.as_pointer())
    if packet is None:
        return None
    if (packet[2] != scene.session_uid or packet[3] != scene.as_pointer()
            or packet[4] != scene.get(CACHE_KEY) or packet[5] != scene.frame_current
            or packet[6] != scene.frame_subframe):
        clear_pose_completion(scene)
        return None
    return packet




@contextmanager
def capture_verified_playback_for_bake(scene, key, frame_index, digest):
    global _capture_for_bake
    previous = _capture_for_bake
    _capture_for_bake = (scene.as_pointer(), key, frame_index, digest)
    try:
        yield
    except BaseException:
        _verified_playback.pop(scene.as_pointer(), None)
        raise
    finally:
        _capture_for_bake = previous


def _evaluated_packet_matrices(scene, packet, depsgraph):
    rows = []
    for name, session_uid, _arm_uuid, bone_name, _bone_uuid, _body_uuid, _kind, _expected in packet[9]:
        arm = scene.objects.get(name)
        if arm is None or arm.session_uid != session_uid:
            return None
        evaluated = arm.evaluated_get(depsgraph)
        pb = evaluated.pose.bones.get(bone_name)
        if pb is None:
            return None
        world = evaluated.matrix_world @ pb.matrix
        rows.append(tuple(float(world[i][j]) for i in range(4) for j in range(4)))
    return tuple(rows) if rows else None

def verified_playback_matches(scene, key, frame_index, digest, depsgraph, receipt):
    """Prove the last verified handler result survived later frame handlers."""
    packet = pose_completion(scene)
    proof = _verified_playback.pop(scene.as_pointer(), None)
    if packet is None or proof is None:
        return False
    try:
        if (proof[:4] != (packet[1], key, frame_index, digest)
                or proof[6] != json.dumps(receipt, sort_keys=True, separators=(',', ':'))):
            return False
        path = _io_path(Path(key).parent / f'pose_{frame_index}.cbor')
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != proof[4]:
            return False
        return _evaluated_packet_matrices(scene, packet, depsgraph) == proof[5]
    except (OSError, ValueError, RuntimeError, ReferenceError, AttributeError):
        return False

def _publish_pose_completion(scene, receipt, pose, targets, plan):
    """Observer only: failed serialization must never roll back a committed pose."""
    global _completion_serial
    try:
        from .fem_pose import binding_kind
        frame = int(pose['frame_id'])
        start = int(receipt['start_frame'])
        cache_key = scene.get(CACHE_KEY)
        if (not isinstance(cache_key, str) or not cache_key or scene.frame_subframe != 0.0
                or scene.frame_current != start + frame):
            return
        rows = []
        for row, arm, pb in targets:
            actual = plan[row['bone_uuid']][1]
            world = arm.matrix_world @ actual
            values = tuple(float(world[i][j]) for i in range(4) for j in range(4))
            if not all(math.isfinite(value) for value in values):
                raise ValueError('non-finite completed world matrix')
            rows.append((str(arm.name), int(arm.session_uid), str(row['armature_uuid']), str(pb.name),
                         str(row['bone_uuid']), str(row['body_uuid']), str(binding_kind(row)), values))
        _completion_serial += 1
        _completion[scene.as_pointer()] = (1, _completion_serial, int(scene.session_uid), int(scene.as_pointer()),
                                            cache_key, int(scene.frame_current), float(scene.frame_subframe),
                                            frame, start, tuple(rows))
    except Exception as exc:
        clear_pose_completion(scene)
        print('PPF bone pose: completion unavailable:', exc)
        return
    # The UI is an observer: a failed notification never undoes the pose or
    # invalidates the successfully committed core completion packet.
    try:
        from ..ui import bone_overlay
        bone_overlay.native_pose_completed(scene)
    except Exception as exc:
        print('PPF bone pose: overlay notification unavailable:', exc)


CACHE_KEY = "ppf_bone_pose_cache_index"
# The authored ``use_connect`` of every bound bone, stored in the .blend
# itself (a scene ID property) rather than only beside the pose cache, so it
# survives the external cache being deleted or moved. An older cache may
# carry the value in its index instead, which is still read as a fallback.
CONNECT_KEY = "author_use_connect"
CONNECT_PROP = "ppf_bone_pose_author_connect"


def _load(data):
    import cbor2
    stream = io.BytesIO(data)
    value = cbor2.CBORDecoder(stream,read_size=1).decode()
    if stream.read(1):
        raise ValueError("bone pose: trailing CBOR data")
    return value


def _uuid(value):
    if not isinstance(value, str) or str(uuid.UUID(value)) != value:
        raise ValueError("bone pose: invalid UUID")
    return value


def _hash(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("bone pose: invalid digest")
    return value


def _rigid(value):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or np.max(np.abs(matrix[3] - [0, 0, 0, 1])) > 1e-8:
        raise ValueError("bone pose: invalid transform")
    r = matrix[:3, :3]
    if np.max(np.abs(r.T @ r - np.eye(3))) > 5e-4 or abs(np.linalg.det(r) - 1) > 5e-4:
        raise ValueError("bone pose: scale/shear is not a rigid pose")
    return matrix


def _display_armature_transform(value):
    """An armature display transform may have nonuniform scale or shear."""
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all() or
            not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0) or
            not math.isfinite(float(np.linalg.cond(matrix[:3, :3]))) or
            np.linalg.cond(matrix[:3, :3]) > 1e12):
        raise ValueError('bone pose: invalid or singular armature display transform')
    return matrix


def decode_bindings(data):
    from .fem_pose import binding_kind, validate_binding
    envelope = _load(data)
    if not isinstance(envelope, dict) or envelope.get("version") != 1 or envelope.get("kind") != "ppf.blender_bone_bindings":
        raise ValueError("bone pose: unsupported binding receipt")
    value = envelope.get("payload")
    if not isinstance(value, dict) or value.get("schema") != "PPF_BLENDER_BONES_V1":
        raise ValueError("bone pose: unsupported binding schema")
    _uuid(value["scene_uuid"]); _hash(value["scene_inputs_sha256"])
    if type(value["start_frame"]) is not int or not math.isfinite(value["fps"]) or value["fps"] <= 0 or value["world_scaling"] != 1:
        raise ValueError("bone pose: invalid frame scale")
    seen_bones, seen_bodies = set(), set()
    if not isinstance(value["bindings"], list) or not value["bindings"]:
        raise ValueError("bone pose: empty bindings")
    for row in value["bindings"]:
        author_scale = row.get('author_uniform_scale', 1.)
        if (type(author_scale) not in (int, float) or not math.isfinite(author_scale)
                or author_scale <= 1.e-8):
            raise ValueError('bone pose: invalid frozen author uniform scale')
        aid, bid, body = (_uuid(row[key]) for key in ("armature_uuid", "bone_uuid", "body_uuid"))
        if (aid, bid) in seen_bones or body in seen_bodies or row["proxy_object_uuid"] != body:
            raise ValueError("bone pose: duplicate or mismatched binding")
        parent = row.get("parent_bone_uuid")
        if parent and (aid, _uuid(parent)) not in seen_bones:
            raise ValueError("bone pose: bindings must be parent before child")
        if binding_kind(row) == "PDRD":
            if type(row["runtime_body_index"]) is not int or row["runtime_body_index"] < 0:
                raise ValueError("bone pose: invalid runtime index")
        else:
            validate_binding(row)
            _hash(value["fem_payload_sha256"])
        _rigid(row["body_to_bone_initial_solver"])
        _rigid(row["bone_initial_solver"])
        seen_bones.add((aid, bid)); seen_bodies.add(body)
    return value


def _validate_native_bodies_scalar(rows):
    bodies, seen, indices = {}, set(), {}
    for row in rows:
        uid = _uuid(row['body_uuid'])
        if uid in bodies or row['runtime_body_index'] in seen or row['vertex_count'] < 4:
            raise ValueError('bone pose: duplicate or invalid native body')
        r = np.asarray(row['rotation_column_major'], dtype=float).reshape(3, 3, order='F')
        matrix = np.eye(4); matrix[:3, :3] = r; matrix[:3, 3] = row['centroid_m']
        _rigid(matrix)
        bodies[uid] = matrix; seen.add(row['runtime_body_index']); indices[uid] = row['runtime_body_index']
    return bodies, indices


def _validate_native_bodies(rows):
    """Fast valid frames; use original ordered checks for every uncertain case."""
    if not isinstance(rows, (list, tuple)):
        return _validate_native_bodies_scalar(rows)
    try:
        bodies, indices, seen, rotations, centroids = {}, {}, set(), [], []
        for row in rows:
            uid = _uuid(row['body_uuid']); idx = row['runtime_body_index']
            if uid in indices or idx in seen or row['vertex_count'] < 4:
                raise ValueError('fallback')
            rotations.append(np.asarray(row['rotation_column_major'], dtype=float).reshape(3, 3, order='F'))
            centroid = np.asarray(row['centroid_m'], dtype=float)
            if centroid.shape != (3,):
                raise ValueError('fallback')
            centroids.append(centroid)
            indices[uid] = idx; seen.add(idx)
        if not rotations:
            return bodies, indices
        r = np.stack(rotations); c = np.stack(centroids)
        with np.errstate(over='ignore', invalid='ignore'):
            orth = np.max(np.abs(np.swapaxes(r, 1, 2) @ r - np.eye(3)), axis=(1, 2))
            det = np.abs(np.linalg.det(r) - 1)
        if (not np.isfinite(r).all() or not np.isfinite(c).all() or
                not np.isfinite(orth).all() or not np.isfinite(det).all() or
                np.any(orth >= 5e-4 - 1e-10) or np.any(det >= 5e-4 - 1e-10)):
            raise ValueError('fallback')
        for uid, rotation, centroid in zip(indices, r, c):
            matrix = np.eye(4); matrix[:3, :3] = rotation; matrix[:3, 3] = centroid
            bodies[uid] = matrix
        return bodies, indices
    except Exception:
        return _validate_native_bodies_scalar(rows)


def validate_frame(pose, receipt, frame_id, vertex_bytes=None):
    from .fem_pose import binding_kind, validate_combined
    if pose.get("kind") == "PPF_DUAL_NATIVE_POSE_V2":
        return validate_combined(pose, receipt, frame_id, vertex_bytes)
    if pose.get("version") != 1 or pose.get("kind") != "PdrdFramePoseMeasurement" or pose.get("measurement") != "read_only_native_pdrd_fit_of_vertex_frame" or pose.get("coordinate_system") != "solver_y_up_meters" or pose.get("constrained_root_state_authority") != "not_reported":
        raise ValueError("bone pose: unsupported native measurement")
    if pose["scene_uuid"] != receipt["scene_uuid"] or pose["scene_inputs_sha256"] != receipt["scene_inputs_sha256"] or pose["frame_id"] != frame_id:
        raise ValueError("bone pose: scene or frame identity mismatch")
    _hash(pose["vertex_frame_sha256"])
    if vertex_bytes is not None:
        vertices = np.frombuffer(vertex_bytes, dtype="<f4")
        if not vertex_bytes or len(vertex_bytes) % 12 or not np.isfinite(vertices).all() or len(vertex_bytes) != pose["vertex_frame_byte_length"] or hashlib.sha256(vertex_bytes).hexdigest() != pose["vertex_frame_sha256"]:
            raise ValueError("bone pose: vertex frame generation mismatch")
    t, a, b = (pose[key] for key in ("frame_time_seconds", "accepted_bracket_start_seconds", "accepted_solver_time_seconds"))
    if not all(math.isfinite(v) for v in (t, a, b)) or not 0 <= a <= t <= b or abs(t - frame_id / receipt["fps"]) > 1e-8:
        raise ValueError("bone pose: invalid accepted frame time")
    source, steps = pose["source"], pose["accepted_steps"]
    valid = ((source == "InitialFrame" and frame_id == 0 and steps == 0 and t == a == b == 0)
             or (source == "AcceptedEndpoint" and frame_id > 0 and steps > 0 and a < b and t == b)
             or (source == "InterpolatedAcceptedBracket" and frame_id > 0 and steps > 0 and a < b and t < b))
    if not valid:
        raise ValueError("bone pose: unaccepted frame source")
    bodies, indices = _validate_native_bodies(pose['bodies'])
    for row in receipt["bindings"]:
        if binding_kind(row) != "PDRD":
            continue
        if row["body_uuid"] not in indices or indices[row["body_uuid"]] != row["runtime_body_index"]:
            raise ValueError("bone pose: native body differs from build binding")
    return bodies


def decode_frame(data, vertices, frame_id, receipt):
    pose = _load(data)
    if not isinstance(pose, dict):
        raise ValueError("bone pose: expected native map")
    validate_frame(pose, receipt, frame_id, vertices)
    return {"receipt": receipt, "pose": pose, "native_bytes": data}


def _resolve_armatures(scene, receipt, *, current):
    candidates = {}
    requested={row['armature_uuid'] for row in receipt['bindings']}
    for obj in scene.objects:
        if obj.type == "ARMATURE" and getattr(obj, "ppf_armature_uuid", "") in requested and (not current or any(b.ppf_bone_physics.enabled for b in obj.data.bones)):
            uid = obj.ppf_armature_uuid
            candidates.setdefault(uid,[]).append(obj)
    arms={}
    for uid,objects in candidates.items():
        if len(objects)>1 and not current:
            # Recovery may include the now-disabled authored rig. A disabled
            # object copy keeps pointers into the original's owned proxies;
            # only the actual proxy parent may receive authored channel writes.
            rows=[row for row in receipt['bindings'] if row['armature_uuid']==uid]
            def owns_all_rows(arm):
                for row in rows:
                    bones=[b for b in arm.data.bones if b.ppf_bone_physics.bone_uuid==row['bone_uuid']]
                    if len(bones)!=1:return False
                    settings=bones[0].ppf_bone_physics;proxy=settings.proxy
                    if proxy is None or settings.body_uuid!=row['body_uuid']:return False
                    parent = proxy.parent
                    if parent != arm and not (
                        parent is not None and parent.type == 'EMPTY'
                        and parent.parent == arm
                        and parent.get('ppf_bone_proxy_object_container') == arm.ppf_armature_uuid
                    ):return False
                    if proxy.get('_solver_uuid')!=row['body_uuid'] or proxy.get('ppf_bone_uuid')!=row['bone_uuid']:return False
                return bool(rows)
            objects=[arm for arm in objects if owns_all_rows(arm)]
        if len(objects)!=1:raise ValueError('bone pose: duplicate or unavailable armature UUID')
        arms[uid]=objects[0]
    return arms


def _targets(scene, receipt, *, current=True):
    from .fem_pose import binding_kind, FEM
    from .encoder.pmx_joints import effective_parent_joint_pair, parent_joint_pairs
    arms=_resolve_armatures(scene,receipt,current=current)
    # Resolve once within this read-only call. No RNA reference survives the
    # call, so Edit-mode rebuilds/Undo and later author changes still revalidate.
    by_arm = {}
    pmx_pairs_by_arm = {}
    for uid, arm in arms.items():
        by_uuid = {}
        for pb in arm.pose.bones:
            by_uuid.setdefault(pb.bone.ppf_bone_physics.bone_uuid, []).append(pb)
        by_arm[uid] = by_uuid
        if current:
            pmx_pairs_by_arm[uid] = parent_joint_pairs(arm)
    checked_transforms = set()
    targets = []
    for row in receipt["bindings"]:
        arm = arms.get(row["armature_uuid"])
        bones = by_arm.get(row["armature_uuid"], {}).get(row["bone_uuid"], [])
        if len(bones) != 1 or (current and (not bones[0].bone.ppf_bone_physics.enabled or bones[0].bone.ppf_bone_physics.body_uuid != row["body_uuid"])):
            raise ValueError("bone pose: author binding changed; transfer again")
        pb = bones[0]
        if current:
            has_physical_parent = (pb.bone.ppf_bone_physics.use_hierarchy_joint and pb.parent is not None
                                   and pb.parent.bone.ppf_bone_physics.enabled
                                   and effective_parent_joint_pair(pb.bone, pmx_pairs_by_arm[row["armature_uuid"]]) is None)
            parent = pb.parent.bone.ppf_bone_physics.bone_uuid if has_physical_parent else None
            if parent != row.get("parent_bone_uuid"):
                raise ValueError("bone pose: parent binding changed")
        if current and any(c.mute is False for c in pb.constraints):
            raise ValueError(f"{pb.name}: disable pose constraints before applying physical frames")
        if current:
            kind = binding_kind(row)
            key = (row["armature_uuid"], kind)
            if key not in checked_transforms:
                if kind == FEM:
                    _display_armature_transform(arm.matrix_world)
                else:
                    from ..models.bone_bindings import similarity_frame_scale
                    similarity_frame_scale(arm.matrix_world, arm.name)
                checked_transforms.add(key)
        targets.append((row, arm, pb))
    return targets


def has_enabled_bones(scene):
    return any(b.ppf_bone_physics.enabled for o in scene.objects if o.type == "ARMATURE" for b in o.data.bones if hasattr(b, "ppf_bone_physics"))


def _armature_key(arm):
    return getattr(arm, "ppf_armature_uuid", "") or arm.name


def _bound_targets(scene, index):
    """Bound bones of a cache generation, without current-author re-checks."""
    return _targets(scene, index["receipt"], current=False)


def _as_plain(value):
    """A Blender ID property reads back as an IDPropertyGroup, not a dict."""
    from idprop.types import IDPropertyGroup
    if isinstance(value, IDPropertyGroup):
        return value.to_dict()
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _as_plain(to_dict())
    if isinstance(value, dict):
        return {key: _as_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_plain(item) for item in value]
    return value


_connection_scopes = []  # One synchronous apply only; pure values, never RNA.


def _connect_records(scene):
    token = scene.as_pointer()
    if _connection_scopes and token in _connection_scopes[-1]:
        return _connection_scopes[-1][token]
    value = scene.get(CONNECT_PROP)
    if value is None:
        return {}
    plain = _as_plain(value)
    result = plain if isinstance(plain, dict) else {}
    if _connection_scopes:
        _connection_scopes[-1][token] = result
    return result


_NATIVE_POSE_FIELDS = {"location": 3, "rotation_euler": 3,
                       "rotation_quaternion": 4, "rotation_axis_angle": 4, "scale": 3}


def _capture_native_channels(pb):
    return {"rotation_mode": pb.rotation_mode,
            **{name: list(getattr(pb, name)) for name in _NATIVE_POSE_FIELDS}}


def _validate_native_channels(value):
    if not isinstance(value, dict) or value.get("rotation_mode") not in {
            'XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX', 'QUATERNION', 'AXIS_ANGLE'}:
        raise ValueError("Bone author native channels have an invalid rotation mode")
    for name, size in _NATIVE_POSE_FIELDS.items():
        row = value.get(name)
        if (not isinstance(row, (list, tuple)) or len(row) != size
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) or abs(v) > 3.4028234663852886e38 for v in row)):
            raise ValueError("Bone author native channels are malformed: " + name)
    return value


def _restore_native_channels(pb, value):
    value = _validate_native_channels(value)  # Validate the whole row before writing.
    pb.rotation_mode = value["rotation_mode"]
    for name in _NATIVE_POSE_FIELDS:
        setattr(pb, name, value[name])


def _restore_author_pose(pb, index, bone_uuid):
    from mathutils import Matrix
    if "author_channels" in index:
        channels = index["author_channels"]
        if not isinstance(channels, dict) or bone_uuid not in channels:
            raise ValueError("Bone author native channels are missing")
        _restore_native_channels(pb, channels[bone_uuid])
    else:
        pb.matrix_basis = Matrix(index["author_basis"][bone_uuid])


def _record_author_connections(scene, targets, index=None):
    """Persist the authored connection of every bound bone, before any detach.

    Keyed by armature UUID then bone UUID, with the name and parent UUID kept
    beside it so a bone can still be identified after a rename or a re-parent. A
    bone that already has a record keeps it: the recorded value IS the authored
    one, and re-reading the live flag after a display detach would store our own
    edit as if the author had made it.
    """
    if index is not None and "author_channels" in index:
        channels = index["author_channels"]
        if not isinstance(channels, dict):
            raise ValueError("Bone author native channels must be a table")
        for row, _arm, _pb in targets:
            _validate_native_channels(channels.get(row["bone_uuid"]))
    records = dict(_connect_records(scene))
    changed = False
    for row, arm, pb in targets:
        per_bone = dict(records.get(_armature_key(arm)) or {})
        if row["bone_uuid"] in per_bone:
            # Old cache records predate the display-only inheritance override.
            # Capture the still-authored enum once, never a later display value.
            record = per_bone[row["bone_uuid"]]
            record = dict(record) if isinstance(record, dict) else {"connect": bool(record)}
            if "channels" not in record and index is not None and "author_channels" in index:
                record["channels"] = _validate_native_channels(index["author_channels"].get(row["bone_uuid"]))
                per_bone[row["bone_uuid"]] = record
                records[_armature_key(arm)] = per_bone
                changed = True
            if "inherit_scale" not in record and not (scene.get(CACHE_KEY) and pb.bone.inherit_scale == 'NONE'):
                record["inherit_scale"] = pb.bone.inherit_scale
                per_bone[row["bone_uuid"]] = record
                records[_armature_key(arm)] = per_bone
                changed = True
            continue
        author_connect = (_author_connection(records, index.get(CONNECT_KEY) or {}, arm, row, pb)
                          if index is not None else None)
        basis = (index.get('author_basis') or {}).get(row['bone_uuid']) if index is not None else None
        if basis is None:
            basis = [list(values) for values in pb.matrix_basis]
        per_bone[row["bone_uuid"]] = {"name": pb.name, "parent_uuid": row.get("parent_bone_uuid") or "",
                                      "connect": bool(pb.bone.use_connect) if author_connect is None else author_connect,
                                      # The authored pose channel too, so the authored
                                      # state survives the external cache disappearing:
                                      # a detached head makes a display basis effective,
                                      # and without this record it could leak into a later
                                      # authored export once the cache is gone.
                                      "basis": [float(v) for values in basis for v in values]}
        if index is not None and "author_channels" in index:
            channel = index["author_channels"].get(row['bone_uuid'])
            per_bone[row['bone_uuid']]['channels'] = _validate_native_channels(channel)
        elif index is None and not scene.get(CACHE_KEY):
            per_bone[row['bone_uuid']]['channels'] = _capture_native_channels(pb)
        per_bone[row['bone_uuid']]['inherit_scale'] = pb.bone.inherit_scale
        records[_armature_key(arm)] = per_bone
        changed = True
    if changed:
        scene[CONNECT_PROP] = records
        if _connection_scopes:
            _connection_scopes[-1][scene.as_pointer()] = records
    return records


def _author_connection(records, legacy, arm, row, pb):
    """The authored Boolean for one bone, or None when it was never recorded.

    None means "unknown", and an unknown bone is left exactly as it is: inferring
    ``True`` from a head this display already detached would invent author intent.
    """
    record = (records.get(_armature_key(arm)) or {}).get(row["bone_uuid"])
    if record is None:
        fallback = legacy.get(_armature_key(arm))
        record = (fallback or {}).get(pb.name) if isinstance(fallback, dict) else None
    if record is None:
        return None
    if isinstance(record, dict):
        if record.get("name") not in (None, pb.name) or record.get("parent_uuid") not in (None, row.get("parent_bone_uuid") or ""):
            return None
        return bool(record.get("connect"))
    return bool(record)


def _connection_plan(scene, index, targets, *, connected):
    """``{armature: {bone: use_connect}}`` for the bound bones that differ."""
    from .fem_pose import binding_kind, FEM
    records = _connect_records(scene)
    legacy = index.get(CONNECT_KEY) or {}
    plan = {}
    for row, arm, pb in targets:
        author = _author_connection(records, legacy, arm, row, pb)
        if author is None:
            continue
        wanted = author
        if bool(pb.bone.use_connect) != wanted:
            plan.setdefault(arm, {})[pb.name] = wanted
    return plan


# Pending entries contain only Python values, never retained Blender RNA objects.
_pending_connection_recoveries = {}
_exported_connection_recoveries = set()
_connection_recovery_generation = 0
_connection_recovery_load_generation = 0


def _connection_identity(arm):
    try:
        object_uid, data_uid = int(arm.session_uid), int(arm.data.session_uid)
        if object_uid and data_uid:
            return ("uid", object_uid, data_uid)
    except (AttributeError, ReferenceError):
        pass
    return ("pointer", arm.as_pointer(), arm.data.as_pointer())


def _connection_state(arm):
    edit = arm.mode == "EDIT"
    bones = arm.data.edit_bones if edit else arm.data.bones
    result = {}
    for bone in bones:
        head = bone.head if edit else bone.head_local
        tail = bone.tail if edit else bone.tail_local
        matrix = bone.matrix if edit else bone.matrix_local
        result[bone.name] = {
            "parent": bone.parent.name if bone.parent else None,
            "connect": bool(bone.use_connect),
            "head": tuple(head), "tail": tuple(tail),
            "matrix": tuple(tuple(row) for row in matrix),
        }
    return {"mode": arm.mode, "bones": result}


def _connection_pure_geometry(geometry):
    return {name: (tuple(head), tuple(tail), tuple(tuple(row) for row in matrix), roll)
            for name, (head, tail, matrix, roll) in geometry.items()}


def pending_connection_recoveries():
    """Return diagnostic copies, including stale or conflicting transactions."""
    import copy
    return copy.deepcopy(list(_pending_connection_recoveries.values()))


def _set_connection_resolved(record, value):
    if record["resolved"] != value:
        record["resolved"] = value
        record["revision"] += 1
    if value:
        record["resolved_generation"] = _connection_recovery_generation


def _reactivate_connection_recoveries(arm):
    identity = _connection_identity(arm)
    matching = [record for record in _pending_connection_recoveries.values()
                if record["identity"] == identity
                and record["load_generation"] == _connection_recovery_load_generation]
    for record in matching:
        if record["resolved"] and record["resolved_generation"] != _connection_recovery_generation:
            if identity[0] == "pointer":
                _set_connection_resolved(record, False)  # conservatively block across Undo
            elif any(_connection_state(arm)["bones"] == failed["bones"]
                     for failed in record["failed_states"]):
                _set_connection_resolved(record, False)
    return matching


def pending_connection_status(arm=None):
    if arm is not None:
        _reactivate_connection_recoveries(arm)
    current = [record for record in _pending_connection_recoveries.values()
               if not record["resolved"] and record["load_generation"] == _connection_recovery_load_generation]
    names = [record["object_name"] for record in current]
    matched = bool(arm and any(record["identity"] == _connection_identity(arm) for record in current))
    return names, matched


def export_pending_connection_recoveries():
    """Save unresolved diagnostic data before teardown; never auto-apply it."""
    pending = [record for record in _pending_connection_recoveries.values()
               if (record["transaction_id"], record["revision"]) not in _exported_connection_recoveries]
    if not pending:
        return None
    import bpy, json, uuid
    from pathlib import Path
    records = [{k: v for k, v in record.items() if k != "identity"}
               for record in pending]
    destination = None
    try:
        config_dir = bpy.utils.user_resource('CONFIG', path='ppf/connection_recovery', create=True)
        if not config_dir or not Path(config_dir).is_absolute():
            raise RuntimeError('Blender returned an empty or non-absolute configuration directory')
        directory = Path(config_dir)
        destination = directory / ("connection_recovery_" + uuid.uuid4().hex + ".json")
        with destination.open('x', encoding='utf-8') as stream:
            json.dump({"format": "PPF_CONNECTION_RECOVERY_DIAGNOSTIC_V1",
                       "note": "Diagnostic only; identities are not persisted and no automatic restoration is safe.",
                       "transactions": records}, stream, indent=2)
        _exported_connection_recoveries.update((record["transaction_id"], record["revision"]) for record in pending)
        print(f"Unresolved bone connection recovery saved to {destination}")
        try:
            from ..models.console import console
            console.write(f"Unresolved bone connection recovery saved to {destination}")
        except Exception:
            pass
        return str(destination)
    except Exception as exc:
        message = f"ERROR: unresolved bone connection recovery could not be saved to {destination}: {exc}"
        print(message)
        try:
            from ..models.console import console
            console.write(message)
        except Exception:
            pass
        raise RuntimeError(message) from exc


def recover_pending_connections(arm):
    """Resolve one exact failed state while retaining history for Undo/Redo."""
    identity = _connection_identity(arm)
    matching = _reactivate_connection_recoveries(arm)
    state = _connection_state(arm)
    candidates = [record for record in matching if not record["resolved"] and
                  (state["bones"] == record["original_state"]["bones"] or any(state["bones"] == failed["bones"]
                   for failed in record["failed_states"]))]
    if len(candidates) != 1:
        raise RuntimeError("Pending connection recovery unavailable, conflicting or ambiguous")
    record = candidates[0]
    if identity[0] == "pointer" and record["generation"] != _connection_recovery_generation:
        raise RuntimeError("Pending connection recovery identity is stale after undo or redo")
    if state["bones"] == record["original_state"]["bones"]:
        _set_connection_resolved(record, True)
        return "already_restored"
    if not record["trusted"]:
        raise RuntimeError("Pending connection recovery conflicts with current rig state")
    try:
        _apply_connection_values(arm, record["connections"],
                                 restore_geometry=record["geometry"],
                                 restore_mode=record["original_mode"])
        if _connection_state(arm) != record["original_state"]:
            raise RuntimeError("Connection recovery verification failed")
    except BaseException as exc:
        record["revision"] += 1
        record["failure"] = str(exc)
        try:
            record["post"] = _connection_state(arm)
            if record["post"] not in record["failed_states"]:
                record["failed_states"].append(record["post"])
            record["trusted"] = True
        except Exception:
            record["trusted"] = False
        raise
    _set_connection_resolved(record, True)
    return True


def _apply_connection_values(arm, values, *, restore_geometry=None, capture_geometry=None, restore_mode=None):
    """Use a view layer containing this rig, including non-active scenes."""
    import bpy
    scene = bpy.context.scene
    layer = bpy.context.view_layer
    if layer is not None and arm.name in layer.objects:
        return _apply_connection_values_in_layer(arm, values, restore_geometry=restore_geometry, capture_geometry=capture_geometry, restore_mode=restore_mode)
    owners = list(arm.users_scene)
    owners.sort(key=lambda owner: owner != scene)
    for owner in owners:
        for candidate in owner.view_layers:
            if arm.name in candidate.objects:
                candidate.update()
                with bpy.context.temp_override(scene=owner, view_layer=candidate):
                    return _apply_connection_values_in_layer(arm, values, restore_geometry=restore_geometry, capture_geometry=capture_geometry, restore_mode=restore_mode)
    raise RuntimeError("bone pose connections: rig is excluded from every view layer")


def _apply_connection_values_in_layer(arm, values, *, restore_geometry=None, capture_geometry=None, restore_mode=None):
    """Set ``use_connect`` on EditBones; return the previous values.

    Main-thread only, and the one public route: ``Bone.use_connect`` is read-only
    at runtime. Normal calls write that flag; rollback may also restore rest
    geometry. The armature's mode, selection and active bone are restored.
    """
    import bpy
    view_layer = bpy.context.view_layer
    previous = {}
    was_selected = arm.select_get()
    active = arm.data.bones.active
    was_active_name=active.name if active is not None else None
    del active
    previous_active = view_layer.objects.active
    layer_selection = [(obj, obj.select_get()) for obj in view_layer.objects]
    pose_objects = [obj for obj in view_layer.objects if obj.type == "ARMATURE" and obj.mode == "POSE"
                    and not (obj == arm and restore_mode and restore_mode != "POSE")]
    if any(obj != arm and obj.mode == "EDIT" for obj in view_layer.objects):
        raise RuntimeError("bone pose connections: another object is being edited")
    previous_mode = restore_mode or (arm.mode if arm.mode in {"OBJECT", "POSE"} else "OBJECT")

    def edit():
        view_layer.objects.active = arm
        arm.select_set(True)
        with bpy.context.temp_override(object=arm, active_object=arm):
            bpy.ops.object.mode_set(mode="EDIT")
            try:
                if capture_geometry is not None:
                    # Save actual EditBone roll once, before mutation. Rebuilding
                    # roll from a rest matrix drifts near the negative Y axis.
                    for bone in arm.data.edit_bones:
                        old = capture_geometry.get(bone.name)
                        if old is not None and old[3] is None:
                            capture_geometry[bone.name] = (
                                bone.head.copy(), bone.tail.copy(),
                                bone.matrix.copy(), float(bone.roll))
                if restore_geometry is not None:
                    # Connected EditBone heads follow their parent tails. Restore
                    # all rest geometry while disconnected, then the flags.
                    for bone in arm.data.edit_bones:
                        bone.use_connect = False
                    for name, (head, tail, matrix, roll) in restore_geometry.items():
                        bone = arm.data.edit_bones.get(name)
                        if bone is not None:
                            if roll is None:
                                bone.matrix = matrix
                            bone.head = head
                            bone.tail = tail
                            if roll is not None:
                                bone.roll = roll
                for name, value in values.items():
                    bone = arm.data.edit_bones.get(name)
                    if bone is None:
                        continue
                    previous.setdefault(name, bool(bone.use_connect))
                    if bool(bone.use_connect) != bool(value):
                        bone.use_connect = bool(value)
            finally:
                bpy.ops.object.mode_set(mode=previous_mode)

    try:
        edit()
    except RuntimeError:
        # A timer-driven tick may hold no window context; retry with one.
        window = bpy.context.window
        area = next((a for a in window.screen.areas if a.type == "VIEW_3D"), None) if window else None
        if window is None or area is None:
            raise
        with bpy.context.temp_override(window=window, area=area, scene=bpy.context.scene, view_layer=view_layer):
            edit()
    finally:
        try:
            arm.data.bones.active = arm.data.bones.get(was_active_name) if was_active_name else None
        except Exception:
            pass
        try:
            if pose_objects:
                for obj, _ in layer_selection:
                    obj.select_set(obj in pose_objects)
                pose_active = previous_active if previous_active in pose_objects else pose_objects[0]
                view_layer.objects.active = pose_active
                with bpy.context.temp_override(object=pose_active, active_object=pose_active):
                    if pose_active.mode != "POSE":
                        bpy.ops.object.mode_set(mode="POSE")
        finally:
            for obj, selected in layer_selection:
                obj.select_set(selected)
            view_layer.objects.active = previous_active
    return previous


def _write_connections(plan):
    """Apply a connection plan atomically, rolling back a partial failure."""
    if not plan:
        return False
    import bpy
    view_layer = bpy.context.view_layer
    previous_active = view_layer.objects.active
    selection = [(obj, obj.select_get()) for obj in view_layer.objects]
    applied = []
    for arm in plan:
        identity = _connection_identity(arm)
        matching = _reactivate_connection_recoveries(arm)
        blocked = any(not record["resolved"] for record in matching)
        # A changed object or data identity must not inherit unresolved edits.
        blocked = blocked or any(not record["resolved"] and
                                 record["load_generation"] == _connection_recovery_load_generation and
                                 record["identity"][0] == identity[0] and
                                 (record["identity"][1] == identity[1] or record["identity"][2] == identity[2])
                                 for record in _pending_connection_recoveries.values())
        if blocked:
            raise RuntimeError("Pending connection recovery; call recover_pending_connections(arm)")
    try:
        for arm, wanted in plan.items():
            if arm.mode == "EDIT":
                continue  # never fight an edit the author has open
            # Capture before calling: failures during edit cleanup can occur
            # after flags changed but before the helper returns.
            previous = {
                name: bool(arm.data.bones[name].use_connect)
                for name in arm.data.bones.keys()
            }
            geometry = {
                bone.name: (bone.head_local.copy(), bone.tail_local.copy(),
                            bone.matrix_local.copy(), None)
                for bone in arm.data.bones
            }
            original_state = _connection_state(arm)
            applied.append((arm, previous, geometry, original_state))
            _apply_connection_values(arm, wanted, capture_geometry=geometry)
    except BaseException:
        # A half-detached rig would be worse than either state: undo what this
        # call already changed, then let the caller see the failure.
        for arm, previous, geometry, original_state in reversed(applied):
            try:
                # A preflight rejection (for example another rig in Edit)
                # changed no flags and must not trigger another mode attempt.
                if arm.mode == original_state["mode"] and all(
                    name in arm.data.bones
                    and bool(arm.data.bones[name].use_connect) == value
                    and (arm.data.bones[name].head_local - geometry[name][0]).length < 1e-7
                    and (arm.data.bones[name].tail_local - geometry[name][1]).length < 1e-7
                    and all(abs(arm.data.bones[name].matrix_local[i][j] - geometry[name][2][i][j]) < 1e-7
                            for i in range(4) for j in range(4))
                    for name, value in previous.items()
                ):
                    continue
                _apply_connection_values(arm, previous, restore_geometry=geometry,
                                         restore_mode=original_state["mode"])
                if _connection_state(arm) != original_state:
                    raise RuntimeError("Connection rollback verification failed")
            except Exception as exc:  # noqa: BLE001
                try:
                    post = _connection_state(arm)
                    trusted = True
                except Exception:
                    post = None
                    trusted = False
                import uuid
                transaction_id = uuid.uuid4().hex
                _pending_connection_recoveries[transaction_id] = {
                    "transaction_id": transaction_id, "revision": 0,
                    "object_name": arm.name, "data_name": arm.data.name,
                    "identity": _connection_identity(arm),
                    "generation": _connection_recovery_generation,
                    "load_generation": _connection_recovery_load_generation,
                    "original_mode": original_state["mode"],
                    "original_state": original_state,
                    "connections": previous, "geometry": _connection_pure_geometry(geometry),
                    "post": post, "failed_states": [post] if trusted else [],
                    "resolved": False, "resolved_generation": None,
                    "trusted": trusted, "failure": str(exc),
                }
                from ..models.console import console
                console.write(f"bone pose connection rollback failed; recover_pending_connections required: {exc}")
        raise
    finally:
        for obj, selected in selection:
            try:
                obj.select_set(selected)
            except Exception:
                pass
        view_layer.objects.active = previous_active
    return bool(applied)


def _rigid_parent_targets(targets):
    """Native rigid global targets never inherit display-only parent scale."""
    from .fem_pose import binding_kind, FEM
    return [pb for row, _arm, pb in targets if binding_kind(row) != FEM and pb.parent is not None]


@contextmanager
def _display_inherit_scales(targets, author_flags, *, restore=True):
    """Plan with representable rigid TRS; retain overrides only after apply."""
    saved = []
    succeeded = False
    try:
        rigid = set(_rigid_parent_targets(targets))
        for row, arm, pb in targets:
            connected, authored = author_flags[(arm.as_pointer(), pb.name)]
            wanted = authored if connected else ('NONE' if pb in rigid else pb.bone.inherit_scale)
            if pb.bone.inherit_scale != wanted:
                saved.append((pb.bone, pb.bone.inherit_scale))
                pb.bone.inherit_scale = wanted
        yield
        succeeded = True
    finally:
        if restore or not succeeded:
            for bone, value in reversed(saved):
                bone.inherit_scale = value


def _restore_author_inherit_scales(scene, targets):
    records = _connect_records(scene)
    for row, arm, pb in targets:
        record = (records.get(_armature_key(arm)) or {}).get(row['bone_uuid'])
        if isinstance(record, dict) and 'inherit_scale' in record:
            pb.bone.inherit_scale = record['inherit_scale']


def free_bound_heads(scene, index, targets):
    """Disconnect bound bones so a physical frame can place their heads."""
    return _write_connections(_connection_plan(scene, index, targets, connected=False))


def restore_authored_connections(scene, index, targets):
    """Put the bound bones back on the connection the author authored."""
    # Restore enum first even if the Edit-mode connection restoration fails.
    _restore_author_inherit_scales(scene, targets)
    _write_connections(_connection_plan(scene, index, targets, connected=True))


def restore_records_now(scene):
    """Restore the authored state from the .blend records alone.

    Used wherever no usable cache index is available: a deleted or malformed
    cache, a save, or an add-on unload. Both the authored connections and the
    authored pose channels come back, so a display pose cannot leak into the
    authored document once the external cache is gone. A bone with no record is
    left exactly as it is.
    """
    from mathutils import Matrix
    records = _connect_records(scene)
    if not records:
        return
    plan = {}
    poses = {}
    for arm in (obj for obj in scene.objects if obj.type == "ARMATURE"):
        by_uuid = records.get(_armature_key(arm)) or {}
        values = {}
        for entry in by_uuid.values():
            if isinstance(entry, dict) and entry.get("name"):
                values[entry["name"]] = bool(entry.get("connect"))
                bone = arm.data.bones.get(entry['name'])
                if bone is not None and 'inherit_scale' in entry:
                    bone.inherit_scale = entry['inherit_scale']
        if values:
            plan[arm] = values
        bones = {pb.bone.name: pb for pb in arm.pose.bones}
        for entry in by_uuid.values():
            if not isinstance(entry, dict):
                continue
            basis = entry.get("basis")
            pb = bones.get(entry.get("name") or "")
            if pb is not None and isinstance(basis, list) and len(basis) == 16:
                channels = entry.get("channels") if "channels" in entry else None
                if "channels" in entry:
                    _validate_native_channels(channels)
                poses[pb] = (Matrix([basis[0:4], basis[4:8], basis[8:12], basis[12:16]]), channels)
    _write_connections(plan)
    # Pose channels last, so a mode round trip cannot undo them.
    for pb, (matrix, channels) in poses.items():
        if channels is not None:
            _restore_native_channels(pb, channels)
        else:
            pb.matrix_basis = matrix


def _prepare_pose(scene, receipt, pose, *, targets=None):
    """Validate all measurements before cache or authored pose mutation."""
    from mathutils import Matrix
    from .transform import zup_to_yup
    from .fem_pose import binding_kind,FEM
    from .fem_scale import measure
    from .fem_bone_mapping import map_bone
    bodies=validate_frame(pose,receipt,pose['frame_id'])
    if targets is None:
        targets=_targets(scene,receipt)
    inverse_axis=zup_to_yup().inverted();desired={};diagnostics={}; arm_axes={}
    for row,arm,pb in targets:
        key=arm.as_pointer()
        if key not in arm_axes: arm_axes[key]=arm.matrix_world.inverted()@inverse_axis
        arm_axis=arm_axes[key]
        settings=pb.bone.ppf_bone_physics
        author_scale = row.get('author_uniform_scale', 1.)
        if binding_kind(row)==FEM:
            axes=np.asarray(row['bone_initial_solver'],float)[:3,:3].T.reshape(-1).tolist()
            metric=(measure(row['fem_scale_reference'],pose['vertex_native_bytes'],expected_rest_axes=axes)
                    if settings.fem_scale_enabled else None)
            solver_target,solver_tail=map_bone(
                row,pose['vertex_native_bytes'],settings.fem_scale_enabled,
                metric['scale_xyz'] if metric else (1.,1.,1.),pb.bone.length * author_scale)
            solver_target[:3,:3] *= author_scale
            target=arm_axis@Matrix(solver_target.tolist())
            tail=arm_axis@Matrix.Translation(solver_tail.tolist())
            diagnostics[row['bone_uuid']]={
                'metric':metric,'tail':tail.translation,
                'applied_scales':tuple(np.linalg.norm(solver_target[:3,:3],axis=0)),
            }
            if 'fem_display_rest_v1' in row and not settings.fem_scale_enabled:
                diagnostics[row['bone_uuid']]['fixed_world_length'] = pb.bone.length * author_scale
        else:
            target=arm_axis@Matrix(bodies[row['body_uuid']].tolist())@Matrix(row['body_to_bone_initial_solver'])
            target = target @ Matrix.Diagonal((author_scale, author_scale, author_scale, 1.))
        desired[row['bone_uuid']]=target
    return desired,diagnostics


def _fem_trs_preserving_local_y(basis):
    """Project a sheared FEM local basis to TRS without moving its bone axis.

    PoseBone only stores location/rotation/scale.  When an inherited parent
    transform makes the inverse-converted FEM basis sheared, generic
    Matrix.decompose() can rotate the local Y axis slightly.  FEM's material
    head/tail mapping depends on that axis, so retain the exact target Y
    column and recover twist from the projected X column.  Reflections are
    represented explicitly by a signed Z scale; degenerate bases fail closed.
    """
    from mathutils import Matrix, Vector

    source = basis.to_3x3()
    columns = [source.col[i].copy() for i in range(3)]
    values = [float(component) for column in columns for component in column]
    if not all(math.isfinite(value) for value in values):
        raise ValueError('FEM pose: non-finite inverse-converted basis')

    location = basis.translation.copy()
    if not all(math.isfinite(float(component)) for component in location):
        raise ValueError('FEM pose: non-finite inverse-converted translation')
    eps = 1e-12
    lengths = [column.length for column in columns]
    if any(not math.isfinite(length) or length <= eps for length in lengths):
        raise ValueError('FEM pose: degenerate inverse-converted basis')

    y = columns[1] / lengths[1]
    x_projected = columns[0] - y * y.dot(columns[0])
    if x_projected.length <= eps * max(lengths):
        # If X is parallel to Y, use projected Z as the stable transverse
        # reference.  This still produces a right-handed rotation basis.
        z_projected = columns[2] - y * y.dot(columns[2])
        if z_projected.length <= eps * max(lengths):
            raise ValueError('FEM pose: no stable transverse axis for TRS projection')
        z_projected.normalize()
        x = y.cross(z_projected)
        if x.length <= eps:
            raise ValueError('FEM pose: degenerate fallback transverse axis')
        x.normalize()
    else:
        x = x_projected.normalized()

    z = x.cross(y)
    if z.length <= eps:
        raise ValueError('FEM pose: projected axes do not form a stable frame')
    z.normalize()
    orientation = z.dot(columns[2])
    if not math.isfinite(orientation) or abs(orientation) <= eps * lengths[2]:
        raise ValueError('FEM pose: ambiguous reflection/handedness in TRS projection')

    rotation_matrix = Matrix.Identity(3)
    rotation_matrix.col[0] = x
    rotation_matrix.col[1] = y
    rotation_matrix.col[2] = z
    rotation = rotation_matrix.to_quaternion()
    scale = Vector((lengths[0], lengths[1], math.copysign(lengths[2], orientation)))
    return location, rotation, scale


def _fem_parent_anchor_target(target, material_tail, parent_matrix, parent_rest_matrix,
                            child_rest_head, rest_length):
    """Follow the authored parent-local anchor; retain the material tail.

    All inputs are in armature pose space. Only the Y length is recomputed;
    nothing multiplies the preceding frame's pose channels or authored length.
    A rest-connected head naturally maps to the parent tail. Offset hair roots
    retain their own anchor whether or not the author's use_connect is checked.
    """
    from mathutils import Matrix
    head = parent_matrix @ (parent_rest_matrix.inverted() @ child_rest_head)
    direction = material_tail - head
    if (not math.isfinite(direction.length) or direction.length <= 1e-12 or
            not math.isfinite(rest_length) or rest_length <= 0):
        raise ValueError('FEM pose: parent anchor and material tail cannot define a bone axis')
    y = direction.normalized()
    source = target.to_3x3()
    x = source.col[0] - y * y.dot(source.col[0])
    if x.length <= 1e-12:
        x = y.cross(source.col[2])
    if x.length <= 1e-12:
        raise ValueError('FEM pose: cannot retain transverse roll at parent anchor')
    x.normalize()
    z = x.cross(y).normalized()
    sign = -1. if source.determinant() < 0 else 1.
    result = Matrix.Identity(4)
    result.translation = head
    result.col[0].xyz = x * source.col[0].length
    result.col[1].xyz = direction / rest_length
    result.col[2].xyz = z * (sign * source.col[2].length)
    return result


def _authored_flags(scene, targets, index=None):
    from .encoder.pmx_joints import effective_parent_joint_pair, parent_joint_pairs
    records = _connect_records(scene)
    legacy = (index or {}).get(CONNECT_KEY) or {}
    flags = {}
    for row, arm, pb in targets:
        record = (records.get(_armature_key(arm)) or {}).get(row['bone_uuid'])
        connected = _author_connection(records, legacy, arm, row, pb)
        connected = bool(pb.bone.use_connect) if connected is None else connected
        connected = bool(connected and pb.bone.ppf_bone_physics.use_hierarchy_joint
                         and effective_parent_joint_pair(pb.bone, parent_joint_pairs(arm)) is None)
        if (connected and (not isinstance(record, dict) or 'inherit_scale' not in record)
                and scene.get(CACHE_KEY) and pb.bone.inherit_scale == 'NONE'):
            raise ValueError('connected bone pose: legacy author inherit_scale is unknown: '+pb.name)
        inherit = record.get('inherit_scale', pb.bone.inherit_scale) if isinstance(record, dict) else pb.bone.inherit_scale
        flags[(arm.as_pointer(), pb.name)] = (bool(connected and pb.parent), inherit)
    return flags


def _pose_application_plan(scene, targets, desired, diagnostics, index=None, author_flags=None):
    """Plan using live Bone data when its inheritance already matches."""
    import bpy
    from .fem_pose import binding_kind, FEM
    copies = {}
    planning_bones = {}
    if author_flags is None:
        author_flags = _authored_flags(scene, targets, index)
    try:
        for row, arm, pb in targets:
            connected, inherit = author_flags[(arm.as_pointer(), pb.name)]
            if pb.parent is not None and (connected or binding_kind(row) != FEM):
                wanted = inherit if connected else 'NONE'
                if pb.bone.inherit_scale == wanted:
                    continue
                if arm.as_pointer() not in copies:
                    copies[arm.as_pointer()] = arm.data.copy()
                bone = copies[arm.as_pointer()].bones[pb.name]
                bone.inherit_scale = wanted
                planning_bones[(arm.as_pointer(), pb.name)] = bone
        return _pose_application_plan_current(targets, desired, diagnostics, planning_bones, author_flags)
    finally:
        for data in copies.values():
            bpy.data.armatures.remove(data)


def _virtual_bridge_parent(arm, bridge, realized, cache):
    """Evaluate frozen disabled FK from this frame's realized physical ancestor."""
    from mathutils import Matrix
    ancestor = arm.pose.bones.get(bridge['ancestor_bone_name'])
    if ancestor is None or ancestor.bone.ppf_bone_physics.bone_uuid != bridge['ancestor_bone_uuid']:
        raise ValueError('bone pose: static bridge ancestor changed; transfer again')
    parent_matrix = realized.get(bridge['ancestor_bone_uuid'])
    if parent_matrix is None:
        raise ValueError('bone pose: static bridge ancestor must be realized first')
    terminal = (arm.as_pointer(), bridge['chain'][-1]['name'])
    if terminal in cache:
        return cache[terminal]
    previous = ancestor
    for item in bridge['chain']:
        pb = arm.pose.bones.get(item['name'])
        if (pb is None or pb.parent != previous or pb.bone.ppf_bone_physics.enabled
                or pb.bone.inherit_scale != item['inherit_scale']
                or bool(pb.bone.use_inherit_rotation) != item['use_inherit_rotation']
                or any(not constraint.mute for constraint in pb.constraints)
                or not np.allclose(np.asarray(pb.bone.matrix_local), item['matrix_local'], atol=1.e-6, rtol=0.)
                or not np.allclose(np.asarray(pb.matrix_basis), item['matrix_basis'], atol=1.e-6, rtol=0.)):
            raise ValueError('bone pose: static nonphysical bridge changed; transfer again')
        key = (arm.as_pointer(), pb.name)
        if key not in cache:
            cache[key] = pb.bone.convert_local_to_pose(Matrix(item['matrix_basis']),
                pb.bone.matrix_local, parent_matrix=parent_matrix,
                parent_matrix_local=previous.bone.matrix_local)
        parent_matrix = cache[key]
        previous = pb
    return parent_matrix


def _pose_application_plan_current(targets, desired, diagnostics, planning_bones, author_flags):
    """Build and validate every local TRS before persistent cache mutation."""
    from mathutils import Matrix, Vector

    realized = {}
    bridge_cache = {}
    plan = {}
    depths = {}
    def depth(item):
        arm, bone = item[1], item[2]
        prefix = arm.as_pointer()
        path = []
        while bone is not None and (prefix, bone.name) not in depths:
            path.append(bone.name)
            bone = bone.parent
        value = -1 if bone is None else depths[(prefix, bone.name)]
        for name in reversed(path):
            value += 1
            depths[(prefix, name)] = value
        return depths[(prefix, item[2].name)]
    for row, arm, pb in sorted(targets, key=depth):
        target = desired[row['bone_uuid']]
        args = {'invert': True}
        if pb.parent:
            parent_id = pb.parent.bone.ppf_bone_physics.bone_uuid
            bridge = row.get('nonphysical_parent_bridge_v1')
            if bridge:
                parent_matrix = _virtual_bridge_parent(arm, bridge, realized, bridge_cache)
            elif parent_id in realized:
                parent_matrix = realized[parent_id]
            else:
                parent_matrix = pb.parent.matrix.copy()
            args.update(parent_matrix=parent_matrix,
                        parent_matrix_local=pb.parent.bone.matrix_local)
        diagnostic = diagnostics.get(row['bone_uuid'])
        connected = author_flags[(arm.as_pointer(), pb.name)][0]
        if connected:
            material_tail = diagnostic['tail'] if diagnostic is not None else target @ Vector((0., pb.bone.length, 0.))
            target = _fem_parent_anchor_target(target, material_tail, args['parent_matrix'],
                                               pb.parent.bone.matrix_local, pb.bone.head_local, pb.bone.length)
            desired[row['bone_uuid']] = target
            if diagnostic is not None:
                diagnostic['parent_anchor_attached'] = True
        elif (diagnostic is not None and pb.parent is not None and
                pb.bone.ppf_bone_physics.root_mode == 'FREE'):
            target = _fem_parent_anchor_target(target, diagnostic['tail'], args['parent_matrix'],
                                               pb.parent.bone.matrix_local,
                                               pb.bone.head_local, pb.bone.length)
            desired[row['bone_uuid']] = target
            diagnostic['parent_anchor_attached'] = True
        if diagnostic and diagnostic.get('parent_anchor_attached') and 'fixed_world_length' in diagnostic:
            direction = target.to_3x3().col[1].normalized()
            world_direction_length = (arm.matrix_world.to_3x3() @ direction).length
            if not math.isfinite(world_direction_length) or world_direction_length <= 1e-12:
                raise ValueError('FEM pose: invalid world direction for fixed bone length')
            target.col[1].xyz = direction * (diagnostic['fixed_world_length'] / (world_direction_length * pb.bone.length))
            desired[row['bone_uuid']] = target
        bone = planning_bones.get((arm.as_pointer(), pb.name), pb.bone)
        basis = bone.convert_local_to_pose(target, bone.matrix_local, **args)
        if diagnostic is not None or connected:
            location, rotation, scale = _fem_trs_preserving_local_y(basis)
        else:
            location, rotation, scale = basis.decompose()
        if connected:
            location = Vector((0., 0., 0.))
        projected = Matrix.LocRotScale(location, rotation, scale)
        forward = {k: v for k, v in args.items() if k != 'invert'}
        actual = bone.convert_local_to_pose(projected, bone.matrix_local, **forward)
        target_axes = target.to_3x3()
        actual_axes = actual.to_3x3()
        corrections = []
        for axis in range(3):
            wanted = target_axes.col[axis].length
            got = actual_axes.col[axis].length
            if not math.isfinite(got) or got <= 0:
                raise ValueError('bone pose: invalid inherited axis length')
            corrections.append(wanted / got)
        scale = Vector(tuple(scale[i] * corrections[i] for i in range(3)))
        projected = Matrix.LocRotScale(location, rotation, scale)
        actual = bone.convert_local_to_pose(projected, bone.matrix_local, **forward)
        scalars = [float(value) for matrix in (projected, actual) for row_values in matrix for value in row_values]
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError('bone pose: non-finite projected transform')
        if connected:
            head_error = (arm.matrix_world.to_3x3() @ (actual.translation - target.translation)).length
            actual_tail = actual @ Vector((0., pb.bone.length, 0.))
            expected_tail = (target @ Vector((0., pb.bone.length, 0.))
                             if diagnostic and 'fixed_world_length' in diagnostic else material_tail)
            tail_error = (arm.matrix_world.to_3x3() @ (actual_tail - expected_tail)).length
            if max(head_error, tail_error) > 3e-5:
                raise ValueError('connected bone pose cannot preserve head/tail: '+pb.name+' head='+str(head_error)+' tail='+str(tail_error))
        if diagnostic is not None:
            actual_axes = actual.to_3x3()
            diagnostic['trs_axis_error_rad'] = max(actual_axes.col[i].angle(target_axes.col[i]) for i in range(3))
        plan[row['bone_uuid']] = (projected.copy(), actual.copy())
        realized[row['bone_uuid']] = actual
    return plan


def _apply(scene, receipt, pose, *, prepared=None):
    _connection_scopes.append({})
    try:
        return _apply_in_scope(scene, receipt, pose, prepared=prepared)
    finally:
        _connection_scopes.pop()


def _apply_in_scope(scene, receipt, pose, *, prepared=None):
    clear_pose_completion(scene)
    targets = _targets(scene, receipt)
    item = _index(scene)
    flags = _authored_flags(scene, targets, item[1] if item else None)
    if prepared is None:
        desired,diagnostics=_prepare_pose(scene,receipt,pose,targets=targets)
        plan=_pose_application_plan(scene,targets,desired,diagnostics,item[1] if item else None, flags)
    else:
        desired,diagnostics,plan=prepared
    previous = [(arm, pb.name, bool(pb.bone.use_connect), pb.bone.inherit_scale, pb.matrix_basis.copy())
                for _row, arm, pb in targets]
    try:
        if item:
            prior_records = _connect_records(scene)
            records = _record_author_connections(scene, targets, item[1])
            bound = targets if item[1]["receipt"] == receipt else _bound_targets(scene, item[1])
            detached = free_bound_heads(scene, item[1], bound)
            if detached:
                targets = _targets(scene, receipt)  # Edit mode invalidates Bone references.
            if detached or records != prior_records:
                flags = _authored_flags(scene, targets, item[1])
        _commit_pose_display(scene, targets, desired, diagnostics, plan, flags)
        _publish_pose_completion(scene, receipt, pose, targets, plan)
        if item:
            _scene_cache_keys.setdefault(scene.as_pointer(), set()).add(item[0])
    except BaseException:
        rollback = {}
        for arm, name, connected, _inherit, _basis in previous:
            rollback.setdefault(arm, {})[name] = connected
        try:
            _write_connections(rollback)
        finally:
            for arm, name, _connected, inherit, basis in previous:
                pb = arm.pose.bones.get(name)
                if pb is not None:
                    pb.bone.inherit_scale = inherit
                    pb.matrix_basis = basis
                    arm.update_tag()
        raise


def _commit_pose_display(scene, targets, desired, diagnostics, plan, author_flags=None):
    """Apply the already validated display plan after connection edits."""
    if author_flags is None:
        author_flags = _authored_flags(scene, targets, (_index(scene) or (None, None))[1])
    with _display_inherit_scales(targets, author_flags, restore=False):
        modified_arms = set()
        try:
            for row,arm,pb in targets:
                target=desired[row['bone_uuid']]
                projected,actual=plan[row['bone_uuid']]
                modified_arms.add(arm)
                pb.matrix_basis=projected
                settings=pb.bone.ppf_bone_physics;diagnostic=diagnostics.get(row['bone_uuid']);metric=diagnostic['metric'] if diagnostic else None
                settings.fem_scale_orientation_error=max(math.degrees(actual.to_3x3().col[i].angle(target.to_3x3().col[i])) for i in range(3))
                if metric:
                    settings.fem_scale_last=tuple(actual.to_3x3().col[i].length for i in range(3))
                    settings.fem_scale_shear=metric['shear_relative'];settings.fem_scale_variation=metric['variation_relative']
                    settings.fem_scale_status='X/Z为全体四面体方向尺度，Y为选定四面体局部伸缩；剪切/弯曲为单骨TRS近似' if metric['approximation'] else 'X/Z为全体四面体方向尺度，Y为选定四面体局部伸缩'
                    if 'fem_display_rest_v1' in row:
                        settings.fem_scale_status='X/Z为全体四面体方向尺度，Y跟随材料支撑头尾；剪切/弯曲为单骨TRS近似'
                else:
                    connected = author_flags[(arm.as_pointer(), pb.name)][0]
                    settings.fem_scale_last=(tuple(actual.to_3x3().col[i].length for i in range(3)) if diagnostic or connected else (1.,1.,1.))
                    settings.fem_scale_shear=0.;settings.fem_scale_variation=0.
                    settings.fem_scale_status='软体缩放关闭：保留初始作者骨长，头点跟随选定四面体' if diagnostic else '刚体或已关闭软体方向缩放'
                    if connected and diagnostic is None:
                        settings.fem_scale_status='连接骨：头点跟随父尾，Y长度跟随材料尾；横截面为TRS近似'
                if diagnostic:
                    if diagnostic.get('parent_anchor_attached'):
                        settings.fem_scale_status='头点跟随作者父局部锚点，Y长度跟随材料尾点，X/Z沿用原映射与作者缩放设置'
                        if 'fixed_world_length' in diagnostic:
                            settings.fem_scale_status='头点跟随作者父局部锚点，Y方向朝向材料尾点，骨长保持作者长度'
                    tail_actual=actual.translation+actual.to_3x3().col[1]*pb.bone.length
                    residual=(arm.matrix_world.to_3x3() @ (tail_actual-diagnostic['tail'])).length
                    support_label = '材料支撑' if 'fem_display_rest_v1' in row else '单四面体仿射外推'
                    settings.fem_scale_status+=f'；材料尾点显示残差 {residual:.6g} m（{support_label}）'
                    if diagnostic.get('trs_axis_error_rad', 0.) > 1e-6:
                        settings.fem_scale_status+='；非均匀继承/剪切的横截面姿态为TRS近似'
        finally:
            for arm in modified_arms:
                arm.update_tag()


def _io_path(path):
    """Use Windows extended paths for internal cache IO, keeping logical keys."""
    path = Path(path)
    if os.name != 'nt':
        return path
    value = os.path.abspath(os.fspath(path))
    if value.startswith('\\\\?\\'):
        return Path(value)
    if value.startswith('\\\\'):
        return Path('\\\\?\\UNC\\' + value[2:])
    return Path('\\\\?\\' + value)


def _write(path, data):
    path = _io_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data); temporary.replace(path)


def _capture_author_pose(scene, receipt, targets):
    """Sample source channels at start, restoring exact current channels last."""
    frame, subframe = scene.frame_current, scene.frame_subframe
    original = [(arm, pb, _capture_native_channels(pb)) for _row, arm, pb in targets]
    try:
        with suspend_playback():
            try:
                scene.frame_set(receipt["start_frame"])
                basis = {row["bone_uuid"]: [list(v) for v in pb.matrix_basis]
                         for row, _arm, pb in targets}
                channels = {row["bone_uuid"]: _capture_native_channels(pb)
                            for row, _arm, pb in targets}
                return basis, channels
            finally:
                scene.frame_set(frame, subframe=subframe)
    finally:
        # Also runs after suspend_playback's display restoration, which may
        # decompose matrices. Native Euler/quaternion/scale intent wins last.
        for arm, pb, channels in original:
            _restore_native_channels(pb, channels)
            arm.update_tag()


def _capture_author_basis(scene, receipt, targets):
    return _capture_author_pose(scene, receipt, targets)[0]


def accept_frame(scene, packet):
    """Main-thread application shared by live and manual fetch."""
    clear_pose_completion(scene)
    from .pc2 import get_pc2_dir
    receipt, pose, raw = packet["receipt"], packet["pose"], packet["native_bytes"]
    desired,diagnostics=_prepare_pose(scene,receipt,pose)
    targets = _targets(scene, receipt)
    # Transform projection can reject degenerate/sheared bases. Validate the
    # whole parents-before-children plan before persisting author state, cache,
    # or detaching connected heads.
    prepared=(desired,diagnostics,_pose_application_plan(scene,targets,desired,diagnostics, (_index(scene) or (None, None))[1]))
    root = Path(get_pc2_dir()) / "bone_poses" / receipt["scene_inputs_sha256"]
    index_path = root / "index.json"
    key = str(index_path)
    if key not in _cache:
        # A malformed generation is discarded rather than trusted, and rebuilt
        # below; the pose files already on disk are left exactly where they are.
        index = _read_index_file(key) if _io_path(index_path).exists() else None
        if index is not None and index["receipt"] != receipt:
            raise ValueError("bone pose: conflicting cache binding")
        if index is None:
            basis, channels = _capture_author_pose(scene, receipt, targets)
            index = {"receipt": receipt, "frames": {}, "author_basis": basis,
                     "author_channels": channels}
        _cache[key] = index
    index = _cache[key]
    # The authored connections are recorded BEFORE the first detach, so the value
    # stored is the author's and never our own display edit.
    _record_author_connections(scene, targets, index)
    frame = str(pose["frame_id"]); digest = hashlib.sha256(raw).hexdigest()
    if frame in index["frames"] and index["frames"][frame] != digest:
        raise ValueError("bone pose: conflicting cached frame generation")
    _write(root / f"pose_{frame}.cbor", raw)
    index["frames"][frame] = digest
    _write(index_path, json.dumps(index, ensure_ascii=False).encode("utf-8"))
    scene[CACHE_KEY] = key
    _scene_cache_keys.setdefault(scene.as_pointer(), set()).add(key)
    _apply(scene, receipt, pose, prepared=prepared)


def _read_index_file(key):
    """The cache index at ``key``, or None when it is absent or malformed.

    A malformed generation is never trusted and never partially accepted, and no
    file on disk is deleted: the caller simply treats the generation as absent.
    """
    try:
        value = json.loads(_io_path(key).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict) or not isinstance(value.get("receipt"), dict) \
            or not isinstance(value.get("frames"), dict) or not isinstance(value.get("author_basis"), dict):
        return None
    if any(not isinstance(k,str) or not k.isascii() or not k.isdecimal() or str(int(k))!=k or not 0<=int(k)<=2147483647 for k in value['frames']):return None
    if any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in value['frames'].values()):return None
    try:
        import cbor2
        decode_bindings(cbor2.dumps({'version':1,'kind':'ppf.blender_bone_bindings','payload':value['receipt']}))
    except (ValueError,KeyError,TypeError,OverflowError):return None
    return value


def _detach_index(scene, key, reason):
    clear_pose_completion(scene)
    """Forget an unusable cache generation, keeping the files and author data."""
    restore_records_now(scene)
    if scene.get(CACHE_KEY) == key:
        del scene[CACHE_KEY]
    owned = _scene_cache_keys.get(scene.as_pointer())
    if owned:
        owned.discard(key)
    from ..models.console import console
    console.write(f"bone pose cache: {reason}; playback detached, authored connections kept")


def _index(scene):
    key = scene.get(CACHE_KEY)
    if not key:
        return None
    if key not in _cache:
        index = _read_index_file(key)
        if index is None:
            # Detaching here keeps every later read clean and stops a per-frame
            # playback loop from retrying a generation that cannot be used.
            _detach_index(scene, key, "index missing or malformed")
            return None
        _cache[key] = index
    return key, _cache[key]


def has_bone_pose_cache(scene):
    """Read-only scene-local predicate suitable for the original Clear poll."""
    return bool(scene.get(CACHE_KEY))


def clear_bone_pose_cache(scene):
    clear_pose_completion(scene)
    from .fem_display_support import clear_cache as clear_fem_display_support
    clear_fem_display_support()
    """Detach playback and restore author channels without touching Actions.

    Return warnings for source channels that cannot be recovered. This recovery
    path always detaches the scene, even after its index or targets were deleted.
    Strict playback and transfer validation remain in their original paths.
    """
    from mathutils import Matrix
    import bpy
    global _suspended
    warnings = []
    key = scene.get(CACHE_KEY)
    owned_keys = _scene_cache_keys.pop(scene.as_pointer(), set())
    if not key and not owned_keys:
        _restore_detached_display(scene)
        return warnings
    if key:
        owned_keys.add(key)
    index = _cache.get(key)
    if index is None and key:
        try:
            index = json.loads(_io_path(key).read_text(encoding="utf-8"))
        except Exception as exc:
            warnings.append(f"Bone cache index unavailable; playback detached: {exc}")
            generation = Path(key).parent.name
            index = next((value for k, value in _cache.items()
                          if k in owned_keys and isinstance(value, dict)
                          and value.get("receipt", {}).get("scene_inputs_sha256") == generation), None)
    # Detach before recovery can fail. A subsequent transfer never traverses
    # this scene's broken index through suspend_playback.
    # Clear restores the authored graph before the cache is forgotten, so the
    # source exports on the bones the author wired even with no cache left.
    try:
        if index is not None:
            restore_authored_connections(scene, index, _bound_targets(scene, index))
        else:
            restore_records_now(scene)
    except Exception as exc:  # noqa: BLE001 - recovery still detaches
        warnings.append(f"Bone connections not restored: {exc}")
    if CACHE_KEY in scene:
        del scene[CACHE_KEY]
    shared_keys = {other.get(CACHE_KEY) for other in bpy.data.scenes if other != scene}
    for cached_key, cached_index in list(_cache.items()):
        if (cached_key in owned_keys or (index is not None and cached_index is index)) and cached_key not in shared_keys:
            _cache.pop(cached_key, None)
    _suspended += 1
    try:
        if index is None:
            warnings.append("Author bone basis unavailable; authored animation will be re-evaluated")
        else:
            try:
                rows = index["receipt"]["bindings"]
                author_basis = index["author_basis"]
                arms=_resolve_armatures(scene,index['receipt'],current=False)
                for row in rows:
                    try:
                        arm = arms.get(row["armature_uuid"])
                        label = row.get("bone_name", row["bone_uuid"])
                        if arm is None:
                            warnings.append(f"{label}: armature UUID missing or ambiguous; author basis not restored")
                            continue
                        bones = [p for p in arm.pose.bones if p.bone.ppf_bone_physics.bone_uuid == row["bone_uuid"]]
                        if len(bones) != 1:
                            warnings.append(f"{label}: bone UUID missing or ambiguous; author basis not restored")
                            continue
                        basis = np.asarray(author_basis[row["bone_uuid"]], dtype=float)
                        if basis.shape != (4, 4) or not np.isfinite(basis).all():
                            raise ValueError("invalid author basis")
                        _restore_author_pose(bones[0], index, row["bone_uuid"])
                        settings=bones[0].bone.ppf_bone_physics
                        settings.fem_scale_last=(1.,1.,1.)
                        settings.fem_scale_shear=0.;settings.fem_scale_variation=0.
                        settings.fem_scale_orientation_error=0.
                        settings.fem_scale_status='已清除物理解算显示，恢复作者姿态'
                        arm.update_tag()
                    except Exception as exc:
                        warnings.append(f"Bone author basis not restored: {exc}")
            except Exception as exc:
                warnings.append(f"Bone author receipt unavailable; playback detached: {exc}")
        try:
            scene.frame_set(scene.frame_current, subframe=scene.frame_subframe)
        except Exception as exc:
            warnings.append(f"Author animation re-evaluation failed after playback detached: {exc}")
    finally:
        _suspended -= 1
    return warnings


SAVE_WARNING_PROP = "ppf_bone_pose_save_warning"
_save_transaction = None


def _save_snapshot():
    import bpy
    return [(scene, scene.get(CACHE_KEY), scene.get(SAVE_WARNING_PROP))
            for scene in bpy.data.scenes]


def _restore_save_snapshot(snapshot):
    for scene, key, warning in snapshot:
        for prop, value in ((CACHE_KEY, key), (SAVE_WARNING_PROP, warning)):
            if value is None:
                scene.pop(prop, None)
            else:
                scene[prop] = value


def _file_sha256(path):
    digest = hashlib.sha256()
    with _io_path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_bone_pose_save(filepath):
    """Publish verified copies and references BEFORE Blender serializes the file.

    Never delete or overwrite a pre-existing generation. A partial target is
    deliberately left for inspection; retry with another blend filename.
    """
    import bpy
    import shutil
    global _save_transaction
    if _save_transaction is not None:
        _restore_save_snapshot(_save_transaction[1])
    snapshot = _save_snapshot()
    _save_transaction = (os.path.abspath(filepath) if filepath else None, snapshot)
    try:
        if not isinstance(filepath, str) or not filepath:
            raise ValueError("save target is unavailable")
        target = Path(os.path.abspath(filepath))
        root = target.parent / "data" / target.stem / "bone_poses"
        plans = []
        by_target = {}
        # Validate EVERY scene/source before creating any destination.
        for scene, old_key, _warning in snapshot:
            if not old_key:
                continue
            source = Path(old_key)
            unchanged = root / source.parent.name / 'index.json'
            if source.name == 'index.json' and source.resolve() == unchanged.resolve():
                # Ordinary Ctrl+S: no migration, no full-cache reads. Keep any
                # prior warning until a successful migration resolves it.
                continue
            index = _read_index_file(old_key)
            if index is None:
                raise ValueError("source index is missing or malformed")
            _hash(index['receipt']['scene_inputs_sha256'])
            generation = index['receipt']['scene_inputs_sha256']
            source = Path(old_key)
            destination = root / generation / 'index.json'
            payload = {'index.json': _file_sha256(source)}
            sources = {'index.json': source}
            for n, digest in index['frames'].items():
                frame_source = source.parent / f'pose_{n}.cbor'
                if _file_sha256(frame_source) != digest:
                    raise ValueError("source frame digest mismatch")
                payload[f'pose_{n}.cbor'] = digest
                sources[f'pose_{n}.cbor'] = frame_source
            key = str(destination)
            if key in by_target and by_target[key][0] != payload:
                raise ValueError("scenes have conflicting cache generations")
            by_target[key] = (payload, sources)
            existing = _io_path(destination.parent)
            if existing.exists():
                if not existing.is_dir() or any(not (existing / name).is_file()
                    or _file_sha256(existing / name) != digest for name, digest in payload.items()):
                    raise ValueError("target generation is incomplete or conflicts; use another filename")
            plans.append((scene, old_key, key, index))
        for key, (payload, sources) in by_target.items():
            folder = _io_path(Path(key).parent)
            if not folder.exists():
                folder.parent.mkdir(parents=True, exist_ok=True)
                folder.mkdir()  # exclusive: never replace another generation
                for name, digest in payload.items():
                    with _io_path(sources[name]).open('rb') as src, (folder / name).open('xb') as dst:
                        shutil.copyfileobj(src, dst, 1024 * 1024)
                    if _file_sha256(folder / name) != digest:
                        raise ValueError("source cache changed during copy")
        for scene, old_key, key, index in plans:
            scene[CACHE_KEY] = key
            _cache[key] = index
            _scene_cache_keys.setdefault(scene.as_pointer(), set()).update((old_key, key))
        for scene, _old_key, _key, _index_value in plans:
            scene.pop(SAVE_WARNING_PROP, None)
    except Exception as exc:
        _restore_save_snapshot(snapshot)
        message = "Bone cache not migrated. Keep the old cache; save to a new filename. " + str(exc)
        for scene, key, _warning in snapshot:
            if key:
                scene[SAVE_WARNING_PROP] = message
        print('PPF:', message)


def finish_bone_pose_save(*_args, failed=False):
    import bpy
    global _save_transaction
    transaction, _save_transaction = _save_transaction, None
    if transaction is not None:
        target, snapshot = transaction
        # Save Copy writes target references into the copy but retains this session.
        if failed or not target or os.path.abspath(bpy.data.filepath) != target:
            _restore_save_snapshot(snapshot)


def bone_pose_save_fail(*_args):
    finish_bone_pose_save(failed=True)
    bone_pose_save_post()


def migrate_bone_poses_on_save(*_args):
    """Compatibility entry point; migration is owned by save_pre now."""
    return


@contextmanager
def _preserve_author_start_pose():
    """Keep an idle Pose Mode edit through the encoder's frame sampling."""
    import bpy
    from .encoder import resolve_start_frame_or_default
    from .playback import is_playing, is_scrubbing

    context = bpy.context
    scene = context.scene
    if (scene is None or context.mode != 'POSE' or scene.frame_subframe
            or scene.frame_current != resolve_start_frame_or_default(scene)
            or has_bone_pose_cache(scene) or _detached_display_pending(scene)
            or is_playing(context) or is_scrubbing(context)):
        yield
        return
    start = scene.frame_current
    channels = [(pb, _capture_native_channels(pb))
                for arm in context.objects_in_mode if arm.type == 'ARMATURE'
                for pb in arm.pose.bones]

    def restore(evaluated_scene, _depsgraph=None):
        if (evaluated_scene != scene or scene.frame_current != start
                or scene.frame_subframe):
            return
        changed = False
        for pb, saved in channels:
            if _capture_native_channels(pb) != saved:
                _restore_native_channels(pb, saved)
                changed = True
        if changed:
            context.view_layer.update()

    handlers = bpy.app.handlers.frame_change_post
    handlers.append(restore)
    try:
        yield
    finally:
        handlers.remove(restore)
        restore(scene)


@contextmanager
def suspend_playback():
    """Restore source bases while transfer samples authored motion."""
    if _suspended:
        yield
        return
    with _preserve_author_start_pose(), _suspend_playback():
        yield


@contextmanager
def _suspend_playback():
    from .frame_time import set_scene_frame
    clear_pose_completion()
    import bpy
    from mathutils import Matrix
    global _suspended
    _suspended += 1
    display = []
    try:
        if _suspended == 1:
            for scene in bpy.data.scenes:
                had_cache = has_bone_pose_cache(scene)
                item = _index(scene)
                # The display matrices are captured BEFORE any Edit-mode work, so
                # the pose that resumes after the suspension is the one the
                # viewer was looking at. The authored graph is then restored while
                # the source is sampled: connections first, then the authored
                # bases, then the frame is re-evaluated.
                #
                # NO CATCH HERE, ON PURPOSE. A restoration failure must reach the
                # caller: entering a source encode on a partially restored display
                # would write display state as if it were the author's. The
                # `finally` below still restores the already-captured display
                # channels and always unwinds `_suspended`.
                if item:
                    _key, index = item
                    targets = _bound_targets(scene, index)
                    for row, _arm, pb in targets:
                        display.append((pb, pb.matrix_basis.copy(), pb.bone.inherit_scale, _capture_native_channels(pb)))
                    restore_authored_connections(scene, index, targets)
                    for row, _arm, pb in targets:
                        _restore_author_pose(pb, index, row["bone_uuid"])
                elif not had_cache and not _restore_detached_display(scene):
                    # Recovery records can outlive playback. Reusing them on
                    # an already-authored rig would replace newer pose edits.
                    continue
                set_scene_frame(scene, scene.frame_current, scene.frame_subframe)
        yield
    finally:
        # Runs even when entry raised, so captured display channels come back and
        # `_suspended` unwinds whatever else happens here.
        try:
            if _suspended == 1:
                for scene in bpy.data.scenes:
                    item = _index(scene)
                    if not item:
                        continue
                    try:
                        free_bound_heads(scene, item[1], _bound_targets(scene, item[1]))
                    except Exception as exc:  # noqa: BLE001 - suspension still ends
                        from ..models.console import console
                        console.write(f"bone pose display connections: {exc}")
            for pb, matrix, inherit_scale, channels in display:
                pb.bone.inherit_scale = inherit_scale
                _restore_native_channels(pb, channels)
        finally:
            _suspended -= 1


def bone_pose_frame_change(scene, _depsgraph=None):
    clear_pose_completion(scene)
    global _busy
    if _suspended or _busy:
        return
    _busy = True
    try:
        item = _index(scene)
        if not item:
            _restore_detached_display(scene)
            return
        key, index = item
        receipt = index["receipt"]
        n = scene.frame_current - receipt["start_frame"]
        if scene.frame_subframe or str(n) not in index["frames"]:
            return  # Missing downloaded frames are not generated or duplicated.
        path = _io_path(Path(key).parent / f"pose_{n}.cbor")
        requested = _capture_for_bake == (scene.as_pointer(), key, n, index["frames"][str(n)])
        before = path.stat() if requested else None
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != index["frames"][str(n)]:
            raise ValueError("bone pose: cached frame digest mismatch")
        _apply(scene, receipt, _load(raw))
        if requested:
            import bpy
            packet = pose_completion(scene)
            after = path.stat()
            if packet is not None and (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns):
                bpy.context.view_layer.update()
                matrices = _evaluated_packet_matrices(scene, packet, bpy.context.evaluated_depsgraph_get())
                if matrices is not None:
                    _verified_playback[scene.as_pointer()] = (packet[1], key, n, digest, (after.st_size, after.st_mtime_ns), matrices, json.dumps(receipt, sort_keys=True, separators=(',', ':')))
    except Exception as exc:
        from ..models.console import console
        console.write(f"bone pose playback: {exc}")
    finally:
        _busy = False



_pending_scene_names=set()

def request_cached_refresh(scene=None):
    import bpy
    if scene is None:_pending_scene_names.update(s.name for s in bpy.data.scenes if s.get(CACHE_KEY))
    elif scene.get(CACHE_KEY):_pending_scene_names.add(scene.name)
    if _pending_scene_names and not bpy.app.timers.is_registered(flush_cached_refresh):
        bpy.app.timers.register(flush_cached_refresh,first_interval=0.)

def _reevaluate_current_frame(scene):
    """Evaluate the active Action without treating internal refresh as scrubbing."""
    from .live_timeline import own_frame_change
    with own_frame_change():
        scene.frame_set(scene.frame_current, subframe=scene.frame_subframe)


def flush_cached_refresh():
    import bpy
    if _suspended or _busy:
        # Keep the queued refresh alive until playback owns the scene again.
        # Clearing it here loses a same-frame Undo/Redo refresh permanently.
        return 0.01 if _pending_scene_names else None
    names=set(_pending_scene_names);_pending_scene_names.clear()
    for scene in bpy.data.scenes:
        if scene.name in names and scene.get(CACHE_KEY):
            # Settle restored author animation before frame_change_post applies
            # the cached display. A direct replay can be overwritten by pending
            # animation evaluation after Undo/Redo, leaving a mixed pose.
            _reevaluate_current_frame(scene)
    return None

def _detached_display_pending(scene):
    return bool(_scene_cache_keys.get(scene.as_pointer()))


def _restore_detached_display(scene):
    if scene.get(CACHE_KEY) or not _detached_display_pending(scene):
        return False
    restore_records_now(scene)
    _scene_cache_keys.pop(scene.as_pointer(), None)
    return True


def bone_pose_undo_redo(*_args):
    clear_pose_completion()
    import bpy
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            _reactivate_connection_recoveries(obj)
    for scene in bpy.data.scenes:
        _restore_detached_display(scene)
    request_cached_refresh()


def _clear_completion_lifecycle(*_args):
    global _connection_recovery_generation
    _connection_recovery_generation += 1
    clear_pose_completion()


def _connection_load_pre(*args):
    global _connection_recovery_load_generation
    _connection_recovery_load_generation += 1
    _clear_completion_lifecycle(*args)


def register():
    import bpy
    if _connection_load_pre not in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.append(bpy.app.handlers.persistent(_connection_load_pre))
    for group in (bpy.app.handlers.undo_pre, bpy.app.handlers.redo_pre):
        if _clear_completion_lifecycle not in group:
            group.append(bpy.app.handlers.persistent(_clear_completion_lifecycle))
    for group in (bpy.app.handlers.undo_post,bpy.app.handlers.redo_post):
        if bone_pose_undo_redo not in group:group.append(bpy.app.handlers.persistent(bone_pose_undo_redo))
    persistent = bpy.app.handlers.persistent(bone_pose_frame_change)
    if not any(getattr(h, "__name__", "") == "bone_pose_frame_change" for h in bpy.app.handlers.frame_change_post):
        bpy.app.handlers.frame_change_post.append(persistent)
    for handlers, handler in ((bpy.app.handlers.save_pre, bone_pose_save_pre),
                              (bpy.app.handlers.save_post, bone_pose_save_post),
                              (bpy.app.handlers.load_post, bone_pose_load_post),
                              (bpy.app.handlers.save_post_fail, bone_pose_save_fail)):
        if not any(getattr(h, "__name__", "") == handler.__name__ for h in handlers):
            handlers.append(bpy.app.handlers.persistent(handler))


def bone_pose_save_pre(filepath=None, *_args):
    """The .blend is saved with the AUTHORED graph, not the display detach.

    Bound bones are recorded here as well, so an author edit to a newly bound
    bone is captured, and every recorded connection is restored even when the
    pose cache is gone.
    """
    import bpy
    prepare_bone_pose_save(filepath)
    for scene in bpy.data.scenes:
        try:
            item = _index(scene)
            if item:
                _record_author_connections(scene, _bound_targets(scene, item[1]), item[1])
            if _connect_records(scene):
                restore_records_now(scene)
                if not item:
                    # A completed Bake keeps recovery records, but its current
                    # Action must win over the old recovery basis when saving.
                    _reevaluate_current_frame(scene)
        except Exception as exc:  # noqa: BLE001 - a save must still proceed
            from ..models.console import console
            console.write(f"bone pose save connections: {exc}")


def bone_pose_save_post(*_args):
    """The session keeps showing the physical pose after the file is written."""
    import bpy
    finish_bone_pose_save()
    for scene in bpy.data.scenes:
        item = _index(scene)
        if not item:
            continue
        try:
            free_bound_heads(scene, item[1], _bound_targets(scene, item[1]))
            request_cached_refresh(scene)
        except Exception as exc:  # noqa: BLE001
            from ..models.console import console
            console.write(f"bone pose save connections: {exc}")


def bone_pose_load_post(*_args):
    global _save_transaction
    _save_transaction = None
    clear_pose_completion()
    from .fem_display_support import clear_cache as clear_fem_display_support
    clear_fem_display_support()
    """A loaded file carries its authored graph, so display resumes from it.

    The external cache is consulted first and defensively: a missing, malformed or
    unreadable index is reported and detached, never trusted. When no usable cache
    remains, the DURABLE local records restore the authored connections AND the
    authored pose channels, so a file saved while the display was detached still
    comes back as the author's rig. No handler path can raise out of the load.
    """
    import bpy
    global _cache, _scene_cache_keys
    _cache = {}
    _scene_cache_keys = {}
    from ..models.console import console
    for scene in bpy.data.scenes:
        try:
            item = _index(scene)
        except Exception as exc:  # noqa: BLE001 - a load must not raise
            console.write(f"bone pose load cache: {exc}")
            item = None
        if item:
            try:
                _key, index = item
                targets = _bound_targets(scene, index)
                free_bound_heads(scene, index, targets)
                request_cached_refresh(scene)
            except Exception as exc:  # noqa: BLE001 - a load must not raise
                console.write(f"bone pose load connections: {exc}")
            continue
        try:
            restore_records_now(scene)
            if _connect_records(scene):
                _reevaluate_current_frame(scene)
        except Exception as exc:  # noqa: BLE001 - a load must not raise
            console.write(f"bone pose load connections: {exc}")


def unregister(*, export_pending=True):
    if export_pending:
        export_pending_connection_recoveries()
    global _save_transaction
    if _save_transaction is not None:
        _restore_save_snapshot(_save_transaction[1])
    _save_transaction = None
    clear_pose_completion()
    import bpy
    if bpy.app.timers.is_registered(flush_cached_refresh):bpy.app.timers.unregister(flush_cached_refresh)
    _pending_scene_names.clear()
    if _connection_load_pre in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(_connection_load_pre)
    for group in (bpy.app.handlers.undo_pre, bpy.app.handlers.redo_pre):
        if _clear_completion_lifecycle in group:
            group.remove(_clear_completion_lifecycle)
    for group in (bpy.app.handlers.undo_post,bpy.app.handlers.redo_post):
        if bone_pose_undo_redo in group:group.remove(bone_pose_undo_redo)
    # Ordinary disable restores author connections. During failed extension
    # registration RestrictBlend exposes no scenes; still finish all teardown.
    scenes = getattr(bpy.data, 'scenes', None)
    for scene in scenes if scenes is not None else ():
        try:
            item = _index(scene)
            if item:
                restore_authored_connections(scene, item[1], _bound_targets(scene, item[1]))
                restore_records_now(scene)
            else:
                restore_records_now(scene)
        except Exception as exc:  # noqa: BLE001 - unregister never raises
            from ..models.console import console
            console.write(f"bone pose unregister connections: {exc}")
    export_pending_connection_recoveries()
    for h in list(bpy.app.handlers.frame_change_post):
        if getattr(h, "__name__", "") == "bone_pose_frame_change":
            bpy.app.handlers.frame_change_post.remove(h)
    for handlers in (bpy.app.handlers.save_post, bpy.app.handlers.load_post):
        for h in list(handlers):
            if getattr(h, "__name__", "") in {"migrate_bone_poses_on_save", "bone_pose_save_post", "bone_pose_load_post"}:
                handlers.remove(h)
    for h in list(bpy.app.handlers.save_pre):
        if getattr(h, "__name__", "") == "bone_pose_save_pre":
            bpy.app.handlers.save_pre.remove(h)
    for h in list(bpy.app.handlers.save_post_fail):
        if getattr(h, "__name__", "") == "bone_pose_save_fail":
            bpy.app.handlers.save_post_fail.remove(h)
    _cache.clear()
    from .fem_display_support import clear_cache as clear_fem_display_support
    clear_fem_display_support()
    _scene_cache_keys.clear()

