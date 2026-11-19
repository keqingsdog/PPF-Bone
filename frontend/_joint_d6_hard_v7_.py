"""Strict V7 D6 authoring and built PPF native-frame preflight.

This module writes no native ABI and makes no claim about a native solve.
``bind_payload_v7`` is called only after ``Scene.build`` has produced the
actual native PDRD initial frames.
"""
from __future__ import annotations

import math
import numbers
from copy import deepcopy

import numpy as np

from ._angular_extensions_ import COORDINATE_DOMAIN_RAD
from ._pdrd_native_frames_ import (
    WORLD_UUID, _qmul, _rotation, native_body_frames, rebind_joint_payload,
)

D6_KEY = "joint_d6_hard_v1"
D6_FEATURE = "pdrd_joint_d6_hard_v1"
COORDINATE_MODEL = "parent_local_d6_swing_twist_x_v1"
AXES = ("x", "y", "z")
CODE_INVALID = "G2_D6_INVALID_INPUT"
CODE_DOMAIN = "G2_D6_UNSUPPORTED_DOMAIN"
CODE_INITIAL = "G2_D6_INITIAL_INFEASIBLE"
CODE_CHART = "G2_D6_CHART_INVALID"
_F32_MAX = float(np.finfo(np.float32).max)
_INITIAL_GATES = (5e-5, 5e-5, 5e-5, 5e-4, 5e-4, 5e-4)


def _number(value, field, jid, fail):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        fail(CODE_INVALID, field, jid)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        fail(CODE_INVALID, field, jid)
    if not math.isfinite(result) or abs(result) > _F32_MAX:
        fail(CODE_INVALID, field, jid)
    narrowed = float(np.float32(result))
    if result != 0.0 and narrowed == 0.0:
        fail(CODE_INVALID, field, jid)
    return result


def validate_d6_v1(value, jid, fail, *, require_rest=False):
    """Validate exact maps; omit rest only in pre-build authoring input."""
    field = D6_KEY
    required = {"coordinate_model", "translation", "rotation"}
    if (not isinstance(value, dict) or
            (set(value) != required and set(value) != required | {"rest_translation_m"}) or
            (require_rest and "rest_translation_m" not in value) or
            value.get("coordinate_model") != COORDINATE_MODEL):
        fail(CODE_INVALID, field, jid)
    result = {"coordinate_model": COORDINATE_MODEL}
    if "rest_translation_m" in value:
        rest = value["rest_translation_m"]
        if not isinstance(rest, (list, tuple)) or len(rest) != 3:
            fail(CODE_INVALID, f"{field}.rest_translation_m", jid)
        result["rest_translation_m"] = [
            _number(v, f"{field}.rest_translation_m[{i}]", jid, fail)
            for i, v in enumerate(rest)]
    for group in ("translation", "rotation"):
        axes = value[group]
        if not isinstance(axes, dict) or set(axes) != set(AXES):
            fail(CODE_INVALID, f"{field}.{group}", jid)
        result[group] = {}
        for axis in AXES:
            name = f"{field}.{group}.{axis}"
            authored = axes[axis]
            if not isinstance(authored, dict):
                fail(CODE_INVALID, name, jid)
            mode = authored.get("mode")
            shape = {"free": {"mode"}, "limited": {"mode", "lower", "upper"},
                     "locked": {"mode", "value"}}.get(mode) if isinstance(mode, str) else None
            if shape is None or set(authored) != shape:
                fail(CODE_INVALID, name, jid)
            entry = {"mode": mode}
            for key in shape - {"mode"}:
                number = _number(authored[key], f"{name}.{key}", jid, fail)
                if group == "rotation" and abs(number) >= COORDINATE_DOMAIN_RAD:
                    fail(CODE_INVALID, f"{name}.{key}", jid)
                entry[key] = number
            if mode == "limited" and not np.float32(entry["lower"]) < np.float32(entry["upper"]):
                fail(CODE_INVALID, f"{name}.lower", jid)
            result[group][axis] = entry
    return result


def _chart(native_a, native_b, joint):
    qa = np.asarray(native_a["world_quaternion_wxyz"], dtype=float)
    qb = np.asarray(native_b["world_quaternion_wxyz"], dtype=float)
    la = np.asarray(joint["parent_local_frame"]["quaternion_wxyz"], dtype=float)
    lb = np.asarray(joint["child_local_frame"]["quaternion_wxyz"], dtype=float)
    rest = np.asarray(joint["rest_relative_quaternion_wxyz"], dtype=float)
    a, b = _qmul(qa, la), _qmul(qb, lb)
    q = _qmul(_qmul(a * [1, -1, -1, -1], b), rest * [1, -1, -1, -1])
    q /= np.linalg.norm(q)
    if (q[0] < 0 or (q[0] == 0 and tuple(q[1:]) < (0.0, 0.0, 0.0))):
        q = -q
    tnorm = math.hypot(q[0], q[1])
    if tnorm < 1e-4:
        return None
    qt = np.array([q[0] / tnorm, q[1] / tnorm, 0., 0.])
    qs = _qmul(q, qt * [1, -1, -1, -1])
    qs /= np.linalg.norm(qs)
    twist = 2 * math.atan2(qt[1], qt[0])
    length = math.hypot(qs[2], qs[3])
    scale = 2 * math.atan2(length, qs[0]) / length if length > 1e-8 else 2 / qs[0]
    swing = (scale * qs[2], scale * qs[3])
    if (not all(math.isfinite(v) for v in (twist, *swing)) or
            abs(twist) >= COORDINATE_DOMAIN_RAD or
            math.hypot(*swing) >= COORDINATE_DOMAIN_RAD):
        return None
    return [twist, *swing]


def _native_anchor_displacement_parent_frame(joint, native_frames):
    """Measure Qa^T(pb-pa) from actual native poses and both local anchors."""
    a = native_frames[joint["parent_body_uuid"]]
    b = native_frames[joint["child_body_uuid"]]
    ra = _rotation(np.asarray(a["world_quaternion_wxyz"]))
    rb = _rotation(np.asarray(b["world_quaternion_wxyz"]))
    la = _rotation(np.asarray(joint["parent_local_frame"]["quaternion_wxyz"]))
    pa = np.asarray(a["world_translation_m"]) + ra @ np.asarray(
        joint["parent_local_frame"]["translation_m"])
    pb = np.asarray(b["world_translation_m"]) + rb @ np.asarray(
        joint["child_local_frame"]["translation_m"])
    return (ra @ la).T @ (pb - pa)


def initial_coordinates(joint, native_frames, fail):
    """Measure initial coordinates at actual native centers and joint frames."""
    jid = joint["joint_uuid"]
    a = native_frames[joint["parent_body_uuid"]]
    b = native_frames[joint["child_body_uuid"]]
    trans = _native_anchor_displacement_parent_frame(joint, native_frames) - np.asarray(
        joint[D6_KEY]["rest_translation_m"])
    angular = _chart(a, b, joint)
    if not np.isfinite(trans).all():
        fail(CODE_INVALID, D6_KEY, jid)
    return [float(v) for v in trans] + (angular if angular is not None else [None] * 3)


def _check_tree(joints, frames, root_uuid, fixed, fail):
    if not 1 <= len(joints) <= 127 or not 2 <= len(frames) <= 128:
        fail(CODE_DOMAIN, "joints")
    d6 = [j for j in joints if D6_KEY in j]
    if len(d6) != 1:
        fail(CODE_DOMAIN, D6_KEY)
    parent_of = {}
    nodes = set()
    edges = set()
    for j in joints:
        jid = j["joint_uuid"]
        p, c = j["parent_body_uuid"], j["child_body_uuid"]
        if p == WORLD_UUID or p not in frames or c not in frames or (p, c) in edges or (c, p) in edges or c in parent_of:
            fail(CODE_DOMAIN, "joints.graph", jid)
        edges.add((p, c)); parent_of[c] = p; nodes.update((p, c))
        k = _number(j["stiffness_N_per_m"], "stiffness_N_per_m", jid, fail)
        if D6_KEY not in j and (not j["enabled"] or np.float32(k) <= 0):
            fail(CODE_DOMAIN, "stiffness_N_per_m", jid)
        if D6_KEY in j and any(j[key] != 0.0 for key in (
                "angle_stiffness_Nm_per_rad", "angle_damping_Nms_per_rad", "max_angle_rad")):
            fail(CODE_DOMAIN, "angle_stiffness_Nm_per_rad", jid)
    sources = nodes - set(parent_of)
    if len(sources) != 1 or len(nodes) != len(joints) + 1 or (root_uuid is not None and root_uuid not in sources):
        fail(CODE_DOMAIN, "joints.graph")
    source = next(iter(sources))
    reachable = {source}
    while True:
        following = reachable | {c for c, p in parent_of.items() if p in reachable}
        if following == reachable:
            break
        reachable = following
    if reachable != nodes or len(set(frames) - nodes) > 1:
        fail(CODE_DOMAIN, "joints.graph")
    rows = np.asarray(fixed._pdrd_body_rows, dtype=float).reshape(-1, 23)
    if any(rows[f["runtime_body_index"], 16] != 0.0 for f in frames.values()):
        fail(CODE_DOMAIN, "pdrd_body.joint_mode")
    used_vertices = set()
    for node in nodes:
        row = rows[frames[node]["runtime_body_index"]]
        start, count = int(row[0]), int(row[1])
        used_vertices.update(int(v) for v in fixed._pdrd_vert_list[start:start + count])
    if any(used_vertices.intersection(int(v) for v in pin.index) for pin in fixed._pin):
        fail(CODE_DOMAIN, "pin.index")
    return d6[0]


def bind_payload_v7(fixed, payload, authored_frames=None, *, root_uuid=None, fail):
    """Rebind authored frames, fill default rest, and preflight one D6 edge."""
    frames = native_body_frames(fixed)
    bound = (rebind_joint_payload(payload, authored_frames, frames,
                                  preserve_world_joint_orientations=True)
             if authored_frames is not None else deepcopy(payload))
    # Caller validates the whole V7 envelope before and after this operation.
    joints = bound["joints"]
    d6 = _check_tree(joints, frames, root_uuid, fixed, fail)
    spec = d6[D6_KEY]
    if "rest_translation_m" not in spec:
        # Derived native rest is the f32 receiver coordinate, not authored f64.
        spec["rest_translation_m"] = [float(np.float32(v)) for v in
            _native_anchor_displacement_parent_frame(d6, frames)]
    d6[D6_KEY] = validate_d6_v1(spec, d6["joint_uuid"], fail, require_rest=True)
    coordinates = initial_coordinates(d6, frames, fail)
    if d6["enabled"]:
        for i, axis in enumerate(AXES):
            _check_axis(coordinates[i], spec["translation"][axis],
                        _INITIAL_GATES[i], d6["joint_uuid"], fail)
        for i, axis in enumerate(AXES, 3):
            _check_axis(coordinates[i], spec["rotation"][axis], _INITIAL_GATES[i], d6["joint_uuid"], fail)
    return bound, frames, coordinates


def _check_axis(coordinate, axis, gate, jid, fail):
    mode = axis["mode"]
    if mode == "free":
        return
    if coordinate is None:
        fail(CODE_CHART, D6_KEY, jid)
    if mode == "locked" and abs(coordinate - axis["value"]) > gate:
        fail(CODE_INITIAL, D6_KEY, jid)
    if mode == "limited" and (coordinate < axis["lower"] - gate or coordinate > axis["upper"] + gate):
        fail(CODE_INITIAL, D6_KEY, jid)


def required_features_v7(joints, has_root):
    from ._angular_extensions_ import required_features
    features = set(required_features(joints))
    features.add(D6_FEATURE)
    if has_root:
        features.add("pdrd_kinematic_roots_v1")
    return sorted(features)
