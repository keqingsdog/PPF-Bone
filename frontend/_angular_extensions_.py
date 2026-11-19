# File: frontend/_angular_extensions_.py
# License: Apache v2.0
"""G2 joint-envelope V4 authoring validation: signed soft limits and drives.

Pure, side-effect-free validators for the two optional version-4 joint keys
defined by ``coordination/checkpoints/G2/angular-controls/next-controls-design-20260926.md``:

* ``angular_soft_limits_v1`` -- per-axis signed compliant soft bounds on the
  parent-local ``[twist_x, swing_y, swing_z]`` coordinates.
* ``angular_drives_v1`` -- independent position, coordinate-rate and
  generalized-effort channels keyed by physical solver seconds.

This module contains no solver, renderer, Blender, or file-writing code.  It
does not import :mod:`frontend._scene_`; the caller passes its stable error
raiser so the two modules cannot form an import cycle and the same
``JOINT_*`` codes reach the Rust boundary.  Nothing here is a hard limit, a
hinge, or a solver guarantee: an enabled soft bound may be violated and is
reported as an excess, and every coefficient/cap may be inert.

Wire conventions (radians, seconds, N*m, SI gains) are already SI.  No degree
conversion happens here; a UI would convert once before authoring.
"""
from __future__ import annotations

import math
import numbers

import numpy as np

# Coordinate model shared with the V3 first slice.  Bounds and position
# targets are expressed in this chart, not in Euler angles or log(q).
COORDINATE_MODEL = "parent_local_swing_twist_x_v1"
CONTINUOUS_MODEL = "parent_local_swing_twist_x_continuous_v2"
MATERIAL_SPIN_MODEL = "material_spin_path_v3"
CONTINUOUS_BOUND_RAD = 512.0

SOFT_LIMITS_KEY = "angular_soft_limits_v1"
DRIVES_KEY = "angular_drives_v1"
CONTROL_KEY = "angular_control_v1"

FEATURE_CONTROL = "pdrd_angular_controls_v1"
FEATURE_SOFT_LIMITS = "pdrd_angular_soft_limits_v1"
FEATURE_SWING_CONE = "pdrd_angular_swing_cone_v1"
FEATURE_DRIVES = "pdrd_angular_drives_v1"

AXES = ("twist_x", "swing_y", "swing_z")

# Principal coordinate domain of the V3 chart.  The native chart refuses at
# ``abs(c) >= pi - 1e-3``; authored bounds and position targets must stay
# strictly inside it so an in-domain authoring value cannot already be a
# refused chart state.
COORDINATE_DOMAIN_RAD = float(np.float32(np.float32(np.pi) - np.float32(1e-3)))

INTERPOLATION = "linear_hold"
DRIVE_MODES = ("position", "velocity", "torque")
DRIVE_MODE_GAIN = {
    "position": "gain_Nm_per_rad",
    "velocity": "gain_Nms_per_rad",
    "torque": "gain",
}
DRIVE_MODE_VALUE = {
    "position": "value_rad",
    "velocity": "value_rad_per_s",
    "torque": "value_Nm",
}

# Explicit bounds for time-keyed target arrays.  The native key POD
# (``AngularDriveKeyGpuV1``) is 16 bytes; 4096 total keys == 65536 bytes ==
# the total byte budget, so the count cap and the byte cap bind together.
# Per-channel cap keeps a single malformed channel from consuming the budget.
MAX_KEYS_PER_CHANNEL = 1024
MAX_TOTAL_KEYS = 4096
KEY_BYTES = 16
MAX_TOTAL_KEY_BYTES = MAX_TOTAL_KEYS * KEY_BYTES

# Stable codes.  Distinct codes keep the failure reason greppable in the
# export-side receipt without changing any Rust-side code.
CODE_INVALID = "JOINT_ANGULAR_EXTENSION_INVALID"
CODE_REQUIRES_CONTROL = "JOINT_ANGULAR_EXTENSION_REQUIRES_CONTROL"
CODE_INTERVAL = "JOINT_ANGULAR_EXTENSION_INVALID_INTERVAL"
CODE_OUT_OF_DOMAIN = "JOINT_ANGULAR_EXTENSION_OUT_OF_DOMAIN"
CODE_INTERPOLATION = "JOINT_ANGULAR_EXTENSION_UNSUPPORTED_INTERPOLATION"
CODE_KEY_ORDER = "JOINT_ANGULAR_EXTENSION_KEY_ORDER"
CODE_KEY_LIMIT = "JOINT_ANGULAR_EXTENSION_KEY_LIMIT"

_F32_MAX = float(np.finfo(np.float32).max)
_F32_MIN_SUBNORMAL = float(np.nextafter(np.float32(0), np.float32(1)))


def _real(value, fail, field, joint_uuid, *, nonnegative=False):
    """Return a finite ``f32``-representable real, else raise the stable error.

    ``bool`` and every non-``numbers.Real`` type (complex, str, bytes, list,
    dict, None, ndarray) are rejected instead of being coerced.  Magnitudes
    that would silently flush to zero in ``f32`` are rejected too, matching
    the V3 coefficient rule.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        fail(CODE_INVALID, field, joint_uuid)
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        fail(CODE_INVALID, field, joint_uuid)
    if (not math.isfinite(number) or number > _F32_MAX
            or 0.0 < abs(number) < _F32_MIN_SUBNORMAL):
        fail(CODE_INVALID, field, joint_uuid)
    if nonnegative and number < 0.0:
        fail(CODE_INVALID, field, joint_uuid)
    return number


def _boolean(value, fail, field, joint_uuid):
    """Strict ``bool`` only; ``0``/``1``/``np.bool_`` are not accepted."""
    if not isinstance(value, bool):
        fail(CODE_INVALID, field, joint_uuid)
    return value


def _exact_map(value, keys, fail, field, joint_uuid):
    if not isinstance(value, dict) or set(value) != set(keys):
        fail(CODE_INVALID, field, joint_uuid)
    return value


def _linear_hold(keys, value_key, time_s):
    """``linear_hold`` sample: linear between keys, hold first/last outside."""
    if time_s <= keys[0]["time_s"]:
        return keys[0][value_key]
    if time_s >= keys[-1]["time_s"]:
        return keys[-1][value_key]
    for index in range(len(keys) - 1):
        t0, t1 = keys[index]["time_s"], keys[index + 1]["time_s"]
        if t0 <= time_s <= t1:
            weight = (time_s - t0) / (t1 - t0)
            return keys[index][value_key] + weight * (
                keys[index + 1][value_key] - keys[index][value_key]
            )
    return keys[-1][value_key]


def validate_soft_limits_v1(value, joint_uuid, fail):
    """Validate ``angular_soft_limits_v1``; all three axis keys are required.

    Endpoints are signed and preserved (never converted to magnitudes).
    ``enabled=false`` or zero stiffness is inert but still fully validated.
    """
    field = SOFT_LIMITS_KEY
    if not isinstance(value, dict):
        fail(CODE_INVALID, field, joint_uuid)
    keys = ("coordinate_model",) + AXES + (("swing_cone",) if "swing_cone" in value else ())
    _exact_map(value, keys, fail, field, joint_uuid)
    if value["coordinate_model"] not in (COORDINATE_MODEL, CONTINUOUS_MODEL, MATERIAL_SPIN_MODEL):
        fail(CODE_INVALID, f"{field}.coordinate_model", joint_uuid)
    result = {"coordinate_model": value["coordinate_model"]}
    for axis in AXES:
        name = f"{field}.{axis}"
        axis_value = _exact_map(
            value[axis],
            ("enabled", "lower_rad", "upper_rad", "stiffness_Nm_per_rad"),
            fail, name, joint_uuid,
        )
        enabled = _boolean(axis_value["enabled"], fail, f"{name}.enabled", joint_uuid)
        lower = _real(axis_value["lower_rad"], fail, f"{name}.lower_rad", joint_uuid)
        upper = _real(axis_value["upper_rad"], fail, f"{name}.upper_rad", joint_uuid)
        stiffness = _real(axis_value["stiffness_Nm_per_rad"], fail,
                          f"{name}.stiffness_Nm_per_rad", joint_uuid, nonnegative=True)
        limit = CONTINUOUS_BOUND_RAD if value["coordinate_model"] == MATERIAL_SPIN_MODEL or (axis == "twist_x" and value["coordinate_model"] == CONTINUOUS_MODEL) else COORDINATE_DOMAIN_RAD
        if (abs(lower) >= limit or abs(upper) >= limit or
                abs(float(np.float32(lower))) >= limit or abs(float(np.float32(upper))) >= limit):
            fail(CODE_OUT_OF_DOMAIN, f"{name}.lower_rad", joint_uuid)
        if lower > upper:
            fail(CODE_INTERVAL, f"{name}.lower_rad", joint_uuid)
        result[axis] = {
            "enabled": enabled,
            "lower_rad": lower,
            "upper_rad": upper,
            "stiffness_Nm_per_rad": stiffness,
        }
    if "swing_cone" in value:
        name = f"{field}.swing_cone"
        cone = _exact_map(value["swing_cone"], ("enabled", "half_angle_rad", "stiffness_Nm_per_rad"), fail, name, joint_uuid)
        enabled = _boolean(cone["enabled"], fail, name+".enabled", joint_uuid)
        alpha = _real(cone["half_angle_rad"], fail, name+".half_angle_rad", joint_uuid)
        stiffness = _real(cone["stiffness_Nm_per_rad"], fail, name+".stiffness_Nm_per_rad", joint_uuid, nonnegative=True)
        alpha32, stiffness32 = float(np.float32(alpha)), float(np.float32(stiffness))
        if any(0. < abs(v) < float(np.finfo(np.float32).tiny) for v in (alpha32, stiffness32)):
            fail(CODE_INVALID, name, joint_uuid)
        if not 0. < alpha32 < COORDINATE_DOMAIN_RAD:
            fail(CODE_OUT_OF_DOMAIN, name+".half_angle_rad", joint_uuid)
        if enabled and any(result[a]["enabled"] for a in AXES[1:]):
            fail(CODE_INVALID, name+".enabled", joint_uuid)
        result["swing_cone"] = dict(enabled=enabled, half_angle_rad=alpha, stiffness_Nm_per_rad=stiffness)
    return result


def _drive_channel(channel, mode, name, joint_uuid, fail, budget, continuous_twist=False):
    gain_key = DRIVE_MODE_GAIN[mode]
    value_key = DRIVE_MODE_VALUE[mode]
    _exact_map(
        channel,
        ("enabled", gain_key, "max_effort_Nm", "interpolation", "target_keys"),
        fail, name, joint_uuid,
    )
    enabled = _boolean(channel["enabled"], fail, f"{name}.enabled", joint_uuid)
    gain = _real(channel[gain_key], fail, f"{name}.{gain_key}", joint_uuid, nonnegative=True)
    max_effort = _real(channel["max_effort_Nm"], fail,
                       f"{name}.max_effort_Nm", joint_uuid, nonnegative=True)
    if channel["interpolation"] != INTERPOLATION:
        fail(CODE_INTERPOLATION, f"{name}.interpolation", joint_uuid)
    keys = channel["target_keys"]
    if not isinstance(keys, list) or not keys:
        fail(CODE_INVALID, f"{name}.target_keys", joint_uuid)
    if len(keys) > MAX_KEYS_PER_CHANNEL:
        fail(CODE_KEY_LIMIT, f"{name}.target_keys", joint_uuid)
    budget["keys"] += len(keys)
    if budget["keys"] > MAX_TOTAL_KEYS or budget["keys"] * KEY_BYTES > MAX_TOTAL_KEY_BYTES:
        fail(CODE_KEY_LIMIT, f"{name}.target_keys", joint_uuid)
    normalized = []
    previous_time = None
    for index, key in enumerate(keys):
        key_name = f"{name}.target_keys[{index}]"
        _exact_map(key, ("time_s", value_key), fail, key_name, joint_uuid)
        time_s = _real(key["time_s"], fail, f"{key_name}.time_s", joint_uuid,
                       nonnegative=True)
        target = _real(key[value_key], fail, f"{key_name}.{value_key}", joint_uuid)
        if index == 0 and time_s != 0.0:
            fail(CODE_KEY_ORDER, f"{key_name}.time_s", joint_uuid)
        if previous_time is not None and time_s <= previous_time:
            fail(CODE_KEY_ORDER, f"{key_name}.time_s", joint_uuid)
        if mode == "position":
            limit = CONTINUOUS_BOUND_RAD if continuous_twist else COORDINATE_DOMAIN_RAD
            if abs(target) >= limit or abs(float(np.float32(target))) >= limit:
                fail(CODE_OUT_OF_DOMAIN, f"{key_name}.{value_key}", joint_uuid)
        previous_time = time_s
        normalized.append({"time_s": time_s, value_key: target})
    return {
        "enabled": enabled,
        gain_key: gain,
        "max_effort_Nm": max_effort,
        "interpolation": INTERPOLATION,
        "target_keys": normalized,
    }


def _check_combined_swing_disk(drives, joint_uuid, fail):
    """Both enabled swing position targets must stay inside the swing disk.

    The combined vector is checked at every union key time.  The disk is
    convex, so a linear-hold segment interior cannot exceed its endpoints and
    key-time sampling is sufficient.  Applied whenever both swing position
    channels are ``enabled`` (the design wording); inert channels are still
    structurally validated separately.
    """
    swing_y = drives.get("swing_y", {}).get("position")
    swing_z = drives.get("swing_z", {}).get("position")
    if not swing_y or not swing_z or not swing_y["enabled"] or not swing_z["enabled"]:
        return
    y_keys, z_keys = swing_y["target_keys"], swing_z["target_keys"]
    union_times = sorted(
        {key["time_s"] for key in y_keys} | {key["time_s"] for key in z_keys}
    )
    for time_s in union_times:
        y_value = _linear_hold(y_keys, "value_rad", time_s)
        z_value = _linear_hold(z_keys, "value_rad", time_s)
        if math.hypot(y_value, z_value) >= COORDINATE_DOMAIN_RAD:
            fail(CODE_OUT_OF_DOMAIN, f"{DRIVES_KEY}.swing_y_swing_z", joint_uuid)


def validate_drives_v1(value, joint_uuid, fail, budget):
    """Validate ``angular_drives_v1``; axis and channel keys are optional.

    ``budget`` is a mutable ``{"keys": int}`` accumulator shared across the
    whole payload so the total-key/byte cap is enforced over all joints.
    """
    field = DRIVES_KEY
    if not isinstance(value, dict) or "coordinate_model" not in value:
        fail(CODE_INVALID, field, joint_uuid)
    extras = set(value) - {"coordinate_model"} - set(AXES)
    if extras:
        fail(CODE_INVALID, f"{field}.{sorted(extras)[0]}", joint_uuid)
    if value["coordinate_model"] not in (COORDINATE_MODEL, CONTINUOUS_MODEL, MATERIAL_SPIN_MODEL):
        fail(CODE_INVALID, f"{field}.coordinate_model", joint_uuid)
    result = {"coordinate_model": value["coordinate_model"]}
    for axis in AXES:
        if axis not in value:
            continue
        name = f"{field}.{axis}"
        axis_value = value[axis]
        if not isinstance(axis_value, dict):
            fail(CODE_INVALID, name, joint_uuid)
        axis_extras = set(axis_value) - set(DRIVE_MODES)
        if axis_extras:
            fail(CODE_INVALID, f"{name}.{sorted(axis_extras)[0]}", joint_uuid)
        channels = {}
        for mode in DRIVE_MODES:
            if mode not in axis_value:
                continue
            channels[mode] = _drive_channel(
                axis_value[mode], mode, f"{name}.{mode}", joint_uuid, fail, budget,
                continuous_twist=(value["coordinate_model"] == MATERIAL_SPIN_MODEL or (axis == "twist_x" and value["coordinate_model"] == CONTINUOUS_MODEL))
            )
        result[axis] = channels
    if value["coordinate_model"] != MATERIAL_SPIN_MODEL:
        _check_combined_swing_disk(result, joint_uuid, fail)
    return result


def required_features(joints):
    """Exact lexicographically sorted feature set for a version-4 payload."""
    features = []
    if any(CONTROL_KEY in joint for joint in joints):
        features.append(FEATURE_CONTROL)
    if any(SOFT_LIMITS_KEY in joint for joint in joints):
        features.append(FEATURE_SOFT_LIMITS)
    if any("swing_cone" in joint.get(SOFT_LIMITS_KEY, {}) for joint in joints):
        features.append(FEATURE_SWING_CONE)
    if any(DRIVES_KEY in joint for joint in joints):
        features.append(FEATURE_DRIVES)
    return sorted(features)
