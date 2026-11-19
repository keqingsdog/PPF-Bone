# File: frontend/_pdrd_root_motion_v1_.py
# License: Apache v2.0
"""G2 D03 root-motion V6 sidecar: strict authoring, native-frame conversion,
canonical CBOR payload and a pure CPU sampler.

Frozen by
``coordination/checkpoints/G2/root-motion/implementation-contract-20260927.md``
(V6 input/sampler wire, section 1).  This module implements **only** the
CPU-side first slice of that contract:

* a strict authoring schema for one known PDRD root (``static`` or
  ``kinematic``) with authored object/bone-world TRS keys or explicit native
  frames;
* conversion of authored object-world keys into the native PDRD rigid frame
  through the released adapter :mod:`frontend._pdrd_native_frames_`;
* the exact version-1 ``ppf.pdrd_root_motion`` CBOR sidecar payload plus the
  closed-input-inventory hook the V6 scene context needs;
* a stateless CPU sampler (linear center, shortest slerp, endpoint hold,
  key-boundary exactness) together with the key-boundary split and
  arc-to-chord bound the native contact sweep needs; and
* the V6 required-feature / known-body joint preflight helpers that the
  ``_scene_``/``joint.rs`` owners call.  This file never edits those owners'
  files.

There is no solver, CUDA, native ABI, renderer or Blender code here, and no
mutable key cursor on the wire.  Nothing in this module proves native root
motion: a V6 sidecar is authoring intent plus a CPU reference sampler until
the reviewed native ABI and the swept-contact transaction exist.

Duplicate CBOR map keys are refused by the Rust decoder
(:mod:`joint_root_motion_wire`) and by a raw-byte pre-scan there; the Python
reader below uses ``cbor2``, which keeps the last duplicate, so Python alone
does not detect that class of malformed input.
"""
from __future__ import annotations

import hashlib
import math
import numbers
import os
import uuid
from collections.abc import Mapping
from typing import NamedTuple

import numpy as np

from ._pdrd_native_frames_ import (
    WORLD_UUID,
    _matrix_quat,
    _rotation,
    _vector,
    native_body_frames,
)

# --- wire identity ---------------------------------------------------------

SIDECAR_RELATIVE_PATH = "bin/pdrd_root_motion_v1.cbor"
SIDECAR_VERSION = 1
SIDECAR_KIND = "ppf.pdrd_root_motion"
#: Joint envelope version that first advertises the root feature.
JOINT_WIRE_VERSION = 6
FEATURE_KINEMATIC_ROOTS = "pdrd_kinematic_roots_v1"

INTERPOLATION = "linear_center_shortest_slerp_v1"
EXTRAPOLATION = "hold"
MODES = ("static", "kinematic")

#: Authoring provenance.  ``object_world_trs`` keys are converted through the
#: built native frames; ``native_frame`` keys are already in the native PDRD
#: rigid frame and are used directly (explicitly identified, never inferred).
AUTHORING_OBJECT_WORLD = "object_world_trs"
AUTHORING_NATIVE_FRAME = "native_frame"
AUTHORING_MODES = (AUTHORING_OBJECT_WORLD, AUTHORING_NATIVE_FRAME)

#: First V6 slice carries exactly one known body.  More roots, mode switching
#: and velocity inheritance are later work and must fail here, not be ignored.
MAX_BODIES = 1

#: Unit-quaternion input tolerance, then f64 normalization for sampling.
QUAT_NORM_TOL = 1e-6
#: After sign flip an adjacent dot at or below this is an ambiguous half turn.
HALFTURN_DOT = 1e-8
#: Native sweep subinterval bound: ``theta_i <= pi/18`` (10 degrees).
SWEEP_MAX_ANGLE_RAD = math.pi / 18.0

# --- stable refusal codes (one per condition) ------------------------------

CODE_VERSION_OR_KIND = "ROOT_MOTION_VERSION_OR_KIND"
CODE_REQUIRED_FEATURES = "ROOT_MOTION_REQUIRED_FEATURES"
CODE_SIDECAR_MISSING = "ROOT_MOTION_SIDECAR_MISSING"
CODE_INPUT_HASH_MISMATCH = "ROOT_MOTION_INPUT_HASH_MISMATCH"
CODE_BODY_BINDING = "ROOT_MOTION_BODY_BINDING"
CODE_INVALID_KEYS = "ROOT_MOTION_INVALID_KEYS"
CODE_INITIAL_POSE = "ROOT_MOTION_INITIAL_POSE"
CODE_OLD_PIN_OVERLAP = "ROOT_MOTION_OLD_PIN_OVERLAP"
CODE_UNSUPPORTED_DOMAIN = "ROOT_MOTION_UNSUPPORTED_DOMAIN"
CODE_NATIVE_CAPABILITY = "ROOT_MOTION_NATIVE_CAPABILITY"
CODE_SWEEP_UNRESOLVED = "ROOT_MOTION_SWEEP_UNRESOLVED"
CODE_TIME_MISMATCH = "ROOT_MOTION_TIME_MISMATCH"

_AUTHORED_BODY_KEYS = ("body_uuid", "mode", "authoring", "initial_frame", "keys")
_AUTHORED_KEY_KEYS = ("time_s", "translation_m", "quaternion_wxyz")
_FRAME_KEYS = ("translation_m", "quaternion_wxyz")
_NATIVE_BODY_KEYS = ("body_uuid", "runtime_body_index", "mode", "keys")
_NATIVE_KEY_KEYS = ("time_s", "center_world_m", "rotation_world_wxyz")
PAYLOAD_KEYS = ("scene_uuid", "interpolation", "extrapolation", "bodies")


class RootMotionError(ValueError):
    """Stable export-side refusal mirroring the Rust ingest fields."""

    def __init__(self, code, field, body_uuid=None):
        super().__init__(f"{code}: field={field} body_uuid={body_uuid}")
        self.stable_code = code
        self.field = field
        self.body_uuid = body_uuid
        self.violations = [{"stage": "export", "stable_code": code,
                            "field": field, "body_uuid": body_uuid}]


class RootPose(NamedTuple):
    """Sampled native rigid pose: world centre (m) and active ``w,x,y,z``."""
    center_world_m: tuple
    rotation_world_wxyz: tuple


def _default_fail(code, field, body_uuid=None):
    raise RootMotionError(code, field, body_uuid)


def _raiser(fail):
    return _default_fail if fail is None else fail


# --- strict scalar helpers -------------------------------------------------

def _uuid_canonical(value, field, fail, body_uuid=None):
    try:
        parsed = uuid.UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        fail(CODE_BODY_BINDING, field, body_uuid=body_uuid)
    canonical = str(parsed)
    if canonical != value or canonical == WORLD_UUID:
        fail(CODE_BODY_BINDING, field, body_uuid=body_uuid)
    return canonical


def _finite_real(value, field, fail, body_uuid=None, *, nonnegative=False):
    """Strict finite real; ``bool`` and every non-real type are refused.

    The wire is binary64, so the whole finite f64 range is allowed; only
    non-finite values and an optional sign constraint are refused.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        fail(CODE_INVALID_KEYS, field, body_uuid=body_uuid)
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        fail(CODE_INVALID_KEYS, field, body_uuid=body_uuid)
    if not math.isfinite(number):
        fail(CODE_INVALID_KEYS, field, body_uuid=body_uuid)
    if nonnegative and number < 0.0:
        fail(CODE_INVALID_KEYS, field, body_uuid=body_uuid)
    return number


def _real_list(value, length, field, fail, body_uuid=None):
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
        fail(CODE_INVALID_KEYS, field, body_uuid=body_uuid)
    if len(value) != length:
        fail(CODE_INVALID_KEYS, field, body_uuid=body_uuid)
    return [_finite_real(item, f"{field}[{index}]", fail, body_uuid)
            for index, item in enumerate(value)]


def _u32(value, field, fail, body_uuid=None):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xFFFF_FFFF:
        fail(CODE_BODY_BINDING, field, body_uuid=body_uuid)
    return value


def _exact_map(value, keys, field, fail, body_uuid=None, code=CODE_INVALID_KEYS):
    if not isinstance(value, Mapping) or set(value) != set(keys):
        fail(code, field, body_uuid=body_uuid)
    return value


def _unit_quaternion(value, field, fail, body_uuid=None):
    """Validate ``|q|-1| <= 1e-6`` then normalize in f64 (contract rule)."""
    # The native-frame adapter's _vector casts bool and non-real objects to
    # float. Apply the wire's strict real policy before that coercion.
    components = _real_list(value, 4, field, fail, body_uuid)
    try:
        quat = _vector(components, 4, field)
    except ValueError:
        fail(CODE_INVALID_KEYS, field, body_uuid=body_uuid)
    norm = float(np.linalg.norm(quat))
    if not math.isfinite(norm) or norm <= 0.0 or abs(norm - 1.0) > QUAT_NORM_TOL:
        fail(CODE_INVALID_KEYS, field, body_uuid=body_uuid)
    return quat / norm


def _frame(value, field, fail, body_uuid=None):
    """Exact ``{translation_m:[3], quaternion_wxyz:[4]}`` frame value."""
    frame = _exact_map(value, _FRAME_KEYS, field, fail, body_uuid)
    return {
        "translation_m": _real_list(frame["translation_m"], 3, f"{field}.translation_m",
                                    fail, body_uuid),
        "quaternion_wxyz": _unit_quaternion(frame["quaternion_wxyz"],
                                            f"{field}.quaternion_wxyz", fail, body_uuid),
    }


# --- sampler primitives ----------------------------------------------------

def quaternion_dot(a, b):
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


def rotation_angle_rad(q0, q1):
    """Shortest proper-rotation angle between two (unit) quaternions, in rad.

    Uses the absolute dot so both sign representations of one rotation report
    the same angle; the result lies in ``[0, pi]``.
    """
    dot = min(1.0, max(0.0, abs(quaternion_dot(q0, q1))))
    return 2.0 * math.acos(dot)


def arc_chord_bound_m(radius_m, theta_rad):
    """``delta = r * (1 - cos(theta/2))``: arc-to-endpoint-chord deviation."""
    if not math.isfinite(radius_m) or radius_m < 0.0:
        raise RootMotionError(CODE_INVALID_KEYS, "radius_m")
    if not math.isfinite(theta_rad) or not 0.0 <= theta_rad <= math.pi:
        raise RootMotionError(CODE_INVALID_KEYS, "theta_rad")
    return float(radius_m * (1.0 - math.cos(theta_rad / 2.0)))


def slerp_shortest(q0, q1, weight):
    """Shortest-arc slerp of two unit quaternions.

    A negative adjacent dot flips the second quaternion (the rotation is
    unchanged); an exactly or numerically ambiguous half turn is refused
    because the connecting arc is then not unique.
    """
    first = np.asarray(q0, dtype=np.float64)
    second = np.asarray(q1, dtype=np.float64)
    dot = quaternion_dot(first, second)
    if dot < 0.0:
        second = -second
        dot = -dot
    if dot <= HALFTURN_DOT:
        raise RootMotionError(CODE_INVALID_KEYS, "rotation_world_wxyz")
    if not 0.0 <= weight <= 1.0:
        raise RootMotionError(CODE_INVALID_KEYS, "weight")
    if dot > 1.0 - 1e-12:
        mixed = first + weight * (second - first)
        norm = float(np.linalg.norm(mixed))
        if norm <= 0.0:
            raise RootMotionError(CODE_INVALID_KEYS, "rotation_world_wxyz")
        return mixed / norm
    theta = math.acos(min(1.0, dot))
    sin_theta = math.sin(theta)
    return (math.sin((1.0 - weight) * theta) / sin_theta) * first + \
           (math.sin(weight * theta) / sin_theta) * second


def require_forward_time(t0_s, t1_s):
    """Refuse ``T<0``, nonfinite ``T`` and a step that does not advance."""
    if (not math.isfinite(t0_s) or not math.isfinite(t1_s)
            or t0_s < 0.0 or not t1_s > t0_s):
        raise RootMotionError(CODE_TIME_MISMATCH, "physical_T_s")
    return float(t1_s)


def _pose(key):
    return RootPose(tuple(float(v) for v in key["center_world_m"]),
                    tuple(float(v) for v in key["rotation_world_wxyz"]))


def sample_root_body(body, time_s):
    """Sample one native body at physical ``time_s`` (seconds).

    Center is linear inside a key interval and rotation is shortest slerp.
    Outside the keyed span the nearest endpoint key is held exactly, and a
    key boundary returns that key's exact pose.
    """
    if not math.isfinite(time_s) or time_s < 0.0:
        raise RootMotionError(CODE_TIME_MISMATCH, "time_s", body_uuid=body["body_uuid"])
    keys = body["keys"]
    if time_s <= keys[0]["time_s"]:
        return _pose(keys[0])
    if time_s >= keys[-1]["time_s"]:
        return _pose(keys[-1])
    for index in range(len(keys) - 1):
        left, right = keys[index], keys[index + 1]
        if left["time_s"] <= time_s <= right["time_s"]:
            if time_s == left["time_s"]:
                return _pose(left)
            if time_s == right["time_s"]:
                return _pose(right)
            weight = (time_s - left["time_s"]) / (right["time_s"] - left["time_s"])
            center = tuple(
                float(left["center_world_m"][axis]
                      + weight * (right["center_world_m"][axis]
                                  - left["center_world_m"][axis]))
                for axis in range(3))
            quat = slerp_shortest(left["rotation_world_wxyz"],
                                  right["rotation_world_wxyz"], weight)
            norm = float(np.linalg.norm(quat))
            return RootPose(center, tuple(float(v / norm) for v in quat))
    return _pose(keys[-1])


def split_at_key_times(body, t0_s, t1_s):
    """Partition ``[t0, t1]`` at every key time strictly inside it.

    Returns ``(a, b)`` pieces with ``a < b``.  A boundary that lands on a key
    time terminates one piece and starts the next, so each piece belongs to
    exactly one key interval and endpoint sampling is exact.
    """
    require_forward_time(t0_s, t1_s)
    boundaries = [float(t0_s)]
    for key in body["keys"]:
        time_s = float(key["time_s"])
        if float(t0_s) < time_s < float(t1_s) and time_s != boundaries[-1]:
            boundaries.append(time_s)
    boundaries.append(float(t1_s))
    return [(boundaries[index], boundaries[index + 1])
            for index in range(len(boundaries) - 1)]


def sweep_subintervals(body, t0_s, t1_s, radius_m):
    """Split key pieces until every piece rotates by at most ``pi/18``.

    Pure CPU reference for the native sweep's subinterval rule.  Rotation
    along one slerp is a geodesic, so angles of adjacent pieces add exactly
    and bisection halves the endpoint angle; each emitted piece therefore
    reports its own ``theta`` and arc-to-chord ``delta``.  A piece that cannot
    be subdivided further is refused rather than silently accepted.
    """
    require_forward_time(t0_s, t1_s)
    if not math.isfinite(radius_m) or radius_m < 0.0:
        raise RootMotionError(CODE_INVALID_KEYS, "radius_m")
    pieces = []
    for start, end in split_at_key_times(body, t0_s, t1_s):
        stack = [(start, end)]
        while stack:
            low, high = stack.pop()
            theta = rotation_angle_rad(sample_root_body(body, low).rotation_world_wxyz,
                                       sample_root_body(body, high).rotation_world_wxyz)
            if theta <= SWEEP_MAX_ANGLE_RAD:
                pieces.append({"t0_s": low, "t1_s": high, "theta_rad": theta,
                               "delta_m": arc_chord_bound_m(radius_m, theta)})
                continue
            middle = 0.5 * (low + high)
            if not low < middle < high:
                raise RootMotionError(CODE_SWEEP_UNRESOLVED, "sweep_subintervals")
            stack.append((middle, high))
            stack.append((low, middle))
    pieces.sort(key=lambda piece: piece["t0_s"])
    return pieces


# --- authored-schema validation -------------------------------------------

def validate_authored_v1(authored, fail=None):
    """Validate the strict authoring document and return a normalized copy.

    Shape: ``{"version": 1, "bodies": [body]}`` with exactly one body.  Each
    body is exactly ``{body_uuid, mode, authoring, initial_frame, keys}``.
    Refusing an exact key set is what stops a keyed scale/shear from being
    silently dropped: there is no field for it, and an extra one fails.
    """
    fail = _raiser(fail)
    if not isinstance(authored, Mapping) or set(authored) != {"version", "bodies"}:
        fail(CODE_INVALID_KEYS, "authored")
    if type(authored["version"]) is not int or authored["version"] != SIDECAR_VERSION:
        fail(CODE_VERSION_OR_KIND, "authored.version")
    bodies = authored["bodies"]
    if not isinstance(bodies, list) or len(bodies) != MAX_BODIES:
        fail(CODE_UNSUPPORTED_DOMAIN, "authored.bodies")
    seen = set()
    normalized = []
    for index, body in enumerate(bodies):
        field = f"authored.bodies[{index}]"
        body = _exact_map(body, _AUTHORED_BODY_KEYS + (("control_events",) if isinstance(body, Mapping) and "control_events" in body else ()), field, fail)
        body_uuid = _uuid_canonical(body["body_uuid"], f"{field}.body_uuid", fail)
        if body_uuid in seen:
            fail(CODE_BODY_BINDING, f"{field}.body_uuid", body_uuid=body_uuid)
        seen.add(body_uuid)
        mode = body["mode"]
        if mode not in MODES:
            fail(CODE_INVALID_KEYS, f"{field}.mode", body_uuid=body_uuid)
        authoring = body["authoring"]
        if authoring not in AUTHORING_MODES:
            fail(CODE_INVALID_KEYS, f"{field}.authoring", body_uuid=body_uuid)
        initial = _frame(body["initial_frame"], f"{field}.initial_frame", fail, body_uuid)
        raw_keys = body["keys"]
        if not isinstance(raw_keys, list) or not raw_keys:
            fail(CODE_INVALID_KEYS, f"{field}.keys", body_uuid=body_uuid)
        keys = []
        previous = None
        for key_index, key in enumerate(raw_keys):
            key_field = f"{field}.keys[{key_index}]"
            key = _exact_map(key, _AUTHORED_KEY_KEYS, key_field, fail, body_uuid)
            time_s = _finite_real(key["time_s"], f"{key_field}.time_s", fail,
                                  body_uuid, nonnegative=True)
            if key_index == 0 and time_s != 0.0:
                fail(CODE_INVALID_KEYS, f"{key_field}.time_s", body_uuid=body_uuid)
            if previous is not None and time_s <= previous:
                fail(CODE_INVALID_KEYS, f"{key_field}.time_s", body_uuid=body_uuid)
            previous = time_s
            keys.append({
                "time_s": time_s,
                "translation_m": _real_list(key["translation_m"], 3,
                                            f"{key_field}.translation_m", fail, body_uuid),
                "quaternion_wxyz": _unit_quaternion(
                    key["quaternion_wxyz"], f"{key_field}.quaternion_wxyz", fail, body_uuid),
            })
        if mode == "static" and not body.get("control_events") and len(keys) != 1:
            fail(CODE_INVALID_KEYS, f"{field}.keys", body_uuid=body_uuid)
        if mode == "kinematic" and len(keys) < 2:
            fail(CODE_INVALID_KEYS, f"{field}.keys", body_uuid=body_uuid)
        # The initial frame is the trajectory anchor and must be the first key
        # exactly: for object authoring it is A0, for native authoring N0.
        first = keys[0]
        if not (first["time_s"] == 0.0
                and np.array_equal(np.asarray(first["translation_m"]),
                                   np.asarray(initial["translation_m"]))
                and abs(quaternion_dot(first["quaternion_wxyz"],
                                       initial["quaternion_wxyz"])) >= 1.0 - QUAT_NORM_TOL):
            fail(CODE_INITIAL_POSE, f"{field}.initial_frame", body_uuid=body_uuid)
        normalized.append({"body_uuid": body_uuid, "mode": mode, "authoring": authoring,
                           "initial_frame": initial, "keys": keys, **_normalized_controls(body, fail, body_uuid)})
    return {"version": SIDECAR_VERSION, "bodies": normalized}


def sign_continuous_quaternions(keys, fail, body_uuid):
    """Flip later quaternions for positive adjacent dot; refuse half turns.

    A quaternion sign is not part of the rotation, so the wire carries one
    canonical sign-continuous form and each key segment is then unique and
    shorter than ``pi``.
    """
    output = []
    previous = None
    for index, key in enumerate(keys):
        quat = np.asarray(key["rotation_world_wxyz"], dtype=np.float64)
        if previous is not None:
            dot = quaternion_dot(previous, quat)
            if dot < 0.0:
                quat = -quat
                dot = -dot
            if dot <= HALFTURN_DOT:
                fail(CODE_INVALID_KEYS, f"keys[{index}].rotation_world_wxyz",
                     body_uuid=body_uuid)
        output.append({"time_s": key["time_s"],
                       "center_world_m": [float(v) for v in key["center_world_m"]],
                       "rotation_world_wxyz": [float(v) for v in quat]})
        previous = quat
    return output


def validate_native_payload_v1(payload, fail=None):
    """Validate an already-native sidecar payload; return a normalized f64 copy."""
    fail = _raiser(fail)
    payload = _exact_map(payload, PAYLOAD_KEYS, "payload", fail, code=CODE_VERSION_OR_KIND)
    scene_uuid = _uuid_canonical(payload["scene_uuid"], "payload.scene_uuid", fail)
    if payload["interpolation"] != INTERPOLATION:
        fail(CODE_VERSION_OR_KIND, "payload.interpolation")
    if payload["extrapolation"] != EXTRAPOLATION:
        fail(CODE_VERSION_OR_KIND, "payload.extrapolation")
    bodies = payload["bodies"]
    if not isinstance(bodies, list) or len(bodies) != MAX_BODIES:
        fail(CODE_UNSUPPORTED_DOMAIN, "payload.bodies")
    seen = set()
    normalized = []
    for index, body in enumerate(bodies):
        field = f"payload.bodies[{index}]"
        body = _exact_map(body, _NATIVE_BODY_KEYS + (("control_events",) if isinstance(body, Mapping) and "control_events" in body else ()), field, fail, code=CODE_BODY_BINDING)
        body_uuid = _uuid_canonical(body["body_uuid"], f"{field}.body_uuid", fail)
        if body_uuid in seen:
            fail(CODE_BODY_BINDING, f"{field}.body_uuid", body_uuid=body_uuid)
        seen.add(body_uuid)
        runtime_index = _u32(body["runtime_body_index"], f"{field}.runtime_body_index",
                             fail, body_uuid)
        mode = body["mode"]
        if mode not in MODES:
            fail(CODE_INVALID_KEYS, f"{field}.mode", body_uuid=body_uuid)
        raw_keys = body["keys"]
        if not isinstance(raw_keys, list) or not raw_keys:
            fail(CODE_INVALID_KEYS, f"{field}.keys", body_uuid=body_uuid)
        keys = []
        previous = None
        for key_index, key in enumerate(raw_keys):
            key_field = f"{field}.keys[{key_index}]"
            key = _exact_map(key, _NATIVE_KEY_KEYS, key_field, fail, body_uuid)
            time_s = _finite_real(key["time_s"], f"{key_field}.time_s", fail,
                                  body_uuid, nonnegative=True)
            if key_index == 0 and time_s != 0.0:
                fail(CODE_INVALID_KEYS, f"{key_field}.time_s", body_uuid=body_uuid)
            if previous is not None and time_s <= previous:
                fail(CODE_INVALID_KEYS, f"{key_field}.time_s", body_uuid=body_uuid)
            previous = time_s
            keys.append({"time_s": time_s,
                         "center_world_m": _real_list(key["center_world_m"], 3,
                                                      f"{key_field}.center_world_m",
                                                      fail, body_uuid),
                         "rotation_world_wxyz": _unit_quaternion(
                             key["rotation_world_wxyz"],
                             f"{key_field}.rotation_world_wxyz", fail, body_uuid)})
        if mode == "static" and not body.get("control_events") and len(keys) != 1:
            fail(CODE_INVALID_KEYS, f"{field}.keys", body_uuid=body_uuid)
        if mode == "kinematic" and len(keys) < 2:
            fail(CODE_INVALID_KEYS, f"{field}.keys", body_uuid=body_uuid)
        normalized.append({"body_uuid": body_uuid, "runtime_body_index": runtime_index,
                           "mode": mode,
                           "keys": sign_continuous_quaternions(keys, fail, body_uuid), **_normalized_controls(body, fail, body_uuid)})
    return {"scene_uuid": scene_uuid, "interpolation": INTERPOLATION,
            "extrapolation": EXTRAPOLATION, "bodies": normalized}


# --- authored -> native conversion ----------------------------------------

def convert_to_native_bodies_v1(authored, native_frames, fail=None):
    """Convert a validated authoring document into native body records.

    ``native_frames`` is the released adapter's ``native_body_frames(fixed)``
    table (UUID -> fitted native centre/rotation/residual).  For an authored
    object key ``(t_o(t), R_o(t))`` and initial frames ``(t_o0, R_o0)`` /
    ``(c_n0, R_n0)``::

        D(t)   = R_o(t) R_o0^T
        c_n(t) = t_o(t) + D(t) (c_n0 - t_o0)
        R_n(t) = D(t) R_n0

    Direct native authoring is passed through unchanged.  The runtime body
    index always comes from the adapter, never from the author, so a
    contradictory index cannot be smuggled in.
    """
    fail = _raiser(fail)
    document = validate_authored_v1(authored, fail)
    if not isinstance(native_frames, Mapping):
        fail(CODE_BODY_BINDING, "native_frames")
    bodies = []
    for body in document["bodies"]:
        body_uuid = body["body_uuid"]
        if body_uuid not in native_frames:
            fail(CODE_BODY_BINDING, "native_frames", body_uuid=body_uuid)
        fitted = native_frames[body_uuid]
        runtime_index = _u32(fitted["runtime_body_index"],
                             "native_frames.runtime_body_index", fail, body_uuid)
        if body["authoring"] == AUTHORING_NATIVE_FRAME:
            keys = [{"time_s": key["time_s"],
                     "center_world_m": list(key["translation_m"]),
                     "rotation_world_wxyz": list(key["quaternion_wxyz"])}
                    for key in body["keys"]]
        else:
            authored_origin = np.asarray(body["initial_frame"]["translation_m"],
                                         dtype=np.float64)
            authored_rotation = _rotation(body["initial_frame"]["quaternion_wxyz"])
            native_origin = np.asarray(fitted["world_translation_m"], dtype=np.float64)
            native_rotation = _rotation(fitted["world_quaternion_wxyz"])
            keys = []
            for key in body["keys"]:
                delta = _rotation(key["quaternion_wxyz"]) @ authored_rotation.T
                center = np.asarray(key["translation_m"], dtype=np.float64) + \
                    delta @ (native_origin - authored_origin)
                quat = _matrix_quat(delta @ native_rotation)
                keys.append({"time_s": key["time_s"],
                             "center_world_m": [float(v) for v in center],
                             "rotation_world_wxyz": [float(v) for v in quat]})
        bodies.append({"body_uuid": body_uuid, "runtime_body_index": runtime_index,
                       "mode": body["mode"], "keys": keys, **_normalized_controls(body, fail, body_uuid)})
    return bodies


def build_sidecar_payload_v1(authored, scene_uuid, native_frames, fail=None):
    """Full authoring path: validate, convert, brand the scene and re-validate."""
    fail = _raiser(fail)
    bodies = convert_to_native_bodies_v1(authored, native_frames, fail)
    payload = validate_native_payload_v1(
        {"scene_uuid": scene_uuid, "interpolation": INTERPOLATION,
         "extrapolation": EXTRAPOLATION, "bodies": bodies}, fail)
    first = payload["bodies"][0]
    if first["runtime_body_index"] != native_frames[first["body_uuid"]]["runtime_body_index"]:
        fail(CODE_BODY_BINDING, "runtime_body_index", body_uuid=first["body_uuid"])
    return payload


# --- initial-pose geometry check ------------------------------------------

def verify_initial_pose_v1(payload, fixed, fail=None):
    """Rebuild ``x0[v] = c(0) + R(0) ybar[k]`` from the built scene.

    The check compares geometry, not only centroid/quaternion, against the
    per-body residual the released adapter already accepted.  The adapter also
    validates row/binding/UUID membership, so a sidecar body missing from the
    built scene is refused here as a binding error.
    """
    fail = _raiser(fail)
    payload = validate_native_payload_v1(payload, fail)
    fitted = native_body_frames(fixed)
    rows = np.asarray(fixed._pdrd_body_rows, dtype=np.float64).reshape(-1, 23)
    rest = np.asarray(fixed._pdrd_rest_centered, dtype=np.float64)
    ids = np.asarray(fixed._pdrd_vert_list).astype(np.int64)
    vertices = np.asarray(fixed.vertex(), dtype=np.float64)
    report = {}
    for body in payload["bodies"]:
        body_uuid = body["body_uuid"]
        if body_uuid not in fitted:
            fail(CODE_BODY_BINDING, "payload.bodies", body_uuid=body_uuid)
        if body["runtime_body_index"] != fitted[body_uuid]["runtime_body_index"]:
            fail(CODE_BODY_BINDING, "runtime_body_index", body_uuid=body_uuid)
        index = fitted[body_uuid]["runtime_body_index"]
        start, count = int(rows[index, 0]), int(rows[index, 1])
        body_ids = ids[start:start + count]
        rest_rows = rest[start:start + count]
        key0 = body["keys"][0]
        center = np.asarray(key0["center_world_m"], dtype=np.float64)
        rotation = _rotation(key0["rotation_world_wxyz"])
        rebuilt = center + (rotation @ rest_rows.T).T
        residual = float(np.max(np.abs(rebuilt - vertices[body_ids])))
        tolerance = float(fitted[body_uuid]["fit_tolerance_m"])
        if not math.isfinite(residual) or residual > tolerance:
            fail(CODE_INITIAL_POSE, "payload.bodies[0].keys[0]", body_uuid=body_uuid)
        report[body_uuid] = {"max_reconstruction_error_m": residual,
                             "fit_tolerance_m": tolerance,
                             "runtime_body_index": index}
    return report


# --- CBOR envelope --------------------------------------------------------

def encode_sidecar_v1(payload):
    """Canonical ``{version:1, kind:ppf.pdrd_root_motion, payload}`` CBOR bytes."""
    from . import _cbor_bridge_ as _cbor
    normalized = validate_native_payload_v1(payload)
    ordered = {key: normalized[key] for key in PAYLOAD_KEYS}
    return _cbor.dumps_envelope_with_version(SIDECAR_VERSION, SIDECAR_KIND, ordered)


def decode_sidecar_v1(blob, fail=None):
    """Decode sidecar bytes and re-validate the payload.

    Rust repeats this validation independently; this reader exists so the
    Python side can round-trip its own bytes and so envelope tests do not need
    a native module.  ``cbor2`` keeps the last duplicate map key, so duplicate
    refusal is a Rust-side guarantee (see the module docstring).
    """
    fail = _raiser(fail)
    import cbor2
    try:
        envelope = cbor2.loads(bytes(blob))
    except Exception:
        fail(CODE_VERSION_OR_KIND, "envelope")
    if not isinstance(envelope, Mapping) or set(envelope) != {"version", "kind", "payload"}:
        fail(CODE_VERSION_OR_KIND, "envelope")
    if envelope["version"] != SIDECAR_VERSION or envelope["kind"] != SIDECAR_KIND:
        fail(CODE_VERSION_OR_KIND, "envelope")
    return validate_native_payload_v1(envelope["payload"], fail)


def write_sidecar_v1(scene_path, payload):
    """Write the sidecar under ``scene_path`` and return its inventory entry.

    Must be called before ``seal_pdrd_scene_context`` so the closed inventory
    snapshot already contains this named file.
    """
    data = encode_sidecar_v1(payload)
    absolute = os.path.join(scene_path, SIDECAR_RELATIVE_PATH.replace("/", os.sep))
    os.makedirs(os.path.dirname(absolute), exist_ok=True)
    with open(absolute, "wb") as handle:
        handle.write(data)
    return {"relative_path": SIDECAR_RELATIVE_PATH, "byte_length": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def verify_sidecar_inventory_v1(scene_path, inventory, fail=None):
    """Require exact sidecar bytes/hash in the sealed context-v2 inventory."""
    fail = _raiser(fail)
    matches = [item for item in inventory
               if item.get("relative_path") == SIDECAR_RELATIVE_PATH]
    if len(matches) != 1:
        fail(CODE_SIDECAR_MISSING, SIDECAR_RELATIVE_PATH)
    absolute = os.path.join(scene_path, SIDECAR_RELATIVE_PATH.replace("/", os.sep))
    if not os.path.exists(absolute):
        fail(CODE_SIDECAR_MISSING, SIDECAR_RELATIVE_PATH)
    with open(absolute, "rb") as handle:
        data = handle.read()
    entry = matches[0]
    if entry.get("byte_length") != len(data) or \
            entry.get("sha256") != hashlib.sha256(data).hexdigest():
        fail(CODE_INPUT_HASH_MISMATCH, SIDECAR_RELATIVE_PATH)
    return entry


# --- V6 feature and known-body preflight ----------------------------------

def required_features_v6(base_features, has_sidecar, fail=None):
    """Exact lexicographically sorted, unique V6 feature list for one payload.

    ``base_features`` is the already-sorted V2..V5 list implied by the joint
    records.  The root feature is implied by the sidecar, so a sidecar without
    it and the feature without a sidecar are both refused rather than merged
    silently.
    """
    fail = _raiser(fail)
    features = list(base_features)
    if FEATURE_KINEMATIC_ROOTS in features and not has_sidecar:
        fail(CODE_REQUIRED_FEATURES, "required_features")
    if has_sidecar:
        features.append(FEATURE_KINEMATIC_ROOTS)
    return sorted(set(features))


def validate_known_body_joint_v6(joints, sidecar_body_uuid, fail=None):
    """V6 first slice needs one enabled joint whose non-world parent is the root."""
    fail = _raiser(fail)
    if sidecar_body_uuid == WORLD_UUID:
        fail(CODE_BODY_BINDING, "sidecar_body_uuid")
    for joint in joints:
        if joint.get("enabled") is True and \
                joint.get("parent_body_uuid") == sidecar_body_uuid:
            return joint
    fail(CODE_UNSUPPORTED_DOMAIN, "joints.parent_body_uuid", body_uuid=sidecar_body_uuid)


def reject_hard_kinematic_composition_v6(joints, fail=None):
    """Hard-hinge + kinematic composition is explicitly unsupported in V6."""
    fail = _raiser(fail)
    for joint in joints:
        if "angular_hard_hinge_v1" in joint:
            fail(CODE_UNSUPPORTED_DOMAIN, "angular_hard_hinge_v1")


def validate_v5_sidecar_absence(version, sidecar_present, fail=None):
    """V2..V5 inputs refuse a root sidecar instead of silently ignoring it."""
    fail = _raiser(fail)
    if version != JOINT_WIRE_VERSION and sidecar_present:
        fail(CODE_VERSION_OR_KIND, "bin/pdrd_root_motion_v1.cbor")


def _normalized_controls(body, fail, body_uuid):
    """Optional physical-time mode events. Missing field preserves old bytes."""
    if "control_events" not in body:
        return {}
    events = body["control_events"]
    if not isinstance(events, list) or len(events) > 4096:
        fail(CODE_INVALID_KEYS, "control_events", body_uuid=body_uuid)
    previous = 0.0
    mode = body["mode"]
    normalized = []
    for event in events:
        if not isinstance(event, dict) or set(event) != {"time_s", "mode"}:
            fail(CODE_INVALID_KEYS, "control_events", body_uuid=body_uuid)
        time = _finite_real(event["time_s"], "control_events.time_s", fail, body_uuid)
        selected = event["mode"]
        if time <= previous or selected not in ("static", "kinematic", "dynamic") or selected == mode:
            fail(CODE_INVALID_KEYS, "control_events", body_uuid=body_uuid)
        normalized.append({"time_s": time, "mode": selected})
        previous, mode = time, selected
    return {"control_events": normalized} if normalized else {}
