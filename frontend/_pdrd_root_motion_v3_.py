"""Root-motion sidecar v3 with an immutable body-local reference transform.

The author record intentionally uses the v2 author vocabulary.  Its keys remain
the author path P(t); the required wire ``reference_transform`` D is composed
on the right only after evaluating P, so the physical path is P(t) * D.
"""
from __future__ import annotations

import hashlib
import io
import math
import os
from collections.abc import Mapping

import numpy as np

from . import _pdrd_root_motion_v1_ as _v1
from . import _pdrd_root_motion_v2_ as _v2
from . import _pdrd_native_frames_ as _frames

VERSION = 3
KIND = _v1.SIDECAR_KIND
SIDECAR_RELATIVE_PATH = _v1.SIDECAR_RELATIVE_PATH
INTERPOLATION = _v1.INTERPOLATION
EXTRAPOLATION = _v1.EXTRAPOLATION
ROOT_FEATURE = "pdrd_kinematic_roots_v3"
ROOT_FEATURE_V1 = _v2.ROOT_FEATURE_V1
ROOT_FEATURE_V2 = _v2.ROOT_FEATURE
BODY_CONTROLS_FEATURE = _v2.BODY_CONTROLS_FEATURE
MAX_BODIES = _v2.MAX_BODIES
MAX_KEYS_PER_ROOT = _v2.MAX_KEYS_PER_ROOT
MAX_KEY_RECORDS_TOTAL = _v2.MAX_KEY_RECORDS_TOTAL

CODE_VERSION_OR_KIND = _v2.CODE_VERSION_OR_KIND
CODE_REQUIRED_FEATURES = _v2.CODE_REQUIRED_FEATURES
CODE_SIDECAR_MISSING = _v2.CODE_SIDECAR_MISSING
CODE_BODY_BINDING = _v2.CODE_BODY_BINDING
CODE_INVALID_KEYS = _v2.CODE_INVALID_KEYS
CODE_INITIAL_POSE = _v2.CODE_INITIAL_POSE
CODE_INPUT_HASH_MISMATCH = _v2.CODE_INPUT_HASH_MISMATCH
CODE_UNSUPPORTED_DOMAIN = _v2.CODE_UNSUPPORTED_DOMAIN
CODE_ROOT_BATCH = _v2.CODE_ROOT_BATCH

_PAYLOAD_KEYS = ("scene_uuid", "interpolation", "extrapolation", "bodies")
_REFERENCE_KEYS = {"translation_local_m", "rotation_local_wxyz"}
_BODY_KEYS = {"body_uuid", "runtime_body_index", "mode", "keys", "reference_transform"}
_BODY_OPTIONAL = {"control_events"}


def _raiser(fail):
    return _v2._raiser(fail)


def _ref_transform(value, fail, body_uuid=None):
    if not isinstance(value, dict) or set(value) != _REFERENCE_KEYS:
        fail(CODE_INVALID_KEYS, "reference_transform", body_uuid=body_uuid)
    translation = _v1._real_list(value["translation_local_m"], 3,
                                 "reference_transform.translation_local_m", fail, body_uuid)
    rotation = _v1._unit_quaternion(value["rotation_local_wxyz"],
                                    "reference_transform.rotation_local_wxyz", fail, body_uuid)
    # Convert every scalar to a built-in float before CBOR encoding.  The wire
    # schema requires binary64 values, including values supplied as integers.
    return {"translation_local_m": [float(v) for v in translation],
            "rotation_local_wxyz": [float(v) for v in rotation]}


def validate_authored_v3(authored, fail=None):
    """Validate the unchanged v2 author document; D is never author input."""
    return _v2.validate_authored_v2(authored, fail)


def _reference_from_initial(root, fitted, fail):
    uuid = root["body_uuid"]
    if root.get("authoring") != _v1.AUTHORING_OBJECT_WORLD:
        # Explicit native-frame roots (and legacy v2 roots without provenance)
        # already describe the native rigid frame and therefore use D=I.
        return {"translation_local_m": [0.0, 0.0, 0.0],
                "rotation_local_wxyz": [1.0, 0.0, 0.0, 0.0]}
    initial = root.get("initial_frame")
    if not isinstance(initial, Mapping):
        fail(CODE_BODY_BINDING, "initial_frame", uuid)
    authored_t = np.asarray(initial["translation_m"], dtype=np.float64)
    authored_r = _frames._rotation(initial["quaternion_wxyz"])
    native_t = np.asarray(fitted["world_translation_m"], dtype=np.float64)
    native_r = _frames._rotation(fitted["world_quaternion_wxyz"])
    translation = authored_r.T @ (native_t - authored_t)
    rotation = _frames._matrix_quat(authored_r.T @ native_r)
    if not np.isfinite(translation).all() or not np.isfinite(rotation).all():
        fail(CODE_BODY_BINDING, "reference_transform", uuid)
    return {"translation_local_m": [float(v) for v in translation],
            "rotation_local_wxyz": [float(v) for v in rotation]}


def convert_to_native_bodies_v3(authored, native_frames, fail=None):
    """Bind actual body indices and D while leaving validated author keys intact."""
    _fail = _raiser(fail)
    validated = validate_authored_v3(authored, _fail)
    if not isinstance(native_frames, Mapping) or not native_frames:
        _fail(CODE_BODY_BINDING, "native_frames")
    bodies = []
    for root in validated["roots"]:
        uuid = root["body_uuid"]
        frame = native_frames.get(uuid)
        if not isinstance(frame, Mapping):
            _fail(CODE_BODY_BINDING, "native_frames", uuid)
        ref = _ref_transform(_reference_from_initial(root, frame, _fail), _fail, uuid)
        keys = [{"time_s": float(k["time_s"]),
                 "center_world_m": [float(v) for v in k["center_world_m"]],
                 "rotation_world_wxyz": [float(v) for v in k["rotation_world_wxyz"]]}
                for k in root["keys"]]
        body = {"body_uuid": uuid,
                "runtime_body_index": _v1._u32(frame["runtime_body_index"],
                                                "runtime_body_index", _fail, uuid),
                "mode": root["mode"], "keys": keys,
                "reference_transform": ref,
                **_v2._normalized_controls(root, _fail, uuid)}
        bodies.append(body)
    bodies.sort(key=lambda body: body["runtime_body_index"])
    indices = [body["runtime_body_index"] for body in bodies]
    if len(set(indices)) != len(indices):
        _fail(CODE_BODY_BINDING, "runtime_body_index")
    return bodies


def build_sidecar_payload_v3(authored, scene_uuid, native_frames, fail=None):
    _fail = _raiser(fail)
    payload = {"scene_uuid": _v2._canonical_uuid(scene_uuid, "scene_uuid", _fail),
               "interpolation": INTERPOLATION, "extrapolation": EXTRAPOLATION,
               "bodies": convert_to_native_bodies_v3(authored, native_frames, _fail)}
    return validate_native_payload_v3(payload, _fail)


def validate_native_payload_v3(payload, fail=None):
    """Strict v3 schema check, delegating only common fields to v2 validation."""
    _fail = _raiser(fail)
    if not isinstance(payload, dict) or set(payload) != set(_PAYLOAD_KEYS):
        _fail(CODE_INVALID_KEYS, "payload")
    bodies = payload.get("bodies")
    if not isinstance(bodies, list) or not 1 <= len(bodies) <= MAX_BODIES:
        _fail(CODE_ROOT_BATCH, "bodies")
    common_bodies = []
    for body in bodies:
        if not isinstance(body, dict) or set(body) not in (
                _BODY_KEYS, _BODY_KEYS | _BODY_OPTIONAL):
            _fail(CODE_INVALID_KEYS, "bodies")
        uid = body.get("body_uuid")
        _ref_transform(body["reference_transform"], _fail, uid)
        common = {key: value for key, value in body.items()
                  if key != "reference_transform"}
        common_bodies.append(common)
    _v2.validate_native_payload_v2({"scene_uuid": payload["scene_uuid"],
                                    "interpolation": payload["interpolation"],
                                    "extrapolation": payload["extrapolation"],
                                    "bodies": common_bodies}, _fail)
    # Validate UUID uniqueness and transform entries were checked above; emit
    # a normalized detached payload so all D components are built-in f64s.
    result = {key: payload[key] for key in _PAYLOAD_KEYS}
    result["bodies"] = []
    for body in bodies:
        item = dict(body)
        item["reference_transform"] = _ref_transform(body["reference_transform"], _fail,
                                                      body["body_uuid"])
        result["bodies"].append(item)
    return result


def verify_initial_pose_v3(payload, fixed, fail=None):
    """Verify (P0 * D) reconstructs each fitted native rest-centered body."""
    _fail = _raiser(fail)
    payload = validate_native_payload_v3(payload, _fail)
    fitted = _frames.native_body_frames(fixed)
    rows = np.asarray(fixed._pdrd_body_rows, dtype=np.float64).reshape(-1, 23)
    rest = np.asarray(fixed._pdrd_rest_centered, dtype=np.float64)
    ids = np.asarray(fixed._pdrd_vert_list).astype(np.int64)
    vertices = np.asarray(fixed.vertex(), dtype=np.float64)
    report = {}
    for position, body in enumerate(payload["bodies"]):
        uid = body["body_uuid"]
        frame = fitted.get(uid)
        if frame is None or body["runtime_body_index"] != frame["runtime_body_index"]:
            _fail(CODE_BODY_BINDING, "runtime_body_index", uid)
        index = body["runtime_body_index"]
        start, count = int(rows[index, 0]), int(rows[index, 1])
        body_ids = ids[start:start + count]
        rest_rows = rest[start:start + count]
        key0 = body["keys"][0]
        ref = body["reference_transform"]
        authored_R = _frames._rotation(key0["rotation_world_wxyz"])
        ref_R = _frames._rotation(ref["rotation_local_wxyz"])
        center = np.asarray(key0["center_world_m"], dtype=np.float64) + \
            authored_R @ np.asarray(ref["translation_local_m"], dtype=np.float64)
        rotation = authored_R @ ref_R
        rebuilt = center + (rotation @ rest_rows.T).T
        residual = float(np.max(np.abs(rebuilt - vertices[body_ids])))
        tolerance = float(frame["fit_tolerance_m"])
        if not np.isfinite(residual) or residual > tolerance:
            _fail(CODE_INITIAL_POSE, f"payload.bodies[{position}].keys[0]*reference_transform", uid)
        report[uid] = {"max_reconstruction_error_m": residual,
                       "fit_tolerance_m": tolerance, "runtime_body_index": index}
    return report


def sample_composed_pose_v3(body, time_s):
    """Sample authored P(t), then compose immutable local D (never key-bake D)."""
    normalized = validate_native_payload_v3({
        "scene_uuid": "00000000-0000-0000-0000-000000000001",
        "interpolation": INTERPOLATION, "extrapolation": EXTRAPOLATION,
        "bodies": [body]})["bodies"][0]
    raw = _v1.sample_root_body(normalized, time_s)
    ref = normalized["reference_transform"]
    R = _frames._rotation(raw.rotation_world_wxyz)
    center = np.asarray(raw.center_world_m, dtype=np.float64) + \
        R @ np.asarray(ref["translation_local_m"], dtype=np.float64)
    rotation = _frames._matrix_quat(R @ _frames._rotation(ref["rotation_local_wxyz"]))
    if not np.isfinite(center).all() or not np.isfinite(rotation).all():
        raise ValueError("ROOT_MOTION_INVALID_KEYS: field=reference_transform.composition")
    return _v1.RootPose(tuple(float(v) for v in center),
                        tuple(float(v) for v in rotation))


def required_features_v3(base_features, root_present=True, fail=None):
    _fail = _raiser(fail)
    features = set(base_features)
    roots = {ROOT_FEATURE_V1, ROOT_FEATURE_V2, ROOT_FEATURE}
    advertised = features.intersection(roots)
    if len(advertised) > 1 or (root_present and advertised and advertised != {ROOT_FEATURE}):
        _fail(CODE_REQUIRED_FEATURES, "required_features")
    if root_present:
        features.add(ROOT_FEATURE)
    return sorted(features)


def validate_sidecar_version_v3(envelope_version, root_feature, fail=None):
    _fail = _raiser(fail)
    expected = {ROOT_FEATURE_V1: 1, ROOT_FEATURE_V2: 2, ROOT_FEATURE: 3}
    if root_feature not in expected or type(envelope_version) is not int or envelope_version != expected[root_feature]:
        _fail(CODE_VERSION_OR_KIND, SIDECAR_RELATIVE_PATH)
    return envelope_version


def validate_sidecar_obligation_v3(root_feature, sidecar_present, fail=None):
    _fail = _raiser(fail)
    valid = {None, ROOT_FEATURE_V1, ROOT_FEATURE_V2, ROOT_FEATURE}
    if root_feature not in valid:
        _fail(CODE_REQUIRED_FEATURES, "root_feature")
    if ((root_feature is None and sidecar_present)
            or (root_feature is not None and not sidecar_present)):
        _fail(CODE_VERSION_OR_KIND if root_feature is None else CODE_SIDECAR_MISSING,
              SIDECAR_RELATIVE_PATH)
    return True


def encode_sidecar_v3(payload, fail=None):
    from . import _cbor_bridge_ as _cbor
    normalized = validate_native_payload_v3(payload, fail)
    ordered = {key: normalized[key] for key in _PAYLOAD_KEYS}
    return _cbor.dumps_envelope_with_version(VERSION, KIND, ordered)


def write_sidecar_v3(scene_path, payload, fail=None):
    data = encode_sidecar_v3(payload, fail)
    absolute = os.path.join(scene_path, SIDECAR_RELATIVE_PATH.replace("/", os.sep))
    os.makedirs(os.path.dirname(absolute), exist_ok=True)
    with open(absolute, "wb") as handle:
        handle.write(data)
    return {"relative_path": SIDECAR_RELATIVE_PATH, "byte_length": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def verify_sidecar_inventory_v3(scene_path, inventory, fail=None):
    _fail = _raiser(fail)
    matches = [item for item in inventory if item.get("relative_path") == SIDECAR_RELATIVE_PATH]
    if len(matches) != 1:
        _fail(CODE_SIDECAR_MISSING, SIDECAR_RELATIVE_PATH)
    absolute = os.path.join(scene_path, SIDECAR_RELATIVE_PATH.replace("/", os.sep))
    if not os.path.isfile(absolute):
        _fail(CODE_SIDECAR_MISSING, SIDECAR_RELATIVE_PATH)
    with open(absolute, "rb") as handle:
        data = handle.read()
    entry = matches[0]
    if entry.get("byte_length") != len(data) or entry.get("sha256") != hashlib.sha256(data).hexdigest():
        _fail(CODE_INPUT_HASH_MISMATCH, SIDECAR_RELATIVE_PATH)
    return entry


def decode_sidecar_v3(blob, fail=None):
    """Decode one complete CBOR item; duplicate-map rejection remains native-owned."""
    _fail = _raiser(fail)
    import cbor2
    try:
        stream = io.BytesIO(bytes(blob))
        envelope = cbor2.CBORDecoder(stream).decode()
    except Exception:
        _fail(CODE_VERSION_OR_KIND, "envelope")
    if stream.read(1):
        _fail(CODE_VERSION_OR_KIND, "trailing_cbor")
    if (not isinstance(envelope, dict) or set(envelope) != {"version", "kind", "payload"}
            or type(envelope["version"]) is not int or envelope["kind"] != KIND
            or envelope["version"] != VERSION):
        _fail(CODE_VERSION_OR_KIND, "envelope")
    return validate_native_payload_v3(envelope["payload"], _fail)
