"""Root-motion sidecar schema **version 2**: 1..128 prescribed roots, one path.

CONTRACT.md section 4.1 / decision **U1**: the physical path stays
``bin/pdrd_root_motion_v1.cbor`` for both schema versions; the CBOR envelope
``version`` distinguishes them.  The existing closed context-v2 consumed-input
inventory and digest field stream remain authoritative and **unchanged**, so this
module adds no new ``optional_absent`` name and no digest-format change.

Kind ``ppf.pdrd_root_motion``, envelope ``version: 2``::

    { scene_uuid, interpolation: "linear_center_shortest_slerp_v1",
      extrapolation: "hold",
      bodies: [ { body_uuid, runtime_body_index, mode: "static"|"kinematic",
                  keys: [ { time_s, center_world_m[3], rotation_world_wxyz[4] } ] } ] }

Rules (CONTRACT.md section 4.1):

* ``1 <= len(bodies) <= 128``;
* bodies sorted by ``runtime_body_index`` ascending and unique;
* per-body keys strictly increasing from exactly ``time_s = 0`` with the v1
  initial-pose and unit-quaternion rules preserved;
* canonical non-world UUIDs;
* a present sidecar whose envelope version does not match the required root feature
  **must refuse before any solve** - an unexpected version is never ignored and
  never treated as absent.

The v1 sidecar (envelope version 1, ``MAX_BODIES = 1``) stays valid and
byte-unchanged; :mod:`frontend._pdrd_root_motion_v1_` remains its only owner.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from . import _pdrd_root_motion_v1_ as _v1
from ._pdrd_native_frames_ import _rotation, native_body_frames

VERSION = 2
KIND = _v1.SIDECAR_KIND
SIDECAR_RELATIVE_PATH = _v1.SIDECAR_RELATIVE_PATH
INTERPOLATION = _v1.INTERPOLATION
EXTRAPOLATION = _v1.EXTRAPOLATION
ROOT_FEATURE = "pdrd_kinematic_roots_v2"
ROOT_FEATURE_V1 = "pdrd_kinematic_roots_v1"
BODY_CONTROLS_FEATURE = "pdrd_body_controls_v1"

MAX_BODIES = 128
MAX_KEYS_PER_ROOT = 4096
MAX_KEY_RECORDS_TOTAL = 65536
BODY_RECORD_SIZE = 64
KEY_RECORD_SIZE = 64
STEP_RECORD_SIZE = 128
CAPABILITIES_SIZE = 48
ABI_VERSION = 2

MODES = {"dynamic": 0, "static": 1, "kinematic": 2}

CODE_VERSION_OR_KIND = _v1.CODE_VERSION_OR_KIND
CODE_REQUIRED_FEATURES = _v1.CODE_REQUIRED_FEATURES
CODE_SIDECAR_MISSING = _v1.CODE_SIDECAR_MISSING
CODE_BODY_BINDING = _v1.CODE_BODY_BINDING
CODE_INVALID_KEYS = _v1.CODE_INVALID_KEYS
CODE_INITIAL_POSE = _v1.CODE_INITIAL_POSE
CODE_INPUT_HASH_MISMATCH = _v1.CODE_INPUT_HASH_MISMATCH
CODE_OLD_PIN_OVERLAP = _v1.CODE_OLD_PIN_OVERLAP
CODE_UNSUPPORTED_DOMAIN = _v1.CODE_UNSUPPORTED_DOMAIN
CODE_NATIVE_CAPABILITY = _v1.CODE_NATIVE_CAPABILITY
CODE_ROOT_BATCH = "G2_D6_ROOT_BATCH_INVALID"
CODE_CAPACITY = "G2_D6_CAPACITY_OVERFLOW"

_PAYLOAD_ORDER = ("scene_uuid", "interpolation", "extrapolation", "bodies")
_ENVELOPE_KEYS = {"version", "kind", "payload"}
_BODY_KEYS = {"body_uuid", "runtime_body_index", "mode", "keys"}
_KEY_KEYS = {"time_s", "center_world_m", "rotation_world_wxyz"}
#: The author vocabulary of a deferred-forest root record, accepted on input only.
_AUTHOR_ROOT_KEYS = {"body_uuid", "mode", "keys", "root_index", "authoring",
                     "initial_frame", "fit_tolerance_m", "control_events"}
_AUTHOR_KEY_ALIASES = (("translation_m", "center_world_m"),
                       ("quaternion_wxyz", "rotation_world_wxyz"))


def _raiser(fail):
    return _v1._raiser(fail)


def _canonical_uuid(value, field, fail, body_uuid=None):
    return _v1._uuid_canonical(value, field, fail, body_uuid)


def _normalize_key(key, fail, body_uuid):
    """Accept the author key vocabulary and emit the closed wire key map."""
    if not isinstance(key, dict):
        fail(CODE_INVALID_KEYS, "keys", body_uuid=body_uuid)
    author_keys = {"time_s", "translation_m", "quaternion_wxyz"}
    wire_keys = {"time_s", "center_world_m", "rotation_world_wxyz"}
    if set(key) not in (author_keys, wire_keys):
        fail(CODE_INVALID_KEYS, "keys", body_uuid=body_uuid)
    entry = {}
    for alias, wire in _AUTHOR_KEY_ALIASES:
        if alias in key and wire in key:
            fail(CODE_INVALID_KEYS, wire, body_uuid=body_uuid)
        if alias in key:
            entry[wire] = key[alias]
    for wire in ("center_world_m", "rotation_world_wxyz"):
        if wire in key:
            entry[wire] = key[wire]
    entry["time_s"] = key.get("time_s")
    return entry


def _validate_mode_key_count(mode, key_count, fail, body_uuid):
    if ((mode == "static" and key_count != 1)
            or (mode == "kinematic" and key_count < 2)):
        fail(CODE_INVALID_KEYS, "keys", body_uuid)


def _normalized_controls(body, fail, body_uuid):
    """V2 explicit current-pose binding; omitted events keep legacy bytes."""
    if "control_events" not in body:
        return {}
    events = body["control_events"]
    if not isinstance(events, list) or len(events) > 4096:
        fail(CODE_INVALID_KEYS, "control_events", body_uuid=body_uuid)
    previous, mode, normalized = 0.0, body["mode"], []
    for event in events:
        if (not isinstance(event, dict) or set(event) not in (
                {"time_s", "mode"}, {"time_s", "mode", "pose_binding"})):
            fail(CODE_INVALID_KEYS, "control_events", body_uuid=body_uuid)
        time = _v1._finite_real(event["time_s"], "control_events.time_s", fail, body_uuid)
        selected = event["mode"]
        if time <= previous or selected not in MODES or selected == mode:
            fail(CODE_INVALID_KEYS, "control_events", body_uuid=body_uuid)
        entry = {"time_s": time, "mode": selected}
        if "pose_binding" in event:
            policy = event["pose_binding"]
            allowed = {"static": {"absolute", "current_hold"},
                       "kinematic": {"absolute", "rigid_delta_v1"}}
            if policy not in allowed.get(selected, set()):
                fail(CODE_INVALID_KEYS, "control_events.pose_binding", body_uuid=body_uuid)
            entry["pose_binding"] = policy
        normalized.append(entry)
        previous, mode = time, selected
    return {"control_events": normalized} if normalized else {}


def requires_body_controls(payload):
    """New modes/policies may never enter the legacy V2 setter implicitly."""
    return any(body["mode"] == "dynamic" or any(
        "pose_binding" in event for event in body.get("control_events", []))
        for body in payload["bodies"])


# ---------------------------------------------------------------------------
# Authoring: validate the deferred-forest root vocabulary
# ---------------------------------------------------------------------------
def validate_authored_v2(authored, fail=None):
    """Validate the authored multi-root block before native frames exist.

    Accepted shape: ``{"roots": [ {body_uuid, mode, keys:[...]}, ... ]}`` or a bare
    list of such records, using either the wire key names or the author aliases
    ``translation_m`` / ``quaternion_wxyz``.
    """
    _fail = _raiser(fail)
    if isinstance(authored, dict) and set(authored) not in (
            {"roots"}, {"roots", "total_keys"}):
        _fail(CODE_BODY_BINDING, "authored")
    roots = authored.get("roots") if isinstance(authored, dict) else authored
    if not isinstance(roots, (list, tuple)) or not 1 <= len(roots) <= MAX_BODIES:
        _fail(CODE_ROOT_BATCH, "roots")
    seen = set()
    seen_root_indices = set()
    result = []
    total_keys = 0
    for position, root in enumerate(roots):
        if not isinstance(root, dict):
            _fail(CODE_BODY_BINDING, "roots")
        extras = set(root) - _AUTHOR_ROOT_KEYS - {"fit_tolerance_m"}
        if extras:
            _fail(CODE_BODY_BINDING, sorted(extras)[0])
        uuid = _canonical_uuid(root.get("body_uuid"), "body_uuid", _fail)
        if uuid in seen:
            _fail(CODE_BODY_BINDING, "body_uuid", uuid)
        seen.add(uuid)
        mode = root.get("mode", "kinematic")
        if mode not in MODES:
            _fail(CODE_UNSUPPORTED_DOMAIN, "mode", uuid)
        root_index = root.get("root_index")
        if root_index is not None:
            if (isinstance(root_index, bool) or not isinstance(root_index, int)
                    or root_index < 0 or root_index in seen_root_indices):
                _fail(CODE_ROOT_BATCH, "root_index", uuid)
            seen_root_indices.add(root_index)
        authoring = root.get("authoring")
        initial_frame = root.get("initial_frame")
        if (authoring is None) != (initial_frame is None):
            _fail(CODE_BODY_BINDING, "authoring/initial_frame", uuid)
        if authoring is not None and authoring not in _v1.AUTHORING_MODES:
            _fail(CODE_BODY_BINDING, "authoring", uuid)
        if initial_frame is not None:
            initial_frame = _v1._frame(initial_frame, "initial_frame", _fail, uuid)
        fit_tolerance = root.get("fit_tolerance_m")
        if fit_tolerance is not None:
            fit_tolerance = _v1._finite_real(
                fit_tolerance, "fit_tolerance_m", _fail, uuid)
            if fit_tolerance <= 0.0:
                _fail(CODE_BODY_BINDING, "fit_tolerance_m", uuid)
        keys = root.get("keys")
        if not isinstance(keys, (list, tuple)) or not 1 <= len(keys) <= MAX_KEYS_PER_ROOT:
            _fail(CODE_INVALID_KEYS, "keys", uuid)
        if mode != "static" or not root.get("control_events"):
            _validate_mode_key_count(mode, len(keys), _fail, uuid)
        normalized = []
        previous = None
        for position, key in enumerate(keys):
            entry = _normalize_key(key, _fail, uuid)
            time_s = _v1._finite_real(entry["time_s"], "time_s", _fail, uuid)
            if position == 0 and time_s != 0.0:
                _fail(CODE_INVALID_KEYS, "time_s", uuid)
            if previous is not None and time_s <= previous:
                _fail(CODE_INVALID_KEYS, "time_s", uuid)
            previous = time_s
            center = _v1._real_list(entry["center_world_m"], 3, "center_world_m",
                                    _fail, uuid)
            rotation = _v1._unit_quaternion(entry["rotation_world_wxyz"],
                                            "rotation_world_wxyz", _fail, uuid)
            normalized.append({"time_s": time_s, "center_world_m": center,
                               "rotation_world_wxyz": [float(v) for v in rotation]})
        if initial_frame is not None:
            first = normalized[0]
            initial_q = initial_frame["quaternion_wxyz"]
            if (first["time_s"] != 0.0
                    or not np.array_equal(np.asarray(first["center_world_m"]),
                                          np.asarray(initial_frame["translation_m"]))
                    or abs(_v1.quaternion_dot(first["rotation_world_wxyz"], initial_q))
                    < 1.0 - _v1.QUAT_NORM_TOL):
                _fail(CODE_INITIAL_POSE, "initial_frame", uuid)
        total_keys += len(normalized)
        if total_keys > MAX_KEY_RECORDS_TOTAL:
            _fail(CODE_CAPACITY, "keys", uuid)
        entry = {"body_uuid": uuid, "mode": mode, "keys": normalized, **_normalized_controls(root, _fail, uuid)}
        if root_index is not None:
            entry["root_index"] = root_index
        if authoring is not None:
            entry["authoring"] = authoring
            entry["initial_frame"] = initial_frame
        if fit_tolerance is not None:
            entry["fit_tolerance_m"] = fit_tolerance
        result.append(entry)
    if isinstance(authored, dict) and "total_keys" in authored:
        if (type(authored["total_keys"]) is not int
                or authored["total_keys"] != total_keys):
            _fail(CODE_CAPACITY, "total_keys")
    return {"roots": result, "total_keys": total_keys}


def convert_to_native_bodies_v2(authored, native_frames, fail=None):
    """Bind authored roots to actual native PDRD frames and sort by body index."""
    _fail = _raiser(fail)
    if not isinstance(native_frames, dict) or not native_frames:
        _fail(CODE_BODY_BINDING, "native_frames")
    bodies = []
    for root in authored["roots"]:
        uuid = root["body_uuid"]
        if uuid not in native_frames:
            _fail(CODE_BODY_BINDING, "body_uuid", uuid)
        frame = native_frames[uuid]
        if root.get("authoring") == _v1.AUTHORING_OBJECT_WORLD:
            initial = root["initial_frame"]
            authored_origin = np.asarray(initial["translation_m"], dtype=np.float64)
            authored_rotation = _rotation(initial["quaternion_wxyz"])
            native_origin = np.asarray(frame["world_translation_m"], dtype=np.float64)
            native_rotation = _rotation(frame["world_quaternion_wxyz"])
            keys = []
            for key in root["keys"]:
                delta = _rotation(key["rotation_world_wxyz"]) @ authored_rotation.T
                center = np.asarray(key["center_world_m"], dtype=np.float64) + \
                    delta @ (native_origin - authored_origin)
                quaternion = _v1._matrix_quat(delta @ native_rotation)
                keys.append({"time_s": key["time_s"],
                             "center_world_m": [float(v) for v in center],
                             "rotation_world_wxyz": [float(v) for v in quaternion]})
        else:
            keys = root["keys"]
        bodies.append({
            "body_uuid": uuid,
            "runtime_body_index": _v1._u32(frame["runtime_body_index"],
                                           "runtime_body_index", _fail, uuid),
            "mode": root["mode"],
            "keys": _v1.sign_continuous_quaternions(keys, _fail, uuid),
            **_normalized_controls(root, _fail, uuid),
        })
    bodies.sort(key=lambda body: body["runtime_body_index"])
    indices = [body["runtime_body_index"] for body in bodies]
    if len(set(indices)) != len(indices):
        _fail(CODE_BODY_BINDING, "runtime_body_index")
    return bodies


def build_sidecar_payload_v2(authored, scene_uuid, native_frames, fail=None):
    """Build the version-2 sidecar payload (same physical path as v1, U1)."""
    _fail = _raiser(fail)
    validated = validate_authored_v2(authored, _fail)
    bodies = convert_to_native_bodies_v2(validated, native_frames, _fail)
    payload = {
        "scene_uuid": _canonical_uuid(scene_uuid, "scene_uuid", _fail),
        "interpolation": INTERPOLATION,
        "extrapolation": EXTRAPOLATION,
        "bodies": bodies,
    }
    return validate_native_payload_v2(payload, _fail)


def validate_native_payload_v2(payload, fail=None):
    """Exact-map validation of a version-2 sidecar payload (closed vocabulary)."""
    _fail = _raiser(fail)
    if not isinstance(payload, dict) or set(payload) != set(_PAYLOAD_ORDER):
        _fail(CODE_INVALID_KEYS, "payload")
    if payload["interpolation"] != INTERPOLATION or payload["extrapolation"] != EXTRAPOLATION:
        _fail(CODE_VERSION_OR_KIND, "interpolation/extrapolation")
    bodies = payload["bodies"]
    if not isinstance(bodies, list) or not 1 <= len(bodies) <= MAX_BODIES:
        _fail(CODE_ROOT_BATCH, "bodies")
    if any(not isinstance(body, dict) or set(body) not in (_BODY_KEYS, _BODY_KEYS | {"control_events"}) for body in bodies):
        _fail(CODE_INVALID_KEYS, "bodies")
    indices = [_v1._u32(body["runtime_body_index"], "runtime_body_index", _fail,
                        body.get("body_uuid")) for body in bodies]
    if indices != sorted(indices) or len(set(indices)) != len(indices):
        _fail(CODE_ROOT_BATCH, "bodies.runtime_body_index")
    total = 0
    for body in bodies:
        uuid = _canonical_uuid(body["body_uuid"], "body_uuid", _fail)
        if uuid == _v1.WORLD_UUID:
            _fail(CODE_BODY_BINDING, "body_uuid", uuid)
        if body["mode"] not in MODES:
            _fail(CODE_UNSUPPORTED_DOMAIN, "mode", uuid)
        keys = body["keys"]
        if not isinstance(keys, list) or not 1 <= len(keys) <= MAX_KEYS_PER_ROOT:
            _fail(CODE_INVALID_KEYS, "keys", uuid)
        _normalized_controls(body, _fail, uuid)
        if body["mode"] != "static" or not body.get("control_events"):
            _validate_mode_key_count(body["mode"], len(keys), _fail, uuid)
        previous = None
        for position, key in enumerate(keys):
            if not isinstance(key, dict) or set(key) != _KEY_KEYS:
                _fail(CODE_INVALID_KEYS, "keys", uuid)
            time_s = _v1._finite_real(key["time_s"], "time_s", _fail, uuid)
            if position == 0 and time_s != 0.0:
                _fail(CODE_INVALID_KEYS, "time_s", uuid)
            if previous is not None and time_s <= previous:
                _fail(CODE_INVALID_KEYS, "time_s", uuid)
            previous = time_s
            _v1._real_list(key["center_world_m"], 3, "center_world_m", _fail, uuid)
            _v1._unit_quaternion(key["rotation_world_wxyz"],
                                 "rotation_world_wxyz", _fail, uuid)
        total += len(keys)
    if total > MAX_KEY_RECORDS_TOTAL:
        _fail(CODE_CAPACITY, "keys")
    return payload


# ---------------------------------------------------------------------------
# Surface selection, version obligation, encode/decode
# ---------------------------------------------------------------------------
def root_surface_for(bodies, root_v2_symbols_loaded, v1_symbols_loaded, fail=None):
    """Contract sections 4.0/10: which root surface the authored root count requires."""
    _fail = _raiser(fail)
    if len(bodies) == 0:
        return "none"
    if len(bodies) == 1:
        if not v1_symbols_loaded:
            _fail(CODE_NATIVE_CAPABILITY, "pdrd_root_motion_query_v1")
        return "root-v1"
    if not root_v2_symbols_loaded:
        _fail(CODE_ROOT_BATCH, "pdrd_root_motion_v2_query_v1")
    return "root-v2"


def required_features_v2(base_features, root_feature, fail=None):
    """Sorted unique union including exactly one root feature, or neither."""
    _fail = _raiser(fail)
    features = set(base_features)
    if root_feature is not None:
        if root_feature not in (ROOT_FEATURE_V1, ROOT_FEATURE):
            _fail(CODE_REQUIRED_FEATURES, "root_feature")
        other = ROOT_FEATURE_V1 if root_feature == ROOT_FEATURE else ROOT_FEATURE
        if other in features:
            _fail(CODE_REQUIRED_FEATURES, "required_features")
        features.add(root_feature)
    return sorted(features)


def validate_sidecar_version(envelope_version, required_root_feature, fail=None):
    """One path, version-distinguished (U1): a mismatch refuses before any solve."""
    _fail = _raiser(fail)
    if required_root_feature == ROOT_FEATURE:
        if envelope_version != VERSION:
            _fail(CODE_VERSION_OR_KIND, SIDECAR_RELATIVE_PATH)
    elif required_root_feature == ROOT_FEATURE_V1:
        if envelope_version != _v1.SIDECAR_VERSION:
            _fail(CODE_VERSION_OR_KIND, SIDECAR_RELATIVE_PATH)
    else:
        _fail(CODE_REQUIRED_FEATURES, "required_features")
    return envelope_version


def validate_sidecar_obligation_v2(root_feature, sidecar_present, fail=None):
    """The feature and the sidecar must agree in both directions (section 10, U1)."""
    _fail = _raiser(fail)
    if root_feature in (ROOT_FEATURE, ROOT_FEATURE_V1) and not sidecar_present:
        _fail(CODE_SIDECAR_MISSING, SIDECAR_RELATIVE_PATH)
    if root_feature is None and sidecar_present:
        _fail(CODE_VERSION_OR_KIND, SIDECAR_RELATIVE_PATH)
    if root_feature not in (None, ROOT_FEATURE, ROOT_FEATURE_V1):
        _fail(CODE_REQUIRED_FEATURES, "root_feature")
    return True


def encode_sidecar_v2(payload, fail=None):
    """Encode with the live frontend CBOR bridge at envelope version 2."""
    from . import _cbor_bridge_ as _cbor
    normalized = validate_native_payload_v2(payload, fail)
    ordered = {key: normalized[key] for key in _PAYLOAD_ORDER}
    return _cbor.dumps_envelope_with_version(VERSION, KIND, ordered)


def write_sidecar_v2(scene_path, payload, fail=None):
    """Write to the one existing sidecar path (U1) and return its inventory entry."""
    import hashlib
    import os
    data = encode_sidecar_v2(payload, fail)
    absolute = os.path.join(scene_path, SIDECAR_RELATIVE_PATH.replace("/", os.sep))
    os.makedirs(os.path.dirname(absolute), exist_ok=True)
    with open(absolute, "wb") as handle:
        handle.write(data)
    return {"relative_path": SIDECAR_RELATIVE_PATH, "byte_length": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def verify_sidecar_inventory_v2(scene_path, inventory, fail=None):
    """Require the exact shared-path sidecar bytes/hash in context-v2 inventory."""
    import hashlib
    import os
    _fail = _raiser(fail)
    matches = [item for item in inventory
               if item.get("relative_path") == SIDECAR_RELATIVE_PATH]
    if len(matches) != 1:
        _fail(CODE_SIDECAR_MISSING, SIDECAR_RELATIVE_PATH)
    absolute = os.path.join(scene_path, SIDECAR_RELATIVE_PATH.replace("/", os.sep))
    if not os.path.isfile(absolute):
        _fail(CODE_SIDECAR_MISSING, SIDECAR_RELATIVE_PATH)
    with open(absolute, "rb") as handle:
        data = handle.read()
    entry = matches[0]
    if (entry.get("byte_length") != len(data)
            or entry.get("sha256") != hashlib.sha256(data).hexdigest()):
        _fail(CODE_INPUT_HASH_MISMATCH, SIDECAR_RELATIVE_PATH)
    return entry


def verify_initial_pose_v2(payload, fixed, fail=None):
    """Check every prescribed root key against its built native PDRD geometry."""
    _fail = _raiser(fail)
    payload = validate_native_payload_v2(payload, _fail)
    fitted = native_body_frames(fixed)
    rows = np.asarray(fixed._pdrd_body_rows, dtype=np.float64).reshape(-1, 23)
    rest = np.asarray(fixed._pdrd_rest_centered, dtype=np.float64)
    ids = np.asarray(fixed._pdrd_vert_list).astype(np.int64)
    vertices = np.asarray(fixed.vertex(), dtype=np.float64)
    report = {}
    for position, body in enumerate(payload["bodies"]):
        body_uuid = body["body_uuid"]
        if body_uuid not in fitted:
            _fail(CODE_BODY_BINDING, "payload.bodies", body_uuid)
        if body["runtime_body_index"] != fitted[body_uuid]["runtime_body_index"]:
            _fail(CODE_BODY_BINDING, "runtime_body_index", body_uuid)
        index = body["runtime_body_index"]
        start, count = int(rows[index, 0]), int(rows[index, 1])
        body_ids = ids[start:start + count]
        rest_rows = rest[start:start + count]
        key0 = body["keys"][0]
        center = np.asarray(key0["center_world_m"], dtype=np.float64)
        rotation = _rotation(key0["rotation_world_wxyz"])
        rebuilt = center + (rotation @ rest_rows.T).T
        residual = float(np.max(np.abs(rebuilt - vertices[body_ids])))
        tolerance = float(fitted[body_uuid]["fit_tolerance_m"])
        if not np.isfinite(residual) or residual > tolerance:
            _fail(CODE_INITIAL_POSE, f"payload.bodies[{position}].keys[0]", body_uuid)
        report[body_uuid] = {"max_reconstruction_error_m": residual,
                             "fit_tolerance_m": tolerance,
                             "runtime_body_index": index}
    return report


def peek_sidecar_version(blob, fail=None):
    """Return ``(version, kind)`` without validating the payload (U1 dispatch)."""
    _fail = _raiser(fail)
    import cbor2
    try:
        envelope = cbor2.loads(bytes(blob))
    except Exception:
        _fail(CODE_VERSION_OR_KIND, "envelope")
    if not isinstance(envelope, dict) or set(envelope) != _ENVELOPE_KEYS:
        _fail(CODE_VERSION_OR_KIND, "envelope")
    if type(envelope["version"]) is not int or envelope["kind"] != KIND:
        _fail(CODE_VERSION_OR_KIND, "kind")
    return int(envelope["version"]), envelope["kind"]


def decode_sidecar_v2(blob, fail=None):
    """Decode+validate a version-2 sidecar blob; a wrong version is a refusal.

    ``cbor2`` keeps the last duplicate map key, so duplicate refusal stays a
    Rust-side guarantee; this reader exists so the Python side can round-trip its
    own bytes and so envelope tests do not need a native module.
    """
    _fail = _raiser(fail)
    version, _kind = peek_sidecar_version(blob, _fail)
    if version != VERSION:
        _fail(CODE_VERSION_OR_KIND, SIDECAR_RELATIVE_PATH)
    import cbor2
    envelope = cbor2.loads(bytes(blob))
    return validate_native_payload_v2(envelope["payload"], _fail)


def min_dt_rule(bodies, t0_s, limit_dt_s):
    """One atomic clock (section 4.3): the trial dt is the minimum over all roots."""
    trial = float(limit_dt_s)
    for body in bodies:
        keys = body["keys"]
        for index in range(len(keys) - 1):
            if keys[index]["time_s"] <= t0_s < keys[index + 1]["time_s"]:
                trial = min(trial, float(keys[index + 1]["time_s"]) - float(t0_s))
                break
        else:
            if t0_s < keys[0]["time_s"]:
                trial = min(trial, float(keys[0]["time_s"]) - float(t0_s))
    return trial


def buffer_layout(counts):
    """Contract section 8 sizing for the root-v2 surface."""
    bodies = counts.get("bodies", 0)
    keys = counts.get("keys", 0)
    return {
        "root_body_records": bodies,
        "root_body_bytes": bodies * BODY_RECORD_SIZE,
        "root_key_records": keys,
        "root_key_bytes": keys * KEY_RECORD_SIZE,
        "root_step_records": bodies,
        "root_step_bytes": bodies * STEP_RECORD_SIZE,
        "capabilities_size": CAPABILITIES_SIZE,
        "body_record_size": BODY_RECORD_SIZE,
        "key_record_size": KEY_RECORD_SIZE,
        "step_record_size": STEP_RECORD_SIZE,
        "max_roots": MAX_BODIES,
        "max_keys_per_root": MAX_KEYS_PER_ROOT,
        "max_key_records_total": MAX_KEY_RECORDS_TOTAL,
        "abi_version": ABI_VERSION,
    }


def copy_as_v1(payload, fail=None):
    """Project a single-root version-2 payload onto the v1 schema (not written)."""
    _fail = _raiser(fail)
    if len(payload["bodies"]) != 1:
        _fail(CODE_ROOT_BATCH, "bodies")
    body = payload["bodies"][0]
    return {
        "scene_uuid": payload["scene_uuid"],
        "interpolation": INTERPOLATION,
        "extrapolation": EXTRAPOLATION,
        "bodies": [{"body_uuid": body["body_uuid"],
                    "runtime_body_index": body["runtime_body_index"],
                    "mode": body["mode"],
                    "keys": deepcopy(body["keys"])}],
    }


def body_mode_number(mode):
    return MODES[mode]


def finite(value):
    return bool(np.isfinite(value))
