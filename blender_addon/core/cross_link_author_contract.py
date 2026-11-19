"""Serialized cross-link author validation at the body-control upload boundary.

Matches frontend._cross_links_.validate_author; no Blender sampling or imports.
The frontend still validates/binds the complete network against the native scene.
"""
import math
import uuid
import numpy as np

AUTHOR_SCHEMA = "PPF_BONE_CROSS_LINKS_V1"
MAX_LINKS = 65536
FLOAT32_MAX = float(np.finfo(np.float32).max)

def fail(field, detail):
    raise ValueError(f"cross_links_v1: {field}: {detail}")

def canonical_uuid(value, field):
    try:
        parsed = uuid.UUID(value)
        if str(parsed) != value or parsed.int == 0:
            fail(field, "expected canonical nonzero UUID")
    except (ValueError, TypeError, AttributeError):
        fail(field, "expected canonical nonzero UUID")
    return value

def _number(value, field, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        fail(field, "expected a finite number")
    result = float(value)
    if result < 0 or positive and result == 0 or result > FLOAT32_MAX:
        fail(field, "expected a representable positive value" if positive else
             "expected a representable nonnegative value")
    if result != 0 and np.float32(result) == 0:
        fail(field, "nonzero value underflows native float32")
    return result


def _vector(value, field):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        fail(field, "expected three coordinates")
    if any(type(v) not in (int, float) or not math.isfinite(v)
           or abs(v) > FLOAT32_MAX
           or v != 0 and np.float32(v) == 0 for v in value):
        fail(field, "expected finite float32-representable coordinates")
    return np.asarray(value, dtype=np.float64)


def _validate_author(author, bindings):
    """Pure preflight: refuse the entire network before touching FixedScene."""
    if (not isinstance(author, dict) or set(author) != {"schema", "links"}
            or author["schema"] != AUTHOR_SCHEMA):
        fail("schema", "expected " + AUTHOR_SCHEMA)
    links = author["links"]
    if not isinstance(links, list) or not links or len(links) > MAX_LINKS:
        fail("links", "expected a nonempty bounded list")
    by_body = {row["body_uuid"]: row for row in bindings}
    seen, pairs = set(), set()
    normalized = []
    for row in links:
        fields = {"link_uuid", "model", "endpoint_a", "endpoint_b", "rest_length_m",
                  "stiffness_N_per_m", "damping_Ns_per_m"}
        directional = {"tension_stiffness_N_per_m", "compression_stiffness_N_per_m",
                       "tension_enabled", "compression_enabled"}
        if not isinstance(row, dict) or set(row) not in (fields, fields | directional):
            fail("link", "unknown or missing fields")
        uid = canonical_uuid(row["link_uuid"], "cross_link.link_uuid")
        if uid in seen or row["model"] != "AXIAL_DISTANCE":
            fail("link", "duplicate identity or unsupported spring model")
        seen.add(uid)
        endpoints = []
        for key in ("endpoint_a", "endpoint_b"):
            end = row[key]
            if not isinstance(end, dict) or set(end) != {"body_uuid", "anchor_world_blender"}:
                fail(key, "expected body UUID and initial world anchor")
            body_uuid = canonical_uuid(end["body_uuid"], "cross_link.body_uuid")
            binding = by_body.get(body_uuid)
            if binding is None or binding.get("enabled") is not True:
                fail(key, "endpoint must belong to an enabled bone binding")
            point = _vector(end["anchor_world_blender"], key + ".anchor_world_blender")
            endpoints.append((body_uuid, point, binding["armature_uuid"]))
        if endpoints[0][0] == endpoints[1][0] or endpoints[0][2] != endpoints[1][2]:
            fail("endpoints", "expected two different bodies in the same armature")
        pair = tuple(sorted((endpoints[0][0], endpoints[1][0])))
        if pair in pairs:
            fail("endpoints", "duplicate body pair")
        pairs.add(pair)
        k = _number(row["stiffness_N_per_m"], "stiffness_N_per_m")
        c = _number(row["damping_Ns_per_m"], "damping_Ns_per_m")
        kt = _number(row.get("tension_stiffness_N_per_m", k), "tension_stiffness_N_per_m")
        kc = _number(row.get("compression_stiffness_N_per_m", k), "compression_stiffness_N_per_m")
        directional_enabled = directional.issubset(row)
        if directional_enabled and any(type(row[key]) is not bool for key in ("tension_enabled", "compression_enabled")):
            fail("direction", "enabled flags must be booleans")
        te = row.get("tension_enabled", True)
        ce = row.get("compression_enabled", True)
        if not directional_enabled and k == 0 and c == 0:
            fail("coefficients", "stiffness and damping cannot both be zero")
        length = _number(row["rest_length_m"], "rest_length_m")
        initial_length = float(np.linalg.norm(endpoints[0][1] - endpoints[1][1]))
        if not math.isclose(length, initial_length, rel_tol=2e-5, abs_tol=2e-6):
            fail("rest_length_m", "reference must match this simulation's initial pose")
        normalized.append((uid, endpoints, k, c, kt, kc, te, ce, directional_enabled))
    return normalized


def validate_author(author, bindings, reject):
    """Reject only malformed requested cross-links; never mutate author bytes."""
    try:
        _validate_author(author, bindings)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        reject(str(exc))
