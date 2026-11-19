"""Bind authored cross-chain springs to actual native material endpoints.

Cross-links are a separate constraint graph.  They never become skeleton
parents and do not change collision filtering or add simulated bodies.
"""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np

FEATURE = "cross_links_v1"
SIDECAR = "bin/cross_links_v1.json"
AUTHOR_SCHEMA = "PPF_BONE_CROSS_LINKS_V1"
MAX_LINKS = 65536
FLOAT32_MAX = float(np.finfo(np.float32).max)


def fail(field, detail):
    raise ValueError(f"BONE_CROSS_LINK_INVALID: {field}: {detail}")


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


def validate_author(author, bindings):
    """Pure preflight: refuse the entire network before touching FixedScene."""
    from ._blender_bones_ import canonical_uuid
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
        directional = {"tension_stiffness_N_per_m", "compression_stiffness_N_per_m", "tension_enabled", "compression_enabled"}
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


def capability(contract, *, directional=False, collision_pairs=False):
    """An explicit new request must be accepted by the frozen native product."""
    from ._build_contract_ import verify_build_contract
    from ._r14_product_profile import solver_child_env
    verify_build_contract(contract)
    profile = contract.profile
    env = solver_child_env(profile, os.environ)
    try:
        result = subprocess.run(
            [str(contract.exe_path), "--cross-links-capability-json"],
            cwd=profile.runtime_dir, env=env, capture_output=True, timeout=30,
            check=True)
        document = json.loads(result.stdout)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise ValueError("BONE_CROSS_LINK_CAPABILITY_UNAVAILABLE: select a matching "
                         "solver with cross_links_v1 and Build again") from exc
    expected = {"kind": "ppf.cross_links.capability", "version": 1,
                "required_feature": FEATURE, "supported": True,
                "semantic_version": 1, "model": "AXIAL_DISTANCE"}
    if (not isinstance(document, dict)
            or any(document.get(key) != value for key, value in expected.items())
            or type(document.get("version")) is not int
            or type(document.get("semantic_version")) is not int
            or document.get("supported") is not True):
        raise ValueError("BONE_CROSS_LINK_CAPABILITY_UNAVAILABLE: incompatible native spring law")
    if directional and (type(document.get("directional_controls_version")) is not int
                        or document.get("directional_controls_version") != 1
                        or document.get("full_force_direction_switches") is not True):
        raise ValueError("BONE_CROSS_LINK_DIRECTION_CAPABILITY_UNAVAILABLE: select a solver with directional force controls and Build again")
    if collision_pairs and (type(document.get("cross_link_collision_pairs_version")) is not int
                            or document.get("cross_link_collision_pairs_version") != 1):
        raise ValueError("BONE_CROSS_LINK_COLLISION_CAPABILITY_UNAVAILABLE: select a solver with independent collision pairs and Build again")
    return document


def install(fixed, author, bone_document, receipt):
    """Stage the complete endpoint network, then attach it in one assignment."""
    from ._blender_bones_ import S, frame_matrix
    from ._pdrd_native_frames_ import native_body_frames
    from ._fem_material_precision_ import native_rest_positions, material_anchor_bary
    links = validate_author(author, bone_document["bindings"])
    native = native_body_frames(fixed)
    material = {body["body_uuid"]: body for body in
                (getattr(fixed, "_fem_bone_joint_payload", None) or {}).get("bodies", [])}
    rest = None
    descriptors, output = {}, []

    def bind(endpoint):
        nonlocal rest
        uid, initial_world, _ = endpoint
        point = S[:3, :3] @ initial_world
        if uid in native:
            transform = frame_matrix(native[uid])
            local = transform[:3, :3].T @ (point - transform[:3, 3])
            descriptor = {"kind": "PDRD", "body_uuid": uid,
                          "runtime_body_index": native[uid]["runtime_body_index"]}
            bound = {"body_uuid": uid, "local_anchor_m": local.tolist()}
        else:
            descriptor = deepcopy(material.get(uid))
            if not descriptor or descriptor.get("kind") != "FEM_TET_MATERIAL":
                fail("body_uuid", "missing actual native material support: " + uid)
            if rest is None:
                rest = native_rest_positions(fixed)
            nodes = rest[np.asarray(descriptor["vertex_indices"], dtype=np.int64)]
            center = np.asarray(descriptor["center_bary"], dtype=np.float32).astype(np.float64)
            rotation = np.asarray(descriptor["frame_rest_column_major"], dtype=np.float32
                                  ).astype(np.float64).reshape(3, 3, order="F")
            local = np.linalg.solve(rotation, point - center @ nodes)
            bary = material_anchor_bary(nodes, descriptor["center_bary"],
                                        descriptor["frame_rest_column_major"], local)
            bound = {"body_uuid": uid, "local_anchor_m": local.tolist(),
                     "anchor_bary": bary.tolist()}
            point = bary @ nodes
        _vector(bound["local_anchor_m"], "native.local_anchor_m")
        descriptors[uid] = descriptor
        return bound, point

    for uid, endpoints, stiffness, damping, kt, kc, te, ce, has_directional in links:
        a, pa = bind(endpoints[0])
        b, pb = bind(endpoints[1])
        length = _number(float(np.linalg.norm(pa - pb)), "native.rest_length_m")
        output.append({"link_uuid": uid, "enabled": True, "model": "AXIAL_DISTANCE",
                       "a": a, "b": b, "rest_length_m": length,
                       "epsilon_m": max(1e-6, length * 1e-4),
                       "stiffness_N_per_m": stiffness, "damping_Ns_per_m": damping})
        if has_directional:
            output[-1].update(tension_stiffness_N_per_m=kt, compression_stiffness_N_per_m=kc,
                              tension_enabled=te, compression_enabled=ce)
    payload = {"version": 1, "kind": "ppf.cross_links", "scene_uuid": fixed._scene_uuid,
               "bodies": [descriptors[key] for key in sorted(descriptors)], "links": output}
    # Parent joints and their receipt remain intact.  This adds no pose authority.
    updated_receipt = deepcopy(receipt)
    updated_receipt["required_features"] = sorted(set(
        updated_receipt.get("required_features", []) + [FEATURE]))
    updated_receipt["cross_links"] = [dict(link_uuid=row["link_uuid"],
                                         body_a_uuid=row["a"]["body_uuid"],
                                         body_b_uuid=row["b"]["body_uuid"],
                                         rest_length_m=row["rest_length_m"])
                                    for row in output]
    fixed._cross_links_payload = payload
    fixed._blender_bone_bindings = updated_receipt
    return updated_receipt


def write_sidecar(fixed, path):
    payload = getattr(fixed, "_cross_links_payload", None)
    if payload is None:
        return False
    if payload["scene_uuid"] != fixed._scene_uuid:
        fail("scene_uuid", "cross-links belong to a different scene")
    target = Path(path) / SIDECAR
    target.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False), encoding="utf-8")
    return True
