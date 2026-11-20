"""Blender authoring adapter; physics stays in the existing PDRD APIs."""
from __future__ import annotations
from ._root_emitter_ import emit_roots

from copy import deepcopy
import os
import uuid

import numpy as np

from ._pdrd_native_frames_ import native_body_frames, _rotation, _matrix_quat

SCHEMA = "PPF_BLENDER_BONES_V1"
RECEIPT_KIND = "ppf.blender_bone_bindings"
RECEIPT_FILE = "blender_bone_bindings.cbor"
AXES = ("twist_x", "swing_y", "swing_z")
MODEL = "parent_local_swing_twist_x_v1"
S = np.array([[1., 0., 0., 0.], [0., 0., 1., 0.],
              [0., -1., 0., 0.], [0., 0., 0., 1.]])


def fail(field, detail=""):
    raise ValueError(f"BLENDER_BONE_INVALID: field={field} {detail}")


def canonical_uuid(value, field):
    try:
        result = str(uuid.UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        fail(field, "expected canonical UUID")
    if result != value or result == "00000000-0000-0000-0000-000000000000":
        fail(field, "expected canonical non-world UUID")
    return result


def matrix(value, field):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError):
        fail(field, "expected rigid 4x4 matrix")
    if result.shape != (4, 4) or not np.isfinite(result).all():
        fail(field, "expected finite 4x4 matrix")
    if (not np.allclose(result[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or abs(np.linalg.det(result[:3, :3]) - 1.) > 1e-6):
        fail(field, "scale/shear/reflection unsupported; apply object transforms")
    # mathutils serializes f32 rotation coefficients; remove only accepted
    # authoring roundoff before taking inverse rigid frames.
    result = result.copy()
    U, _, Vt = np.linalg.svd(result[:3, :3])
    result[:3, :3] = U @ Vt
    return result


def positive_similarity_matrix(value, field):
    """Validate raw armature metadata without changing its authored scale.

    Bone, proxy, joint and native frames continue through the rigid validator.
    """
    try:
        result = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError):
        fail(field, 'expected positive uniform 4x4 transform')
    if result.shape != (4, 4) or not np.isfinite(result).all():
        fail(field, 'expected finite positive uniform transform')
    linear = result[:3, :3]
    scales = np.linalg.norm(linear, axis=0)
    scale = float(np.mean(scales))
    determinant = np.linalg.det(linear)
    if (not np.allclose(result[3], [0, 0, 0, 1], atol=1.e-8, rtol=0)
            or not np.isfinite(scale) or scale <= 1.e-8
            or not np.isfinite(determinant) or determinant <= 0
            or np.any(scales <= 0)
            or np.max(np.abs(scales / scale - 1.)) > 1.e-5
            or not np.allclose((linear / scales).T @ (linear / scales), np.eye(3), atol=1.e-5, rtol=0)):
        fail(field, 'only static positive uniform scale is supported; remove nonuniform scale, shear or reflection')
    return result


def frame_matrix(frame):
    result = np.eye(4)
    result[:3, :3] = _rotation(frame["world_quaternion_wxyz"])
    result[:3, 3] = frame["world_translation_m"]
    return result


def local_frame(body, joint):
    result = np.linalg.inv(body) @ joint
    return {"translation_m": result[:3, 3].tolist(),
            "quaternion_wxyz": _matrix_quat(result[:3, :3]).tolist()}


def _friction_nonnegative(value, field):
    if isinstance(value, dict):
        for key, entry in value.items():
            _friction_nonnegative(entry, f"{field}.{key}")
    elif isinstance(value, (list, tuple)):
        for i, entry in enumerate(value):
            _friction_nonnegative(entry, f"{field}.{i}")
    else:
        try:
            if isinstance(value, (bool, np.bool_)):
                raise ValueError
            coefficient = float(value)
        except (TypeError, ValueError, OverflowError):
            fail(field, "friction must be a finite nonnegative coefficient")
        if not np.isfinite(coefficient) or coefficient < 0.:
            fail(field, "friction must be a finite nonnegative coefficient")


def validate_environment(data):
    if float(data["scene"].get("world-scaling", 1.)) != 1.:
        fail("world-scaling", "native joint scenes support exactly 1")
    for group in data["group"]:
        # The decoder zips object names/UUIDs; an empty authoring group does
        # not produce material inputs and cannot constrain a native scene.
        if not group[1] or not group[2]:
            continue
        for key, value in group[0].items():
            if "friction" in key:
                _friction_nonnegative(value, key)
            if key == "param-anim":
                for animated, series in value.items():
                    if "friction" in animated:
                        _friction_nonnegative(series, animated)
    colliders = data.get("invisible_colliders", {})
    for kind in ("walls", "spheres"):
        for item in colliders.get(kind, []):
            _friction_nonnegative(item["friction"], f"invisible_colliders.{kind}.friction")


def install(fixed, document, param_data):
    from ._soft_limit_auto_ import prepare_author
    from ._bone_connected_collision_ import cross_link_pairs
    pmx_joints = None
    if isinstance(document, dict) and 'pmx_joints_v1' in document:
        from ._pmx_joints_ import validate_author
        document = deepcopy(document)
        pmx_joints = document.pop('pmx_joints_v1')
        validate_author(pmx_joints, document.get('bindings', []))
    collision_pairs = cross_link_pairs(document)
    fixed._cross_link_collision_uuid_pairs = collision_pairs
    if isinstance(document, dict) and 'cross_link_collision_pairs_v1' in document:
        document = deepcopy(document)
        del document['cross_link_collision_pairs_v1']
    cross_links = None
    if isinstance(document, dict) and 'cross_links_v1' in document:
        from ._cross_links_ import validate_author
        cross_links = document['cross_links_v1']
        validate_author(cross_links, document.get('bindings', []))
        document = deepcopy(document)
        del document['cross_links_v1']
    document,intent=prepare_author(document)
    if isinstance(document,dict) and 'bone_mass_intent_v1' in document:
        from ._bone_mass_authority_ import transaction
        result=transaction(fixed,document,param_data,_install)
    else:
        result=_install(fixed,document,param_data)
    if cross_links is not None:
        from ._cross_links_ import install as install_cross_links
        result = install_cross_links(fixed, cross_links, document, result)
    if pmx_joints is not None:
        from ._pmx_joints_ import install as install_pmx_joints
        install_pmx_joints(fixed, pmx_joints, document['bindings'], result)
    if collision_pairs:
        from ._bone_connected_collision_ import CROSS_FEATURE
        result['required_features'] = sorted(set(result.get('required_features', []) + [CROSS_FEATURE]))
        fixed._blender_bone_bindings = result
    wires=list((getattr(fixed,'_pdrd_ball_anchor_payload',None) or {}).get('joints',[]))
    wires += [item['wire'] for item in (getattr(fixed,'_fem_bone_joint_payload',None) or {}).get('joints',[])]
    if any('swing_cone' in wire.get('angular_soft_limits_v1',{}) for wire in wires):
        result['required_features']=sorted(set(result.get('required_features',[])+['pdrd_angular_swing_cone_v1']))
        fixed._blender_bone_bindings=result
    fixed._soft_limit_auto_intent=intent
    return result


def _install(fixed, document, param_data):
    """Bind UUIDs after Scene.build, then call the canonical joint/root setters."""
    fem_objects = {s['object_uuid'] for s in fixed._statistics_objects if s['dynamics_type'] == 'SOLID'}
    if (isinstance(document, dict) and isinstance(document.get('bindings'), list)
            and any(isinstance(r, dict) and r.get('body_uuid') in fem_objects for r in document['bindings'])):
        from ._fem_bone_joints_ import install_mixed
        return install_mixed(fixed, document, param_data)
    from ._scene_ import _validate_r1_payload
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        fail("schema")
    if set(document) - {"collision_filter_v1", "body_controls_v1", "body_controls_author_intent_v1", "bone_mass_intent_v1"} != {"schema", "armatures", "bindings", "start_frame", "fps", "world_scaling"}:
        fail("document", "unknown or missing requested field")
    body_controls = document.get("body_controls_v1", False)
    if "body_controls_v1" in document and body_controls is not True:
        fail("body_controls_v1", "explicit capability request must be true")
    if not body_controls and "body_controls_author_intent_v1" in document:
        fail("body_controls_author_intent_v1", "intent binding requires explicit body controls")
    validate_environment(param_data)
    _verify_installed_collision_filters(fixed, document)
    if (type(document["start_frame"]) is not int or document["fps"] != param_data["scene"]["fps"]
            or document["world_scaling"] != 1.):
        fail("time/units")
    armatures = {}
    for arm in document["armatures"]:
        if not isinstance(arm, dict) or set(arm) != {"armature_uuid", "object_name", "matrix_world_blender"}:
            fail("armature", "unknown or missing requested field")
        aid = canonical_uuid(arm["armature_uuid"], "armature_uuid")
        if aid in armatures:
            fail("armature_uuid", "duplicate")
        positive_similarity_matrix(arm["matrix_world_blender"], "armature.matrix_world_blender")
        armatures[aid] = arm
    native = native_body_frames(fixed)
    records = document["bindings"]
    if not isinstance(records, list) or not records:
        fail("bindings", "expected nonempty list")
    seen_bone, seen_body, by_bone = set(), set(), {}
    allowed = {"armature_uuid", "bone_uuid", "bone_name", "body_uuid", "proxy_object_uuid",
               "parent_bone_uuid", "parent_body_uuid", "joint_uuid", "enabled", "physics",
               "root_boundary", "root_mode", "proxy_object_name", "group_uuid", "shape", "density", "use_auto_frames",
               "bone_world_matrix_blender", "proxy_world_matrix_blender", "joint_world_matrix_blender",
               "body_to_bone_rest_blender", "parent_local_frame", "child_local_frame", "root_motion", "root_motion_source", "author_uniform_scale"}
    for row in records:
        if isinstance(row, dict) and "automatic_parent_anchor_v1" in row:
            fail("automatic_parent_anchor_v1", "automatic anchors must be built as real PDRD endpoints before installation")
        if not isinstance(row, dict) or set(row) - allowed:
            fail("binding", f"unknown requested fields {set(row) - allowed if isinstance(row, dict) else ''}")
        required = {"armature_uuid", "bone_uuid", "bone_name", "body_uuid", "parent_bone_uuid",
                    "parent_body_uuid", "joint_uuid", "enabled", "physics", "root_boundary", "root_mode",
                    "use_auto_frames", "bone_world_matrix_blender", "proxy_world_matrix_blender", "joint_world_matrix_blender"}
        if not required.issubset(row):
            fail("binding", f"missing fields {sorted(required-set(row))}")
        bid = canonical_uuid(row["bone_uuid"], "bone_uuid")
        uid = canonical_uuid(row["body_uuid"], "body_uuid")
        if bid in seen_bone or uid in seen_body:
            fail("binding", "duplicate bone/body UUID")
        if row["armature_uuid"] not in armatures or uid not in native:
            fail("body_uuid", f"missing armature or actual PDRD runtime: {uid}")
        if row.get("proxy_object_uuid", uid) != uid:
            fail("proxy_object_uuid", "must equal original proxy object UUID")
        if row.get("enabled", True) is not True:
            fail("enabled", "disabled bones must be omitted from transport")
        if row.get("root_mode") not in ("FREE", "KINEMATIC", "STATIC"):
            fail("root_mode", "unsupported root mode")
        if type(row.get("use_auto_frames")) is not bool:
            fail("use_auto_frames", "expected bool")
        boundary = row.get("root_boundary")
        if (not isinstance(boundary, dict) or set(boundary) != {"is_root", "nonphysical_parent_bone"}
                or type(boundary["is_root"]) is not bool
                or boundary["is_root"] != (not bool(row.get("parent_body_uuid")))):
            fail("root_boundary", "root/parent identity differs")
        seen_bone.add(bid); seen_body.add(uid); by_bone[bid] = row
    # Resolve graph by UUID, including same-armature ownership, and detect cycles.
    ordered, pending = [], dict(by_bone)
    while pending:
        progressed = False
        for bid, row in list(pending.items()):
            parent = row.get("parent_bone_uuid")
            if parent:
                if parent not in by_bone:
                    fail("parent_bone_uuid", "missing enabled parent")
                prow = by_bone[parent]
                if prow["body_uuid"] != row.get("parent_body_uuid") or prow["armature_uuid"] != row["armature_uuid"]:
                    fail("parent_body_uuid", "parent ownership differs")
                if parent in pending:
                    continue
            elif row.get("parent_body_uuid"):
                fail("parent_body_uuid", "root has unexpected parent")
            ordered.append(row); del pending[bid]; progressed = True
        if not progressed:
            fail("parent_bone_uuid", "cycle")
    joints, receipt_rows, roots = [], [], []
    for row in ordered:
        uid = row["body_uuid"]
        T = frame_matrix(native[uid])
        B = S @ matrix(row["bone_world_matrix_blender"], "bone_world_matrix_blender")
        physics = deepcopy(row["physics"])
        allowed_physics = {"density_kg_per_m3", "anchor_stiffness_N_per_m", "angular_control_v1", "angular_soft_limits_v1", "angular_drives_v1", "angular_hard_hinge_v1", "joint_d6_hard_v1", "collide_connected"}
        if not isinstance(physics, dict):
            fail("physics", "expected settings map")
        if set(physics) - allowed_physics:
            fail("physics", f"unsupported requested settings {set(physics)-allowed_physics}")
        if not {"density_kg_per_m3", "anchor_stiffness_N_per_m", "angular_control_v1", "collide_connected"}.issubset(physics):
            fail("physics", "missing required setting")
        density = physics.pop("density_kg_per_m3")
        source_groups = [g for g in param_data["group"] if len(g) > 2 and uid in g[2]]
        if (len(source_groups) != 1 or source_groups[0][0].get("model") != "pdrd"
                or float(source_groups[0][0].get("density", 0)) != float(np.float32(density))):
            fail("density_kg_per_m3", "author settings differ from original PDRD group")
        physics["stiffness_N_per_m"] = physics.pop("anchor_stiffness_N_per_m")
        parent = row.get("parent_body_uuid")
        if not parent and any(key in physics for key in (
                "angular_drives_v1", "angular_hard_hinge_v1", "joint_d6_hard_v1")):
            fail("physics", "joint extension requires a physical parent joint")
        if parent:
            jid = canonical_uuid(row["joint_uuid"], "joint_uuid")
            if row.get("use_auto_frames", True):
                J = S @ matrix(row["joint_world_matrix_blender"], "joint_world_matrix_blender")
                pf, cf = local_frame(frame_matrix(native[parent]), J), local_frame(T, J)
                # Both endpoint frames represent the same authored world J.
                # Its initial relative rotation is exactly identity; multiplying
                # the independently rounded frames can manufacture sub-f32
                # quaternion components that the native transport must reject.
                rest_relative = [1., 0., 0., 0.]
            else:
                # Manual frames are in the authored proxy rigid basis. Rebind to
                # rest-centered native frames using the existing input adapter.
                from ._pdrd_native_frames_ import rebind_joint_payload
                authored = {}
                for endpoint in (parent, uid):
                    source = next(r for r in ordered if r["body_uuid"] == endpoint)
                    P = S @ matrix(source["proxy_world_matrix_blender"], "proxy_world_matrix_blender")
                    authored[endpoint] = {"world_translation_m": P[:3, 3].tolist(), "world_quaternion_wxyz": _matrix_quat(P[:3, :3]).tolist()}
                raw = {"joints": [{"parent_body_uuid": parent, "child_body_uuid": uid,
                                  "parent_local_frame": row["parent_local_frame"], "child_local_frame": row["child_local_frame"]}]}
                rebound = rebind_joint_payload(raw, authored, native, preserve_world_joint_orientations=True)["joints"][0]
                pf, cf = rebound["parent_local_frame"], rebound["child_local_frame"]
                rest_relative = _matrix_quat(
                    (frame_matrix(native[parent])[:3, :3] @ _rotation(pf["quaternion_wxyz"])).T
                    @ T[:3, :3] @ _rotation(cf["quaternion_wxyz"])).tolist()
            joints.append({"joint_uuid": jid, "parent_body_uuid": parent,
                           "child_body_uuid": uid, "parent_local_frame": pf,
                           "child_local_frame": cf, "enabled": True,
                           "rest_relative_quaternion_wxyz": rest_relative, **physics})
        # A controller belongs to a physical body, including a body with an
        # incoming joint. Keep that joint and let the canonical native-frame
        # installer validate the controller's actual supported domain.
        mode = row["root_mode"]
        if mode != "FREE":
            root = deepcopy(row.get("root_motion"))
            if (not isinstance(root, dict) or root.get("body_uuid") != uid
                    or root.get("mode") != mode.lower()):
                fail("root_motion", "missing or inconsistent sampled authoring")
            roots.append(root)
        elif "root_motion" in row:
            root = deepcopy(row["root_motion"])
            if (not body_controls or not isinstance(root, dict)
                    or root.get("body_uuid") != uid or root.get("mode") != "dynamic"):
                fail("root_motion", "FREE body requires explicit dynamic body control")
            roots.append(root)
        author_scale = row.get('author_uniform_scale', 1.)
        if type(author_scale) not in (int, float) or not np.isfinite(author_scale) or author_scale <= 1.e-8:
            fail('author_uniform_scale', 'expected finite positive static scale')
        receipt_rows.append({"armature_uuid": row["armature_uuid"],
                             "armature_name": armatures[row["armature_uuid"]]["object_name"],
                             "bone_uuid": row["bone_uuid"], "bone_name": row["bone_name"],
                             "body_uuid": uid, "proxy_object_uuid": uid,
                             "runtime_body_index": native[uid]["runtime_body_index"],
                             "parent_bone_uuid": row.get("parent_bone_uuid"),
                             "bone_initial_solver": B.tolist(),
                             "body_to_bone_initial_solver": (np.linalg.inv(T) @ B).tolist()})
        if author_scale != 1.:
            receipt_rows[-1]['author_uniform_scale'] = author_scale
    payload = _validate_r1_payload({"joints": joints})
    if body_controls and not roots:
        fail("body_controls_v1", "capability request has no controlled bodies")
    if payload.get("version") == 7 and roots:
        if body_controls:
            fail("body_controls_v1", "new body control and hard D6 union requires separate native qualification")
        # The released hard/root union is exactly one D6 edge and one V1
        # source controller. Its setter checks this domain against real
        # native frames; do not silently promote it to a V2 root batch.
        for root in roots:
            # The bone encoder uses V2 wire-key names. V1 authoring requires
            # the original object-TRS vocabulary; translate only that exact
            # key map so unknown author fields still reach a strict refusal.
            root["keys"] = [
                {"time_s": key["time_s"], "translation_m": key["center_world_m"],
                 "quaternion_wxyz": key["rotation_world_wxyz"]}
                if isinstance(key, dict) and set(key) == {
                    "time_s", "center_world_m", "rotation_world_wxyz"} else key
                for key in root.get("keys", [])]
        fixed.set_pdrd_joint_d6_hard_v7(payload, root_motion={"version": 1, "bodies": roots})
    else:
        fixed.set_pdrd_ball_anchor_payload(payload)
        if roots:
            if body_controls:
                emit_roots(fixed, roots, True)
                from ._body_control_receipt_ import record_author
                record_author(fixed, document)
            else:
                emit_roots(fixed, roots)
    receipt = {"schema": SCHEMA, "collision_filter_v1": deepcopy(document.get("collision_filter_v1")), "start_frame": document["start_frame"],
               "fps": float(document["fps"]), "world_scaling": 1., "bindings": receipt_rows}
    fixed._blender_bone_bindings = receipt
    return receipt


def write_receipt(fixed_session, receipt):
    """Write display receipt after the original Session has sealed its inputs."""
    import cbor2
    from ._cbor_bridge_ import dumps_envelope_with_version
    path = fixed_session.info.path
    from ._bone_connected_collision_ import CROSS_FEATURE, verify_new_seal, verify_cross_seal, require_selected_capability
    if verify_cross_seal(fixed_session.fixed_scene, path) and CROSS_FEATURE not in receipt.get('required_features', []):
        fail('required_features', 'cross-link collision receipt is missing its feature')
    if verify_new_seal(fixed_session.fixed_scene, path, receipt):
        require_selected_capability(contract=fixed_session.fixed_scene._build_contract)
    with open(os.path.join(path, "bin", "pdrd_scene_context.cbor"), "rb") as file:
        context = cbor2.loads(file.read())["payload"]
    if context.get("complete") is not True:
        fail("context.complete", "Session final seal missing")
    result = deepcopy(receipt)
    result.update(scene_uuid=context["scene_uuid"], scene_inputs_sha256=context["scene_inputs_sha256"])
    if 'fem_material_joints_v1' in result.get('required_features', []):
        import hashlib
        with open(os.path.join(path, 'bin/fem_bone_joints.json'), 'rb') as fem_file:
            result['fem_payload_sha256'] = hashlib.sha256(fem_file.read()).hexdigest()
    with open(os.path.join(path, RECEIPT_FILE), "wb") as file:
        file.write(dumps_envelope_with_version(1, RECEIPT_KIND, result))
    from ._body_control_receipt_ import write_receipt as write_body_control_receipt
    write_body_control_receipt(fixed_session)
    return result


def prepare_collision_filters(scene, document, param_data):
    """Install immutable object masks before Scene.build's initial checks."""
    from ._bone_connected_collision_ import authored_pairs
    scene._connected_no_collision_uuid_pairs = authored_pairs(document)
    config = document.get("collision_filter_v1")
    if config is not None:
        from ._collision_filter_ import prepare
        prepare(scene, config, param_data)


def _verify_installed_collision_filters(fixed, document):
    from ._bone_connected_collision_ import canonical_pairs, verify_build_pairs
    if hasattr(fixed, '_connected_no_collision_uuid_pairs'):
        pairs = sorted(set(canonical_pairs(document.get('bindings', []), authored=True)) |
                       set(getattr(fixed, '_cross_link_collision_uuid_pairs', [])))
        verify_build_pairs(fixed, pairs)
    config = document.get("collision_filter_v1")
    if config is not None:
        from ._collision_filter_ import verify
        verify(fixed, config)
