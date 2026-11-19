"""Validate native FEM material measurements; never fit or alter geometry."""
import hashlib
import math
import numpy as np

FEM = "FEM_TET_MATERIAL"
SYNC_FIELDS = ("scene_uuid", "scene_inputs_sha256", "vertex_frame_sha256",
               "vertex_frame_byte_length", "frame_id", "frame_time_seconds",
               "accepted_bracket_start_seconds", "accepted_solver_time_seconds",
               "accepted_steps", "source", "coordinate_system")


def binding_kind(row):
    kind = row.get("body_type", "PDRD")
    if kind not in ("PDRD", FEM):
        raise ValueError("bone pose: unsupported explicit physical body type")
    return kind


def validate_binding(row):
    from .rigid_pose import _rigid
    if binding_kind(row) != FEM:
        return
    if "runtime_body_index" in row:
        raise ValueError("FEM material binding cannot carry a PDRD index")
    if type(row["tet_index"]) is not int or row["tet_index"] < 0:
        raise ValueError("invalid FEM tet index")
    ids = row["vertex_indices"]
    if not isinstance(ids, list) or len(ids) != 4 or len(set(ids)) != 4 or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("invalid FEM actual vertex namespace")
    dm = np.asarray(row["dm_inv_column_major"], float)
    bary = np.asarray(row["center_bary"], float)
    rest = np.asarray(row["frame_rest_column_major"], float)
    if dm.shape != (9,) or not np.isfinite(dm).all() or np.linalg.det(dm.reshape(3,3,order="F")) == 0 or bary.shape != (4,) or not np.isfinite(bary).all() or abs(bary.sum()-1) > 1e-8 or rest.shape != (9,):
        raise ValueError("invalid FEM reference material frame")
    m = np.eye(4); m[:3,:3] = rest.reshape(3,3,order="F"); _rigid(m)
    from .fem_scale import compile_reference
    initial=_rigid(row['bone_initial_solver'])
    axes=initial[:3,:3].T.reshape(-1).tolist()
    compile_reference(row['fem_scale_reference'],expected_rest_axes=axes)
    if 'fem_display_rest_v1' in row:
        display = row['fem_display_rest_v1']
        if (not isinstance(display, dict) or set(display) != {'version', 'vertex_indices', 'positions_m'}
                or type(display['version']) is not int or display['version'] != 1):
            raise ValueError('invalid FEM display rest support')
        ids = display['vertex_indices']
        if (not isinstance(ids, list) or not ids or
                any(type(i) is not int or i < 0 for i in ids) or
                any(a >= b for a, b in zip(ids, ids[1:]))):
            raise ValueError('invalid FEM display rest vertex indices')
        positions = np.asarray(display['positions_m'], dtype=float)
        if positions.shape != (len(ids), 3) or not np.isfinite(positions).all():
            raise ValueError('invalid FEM display rest positions')
        from .fem_display_support import validate_rest
        validate_rest(row)


def validate_frame(pose, receipt, frame_id, vertices=None):
    from .rigid_pose import _hash, _uuid, _rigid
    event_context=receipt.get('fem_control_events_v2')
    version=2 if event_context is not None else 1
    authority='native_fem_control_segments_v2' if event_context is not None else 'not_reported'
    if not isinstance(pose, dict) or type(pose.get('version')) is not int or pose.get("version") != version or pose.get("kind") != "FEMMaterialFramePoseMeasurement" or pose.get("measurement") != "read_only_fem_material_polar_of_vertex_frame" or pose.get("coordinate_system") != "solver_y_up_meters" or pose.get("constrained_root_state_authority") != authority:
        raise ValueError("bone pose: unsupported FEM native measurement")
    if pose["scene_uuid"] != receipt["scene_uuid"] or pose["scene_inputs_sha256"] != receipt["scene_inputs_sha256"] or type(pose["frame_id"]) is not int or pose["frame_id"] != frame_id:
        raise ValueError("bone pose: FEM scene/frame mismatch")
    _hash(pose["fem_payload_sha256"]); _hash(pose["vertex_frame_sha256"])
    if pose["fem_payload_sha256"] != receipt["fem_payload_sha256"]:
        raise ValueError("bone pose: FEM payload generation mismatch")
    if type(pose["vertex_frame_byte_length"]) is not int or pose["vertex_frame_byte_length"] <= 0 or pose["vertex_frame_byte_length"] % 12:
        raise ValueError("bone pose: FEM vertex frame length")
    if vertices is not None and (len(vertices) != pose["vertex_frame_byte_length"] or hashlib.sha256(vertices).hexdigest() != pose["vertex_frame_sha256"] or not np.isfinite(np.frombuffer(vertices,dtype="<f4")).all()):
        raise ValueError("bone pose: FEM accepted vertex generation mismatch")
    t,a,b=(pose[k] for k in ("frame_time_seconds","accepted_bracket_start_seconds","accepted_solver_time_seconds"))
    if not all(type(v) in (int,float) and math.isfinite(v) for v in (t,a,b)) or not 0 <= a <= t <= b or abs(t-frame_id/receipt["fps"])>1e-8:
        raise ValueError("bone pose: FEM accepted frame time")
    steps=pose["accepted_steps"]; source=pose["source"]
    if type(steps) is not int or not ((source=="InitialFrame" and frame_id==0 and steps==0 and t==a==b==0) or (source=="AcceptedEndpoint" and frame_id>0 and steps>0 and a<b and t==b) or (source=="InterpolatedAcceptedBracket" and frame_id>0 and steps>0 and a<=t<b)):
        raise ValueError("bone pose: unaccepted FEM source")
    if event_context is not None:
        from .fem_control_segments import validate as validate_segments
        validate_segments(pose,event_context)
    elif any(k in pose for k in ('control_program_sha256','control_segments')):
        raise ValueError('bone pose: unadvertised FEM event output')
    expected={r["body_uuid"]:r for r in receipt["bindings"] if binding_kind(r)==FEM}
    bodies={}
    for p in pose["bodies"]:
        uid=_uuid(p["body_uuid"]); row=expected.get(uid)
        if uid in bodies or row is None or p.get("body_type")!=FEM or "runtime_body_index" in p or p["tet_index"]!=row["tet_index"] or p["vertex_indices"]!=row["vertex_indices"] or max(p["vertex_indices"])>=pose["vertex_frame_byte_length"]//12:
            raise ValueError("bone pose: FEM material differs from sealed binding")
        r=np.asarray(p["rotation_column_major"],float); center=np.asarray(p["centroid_m"],float)
        if r.shape!=(9,) or center.shape!=(3,):raise ValueError("bone pose: FEM matrix dimensions")
        m=np.eye(4);m[:3,:3]=r.reshape(3,3,order="F");m[:3,3]=center;_rigid(m)
        bodies[uid]=m
    if set(bodies)!=set(expected) or not bodies:raise ValueError("bone pose: missing FEM measurements")
    return bodies


def validate_combined(wrapper, receipt, frame_id, vertices=None):
    from .rigid_pose import _load, validate_frame as validate_rigid
    if wrapper.get('version')!=2 or wrapper.get('kind')!='PPF_DUAL_NATIVE_POSE_V2':raise ValueError('unsupported dual native pose cache')
    retained=wrapper.get('vertex_native_bytes')
    if not isinstance(retained,bytes):raise ValueError('missing original FEM accepted vertex bytes')
    if vertices is not None and vertices!=retained:raise ValueError('FEM cached vertex bytes differ from accepted frame')
    vertices=retained
    bodies={}; common=None
    need_rigid=any(binding_kind(r)=="PDRD" for r in receipt["bindings"])
    need_fem=any(binding_kind(r)==FEM for r in receipt["bindings"])
    for key,need,validator in (("rigid_native_bytes",need_rigid,validate_rigid),("fem_native_bytes",need_fem,validate_frame)):
        raw=wrapper.get(key)
        if need and not isinstance(raw,bytes):raise ValueError("missing required native physical body sidecar")
        if raw is None:continue
        pose=_load(raw)
        values=validator(pose,receipt,frame_id,vertices)
        if key == "rigid_native_bytes":
            required={r["body_uuid"] for r in receipt["bindings"] if binding_kind(r)=="PDRD"}
            values={uid:m for uid,m in values.items() if uid in required}
        current=tuple(pose[k] for k in SYNC_FIELDS)
        if common is not None and current!=common:raise ValueError("native rigid/FEM accepted frame mismatch")
        common=current
        if set(values)&set(bodies):raise ValueError("duplicate dual native body identity")
        bodies.update(values)
    if not bodies or set(bodies)!={r["body_uuid"] for r in receipt["bindings"]}:
        raise ValueError("dual native body coverage mismatch")
    if wrapper["frame_id"]!=frame_id:raise ValueError("dual cache frame mismatch")
    return bodies


def decode_native_frames(rigid_data, fem_data, vertices, frame_id, receipt):
    """Packet retains both original native sidecars for exact cached playback."""
    from .rigid_pose import decode_frame
    if fem_data is None and not any(binding_kind(r)==FEM for r in receipt["bindings"]):
        return decode_frame(rigid_data,vertices,frame_id,receipt)
    import cbor2
    wrapper={"version":2,"kind":"PPF_DUAL_NATIVE_POSE_V2","frame_id":frame_id,
             "rigid_native_bytes":rigid_data,"fem_native_bytes":fem_data,"vertex_native_bytes":vertices}
    validate_combined(wrapper,receipt,frame_id,vertices)
    return {"receipt":receipt,"pose":wrapper,"native_bytes":cbor2.dumps(wrapper)}
