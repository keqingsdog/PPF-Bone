"""Build real hidden PDRD endpoints for nonphysical Blender parents.

Only author input is extended. Ordinary joint/root installers remain the
authority for fitted body frames and every returned user bone pose.
"""
from copy import deepcopy
import uuid

import numpy as np

from . import _blender_bones_ as bb

ANCHOR_EDGE_M = 0.02
ANCHOR_DENSITY = 1000.0
META_KEYS = {
    'version', 'anchor_body_uuid', 'parent_bone_uuid', 'parent_bone_name',
    'parent_world_matrix_blender', 'anchor_proxy_world_matrix_blender',
    'root_motion',
}


def _anchor(row):
    meta = row['automatic_parent_anchor_v1']
    if (not isinstance(meta, dict) or set(meta) != META_KEYS
            or type(meta['version']) is not int or meta['version'] != 1):
        bb.fail('automatic_parent_anchor_v1', 'invalid metadata')
    for key in ('anchor_body_uuid', 'parent_bone_uuid'):
        bb.canonical_uuid(meta[key], 'automatic_parent_anchor_v1.' + key)
    if (not isinstance(meta['parent_bone_name'], str) or not meta['parent_bone_name']
            or row.get('parent_body_uuid') is not None
            or row.get('parent_bone_uuid') is not None
            or row.get('root_mode') != 'FREE'
            or row.get('root_motion') is not None
            or row.get('root_boundary') != {
                'is_root': True, 'nonphysical_parent_bone': meta['parent_bone_name']}):
        bb.fail('automatic_parent_anchor_v1', 'parent/root intent differs')
    parent = bb.matrix(meta['parent_world_matrix_blender'], 'automatic parent pose')
    proxy = bb.matrix(meta['anchor_proxy_world_matrix_blender'], 'automatic parent proxy')
    if not np.allclose(parent, proxy, atol=1e-8, rtol=0):
        bb.fail('automatic_parent_anchor_v1', 'anchor frame differs from parent pose')
    motion = deepcopy(meta['root_motion'])
    if not isinstance(motion, dict):
        bb.fail('automatic_parent_anchor_v1.root_motion')
    if motion.get('body_uuid', meta['anchor_body_uuid']) != meta['anchor_body_uuid']:
        bb.fail('automatic_parent_anchor_v1.root_motion', 'body identity differs')
    motion['body_uuid'] = meta['anchor_body_uuid']
    if motion.get('mode') != 'kinematic':
        bb.fail('automatic_parent_anchor_v1.root_motion', 'parent must be kinematic')
    from ._pdrd_root_motion_v2_ import validate_authored_v2
    from ._scene_ import _r1_fail
    validate_authored_v2({'roots': [motion]}, _r1_fail)
    initial = motion['initial_frame']
    expected = bb.S @ parent
    if (not np.allclose(initial['translation_m'], expected[:3, 3], atol=1e-8, rtol=0)
            or not np.allclose(bb._rotation(initial['quaternion_wxyz']), expected[:3, :3],
                               atol=1e-6, rtol=0)):
        bb.fail('automatic_parent_anchor_v1.root_motion', 'initial frame differs from parent')
    return meta, parent, motion


def _cube_local_vertices():
    local = np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                      [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]], dtype=float)
    return local * (ANCHOR_EDGE_M / 2)


def _wall_violations(scene, vertices):
    """Use the same static-wall predicate as normal scene validation."""
    walls = list(scene.wall_list)
    if not walls:
        return []
    from ._invisible_collider_ import check_wall_violations
    return check_wall_violations(np.ascontiguousarray(vertices, dtype=np.float64), walls)


def _static_wall_records(scene):
    """Return supported wall geometry; None means no migration is allowed."""
    if scene.sphere_list:
        return None
    records = []
    for wall in scene.wall_list:
        if not wall.is_static_collider():
            return None
        entry = wall.get_entry()
        params = wall.param.list()
        if len(entry) != 1 or 'contact-gap' not in params:
            return None
        gap = float(params['contact-gap'])
        if not np.isfinite(gap) or gap < 0.0:
            return None
        normal = np.asarray(wall.normal, dtype=np.float64)
        if normal.shape != (3,) or not np.isfinite(normal).all():
            return None
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm <= 0.0:
            return None
        normal /= norm
        position = np.asarray(entry[0][0], dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            return None
        records.append((position, normal, gap))
    return records


def _outside_wall_activation_band(scene, vertices):
    """Require clearance beyond contact-gap plus float32 plane arithmetic.

    Thickness is a negative-side penetration cutoff, not an activation distance.
    Existing native wall/CCD checks remain authoritative after this guard.
    """
    walls = _static_wall_records(scene)
    if walls is None:
        return False
    points = np.asarray(vertices, dtype=np.float64)
    for position, normal, gap in walls:
        signed = (points - position) @ normal
        scale = max(1.0, float(np.max(np.abs(points))),
                    float(np.max(np.abs(position))), abs(gap))
        rounding_margin = 16.0 * np.finfo(np.float32).eps * scale
        if not np.isfinite(signed).all() or np.any(signed <= gap + rounding_margin):
            return False
    return True


def _swept_carrier_clearance(scene, keys, local, offset):
    """Prove every carrier corner clears every supported static wall segment.

    The v3 root setter sends unchanged author keys plus one native reference D.
    Here local points are in the authored frame, so the desired world trajectory
    is P(t)*(offset+corner); D must not be applied a second time.
    """
    from ._root_motion_wall_sweep_ import (
        minimum_signed_plane_distance, float32_clearance_margin)
    walls = _static_wall_records(scene)
    if walls is None or not isinstance(keys, list) or len(keys) < 2:
        return False
    points = np.asarray(local, dtype=np.float64) + np.asarray(offset, dtype=np.float64)
    for first, last in zip(keys, keys[1:]):
        t0, t1 = first['center_world_m'], last['center_world_m']
        q0, q1 = first['rotation_world_wxyz'], last['rotation_world_wxyz']
        for position, normal, gap in walls:
            margin = float32_clearance_margin(t0, t1, points, position, gap)
            for point in points:
                bound = minimum_signed_plane_distance(t0, q0, t1, q1, point, position, normal)
                if bound['lower_bound_m'] <= gap + margin:
                    return False
    return True


def _candidate_carrier_offset(scene, parent, motion, child_rows, local):
    """Return a reference offset only for a failing, supported KIN carrier.

    None means preserve the original carrier frame and leave normal Build
    validation to accept or reject it. Requires the v3 sample-then-compose root
    path; no user geometry, author key or wall is changed.
    """
    S_parent = bb.S @ parent
    original_world = local @ S_parent[:3, :3].T + S_parent[:3, 3]
    if not _wall_violations(scene, original_world):
        return np.eye(4)
    if (motion.get('mode') != 'kinematic'
            or motion.get('authoring') != 'object_world_trs'
            or motion.get('control_events')):
        print('PPF internal parent anchor retained original frame: unsupported root-motion mode/authoring/events')
        return None
    keys = motion.get('keys')
    if not isinstance(keys, list) or not keys:
        print('PPF internal parent anchor retained original frame: no authored root-motion keys')
        return None
    if not child_rows:
        print('PPF internal parent anchor retained original frame: no child joint anchors')
        return None

    child_positions = []
    for row in child_rows:
        joint = bb.matrix(row['joint_world_matrix_blender'], 'child joint world anchor')
        child_positions.append(joint[:3, 3])
    center = np.mean(np.asarray(child_positions, dtype=np.float64), axis=0)
    target = np.eye(4)
    target[:3, 3] = center
    offset_local = np.linalg.inv(parent) @ target
    D = np.eye(4)
    D[:3, 3] = offset_local[:3, 3]
    moved = bb.S @ parent @ D
    initial_vertices = local @ moved[:3, :3].T + moved[:3, 3]
    if (_wall_violations(scene, initial_vertices)
            or not _outside_wall_activation_band(scene, initial_vertices)
            or not _swept_carrier_clearance(scene, keys, local, D[:3, 3])):
        print('PPF internal parent anchor retained original frame: shifted carrier fails full-interval wall-domain or contact-clearance check')
        return None
    return D


def _prepare_static_bridges(rows):
    """Reparent private joint endpoints, retaining each authored world frame."""
    by_body = {row['body_uuid']: row for row in rows}
    for row in rows:
        meta = row.get('automatic_parent_anchor_v1')
        if not isinstance(meta, dict) or 'physical_ancestor_bridge_v1' not in meta:
            continue
        bridge = meta['physical_ancestor_bridge_v1']
        if not isinstance(bridge, dict) or bridge.get('version') != 1 or not bridge.get('chain'):
            bb.fail('physical_ancestor_bridge_v1', 'invalid static bridge')
        ancestor = by_body.get(bridge.get('ancestor_body_uuid'))
        if (ancestor is None or ancestor['bone_uuid'] != bridge.get('ancestor_bone_uuid')
                or ancestor['armature_uuid'] != row['armature_uuid'] or row['root_mode'] != 'FREE'
                or row.get('parent_body_uuid') is not None):
            bb.fail('physical_ancestor_bridge_v1', 'physical ancestor differs')
        parent = bb.matrix(meta['parent_world_matrix_blender'], 'bridge parent frame')
        relation = bb.matrix(bridge['ancestor_to_parent_blender'], 'bridge relative frame')
        initial = bb.matrix(ancestor['bone_world_matrix_blender'], 'bridge ancestor frame')
        if not np.allclose(initial @ relation, parent, atol=2.e-6, rtol=0.):
            bb.fail('physical_ancestor_bridge_v1', 'initial parent relation differs')
        if not row['use_auto_frames']:
            # Manual parent J is relative to the former B endpoint, not A.
            old = row['parent_local_frame']
            local = np.eye(4)
            local[:3, :3] = bb._rotation(old['quaternion_wxyz'])
            local[:3, 3] = old['translation_m']
            world_joint = (bb.S @ parent) @ local
            row['parent_local_frame'] = bb.local_frame(
                bb.S @ bb.matrix(ancestor['proxy_world_matrix_blender'], 'bridge ancestor proxy'), world_joint)
        row.pop('automatic_parent_anchor_v1')
        row['parent_body_uuid'] = ancestor['body_uuid']
        row['parent_bone_uuid'] = ancestor['bone_uuid']
        row['root_boundary'] = {'is_root': False, 'nonphysical_parent_bone': None}
        # Child frame, angular drive/rest intent and automatic J0 stay unchanged.


def prepare(scene, param_data):
    from ._build_contract_ import contract_for, validate_requests
    contract = contract_for(scene)
    validate_requests(contract, param_data)
    if contract.producer_version == 2:
        from ._automatic_parent_bridge_v2_ import has_bridges, prepare as prepare_bridges
        if has_bridges(param_data):
            return prepare_bridges(scene, param_data, contract)
        from ._automatic_parent_anchor_legacy_ import prepare as legacy_prepare
        return legacy_prepare(scene, param_data)
    """Return a private expanded upload and add endpoints before Scene.build.

    Validate the whole batch before changing the scene. The decoded original
    upload and Blender objects/material groups are never modified.
    """
    document = param_data.get('blender_bones_v1')
    if document is None or not any('automatic_parent_anchor_v1' in r
                                    for r in document.get('bindings', [])):
        return param_data
    prepared = deepcopy(param_data)
    doc = prepared['blender_bones_v1']
    originals = doc['bindings']
    original_bodies = {r['body_uuid'] for r in originals}
    original_bones = {r['bone_uuid'] for r in originals}
    _prepare_static_bridges(originals)
    anchors = {}
    local = _cube_local_vertices()
    for row in originals:
        if 'automatic_parent_anchor_v1' not in row:
            continue
        meta, matrix, motion = _anchor(row)
        uid = meta['anchor_body_uuid']
        if uid in original_bodies or meta['parent_bone_uuid'] in original_bones:
            bb.fail('automatic_parent_anchor_v1', 'internal endpoint identity collides')
        candidate = (row['armature_uuid'], deepcopy(meta), matrix, motion, [row])
        if uid in anchors:
            prior = anchors[uid]
            if prior[0] != candidate[0] or prior[1] != candidate[1]:
                bb.fail('automatic_parent_anchor_v1', 'shared parent samples differ')
            prior[4].append(row)
        else:
            if uid in scene.object_dict or uid in scene.asset_manager.list():
                bb.fail('automatic_parent_anchor_v1', 'internal endpoint already exists')
            anchors[uid] = candidate

    offsets = {}
    for uid, (arm_uuid, meta, parent, motion, child_rows) in sorted(anchors.items()):
        D = _candidate_carrier_offset(scene, parent, motion, child_rows, local)
        offsets[uid] = D
        if D is not None:
            meta['anchor_proxy_world_matrix_blender'] = (parent @ D).tolist()

    internal_rows = []
    for uid, (arm_uuid, meta, parent, motion, child_rows) in sorted(anchors.items()):
        D = offsets[uid]
        carrier = parent @ D if D is not None else parent
        internal_rows.append({
            'armature_uuid': arm_uuid, 'bone_uuid': meta['parent_bone_uuid'],
            'bone_name': meta['parent_bone_name'], 'body_uuid': uid,
            'proxy_object_uuid': uid, 'proxy_object_name': '__PPF_parent_anchor_' + uid,
            'group_uuid': str(uuid.uuid5(uuid.UUID(uid), 'internal-material')),
            'parent_bone_uuid': None, 'parent_body_uuid': None,
            'joint_uuid': str(uuid.uuid5(uuid.UUID(uid), 'unused-root-joint')),
            'enabled': True, 'root_mode': 'KINEMATIC', 'root_motion': motion,
            'root_motion_source': 'POSE',
            'root_boundary': {'is_root': True, 'nonphysical_parent_bone': None},
            'bone_world_matrix_blender': parent.tolist(),
            'proxy_world_matrix_blender': carrier.tolist(),
            'body_to_bone_rest_blender': np.linalg.inv(D).tolist() if D is not None else np.eye(4).tolist(),
            'joint_world_matrix_blender': parent.tolist(), 'use_auto_frames': True,
            'physics': {
                'density_kg_per_m3': ANCHOR_DENSITY, 'anchor_stiffness_N_per_m': 100000.0,
                'angular_control_v1': {
                    'model': 'parent_local_swing_twist_x_v1',
                    'stiffness_Nm_per_rad': dict.fromkeys(bb.AXES, 0.0),
                    'damping_Nms_per_rad': dict.fromkeys(bb.AXES, 0.0)},
                'collide_connected': True},
        })
    for row in originals:
        meta = row.pop('automatic_parent_anchor_v1', None)
        if meta is None:
            continue
        row['parent_bone_uuid'] = meta['parent_bone_uuid']
        row['parent_body_uuid'] = meta['anchor_body_uuid']
        row['root_boundary'] = {'is_root': False, 'nonphysical_parent_bone': None}
        # This is an invisible kinematic endpoint, never a physical collider.
        # Mixed FEM's existing interface requires this flag true; its actual
        # contacts are excluded independently by the upstream object policy.
        row['physics']['collide_connected'] = True
    doc['bindings'] = internal_rows + originals
    if doc.get('body_controls_v1'):
        controls = doc['body_controls_author_intent_v1']['controls']
        controls.extend({'body_uuid': r['body_uuid'], 'mode': 'KINEMATIC',
                         'source': 'POSE', 'events': []} for r in internal_rows)
        controls.sort(key=lambda r: r['body_uuid'])

    # Eight vertices / twelve outward faces give a genuine nondegenerate
    # closed body; no collision geometry is exposed in Blender or PPF preview.
    faces = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7], [0,1,5],[0,5,4],
                      [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]], dtype=np.int64)
    group_names, group_uuids = [], []
    for uid, (_, meta, parent, _, _) in sorted(anchors.items()):
        D = offsets[uid]
        solver = bb.S @ parent @ (D if D is not None else np.eye(4))
        world = local @ solver[:3, :3].T + solver[:3, 3]
        scene.asset_manager.add.tri(uid, np.ascontiguousarray(world), faces)
        obj = scene.add(uid).as_pdrd().invisible()
        obj._statistics_uuid = uid
        obj._statistics_name = '__PPF_parent_anchor_' + meta['parent_bone_name']
        obj._statistics_type = 'PDRD'
        obj.group('__PPF_internal_parent_anchors')
        obj.param.set('density', ANCHOR_DENSITY)
        obj.param.set('friction', 0.0)
        for key in ('allow-self-intersection', 'allow-inter-object-intersection',
                    'allow-inter-group-intersection'):
            obj.param.set(key, 1.0)
        group_names.append(obj._statistics_name)
        group_uuids.append(uid)
    if group_names:
        prepared['group'].append([{'model': 'pdrd', 'density': ANCHOR_DENSITY,
                                   'friction': 0.0}, group_names, group_uuids])
    # Build's dense exclusions and the installed wire must describe the same
    # expanded physical edges, including static ancestor bridges.
    from ._bone_connected_collision_ import authored_pairs
    scene._connected_no_collision_uuid_pairs = authored_pairs(doc)
    return prepared


def visible_receipt(receipt, original_document):
    """Hide internal endpoints without changing any solved bone pose."""
    authored = {r['body_uuid']: r for r in original_document['bindings']}
    rows = []
    for row in receipt['bindings']:
        if row['body_uuid'] not in authored:
            continue
        row = deepcopy(row)
        row['parent_bone_uuid'] = authored[row['body_uuid']]['parent_bone_uuid']
        meta = authored[row['body_uuid']].get('automatic_parent_anchor_v1', {})
        if 'physical_ancestor_bridge_v1' in meta:
            row['nonphysical_parent_bridge_v1'] = deepcopy(meta['physical_ancestor_bridge_v1'])
        rows.append(row)
    if len(rows) != len(authored):
        bb.fail('automatic_parent_anchor_v1', 'user pose receipt lost a body')
    result = deepcopy(receipt)
    result['bindings'] = rows
    return result
