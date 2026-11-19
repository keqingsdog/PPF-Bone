"""Bind exact authored neighbor exclusions to CPU Build and sealed JointWire.

No collision groups, topology edits or solver emulation. Native ownership proof
and full-member link installation remain authoritative.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

FEATURE = 'bone_connected_collision_v1'
CROSS_FEATURE = 'cross_link_collision_pairs_v1'
CROSS_FILE = 'bin/cross_link_collision_pairs_v1.json'
WORLD = '00000000-0000-0000-0000-000000000000'


def fail(detail, code='BONE_CONNECTED_COLLISION_MISMATCH'):
    raise ValueError(f'{code}: {detail}')


def canonical_pairs(rows, authored=False):
    pairs = set()
    for row in rows:
        parent = row.get('parent_body_uuid')
        child = row.get('body_uuid' if authored else 'child_body_uuid')
        physics = row.get('physics', {}) if authored else row
        if (row.get('enabled') is True and parent not in (None, WORLD)
                and physics.get('collide_connected') is False):
            from ._blender_bones_ import canonical_uuid
            canonical_uuid(parent, 'connected.parent_body_uuid')
            canonical_uuid(child, 'connected.child_body_uuid')
            if parent == child:
                fail('connected body cannot exclude itself')
            pairs.add(tuple(sorted((parent, child))))
    return sorted(pairs)


def cross_link_pairs(document):
    """Validate explicit authored cross-link exclusions before Build."""
    from ._blender_bones_ import canonical_uuid
    if not isinstance(document, dict):
        fail('cross-link pairs require a bone document')
    rows = document.get('cross_link_collision_pairs_v1')
    if rows is None:
        return []
    if not isinstance(rows, list) or not rows or len(rows) > 65536:
        fail('cross-link pairs must be a nonempty bounded list')
    bindings = document.get('bindings')
    if not isinstance(bindings, list):
        fail('cross-link pairs require bone bindings')
    bodies = {}
    for row in bindings:
        if not isinstance(row, dict) or row.get('enabled') is not True:
            continue
        body = canonical_uuid(row.get('body_uuid'), 'cross-link body_uuid')
        if body in bodies:
            fail('duplicate body UUID in cross-link bindings')
        bodies[body] = row.get('armature_uuid')
    pairs = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 2:
            fail('cross-link pair must contain two UUIDs')
        a = canonical_uuid(row[0], 'cross-link pair a')
        b = canonical_uuid(row[1], 'cross-link pair b')
        if a >= b or a not in bodies or b not in bodies or bodies[a] != bodies[b]:
            fail('cross-link pair must name two ordered enabled bodies in one armature')
        pairs.append((a, b))
    if pairs != sorted(set(pairs)):
        fail('cross-link pairs must be sorted and unique')
    return pairs


def authored_pairs(document):
    return sorted(set(canonical_pairs(document.get('bindings', []), authored=True))
                  | set(cross_link_pairs(document)))


def write_cross_sidecar(fixed, path):
    pairs = list(getattr(fixed, '_cross_link_collision_uuid_pairs', []))
    target = Path(path) / CROSS_FILE
    if not pairs:
        if target.exists():
            fail('unexpected stale cross-link collision sidecar')
        return False
    payload = {'version': 1, 'kind': 'ppf.cross_link_collision_pairs',
               'scene_uuid': fixed._scene_uuid,
               'pairs': [list(pair) for pair in pairs]}
    target.write_text(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                 allow_nan=False), encoding='utf8')
    return True


def verify_cross_seal(fixed, path):
    pairs = list(getattr(fixed, '_cross_link_collision_uuid_pairs', []))
    if not pairs:
        return False
    import cbor2
    root = Path(path)
    raw = (root / CROSS_FILE).read_bytes()
    body = json.loads(raw)
    expected = {'version': 1, 'kind': 'ppf.cross_link_collision_pairs',
                'scene_uuid': fixed._scene_uuid, 'pairs': [list(p) for p in pairs]}
    joint = cbor2.loads((root / 'bin/pdrd_ball_anchor.cbor').read_bytes())['payload']
    context = cbor2.loads((root / 'bin/pdrd_scene_context.cbor').read_bytes())['payload']
    entries = [i for i in context.get('inputs', []) if i.get('relative_path') == CROSS_FILE]
    if (body != expected or CROSS_FEATURE not in joint.get('required_features', [])
            or context.get('scene_uuid') != fixed._scene_uuid
            or context.get('complete') is not True
            or len(entries) != 1 or entries[0].get('byte_length') != len(raw)
            or entries[0].get('sha256') != hashlib.sha256(raw).hexdigest()):
        fail('sealed cross-link collision source does not match Build')
    verify_wire_pairs(fixed, getattr(fixed, '_fem_bone_joint_payload', None))
    return True


def pairs_digest(pairs):
    return hashlib.sha256(''.join(f'{a}\t{b}\n' for a, b in sorted(set(pairs))).encode('utf8')).hexdigest()


def required(fem_payload):
    if fem_payload is None:
        return False
    kinds = {body['body_uuid']: body['kind'] for body in fem_payload['bodies']}
    for item in fem_payload['joints']:
        row = item['wire']
        if (canonical_pairs([row]) and
                any(kinds.get(row[side + '_body_uuid']) == 'FEM_TET_MATERIAL'
                    for side in ('parent', 'child'))):
            return True
    return False


def verify_build_pairs(fixed, expected):
    """Compare authored UUIDs and the dense object pairs actually used by CPU Build."""
    stored = getattr(fixed, '_connected_no_collision_uuid_pairs', None)
    if stored is None:
        if expected:
            fail('missing CPU Build connected-pair provenance; rebuild from author input')
        return
    if list(stored) != expected:
        fail('author/CPU Build UUID pairs differ; rebuild after changing collide_connected')
    object_ids = {}
    owners = fixed._object_vert_index
    for name, vertices in fixed._map_by_name.items():
        ids = {int(owners[int(i)]) for i in vertices}
        if len(ids) != 1:
            fail('CPU object map is not a unique owner: ' + str(name))
        object_ids[name] = ids.pop()
    if any(a not in object_ids or b not in object_ids for a, b in expected):
        fail('connected body UUID is absent from the CPU object map')
    dense = sorted(set(tuple(sorted((object_ids[a], object_ids[b]))) for a, b in expected))
    if dense != list(fixed._excluded_object_pairs):
        fail('CPU dense exclusion table differs from authored UUID pairs')


def verify_wire_pairs(fixed, fem_payload):
    carrier = getattr(fixed, '_pdrd_ball_anchor_payload', None)
    rows = list(carrier['joints']) if carrier else []
    if fem_payload is not None:
        rows += [item['wire'] for item in fem_payload['joints']]
    expected = sorted(set(canonical_pairs(rows)) |
                      set(getattr(fixed, '_cross_link_collision_uuid_pairs', [])))
    # Keep manually constructed legacy setter exports byte-compatible. Actual
    # Scene.build always supplies provenance; the new FEM path requires it.
    if required(fem_payload) or hasattr(fixed, '_connected_no_collision_uuid_pairs'):
        verify_build_pairs(fixed, expected)
    return expected


def verify_new_seal(fixed, path, receipt):
    """Refuse a missing/tampered new feature before publishing a successful Build."""
    payload = getattr(fixed, '_fem_bone_joint_payload', None)
    if not required(payload):
        if FEATURE in receipt.get('required_features', []):
            fail('receipt requests an exclusion absent from current FEM wire')
        return False
    import cbor2
    path = Path(path)
    context = cbor2.loads((path / 'bin/pdrd_scene_context.cbor').read_bytes())['payload']
    joint = cbor2.loads((path / 'bin/pdrd_ball_anchor.cbor').read_bytes())['payload']
    raw = (path / 'bin/fem_bone_joints.json').read_bytes()
    fem = json.loads(raw)
    features = joint.get('required_features', [])
    if (not isinstance(features, list) or any(not isinstance(f, str) for f in features)
            or FEATURE not in features or FEATURE not in receipt.get('required_features', [])
            or features != sorted(set(features)) or not required(fem)):
        fail('sealed FEM exclusion requires ' + FEATURE, 'BONE_CONNECTED_COLLISION_FEATURE_REQUIRED')
    if (context.get('complete') is not True or context.get('scene_uuid') != fixed._scene_uuid
            or joint.get('scene_uuid') != context['scene_uuid']
            or joint.get('scene_inputs_sha256') != context['scene_inputs_sha256']
            or fem != payload
            or joint['joints'] != fixed._pdrd_ball_anchor_payload['joints']):
        fail('sealed author, scene or context changed')
    entries = [item for item in context['inputs'] if item['relative_path'] == 'bin/fem_bone_joints.json']
    if (len(entries) != 1 or entries[0]['byte_length'] != len(raw)
            or entries[0]['sha256'] != hashlib.sha256(raw).hexdigest()):
        fail('sealed FEM wire bytes differ from context inputs')
    verify_wire_pairs(fixed, fem)
    return True


def capability(exe, context=None):
    """Pure-host query, handled by native main before device/backend loading."""
    from . import _r14_product_adapter as r14
    exe = Path(exe).resolve()
    tree = Path(__file__).resolve().parent.parent
    environment = os.environ.copy()
    environment['PATH'] = os.pathsep.join([str(exe.parent), str(tree / 'bin'), environment.get('PATH', '')])
    child_cwd = tree
    product_context = context if context is not None else r14.solver_context(exe)
    if product_context is not None:
        selected_exe, child_cwd, environment = product_context
        exe = Path(selected_exe)
    try:
        result = subprocess.run([str(exe), '--bone-connected-collision-capability-json'],
                                cwd=child_cwd, env=environment, capture_output=True, timeout=30)
        if result.returncode != 0:
            fail('selected backend does not support ' + FEATURE + '; update the matching backend and Build again',
                 'BONE_CONNECTED_COLLISION_CAPABILITY_UNAVAILABLE')
        doc = json.loads(result.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        fail('selected backend capability unavailable for ' + FEATURE + ': ' + str(exc),
             'BONE_CONNECTED_COLLISION_CAPABILITY_UNAVAILABLE')
    expected = {'kind': 'ppf.bone_connected_collision.capability', 'version': 1,
                'required_feature': FEATURE, 'supported': True, 'semantic_version': 1}
    if (not isinstance(doc, dict) or set(doc) != set(expected) or doc != expected
            or type(doc['version']) is not int or type(doc['semantic_version']) is not int
            or doc['supported'] is not True):
        fail('selected backend returned an incompatible ' + FEATURE + ' capability',
             'BONE_CONNECTED_COLLISION_CAPABILITY_UNAVAILABLE')
    return doc


def require_selected_capability(exe=None, context=None, contract=None):
    if contract is not None:
        from ._build_contract_ import verify_build_contract
        verify_build_contract(contract)
        if FEATURE not in contract.capabilities:
            fail("frozen backend lacks " + FEATURE, "BONE_CONNECTED_COLLISION_CAPABILITY_UNAVAILABLE")
        return dict(contract.capability_snapshot)
    if exe is not None:
        return capability(exe, context=context)
    fail('a frozen BuildContract or explicit frozen EXE/context is required; Build again',
         'BONE_CONNECTED_COLLISION_CAPABILITY_UNAVAILABLE')
