"""Immutable body-control upload/seal checks; no Blender or network work."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import PurePosixPath
import hashlib, json, math, re, uuid
from .body_control_intent import validate_author_intent

CAPABILITY = 'pdrd_body_controls_v1'
RECEIPT_FILE = 'body_control_receipt.cbor'
CONTEXT_FILE = 'bin/pdrd_scene_context.cbor'
JOINT_FILE = 'bin/pdrd_ball_anchor.cbor'
SIDECAR_FILE = 'bin/pdrd_root_motion_v1.cbor'

class BodyControlReceiptError(ValueError): pass

def fail(detail):
    raise BodyControlReceiptError('Bone control validation failed: ' + detail
        + '. Transfer again using a frontend supporting body controls.')

def _hash(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value): fail('invalid SHA256')
    return value

def _uuid(value):
    try:
        parsed = uuid.UUID(value)
        if str(parsed) != value or parsed.int == 0: fail('noncanonical body/scene UUID')
    except (TypeError, ValueError, AttributeError): fail('invalid body/scene UUID')
    return value

def token(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{12}', value): fail('invalid upload_id')
    return value

def _json_hash(value):
    try:
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
            ensure_ascii=True, allow_nan=False).encode('utf-8')).hexdigest()
    except (TypeError, ValueError): fail('invalid immutable author contract')

def _envelope(raw, kind, version):
    from .module import get_cbor2
    try:
        cbor = get_cbor2(); value = cbor.loads(raw)
        if cbor.dumps(value) != raw: fail('trailing, duplicate or noncanonical CBOR')
    except BodyControlReceiptError: raise
    except Exception: fail('missing or damaged ' + kind)
    if (not isinstance(value, dict) or set(value) != {'version', 'kind', 'payload'}
            or type(value['version']) is not int or value['version'] != version
            or value['kind'] != kind or not isinstance(value['payload'], dict)):
        fail('old or unsupported ' + kind)
    return value['payload']

def required_contract(intent):
    if (not isinstance(intent, dict) or set(intent) != {'body_controls_v1', 'controls'}
            or type(intent['body_controls_v1']) is not bool or not isinstance(intent['controls'], list)):
        fail('invalid current author intent')
    return _json_hash(intent) if intent['body_controls_v1'] else None

def _real(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        fail('control transform requires finite real values')
    return float(value)


def _vector(value, size, quaternion=False):
    if not isinstance(value, list) or len(value) != size:
        fail('invalid control transform components')
    result = [_real(v) for v in value]
    if quaternion and abs(math.sqrt(sum(v*v for v in result)) - 1.) > 1e-6:
        fail('invalid control quaternion')
    return result


def _rotation(value):
    w, x, y, z = _vector(value, 4, quaternion=True)
    norm = math.sqrt(w*w+x*x+y*y+z*z)
    w, x, y, z = (v/norm for v in (w, x, y, z))
    return ((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
            (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
            (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))


def _rigid_matrix(value):
    import numpy as np
    if not isinstance(value, list) or len(value) != 4 or any(not isinstance(r, list) or len(r) != 4 for r in value):
        fail('invalid automatic parent matrix')
    a = np.array([[_real(v) for v in row] for row in value])
    if (not np.allclose(a[3], [0., 0., 0., 1.], atol=1e-8, rtol=0)
            or not np.allclose(a[:3,:3].T @ a[:3,:3], np.eye(3), atol=1e-6, rtol=0)
            or abs(float(np.linalg.det(a[:3,:3]))-1.) > 1e-6):
        fail('automatic parent matrix is not rigid')
    return a


def _control_signature(motion, uid):
    if not isinstance(motion, dict) or motion.get('body_uuid', uid) != uid:
        fail('control trajectory body identity differs')
    mode, keys = motion.get('mode'), motion.get('keys')
    if not isinstance(mode, str) or mode not in ('dynamic', 'static', 'kinematic'):
        fail('invalid control mode')
    if not isinstance(keys, list) or not 1 <= len(keys) <= 4096:
        fail('invalid control key count')
    times, previous = [], -1.
    for key in keys:
        if not isinstance(key, dict) or set(key) != {'time_s','center_world_m','rotation_world_wxyz'}:
            fail('invalid control key schema')
        time = _real(key['time_s'])
        if (not times and time != 0.) or time <= previous:
            fail('invalid control key time order')
        _vector(key['center_world_m'], 3); _vector(key['rotation_world_wxyz'], 4, quaternion=True)
        times.append(key['time_s']); previous = time
    events, previous, selected = motion.get('control_events', []), 0., mode
    if not isinstance(events, list) or len(events) > 4096:
        fail('invalid control event count')
    program = []
    for event in events:
        if not isinstance(event, dict) or set(event) not in ({'time_s','mode'}, {'time_s','mode','pose_binding'}):
            fail('invalid control event schema')
        time, target = _real(event['time_s']), event['mode']
        if time <= previous or not isinstance(target, str) or target not in ('dynamic','static','kinematic') or target == selected:
            fail('invalid control event order/mode')
        if 'pose_binding' in event:
            allowed = {'static': ('absolute','current_hold'), 'kinematic': ('absolute','rigid_delta_v1')}
            if event['pose_binding'] not in allowed.get(target, ()):
                fail('invalid control event pose binding')
        program.append((event['time_s'], target, event.get('pose_binding')))
        previous, selected = time, target
    if (mode == 'kinematic' and len(keys) < 2) or (mode == 'static' and not events and len(keys) != 1):
        fail('invalid mode trajectory key count')
    return (uid, mode, tuple(times), tuple(program))


def _automatic_anchor_controls(document):
    rows = document['bindings']
    if not any('automatic_parent_anchor_v1' in row for row in rows):
        return []
    import numpy as np
    bodies, bones = set(), set()
    for row in rows:
        body, bone = _uuid(row.get('body_uuid')), _uuid(row.get('bone_uuid'))
        if body in bodies or bone in bones: fail('duplicate original automatic parent owner')
        bodies.add(body); bones.add(bone)
    parents, anchors = {}, {}
    meta_keys = {'version','anchor_body_uuid','parent_bone_uuid','parent_bone_name',
                 'parent_world_matrix_blender','anchor_proxy_world_matrix_blender','root_motion'}
    for row in rows:
        if 'automatic_parent_anchor_v1' not in row: continue
        meta = row['automatic_parent_anchor_v1']
        if (not isinstance(meta, dict) or set(meta) != meta_keys
                or type(meta['version']) is not int or meta['version'] != 1):
            fail('invalid automatic parent metadata')
        uid, parent = _uuid(meta['anchor_body_uuid']), _uuid(meta['parent_bone_uuid'])
        arm = _uuid(row.get('armature_uuid'))
        if uid in bodies or uid in bones or parent in bones or uid == parent:
            fail('automatic parent identity collides with original body/bone')
        name = meta['parent_bone_name']
        if (not isinstance(name, str) or not name or row.get('enabled') is not True
                or row.get('root_mode') != 'FREE' or row.get('root_motion') is not None
                or row.get('parent_body_uuid') is not None or row.get('parent_bone_uuid') is not None
                or row.get('root_boundary') != {'is_root':True,'nonphysical_parent_bone':name}):
            fail('automatic parent declaration differs from child root intent')
        matrix = _rigid_matrix(meta['parent_world_matrix_blender'])
        proxy = _rigid_matrix(meta['anchor_proxy_world_matrix_blender'])
        if not np.allclose(matrix, proxy, atol=1e-8, rtol=0):
            fail('automatic parent proxy differs from parent pose')
        # Detached local projection supplies the frontend's documented default;
        # never rewrite the authored document or its full-contract digest.
        motion = meta['root_motion']
        allowed = {'body_uuid','mode','authoring','initial_frame','keys','control_events','root_index','fit_tolerance_m'}
        if not isinstance(motion, dict) or set(motion)-allowed or motion.get('mode') != 'kinematic':
            fail('invalid automatic parent trajectory')
        if 'root_index' in motion and (type(motion['root_index']) is not int or motion['root_index'] < 0):
            fail('invalid automatic parent root index')
        if 'fit_tolerance_m' in motion and _real(motion['fit_tolerance_m']) <= 0:
            fail('invalid automatic parent fit tolerance')
        projected = dict(motion, body_uuid=motion.get('body_uuid',uid))
        signature = _control_signature(projected,uid)
        initial = motion.get('initial_frame')
        if (motion.get('authoring') not in ('object_world_trs','native_frame')
                or not isinstance(initial, dict) or set(initial) != {'translation_m','quaternion_wxyz'}):
            fail('invalid automatic parent initial frame')
        xyz = _vector(initial['translation_m'],3); rotation = _rotation(initial['quaternion_wxyz'])
        solver = np.array([[1.,0.,0.,0.],[0.,0.,1.,0.],[0.,-1.,0.,0.],[0.,0.,0.,1.]]) @ matrix
        if (not np.allclose(xyz,solver[:3,3],atol=1e-8,rtol=0)
                or not np.allclose(rotation,solver[:3,:3],atol=1e-6,rtol=0)):
            fail('automatic parent initial frame differs from parent pose')
        first = motion['keys'][0]
        if (first['center_world_m'] != initial['translation_m']
                or not np.allclose(_rotation(first['rotation_world_wxyz']),rotation,atol=1e-6,rtol=0)):
            fail('automatic parent first key differs from initial frame')
        owner = (arm,parent); fingerprint = _json_hash(meta)
        if owner in parents and parents[owner] != (uid,fingerprint):
            fail('same automatic parent declares different anchor or metadata')
        if uid in anchors and anchors[uid][:2] != (owner,fingerprint):
            fail('automatic anchor is claimed by different parent or metadata')
        parents[owner] = (uid,fingerprint); anchors[uid] = (owner,fingerprint,signature)
    return [entry[2] for entry in anchors.values()]


def uploaded_contract(raw):
    param = _envelope(raw, 'Param', 2)
    document = param.get('blender_bones_v1')
    if document is None or 'body_controls_v1' not in document: return None
    if document['body_controls_v1'] is not True: fail('body control request must be true')
    intent_hash = _json_hash(validate_author_intent(document, fail))
    models={uid:g[0].get('model') for g in param.get('group',[]) if isinstance(g,list) and len(g)>2 for uid in g[2]}
    controls = []
    for row in document.get('bindings', []):
        root = row.get('root_motion')
        if root is not None:
            family=[key for key in ('fem_root_control_v1','fem_root_control_events_v1') if key in row]
            is_fem=models.get(row.get('body_uuid')) in ('snhk','arap')
            if (is_fem and (len(family)!=1 or row[family[0]] is not True)) or (family and not is_fem):fail('FEM root marker differs from physical material')
            if root.get('body_uuid') != row.get('body_uuid'): fail('uploaded control body differs from binding')
            if 'fem_root_control_events_v1' in row:
                from .fem_body_control_events import signature as event_signature
                controls.append(event_signature(row))
            elif 'fem_root_control_v1' in row:
                from .fem_body_control_receipt import signature
                if row['fem_root_control_v1'] is not True:fail('invalid FEM root family marker')
                controls.append(signature(root,_uuid(root['body_uuid']),row.get('root_motion_source')))
            else:controls.append(_control_signature(root, _uuid(root['body_uuid'])))
    if any(len(r)==8 for r in controls) and any(len(r)==6 for r in controls):fail('FEM v1/v2 scene mixing is unsupported')
    controls.extend(_automatic_anchor_controls(document))
    if not controls or len({r[0] for r in controls}) != len(controls): fail('missing or duplicate uploaded controls')
    return _json_hash(document), tuple(sorted(controls)), intent_hash

@dataclass(frozen=True)
class Expectation:
    author_hash: str
    intent_hash: str
    controls: tuple
    data_hash: str
    param_hash: str
    upload_id: str = ''
    root: str = ''

@dataclass(frozen=True)
class VerifiedSeal:
    scene_uuid: str
    context_hash: str
    joint_hash: str
    sidecar_hash: str
    inputs_hash: str

def _verify_pdrd(receipt_raw, context_raw, joint_raw, sidecar_raw, expected, previous=None):
    receipt = _envelope(receipt_raw, 'BodyControlReceipt', 1)
    if (set(receipt) != {'capability', 'request_token', 'scene_uuid',
            'author_contract_sha256', 'scene_inputs_sha256', 'context_sha256',
            'joint_sha256', 'sidecar_sha256'} or receipt['capability'] != CAPABILITY):
        fail('missing new body-control capability')
    if token(receipt['request_token']) != token(expected.upload_id): fail('stale upload receipt')
    if _hash(receipt['author_contract_sha256']) != expected.author_hash: fail('wrong author contract')
    raw = {'context': context_raw, 'joint': joint_raw, 'sidecar': sidecar_raw}
    hashes = {k: hashlib.sha256(v).hexdigest() for k, v in raw.items()}
    if any(_hash(receipt[k + '_sha256']) != v for k, v in hashes.items()): fail('downloaded seal bytes mismatch')
    ctx = _envelope(context_raw, 'ppf.pdrd_scene_context', 2)
    joint = _envelope(joint_raw, 'ppf.pdrd_ball_anchor', 9)
    base_fields = {'context_version', 'scene_uuid', 'builder_contract_version',
                   'complete', 'bodies', 'inputs', 'optional_absent', 'scene_inputs_sha256'}
    if (set(ctx) not in (base_fields, base_fields | {'restoring_curves_v1'})
            or ctx['complete'] is not True or type(ctx['context_version']) is not int
            or ctx['context_version'] != 2 or ctx['builder_contract_version'] != 2):
        fail('incomplete or unsupported scene context')
    if 'restoring_curves_v1' in ctx:
        marker = ctx['restoring_curves_v1']
        if (not isinstance(marker, dict) or set(marker) != {'required_feature',
                'sidecar_sha256', 'sidecar_byte_length', 'combined_inputs_sha256'}
                or marker['required_feature'] != 'restoring_curves_v1'
                or type(marker['sidecar_byte_length']) is not int
                or not 0 < marker['sidecar_byte_length'] <= 32 * 1024 * 1024):
            fail('invalid restoring curves context marker')
        combined = hashlib.sha256((_hash(ctx['scene_inputs_sha256'])
                    + _hash(marker['sidecar_sha256'])).encode('ascii')).hexdigest()
        if _hash(marker['combined_inputs_sha256']) != combined:
            fail('restoring curves combined input digest mismatch')
    scene = _uuid(ctx['scene_uuid'])
    if any(p.get('scene_uuid') != scene for p in (receipt, joint)): fail('scene build identity mismatch')
    features = joint.get('required_features')
    root_features = {2:'pdrd_kinematic_roots_v2',3:'pdrd_kinematic_roots_v3'}
    if (not isinstance(features, list) or not all(isinstance(f,str) for f in features)
            or features != sorted(set(features)) or CAPABILITY not in features):
        fail('sealed joint manifest does not authorize new body controls')
    advertised = {feature for feature in features if feature.startswith('pdrd_kinematic_roots_v')}
    versions = [version for version, feature in root_features.items() if advertised == {feature}]
    if len(versions) != 1: fail('sealed joint manifest has unsupported or mixed root versions')
    root_version = versions[0]
    side = _envelope(sidecar_raw, 'ppf.pdrd_root_motion', root_version)
    if side.get('scene_uuid') != scene: fail('scene build identity mismatch')
    lines = ['ppf.pdrd_scene_context/v2', scene]
    bodies, indices = {}, set()
    previous_uid = ''
    if not isinstance(ctx['bodies'], list): fail('invalid body manifest')
    for b in ctx['bodies']:
        if not isinstance(b, dict) or set(b) != {'body_uuid', 'runtime_body_index', 'vertex_count'}:
            fail('invalid native body membership')
        uid = _uuid(b['body_uuid']); index, count = b['runtime_body_index'], b['vertex_count']
        if (uid <= previous_uid or type(index) is not int or index < 0 or index in indices
                or type(count) is not int or count <= 0): fail('duplicate or invalid body membership')
        previous_uid = uid; bodies[uid] = index; indices.add(index)
        lines.append(f'{uid}\t{index}\t{count}')
    if joint.get('bodies') != [{'body_uuid': u, 'runtime_body_index': i} for u, i in bodies.items()]:
        fail('joint and context body membership differ')
    inventory, last = {}, ''
    if not isinstance(ctx['inputs'], list): fail('invalid consumed inventory')
    for row in ctx['inputs']:
        if not isinstance(row, dict) or set(row) != {'relative_path', 'byte_length', 'sha256'}: fail('invalid input record')
        path = row['relative_path']
        if (not isinstance(path, str) or not path or path <= last or '\\' in path or '\t' in path or '\n' in path
                or PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts
                or type(row['byte_length']) is not int or row['byte_length'] < 0): fail('invalid input path')
        last = path; inventory[path] = row
        if path == 'bin/restoring_curves_v1.json':
            fail('restoring curves extension cannot reference itself in base inputs')
        lines.append(f'{path}\t{row["byte_length"]}\t{_hash(row["sha256"])}')
    absent = ctx['optional_absent']
    if (not isinstance(absent, list) or absent != sorted(set(absent))
            or any(p not in {'dyn_param.txt', 'bin/rest_vert.bin', 'bin/rest_vert_anim.bin'} or p in inventory for p in absent)):
        fail('invalid absent input declaration')
    lines.extend('optional_absent\t' + p for p in absent)
    digest = hashlib.sha256(('\n'.join(lines) + '\n').encode('utf-8')).hexdigest()
    if any(_hash(p.get('scene_inputs_sha256')) != digest for p in (ctx, joint, receipt)): fail('sealed input digest mismatch')
    consumed = inventory.get(SIDECAR_FILE)
    if not consumed or consumed['sha256'] != hashes['sidecar'] or consumed['byte_length'] != len(sidecar_raw):
        fail('body trajectory is absent from consumed inputs')
    if set(side) != {'scene_uuid', 'interpolation', 'extrapolation', 'bodies'} or not isinstance(side['bodies'], list):
        fail('invalid sidecar')
    if side['interpolation'] != 'linear_center_shortest_slerp_v1' or side['extrapolation'] != 'hold':
        fail('unsupported sidecar interpolation/extrapolation')
    controls = []
    last_index = -1
    for b in side['bodies']:
        if not isinstance(b,dict): fail('invalid sidecar body schema')
        uid = _uuid(b.get('body_uuid')); index = b.get('runtime_body_index')
        if bodies.get(uid) != index or type(index) is not int or index <= last_index: fail('sidecar body membership differs')
        last_index = index
        base_fields = {'body_uuid','runtime_body_index','mode','keys'}
        if root_version == 3: base_fields.add('reference_transform')
        if set(b) not in (base_fields, base_fields | {'control_events'}): fail('invalid sidecar body schema')
        if root_version == 3:
            reference = b['reference_transform']
            if not isinstance(reference,dict) or set(reference) != {'translation_local_m','rotation_local_wxyz'}:
                fail('invalid v3 reference transform')
            _vector(reference['translation_local_m'],3)
            _vector(reference['rotation_local_wxyz'],4,quaternion=True)
        controls.append(_control_signature(b,uid))
    if tuple(sorted(controls)) != expected.controls: fail('sealed body controls differ from uploaded request')
    seal = VerifiedSeal(scene, hashes['context'], hashes['joint'], hashes['sidecar'], digest)
    if previous is not None and seal != previous: fail('sealed build changed after READY')
    return seal


def required_files(expected):
    if any(isinstance(row,tuple) and len(row)==8 for row in expected.controls):
        from .fem_body_control_events_receipt import required_files as event_files
        return event_files(expected)
    from .fem_body_control_receipt import partition,RECEIPT_FILE as FR,SIDECAR_FILE as FS
    rigid,fem=partition(expected.controls)
    return tuple(dict.fromkeys(([RECEIPT_FILE,CONTEXT_FILE,JOINT_FILE,SIDECAR_FILE] if rigid else [])+([FR,CONTEXT_FILE,JOINT_FILE,FS,'map.pickle'] if fem else [])))

def verify_downloads(raw,expected,previous=None):
    if any(isinstance(row,tuple) and len(row)==8 for row in expected.controls):
        from .fem_body_control_events_receipt import verify_downloads as event_verify
        return event_verify(raw,expected,previous)
    from .fem_body_control_receipt import partition,verify as verify_fem
    from dataclasses import replace
    rigid,fem=partition(expected.controls)
    if not fem:return verify(raw[RECEIPT_FILE],raw[CONTEXT_FILE],raw[JOINT_FILE],raw[SIDECAR_FILE],expected,previous)
    rigid_seal=(_verify_pdrd(raw[RECEIPT_FILE],raw[CONTEXT_FILE],raw[JOINT_FILE],raw[SIDECAR_FILE],replace(expected,controls=rigid)) if rigid else None)
    seal=(rigid_seal,verify_fem(raw,expected))
    if previous is not None and previous!=seal:fail('FEM/PDRD seal changed after validation')
    return seal

def verify(receipt_raw,context_raw,joint_raw,sidecar_raw,expected,previous=None):
    return _verify_pdrd(receipt_raw,context_raw,joint_raw,sidecar_raw,expected,previous)
