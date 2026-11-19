"""Receipt of the explicit author body-control contract and final real seal."""
from pathlib import Path
import hashlib, json, re
import cbor2
from ._body_control_intent_ import validate_author_intent

CAPABILITY = 'pdrd_body_controls_v1'
ROOT_FEATURES = {2: 'pdrd_kinematic_roots_v2',
                 3: 'pdrd_kinematic_roots_v3'}
FILE = 'body_control_receipt.cbor'
KIND = 'BodyControlReceipt'

def fail(detail):
    raise ValueError('BODY_CONTROL_RECEIPT_INVALID: ' + detail)

def contract_sha256(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode('utf-8')).hexdigest()

def record_author(fixed, document):
    validate_author_intent(document, fail)
    context = getattr(fixed, '_collision_filter_upload_context', None)
    if not isinstance(context, dict) or not re.fullmatch('[0-9a-f]{12}', context.get('request_token', '')):
        fail('new body controls require a current atomic Transfer upload identity')
    fixed._body_controls_author_contract_sha256 = contract_sha256(document)

def write_receipt(fixed_session):
    fixed = fixed_session.fixed_scene
    if not getattr(fixed, '_pdrd_body_controls_v1', False):
        return
    upload = getattr(fixed, '_collision_filter_upload_context', None)
    author_hash = getattr(fixed, '_body_controls_author_contract_sha256', None)
    if not upload or not author_hash:
        fail('missing frozen author/upload contract')
    root_version = getattr(fixed, '_pdrd_root_motion_version', None)
    root_feature = ROOT_FEATURES.get(root_version)
    if root_feature is None:
        fail('body controls require an explicitly supported root sidecar version')
    root = Path(upload['root'])
    token = (root / 'upload_id.txt').read_text(encoding='ascii').strip()
    if token != upload['request_token']:
        fail('upload changed during body-control build; Transfer again')
    path = Path(fixed_session.info.path)
    rels = {'context': 'bin/pdrd_scene_context.cbor', 'joint': 'bin/pdrd_ball_anchor.cbor',
            'sidecar': 'bin/pdrd_root_motion_v1.cbor'}
    raw = {k: (path / v).read_bytes() for k, v in rels.items()}
    decoded = {k: cbor2.loads(v) for k, v in raw.items()}
    ctx, joint, side = (decoded[k]['payload'] for k in ('context', 'joint', 'sidecar'))
    advertised_roots = set(joint.get('required_features', [])) & {
        'pdrd_kinematic_roots_v1', 'pdrd_kinematic_roots_v2',
        'pdrd_kinematic_roots_v3'}
    if (decoded['context']['version'] != 2 or decoded['joint']['version'] != 9
            or decoded['sidecar']['version'] != root_version or ctx.get('complete') is not True
            or CAPABILITY not in joint.get('required_features', [])
            or advertised_roots != {root_feature}
            or any(p.get('scene_uuid') != fixed._scene_uuid for p in (ctx, joint, side))
            or joint.get('scene_inputs_sha256') != ctx['scene_inputs_sha256']
            or side != fixed._pdrd_root_motion_payload):
        fail('final sealed context/joint/sidecar does not prove the new contract')
    inventory = {row['relative_path']: row for row in ctx['inputs']}
    if inventory.get(rels['sidecar'], {}).get('sha256') != hashlib.sha256(raw['sidecar']).hexdigest():
        fail('body sidecar was not consumed by the final seal')
    payload = {'capability': CAPABILITY, 'request_token': token,
               'scene_uuid': fixed._scene_uuid, 'author_contract_sha256': author_hash,
               'scene_inputs_sha256': ctx['scene_inputs_sha256'],
               **{k + '_sha256': hashlib.sha256(v).hexdigest() for k, v in raw.items()}}
    # Read identity again immediately before publishing; a late old build
    # cannot borrow a newer upload token with the same author contract.
    if (root / 'upload_id.txt').read_text(encoding='ascii').strip() != token:
        fail('upload changed before receipt publication')
    (path / FILE).write_bytes(cbor2.dumps({'version': 1, 'kind': KIND, 'payload': payload}))
