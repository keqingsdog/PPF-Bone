"""Immutable engine/producer admission shared by Build and Run.

No backend probe occurs here. Only the reviewed unified executable is asked
for its host-only SOFT capability; native format/device probes retain their
existing launch gate in frontend._run_directory.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from . import _r14_product_profile as profiles

UNIFIED_EXE_SHA256 = 'f04bf8368082329acbe995c37bcf8211b639d9682444d2931ef11da3a5fe44c5'
SOFT_FEATURE = 'bone_connected_collision_v1'
from ._build_receipt_validation_ import RECEIPT, DOMAIN, scan_inputs, inventory_digest, validate_receipt

@dataclass(frozen=True)
class BuildContract:
    profile: profiles.ProductProfile
    exe_sha256: str
    dll_sha256: str
    format_identity: str
    producer_version: int
    capabilities: tuple[str, ...]
    capability_snapshot: tuple[tuple[str, object], ...] = ()
    fem_capability_snapshot: tuple[tuple[str, object], ...] = ()
    event_capability_snapshot: tuple[tuple[str, object], ...] = ()

    @property
    def exe_path(self):
        return self.profile.exe_path

    @property
    def dll_path(self):
        return self.profile.dll_path

    @property
    def product_root(self):
        return self.profile.product_root

    def receipt(self):
        # A fresh object each call; never expose mutable contract internals.
        return {'schema': 'ppf.build_contract.v1', 'product': self.profile.product_name,
                'product_root': str(self.product_root), 'exe_path': str(self.exe_path),
                'dll_path': str(self.dll_path), 'exe_sha256': self.exe_sha256,
                'dll_sha256': self.dll_sha256, 'format_identity': self.format_identity,
                'producer_version': self.producer_version,
                'capabilities': list(self.capabilities), 'capability_snapshot': {**dict(self.capability_snapshot), **({'fem_body_controls':dict(getattr(self,'fem_capability_snapshot',()))} if getattr(self,'fem_capability_snapshot',()) else {}), **({'fem_body_control_events':dict(self.event_capability_snapshot)} if self.event_capability_snapshot else {})}}


def _event_product_pair(profile, exe_hash, dll_hash):
    if profile.local_source:
        identity = profiles.verify_binaries(profile)
        return (exe_hash, dll_hash) == (identity['exe_sha256'], identity['dll_sha256'])
    if profile.product_name == 'R17 FEM ABS Events Candidate':
        pair = profiles.FEM_ABS_EVENTS_EXE_SHA256, profiles.FEM_ABS_EVENTS_DLL_SHA256
    elif profile.product_name == 'R17 Bone Resume Candidate':
        pair = profiles.BONE_RESUME_EXE_SHA256, profiles.BONE_RESUME_DLL_SHA256
    elif profile.product_name == 'R17 Bone Rewind Candidate':
        pair = profiles.BONE_REWIND_EXE_SHA256, profiles.BONE_REWIND_DLL_SHA256
    else:
        return False
    return (exe_hash, dll_hash) == pair


def freeze_build_contract(profile=None):
    profile = profiles.load_profile() if profile is None else profile
    identity = profiles.verify_binaries(profile)
    local = profiles.verify_local_capabilities(profile) if profile.local_source else None
    from . import _rust
    stamp = getattr(_rust, '__source_stamp__', None)
    if not isinstance(stamp, str) or not stamp:
        raise ValueError('BUILD_CONTRACT_FORMAT_IDENTITY_UNAVAILABLE')
    if local is not None and stamp != local['format_identity']:
        raise ValueError('LOCAL_SOURCE_EXTENSION_FORMAT_MISMATCH')
    local_module=profiles.local_source_module() if profile.local_source else None
    local_doc=local_module.load(profile.product_root) if local_module else None
    if local_doc is not None and local_doc['schema']==local_module.RELATIVE_SCHEMA:
        features=local_module.contract_features(local_doc,local)
        docs=local['capabilities']
        result=BuildContract(profile,identity['exe_sha256'],identity['dll_sha256'],stamp,
            3 if 'pdrd_kinematic_roots_v3' in features else 2,features,
            tuple(sorted(docs['bone'].items())) if SOFT_FEATURE in features else (),
            tuple(sorted(docs['fem'].items())) if 'fem_prescribed_roots_v1' in features else (),
            tuple(sorted(docs['events'].items())) if 'fem_root_control_events_v1' in features else ())
        return result
    unified = (profile.product_name == 'R17 Unified Candidate BJ' and identity['exe_sha256'] == UNIFIED_EXE_SHA256)
    fem = (profile.product_name == 'R17 FEM Root Candidate' and identity['exe_sha256'] == profiles.FEM_ROOT_EXE_SHA256 and identity['dll_sha256'] == profiles.FEM_ROOT_DLL_SHA256)
    events = _event_product_pair(profile, identity['exe_sha256'], identity['dll_sha256'])
    fem = fem or events
    unified = unified or fem
    event_snapshot=()
    fem_snapshot=()
    snapshot = ()
    capabilities = ()
    if unified:
        from ._bone_connected_collision_ import capability
        context = (str(profile.exe_path), str(profile.runtime_dir), profiles.solver_child_env(profile, os.environ))
        doc = capability(profile.exe_path, context=context)
        snapshot = tuple(sorted(doc.items()))
        capabilities = (SOFT_FEATURE, 'pdrd_kinematic_roots_v3')
        if fem:
            from ._fem_body_controls_ import capability
            fem_snapshot=tuple(sorted(capability(profile).items()))
            capabilities=tuple(sorted(capabilities+('fem_prescribed_roots_v1',)))
    if events:
        from ._fem_body_control_events_ import capability as event_capability
        event_snapshot=tuple(sorted(event_capability(profile).items()))
        capabilities=tuple(sorted(capabilities+('fem_root_control_events_v1',)))
    result = BuildContract(profile, identity['exe_sha256'], identity['dll_sha256'], stamp,
                           3 if unified else 2, capabilities, snapshot, fem_snapshot,event_snapshot)
    verify_build_contract(result)
    return result


def verify_build_contract(contract, recheck_selection=True):
    if not isinstance(contract, BuildContract):
        raise ValueError('BUILD_CONTRACT_REQUIRED: Build again with a frozen product')
    identity = profiles.verify_binaries(contract.profile)
    if (identity['exe_sha256'], identity['dll_sha256']) != (contract.exe_sha256, contract.dll_sha256):
        raise ValueError('BUILD_CONTRACT_BINARY_CHANGED: Build again')
    from . import _rust
    if getattr(_rust, '__source_stamp__', None) != contract.format_identity:
        raise ValueError('BUILD_CONTRACT_FORMAT_CHANGED: Build again')
    if contract.profile.local_source:
        local_module=profiles.local_source_module();doc=local_module.load(contract.product_root)
        if doc['schema']==local_module.RELATIVE_SCHEMA:
            actual=profiles.verify_local_capabilities(contract.profile)
            features=local_module.contract_features(doc,actual);docs=actual['capabilities']
            expected=(tuple(sorted(docs['bone'].items())) if SOFT_FEATURE in features else (),
                      tuple(sorted(docs['fem'].items())) if 'fem_prescribed_roots_v1' in features else (),
                      tuple(sorted(docs['events'].items())) if 'fem_root_control_events_v1' in features else ())
            if contract.capabilities!=features or contract.producer_version!=(3 if 'pdrd_kinematic_roots_v3' in features else 2) or (contract.capability_snapshot,contract.fem_capability_snapshot,contract.event_capability_snapshot)!=expected:
                raise ValueError('LOCAL_SOURCE_CAPABILITY_SNAPSHOT_CHANGED')
            if recheck_selection and profiles.load_profile()!=contract.profile:
                raise ValueError('BUILD_CONTRACT_SELECTION_CHANGED: select the Build engine or Build again')
            return contract
    unified = contract.profile.product_name == 'R17 Unified Candidate BJ' and contract.exe_sha256 == UNIFIED_EXE_SHA256
    fem = contract.profile.product_name == 'R17 FEM Root Candidate' and contract.exe_sha256 == profiles.FEM_ROOT_EXE_SHA256 and contract.dll_sha256 == profiles.FEM_ROOT_DLL_SHA256
    events=_event_product_pair(contract.profile, contract.exe_sha256, contract.dll_sha256)
    fem=fem or events
    unified = unified or fem
    expected = ((SOFT_FEATURE,'fem_prescribed_roots_v1','pdrd_kinematic_roots_v3') if fem else ((SOFT_FEATURE, 'pdrd_kinematic_roots_v3') if unified else ()))
    if events:expected=tuple(sorted(expected+('fem_root_control_events_v1',)))
    from ._fem_body_control_events_ import CAPABILITY_DOCUMENT as EVENT_DOCUMENT
    if dict(contract.event_capability_snapshot)!=(EVENT_DOCUMENT if events else {}):
        raise ValueError('FEM_EVENT_BUILD_CONTRACT_CAPABILITY_MISMATCH')
    from ._fem_body_controls_ import CAPABILITY_DOCUMENT
    if dict(getattr(contract,'fem_capability_snapshot',()))!=(CAPABILITY_DOCUMENT if fem else {}):
        raise ValueError('FEM_BUILD_CONTRACT_CAPABILITY_MISMATCH')
    if contract.producer_version != (3 if unified else 2) or contract.capabilities != expected:
        raise ValueError('BUILD_CONTRACT_PRODUCER_MISMATCH')
    if unified and dict(contract.capability_snapshot) != {'kind': 'ppf.bone_connected_collision.capability', 'version': 1,
            'required_feature': SOFT_FEATURE, 'supported': True, 'semantic_version': 1}:
        raise ValueError('BUILD_CONTRACT_CAPABILITY_MISMATCH')
    if recheck_selection and profiles.load_profile() != contract.profile:
        raise ValueError('BUILD_CONTRACT_SELECTION_CHANGED: select the Build engine or Build again')
    return contract


def _event_pose_binding_supported(contract, binding):
    """Allow ABS only for the exact candidate/native capability pair."""
    if binding in ('current_hold', 'rigid_delta_v1'):
        return True
    if binding != 'absolute':
        return False
    snapshot = dict(getattr(contract, 'event_capability_snapshot', ()))
    return ((contract.profile.local_source or contract.profile.product_name in ('R17 FEM ABS Events Candidate', 'R17 Bone Resume Candidate', 'R17 Bone Rewind Candidate'))
            and 'fem_root_control_events_v1' in contract.capabilities
            and 'absolute' in snapshot.get('supported_event_pose_bindings', ())
            and snapshot.get('resume_supported') is False)


def product_selected():
    """Selection is explicit; an absent selection never means an E-drive default."""
    if 'PPF_CTS_PRODUCT_ROOT' not in os.environ:
        return False
    root = os.environ['PPF_CTS_PRODUCT_ROOT']
    if root and profiles.has_local_source(root):
        local=profiles.local_source_module()
        doc=local.read_json(Path(root)/local.CONFIG)
        if doc.get('schema') == local.RELATIVE_SCHEMA and doc.get('scope') == 'source':
            return False  # Explicit source locator still is not extension intent.
    return True


_NATIVE_EXTENSION_ATTRS = (
    '_pdrd_hard_authoring', '_pdrd_root_motion_authoring',
    '_pdrd_ball_anchor_payload', '_pdrd_root_motion_payload',
    '_fem_bone_joint_payload', '_fem_body_controls_payload',
    '_fem_body_control_events_payload', '_blender_bone_bindings',
    '_cross_links_payload',
    '_soft_limit_auto_intent',
)
_EXTENSION_SIDECARS = (
    RECEIPT, 'blender_bone_bindings.cbor', 'bin/pdrd_ball_anchor.cbor',
    'bin/pdrd_scene_context.cbor', 'bin/restoring_curves_v1.json',
    'bin/fem_bone_joints.json', 'bin/fem_body_controls_v1.json',
    'bin/fem_body_controls_v2.json', 'bin/pdrd_root_motion_v1.cbor',
    'bin/cross_links_v1.json',
    'bin/pdrd_root_motion_v2.cbor', 'bin/pdrd_root_motion_v3.cbor',
)


def requested_extensions(data=None, holder=None, path=None):
    """Ordinary PDRD material, pins and collision allowances are not bones.

    Test presence, not truthiness: malformed extension intent must be refused
    by its original validator, never quietly converted into an ordinary scene.
    """
    if data is not None:
        if not isinstance(data, dict):
            raise ValueError('BUILD_CONTRACT_INVALID_PARAM_DOCUMENT')
        if 'blender_bones_v1' in data:
            if not isinstance(data['blender_bones_v1'], dict):
                raise ValueError('BLENDER_BONE_INVALID: expected document object')
            return True
        if 'restoring_curves_v1' in data:
            if data['restoring_curves_v1'] is None:
                raise ValueError('RESTORING_SCHEMA_INVALID: null author document')
            from ._restoring_curves_ import active
            if active(data['restoring_curves_v1']):
                return True
    if holder is not None:
        if any(getattr(holder, key, None) is not None for key in _NATIVE_EXTENSION_ATTRS):
            return True
        author = getattr(holder, '_restoring_curves_author', None)
        if author is not None:
            from ._restoring_curves_ import active
            if active(author):
                return True
    return bool(path is not None and any((Path(path) / key).exists() for key in _EXTENSION_SIDECARS))


def optional_contract_for(holder, data=None, path=None):
    """General boundary: preserve official native None, keep extensions strict."""
    prior = getattr(holder, '_build_contract', None)
    if prior is not None:
        # Removing a selection or choosing CPU cannot retire a sealed contract.
        verify_build_contract(prior)
        return prior
    if product_selected() or requested_extensions(data, holder, path):
        return contract_for(holder)
    return None


def contract_for(holder):
    contract = getattr(holder, '_build_contract', None)
    if contract is None:
        contract = freeze_build_contract()
        holder._build_contract = contract
    verify_build_contract(contract)
    return contract


def validate_requests(contract, param_data):
    verify_build_contract(contract)
    document = (param_data or {}).get('blender_bones_v1')
    if not isinstance(document, dict):
        if 'blender_bones_v1' in (param_data or {}):
            raise ValueError('BLENDER_BONE_INVALID: expected document object')
        return
    if 'cross_links_v1' in document:
        from ._cross_links_ import validate_author, capability
        validate_author(document['cross_links_v1'], document.get('bindings', []))
        capability(contract, directional=any('tension_enabled' in row for row in document['cross_links_v1']['links']))
    if document.get('cross_link_collision_pairs_v1'):
        from ._cross_links_ import capability
        capability(contract, collision_pairs=True)
    # Determine the real material endpoint from source groups before CPU
    # collision exclusion, automatic carrier creation or author installation.
    models = {uid: group[0].get('model') for group in param_data.get('group', [])
              if len(group) > 2 for uid in group[2]}
    rows = {row.get('body_uuid'): row for row in document.get('bindings', [])}
    for row in rows.values():
        if 'fem_root_control_events_v1' in row:
            if row['fem_root_control_events_v1'] is not True or 'fem_root_control_events_v1' not in contract.capabilities:
                raise ValueError('FEM_ROOT_EVENTS_CAPABILITY_UNAVAILABLE')
            for event in row.get('root_motion',{}).get('control_events',[]):
                binding=event.get('pose_binding')
                if event.get('mode')!='dynamic' and not _event_pose_binding_supported(contract,binding):
                    raise ValueError('FEM_ROOT_EVENT_POLICY_UNSUPPORTED: pose binding is not supported by the selected paired product')
        if 'fem_root_control_v1' in row and 'fem_prescribed_roots_v1' not in contract.capabilities:
            raise ValueError('FEM_BODY_CONTROL_CAPABILITY_UNAVAILABLE')
        parent = row.get('parent_body_uuid')
        if (row.get('enabled') is True and parent and
                row.get('physics', {}).get('collide_connected') is False and
                any(models.get(uid) != 'pdrd' and uid in models for uid in (parent, row.get('body_uuid')))):
            if SOFT_FEATURE not in contract.capabilities:
                raise ValueError('BONE_CONNECTED_COLLISION_CAPABILITY_UNAVAILABLE: FEM neighbor exclusion requires the reviewed unified engine; Build again after selecting it')


def validate_fixed(contract, fixed):
    if getattr(fixed, '_cross_links_payload', None) is not None:
        from ._cross_links_ import capability
        capability(contract, directional=any('tension_enabled' in row for row in fixed._cross_links_payload['links']))
    if getattr(fixed, '_cross_link_collision_uuid_pairs', None):
        from ._cross_links_ import capability
        capability(contract, collision_pairs=True)
    if getattr(fixed,'_fem_body_control_events_payload',None) is not None and 'fem_root_control_events_v1' not in contract.capabilities:
        raise ValueError('FEM_ROOT_EVENTS_CAPABILITY_UNAVAILABLE')
    if getattr(fixed,'_fem_body_controls_payload',None) is not None and 'fem_prescribed_roots_v1' not in contract.capabilities:
        raise ValueError('FEM_BODY_CONTROL_CAPABILITY_UNAVAILABLE')
    version = getattr(fixed, '_pdrd_root_motion_version', None)
    if version == 3 and contract.producer_version != 3:
        raise ValueError('BUILD_CONTRACT_FEATURE_UNAVAILABLE: pdrd_kinematic_roots_v3')
    from ._bone_connected_collision_ import required
    if required(getattr(fixed, '_fem_bone_joint_payload', None)) and SOFT_FEATURE not in contract.capabilities:
        raise ValueError('BONE_CONNECTED_COLLISION_CAPABILITY_UNAVAILABLE: frozen product cannot consume this Build')


def _sealed_context(path):
    import cbor2
    target = Path(path) / 'bin/pdrd_scene_context.cbor'
    if target.exists():
        return cbor2.loads(target.read_bytes())['payload']
    rows = scan_inputs(path)
    return {'complete': True, 'scene_uuid': None,
            'scene_inputs_sha256': inventory_digest(rows), 'inputs': rows}


def _required_features(path):
    import cbor2
    target = Path(path) / 'bin/pdrd_ball_anchor.cbor'
    return cbor2.loads(target.read_bytes())['payload'].get('required_features', []) if target.exists() else []


def _context_sha(path):
    target = Path(path) / 'bin/pdrd_scene_context.cbor'
    return hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None


def _verify_native_inventory(path, context):
    # Preserve the exact native hash domain; the stronger receipt inventory
    # additionally covers its three excluded sidecars and command.sh.
    if _context_sha(path) is not None:
        native = [row for row in scan_inputs(path) if row['relative_path'] not in {
            'bin/pdrd_scene_context.cbor', 'bin/pdrd_ball_anchor.cbor',
            'bin/restoring_curves_v1.json', 'command.sh'}]
        if context.get('inputs') != native:
            raise ValueError('BUILD_CONTRACT_NATIVE_INVENTORY_MISMATCH: Build again')
        if context.get('complete') is not True:
            raise ValueError('BUILD_CONTRACT_INCOMPLETE_SEAL')


def write_receipt(contract, fixed, path):
    context = _sealed_context(path)
    if context.get('complete') is not True:
        raise ValueError('BUILD_CONTRACT_INCOMPLETE_SEAL')
    validate_fixed(contract, fixed)
    _verify_native_inventory(path, context)
    features = _required_features(path)
    if ('pdrd_kinematic_roots_v3' in features and contract.producer_version != 3
            or SOFT_FEATURE in features and SOFT_FEATURE not in contract.capabilities
            or 'fem_prescribed_roots_v1' in features and 'fem_prescribed_roots_v1' not in contract.capabilities
            or 'fem_root_control_events_v1' in features and 'fem_root_control_events_v1' not in contract.capabilities):
        raise ValueError('BUILD_CONTRACT_FEATURE_UNAVAILABLE')
    result = contract.receipt()
    result.update(inventory_domain=DOMAIN, context_sha256=_context_sha(path), scene_uuid=context['scene_uuid'],
                  scene_inputs_sha256=context['scene_inputs_sha256'], required_features=features,
                  inputs=scan_inputs(path))
    target = Path(path) / RECEIPT
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding='utf-8')
    os.replace(temporary, target)
    try:
        validate_receipt(path, contract.receipt())
    except Exception:
        target.unlink()
        raise


def verify_session(fixed, path):
    contract = optional_contract_for(fixed, path=path)
    if contract is None:
        return None
    validate_fixed(contract, fixed)
    receipt = validate_receipt(path, contract.receipt())
    context = _sealed_context(path)
    _verify_native_inventory(path, context)
    if (receipt['scene_uuid'] != context['scene_uuid']
            or receipt['scene_inputs_sha256'] != context['scene_inputs_sha256']
            or receipt['required_features'] != _required_features(path)):
        raise ValueError('BUILD_CONTRACT_SEAL_CHANGED: Build again')
    return contract
