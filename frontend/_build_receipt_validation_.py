"""Strict stdlib Build receipt validation shared by frontend and launcher.



The receipt lives outside native bin input inventory. Only native configuration

files, both launcher spellings and the entire bin subtree are input-domain

members; mutable outputs and diagnostic/summary files are deliberately outside.

"""

import hashlib

import json

import re

import stat

from pathlib import Path, PurePosixPath



RECEIPT = 'build_contract_receipt.json'

SCHEMA = 'ppf.build_contract.v1'

DOMAIN = 'ppf.solver-inputs.v1'

FORMATS_IDENTITY = 'e70f0a681c30724a'

UNIFIED_EXE_SHA256 = 'f04bf8368082329acbe995c37bcf8211b639d9682444d2931ef11da3a5fe44c5'

TOP_LEVEL = frozenset({'info.toml', 'param.toml', 'dyn_param.txt', 'map.pickle', 'command.bat', 'command.sh'})

REQUIRED = frozenset({'info.toml', 'param.toml', 'map.pickle', 'bin/displacement.bin',

                      'bin/vert_dmap.bin', 'bin/vert.bin', 'bin/vel.bin', 'bin/tri.bin'})

PDRD_REQUIRED = frozenset({'bin/uv.bin', 'bin/pdrd_body.bin', 'bin/pdrd_vert_index.bin', 'bin/pdrd_vert_list.bin', 'bin/pdrd_rest_centered.bin', 'bin/pdrd_ball_anchor.cbor'})

CONTEXT = 'bin/pdrd_scene_context.cbor'

IDENTITY_KEYS = frozenset({'schema', 'product', 'product_root', 'exe_path', 'dll_path', 'exe_sha256', 'dll_sha256', 'format_identity', 'producer_version', 'capabilities', 'capability_snapshot'})

RECEIPT_KEYS = IDENTITY_KEYS | {'inventory_domain', 'context_sha256', 'scene_uuid', 'scene_inputs_sha256', 'required_features', 'inputs'}

HEX = re.compile(r'^[0-9a-f]{64}$')

UUID = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')





def fail(detail):

    raise ValueError('BUILD_CONTRACT_RECEIPT_INVALID: ' + detail + '; Build again')





def _regular(path):

    metadata = path.lstat()

    if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, 'st_file_attributes', 0) & 0x400:

        fail('symlink/reparse input ' + str(path))

    return stat.S_ISREG(metadata.st_mode)





def _safe_relative(value):

    return (isinstance(value, str) and bool(value) and '\\' not in value and ':' not in value

            and not value.startswith('/') and PurePosixPath(value).as_posix() == value

            and all(part not in ('', '.', '..') for part in value.split('/')))





def scan_inputs(path):

    root = Path(path)

    inputs = []

    def append(file):

        if not _regular(file):

            fail('input is not a regular file ' + str(file))

        data = file.read_bytes()

        inputs.append({'relative_path': file.relative_to(root).as_posix(),

                       'byte_length': len(data), 'sha256': hashlib.sha256(data).hexdigest()})

    for name in sorted(TOP_LEVEL):

        file = root / name

        if file.exists() or file.is_symlink():

            append(file)

    folder = root / 'bin'

    if folder.exists() or folder.is_symlink():

        if _regular(folder) or not folder.is_dir():

            fail('bin is not a directory')

        def walk(directory):

            for file in sorted(directory.iterdir()):

                if _regular(file):

                    append(file)

                elif file.is_dir():

                    walk(file)

                else:

                    fail('nonregular input ' + str(file))

        walk(folder)

    return sorted(inputs, key=lambda row: row['relative_path'])





def inventory_digest(rows):

    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()





def contract_identity(profile, binary_identity, format_identity=None):

    if profile.local_source:
        import importlib.util
        import sys
        # Reuse the package profile so App.set_backend's in-process choice is
        # preserved. Standalone launchers instead receive the frozen backend
        # in their own child environment and load trusted sibling code.
        profiles = sys.modules.get((__package__ or '')+'._r14_product_profile')
        if profiles is None:
            name = '_ppf_local_receipt_profile'
            spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name('_r14_product_profile.py'))
            profiles = importlib.util.module_from_spec(spec)
            sys.modules[name] = profiles
            spec.loader.exec_module(profiles)
        local_profile = profiles.load_profile(profile.product_root)
        actual_identity = profiles.verify_binaries(local_profile)
        if actual_identity != binary_identity:
            fail('local binary identity changed')
        actual = profiles.verify_local_capabilities(local_profile)
        if format_identity is not None and format_identity != actual['format_identity']:
            fail('local format identity changed')
        docs = actual['capabilities']
        local=profiles.local_source_module();doc=local.load(profile.product_root)
        if doc['schema']==local.RELATIVE_SCHEMA:
            features=local.contract_features(doc,actual)
            snapshot=dict(docs['bone']) if 'bone_connected_collision_v1' in features else {}
            if 'fem_prescribed_roots_v1' in features:snapshot['fem_body_controls']=docs['fem']
            if 'fem_root_control_events_v1' in features:snapshot['fem_body_control_events']=docs['events']
            return {'schema':SCHEMA,'product':profile.product_name,'product_root':str(profile.product_root),
                    'exe_path':str(profile.exe_path),'dll_path':str(profile.dll_path),
                    'exe_sha256':binary_identity['exe_sha256'],'dll_sha256':binary_identity['dll_sha256'],
                    'format_identity':actual['format_identity'],'producer_version':3 if 'pdrd_kinematic_roots_v3' in features else 2,
                    'capabilities':list(features),'capability_snapshot':snapshot}
        return {'schema': SCHEMA, 'product': profile.product_name, 'product_root': str(profile.product_root),
                'exe_path': str(profile.exe_path), 'dll_path': str(profile.dll_path),
                'exe_sha256': binary_identity['exe_sha256'], 'dll_sha256': binary_identity['dll_sha256'],
                'format_identity': actual['format_identity'], 'producer_version': 3,
                'capabilities': ['bone_connected_collision_v1','fem_prescribed_roots_v1','fem_root_control_events_v1','pdrd_kinematic_roots_v3'],
                'capability_snapshot': {**docs['bone'], 'fem_body_controls': docs['fem'], 'fem_body_control_events': docs['events']}}
    if format_identity is None:
        format_identity = FORMATS_IDENTITY
    unified = profile.product_name == 'R17 Unified Candidate BJ' and binary_identity['exe_sha256'] == UNIFIED_EXE_SHA256

    capabilities = ['bone_connected_collision_v1', 'pdrd_kinematic_roots_v3'] if unified else []

    snapshot = {'kind': 'ppf.bone_connected_collision.capability', 'version': 1,

                'required_feature': 'bone_connected_collision_v1', 'supported': True, 'semantic_version': 1} if unified else {}

    if profile.product_name == 'R17 FEM Root Candidate':
        # This module is also loaded as a standalone trusted launcher sibling.
        # Keep the identity constructor stdlib-only: no frontend/package import.
        expected_exe='3617a596e006c479a02b206de2dc2222bf05515bf2f7c54b7cc01fb429b7545a'
        expected_dll='794e059be9c125082a79313711a44b5ad39e45bbaa508bf405f266ebaad0492c'
        CAPABILITY_DOCUMENT={'kind':'ppf.fem_body_controls.capability','version':1,'required_feature':'fem_prescribed_roots_v1','supported':True,'semantic_version':1,'modes':['STATIC','KINEMATIC'],'scope':'initial_only','midrun_switching':False,'requires_matching_native_kernel_build':True,'supported_backends':['cuda'],'backend':'cuda'}
        if binary_identity['exe_sha256'] != expected_exe or binary_identity['dll_sha256'] != expected_dll:
            fail('FEM root binary pair not frozen')
        unified=True
        capabilities=['bone_connected_collision_v1','fem_prescribed_roots_v1','pdrd_kinematic_roots_v3']
        snapshot={'kind':'ppf.bone_connected_collision.capability','version':1,'required_feature':'bone_connected_collision_v1','supported':True,'semantic_version':1,'fem_body_controls':CAPABILITY_DOCUMENT}
    if profile.product_name in ('R17 FEM ABS Events Candidate', 'R17 Bone Resume Candidate', 'R17 Bone Rewind Candidate'):
        # Pending actual new event native pair; never authorize the old v1 pair.
        expected_exe = {
            'R17 FEM ABS Events Candidate': 'cda0527b29b098954d0f1144672c223644c8a3ad6f07991940c6d512d8f28ed5',
            'R17 Bone Resume Candidate': '4bbb557ff3f02656ac693b67ffa902b13dd33ff40ad9186661ff491188b2ea9a',
            'R17 Bone Rewind Candidate': '00b996d0aa513da051e05a9ea5d4f4efbc1eadfceb135781e0c214e0ced3e28d',
        }[profile.product_name]
        expected_dll='8a4df3f8b964dfd817221f9a8c1d11c2cd0e6fa7a434acf22b6843c6dbe63b5c'
        if not expected_exe or not expected_dll or binary_identity['exe_sha256']!=expected_exe or binary_identity['dll_sha256']!=expected_dll:
            fail('FEM event binary pair not frozen')
        unified=True
        capabilities=['bone_connected_collision_v1','fem_prescribed_roots_v1','fem_root_control_events_v1','pdrd_kinematic_roots_v3']
        initial={'kind':'ppf.fem_body_controls.capability','version':1,'required_feature':'fem_prescribed_roots_v1','supported':True,'semantic_version':1,'modes':['STATIC','KINEMATIC'],'scope':'initial_only','midrun_switching':False,'requires_matching_native_kernel_build':True,'supported_backends':['cuda'],'backend':'cuda'}
        events={'kind':'ppf.fem_body_control_events.capability','version':1,'semantic_version':2,'required_feature':'fem_root_control_events_v1','supported':True,'supported_backends':['cuda'],'backend':'cuda','scope':'fem_roots_only','modes':['FREE','STATIC','KINEMATIC'],'midrun_switching':True,'supported_event_pose_bindings':['current_hold','current_follow','absolute'],'resume_supported':False,'event_clock':'accepted_physical_boundary_v1','requires_matching_native_kernel_build':True}
        snapshot={'kind':'ppf.bone_connected_collision.capability','version':1,'required_feature':'bone_connected_collision_v1','supported':True,'semantic_version':1,'fem_body_controls':initial,'fem_body_control_events':events}
    return {'schema': SCHEMA, 'product': profile.product_name, 'product_root': str(profile.product_root),

            'exe_path': str(profile.exe_path), 'dll_path': str(profile.dll_path),

            'exe_sha256': binary_identity['exe_sha256'], 'dll_sha256': binary_identity['dll_sha256'],

            'format_identity': format_identity, 'producer_version': 3 if unified else 2,

            'capabilities': capabilities, 'capability_snapshot': snapshot}





def _unique_object(pairs):

    result = {}

    for key, value in pairs:

        if key in result:

            fail('duplicate JSON field ' + key)

        result[key] = value

    return result





def _context_document(blob):
    """Read the restoring-only context with stdlib; native owns full schema.

    Keep this reader out of the existing nonempty PDRD path. Limits apply
    before allocation, and duplicate maps/trailing data never gain authority.
    """
    import math
    import struct
    if len(blob) > 32 * 1024 * 1024:
        fail('restoring context byte budget')
    offset, remaining = 0, 1000000
    def take(count):
        nonlocal offset
        if count < 0 or count > len(blob) - offset:
            fail('truncated restoring context')
        value = blob[offset:offset + count]
        offset += count
        return value
    def read(depth=0):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 32:
            fail('restoring context structural budget')
        first = take(1)[0]
        major, ai = first >> 5, first & 31
        if major == 7:
            if ai in (20, 21): return ai == 21
            if ai == 22: return None
            if ai in (25, 26, 27):
                value = struct.unpack({25:'>e', 26:'>f', 27:'>d'}[ai], take({25:2,26:4,27:8}[ai]))[0]
                if not math.isfinite(value): fail('nonfinite restoring context value')
                return value
            fail('unsupported restoring context simple value')
        if ai < 24: count = ai
        elif ai in (24,25,26,27): count = int.from_bytes(take({24:1,25:2,26:4,27:8}[ai]),'big')
        else: fail('indefinite restoring context item')
        if major == 0: return count
        if major == 1: return -1-count
        if major == 2: return take(count)
        if major == 3:
            try: return take(count).decode('utf-8')
            except UnicodeDecodeError: fail('invalid restoring context UTF-8')
        if major == 4:
            if count > remaining: fail('restoring context array budget')
            return [read(depth+1) for _ in range(count)]
        if major == 5:
            if count > remaining//2: fail('restoring context map budget')
            result = {}
            for _ in range(count):
                key = read(depth+1)
                if not isinstance(key,str) or key in result: fail('duplicate/nontext restoring context key')
                result[key] = read(depth+1)
            return result
        fail('unsupported restoring context type')
    document = read()
    if offset != len(blob): fail('trailing restoring context bytes')
    return document


def _validate_restoring_only_context(path, receipt, mapping):
    context_path = Path(path)/CONTEXT
    if context_path.stat().st_size > 32*1024*1024: fail('restoring context byte budget')
    context = _context_document(context_path.read_bytes())
    if (not isinstance(context,dict) or set(context) != {'version','kind','payload'}
            or type(context['version']) is not int or context['version'] != 2
            or context['kind'] != 'ppf.pdrd_scene_context'):
        fail('invalid restoring-only context envelope')
    context = context['payload']
    if (not isinstance(context,dict) or type(context.get('context_version')) is not int
            or context['context_version'] != 2 or type(context.get('builder_contract_version')) is not int
            or context['builder_contract_version'] != 2 or context.get('complete') is not True
            or context.get('bodies') != [] or type(context.get('bodies')) is not list
            or context.get('scene_uuid') != receipt['scene_uuid']
            or context.get('scene_inputs_sha256') != receipt['scene_inputs_sha256']):
        fail('invalid restoring-only context binding')
    marker = context.get('restoring_curves_v1')
    relative = 'bin/restoring_curves_v1.json'
    row = mapping.get(relative)
    if (not isinstance(marker,dict) or set(marker) != {'required_feature','sidecar_sha256','sidecar_byte_length','combined_inputs_sha256'}
            or marker.get('required_feature') != 'restoring_curves_v1' or row is None
            or type(marker.get('sidecar_byte_length')) is not int or marker['sidecar_byte_length'] <= 0
            or marker['sidecar_byte_length'] != row['byte_length']
            or marker.get('sidecar_sha256') != row['sha256']
            or marker.get('combined_inputs_sha256') != hashlib.sha256((receipt['scene_inputs_sha256']+row['sha256']).encode('ascii')).hexdigest()):
        fail('invalid restoring-only sidecar seal')
    if row['byte_length'] > 32*1024*1024: fail('restoring sidecar byte budget')
    try:
        sidecar = json.loads((Path(path)/relative).read_text(encoding='utf-8'),object_pairs_hook=_unique_object,
                             parse_constant=lambda value: fail('nonfinite restoring JSON constant'))
    except (UnicodeError,json.JSONDecodeError,RecursionError): fail('invalid restoring sidecar JSON')
    if (not isinstance(sidecar,dict) or sidecar.get('kind') != 'ppf.restoring_curves'
            or type(sidecar.get('version')) is not int or sidecar['version'] != 1
            or sidecar.get('required_feature') != 'restoring_curves_v1'
            or sidecar.get('scene_uuid') != receipt['scene_uuid']
            or sidecar.get('basis_sha256') != receipt['scene_inputs_sha256']
            or type(sidecar.get('bindings')) is not list or not sidecar['bindings']
            or type(sidecar.get('tables')) is not list or not sidecar['tables']
            or type(sidecar.get('mass_receipts')) is not list):
        fail('invalid restoring-only sidecar binding')


def _validate_context_binding(path,receipt,mapping,names):
    if (receipt['context_sha256'] != mapping[CONTEXT]['sha256']
            or not isinstance(receipt['scene_uuid'],str) or not UUID.fullmatch(receipt['scene_uuid'])):
        fail('invalid PDRD seal binding')
    if not PDRD_REQUIRED.issubset(names):
        # Ordinary FEM/stitch restoring curves also carry a scene context.
        # Any dedicated PDRD input keeps the historical six-file gate.
        if (PDRD_REQUIRED-{'bin/uv.bin'}).intersection(names):
            fail('invalid PDRD seal binding')
        _validate_restoring_only_context(path,receipt,mapping)


def validate_receipt(path, expected_contract, *, resume_exe_rebind=False, resume_source_stamps=()):

    if not isinstance(expected_contract, dict) or set(expected_contract) != IDENTITY_KEYS:

        fail('expected identity fields are incomplete')

    receipt_path = Path(path) / RECEIPT

    if not _regular(receipt_path):

        fail('receipt is not a regular file')

    try:

        receipt = json.loads(receipt_path.read_text(encoding='utf-8'), object_pairs_hook=_unique_object)

    except (UnicodeError, json.JSONDecodeError) as exc:

        fail('malformed JSON: ' + str(exc))

    if not isinstance(receipt, dict) or set(receipt) != RECEIPT_KEYS:

        fail('receipt fields are incomplete or unknown')

    if receipt['schema'] != SCHEMA or receipt['inventory_domain'] != DOMAIN:

        fail('schema/inventory domain mismatch')

    if (type(receipt['producer_version']) is not int or type(receipt['capabilities']) is not list

            or receipt['capabilities'] != sorted(set(receipt['capabilities']))

            or type(receipt['capability_snapshot']) is not dict):

        fail('malformed producer/capability contract')

    # JSON bool/int compare equal in Python, so snapshot types are checked too.

    snapshot = receipt['capability_snapshot']

    if snapshot and (type(snapshot.get('version')) is not int or type(snapshot.get('semantic_version')) is not int or type(snapshot.get('supported')) is not bool):

        fail('malformed capability snapshot types')

    # Only the specifically tested legacy source stamp may move to the
    # selected executable's actual current reader stamp. A capability claim
    # is required from that same selected executable, and does not waive the
    # remaining receipt, input, or native checkpoint validations.
    legacy_stamp = 'd45426411338a803'
    old_stamp = receipt.get('format_identity')
    current_stamp = expected_contract.get('format_identity')
    known_stamp_upgrade = (resume_exe_rebind is True and old_stamp == legacy_stamp
        and isinstance(current_stamp, str) and re.fullmatch(r'[0-9a-f]{16}', current_stamp)
        and isinstance(resume_source_stamps, list)
        and all(isinstance(stamp, str) and re.fullmatch(r'[0-9a-f]{16}', stamp) for stamp in resume_source_stamps)
        and len(resume_source_stamps) == len(set(resume_source_stamps))
        and old_stamp in resume_source_stamps and current_stamp in resume_source_stamps)

    for key in IDENTITY_KEYS:

        if receipt[key] != expected_contract[key]:

            if (key in ('exe_sha256', 'dll_sha256') and resume_exe_rebind is True
                    and (key == 'exe_sha256' or known_stamp_upgrade)
                    and isinstance(receipt[key], str) and HEX.fullmatch(receipt[key])
                    and isinstance(expected_contract[key], str) and HEX.fullmatch(expected_contract[key])):
                continue
            if key == 'format_identity' and known_stamp_upgrade:
                continue
            fail('frozen identity mismatch: ' + key)

    if not isinstance(receipt['scene_inputs_sha256'], str) or not HEX.fullmatch(receipt['scene_inputs_sha256']):

        fail('invalid scene input hash')

    features = receipt['required_features']

    if not isinstance(features, list) or any(not isinstance(item, str) or not item for item in features) or features != sorted(set(features)):

        fail('invalid required_features')

    if ('pdrd_kinematic_roots_v3' in features and receipt['producer_version'] != 3

            or 'bone_connected_collision_v1' in features and 'bone_connected_collision_v1' not in receipt['capabilities']
            or 'fem_prescribed_roots_v1' in features and 'fem_prescribed_roots_v1' not in receipt['capabilities']
            or 'fem_root_control_events_v1' in features and 'fem_root_control_events_v1' not in receipt['capabilities']):

        fail('unsupported required_features')

    rows = receipt['inputs']

    if not isinstance(rows, list) or not rows:

        fail('missing/nonempty input inventory required')

    names = []

    for row in rows:

        if (not isinstance(row, dict) or set(row) != {'relative_path', 'byte_length', 'sha256'}

                or not _safe_relative(row['relative_path']) or type(row['byte_length']) is not int

                or row['byte_length'] < 0 or not isinstance(row['sha256'], str) or not HEX.fullmatch(row['sha256'])):

            fail('malformed input inventory entry')

        relative = row['relative_path']

        if relative not in TOP_LEVEL and not relative.startswith('bin/'):

            fail('input outside solver domain')

        names.append(relative)

    if names != sorted(set(names)) or len({name.casefold() for name in names}) != len(names):

        fail('noncanonical input inventory order/alias')

    if not REQUIRED.issubset(names):

        fail('required solver inputs missing')

    if rows != scan_inputs(path):

        fail('actual solver input inventory/bytes changed')

    mapping = {row['relative_path']: row for row in rows}

    if CONTEXT in mapping:

        _validate_context_binding(path,receipt,mapping,names)

    elif (receipt['context_sha256'] is not None or receipt['scene_uuid'] is not None

          or features or receipt['scene_inputs_sha256'] != inventory_digest(rows)):

        fail('invalid ordinary scene inventory binding')

    return receipt

