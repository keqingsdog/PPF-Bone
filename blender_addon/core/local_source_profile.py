"""Explicit local CUDA source products. Stdlib only; never edits parent env.

A local configuration binds artifacts, not capability claims: admission probes
the actual executable with its pinned backend before Build or standalone Run.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

SCHEMA = 'ppf.local-source-product.v1'
RELATIVE_SCHEMA = 'ppf.local-source-product.v2'
LABEL = 'Local Source Build'
CONFIG = 'LOCAL_SOURCE_BUILD.json'
POLICY = 'LOCAL_SOURCE_POLICY.json'
HEX = re.compile(r'^[0-9a-f]{64}$')
PINS = {'PPF_PRECOND': 'block-jacobi', 'PPF_ADAPTIVE_CG_RETRY': '1'}
CAPABILITY_ARGS = {
    'restoring': '--restoring-curves-capability-json',
    'bone': '--bone-connected-collision-capability-json',
    'fem': '--fem-body-controls-capability-json',
    'events': '--fem-body-control-events-capability-json',
    'resume': '--resume-capability-json',
}

def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _pairs(rows):
    result = {}
    for key, value in rows:
        if key in result:
            raise ValueError('LOCAL_SOURCE_DUPLICATE_JSON_KEY: ' + key)
        result[key] = value
    return result

def read_json(path):
    data = Path(path).read_bytes()
    if len(data) > 1024 * 1024:
        raise ValueError('LOCAL_SOURCE_CONFIG_TOO_LARGE')
    return json.loads(data.decode('utf-8'), object_pairs_hook=_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

def relative_file(root, value):
    if (not isinstance(value, str) or not value or '\\' in value or ':' in value
            or PurePosixPath(value).is_absolute()
            or any(part in ('', '.', '..') for part in value.split('/'))):
        raise ValueError('LOCAL_SOURCE_UNSAFE_RELATIVE_PATH')
    path = (root / value).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('LOCAL_SOURCE_RUNTIME_ESCAPE_OR_MISSING')
    return path

def has_config(root):
    return (Path(root) / CONFIG).is_file()

def load(root):
    selected = Path(root).resolve(strict=True)
    doc = read_json(selected / CONFIG)
    if doc.get('schema') == RELATIVE_SCHEMA:
        return load_relative(selected, doc)
    if doc.get('schema') != SCHEMA or doc.get('product') != LABEL:
        raise ValueError('LOCAL_SOURCE_SCHEMA_OR_LABEL_INVALID')
    required = {'schema','product','platform','backend','source_root','build_dir','product_root','python','cuda_bin','library_dirs','exe','backend_library','source_identity','exe_sha256','dll_sha256','policy_sha256','format_identity','probe_capabilities'}
    if set(doc) != required:
        raise ValueError('LOCAL_SOURCE_CONFIG_FIELDS_INVALID')
    platform = doc.get('platform')
    if platform not in ('win32', 'linux') or doc.get('backend') != 'cuda':
        raise ValueError('LOCAL_SOURCE_PLATFORM_BACKEND_UNSUPPORTED')
    if platform != sys.platform:
        raise ValueError('LOCAL_SOURCE_PLATFORM_MISMATCH')
    for field in ('product_root', 'source_root', 'build_dir', 'python', 'cuda_bin'):
        value = doc.get(field)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError('LOCAL_SOURCE_ABSOLUTE_PATH_REQUIRED: ' + field)
        if not Path(value).exists():
            raise ValueError('LOCAL_SOURCE_PATH_MISSING: ' + field)
    product = Path(doc['product_root']).resolve(strict=True)
    source = Path(doc['source_root']).resolve(strict=True)
    if selected not in (product, source):
        raise ValueError('LOCAL_SOURCE_SELECTION_ROOT_MISMATCH')
    if selected != product and read_json(product / CONFIG) != doc:
        raise ValueError('LOCAL_SOURCE_SELECTION_CHANGED')
    exe = relative_file(product, doc.get('exe'))
    dll = relative_file(product, doc.get('backend_library'))
    names = ('ppf-contact-solver.exe', 'libsimbackend_cuda.dll') if platform == 'win32' else ('ppf-contact-solver', 'libppfbe_cuda.so')
    if (exe.name, dll.name) != names or exe.parent != dll.parent:
        raise ValueError('LOCAL_SOURCE_RUNTIME_NAMES_INVALID')
    if not isinstance(doc['library_dirs'], list) or any(not isinstance(value, str) or not Path(value).is_absolute() or not Path(value).is_dir() for value in doc['library_dirs']):
        raise ValueError('LOCAL_SOURCE_LIBRARY_DIR_INVALID')
    if not Path(doc['cuda_bin']).is_dir() or not Path(doc['source_root']).is_dir() or not Path(doc['build_dir']).is_dir():
        raise ValueError('LOCAL_SOURCE_DIRECTORY_REQUIRED')
    if not Path(doc['python']).is_file():
        raise ValueError('LOCAL_SOURCE_PYTHON_MISSING')
    for field in ('exe_sha256', 'dll_sha256', 'policy_sha256'):
        if not isinstance(doc.get(field), str) or not HEX.fullmatch(doc[field]):
            raise ValueError('LOCAL_SOURCE_PIN_INVALID: ' + field)
    identity = doc.get('source_identity', {})
    if any(type(identity.get(key)) is not int for key in ('ordered_entry_count','contract_version','channel_mask')) or identity.get('ordered_entry_count') != 366 or identity.get('contract_version') != 1 or identity.get('channel_mask') != 15:
        raise ValueError('LOCAL_SOURCE_GENERATED_IDENTITY_INVALID')
    for field in ('source_stamp', 'layout_sha256', 'kernel_table_sha256'):
        if not HEX.fullmatch(str(identity.get(field, ''))):
            raise ValueError('LOCAL_SOURCE_GENERATED_IDENTITY_INVALID: ' + field)
    if not re.fullmatch('[0-9a-f]{16}', str(doc.get('format_identity', ''))):
        raise ValueError('LOCAL_SOURCE_FORMAT_IDENTITY_INVALID')
    return doc

def artifact_library_name(backend, platform=None):
    platform = sys.platform if platform is None else platform
    if backend == 'cpu':
        return '_ppf_cts_py.dll' if platform == 'win32' else 'lib_ppf_cts_py' + ('.dylib' if platform == 'darwin' else '.so')
    if platform == 'win32':return 'libsimbackend_' + backend + '.dll'
    return 'libppfbe_' + backend + ('.dylib' if platform == 'darwin' else '.so')


def load_relative(root, raw):
    required = {'schema','scope','product','platform','backend','product_root','source_root','build_dir','python','cuda_bin','library_dirs','exe','backend_library'}
    if set(raw) not in (required, required | {'backend_routes'}) or raw['scope'] != 'source' or raw['product'] != LABEL:
        raise ValueError('LOCAL_SOURCE_RELATIVE_FIELDS_INVALID')
    if raw['platform'] != sys.platform or raw['backend'] not in ('cpu','cuda','rocm','metal'):
        raise ValueError('LOCAL_SOURCE_PLATFORM_BACKEND_UNSUPPORTED')
    def resolve(value, kind):
        if value == '.' and kind == 'root':
            return root
        if not isinstance(value,str) or not value or '\\' in value or ':' in value or PurePosixPath(value).is_absolute() or any(p in ('','.','..') for p in value.split('/')):
            raise ValueError('LOCAL_SOURCE_UNSAFE_RELATIVE_PATH')
        path=(root/value).resolve(strict=True)
        if not path.is_relative_to(root) or not (path.is_file() if kind == 'file' else path.is_dir()):
            raise ValueError('LOCAL_SOURCE_RUNTIME_ESCAPE_OR_MISSING')
        return path
    raw=dict(raw)
    # Source profiles follow the chosen official artifact, never switch CPU to CUDA.
    front=sys.modules.get(__name__.split('.')[0])
    choice=getattr(front,'_BACKEND_CHOICE',None)
    selected_directory=None
    resolver=getattr(front,'_resolve',None)
    if callable(resolver):
        choice,selected_directory=resolver(front._run_profile())
    backend=(choice or os.environ.get('PPF_BACKEND',raw['backend'])).lower()
    if backend not in ('cpu','cuda','rocm','metal'):raise ValueError('LOCAL_SOURCE_BACKEND_SELECTION_INVALID')
    routes=raw.get('backend_routes')
    if routes is not None:
        if not isinstance(routes,dict) or not routes or any(k not in ('cpu','cuda','rocm','metal') or not isinstance(v,dict) or set(v) != {'build_dir','exe','backend_library'} for k,v in routes.items()):
            raise ValueError('LOCAL_SOURCE_BACKEND_ROUTES_INVALID')
        if backend not in routes:
            raise ValueError('LOCAL_SOURCE_BACKEND_ROUTE_MISSING: ' + backend)
        raw.update(routes[backend]); raw['backend']=backend
    elif backend != raw['backend']:
        raise ValueError('LOCAL_SOURCE_BACKEND_ROUTE_MISSING: ' + backend)
    if selected_directory is not None:
        directory=Path(selected_directory).resolve(strict=True)
        if not directory.is_relative_to(root):raise ValueError('LOCAL_SOURCE_SELECTED_ARTIFACT_OUTSIDE_ROOT')
        if directory != resolve(raw['build_dir'],'dir'):
            raise ValueError('LOCAL_SOURCE_SELECTED_ROUTE_MISMATCH')
    doc=dict(raw)
    for key in ('product_root','source_root'):
        if raw[key] != '.':raise ValueError('LOCAL_SOURCE_RELATIVE_ROOT_REQUIRED')
        doc[key]=str(resolve(raw[key],'root'))
    for key in ('build_dir','cuda_bin'):
        doc[key]=str(resolve(raw[key],'dir'))
    python_target = resolve(raw['python'], 'file')  # Enforce real-target closure first.
    # POSIX interpreter entry paths carry venv identity; resolving an internal
    # symlink here would launch the base interpreter instead of that entry.
    doc['python'] = str(root / raw['python'] if raw['platform'] in ('linux', 'darwin') else python_target)
    if not isinstance(raw['library_dirs'],list):raise ValueError('LOCAL_SOURCE_LIBRARY_DIR_INVALID')
    doc['library_dirs']=[str(resolve(v,'dir')) for v in raw['library_dirs']]
    exe=resolve(raw['exe'],'file');dll=resolve(raw['backend_library'],'file')
    if exe.name != ('ppf-contact-solver.exe' if sys.platform == 'win32' else 'ppf-contact-solver') or dll.name != artifact_library_name(backend):
        raise ValueError('LOCAL_SOURCE_RUNTIME_NAMES_INVALID')
    if exe.parent != Path(doc['build_dir']) or dll.parent not in (exe.parent, root/'bin'):
        raise ValueError('LOCAL_SOURCE_RUNTIME_NAMES_INVALID')
    return doc


def verify(doc, unset):
    root = Path(doc['product_root']).resolve(strict=True)
    current = load(root)
    if current != doc:
        raise ValueError('LOCAL_SOURCE_CONFIG_CHANGED')
    exe, dll = relative_file(root, doc['exe']), relative_file(root, doc['backend_library'])
    if doc['schema'] == RELATIVE_SCHEMA:
        # Only a real extension contract calls verify: its receipt must detect
        # replacement of these two engine artifacts after Build. No tree/policy hash.
        return {'product':LABEL,'entry_count':366,'exe_path':str(exe),'dll_path':str(dll),
                'exe_sha256':sha256(exe),'dll_sha256':sha256(dll),'environment_policy_sha256':None}
    policy = read_json(root / POLICY)
    expected_policy = {'schema': 'ppf.local-source-policy.v1', 'product': LABEL,
                       'unset': sorted(unset), 'set': PINS}
    if policy != expected_policy or sha256(root / POLICY) != doc['policy_sha256']:
        raise ValueError('LOCAL_SOURCE_POLICY_CHANGED')
    if sha256(exe) != doc['exe_sha256'] or sha256(dll) != doc['dll_sha256']:
        raise ValueError('LOCAL_SOURCE_BINARY_PAIR_CHANGED')
    return {'product': LABEL, 'entry_count': 366, 'exe_path': str(exe), 'dll_path': str(dll),
            'exe_sha256': doc['exe_sha256'], 'dll_sha256': doc['dll_sha256'],
            'environment_policy_sha256': doc['policy_sha256']}

def child_env(doc, base_env, unset):
    windows = doc['platform'] == 'win32'
    managed = set(unset) | set(PINS) | {'PATH', 'CUDA_PATH', 'CARGO_TARGET_DIR'}
    key_of = str.upper if windows else (lambda value: value)
    env = {key: value for key, value in base_env.items() if key_of(key) not in managed}
    old_path = next((value for key, value in base_env.items() if key_of(key) == 'PATH'), '')
    root = Path(doc['product_root']).resolve()
    runtime = relative_file(root, doc['exe']).parent
    separator = ';' if windows else ':'
    env.update(PINS)
    if doc['schema'] == RELATIVE_SCHEMA:env['PPF_BACKEND']=doc['backend']
    env['PATH'] = separator.join([str(runtime), doc['cuda_bin']] + ([old_path] if old_path else []))
    env['CUDA_PATH'] = str(Path(doc['cuda_bin']).parent)
    env['CARGO_TARGET_DIR'] = str(Path(doc['build_dir']).parent)
    if not windows:
        library_key = 'DYLD_LIBRARY_PATH' if doc['platform'] == 'darwin' else 'LD_LIBRARY_PATH'
        env[library_key] = ':'.join([str(runtime), str(relative_file(root, doc['backend_library']).parent), *doc.get('library_dirs', [])] + ([env[library_key]] if env.get(library_key) else []))
    return env

def probe(doc, env):
    """No config capability field is an authority; ask the pinned executable."""
    root = Path(doc['product_root']).resolve()
    exe = relative_file(root, doc['exe'])
    def run(flag):
        result = subprocess.run([str(exe), flag], cwd=str(exe.parent), env=env,
                                capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise ValueError('LOCAL_SOURCE_PROBE_FAILED: ' + flag + ': ' + (result.stdout + result.stderr)[-4000:])
        return result.stdout.strip()
    backend = doc['backend']
    backend_text = run('--backend')
    lines = [line.strip().lower() for line in backend_text.splitlines() if line.strip()]
    if not lines or any(line not in (backend, 'backend: '+backend, 'linked: '+backend) for line in lines):
        raise ValueError('LOCAL_SOURCE_BACKEND_DISAGREEMENT: ' + backend_text)
    probe_text = run('--probe')
    fields = {}
    for line in probe_text.splitlines():
        key, separator, value = line.partition(': ')
        if separator:
            if key in fields and fields[key] != value:
                raise ValueError('LOCAL_SOURCE_PROBE_CONTRADICTORY_FIELDS')
            fields[key] = value
    if fields.get('backend') != backend or (backend != 'cpu' and fields.get('linked') != backend) or not fields.get('device'):
        raise ValueError('LOCAL_SOURCE_DRIVER_PROBE_INVALID')
    stamp = fields.get('stamp')
    if not re.fullmatch('[0-9a-f]{16}', stamp or ''):
        raise ValueError('LOCAL_SOURCE_FORMAT_PROBE_INVALID')
    docs = {key: json.loads(run(flag), object_pairs_hook=_pairs) for key, flag in CAPABILITY_ARGS.items()}
    if doc['schema'] == RELATIVE_SCHEMA:
        restoring=docs['restoring']
        expected={'kind':'ppf.restoring_curves.capability','version':1,'capability':'restoring_curves_v1','contract_version':1,'channel_mask':15,'backend':backend}
        if any(restoring.get(k)!=v for k,v in expected.items()) or type(restoring.get('abi_version')) is not int or restoring['abi_version'] <= 0:
            raise ValueError('LOCAL_SOURCE_RESTORING_ABI_IDENTITY_MISMATCH')
        for key,kind,feature in [('bone','ppf.bone_connected_collision.capability','bone_connected_collision_v1'),('fem','ppf.fem_body_controls.capability','fem_prescribed_roots_v1'),('events','ppf.fem_body_control_events.capability','fem_root_control_events_v1')]:
            value=docs[key]
            if value.get('kind')!=kind or type(value.get('version')) is not int or value['version']!=1 or value.get('required_feature')!=feature or type(value.get('supported')) is not bool:
                raise ValueError('LOCAL_SOURCE_CAPABILITY_DOCUMENT_INVALID: '+key)
            if key!='bone' and value.get('backend')!=backend:raise ValueError('LOCAL_SOURCE_CAPABILITY_BACKEND_MISMATCH')
        known={'bone': {'kind': 'ppf.bone_connected_collision.capability', 'version': 1, 'required_feature': 'bone_connected_collision_v1', 'supported': True, 'semantic_version': 1}, 'fem': {'kind': 'ppf.fem_body_controls.capability', 'version': 1, 'required_feature': 'fem_prescribed_roots_v1', 'supported': True, 'semantic_version': 1, 'modes': ['STATIC', 'KINEMATIC'], 'scope': 'initial_only', 'midrun_switching': False, 'requires_matching_native_kernel_build': True, 'supported_backends': ['cuda'], 'backend': 'cuda'}, 'events': {'kind': 'ppf.fem_body_control_events.capability', 'version': 1, 'semantic_version': 2, 'required_feature': 'fem_root_control_events_v1', 'supported': True, 'supported_backends': ['cuda'], 'backend': 'cuda', 'scope': 'fem_roots_only', 'modes': ['FREE', 'STATIC', 'KINEMATIC'], 'midrun_switching': True, 'supported_event_pose_bindings': ['current_hold', 'current_follow', 'absolute'], 'resume_supported': False, 'event_clock': 'accepted_physical_boundary_v1', 'requires_matching_native_kernel_build': True}}
        for key,expected_doc in known.items():
            if docs[key]['supported'] is True and (docs[key]!=expected_doc or type(docs[key].get('semantic_version')) is not int):
                raise ValueError('LOCAL_SOURCE_CAPABILITY_SEMANTICS_UNSUPPORTED: '+key)
        return {'format_identity':stamp,'capabilities':docs,'device':fields['device']}
    restoring = docs['restoring']
    expected = {'kind': 'ppf.restoring_curves.capability', 'version': 1,
                'capability': 'restoring_curves_v1', 'contract_version': 1,
                'channel_mask': 15, 'backend': 'cuda'}
    if doc['schema'] != RELATIVE_SCHEMA:
        expected.update({key: doc['source_identity'][key] for key in ('source_stamp', 'layout_sha256', 'kernel_table_sha256')})
    if any(restoring.get(key) != value for key, value in expected.items()) or type(restoring.get('abi_version')) is not int or restoring['abi_version'] <= 0:
        raise ValueError('LOCAL_SOURCE_RESTORING_ABI_IDENTITY_MISMATCH')
    checks = {
        'bone': ('ppf.bone_connected_collision.capability', 'bone_connected_collision_v1'),
        'fem': ('ppf.fem_body_controls.capability', 'fem_prescribed_roots_v1'),
        'events': ('ppf.fem_body_control_events.capability', 'fem_root_control_events_v1'),
    }
    for key, (kind, feature) in checks.items():
        value = docs[key]
        if value.get('kind') != kind or value.get('version') != 1 or value.get('required_feature') != feature or value.get('supported') is not True:
            raise ValueError('LOCAL_SOURCE_CAPABILITY_UNSUPPORTED: ' + key)
    bone_expected = {'kind':'ppf.bone_connected_collision.capability','version':1,
                     'required_feature':'bone_connected_collision_v1','supported':True,'semantic_version':1}
    fem_expected = {'kind':'ppf.fem_body_controls.capability','version':1,'required_feature':'fem_prescribed_roots_v1','supported':True,'semantic_version':1,'modes':['STATIC','KINEMATIC'],'scope':'initial_only','midrun_switching':False,'requires_matching_native_kernel_build':True,'supported_backends':['cuda'],'backend':'cuda'}
    events_expected = {'kind':'ppf.fem_body_control_events.capability','version':1,'semantic_version':2,'required_feature':'fem_root_control_events_v1','supported':True,'supported_backends':['cuda'],'backend':'cuda','scope':'fem_roots_only','modes':['FREE','STATIC','KINEMATIC'],'midrun_switching':True,'supported_event_pose_bindings':['current_hold','current_follow','absolute'],'resume_supported':False,'event_clock':'accepted_physical_boundary_v1','requires_matching_native_kernel_build':True}
    if docs['bone'] != bone_expected or docs['fem'] != fem_expected or docs['events'] != events_expected:
        raise ValueError('LOCAL_SOURCE_CAPABILITY_SEMANTICS_UNSUPPORTED')
    for key in ('bone','fem','events'):
        if type(docs[key]['version']) is not int or type(docs[key]['semantic_version']) is not int:
            raise ValueError('LOCAL_SOURCE_CAPABILITY_NUMERIC_TYPE_INVALID')
    if docs['events'].get('supported_event_pose_bindings') != ['current_hold', 'current_follow', 'absolute'] or docs['events'].get('resume_supported') is not False:
        raise ValueError('LOCAL_SOURCE_EVENT_SEMANTICS_UNSUPPORTED')
    resume = docs['resume']
    if resume.get('kind') != 'ppf.bone_resume_capability' or resume.get('version') != 1 or resume.get('feature') != 'bone_checkpoint_resume_v1' or resume.get('supported') is not True or resume.get('target_frame_override') is not True:
        raise ValueError('LOCAL_SOURCE_RESUME_UNSUPPORTED')
    return {'format_identity': stamp, 'capabilities': docs, 'device': fields['device']}

def verify_probes(profile, unset, *, already_verified=False):
    doc = load(profile.product_root)
    if not already_verified:
        verify(doc, unset)
    actual = probe(doc, child_env(doc, os.environ, unset))
    if doc['schema'] != RELATIVE_SCHEMA and (actual['format_identity'] != doc['format_identity'] or actual['capabilities'] != doc['probe_capabilities']):
        raise ValueError('LOCAL_SOURCE_FROZEN_PROBE_CHANGED')
    return actual


def contract_features(doc, actual):
    docs=actual['capabilities'];features=[]
    if docs['bone'].get('supported') is True:features.append('bone_connected_collision_v1')
    if docs['fem'].get('supported') is True:features.append('fem_prescribed_roots_v1')
    if docs['events'].get('supported') is True:features.append('fem_root_control_events_v1')
    # Retained v3 producer has CUDA implementation evidence; CPU bone probe
    # alone does not establish kinematic-root-v3 semantics.
    if doc['backend']=='cuda':features.append('pdrd_kinematic_roots_v3')
    return tuple(sorted(features))
