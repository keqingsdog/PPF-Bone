"""Fixed R14/R15/R17 product configuration; only reads and child-env copies.

Copy this file byte for byte into the addon core and frontend packages.
No solver probe, DLL import, process launch or parent-environment mutation.
"""

from dataclasses import dataclass
import hashlib
import json
import ntpath
import os
from pathlib import Path
from typing import Mapping


R14_PRODUCT_ROOT = None
DEFAULT_PRODUCT_ROOT = None
ENVIRONMENT_POLICY_SHA256 = "7383b589151e8331386e2c2c36927f3b583d84ed034d53e3e2b204855fc7a282"
EXE_SHA256 = "aae56122fb80df60732a08ff180bc48e956bbfb74cd76b466c19672a8ce4bcec"
DLL_SHA256 = "805d6c845aef55db0bd990bd75059f0b36e5ce213d196f09a1f790ea8911c76a"
_UNSET = frozenset({
    "PPF_ANALYTIC_GAP_CAPTURE_DIR",
    "PPF_ANALYTIC_STALL_TERMINAL_CAPTURE_DIR",
    "PPF_CONTACT_TRIAL_DT_EXPERIMENT", "PPF_DUMP_DIR", "PPF_DUMP_LINSYS",
    "PPF_HARD_COEFFICIENT_CAPTURE", "PPF_PCG_TRUE_RESIDUAL_CAPTURE_DIR",
    "PPF_PCG_TRUE_RESIDUAL_CAPTURE_STOP", "PPF_PCG_TRUE_RESIDUAL_DIAGNOSTICS",
    "PPF_PHASE_STATS", "PPF_RIGID_PCG_FAILURE_CAPTURE_DIR",
    "PPF_ROOT_ACCEPTANCE_RETRY_EXPERIMENT", "PPF_SAND_NO_ROLL",
    "PPF_SAND_OFFSET_CAPTURE", "PPF_SAND_OFFSET_CAPTURE_BEGIN",
    "PPF_SAND_OFFSET_CAPTURE_END", "PPF_SAND_ROLL_RESIST",
    "PPF_SAND_SPIN_COUPLE", "PPF_SCHWARZ_CONSTRAINT_CHECK_NEWTON",
    "PPF_SCHWARZ_FROZEN_CHECK", "PPF_SLOT_REPLAY", "PPF_SORT_VERIFY",
    "PPF_STEP_DELAY_MS", "PPF_TET_SPLIT",
})


@dataclass(frozen=True)
class ProductProfile:
    product_root: Path
    runtime_dir: Path
    exe_path: Path
    dll_path: Path
    cuda_bin: Path
    product_name: str = 'R14 BJ'
    local_source: bool = False
    python_path: Path | None = None
    source_root: Path | None = None
    build_dir: Path | None = None
    platform: str = 'win32'
    # Freeze the validated target independently of the POSIX entry spelling.
    python_target: Path | None = None


@dataclass(frozen=True)
class _Identity:
    root: str
    product_schema: str | None
    policy_schema: str
    policy_product: str
    policy_hash: str
    exe_hash: str
    dll_hash: str
    cuda_bin: str = ''


# Bind these three source pins only after main builds the matching isolated product.
FEM_ROOT_POLICY_SHA256 = '9004bda097a93be7f3c44d104a5ed20309a479d2180761f835cee9149018ee6e'
FEM_ROOT_EXE_SHA256 = '3617a596e006c479a02b206de2dc2222bf05515bf2f7c54b7cc01fb429b7545a'
FEM_ROOT_DLL_SHA256 = '794e059be9c125082a79313711a44b5ad39e45bbaa508bf405f266ebaad0492c'

# Event product pins remain unassigned until a new actual paired build exists.
FEM_ABS_EVENTS_POLICY_SHA256 = '18481cc9888bc99f8c310ce19bfb3c1146aa31d2dee88660f8ad11afcc367c64'
FEM_ABS_EVENTS_EXE_SHA256 = 'cda0527b29b098954d0f1144672c223644c8a3ad6f07991940c6d512d8f28ed5'
FEM_ABS_EVENTS_DLL_SHA256 = '8a4df3f8b964dfd817221f9a8c1d11c2cd0e6fa7a434acf22b6843c6dbe63b5c'

BONE_RESUME_POLICY_SHA256 = 'f501894ac49d0ece3ab2b46802c51775dc62ffcc675f9420e5cb3b4be71a825d'
BONE_RESUME_EXE_SHA256 = '4bbb557ff3f02656ac693b67ffa902b13dd33ff40ad9186661ff491188b2ea9a'
BONE_RESUME_DLL_SHA256 = '8a4df3f8b964dfd817221f9a8c1d11c2cd0e6fa7a434acf22b6843c6dbe63b5c'

BONE_REWIND_POLICY_SHA256 = '4b487ef51569e077b7777a72f388eccf73fd304424bcb5296dfcc555d022cee3'
BONE_REWIND_EXE_SHA256 = '00b996d0aa513da051e05a9ea5d4f4efbc1eadfceb135781e0c214e0ced3e28d'
BONE_REWIND_DLL_SHA256 = '8a4df3f8b964dfd817221f9a8c1d11c2cd0e6fa7a434acf22b6843c6dbe63b5c'

# Expectations come from reviewed source constants, never from product JSON.
_IDENTITIES = {
    'R17 Bone Rewind Candidate': _Identity(
        None,
        'ppf.r17-bone-rewind-candidate.product.v1',
        'ppf.r17-bone-rewind-candidate.solver-environment.v1', 'R17 Bone Rewind Candidate',
        BONE_REWIND_POLICY_SHA256, BONE_REWIND_EXE_SHA256, BONE_REWIND_DLL_SHA256),
    'R17 Bone Resume Candidate': _Identity(
        None,
        'ppf.r17-bone-resume-candidate.product.v1',
        'ppf.r17-bone-resume-candidate.solver-environment.v1', 'R17 Bone Resume Candidate',
        BONE_RESUME_POLICY_SHA256, BONE_RESUME_EXE_SHA256, BONE_RESUME_DLL_SHA256),
    'R17 FEM ABS Events Candidate': _Identity(
        None,
        'ppf.r17-fem-abs-events-candidate.product.v1',
        'ppf.r17-fem-abs-events-candidate.solver-environment.v1', 'R17 FEM ABS Events Candidate',
        FEM_ABS_EVENTS_POLICY_SHA256,FEM_ABS_EVENTS_EXE_SHA256,FEM_ABS_EVENTS_DLL_SHA256),
    'R17 FEM Root Candidate': _Identity(
        None,
        'ppf.r17-fem-root-candidate.product.v1',
        'ppf.r17-fem-root-candidate.solver-environment.v1', 'R17 FEM Root Candidate',
        FEM_ROOT_POLICY_SHA256, FEM_ROOT_EXE_SHA256, FEM_ROOT_DLL_SHA256),
    'R17 Unified Candidate BJ': _Identity(
        None, 'ppf.r17-unified-candidate.product.v1',
        'ppf.r17-unified-candidate.solver-environment.v1', 'R17 Unified Candidate BJ actual366',
        'c418a1e8b1f6f726f7d613d56a102030b81532574a10b8515ab7af73320f1646', 'f04bf8368082329acbe995c37bcf8211b639d9682444d2931ef11da3a5fe44c5', DLL_SHA256),
    'R15 Adaptive BJ': _Identity(
        None,
        'ppf.r15-adaptive-bj.product.v1', 'ppf.r15-adaptive-bj.solver-environment.v1',
        'R15 Adaptive BJ actual366', '01a4deba7a6a7420c4aea088f9f50480b7c2e2cad66369be2e1337e0f7919b94',
        '0491aaba26adaa5eafd6333199c2a8fd250fc91fc337dc9b120425ec33583122', DLL_SHA256),
    'R14 BJ': _Identity(
        None, None, 'ppf.r14-bj.solver-environment.v1',
        'R14 BJ actual366', ENVIRONMENT_POLICY_SHA256, EXE_SHA256, DLL_SHA256),
    'R17 Reference BJ': _Identity(
        None,
        'ppf.r17-reference-bj.product.v1',
        'ppf.r17-reference-bj.solver-environment.v1',
        'R17 Reference BJ actual366',
        'f947712d254f0add353101b446811d9c6f7692387695efb32f504167133614e0',
        '5df8509db6f5539a93aaa2037c338d3ad9321eef004feb36e2f2c64925e464b8',
        DLL_SHA256),
}


def _identity(root: Path, product: dict) -> _Identity:
    spec = _IDENTITIES.get(product.get('product'))
    if spec is None:
        raise ValueError('PPF product root/name is not in the fixed allowlist')
    if product.get('schema') != spec.product_schema:
        raise ValueError('PPF product schema mismatch')
    if product['product'] == 'R17 Reference BJ' and (
            product.get('backend') != 'cuda'
            or product.get('required_root_motion_feature') != 'pdrd_kinematic_roots_v3'):
        raise ValueError('R17 backend/root-motion contract mismatch')
    if product['product'] == 'R15 Adaptive BJ' and (
            product.get('backend') != 'cuda'
            or product.get('adaptive_policy') != 'sticky-ceiling-v2'
            or product.get('capabilities') != {'root_v3': False, 'soft_connected_collision_v1': False}
            or product.get('required_root_motion_feature') is not None):
        raise ValueError('R15 backend/adaptive/capability contract mismatch')
    if product['product'] == 'R17 Unified Candidate BJ' and (
            product.get('backend') != 'cuda'
            or product.get('capabilities') != {'root_v3': True, 'soft_connected_collision_v1': True}
            or product.get('adaptive_policy') != 'sticky-ceiling-v2'
            or product.get('required_root_motion_feature') != 'pdrd_kinematic_roots_v3'):
        raise ValueError('Unified candidate backend/root/collision contract mismatch')
    if product['product'] == 'R17 FEM Root Candidate':
        if any(not isinstance(v,str) or len(v)!=64 for v in (spec.policy_hash,spec.exe_hash,spec.dll_hash)):
            raise ValueError('FEM root product identity is not yet frozen')
        if (product.get('backend') != 'cuda' or product.get('adaptive_policy') != 'sticky-ceiling-v2'
                or product.get('capabilities') != {'root_v3':True,'soft_connected_collision_v1':True,'fem_prescribed_roots_v1':True}
                or product.get('required_root_motion_feature') != 'pdrd_kinematic_roots_v3'):
            raise ValueError('FEM root product capability mismatch')
    if product['product'] in ('R17 FEM ABS Events Candidate', 'R17 Bone Resume Candidate', 'R17 Bone Rewind Candidate'):
        if any(not isinstance(v,str) or len(v)!=64 for v in (spec.policy_hash,spec.exe_hash,spec.dll_hash)):
            raise ValueError('FEM event product identity is not yet frozen')
        expected_capabilities = {'root_v3':True,'soft_connected_collision_v1':True,'fem_prescribed_roots_v1':True,'fem_root_control_events_v1':True}
        if product['product'] in ('R17 Bone Resume Candidate', 'R17 Bone Rewind Candidate'):
            expected_capabilities['bone_checkpoint_resume_v1'] = True
        if (product.get('backend')!='cuda' or product.get('adaptive_policy')!='sticky-ceiling-v2'
                or product.get('capabilities') != expected_capabilities
                or product.get('required_root_motion_feature')!='pdrd_kinematic_roots_v3'):
            raise ValueError('FEM event product capability mismatch')
    return spec


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same_windows_path(first, second) -> bool:
    return ntpath.normcase(ntpath.normpath(str(first))) == ntpath.normcase(ntpath.normpath(str(second)))


def _pinned_values(product_name: str) -> dict[str, str]:
    pins = {'PPF_PRECOND': 'block-jacobi'}
    if product_name in ('R15 Adaptive BJ', 'R17 Unified Candidate BJ', 'R17 FEM Root Candidate', 'R17 FEM ABS Events Candidate', 'R17 Bone Resume Candidate', 'R17 Bone Rewind Candidate'):
        pins['PPF_ADAPTIVE_CG_RETRY'] = '1'
    return pins


def _validate_legacy_paths(product):
    # Metadata is checked for consistency, never passed as a process path.
    declared_cwd = product.get("launch_working_directory", "")
    declared_cuda = product.get("cuda_runtime_bin", "")
    def safe_declared_path(value):
        if not isinstance(value, str) or not value or '..' in value.replace('\\', '/').split('/'):
            return False
        return True
    if (not safe_declared_path(declared_cwd) or not safe_declared_path(declared_cuda)
            or ntpath.basename(ntpath.normpath(declared_cwd)).lower() != 'runtime'
            or (not ntpath.isabs(declared_cwd) and ntpath.normpath(declared_cwd).lower() != 'runtime')):
        raise ValueError('PPF product runtime path contract invalid')
    prefix = product.get("required_path_prefix", [])
    if (not isinstance(prefix, list) or len(prefix) != 2
            or not all(safe_declared_path(x) for x in prefix)
            or not _same_windows_path(prefix[0], declared_cwd)
            or not _same_windows_path(prefix[1], declared_cuda)):
        raise ValueError('PPF runtime cwd/PATH contract mismatch')

def _read_contract(root: Path) -> dict:
    product = json.loads((root / 'PRODUCT.json').read_text(encoding='utf-8-sig'))
    spec = _identity(root, product)
    policy_path = root / "SOLVER_ENVIRONMENT.json"
    policy_bytes = policy_path.read_bytes()
    if hashlib.sha256(policy_bytes).hexdigest() != spec.policy_hash:
        raise ValueError("PPF environment policy SHA256 mismatch")
    policy = json.loads(policy_bytes.decode("utf-8-sig"))
    if policy.get("schema") != spec.policy_schema:
        raise ValueError("Unsupported PPF environment policy schema")
    if policy.get("product") != spec.policy_product:
        raise ValueError("Unexpected PPF environment product identity")
    identity = policy.get("product_identity", {})
    if identity != {"exe_sha256": spec.exe_hash, "dll_sha256": spec.dll_hash, "entry_count": 366}:
        raise ValueError("PPF environment binary identity mismatch")
    apply = policy.get("apply_policy", {})
    unset = apply.get("unset_exact_names", [])
    if (len(unset) != 24 or frozenset(unset) != _UNSET
            or apply.get("pin_exact_values") != _pinned_values(product["product"])
            or apply.get("blanket_clear_ppf_prefix") is not False
            or apply.get("absence_means_remove_key") is not True
            or apply.get("do_not_substitute_zero_for_unset") is not True):
        raise ValueError("R14 exact child environment policy mismatch")
    if (product.get("compute_entry_count") != 366
            or product.get("preconditioner") != "block-jacobi"
            or product.get("runtime_files_sha256") != {
                "ppf-contact-solver.exe": spec.exe_hash,
                "libsimbackend_cuda.dll": spec.dll_hash,
            }):
        raise ValueError("PPF PRODUCT.json identity mismatch")
    _validate_legacy_paths(product)
    return product


def local_source_module():
    # Works as package module and standalone launcher sibling, including _pth.
    import importlib.util
    import sys
    name = __name__ + '_local_source'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name('local_source_profile.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def has_local_source(root):
    return local_source_module().has_config(root)


def verify_local_capabilities(profile):
    if not profile.local_source:
        raise ValueError('LOCAL_SOURCE_PROFILE_REQUIRED')
    verify_binaries(profile)
    return local_source_module().verify_probes(profile, _UNSET, already_verified=True)


def _configured_path(root, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _legacy_runtime_paths(root, product):
    cuda_value = os.environ.get('PPF_CTS_PRODUCT_CUDA_BIN') or product.get('cuda_runtime_bin')
    if not cuda_value:
        raise ValueError('PPF_PRODUCT_CUDA_REQUIRED: set PPF_CTS_PRODUCT_CUDA_BIN to the CUDA bin directory')
    cuda_bin = _configured_path(root, cuda_value)
    if cuda_bin.name.lower() != 'bin' or not cuda_bin.is_dir():
        raise ValueError('PPF_PRODUCT_CUDA_REQUIRED: set PPF_CTS_PRODUCT_CUDA_BIN to an existing CUDA bin directory')
    configured = os.environ.get('PPF_CTS_PRODUCT_PYTHON')
    candidates = [_configured_path(root, configured)] if configured else [root / 'runtime' / 'python' / 'python.exe', root / 'python' / 'python.exe']
    python = next((p.resolve() for p in candidates if p.is_file()), None)
    if python is None:
        raise ValueError('PPF_PRODUCT_PYTHON_REQUIRED: set PPF_CTS_PRODUCT_PYTHON to the product Python executable')
    return cuda_bin, python


def load_profile(root=None) -> ProductProfile:
    """Read and validate the frozen contracts; do not launch or import binaries."""
    selected_root = os.environ.get('PPF_CTS_PRODUCT_ROOT') if root is None else root
    if selected_root is None:
        # Reached only for actual extension intent. A v2 source locator does not
        # activate product routing for ordinary scenes or artifact discovery.
        source = Path(__file__).resolve().parent.parent
        if source.name == 'blender_addon':source=source.parent
        locator=source/local_source_module().CONFIG
        if locator.is_file() and local_source_module().read_json(locator).get('schema') == local_source_module().RELATIVE_SCHEMA:
            selected_root=source
        else:
            raise ValueError('PRODUCT_SELECTION_REQUIRED: select an explicit product for extended physics')
    if not str(selected_root).strip():
        raise ValueError('PPF explicit product root cannot be empty')
    product_root = Path(selected_root).resolve(strict=True)
    # A published name/root always retains the fixed registry and its failures.
    published = False
    if (product_root / 'PRODUCT.json').exists():
        published = published or local_source_module().read_json(product_root / 'PRODUCT.json').get('product') in _IDENTITIES
    if not published and has_local_source(product_root):
        doc = local_source_module().load(product_root)
        root_path = Path(doc['product_root']).resolve(strict=True)
        exe = local_source_module().relative_file(root_path, doc['exe'])
        dll = local_source_module().relative_file(root_path, doc['backend_library'])
        return ProductProfile(root_path, exe.parent, exe, dll, Path(doc['cuda_bin']).resolve(),
                              doc['product'], True, (Path(os.path.abspath(doc['python'])) if doc['platform'] in ('linux', 'darwin') else Path(doc['python']).resolve()),
                              Path(doc['source_root']).resolve(), Path(doc['build_dir']).resolve(), doc['platform'],
                              python_target=(Path(doc['python']).resolve(strict=True)
                                  if doc['schema'] == local_source_module().RELATIVE_SCHEMA and doc['platform'] in ('linux', 'darwin') else None))
    product = _read_contract(product_root)
    runtime = product_root / "runtime"
    if not runtime.is_dir():
        raise FileNotFoundError(f"Missing R14 runtime directory: {runtime}")
    cuda_bin, python = _legacy_runtime_paths(product_root, product)
    return ProductProfile(product_root, runtime, runtime / "ppf-contact-solver.exe",
                          runtime / "libsimbackend_cuda.dll", cuda_bin,
                          product['product'], python_path=python)


def verify_binaries(profile: ProductProfile) -> dict:
    """Revalidate contracts and hash the two frozen files, without loading them."""
    expected = load_profile(profile.product_root)
    if profile != expected:
        raise ValueError("R14 profile differs from the validated product contract")
    if profile.local_source:
        return local_source_module().verify(local_source_module().load(profile.product_root), _UNSET)
    spec = _IDENTITIES[profile.product_name]
    exe_hash, dll_hash = _sha256(profile.exe_path), _sha256(profile.dll_path)
    if exe_hash != spec.exe_hash or dll_hash != spec.dll_hash:
        raise ValueError("PPF EXE/DLL SHA256 mismatch; refusing an alternate build")
    return {"product": profile.product_name, "entry_count": 366,
            "exe_path": str(profile.exe_path), "dll_path": str(profile.dll_path),
            "exe_sha256": exe_hash, "dll_sha256": dll_hash,
            "environment_policy_sha256": spec.policy_hash}


def solver_child_env(profile: ProductProfile, base_env: Mapping[str, str]) -> dict[str, str]:
    """Return a solver/probe-only copy; preserve formal PPF and device settings.

    Windows environment names are case-insensitive even when the supplied
    mapping is an ordinary Python dict. Never apply this to the server or
    build worker: their deployment settings must remain intact.
    """
    verify_binaries(profile)
    if profile.local_source:
        return local_source_module().child_env(local_source_module().load(profile.product_root), base_env, _UNSET)
    env = {}
    prior_keys = {}
    pins = _pinned_values(profile.product_name)
    managed = _UNSET | set(pins) | {"PATH", "CUDA_PATH"}
    old_path = ""
    for key, value in base_env.items():
        upper = key.upper()
        if upper == "PATH":
            old_path = value
        if upper not in managed:
            prior = prior_keys.get(upper)
            if prior is not None:
                env.pop(prior, None)
            env[key] = value
            prior_keys[upper] = key
    env.update(pins)
    env["PATH"] = ";".join([str(profile.runtime_dir), str(profile.cuda_bin)] + ([old_path] if old_path else []))
    env["CUDA_PATH"] = str(profile.cuda_bin.parent)
    return env
