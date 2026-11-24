"""Standalone R14 launcher: pure stdlib/profile, never import frontend/DLL.

Regular: python -I helper.py ROOT --path SCENE --output OUTPUT -- SOLVER_ARGS
Internal server query: python -I helper.py --product-root ROOT --query-backend --await-parent-job
STATIC DRAFT: NOT_EXECUTED.
"""
import os
import json
from pathlib import Path
import subprocess
import sys
import tomllib


def require_pcg_precision(scene, profile, child_env):
    """Reject retired mixed-precision sessions before launching a solver."""
    with (scene / 'param.toml').open('rb') as stream:
        params = tomllib.load(stream)['param']
    precision = params.get('pcg_precision', 'fp32')
    if precision != 'fp32':
        raise RuntimeError(
            'PCG_PRECISION_UNSUPPORTED: FP16 PCG has been retired; '
            'rebuild and run the scene from the beginning with FP32'
        )


def main(argv):
    solver_mode = 'strict'
    sealed_mode = len(argv) > 1 and argv[1].startswith('--sealed-solver-mode=')
    if sealed_mode:
        solver_mode = argv[1].split('=', 1)[1]
        if solver_mode not in ('strict', 'animation'):
            raise ValueError('R14_SOLVER_MODE_INVALID')
        argv = [argv[0], *argv[2:]]
    # Generated v2 POSIX commands carry a canonical target frozen at Build.
    # Normalize this protected launcher argument before the existing routing
    # parser so solver arguments and query-backend retain their old positions.
    frozen_python_target = None
    if len(argv) > 1 and argv[1] == '--frozen-python-target':
        if len(argv) < 4 or not Path(argv[2]).is_absolute():
            raise ValueError('R14_LAUNCHER_PYTHON_TARGET_INVALID')
        frozen_python_target = Path(argv[2])
        argv = [argv[0], *argv[3:]]
    # Embedded Python's _pth does not expose script siblings. Load exactly
    # the two trusted files beside this launcher without package/DLL imports
    # or changing global producer discovery.
    import importlib.util
    def sibling(name):
        source = Path(__file__).resolve().with_name(name + '.py')
        module_name = '_ppf_standalone' + name
        spec = importlib.util.spec_from_file_location(module_name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    profiles = sibling('_r14_product_profile')
    validation = sibling('_build_receipt_validation_')
    load_profile = profiles.load_profile
    verify_binaries = profiles.verify_binaries
    solver_child_env = profiles.solver_child_env
    query_backend = len(argv) == 5 and argv[1] == '--product-root' and argv[3:] == ['--query-backend', '--await-parent-job']
    selected_root = argv[2] if query_backend else (argv[1] if len(argv) > 1 else '')
    selected_profile = load_profile(selected_root)
    expected_python = selected_profile.python_path
    if expected_python is None:
        raise RuntimeError('PPF_PRODUCT_PYTHON_REQUIRED: configure product Python')
    actual_python = (Path(os.path.abspath(sys.executable)) if selected_profile.local_source and selected_profile.platform in ('linux', 'darwin') else Path(sys.executable).resolve())
    if actual_python != expected_python:
        raise RuntimeError('R14_LAUNCHER_PYTHON_MISMATCH')
    if frozen_python_target is not None:
        if (not selected_profile.local_source or selected_profile.platform not in ('linux', 'darwin')
                or selected_profile.python_target != frozen_python_target
                or Path(sys.executable).resolve(strict=True) != frozen_python_target):
            raise RuntimeError('R14_LAUNCHER_PYTHON_TARGET_MISMATCH')
    elif (not query_backend and selected_profile.local_source
          and selected_profile.platform in ('linux', 'darwin')
          and selected_profile.python_target is not None
          and actual_python != selected_profile.python_target):
        # Existing v2 command files were generated only when the entry was
        # already canonical. Preserve their original retarget protection.
        raise RuntimeError('R14_LAUNCHER_PYTHON_TARGET_MISMATCH')
    if query_backend:
        # Parent must assign this helper to its non-inheritable kill-on-close
        # Job before releasing native execution. Read through EOF (up to four
        # bytes) so missing, wrong or additional gate bytes all fail closed.
        try:
            gate = sys.stdin.buffer.read(4)
        except Exception:
            print('R14_PARENT_JOB_GATE_READ_FAILED', file=sys.stderr)
            return 2
        if gate != b'GO\n':
            print('R14_PARENT_JOB_GATE_REQUIRED', file=sys.stderr)
            return 2
        profile = load_profile(argv[2])
        verify_binaries(profile)
        # Native stdout/stderr and actual exit status are preserved. This is
        # metadata only; it does not replace the real formats-stamp --probe.
        return subprocess.call([str(profile.exe_path), '--backend'],
                               cwd=str(profile.runtime_dir),
                               env=solver_child_env(profile, os.environ))
    if len(argv) < 7 or argv[1].startswith('--') or argv[2] != '--path' or argv[4] != '--output' or argv[6] != '--':
        raise ValueError('R14_LAUNCHER_USAGE: ROOT --path SCENE --output OUTPUT -- SOLVER_ARGS')
    profile = load_profile(argv[1])
    if solver_mode == 'animation' and (
        os.name != 'nt' or not profile.local_source or profile.backend not in ('cpu', 'cuda')
    ):
        raise ValueError('R14_SOLVER_MODE_UNSUPPORTED')
    verify_binaries(profile)
    scene, output = Path(argv[3]), Path(argv[5])
    if not scene.is_absolute() or not output.is_absolute():
        raise ValueError('R14_LAUNCHER_ABSOLUTE_PATHS_REQUIRED')
    identity = verify_binaries(profile)
    forwarded = argv[7:]
    protected = {'--product-root', '--root', '--path', '--output', '--query-backend', '--await-parent-job', '--frozen-python-target', '--sealed-solver-mode'}
    if any(arg.split('=', 1)[0] in protected for arg in forwarded):
        raise ValueError('R14_LAUNCHER_ROUTING_OVERRIDE_FORBIDDEN')
    if any(arg == '--async-output' or arg.startswith('--async-output=') for arg in forwarded):
        raise ValueError('R14_SYNC_OUTPUT_REQUIRED')
    resume = sibling('_bone_resume_')
    load,target = resume.resume_arguments(forwarded)
    mode = resume.twist_mode(forwarded)
    # Explicit V1 migration or ordinary V2 continuation may bind to the new
    # executable, but only after its real continuous capability is confirmed.
    v2_resume = load > 0 and mode is None and resume.v2_history_candidate(output, load)
    rebind = load > 0 and (mode == 'continuous_from_current' or v2_resume)
    child_env = solver_child_env(profile, os.environ)
    child_env.pop('PPF_ANIMATION_BOUNDED_PROGRESS', None)
    if os.name == "nt" and profile.local_source:
        trial_options = sibling('_bone_trial_options')
        if not sealed_mode and load != 0:
            trial_options.check_legacy_r14_animation_resume(scene / 'stdout.log')
        trial_options.solver_env(child_env, profile.backend, solver_mode)
    capability = None
    if target is not None or mode is not None or v2_resume:
        capability = resume.capability(profile, child_env, 'continuous_from_current' if rebind else mode)
    stamps = capability.get('accepted_receipt_source_stamps', ()) if rebind and capability else ()
    validation.validate_receipt(scene, validation.contract_identity(profile, identity),
                                resume_exe_rebind=rebind, resume_source_stamps=stamps)
    require_pcg_precision(scene, profile, child_env)
    # Native validates the checkpoint/coverage before pruning output. Probe is not checkpoint validity.
    # Keep solver flags in original order, including --load and require flags.
    return subprocess.call([str(profile.exe_path), '--path', str(scene), '--output', str(output), *forwarded],
                           cwd=str(profile.runtime_dir),
                           env=child_env)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
