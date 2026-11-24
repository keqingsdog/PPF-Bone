"""Explicit R14 routing. Importing this module does not inspect or launch anything.

STATIC DRAFT: no candidate import, compilation or runtime test was performed.
"""
import os
from pathlib import Path
import sys

from ._r14_product_profile import load_profile, verify_binaries, solver_child_env


def activated_profile():
    root = os.environ.get("PPF_CTS_PRODUCT_ROOT")
    if root is None:
        return None
    if not root.strip():
        raise RuntimeError("R14_PRODUCT_UNAVAILABLE: explicit Windows product root required")
    from ._r14_product_profile import local_source_module
    local=local_source_module()
    if local.has_config(root):
        doc=local.read_json(Path(root)/local.CONFIG)
        if doc.get('schema') == local.RELATIVE_SCHEMA and doc.get('scope') == 'source':
            return None  # A locator must not replace ordinary artifact discovery.
    profile = load_profile(root)
    verify_binaries(profile)
    if not profile.local_source and sys.platform != "win32":
        raise RuntimeError("R14_PRODUCT_UNAVAILABLE: published product requires Windows")
    return profile


def solver_context(requested_exe):
    """Return product child arguments, or None for the unchanged legacy path."""
    profile = activated_profile()
    if profile is None:
        return None
    if Path(requested_exe).resolve() != profile.exe_path.resolve():
        raise RuntimeError("R14_PRODUCT_MISMATCH: solver request is outside selected product")
    return str(profile.exe_path), str(profile.runtime_dir), solver_child_env(profile, os.environ)


def _batch_literal(value):
    value = str(value)
    if any(char in value for char in ('"', '\r', '\n', '\x00')):
        raise ValueError("R14_LAUNCHER_PATH_INVALID: quote/control character")
    return value.replace('%', '%%')


def write_launcher(profile, session_path, output_path, solver_mode=None):
    """Write the final launcher before session seal; never patch sealed files."""
    if solver_mode == 'animation' and (
        profile.platform != 'win32' or not profile.local_source or profile.backend not in ('cpu', 'cuda')
    ):
        raise ValueError('R14_SOLVER_MODE_UNSUPPORTED')
    verify_binaries(profile)
    scene = Path(session_path).resolve()
    output = Path(output_path).resolve()
    interpreter = Path(sys.executable).resolve()
    expected_python = profile.python_path
    if expected_python is None:
        raise RuntimeError('PPF_PRODUCT_PYTHON_REQUIRED: configure product Python')
    if profile.local_source:
        expected_python = profile.python_path
        if profile.platform in ('linux', 'darwin'):
            interpreter = Path(os.path.abspath(sys.executable))
    if interpreter != expected_python:
        raise RuntimeError("R14_LAUNCHER_PYTHON_MISMATCH")
    helper = Path(__file__).resolve().with_name('_r14_solver_launcher.py')
    profile_helper = helper.with_name('_r14_product_profile.py')
    if not interpreter.is_file() or not helper.is_file() or not profile_helper.is_file():
        raise RuntimeError("R14_LAUNCHER_DEPENDENCY_MISSING")
    if profile.local_source and profile.platform in ('linux', 'darwin'):
        import shlex
        from ._r14_product_profile import local_source_module
        local=local_source_module();doc=local.load(profile.product_root)
        prefix = ('export PPF_BACKEND='+shlex.quote(doc['backend'])+'\n' if doc['schema']==local.RELATIVE_SCHEMA else '')
        # Freeze the canonical target in the generated launcher as well as the
        # entry spelling: an internal symlink may be changed after Build.
        frozen = (['--frozen-python-target', profile.python_target]
                  if doc['schema'] == local.RELATIVE_SCHEMA else [])
        command_path = scene / 'command.sh'
        command_path.write_bytes(render_posix_launcher(interpreter, helper, profile.product_root, scene,
                                                       output, profile.runtime_dir, prefix, frozen))
        command_path.chmod(0o755)
        return str(command_path)
    frozen_backend = None
    if profile.local_source:
        from ._r14_product_profile import local_source_module
        local=local_source_module();doc=local.load(profile.product_root)
        if doc['schema']==local.RELATIVE_SCHEMA:frozen_backend=doc['backend']
    content = render_windows_launcher(interpreter, helper, profile.product_root, scene, output, profile.runtime_dir, frozen_backend, solver_mode)
    command_path = scene / 'command.bat'
    command_path.write_bytes(content)
    return str(command_path)


def render_windows_launcher(interpreter, helper, product_root, scene, output, runtime_dir, frozen_backend=None, solver_mode=None):
    """Render the canonical batch file without executing or writing it."""
    if frozen_backend is not None and frozen_backend not in ('cpu', 'cuda', 'metal', 'rocm'):
        raise ValueError('R14_LAUNCHER_BACKEND_INVALID')
    if solver_mode is not None and solver_mode not in ('strict', 'animation'):
        raise ValueError('R14_SOLVER_MODE_INVALID')
    values = [_batch_literal(p) for p in (interpreter, helper, product_root, scene, output, runtime_dir)]
    python, launcher, root, scene_arg, output_arg, runtime = values
    lines = [
        '@echo off',
        *(['@rem PPF_SESSION_SOLVER_MODE=' + solver_mode] if solver_mode else []),
        'setlocal DisableDelayedExpansion',
        'chcp 65001 >nul',
        *(['set "PPF_BACKEND='+frozen_backend+'"'] if frozen_backend else []),
        f'cd /d "{runtime}"',
        'if errorlevel 1 exit /b %errorlevel%',
        f'"{python}" -I "{launcher}"' + (f' --sealed-solver-mode={solver_mode}' if solver_mode else '') + f' "{root}" --path "{scene_arg}" --output "{output_arg}" -- %*',
        'exit /b %errorlevel%',
    ]
    return ('\r\n'.join(lines) + '\r\n').encode('utf-8')


def verify_windows_launcher_bytes(actual, interpreter, helper, product_root, scene, output, runtime_dir, frozen_backend=None, solver_mode=None):
    """Accept only the generated launcher with known capability switches.

    Project command bytes are inspected, never run. Existing build hooks append
    their switches to the single solver line and may normalize CRLF to LF.
    """
    if not isinstance(actual, bytes):
        return False
    canonical = render_windows_launcher(interpreter, helper, product_root, scene, output, runtime_dir, frozen_backend, solver_mode)
    normalize = lambda value: value.replace(b'\r\n', b'\n')
    actual, canonical = normalize(actual), normalize(canonical)
    if b'\r' in actual:
        return False
    marker = b' -- %*\n'
    if canonical.count(marker) != 1:
        return False
    prefix, suffix = canonical.split(marker)
    if not actual.startswith(prefix + b' -- %*') or not actual.endswith(suffix):
        return False
    tail = actual[len(prefix + b' -- %*'):len(actual)-len(suffix) if suffix else len(actual)]
    if not tail.endswith(b'\n'):
        return False
    flags = tail[:-1].split()
    allowed = {b'--require-collision-filter-v1', b'--require-restoring-curves-v1'}
    return len(flags) == len(set(flags)) and all(flag in allowed for flag in flags) and tail[:-1] == b''.join(b' ' + flag for flag in flags)


def render_posix_launcher(interpreter, helper, product_root, scene, output, runtime_dir, prefix='', frozen=()):
    """Render the existing POSIX shell launcher as bytes without writing it."""
    import shlex
    if prefix and prefix not in tuple('export PPF_BACKEND=' + x + '\n' for x in ('cpu','cuda','rocm','metal')):
        raise ValueError('R14_LAUNCHER_BACKEND_INVALID')
    if frozen and (len(frozen) != 2 or frozen[0] != '--frozen-python-target'):
        raise ValueError('R14_LAUNCHER_PYTHON_TARGET_INVALID')
    arguments = [interpreter, '-I', helper, *frozen, product_root, '--path', scene, '--output', output, '--']
    return ('#!/bin/sh\nset -eu\n' + prefix + 'cd ' + shlex.quote(str(runtime_dir)) + '\nexec ' +
            ' '.join(shlex.quote(str(arg)) for arg in arguments) + ' "$@"\n').encode('utf-8')


def verify_posix_launcher_bytes(actual, interpreter, helper, product_root, scene, output, runtime_dir, prefix='', frozen=()):
    if not isinstance(actual, bytes) or b'\r' in actual:
        return False
    canonical = render_posix_launcher(interpreter, helper, product_root, scene, output, runtime_dir, prefix, frozen)
    marker = b' "$@"\n'
    if canonical.count(marker) != 1:
        return False
    head, suffix = canonical.split(marker)
    head += b' "$@"'
    if not actual.startswith(head) or not actual.endswith(suffix):
        return False
    tail = actual[len(head):len(actual)-len(suffix) if suffix else len(actual)]
    if not tail.endswith(b'\n'):
        return False
    flags = tail[:-1].split()
    allowed = {b'--require-collision-filter-v1', b'--require-restoring-curves-v1'}
    return len(flags) == len(set(flags)) and all(x in allowed for x in flags) and tail[:-1] == b''.join(b' '+x for x in flags)
