"""Trusted-profile, read-only Resume capability probe.

The caller supplies a session directory, never an executable or project script.
The profile comes from the server environment and the executable is queried only
once the entire saved launcher matches the trusted renderer. No launcher is run.
"""
import importlib
import json
import os
from pathlib import Path
import sys
import types

_PACKAGE = '_ppf_resume_probe_frontend'

def _sibling(name):
    if _PACKAGE not in sys.modules:
        package = types.ModuleType(_PACKAGE)
        package.__path__ = [str(Path(__file__).resolve().parent)]
        sys.modules[_PACKAGE] = package
    return importlib.import_module(_PACKAGE + '.' + name)



def _trusted_launcher_path():
    # Windows Rust canonicalize emits extended paths while Blender's sealed
    # renderer uses ordinary drive/UNC spelling. Resolve the trusted sibling
    # first; normalize only these two equivalent, fully-qualified spellings.
    path = str(Path(__file__).resolve().with_name('_r14_solver_launcher.py'))
    if os.name == 'nt':
        slash = chr(92)
        prefix = slash * 2 + '?' + slash
        if path.startswith(prefix + 'UNC' + slash):
            path = slash * 2 + path[len(prefix + 'UNC' + slash):]
        elif path.startswith(prefix) and len(path) >= len(prefix) + 3:
            tail = path[len(prefix):]
            if tail[0].isalpha() and tail[1:3] == ':' + slash:
                path = tail
    return Path(path)

def probe(scene, root, profile_loader=None, capability=None):
    if not root or not Path(root).is_absolute():
        raise ValueError('TWIST_RESUME_TRUSTED_PROFILE_REQUIRED')
    scene = Path(scene).resolve(strict=True)
    if not scene.is_dir():
        raise ValueError('TWIST_RESUME_SCENE_INVALID')
    profiles = _sibling('_r14_product_profile')
    adapter = _sibling('_r14_product_adapter')
    resume = _sibling('_bone_resume_')
    profile = (profile_loader or profiles.load_profile)(root)
    interpreter = profile.python_path if profile.local_source else Path(sys.executable).resolve()
    backend = None
    doc = None
    if profile.local_source:
        local = profiles.local_source_module()
        doc = local.load(profile.product_root)
        backend = doc['backend'] if doc['schema'] == local.RELATIVE_SCHEMA else None
    helper = _trusted_launcher_path()
    output = (scene / 'output').resolve()
    if profile.platform == 'win32':
        raw = (scene / 'command.bat').read_bytes()
        valid = any(adapter.verify_windows_launcher_bytes(raw, interpreter, helper,
            profile.product_root, scene, output, profile.runtime_dir, backend, mode)
            for mode in (None, 'strict', 'animation'))
    elif profile.local_source and profile.platform in ('linux', 'darwin'):
        prefix = 'export PPF_BACKEND=' + backend + '\n' if backend else ''
        frozen = ['--frozen-python-target', profile.python_target] if backend else []
        raw = (scene / 'command.sh').read_bytes()
        valid = adapter.verify_posix_launcher_bytes(raw, interpreter, helper,
            profile.product_root, scene, output, profile.runtime_dir, prefix, frozen)
    else:
        raise ValueError('TWIST_RESUME_LAUNCHER_UNSUPPORTED')
    if not valid:
        raise ValueError('TWIST_RESUME_UNTRUSTED_LAUNCHER')
    call = capability or resume.capability
    result = call(profile, profiles.solver_child_env(profile, os.environ), 'continuous_from_current')
    result = dict(result)
    result['selected_backend'] = profile.backend
    return result


def main():
    if len(sys.argv) != 2:
        raise ValueError('TWIST_RESUME_PROBE_USAGE')
    result = probe(sys.argv[1], os.environ.get('PPF_CTS_PRODUCT_ROOT'))
    print(json.dumps(result, separators=(',', ':')))

if __name__ == '__main__':
    main()
