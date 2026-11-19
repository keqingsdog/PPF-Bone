"""Bind an explicitly built CUDA source checkout to its local runtime.

Example: python tools/configure_source_product.py --source-root .
 --build-dir target/cuda/release --product-root ../LocalProduct
 --python /absolute/venv/bin/python --cuda-bin /usr/local/cuda/bin
 --backend-library crates/ppf-cts-compute/cuda/build/lib/libppfbe_cuda.so

The output is local configuration, not a published-release identity. All
capabilities are obtained from the actual paired build before configuration.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value

def atomic_json(path, doc):
    temporary = path.with_name(path.name + '.tmp')
    try:
        temporary.write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def configure_relative(argv):
    """Create a portable source locator; never manufacture capability claims."""
    parser = argparse.ArgumentParser(description='Configure an assembled relative v2 source runtime')
    parser.add_argument('--relative-root', required=True, type=Path)
    parser.add_argument('--backend', required=True, choices=('cpu','cuda','metal','rocm'))
    parser.add_argument('--python-relative', required=True)
    parser.add_argument('--bin-relative', default='bin')
    parser.add_argument('--build-relative')
    parser.add_argument('--backend-library-relative')
    parser.add_argument('--route', action='append', default=[], metavar='BACKEND:BUILD:LIBRARY')
    args=parser.parse_args(argv)
    root=args.relative_root.resolve(strict=True)
    if sys.platform not in ('win32','linux','darwin'):parser.error('unsupported host platform')
    local=module(root/'frontend/local_source_profile.py','_configure_relative_source')
    build=args.build_relative or 'target/'+args.backend+'/release'
    name=local.artifact_library_name(args.backend)
    doc=dict(schema=local.RELATIVE_SCHEMA,scope='source',product=local.LABEL,
             platform=sys.platform,backend=args.backend,product_root='.',source_root='.',
             build_dir=build,python=args.python_relative,cuda_bin=args.bin_relative,
             library_dirs=[],exe=build+('/ppf-contact-solver.exe' if sys.platform=='win32' else '/ppf-contact-solver'),
             backend_library=args.backend_library_relative or build+'/'+name)
    routes={args.backend:dict(build_dir=build,exe=doc['exe'],backend_library=doc['backend_library'])}
    for route in args.route:
        parts=route.split(':')
        if len(parts)!=3 or parts[0] not in ('cpu','cuda','metal','rocm') or parts[0] in routes:
            parser.error('invalid or duplicate --route')
        backend,route_build,library=parts
        routes[backend]=dict(build_dir=route_build,exe=route_build+('/ppf-contact-solver.exe' if sys.platform=='win32' else '/ppf-contact-solver'),backend_library=library)
    doc['backend_routes']=routes
    if (root/local.CONFIG).exists():parser.error('configuration already exists; preserve it and choose a fresh runtime')
    # Validate path closure and artifact existence before creating a locator.
    os.environ['PPF_BACKEND']=args.backend  # Explicit CLI selection, not inherited runtime preference.
    for selected in routes:
        os.environ['PPF_BACKEND']=selected
        local.load_relative(root,doc)
    os.environ['PPF_BACKEND']=args.backend
    resolved=local.load_relative(root,doc)
    env=local.child_env(resolved,os.environ,set())
    result=subprocess.run([str(root/doc['exe']),'--backend'],env=env,cwd=root/build,capture_output=True,text=True,timeout=60)
    if result.returncode or args.backend not in result.stdout.lower().split():
        raise ValueError('relative source backend executable did not confirm selected backend')
    atomic_json(root/local.CONFIG,doc)
    print(json.dumps({'configured':str(root/local.CONFIG),'backend':args.backend,
                      'scope':'locator only; actual extension capability admission remains at Build'},indent=2))
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if '--relative-root' in argv:return configure_relative(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'build-dir', 'product-root', 'python', 'cuda-bin', 'backend-library'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--library-dir', action='append', default=[])
    args = parser.parse_args(argv)
    if sys.platform not in ('win32', 'linux'):
        parser.error('only Windows/Linux CUDA source products are supported')
    source = Path(args.source_root).resolve(strict=True)
    build = Path(args.build_dir)
    build = (source / build).resolve(strict=True) if not build.is_absolute() else build.resolve(strict=True)
    product = Path(args.product_root).resolve()
    # POSIX venv entry paths carry prefix discovery; resolving the symlink
    # to system Python would silently discard that environment.
    python = Path(os.path.abspath(args.python)) if sys.platform == 'linux' else Path(args.python).resolve(strict=True)
    cuda = Path(args.cuda_bin).resolve(strict=True)
    backend = Path(args.backend_library)
    backend = (source / backend).resolve(strict=True) if not backend.is_absolute() else backend.resolve(strict=True)
    if product.exists():
        parser.error('product root already exists; use a new explicit product root')
    if not python.is_file() or not cuda.is_dir():
        parser.error('Python/CUDA path has the wrong type')
    libraries = [str(Path(value).resolve(strict=True)) for value in args.library_dir]
    if any(not Path(value).is_dir() for value in libraries):
        parser.error('library-dir must be a directory')
    win = sys.platform == 'win32'
    exe_name = 'ppf-contact-solver.exe' if win else 'ppf-contact-solver'
    dll_name = 'libsimbackend_cuda.dll' if win else 'libppfbe_cuda.so'
    exe = build / exe_name
    if not exe.is_file() or backend.name != dll_name:
        parser.error('actual solver/backend artifact missing or platform filename wrong')
    profiles = module(source / 'frontend/_r14_product_profile.py', '_configure_source_profiles')
    local = profiles.local_source_module()
    identity = local.read_json(source / 'coordination/windows-resume-20261004/restoring-curve/native-scalar/compiled-identity.json')
    source_identity = {key: identity[key] for key in ('contract_version', 'channel_mask', 'ordered_entry_count', 'source_stamp', 'layout_sha256', 'kernel_table_sha256')}
    if source_identity['ordered_entry_count'] != 366:
        raise ValueError('source generator did not produce the required 366 entries')
    result = subprocess.run([str(python), '-I', '-c', 'import sys; print(sys.executable)'], capture_output=True, text=True, timeout=30)
    returned_python = Path(os.path.abspath(result.stdout.strip())) if sys.platform == 'linux' else Path(result.stdout.strip()).resolve()
    if result.returncode != 0 or returned_python != python:
        raise ValueError('bound Python could not execute or resolved a different interpreter')
    product.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=product.name + '.configuring-', dir=product.parent))
    committed = False
    try:
        runtime = stage / 'runtime'
        runtime.mkdir()
        shutil.copy2(exe, runtime / exe_name)
        shutil.copy2(backend, runtime / dll_name)
        policy = {'schema': 'ppf.local-source-policy.v1', 'product': local.LABEL,
                  'unset': sorted(profiles._UNSET), 'set': local.PINS}
        atomic_json(stage / local.POLICY, policy)
        doc = {'schema': local.SCHEMA, 'product': local.LABEL, 'platform': sys.platform,
               'backend': 'cuda', 'source_root': str(source), 'build_dir': str(build),
               'product_root': str(stage), 'python': str(python), 'cuda_bin': str(cuda),
               'library_dirs': libraries, 'exe': 'runtime/' + exe_name,
               'backend_library': 'runtime/' + dll_name, 'source_identity': source_identity,
               'exe_sha256': local.sha256(runtime / exe_name),
               'dll_sha256': local.sha256(runtime / dll_name),
               'policy_sha256': local.sha256(stage / local.POLICY)}
        actual = local.probe(doc, local.child_env(doc, os.environ, profiles._UNSET))
        doc.update(format_identity=actual['format_identity'], probe_capabilities=actual['capabilities'])
        doc['product_root'] = str(product)
        atomic_json(stage / local.CONFIG, doc)
        # Both destinations are prepared before making the source selection visible.
        os.replace(stage, product)
        committed = True
        profiles.verify_local_capabilities(profiles.load_profile(product))
        atomic_json(source / local.CONFIG, doc)
    except Exception:
        # Own new directory only; never remove another existing source/product.
        if committed:
            shutil.rmtree(product)
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(json.dumps({'configured': str(product), 'source_root': str(source),
                      'backend': 'cuda', 'device': actual['device'],
                      'format_identity': actual['format_identity']}, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
