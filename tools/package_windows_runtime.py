"""Assemble current-source CPU/CUDA artifacts without running destructive bundle.bat."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess

ARTIFACTS = ('ppf-contact-solver.exe', 'ppf-cts-server.exe', '_ppf_cts_py.dll', '.ppf-backend')

def artifact_groups(source, backends, cpu_dir=None, cuda_dir=None):
    """Resolve each complete backend group once; never fall back per file."""
    requested = {'cpu': cpu_dir, 'cuda': cuda_dir}
    groups = {}
    for backend in backends:
        directory = requested[backend] or source / 'target' / backend / 'release'
        try:
            directory = directory.resolve(strict=True)
        except OSError as exc:
            raise SystemExit(f'{backend} artifact directory unavailable: {directory}: {exc}') from exc
        if not directory.is_dir():
            raise SystemExit(f'{backend} artifact directory is not a directory: {directory}')
        for name in ARTIFACTS:
            path = directory / name
            if not path.is_file():
                raise SystemExit('Required runtime input missing: ' + str(path))
            if path.resolve().parent != directory:
                raise SystemExit(f'Artifact resolves outside its {backend} group: {path}')
        marker = (directory / '.ppf-backend').read_text(encoding='utf-8').strip()
        if marker != backend:
            raise SystemExit(f'Backend marker mismatch: {directory}: expected {backend}, got {marker!r}')
        groups[backend] = directory
    return groups

def require_fem_retry(executable, env):
    """Reject a stale solver before copying and again after packaging."""
    try:
        proc = subprocess.run([str(executable), '--probe-fem-retry'], env=env,
                              capture_output=True, text=True, errors='replace', timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SystemExit(f'FEM retry capability probe failed: {executable}: {exc}') from exc
    if proc.returncode != 0 or proc.stdout.strip() != 'typed_b32_v1':
        raise SystemExit(f'FEM retry capability probe rejected {executable}: '
                         f'exit={proc.returncode}, stdout={proc.stdout!r}, stderr={proc.stderr!r}')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--third-party-runtime',type=Path,required=True)
    parser.add_argument('--mingit',type=Path,required=True)
    parser.add_argument('--backend',choices=('all','cpu'),default='all',help='all preserves the existing CPU/CUDA bundle; cpu packages only the CPU runtime')
    parser.add_argument('--include-ffmpeg',action='store_true',help='Include optional FFmpeg; distributor must provide its required licensing and source materials')
    parser.add_argument('--cuda-library',type=Path)
    parser.add_argument('--cuda-bin',type=Path)
    parser.add_argument('--cpu-artifact-dir',type=Path,help='CPU release directory containing the complete backend group')
    parser.add_argument('--cuda-artifact-dir',type=Path,help='CUDA release directory containing the complete backend group')
    parser.add_argument('--msvc-redist',type=Path,required=True,help='Official x64 Microsoft.VC143.CRT redistributable directory')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    backends=('cpu',) if args.backend=='cpu' else ('cpu','cuda')
    default_backend=backends[-1]
    if 'cuda' in backends and (args.cuda_library is None or args.cuda_bin is None):
        parser.error('--cuda-library and --cuda-bin are required for --backend all')
    source=args.source_root.resolve(strict=True)
    output=args.output.resolve()
    if output.exists():raise SystemExit('Output already exists; use a new directory, never clear a user runtime')
    groups=artifact_groups(source,backends,args.cpu_artifact_dir,args.cuda_artifact_dir)
    # The SOFT frontend's reference builder loads this shared NumPy-only module.
    shared_modules=('fem_scale.py','fem_body_control_events.py')
    required=[source/'blender_addon/core'/name for name in shared_modules]
    required.extend(source/name for name in ('LICENSE','NOTICE_BONE_FORK.md','README_BONE_FORK.md','BUILDING_BONES.md','pyproject.toml'))
    required += [args.third_party_runtime/'python/python.exe',
                 args.mingit/'cmd/git.exe',
                 args.msvc_redist/'msvcp140.dll',args.msvc_redist/'vcruntime140.dll',args.msvc_redist/'vcruntime140_1.dll']
    if args.include_ffmpeg:
        required.append(args.third_party_runtime/'bin/ffmpeg.exe')
    if 'cuda' in backends:
        required.extend((args.cuda_library,args.cuda_bin/'cudart64_12.dll'))
    for path in required:
        if not path.is_file():raise SystemExit('Required runtime input missing: '+str(path))
    for directory in (args.third_party_runtime/'python', args.mingit,
                      source/'crates/ppf-cts-core/src',source/'crates/ppf-cts-solver/src'):
        if not directory.is_dir():raise SystemExit('Required runtime directory missing: '+str(directory))
    system=Path(os.environ.get('SystemRoot',r'C:\Windows'))
    # The input solver is linked to the supplied runtime DLLs even though this
    # host-only probe neither initializes a device nor reads a scene.
    checks=[]
    for backend, directory in groups.items():
        probe_env=os.environ.copy()
        dll_dirs=[directory,args.msvc_redist]
        if backend=='cuda':dll_dirs.extend((args.cuda_library.parent,args.cuda_bin))
        probe_env['PATH']=os.pathsep.join(map(str,(*dll_dirs,system/'System32',system)))
        require_fem_retry(directory/'ppf-contact-solver.exe',probe_env)
        checks.append({'backend':backend,'input_solver':str(directory/'ppf-contact-solver.exe'),
                       'fem_retry_capability':'typed_b32_v1'})
    output.mkdir(parents=True)
    ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pyo','.pytest_cache')
    for name in ('frontend','examples','asset','docs'):
        if (source/name).is_dir():shutil.copytree(source/name,output/name,ignore=ignore)
    (output/'blender_addon/core').mkdir(parents=True)
    for name in shared_modules:shutil.copy2(source/'blender_addon/core'/name,output/'blender_addon/core'/name)
    shutil.copytree(args.third_party_runtime/'python',output/'python',ignore=ignore)
    shutil.copytree(args.mingit,output/'mingit',ignore=ignore)
    for crate in ('ppf-cts-core','ppf-cts-solver'):
        shutil.copytree(source/'crates'/crate/'src',output/'crates'/crate/'src',ignore=ignore)
    for name in ('LICENSE','NOTICE_BONE_FORK.md','README_BONE_FORK.md','BUILDING_BONES.md','pyproject.toml'):
        shutil.copy2(source/name,output/name)
    (output/'licenses').mkdir()
    shutil.copy2(source/'LICENSE',output/'licenses/PPF-LICENSE.txt')
    if 'cuda' in backends:
        cuda_license=args.cuda_bin.parent/'LICENSE'
        if cuda_license.is_file():shutil.copy2(cuda_license,output/'licenses/CUDA-LICENSE.txt')
    (output/'bin').mkdir()
    for path in args.msvc_redist.glob('*.dll'):shutil.copy2(path,output/'bin'/path.name)
    runtime_libraries=[args.third_party_runtime/'bin/ffmpeg.exe'] if args.include_ffmpeg else []
    if 'cuda' in backends:
        runtime_libraries.extend((args.cuda_library,args.cuda_bin/'cudart64_12.dll'))
    for path in runtime_libraries:
        shutil.copy2(path,output/'bin'/path.name)
    for backend in backends:
        target=output/'target'/backend/'release';target.mkdir(parents=True)
        for name in ARTIFACTS:shutil.copy2(groups[backend]/name,target/name)
        # The ordinary frontend also resolves backend libraries beside artifacts.
        if backend=='cuda':shutil.copy2(args.cuda_library,target/args.cuda_library.name)
    (output/'.ppf-selfcontained').write_text('selfcontained\n')
    profile={'schema':'ppf.local-source-product.v2','scope':'source','product':'Local Source Build',
             'platform':'win32','backend':default_backend,'product_root':'.','source_root':'.',
             'build_dir':f'target/{default_backend}/release','python':'python/python.exe',
             'cuda_bin':'bin',
             'library_dirs':['bin'],'exe':f'target/{default_backend}/release/ppf-contact-solver.exe',
             'backend_library':('bin/libsimbackend_cuda.dll' if default_backend=='cuda'
                                else 'target/cpu/release/_ppf_cts_py.dll')}
    if len(backends) == 2:
        profile['backend_routes'] = {
            'cpu': {'build_dir': 'target/cpu/release',
                    'exe': 'target/cpu/release/ppf-contact-solver.exe',
                    'backend_library': 'target/cpu/release/_ppf_cts_py.dll'},
            'cuda': {'build_dir': 'target/cuda/release',
                     'exe': 'target/cuda/release/ppf-contact-solver.exe',
                     'backend_library': 'bin/libsimbackend_cuda.dll'},
        }
    (output/'LOCAL_SOURCE_BUILD.json').write_text(json.dumps(profile,indent=2),encoding='utf-8')
    (output/'config.bat').write_text('@echo off\nset PORT=8080\n')
    launch=r'''@echo off
setlocal
set "DIST=%~dp0"
call "%DIST%config.bat"
set "PATH=%DIST%python;%DIST%python\Scripts;%DIST%bin;%DIST%mingit\cmd;%PATH%"
set "PYTHONPATH=%DIST%;%PYTHONPATH%"
set "SSL_CERT_FILE=%DIST%python\Lib\site-packages\certifi\cacert.pem"
set "REQUESTS_CA_BUNDLE=%SSL_CERT_FILE%"
set "JUPYTER_CONFIG_DIR=%DIST%jupyter\config"
set "JUPYTER_DATA_DIR=%DIST%jupyter\data"
set "IPYTHONDIR=%DIST%jupyter\ipython"
"%DIST%python\python.exe" -m jupyterlab --no-browser --port=%PORT% --ServerApp.port_retries=50 --notebook-dir="%DIST%examples"
endlocal
'''
    (output/'start.bat').write_text(launch,encoding='utf-8')
    env=os.environ.copy()
    system=Path(env.get('SystemRoot',r'C:\Windows'))
    # Loader smoke tests must not borrow CUDA/MSVC from the developer PATH.
    env['PATH']=os.pathsep.join(map(str,(output/'bin',output/'python',output/'mingit/cmd',system/'System32',system)))
    for backend in backends:
        executable=output/'target'/backend/'release/ppf-contact-solver.exe'
        proc=subprocess.run([str(executable),'--backend'],env=env,capture_output=True,text=True,errors='replace')
        checks.append({'backend':backend,'exit_code':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr})
        if proc.returncode or proc.stdout.strip()!=backend:raise SystemExit('Backend startup failed: '+str(checks[-1]))
        require_fem_retry(executable,env)
        checks.append({'backend':backend,'packaged_solver':str(executable),
                       'fem_retry_capability':'typed_b32_v1'})
    proc=subprocess.run([str(output/'python/python.exe'),'--version'],env=env,capture_output=True,text=True)
    checks.append({'python_exit_code':proc.returncode,'stdout':proc.stdout})
    if proc.returncode:raise SystemExit('Bundled Python startup failed')
    proc=subprocess.run([str(output/'mingit/cmd/git.exe'),'--version'],env=env,capture_output=True,text=True)
    checks.append({'git_exit_code':proc.returncode,'stdout':proc.stdout})
    if proc.returncode:raise SystemExit('Bundled MinGit startup failed')
    print(json.dumps({'runtime':str(output),'checks':checks,'native_simulation_run':False},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
