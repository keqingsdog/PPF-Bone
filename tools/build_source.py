"""Build the checked-in source with a local toolchain and no prior own artifacts.

Select CPU, CUDA, Metal or ROCm explicitly, or retain upstream auto selection.
Use a Visual Studio x64 developer console on Windows. CUDA builds need 12.8;
CPU builds do not build or load a CUDA backend.
"""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys

ROOT=Path(os.path.abspath(__file__)).parents[1] if os.name == 'nt' else Path(__file__).resolve().parents[1]

def checked(argv,env):
    print('RUN',subprocess.list2cmdline([str(x) for x in argv]),flush=True)
    if env.get('PPF_SOURCE_PRINT_ONLY')=='1':return
    subprocess.run([str(x) for x in argv],cwd=ROOT,env=env,check=True)

def require_cuda_guard_tools(env):
    """Formal source builds must reach the unchanged build.rs SASS guard."""
    if 'PPF_ALLOW_FP64' in env:
        raise SystemExit('Formal CUDA builds cannot use PPF_ALLOW_FP64: the original device single-precision guard must run.')
    for name in ('cuobjdump','nvdisasm'):
        tool=shutil.which(name,path=env.get('PATH',''))
        if tool is None:
            raise SystemExit('Formal CUDA builds require '+name+' on PATH before compilation; install the NVIDIA binary tools. --phase backend is a development-only device build without final Rust guard acceptance.')
        try:
            result=subprocess.run([tool,'--version'],env=env,capture_output=True,text=True,check=True)
        except (OSError,subprocess.CalledProcessError) as error:
            raise SystemExit('CUDA guard tool is not runnable: '+name) from error
        print('CUDA_GUARD_TOOL',name,tool,result.stdout.strip(),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda-root',type=Path)
    parser.add_argument('--rocm-root',type=Path)
    parser.add_argument('--backend',choices=('auto','cpu','cuda','metal','rocm'),default='auto')
    parser.add_argument('--python',type=Path,default=Path(sys.executable))
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--phase',choices=('all','backend','rust','services'),default='all')
    parser.add_argument('--target-dir',type=Path)
    parser.add_argument('--offline',action='store_true')
    native_platform='windows' if os.name=='nt' else 'macos' if sys.platform=='darwin' else 'linux'
    parser.add_argument('--platform',choices=('windows','linux','macos'),default=native_platform)
    parser.add_argument('--print-commands',action='store_true',help='Print the selected platform commands without compiling or probing tools.')
    args=parser.parse_args()
    if args.jobs < 1:parser.error('--jobs must be at least 1')
    env=os.environ.copy()
    windows=args.platform=='windows'
    if not args.print_commands and args.platform!=native_platform:
        raise SystemExit('--platform must match the current host unless --print-commands is used.')
    if args.print_commands:env['PPF_SOURCE_PRINT_ONLY']='1'
    # Preserve a selected venv's executable entry; resolving its symlink loses
    # the venv prefix and silently changes frontend/build dependencies on Unix.
    python=Path(os.path.abspath(args.python))
    if not python.is_file():raise SystemExit('Selected Python entry is not a file: '+str(python))
    env['PPF_KERNELGEN_PYTHON']=str(python)
    env['PYO3_PYTHON']=str(python)
    env['PYTHON']=str(python)
    env['PPF_WIN_BUILD_JOBS']=str(max(1,args.jobs))
    env['CARGO_BUILD_JOBS']=str(max(1,args.jobs))
    # Cargo owns the outer budget. Each nested device build uses one worker,
    # rather than multiplying --jobs by another machine-wide make pool.
    env['PPF_DEVICE_BUILD_JOBS']='1'
    env['PATH']=str(python.parent)+os.pathsep+env.get('PATH','')
    cuda=args.cuda_root or (Path(env['CUDA_PATH']) if env.get('CUDA_PATH') else None)
    if cuda is None and env.get('PPF_CUDA_ROOT'):cuda=Path(env['PPF_CUDA_ROOT'])
    if cuda is None:
        compiler=shutil.which('nvcc.exe' if windows else 'nvcc',path=env.get('PATH',''))
        if compiler:cuda=Path(compiler).parent.parent
        elif not windows and Path('/usr/local/cuda/bin/nvcc').is_file():cuda=Path('/usr/local/cuda')
    backend=args.backend
    if backend=='auto':
        backend='cuda' if cuda is not None else 'metal' if args.platform=='macos' else 'auto'
        if backend=='auto':
            raise SystemExit('No CUDA or Metal backend is available for automatic selection. Install the GPU SDK or select --backend cpu explicitly.')
    if backend=='cuda' and cuda is None:
        raise SystemExit('CUDA backend requested but no CUDA SDK was found. Select --cuda-root, CUDA_PATH or PPF_CUDA_ROOT; use --backend cpu for a CPU build.')
    if windows and backend=='metal':
        raise SystemExit('The Metal backend requires macOS; no Windows backend command is available.')
    target=args.target_dir or Path('target')/backend
    env['CARGO_TARGET_DIR']=os.path.abspath(ROOT/target) if windows else str((ROOT/target).resolve())
    print('BACKEND',backend,'TARGET',env['CARGO_TARGET_DIR'],flush=True)
    if cuda is not None and backend=='cuda':
        cuda=cuda.resolve(strict=not args.print_commands)
        env['CUDA_PATH']=str(cuda);env['PPF_CUDA_ROOT']=str(cuda)
        env['PATH']=str(cuda/'bin')+os.pathsep+env['PATH']
    if backend=='cuda' and args.phase in ('all','rust') and not args.print_commands:
        require_cuda_guard_tools(env)
        env['PPF_FORMAL_CUDA_GUARD']='1'
    generator=ROOT/'coordination/windows-resume-20261004/restoring-curve/native-scalar/generate_identity.py'
    checked([python,'-B',generator,'--check'],env)
    if args.rocm_root:
        env['ROCM_PATH']=str(args.rocm_root.resolve(strict=not args.print_commands))
        env['HIP_PATH']=env['ROCM_PATH']
    # Unix Cargo invokes the selected upstream Makefile inside its own OUT_DIR.
    if not windows and args.phase=='backend' and backend in ('cuda','metal','rocm'):
        checked(['make','-C',ROOT/f'crates/ppf-cts-compute/{backend}','abi',f'-j{max(1,args.jobs)}',f'PYTHON={python}'],env)
    if windows and backend=='cuda' and args.phase in ('all','backend'):
        if not args.print_commands and not shutil.which('cl.exe',path=env['PATH']):
            raise SystemExit('Open a Visual Studio x64 developer console before running this command.')
        if cuda is None:raise SystemExit('Name CUDA 12.8 with --cuda-root or CUDA_PATH.')
        nvcc=cuda/'bin/nvcc.exe'
        if not args.print_commands:
            version=subprocess.check_output([str(nvcc),'--version'],env=env,text=True,errors='replace')
            if 'release 12.8,' not in version:raise SystemExit('This source requires CUDA 12.8; selected compiler: '+version)
        cpp=ROOT/'crates/ppf-cts-compute/cuda'
        output=cpp/'build'
        env.update(BUILD_WIN=str(ROOT/'build-win-native'),SRC_DIR=str(ROOT),CPP_DIR=str(cpp),
                   KERNEL_DIR=str(ROOT/'crates/ppf-cts-solver/src/kernels'),
                   KERNELGEN=str(ROOT/'crates/ppf-cts-compute/seam/kernelgen.py'),
                   OUT_DIR=str(output),KERNELGEN_DIR=str(output/'kernelgen'),LIB_DIR=str(output/'lib'),
                   PYTHON_EXE=str(python))
        checked([env.get('COMSPEC','cmd.exe'),'/d','/c','call',str(ROOT/'build-win-native/scripts/build-cuda.bat')],env)
    if windows and backend=='rocm' and args.phase in ('all','backend'):
        rocm=args.rocm_root or (Path(env['ROCM_PATH']) if env.get('ROCM_PATH') else None)
        if rocm is None:raise SystemExit('Select the ROCm SDK with --rocm-root or ROCM_PATH; no CUDA fallback is used.')
        checked([python,ROOT/'build-win-native/scripts/build_rocm.py','--platform','amd',
                 '--kernel-root',ROOT/'crates/ppf-cts-solver/src/kernels',
                 '--rocm-dir',ROOT/'crates/ppf-cts-compute/rocm',
                 '--cuda-prologue-dir',ROOT/'crates/ppf-cts-compute/cuda',
                 '--kernelgen',ROOT/'crates/ppf-cts-compute/seam/kernelgen.py',
                 '--gen-def',ROOT/'build-win-native/scripts/gen_def.py',
                 '--out-dir',ROOT/'crates/ppf-cts-compute/rocm/build',
                 '--hip-include',rocm/'include','--rocm-path',rocm,'--python',python,'--jobs',max(1,args.jobs)],env)
    if args.phase in ('all','rust','services') or (args.phase=='backend' and backend=='cpu'):
        argv=['cargo','build','--release','--locked','--jobs',str(args.jobs),'--manifest-path',ROOT/'Cargo.toml']
        if args.phase=='services':
            argv+=['-p','ppf-cts-server','-p','ppf-cts-py']
            if backend=='cpu':argv+=['--features','ppf-cts-server/cpu']
            elif backend=='rocm':argv+=['-p','ppf-cts-core','--features','ppf-cts-core/rocm']
        else:
            if backend!='auto':argv+=['--features',backend]
        if args.offline:argv+=['--offline']
        checked(argv,env)

if __name__=='__main__':main()

