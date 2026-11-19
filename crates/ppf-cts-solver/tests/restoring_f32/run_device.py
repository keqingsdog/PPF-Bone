"""Real GPU readback and active-restoring Schwarz body export regression.

Run from a CUDA/C++ developer shell after normal kernelgen has populated
ppf-cts-compute/cuda/build/kernelgen. Requires nvcc and a CUDA device matching
--arch. This small executable does not replace the full backend SASS guard.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda-root',type=Path)
    parser.add_argument('--generated-root',type=Path)
    parser.add_argument('--arch',default='89')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();here=Path(__file__).resolve().parent
    compute=here.parents[1].parent/'ppf-cts-compute/cuda'
    generated=args.generated_root or compute/'build/kernelgen'
    nvcc=args.cuda_root/'bin'/('nvcc.exe' if os.name=='nt' else 'nvcc') if args.cuda_root else shutil.which('nvcc')
    if not nvcc:raise SystemExit('nvcc is required')
    output=args.out.resolve();output.mkdir(parents=True,exist_ok=True)
    exe=output/'channels_probe.exe'
    command=[str(nvcc),'-t','1','-std=c++17','-O3','-arch=sm_'+args.arch,
             '--expt-relaxed-constexpr','--extended-lambda',
             '-I'+str(generated),'-I'+str(compute),'-I'+str(here.parents[1]/'src/kernels'),
             str(here/'device_readback.cu'),'-o',str(exe)]
    if os.name=='nt':command+=['-Xcompiler','/EHsc,/MD,/O2']
    subprocess.run(command,check=True,cwd=output)
    env=os.environ.copy();env['PPF_RESTORING_TEST_BUILD']=str(output);env['PPF_RESTORING_TEST_DEVICE']='1'
    subprocess.run([sys.executable,'-B',str(here/'test_channels.py')],env=env,check=True)
    result=subprocess.run([str(exe),'schwarz'],capture_output=True,text=True)
    (output/'SCHWARZ_RESULT.json').write_text(json.dumps({'exit_code':result.returncode,
        'stdout':result.stdout,'stderr':result.stderr,'active_curves':True,
        'scope':'real CUDA readback of current generated neutral Schwarz bodies; not a solver scene'},indent=2),encoding='utf-8')
    print(result.stdout,result.stderr,end='')
    result.check_returncode()
    print('PASS GPU readback / active restoring legacy+reduced Schwarz export')

if __name__=='__main__':main()
