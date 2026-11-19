"""Compiled host f32 arithmetic versus independent Decimal200 reference.

Run in a C++ developer shell: python -B run.py [--cxx cl|c++].
No CUDA SDK/GPU is used; these results do not replace the device SASS guard.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx',default=os.environ.get('CXX','cl' if os.name=='nt' else 'c++'))
    parser.add_argument('--out',type=Path,help='Keep private test binaries and numeric result JSON in this directory.')
    args=parser.parse_args();here=Path(__file__).resolve().parent
    include=here.parents[1]/'src/kernels'
    def run(folder):
        folder.mkdir(parents=True,exist_ok=True)
        for name in ('coordinate_probe','channels_probe'):
            if Path(args.cxx).stem.lower() in ('cl','clang-cl'):
                command=[args.cxx,'/nologo','/EHsc','/std:c++17','/fp:strict','/I'+str(include),
                         '/Fe:'+str(folder/(name+'.exe')),'/Fo:'+str(folder/(name+'.obj')),str(here/(name+'.cpp'))]
            else:
                command=[args.cxx,'-std=c++17','-O2','-ffp-contract=off','-I'+str(include),str(here/(name+'.cpp')),'-o',str(folder/(name+'.exe'))]
            subprocess.run(command,cwd=folder,check=True)
        subprocess.run([str(folder/'coordinate_probe.exe')],check=True)
        env=os.environ.copy();env['PPF_RESTORING_TEST_BUILD']=str(folder)
        for name in ('test_coordinate.py','test_channels.py'):
            subprocess.run([sys.executable,'-B',str(here/name)],env=env,check=True)
    if args.out:run(args.out.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix='ppf-restoring-f32-') as folder:run(Path(folder))

if __name__=='__main__':main()
