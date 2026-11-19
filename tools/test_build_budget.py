"""Command-construction checks; does not claim native Unix compilation."""
import importlib.util
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('source_build', ROOT/'tools/build_source.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
count = 0
for jobs in (1, 3):
    for platform, backend in (('linux','cuda'),('macos','metal'),('linux','rocm'),('windows','cuda')):
        for phase in ('backend','all'):
            commands = []
            argv = ['build_source.py','--print-commands','--platform',platform,
                    '--backend',backend,'--phase',phase,'--jobs',str(jobs),
                    '--cuda-root',str(ROOT/'mock-cuda'),'--rocm-root',str(ROOT/'mock-rocm')]
            with patch.object(sys,'argv',argv), patch.dict(os.environ,{},clear=True), \
                 patch.object(module,'checked',lambda args,env: commands.append(([str(x) for x in args],env.copy()))):
                module.main()
            for args, env in commands:
                assert env['CARGO_BUILD_JOBS'] == str(jobs)
                assert env['PPF_WIN_BUILD_JOBS'] == str(jobs)
                assert env['PPF_DEVICE_BUILD_JOBS'] == '1'
                if args[0] == 'cargo':
                    assert args[args.index('--jobs')+1] == str(jobs)
                if args[0] == 'make':
                    assert '-j'+str(jobs) in args
            count += 1
rs = (ROOT/'crates/ppf-cts-solver/build.rs').read_text(encoding='utf-8')
assert 'let num_threads = num_cpus::get();' not in rs
assert 'cargo:rerun-if-env-changed=PPF_DEVICE_BUILD_JOBS' in rs
assert '.unwrap_or(1);' in rs
assert 'make.env_remove("MAKEFLAGS")' in rs
bat = (ROOT/'build-win-native/scripts/build-cuda.bat').read_text(encoding='utf-8')
assert 'set NVCC_COMMON=-t 1 ' in bat
assert '-shared -dlto -t 1 ' in bat
assert '-t 0 ' not in bat
with patch.object(sys,'argv',['build_source.py','--jobs','0']):
    try:module.main()
    except SystemExit as error:assert error.code == 2
    else:raise AssertionError('zero jobs accepted')
print(f'PASS {count} command plans; nested make default and CUDA threads inspected; zero jobs rejected')
