"""Focused backend dispatch and extracted link-command tests; no native compilation."""
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('source_build',ROOT/'tools/build_source.py')
build=importlib.util.module_from_spec(spec);spec.loader.exec_module(build)
real_is_file=Path.is_file

def no_cuda_file(path):
    return False if path.as_posix()=='/usr/local/cuda/bin/nvcc' else real_is_file(path)

class BackendDispatchTests(unittest.TestCase):
    def invoke_without_sdk(self,extra):
        commands=[]
        self.commands=commands
        with patch.dict(os.environ,{},clear=True),patch.object(build.shutil,'which',return_value=None), \
             patch.object(Path,'is_file',no_cuda_file), \
             patch.object(sys,'argv',['build_source.py','--phase','backend',*extra]), \
             patch.object(build,'checked',lambda args,env: commands.append(([str(value) for value in args],env.copy()))):
            build.main()
        return commands

    def test_no_cuda_auto_refuses_and_explicit_cpu_dispatches(self):
        with self.assertRaisesRegex(SystemExit,'select --backend cpu explicitly'):
            self.invoke_without_sdk(['--platform','linux','--print-commands'])
        self.assertEqual(self.commands,[],'auto must not report identity-only success')
        commands=self.invoke_without_sdk(['--backend','cpu'])
        cargo=[(args,env) for args,env in commands if args[0]=='cargo']
        self.assertEqual(len(cargo),1,'backend phase must schedule a backend command beyond identity')
        args,env=cargo[0]
        self.assertNotEqual(env.get('PPF_SOURCE_PRINT_ONLY'),'1','normal execution must dispatch the CPU build')
        self.assertEqual(args[args.index('--features')+1],'cpu')
        self.assertTrue(Path(env['CARGO_TARGET_DIR']).as_posix().endswith('/target/cpu'))
        self.assertFalse(any(args[0]=='make' for args,_env in commands))

    def test_explicit_unsupported_backend_refuses_before_any_command(self):
        with self.assertRaisesRegex(SystemExit,'CUDA backend requested but no CUDA SDK'):
            self.invoke_without_sdk(['--backend','cuda'])
        self.assertEqual(self.commands,[])
        # The same early refusal applies to a requested hypothetical command
        # plan unless the caller supplies a CUDA root for that plan.
        with self.assertRaisesRegex(SystemExit,'CUDA backend requested but no CUDA SDK'):
            self.invoke_without_sdk(['--backend','cuda','--platform','linux','--print-commands'])
        self.assertEqual(self.commands,[])
        with self.assertRaisesRegex(SystemExit,'Metal backend requires macOS'):
            self.invoke_without_sdk(['--backend','metal','--platform','windows','--print-commands'])
        self.assertEqual(self.commands,[])

    def test_cuda_link_recipe_uses_one_worker(self):
        makefile=(ROOT/'crates/ppf-cts-compute/cuda/Makefile').read_text(encoding='utf-8')
        recipe=next(line.strip() for line in makefile.splitlines() if line.startswith('\t$(NVCC) -shared -dlto '))
        command=recipe.removeprefix('$(NVCC) ').replace('$(CUDA_GENCODE)','-gencode arch=compute_61,code=sm_61')
        command=command.replace('$(filter-out $(CUDA_ARCH_FILE),$^)','mechanism.o entry.o').replace('$@','backend.so')
        # Run the exact extracted argument sequence through an inert compiler
        # stand-in. This checks command construction, not a Linux CUDA build.
        with tempfile.TemporaryDirectory(prefix='ppf-link-command-') as folder:
            compiler=Path(folder)/'nvcc_stub.py'
            compiler.write_text('import json,sys;print(json.dumps(sys.argv[1:]))\n',encoding='utf-8')
            result=subprocess.run([sys.executable,'-B','-X','utf8',str(compiler),*shlex.split(command)],
                                  capture_output=True,text=True,encoding='utf-8',check=True)
        args=json.loads(result.stdout)
        self.assertEqual(args[args.index('-t')+1],'1')
        self.assertIn('-dlto',args)
        windows=(ROOT/'build-win-native/scripts/build-cuda.bat').read_text(encoding='utf-8')
        self.assertIn('set NVCC_COMMON=-t 1 ',windows)
        self.assertIn('-shared -dlto -t 1 ',windows)

if __name__=='__main__':unittest.main(verbosity=2)
