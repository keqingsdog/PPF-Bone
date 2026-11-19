"""Formal CUDA guard tool preflight; inert tools only, no native compilation."""
import importlib.util
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('build_source',Path(__file__).with_name('build_source.py'))
build=importlib.util.module_from_spec(spec);spec.loader.exec_module(build)

class GuardPreflightTests(unittest.TestCase):
    def test_missing_tool_and_bypass_refuse(self):
        with patch.object(build.shutil,'which',return_value=None),patch.object(build.subprocess,'run') as run:
            with self.assertRaisesRegex(SystemExit,'require cuobjdump'):
                build.require_cuda_guard_tools({'PATH':''})
            run.assert_not_called()
        with self.assertRaisesRegex(SystemExit,'cannot use PPF_ALLOW_FP64'):
            build.require_cuda_guard_tools({'PPF_ALLOW_FP64':'','PATH':''})

    def test_each_tool_must_run_successfully(self):
        with patch.object(build.shutil,'which',side_effect=lambda name,**kw:'/tools/'+name), \
             patch.object(build.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'version','')) as run:
            build.require_cuda_guard_tools({'PATH':'/tools'})
            self.assertEqual([call.args[0] for call in run.call_args_list],
                             [['/tools/cuobjdump','--version'],['/tools/nvdisasm','--version']])
            self.assertTrue(all(call.kwargs['check'] for call in run.call_args_list))
        with patch.object(build.shutil,'which',return_value='/tools/broken'), \
             patch.object(build.subprocess,'run',side_effect=OSError('not executable')):
            with self.assertRaisesRegex(SystemExit,'not runnable'):
                build.require_cuda_guard_tools({'PATH':''})

    def test_only_formal_cuda_execution_requires_tools(self):
        # Exercise the actual main control flow with all build commands inert.
        for backend,phase,planned,wanted in [('cpu','all',False,False),('cuda','backend',False,False),
                ('cuda','rust',False,True),('cuda','all',False,True),('cuda','all',True,False)]:
            with self.subTest(backend=backend,phase=phase,planned=planned):
                argv=['build_source.py','--backend',backend,'--phase',phase]
                if backend=='cuda':argv+=['--cuda-root',str(Path(__file__).parent)]
                if planned:argv+=['--print-commands']
                with patch.dict(os.environ,{},clear=True),patch.object(build.sys,'argv',argv),patch.object(build,'checked') as checked, \
                     patch.object(build,'require_cuda_guard_tools') as preflight, \
                     patch.object(build.shutil,'which',return_value='tool'), \
                     patch.object(build.subprocess,'check_output',return_value='release 12.8,'):
                    build.main()
                    self.assertEqual(preflight.called,wanted)
                    for call in checked.call_args_list:
                        self.assertEqual(call.args[1].get('PPF_FORMAL_CUDA_GUARD'), '1' if wanted else None)

if __name__=='__main__':unittest.main(verbosity=2)
