import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

def load(rel, name):
    spec=importlib.util.spec_from_file_location(name, ROOT / rel)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

class PublicProductPaths(unittest.TestCase):
    def test_both_profiles_legacy_portable_configuration_and_refusals(self):
        for index, rel in enumerate(('frontend/_r14_product_profile.py','blender_addon/core/r14_product_profile.py')):
            m=load(rel, 'public_profile_'+str(index))
            with self.subTest(module=rel), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve(); (root/'runtime'/'python').mkdir(parents=True)
                (root/'cuda'/'bin').mkdir(parents=True)
                python=root/'runtime'/'python'/'python.exe';python.touch()
                product={'cuda_runtime_bin':'cuda/bin'}
                with patch.dict(os.environ, {}, clear=True):
                    cuda, found=m._legacy_runtime_paths(root,product)
                    self.assertEqual(cuda,root/'cuda'/'bin');self.assertEqual(found,python)
                    python.unlink()
                    with self.assertRaisesRegex(ValueError,'PPF_PRODUCT_PYTHON_REQUIRED'):
                        m._legacy_runtime_paths(root,product)
                    alt=root/'alternate.exe';alt.touch()
                    with patch.dict(os.environ,{'PPF_CTS_PRODUCT_PYTHON':'alternate.exe','PPF_CTS_PRODUCT_CUDA_BIN':'cuda/bin'}):
                        self.assertEqual(m._legacy_runtime_paths(root,product),(root/'cuda'/'bin',alt))
                    with self.assertRaisesRegex(ValueError,'PPF_PRODUCT_CUDA_REQUIRED'):
                        m._legacy_runtime_paths(root,{'cuda_runtime_bin':'missing/bin'})
                with self.assertRaisesRegex(ValueError,'allowlist'):
                    m._identity(root,{'product':'unknown'})

    def test_metadata_paths_fail_closed_in_both_entries(self):
        for index, rel in enumerate(('frontend/_r14_product_profile.py','blender_addon/core/r14_product_profile.py')):
            m=load(rel, 'metadata_profile_'+str(index))
            good={'launch_working_directory':r'C:\old\runtime', 'cuda_runtime_bin':r'C:\old\cuda\bin',
                  'required_path_prefix':[r'C:\old\runtime',r'C:\old\cuda\bin']}
            m._validate_legacy_paths(good)
            for bad in [dict(good,required_path_prefix=good['required_path_prefix']+['extra']),
                        dict(good,launch_working_directory=r'C:\old\elsewhere'),
                        dict(good,launch_working_directory=r'C:\old\..\runtime'),
                        dict(good,required_path_prefix=[r'C:\other\runtime',good['cuda_runtime_bin']])]:
                with self.assertRaises(ValueError): m._validate_legacy_paths(bad)

    def test_no_selected_product_does_not_select_old_default(self):
        for index, rel in enumerate(('frontend/_r14_product_profile.py','blender_addon/core/r14_product_profile.py')):
            m=load(rel, 'unselected_profile_'+str(index))
            with patch.dict(os.environ, {}, clear=True), patch.object(m, 'local_source_module') as local:
                local.return_value.CONFIG='nonexistent-profile-config.json'
                with self.assertRaisesRegex(ValueError,'PRODUCT_SELECTION_REQUIRED'):
                    m.load_profile()

if __name__ == '__main__': unittest.main()
