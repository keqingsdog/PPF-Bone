"""Load the standalone shared display reference builder without importing bpy."""
import importlib.util
import sys
from pathlib import Path
_NAME='_ppf_fem_scale_shared_reference'
if _NAME not in sys.modules:
    _path=Path(__file__).resolve().parents[1]/'blender_addon/core/fem_scale.py'
    _spec=importlib.util.spec_from_file_location(_NAME,_path)
    _module=importlib.util.module_from_spec(_spec);sys.modules[_NAME]=_module
    _spec.loader.exec_module(_module)
build_reference=sys.modules[_NAME].build_reference
