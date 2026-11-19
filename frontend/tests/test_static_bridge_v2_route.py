"""Focused v2 static-bridge schema and routing checks; no native backend."""
from copy import deepcopy
import importlib, sys, types, unittest
from unittest.mock import patch
from pathlib import Path
root = Path(__file__).resolve().parents[2]
pkg = types.ModuleType('_bridge_test_frontend'); pkg.__path__ = [str(root/'frontend')]; sys.modules[pkg.__name__] = pkg
bridge = importlib.import_module(pkg.__name__+'._automatic_parent_bridge_v2_')
build = importlib.import_module(pkg.__name__+'._automatic_parent_anchor_build_')
I = [[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]
def uid(n): return f'{n:08x}-0000-4000-8000-000000000000'
def fixture():
    ancestor = dict(body_uuid=uid(1),bone_uuid=uid(2),armature_uuid=uid(3),bone_name='ancestor',enabled=True,bone_world_matrix_blender=I,proxy_world_matrix_blender=I)
    child = dict(body_uuid=uid(4),bone_uuid=uid(5),armature_uuid=uid(3),enabled=True,root_mode='FREE',parent_body_uuid=None,parent_bone_uuid=None,root_motion=None,root_boundary={'is_root':True,'nonphysical_parent_bone':'middle'},use_auto_frames=True,
      automatic_parent_anchor_v1={'version':1,'anchor_body_uuid':uid(6),'parent_bone_uuid':uid(7),'parent_bone_name':'middle','parent_world_matrix_blender':I,'anchor_proxy_world_matrix_blender':I,'physical_ancestor_bridge_v1':{'version':1,'ancestor_body_uuid':uid(1),'ancestor_bone_uuid':uid(2),'ancestor_bone_name':'ancestor','ancestor_to_parent_blender':I,'chain':[{'name':'middle','parent_name':'ancestor','matrix_basis':I,'matrix_local':I,'inherit_scale':'FULL','use_inherit_rotation':True}]}})
    return [ancestor, child]
class StaticBridge(unittest.TestCase):
    def test_valid_reparent_and_original_untouched(self):
        rows=fixture(); rows[1]['physics']={'translation_constraint_v1':{'mode':'FIXED'}}; original=deepcopy(rows); bridge.validate_bridges(rows); prepared=deepcopy(rows); build._prepare_static_bridges(prepared)
        self.assertEqual(rows,original); self.assertEqual(prepared[1]['parent_body_uuid'],uid(1)); self.assertNotIn('automatic_parent_anchor_v1',prepared[1]); self.assertEqual(prepared[1]['physics']['translation_constraint_v1'],{'mode':'FIXED'})
    def test_dispatches_v2_bridge_only_and_leaves_v3_path(self):
        contracts=importlib.import_module(pkg.__name__+'._build_contract_')
        data={'blender_bones_v1':{'bindings':fixture()}}
        contract=types.SimpleNamespace(producer_version=2)
        with patch.object(contracts,'contract_for',return_value=contract), patch.object(contracts,'validate_requests'), patch.object(bridge,'prepare',return_value='v2 bridge') as selected:
            self.assertEqual(build.prepare(object(),data),'v2 bridge')
            selected.assert_called_once()
        contract.producer_version=3
        empty={'blender_bones_v1':{'bindings':[]}}
        with patch.object(contracts,'contract_for',return_value=contract), patch.object(contracts,'validate_requests'), patch.object(bridge,'prepare') as selected:
            self.assertIs(build.prepare(object(),empty),empty)
            selected.assert_not_called()
    def test_invalid_metadata_refused(self):
        for mutate in (lambda r:r[1]['automatic_parent_anchor_v1'].update(version=True), lambda r:r[1]['automatic_parent_anchor_v1']['physical_ancestor_bridge_v1']['chain'][0].update(parent_name='wrong'), lambda r:r[1].update(root_mode='KINEMATIC'), lambda r:r[1]['automatic_parent_anchor_v1']['physical_ancestor_bridge_v1'].update(ancestor_body_uuid=uid(99)), lambda r:r[1]['automatic_parent_anchor_v1']['physical_ancestor_bridge_v1']['chain'][0].update(matrix_basis=[[0]])):
            with self.subTest(mutate=mutate):
                rows=fixture(); mutate(rows)
                with self.assertRaises(Exception): bridge.validate_bridges(rows)
    def test_initial_relative_mismatch_refused(self):
        rows=fixture(); rows[1]['automatic_parent_anchor_v1']['physical_ancestor_bridge_v1']['ancestor_to_parent_blender']=deepcopy(I); rows[1]['automatic_parent_anchor_v1']['physical_ancestor_bridge_v1']['ancestor_to_parent_blender'][0][3]=1
        bridge.validate_bridges(rows)
        with self.assertRaises(Exception): build._prepare_static_bridges(rows)
if __name__=='__main__': unittest.main()
