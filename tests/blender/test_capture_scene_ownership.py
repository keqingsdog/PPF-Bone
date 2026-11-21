"""Isolated regression retained from the 2026-10-25 repair.

Run in a separate factory process:
  blender --background --factory-startup --python-exit-code 1 --python <this file>
Creates only a private temporary test project; no solver or network is required.
"""
from pathlib import Path
import bpy,sys,json,numpy as np
import tempfile
S=Path(__file__).resolve().parents[2]
W=Path(tempfile.mkdtemp(prefix='ppf-scene-regression-'))
bpy.context.preferences.filepaths.temporary_directory=str(W)
sys.path.insert(0,str(S));import blender_addon;blender_addon.register()
from blender_addon.core import pc2,rigid_pose
from blender_addon.models.groups import get_addon_data
try:
 bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
 a=bpy.context.scene;a.name='A';b=bpy.data.scenes.new('B')
 objects=[]
 for scene in (a,b):
  mesh=bpy.data.meshes.new(scene.name+'mesh');mesh.from_pydata([(0,0,0),(1,0,0),(0,1,0)],[],[(0,1,2)]);obj=bpy.data.objects.new(scene.name+'object',mesh);scene.collection.objects.link(obj);obj.vertex_groups.new(name='Pin');objects.append(obj)
 bpy.context.window.scene=a;bpy.ops.wm.save_as_mainfile(filepath=str(W/'two-scenes.blend'),check_existing=False)
 keys={};bones={}
 for scene,obj in zip((a,b),objects):
  pc2.write_static_deform_pc2(obj,np.zeros((2,3,3)));pc2.write_pin_anim_pc2(obj,'Pin',np.ones((2,3,3)))
  keys[scene.name]=[pc2.static_deform_pc2_key(obj),pc2.pin_anim_pc2_key(obj,'Pin')]
  index=Path(pc2.get_pc2_dir())/'bone_poses'/(('a' if scene.name=='A' else 'b')*64)/'index.json';index.parent.mkdir(parents=True,exist_ok=True);index.write_text(json.dumps({'receipt':{'scene_inputs_sha256':('a' if scene.name=='A' else 'b')*64},'test':'independent bone index control','scene':scene.name}));scene[rigid_pose.CACHE_KEY]=str(index);rigid_pose._cache[str(index)]={'receipt':{'scene_inputs_sha256':('a' if scene.name=='A' else 'b')*64},'test_control':scene.name};get_addon_data(scene).state.last_session_id='session-'+scene.name;bones[scene.name]=str(index)
 bpy.ops.wm.save_as_mainfile(filepath=str(W/'two-scenes.blend'),check_existing=False)
 originalB=Path(bones['B']).read_text();result=bpy.ops.solver.clear_all_deformations()
 out={'operator':list(result),'A':{'files':[Path(pc2.get_pc2_path(k)).exists() for k in keys['A']],'memory':[keys['A'][0] in pc2._static_deform_cache,keys['A'][1] in pc2._pin_anim_cache]},'B':{'files':[Path(pc2.get_pc2_path(k)).exists() for k in keys['B']],'memory':[keys['B'][0] in pc2._static_deform_cache,keys['B'][1] in pc2._pin_anim_cache],'bone_file_unchanged':Path(bones['B']).read_text()==originalB,'bone_memory_unchanged':rigid_pose._cache.get(bones['B'],{}).get('test_control')=='B','bone_scene_index_unchanged':b.get(rigid_pose.CACHE_KEY)==bones['B'],'session_state':get_addon_data(b).state.last_session_id},'scope':'saved private .blend A/B real static/pin PC2 files+arrays; bone index/memory/state preservation controls, no bone playback or native solver'}
 assert out['A']['files']==[False,False] and out['A']['memory']==[False,False]
 assert out['B']['files']==[True,True] and out['B']['memory']==[True,True]
 assert all(out['B'][name] for name in ('bone_file_unchanged','bone_memory_unchanged','bone_scene_index_unchanged')) and out['B']['session_state']=='session-B'
 # One focused ownership batch: memory-only, tombstone, unknown, shared and copied record.
 def mesh_object(name,scenes):
  mesh=bpy.data.meshes.new(name+'mesh');mesh.from_pydata([(0,0,0)],[],[]);obj=bpy.data.objects.new(name,mesh)
  for scene in scenes:scene.collection.objects.link(obj)
  obj.vertex_groups.new(name='Pin');return obj
 mem=mesh_object('MemoryOnlyA',(a,));pc2.write_static_deform_pc2(mem,np.zeros((2,1,3)));mk=pc2.static_deform_pc2_key(mem);Path(pc2.get_pc2_path(mk)).unlink()
 orphan=mesh_object('DeletedOwnerA',(a,));pc2.write_static_deform_pc2(orphan,np.zeros((2,1,3)));ok=pc2.static_deform_pc2_key(orphan);bpy.data.objects.remove(orphan,do_unlink=True)
 shared=mesh_object('SharedAB',(a,b));pc2.write_static_deform_pc2(shared,np.zeros((2,1,3)));pc2.write_pin_anim_pc2(shared,'Pin',np.ones((2,1,3)));sk=pc2.static_deform_pc2_key(shared);pk=pc2.pin_anim_pc2_key(shared,'Pin')
 # Reach the same guarded helper via the real operator's per-object STATIC path.
 from blender_addon.models.groups import iterate_object_groups
 group=next(iterate_object_groups(a));group.active=True;group.object_type='STATIC';entry=group.assigned_objects.add();entry.uuid=shared.get('_solver_uuid');entry.name=shared.name;entry.included=True
 assert pc2.remove_static_deform_pc2(shared,a) is False and pc2.remove_pin_anim_pc2(shared,'Pin',a) is False
 copied=mesh_object('CopiedRecordA',(a,));pc2.write_static_deform_pc2(copied,np.zeros((2,1,3)));ck=pc2.static_deform_pc2_key(copied);records=pc2._capture_records(b);records[ck]=copied.get('_solver_uuid');b[pc2._CAPTURE_OWNERS]=json.dumps(records)
 unknown=Path(pc2.get_pc2_path(objects[0].get('_solver_uuid')+'_unknown_staticdeform'));pc2.write_pc2(str(unknown),[np.zeros((1,3))])
 assert bpy.ops.solver.clear_all_deformations()=={'FINISHED'}
 assert mk not in pc2._static_deform_cache and not Path(pc2.get_pc2_path(ok)).exists() and ok not in pc2._static_deform_cache
 assert all(Path(pc2.get_pc2_path(k)).exists() for k in (sk,pk,ck)) and sk in pc2._static_deform_cache and pk in pc2._pin_anim_cache and ck in pc2._static_deform_cache
 assert unknown.exists()
 # Redraw poll skips Pin-group work and catches a legacy file that appears
 # or disappears externally. A modifier in a different scene still protects it.
 from unittest.mock import patch
 with patch.object(pc2, '_safe_vg_for_key', side_effect=AssertionError('poll walked Pin keys')):
  assert not pc2.scene_has_static_deform_cache(a)
  assert pc2.scene_has_static_deform_cache(b)
  legacy_key = keys['A'][0]
  assert legacy_key not in pc2._capture_records(a)
  legacy_path = Path(pc2.get_pc2_path(legacy_key))
  pc2.write_pc2(str(legacy_path), [np.zeros((3, 3))])
  assert pc2.scene_has_static_deform_cache(a)
  reference = objects[1].modifiers.new('Foreign cache reference', 'MESH_CACHE')
  reference.filepath = str(legacy_path)
  assert not pc2.scene_has_static_deform_cache(a)
  objects[1].modifiers.remove(reference)
  assert pc2.scene_has_static_deform_cache(a)
  legacy_path.unlink()
  assert not pc2.scene_has_static_deform_cache(a)
 # A denied unlink keeps its memory and persistent exact-owner record.
 denied=mesh_object('DeniedA',(a,));pc2.write_static_deform_pc2(denied,np.zeros((2,1,3)));dk=pc2.static_deform_pc2_key(denied)
 with patch.object(pc2.os,'remove',side_effect=PermissionError('fixture refusal')):assert pc2.remove_static_deform_pc2(denied,a) is False
 assert dk in pc2._static_deform_cache and dk in pc2._capture_records(a) and Path(pc2.get_pc2_path(dk)).exists()
 out['focused_ownership']={'memory_only_A_cleared':True,'recorded_deleted_owner_A_cleared':True,'shared_A_B_preserved_via_operator_and_per_object_helpers':True,'copied_scene_record_conflict_preserved':True,'unknown_prefix_file_preserved':True,'unlink_failure_retains_memory_and_owner_record':True}
 (W/'clear-after.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
finally:blender_addon.unregister()


