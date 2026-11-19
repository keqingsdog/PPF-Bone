import os, bpy, sys, pathlib, json, hashlib, uuid, shutil
ROOT=pathlib.Path(os.environ['PPF_SAVEAS_TEST_ROOT'])
sys.path.insert(0,os.environ['PPF_SAVEAS_SOURCE'])
sys.path[:0]=[str(ROOT),str(ROOT/'test_deps')]
import blender_addon as addon
import cbor2
from blender_addon.core import rigid_pose as r
from mathutils import Matrix
addon.register()
OUT=ROOT/'run_integrated';OUT.mkdir(exist_ok=True)
results=[]
def check(label, ok):
 results.append([label,bool(ok)]); print('CHECK',label,bool(ok)); assert ok,label

def setup(scene,name):
 bpy.context.window.scene=scene
 bpy.ops.object.armature_add(); arm=bpy.context.object;arm.name=name
 aid,bid,body=[str(uuid.uuid4()) for _ in range(3)]
 arm.ppf_armature_uuid=aid;pb=arm.pose.bones[0];settings=pb.bone.ppf_bone_physics
 settings.bone_uuid=bid;settings.body_uuid=body;settings.enabled=True
 row={'armature_uuid':aid,'bone_uuid':bid,'body_uuid':body,'proxy_object_uuid':body,'runtime_body_index':0,'body_to_bone_initial_solver':list(map(list,Matrix.Identity(4))),'bone_initial_solver':list(map(list,Matrix.Identity(4))),'bone_name':pb.name}
 receipt={'schema':'PPF_BLENDER_BONES_V1','scene_uuid':str(uuid.uuid4()),'scene_inputs_sha256':hashlib.sha256(name.encode()).hexdigest(),'start_frame':1,'fps':24.,'world_scaling':1,'bindings':[row]}
 for f in (0,1):
  scene.frame_set(f+1)
  pose={'version':1,'kind':'PdrdFramePoseMeasurement','measurement':'read_only_native_pdrd_fit_of_vertex_frame','coordinate_system':'solver_y_up_meters','constrained_root_state_authority':'not_reported','scene_uuid':receipt['scene_uuid'],'scene_inputs_sha256':receipt['scene_inputs_sha256'],'frame_id':f,'vertex_frame_sha256':'a'*64,'vertex_frame_byte_length':48,'frame_time_seconds':f/24.,'accepted_bracket_start_seconds':0.,'accepted_solver_time_seconds':f/24.,'source':'InitialFrame' if not f else 'AcceptedEndpoint','accepted_steps':f,'bodies':[{'body_uuid':body,'runtime_body_index':0,'vertex_count':4,'rotation_column_major':[1.,0.,0.,0.,1.,0.,0.,0.,1.],'centroid_m':[float(f+2),0.,0.]}]}
  raw=cbor2.dumps(pose);r.accept_frame(scene,{'receipt':receipt,'pose':pose,'native_bytes':raw})
 return arm

bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'source.blend'))
a=setup(bpy.context.scene,'A');s1=bpy.context.scene
s2=bpy.data.scenes.new('B');b=setup(s2,'B')
old={s.name:s[r.CACHE_KEY] for s in (s1,s2)}
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'source.blend'))
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'另存 测试.blend'))
new={s.name:s[r.CACHE_KEY] for s in (s1,s2)}
check('references changed before save',all(new[n]!=old[n] and '另存 测试' in pathlib.Path(new[n]).parts for n in new))
shutil.move(OUT/'data/source',OUT/'source_cache_parked')
# Cold verification occurs in separate Blender process below.
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'copy.blend'),copy=True)
check('save copy session refs restored',all(s[r.CACHE_KEY]==new[s.name] for s in (s1,s2)))
check('save copy session filepath restored',bpy.data.filepath==str(OUT/'另存 测试.blend'))
# Partial target is retained and warns, rather than overwritten.
partial=OUT/'data/partial/bone_poses'/pathlib.Path(new[s1.name]).parent.name
partial.mkdir(parents=True);(partial/'sentinel').write_text('keep')
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'partial.blend'))
check('partial destination warning',all(s.get(r.SAVE_WARNING_PROP) for s in (s1,s2)))
check('partial keeps old refs',all(s[r.CACHE_KEY]==new[s.name] for s in (s1,s2)))
check('partial sentinel intact',(partial/'sentinel').read_text()=='keep')
# Fresh target succeeds and clears warning.
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'recovered.blend'))
check('new target clears warning',all(not s.get(r.SAVE_WARNING_PROP) for s in (s1,s2)))
# Corrupt B only, verify all-source prevalidation leaves A untouched.
key=pathlib.Path(s2[r.CACHE_KEY]);p=key.parent/'pose_1.cbor';raw=p.read_bytes();p.write_bytes(b'corrupt')
prior={s.name:s[r.CACHE_KEY] for s in (s1,s2)}
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'badsource.blend'))
check('bad B leaves both refs',all(s[r.CACHE_KEY]==prior[s.name] for s in (s1,s2)))
check('bad B creates no target cache',not (OUT/'data/badsource/bone_poses').exists())
check('bad B warns both',all(s.get(r.SAVE_WARNING_PROP) for s in (s1,s2)))
p.write_bytes(raw)
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'badsource.blend'))
check('same target retry after prevalidation failure',all('badsource' in pathlib.Path(s[r.CACHE_KEY]).parts and not s.get(r.SAVE_WARNING_PROP) for s in (s1,s2)))
(OUT/'results.json').write_text(json.dumps(results,indent=2))
(OUT/'metadata.json').write_text(json.dumps({'old':old,'new':new},indent=2))
addon.unregister()
