import os, bpy,sys,pathlib,json
ROOT=pathlib.Path(os.environ['PPF_SAVEAS_TEST_ROOT'])
sys.path.insert(0,os.environ['PPF_SAVEAS_SOURCE']);sys.path[:0]=[str(ROOT),str(ROOT/'test_deps')]
from blender_addon.core import rigid_pose as r
import blender_addon as addon
addon.register()
label=sys.argv[-1];label='另存 测试' if label=='saved' else label;out=ROOT/'run_integrated';bpy.ops.wm.open_mainfile(filepath=str(out/(label+'.blend')))
results=[]
for s in bpy.data.scenes:
 bpy.context.window.scene=s
 arm=next(o for o in s.objects if o.type=='ARMATURE'); vals=[]
 for frame in (1,2):
  s.frame_set(frame);r.bone_pose_frame_change(s);bpy.context.view_layer.update();vals.append(float(arm.pose.bones[0].matrix.translation.x))
 assert all(abs(a-b)<1e-4 for a,b in zip(vals,(2.,3.))),(s.name,vals)
 assert r.has_bone_pose_cache(s)
 expected='另存 测试' if label=='partial' else label
 assert expected in pathlib.Path(s[r.CACHE_KEY]).parts
 assert bool(s.get(r.SAVE_WARNING_PROP))==(label=='partial')
 results.append({'scene':s.name,'positions':vals,'warning':s.get(r.SAVE_WARNING_PROP),'key':s[r.CACHE_KEY]})
if label=='shared':
 scenes=list(bpy.data.scenes);b=bpy.data.scenes['shared'];scenes[0]=bpy.data.scenes['Scene'];prior=b[r.CACHE_KEY]
 bpy.context.window.scene=scenes[0];r.clear_bone_pose_cache(scenes[0]);assert not scenes[0].get(r.CACHE_KEY)
 assert b[r.CACHE_KEY]==prior
 bpy.context.window.scene=b;
 for f,x in ((1,2),(2,3)):
  b.frame_set(f);r.bone_pose_frame_change(b);bpy.context.view_layer.update();assert abs(next(o for o in b.objects if o.type=='ARMATURE').pose.bones[0].matrix.translation.x-x)<1e-4
print('COLD_PASS',label,results);(out/(label+'_cold.json')).write_text(json.dumps(results,indent=2));addon.unregister()
