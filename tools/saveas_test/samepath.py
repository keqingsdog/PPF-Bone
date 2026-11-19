import os, bpy,sys,pathlib,json
ROOT=pathlib.Path(os.environ['PPF_SAVEAS_TEST_ROOT'])
sys.path.insert(0,os.environ['PPF_SAVEAS_SOURCE']);sys.path[:0]=[str(ROOT),str(ROOT/'test_deps')]
import blender_addon as addon
from blender_addon.core import rigid_pose as r
addon.register();run=sys.argv[-1];out=ROOT/run;bpy.ops.wm.open_mainfile(filepath=str(out/'另存 测试.blend'))
count=[0];original=r._file_sha256
def counted(path):count[0]+=1;return original(path)
r._file_sha256=counted
prior={s.name:s[r.CACHE_KEY] for s in bpy.data.scenes}
bpy.ops.wm.save_as_mainfile(filepath=str(out/'另存 测试.blend'))
assert count[0]==0,count
assert all(s[r.CACHE_KEY]==prior[s.name] for s in bpy.data.scenes)
(out/'samepath_probe.json').write_text(json.dumps({'full_cache_hash_calls':count[0],'refs_unchanged':True}));print('SAMEPATH_PASS',count)
addon.unregister()
