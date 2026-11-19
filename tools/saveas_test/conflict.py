import os, bpy,sys,pathlib,shutil,json
ROOT=pathlib.Path(os.environ['PPF_SAVEAS_TEST_ROOT']);sys.path[:0]=[os.environ['PPF_SAVEAS_SOURCE'],str(ROOT/'test_deps')]
import blender_addon as addon
from blender_addon.core import rigid_pose as r
addon.register();out=ROOT/'run_integrated';bpy.ops.wm.open_mainfile(filepath=str(out/'shared.blend'))
a=bpy.data.scenes['Scene'];b=bpy.data.scenes['shared'];src=pathlib.Path(a[r.CACHE_KEY]).parent;alt=out/'conflict_source';shutil.copytree(src,alt)
i=alt/'index.json';d=json.loads(i.read_text());d['frames'].pop('1');i.write_text(json.dumps(d));b[r.CACHE_KEY]=str(i);before=[a[r.CACHE_KEY],b[r.CACHE_KEY]]
bpy.ops.wm.save_as_mainfile(filepath=str(out/'conflicting.blend'))
assert [a[r.CACHE_KEY],b[r.CACHE_KEY]]==before
assert a.get(r.SAVE_WARNING_PROP) and b.get(r.SAVE_WARNING_PROP)
assert not (out/'data/conflicting/bone_poses').exists()
print('SHARED_CONFLICT_REJECTED');addon.unregister()
