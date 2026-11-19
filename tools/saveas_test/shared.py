import os, bpy,sys,pathlib
ROOT=pathlib.Path(os.environ['PPF_SAVEAS_TEST_ROOT'])
sys.path[:0]=[os.environ['PPF_SAVEAS_SOURCE'],str(ROOT/'test_deps')]
import blender_addon as addon
from blender_addon.core import rigid_pose as r
addon.register();out=ROOT/'run_integrated';bpy.ops.wm.open_mainfile(filepath=str(out/'另存 测试.blend'))
a=bpy.data.scenes['Scene'];b=bpy.data.scenes.new('shared');arm=next(o for o in a.objects if o.type=='ARMATURE');b.collection.objects.link(arm);b[r.CACHE_KEY]=a[r.CACHE_KEY]
bpy.ops.wm.save_as_mainfile(filepath=str(out/'shared.blend'))
assert a[r.CACHE_KEY]==b[r.CACHE_KEY];print('SHARED_SAVE_PASS');addon.unregister()
