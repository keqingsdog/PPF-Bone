import os, bpy,sys,pathlib,json
ROOT=pathlib.Path(os.environ['PPF_SAVEAS_TEST_ROOT'])
sys.path.insert(0,os.environ['PPF_SAVEAS_SOURCE']);sys.path[:0]=[str(ROOT),str(ROOT/'test_deps')]
import blender_addon as addon
from blender_addon.core import rigid_pose as r
addon.register();OUT=ROOT/'run_integrated';bpy.ops.wm.open_mainfile(filepath=str(OUT/'另存 测试.blend'))
prior={s.name:s[r.CACHE_KEY] for s in bpy.data.scenes}
for s in bpy.data.scenes:s[r.SAVE_WARNING_PROP]='old warning'
import ctypes
folder=OUT/'locked_parent';folder.mkdir(exist_ok=True)
target=folder/'failure.blend'
k=ctypes.WinDLL('kernel32',use_last_error=True); k.CreateFileW.restype=ctypes.c_void_p
handle=k.CreateFileW(str(target)+'@',0x40000000,0,None,2,0x80,None)
assert handle != ctypes.c_void_p(-1).value
failed=False
try:
 try:bpy.ops.wm.save_as_mainfile(filepath=str(target))
 except RuntimeError as e: failed=True;print('EXPECTED_FAILURE',e)
finally:k.CloseHandle(ctypes.c_void_p(handle))
assert failed, 'expected OS write failure after preparation'
assert all(s[r.CACHE_KEY]==prior[s.name] and s[r.SAVE_WARNING_PROP]=='old warning' for s in bpy.data.scenes)
assert (folder/'data/failure/bone_poses').exists()
assert r._save_transaction is None
print('POST_PREPARE_WRITE_FAILURE_PASS')
addon.unregister()
