import bpy,sys,types,os,importlib
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
root=Path(os.environ['PPF_CANONICAL'])/'blender_addon'
m=types.ModuleType('ppf_live_test');m.__path__=[str(root)];sys.modules[m.__name__]=m
mod=importlib.import_module('ppf_live_test.ui.solver')
class Engine:
 def __init__(self):self.state=NS(server_upload_id=3);self.listeners=[]
 def on_change(self,x):self.listeners.append(x)
 def remove_listener(self,x):self.listeners.remove(x)
class Fake:
 def __init__(self):self._payload={'data':b'a'*10000,'param':b'b'*5000,'data_hash':'a','param_hash':'b'};self._transfer_frame=(1,0.0);self._stage_connection_ticket=None;self.marked=False
 def _origin_changed(self,c):return False
 def mark_remote_requested(self):self.marked=True
ctx=NS(scene=NS(frame_current=1,frame_subframe=0.0)); eng=Engine(); captures=[]
state=NS(commit_pending_mesh_hash=lambda:None)
com=NS(_engine=eng,animation=NS(clear=lambda:None),build_pipeline=lambda **kw:captures.append(kw))
with patch.object(mod,'com',com),patch.object(mod,'get_addon_data',lambda s:NS(state=state)):
 x=Fake();mod.SOLVER_OT_Transfer._stage_upload(x,ctx)
 assert x.marked and x._payload=={} and captures[0]['data']==b'a'*10000 and len(eng.listeners)==1
 eng.remove_listener(x._transfer_upload_listener);x._transfer_upload_listener=None
 def fail(**kw):raise RuntimeError('upload refused')
 com.build_pipeline=fail;y=Fake()
 try:mod.SOLVER_OT_Transfer._stage_upload(y,ctx)
 except RuntimeError as e:assert str(e)=='upload refused'
 else:raise AssertionError('failure swallowed')
 assert not eng.listeners and y._payload['data']
 with patch.object(mod,'_remove_transfer_upload_listener',lambda self:None),patch.object(mod,'super',lambda: NS(cleanup_modal=lambda c:None),create=True):
  mod.SOLVER_OT_Transfer.cleanup_modal(y,ctx)
 assert y._payload=={}
print('UPLOAD_REFERENCE_SUCCESS_FAILURE_CLEANUP_PASS',flush=True)
