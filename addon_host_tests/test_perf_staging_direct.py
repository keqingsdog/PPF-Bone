import importlib, importlib.util, sys, types, os, tempfile, unittest, io
from pathlib import Path
from unittest.mock import patch
import numpy as np, cbor2
root=Path(os.environ['PPF_TEST_ROOT'])/'blender_addon'
for name,loc in [('ppf_test',root),('ppf_test.core',root/'core'),('ppf_test.core.encoder',root/'core'/'encoder')]:
 m=types.ModuleType(name);m.__path__=[str(loc)];sys.modules[name]=m
mod=types.ModuleType('ppf_test.core.module');mod.get_cbor2=lambda:cbor2;sys.modules[mod.__name__]=mod
enc=importlib.import_module('ppf_test.core.encoder.cbor_encode')
events=importlib.import_module('ppf_test.core.events')
repro=importlib.import_module('ppf_test.core.event_repr')
class PerfTests(unittest.TestCase):
 def test_cbor_wire_compatibility_and_cleanup(self):
  p={'a':np.arange(140000,dtype=np.int64).reshape(-1,2),'b':np.float32(1.5),'c':np.array([1,2],dtype=np.int8)}
  old=lambda e,v:e.encode(v.tolist() if isinstance(v,np.ndarray) else v.item())
  expected=cbor2.dumps({'version':2,'kind':'Scene','payload':p},default=old)
  self.assertEqual(enc.dumps_envelope('Scene',p),expected)
  opened=[];real=tempfile.TemporaryFile
  def tracked(*a,**kw):
   f=real(*a,**kw);opened.append(f);return f
  with patch('tempfile.TemporaryFile',tracked):
   with self.assertRaises(TypeError):enc.dumps_envelope('Param',{'bad':object()})
  self.assertTrue(opened and all(x.closed for x in opened))
 def test_events_prefix(self):
  for payload in (b'',b"a'b\"c\n\\\x00"*100000,b"'"*1000,b'"'*1000,bytes(range(256))*100):
   for cls in (events.SendDataRequested,events.ReceiveDataComplete,events.BuildPipelineRequested,events.UploadOnlyRequested):
    e=cls(data=payload)
    self.assertEqual(repro.event_repr_prefix(e),repr(e)[:200])
 def test_spill_cleanup_and_order(self):
  bpy=types.ModuleType('bpy');sys.modules['bpy']=bpy
  transform=types.ModuleType('ppf_test.core.transform')
  for n in ('_normalize_and_scale','_swap_axes','_to_solver'):setattr(transform,n,lambda *a:None)
  sys.modules[transform.__name__]=transform
  for n,names in [('mesh',['compute_data_hash','compute_mesh_hash','detect_stitch_edges','encode_obj','encode_obj_with_hash']),('params',['compute_param_hash','encode_param','encode_param_with_hash'])]:
   m=types.ModuleType('ppf_test.core.encoder.'+n)
   for name in names:setattr(m,name,lambda *a:None)
   sys.modules[m.__name__]=m
  spec=importlib.util.spec_from_file_location('ppf_test.core.encoder_real',root/'core'/'encoder'/'__init__.py',submodule_search_locations=[str(root/'core'/'encoder')])
  m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m
  # resolve the relative mesh/params imports under this package name
  for n in ('mesh','params'):sys.modules[spec.name+'.'+n]=sys.modules['ppf_test.core.encoder.'+n]
  spec.loader.exec_module(m);order=[];opened=[];real=tempfile.TemporaryFile
  def tracked(*a,**kw):
   f=real(*a,**kw);opened.append(f);return f
  m.encode_obj_with_hash=lambda c:(order.append('scene') or (b'scene','scene-token'))
  m.encode_param_with_hash=lambda c:(order.append('param') or (b'param','param-token'))
  context=types.SimpleNamespace(scene={})
  with patch('tempfile.TemporaryFile',tracked):
   self.assertEqual(m._encode_upload_payloads(context,want_data=True,want_param=True),(b'scene',b'param','scene-token','param-token'))
   self.assertEqual(order,['scene','param']);self.assertTrue(all(f.closed for f in opened))
   def fail(c):raise RuntimeError('param failed')
   m.encode_param_with_hash=fail
   with self.assertRaisesRegex(RuntimeError,'param failed'):m._encode_upload_payloads(context,want_data=True,want_param=True)
   self.assertTrue(all(f.closed for f in opened))
 def test_capture_ownership_index(self):
  from types import SimpleNamespace as NS
  bpy=types.ModuleType('bpy');bpy.app=NS(handlers=NS(persistent=lambda f:f));bpy.path=NS(abspath=lambda p,library=None:p);sys.modules['bpy']=bpy
  trans=types.ModuleType('ppf_test.i18n');trans.pgettext_iface=trans.pgettext_tip=lambda x:x;sys.modules[trans.__name__]=trans
  uuid=types.ModuleType('ppf_test.core.uuid_registry');uuid.get_object_uuid=lambda o:o.uid;sys.modules[uuid.__name__]=uuid
  pc2=importlib.import_module('ppf_test.core.pc2');pc2.get_pc2_path=lambda k:os.path.join(tempfile.gettempdir(),k+'.pc2')
  class Objects(list):
   def get(self,name):return next((o for o in self if o.name==name),None)
  class Scene(dict):
   def __init__(self,objs):super().__init__();self.objects=Objects(objs)
  def obj(name,uid,groups=(),mods=()):
   return NS(name=name,uid=uid,type='MESH',vertex_groups=[NS(name=n) for n in groups],modifiers=list(mods),users_scene=[1],library=None,data=NS(library=None),is_editable=True)
  a=obj('a','alpha',['A B','a/b','Unique']);duplicate=obj('dup','dup');duplicate2=obj('dup2','dup');foreign=obj('foreign','foreign')
  own=Scene([a,duplicate]);other=Scene([foreign]);bpy.data=NS(objects=[a,duplicate,duplicate2,foreign],scenes=[own,other]);bpy.context=NS(scene=own)
  key='alpha_staticdeform'; other_key='foreign_staticdeform'; orphan='missing_staticdeform';referenced='ref_staticdeform'
  own[pc2._CAPTURE_OWNERS]=__import__('json').dumps({orphan:'missing',referenced:'ref',other_key:'foreign'})
  a.modifiers.append(NS(type='MESH_CACHE',filepath=pc2.get_pc2_path(referenced)))
  got=pc2._owned_capture_keys(own)
  self.assertIn(key,got);self.assertIn(orphan,got);self.assertNotIn(referenced,got);self.assertNotIn(other_key,got)
  self.assertFalse(any(k.startswith('dup') for k in got));live=pc2._live_capture_keys(a)
  self.assertNotIn('alpha__A_B__pindeform',live);self.assertNotIn('alpha__a_b__pindeform',live)
  self.assertIn('alpha__Unique__pindeform',live)
if __name__=='__main__': unittest.main(verbosity=2)
