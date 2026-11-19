import os,sys,pathlib,bpy
ROOT=pathlib.Path(os.environ['PPF_SAVEAS_TEST_ROOT']);sys.path[:0]=[os.environ['PPF_SAVEAS_SOURCE'],str(ROOT/'test_deps')]
import blender_addon as addon
from blender_addon.ui.solver import SOLVER_PT_SolverPanel
addon.register()
assert bpy.ops.solver.resume_from.get_rna_type() and bpy.ops.solver.twist_upgrade_query.get_rna_type()
bpy.context.scene['ppf_bone_pose_save_warning']='save-as-test-warning'
class Sink:
 def __init__(self):self.labels=[]
 def __getattr__(self,name):
  def call(*args,**kwargs):
   if name=='label':self.labels.append(kwargs.get('text',''))
   return self
  return call
sink=Sink();obj=type('PanelProxy',(),{})();obj.layout=sink
SOLVER_PT_SolverPanel.draw(obj,bpy.context)
assert 'save-as-test-warning' in sink.labels,sink.labels
print('PANEL_DRAW_AND_UPGRADE_RNA_PASS');addon.unregister()
