"""Focused static uniform pose scale regression. No native solver run.
Blender --background --factory-startup --python-exit-code 1 --python this.py -- input.blend result.json
"""
import bpy,sys,json,uuid,types,importlib,copy
from pathlib import Path
import numpy as np
from mathutils import Matrix,Vector
SOURCE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(SOURCE))
import blender_addon
blender_addon.register()
from blender_addon.models.bone_bindings import bone_world_frame_scale,collect_bindings
from blender_addon.ops.bone_proxy import update_proxies
from blender_addon.core.encoder.bones import encode_bones
from blender_addon.models.groups import get_addon_data
from blender_addon.core.compat import set_pose_bone_selected
from blender_addon.core import rigid_pose as rp
args=sys.argv[sys.argv.index('--')+1:];bpy.ops.wm.open_mainfile(filepath=args[0])
arm=bpy.data.objects['Armature'];bone=arm.data.bones['Bone.005'];pb=arm.pose.bones[bone.name]
object_case=len(args)>2 and args[2] in ('object_unit','object_pose')
unit_case=len(args)>2 and args[2] in ('unit','object_unit')
if object_case:arm.scale=(1.4414172,)*3;bpy.context.view_layer.update()
if unit_case:
 arm.pose.bones['Bone.001'].scale=(1.,1.,1.);bpy.context.view_layer.update()
for p in arm.pose.bones:set_pose_bone_selected(p,p==pb)
arm.data.bones.active=bone
before={p.name:p.matrix_basis.copy() for p in arm.pose.bones}
head=arm.matrix_world@pb.head;tail=arm.matrix_world@pb.tail
assert update_proxies(bpy.context)==1
proxy=bone.ppf_bone_physics.proxy
assert (proxy.matrix_world.translation-(head+tail)/2).length<3e-6
actual_length=max(v.co.x for v in proxy.data.vertices)-min(v.co.x for v in proxy.data.vertices)
assert abs(actual_length-(tail-head).length)<3e-6
assert all(max(abs(p.matrix_basis[i][j]-before[p.name][i][j]) for i in range(4) for j in range(4))<1e-7 for p in arm.pose.bones)
state=get_addon_data(bpy.context.scene).state;state.frame_count=2
source=collect_bindings(bpy.context)
assert np.max(np.abs(np.array(next(a for a in source['armatures'] if a['armature_uuid']==arm.ppf_armature_uuid)['matrix_world_blender'])-np.array(arm.matrix_world)))<1e-7
doc=encode_bones(bpy.context,state,60,bpy.context.scene.frame_current)
row=next(r for r in doc['bindings'] if r['bone_uuid']==bone.ppf_bone_physics.bone_uuid)
s=row.get('author_uniform_scale',1.);assert abs(s-(1. if unit_case else 1.363726)*(1.4414172 if object_case else 1.))<1e-5
assert max(abs(v-(1.4414172 if object_case else 1.)) for v in arm.scale)<1e-6
# Root and nonphysical-parent samplers consume the same normalized world frames.
assert row.get('automatic_parent_anchor_v1',{}).get('root_motion',{}).get('keys')
settings=bone.ppf_bone_physics
settings.auto_follow_parent=False;settings.root_mode='KINEMATIC'
root_doc=encode_bones(bpy.context,state,60,bpy.context.scene.frame_current)
root_row=next(r for r in root_doc['bindings'] if r['bone_uuid']==settings.bone_uuid)
assert len(root_row['root_motion']['keys'])==int(state.frame_count),(len(root_row['root_motion']['keys']),state.frame_count)
settings.root_mode='FREE';settings.auto_follow_parent=True
# Isolate one binding from unrelated pre-existing rigs for receipt assembly.
doc['bindings']=[row];row.pop('automatic_parent_anchor_v1',None)
row.update(parent_body_uuid=None,parent_bone_uuid=None,root_boundary={'is_root':True,'nonphysical_parent_bone':None})
doc.pop('soft_limit_auto_v1',None);doc.pop('collision_filter_v1',None)
# Load real frontend receipt assemblers without initializing Rust application.
pkg=types.ModuleType('scale_frontend');pkg.__path__=[str(SOURCE/'frontend')];sys.modules[pkg.__name__]=pkg
scene_mod=types.ModuleType('scale_frontend._scene_');scene_mod._validate_r1_payload=lambda x:x;sys.modules[scene_mod.__name__]=scene_mod
pin_mod=types.ModuleType('scale_frontend._scene_pin_');pin_mod.PinData=object;pin_mod.TransformKeyframeOperation=object;sys.modules[pin_mod.__name__]=pin_mod
bb=importlib.import_module('scale_frontend._blender_bones_');fm=importlib.import_module('scale_frontend._pdrd_native_frames_')
fixed=types.SimpleNamespace(_scene_uuid=str(uuid.uuid4()),_pin=[],_tet=np.array([[0,1,2,3]]))
P=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=float)
fixed._vert=(np.zeros(4,dtype=int),P);fixed._displacement=np.zeros((1,3));fixed.vertex=lambda:P
fixed.set_pdrd_ball_anchor_payload=lambda payload:setattr(fixed,'_pdrd_ball_anchor_payload',payload)
uid=row['body_uuid'];frames={uid:{'world_translation_m':(bb.S@np.array(row['proxy_world_matrix_blender']))[:3,3].tolist(),'world_quaternion_wxyz':list((Matrix(bb.S.tolist())@proxy.matrix_world).to_quaternion()),'runtime_body_index':0}}
param={'scene':{'fps':60},'group':[[{'model':'snhk','density':bone.ppf_bone_physics.density},['proxy'],[uid]]]}
fixed._statistics_objects=[{'object_uuid':uid,'dynamics_type':'SOLID','vertex_indices':[0,1,2,3],'tet_range':(0,1)}]
fm.native_body_frames=lambda _:{}
fem=importlib.import_module('scale_frontend._fem_bone_joints_');fem_receipt=fem.install_mixed(fixed,doc,param)
assert fem_receipt['bindings'][0].get('author_uniform_scale',1.)==s
legacy_doc=copy.deepcopy(doc);legacy_doc['bindings'][0].pop('author_uniform_scale',None)
assert 'author_uniform_scale' not in fem.install_mixed(fixed,legacy_doc,param)['bindings'][0]
fixed._statistics_objects=[{'object_uuid':uid,'dynamics_type':'PDRD'}];bb.native_body_frames=lambda _:frames
param['group'][0][0]['model']='pdrd';receipt=bb._install(fixed,doc,param)
assert receipt['bindings'][0].get('author_uniform_scale',1.)==s
assert 'author_uniform_scale' not in bb._install(fixed,legacy_doc,param)['bindings'][0]
# Real display target resolution and mapping; accepted-frame validation is outside fixture.
assert len(rp._targets(bpy.context.scene,receipt))==1
assert len(rp._targets(bpy.context.scene,fem_receipt))==1
old_validate,old_targets=rp.validate_frame,rp._targets
rp.validate_frame=lambda *a:{uid:bb.frame_matrix(frames[uid])}
# Keep real _targets: uniform armature metadata must pass the actual display gate.
results=[]
try:
 for kind,r in [('RIGID',receipt),('FEM',fem_receipt)]:
  for enabled in ([True,False] if kind=='FEM' else [True]):
   bone.ppf_bone_physics.fem_scale_enabled=enabled
   for repeat in range(2):
    desired,diag=rp._prepare_pose(bpy.context.scene,r,{'frame_id':0,'vertex_native_bytes':P.astype('<f4').tobytes()})
    target=desired[row['bone_uuid']]
    assert (arm.matrix_world@target.translation-head).length<3e-6
    mapped_tail=arm.matrix_world@(target@Vector((0,bone.length,0)))
    assert (mapped_tail-tail).length<5e-6,(kind,enabled,list(mapped_tail),list(tail))
    targets=[(r['bindings'][0],arm,pb)]
    plan=rp._pose_application_plan(targets,desired,diag)
    rp._commit_pose_display(targets,desired,diag,plan);bpy.context.view_layer.update()
    assert (arm.matrix_world@pb.head-head).length<5e-6
    assert (arm.matrix_world@pb.tail-tail).length<8e-6,(kind,enabled,list(pb.tail),list(tail))
   results.append({'kind':kind,'fem_scale_enabled':enabled,'head_tail_preserved':True,'repeat_no_accumulation':True})
finally:
 rp.validate_frame,rp._targets=old_validate,old_targets
 for p in arm.pose.bones:p.matrix_basis=before[p.name]
 bone.inherit_scale='FULL';bpy.context.view_layer.update()
# Real receipt decoding: missing metadata stays backward compatible, malformed fails.
import cbor2
for value in ['missing',s,0.,-1.,float('nan'),True,'1']:
 decoded=copy.deepcopy(receipt);decoded.update(scene_uuid=str(uuid.uuid4()),scene_inputs_sha256='0'*64)
 if value=='missing':decoded['bindings'][0].pop('author_uniform_scale',None)
 else:decoded['bindings'][0]['author_uniform_scale']=value
 try:rp.decode_bindings(cbor2.dumps({'version':1,'kind':'ppf.blender_bone_bindings','payload':decoded}))
 except ValueError:
  assert value != 'missing' and not(type(value) is float and value==s),value
 else:assert value=='missing' or type(value) is float and value==s,value
old_arm_scale=arm.scale.copy()
for bad_scale in [(1,1.2,1),(-1,1,1)]:
 arm.scale=bad_scale;bpy.context.view_layer.update()
 try:bone_world_frame_scale(arm,bone)
 except ValueError:pass
 else:raise AssertionError(('object transform accepted',bad_scale))
arm.scale=old_arm_scale;bpy.context.view_layer.update()
parent=arm.pose.bones['Bone.001'];old=parent.scale.copy()
for scale in [(1,1.2,1),(-1,1,1)]:
 parent.scale=scale;bpy.context.view_layer.update()
 try:bone_world_frame_scale(arm,bone)
 except ValueError:pass
 else:raise AssertionError(('unsupported accepted',scale))
parent.scale=old;bpy.context.view_layer.update()
shear=Matrix.Identity(4);shear[0][1]=.2
fake=types.SimpleNamespace(name='ShearArm',matrix_world=Matrix.Identity(4),pose=types.SimpleNamespace(bones={'Shear':types.SimpleNamespace(matrix=shear)}))
try:bone_world_frame_scale(fake,types.SimpleNamespace(name='Shear',matrix_local=Matrix.Identity(4)))
except ValueError:pass
else:raise AssertionError('shear accepted')
frame=bpy.context.scene.frame_current;sub=bpy.context.scene.frame_subframe
def insert_changing_scale(target, frame):
    original = target.scale.copy()
    target.keyframe_insert('scale', frame=frame)
    target.scale = original * 1.25
    target.keyframe_insert('scale', frame=frame + 1)
    target.scale = original
    bpy.context.view_layer.update()

insert_changing_scale(arm, frame)
try:encode_bones(bpy.context,state,60,frame)
except ValueError as e:assert 'animated' in str(e),str(e)
else:raise AssertionError('animated armature object accepted')
assert bpy.context.scene.frame_current==frame and bpy.context.scene.frame_subframe==sub
arm.animation_data_clear()
insert_changing_scale(parent, frame)
try:encode_bones(bpy.context,state,60,frame)
except ValueError as e:assert 'animated' in str(e),str(e)
else:raise AssertionError('animated ancestor accepted')
assert bpy.context.scene.frame_current==frame and bpy.context.scene.frame_subframe==sub
# Object scale retains the narrowly bounded matching-rest drift correction.
from blender_addon.models.bone_bindings import similarity_frame_scale
drift=Matrix.Rotation(2.9670597,4,'Z');drift.col[0].xyz=drift.col[0].xyz*(1.+1.28e-5);drift.translation=(.1,.2,.3)
scaled_world=Matrix.Translation((.4,.5,.6))@Matrix.Diagonal((1.4414172,)*3+(1.,))
fake_rest_arm=types.SimpleNamespace(name='DriftArm',matrix_world=scaled_world,pose=types.SimpleNamespace(bones={'Drift':types.SimpleNamespace(matrix=drift)}))
frame_drift,scale_drift=bone_world_frame_scale(fake_rest_arm,types.SimpleNamespace(name='Drift',matrix_local=drift))
assert abs(scale_drift-1.4414172)<1e-6
assert (frame_drift.translation-scaled_world@drift.translation).length<1e-6
# Isolated author-follow bake: preserve four world vertices during static scale freeze.
if bpy.context.object and bpy.context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
scene=bpy.data.scenes.new('StaticScaleBake');bpy.context.window.scene=scene
bake_arm=bpy.data.objects.new('ScaleBake',bpy.data.armatures.new('ScaleBake'));scene.collection.objects.link(bake_arm)
bake_arm.select_set(True);bpy.context.view_layer.objects.active=bake_arm
bpy.ops.object.mode_set(mode='EDIT');eb=bake_arm.data.edit_bones.new('Root');eb.head=(0,0,0);eb.tail=(0,1,0)
bpy.ops.object.mode_set(mode='POSE');bake_pb=bake_arm.pose.bones['Root'];bake_pb.scale=(1.363726,)*3
if object_case:bake_arm.scale=(1.4414172,)*3;bpy.context.view_layer.update()
assert update_proxies(bpy.context,all_bones=True)==1
from blender_addon.core import bone_author_follow as follow
follow._snapshot_all()
bake_proxy=bake_pb.bone.ppf_bone_physics.proxy
indices=[0,1,3,len(bake_proxy.data.vertices)-1]
world=[bake_proxy.matrix_world@bake_proxy.data.vertices[i].co for i in indices]
old_world=(bake_arm.matrix_world@bake_pb.matrix).copy();bake_pb.scale=(1.5,)*3;bpy.context.view_layer.update()
expected=[bake_arm.matrix_world@bake_pb.matrix@old_world.inverted()@v for v in world]
assert follow.sync_for_transfer(bpy.context)==1
errors=[(bake_proxy.matrix_world@bake_proxy.data.vertices[i].co-v).length for i,v in zip(indices,expected)]
assert max(errors)<2e-6,errors
assert max(abs(c.length-1) for c in bake_proxy.matrix_world.to_3x3().col)<1e-6
assert max(abs(v-1.5) for v in bake_pb.scale)<1e-6
# Explicit Apply Scale on the private fixture preserves baked world geometry.
world_before_apply=[bake_proxy.matrix_world@bake_proxy.data.vertices[i].co for i in indices]
bpy.ops.object.mode_set(mode='OBJECT');bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
bpy.context.view_layer.update();follow.apply_pending(bpy.context)
apply_errors=[(bake_proxy.matrix_world@bake_proxy.data.vertices[i].co-v).length for i,v in zip(indices,world_before_apply)]
assert max(apply_errors)<2e-6,apply_errors
empty=bpy.data.objects.new('AnimatedParent',None);scene.collection.objects.link(empty);bake_arm.parent=empty
insert_changing_scale(empty, scene.frame_current)
try:encode_bones(bpy.context,get_addon_data(scene).state,60,scene.frame_current)
except ValueError as e:assert 'ancestor object scale' in str(e),str(e)
else:raise AssertionError('FREE body animated object ancestor accepted')
# Shared layered Action: another slot's object scale is unrelated to this rig.
bake_arm.parent=None
shared=bpy.data.actions.new('SharedAction');own=shared.slots.new('OBJECT','RigSlot');other=shared.slots.new('OBJECT','UnrelatedSlot')
layer=shared.layers.new('Layer');strip=layer.strips.new(type='KEYFRAME')
strip.channelbags.new(own).fcurves.new('rotation_euler',index=0).keyframe_points.insert(scene.frame_current,0.)
strip.channelbags.new(other).fcurves.new('scale',index=0).keyframe_points.insert(scene.frame_current,2.)
ad=bake_arm.animation_data_create();ad.action=shared;ad.action_slot=own
encode_bones(bpy.context,get_addon_data(scene).state,60,scene.frame_current)

# A real pose keying set includes scale when all the author wanted was pose motion.
bake_arm.animation_data_clear()
bpy.ops.object.mode_set(mode='POSE')
set_pose_bone_selected(bake_pb, True)
bake_arm.data.bones.active = bake_pb.bone
area = next(area for area in bpy.context.screen.areas if area.type == 'VIEW_3D')
region = next(region for region in area.regions if region.type == 'WINDOW')
with bpy.context.temp_override(area=area, region=region):
    assert bpy.ops.anim.keyframe_insert_by_name(type='LocRotScale') == {'FINISHED'}
from blender_addon.core.utils import get_id_fcurves
scale_path = bake_pb.path_from_id('scale')
scale_curves = [curve for curve in get_id_fcurves(bake_arm) if curve.data_path == scale_path]
assert len(scale_curves) == 3 and all(len(curve.keyframe_points) == 1 for curve in scale_curves)
from blender_addon.core.encoder.bones import _validate_static_pose_scales
_validate_static_pose_scales(bpy.context, None, None, explicit_only=True)
first_key_doc = encode_bones(bpy.context, get_addon_data(scene).state, 60, scene.frame_current)
assert first_key_doc['bindings'][0]['author_uniform_scale'] > 0

# A scale excursion between two integer frame samples must still be rejected.
for curve in scale_curves:
    initial_scale = curve.keyframe_points[0].co.y
    curve.keyframe_points.insert(scene.frame_current + 0.5, initial_scale * 1.25)
    curve.keyframe_points.insert(scene.frame_current + 1, initial_scale)
try:
    _validate_static_pose_scales(bpy.context, None, None, explicit_only=True)
except ValueError as exc:
    assert 'animated pose/object scale' in str(exc)
else:
    raise AssertionError('subframe pose scale change accepted')
# Cached physical display must skip author-scale preflight before its nonuniform display is read.
from blender_addon.core.encoder import bones as enc
old_cache,old_gate=rp.has_bone_pose_cache,enc._validate_static_pose_scales
rp.has_bone_pose_cache=lambda scene:True
def unexpected(*args):raise AssertionError('cached display entered author scale validation')
enc._validate_static_pose_scales=unexpected
try:assert follow.sync_for_transfer(bpy.context)==0
finally:rp.has_bone_pose_cache,enc._validate_static_pose_scales=old_cache,old_gate
Path(args[1]).write_text(json.dumps({'status':'PASS','source_scale':s,'actual_length':actual_length,'expected_length':(tail-head).length,'display':results,'bake_world_vertex_max_error':max(errors),'explicit_apply_scale_world_vertex_max_error':max(apply_errors),'raw_armature_metadata_retained':True,'animated_object_scale_refused':True,'object_scaled_rest_drift_preserved':True,'real_targets_gate_used':True,'root_and_auto_parent_samplers':True,'object_scale_preserved':True,'real_pose_commit_preserved':True,'legacy_receipt_shape_preserved':True,'legacy_receipt_and_invalid_scale_decode':True,'unrelated_action_slot_ignored':True,'cached_display_skips_author_gate':True,'ancestor_object_animation_refused':True,'ancestor_animation_refused':True,'nonuniform_shear_reflection_refused':True,'first_pose_key_accepted':True,'subframe_scale_animation_refused':True},indent=2),encoding='utf8')
print('STATIC_SCALE_PASS',args[1])
