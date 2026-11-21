import sys
from pathlib import Path
import bpy
from mathutils import Vector
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import blender_addon
blender_addon.register()
from blender_addon.ops.bone_proxy import validate_closed_mesh
arm=bpy.data.objects.new('Arm',bpy.data.armatures.new('Arm'))
bpy.context.scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active=arm
arm.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
b=arm.data.edit_bones.new('Bone');b.head=(0,0,0);b.tail=(0,2,0)
bpy.ops.object.mode_set(mode='OBJECT')
mesh=bpy.data.meshes.new('source');mesh.from_pydata([(1,0.2,-.3),(2,1.7,.4),(100,100,100)],[],[])
obj=bpy.data.objects.new('source',mesh);bpy.context.scene.collection.objects.link(obj)
vg=obj.vertex_groups.new(name='Bone');vg.add([0,1],1,'REPLACE');vg.add([2],.001,'REPLACE')
mod=obj.modifiers.new('arm','ARMATURE');mod.object=arm
bpy.context.view_layer.objects.active=arm
bpy.ops.object.mode_set(mode='POSE')
arm.data.bones.active=arm.data.bones['Bone']
if hasattr(arm.pose.bones['Bone'], 'select'):
    arm.pose.bones['Bone'].select=True
else:
    arm.data.bones['Bone'].select=True
for shape in ('BOX','SPHERE','CAPSULE'):
 for quality in ('VERY_LOW','HIGH'):
  for mode in ('QUAD','TRI'):
   s=arm.data.bones['Bone'].ppf_bone_physics
   s.shape=shape;s.proxy_quality=quality;s.proxy_face_mode=mode
   assert bpy.ops.ppf.fit_bone_proxies_to_weights(weight_threshold=.01)=={'FINISHED'}
   proxy=s.proxy;validate_closed_mesh(proxy.data)
   expected = {('BOX','VERY_LOW'):8, ('BOX','HIGH'):26, ('SPHERE','VERY_LOW'):26, ('SPHERE','HIGH'):290, ('CAPSULE','VERY_LOW'):34, ('CAPSULE','HIGH'):642}
   assert len(proxy.data.vertices) == expected[shape, quality]
   assert (proxy.matrix_world.translation - Vector((1.5,.95,.05))).length < 1e-5
   axis = proxy.matrix_world.to_3x3().col[0].normalized()
   assert abs(axis.dot(Vector((0,1,0)))) > .99999
   assert max((proxy.matrix_world @ v.co).x for v in proxy.data.vertices)<10
   assert all(len(p.vertices)==3 for p in proxy.data.polygons) if mode=='TRI' else True
   print('PASS',shape,quality,mode,len(proxy.data.vertices),flush=True)


# A single included point must still produce a valid closed proxy.
vg.remove([1])
for shape in ('BOX', 'SPHERE', 'CAPSULE'):
    s.shape = shape
    assert bpy.ops.ppf.fit_bone_proxies_to_weights(weight_threshold=.01) == {'FINISHED'}
    validate_closed_mesh(s.proxy.data)
    assert (s.proxy.matrix_world.translation - Vector((1,.2,-.3))).length < 1e-5
    print('PASS degenerate', shape, flush=True)