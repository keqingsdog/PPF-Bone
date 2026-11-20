"""Fit new PPF bone proxies to weighted source vertices."""
import bpy
from bpy.props import FloatProperty
from .bone_proxy import selected_bones, update_proxies, box_geometry
from ..models.bone_bindings import bone_world_frame_scale


class PPF_OT_FitBoneProxiesToWeights(bpy.types.Operator):
    bl_idname = 'ppf.fit_bone_proxies_to_weights'
    bl_label = 'Fit Bone Proxies to Skin Weights'
    bl_options = {'REGISTER', 'UNDO'}

    weight_threshold: FloatProperty(name='Minimum Weight', default=0.01, min=0.0, max=1.0)

    def execute(self, context):
        try:
            arm, bones = selected_bones(context)
            sources = [obj for obj in context.scene.objects if obj.type == 'MESH'
                       and any(m.type == 'ARMATURE' and m.object == arm for m in obj.modifiers)]
            bounds = {}
            arm_inverse = arm.matrix_world.inverted()
            for obj in sources:
                groups = {vg.index: bone for bone in bones
                          if (vg := obj.vertex_groups.get(bone.name)) is not None}
                if not groups:
                    continue
                transforms = {index: bone.matrix_local.inverted() @ arm_inverse @ obj.matrix_world
                              for index, bone in groups.items()}
                for vertex in obj.data.vertices:
                    for membership in vertex.groups:
                        index = membership.group
                        if (index not in groups or membership.weight <= 0
                                or membership.weight < self.weight_threshold):
                            continue
                        bone = groups[index]
                        point = transforms[index] @ vertex.co
                        if bone.name not in bounds:
                            bounds[bone.name] = ([float(v) for v in point], [float(v) for v in point])
                        else:
                            lo, hi = bounds[bone.name]
                            for axis in range(3):
                                lo[axis] = min(lo[axis], point[axis])
                                hi[axis] = max(hi[axis], point[axis])
            rows, missing = {}, []
            for bone in bones:
                if bone.name not in bounds:
                    missing.append(bone.name)
                    continue
                lo, hi = bounds[bone.name]
                center = [(a+b)/2 for a,b in zip(lo,hi)]
                size = [max(b-a, 0.001) for a,b in zip(lo,hi)]
                frame, scale = bone_world_frame_scale(arm, bone)
                vertices, faces = box_geometry(*(value * scale for value in size))
                vertices = [tuple(v[i]+center[i]*scale for i in range(3)) for v in vertices]
                # Rest-frame samples are expressed in bone local coordinates;
                # place the result in the current pose using the same bone frame.
                rows[bone.name] = {'settings': {}, 'geometry': (vertices, faces, frame)}
            if not rows:
                raise ValueError('No weighted vertices found for selected bones')
            count = update_proxies(context, authoring_rows=rows)
            self.report({'WARNING'} if missing else {'INFO'},
                        f'Fitted {count} proxies' + (f'; no weighted vertices: {", ".join(missing)}' if missing else ''))
            return {'FINISHED'}
        except (ValueError, RuntimeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


def register():
    bpy.utils.register_class(PPF_OT_FitBoneProxiesToWeights)


def unregister():
    bpy.utils.unregister_class(PPF_OT_FitBoneProxiesToWeights)
