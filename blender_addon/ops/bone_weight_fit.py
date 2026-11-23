"""Fit PPF bone proxies to weighted source vertices."""
import math

import bpy
from mathutils import Matrix
from ..i18n import register_class
from ..i18n.props import FloatProperty
from .bone_proxy import selected_bones, update_proxies, box_geometry, capsule_geometry, sphere_geometry
from ..models.proxy_generation import effective_proxy_quality, effective_proxy_face_mode
from ..models.bone_bindings import bone_world_frame_scale
from .bone_weight_deform import deform_weight_contour


class PPF_OT_FitBoneProxiesToWeights(bpy.types.Operator):
    bl_idname = 'ppf.fit_bone_proxies_to_weights'
    bl_label = 'Fit Bone Proxies to Skin Weights'
    bl_description = 'Fit proxies to mesh vertices weighted to the selected bones.'
    bl_options = {'REGISTER', 'UNDO'}

    weight_threshold: FloatProperty(name='Minimum Weight', description='Use every positive skin weight at zero; otherwise include weights at or above the threshold.', default=0.0, min=0.0, max=1.0)

    fit_strength: FloatProperty(
        name='Contour Fit',
        description='Reshape the starter proxy toward the convex skin-weight footprint; zero keeps the starter shape and adjusts its size if needed',
        default=0.0, min=0.0, max=1.0, subtype='FACTOR',
    )

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        self.layout.prop(self, 'weight_threshold')
        self.layout.prop(self, 'fit_strength', slider=True)

    def execute(self, context):
        try:
            arm, bones = selected_bones(context)
            sources = [obj for obj in context.scene.objects if obj.type == 'MESH'
                       and any(m.type == 'ARMATURE' and m.object == arm for m in obj.modifiers)]
            fit_strength = getattr(self, 'fit_strength', 0.0)
            threshold = float(self.weight_threshold)
            bounds, points = {}, {}
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
                                or membership.weight < threshold):
                            continue
                        bone = groups[index]
                        point = transforms[index] @ vertex.co
                        points.setdefault(bone.name, []).append((point, membership.weight))
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
                frame, scale = bone_world_frame_scale(arm, bone)
                size = [max((b-a)*scale, 0.001) for a,b in zip(lo,hi)]
                settings = bone.ppf_bone_physics
                shape = settings.shape
                quality = effective_proxy_quality(settings)
                face_mode = effective_proxy_face_mode(settings)
                if shape == 'CUSTOM':
                    raise ValueError(f'{bone.name}: skin weight fitting does not support custom proxy shape')
                if shape == 'BOX':
                    vertices, faces = box_geometry(size[1], size[0], size[2], face_mode, quality)
                elif shape == 'SPHERE':
                    radius = max(0.0005, max(math.dist(point, center) * scale
                                             for point, _weight in points[bone.name]))
                    vertices, faces = sphere_geometry(radius, quality, face_mode)
                elif shape == 'CAPSULE':
                    cx, cy, cz = center
                    samples = [(math.hypot(point.x-cx, point.z-cz) * scale,
                                abs(point.y-cy) * scale) for point, _weight in points[bone.name]]
                    radius = max(0.0005, max(radial for radial, _ in samples))
                    # Each cap covers this far beyond the straight segment at
                    # a given radial distance from the bone's local Y axis.
                    radius_squared = radius * radius
                    half_shaft = max(axial - math.sqrt(max(0., radius_squared - radial**2))
                                     for radial, axial in samples)
                    length = 2 * radius + max(2 * half_shaft, 0.001)
                    vertices, faces = capsule_geometry(length, radius, quality, face_mode=face_mode)
                else:
                    raise ValueError(f'{bone.name}: unsupported proxy shape {shape}')
                cx, cy, cz = center
                weighted = [((point.y-cy)*scale, -(point.x-cx)*scale, (point.z-cz)*scale, weight)
                            for point, weight in points[bone.name]]
                vertices = deform_weight_contour(vertices, faces, weighted, fit_strength)
                frame = frame @ Matrix.Translation(tuple(v*scale for v in center)) @ Matrix.Rotation(math.pi/2, 4, 'Z')
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
    register_class(PPF_OT_FitBoneProxiesToWeights)


def unregister():
    bpy.utils.unregister_class(PPF_OT_FitBoneProxiesToWeights)
