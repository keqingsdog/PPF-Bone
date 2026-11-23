"""Fit PPF bone proxies to weighted source vertices."""
import math

import bpy
import numpy as np
from mathutils import Matrix
from ..i18n import register_class, pgettext_iface as iface
from ..i18n.props import BoolProperty, FloatProperty
from .bone_proxy import selected_bones, update_proxies, box_geometry, capsule_geometry, sphere_geometry
from ..models.proxy_generation import effective_proxy_quality, effective_proxy_face_mode
from ..models.bone_bindings import bone_world_frame_scale
from .bone_weight_deform import deform_weight_contour
from .bone_weight_precision import MAX_SAMPLES, fit_height_envelope


def _minimum_weight_property(**kwargs):
    return FloatProperty(name='Minimum Weight', description='Use every positive skin weight at zero; otherwise include weights at or above the threshold.',
                         default=0.0, min=0.0, max=1.0, **kwargs)


def _contour_fit_property(**kwargs):
    return FloatProperty(name='Contour Fit',
                         description='Reshape the starter proxy toward the convex skin-weight footprint; zero keeps the starter shape and adjusts its size if needed',
                         default=0.0, min=0.0, max=1.0, subtype='FACTOR', **kwargs)


def _precision_fit_property(**kwargs):
    return BoolProperty(
        name='Precision Fit',
        description='Try to follow concave details where the source supports it. If a valid proxy cannot be built, use the convex fit and report the fallback.',
        default=False,
        **kwargs,
    )


def draw_inline_fit(layout, context):
    wm = context.window_manager
    controls = layout.column(align=True)
    settings = controls.row(align=True)
    settings.prop(wm, 'ppf_bone_fit_weight_threshold')
    settings.prop(wm, 'ppf_bone_fit_strength', slider=True)
    settings.prop(wm, 'ppf_bone_fit_precision', toggle=True)
    controls.operator_context = 'EXEC_DEFAULT'
    action = controls.row()
    action.enabled = context.mode in {'POSE', 'OBJECT'}
    op = action.operator('ppf.fit_bone_proxies_to_weights', text='按蒙皮权重拟合代理', icon='MOD_ARMATURE')
    op.weight_threshold = wm.ppf_bone_fit_weight_threshold
    op.fit_strength = wm.ppf_bone_fit_strength
    op.precision_fit = wm.ppf_bone_fit_precision



def _known_absolute_contact_offset(scene, arm, bone):
    """Read only this existing owner's definite static absolute contact setting."""
    from ..core.utils import get_id_fcurves
    from ..models.groups import get_group_by_uuid, iterate_object_groups
    from .bone_proxy import is_owned_bone_group
    settings = bone.ppf_bone_physics
    group = get_group_by_uuid(scene, settings.group_uuid) if settings.group_uuid else None
    proxy = settings.proxy
    if (settings.physical_model != 'RIGID' or not settings.body_uuid
            or not is_owned_bone_group(group, 'PDRD', settings.body_uuid)
            or group.use_group_bounding_box_diagonal
            or proxy is None or proxy.type != 'MESH' or not proxy.get('ppf_bone_proxy_owned')
            or proxy.get('_solver_uuid') != settings.body_uuid
            or proxy.get('ppf_bone_uuid') != settings.bone_uuid
            or proxy.get('ppf_armature_uuid') != arm.ppf_armature_uuid
            or proxy.data is None or proxy.data.shape_keys is not None
            or any(mod.show_viewport for mod in proxy.modifiers)):
        return 0.0
    if sum(g.active and g.uuid == settings.group_uuid for g in iterate_object_groups(scene)) != 1:
        return 0.0
    offset = float(group.contact_offset)
    if not math.isfinite(offset) or offset <= 0.0:
        return 0.0
    anim = scene.animation_data
    if anim:
        if anim.nla_tracks:
            return 0.0
        paths = {group.path_from_id('contact_offset'), group.path_from_id('use_group_bounding_box_diagonal')}
        if any(c.data_path in paths for c in (*get_id_fcurves(scene), *anim.drivers)):
            return 0.0
    # group_contact_lengths scales both this absolute offset and world-space
    # geometry by the same solver-wide scale. Their relation is unchanged in
    # the already world-sized proxy frame; never infer a group-relative value.
    return offset

class PPF_OT_FitBoneProxiesToWeights(bpy.types.Operator):
    bl_idname = 'ppf.fit_bone_proxies_to_weights'
    bl_label = 'Fit Bone Proxies to Skin Weights'
    bl_description = 'Fit proxies to mesh vertices weighted to the selected bones.'
    bl_options = {'REGISTER', 'UNDO'}

    weight_threshold: _minimum_weight_property()
    fit_strength: _contour_fit_property()
    precision_fit: _precision_fit_property()

    def draw(self, context):
        self.layout.prop(self, 'weight_threshold')
        self.layout.prop(self, 'fit_strength', slider=True)
        self.layout.prop(self, 'precision_fit', toggle=True)

    def execute(self, context):
        try:
            arm, bones = selected_bones(context)
            sources = [obj for obj in context.scene.objects if obj.type == 'MESH'
                       and any(m.type == 'ARMATURE' and m.object == arm for m in obj.modifiers)]
            fit_strength = getattr(self, 'fit_strength', 0.0)
            threshold = float(self.weight_threshold)
            precision = bool(fit_strength > 0.0 and getattr(self, 'precision_fit', False))
            bounds, points, source_faces = {}, {}, {}
            arm_inverse = arm.matrix_world.inverted()
            for obj in sources:
                groups = {vg.index: bone for bone in bones
                          if (vg := obj.vertex_groups.get(bone.name)) is not None}
                if not groups:
                    continue
                transforms = {index: bone.matrix_local.inverted() @ arm_inverse @ obj.matrix_world
                              for index, bone in groups.items()}
                membership_indices = {} if precision else None
                for vertex in obj.data.vertices:
                    for membership in vertex.groups:
                        index = membership.group
                        if (index not in groups or membership.weight <= 0
                                or membership.weight < threshold):
                            continue
                        bone = groups[index]
                        point = transforms[index] @ vertex.co
                        if membership_indices is not None:
                            ix = len(points.get(bone.name, []))
                            if ix < MAX_SAMPLES:
                                membership_indices.setdefault(vertex.index, {})[bone.name] = ix
                        points.setdefault(bone.name, []).append((point, membership.weight))
                        if bone.name not in bounds:
                            bounds[bone.name] = ([float(v) for v in point], [float(v) for v in point])
                        else:
                            lo, hi = bounds[bone.name]
                            for axis in range(3):
                                lo[axis] = min(lo[axis], point[axis])
                                hi[axis] = max(hi[axis], point[axis])
                if membership_indices:
                    obj.data.calc_loop_triangles()
                    for face in obj.data.loop_triangles:
                        a, b, c = (membership_indices.get(i, {}) for i in face.vertices)
                        for bn in a.keys() & b.keys() & c.keys():
                            if len(points[bn]) <= MAX_SAMPLES:
                                source_faces.setdefault(bn, []).append((a[bn], b[bn], c[bn]))
            rows, missing, fell_back, repaired = {}, [], [], []
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
                if precision:
                    try:
                        vertices, faces = fit_height_envelope(
                            [point[:3] for point in weighted], source_faces.get(bone.name, []),
                            sum(len(face)-2 for face in faces), fit_strength)
                    except (ValueError, RuntimeError, np.linalg.LinAlgError):
                        fell_back.append(bone.name)
                        vertices = deform_weight_contour(vertices, faces, weighted, fit_strength)
                    else:
                        offset = _known_absolute_contact_offset(context.scene, arm, bone)
                        if offset > 0.0:
                            from .bone_weight_repair import repair
                            try:
                                candidate_v, candidate_f, changes = repair(
                                    [point[:3] for point in weighted], source_faces[bone.name],
                                    vertices, faces, offset)
                            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                                candidate_v = candidate_f = None
                            if candidate_v is None:
                                raise ValueError(iface(
                                    'Precision Fit could not repair {bone} at its current contact offset while preserving the contour and coverage; no proxies were changed.'
                                ).format(bone=bone.name))
                            if len(changes) > 1:
                                vertices, faces = candidate_v.tolist(), candidate_f.tolist()
                                repaired.append(bone.name)
                else:
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
            if repaired:
                self.report({'INFO'}, iface(
                    'Precision Fit repaired local face clearance for {bones} without changing contact offset.'
                ).format(bones=', '.join(repaired[:8]) + (f' (+{len(repaired)-8})' if len(repaired) > 8 else '')))
            if fell_back:
                names = ', '.join(fell_back[:8])
                if len(fell_back) > 8:
                    names += f' (+{len(fell_back)-8})'
                self.report({'WARNING'}, iface(
                    'Precision Fit fell back to convex fitting for {bones}; the selected skin or triangle budget did not meet the precision checks.'
                ).format(bones=names))
            return {'FINISHED'}
        except (ValueError, RuntimeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


def register():
    register_class(PPF_OT_FitBoneProxiesToWeights)
    bpy.types.WindowManager.ppf_bone_fit_weight_threshold = _minimum_weight_property(options={'SKIP_SAVE'})
    bpy.types.WindowManager.ppf_bone_fit_strength = _contour_fit_property(options={'SKIP_SAVE'})
    bpy.types.WindowManager.ppf_bone_fit_precision = _precision_fit_property(options={'SKIP_SAVE'})


def unregister():
    del bpy.types.WindowManager.ppf_bone_fit_precision
    del bpy.types.WindowManager.ppf_bone_fit_strength
    del bpy.types.WindowManager.ppf_bone_fit_weight_threshold
    bpy.utils.unregister_class(PPF_OT_FitBoneProxiesToWeights)
