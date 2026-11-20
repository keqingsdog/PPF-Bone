"""Create actual closed mesh objects in the original PPF authoring groups.

Candidate (jointed-material-addon-staging): the proxy is placed in a PDRD group
(RIGID, the original behaviour and the default) or in a real SOLID group (SOFT),
which is the official TetGen + Stable NeoHookean path. One bone still owns
exactly ONE material group. An existing proxy switches only through the explicit
batch operator, whose synchronous core retirement requires a fresh Transfer and
Build before its new material can authorize frames.
"""
from .. import i18n as _ppf_i18n
import math
import uuid

import bpy
from ..core.compat import pose_bone_selected,set_pose_bone_selected
from ..i18n import pgettext_iface as iface
import bmesh
from ..i18n.props import BoolProperty, EnumProperty, IntProperty, StringProperty
from mathutils import Matrix, Vector

from ..models.bone_bindings import active_armature, has_physical_joint_parent, check_soft_supported, ensure_binding_ids, rigid_matrix, bone_world_matrix, bone_world_length, soft_material_refusal, sync_owned_material, validate_proxy_owner, validate_final_binding_ids
from ..models.groups import assign_display_indices, get_addon_data, get_group_by_uuid, has_addon_data, invalidate_overlays, iterate_active_object_groups, migrate_bone_material_groups, remove_bone_material_group
from ..models.new_binding_defaults import new_binding_plan, NEW_SOFT_DAMPING_SECONDS, NEW_GENERATED_SOFT_TET_BACKEND, record_auto_geometry, geometry_snapshot_matches, deletion_geometry_intent, recreate_geometry_intent
from ..models.proxy_generation import round_geometry, triangulate_faces, effective_proxy_quality, effective_proxy_face_mode
from ..core.uuid_registry import get_object_uuid, get_or_create_object_uuid


SCOPES = [('SELECTED', 'Selected Bones', 'Selected pose bones'), ('CHAIN', 'Active Chain', 'Active bone and all descendants'), ('ALL', 'All Bones', 'Entire armature')]
OBJECT_TYPES = {'RIGID': 'PDRD', 'SOFT': 'SOLID'}
PHYSICAL_LABELS = {'RIGID': 'Rigid Body', 'SOFT': 'Soft Body'}
_RECREATE_SIZE_POLICY = 'ppf_bone_recreate_size_policy'
# Properties whose value is material-group state rather than per-bone authoring.
# They are never copied field by field: the material group is the single source
# of truth and is validated as a whole before a batch writes anything.
MATERIAL_OWNED_PROPERTIES = ('physical_model', 'solid_model', 'solid_young_modulus', 'solid_poisson_ratio', 'solid_deformation_damping')


def object_type_for(settings):
    return OBJECT_TYPES[settings.physical_model]


def enter_proxy_pose(context, scope='SELECTED'):
    """Resolve names before an Edit/Object transition; never retain stale RNA."""
    arm=active_armature(context)
    if arm is None:raise ValueError('请先选择一个骨架')
    if sum(obj.type=='ARMATURE' for obj in context.selected_objects)>1:
        raise ValueError('请选择一个骨架，避免跨骨架选择歧义')
    original=context.mode
    if original=='EDIT_ARMATURE':
        names=[b.name for b in arm.data.edit_bones if b.select]
        active=arm.data.edit_bones.active
    else:
        names=[p.name for p in arm.pose.bones if pose_bone_selected(p)]
        active=arm.data.bones.active
    active_name=active.name if active else None
    if scope!='ALL' and (not names or scope=='CHAIN' and active_name not in names):
        raise ValueError('请先选中骨骼；仅选择骨架物体不会指定要添加的骨骼')
    if original!='POSE':
        if original!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.mode_set(mode='POSE')
        for pb in arm.pose.bones:set_pose_bone_selected(pb,pb.name in names)
        if active_name:arm.data.bones.active=arm.data.bones.get(active_name)
    return original


def selected_bones(context, all_bones=False, scope='SELECTED'):
    arm = active_armature(context)
    if not arm:
        raise ValueError("Select an armature in Object or Pose mode")
    if context.mode not in {"OBJECT", "POSE"}:
        raise ValueError("Finish Edit mode before creating physics proxies")
    if all_bones or scope == 'ALL':
        bones = list(arm.data.bones)
    elif scope == 'CHAIN':
        active = arm.data.bones.active
        selected = {p.name for p in arm.pose.bones if pose_bone_selected(p)}
        bones = [active, *active.children_recursive] if active and active.name in selected else []
    elif context.mode == "POSE":
        selected=list(context.selected_pose_bones or [])
        if any(getattr(p,'id_data',arm)!=arm for p in selected):
            raise ValueError('请选择一个骨架，避免跨骨架选择歧义')
        bones = [p.bone for p in selected]
    else:
        bones = [p.bone for p in arm.pose.bones if pose_bone_selected(p)]
    if not bones:
        raise ValueError("Select a pose bone, or choose All Bones")
    return arm, bones


def capsule_geometry(length, radius, quality='HIGH', physical_model='RIGID', face_mode='QUAD', legacy=False):
    # physical_model retained for source compatibility; presets are explicit.
    return round_geometry(length, radius, quality, face_mode, legacy=legacy)


def sphere_geometry(radius, quality='MEDIUM', face_mode='QUAD'):
    return round_geometry(2*radius, radius, quality, face_mode, sphere=True)


def box_geometry(length, width, depth, face_mode='QUAD', quality='LOW'):
    vertices = [(x*length/2, y*width/2, z*depth/2) for x, y, z in [(-1,-1,-1),(-1,-1,1),(-1,1,-1),(-1,1,1),(1,-1,-1),(1,-1,1),(1,1,-1),(1,1,1)]]
    faces = [(0,2,3,1),(4,5,7,6),(0,1,5,4),(2,6,7,3),(0,4,6,2),(1,3,7,5)]
    subdivisions = 4 if quality == 'VERY_HIGH' else 2 if quality == 'HIGH' else 1
    if subdivisions > 1:
        corners=[tuple(round(2*v/d) for v,d in zip(p,(length,width,depth))) for p in vertices]
        points=[]; indices={}; patches=[]; n=subdivisions
        for face in faces:
            a,b,_c,d=[corners[k] for k in face]
            grid={}
            for j in range(n+1):
                for i in range(n+1):
                    key=tuple(n*a[k]+i*(b[k]-a[k])+j*(d[k]-a[k]) for k in range(3))
                    if key not in indices:
                        indices[key]=len(points)
                        points.append(tuple(key[k]*size/(2*n) for k,size in enumerate((length,width,depth))))
                    grid[i,j]=indices[key]
            for j in range(n):
                for i in range(n):patches.append((grid[i,j],grid[i+1,j],grid[i+1,j+1],grid[i,j+1]))
        vertices,faces=points,patches
    return vertices, triangulate_faces(faces,vertices) if face_mode == 'TRI' else faces


def builtin_geometry(length, settings, plan=None):
    plan = plan or {}
    shape = plan.get('shape', settings.shape)
    quality = plan.get('proxy_quality', effective_proxy_quality(settings))
    face_mode = plan.get('proxy_face_mode', effective_proxy_face_mode(settings))
    if shape == 'CAPSULE':
        return capsule_geometry(length, plan.get('radius', settings.radius), quality, plan.get('physical_model', settings.physical_model), face_mode)
    if shape == 'SPHERE':
        return sphere_geometry(plan.get('radius', settings.radius), quality, face_mode)
    if shape == 'BOX':
        return box_geometry(length, plan.get('box_width', settings.box_width), plan.get('box_depth', settings.box_depth), face_mode, quality)
    raise ValueError('Unknown builtin proxy shape')


def validate_closed_mesh(mesh):
    bm = bmesh.new()
    triangles = None
    try:
        bm.from_mesh(mesh)
        if len(bm.verts) < 4 or not bm.faces or any(not e.is_manifold for e in bm.edges):
            raise ValueError("Proxy must be a closed manifold mesh")
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
        triangles = bm.copy()
        bmesh.ops.triangulate(triangles, faces=list(triangles.faces))
        if any(f.calc_area() <= 1e-14 for f in triangles.faces):
            raise ValueError("Proxy contains a degenerate face")
        if triangles.calc_volume(signed=True) <= 1e-15:
            raise ValueError("Proxy must enclose a positive volume")
        # Retain historical outward-normal repair, but never write the
        # triangulated validation copy back to the author's mesh.
        bm.to_mesh(mesh)
        mesh.update()
    finally:
        if triangles is not None:
            triangles.free()
        bm.free()


def _validate_bone_proxy_mesh(mesh, arm, bone, *, generated=False):
    """Keep strict mesh gates and report the exact author target on refusal."""
    try:
        validate_closed_mesh(mesh)
    except ValueError as error:
        if generated:
            size = tuple(max(v.co[i] for v in mesh.vertices) - min(v.co[i] for v in mesh.vertices)
                         for i in range(3)) if mesh.vertices else (0., 0., 0.)
            raise ValueError(iface(
                "Generated proxy for {armature}/{bone} rejected (bone length {length}; mesh size {size}): {reason}. "
                "Check this bone's length and proxy dimensions, or select only the bones that need physics."
            ).format(armature=arm.name, bone=bone.name,
                     length=f"{bone_world_length(arm, bone):.6g}",
                     size=" x ".join(f"{value:.6g}" for value in size),
                     reason=iface(str(error)))) from error
        proxy = bone.ppf_bone_physics.proxy
        raise ValueError(iface(
            "Proxy {proxy} for {armature}/{bone} rejected: {reason}. "
            "Repair this proxy mesh and retry; authored geometry is retained."
        ).format(proxy=proxy.name if proxy else mesh.name, armature=arm.name,
                 bone=bone.name, reason=iface(str(error)))) from error


def _generation_frame(arm, bone):
    settings=bone.ppf_bone_physics
    if settings.proxy is not None:
        return rigid_matrix(settings.proxy.matrix_world.copy(), settings.proxy.name)
    return bone_world_matrix(arm,bone) @ Matrix.Translation((0.,bone_world_length(arm,bone)/2,0.)) @ Matrix.Rotation(math.pi/2,4,'Z')


def _make_mesh(context, arm, bone, creation_plan=None):
    settings = bone.ppf_bone_physics
    plan=creation_plan or {}
    shape=plan.get('shape',settings.shape)
    frame = _generation_frame(arm,bone)
    mesh = bpy.data.meshes.new(f"PPF_{bone.name}_ProxyMesh")
    try:
        if shape == "CUSTOM":
            source = settings.custom_object
            if not source or source.type != "MESH":
                raise ValueError("Choose a closed custom mesh first")
            evaluated = source.evaluated_get(context.evaluated_depsgraph_get())
            copied = bpy.data.meshes.new_from_object(evaluated, depsgraph=context.evaluated_depsgraph_get())
            bpy.data.meshes.remove(mesh)
            mesh = copied
            # Bake the source's actual world geometry into a unit-scale body;
            # original source data/modifiers/transform are never edited.
            mesh.transform(frame.inverted() @ source.matrix_world)
        else:
            length = bone_world_length(arm, bone) * plan.get('length_ratio',settings.length_ratio)
            vertices, faces = builtin_geometry(length, settings, plan)
            mesh.from_pydata(vertices, [], faces)
        _validate_bone_proxy_mesh(mesh, arm, bone, generated=shape != "CUSTOM")
        return mesh, frame
    except Exception:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
        raise


def is_owned_bone_group(group, object_type, body_uuid):
    return bool(group and group.active and group.object_type == object_type
                and len(group.assigned_objects) == 1
                and group.assigned_objects[0].uuid == body_uuid)


def _existing_generated_capsule(context, arm, bone, group, *, mesh_dimensions=False):
    """Describe an unchanged generated capsule from its mesh and live transform.

    A proxy is treated as a fixed capsule only when its actual local mesh still
    matches the generator output. Edited/custom meshes are never reconstructed
    from RNA dimensions and remain untouched.
    """
    s=bone.ppf_bone_physics;obj=s.proxy
    if not obj or s.shape!='CAPSULE':return None
    # A canonical base mesh does not prove its evaluated collision surface.
    # Do not guess whether a modifier or key happens to be deforming today;
    # these owners must use the evaluated-world conservative bound below.
    if obj.data.shape_keys is not None or any(mod.show_viewport for mod in obj.modifiers):
        return None
    try:
        from ..core.encoder.params import group_contact_lengths
        quality=effective_proxy_quality(s)
        candidates=[(quality,effective_proxy_face_mode(s),False),(quality,'TRI',False),(quality,'TRI',True)]
        # Recognize original generated meshes without rewriting their author
        # settings when their old SOFT LOW identifier has a new LOD name.
        if s.proxy_quality=='LOW' and s.physical_model=='SOFT':candidates.append(('MEDIUM','TRI',True))
        mesh=obj.data
        if any(not math.isfinite(c) for v in mesh.vertices for c in v.co):return None
        actual_faces={tuple(sorted(p.vertices)) for p in mesh.polygons}
        length=bone_world_length(arm,bone)*s.length_ratio
        radius=s.radius
        if mesh_dimensions:
            # Old author-follow baked uniform size into vertices without updating
            # dimension RNA. Delete is an explicit default-rebuild action; admit
            # that template only after matching every vertex and polygon below.
            # Bounds propose parameters, never prove a generated shape or AUTO.
            if not mesh.vertices:return None
            length=max(v.co.x for v in mesh.vertices)-min(v.co.x for v in mesh.vertices)
            radius=max(v.co.y for v in mesh.vertices)
        scale=max(radius,length) if mesh_dimensions else max(1.,radius,length)
        tolerance=scale*(16.*2.**-23 if mesh_dimensions else 2.e-7)
        for candidate_quality,candidate_mode,legacy in dict.fromkeys(candidates):
            expected_vertices,expected_faces=capsule_geometry(length,radius,candidate_quality,s.physical_model,candidate_mode,legacy=legacy)
            if len(mesh.vertices)!=len(expected_vertices) or len(mesh.polygons)!=len(expected_faces):continue
            if any((mesh.vertices[i].co-Vector(v)).length>tolerance for i,v in enumerate(expected_vertices)):continue
            if actual_faces!={tuple(sorted(f)) for f in expected_faces}:continue
            break
        else:return None
        matrix=rigid_matrix(obj.matrix_world.copy(),obj.name)
        axis=matrix.to_3x3()@Vector((1.,0.,0.))
        length=expected_vertices[-1][0]-expected_vertices[0][0]
        center=matrix.translation
        bone_head=bone_world_matrix(arm,bone).translation
        direction=axis.normalized()
        gap,offset=group_contact_lengths(group)
        return dict(eligible=False,length=length,radius=radius,shape='CAPSULE',gap=gap,offset=offset,
            coordinate_scale=max(abs(v) for p in (center-direction*(length/2),center+direction*(length/2)) for v in p),capsule_known=True,
            head=tuple(center-direction*(length/2)),tail=tuple(center+direction*(length/2)),
            direction=tuple(direction),bone_head=tuple(bone_head))
    except (ValueError, RuntimeError, TypeError, AttributeError):
        return None


def _group(scene, bone, creation_plan=None, *, initialize_material=True):
    """The bone's single material group, in the selected physical body.

    A group that already exists with the OTHER object type is a refusal rather
    than a silent conversion: the two bodies need different geometry processing
    (a SOLID body is tetrahedralized; a PDRD body is not), so reinterpreting one
    as the other would ship a scene the author never asked for.
    """
    s = bone.ppf_bone_physics
    object_type = object_type_for(s)
    group = get_group_by_uuid(scene, s.group_uuid) if s.group_uuid else None
    existing_group = bool(group and group.active)
    if existing_group:
        if group.object_type != object_type:
            raise ValueError(iface(
                "{name}: this proxy is already a {current} body; delete the proxy before switching the physical body to {wanted}"
            ).format(name=bone.name, current=iface(PHYSICAL_LABELS['SOFT' if group.object_type == 'SOLID' else 'RIGID']),
                     wanted=iface(PHYSICAL_LABELS[s.physical_model])))
        if any(a.uuid != s.body_uuid for a in group.assigned_objects):
            raise ValueError(f"{bone.name}: its dedicated physics group was reassigned")
        # The bone writes material values only into a group it owns outright: a
        # shared or mixed group is refused rather than steered from here.
        if (len(group.assigned_objects) != 1 or group.assigned_objects[0].uuid != s.body_uuid):
            raise ValueError(f"{bone.name}: its material group must hold exactly this proxy before settings are synced")
    else:
        group = get_addon_data(scene).bone_material_groups.add()
        group.reset_to_defaults();group.active = True;group.ensure_uuid()
        group.name = f"Bone · {bone.name}"
        s.group_uuid = group.uuid
        if creation_plan and creation_plan.get('physical_model')=='SOFT':group.deformation_damping=NEW_SOFT_DAMPING_SECONDS
    group.object_type = object_type
    # Preset transactions supply a separately prevalidated material plan.
    # Do not first write an unrepresentable true modulus into fresh SOLID RNA.
    if not initialize_material:return group
    if object_type == 'PDRD':
        group.pdrd_density = s.density
    elif not existing_group:
        # Real SOLID material. Density is the author's value in kg/m^3; the
        # modulus is a true pascal value paired with the non-normalized switch,
        # so the encoder applies the solver's density-normalized convention.
        group.solid_model = s.solid_model
        group.solid_density = s.density
        group.solid_young_modulus = s.solid_young_modulus
        group.solid_poisson_ratio = s.solid_poisson_ratio
        group.young_mod_density_normalized = False
        if s.is_property_set('solid_deformation_damping'): group.deformation_damping = s.solid_deformation_damping
    return group


def _proxy_geometry_state(proxy):
    if proxy is None:return None
    return (tuple(tuple(v.co) for v in proxy.data.vertices),
            tuple(tuple(edge.vertices) for edge in proxy.data.edges),
            tuple(tuple(p.vertices) for p in proxy.data.polygons),
            tuple(tuple(row) for row in proxy.matrix_world))


def update_proxies(context, all_bones=False, scope='SELECTED', rebuild=False, reset_default_size=False, shape_changed_from=None, *, authoring_rows=None, after_apply=None):
    """Keep author follow and proxy updates inside one rollback boundary."""
    from ..core.bone_geometry_edit import preflight
    from ..core.bone_author_follow import transfer_author_transaction
    preflight(context)
    # Follow may move unselected rows too, just as the ordinary FramePump does.
    # Hold every existing author mesh/baseline until the selected update commits.
    from ..models.bone_collision import collision_edit_batch
    with transfer_author_transaction(context) as commit_author, collision_edit_batch():
        count = _update_proxies_author_transaction(
            context, all_bones, scope, rebuild, reset_default_size, shape_changed_from, authoring_rows, after_apply)
        commit_author()
        return count


def _update_proxies_author_transaction(context, all_bones=False, scope='SELECTED', rebuild=False, reset_default_size=False, shape_changed_from=None, authoring_rows=None, after_apply=None):
    """Create new curve authority in the same undoable proxy transaction."""
    from ..core.bone_geometry_edit import preflight
    preflight(context)
    if reset_default_size:
        _reset_arm, reset_bones = selected_bones(context, all_bones, scope)
        if any(b.ppf_bone_physics.shape == 'CUSTOM' for b in reset_bones):
            raise ValueError('自定义网格不支持重算默认尺寸；批量操作未更改任何骨骼')
    reconcile_deleted_proxies(context.scene)
    from .restoring_curve_operators import read_state,write_state,snapshot_rna,restore_rna
    from .restoring_curves import add_new_common
    arm,bones=selected_bones(context,all_bones,scope)
    if authoring_rows is not None:
        unknown = set(authoring_rows) - {bone.name for bone in bones}
        if unknown or not authoring_rows:
            raise ValueError('Authoring rows must name selected bones')
        bones = [bone for bone in bones if bone.name in authoring_rows]
    rebuild = rebuild or reset_default_size
    before_geometry=[_proxy_geometry_state(b.ppf_bone_physics.proxy) for b in bones]
    owner=context.scene.ppf_restoring_curves
    dormant=owner.mode=='LEGACY' and bool(owner.document_json or owner.editor_target_json or len(owner.nodes))
    previous={'mode':'LEGACY','document':None} if dormant else read_state(context.scene)
    author_snapshot=snapshot_rna(owner)
    settings=[(bone,snapshot_rna(bone.ppf_bone_physics)) for bone in bones]
    arm_id=arm.ppf_armature_uuid
    arm_id_set=arm.is_property_set('ppf_armature_uuid')
    existing_objects={obj.as_pointer() for obj in bpy.data.objects}
    data=get_addon_data(context.scene)
    group_snapshot=snapshot_rna(data)
    backups=[(b.ppf_bone_physics.proxy,b.ppf_bone_physics.proxy.data,_proxy_channels(b.ppf_bone_physics.proxy,context.scene))
             for b in bones if b.ppf_bone_physics.proxy]
    retained_meshes={mesh.as_pointer() for _obj,mesh,_channels in backups}
    existing_meshes={mesh.as_pointer() for mesh in bpy.data.meshes}
    new_visibility=[]
    new_resources={'objects': [], 'collections': []}
    try:
        if authoring_rows is not None:
            for bone in bones:
                for key, value in authoring_rows[bone.name].get('settings', {}).items():
                    setattr(bone.ppf_bone_physics, key, value)
        fresh={bone.name for bone in bones if not bone.ppf_bone_physics.enabled and not bone.ppf_bone_physics.proxy
               and not any(getattr(bone.ppf_bone_physics,key) for key in ('bone_uuid','body_uuid','joint_uuid','group_uuid'))}
        planned_ids={name:str(uuid.uuid4()) for name in fresh}
        enabled={b.name for b in arm.data.bones if b.ppf_bone_physics.enabled}|{b.name for b in bones}
        targets=[{'kind':'JOINT','joint_uuid':planned_ids[b.name],'channel':channel}
                 for b in bones if b.name in fresh and b.parent
                 and has_physical_joint_parent(b, enabled)
                 and b.ppf_bone_physics.joint_mode in {'SOFT','BALL_FIXED'}
                 for channel in (('twist_x','swing_y','swing_z') if b.ppf_bone_physics.joint_mode=='BALL_FIXED' else ('twist_x','swing_y','swing_z','anchor'))]
        data=get_addon_data(context.scene)
        candidate=add_new_common(previous,targets,[p.pair_id for p in data.state.merge_pairs],
                                 has_draft=dormant or bool(owner.editor_target_json or len(owner.nodes)))
        from ..core.bone_author_follow import flush_author_before_update
        follow_changed = flush_author_before_update(context)
        count=_update_proxies(context,all_bones,scope,rebuild,planned_ids,retained_meshes,new_visibility,reset_default_size,shape_changed_from,new_resources,authoring_rows)
        if after_apply is not None:
            after_apply(arm, bones)
        for bone in bones:
            if bone.name in fresh and not bone.ppf_bone_physics.is_property_set('soft_limit_policy'):
                bone.ppf_bone_physics.soft_limit_policy='AUTO_INERTIA'
                bone.ppf_bone_physics.soft_limit_frequency_hz=10.
                bone.ppf_bone_physics.soft_limit_strength=1.
        if candidate!=previous:write_state(context.scene,candidate)
        if not follow_changed and before_geometry!=[_proxy_geometry_state(b.ppf_bone_physics.proxy) for b in bones]:
            _facade_communicator().bone_physical_body_changed()
        return count
    except Exception:
        from ..core.bone_proxy_visibility import restore_proxy
        for obj,token in reversed(new_visibility):restore_proxy(obj,token)
        restore_rna(owner,author_snapshot)
        with _rollback_guard():
            restore_rna(data,group_snapshot)
            if not group_snapshot['bone_material_groups'][0] and not data.bone_material_groups:
                data.property_unset('bone_material_groups')
            for bone,snapshot in settings:restore_rna(bone.ppf_bone_physics,snapshot)
            arm.ppf_armature_uuid=arm_id
            if not arm_id_set:arm.property_unset('ppf_armature_uuid')
        replacements=[]
        for obj,mesh,channels in backups:
            replacement=obj.data
            obj.data=mesh
            _restore_proxy_channels(obj,channels)
            if replacement!=mesh and replacement.as_pointer() not in existing_meshes:
                replacements.append(replacement)
        for obj in list(bpy.data.objects):
            if obj.as_pointer() not in existing_objects and obj.get('ppf_bone_proxy_owned',False):
                mesh=obj.data;bpy.data.objects.remove(obj,do_unlink=True)
                if mesh.users==0:bpy.data.meshes.remove(mesh)
        for mesh in replacements:
            if mesh.users==0:bpy.data.meshes.remove(mesh)
        # Only auxiliary containers created by this transaction are retired.
        # Existing or unexpectedly nonempty user containers remain untouched.
        for obj in reversed(new_resources['objects']):
            try:
                if bpy.data.objects.get(obj.name)==obj and not obj.children:
                    bpy.data.objects.remove(obj,do_unlink=True)
            except ReferenceError:pass  # Presentation rollback may already remove it.
        for collection in reversed(new_resources['collections']):
            try:
                if not collection.objects and not collection.children:
                    bpy.data.collections.remove(collection)
            except ReferenceError:pass
        raise
    finally:
        # Resolve shared backup identities before releasing any live RNA.
        backup_meshes={mesh.as_pointer():mesh for _obj,mesh,_channels in backups}
        for mesh in backup_meshes.values():
            if mesh.users==0:bpy.data.meshes.remove(mesh)


def _proxy_channels(obj, scene=None):
    from ..core.bone_proxy_visibility import snapshot_proxy
    scalars=('parent','parent_type','parent_bone','rotation_mode','display_type','show_in_front','hide_render')
    vectors=('location','rotation_euler','rotation_quaternion','rotation_axis_angle','scale',
             'delta_location','delta_rotation_euler','delta_rotation_quaternion','delta_scale','color')
    return {**{key:getattr(obj,key) for key in scalars},
            **{key:tuple(getattr(obj,key)) for key in vectors},
            'matrix_parent_inverse':obj.matrix_parent_inverse.copy(),
            '_recreate_size_policy':(_RECREATE_SIZE_POLICY in obj,obj.get(_RECREATE_SIZE_POLICY)),
            '_visibility':snapshot_proxy(obj,scene)}


def _restore_proxy_channels(obj,channels):
    # Restore channels directly: matrix decomposition would alter untouched
    # rotation/scale and lose exact authored values after a failed rebuild.
    from ..core.bone_proxy_visibility import restore_proxy
    for key,value in channels.items():
        if key=='_recreate_size_policy':
            if value[0]:obj[_RECREATE_SIZE_POLICY]=value[1]
            elif _RECREATE_SIZE_POLICY in obj:del obj[_RECREATE_SIZE_POLICY]
        elif key!='_visibility':setattr(obj,key,value)
    restore_proxy(obj,channels['_visibility'])


def _rollback_guard():
    from ..core.bone_mass import material_write_guard
    return material_write_guard()


def _update_proxies(context, all_bones=False, scope='SELECTED', rebuild=False, planned_joint_ids=None, retained_meshes=None, new_visibility=None, reset_default_size=False, shape_changed_from=None, new_resources=None, authoring_rows=None):
    arm, bones = selected_bones(context, all_bones, scope)
    if authoring_rows is not None:
        bones = [bone for bone in bones if bone.name in authoring_rows]
    if not has_addon_data(context.scene):
        raise ValueError("Register the original PPF addon before creating proxies")
    validate_final_binding_ids(context.scene,arm,bones)
    for bone in bones:validate_proxy_owner(arm,bone)
    selected_names={b.name for b in bones}
    enabled_names=selected_names|{b.name for b in arm.data.bones if b.ppf_bone_physics.enabled}
    creation_plans={b.name:new_binding_plan(b.ppf_bone_physics,bone_world_length(arm, b),reset_default_size,arm.ppf_armature_uuid) for b in bones}
    if authoring_rows is not None:
        for bone in bones:
            if bone.ppf_bone_physics.proxy is not None:
                creation_plans[bone.name] = {}
    # Authoring may intentionally overlap. Collision clearance belongs to the
    # real build/solver policy, never a coarse pre-creation fit or refusal.
    planned_models={b.name:creation_plans.get(b.name,{}).get('physical_model',b.ppf_bone_physics.physical_model) for b in arm.data.bones}
    for bone in arm.data.bones:
        if bone.name not in enabled_names:continue
        check_soft_supported(bone.ppf_bone_physics,bone.name,has_physical_joint_parent(bone, enabled_names),assumed_model=planned_models[bone.name])
        for child_name,kind,mode in incident_edge_refusals(bone,planned_models,enabled_names):
            if bone.name in selected_names or child_name in selected_names:
                raise ValueError(f'{bone.name}: unsupported {kind} edge to {child_name} ({mode})')
    prepared = []
    from ..core.bone_mass import prepare_mass_batch, apply_mass_batch, invalidate_all, material_write_guard
    try:
        # Validate every selected shape before writing any proxy/group.
        for bone in bones:
            s = bone.ppf_bone_physics
            check_soft_supported(s, bone.name, has_physical_joint_parent(bone, enabled_names),assumed_model=planned_models[bone.name])
            if s.proxy and not s.proxy.get("ppf_bone_proxy_owned", False):
                raise ValueError(f"{bone.name}: refusing to replace an unrelated object")
            if s.proxy and get_object_uuid(s.proxy) != s.body_uuid:
                raise ValueError(f"{bone.name}: proxy identity changed")
            if s.proxy and s.proxy.name not in context.scene.objects and s.proxy.users_scene:
                raise ValueError('Bone proxy exists in another scene; restore it to this scene before Add / Update')
            if not s.proxy and s.body_uuid and any(o.get('_solver_uuid') == s.body_uuid for o in bpy.data.objects):
                raise ValueError(f"{bone.name}: persistent body identity is owned by another object")
            group = get_group_by_uuid(context.scene, s.group_uuid) if s.group_uuid else None
            if group and group.active:
                if group.object_type != object_type_for(s):
                    raise ValueError(iface(
                        "{name}: this proxy is already a {current} body; delete the proxy before switching the physical body to {wanted}"
                    ).format(name=bone.name, current=iface(PHYSICAL_LABELS['SOFT' if group.object_type == 'SOLID' else 'RIGID']),
                             wanted=iface(PHYSICAL_LABELS[s.physical_model])))
                if any(a.uuid != s.body_uuid for a in group.assigned_objects):
                    raise ValueError(f"{bone.name}: dedicated group has unrelated assignments")
                if len(group.assigned_objects)!=1:
                    raise ValueError(f'{bone.name}: its material group must hold exactly this proxy before settings are synced')
            if authoring_rows is not None and authoring_rows[bone.name].get("geometry") is not None:
                vertices, faces, frame = authoring_rows[bone.name]["geometry"]
                mesh = bpy.data.meshes.new(f"PPF_Bone_{bone.name}_fit")
                try:
                    mesh.from_pydata(vertices, [], faces)
                    mesh.update()
                    _validate_bone_proxy_mesh(mesh, arm, bone, generated=True)
                except Exception:
                    bpy.data.meshes.remove(mesh)
                    raise
            elif s.proxy and not rebuild:
                # Update authoring metadata without erasing edited geometry or pose.
                # Validate a disposable copy without changing authored topology.
                checked = s.proxy.data.copy()
                try:
                    _validate_bone_proxy_mesh(checked, arm, bone)
                    frame = rigid_matrix(s.proxy.matrix_world.copy(), s.proxy.name)
                finally:
                    bpy.data.meshes.remove(checked)
                mesh = None
            else:
                mesh, frame = _make_mesh(context, arm, bone,creation_plans[bone.name])
            prepared.append((bone, mesh, frame))
        mass_plan = prepare_mass_batch(context.scene, [(arm, bone, mesh, frame, creation_plans[bone.name].get('density')) for bone, mesh, frame in prepared])
        planned_rho = {bone.name: rho for candidate_arm, bone, _group, rho, _result in mass_plan if candidate_arm == arm}
        # Preflight every newly initialized SOFT modulus before IDs, authors,
        # proxies or groups are committed. MASS uses the final planned volume.
        from types import SimpleNamespace
        group_type = get_addon_data(context.scene).bl_rna.properties['bone_material_groups'].fixed_type
        new_solid_plans = {}
        for bone in bones:
            settings = bone.ppf_bone_physics
            group = get_group_by_uuid(context.scene, settings.group_uuid) if settings.group_uuid else None
            if planned_models[bone.name] != 'SOFT' or group is not None and group.active:
                continue
            values = creation_plans[bone.name]
            density = planned_rho.get(bone.name, values.get('density', float(settings.density)))
            material = {key: values.get(key, getattr(settings, key)) for key in
                        ('solid_model', 'solid_young_modulus', 'solid_poisson_ratio', 'solid_deformation_damping')}
            try:
                writes, expected = _plan_copied_soft_material(
                    settings, SimpleNamespace(bl_rna=group_type), material, density, False)
            except ValueError as error:
                raise ValueError(bone.name + '：无法创建保真的软体材质；' + str(error)) from error
            # Preserve existing non-E initialization semantics, including unset
            # damping on legacy explicit SOFT authors. This is not drift repair.
            new_solid_plans[bone.name] = (
                {key: value for key, value in writes.items() if key in
                 ('solid_density', 'solid_young_modulus', 'young_mod_density_normalized')},
                {'solid_young_modulus': expected['solid_young_modulus']})
        for bone in bones:
            if planned_joint_ids and bone.name in planned_joint_ids:
                bone.ppf_bone_physics.joint_uuid=planned_joint_ids[bone.name]
        # Validation has completed. Retain legacy UUIDs/materials and free the
        # original user slots before generating any new dedicated material.
        migrate_bone_material_groups(context.scene)
        for bone, mesh, frame in prepared:
            old_intent=recreate_geometry_intent(bone.ppf_bone_physics,arm.ppf_armature_uuid)
            with material_write_guard():
                for key,value in creation_plans[bone.name].items():
                    if not key.startswith('_'):setattr(bone.ppf_bone_physics,key,value)
                if creation_plans[bone.name].get('_auto_geometry_defaults'):record_auto_geometry(bone.ppf_bone_physics)
                if bone.name in planned_rho: bone.ppf_bone_physics.density = planned_rho[bone.name]
            s = ensure_binding_ids(arm, bone)
            proxy = s.proxy
            created = proxy is None
            if not proxy:
                proxy = bpy.data.objects.new(f"PPF_Bone_{bone.name}", mesh)
                context.scene.collection.objects.link(proxy)
                if s.body_uuid:
                    if any(o.get('_solver_uuid') == s.body_uuid for o in bpy.data.objects if o != proxy):
                        raise ValueError("Persistent body UUID is already owned by another object")
                    proxy['_solver_uuid'] = s.body_uuid
                s.proxy = proxy
            elif mesh is not None:
                previous = proxy.data;proxy.data = mesh
                if previous.users == 0 and (not retained_meshes or previous.as_pointer() not in retained_meshes):
                    bpy.data.meshes.remove(previous)
            if proxy.name not in context.scene.objects:context.scene.collection.objects.link(proxy)
            proxy['ppf_bone_proxy_owned'] = True
            proxy['ppf_bone_uuid'] = s.bone_uuid
            proxy['ppf_armature_uuid'] = arm.ppf_armature_uuid
            if authoring_rows is not None and authoring_rows[bone.name].get('geometry') is not None:
                proxy[_RECREATE_SIZE_POLICY] = 'PRESERVE'
            if creation_plans[bone.name].get('_auto_geometry_defaults') and not (authoring_rows is not None and authoring_rows[bone.name].get('geometry') is not None):
                if _RECREATE_SIZE_POLICY in proxy:del proxy[_RECREATE_SIZE_POLICY]
            elif created and (old_intent=='PRESERVE' or not geometry_snapshot_matches(s)):
                # Reconstructing stored author dimensions does not turn them
                # into default dimensions merely because the new mesh is canonical.
                proxy[_RECREATE_SIZE_POLICY]='PRESERVE'
            if mesh is not None:
                proxy.parent = arm;proxy.matrix_world = frame
            proxy.display_type = 'WIRE';proxy.show_in_front = True;proxy.hide_render = True
            s.body_uuid = get_or_create_object_uuid(proxy)
            group = _group(context.scene, bone, creation_plans[bone.name],
                           initialize_material=bone.name not in new_solid_plans)
            if bone.name in new_solid_plans:
                group.solid_model = s.solid_model
                group.solid_poisson_ratio = s.solid_poisson_ratio
                if s.is_property_set('solid_deformation_damping'):
                    group.deformation_damping = s.solid_deformation_damping
                _apply_copied_soft_material(s, group, new_solid_plans[bone.name])
            if not any(a.uuid == s.body_uuid for a in group.assigned_objects):
                assigned = group.assigned_objects.add();assigned.uuid = s.body_uuid;assigned.name = proxy.name;assigned.included = True
                if creation_plans[bone.name].get('physical_model')=='SOFT' and s.shape!='CUSTOM':assigned.tet_backend=NEW_GENERATED_SOFT_TET_BACKEND
            for assigned in group.assigned_objects:
                if assigned.uuid == s.body_uuid:
                    assigned.included = True;assigned.name = proxy.name
            s.enabled = True
            if created:s.recreate_geometry_intent_json = ''
            from ..core.bone_proxy_visibility import manage_proxy,snapshot_proxy
            if created and new_visibility is not None:new_visibility.append((proxy,snapshot_proxy(proxy,context.scene)))
            manage_proxy(context.scene,arm,bone,proxy,
                         created_objects=new_resources['objects'] if new_resources is not None else None,
                         created_collections=new_resources['collections'] if new_resources is not None else None)
        apply_mass_batch(mass_plan, context.scene)
        invalidate_all()
        assign_display_indices(context.scene);invalidate_overlays();context.view_layer.update()
        from ..core.bone_author_follow import baseline_proxy
        for bone in bones:
            baseline_proxy(context.scene, arm, bone)
        return len(bones)
    finally:
        for _bone, mesh, _frame in prepared:
            if mesh is not None and mesh.users == 0:
                bpy.data.meshes.remove(mesh)


def isolate_copied_armature_bindings(scene):
    """Detach only proven borrowed bindings on an independent Object duplicate.

    The proxy's actual armature parent proves the source; names/selection do
    not. Never repair a linked armature datablock by mutating shared Bone RNA.
    A user can make linked data single-user before this same cleanup applies.
    Ordinary Blender duplicates become empty physics rigs. Use the dedicated
    rig-copy operator to retain physics authoring. No source binding, proxy,
    material group or user object is removed.
    """
    from ..ui.dynamics.group_ops import cleanup_identity_rebuilding
    from ..core.bone_author_follow import _transform_active
    if (scene is None or cleanup_identity_rebuilding() or _transform_active(scene)):
        return 0
    from .restoring_curve_operators import snapshot_rna, restore_rna
    plans = []
    for arm in scene.objects:
        if (arm.type != 'ARMATURE' or arm.mode == 'EDIT' or arm.library
                or arm.override_library or arm.data.library or arm.data.users != 1
                or len(arm.users_scene) != 1 or not arm.ppf_armature_uuid):
            continue
        rows = [(bone, bone.ppf_bone_physics) for bone in arm.data.bones]
        references = [(bone, settings, settings.proxy) for bone, settings in rows
                      if settings.proxy is not None]
        if not references:
            continue
        from ..models.bone_bindings import proxy_parent_armature
        source = proxy_parent_armature(references[0][2])
        if (source is None or source == arm or source.type != 'ARMATURE'
                or scene.objects.get(source.name) != source or source.data == arm.data
                or source.ppf_armature_uuid != arm.ppf_armature_uuid
                or source.library or source.data.library or source.data.users != 1
                or len(source.users_scene) != 1):
            continue
        originals = {}
        ambiguous = False
        for bone in source.data.bones:
            uid = bone.ppf_bone_physics.bone_uuid
            if uid:
                if uid in originals: ambiguous = True; break
                originals[uid] = bone
        if ambiguous:
            continue
        for bone, settings in rows:
            original = originals.get(settings.bone_uuid)
            if original is None:
                if settings.proxy is not None or any(getattr(settings, key) for key in
                        ('bone_uuid', 'body_uuid', 'joint_uuid', 'group_uuid')):
                    ambiguous = True; break
                continue
            owner = original.ppf_bone_physics
            if (any(getattr(settings, key) != getattr(owner, key) for key in
                    ('proxy', 'body_uuid', 'joint_uuid', 'group_uuid'))
                    or settings.proxy is not None and proxy_parent_armature(settings.proxy) != source):
                ambiguous = True; break
            try:
                validate_proxy_owner(source, original)
            except (ValueError, RuntimeError, ReferenceError):
                ambiguous = True; break
        if not ambiguous:
            plans.append((arm, [(settings, snapshot_rna(settings))
                                for _bone, settings in rows],
                          [snapshot_rna(link) for link in arm.ppf_cross_links],
                          snapshot_rna(arm.ppf_cross_link_draft)))
    def clear_physics(owner, snapshot):
        for key, (_was_set, (kind, value)) in snapshot.items():
            if kind == 'COLLECTION': getattr(owner, key).clear()
            elif kind == 'GROUP': clear_physics(getattr(owner, key), value)
            else: owner.property_unset(key)
    changed = 0
    for arm, rows, links, draft in plans:
        old_uuid = arm.ppf_armature_uuid
        try:
            arm.ppf_armature_uuid = str(uuid.uuid4())
            arm.ppf_cross_links.clear()
            clear_physics(arm.ppf_cross_link_draft, draft)
            for settings, snapshot in rows:
                # Drop the borrowed pointer before callbacks can inspect a group.
                settings.proxy = None
                clear_physics(settings, snapshot)
        except Exception:
            arm.ppf_armature_uuid = old_uuid
            for settings, snapshot in rows:
                restore_rna(settings, snapshot)
            arm.ppf_cross_links.clear()
            for snapshot in links:
                restore_rna(arm.ppf_cross_links.add(), snapshot)
            restore_rna(arm.ppf_cross_link_draft, draft)
            raise
        changed += 1
    if changed:
        invalidate_overlays()
    return changed


def reconcile_deleted_proxies(scene, *, strict=True):
    """Retire only proven deleted owners; never infer deletion from validation failure.

    Called synchronously before Transfer and by the existing deferred cleanup.
    Unknown/shared/duplicated identities are left intact for the normal error
    path. Missing proxy bindings keep authored parameters and persistent IDs.
    """
    from ..models.groups import iterate_object_groups
    from ..ui.dynamics.utils import cleanup_group_references_for_object
    from ..ui.dynamics.group_ops import cleanup_identity_rebuilding
    if cleanup_identity_rebuilding():
        if strict:
            raise RuntimeError('Wait for Undo/Redo identity reconstruction before Transfer')
        return 0
    if scene is None or not has_addon_data(scene):
        return 0
    arms = [obj for obj in bpy.data.objects if obj.type == 'ARMATURE']
    objects = list(bpy.data.objects)
    missing = []
    orphans = []
    ambiguous = []
    for arm in scene.objects:
        if arm.type != 'ARMATURE':
            continue
        arm_data = arm.data
        if arm_data is None or arm.library or arm_data.library:
            continue
        if len(arm.users_scene) != 1 or arm_data.users != 1:
            continue
        for bone in arm.data.bones:
            settings = bone.ppf_bone_physics
            if settings.proxy is not None or not settings.body_uuid:
                continue
            if any(obj.get('_solver_uuid') == settings.body_uuid for obj in objects):
                continue
            if settings.enabled or settings.group_uuid:
                missing.append(settings)
    for proxy in list(scene.objects):
        if proxy.type != 'MESH' or not proxy.get('ppf_bone_proxy_owned'):
            continue
        mesh = proxy.data
        if (mesh is None or proxy.library or mesh.library or not proxy.is_editable
                or len(proxy.users_scene) != 1 or mesh.users != 1):
            continue
        aid, bid, body = (proxy.get(key, '') for key in
                          ('ppf_armature_uuid', 'ppf_bone_uuid', '_solver_uuid'))
        try:
            if any(str(uuid.UUID(value)) != value for value in (aid, bid, body)):
                continue
        except (ValueError, TypeError, AttributeError):
            continue
        possible = [arm for arm in arms if arm.ppf_armature_uuid == aid]
        references = [(arm, bone) for arm in arms for bone in arm.data.bones
                      if bone.ppf_bone_physics.proxy == proxy
                      or bone.ppf_bone_physics.body_uuid == body]
        if references:
            continue  # Includes disabled owners and owners in another scene.
        if len(possible) > 1 or sum(obj.get('_solver_uuid') == body for obj in objects) != 1:
            ambiguous.append(proxy.name)
            continue
        if possible and any(bone.ppf_bone_physics.bone_uuid == bid for bone in possible[0].data.bones):
            continue
        # Automatic deletion may only unlink the plugin's own collections.
        if any(collection.get('ppf_bone_proxy_collection_owner') != aid
               for collection in proxy.users_collection):
            continue
        orphans.append(proxy)
    if ambiguous:
        if strict:
            raise ValueError('骨骼代理归属不唯一，未自动清理：' + ', '.join(ambiguous))
        return 0
    if not missing and not orphans:
        return 0
    retired = {settings.body_uuid for settings in missing}
    retired.update(proxy.get('_solver_uuid') for proxy in orphans)
    # All ownership decisions finish before any assignment or object is changed.
    _facade_communicator().bone_physical_body_changed()
    from .restoring_curve_operators import snapshot_rna, restore_rna
    root = get_addon_data(scene)
    group_snapshot = snapshot_rna(root)
    binding_snapshot = [(settings, settings.enabled, settings.group_uuid) for settings in missing]
    unlinked = []
    try:
        empty_bone_groups = []
        for group in list(iterate_object_groups(scene)):
            removed = False
            for index in reversed(range(len(group.assigned_objects))):
                body = group.assigned_objects[index].uuid
                if body in retired:
                    group.assigned_objects.remove(index)
                    cleanup_group_references_for_object(group, body)
                    removed = True
            if removed and not group.assigned_objects:
                empty_bone_groups.append(group.uuid)
        # Removing a CollectionProperty item relocates later RNA item storage.
        # Finish reading/mutating every assignment before removing any group;
        # retain only immutable UUID strings across that second pass.
        for group_uuid in empty_bone_groups:
            remove_bone_material_group(scene, group_uuid)
        for settings in missing:
            settings.enabled = False
            settings.group_uuid = ''
        for proxy in orphans:
            for collection in list(proxy.users_collection):
                collection.objects.unlink(proxy)
                unlinked.append((collection, proxy))
    except Exception:
        for collection, proxy in reversed(unlinked):
            if collection.objects.get(proxy.name) != proxy:
                collection.objects.link(proxy)
        restore_rna(root, group_snapshot)
        for settings, enabled, group_uuid in binding_snapshot:
            settings.enabled = enabled
            settings.group_uuid = group_uuid
        raise
    # The transaction has retired scene membership. A failed datablock purge
    # can safely leave an unlinked zero-user datablock; it cannot keep solving.
    for proxy in orphans:
        mesh = proxy.data
        try:
            bpy.data.objects.remove(proxy)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        except RuntimeError:
            pass
    assign_display_indices(scene)
    invalidate_overlays()
    from ..core.bone_proxy_visibility import cleanup_collections
    cleanup_collections(scene)
    return len(missing) + len(orphans)


def _deletion_recreate_mode(context, arm, bone):
    """Explicit Delete policy, not a guess about legacy size provenance."""
    s=bone.ppf_bone_physics;obj=s.proxy
    if (not obj or s.shape=='CUSTOM' or obj.data.shape_keys is not None
            or any(mod.show_viewport for mod in obj.modifiers)
            or obj.get(_RECREATE_SIZE_POLICY)=='PRESERVE'):
        return 'PRESERVE'
    if s.auto_geometry_snapshot_json and not geometry_snapshot_matches(s):
        return 'PRESERVE'
    if s.shape=='CAPSULE':
        group=get_group_by_uuid(context.scene,s.group_uuid)
        return ('DEFAULTS' if group and (
            _existing_generated_capsule(context,arm,bone,group)
            or _existing_generated_capsule(context,arm,bone,group,mesh_dimensions=True)) else 'PRESERVE')
    # Sphere and Box have no pre-creation size provenance either. Only an
    # unchanged generated template opts into the explicit default-rebuild action.
    length=bone_world_length(arm,bone)*s.length_ratio
    candidates=[]
    for mode in dict.fromkeys((effective_proxy_face_mode(s),'TRI','QUAD')):
        candidates.append(builtin_geometry(length,s,{'proxy_face_mode':mode}))
        if s.shape=='BOX':candidates.append(box_geometry(length,s.box_width,s.box_depth,mode,'LOW'))
    actual_faces={tuple(sorted(p.vertices)) for p in obj.data.polygons}
    scale=max(1.,length,s.radius,s.box_width,s.box_depth)
    for vertices,faces in candidates:
        if len(obj.data.vertices)!=len(vertices) or len(obj.data.polygons)!=len(faces):continue
        if any((obj.data.vertices[i].co-Vector(v)).length>scale*2.e-7 for i,v in enumerate(vertices)):continue
        if actual_faces=={tuple(sorted(face)) for face in faces}:return 'DEFAULTS'
    return 'PRESERVE'


def delete_proxies(context, all_bones=False, scope='SELECTED'):
    from ..core.bone_geometry_edit import preflight
    preflight(context)
    arm, bones = selected_bones(context, all_bones, scope)
    for bone in bones:
        validate_proxy_owner(arm,bone)
        s = bone.ppf_bone_physics;proxy = s.proxy
        if proxy and not proxy.get("ppf_bone_proxy_owned", False):
            raise ValueError("Refusing to delete an unrelated mesh")
    deletion_intents={}
    for bone in bones:
        s=bone.ppf_bone_physics
        if s.proxy:
            mode=_deletion_recreate_mode(context,arm,bone)
            deletion_intents[bone.name]=deletion_geometry_intent(s,arm.ppf_armature_uuid,mode)
    deleted=sum(b.ppf_bone_physics.proxy is not None for b in bones)
    from .restoring_curve_operators import read_state,retire_joint_or_pair,write_state
    # Removing a parent proxy also retires incident child joint curves: that
    # physical joint no longer exists after the explicit Delete operation.
    removed={bone.name for bone in bones}
    incident=[bone.ppf_bone_physics.joint_uuid for bone in arm.data.bones
              if bone.name in removed or bone.parent and bone.parent.name in removed]
    removed_body_ids={bone.ppf_bone_physics.body_uuid for bone in bones
                      if bone.ppf_bone_physics.body_uuid}
    # Preserve existing merge-pair records and their invalid-endpoint warning,
    # but explicitly retire restoration attached to the removed body. Reusing
    # its persistent body UUID on rebuild must not resurrect that curve.
    incident_pair_ids=[pair.pair_id for pair in get_addon_data(context.scene).state.merge_pairs
                       if pair.pair_id and (pair.object_a_uuid in removed_body_ids
                                            or pair.object_b_uuid in removed_body_ids)]
    curve_state=retire_joint_or_pair(read_state(context.scene),joint_ids=incident,
                                   pair_ids=incident_pair_ids)
    for bone in bones:
        s = bone.ppf_bone_physics;proxy = s.proxy
        if proxy is None:
            if s.enabled:s.enabled=False
            if s.group_uuid:s.group_uuid=''
            continue
        for group in list(iterate_active_object_groups(context.scene)):
            for index in reversed(range(len(group.assigned_objects))):
                if proxy is not None and s.body_uuid and group.assigned_objects[index].uuid == s.body_uuid:
                    group.assigned_objects.remove(index)
            if group.uuid == s.group_uuid and not group.assigned_objects:
                if not remove_bone_material_group(context.scene, group.uuid):
                    group.reset_to_defaults()
        if proxy:
            mesh = proxy.data;bpy.data.objects.remove(proxy, do_unlink=True)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        s.proxy = None;s.group_uuid = '';s.enabled = False
        s.recreate_geometry_intent_json=deletion_intents[bone.name]
        # Bone/body/joint identity intentionally survives deletion/recreation,
        # including the physical body choice, so a rebuilt proxy returns to the
        # same material without re-authoring.
    assign_display_indices(context.scene);invalidate_overlays()
    write_state(context.scene,curve_state)
    from ..core.bone_proxy_visibility import cleanup_collections
    cleanup_collections(context.scene)
    if deleted:_facade_communicator().bone_physical_body_changed()
    return deleted


class PPF_OT_BoneProxyUpdate(bpy.types.Operator):
    bl_idname = "ppf.bone_proxy_update"
    bl_label = "添加或更新骨骼代理"
    bl_description = '添加或更新保留手工代理网格、变换和本地轴；按参数重建会替换手工网格和变换。代理编辑快捷键C添加、S缩放、E编辑、Esc退出'
    bl_options = {"REGISTER", "UNDO"}
    all_bones: BoolProperty(name="All Bones", default=False)
    scope: EnumProperty(name="Bones", items=SCOPES, default='SELECTED')
    rebuild: BoolProperty(name="Rebuild Shape", description="Replace existing proxy geometry and transform from shape settings; discards proxy edits", default=False)

    @classmethod
    def poll(cls, context):
        return active_armature(context) is not None and context.mode in {"OBJECT", "POSE", "EDIT_ARMATURE"}

    def execute(self, context):
        original=context.mode
        try:
            enter_proxy_pose(context,'ALL' if self.all_bones else self.scope)
            count = update_proxies(context, self.all_bones, self.scope, self.rebuild)
        except (ValueError, RuntimeError) as error:
            if context.mode!=original:
                bpy.ops.object.mode_set(mode='EDIT' if original=='EDIT_ARMATURE' else original)
            self.report({'ERROR'}, iface(str(error)));return {'CANCELLED'}
        self.report({'INFO'}, iface("Updated {count} real bone proxies").format(count=count))
        return {'FINISHED'}


class PPF_OT_BoneProxyDelete(bpy.types.Operator):
    bl_idname = "ppf.bone_proxy_delete"
    bl_label = "删除骨骼代理"
    bl_description = '删除指定范围内的物理代理并停用参与；可撤销'
    bl_options = {"REGISTER", "UNDO"}
    all_bones: BoolProperty(name="All Bones", default=False)
    scope: EnumProperty(name="Bones", items=SCOPES, default='SELECTED')

    @classmethod
    def poll(cls, context):
        return PPF_OT_BoneProxyUpdate.poll(context)

    def execute(self, context):
        try:
            count = delete_proxies(context, self.all_bones, self.scope)
        except (ValueError, RuntimeError) as error:
            self.report({'ERROR'}, iface(str(error)));return {'CANCELLED'}
        self.report({'INFO'}, iface("Deleted {count} bone proxies; persistent IDs retained").format(count=count))
        return {'FINISHED'}


class PPF_OT_BoneProxySelect(bpy.types.Operator):
    bl_idname = 'ppf.bone_proxy_select'
    bl_label = '编辑代理网格'
    bl_description = '进入活动骨骼的代理网格编辑；代理编辑模式中按 Tab/E 返回，网格须保持闭合'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        arm = active_armature(context)
        return bool(arm and arm.data.bones.active and arm.data.bones.active.ppf_bone_physics.proxy)

    def execute(self, context):
        arm = active_armature(context)
        bone=arm.data.bones.active;name=bone.name
        proxy = bone.ppf_bone_physics.proxy
        names=[p.name for p in arm.pose.bones if pose_bone_selected(p)]
        from ..core.bone_proxy_visibility import begin_mesh_edit,manage_proxy,snapshot_proxy,restore_proxy
        from ..core.bone_geometry_edit import acquire,finish
        visibility=snapshot_proxy(proxy,context.scene)
        lease=None
        original=context.mode
        orientation=context.scene.transform_orientation_slots[0].type
        try:
            validate_proxy_owner(arm,bone)
            lease=acquire(context,[proxy],kind='MESH')
            manage_proxy(context.scene,arm,bone,proxy)
            begin_mesh_edit(context.scene,arm,bone,proxy,context.view_layer)
            if proxy.name not in context.view_layer.objects:
                raise ValueError('代理不在当前视图层；请先恢复可见集合')
            if context.mode != 'OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
            bpy.ops.object.select_all(action='DESELECT')
            proxy.select_set(True);context.view_layer.objects.active = proxy
            proxy['ppf_proxy_edit_selected_bones']=names or [name]
            proxy['ppf_proxy_edit_orientation']=orientation
            context.scene.transform_orientation_slots[0].type = 'LOCAL'
            bpy.ops.object.mode_set(mode='EDIT');bpy.ops.mesh.select_all(action='SELECT')
        except Exception as error:
            if lease is not None:
                finish(lease,commit=False)
                restore_proxy(proxy,visibility)
                context.scene.transform_orientation_slots[0].type=orientation
                proxy.pop('ppf_proxy_edit_orientation',None)
                proxy.pop('ppf_proxy_edit_selected_bones',None)
                if context.object!=arm or context.mode!=original:
                    if context.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
                    bpy.ops.object.select_all(action='DESELECT')
                    arm.select_set(True);context.view_layer.objects.active=arm
                    if original!='OBJECT':bpy.ops.object.mode_set(mode=original)
                    for pb in arm.pose.bones:set_pose_bone_selected(pb,pb.name in names)
            if isinstance(error,(ValueError,RuntimeError)):
                self.report({'ERROR'},str(error));return {'CANCELLED'}
            raise
        return {'FINISHED'}


class PPF_OT_BoneProxyReturn(bpy.types.Operator):
    bl_idname = 'ppf.bone_proxy_return'
    bl_label = '返回绑定骨骼'
    bl_description = '检查网格闭合并返回绑定骨骼；网格无效时保留编辑状态'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bool(context.object and context.object.get('ppf_bone_proxy_owned') and context.mode in {'OBJECT', 'EDIT_MESH'})

    def execute(self, context):
        proxy = context.object
        target = next(((arm, bone) for arm in context.scene.objects if arm.type == 'ARMATURE' for bone in arm.data.bones if bone.ppf_bone_physics.proxy == proxy), None)
        if not target:
            self.report({'ERROR'}, iface('Proxy has no bone binding in this scene'));return {'CANCELLED'}
        arm, bone = target
        name=bone.name
        names=list(proxy.get('ppf_proxy_edit_selected_bones',[name]))
        try:validate_proxy_owner(arm,bone)
        except ValueError as error:self.report({'ERROR'},str(error));return {'CANCELLED'}
        if arm.name not in context.view_layer.objects:
            self.report({'ERROR'},'骨架不在当前视图层；请先恢复可见集合');return {'CANCELLED'}
        if context.mode != 'OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
        checked=proxy.data.copy()
        try:
            try:_validate_bone_proxy_mesh(checked, arm, bone)
            except ValueError as error:
                bpy.ops.object.mode_set(mode='EDIT');self.report({'ERROR'},str(error));return {'CANCELLED'}
        finally:
            bpy.data.meshes.remove(checked)
        from ..core.bone_geometry_edit import token_for,finish
        lease=token_for(proxy,kind='MESH')
        try:
            bpy.ops.object.select_all(action='DESELECT')
            arm.hide_set(False);arm.select_set(True);context.view_layer.objects.active = arm
            arm.data.bones.active = arm.data.bones.get(name)
            bpy.ops.object.mode_set(mode='POSE')
            for pb in arm.pose.bones:set_pose_bone_selected(pb,pb.name in names)
            from ..core.bone_proxy_visibility import end_mesh_edit
            end_mesh_edit(context.scene,arm,arm.data.bones[name],proxy,context.view_layer)
            if lease is not None:finish(lease,commit=True)
        except Exception as error:
            # A failed commit retains its lease; keep the authored edit available
            # for repair/retry rather than exposing it to solver operations.
            if lease is not None:
                from ..core.bone_proxy_visibility import begin_mesh_edit
                if context.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
                begin_mesh_edit(context.scene,arm,bone,proxy,context.view_layer)
                bpy.ops.object.select_all(action='DESELECT')
                proxy.select_set(True);context.view_layer.objects.active=proxy
                bpy.ops.object.mode_set(mode='EDIT')
            if isinstance(error,(ValueError,RuntimeError)):
                self.report({'ERROR'},str(error));return {'CANCELLED'}
            raise
        if 'ppf_proxy_edit_orientation' in proxy:
            context.scene.transform_orientation_slots[0].type=proxy.pop('ppf_proxy_edit_orientation')
        proxy.pop('ppf_proxy_edit_selected_bones',None)
        return {'FINISHED'}


class PPF_OT_BoneCustomFromSelection(bpy.types.Operator):
    bl_idname = 'ppf.bone_custom_from_selection'
    bl_label = 'Use Selected Mesh'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        arm = active_armature(context)
        return bool(arm and arm.data.bones.active and any(o.type == 'MESH' and not o.get('ppf_bone_proxy_owned') for o in context.selected_objects))

    def execute(self, context):
        meshes = [o for o in context.selected_objects if o.type == 'MESH' and not o.get('ppf_bone_proxy_owned')]
        if len(meshes) != 1:
            self.report({'ERROR'}, iface('Select exactly one source mesh with the armature active'))
            return {'CANCELLED'}
        s = active_armature(context).data.bones.active.ppf_bone_physics
        s.custom_object = meshes[0];s.shape = 'CUSTOM'
        return {'FINISHED'}


def apply_custom_shape(context, scope='SELECTED', rebuild=False, source_name=''):
    """Use the ordinary proxy transaction; only explicit rebuild replaces a mesh."""
    from ..core.bone_geometry_edit import preflight
    preflight(context)
    arm,bones=selected_bones(context,scope=scope)
    active=arm.data.bones.active
    source=(context.scene.objects.get(source_name) if source_name else
            getattr(context.window_manager,'ppf_bone_editor_custom_source',None) or active.ppf_bone_physics.custom_object)
    if not source or source.type!='MESH' or source.get('ppf_bone_proxy_owned'):
        raise ValueError('请选择一个非物理代理的闭合自定义源网格')
    from .restoring_curve_operators import snapshot_rna,restore_rna
    from ..core.bone_mass import material_write_guard
    snapshots=[(b.ppf_bone_physics,snapshot_rna(b.ppf_bone_physics)) for b in bones]
    try:
        for bone in bones:
            settings=bone.ppf_bone_physics
            settings.shape='CUSTOM';settings.custom_object=source
        return update_proxies(context,scope=scope,rebuild=rebuild)
    except Exception:
        with material_write_guard():
            for owner,snapshot in snapshots:restore_rna(owner,snapshot)
        raise


class PPF_OT_BoneCustomApply(bpy.types.Operator):
    bl_idname='ppf.bone_custom_apply'
    bl_label='应用自定义源网格'
    bl_description='为范围内骨骼设置自定义源网格并创建缺少的代理；已有手调网格仅在明确重建时替换'
    bl_options={'REGISTER','UNDO'}
    scope:EnumProperty(name='Bones',items=SCOPES,default='SELECTED')
    rebuild:BoolProperty(name='重建为自定义网格',default=False)
    source_name:StringProperty(name='自定义源网格名称',default='')

    @classmethod
    def poll(cls,context):
        arm=active_armature(context)
        return bool(arm and arm.data.bones.active and context.mode in {'OBJECT','POSE'})

    def invoke(self,context,event):
        if self.rebuild:
            return context.window_manager.invoke_confirm(self,event,
                message='重建会替换范围内已有手调代理网格，是否继续？')
        return self.execute(context)

    def execute(self,context):
        original=context.mode
        try:
            enter_proxy_pose(context,self.scope)
            count=apply_custom_shape(context,self.scope,self.rebuild,self.source_name)
        except (ValueError,RuntimeError) as error:
            if context.mode!=original:
                bpy.ops.object.mode_set(mode='EDIT' if original=='EDIT_ARMATURE' else original)
            self.report({'ERROR'},str(error));return {'CANCELLED'}
        self.report({'INFO'},f'已应用自定义源网格到 {count} 根骨骼')
        return {'FINISHED'}


def _owned_bone_material(scene, arm, bone, *, groups=None, **owner_counts):
    try:validate_proxy_owner(arm,bone,**owner_counts)
    except ValueError:return None
    s = bone.ppf_bone_physics
    group = None
    if s.group_uuid:
        group = get_group_by_uuid(scene, s.group_uuid) if groups is None else groups.get(s.group_uuid)
    proxy = s.proxy
    if (group and group.active and group.object_type in ('PDRD', 'SOLID') and proxy
            and proxy.get('ppf_bone_proxy_owned')
            and proxy.get('_solver_uuid') == s.body_uuid
            and proxy.get('ppf_bone_uuid') == s.bone_uuid
            and proxy.get('ppf_armature_uuid') == arm.ppf_armature_uuid
            and len(group.assigned_objects) == 1
            and group.assigned_objects[0].uuid == s.body_uuid):
        return group
    return None


def _copy_bone_author_rna(source, target):
    for prop in source.bl_rna.properties:
        key=prop.identifier
        if key=='rna_type':continue
        value=getattr(source,key)
        if prop.type=='COLLECTION':
            collection=getattr(target,key);collection.clear()
            for item in value:_copy_bone_author_rna(item,collection.add())
        elif prop.type=='POINTER' and isinstance(value,bpy.types.PropertyGroup):
            _copy_bone_author_rna(value,getattr(target,key))
        elif not prop.is_readonly:
            setattr(target,key,tuple(value) if getattr(prop,'is_array',False) else value)


BONE_COPY_CATEGORIES = [
    ('PARAMETERS', '物理参数', '复制材质、质量、恢复和软限位，保留形状与控制轨迹'),
    ('SHAPE', '形状参数', '只复制创建形状设置，保留已有手调网格'),
    ('COLLISION', '碰撞设置', '只复制碰撞层、遮罩和相邻碰撞'),
    ('ALL', '完整设置', '兼容原有完整参数复制'),
]
SHAPE_COPY_FIELDS = {'shape', 'custom_object', 'length_ratio', 'radius',
                     'box_width', 'box_depth', 'proxy_quality', 'new_proxy_quality', 'proxy_face_mode'}
COLLISION_COPY_FIELDS = {'collision_filter_enabled', 'collision_group_layers',
                         'collision_mask_layers', 'collide_connected'}
PARAMETER_COPY_FIELDS = {'density', 'mass_mode', 'target_mass_kg', 'material_preset',
    'solid_model', 'solid_young_modulus', 'solid_poisson_ratio', 'solid_deformation_damping',
    'anchor_stiffness', 'angular_stiffness', 'angular_damping', 'soft_limits_enabled',
    'soft_lower', 'soft_upper', 'soft_limit_policy', 'soft_limit_strength',
    'soft_limit_frequency_hz', 'soft_stiffness', 'soft_limit_mode', 'swing_cone_half_angle', 'swing_cone_stiffness'}


def _copy_bone_fields(context, active, targets, fields):
    # Shape metadata does not rebuild geometry and needs no material/mass/curve gate.
    from .restoring_curve_operators import snapshot_rna, restore_rna
    from ..models.bone_collision import collision_edit_batch
    from ..core.bone_mass import material_write_guard
    source=active.ppf_bone_physics
    values=[(p.identifier, tuple(getattr(source,p.identifier)) if getattr(p,'is_array',False) else getattr(source,p.identifier))
            for p in source.bl_rna.properties if p.identifier in fields]
    snapshots=[(b.ppf_bone_physics,snapshot_rna(b.ppf_bone_physics)) for b,_group in targets]
    with collision_edit_batch():
        try:
            for bone,_group in targets:
                for key,value in values:setattr(bone.ppf_bone_physics,key,value)
        except Exception:
            with material_write_guard():
                for owner,snapshot in snapshots:restore_rna(owner,snapshot)
            raise
    invalidate_overlays()
    return len(targets),0,0,0,0


def _plan_copied_soft_material(settings, group, material, density, normalized):
    """Read-only exact material plan; choose a representable unit convention."""
    from ..core.bone_preset_apply import _validate_rna
    keys=('solid_model','solid_young_modulus','solid_poisson_ratio','solid_deformation_damping')
    expected={key:material[key] for key in keys}
    for key,value in expected.items():_validate_rna(settings,key,value)
    _validate_rna(settings,'density',density)
    _validate_rna(group,'solid_density',density)
    if not math.isfinite(density) or density<=0:raise ValueError('复制材质要求有限正密度')
    young=expected['solid_young_modulus']
    stored=None
    for convention in (bool(normalized),not bool(normalized)):
        value=young/density if convention else young
        try:_validate_rna(group,'solid_young_modulus',value)
        except ValueError:continue
        normalized=convention;stored=value;break
    if stored is None:
        raise ValueError('目标密度下的真实杨氏模量无法由材质字段无损表示；整批未复制')
    writes={'solid_model':expected['solid_model'],'solid_density':density,
            'solid_young_modulus':stored,'young_mod_density_normalized':normalized,
            'solid_poisson_ratio':expected['solid_poisson_ratio'],
            'deformation_damping':expected['solid_deformation_damping']}
    for key,value in writes.items():_validate_rna(group,key,value)
    return writes,expected


def _apply_copied_soft_material(settings, group, plan):
    """Commit a preflighted plan and reject unexpected RNA clamping."""
    from ..models.bone_bindings import effective_solid_material
    writes,expected=plan
    for key,value in writes.items():setattr(group,key,value)
    def equal(actual,value):
        return math.isclose(actual,value,rel_tol=2e-6,abs_tol=1e-9) if type(value) in (int,float) else actual==value
    if any(not equal(getattr(group,key),value) for key,value in writes.items()):
        raise ValueError('复制材质值被限制；整批已撤回')
    actual=effective_solid_material(settings,group)
    if (not equal(settings.density,writes['solid_density'])
            or any(not equal(actual[key],value) or not equal(getattr(settings,key),value) for key,value in expected.items())):
        raise ValueError('复制后的作者与真实材质不一致；整批已撤回')


def copy_bone_settings(context, scope='SELECTED', category='ALL'):
    if category not in {'ALL','PARAMETERS','SHAPE','COLLISION'}:
        raise ValueError('未知参数复制类别')
    arm, bones = selected_bones(context, scope=scope)
    active = arm.data.bones.active
    source = active.ppf_bone_physics
    if category == 'ALL' and source.joint_mode != 'SOFT':
        raise ValueError('硬关节不再支持，不能通过完整复制创建；请先显式选择柔性关节')
    source_plan = new_binding_plan(source)
    source_model = source_plan.get('physical_model',source.physical_model)
    source_group = _owned_bone_material(context.scene, arm, active)
    targets = [(bone, _owned_bone_material(context.scene, arm, bone)) for bone in bones if bone != active]
    if category in {'SHAPE','COLLISION'}:
        fields=SHAPE_COPY_FIELDS if category=='SHAPE' else COLLISION_COPY_FIELDS
        return _copy_bone_fields(context,active,targets,fields)
    # Existing physical targets follow the same pre-write gate as material
    # presets. Pure future-author and source-only copies retain their old path.
    if any(group is not None or (bone.ppf_bone_physics.enabled and bone.ppf_bone_physics.proxy)
           for bone,group in targets):
        _facade_communicator().assert_bone_body_switch_idle()
        if stale_simulation_output(context):
            raise ValueError('请先 Clear 已有模拟帧，再复制已有物理骨骼参数')
    from .restoring_curve_operators import prepare_bone_copy,write_state,snapshot_rna,restore_rna
    curve_candidate=prepare_bone_copy(context,active,[bone for bone,_group in targets])
    # A cross-material batch is refused BEFORE any write. The physical body is a
    # property of the bone's own material group, so copying it onto a bone whose
    # group is the other type would leave settings and group disagreeing -- SOFT
    # settings over a PDRD group -- and the next export or update would refuse
    # the scene. Reporting a skip and writing the mismatch anyway would be worse
    # than refusing, because the operator would still claim success.
    for bone, _group in targets:
        target_model=new_binding_plan(bone.ppf_bone_physics).get('physical_model',bone.ppf_bone_physics.physical_model)
        if target_model != source_model:
            raise ValueError(iface(
                "{name}: cannot copy settings from a {source} body to a {target} body; "
                "material settings belong to the bone's own material group"
            ).format(name=bone.name, source=iface(PHYSICAL_LABELS[source_model]),
                     target=iface(PHYSICAL_LABELS[target_model])))
    for bone,group in [(active,source_group),*targets]:
        model=new_binding_plan(bone.ppf_bone_physics).get('physical_model',bone.ppf_bone_physics.physical_model)
        expected='SOLID' if model=='SOFT' else 'PDRD'
        if group is not None and group.object_type!=expected:
            raise ValueError(f'{bone.name}：请先应用已选物理体，再复制材质参数')
    from ..core.utils import get_id_fcurves
    from ..core.encoder.curve_refusal import _nla_curves
    paths = [curve.data_path for curve in get_id_fcurves(context.scene)]
    animation = context.scene.animation_data
    if animation:
        paths.extend(curve.data_path for curve in animation.drivers)
        paths.extend(curve.data_path for _track, _strip, curve in _nla_curves(context.scene))
    # Validate all actual material writes before changing any bone or group.
    # Animation paths belong to their author; retain them and refuse the batch.
    for group in [source_group, *(group for _bone, group in targets)]:
        if group is not None:
            prefix = group.path_from_id() + '.'
            if any(path.startswith(prefix) for path in paths):
                raise ValueError(iface('Cannot copy bone materials with keyframes or drivers: {name}').format(name=group.name))
    # The material-owned properties are copied through ``sync_owned_material``
    # rather than as ordinary fields: their write must land on the target's own
    # group, and a target that owns no compatible group must stay unchanged and
    # be reported instead of being written blind.
    excluded = {'rna_type', 'soft_stiffness_degree', 'swing_cone_stiffness_degree', 'new_binding_physical_model', 'enabled', 'proxy', 'bone_uuid', 'body_uuid', 'joint_uuid', 'group_uuid', 'parent_frame_translation', 'parent_frame_quaternion', 'child_frame_translation', 'child_frame_quaternion'}
    nested = {'root_control_events', 'drive_twist_x', 'drive_swing_y', 'drive_swing_z', *(f'd6_{kind}_{axis}' for kind in ('translation','rotation') for axis in 'xyz')}
    excluded.update(nested)
    excluded.update(MATERIAL_OWNED_PROPERTIES)
    # Measurements belong to each body's accepted frame, not its author settings.
    excluded.update({'fem_scale_last','fem_scale_shear','fem_scale_variation',
                     'fem_scale_orientation_error','fem_scale_status',
                     'mass_volume_m3','mass_measured_kg','mass_valid','mass_status'})
    from ..core.bone_mass import prepare_mass_batch, material_write_guard, _measure, mark_settings
    from ..core.bone_mass_math import density_for_mass
    prepare_mass_batch(context.scene)  # Full read-only MASS batch before any copy mutation.
    target_density = {}
    if source.mass_mode == 'MASS':
        for bone, group in targets:
            if group is None: raise ValueError(f'{bone.name}：固定质量复制要求独立的代理和材质')
            target_density[bone.name] = density_for_mass(source.target_mass_kg, _measure(context.scene, bone.ppf_bone_physics)['volume_m3'])
    def effective_source_value(prop):
        group_fields = {'density':'solid_density' if source_model == 'SOFT' else 'pdrd_density',
                        'solid_model':'solid_model','solid_young_modulus':'solid_young_modulus',
                        'solid_poisson_ratio':'solid_poisson_ratio','solid_deformation_damping':'deformation_damping'}
        if source_group is not None and source_model == 'SOFT' and prop.identifier in {'solid_model','solid_young_modulus','solid_poisson_ratio','solid_deformation_damping'}:
            from ..models.bone_bindings import effective_solid_material
            value = effective_solid_material(source, source_group)[prop.identifier]
        elif source_group is not None and prop.identifier in group_fields:
            value = getattr(source_group, group_fields[prop.identifier])
        else: value=source_plan.get(prop.identifier,getattr(source,prop.identifier))
        return tuple(value) if getattr(prop,'is_array',False) else value
    values = [(p.identifier,effective_source_value(p)) for p in source.bl_rna.properties if p.identifier not in excluded and not p.is_readonly]
    if category=='PARAMETERS':
        values=[(key,value) for key,value in values if key in PARAMETER_COPY_FIELDS]
    material_source = [(p.identifier,effective_source_value(p)) for p in source.bl_rna.properties if p.identifier in MATERIAL_OWNED_PROPERTIES and not p.is_readonly and (category=='ALL' or p.identifier in PARAMETER_COPY_FIELDS)]
    # Plan the whole batch before writing. A normalized source may represent
    # a true modulus above the non-normalized group's hard RNA maximum.
    # MASS targets need E divided by their own derived density, not source rho.
    solid_plans={};source_material=dict(material_source);source_values=dict(values)
    preferred_normalized=bool(source_group and source_group.object_type=='SOLID' and source_group.young_mod_density_normalized)
    for bone,group in targets:
        if group is not None and group.object_type=='SOLID':
            density=target_density.get(bone.name,source_values.get('density',bone.ppf_bone_physics.density))
            try:solid_plans[bone.name]=_plan_copied_soft_material(bone.ppf_bone_physics,group,source_material,density,preferred_normalized)
            except ValueError as error:raise ValueError(bone.name+'：'+str(error)) from error
    material_copied = 0
    density_skipped = 0
    material_parameter_skipped = 0
    curve_owner=context.scene.ppf_restoring_curves
    curve_snapshot=(curve_owner.document_json,curve_owner.mode,curve_owner.requested_mode)
    snapshots=[(bone.ppf_bone_physics,snapshot_rna(bone.ppf_bone_physics)) for bone,_group in targets]
    snapshots.extend((group,snapshot_rna(group)) for _,group in targets if group is not None)
    from ..models.bone_collision import collision_edit_batch
    # Defer collision retirement until writes AND any rollback have completed.
    with collision_edit_batch():
        try:
          with material_write_guard():
            for bone, group in targets:
                settings = bone.ppf_bone_physics
                referenced = get_group_by_uuid(context.scene, settings.group_uuid) if settings.group_uuid else None
                for key, value in values:
                    # Density's RNA callback writes the referenced group. An unrelated
                    # or mixed group must not be altered through a stale bone pointer.
                    if key == 'density' and referenced is not None and group is None:
                        density_skipped += 1
                        continue
                    setattr(settings, key, value)
                # The requested material copy: the physical body and the soft model/E/nu
                # move to the target, but only when the target OWNS a compatible group.
                # A shared, unrelated or wrong-type group is left exactly as it was and
                # counted, so the operator reports the skip instead of claiming success.
                if group is not None:
                    for key, value in material_source:
                        setattr(settings, key, value)
                    # Write both directions in one place: the callbacks mirror the fields
                    # onto the owned group, and this makes the group's material state
                    # explicit so a later `_group` rebuild reads the same values.
                    if bone.name in target_density: settings.density = target_density[bone.name]
                    if bone.name in solid_plans:_apply_copied_soft_material(settings,group,solid_plans[bone.name])
                    else:sync_owned_material(settings)
                elif referenced is not None:
                    material_parameter_skipped += 1
                if category=='ALL':
                    settings.root_control_events.clear()
                    for event in source.root_control_events:_copy_bone_author_rna(event, settings.root_control_events.add())
                    for key in nested - {'root_control_events'}:_copy_bone_author_rna(getattr(source,key), getattr(settings,key))
                if source_group is not None and group is not None:
                    group.friction = source_group.friction
                    if group.object_type==source_group.object_type=='SOLID':group.deformation_damping=source_group.deformation_damping
                    material_copied += 1
          if curve_candidate is not None:write_state(context.scene,curve_candidate)
        except Exception:
          with material_write_guard():
            for owner,snapshot in snapshots:restore_rna(owner,snapshot)
          curve_owner.document_json,curve_owner.mode,curve_owner.requested_mode=curve_snapshot
          raise
    for bone, _group in targets: mark_settings(bone.ppf_bone_physics, context)
    return len(targets), material_copied, len(targets) - material_copied, density_skipped, material_parameter_skipped


def _set_participation(settings,enabled):
    """Use the real RNA update callback, including owned material membership."""
    settings.enabled=enabled


def set_bone_participation(context,scope='SELECTED',enabled=True):
    """Only existing proxies; validate all owners before one undoable batch."""
    if type(enabled) is not bool:raise ValueError('参与状态必须为布尔值')
    if scope not in {'SELECTED','CHAIN','ALL'}:raise ValueError('未知骨骼范围')
    arm,bones=selected_bones(context,scope=scope)
    planned=[];skipped=[]
    for bone in bones:
        settings=bone.ppf_bone_physics
        if settings.proxy is None:
            skipped.append(bone.name);continue
        validate_proxy_owner(arm,bone)
        group=_owned_bone_material(context.scene,arm,bone)
        if group is None or group.object_type!=OBJECT_TYPES[settings.physical_model]:
            raise ValueError(bone.name+'：物理代理缺少匹配的独占材质，请先修复绑定')
        item=group.assigned_objects[0]
        if settings.enabled!=enabled or item.included!=enabled:
            planned.append((settings,item,bool(settings.enabled),bool(item.included),settings.is_property_set('enabled'),item.is_property_set('included')))
    if not planned:return 0,skipped
    com=_facade_communicator();com.assert_bone_body_switch_idle()
    reasons=stale_simulation_output(context)
    if reasons:raise ValueError('请先清除已有模拟帧再修改参与范围：'+', '.join(reasons))
    from ..models.bone_collision import collision_edit_batch
    with collision_edit_batch():
        try:
            for settings,item,*_before in planned:
                _set_participation(settings,enabled)
                if bool(settings.enabled)!=enabled or bool(item.included)!=enabled:
                    raise ValueError('参与状态未能同步到独占材质；批量修改已撤回')
            com.bone_physical_body_changed()
        except BaseException:
            # A callback may have failed after changing RNA. Restore BOTH sides,
            # before the existing batch invalidator observes the final contract.
            for settings,item,old_enabled,old_included,enabled_set,included_set in reversed(planned):
                settings.enabled=old_enabled
                if not enabled_set:settings.property_unset('enabled')
                item.included=old_included
                if not included_set:item.property_unset('included')
            raise
    invalidate_overlays()
    return len(planned),skipped


class PPF_OT_BoneParticipation(bpy.types.Operator):
    bl_idname='ppf.bone_participation'
    bl_label='批量参与骨骼物理'
    bl_description='仅启用或停用已有物理代理；不删除、重建或复制参数。无代理骨骼跳过；修改后需重新传输构建'
    bl_options={'REGISTER','UNDO'}
    scope:EnumProperty(items=SCOPES,default='SELECTED',options={'HIDDEN'})
    enabled:BoolProperty(default=True,options={'HIDDEN'})

    @classmethod
    def poll(cls,context):
        return has_addon_data(context.scene) and active_armature(context) is not None and context.mode in {'OBJECT','POSE'}

    def execute(self,context):
        try:changed,skipped=set_bone_participation(context,self.scope,self.enabled)
        except (ValueError,RuntimeError) as error:
            self.report({'ERROR'},str(error));return {'CANCELLED'}
        label='启用' if self.enabled else '停用'
        detail=f'{label} {changed} 个已有代理'
        if skipped:detail+=f'；跳过 {len(skipped)} 个无代理骨骼（'+ '、'.join(skipped[:5])+('…' if len(skipped)>5 else '')+'）'
        self.report({'WARNING'} if skipped else {'INFO'},detail)
        return {'FINISHED'}


class PPF_OT_BoneSettingsCopy(bpy.types.Operator):
    bl_idname = 'ppf.bone_settings_copy'
    bl_label = '批量复制骨骼设置'
    bl_description = '复制当前参数与恢复曲线，保留目标代理身份及手调本地轴；整批失败时恢复设置和原有效缓存'
    bl_options = {'REGISTER', 'UNDO'}
    scope: EnumProperty(name='Bones', items=SCOPES, default='SELECTED')
    category: EnumProperty(name='类别',items=BONE_COPY_CATEGORIES,default='ALL')

    @classmethod
    def poll(cls, context):
        arm = active_armature(context)
        return bool(arm and arm.data.bones.active and context.mode in {'OBJECT', 'POSE'})

    def execute(self, context):
        try:
            count, copied, skipped, density_skipped, material_parameter_skipped = copy_bone_settings(context, self.scope, self.category)
        except (ValueError, RuntimeError) as error:
            self.report({'ERROR'}, iface(str(error)));return {'CANCELLED'}
        if self.category in {'SHAPE','COLLISION'}:
            self.report({'INFO'},f'已复制{dict((key,label) for key,label,_description in BONE_COPY_CATEGORIES)[self.category]}到 {count} 根骨骼')
        else:
            self.report({'WARNING'} if skipped else {'INFO'}, iface('Copied settings to {count} bones; friction copied to {copied}, skipped for {skipped} without owned source or target materials').format(count=count, copied=copied, skipped=skipped))
        if density_skipped:
            self.report({'WARNING'}, iface('Density skipped for {count} bones to preserve unrelated or mixed material groups').format(count=density_skipped))
        if material_parameter_skipped:
            self.report({'WARNING'}, iface('Material model, Young modulus and Poisson ratio skipped for {count} bones: their group is not owned by the bone').format(count=material_parameter_skipped))
        return {'FINISHED'}


class PPF_OT_BoneControlEventEdit(bpy.types.Operator):
    bl_idname = 'ppf.bone_control_event_edit'
    bl_label = 'Edit Bone Control Events'
    bl_options = {'REGISTER', 'UNDO'}
    action: EnumProperty(items=[('ADD','Add',''),('REMOVE','Remove','')])
    index: IntProperty(default=-1)

    @classmethod
    def poll(cls, context):return PPF_OT_BoneSettingsCopy.poll(context)

    def execute(self, context):
        bone = active_armature(context).data.bones.active
        s = bone.ppf_bone_physics
        if self.action == 'REMOVE':
            if not 0 <= self.index < len(s.root_control_events):return {'CANCELLED'}
            s.root_control_events.remove(self.index)
            return {'FINISHED'}
        else:
            events_were_set = s.is_property_set('root_control_events')
            from ..core.encoder import resolve_start_frame
            start = resolve_start_frame(get_addon_data(context.scene).state)
            frame = max(start + 1., context.scene.frame_current, s.root_control_events[-1].frame + 1. if s.root_control_events else start + 1.)
            prior = [event for event in s.root_control_events if event.frame <= frame]
            previous_mode = max(enumerate(prior), key=lambda item: (item[1].frame, item[0]))[1].mode if prior else s.root_mode
            event = s.root_control_events.add();event.frame = frame
            event.mode = 'KINEMATIC' if previous_mode == 'FREE' else 'FREE'
        # Validate actual physical parent/material support before committing.
        # SOFT root events remain explicit author intent; the selected product
        # capability is checked during Transfer/Build, never silently dropped.
        try:
            check_soft_supported(s, bone.name, has_physical_joint_parent(bone))
        except ValueError as error:
            if self.action == 'ADD':
                s.root_control_events.remove(len(s.root_control_events) - 1)
                if not events_were_set:
                    s.property_unset('root_control_events')
            self.report({'ERROR'}, iface(str(error)))
            return {'CANCELLED'}
        return {'FINISHED'}


class PPF_OT_BoneDriveTargetEdit(bpy.types.Operator):
    bl_idname = 'ppf.bone_drive_target_edit'
    bl_label = 'Edit Angular Drive Targets'
    bl_options = {'REGISTER', 'UNDO'}
    action: EnumProperty(items=[('ADD','Add',''),('REMOVE','Remove','')])
    axis: EnumProperty(items=[(a,a,'') for a in ('twist_x','swing_y','swing_z')])
    channel: EnumProperty(items=[(a,a,'') for a in ('position','velocity','torque')])
    index: IntProperty(default=-1)

    @classmethod
    def poll(cls, context):return PPF_OT_BoneSettingsCopy.poll(context)

    def execute(self, context):
        s = active_armature(context).data.bones.active.ppf_bone_physics
        channel = getattr(getattr(s, 'drive_' + self.axis), self.channel)
        if self.action == 'REMOVE':
            if not 0 <= self.index < len(channel.targets):return {'CANCELLED'}
            channel.targets.remove(self.index)
        else:
            from ..core.encoder import resolve_start_frame
            start = resolve_start_frame(get_addon_data(context.scene).state)
            frame = start if not channel.targets else channel.targets[-1].frame + 1.
            key = channel.targets.add();key.frame = frame
        return {'FINISHED'}



# ---------------------------------------------------------------------------
# Explicit physical-body change (RIGID <-> SOFT) on an EXISTING owned proxy.
#
# The bone keeps ONE owned material group for its whole life: the group OBJECT is
# reused (same uuid, same assignment, same body_uuid), and only its object_type and
# the material fields of the target kind are rewritten. Nothing rebuilds the proxy,
# so edited geometry, the proxy transform, the authored local frames, the joint
# identity and the group's friction survive by construction.
# ---------------------------------------------------------------------------

BODY_ITEMS = [
    ('RIGID', 'Rigid Body', 'The original rigid PDRD body'),
    ('SOFT', 'Soft Body', 'A real SOLID body; the frontend tetrahedralizes it at Build'),
]

# Every user-facing string this feature adds, so the catalogue gate in the staged
# tests can prove each one is translated.
CHANGE_BODY_STRINGS = (
    "Change Physical Body",
    "Switch to Soft Body",
    "Switch to Rigid Body",
    "Keeps the proxy, its geometry, transforms, local frames and identities",
    "Switching needs a new Transfer, Build and run",
    "Physical body changed on {count} bone(s); Transfer, Build and run again",
    "{count} bone(s) already used this physical body",
    "Every selected bone is already a {wanted} body",
    "Unknown physical body: {name}",
    "{name}: no owned physics proxy to switch; create or update it first",
    "{name}: its declared physical body does not match its material group; press Add / Update first",
    "{name}: cannot switch to a soft body. {detail}",
    "{name}: cannot change a physical body whose material state is keyframed or driven",
    "Clear the previous solve first: its frames belong to the old physical body and can never be applied to the new one",
    "Declared physical body and material group disagree",
    "Register the original PPF addon before switching a physical body",
    "Select an armature in Object or Pose mode",
    "Finish or stop the current task and Clear its frames before changing a physical body",
    "The core facade hook {name} is missing; this addon build cannot retire the previous Transfer",
    "The core facade communicator is unavailable; a physical body cannot be changed",
    "{name}: the joint to its enabled child '{child}' is a hard {mode} joint; a soft body supports only a compliant joint",
    "{name}: the edge to its enabled child '{child}' excludes collision with its connected neighbor; a soft body cannot",
    "{label}: cannot read animation curves ({error})",
    "{label}: cannot read animation drivers ({error})",
    "{label}: cannot read NLA curves ({error})",
    "Physical body change refused: {error}",
)


def owned_bone_material(scene, arm, bone):
    """The group this bone owns outright, or None (public read-only view)."""
    return _owned_bone_material(scene, arm, bone)


def selected_bone_materials(scene, rows):
    """Check one draw's selected rows with a shared ownership inventory."""
    from collections import Counter
    from ..models.bone_bindings import armature_data_owner_counts
    from ..models.groups import iterate_object_groups

    groups = {}
    for group in iterate_object_groups(scene):
        groups.setdefault(group.uuid, group)
    data_counts = armature_data_owner_counts()
    scene_counts = Counter(obj.as_pointer() for candidate in bpy.data.scenes for obj in candidate.objects)
    owners = {}
    for arm, bone in rows:
        if bone.ppf_bone_physics.proxy is None:
            continue
        identity = arm.data.as_pointer()
        if identity not in owners:
            owners[identity] = Counter(proxy.as_pointer() for sibling in arm.data.bones
                                       if (proxy := sibling.ppf_bone_physics.proxy) is not None)
        group = _owned_bone_material(scene, arm, bone, groups=groups, data_owner_counts=data_counts,
                                     scene_user_counts=scene_counts, proxy_counts=owners[identity])
        if group is not None:
            yield group


def escaped_bone_path(bone):
    """The real RNA path of a Bone inside its armature datablock."""
    try:
        return bone.path_from_id()
    except Exception as exc:
        raise ValueError(iface("{label}: cannot read animation curves ({error})").format(label=bone.name, error=exc))


def _owner_animation_paths(block, label):
    """Active-slot curves, drivers and NLA curves of ONE ID owner.

    Unreadable metadata is REFUSED rather than read as "no animation": a guard
    that cannot see a curve must never authorize a material rewrite.
    """
    try:
        animation = block.animation_data
        if animation is None:
            return []
        action = animation.action
        curves = []
        if action is not None:
            slot = getattr(animation,'action_slot',None)
            # An action with layered channels must have an assigned slot. A
            # missing slot must not silently hide another material curve.
            if getattr(action,'layers',()) and slot is None:
                raise ValueError("layered action has no assigned action slot")
            for layer in getattr(action,'layers',()):
                for strip in layer.strips:
                    bag = strip.channelbag(slot) if slot is not None else None
                    if bag is not None:
                        curves.extend(bag.fcurves)
            if not curves:curves.extend(getattr(action, 'fcurves', ()) or ())
        paths = [curve.data_path for curve in curves]
    except Exception as exc:  # noqa: BLE001 - unreadable, not absent
        raise ValueError(iface("{label}: cannot read animation curves ({error})").format(label=iface(label), error=exc))
    try:
        paths.extend(curve.data_path for curve in animation.drivers)
    except Exception as exc:  # noqa: BLE001 - unreadable, not absent
        raise ValueError(iface("{label}: cannot read animation drivers ({error})").format(label=iface(label), error=exc))
    try:
        for track in animation.nla_tracks:
            for nla_strip in track.strips:
                action = nla_strip.action
                if action is None:
                    continue
                slot = getattr(nla_strip,'action_slot',None)
                if getattr(action,'layers',()) and slot is None:
                    raise ValueError("NLA layered action has no assigned action slot")
                for layer in getattr(action,'layers',()):
                    for strip in layer.strips:
                        bag = strip.channelbag(slot) if slot is not None else None
                        if bag is not None:
                            paths.extend(curve.data_path for curve in bag.fcurves)
                paths.extend(curve.data_path for curve in (getattr(action, 'fcurves', ()) or ()))
    except Exception as exc:  # noqa: BLE001 - unreadable, not absent
        raise ValueError(iface("{label}: cannot read NLA curves ({error})").format(label=iface(label), error=exc))
    return paths


def _material_animation_paths(context, arm):
    """Animated/driven paths that own bone or group material state.

    ``PPF_BonePhysics`` is registered on ``bpy.types.Bone``, so a bone material
    curve or driver lives on the ARMATURE DATABLOCK (``arm.data`` / ``bone.id_data``)
    and not on the armature object. The scene is read as well because it owns the
    material groups. All three owners are scanned, and a failure to read any of them
    is refused instead of being treated as "no animation".
    """
    owners = ((context.scene, "scene"), (arm.data, "armature data"), (arm, "armature object"))
    paths = []
    for block, label in owners:
        if block is None:
            raise ValueError(iface("{label}: cannot read animation curves ({error})").format(label=label, error="missing ID owner"))
        paths.extend(_owner_animation_paths(block, label))
    return paths


def stale_simulation_output(context):
    """Reasons the previous solve's output belongs to the OLD physical body."""
    scene = context.scene
    reasons = []
    try:
        from ..core.rigid_pose import has_bone_pose_cache
        if has_bone_pose_cache(scene):
            reasons.append("bone pose frames")
    except Exception:
        pass
    try:
        from ..core.pc2 import scene_has_solver_cache
        if scene_has_solver_cache(scene):
            reasons.append("solver mesh cache")
    except Exception:
        pass
    if has_addon_data(scene):
        state = get_addon_data(scene).state
        convert = getattr(state, "convert_fetched_frames_to_list", None)
        fetched = convert() if callable(convert) else []
        if fetched:
            reasons.append("{count} fetched animation frames".format(count=len(fetched)))
    return reasons


def change_physical_body(context, scope='SELECTED', physical_body='SOFT', repair_from_material=False):
    """Switch an existing owned proxy between the two physical bodies.

    Returns ``(changed, already)``. Raises ValueError, with every reason named,
    BEFORE the first write: a batch that cannot complete leaves every member of the
    selection exactly as it was.
    """
    scene = context.scene
    if not has_addon_data(scene):
        raise ValueError(iface("Register the original PPF addon before switching a physical body"))
    repair = repair_from_material
    if physical_body not in OBJECT_TYPES:
        raise ValueError(iface("Unknown physical body: {name}").format(name=physical_body))
    arm, bones = selected_bones(context, scope=scope)
    target_type = OBJECT_TYPES.get(physical_body)
    wanted_label = 'current owned material' if repair else iface(PHYSICAL_LABELS[physical_body])

    # ---- Phase 1: preflight. No scene, group or bone state is written. ----
    plan = []
    already = 0
    for bone in bones:
        s = bone.ppf_bone_physics
        if repair and (not s.enabled or s.proxy is None):
            continue
        group = _owned_bone_material(scene, arm, bone)
        if group is None:
            raise ValueError(iface(
                "{name}: no owned physics proxy to switch; create or update it first"
            ).format(name=bone.name))
        if repair and sum(other.uuid == group.uuid for other in iterate_active_object_groups(scene)) != 1:
            raise ValueError('{name}: duplicate material-group UUID; repair needs manual review'.format(name=bone.name))
        current_model = 'SOFT' if group.object_type == 'SOLID' else 'RIGID'
        wanted_model = current_model if repair else physical_body
        if repair and s.physical_model != current_model:
            # The current dedicated material is authoritative for legacy drift.
            # A differing hand-edited material is ambiguous and never overwritten.
            if s.is_property_set('density'):
                actual = group.solid_density if current_model == 'SOFT' else group.pdrd_density
                if abs(actual - s.density) > 1e-4:
                    raise ValueError(f'{bone.name}: density differs; repair needs manual review')
            if current_model == 'SOFT':
                from ..models.bone_bindings import effective_solid_material
                from ..core.bone_preset_apply import _validate_rna
                for field, value in effective_solid_material(s, group).items():
                    _validate_rna(s, field, value)
        if current_model == wanted_model and s.physical_model == wanted_model:
            already += 1
            continue
        if wanted_model == 'SOFT':
            has_parent = has_physical_joint_parent(bone)
            refusal = soft_material_refusal(s, bone.name, has_parent, assumed_model='SOFT')
            if refusal is not None:
                # The real gate, evaluated for the TARGET body before any write.
                # Nothing is relaxed to make SOFT fit.
                raise ValueError(iface(
                    "{name}: cannot switch to a soft body. {detail}"
                ).format(name=bone.name, detail=iface(refusal[1])))
        plan.append((bone, s, group, wanted_model))
    if not plan:
        raise ValueError(iface("Every selected bone is already a {wanted} body").format(wanted=wanted_label))

    # Plan every incoming SOFT material before any switch. Keep true E even
    # when only the density-normalized representation fits the group's RNA.
    # Leaving SOFT must not silently discard pre-existing author/material drift.
    solid_plans = {}
    if not repair:
        from ..models.bone_bindings import effective_solid_material
        material_keys = ('solid_model', 'solid_young_modulus', 'solid_poisson_ratio', 'solid_deformation_damping')
        for bone, settings, group, wanted_model in plan:
            authored = {key: getattr(settings, key) for key in material_keys}
            actual_density = group.solid_density if group.object_type == 'SOLID' else group.pdrd_density
            if not math.isclose(float(settings.density), float(actual_density), rel_tol=2e-6, abs_tol=1e-9):
                raise ValueError(bone.name + '：作者与当前材质密度不同，请先人工核对；整批未切换')
            if wanted_model == 'SOFT' and settings.mass_mode == 'MASS':
                from ..core.bone_mass import _measure
                from ..core.bone_mass_math import density_for_mass
                density = density_for_mass(settings.target_mass_kg, _measure(scene, settings)['volume_m3'])
                if not math.isclose(float(settings.density), density, rel_tol=2e-7):
                    raise ValueError(bone.name + '：质量密度尚未同步，请等待刷新后再切换；整批未切换')
            if group.object_type == 'SOLID':
                actual = effective_solid_material(settings, group)
                if any((not math.isclose(actual[key], value, rel_tol=2e-6, abs_tol=1e-9))
                       if type(value) in (int, float) else actual[key] != value
                       for key, value in authored.items()):
                    raise ValueError(bone.name + '：作者与当前软体材质不同，请先人工核对；整批未切换')
            if wanted_model == 'SOFT':
                try:
                    solid_plans[bone.name] = _plan_copied_soft_material(
                        settings, group, authored, float(settings.density), group.young_mod_density_normalized)
                except ValueError as error:
                    raise ValueError(bone.name + '：无法安全切换软体材质；' + str(error)) from error

    # The child-owned side of every incident enabled edge, against the PLANNED
    # final kinds. Refused before any write, naming the bone and the edge.
    planned_models = {bone_.name: ('SOFT' if bone_.ppf_bone_physics.physical_model == 'SOFT' else 'RIGID')
                      for bone_ in arm.data.bones}
    for bone_, _s_, _g_, wanted_model in plan:
        planned_models[bone_.name] = wanted_model
    # Include both sides of each enabled edge, even when the soft endpoint is
    # unchanged and only its selected rigid neighbor is changing.
    changed_names = {bone_.name for bone_, _s_, _g_, _m_ in plan}
    for bone_ in arm.data.bones:
        if not bone_.ppf_bone_physics.enabled:
            continue
        for child_name, kind, mode in incident_edge_refusals(bone_, planned_models):
            if bone_.name not in changed_names and child_name not in changed_names:
                continue
            if kind == 'joint':
                raise ValueError(iface(
                    "{name}: the joint to its enabled child '{child}' is a hard {mode} joint; "
                    "a soft body supports only a compliant joint"
                ).format(name=bone_.name, child=child_name, mode=mode))
            raise ValueError(iface(
                "{name}: the edge to its enabled child '{child}' excludes collision with its connected neighbor; "
                "a soft body cannot"
            ).format(name=bone_.name, child=child_name))

    # Main's core hook: an acknowledged old upload, or a running task with queued
    # frames, must be settled BEFORE any authored state changes.
    com = _facade_communicator()
    com.assert_bone_body_switch_idle()

    reasons = stale_simulation_output(context)
    if reasons:
        raise ValueError("{notice} ({reasons})".format(
            notice=iface("Clear the previous solve first: its frames belong to the old physical body "
                         "and can never be applied to the new one"),
            reasons=", ".join(reasons)))

    paths = _material_animation_paths(context, arm)
    for bone, s, group, _wanted_model in plan:
        prefix = group.path_from_id() + '.'
        if any(path.startswith(prefix) for path in paths):
            raise ValueError(iface(
                "{name}: cannot change a physical body whose material state is keyframed or driven"
            ).format(name=bone.name))
        bone_prefix = escaped_bone_path(bone) + '.ppf_bone_physics.'
        for path in paths:
            if not path.startswith(bone_prefix):
                continue
            if path.rsplit('.', 1)[-1] in set(MATERIAL_OWNED_PROPERTIES) | {'density'}:
                raise ValueError(iface(
                    "{name}: cannot change a physical body whose material state is keyframed or driven"
                ).format(name=bone.name))

    # ---- Phase 2: commit. Only in-memory RNA on the group this bone already owns.
    group_fields = ('object_type', 'pdrd_density', 'solid_model', 'solid_density',
                    'solid_young_modulus', 'solid_poisson_ratio',
                    'young_mod_density_normalized', 'deformation_damping')
    undo = [(s, s.physical_model, s.is_property_set('physical_model'), group,
                     {key: getattr(group, key) for key in group_fields},
                     {key: group.is_property_set(key) for key in group_fields})
            for _bone, s, group, _wanted_model in plan]
    # Retirement is part of the successful change, not an afterthought: if it
    # fails the RNA writes above are rolled back so the scene can never disagree
    # with an upload the core still considers acknowledged.
    try:
        for bone, s, group, wanted_model in plan:
            s.physical_model = wanted_model
            if not repair:
                group.object_type = target_type
                if target_type == 'PDRD':
                    group.pdrd_density = s.density
                    sync_owned_material(s)
                else:
                    _apply_copied_soft_material(s, group, solid_plans[bone.name])
        assign_display_indices(scene)
        invalidate_overlays()
        context.view_layer.update()
        from ..core.bone_mass import mark_settings
        for _bone, settings, _group, _wanted_model in plan:
            mark_settings(settings, context)
        com.bone_physical_body_changed()
    except Exception:
        for s, model, model_set, group, values, property_set in reversed(undo):
            s.physical_model = model
            if not model_set: s.property_unset('physical_model')
            for key, value in values.items():
                setattr(group, key, value)
                if not property_set[key]: group.property_unset(key)
            # Restore the exact saved group fields, including deliberate
            # non-default inactive material values; sync would overwrite them.
        context.view_layer.update()
        raise
    return len(plan), already


class PPF_OT_BonePhysicalBodyChange(bpy.types.Operator):
    bl_idname = 'ppf.bone_physical_body_change'
    bl_label = '切换骨骼物理体'
    bl_description = '切换已有代理的刚体或软体类型，保留网格、变换、本地轴和身份；更改后需重新传输和构建'
    bl_options = {'REGISTER', 'UNDO'}
    scope: EnumProperty(name='Bones', items=SCOPES, default='SELECTED')
    physical_body: EnumProperty(name='Physical Body', items=BODY_ITEMS, default='SOFT')
    repair_from_material: BoolProperty(name='Repair declaration from dedicated material', default=False, options={'SKIP_SAVE'})

    @classmethod
    def poll(cls, context):
        return context.mode in {'OBJECT', 'POSE'} and active_armature(context) is not None

    def execute(self, context):
        try:
            changed, already = change_physical_body(
                context, scope=self.scope, physical_body=self.physical_body,
                repair_from_material=self.repair_from_material)
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        except Exception as exc:  # noqa: BLE001 - a core refusal is surfaced, never swallowed
            # A core hook/refusal failure must reach the user with its own message;
            # reporting FINISHED here would claim a retirement that did not happen.
            self.report({'ERROR'}, iface("Physical body change refused: {error}").format(error=exc))
            return {'CANCELLED'}
        if already:
            self.report({'INFO'}, iface("{count} bone(s) already used this physical body").format(count=already))
        self.report({'INFO'}, iface(
            "Physical body changed on {count} bone(s); Transfer, Build and run again").format(count=changed))
        return {'FINISHED'}


def incident_edge_refusals(bone, planned_models, enabled_names=None):
    """Outgoing enabled edges of *bone* that cannot be incident to a soft body.

    A joint connecting two bones is stored on the CHILD, so a selected parent that
    becomes FEM makes its child's hard hinge/D6 incident to FEM even when the parent's own
    incoming joint is compliant. The incoming side of every planned soft body is
    already refused by ``soft_material_refusal(..., assumed_model='SOFT')``; this
    covers the child-owned side, against the planned final kinds. Nothing is
    changed here: the values only decide whether the whole batch is refused.
    """
    problems = []
    final = planned_models.get(bone.name)
    for child in bone.children:
        settings = child.ppf_bone_physics
        if not (settings.enabled or enabled_names is not None and child.name in enabled_names):
            continue
        if final != 'SOFT' and planned_models.get(child.name) != 'SOFT':
            continue
        if settings.joint_mode != 'SOFT':
            problems.append((child.name, 'joint', settings.joint_mode))
    return problems


def _facade_communicator():
    """Main's core facade communicator, which owns upload retirement.

    Both hooks are REQUIRED: without them the operator cannot stop an
    already-acknowledged upload from authorizing the previous physical body, and
    silently continuing would be exactly the gap this seam closes.
    """
    from ..core import facade as facade_module
    com = getattr(facade_module, "communicator", None)
    if com is None:
        raise ValueError(iface("The core facade communicator is unavailable; a physical body cannot be changed"))
    for hook in ("assert_bone_body_switch_idle", "bone_physical_body_changed"):
        if not callable(getattr(com, hook, None)):
            raise ValueError(iface(
                "The core facade hook {name} is missing; this addon build cannot retire the previous Transfer"
            ).format(name=hook))
    return com


classes = (PPF_OT_BoneParticipation, PPF_OT_BoneProxyUpdate, PPF_OT_BoneProxyDelete, PPF_OT_BoneProxySelect, PPF_OT_BoneProxyReturn, PPF_OT_BoneCustomFromSelection, PPF_OT_BoneCustomApply, PPF_OT_BoneSettingsCopy, PPF_OT_BoneControlEventEdit, PPF_OT_BoneDriveTargetEdit, PPF_OT_BonePhysicalBodyChange)


def register():
    bpy.types.WindowManager.ppf_bone_participation_scope=EnumProperty(name='范围',items=[('SELECTED','选中骨骼',''),('CHAIN','当前链',''),('ALL','整个骨架','')],default='SELECTED',options={'SKIP_SAVE'})
    for cls in classes:_ppf_i18n.register_class(cls)
    from . import bone_material
    bone_material.register()
    from . import bone_proxy_editor
    bone_proxy_editor.register()
    from . import bone_weight_fit
    bone_weight_fit.register()


def unregister():
    if hasattr(bpy.types.WindowManager,'ppf_bone_participation_scope'):del bpy.types.WindowManager.ppf_bone_participation_scope
    from . import bone_proxy_editor
    from . import bone_weight_fit
    bone_weight_fit.unregister()
    bone_proxy_editor.unregister()
    from . import bone_material
    bone_material.unregister()
    for cls in reversed(classes):bpy.utils.unregister_class(cls)
