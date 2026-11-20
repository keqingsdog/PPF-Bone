"""Read-only GPU display of actual bone proxies, independent of solver data."""
from .. import i18n as _ppf_i18n
import bpy
import bmesh
import gpu
from time import perf_counter
from collections import deque
from mathutils import Matrix, Vector
from ..core.compat import pose_bone_selected
from bpy.app.handlers import persistent
from ..i18n.props import BoolProperty, FloatProperty, FloatVectorProperty, IntProperty, PointerProperty
from gpu_extras.batch import batch_for_shader

_handler = None
_registered = False
_rebuilding = False
_generation = 0
_snapshots = {}  # Primitive viewport keys -> immutable primitive draw data.
_geometry_snapshots = {}  # Viewport -> primitive proxy tokens and copied geometry.
_known_proxy_meshes = set()
_geometry_serial = 0
_mesh_revisions = {}  # Mesh ID pointers -> monotonic dirtiness, shared by all viewports.
_object_revisions = {}  # Evaluated deformation dirtiness; never consumed by one viewport.
_view_publications = {}  # Primitive matrix/geometry signature, revision and original birth.
_publication_serial = 0
_cpu_stats = {key: deque(maxlen=2048) for key in
              ('producer_ms', 'capture_ms', 'evaluation_ms', 'local_geometry_ms')}
_gpu_views = {}  # Viewport -> GPU resources; never RNA. Created only inside draw.
_dynamic_plans = {}  # viewport -> primitive object identity/topology plans
_dynamic_packets = {}  # viewport -> immutable current-frame vertex/matrix data
_dynamic_serial = 0
_native_consumed = {}  # scene -> generation/epoch/frame/completion serial
_fast_views = {}  # viewport -> generation for plans covering all visible entries
_native_overlay_signature = ()
_gpu_stats = {'shaders': 0, 'batches': 0, 'draw_ms': deque(maxlen=2048),
              'upload_ms': deque(maxlen=2048)}
_GPU_UPLOAD_BUDGET = .002
_producer_running = False
_visibility_pending = True
_reported_errors = set()
_REFRESH_INTERVAL = 1. / 30.  # Normal display sampling cadence, not an Undo cooldown.
_snapshot_dirty = True
_ui_signature = None


def _redraw():
    manager = getattr(bpy.context, 'window_manager', None)
    if manager is None:
        return
    for window in manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        region.tag_redraw()


def _redraw_pending_views():
    """Wake only viewports able to consume a pending upload."""
    global _native_overlay_signature
    manager = getattr(bpy.context, 'window_manager', None)
    if manager is None:
        _native_overlay_signature = ()
        return
    signature = []
    for window in manager.windows:
        for area in window.screen.areas:
            if area.type != 'VIEW_3D':
                continue
            region = next((r for r in area.regions if r.type == 'WINDOW'), None)
            if region is None:
                continue
            with bpy.context.temp_override(window=window, area=area, region=region):
                key = _viewport_key(bpy.context)
                visible = bool(bpy.context.space_data.overlay.show_overlays)
                signature.append((key, visible))
                view = _gpu_views.get(key)
                if visible and view and view.get('inflight') is not None:
                    region.tag_redraw()
    _native_overlay_signature = tuple(signature)


def _remove_handler():
    global _handler
    if _handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handler, 'WINDOW')
        _handler = None


def _sync_handler(*_args):
    global _handler
    if _rebuilding or not _registered:
        return
    wanted = any(getattr(scene, 'ppf_bone_display', None) and
                 scene.ppf_bone_display.enabled for scene in bpy.data.scenes)
    if wanted and _handler is None:
        _handler = bpy.types.SpaceView3D.draw_handler_add(draw, (), 'WINDOW', 'POST_VIEW')
    elif not wanted:
        _remove_handler()


def _update_display(self, context):
    """RNA property updates require exactly two positional parameters."""
    _invalidate_snapshots(geometry=False)
    if _registered and not _rebuilding:
        _ensure_producer()


class PPF_PG_BoneDisplay(bpy.types.PropertyGroup):
    enabled: BoolProperty(name='GPU 总闸', default=False, update=_update_display)
    proxy_enabled: BoolProperty(name='显示代理', default=True, update=_update_display)
    faces: BoolProperty(name='透明表面', default=True, update=_update_display)
    edges: BoolProperty(name='叠加线框', default=True, update=_update_display)
    in_front: BoolProperty(name='叠加层透视', default=False, update=_update_display)
    selected_only: BoolProperty(name='叠加层仅选中', default=False, update=_update_display)
    opacity: FloatProperty(name='表面不透明度', default=0.18, min=0.0, max=1.0, update=_update_display)
    cross_links: BoolProperty(name='链间弹簧', description='连接示意，非力大小；软体锚点位置为近似', default=True, update=_update_display)
    cross_link_color: FloatVectorProperty(name='连接颜色', subtype='COLOR', size=4, default=(1.0, 0.55, 0.08, 0.9), min=0.0, max=1.0, update=_update_display)
    cross_link_width: FloatProperty(name='连接线宽', default=2.0, min=1.0, max=10.0, update=_update_display)
    cross_link_in_front: BoolProperty(name='连接透视显示', description='仅让链间弹簧穿过实体显示，避免骨内锚点被代理或角色表面遮挡', default=True, update=_update_display)
    refresh_fps: IntProperty(name='共享刷新频率', description='所有已启用场景共用采样计时，采用其中最高频率；空闲时不重复生成几何。不影响解算步长和时间轴播放速度。', default=30, min=1, max=60, update=_update_display)


def _edit_record(proxy):
    import json
    from ..core import bone_author_follow
    record = json.loads(proxy.get('ppf_bone_author_follow_v1', '{}'))
    metadata = bone_author_follow._edit_metadata.get((proxy.get('ppf_armature_uuid'),
                                                    proxy.get('ppf_bone_uuid'),
                                                    proxy.get('_solver_uuid')))
    if metadata:
        record.update(metadata)
    return record


def bindings(context):
    """Resolve live owner identities in this scene and view layer; never repair."""
    if not _identity_ready():
        return
    from collections import Counter
    from ..core.bone_proxy_visibility import overlay_visible
    collection_visibility = {}
    body_counts = Counter(obj.get('_solver_uuid') for obj in context.scene.objects if obj.get('_solver_uuid'))
    proxy_counts = Counter(
        bone.ppf_bone_physics.proxy.as_pointer()
        for arm in context.scene.objects if arm.type == 'ARMATURE' and arm.mode != 'EDIT'
        for bone in arm.data.bones if bone.ppf_bone_physics.proxy is not None)
    for arm in context.scene.objects:
        if arm.type != 'ARMATURE' or arm.name not in context.view_layer.objects:
            continue
        if not arm.visible_get(view_layer=context.view_layer, viewport=context.space_data):
            continue
        if arm.mode == 'EDIT':
            from types import SimpleNamespace
            from ..models.bone_bindings import proxy_parent_armature
            import json
            for proxy in context.scene.objects:
                if not proxy.get('ppf_bone_proxy_owned') or proxy.type != 'MESH':
                    continue
                try:
                    if proxy_parent_armature(proxy) != arm:
                        continue
                    record = _edit_record(proxy)
                    display = record.get('display', {})
                    edit = arm.data.edit_bones.get(record.get('bone_name', ''))
                    if (edit is None or not display.get('enabled') or edit.hide
                            or display.get('body_uuid') != proxy.get('_solver_uuid')
                            or display.get('bone_uuid') != proxy.get('ppf_bone_uuid')
                            or body_counts[display['body_uuid']] != 1
                            or not overlay_visible(context, proxy, collection_visibility=collection_visibility)):
                        continue
                    selected = bool(edit.select or edit.select_head or edit.select_tail)
                    yield proxy, display['physical_model'], selected, SimpleNamespace(**display)
                except (ValueError, TypeError, KeyError, ReferenceError):
                    continue
            continue
        for bone in arm.data.bones:
            s = bone.ppf_bone_physics
            proxy = s.proxy
            if (not s.enabled or bone.hide or not proxy or proxy.type != 'MESH'
                    or (proxy.mode=='EDIT' and proxy.get('ppf_bone_proxy_visibility_editing'))
                    or proxy.name not in context.scene.objects
                    or proxy.name not in context.view_layer.objects
                    or not overlay_visible(context, proxy, collection_visibility=collection_visibility)
                    or not proxy.get('ppf_bone_proxy_owned')
                    or not s.body_uuid or not s.bone_uuid or not arm.ppf_armature_uuid
                    or body_counts[s.body_uuid] != 1
                    or proxy_counts[proxy.as_pointer()] != 1
                    or proxy.get('_solver_uuid') != s.body_uuid
                    or proxy.get('ppf_bone_uuid') != s.bone_uuid
                    or proxy.get('ppf_armature_uuid') != arm.ppf_armature_uuid):
                continue
            if bone.collections and not any(c.is_visible_effectively for c in bone.collections):
                continue
            # Copies with a stale shared pointer must not draw twice.
            from ..models.bone_bindings import proxy_parent_armature
            try:
                owner = proxy_parent_armature(proxy)
            except ValueError:
                continue
            if owner is not None and owner != arm:
                continue
            pose_bone = arm.pose.bones.get(bone.name)
            edit_bone = arm.data.edit_bones.get(bone.name) if context.mode == 'EDIT_ARMATURE' else None
            selected = bool(edit_bone and (edit_bone.select or edit_bone.select_head or edit_bone.select_tail)) or (proxy.visible_get(view_layer=context.view_layer, viewport=context.space_data) and proxy.select_get(view_layer=context.view_layer)) or bool(
                context.object == arm and context.mode == 'POSE' and pose_bone and pose_bone_selected(pose_bone))
            yield proxy, s.physical_model, selected, s


def edit_preview_matrix(context, proxy):
    """Project the author's offset into live Edit rest space, without writes.

    The raw offset carries inverse pose scale when proxy geometry was already
    baked at author scale. Applying it once also keeps offset translations in
    rest-bone space; multiplying pose scale again would displace the center.
    Length changes use the same reference as author-follow commit: shift the
    default center by half the length delta and stretch proxy-local X, retaining
    the author's extra translation, rotation and transverse dimensions. The
    shared source-mesh pivot also preserves translations authored in Edit Mesh.
    Legacy offsets describe current placement only, never historical recovery.
    """
    if (not _identity_ready() or context.mode != 'EDIT_ARMATURE'
            or getattr(getattr(context, 'screen', None), 'is_animation_playing', False)):
        return None
    from ..core import bone_author_follow
    try:
        if (not bone_author_follow._finite_invertible(proxy.matrix_world)
                or not bone_author_follow._author_scene(context.scene)):
            return None  # Native frames, playback caches and busy jobs retain authority.
        for arm in context.scene.objects:
            if arm.type != 'ARMATURE' or arm.mode != 'EDIT':
                continue
            import json
            from mathutils import Matrix, Vector
            from ..models.bone_bindings import proxy_parent_armature
            if proxy_parent_armature(proxy) != arm:
                continue
            record = _edit_record(proxy)
            for edit_bone in arm.data.edit_bones:
                if edit_bone.name != record.get('bone_name'):
                    continue
                if edit_bone is None:
                    return None
                offset = Matrix([record['offset'][i:i+4] for i in range(0,16,4)])
                reference_length = bone_author_follow._reference_length((
                    Matrix([record['arm_world'][i:i+4] for i in range(0, 16, 4)]),
                    Vector(record['rest_head']), Vector(record['rest_tail'])))
                from math import isfinite
                from mathutils import Matrix
                length = float(edit_bone.length)
                if (reference_length is None or not isfinite(reference_length)
                        or reference_length <= 1.e-8 or not isfinite(length) or length <= 1.e-8):
                    return None
                ratio = length / reference_length
                center_delta = Matrix.Translation((0., (length - reference_length) * .5, 0.))
                axial_scale = bone_author_follow.author_length_stretch(proxy, ratio)
                matrix = arm.matrix_world @ edit_bone.matrix @ center_delta @ offset @ axial_scale
                if bone_author_follow._finite_invertible(matrix):
                    return matrix
                return None
    except (ReferenceError, RuntimeError, AttributeError, KeyError, TypeError, ValueError):
        return None  # A retiring/rebuilding owner must not leave a stale preview.
    return None


def _identity_ready():
    """Check primitive lifecycle flags before touching Scene/Object/Mesh RNA."""
    if _rebuilding:
        return False
    from ..core import bone_author_follow
    from .dynamics.group_ops import cleanup_identity_rebuilding
    return not bone_author_follow._identity_rebuilding and not cleanup_identity_rebuilding()


def geometry(proxy, depsgraph, *, world_matrix=None, _local=False):
    """Copy geometry synchronously; no source mesh or RNA escapes this call.

    Production callers run after a completed fresh view-layer evaluation, outside
    the region draw callback. Even hidden managed meshes use an owned evaluated
    temporary; drawing must never iterate the original Mesh.vertices collection.
    """
    if not _identity_ready():
        return (), (), ()
    if proxy.mode == 'EDIT':
        # Live Edit Mesh is copied only in the stable main-loop producer. Never
        # retain the BMesh/verts/loops through an event or an Undo boundary.
        mesh = bmesh.from_edit_mesh(proxy.data)
        verts = list(mesh.verts)
        indices = {v: i for i, v in enumerate(verts)}
        matrix = world_matrix if world_matrix is not None else proxy.matrix_world.copy()
        positions = (tuple(tuple(v.co) for v in verts) if _local
                     else tuple(tuple(matrix @ v.co) for v in verts))
        edges = tuple((indices[e.verts[0]], indices[e.verts[1]]) for e in mesh.edges)
        triangles = tuple(tuple(indices[loop.vert] for loop in tri) for tri in mesh.calc_loop_triangles())
        return positions, edges, triangles
    from array import array
    from mathutils import Vector
    evaluated = proxy.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
    if mesh is None:
        return (), (), ()
    try:
        coordinates = array('f', [0.]) * (len(mesh.vertices) * 3)
        mesh.vertices.foreach_get('co', coordinates)
        edge_indices = array('i', [0]) * (len(mesh.edges) * 2)
        mesh.edges.foreach_get('vertices', edge_indices)
        mesh.calc_loop_triangles()
        triangle_indices = array('i', [0]) * (len(mesh.loop_triangles) * 3)
        mesh.loop_triangles.foreach_get('vertices', triangle_indices)
        matrix = evaluated.matrix_world.copy()
        if world_matrix is not None:
            matrix = world_matrix @ proxy.matrix_world.inverted() @ matrix
        positions = (tuple(tuple(coordinates[i:i+3]) for i in range(0, len(coordinates), 3))
                     if _local else tuple(tuple(matrix @ Vector(coordinates[i:i+3]))
                                          for i in range(0, len(coordinates), 3)))
        edges = tuple(tuple(edge_indices[i:i+2]) for i in range(0, len(edge_indices), 2))
        triangles = tuple(tuple(triangle_indices[i:i+3]) for i in range(0, len(triangle_indices), 3))
        return positions, edges, triangles
    finally:
        evaluated.to_mesh_clear()


def _viewport_key(context):
    return (context.window.as_pointer(), context.area.as_pointer(), context.region.as_pointer(),
            context.scene.as_pointer(), context.view_layer.as_pointer())


def _matrix_token(matrix):
    return tuple(float(value) for row in matrix for value in row)


def _model_matrix(context, proxy, depsgraph):
    """Copy only the placement; local evaluated vertices remain independently cached."""
    preview = edit_preview_matrix(context, proxy)
    if proxy.mode == 'EDIT':
        return _matrix_token(preview if preview is not None else proxy.matrix_world)
    evaluated_world = proxy.evaluated_get(depsgraph).matrix_world.copy()
    if preview is not None:
        evaluated_world = preview @ proxy.matrix_world.inverted() @ evaluated_world
    return _matrix_token(evaluated_world)



def _cross_link_segments(context, depsgraph=None, plan=None):
    """Copy schematic anchors into world coordinates in the stable producer.

    Only endpoints with the same valid visible proxy ownership as the overlay
    participate. No Bone or PoseBone wrapper survives this function.
    """
    from collections import Counter
    from math import isfinite
    from ..models.groups import iterate_object_groups
    from ..models.bone_bindings import armature_data_owner_counts
    data_owners = armature_data_owner_counts()
    groups = {g.uuid: g for g in iterate_object_groups(context.scene)}
    visible = {}
    for proxy, model, selected, owner in bindings(context):
        group = groups.get(owner.group_uuid)
        if (group is None or not group.active
                or group.object_type != ('SOLID' if model == 'SOFT' else 'PDRD')
                or len(group.assigned_objects) != 1
                or group.assigned_objects[0].uuid != owner.body_uuid
                or not group.assigned_objects[0].included
                or len(proxy.users_scene) != 1):
            continue
        key = (proxy.get('ppf_armature_uuid'), owner.bone_uuid)
        visible[key] = bool(selected)
    arms = [a for a in context.scene.objects if a.type == 'ARMATURE' and a.ppf_armature_uuid]
    arm_counts = Counter(a.ppf_armature_uuid for a in arms)
    link_counts = Counter(row.link_uuid for a in arms for row in a.ppf_cross_links if row.link_uuid)
    result = []
    for arm in arms:
        if (arm_counts[arm.ppf_armature_uuid] != 1 or arm.library or arm.data.library
                or data_owners[arm.data.as_pointer()] != 1 or len(arm.users_scene) != 1):
            continue
        bones = {}
        duplicates = set()
        if arm.mode == 'EDIT':
            import json
            primitive_bones = []
            for proxy in context.scene.objects:
                if proxy.get('ppf_armature_uuid') != arm.ppf_armature_uuid:
                    continue
                try:
                    record = _edit_record(proxy)
                    if record.get('bone_name'):
                        primitive_bones.append((record.get('bone_uuid'), record['bone_name']))
                except (ValueError, TypeError):
                    continue
        else:
            primitive_bones = [(bone.ppf_bone_physics.bone_uuid, bone.name) for bone in arm.data.bones]
        for uid, name in primitive_bones:
            if uid in bones:
                duplicates.add(uid)
            elif uid:
                bones[uid] = name
        for row in arm.ppf_cross_links:
            a, b = row.bone_a_uuid, row.bone_b_uuid
            if (not row.enabled or not row.link_uuid or link_counts[row.link_uuid] != 1
                    or not (row.tension_enabled or row.compression_enabled)
                    or not a or not b or a == b or a in duplicates or b in duplicates
                    or (arm.ppf_armature_uuid, a) not in visible
                    or (arm.ppf_armature_uuid, b) not in visible):
                continue
            selected = bool(visible[(arm.ppf_armature_uuid, a)] or visible[(arm.ppf_armature_uuid, b)])
            if context.scene.ppf_bone_display.selected_only and not selected:
                continue
            try:
                points = []
                for uid, anchor in ((a, row.anchor_a_bone), (b, row.anchor_b_bone)):
                    name = bones[uid]
                    eval_arm = arm.evaluated_get(depsgraph) if depsgraph is not None and arm.mode != 'EDIT' else arm
                    bone = (arm.data.edit_bones.get(name) if arm.mode == 'EDIT' else eval_arm.pose.bones.get(name))
                    if bone is None:
                        break
                    point = eval_arm.matrix_world @ bone.matrix @ Vector(anchor)
                    if not all(isfinite(v) for v in point):
                        break
                    points.append(tuple(point))
                if len(points) == 2:
                    color = (row.preview_color if row.preview_color_custom
                             else context.scene.ppf_bone_display.cross_link_color)
                    color = tuple(float(v) for v in color)
                    result.append((points[0], points[1], selected, color))
                    if plan is not None and arm.mode != 'EDIT':
                        endpoints = []
                        for uid in (a, b):
                            source_bone = arm.data.bones[bones[uid]]
                            settings = source_bone.ppf_bone_physics
                            proxy = settings.proxy
                            if proxy is None:
                                break
                            endpoints.append((source_bone.name, settings.bone_uuid, settings.body_uuid,
                                              proxy.name, proxy.session_uid, proxy.as_pointer()))
                        if len(endpoints) == 2:
                            plan.append((arm.name, arm.session_uid, arm.ppf_armature_uuid,
                                         endpoints[0], endpoints[1], tuple(row.anchor_a_bone),
                                         tuple(row.anchor_b_bone), selected, color))
            except (KeyError, ReferenceError, RuntimeError, ValueError, TypeError):
                continue
    return tuple(result)

def _capture_view(context, generation):
    started = perf_counter()
    try:
        return _capture_view_impl(context, generation)
    finally:
        _cpu_stats['capture_ms'].append((perf_counter()-started)*1000.)


_pose_offset_diagnostics = set()
_frame_epochs = {}
_frame_publications = {}
_frame_known_targets = {}
_frame_publishing = False
_frame_plans = {}
_frame_link_plans = {}


def _pose_display_matrices(context, depsgraph, *, allow_cache=True, targets=None, plan=None):
    """Build evaluated rigid placements once per capture, without RNA writes."""
    from ..core import bone_author_follow as follow
    from ..core import rigid_pose
    from ..core.single_bake_restore import frozen_proxy
    from ..models.bone_bindings import proxy_parent_armature, armature_data_owner_counts
    data_owners = armature_data_owner_counts()
    from collections import Counter
    arm_counts = Counter(o.ppf_armature_uuid for o in bpy.data.objects if o.type == 'ARMATURE')
    result = {}
    key = context.scene.get(rigid_pose.CACHE_KEY)
    index = rigid_pose._cache.get(key) if key else None
    bindings = (index or {}).get('receipt', {}).get('bindings', ()) if allow_cache else ()
    cached = {(r['armature_uuid'], r['bone_uuid'], r['body_uuid']) for r in bindings}
    for arm in context.scene.objects:
        if arm.type != 'ARMATURE' or arm.mode == 'EDIT':
            continue
        if arm.library or arm.data.library or len(arm.users_scene) > 1 or data_owners[arm.data.as_pointer()] > 1:
            continue
        proxy_counts = Counter(b.ppf_bone_physics.proxy.as_pointer() for b in arm.data.bones if b.ppf_bone_physics.proxy)
        driven = follow.time_driven_bones(arm)
        evaluated = arm.evaluated_get(depsgraph)
        for bone in arm.data.bones:
            settings = bone.ppf_bone_physics
            proxy = settings.proxy
            if (not settings.enabled or settings.physical_model != 'RIGID' or proxy is None
                    or proxy.type != 'MESH' or proxy.mode == 'EDIT' or frozen_proxy(context.scene, proxy)
                    or not ((arm.ppf_armature_uuid, settings.bone_uuid, settings.body_uuid) in cached or bone.name in driven)
                    or any(m.type == 'MESH_CACHE' and m.show_viewport for m in proxy.modifiers)
                                        or (proxy.animation_data and (proxy.animation_data.action or proxy.animation_data.drivers or proxy.animation_data.nla_tracks))
                    or any(not c.mute for c in proxy.constraints)
                    or (proxy.data.shape_keys is not None and proxy.data.shape_keys.animation_data
                        and (proxy.data.shape_keys.animation_data.action or proxy.data.shape_keys.animation_data.drivers
                             or proxy.data.shape_keys.animation_data.nla_tracks))):
                continue
            if targets is not None:
                targets.add(proxy.as_pointer())
            try:
                owner = proxy_parent_armature(proxy)
                valid = (context.scene.objects.get(proxy.name) == proxy and len(proxy.users_scene) <= 1
                         and proxy_counts[proxy.as_pointer()] == 1 and proxy.get('ppf_bone_proxy_owned')
                         and proxy.get('ppf_bone_uuid') == settings.bone_uuid
                         and proxy.get('ppf_armature_uuid') == arm.ppf_armature_uuid
                         and proxy.get('_solver_uuid') == settings.body_uuid
                         and (owner == arm or (owner is None and arm_counts[arm.ppf_armature_uuid] == 1)))
                if not valid:
                    continue
                record = follow._load_author_record(settings, proxy)
                row_key = (context.scene.as_pointer(), arm.as_pointer(), bone.name, proxy.as_pointer())
                if record is not None:
                    offset, authority = record[0][2], 'durable'
                elif row_key in follow._seen and row_key not in follow._unproven:
                    offset, authority = follow._seen[row_key][2], 'baseline'
                else:
                    offset, authority = None, 'unavailable'
            except (AttributeError, KeyError, ReferenceError, RuntimeError, TypeError, ValueError):
                continue
            pose = evaluated.pose.bones.get(bone.name)
            if authority not in ('durable', 'baseline') or pose is None:
                diagnostic = (arm.ppf_armature_uuid, settings.bone_uuid, settings.body_uuid, authority)
                if diagnostic not in _pose_offset_diagnostics:
                    _pose_offset_diagnostics.add(diagnostic)
                    print('PPF bone overlay: trusted author offset unavailable:', diagnostic)
                continue
            matrix = evaluated.matrix_world @ pose.matrix @ offset
            if follow._finite_invertible(matrix):
                result[proxy.as_pointer()] = _matrix_token(matrix)
                if plan is not None:
                    plan.append((arm.name, arm.session_uid, bone.name, proxy.name, proxy.session_uid, proxy.as_pointer(), settings.bone_uuid, settings.body_uuid, _matrix_token(offset), authority, proxy.get(follow._AUTHOR_RECORD)))
    return result

@persistent
def _frame_placement_pre(scene, _depsgraph=None):
    key = scene.as_pointer()
    _frame_epochs[key] = _frame_epochs.get(key, 0) + 1
    _frame_publications.pop(key, None)
    for viewport in tuple(_dynamic_packets):
        if viewport[3] == key:
            _dynamic_packets.pop(viewport, None)


def _dynamic_frame(scene, depsgraph, epoch, viewport=None):
    """Copy supported evaluated meshes into a scene/frame-scoped packet."""
    global _dynamic_serial
    viewports = (viewport,) if viewport is not None else tuple(k for k in _dynamic_plans if k[3] == scene.as_pointer() and k[4] == depsgraph.view_layer.as_pointer())
    for view_key in viewports:
        _dynamic_frame_view(scene, depsgraph, epoch, view_key)


def _dynamic_frame_view(scene, depsgraph, epoch, viewport):
    global _dynamic_serial
    plan = _dynamic_plans.get(viewport)
    if not plan or plan[0] != _generation:
        return
    rows = {}
    for name, uid, identity, mesh_uid, vertex_count, mesh_ptr, mesh_revision in plan[1]:
        obj = scene.objects.get(name)
        if (obj is None or obj.session_uid != uid or obj.as_pointer() != identity
                or obj.type != 'MESH' or obj.data.session_uid != mesh_uid
                or obj.data.as_pointer() != mesh_ptr
                or _mesh_revisions.get(mesh_ptr, 0) != mesh_revision):
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            if len(mesh.vertices) != vertex_count:
                continue
            coords = [0.] * (vertex_count * 3)
            mesh.vertices.foreach_get('co', coords)
            positions = tuple(zip(coords[0::3], coords[1::3], coords[2::3]))
            from math import isfinite
            matrix = _matrix_token(evaluated.matrix_world)
            if not all(isfinite(x) for x in coords) or not all(isfinite(x) for x in matrix):
                continue
            rows[identity] = (positions, matrix, mesh_ptr, mesh_revision)
        finally:
            evaluated.to_mesh_clear()
    if epoch == _frame_epochs.get(scene.as_pointer()):
        _dynamic_serial += 1
        _dynamic_packets[viewport] = (_generation, epoch, float(scene.frame_current_final), rows, _dynamic_serial)


def _native_frame_publication(scene, packet):
    """Compose completed native pose primitives with the stable author plan."""
    from math import isfinite
    from ..core import bone_author_follow as follow
    from ..core.single_bake_restore import frozen_proxy
    scene_key = scene.as_pointer()
    planned = _frame_plans.get(scene_key)
    if planned is None or planned[0] != _generation:
        return None
    by_id = {(r[2], r[4], r[5]): r for r in packet[9]}
    matrices, worlds = {}, {}
    for row in packet[9]:
        arm = scene.objects.get(row[0])
        if arm is None or arm.mode == 'EDIT' or arm.session_uid != row[1] or arm.ppf_armature_uuid != row[2]:
            continue
        bone = arm.data.bones.get(row[3])
        if bone is None or bone.ppf_bone_physics.bone_uuid != row[4] or bone.ppf_bone_physics.body_uuid != row[5]:
            continue
        matrix = Matrix(tuple(row[7][i:i+4] for i in range(0,16,4)))
        if follow._finite_invertible(matrix):
            worlds[(row[2], row[4], row[5])] = matrix
    for arm_name, arm_uid, bone_name, proxy_name, proxy_uid, identity, bone_uuid, body_uuid, offset, authority, record in planned[1]:
        arm = scene.objects.get(arm_name); proxy = scene.objects.get(proxy_name)
        if (arm is None or proxy is None or frozen_proxy(scene, proxy) or arm.mode == 'EDIT' or arm.session_uid != arm_uid
                or proxy.session_uid != proxy_uid or proxy.as_pointer() != identity
                or proxy.get('ppf_armature_uuid') != arm.ppf_armature_uuid
                or proxy.get('ppf_bone_uuid') != bone_uuid or proxy.get('_solver_uuid') != body_uuid
                or proxy.get(follow._AUTHOR_RECORD) != record):
            continue
        bone = arm.data.bones.get(bone_name)
        if (bone is None or not bone.ppf_bone_physics.enabled or bone.ppf_bone_physics.proxy != proxy
                or bone.ppf_bone_physics.bone_uuid != bone_uuid or bone.ppf_bone_physics.body_uuid != body_uuid):
            continue
        if authority == 'baseline':
            baseline_key = (scene_key, arm.as_pointer(), bone_name, identity)
            if baseline_key not in follow._seen or baseline_key in follow._unproven or _matrix_token(follow._seen[baseline_key][2]) != offset:
                continue
        item = by_id.get((arm.ppf_armature_uuid, bone_uuid, body_uuid))
        world = worlds.get((arm.ppf_armature_uuid, bone_uuid, body_uuid))
        if item is None or item[6] != 'PDRD' or world is None:
            continue
        matrix = world @ Matrix(tuple(offset[i:i+4] for i in range(0,16,4)))
        if follow._finite_invertible(matrix):
            matrices[identity] = _matrix_token(matrix)
    links = {}
    arm_cache, endpoint_cache, point_cache = {}, {}, {}
    def endpoint(arm_key, ep):
        key = (arm_key, ep)
        if key in endpoint_cache:
            return endpoint_cache[key]
        arm_name, arm_uid, arm_uuid = arm_key
        if arm_key not in arm_cache:
            arm = scene.objects.get(arm_name)
            arm_cache[arm_key] = (arm if arm is not None and arm.mode != 'EDIT'
                                  and arm.session_uid == arm_uid and arm.ppf_armature_uuid == arm_uuid else None)
        arm = arm_cache[arm_key]
        if arm is None:
            endpoint_cache[key] = None
            return None
        name, bone_uuid, body_uuid, proxy_name, proxy_uid, identity = ep
        bone = arm.data.bones.get(name); proxy = scene.objects.get(proxy_name)
        world = worlds.get((arm_uuid, bone_uuid, body_uuid))
        if (bone is None or proxy is None or world is None or proxy.session_uid != proxy_uid
                or proxy.as_pointer() != identity or proxy.get('ppf_armature_uuid') != arm_uuid
                or proxy.get('ppf_bone_uuid') != bone_uuid or proxy.get('_solver_uuid') != body_uuid
                or not proxy.get('ppf_bone_proxy_owned') or not bone.ppf_bone_physics.enabled
                or bone.ppf_bone_physics.proxy != proxy or bone.ppf_bone_physics.bone_uuid != bone_uuid
                or bone.ppf_bone_physics.body_uuid != body_uuid):
            world = None
        endpoint_cache[key] = world
        return world
    for view_key, rows in _frame_link_plans.get(scene_key, {}).items():
        segments = []
        for arm_name, arm_uid, arm_uuid, ep_a, ep_b, anchor_a, anchor_b, selected, color in rows:
            arm_key = (arm_name, arm_uid, arm_uuid)
            points = []
            for ep, anchor in ((ep_a, anchor_a), (ep_b, anchor_b)):
                key = (arm_key, ep, tuple(anchor))
                if key not in point_cache:
                    world = endpoint(arm_key, ep)
                    point = tuple(world @ Vector(anchor)) if world is not None else None
                    point_cache[key] = point if point is not None and all(isfinite(v) for v in point) else None
                point = point_cache[key]
                if point is None:
                    break
                points.append(point)
            if len(points) == 2:
                segments.append((points[0], points[1], selected, color))
        links[view_key] = tuple(segments)
    return matrices, links


def _consume_native_completion(scene, packet):
    """Publish exactly this completed pose for the current overlay epoch."""
    from ..core import rigid_pose
    key = scene.as_pointer()
    epoch = _frame_epochs.get(key)
    if (not _registered or _rebuilding or epoch is None or not _identity_ready()
            or packet is None or rigid_pose.pose_completion(scene) is not packet):
        return False
    generation = _generation
    token = (generation, epoch, scene.frame_current, float(scene.frame_subframe), packet[1])
    if _native_consumed.get(key) == token and key in _frame_publications:
        return True
    try:
        completed = _native_frame_publication(scene, packet)
        if (completed is None or generation != _generation or _frame_epochs.get(key) != epoch
                or rigid_pose.pose_completion(scene) is not packet):
            return False
        _frame_publications[key] = (generation, epoch, float(scene.frame_current_final), completed[0], completed[1])
        _native_consumed[key] = token
        _redraw()
        return True
    except Exception as exc:
        _frame_publications.pop(key, None)
        _native_consumed.pop(key, None)
        print('PPF bone overlay: native completion unavailable:', exc)
        return False


def native_pose_completed(scene):
    """Observer invoked after core pose commit; never touches dynamic PC2."""
    from ..core import rigid_pose
    if _frame_publishing:
        return False
    return _consume_native_completion(scene, rigid_pose.pose_completion(scene))


@persistent
def _frame_placement_post(scene, depsgraph=None):
    global _frame_publishing
    if not _registered or _rebuilding or _frame_publishing or depsgraph is None or not _identity_ready():
        _frame_publications.pop(scene.as_pointer(), None)
        return
    from ..core import rigid_pose
    key = scene.as_pointer()
    epoch = _frame_epochs.get(key)
    if epoch is None:
        return
    generation = _generation
    _frame_publishing = True
    try:
        try:
            _dynamic_frame(scene, depsgraph, epoch)
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError) as exc:
            for viewport in tuple(_dynamic_packets):
                if viewport[3] == key:
                    _dynamic_packets.pop(viewport, None)
            print('PPF dynamic overlay frame unavailable:', exc)
        if rigid_pose.has_bone_pose_cache(scene):
            packet = rigid_pose.pose_completion(scene)
            if packet is None:
                return
            _consume_native_completion(scene, packet)
            return
        planned = _frame_plans.get(key)
        if planned is None or planned[0] != generation:
            return
        matrices = {}
        evaluated_arms = {}
        from ..core import bone_author_follow as follow
        from ..core.single_bake_restore import frozen_proxy
        for arm_name, arm_uid, bone_name, proxy_name, proxy_uid, identity, bone_uuid, body_uuid, offset, authority, record in planned[1]:
            arm = scene.objects.get(arm_name)
            proxy = scene.objects.get(proxy_name)
            if (arm is None or proxy is None or frozen_proxy(scene, proxy) or arm.mode == 'EDIT' or arm.session_uid != arm_uid or proxy.session_uid != proxy_uid
                    or not arm.ppf_armature_uuid or proxy.get('ppf_armature_uuid') != arm.ppf_armature_uuid
                    or proxy.as_pointer() != identity or proxy.get('ppf_bone_uuid') != bone_uuid
                    or proxy.get('_solver_uuid') != body_uuid
                    or proxy.get(follow._AUTHOR_RECORD) != record):
                continue
            bone = arm.data.bones.get(bone_name)
            if (bone is None or not bone.ppf_bone_physics.enabled or bone.ppf_bone_physics.proxy != proxy
                    or bone.ppf_bone_physics.bone_uuid != bone_uuid or bone.ppf_bone_physics.body_uuid != body_uuid):
                continue
            if authority == 'baseline':
                baseline_key = (key, arm.as_pointer(), bone_name, identity)
                if baseline_key not in follow._seen or baseline_key in follow._unproven or _matrix_token(follow._seen[baseline_key][2]) != offset:
                    continue
            ev = evaluated_arms.get(arm_uid)
            if ev is None:
                ev = arm.evaluated_get(depsgraph)
                evaluated_arms[arm_uid] = ev
            pose = ev.pose.bones.get(bone_name)
            if pose is None:
                continue
            mat = ev.matrix_world @ pose.matrix @ Matrix(tuple(offset[i:i+4] for i in range(0,16,4)))
            if follow._finite_invertible(mat):
                matrices[identity] = _matrix_token(mat)
        links = {}
        # Cross-link plans are prepared by the stable producer per viewport.
        for view_key, rows in _frame_link_plans.get(key, {}).items():
            segments = []
            for arm_name, arm_uid, arm_uuid, endpoint_a, endpoint_b, anchor_a, anchor_b, selected, color in rows:
                arm = scene.objects.get(arm_name)
                if (arm is None or arm.mode == 'EDIT' or arm.session_uid != arm_uid
                        or not arm_uuid or arm.ppf_armature_uuid != arm_uuid):
                    continue
                valid = True
                for bone_name, bone_uuid, body_uuid, proxy_name, proxy_uid, identity in (endpoint_a, endpoint_b):
                    bone = arm.data.bones.get(bone_name)
                    proxy = scene.objects.get(proxy_name)
                    if (bone is None or proxy is None or proxy.session_uid != proxy_uid
                            or proxy.as_pointer() != identity or not proxy.get('ppf_bone_proxy_owned')
                            or proxy.get('ppf_armature_uuid') != arm_uuid
                            or proxy.get('ppf_bone_uuid') != bone_uuid or proxy.get('_solver_uuid') != body_uuid
                            or not bone.ppf_bone_physics.enabled or bone.ppf_bone_physics.proxy != proxy
                            or bone.ppf_bone_physics.bone_uuid != bone_uuid
                            or bone.ppf_bone_physics.body_uuid != body_uuid):
                        valid = False
                        break
                if not valid:
                    continue
                a_name, b_name = endpoint_a[0], endpoint_b[0]
                ev = evaluated_arms.get(arm_uid)
                if ev is None:
                    ev = arm.evaluated_get(depsgraph)
                    evaluated_arms[arm_uid] = ev
                a, b = ev.pose.bones.get(a_name), ev.pose.bones.get(b_name)
                if a is None or b is None:
                    continue
                from math import isfinite
                point_a = tuple(ev.matrix_world @ a.matrix @ Vector(anchor_a))
                point_b = tuple(ev.matrix_world @ b.matrix @ Vector(anchor_b))
                if all(isfinite(v) for v in (*point_a, *point_b)):
                    segments.append((point_a, point_b, selected, color))
            links[view_key] = tuple(segments)
        if _generation == generation and _frame_epochs.get(key) == epoch:
            _frame_publications[key] = (_generation, epoch, float(scene.frame_current_final), matrices, links)
            _redraw()
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError) as exc:
        _frame_publications.pop(key, None)
        print('PPF bone overlay: frame placement unavailable:', exc)
    finally:
        _frame_publishing = False


def _capture_view_impl(context, generation):
    """Build one immutable viewport snapshot after an actual evaluation barrier."""
    global _geometry_serial, _publication_serial
    if generation != _generation or not _identity_ready():
        return None
    # Undo post only schedules this future event-loop operation. Successful
    # evaluation, rather than elapsed time, qualifies the newly restored IDs.
    evaluation_started = perf_counter()
    context.view_layer.update()
    _cpu_stats['evaluation_ms'].append((perf_counter()-evaluation_started)*1000.)
    if generation != _generation or not _identity_ready():
        return None
    depsgraph = context.evaluated_depsgraph_get()
    settings = context.scene.ppf_bone_display
    options = (bool(settings.faces), bool(settings.edges), bool(settings.in_front), float(settings.opacity))
    key = _viewport_key(context)
    pose_targets = set()
    pose_plan = []
    pose_matrices = _pose_display_matrices(context, depsgraph, targets=pose_targets, plan=pose_plan)
    _frame_known_targets[context.scene.as_pointer()] = frozenset(pose_targets)
    _frame_plans[context.scene.as_pointer()] = (_generation, tuple(pose_plan))
    geometry_cache = _geometry_snapshots.get(key, {})
    staged_geometry = {}
    source_rows = []
    entries = []
    dynamic_plan = []
    fast_unsupported = False
    if settings.enabled and settings.proxy_enabled and (settings.faces or settings.edges):
        from ..models.bone_bindings import effective_preview_color
        for proxy, model, selected, owner in bindings(context):
            if generation != _generation or not _identity_ready():
                return None
            # The shared source-pivot API stays synchronous in this stable
            # producer; it is never called by draw and no source RNA is cached.
            identity = proxy.as_pointer()
            mesh_identity = proxy.data.as_pointer()
            # Frame-sensitive evaluated geometry stays conservative. Plain
            # rigid proxies have no frame/pose/world matrix in their mesh token.
            visible_modifiers = [m for m in proxy.modifiers if m.show_viewport]
            dynamic_fast = (all(m.type == 'MESH_CACHE' for m in visible_modifiers)
                            and bool(visible_modifiers or proxy.data.shape_keys))
            if not dynamic_fast and (visible_modifiers or proxy.data.shape_keys):
                fast_unsupported = True
            dynamic_frame = (float(context.scene.frame_current_final)
                             if not dynamic_fast and (proxy.modifiers or proxy.data.shape_keys) else None)
            token = (getattr(proxy, 'session_uid', 0), proxy.get('_solver_uuid'),
                     proxy.get('ppf_bone_uuid'), mesh_identity,
                     getattr(proxy.data, 'session_uid', 0), proxy.mode,
                     _mesh_revisions.get(mesh_identity, 0),
                     0 if dynamic_fast else _object_revisions.get(identity, 0), dynamic_frame,
                     tuple((modifier.as_pointer(), modifier.type, bool(modifier.show_viewport))
                           for modifier in proxy.modifiers),
                     proxy.data.shape_keys.as_pointer() if proxy.data.shape_keys else 0)
            cached = geometry_cache.get(identity)
            if cached is not None and cached[0] == token:
                positions, edges, triangles = cached[1]
                serial = cached[2]
            else:
                geometry_started = perf_counter()
                positions, edges, triangles = geometry(proxy, depsgraph, _local=True)
                _cpu_stats['local_geometry_ms'].append((perf_counter()-geometry_started)*1000.)
                _geometry_serial += 1
                serial = _geometry_serial
            staged_geometry[identity] = (token, (positions, edges, triangles), serial)
            matrix = pose_matrices.get(identity) or _model_matrix(context, proxy, depsgraph)
            source_rows.append((identity, serial, matrix))
            if positions and (not settings.selected_only or selected):
                color = tuple(float(value) for value in effective_preview_color(owner, model))
                entries.append((positions, edges, triangles, color, bool(selected), (identity, serial), matrix))
                if dynamic_fast:
                    dynamic_plan.append((proxy.name, proxy.session_uid, identity,
                                         proxy.data.session_uid, len(positions), mesh_identity,
                                         _mesh_revisions.get(mesh_identity, 0)))
    if generation != _generation or not _identity_ready():
        return None
    if settings.enabled and settings.proxy_enabled and (settings.faces or settings.edges):
        _geometry_snapshots[key] = staged_geometry
    link_plan = []
    segments = _cross_link_segments(context, plan=link_plan) if settings.enabled and settings.cross_links else ()
    scene_key = context.scene.as_pointer()
    new_dynamic_plan = (_generation, tuple(dynamic_plan))
    plan_changed = _dynamic_plans.get(key) != new_dynamic_plan
    _dynamic_plans[key] = new_dynamic_plan
    if not dynamic_plan:
        _dynamic_packets.pop(key, None)
    # The first producer runs after an evaluation barrier, including when no
    # frame-change callback has fired since opening a saved scene.
    epoch = _frame_epochs.setdefault(scene_key, 0)
    if dynamic_plan:
        try:
            _dynamic_frame(context.scene, depsgraph, epoch, key)
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError) as exc:
            _dynamic_packets.pop(key, None)
            print('PPF dynamic overlay frame unavailable:', exc)
    _frame_link_plans.setdefault(scene_key, {})[key] = tuple(link_plan)
    covered = pose_targets | {row[2] for row in dynamic_plan}
    if (context.mode != 'EDIT_ARMATURE' and not fast_unsupported
            and all(item[5][0] in covered for item in entries)
            and len(segments) == len(link_plan)):
        _fast_views[key] = _generation
    else:
        _fast_views.pop(key, None)
    if epoch is not None and generation == _generation:
        previous_frame = _frame_publications.get(scene_key)
        frame_links = dict(previous_frame[4]) if previous_frame and previous_frame[:2] == (generation, epoch) else {}
        frame_links[key] = segments
        _frame_publications[scene_key] = (generation, epoch, float(context.scene.frame_current_final), pose_matrices, frame_links)
    link_style = (tuple(float(v) for v in settings.cross_link_color), float(settings.cross_link_width),
                  bool(settings.cross_link_in_front))
    previous = _view_publications.get(key)
    source_token = tuple(source_rows)
    source_token = (source_token, segments)
    if previous is None or previous[0] != source_token:
        _publication_serial += 1
        previous = (source_token, _publication_serial, perf_counter())
        _view_publications[key] = previous
    return key, (options, tuple(entries), previous[2], previous[1],
                 tuple(sorted(row[0] for row in source_rows)), segments, link_style)


def _refresh_snapshots(generation):
    """Resolve fresh UI contexts per tick; never cache their RNA wrappers."""
    global _snapshots
    staged = {}
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != 'VIEW_3D':
                continue
            region = next((r for r in area.regions if r.type == 'WINDOW'), None)
            if region is None:
                continue
            with bpy.context.temp_override(window=window, area=area, region=region):
                result = _capture_view(bpy.context, generation)
            if result is not None:
                key, snapshot = result
                staged[key] = snapshot
    if generation == _generation and _identity_ready():
        _snapshots = staged
        for key in tuple(_geometry_snapshots):
            if key not in staged:
                _geometry_snapshots.pop(key, None)
                _view_publications.pop(key, None)
        for key in tuple(_gpu_views):
            if key not in staged:
                _gpu_views.pop(key, None)
        _known_proxy_meshes.clear()
        _known_proxy_meshes.update(
            cached[0][3] for viewport in _geometry_snapshots.values() for cached in viewport.values()
        )


def gpu_cache_metrics(reset=False):
    """Bounded primitive diagnostics for private profiling; no mesh/RNA reads."""
    result = {key: value for key, value in _gpu_stats.items()
              if key not in ('draw_ms', 'upload_ms')}
    for key in ('draw_ms', 'upload_ms'):
        samples = sorted(_gpu_stats[key])
        result[key] = {'n': len(samples),
                       'p95': samples[min(len(samples)-1, int(len(samples)*.95))] if samples else 0.,
                       'max': max(samples, default=0.)}
    for key, values in _cpu_stats.items():
        samples = sorted(values)
        result[key] = {'n': len(samples),
                       'p95': samples[min(len(samples)-1, int(len(samples)*.95))] if samples else 0.,
                       'max': max(samples, default=0.)}
    result['viewports'] = len(_gpu_views)
    now = perf_counter()
    result['displays'] = [
        {'viewport': key, 'revision': view['display_revision'],
         'age_ms': (now-view['display_birth'])*1000. if view['display'] else None,
         'inflight_revision': (view['inflight'][3] if len(view['inflight']) > 3
                               else max((item[5][1] for item in view['inflight'][1]), default=0))
                              if view['inflight'] else None}
        for key, view in _gpu_views.items()]
    if reset:
        _gpu_stats['shaders'] = _gpu_stats['batches'] = 0
        _gpu_stats['draw_ms'].clear()
        _gpu_stats['upload_ms'].clear()
        for samples in _cpu_stats.values():
            samples.clear()
    return result


def _upload_view(key, snapshot):
    """Bound GPU uploads per callback; keep the last complete display meanwhile.

    A single driver call cannot be interrupted. Record its actual duration so a
    slow first allocation cannot be hidden by the soft between-proxy budget.
    """
    view = _gpu_views.get(key)
    if view is None:
        view = {'shader': gpu.shader.from_builtin('UNIFORM_COLOR'),
                'link_shader': gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR'),
                'batches': {}, 'display': None, 'inflight': None,
                'latest': None, 'latest_birth': 0., 'inflight_birth': 0.,
                'display_birth': 0., 'display_revision': 0, 'identities': None,
                'link_key': None, 'link_batches': (), 'dynamic': {}}
        _gpu_views[key] = view
        _gpu_stats['shaders'] += 2
    shader = view['shader']
    now = perf_counter()
    # CPU publication removes a retired owner immediately. Geometry serial
    # changes alone never cancel a valid upload. Lifecycle clears the whole view.
    # Owner membership is published with the completed CPU snapshot. A dirty
    # geometry cache may be empty while this snapshot remains safe to display.
    identities = snapshot[4]
    if view['identities'] != identities:
        view.update(identities=identities, inflight=None, display=None,
                    display_revision=0, display_birth=now)
        view['batches'].clear()
    if (view['display'] is snapshot and view['latest'] is snapshot
            and view['inflight'] is None):
        return view
    if view['latest'] is not snapshot:
        view['latest'] = snapshot
        view['latest_birth'] = snapshot[2] if len(snapshot) > 2 else now
    if view['inflight'] is None:
        view['inflight'] = view['latest']
        view['inflight_birth'] = view['latest_birth']
    target = view['inflight']  # Fixed primitive snapshot until fully uploaded.
    started = perf_counter()
    needed = {item[5] for item in target[1]}
    current_geometry = {(identity, cached[2])
                        for identity, cached in _geometry_snapshots.get(key, {}).items()}
    retained = needed | current_geometry | ({item[5] for item in view['display'][1]}
                         if view['display'] else set())
    for geometry_key in tuple(view['batches']):
        if geometry_key not in retained:
            view['batches'].pop(geometry_key)
    for item in target[1]:
        positions, edges, triangles, _color, _selected, geometry_key = item[:6]
        if geometry_key in view['batches']:
            continue
        upload_start = perf_counter()
        face_batch = (batch_for_shader(shader, 'TRIS', {'pos': positions}, indices=triangles)
                      if triangles else None)
        # Build fallback wires once, independent of selection/style changes.
        draw_edges = edges or tuple(sorted({tuple(sorted((tri[i], tri[(i+1) % 3])))
                                            for tri in triangles for i in range(3)}))
        edge_batch = (batch_for_shader(shader, 'LINES', {'pos': positions}, indices=draw_edges)
                      if draw_edges else None)
        view['batches'][geometry_key] = (face_batch, edge_batch, bool(edges))
        _gpu_stats['batches'] += int(face_batch is not None) + int(edge_batch is not None)
        _gpu_stats['upload_ms'].append((perf_counter()-upload_start)*1000.)
        if perf_counter()-started >= _GPU_UPLOAD_BUDGET:
            break
    if needed.issubset(view['batches']):
        revision = (target[3] if len(target) > 3
                    else max((item[5][1] for item in target[1]), default=0))
        # Style-only publication does not pretend old geometry became younger.
        if view['display'] is None or revision != view['display_revision']:
            view['display_birth'] = view['inflight_birth']
        view['display_revision'] = revision
        view['display'] = target
        view['inflight'] = None
        latest_needed = {item[5] for item in view['latest'][1]}
        for geometry_key in tuple(view['batches']):
            if geometry_key not in needed | current_geometry | latest_needed:
                view['batches'].pop(geometry_key)
        if view['latest'] is not target:
            bpy.context.region.tag_redraw()
    else:
        bpy.context.region.tag_redraw()
    return view


def _dynamic_vertex_buffer(fmt, positions):
    vertex = gpu.types.GPUVertBuf(format=fmt, len=len(positions))
    vertex.attr_fill(id='pos', data=positions)
    return vertex


def _dynamic_packet_row(packet, identity):
    row = packet[3].get(identity)
    if row is None or _mesh_revisions.get(row[2], 0) != row[3]:
        return None
    return row


def _dynamic_display_row(packet, viewport, geometry_key):
    identity, serial = geometry_key
    row = _dynamic_packet_row(packet, identity)
    cached = _geometry_snapshots.get(viewport, {}).get(identity)
    if (row is None or cached is None or cached[2] != serial
            or cached[0][3] != row[2] or cached[0][6] != row[3]):
        return None
    return row


def draw():
    # Completed primitive snapshots only; no retired Mesh/Object/owner reads.
    if _rebuilding or _producer_running or not _registered or not _identity_ready():
        return
    context = bpy.context
    if (not context.area or context.area.type != 'VIEW_3D'
            or not context.region or context.region.type != 'WINDOW'
            or not context.space_data.overlay.show_overlays):
        return
    key = _viewport_key(context)
    snapshot = _snapshots.get(key)
    if snapshot is None:
        return
    started = perf_counter()
    blend = gpu.state.blend_get()
    depth = gpu.state.depth_test_get()
    mask = gpu.state.depth_mask_get()
    width = gpu.state.line_width_get()
    try:
        view = _upload_view(key, snapshot)
        if view['display'] is None:
            return
        (faces, edges_enabled, in_front, opacity), items = view['display'][:2]
        shader = view['shader']
        segments, (link_color, link_width, link_in_front) = view['display'][5:7]
        scene_key = context.scene.as_pointer()
        epoch = _frame_epochs.get(scene_key)
        publication = _frame_publications.get(scene_key)
        current = (publication is not None and publication[0] == _generation and publication[1] == epoch)
        packet = _dynamic_packets.get(key)
        dynamic_current = (packet is not None and packet[0] == _generation and packet[1] == epoch)
        dynamic_ids = {r[2] for r in _dynamic_plans.get(key, (None, ()))[1]}
        if current:
            segments = publication[4].get(key, segments)
        elif epoch is not None and scene_key in _frame_known_targets:
            segments = ()

        link_shader = view['link_shader']
        if view['link_key'] != segments:
            groups = {}
            for a, b, selected, color in segments:
                # Keep authored color even while editing selected connections.
                # All keys/positions are copied scalars, never live RNA.
                groups.setdefault((selected, color), []).extend((a, b))
            view['link_batches'] = tuple(
                (selected, color, batch_for_shader(link_shader, 'LINES', {'pos': pts}))
                for (selected, color), pts in groups.items())
            view['link_key'] = segments
        # Build every dynamic buffer before drawing any dynamic member, so
        # allocation/upload failure cannot leave a partially rendered frame.
        dynamic_ready = {}
        try:
            if dynamic_current:
                for entry in items:
                    positions0, edges0, triangles0, _color, _selected, geometry_key0, _matrix = entry
                    identity0 = geometry_key0[0]
                    if identity0 not in dynamic_ids or identity0 not in packet[3]:
                        continue
                    row = _dynamic_display_row(packet, key, geometry_key0)
                    if row is None:
                        continue
                    positions, _current_matrix, mesh_ptr, mesh_revision = row
                    topology = (len(positions), edges0, triangles0)
                    resource = view['dynamic'].get(identity0)
                    if resource is None or resource[0] != topology:
                        fmt = gpu.types.GPUVertFormat()
                        fmt.attr_add(id='pos', comp_type='F32', len=3, fetch_mode='FLOAT')
                        faces_index = gpu.types.GPUIndexBuf(type='TRIS', seq=triangles0) if triangles0 else None
                        edge_index = gpu.types.GPUIndexBuf(type='LINES', seq=edges0) if edges0 else None
                        resource = [topology, fmt, faces_index, edge_index, None, None, None, None]
                        view['dynamic'][identity0] = resource
                    stamp = (packet[0], packet[1], packet[4])
                    if resource[4] != stamp:
                        vertex = _dynamic_vertex_buffer(resource[1], positions)
                        resource[5] = vertex
                        resource[6] = gpu.types.GPUBatch(type='TRIS', buf=vertex, elem=resource[2]) if resource[2] else None
                        resource[7] = gpu.types.GPUBatch(type='LINES', buf=vertex, elem=resource[3]) if resource[3] else None
                        resource[4] = stamp
                    dynamic_ready[identity0] = (resource[6], resource[7], bool(edges0))
        except (RuntimeError, TypeError, ValueError) as exc:
            print('PPF dynamic overlay GPU upload unavailable:', exc)
            return
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE' if in_front else 'LESS_EQUAL')
        gpu.state.depth_mask_set(False)
        for item in sorted(items, key=lambda item: item[4]):
            _positions, _edges, _triangles, color, selected, geometry_key, matrix = item
            identity = geometry_key[0]
            dynamic_batches = dynamic_ready.get(identity)
            if identity in dynamic_ids:
                if dynamic_batches is None:
                    continue
                matrix = packet[3][identity][1]
            if epoch is not None and identity in _frame_known_targets.get(scene_key, ()):
                if not current or identity not in publication[3]:
                    continue
                cached = _geometry_snapshots.get(key, {}).get(identity)
                if cached is None or cached[2] != geometry_key[1]:
                    continue
                token = cached[0]
                if (token[6] != _mesh_revisions.get(token[3], 0)
                        or token[7] != _object_revisions.get(identity, 0)):
                    continue
                matrix = publication[3][identity]
            with gpu.matrix.push_pop():
                gpu.matrix.multiply_matrix(Matrix(tuple(matrix[i:i+4] for i in range(0, 16, 4))))
                face_batch, edge_batch, source_edges = dynamic_batches or view['batches'][geometry_key]
                shader.bind()
                if faces and face_batch is not None:
                    shader.uniform_float('color', (*color, opacity))
                    face_batch.draw(shader)
                if edge_batch is not None and ((edges_enabled and source_edges) or (selected and faces)):
                    gpu.state.line_width_set(2.0 if selected else 1.0)
                    edge_color = (1.0, 0.65, 0.12) if selected else color
                    shader.uniform_float('color', (*edge_color, 0.95 if selected else 0.65))
                    edge_batch.draw(shader)

        # Anchors often lie inside opaque proxy/character surfaces. Only links
        # get this independent depth policy; proxy faces/wires retain theirs.
        gpu.state.depth_test_set('NONE' if link_in_front or in_front else 'LESS_EQUAL')
        for selected, color, batch in view['link_batches']:
            link_shader.bind()
            link_shader.uniform_float('viewportSize', gpu.state.viewport_get()[2:])
            if selected:
                link_shader.uniform_float('lineWidth', link_width + min(1.5, link_width * 0.5))
                link_shader.uniform_float('color', (1.0, 0.85, 0.2, 1.0))
                batch.draw(link_shader)
            link_shader.uniform_float('lineWidth', link_width)
            link_shader.uniform_float('color', color)
            batch.draw(link_shader)

    finally:
        gpu.state.line_width_set(width)
        gpu.state.depth_mask_set(mask)
        gpu.state.depth_test_set(depth)
        gpu.state.blend_set(blend)
        _gpu_stats['draw_ms'].append((perf_counter()-started)*1000.)


def _invalidate_snapshots(*, geometry=True):
    _dynamic_plans.clear()
    _dynamic_packets.clear()
    _fast_views.clear()
    _pose_offset_diagnostics.clear()
    _frame_epochs.clear()
    _native_consumed.clear()
    _frame_publications.clear()
    _frame_known_targets.clear()
    _frame_plans.clear()
    _frame_link_plans.clear()
    global _generation, _snapshots, _snapshot_dirty, _ui_signature
    _generation += 1
    _snapshots = {}
    if geometry:
        _geometry_snapshots.clear()
        _known_proxy_meshes.clear()
        _gpu_views.clear()
        _mesh_revisions.clear()
        _object_revisions.clear()
        _view_publications.clear()
    _snapshot_dirty = True
    _ui_signature = None


_layer_signatures = {}
_layer_pending = set()
_full_layer_sync_pending = True
_layer_msgbus_owner = object()


def _layer_window_changed():
    global _full_layer_sync_pending
    if _registered:
        _full_layer_sync_pending = True
        _ensure_producer()


def _subscribe_layer_window():
    bpy.msgbus.clear_by_owner(_layer_msgbus_owner)
    bpy.msgbus.subscribe_rna(key=(bpy.types.Window, 'view_layer'),
                             owner=_layer_msgbus_owner, args=(),
                             notify=_layer_window_changed)


@persistent
def _depsgraph_changed(_scene, depsgraph):
    """Primitive dirtiness only; mesh reads remain in the stable producer."""
    global _snapshot_dirty
    if not _identity_ready():
        _geometry_snapshots.clear()
        _known_proxy_meshes.clear()
        _gpu_views.clear()
        return
    scene_key = _scene.as_pointer()
    signature = tuple(layer.as_pointer() for layer in _scene.view_layers)
    if _layer_signatures.get(scene_key) != signature:
        _layer_signatures[scene_key] = signature
        _layer_pending.add(scene_key)
        _snapshot_dirty = True
        _ensure_producer()
    for update in depsgraph.updates:
        item = update.id.original
        # Preserve copied local geometry for unrelated proxies. Epochs are not
        # consumed, so every viewport independently observes the same update.
        if isinstance(item, bpy.types.Mesh):
            identity = item.as_pointer()
            if identity in _known_proxy_meshes:
                _mesh_revisions[identity] = _mesh_revisions.get(identity, 0) + 1
                _snapshot_dirty = True
        elif isinstance(item, bpy.types.Object):
            if item.type == 'MESH' and item.get('ppf_bone_proxy_owned'):
                # The leased Edit mesh is rendered by Blender, and bindings()
                # excludes it. Its vertices cannot change any overlay row.
                if item.mode == 'EDIT' and item.get('ppf_bone_proxy_visibility_editing'):
                    continue
                if update.is_updated_geometry:
                    identity = item.as_pointer()
                    _object_revisions[identity] = _object_revisions.get(identity, 0) + 1
                _snapshot_dirty |= update.is_updated_geometry or update.is_updated_transform
            elif item.type == 'ARMATURE':
                _snapshot_dirty |= update.is_updated_geometry or update.is_updated_transform
        elif isinstance(item, bpy.types.Collection) or (
                isinstance(item, bpy.types.Armature) and update.is_updated_geometry):
            _snapshot_dirty = True


def _display_interval():
    rates = [getattr(scene.ppf_bone_display, 'refresh_fps', 30)
             for scene in bpy.data.scenes
             if getattr(scene, 'ppf_bone_display', None) and scene.ppf_bone_display.enabled]
    return 1. / max(rates) if rates else _REFRESH_INTERVAL


def _display_signature():
    """Cheap UI state that can change without a depsgraph geometry update."""
    rows = []
    scene_rows = {}
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != 'VIEW_3D':
                continue
            region = next((r for r in area.regions if r.type == 'WINDOW'), None)
            if region is None:
                continue
            with bpy.context.temp_override(window=window, area=area, region=region):
                context = bpy.context
                obj = context.object
                scene_key = context.scene.as_pointer()
                if scene_key not in scene_rows:
                    # Preview color callbacks request redraw only, with no ID
                    # geometry tag. Read scalars, never mesh or evaluated data.
                    scene_rows[scene_key] = tuple(
                        (arm.as_pointer(), bone.name, bool(bone.hide), bool(settings.enabled),
                         settings.physical_model, bool(settings.preview_color_custom),
                         tuple(settings.preview_color), bool(settings.collision_filter_enabled),
                         tuple(settings.collision_group_layers), bool(settings.proxy.hide_viewport))
                        for arm in context.scene.objects if arm.type == 'ARMATURE' and arm.mode != 'EDIT'
                        for bone in arm.data.bones for settings in (bone.ppf_bone_physics,)
                        if settings.proxy is not None)
                    scene_rows[scene_key] += tuple(
                        (arm.as_pointer(), bone.name, bool(bone.hide),
                         tuple(bone.head), tuple(bone.tail), float(bone.roll),
                         bool(bone.select or bone.select_head or bone.select_tail))
                        for arm in context.scene.objects if arm.type == 'ARMATURE' and arm.mode == 'EDIT'
                        for bone in arm.data.edit_bones)
                    link_rows = tuple((arm.as_pointer(), tuple((r.link_uuid, r.enabled, r.selected,
                        r.bone_a_uuid, r.bone_b_uuid, tuple(r.anchor_a_bone), tuple(r.anchor_b_bone),
                        r.tension_enabled, r.compression_enabled,
                        r.preview_color_custom, tuple(r.preview_color)) for r in arm.ppf_cross_links))
                        for arm in context.scene.objects if arm.type == 'ARMATURE')
                    scene_rows[scene_key] = (scene_rows[scene_key], link_rows)
                bones = (getattr(context, 'selected_pose_bones', None) or
                         getattr(context, 'selected_bones', None) or ())
                viewport = _viewport_key(context)
                frame = None if _fast_views.get(viewport) == _generation else float(context.scene.frame_current_final)
                arm_visibility = tuple(
                    (arm.as_pointer(), arm.visible_get(view_layer=context.view_layer, viewport=context.space_data))
                    for arm in context.view_layer.objects if arm.type == 'ARMATURE'
                )
                layer_visibility = []
                layers = [context.view_layer.layer_collection]
                while layers:
                    layer = layers.pop()
                    layer_visibility.append((layer.as_pointer(), layer.exclude, layer.hide_viewport,
                                             layer.collection.hide_viewport))
                    layers.extend(layer.children)
                rows.append((viewport, context.mode, frame,
                             obj.as_pointer() if obj else 0,
                             tuple(item.as_pointer() for item in context.selected_objects),
                             tuple(bone.name for bone in bones),
                             bool(context.space_data.local_view), arm_visibility, tuple(layer_visibility),
                             scene_rows[scene_key]))
    return tuple(rows)


@persistent
def _identity_pre(*_args):
    global _rebuilding
    _rebuilding = True
    _layer_signatures.clear()
    _layer_pending.clear()
    _invalidate_snapshots()
    _cancel_startup()


@persistent
def _identity_post(*_args):
    global _rebuilding, _full_layer_sync_pending
    _rebuilding = False
    if _registered:
        _full_layer_sync_pending = True
        _ensure_producer()


@persistent
def _load_pre(*_args):
    _identity_pre()
    _remove_handler()


@persistent
def _load_post(*_args):
    global _visibility_pending
    _subscribe_layer_window()
    _visibility_pending = True
    _identity_post()  # No Scene/Object/mesh reads before the future stable producer.


def _cancel_startup():
    if bpy.app.timers.is_registered(_startup_sync):
        bpy.app.timers.unregister(_startup_sync)


def _ensure_producer():
    if not bpy.app.timers.is_registered(_startup_sync):
        bpy.app.timers.register(_startup_sync, first_interval=0.0)


def _sync_visibility():
    from ..core.bone_proxy_visibility import sync_scene
    for scene in bpy.data.scenes:
        sync_scene(scene)


def _startup_sync():
    global _producer_running, _visibility_pending, _snapshot_dirty, _ui_signature, _full_layer_sync_pending
    if not _registered:
        return None
    if _producer_running or not _identity_ready():
        return _REFRESH_INTERVAL
    if getattr(bpy.data, 'scenes', None) is None:
        return _REFRESH_INTERVAL
    generation = _generation
    producer_started = perf_counter()
    _producer_running = True
    try:
        layer = getattr(bpy.context, 'view_layer', None)
        if layer is None:
            return _REFRESH_INTERVAL
        if _visibility_pending:
            layer.update()
            _sync_visibility()
            if generation != _generation or not _identity_ready():
                return _REFRESH_INTERVAL
            _visibility_pending = False
        if _layer_pending or _full_layer_sync_pending:
            from ..core.bone_proxy_visibility import protect_scene_layers
            scenes = tuple(bpy.data.scenes)
            live = {scene.as_pointer() for scene in scenes}
            if _full_layer_sync_pending:
                _layer_pending.update(live)
                _full_layer_sync_pending = False
            _layer_pending.intersection_update(live)
            for scene in scenes:
                key = scene.as_pointer()
                if key in _layer_pending:
                    protect_scene_layers(scene)
                    _layer_pending.discard(key)
        _sync_handler()
        if _handler is None:
            _snapshots.clear()
            _gpu_views.clear()
            return None  # Disabled displays sleep until their property callback wakes us.
        signature = _display_signature()
        if _snapshot_dirty or signature != _ui_signature:
            # Clear before sampling so a new event during evaluation survives.
            _snapshot_dirty = False
            _refresh_snapshots(generation)
            _ui_signature = signature
            _redraw()
        elif any(view.get('inflight') is not None for view in _gpu_views.values()):
            # A redraw requested from inside a draw callback can be coalesced
            # by Blender. Keep bounded uploads moving without recapturing RNA.
            _redraw_pending_views()
    except Exception as exc:
        _snapshots.clear()
        _snapshot_dirty = True
        message = str(exc)
        if message not in _reported_errors:
            _reported_errors.add(message)
            import traceback
            print('[PPF bone overlay] Snapshot rebuild failed:', message)
            traceback.print_exc()
    finally:
        _producer_running = False
        _cpu_stats['producer_ms'].append((perf_counter()-producer_started)*1000.)
    return _display_interval()


def _lifecycle_handlers():
    return ((bpy.app.handlers.frame_change_pre, _frame_placement_pre),
            (bpy.app.handlers.frame_change_post, _frame_placement_post),
            (bpy.app.handlers.depsgraph_update_post, _depsgraph_changed),
            (bpy.app.handlers.load_pre, _load_pre),
            (bpy.app.handlers.load_post, _load_post),
            (bpy.app.handlers.undo_pre, _identity_pre),
            (bpy.app.handlers.undo_post, _identity_post),
            (bpy.app.handlers.redo_pre, _identity_pre),
            (bpy.app.handlers.redo_post, _identity_post))


def register():
    global _registered, _rebuilding, _visibility_pending
    if _registered:
        return
    _ppf_i18n.register_class(PPF_PG_BoneDisplay)
    bpy.types.Scene.ppf_bone_display = PointerProperty(type=PPF_PG_BoneDisplay)
    for collection, callback in _lifecycle_handlers():
        if callback not in collection:
            collection.append(callback)
    _registered = True
    _subscribe_layer_window()
    _rebuilding = False
    _visibility_pending = True
    _invalidate_snapshots()
    _reported_errors.clear()
    _ensure_producer()


def unregister():
    _pose_offset_diagnostics.clear()
    global _registered, _rebuilding
    _registered = False
    bpy.msgbus.clear_by_owner(_layer_msgbus_owner)
    _rebuilding = True
    _invalidate_snapshots()
    _cancel_startup()
    _remove_handler()
    for collection, callback in _lifecycle_handlers():
        if callback in collection:
            collection.remove(callback)
    del bpy.types.Scene.ppf_bone_display
    bpy.utils.unregister_class(PPF_PG_BoneDisplay)
    _redraw()


