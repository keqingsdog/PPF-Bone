"""Active joint axes, sampled outside the viewport draw callback."""
import math
import json

import bpy
import gpu
from bpy.app.handlers import persistent
from mathutils import Euler, Matrix, Quaternion
from gpu_extras.batch import batch_for_shader

from ..models.bone_bindings import (active_armature, active_bone, bone_world_matrix,
                                    has_physical_joint_parent, rigid_matrix, _frame)

_handler = None
_frames = {}
_pmx_cache = {}
_proxy_index = {}
_legacy_offsets = {}
_gpu_axes = {}


@persistent
def _reset_for_load(_dummy):
    _frames.clear()
    _pmx_cache.clear()
    _proxy_index.clear()
    _legacy_offsets.clear()
    _gpu_axes.clear()


@persistent
def _invalidate_armature_index(_scene, depsgraph):
    # Bone membership, names and custom body identities belong to Armature data.
    # Proxy transforms update objects and do not require an index rebuild.
    if (_pmx_cache or _proxy_index) and any(isinstance(update.id, bpy.types.Armature)
                          for update in depsgraph.updates):
        _pmx_cache.clear()
        _proxy_index.clear()


def _selected_bones(context, arm):
    """Resolve selection once per sample, never during GPU drawing."""
    result = {p.name: p.bone for p in (context.selected_pose_bones or ())
              if p.id_data == arm}
    objects = context.selected_objects or ()
    proxies = [obj for obj in objects if obj.get('ppf_bone_uuid')]
    if proxies:
        key = arm.as_pointer()
        index = _proxy_index.get(key)
        if index is None:
            index = {b.ppf_bone_physics.bone_uuid: b.name for b in arm.data.bones
                     if b.ppf_bone_physics.bone_uuid}
            _proxy_index[key] = index
        for obj in proxies:
            bone = arm.data.bones.get(index.get(obj.get('ppf_bone_uuid'), ''))
            if bone and bone.ppf_bone_physics.proxy == obj:
                result[bone.name] = bone
    return tuple(result.values())


def _pose_world(context, arm, bone):
    evaluated = arm.evaluated_get(context.evaluated_depsgraph_get())
    return evaluated.matrix_world @ evaluated.pose.bones[bone.name].matrix


def _proxy_world(context, arm, bone, owner_snapshot=None):
    from ..core.bone_author_follow import read_author_offset
    proxy = bone.ppf_bone_physics.proxy
    deps = context.evaluated_depsgraph_get()
    if proxy.animation_data or proxy.constraints:
        return rigid_matrix(proxy.evaluated_get(deps).matrix_world, proxy.name)
    offset, authority = read_author_offset(context.scene, arm, bone, owner_snapshot=owner_snapshot)
    if offset is None:
        return None
    key = (context.scene.as_pointer(), arm.as_pointer(), bone.name, proxy.as_pointer())
    if authority == 'legacy_current_pose':
        offset = _legacy_offsets.setdefault(key, offset.copy())
    else:
        _legacy_offsets.pop(key, None)
    return rigid_matrix(_pose_world(context, arm, bone) @ offset, proxy.name)


def native_frame(arm, bone, context=None, owner_snapshot=None):
    context = context or bpy.context
    s = bone.ppf_bone_physics
    if (not s.enabled or not s.proxy or s.joint_control_mode != 'NATIVE'
            or not has_physical_joint_parent(bone, arm=arm)):
        return None
    if s.use_auto_frames:
        evaluated = arm.evaluated_get(context.evaluated_depsgraph_get())
        return (bone_world_matrix(evaluated, evaluated.data.bones[bone.name]) @ Matrix.Rotation(math.pi / 2, 4, 'Z')
                @ Euler(tuple(s.joint_axis_euler)).to_matrix().to_4x4())
    parent = bone.parent
    ps = parent.ppf_bone_physics
    if ps.enabled:
        if not ps.proxy:
            return None
        world = _proxy_world(context, arm, parent, owner_snapshot)
        if world is None:
            return None
    else:
        evaluated = arm.evaluated_get(context.evaluated_depsgraph_get())
        world = bone_world_matrix(evaluated, evaluated.data.bones[parent.name])
    frame = _frame(s, 'parent')
    _frame(s, 'child')
    local = Quaternion(frame['quaternion_wxyz']).to_matrix().to_4x4()
    local.translation = frame['translation_m']
    return world @ local


def pmx_frame(context, arm, bone, owner_snapshot=None):
    from ..core.encoder.pmx_joints import _mmd_endpoint
    s = bone.ppf_bone_physics
    if not _mmd_endpoint(bone) or s.physical_model != 'RIGID':
        return None
    raw = arm.get('ppf_mmd_joints_v1', '[]')
    key = (arm.as_pointer(), bone.name, s.body_uuid, raw)
    cached = _pmx_cache.get(key)
    if cached is None:
        records = json.loads(raw)
        names = {b.ppf_bone_physics.body_uuid: b.name for b in arm.data.bones
                 if b.ppf_bone_physics.body_uuid}
        cached = []
        for row in records:
            a, b = row['body_a_uuid'], row['body_b_uuid']
            if a == b or s.body_uuid not in (a, b):
                continue
            other = names.get(b if a == s.body_uuid else a)
            if other and names.get(a):
                cached.append((other, names[a], row))
        if len(_pmx_cache) >= 16:
            _pmx_cache.clear()
        _pmx_cache[key] = cached
    choices = []
    for other_name, a_name, row in cached:
        other, side_a = arm.data.bones.get(other_name), arm.data.bones.get(a_name)
        if (other is None or side_a is None
                or side_a.ppf_bone_physics.body_uuid != row['body_a_uuid']
                or other.ppf_bone_physics.body_uuid not in (row['body_a_uuid'], row['body_b_uuid'])):
            _pmx_cache.pop(key, None)
            continue
        if not _mmd_endpoint(other) or other.ppf_bone_physics.physical_model != 'RIGID':
            continue
        if other == bone.parent:
            child, order = bone, 0
        elif other.parent == bone:
            child, order = other, 1
        else:
            child, order = None, 2
        if child is not None and not child.ppf_bone_physics.pmx_parent_joint_enabled:
            continue
        choices.append((order, other.name, row['joint_uuid'], side_a, row))
    if not choices:
        return None
    choices.sort(key=lambda item: item[:3])
    selected = context.window_manager.ppf_display_pmx_joint
    choice = next((item for item in choices if item[2] == selected), choices[0])
    side_a, record = choice[3:]
    frame = record['frame_a']
    local = Quaternion(frame['quaternion_wxyz']).to_matrix().to_4x4()
    local.translation = frame['translation_m']
    world = _proxy_world(context, arm, side_a, owner_snapshot)
    return world @ local if world is not None else None


def _sample():
    if _handler is None:
        return None
    manager = bpy.context.window_manager
    fresh = {}
    for window in manager.windows:
        for area in window.screen.areas:
            if area.type != 'VIEW_3D':
                continue
            region = next((r for r in area.regions if r.type == 'WINDOW'), None)
            if region is None:
                continue
            key = (window.as_pointer(), area.as_pointer())
            try:
                with bpy.context.temp_override(window=window, area=area, region=region):
                    context = bpy.context
                    if not context.scene.ppf_bone_display.show_joint_axes:
                        continue
                    arm = active_armature(context)
                    if arm is None:
                        continue
                    axes = [[], [], []]
                    selected = _selected_bones(context, arm)
                    if not selected:
                        continue
                    from collections import Counter, defaultdict
                    from ..models.bone_bindings import armature_data_owner_counts
                    arms_by_uuid = defaultdict(list)
                    for obj in bpy.data.objects:
                        if obj.type == 'ARMATURE':
                            arms_by_uuid[obj.ppf_armature_uuid].append(obj)
                    counts = Counter(b.ppf_bone_physics.proxy.as_pointer() for b in arm.data.bones
                                     if b.ppf_bone_physics.proxy)
                    snapshot = {'data_owner_counts': armature_data_owner_counts(),
                                'proxy_counts': counts, 'arms_by_uuid': arms_by_uuid}
                    for bone in selected:
                        try:
                            frame = (pmx_frame(context, arm, bone, snapshot) if bone.ppf_bone_physics.joint_control_mode == 'MMD'
                                     else native_frame(arm, bone, context, snapshot))
                            if frame is None:
                                continue
                            size = max(0.05, min(1.0, bone.length * 0.5))
                            origin = frame.translation
                            for i in range(3):
                                axes[i].extend((tuple(origin), tuple(origin + frame.to_3x3().col[i] * size)))
                        except (ValueError, RuntimeError, ReferenceError, AttributeError, KeyError, TypeError):
                            continue
                    if axes[0]:
                        fresh[key] = tuple(tuple(points) for points in axes)
            except (ValueError, RuntimeError, ReferenceError, AttributeError, KeyError, TypeError):
                continue
            if fresh.get(key) != _frames.get(key):
                region.tag_redraw()
    for window in manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D' and (window.as_pointer(), area.as_pointer()) in _frames and (window.as_pointer(), area.as_pointer()) not in fresh:
                area.tag_redraw()
    _frames.clear()
    _frames.update(fresh)
    return 0.1


def _draw():
    context = bpy.context
    if not context.space_data.overlay.show_overlays:
        return
    key = (context.window.as_pointer(), context.area.as_pointer())
    axes = _frames.get(key)
    if axes is None:
        return
    cached = _gpu_axes.get(key)
    if cached is None or cached[0] != axes:
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        cached = (axes, shader, tuple(batch_for_shader(shader, 'LINES', {'pos': points}) for points in axes))
        _gpu_axes[key] = cached
    shader, batches = cached[1:]
    depth, width = gpu.state.depth_test_get(), gpu.state.line_width_get()
    try:
        gpu.state.depth_test_set('NONE')
        gpu.state.line_width_set(3.0)
        for batch, color in zip(batches, ((1, .15, .15, 1), (.2, 1, .2, 1), (.2, .45, 1, 1))):
            shader.bind()
            shader.uniform_float('color', color)
            batch.draw(shader)
    finally:
        gpu.state.depth_test_set(depth)
        gpu.state.line_width_set(width)


def register():
    global _handler
    if _handler is None:
        _handler = bpy.types.SpaceView3D.draw_handler_add(_draw, (), 'WINDOW', 'POST_VIEW')
    if not bpy.app.timers.is_registered(_sample):
        bpy.app.timers.register(_sample, first_interval=0.0, persistent=True)
    if _invalidate_armature_index not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_invalidate_armature_index)
    for handlers in (bpy.app.handlers.load_pre, bpy.app.handlers.undo_pre, bpy.app.handlers.redo_pre):
        if _reset_for_load not in handlers:
            handlers.append(_reset_for_load)


def unregister():
    global _handler
    if bpy.app.timers.is_registered(_sample):
        bpy.app.timers.unregister(_sample)
    if _invalidate_armature_index in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_invalidate_armature_index)
    for handlers in (bpy.app.handlers.load_pre, bpy.app.handlers.undo_pre, bpy.app.handlers.redo_pre):
        if _reset_for_load in handlers:
            handlers.remove(_reset_for_load)
    if _handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handler, 'WINDOW')
        _handler = None
    _frames.clear()
    _pmx_cache.clear()
    _proxy_index.clear()
    _legacy_offsets.clear()
    _gpu_axes.clear()
