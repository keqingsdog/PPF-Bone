"""Draw-only proxy scale override. Owns no Blender IDs or scene data."""
import math
import bpy
from mathutils import Matrix, Vector

# Only the modal operation owning the lease can write this transient packet.
_packet = None
_handoffs = {}  # View -> stale immutable GPU geometry key -> display matrix.
_MAX_FLOAT32 = 3.3e38


def _redraw_scene(scene_id):
    for window in bpy.context.window_manager.windows:
        if window.scene.as_pointer() != scene_id:
            continue
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def preview_basis(proxies, shapes):
    """Conservative opt-in for entire batches of owned, baked plain meshes."""
    if len(proxies) != len(shapes) or not proxies:
        return False
    return all(shape in {'CAPSULE', 'BOX', 'SPHERE', 'CUSTOM'}
               and proxy.data.shape_keys is None and not proxy.modifiers
               and not proxy.constraints and not proxy.animation_data
               for proxy,shape in zip(proxies,shapes))


def basis_pair(matrix):
    """Freeze the exact world-space transform/inverse used by stock scale."""
    if not all(math.isfinite(x) for row in matrix for x in row):
        raise ValueError('代理世界矩阵无效')
    try:
        return matrix.copy(),matrix.inverted()
    except ValueError as exc:
        raise ValueError('代理世界矩阵不可逆') from exc


def local_transform(factors, basis=None):
    if len(factors)!=3 or any(not math.isfinite(x) or x<=0 for x in factors):
        raise ValueError('缩放比例必须是大于零的有限数值')
    stretch = Matrix.Diagonal(Vector(factors))
    if basis is not None:
        world, inverse = basis
        stretch = inverse @ stretch @ world
    if not all(math.isfinite(x) for row in stretch for x in row):
        raise ValueError('缩放后代理坐标超出有限范围')
    return stretch


def publish(token, view, targets, factors, world):
    global _packet
    transforms={}
    local_matrix = None if world else local_transform(factors)
    for identity, bounds, basis in targets:
        matrix = local_transform(factors,basis) if world else local_matrix
        if any(sum(abs(x)*y for x,y in zip(row,bounds)) > _MAX_FLOAT32 for row in matrix):
            raise ValueError('缩放后代理坐标超出有限范围')
        transforms[identity]=matrix.to_4x4()
    _packet=(token,view[3],transforms)
    _redraw_scene(view[3])


def matrices(view):
    packet=_packet
    return packet[2] if packet is not None and packet[1]==view[3] else None


def commit_handoff(token, view, drawable_keys):
    active = _packet
    if active is None or active[0] is not token or active[1] != view[3]:
        return
    previous = _handoffs.get(view, {})
    staged = {}
    for key in drawable_keys:
        older = previous.get(key)
        change = active[2].get(key[0])
        if change is not None:
            staged[key] = change if older is None else change @ older
        elif older is not None:
            staged[key] = older
    if staged:
        _handoffs[view] = staged
    else:
        _handoffs.pop(view, None)


def handoff_matrices(view):
    return _handoffs.get(view)


def limit_handoff(view, drawable_keys):
    pending = _handoffs.get(view)
    if pending is not None:
        for key in pending.keys() - drawable_keys:
            del pending[key]
        if not pending:
            _handoffs.pop(view, None)


def forget_view(view):
    _handoffs.pop(view, None)


def clear_handoffs():
    _handoffs.clear()


def clear(token=None):
    global _packet
    if token is None:
        clear_handoffs()
    if token is None or (_packet is not None and _packet[0] is token):
        scene_id = _packet[1] if _packet is not None else None
        _packet=None
        if scene_id is not None:
            _redraw_scene(scene_id)
