"""Conservative fixed-mesh capsule bounds for read-only clearance planning.

This does not identify generated geometry or edit any existing proxy. A convex
capsule containing every vertex also contains every polygon of that mesh.
"""
import math

from .new_binding_defaults import capsule_segment_distance


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _unit(value):
    length = math.sqrt(_dot(value, value))
    if not math.isfinite(length) or length <= 1.e-12:
        raise ValueError('Cannot bound a proxy with a degenerate longitudinal axis')
    return tuple(x / length for x in value)


def enclosing_capsule(points, axis, origin=(0., 0., 0.)):
    """Bound finite world points without topology, vertex-order or LOD guesses."""
    points = [tuple(float(x) for x in point) for point in points]
    if not points or len(points) > 20002 or any(len(p) != 3 or not all(math.isfinite(x) for x in p) for p in points):
        raise ValueError('Existing proxy capsule bound needs 1..20002 finite vertices')
    if len(origin) != 3 or not all(math.isfinite(x) for x in origin):
        raise ValueError('Existing proxy capsule bound has invalid origin')
    direction = _unit(tuple(axis))
    seed = min(((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), key=lambda v: abs(_dot(v, direction)))
    y = _unit(_cross(direction, seed))
    z = _cross(direction, y)
    local = []
    for point in points:
        delta = tuple(point[i] - origin[i] for i in range(3))
        local.append((_dot(delta, direction), _dot(delta, y), _dot(delta, z)))
    center_y = (min(p[1] for p in local) + max(p[1] for p in local)) / 2.
    center_z = (min(p[2] for p in local) + max(p[2] for p in local)) / 2.
    radial_squared = [(p[1]-center_y)**2 + (p[2]-center_z)**2 for p in local]
    extent = max(max(p[i] for p in local)-min(p[i] for p in local) for i in range(3))
    coordinate_scale = max(abs(v) for p in points for v in p)
    padding = 64. * 2. ** -23 * max(extent, coordinate_scale, 1.e-3)
    radius = math.sqrt(max(radial_squared)) + padding
    caps = [math.sqrt(max(0., radius*radius-r2)) for r2 in radial_squared]
    first = min(p[0]+cap for p, cap in zip(local, caps))
    last = max(p[0]-cap for p, cap in zip(local, caps))
    if first > last:
        first = last = (first+last)/2.
    center = tuple(origin[i] + center_y*y[i] + center_z*z[i] for i in range(3))
    a = tuple(center[i] + direction[i]*first for i in range(3))
    b = tuple(center[i] + direction[i]*last for i in range(3))
    # Endpoint construction can land exactly on the inner sphere in f64.
    # Keep a separate outward roundoff reserve for the actual containment gate.
    radius += padding
    # This is the actual containment proof, independent of the fit formulas.
    maximum = max(capsule_segment_distance(p, p, a, b) for p in points)
    if not math.isfinite(maximum) or maximum > radius:
        raise ValueError('Existing proxy cannot be conservatively bounded at this coordinate precision')
    return {
        'eligible': False, 'capsule_known': True, 'capsule_bound': True, 'canonical': False,
        'shape': 'CAPSULE', 'length': (last-first)+2.*radius, 'radius': radius,
        'head': tuple(a[i]-direction[i]*radius for i in range(3)),
        'tail': tuple(b[i]+direction[i]*radius for i in range(3)),
        'direction': direction, 'coordinate_scale': max(coordinate_scale, (last-first)+2.*radius),
        'bound_vertex_count': len(points), 'bound_containment_max_distance': maximum,
        'bound_numerical_padding': 2.*padding,
    }


def proxy_capsule_bound(context, proxy):
    """Use the same evaluated geometry domain as export, then release it.

    Modifiers, shape keys, object scale and author edits are included. Returned
    values are primitive tuples/numbers; no evaluated RNA escapes this call.
    """
    depsgraph = context.evaluated_depsgraph_get()
    evaluated = proxy.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
    try:
        if mesh is None:
            raise ValueError('Existing proxy has no evaluated mesh to bound')
        if len(mesh.vertices) > 20002:
            raise ValueError('Existing proxy exceeds the capsule-bound vertex budget')
        matrix = evaluated.matrix_world.copy()
        return enclosing_capsule(
            (tuple(matrix @ v.co) for v in mesh.vertices),
            tuple(matrix.to_3x3().col[0]), tuple(matrix.translation))
    finally:
        evaluated.to_mesh_clear()


def collision_pair_admitted(first, second, *, adjacent_child=None):
    """Encoder author records, native symmetric AND, then adjacent exclusion."""
    from .bone_collision import settings_record
    a,b=settings_record(first),settings_record(second)
    ga,ma=(a['groups'],a['mask']) if a['enabled'] else (65535,65535)
    gb,mb=(b['groups'],b['mask']) if b['enabled'] else (65535,65535)
    if not (ga & mb and gb & ma):
        return False
    return adjacent_child is None or bool(adjacent_child.collide_connected)


def evaluated_world_geometry(context, proxy):
    from ..core.numpy_mesh_utils import loop_triangle_indices
    depsgraph=context.evaluated_depsgraph_get()
    evaluated=proxy.evaluated_get(depsgraph)
    mesh=evaluated.to_mesh(preserve_all_data_layers=False,depsgraph=depsgraph)
    try:
        if mesh is None or len(mesh.vertices)>20002:
            raise ValueError('Existing proxy exceeds the clearance proof vertex budget')
        matrix=evaluated.matrix_world.copy()
        return [tuple(matrix@v.co) for v in mesh.vertices],loop_triangle_indices(mesh).tolist()
    finally:
        evaluated.to_mesh_clear()


def separating_plane_clearance(points_a, faces_a, points_b, faces_b, required_gap):
    """Sufficient all-vertex proof, independent of convexity or BVH overlap.

    A separating projection contains every polygon because each polygon lies
    in its vertices' convex hull. No axis found means unproven, never collision.
    Candidate axes are bounded face normals, centers and world basis axes.
    """
    if not math.isfinite(required_gap) or required_gap<0:
        raise ValueError('Invalid clearance proof distance')
    a=[tuple(float(v) for v in p) for p in points_a]
    b=[tuple(float(v) for v in p) for p in points_b]
    if (not a or not b or len(a)>20002 or len(b)>20002
            or any(len(p)!=3 or not all(math.isfinite(v) for v in p) for p in a+b)):
        raise ValueError('Clearance proof requires bounded finite world vertices')
    coordinate_scale=max(1.e-3,max(abs(v) for p in a+b for v in p))
    reserve=128.*2.**-23*coordinate_scale
    center_a=tuple(sum(p[k] for p in a)/len(a) for k in range(3))
    center_b=tuple(sum(p[k] for p in b)/len(b) for k in range(3))
    axes=[tuple(x-y for x,y in zip(center_b,center_a)),(1.,0.,0.),(0.,1.,0.),(0.,0.,1.)]
    for points,faces in ((a,faces_a),(b,faces_b)):
        stride=max(1,math.ceil(len(faces)/250))
        for face in faces[::stride][:250]:
            if len(face)<3:continue
            p,q,r=(points[int(i)] for i in face[:3])
            axes.append(_cross(tuple(y-x for x,y in zip(p,q)),tuple(y-x for x,y in zip(p,r))))
    for axis in axes:
        squared=_dot(axis,axis)
        if not math.isfinite(squared) or squared<=1.e-24:continue
        unit=tuple(v/math.sqrt(squared) for v in axis)
        pa=[_dot(p,unit) for p in a];pb=[_dot(p,unit) for p in b]
        gap=max(min(pb)-max(pa),min(pa)-max(pb))
        if gap>=required_gap+reserve:
            return dict(axis=unit,projection_gap=gap,required_gap=required_gap,
                        numerical_reserve=reserve,vertex_counts=(len(a),len(b)))
    return None
