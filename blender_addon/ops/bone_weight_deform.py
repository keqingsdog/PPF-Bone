"""Sample the padded selected-weight convex footprint on the starter mesh."""
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def _planes(points):
    mesh = bmesh.new()
    try:
        verts = [mesh.verts.new(tuple(point)) for point in points]
        bmesh.ops.convex_hull(mesh, input=verts, use_existing_faces=False)
        mesh.normal_update()
        return np.asarray([tuple(f.normal) for f in mesh.faces], dtype=float).reshape(-1, 3)
    finally:
        mesh.free()


def _dense_hull_planes(points):
    """Return float32 hull planes with padding for vertex quantization."""
    mesh = bmesh.new()
    try:
        verts = [mesh.verts.new(tuple(point)) for point in points]
        bmesh.ops.convex_hull(mesh, input=verts, use_existing_faces=False)
        mesh.normal_update()
        if not mesh.faces or any(not edge.is_manifold for edge in mesh.edges):
            return None
        coordinates = np.asarray([[tuple(v.co) for v in face.verts[:3]]
                                  for face in mesh.faces], dtype=float)
        anchors = coordinates[:, 0]
        normals = np.cross(coordinates[:, 1] - anchors, coordinates[:, 2] - anchors)
        authored_normals = np.asarray([tuple(face.normal) for face in mesh.faces], dtype=float)
        normals[np.einsum('ij,ij->i', normals, authored_normals) < 0] *= -1
        lengths = np.linalg.norm(normals, axis=1)
        if (not np.isfinite(normals).all() or not np.isfinite(anchors).all()
                or np.any(lengths < 1.e-12)):
            return None
        normals /= lengths[:, None]
        error = 32 * np.finfo(np.float32).eps * max(1.0, float(np.max(np.abs(points))))
        return normals, np.einsum('ij,ij->i', normals, anchors) + error
    finally:
        mesh.free()


def _covered(vertices, faces, points):
    triangles = [(face[0], face[i], face[i+1])
                 for face in faces for i in range(1, len(face)-1)]
    if not triangles or not np.isfinite(vertices).all():
        return False
    edges = {}
    for face in faces:
        for index, start in enumerate(face):
            edge = tuple(sorted((start, face[(index+1) % len(face)])))
            edges[edge] = edges.get(edge, 0) + 1
    if any(count != 2 for count in edges.values()):
        return False
    tri = np.asarray(triangles, dtype=int)
    if np.any(np.linalg.norm(np.cross(vertices[tri[:, 1]] - vertices[tri[:, 0]],
                                      vertices[tri[:, 2]] - vertices[tri[:, 0]]), axis=1) <= 1.e-16):
        return False
    tree = BVHTree.FromPolygons(list(map(tuple, vertices)), triangles, all_triangles=True)
    direction = Vector((1.0, 0.371, 0.217)).normalized()
    for point in points:
        origin = Vector(point)
        nearest = tree.find_nearest(origin)
        if nearest[3] is None:
            return False
        if nearest[3] < 1.e-7:
            continue
        count = 0
        for _ in range(len(triangles)+1):
            hit = tree.ray_cast(origin, direction)
            if hit[0] is None:
                break
            count += 1
            origin = hit[0] + direction * 1.e-7
        else:
            return False
        if count % 2 == 0:
            return False
    return True


def _coverage_gain(rays, radii, faces, points):
    """Expand the few intercepted triangles, not the entire fitted proxy."""
    verts = list(map(tuple, rays*radii[:, None]))
    gain = np.ones(len(verts), dtype=float)
    required = np.linalg.norm(points, axis=1)
    for alternate in (False, True) if any(len(face) == 4 for face in faces) else (False,):
        if alternate:
            triangles = [tri for face in faces for tri in
                         ((face[:3],) if len(face) == 3 else
                          ((face[1], face[2], face[3]), (face[1],face[3],face[0])))]
        else:
            triangles = [(face[0], face[i], face[i+1])
                         for face in faces for i in range(1,len(face)-1)]
        tree = BVHTree.FromPolygons(verts, triangles, all_triangles=True)
        for point, distance in zip(points, required):
            if distance < 1.e-8:
                continue
            _, _, i, hit = tree.ray_cast((0.0,0.0,0.0), tuple(point / distance))
            if i is not None and 0.0 < hit < distance:
                ids = list(triangles[i])
                gain[ids] = np.maximum(gain[ids], (distance / hit) * 1.00002)
    return gain


def deform_weight_contour(vertices, faces, samples, strength, *, _exact=False):
    """Samples are (X,Y,Z,weight) in the proxy frame, length in metres."""
    sample = np.asarray(samples, dtype=float)
    if len(sample) == 0 or not np.isfinite(sample).all() or not 0.0 <= strength <= 1.0:
        return vertices
    # Selection is already established by the operator; no internal weight cutoff.
    points = np.unique(sample[:, :3], axis=0)
    stock = np.asarray(vertices, dtype=float)
    stock_extent = np.max(np.abs(stock), axis=0)
    ray = stock / np.maximum(stock_extent, 1.e-12)
    if strength == 0.0:
        # Preserve the exact starter shape; if coarse facets cut into the
        # selected points, dilate it uniformly without changing its topology.
        gain = float(np.max(_coverage_gain(ray, np.ones(len(ray)), faces, points/stock_extent)))
        return vertices if gain == 1.0 else list(map(tuple, stock*gain))
    origin = (points.min(axis=0) + points.max(axis=0)) * 0.5
    extents = np.ptp(points, axis=0)
    fit_scale = max(float(max(extents)), 0.001)
    fit_extents = np.maximum(extents * 0.5, max(0.00025, 0.0125 * fit_scale))
    local = (points - origin) / fit_extents
    centroid = local.mean(axis=0)
    local -= centroid
    center = origin + centroid * fit_extents

    # Support planes of the actual footprint. Planar sets do not have a
    # closed hull, so also sample uniformly oriented planes and the two
    # directions of every principal axis (including the thin normal).
    axes = np.linalg.eigh(local.T @ local)[1].T
    z = 1 - (2*np.arange(50)+1)/50
    azimuth = np.arange(50)*np.pi*(3-np.sqrt(5.0))
    xy = np.sqrt(1-z*z)
    directions = [np.column_stack((xy*np.cos(azimuth), xy*np.sin(azimuth), z)),
                  axes, -axes, np.eye(3), -np.eye(3)]
    dense_hull = None
    if np.linalg.matrix_rank(local, tol=1.e-5) == 3:
        if len(points) > 8192 and not _exact:
            dense_hull = _dense_hull_planes(local)
        if dense_hull is None:
            hull = _planes(local)
            if len(hull):
                directions.append(hull)
    normals = np.unique(np.round(np.concatenate(directions), 6), axis=0)
    normals = normals[np.linalg.norm(normals, axis=1) > 0.5]
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    # Keep a physical minimum total thickness of 0.5 mm, or 2.5% of
    # footprint size if larger. This is hull padding, not skin offset.
    padding = max(0.00025, 0.0125 * fit_scale)
    support = []
    for chunk in np.array_split(normals, max(1, (len(normals)+127)//128)):
        support.extend(np.max(local @ chunk.T, axis=0) + padding * np.linalg.norm(chunk / fit_extents, axis=1))
    support = np.asarray(support)
    if dense_hull is not None:
        hull_normals, hull_support = dense_hull
        hull_support += padding * np.linalg.norm(hull_normals / fit_extents, axis=1)
        normals = np.concatenate((normals, hull_normals))
        support = np.concatenate((support, hull_support))

    # The same starter ray is used at 0% and 100%, and relative vertex
    # order stays unchanged. Interpolate positive radii, not the angles.
    if dense_hull is None and len(normals) <= 8192:
        dot = ray @ normals.T
        hit = np.min(np.where(dot > 1.e-10, support / np.maximum(dot, 1.e-10), np.inf), axis=1)
    else:
        hit = np.full(len(ray), np.inf)
        for start in range(0, len(normals), 128):
            dot = ray @ normals[start:start+128].T
            distance = np.where(dot > 1.e-10,
                                support[start:start+128] / np.maximum(dot, 1.e-10),
                                np.inf)
            hit = np.minimum(hit, np.min(distance, axis=1))
    if not np.all(np.isfinite(hit)):
        return vertices
    # Normalize the 0% endpoint before morphing so an expanded starter
    # doesn't snap down toward nominal dimensions at an infinitesimal fit.
    baseline = float(np.max(_coverage_gain(ray, np.ones(len(ray)), faces, points/stock_extent)))
    extent = (1.0-strength)*stock_extent*baseline + strength*fit_extents
    radii = 1.0-strength + strength*hit
    position = strength * center
    radii *= _coverage_gain(ray, radii, faces, (points-position)/extent)
    result = position + ray * extent * radii[:, None]
    if dense_hull is not None and not _covered(result, faces, points):
        exact = np.asarray(deform_weight_contour(vertices, faces, samples, strength, _exact=True))
        if not _covered(exact, faces, points):
            raise ValueError('The fitted proxy does not cover all selected weighted vertices')
        return list(map(tuple, exact))
    return list(map(tuple, result))
