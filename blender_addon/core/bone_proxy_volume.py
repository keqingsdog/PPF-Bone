"""Author base volume from encoder-canonical loop triangles and world scale."""
import math
from array import array
import bmesh
from .bone_mass_math import strict_volume, validate_mesh_counts, MAX_TRIANGLES, MAX_VERTICES
from .numpy_mesh_utils import loop_triangle_indices


def matrix_determinant(matrix_world):
    if any(not math.isfinite(float(v)) for row in matrix_world for v in row):
        raise ValueError('代理对象变换包含非有限值')
    if any(abs(float(matrix_world[3][i])-v) > 1e-12 for i, v in enumerate((0., 0., 0., 1.))):
        raise ValueError('代理对象变换必须是仿射变换')
    determinant = float(matrix_world.to_3x3().determinant())
    if not math.isfinite(determinant) or determinant <= 0:
        raise ValueError('代理包含镜像、零缩放或无效变换；不能使用绝对体积')
    return determinant


def measure_volume(mesh, matrix_world, *, edit_mesh=False, moments=False):
    determinant = matrix_determinant(matrix_world)
    temporary_mesh = None
    if edit_mesh:
        original = bmesh.from_edit_mesh(mesh)
        # Read the existing Edit BMesh counts first. Never allocate a copy of a
        # large edited proxy just to discover afterwards that it exceeds budget.
        if len(original.verts) > MAX_VERTICES or len(original.faces) > MAX_TRIANGLES or len(original.edges) > 3*MAX_TRIANGLES:
            raise ValueError('编辑代理超过质量换算资源预算；请简化代理')
        loops = 0
        for face in original.faces:
            loops += len(face.verts)
            if loops > 3*MAX_TRIANGLES:
                raise ValueError('编辑代理环数超过质量换算资源预算；请简化代理')
        validate_mesh_counts(len(original.verts), len(original.faces), loops)
        bm = original.copy()
    else:
        validate_mesh_counts(len(mesh.vertices), len(mesh.polygons), len(mesh.loops))
        if len(mesh.edges) > 3*MAX_TRIANGLES:
            raise ValueError('代理边数超过质量换算资源预算；请简化代理')
        bm = bmesh.new()
    try:
        if not edit_mesh: bm.from_mesh(mesh)
        # Check original topology before triangulation: do not repair normals.
        if any(not e.is_manifold or not e.is_contiguous for e in bm.edges):
            raise ValueError('代理有开口、非流形边或面方向不一致；请检查网格')
        canonical_mesh = mesh
        if edit_mesh:
            # Read the live Edit BMesh through a disposable Mesh so its loop
            # tessellation uses exactly the encoder's canonical diagonal rules.
            import bpy
            temporary_mesh = bpy.data.meshes.new('PPF_Temporary_AuthorVolume')
            bm.to_mesh(temporary_mesh)
            canonical_mesh = temporary_mesh
        vertices = [tuple(v.co) for v in canonical_mesh.vertices]
        triangles = loop_triangle_indices(canonical_mesh).tolist()
        # Use the bounded streaming sweep instead of BVH.overlap's unbounded
        # result list on heavily overlapping geometry. Positive affine
        # transforms preserve intersections; determinant scales volume.
        result = strict_volume(vertices, triangles, determinant)
        result['local_volume_m3'] = result['volume_m3']/determinant
        result['volume_definition'] = 'author_base_mesh_matrix_world_m3'
        if moments:
            result.update(_rest_moments(vertices, triangles, result['local_volume_m3']))
        return result
    finally:
        if temporary_mesh is not None:
            import bpy
            bpy.data.meshes.remove(temporary_mesh)
        bm.free()


def _rest_moments(vertices, triangles, local_volume):
    """Second moments per unit mass, in local coordinates about point centroid."""
    import numpy as np

    points = np.asarray(vertices, dtype=np.float64)
    points -= points.mean(axis=0)
    a, b, c = points[np.asarray(triangles)].transpose(1, 0, 2)
    total = a + b + c
    fractions = np.einsum('ni,ni->n', a, np.cross(b, c)) / (6. * local_volume)
    first = np.einsum('n,ni->i', fractions, total) / 4.
    second = sum(np.einsum('n,ni,nj->ij', fractions, v, v)
                 for v in (a, b, c, total)) / 20.
    return dict(point_second_moment=points.T @ points / len(points),
                solid_second_moment=second,
                solid_centered_second_moment=second - np.outer(first, first))


def author_geometry_bytes(mesh, *, edit_mesh=False):
    """Exact bounded base geometry bytes, never evaluated modifier geometry.

    Called only by the timer after a dirty event. No RNA/BMesh references escape
    this function. Equality is byte-for-byte, avoiding hash-collision assumptions.
    """
    if edit_mesh:
        bm = bmesh.from_edit_mesh(mesh)
        if len(bm.verts) > MAX_VERTICES or len(bm.faces) > MAX_TRIANGLES or len(bm.edges) > 3*MAX_TRIANGLES:
            raise ValueError('编辑代理超过质量换算资源预算；请简化代理')
        loops = 0
        for face in bm.faces:
            loops += len(face.verts)
            if loops > 3*MAX_TRIANGLES: raise ValueError('编辑代理环数超过质量换算资源预算；请简化代理')
        validate_mesh_counts(len(bm.verts), len(bm.faces), loops)
        indexes = {vertex:index for index,vertex in enumerate(bm.verts)}
        coordinates = array('f', (component for vertex in bm.verts for component in vertex.co))
        edges = array('i', (indexes[vertex] for edge in bm.edges for vertex in edge.verts))
        faces = array('i')
        for face in bm.faces:
            faces.append(len(face.verts)); faces.extend(indexes[vertex] for vertex in face.verts)
        return ('EDIT', len(bm.verts), len(bm.edges), len(bm.faces), coordinates.tobytes(), edges.tobytes(), faces.tobytes())
    validate_mesh_counts(len(mesh.vertices), len(mesh.polygons), len(mesh.loops))
    if len(mesh.edges) > 3*MAX_TRIANGLES:
        raise ValueError('代理边数超过质量换算资源预算；请简化代理')
    coordinates = array('f', [0.])* (3*len(mesh.vertices))
    edges = array('i', [0])* (2*len(mesh.edges))
    loops = array('i', [0])* len(mesh.loops)
    starts = array('i', [0])* len(mesh.polygons)
    totals = array('i', [0])* len(mesh.polygons)
    mesh.vertices.foreach_get('co', coordinates)
    mesh.edges.foreach_get('vertices', edges)
    mesh.loops.foreach_get('vertex_index', loops)
    mesh.polygons.foreach_get('loop_start', starts)
    mesh.polygons.foreach_get('loop_total', totals)
    return ('OBJECT', len(mesh.vertices), len(mesh.edges), len(mesh.polygons),
            coordinates.tobytes(), edges.tobytes(), loops.tobytes(), starts.tobytes(), totals.tobytes())
