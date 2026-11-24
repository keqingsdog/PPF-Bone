"""Outward geometry padding for skin-weight fits, in the fit frame's metres."""
import numpy as np
from mathutils.bvhtree import BVHTree

from .bone_weight_deform import _covered
from .bone_weight_proximity import check_python


def _tris(faces, alternate=False):
    return [tuple(f) for face in faces for f in (
        ((face[1], face[2], face[3]), (face[1], face[3], face[0]))
        if alternate and len(face) == 4 else
        ((face[0], face[i], face[i+1]) for i in range(1, len(face)-1))
    )]


def _tessellations(faces):
    yield _tris(faces)
    if any(len(f) == 4 for f in faces):
        yield _tris(faces, True)


def check_padded_clearance(vertices, faces, contact_offset):
    """Do not accept a padded fit below the existing known contact clearance."""
    if contact_offset <= 0:
        return
    v = np.asarray(vertices, dtype=np.float32).astype(float)
    for tris in _tessellations(faces):
        if check_python(v, np.asarray(tris), contact_offset)[0]:
            raise ValueError('Invalid local clearance')


def _normals(v, triangles):
    a, b, c = v[np.asarray(triangles)].transpose(1, 0, 2)
    return np.cross(b-a, c-a)


def _valid(v, padded, faces, points):
    # Verify the float32 geometry that Blender actually writes, including both
    # possible tessellations of convex-fit quads.
    q = padded.astype(np.float32).astype(float)
    if not np.isfinite(q).all():
        return False
    for tris in _tessellations(faces):
        normals = _normals(v, tris)
        shifted = _normals(q, tris)
        if np.any(np.linalg.norm(shifted, axis=1) < 2.e-14) or np.any(np.sum(normals * shifted, axis=1) <= 0):
            return False
        ref = [frozenset(f) for f in tris]
        tree = BVHTree.FromPolygons(q.tolist(), tris, all_triangles=True)
        if any(i < j and not ref[i] & ref[j] for i, j in tree.overlap(tree)):
            return False
        if not _covered(q, tris, points):
            return False
    return True


def pad_fitted_proxy(vertices, faces, points, offset, *, sheet_axis=None):
    if offset == 0.0:
        return vertices
    if not np.isfinite(offset) or offset < 0:
        raise ValueError('Invalid offset')
    v = np.asarray(vertices, dtype=float)
    if sheet_axis is not None:
        # The two caps have the same projected graph. Translating each as a
        # whole preserves its slopes, footprint, source coverage and topology.
        # Rebuild the corresponding walls by keeping connectivity unchanged.
        padded = v.copy()
        n = len(v)//2
        padded[:n] -= offset * sheet_axis
        padded[n:] += offset * sheet_axis
    else:
        # Preserve the chosen starter topology. Increase both ends of each
        # fit-frame bounding axis by the requested distance, with one
        # positive diagonal affine transform (no local folds).
        lo, hi = v.min(axis=0), v.max(axis=0)
        half = (hi - lo) * 0.5
        if np.any(half <= 1.e-12):
            raise ValueError('Degenerate fitting bounds')
        center = (lo + hi) * 0.5
        padded = center + (v - center) * (1.0 + offset / half)
    if not _valid(v, padded, faces, points):
        raise ValueError('Invalid padded geometry')
    return padded.tolist()
