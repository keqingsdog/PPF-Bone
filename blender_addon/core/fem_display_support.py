"""Material-supported display endpoints for a sealed FEM bone binding.

The points are display data only. Joint forces use their native descriptors.
"""
from collections import OrderedDict
import marshal

import numpy as np

_CACHE = OrderedDict()
_MAX_CACHED_BINDINGS = 4096


def clear_cache():
    """Release retained receipt data when playback caches are detached."""
    _CACHE.clear()


def _invalid(message):
    raise ValueError("FEM display support: " + message)


def _support(point, tet_points, tetrahedra, inverse):
    # An interior point retains its exact barycentric coordinates. Outside
    # points use a convex combination from the nearest candidate tet;
    # the remaining rest offset is carried by that tet's material rotation.
    delta = point - tet_points[:, 0, :]
    beta = np.einsum("nij,nj->ni", inverse, delta)
    bary = np.column_stack((1.0 - beta.sum(axis=1), beta))
    convex = np.maximum(bary, 0.0)
    convex /= convex.sum(axis=1)[:, None]
    candidates = np.einsum("ni,nij->nj", convex, tet_points)
    distance = np.sum((candidates - point) ** 2, axis=1)
    chosen = int(np.argmin(distance))
    offset = point - candidates[chosen]
    zero_tolerance = 64.0 * np.finfo(np.float64).eps * max(
        1.0, float(np.max(np.abs(point))), float(np.max(np.abs(tet_points[chosen]))))
    if np.linalg.norm(offset) <= zero_tolerance:
        offset = np.zeros(3, dtype=np.float64)
    return (tetrahedra[chosen].copy(), inverse[chosen].copy(),
            convex[chosen].copy(), offset)


def validate_rest(row):
    from .fem_scale import compile_reference
    display = row["fem_display_rest_v1"]
    if (not isinstance(display, dict) or set(display) !=
            {"version", "vertex_indices", "positions_m"} or
            type(display["version"]) is not int or display["version"] != 1):
        _invalid("invalid version or fields")
    reference = compile_reference(row["fem_scale_reference"])
    ids = display["vertex_indices"]
    if (not isinstance(ids, list) or not ids or
            any(type(i) is not int or i < 0 or i >= reference.vertex_count for i in ids) or
            any(a >= b for a, b in zip(ids, ids[1:]))):
        _invalid("invalid rest vertex indices")
    positions = np.asarray(display["positions_m"], dtype=np.float64)
    if positions.shape != (len(ids), 3) or not np.isfinite(positions).all():
        _invalid("invalid rest vertex positions")
    ids = np.asarray(ids, dtype=np.int64)
    tet = reference.tet_vertex_indices
    if not np.array_equal(ids, np.unique(tet)):
        _invalid("rest positions differ from the material vertex set")
    lookup = np.searchsorted(ids, tet)
    if np.any(lookup >= len(ids)):
        _invalid("rest positions omit a material vertex")
    if not np.array_equal(ids[lookup], tet):
        _invalid("rest positions omit a material vertex")
    x = positions[lookup]
    dm = np.transpose(x[:, 1:] - x[:, :1], (0, 2, 1))
    identity = np.einsum("nij,njk->nik", dm, reference.dm_inv)
    if not np.isfinite(identity).all() or not np.allclose(identity, np.eye(3), atol=1e-5, rtol=1e-5):
        _invalid("rest positions disagree with material reference")
    initial = np.asarray(row["bone_initial_solver"], dtype=np.float64)
    anchor_ids = np.asarray(row["vertex_indices"], dtype=np.int64)
    anchor_lookup = np.searchsorted(ids, anchor_ids)
    if np.any(anchor_lookup >= len(ids)) or not np.array_equal(ids[anchor_lookup], anchor_ids):
        _invalid("rest positions omit the material anchor")
    center = np.asarray(row["center_bary"], dtype=np.float64) @ positions[anchor_lookup]
    frame = initial @ np.linalg.inv(np.asarray(row["body_to_bone_initial_solver"], dtype=np.float64))
    if not np.allclose(center, frame[:3, 3], atol=2e-5, rtol=0.0):
        _invalid("rest positions disagree with the material frame origin")
    return x, tet, reference.dm_inv


def _compile(row, rest_length):
    x, tet, inverse = validate_rest(row)
    initial = np.asarray(row["bone_initial_solver"], dtype=np.float64)
    head = initial[:3, 3]
    tail = head + initial[:3, 1] * rest_length
    return (_support(head, x, tet, inverse),
            _support(tail, x, tet, inverse))


def endpoints(row, xyz, rest_length):
    """Evaluate sealed material supports on one accepted vertex frame."""
    display = row["fem_display_rest_v1"]
    # The retained row reference prevents Python id reuse across scenes. The
    # snapshot detects edits to this additive untrusted receipt field.
    snapshot = marshal.dumps((display, row["fem_scale_reference"],
                              row["vertex_indices"], row["center_bary"],
                              row["body_to_bone_initial_solver"]), 4)
    initial = tuple(np.asarray(row["bone_initial_solver"], dtype=float).reshape(-1))
    key = (id(row), float(rest_length))
    cached = _CACHE.get(key)
    if (cached is None or cached[0] is not row or cached[1] != snapshot or
            cached[2] != initial or cached[3] is not row["fem_scale_reference"]):
        supports = _compile(row, rest_length)
        _CACHE[key] = (row, snapshot, initial, row["fem_scale_reference"], supports)
        while len(_CACHE) > _MAX_CACHED_BINDINGS:
            _CACHE.popitem(last=False)
    else:
        _CACHE.move_to_end(key)
        supports = cached[4]
    from .fem_material_rotation import proper_rotation
    result = []
    for ids, inverse, bary, offset in supports:
        if np.any(ids >= len(xyz)):
            _invalid("accepted frame omits a support vertex")
        points = xyz[ids].astype(np.float64)
        point = bary @ points
        if np.any(offset):
            deformation = (points[1:] - points[0]).T @ inverse
            point = point + proper_rotation(deformation) @ offset
        result.append(point)
    return result
