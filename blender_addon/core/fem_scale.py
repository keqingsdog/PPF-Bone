"""Read-only directional FEM stretch; reimplemented after workspace loss.

This is a NEW implementation, not the lost R7/R8 source or its verification.
For every body tet, F = Ds @ inv(Dm), C = F.T @ F. Average C with
absolute *rest* volume weights, then evaluate sqrt(a.T @ Cbar @ a) in
the sealed authored bone rest axes. Neither averaged F nor sorted SVD
stretches can stand in for those bone-local directional measurements.

The caller owns scene/frame/hash/accepted-bracket validation. In production,
both receipt decoding and measurement must pass the actual binding's rest
axes as ``expected_rest_axes``; a self-consistent reference is not an
independent authority for a binding. No solver state or geometry is changed.
"""

from collections import OrderedDict
from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
import marshal

import numpy as np


KIND = "PPF_FEM_BONE_SCALE_REFERENCE_V1"
COORDINATE_SYSTEM = "solver_y_up_meters"
MAX_REST_CONDITION = 1.0e8
MAX_FRAME_CONDITION = 1.0 / (64.0 * np.finfo(np.float64).eps)
_MAX_COMPILED = 128
_KEYS = frozenset((
    "version", "kind", "coordinate_system", "vertex_count",
    "tet_vertex_indices", "dm_inv_column_major", "rest_volumes",
    "bone_rest_axes_column_major", "sha256",
))
_compiled = OrderedDict()
_admitted = OrderedDict()


def _remember(reference, compiled):
    snapshot = deepcopy(reference)
    # An in-process exact-typed snapshot comparison, never a transport/seal or
    # executable payload. marshal.loads is deliberately not used. The C encoder
    # distinguishes bool/int, int/float and list/tuple unlike ordinary equality.
    _admitted[id(reference)] = (reference, snapshot, compiled, marshal.dumps(snapshot, 4))
    _admitted.move_to_end(id(reference))
    while len(_admitted) > _MAX_COMPILED:
        _admitted.popitem(last=False)
    return compiled


def _fail(message):
    raise ValueError("FEM bone scale: " + message)


def _numbers(value, shape, label, *, json_only=False):
    if json_only:
        def visit(item):
            if isinstance(item, list):
                return all(visit(child) for child in item)
            return type(item) in (int, float)
        if not isinstance(value, list) or not visit(value):
            _fail(label + " must contain JSON numbers (not bool/string)")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        _fail(label + " is not numeric")
    if array.shape != shape or not np.isfinite(array).all():
        _fail(label + " has invalid shape or nonfinite values")
    return array


def _axes(value, *, stored=False):
    array = _numbers(value, (9,), "rest bone axes", json_only=stored)
    axes = array.reshape(3, 3, order="F")
    tolerance = 5.0e-4
    if (np.max(np.abs(axes.T @ axes - np.eye(3))) > tolerance
            or abs(float(np.linalg.det(axes)) - 1.0) > tolerance):
        _fail("rest bone axes must be a proper orthonormal frame")
    if stored:
        if np.max(np.abs(np.linalg.norm(axes, axis=0) - 1.0)) > 1.0e-10:
            _fail("sealed rest bone axes must be unit directions")
        return axes
    # Normalize only near-unit authored axes, never a scaled/sheared frame.
    return axes / np.linalg.norm(axes, axis=0)


def _digest(reference):
    payload = {key: value for key, value in reference.items() if key != "sha256"}
    try:
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            allow_nan=False, ensure_ascii=True,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError, RecursionError):
        _fail("reference is not finite canonical JSON")
    return hashlib.sha256(encoded).hexdigest()


def _immutable(array, dtype):
    """Immutable bytes owner prevents even setflags(write=True) on the view."""
    array = np.ascontiguousarray(array, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, eq=False)
class CompiledReference:
    sha256: str
    vertex_count: int
    tet_vertex_indices: np.ndarray
    dm_inv: np.ndarray
    rest_volumes: np.ndarray
    rest_weights: np.ndarray
    bone_axes: np.ndarray


def _checked_header(reference, expected_rest_axes):
    """Full sealed admission gate; unchanged snapshots retain these checks."""
    if not isinstance(reference, dict) or set(reference) != _KEYS:
        _fail("reference schema keys differ")
    if (type(reference["version"]) is not int or reference["version"] != 1
            or reference["kind"] != KIND
            or reference["coordinate_system"] != COORDINATE_SYSTEM):
        _fail("unsupported reference version/kind/coordinates")
    count = reference["vertex_count"]
    if type(count) is not int or count < 4:
        _fail("invalid reference vertex count")
    digest = reference["sha256"]
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or _digest(reference) != digest):
        _fail("reference SHA256 mismatch")
    axes = _axes(reference["bone_rest_axes_column_major"], stored=True)
    if expected_rest_axes is not None:
        expected = _axes(expected_rest_axes)
        if not np.allclose(axes, expected, atol=1.0e-12, rtol=0.0):
            _fail("reference axes differ from actual bone rest binding")
    return digest, count, axes


def compile_reference(reference, expected_rest_axes=None):
    """Validate a sealed JSON reference and reuse immutable compiled arrays.

Initial admission verifies every sealed field. An unchanged strong-held
reference is compared with an independent exact-typed deep snapshot before
reusing immutable arrays; expected binding axes are still checked each call.
Any edit takes the complete original digest/schema/semantic validation path.
"""
    key = id(reference)
    admitted = _admitted.get(key)
    if admitted is not None and admitted[0] is reference:
        if marshal.dumps(reference, 4) == admitted[3]:
            cached = admitted[2]
            if expected_rest_axes is not None:
                expected = _axes(expected_rest_axes)
                if not np.allclose(cached.bone_axes, expected, atol=1.0e-12, rtol=0.0):
                    _fail("reference axes differ from actual bone rest binding")
            _admitted.move_to_end(key)
            return cached
        _admitted.pop(key, None)
        _compiled.pop(admitted[2].sha256, None)
    digest, count, axes = _checked_header(reference, expected_rest_axes)
    cached = _compiled.get(digest)
    if cached is not None:
        _compiled.move_to_end(digest)
        return _remember(reference, cached)
    ids = reference["tet_vertex_indices"]
    if (not isinstance(ids, list) or not ids
            or any(not isinstance(tet, list) or len(tet) != 4
                   or any(type(i) is not int or i < 0 or i >= count for i in tet)
                   or len(set(tet)) != 4 for tet in ids)):
        _fail("invalid complete tetrahedron vertex indices")
    if len({tuple(sorted(tet)) for tet in ids}) != len(ids):
        _fail("duplicate tetrahedron in reference")
    n = len(ids)
    inv_wire = _numbers(reference["dm_inv_column_major"], (n, 9),
                        "inverse rest matrices", json_only=True)
    inverse = inv_wire.reshape(n, 3, 3).transpose(0, 2, 1)
    volumes = _numbers(reference["rest_volumes"], (n,), "rest volumes", json_only=True)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        determinants = np.linalg.det(inverse)
        condition = np.linalg.cond(inverse)
        consistency = np.abs(determinants) * volumes * 6.0
    if (not np.isfinite(determinants).all() or np.any(determinants == 0)
            or not np.isfinite(condition).all() or np.any(condition > MAX_REST_CONDITION)
            or np.any(volumes <= 0)
            or not np.allclose(consistency, 1.0, rtol=1.0e-7, atol=0.0)):
        _fail("degenerate/ill-conditioned rest tet or inconsistent rest volume")
    relative = volumes / np.max(volumes)
    weights = relative / np.sum(relative)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        _fail("unrepresentable rest volume weights")
    compiled = CompiledReference(
        digest, count, _immutable(ids, np.int64), _immutable(inverse, np.float64),
        _immutable(volumes, np.float64), _immutable(weights, np.float64),
        _immutable(axes, np.float64),
    )
    _compiled[digest] = compiled
    while len(_compiled) > _MAX_COMPILED:
        _compiled.popitem(last=False)
    return _remember(reference, compiled)


def build_reference(rest_xyz_global, tets_global_Nx4,
                    rest_bone_axes_solver_column_major9):
    """Seal ALL supplied body tets in the global solver vertex namespace.

Pass ``fixed.vertex()`` and the entire actual body's ``fixed._tet[lo:lo+n]``.
Selection completeness is the frontend's responsibility, not inferred from
geometry. Left-handed rest winding is valid: volume uses abs(det(Dm)).
"""
    try:
        vertices = np.asarray(rest_xyz_global, dtype=np.float64)
        tets = np.asarray(tets_global_Nx4)
    except (TypeError, ValueError, OverflowError):
        _fail("invalid rest vertices/topology")
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,)
            or len(vertices) < 4 or not np.isfinite(vertices).all()):
        _fail("invalid finite global rest vertices")
    if (tets.ndim != 2 or tets.shape[1:] != (4,) or len(tets) == 0
            or tets.dtype.kind not in "iu" or np.any(tets < 0)
            or np.any(tets >= len(vertices))):
        _fail("invalid full body tetrahedra")
    tets = tets.astype(np.int64, copy=False)
    if any(len(set(map(int, tet))) != 4 for tet in tets):
        _fail("tetrahedron has repeated vertices")
    axes = _axes(rest_bone_axes_solver_column_major9)
    points = vertices[tets]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        dm = (points[:, 1:] - points[:, :1]).transpose(0, 2, 1)
        if not np.isfinite(dm).all():
            _fail("unrepresentable rest edge matrix")
        det = np.linalg.det(dm)
        condition = np.linalg.cond(dm)
    if (not np.isfinite(det).all() or np.any(det == 0)
            or not np.isfinite(condition).all() or np.any(condition > MAX_REST_CONDITION)):
        _fail("degenerate/ill-conditioned rest tetrahedron")
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            inverse = np.linalg.inv(dm)
            volumes = np.abs(det) / 6.0
    except np.linalg.LinAlgError:
        _fail("singular rest tetrahedron")
    reference = {
        "version": 1,
        "kind": KIND,
        "coordinate_system": COORDINATE_SYSTEM,
        "vertex_count": len(vertices),
        "tet_vertex_indices": tets.tolist(),
        "dm_inv_column_major": inverse.transpose(0, 2, 1).reshape(-1, 9).tolist(),
        "rest_volumes": volumes.tolist(),
        "bone_rest_axes_column_major": axes.T.reshape(-1).tolist(),
    }
    reference["sha256"] = _digest(reference)
    compile_reference(reference, rest_bone_axes_solver_column_major9)
    return reference


def measure(reference, vertices, expected_rest_axes=None):
    """Return current bone-axis stretch and explicit approximation diagnostics.

``vertices`` is the full accepted global Nx3 frame or exact little-endian
float32 vertex bytes. Refuse invalid tets; never skip them or return unit
scale as a fallback. Each call is rest-relative and does not accumulate.
"""
    compiled = compile_reference(reference, expected_rest_axes)
    if isinstance(vertices, (bytes, bytearray, memoryview)):
        if len(vertices) != compiled.vertex_count * 12:
            _fail("vertex frame byte length differs from reference")
        try:
            xyz = np.frombuffer(vertices, dtype="<f4").reshape(-1, 3).astype(np.float64)
        except (TypeError, ValueError, BufferError):
            _fail("invalid contiguous little-endian vertex bytes")
    else:
        xyz = _numbers(vertices, (compiled.vertex_count, 3), "vertex frame")
    if not np.isfinite(xyz).all():
        _fail("vertex frame is nonfinite")
    points = xyz[compiled.tet_vertex_indices]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ds = (points[:, 1:] - points[:, :1]).transpose(0, 2, 1)
        deformation = ds @ compiled.dm_inv
        det_f = np.linalg.det(deformation)
        tensors = deformation.transpose(0, 2, 1) @ deformation
    if (not np.isfinite(deformation).all() or not np.isfinite(det_f).all()
            or not np.isfinite(tensors).all()):
        _fail("nonfinite deformed tetrahedron")
    # Scale is defined by the actual F^T F, including signed or rank-deficient
    # individual tets. It does not require a rotation or an inverse of F;
    # the aggregate directions below must still have finite positive length.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mean = np.einsum("t,tij->ij", compiled.rest_weights, tensors)
        projected = compiled.bone_axes.T @ mean @ compiled.bone_axes
        squared = np.diag(projected)
        scales = np.sqrt(squared)
    if not np.isfinite(projected).all() or not np.isfinite(scales).all() or np.any(squared <= 0):
        _fail("invalid directional scale")
    # Relative off-diagonal metric and RMS F variation flag what a single
    # diagonal bone scale cannot represent, including differing local rigid
    # rotations (bending). Averaged F is DIAGNOSTIC ONLY, never used for scale.
    # Rescale before norms to
    # avoid squaring already large finite C entries into infinity.
    magnitude = float(np.max(np.abs(mean)))
    normalized = mean / magnitude
    denominator = float(np.linalg.norm(normalized))
    off_diagonal = projected - np.diag(squared)
    shear = float(np.linalg.norm(off_diagonal / magnitude) / denominator)
    normalized_f = deformation / float(np.max(np.abs(deformation)))
    mean_f = np.einsum("t,tij->ij", compiled.rest_weights, normalized_f)
    differences = normalized_f - mean_f
    with np.errstate(over="ignore", invalid="ignore"):
        variation = float(np.sqrt(np.einsum(
            "t,tij,tij->", compiled.rest_weights, differences, differences,
        ) / np.einsum("t,tij,tij->", compiled.rest_weights, normalized_f, normalized_f)))
        total_volume = float(np.sum(compiled.rest_volumes))
    if not all(np.isfinite(value) for value in (shear, variation, total_volume)):
        _fail("unrepresentable scale diagnostic")
    return {
        "scale_xyz": scales.tolist(),
        "shear_relative": shear,
        "variation_relative": variation,
        "tet_count": len(compiled.tet_vertex_indices),
        "rest_volume": total_volume,
        "min_det_f": float(np.min(det_f)),
        "max_det_f": float(np.max(det_f)),
        "reference_sha256": compiled.sha256,
        "approximation": shear > 1.0e-8 or variation > 1.0e-8,
    }
