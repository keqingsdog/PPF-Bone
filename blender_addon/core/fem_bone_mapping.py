"""Map accepted FEM material vertices to a display bone without changing physics."""
import numpy as np


def _fail(message):
    raise ValueError("bone pose: " + message)


def map_bone(row, vertex_native_bytes, scale_enabled, scale_xyz=(1., 1., 1.), rest_length=1.):
    """Return solver-space 4x4 display transform and material tail point.

    The caller must first validate the complete accepted native frame, its
    vertex digest and the sealed binding.  This function also checks every
    value it consumes, including the selected tet's index and rank.
    """
    if (not isinstance(vertex_native_bytes, bytes) or len(vertex_native_bytes) % 12 or
            not np.isfinite(rest_length) or rest_length <= 0):
        _fail("invalid accepted FEM vertex bytes")
    xyz = np.frombuffer(vertex_native_bytes, dtype="<f4").reshape(-1, 3)
    ids = row["vertex_indices"]
    if (not isinstance(ids, list) or len(ids) != 4 or
            any(type(i) is not int or i < 0 or i >= len(xyz) for i in ids) or
            len(set(ids)) != 4):
        _fail("FEM tetrahedron indices outside accepted vertex frame")
    points = xyz[ids].astype(float)
    bary = np.asarray(row["center_bary"], dtype=float)
    dm_inv = np.asarray(row["dm_inv_column_major"], dtype=float).reshape(3, 3, order="F")
    rest_r = np.asarray(row["frame_rest_column_major"], dtype=float).reshape(3, 3, order="F")
    initial = np.asarray(row["bone_initial_solver"], dtype=float)
    offset = np.asarray(row["body_to_bone_initial_solver"], dtype=float)
    if (bary.shape != (4,) or not np.isfinite(bary).all() or
            abs(bary.sum() - 1.) > 1e-8 or not np.isfinite(points).all() or
            not np.isfinite(dm_inv).all() or not np.isfinite(rest_r).all() or
            initial.shape != (4, 4) or offset.shape != (4, 4) or
            not np.isfinite(initial).all() or not np.isfinite(offset).all()):
        _fail("invalid FEM material binding or accepted vertices")
    # The scale reference is independently sealed by validate_binding.  Make
    # sure this single-tet anchor uses the same rest topology and Dm inverse.
    reference = row.get("fem_scale_reference")
    if reference is not None:
        matches = [i for i, tet in enumerate(reference["tet_vertex_indices"]) if tet == ids]
        if len(matches) != 1:
            _fail("FEM anchor tet differs from sealed scale reference")
        inverse = np.asarray(reference["dm_inv_column_major"][matches[0]], dtype=float).reshape(3, 3, order="F")
        if not np.allclose(dm_inv, inverse, atol=1e-9, rtol=1e-9):
            _fail("FEM anchor rest inverse differs from sealed scale reference")
    # The rest frame and bone binding must describe the same authored axes.
    if not np.allclose(rest_r @ offset[:3, :3], initial[:3, :3], atol=5e-4, rtol=0):
        _fail("FEM rest frame differs from sealed bone binding")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        deformation = (points[1:] - points[0]).T @ dm_inv
    from .fem_material_rotation import proper_rotation
    try:
        polar = proper_rotation(deformation @ initial[:3, :3])
    except ValueError as exc:
        _fail(str(exc))
    center = bary @ points
    head = center + deformation @ (rest_r @ offset[:3, 3])
    deformed_y = deformation @ initial[:3, 1]
    if "fem_display_rest_v1" in row:
        from .fem_display_support import endpoints
        head, tail = endpoints(row, xyz, rest_length)
        deformed_y = (tail - head) / rest_length
    y_length = np.linalg.norm(deformed_y)
    if not np.isfinite(head).all() or not np.isfinite(y_length) or y_length <= 0:
        _fail("invalid FEM head or deformed bone axis")
    y = deformed_y / y_length
    # Polar orientation supplies twist/roll; the deformed Y axis owns the
    # actual tail direction even under shear.
    x = polar[:, 0] - y * np.dot(y, polar[:, 0])
    x_length = np.linalg.norm(x)
    if not np.isfinite(x_length) or x_length <= 1e-12:
        x = polar[:, 2] - y * np.dot(y, polar[:, 2])
        x_length = np.linalg.norm(x)
        if not np.isfinite(x_length) or x_length <= 1e-12:
            _fail("cannot construct FEM bone roll")
    x /= x_length
    z = np.cross(x, y)
    if scale_enabled:
        scales = np.asarray(scale_xyz, dtype=float)
        if scales.shape != (3,) or not np.isfinite(scales).all() or np.any(scales <= 0):
            _fail("invalid FEM directional scale")
        scales = np.array([scales[0], y_length, scales[2]])
    else:
        scales = np.ones(3)
    target = np.eye(4)
    target[:3, :3] = np.column_stack((x, y, z)) @ np.diag(scales)
    target[:3, 3] = head
    if not np.isfinite(target).all():
        _fail("unrepresentable FEM display transform")
    return target, head + deformed_y * rest_length
