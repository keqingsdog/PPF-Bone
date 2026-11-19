"""Use the native input precision for material support and joint weights."""
import numpy as np


def native_rest_positions(fixed):
    # Export writes these two arrays as f64, but the native constructor narrows
    # each operand to f32 before adding the mapped displacement.
    dmap, local = fixed._vert
    dmap = np.asarray(dmap)
    local = np.asarray(local, dtype=np.float32)
    displacement = np.asarray(fixed._displacement, dtype=np.float32)
    if (local.ndim != 2 or local.shape[1] != 3 or dmap.shape != (len(local),)
            or dmap.dtype.kind not in 'iu' or displacement.ndim != 2
            or displacement.shape[1] != 3
            or np.any(dmap < 0) or np.any(dmap >= len(displacement))):
        raise ValueError('FEM_NATIVE_REST_INVALID: local/displacement mapping dimensions')
    world = local + displacement[dmap]
    if not np.isfinite(world).all():
        raise ValueError('FEM_NATIVE_REST_INVALID: nonfinite native positions')
    return world.astype(np.float64)


def material_anchor_bary(nodes, center_bary, frame_rest_column_major, local_translation):
    """Reconstruct exactly the frame used by soft_joint::prepare, then solve.

    The native endpoint stores center and rotation as f32. Its strict anchor
    check promotes those stored values and the actual f32 rest positions to
    f64; using an unquantized author image here can amplify a ULP through
    extrapolated barycentric coordinates.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    center = np.asarray(center_bary, dtype=np.float32).astype(np.float64)
    rotation = np.asarray(frame_rest_column_major, dtype=np.float32).astype(np.float64).reshape(3, 3, order='F')
    local = np.asarray(local_translation, dtype=np.float64)
    if (nodes.shape != (4, 3) or center.shape != (4,) or local.shape != (3,)
            or not all(np.isfinite(value).all() for value in (nodes, center, rotation, local))):
        raise ValueError('FEM_NATIVE_ANCHOR_INVALID: nonfinite or incorrect support dimensions')
    target = center @ nodes + rotation @ local
    dm = np.column_stack([nodes[k] - nodes[0] for k in (1, 2, 3)])
    beta = np.linalg.solve(dm, target - nodes[0])
    weights = np.r_[1.0 - beta.sum(), beta]
    if not np.isfinite(weights).all() or np.linalg.norm(weights @ nodes - target) > 2e-6:
        raise ValueError('FEM_NATIVE_ANCHOR_INVALID: support cannot reconstruct native material anchor')
    return weights
