"""Closest proper material frame without altering the accepted deformation."""
import numpy as np


def proper_rotation(deformation):
    f = np.asarray(deformation, dtype=np.float64)
    if f.shape[-2:] != (3, 3) or not np.isfinite(f).all():
        raise ValueError("nonfinite or malformed FEM deformation")
    try:
        u, singular, vt = np.linalg.svd(f)
        determinant = np.linalg.det(f)
        orientation = np.linalg.det(u @ vt)
    except np.linalg.LinAlgError as exc:
        raise ValueError("FEM material SVD failed") from exc
    if (not np.isfinite(singular).all() or not np.isfinite(determinant).all()
            or not np.isfinite(orientation).all()
            or np.any(np.abs(np.abs(orientation) - 1.) > 5e-4)):
        raise ValueError("invalid FEM material SVD")
    sign = np.where(orientation < 0., -1., 1.)
    corrected = u.copy()
    corrected[..., :, 2] *= sign[..., None]
    rotation = corrected @ vt
    # np.linalg.svd returns singular values in descending order. Full-rank
    # positive deformations preserve the existing polar path. At rank two or
    # for reflected F, certify the unique SO(3) factor.
    if np.any((singular[..., 0] <= 0.) | (singular[..., 1] <= 2e-6 * singular[..., 0])):
        raise ValueError("FEM material frame is rank deficient")
    special = (determinant <= 0.) | (singular[..., 2] == 0.)
    if np.any(special):
        stretch = np.swapaxes(rotation, -1, -2) @ f
        maxs = np.max(np.abs(stretch), axis=(-2, -1))
        skew = np.max(np.abs(stretch - np.swapaxes(stretch, -1, -2)), axis=(-2, -1))
        if (not np.isfinite(maxs).all() or np.any(maxs <= 0.)
                or not np.isfinite(skew).all() or np.any(skew > 2e-5 * maxs)):
            raise ValueError("invalid signed FEM stretch")
        h = (stretch + np.swapaxes(stretch, -1, -2)) * (0.5 / maxs)[..., None, None]
        km = np.trace(h, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) - h
        first = km[..., 0, 0]
        minor = km[..., 0, 0] * km[..., 1, 1] - km[..., 0, 1] * km[..., 1, 0]
        kd = np.linalg.det(km)
        if np.any(special & ((first <= 1e-6) | (minor <= 1e-6) | (kd <= 1e-6))):
            raise ValueError("FEM material rotation is non-unique or ill-conditioned")
    return rotation
