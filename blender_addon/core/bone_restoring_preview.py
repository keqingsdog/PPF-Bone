"""Author-time angular estimates from the validated rest proxy and mass cache."""
import math

import numpy as np
from mathutils import Euler, Matrix, Quaternion

from .bone_mass import cached_rest_mass
from .bone_physical_units import linear_constants
from ..models.bone_bindings import _frame, bone_world_matrix, rigid_matrix


def _angular_mobility(scene, arm, bone, axes):
    settings = bone.ppf_bone_physics
    measured = cached_rest_mass(scene, arm, bone)
    if measured is None:
        return None
    if settings.root_mode in {'STATIC', 'KINEMATIC'}:
        return np.zeros(3)

    transform = np.asarray(settings.proxy.matrix_world.to_3x3(), dtype=np.float64)
    if settings.physical_model == 'RIGID':
        points = transform @ measured['point_second_moment'] @ transform.T
        total_mass = measured['mass_kg']
        if settings.mass_mode == 'DENSITY':
            # PDRD density mode scales uniform point masses to match the
            # trace of the solid inertia about the point centroid.
            solid = transform @ measured['solid_second_moment'] @ transform.T
            if np.trace(points) <= 0.:
                return None
            total_mass *= np.trace(solid) / np.trace(points)
        second = points * total_mass
    else:
        # The surface defines the enclosed rest-volume inertia. FEM Build
        # uses the actual tetrahedral nodal masses and joint frame Jacobian.
        second = transform @ measured['solid_centered_second_moment'] @ transform.T
        second *= measured['mass_kg']
    inertia = np.eye(3) * np.trace(second) - second
    eigenvalues = np.linalg.eigvalsh(inertia)
    if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= eigenvalues[-1] * 1e-12:
        return None
    return np.einsum('ij,ij->j', axes, np.linalg.solve(inertia, axes))


def _joint_axes(arm, bone):
    settings = bone.ppf_bone_physics
    if settings.use_auto_frames:
        return (bone_world_matrix(arm, bone).to_3x3()
                @ Matrix.Rotation(math.pi / 2., 3, 'Z')
                @ Euler(tuple(settings.joint_axis_euler)).to_matrix())

    # Build removes the authored rest rotation before evaluating the angular
    # chart. Its rest Jacobian therefore uses the parent joint's world axes,
    # even if the two manual frame orientations are different.
    parent = bone.parent
    parent_settings = parent.ppf_bone_physics
    if parent_settings.enabled:
        if parent_settings.proxy is None:
            raise ValueError('Missing parent rest proxy')
        parent_world = rigid_matrix(parent_settings.proxy.matrix_world, parent.name)
    else:
        parent_world = bone_world_matrix(arm, parent)
    _frame(settings, 'child')
    parent_frame = _frame(settings, 'parent')
    return parent_world.to_3x3() @ Quaternion(parent_frame['quaternion_wxyz']).to_matrix()


def angular_metrics(context, arm, bone):
    """Per-axis effective inertias for a hierarchy joint in its rest pose."""
    if arm is None or bone.parent is None:
        return None
    scene = context.scene
    try:
        axes = np.asarray(_joint_axes(arm, bone), dtype=np.float64)
        mobility = _angular_mobility(scene, arm, bone, axes)
        if mobility is None:
            return None
        # Build anchors to the immediate nonphysical parent kinematically.
        # That anchor does not take inertia from a farther simulated ancestor.
        parent = bone.parent
        if parent.ppf_bone_physics.enabled:
            parent_mobility = _angular_mobility(scene, arm, parent, axes)
            if parent_mobility is None:
                return None
            mobility += parent_mobility
        if not np.isfinite(mobility).all() or np.any(mobility <= 0.):
            return None
        return tuple(float(1. / value) for value in mobility)
    except (ValueError, RuntimeError, ReferenceError, np.linalg.LinAlgError):
        return None


def angular_constants(metric, saved):
    """Resolve the same linear AUTO_FREQUENCY coefficients as Build."""
    line = linear_constants(saved['table'], 0.)
    if line is None:
        return None
    binding = saved['binding']
    slope = line[0]
    ratio = binding['damping_ratio']
    if slope == 0. and ratio == 0.:
        return (0., 0.)
    if metric is None:
        return None
    omega = 2. * math.pi * binding['frequency_hz']
    return (slope * metric * omega ** 2, 2. * ratio * metric * omega)
