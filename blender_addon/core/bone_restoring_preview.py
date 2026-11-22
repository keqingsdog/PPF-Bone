"""Author-time angular estimates from the validated rest proxy and mass cache."""
import math

import numpy as np
from mathutils import Euler, Matrix, Quaternion

from .bone_mass import cached_rest_mass
from .bone_physical_units import linear_constants
from ..models.bone_bindings import _frame, rigid_matrix, similarity_frame_scale, bone_world_matrix

_fallback_cache = {}


def _proxy_arm_matrix(arm, proxy):
    """Use authored parent-local transforms without evaluated bone pose."""
    for obj in (proxy, proxy.parent if proxy.parent != arm else None):
        if obj is not None and (any(not c.mute for c in obj.constraints)
                                or (obj.animation_data and (obj.animation_data.action
                                    or obj.animation_data.drivers or obj.animation_data.nla_tracks))):
            raise ValueError("Animated or constrained proxy has no stable author reference")
    local = proxy.matrix_parent_inverse @ proxy.matrix_basis
    if proxy.parent == arm:
        return local
    parent = proxy.parent
    if parent is not None and parent.parent == arm and parent.type == 'EMPTY':
        return (parent.matrix_parent_inverse @ parent.matrix_basis) @ local
    raise ValueError("Proxy parent has no supported stable author reference")


def _angular_mobility(scene, arm, bone, axes, world_scale, *, evaluated=False):
    settings = bone.ppf_bone_physics
    if settings.root_mode in {'STATIC', 'KINEMATIC'}:
        return np.zeros(3)
    measured = cached_rest_mass(scene, arm, bone)
    if measured is None:
        return None

    matrix = settings.proxy.matrix_world if evaluated else _proxy_arm_matrix(arm, settings.proxy)
    transform = np.asarray(matrix.to_3x3(), dtype=np.float64) * world_scale
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


def _joint_axes(arm, bone, *, evaluated=False):
    settings = bone.ppf_bone_physics
    if settings.use_auto_frames:
        orientation = (bone_world_matrix(arm, bone).to_3x3() if evaluated else
                       bone.matrix_local.to_quaternion().normalized().to_matrix())
        return (orientation
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
        rigid_matrix(parent_settings.proxy.matrix_world, parent.name)
        parent_world = (parent_settings.proxy.matrix_world if evaluated else
                        _proxy_arm_matrix(arm, parent_settings.proxy).to_quaternion().normalized().to_matrix().to_4x4())
    else:
        parent_world = bone_world_matrix(arm, parent) if evaluated else parent.matrix_local
    _frame(settings, 'child')
    parent_frame = _frame(settings, 'parent')
    return parent_world.to_3x3() @ Quaternion(parent_frame['quaternion_wxyz']).to_matrix()



def _fallback_proxy_signature(proxy):
    """Author configuration only: exclude evaluated transforms of animated objects."""
    result = []
    seen = set()
    obj = proxy
    while obj is not None and obj.as_pointer() not in seen:
        seen.add(obj.as_pointer())
        ad = obj.animation_data
        animated = bool(ad and (ad.action or ad.drivers or ad.nla_tracks))
        from .utils import get_id_fcurves
        animated_paths = set()
        if ad:
            if ad.action:
                animated_paths.update(fc.data_path for fc in get_id_fcurves(obj))
            animated_paths.update(fc.data_path for fc in ad.drivers)
        constraints = []
        for constraint in obj.constraints:
            values = []
            for prop in constraint.bl_rna.properties:
                if prop.identifier in {'rna_type', 'name'} or prop.is_readonly:
                    continue
                if constraint.path_from_id(prop.identifier) in animated_paths:
                    continue
                value = getattr(constraint, prop.identifier)
                if prop.type == 'POINTER':
                    value = value.as_pointer() if value is not None else 0
                elif getattr(prop, 'is_array', False):
                    value = tuple(value)
                elif prop.type not in {'BOOLEAN', 'INT', 'FLOAT', 'ENUM', 'STRING'}:
                    continue
                values.append((prop.identifier, value))
            constraints.append((constraint.type, tuple(values)))
        result.append((obj.as_pointer(), obj.parent.as_pointer() if obj.parent else 0,
                       obj.parent_type, obj.parent_bone, animated,
                       tuple(float(x) for row in obj.matrix_parent_inverse for x in row),
                       None if animated else tuple(float(x) for row in obj.matrix_basis for x in row),
                       tuple(constraints)))
        obj = obj.parent
    return tuple(result)

def angular_input_signature(context, arm, bone):
    """Small author/rest inputs; cache values are invalidated by the mass producer."""
    if arm is None or bone is None or bone.parent is None:
        return None
    try:
        _, scale = similarity_frame_scale(arm.matrix_world, arm.name)
        rows = []
        for item in (bone, bone.parent):
            s = item.ppf_bone_physics
            mass = cached_rest_mass(context.scene, arm, item) if s.enabled else None
            mass_signature = None if mass is None else (float(mass['mass_kg']),
                tuple(float(x) for field in ('point_second_moment', 'solid_second_moment', 'solid_centered_second_moment')
                      for row in mass[field] for x in row))
            try:
                proxy_signature = ('rest', tuple(float(x) for row in _proxy_arm_matrix(arm, s.proxy) for x in row)) if s.proxy else None
            except ValueError:
                proxy_signature = ('fallback', _fallback_proxy_signature(s.proxy))
            rows.append((item.as_pointer(), s.enabled, s.root_mode, s.physical_model, s.mass_mode,
                         float(s.density), float(s.target_mass_kg), s.material_preset,
                         tuple(float(x) for row in item.matrix_local for x in row),
                         proxy_signature, mass_signature))
        s = bone.ppf_bone_physics
        return (float(scale), tuple(rows), s.use_auto_frames, tuple(s.joint_axis_euler),
                repr(_frame(s, 'parent')) if not s.use_auto_frames else None,
                repr(_frame(s, 'child')) if not s.use_auto_frames else None)
    except (ValueError, RuntimeError, ReferenceError, AttributeError, np.linalg.LinAlgError):
        return None


def angular_mass_pending(context, arm, bone):
    if arm is None or bone is None or bone.parent is None:
        return False
    from . import bone_mass
    for item in (bone, bone.parent):
        s = item.ppf_bone_physics
        if s.enabled and s.root_mode not in {'STATIC', 'KINEMATIC'}:
            key = bone_mass._key(context.scene, arm, item)
            if (not bone_mass.CACHE.clean(key)
                    or bone_mass._metadata_snapshot(s) != bone_mass._snapshots.get(key)):
                return True
    return False


def angular_metrics(context, arm, bone):
    """Per-axis effective inertias for a hierarchy joint in its rest pose."""
    if arm is None or bone.parent is None:
        return None
    scene = context.scene
    signature = angular_input_signature(context, arm, bone)
    cache_key = (context.scene.as_pointer(), arm.as_pointer(), bone.as_pointer())
    prior = _fallback_cache.get(cache_key)
    if prior is not None and prior[0] == signature:
        return prior[1]
    try:
        _, world_scale = similarity_frame_scale(arm.matrix_world, arm.name)
        try:
            axes = np.asarray(_joint_axes(arm, bone), dtype=np.float64)
            mobility = _angular_mobility(scene, arm, bone, axes, world_scale)
            parent = bone.parent
            parent_mobility = (_angular_mobility(scene, arm, parent, axes, world_scale)
                               if parent.ppf_bone_physics.enabled else np.zeros(3))
            fallback = False
        except ValueError:
            # Legacy hierarchy/constraints: freeze the first valid evaluated
            # estimate for this author configuration instead of gating Auto.
            axes = np.asarray(_joint_axes(arm, bone, evaluated=True), dtype=np.float64)
            mobility = _angular_mobility(scene, arm, bone, axes, 1., evaluated=True)
            parent = bone.parent
            parent_mobility = (_angular_mobility(scene, arm, parent, axes, 1., evaluated=True)
                               if parent.ppf_bone_physics.enabled else np.zeros(3))
            fallback = True
        if mobility is None or parent_mobility is None:
            return None
        mobility += parent_mobility
        if not np.isfinite(mobility).all() or np.any(mobility <= 0.):
            return None
        result = tuple(float(1. / value) for value in mobility)
        if fallback and signature is not None:
            _fallback_cache[cache_key] = (signature, result)
        else:
            _fallback_cache.pop(cache_key, None)
        return result
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
