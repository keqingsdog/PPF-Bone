"""Static-plane eligibility for a carrier attached after root interpolation.

This is a producer guard, not a replacement for native wall/CCD validation.
Coordinates are solver-space metres. A local point already includes the fixed
reference transform D. Authored translation is linear; rotation follows the
native shortest-slerp convention. No sampling grid is used to approve clearance.
"""
import math
import numpy as np


def _vector(value, size, field):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"root sweep: {field} requires finite length {size}")
    return result


def _quaternion(value, field):
    result = _vector(value, 4, field)
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm) or abs(norm - 1.0) > 1e-6:
        raise ValueError(f"root sweep: {field} must be a unit quaternion")
    return result / norm


def _rotation(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


def minimum_signed_plane_distance(t0, q0, t1, q1, local_point,
                                  plane_position, plane_normal):
    """Bound min n·(lerp(t0,t1,u)+R(u)local_point-plane), u in [0,1].

    For a normal slerp segment the distance is a+b*u+A*cos(theta*u)+
    B*sin(theta*u). All stationary points and both endpoints are evaluated.
    The native extremely-small-angle branch uses normalized quaternion lerp;
    its full rotation-arc diameter bounds its deviation from exact slerp.
    This intentionally conservative bound affects eligibility only.
    """
    t0 = _vector(t0, 3, 't0')
    t1 = _vector(t1, 3, 't1')
    q0 = _quaternion(q0, 'q0')
    q1 = _quaternion(q1, 'q1')
    point = _vector(local_point, 3, 'local_point')
    plane = _vector(plane_position, 3, 'plane_position')
    normal = _vector(plane_normal, 3, 'plane_normal')
    norm = float(np.linalg.norm(normal))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError('root sweep: plane_normal must be nonzero')
    normal = normal / norm
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(0.0, dot))
    # conj(q0) * q1; the axis is in q0's local coordinates.
    a0, v0 = q0[0], q0[1:]
    a1, v1 = q1[0], q1[1:]
    relative_vector = a0*v1 - a1*v0 - np.cross(v0, v1)
    relative_w = float(a0*a1 + np.dot(v0, v1))
    sine = float(np.linalg.norm(relative_vector))
    theta = 2.0 * math.atan2(sine, max(0.0, relative_w))
    rotation0 = _rotation(q0)
    slope = float(np.dot(normal, t1-t0))
    if sine == 0.0:
        start = float(np.dot(normal, t0 + rotation0 @ point - plane))
        end = start + slope
        u = 0.0 if start <= end else 1.0
        minimum = min(start, end)
        result = {'minimum_m': minimum, 'lower_bound_m': minimum,
                  'minimum_u': u, 'angle_rad': 0.0,
                  'near_linear_bound_m': 0.0, 'stationary_points': 0}
    else:
        axis = relative_vector / sine
        parallel = axis * float(np.dot(axis, point))
        perpendicular = point - parallel
        constant = float(np.dot(normal, t0-plane + rotation0 @ parallel))
        cosine = float(np.dot(normal, rotation0 @ perpendicular))
        sine_coefficient = float(np.dot(normal, rotation0 @ np.cross(axis, perpendicular)))
        amplitude = math.hypot(cosine, sine_coefficient)
        candidates = [0.0, 1.0]
        if amplitude > 0.0:
            ratio = -slope / (theta * amplitude)
            if -1.0 <= ratio <= 1.0:
                phase = math.atan2(cosine, sine_coefficient)
                arc = math.acos(ratio)
                for sign in (-1.0, 1.0):
                    base_angle = sign*arc - phase
                    first = math.ceil(-base_angle / math.tau)
                    last = math.floor((theta-base_angle) / math.tau)
                    for turn in range(first, last+1):
                        u = (base_angle + turn*math.tau) / theta
                        if 0.0 <= u <= 1.0:
                            candidates.append(u)
        def distance(u):
            angle = theta*u
            return constant + slope*u + cosine*math.cos(angle) + sine_coefficient*math.sin(angle)
        values = [distance(u) for u in candidates]
        best = min(range(len(values)), key=values.__getitem__)
        minimum = values[best]
        # Exact native condition (joint_root_motion_wire.rs::shortest_slerp).
        # Both interpolation rotations lie on the same shortest arc. Its full
        # diameter bounds the difference at every common translation parameter.
        near_bound = (2.0*float(np.linalg.norm(perpendicular))*math.sin(theta/2.0)
                      if dot > 1.0-1e-12 else 0.0)
        result = {'minimum_m': minimum, 'lower_bound_m': minimum-near_bound,
                  'minimum_u': candidates[best], 'angle_rad': theta,
                  'near_linear_bound_m': near_bound,
                  'stationary_points': len(candidates)-2}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError('root sweep: nonfinite derived bound')
    return result


def float32_clearance_margin(t0, t1, local_points, plane_position, gap):
    """Conservative coordinate scale for the existing 16-epsilon32 margin."""
    t0 = _vector(t0, 3, 't0')
    t1 = _vector(t1, 3, 't1')
    plane = _vector(plane_position, 3, 'plane_position')
    points = np.asarray(local_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
        raise ValueError('root sweep: local_points requires finite nonempty Nx3')
    gap = float(gap)
    if not math.isfinite(gap) or gap < 0.0:
        raise ValueError('root sweep: contact gap must be finite and nonnegative')
    radius = float(np.max(np.linalg.norm(points, axis=1)))
    scale = max(1.0, float(np.max(np.abs(t0))), float(np.max(np.abs(t1)))) + radius
    scale = max(scale, float(np.max(np.abs(plane))), gap)
    if not math.isfinite(scale):
        raise ValueError('root sweep: nonfinite coordinate scale')
    return 16.0 * np.finfo(np.float32).eps * scale
