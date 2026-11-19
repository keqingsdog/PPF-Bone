"""Bind authored joints to the initial rigid frames of a built PPF scene.

The native PDRD fit uses already placed world vertices against f32
``pdrd_rest_centered`` points. An object transform is therefore not generally
the native rigid frame. This adapter is for input authoring; its fit tolerance
checks f32 export roundoff at scene scale and is not a G1 accuracy threshold.
"""
from __future__ import annotations

from copy import deepcopy
import math
import uuid
from collections.abc import Mapping

import numpy as np


WORLD_UUID = "00000000-0000-0000-0000-000000000000"
_F32_EPS = np.finfo(np.float32).eps


def _uuid(value, label):
    try:
        result = str(uuid.UUID(str(value)))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f"{label}: invalid UUID") from exc
    if result != value:
        raise ValueError(f"{label}: UUID must be canonical")
    return result


def _vector(value, length, label):
    try:
        a = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label}: expected numeric vector") from exc
    if a.shape != (length,) or not np.isfinite(a).all():
        raise ValueError(f"{label}: expected {length} finite numbers")
    return a


def _quat(value, label):
    q = _vector(value, 4, label)
    norm = float(np.linalg.norm(q))
    if abs(norm - 1.0) > 1e-6:
        raise ValueError(f"{label}: nonunit quaternion")
    return q / norm


def _rotation(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)


def _matrix_quat(R):
    """Proper active rotation matrix to canonical unit wxyz quaternion."""
    trace = float(np.trace(R))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = np.array([s / 4, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s,
                      (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(max(0.0, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2
        if s <= 0:
            raise ValueError("native rotation: quaternion conversion failed")
        q = np.empty(4)
        q[i + 1] = s / 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[j + 1] = (R[j, i] + R[i, j]) / s
        q[k + 1] = (R[k, i] + R[i, k]) / s
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dtype=np.float64)


def _frame(table, body_uuid, label):
    if body_uuid == WORLD_UUID:
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
    if body_uuid not in table:
        raise ValueError(f"{label}: missing body frame {body_uuid}")
    frame = table[body_uuid]
    if isinstance(frame, Mapping):
        t = frame.get("world_translation_m")
        q = frame.get("world_quaternion_wxyz")
    else:
        t = getattr(frame, "world_translation_m", None)
        q = getattr(frame, "world_quaternion_wxyz", None)
    return _vector(t, 3, f"{label}.world_translation_m"), _quat(
        q, f"{label}.world_quaternion_wxyz")


def native_body_frames(fixed) -> dict[str, dict]:
    """Fit native initial ``(centroid, proper R)`` for every UUID runtime body.

    The body-row start/count index the flat vertex list and aligned f32 rest
    rows. Vertex membership comes from builder-owned UUID/runtime bindings.
    A fit is refused when its residual exceeds f32 rest roundoff relative to
    the body's baked size (plus a small world-coordinate subtraction floor).
    """
    rows_flat = np.asarray(fixed._pdrd_body_rows, dtype=np.float64)
    if rows_flat.ndim != 1 or rows_flat.size % 23:
        raise ValueError("PDRD body rows: expected flat 23-float records")
    rows = rows_flat.reshape(-1, 23)
    bindings = fixed._pdrd_body_bindings
    if len(bindings) != len(rows):
        raise ValueError("PDRD bindings: count differs from body rows")
    vertices = np.asarray(fixed.vertex(), dtype=np.float64)
    rest = np.asarray(fixed._pdrd_rest_centered, dtype=np.float64)
    ids_raw = np.asarray(fixed._pdrd_vert_list)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("PDRD world vertices: expected finite Nx3")
    # FixedScene deliberately stores rest=None and list=[] when the builder
    # has no PDRD bodies (including an all-FEM bone scene). Normalize only
    # that absent inventory; never hide populated or malformed rigid data.
    if len(rows) == 0 and ids_raw.shape == (0,):
        if fixed._pdrd_rest_centered is None:
            rest = np.empty((0, 3), dtype=np.float64)
        ids_raw = np.empty(0, dtype=np.int64)
    if rest.shape != (len(ids_raw), 3) or not np.isfinite(rest).all():
        raise ValueError("PDRD rest-centered rows: expected finite list-aligned Nx3")
    if ids_raw.ndim != 1 or not np.issubdtype(ids_raw.dtype, np.integer):
        raise ValueError("PDRD vertex list: expected integer indices")
    if np.any(ids_raw < 0) or np.any(ids_raw >= len(vertices)):
        raise ValueError("PDRD vertex list: out-of-range index")
    ids = ids_raw.astype(np.int64)
    if len(set(map(int, ids))) != len(ids):
        raise ValueError("PDRD vertex list: duplicate body vertex")
    vi = getattr(fixed, "_pdrd_vert_index", None)
    if vi is not None:
        vi = np.asarray(vi)
        if vi.shape != (len(vertices),) or not np.issubdtype(vi.dtype, np.integer):
            raise ValueError("PDRD vertex-index map: invalid shape/type")
        if len(rows) == 0 and np.any(vi != 0):
            raise ValueError("PDRD vertex-index map: body references without bodies")

    by_index = {}
    for binding in bindings:
        if not isinstance(binding, Mapping):
            raise ValueError("PDRD binding: expected mapping")
        body_uuid = _uuid(binding.get("body_uuid"), "PDRD body_uuid")
        if body_uuid == WORLD_UUID:
            raise ValueError("PDRD binding: world UUID is not a body")
        index = binding.get("runtime_body_index")
        count = binding.get("vertex_count")
        if type(index) is not int or type(count) is not int or index < 0 or count < 4:
            raise ValueError("PDRD binding: invalid runtime index or vertex count")
        if index in by_index or any(b[0] == body_uuid for b in by_index.values()):
            raise ValueError("PDRD binding: duplicate UUID/runtime index")
        by_index[index] = (body_uuid, count)
    if set(by_index) != set(range(len(rows))):
        raise ValueError("PDRD bindings: runtime indices must be contiguous")

    result = {}
    used_rest = np.zeros(len(ids), dtype=bool)
    for index in range(len(rows)):
        body_uuid, bound_count = by_index[index]
        start_f, count_f = rows[index, :2]
        if not np.isfinite([start_f, count_f]).all() or not float(start_f).is_integer() or not float(count_f).is_integer():
            raise ValueError(f"PDRD row {index}: nonintegral start/count")
        start, count = int(start_f), int(count_f)
        if start < 0 or count != bound_count or start + count > len(ids):
            raise ValueError(f"PDRD row {index}: invalid start/count or binding mismatch")
        if used_rest[start:start + count].any():
            raise ValueError(f"PDRD row {index}: overlapping rest slices")
        used_rest[start:start + count] = True
        body_ids = ids[start:start + count]
        if vi is not None and not np.all(vi[body_ids] == index + 1):
            raise ValueError(f"PDRD row {index}: vertex-index map disagrees")
        x = vertices[body_ids]
        Y = rest[start:start + count]
        c = x.mean(axis=0)
        gram = Y.T @ Y
        gram_s = np.linalg.svd(gram, compute_uv=False)
        if not np.isfinite(gram_s).all() or gram_s[0] <= 0 or gram_s[-1] <= gram_s[0] * 1e-10:
            raise ValueError(f"PDRD row {index}: degenerate rest geometry")
        M = (x - c).T @ Y
        if not np.isfinite(M).all():
            raise ValueError(f"PDRD row {index}: nonfinite covariance")
        U, _, Vt = np.linalg.svd(M)
        parity = float(np.linalg.det(U @ Vt))
        R = U @ np.diag([1.0, 1.0, 1.0 if parity > 0 else -1.0]) @ Vt
        if (not np.isfinite(R).all() or np.linalg.det(R) <= 0 or
                np.max(np.abs(R.T @ R - np.eye(3))) > 1e-8):
            raise ValueError(f"PDRD row {index}: invalid proper rotation")
        fit_error = float(np.max(np.abs(x - (c + (R @ Y.T).T))))
        baked_span = float(np.max(np.linalg.norm(Y, axis=1)))
        world_scale = max(1.0, float(np.max(np.abs(c))))
        # 16 f32 ULPs of baked rest extent plus f64 world subtraction floor.
        # The world-coordinate term uses actual f64 roundoff, so distant
        # scenes do not get a permissive macroscopic fit tolerance.
        fit_tolerance = float(max(16 * _F32_EPS * baked_span,
                                  64 * np.finfo(np.float64).eps * world_scale))
        if fit_error > fit_tolerance:
            raise ValueError(f"PDRD row {index}: visibly nonrigid initial fit "
                             f"{fit_error:g} m > {fit_tolerance:g} m")
        result[body_uuid] = {
            "world_translation_m": [float(v) for v in c],
            "world_quaternion_wxyz": [float(v) for v in _matrix_quat(R)],
            "runtime_body_index": index,
            "vertex_count": count,
            "fit_max_abs_error_m": fit_error,
            "fit_tolerance_m": fit_tolerance,
        }
    if not used_rest.all():
        raise ValueError("PDRD rows: unclaimed rest-centered vertices")
    return result


def rebind_joint_payload(payload, authored_frames, native_frames, *,
                         preserve_world_joint_orientations: bool) -> dict:
    """Deep-copy and re-express each authored joint endpoint in native frames.

    World anchors are first reconstructed from authored body pose and old local
    translation. Angular users also retain each authored world joint orientation
    by converting old local quaternions through the same world frame.
    """
    if type(preserve_world_joint_orientations) is not bool:
        raise ValueError("preserve_world_joint_orientations: expected bool")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("joints"), list):
        raise ValueError("joint payload: expected joints list")
    if not isinstance(authored_frames, Mapping) or not isinstance(native_frames, Mapping):
        raise ValueError("joint frames: expected UUID maps")
    result = deepcopy(payload)
    for joint in result["joints"]:
        if not isinstance(joint, Mapping):
            raise ValueError("joint payload: malformed joint")
        for endpoint in ("parent", "child"):
            body_uuid = _uuid(joint.get(f"{endpoint}_body_uuid"), f"{endpoint}_body_uuid")
            if endpoint == "child" and body_uuid == WORLD_UUID:
                raise ValueError("child_body_uuid: world cannot be child")
            old = joint.get(f"{endpoint}_local_frame")
            if not isinstance(old, Mapping):
                raise ValueError(f"{endpoint}_local_frame: expected mapping")
            authored_t, authored_q = _frame(authored_frames, body_uuid, "authored")
            native_t, native_q = _frame(native_frames, body_uuid, "native")
            old_t = _vector(old.get("translation_m"), 3, f"{endpoint}.translation_m")
            world_anchor = authored_t + _rotation(authored_q) @ old_t
            old["translation_m"] = [float(v) for v in
                                    (_rotation(native_q).T @ (world_anchor - native_t))]
            if preserve_world_joint_orientations:
                old_q = _quat(old.get("quaternion_wxyz"), f"{endpoint}.quaternion_wxyz")
                world_q = _qmul(authored_q, old_q)
                inverse_native = native_q * np.array([1.0, -1.0, -1.0, -1.0])
                new_q = _qmul(inverse_native, world_q)
                new_q /= np.linalg.norm(new_q)
                old["quaternion_wxyz"] = [float(v) for v in new_q]
    return result
