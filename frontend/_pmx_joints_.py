"""Bind Blender-authored PMX frames to the solver's rest-centered rigids."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np

from ._pdrd_native_frames_ import _quat, _rotation, native_body_frames


FEATURE = 'pmx_joints_v1'
FILE = 'bin/pmx_joints_v1.json'
KIND = 'ppf.pmx_joints'
FIELDS = {'joint_uuid', 'body_a_uuid', 'body_b_uuid', 'frame_a', 'frame_b',
          'translation', 'rotation', 'linear_spring_N_per_m', 'angular_spring_Nm_per_rad'}


def fail(detail):
    raise ValueError(f'PMX_JOINT_INVALID: {detail}')


def _number(value):
    return type(value) in (int, float) and -math.inf < value < math.inf


def _vector(value, count):
    return isinstance(value, list) and len(value) == count and all(map(_number, value))


def _axes(axes):
    if not isinstance(axes, dict) or set(axes) != {'x', 'y', 'z'}:
        fail('expected x, y, z axis records')
    for axis in axes.values():
        if not isinstance(axis, dict):
            fail('expected a PMX axis record')
        mode = axis.get('mode')
        if (mode == 'free' and set(axis) == {'mode'}
                or mode == 'locked' and set(axis) == {'mode', 'value'} and _number(axis['value'])
                or mode == 'limited' and set(axis) == {'mode', 'lower', 'upper'}
                and _number(axis['lower']) and _number(axis['upper']) and axis['lower'] < axis['upper']):
            continue
        fail('invalid free, locked, or limited axis')


def validate_author(rows, bindings):
    from ._blender_bones_ import canonical_uuid

    if not isinstance(rows, list) or not 0 < len(rows) <= 65536:
        fail('expected 1–65536 joints')
    by_body = {row['body_uuid']: row for row in bindings}
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != FIELDS:
            fail('unexpected or missing joint fields')
        uid = canonical_uuid(row['joint_uuid'], 'PMX joint_uuid')
        if uid in seen:
            fail(f'duplicate joint {uid}')
        seen.add(uid)
        body_a = canonical_uuid(row['body_a_uuid'], 'PMX body_a_uuid')
        body_b = canonical_uuid(row['body_b_uuid'], 'PMX body_b_uuid')
        if body_a == body_b or body_a not in by_body or body_b not in by_body:
            fail(f'{uid}: expected two different active bodies')
        if by_body[body_a]['armature_uuid'] != by_body[body_b]['armature_uuid']:
            fail(f'{uid}: endpoints belong to different armatures')
        for key in ('frame_a', 'frame_b'):
            frame = row[key]
            if (not isinstance(frame, dict) or set(frame) != {'translation_m', 'quaternion_wxyz'}
                    or not _vector(frame['translation_m'], 3) or not _vector(frame['quaternion_wxyz'], 4)):
                fail(f'{uid}: invalid {key}')
            _quat(frame['quaternion_wxyz'], f'{uid}.{key}.quaternion_wxyz')
        _axes(row['translation'])
        _axes(row['rotation'])
        for key in ('linear_spring_N_per_m', 'angular_spring_Nm_per_rad'):
            if not _vector(row[key], 3) or min(row[key]) < 0:
                fail(f'{uid}: expected nonnegative {key} values')
    return by_body


def capability(contract):
    from ._r14_product_profile import solver_child_env

    profile = contract.profile
    message = 'PMX_JOINT_CAPABILITY_UNAVAILABLE: select a solver that supports pmx_joints_v1 and Build again'
    try:
        result = subprocess.run(
            [str(contract.exe_path), '--pmx-joints-capability-json'],
            cwd=profile.runtime_dir, env=solver_child_env(profile, os.environ),
            capture_output=True, timeout=30, check=True)
        document = json.loads(result.stdout)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise ValueError(message) from exc
    expected = {'kind': KIND + '.capability', 'version': 1, 'required_feature': FEATURE,
                'supported': True, 'semantic_version': 1, 'model': 'COMPLIANT_EULER_XYZ_BOX'}
    if (not isinstance(document, dict) or any(document.get(key) != value for key, value in expected.items())
            or type(document['version']) is not int or type(document['semantic_version']) is not int
            or document['supported'] is not True):
        raise ValueError(message)


def install(fixed, rows, bindings, receipt):
    from ._blender_bones_ import S, frame_matrix, local_frame, matrix

    by_body = validate_author(rows, bindings)
    native = native_body_frames(fixed)
    joints = deepcopy(rows)
    for joint in joints:
        for endpoint in ('a', 'b'):
            uid = joint[f'body_{endpoint}_uuid']
            if uid not in native:
                fail(f'{joint["joint_uuid"]}: endpoint {uid} is not a native PDRD body')
            proxy = matrix(by_body[uid]['proxy_world_matrix_blender'], 'proxy_world_matrix_blender')
            frame = joint[f'frame_{endpoint}']
            proxy_local = np.eye(4)
            proxy_local[:3, :3] = _rotation(_quat(frame['quaternion_wxyz'], 'PMX proxy-local rotation'))
            proxy_local[:3, 3] = frame['translation_m']
            joint[f'frame_{endpoint}'] = local_frame(frame_matrix(native[uid]), S @ proxy @ proxy_local)
    fixed._pmx_joints_payload = {'version': 1, 'kind': KIND, 'scene_uuid': fixed._scene_uuid, 'joints': joints}
    receipt['required_features'] = sorted(set(receipt.get('required_features', []) + [FEATURE]))


def write_sidecar(fixed, path):
    target = Path(path) / FILE
    payload = getattr(fixed, '_pmx_joints_payload', None)
    if payload is None:
        if target.exists():
            fail('orphan PMX sidecar; Build in a fresh directory')
        return False
    if payload['scene_uuid'] != fixed._scene_uuid or fixed._pdrd_ball_anchor_payload is None:
        fail('PMX sidecar requires the current scene joint carrier')
    data = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    if len(data) > 16 * 1024 * 1024:
        fail('PMX sidecar exceeds 16 MiB')
    target.write_bytes(data)
    return True


def require_wire_presence(path, features):
    present = FEATURE in features
    if present != (Path(path) / FILE).is_file():
        fail('PMX sidecar and joint feature must be present together')
    return present
