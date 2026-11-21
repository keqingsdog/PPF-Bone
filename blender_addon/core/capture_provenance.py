"""Fail-closed provenance for captured solver INPUTS, never playback output.

PC2 stays unchanged. A payload-bound JSON proof and a persistent Scene binding
identify the captured facts. A pending marker rejects interrupted paired commits;
ordinary failures restore disk backups, not in-memory copies of large old files.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import tempfile
import uuid

import numpy as np

SCHEMA = 'ppf.capture-input.v1'
BINDINGS = 'ppf_capture_proof_bindings_v1'
PROOF_SUFFIX = '.capture.json'
PENDING_SUFFIX = '.capture.pending'
GROUP_PENDING_NAME = '.ppf-capture-group.pending'


def _assert_group_idle(parent):
    if os.path.lexists(os.path.join(os.fspath(parent), GROUP_PENDING_NAME)):
        raise ValueError('A mixed capture transaction is incomplete or uncertain; cache directory is blocked')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')


def _payload_digest(frames):
    return hashlib.sha256(memoryview(np.ascontiguousarray(frames, dtype='<f')).cast('B')).hexdigest()


def _proof_digest(proof):
    return hashlib.sha256(_canonical({k: v for k, v in proof.items() if k != 'proof_sha256'})).hexdigest()


def make_proof(obj, kind, entry):
    """Facts come from the completed frozen entry, never inferred on reload."""
    return dict(schema=SCHEMA, status='complete', generation=uuid.uuid4().hex,
                object_uuid=entry['obj_uuid'], kind=kind,
                frame_start=int(entry['frame_start']), frame_end=int(entry['frame_end']),
                vertex_indices=[int(i) for i in entry['pin_indices']] if kind == 'PIN' else None)


def bindings(scene):
    try:
        value = json.loads(scene.get(BINDINGS, '{}'))
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def bind(scene, key, proof):
    values = bindings(scene)
    if proof and proof.get('status') == 'complete':
        values[key] = proof['proof_sha256']
    else:
        values.pop(key, None)
    scene[BINDINGS] = json.dumps(values, sort_keys=True)


def _remove_temp(path):
    if path:
        try:
            os.unlink(path)
        except OSError:
            pass  # Retained owned staging evidence is safer than hiding a primary error.


def _temp(parent, suffix):
    fd, name = tempfile.mkstemp(prefix='.ppf-capture-', suffix=suffix, dir=parent)
    os.close(fd)
    return name


def _regular_or_absent(path):
    if os.path.lexists(path) and not stat.S_ISREG(os.lstat(path).st_mode):
        raise ValueError('Capture destination is not an ordinary file: ' + path)


def _commit_pair(path, writer, proof_bytes, publish=None, rollback=None):
    """No global FS atomicity claim: marker + proof binding reject uncertainty."""
    path = os.fspath(path)
    proof_path, pending = path + PROOF_SUFFIX, path + PENDING_SUFFIX
    for target in (path, proof_path, pending):
        _regular_or_absent(target)
    parent = os.path.dirname(path)
    _assert_group_idle(parent)
    os.makedirs(parent, exist_ok=True)
    staged = backups = None
    created_pending = False
    started = False
    rollback_failed = False
    try:
        staged = [_temp(parent, '.pc2'), _temp(parent, '.json')]
        writer(staged[0])
        with open(staged[0], 'r+b') as handle:
            os.fsync(handle.fileno())
        with open(staged[1], 'wb') as handle:
            handle.write(proof_bytes if proof_bytes is not None else b'')
            handle.flush(); os.fsync(handle.fileno())
        backups = []
        for target in (path, proof_path):
            backup = _temp(parent, '.backup') if os.path.exists(target) else None
            backups.append(backup)
            if backup:
                shutil.copy2(target, backup)
        try:
            with open(pending, 'xb') as handle:
                created_pending = True
                handle.write(b'Capture pair commit incomplete; recapture required.\n')
                handle.flush(); os.fsync(handle.fileno())
        except FileExistsError:
            pass  # Explicit recapture may recover a previous uncertain pair.
        started = True
        os.replace(staged[0], path)
        if proof_bytes is not None:
            os.replace(staged[1], proof_path)
        elif os.path.exists(proof_path):
            os.unlink(proof_path)
        if publish:
            publish()
        os.unlink(pending)
    except BaseException as error:
        if started:
            failures = []
            for target, backup in zip((path, proof_path), backups):
                try:
                    if backup is not None:
                        os.replace(backup, target)
                    elif os.path.exists(target):
                        os.unlink(target)
                except Exception as exc:
                    failures.append(str(exc))
            if rollback:
                try:
                    rollback()
                except Exception as exc:
                    failures.append(str(exc))
            rollback_failed = bool(failures)
            if not rollback_failed and created_pending:
                _remove_temp(pending)
            if rollback_failed:
                raise RuntimeError('Capture commit/rollback uncertain; pending marker and disk backups retained; recapture required: ' + '; '.join(failures)) from error
        elif created_pending:
            _remove_temp(pending)
        raise
    finally:
        for temp in staged or ():
            _remove_temp(temp)
        if not rollback_failed:
            for backup in backups or ():
                _remove_temp(backup)


def commit_capture(path, frames, proof=None, *, publish=None, rollback=None):
    from .pc2 import write_pc2
    arr = np.ascontiguousarray(frames, dtype='<f')
    if arr.ndim != 3 or arr.shape[2] != 3 or not arr.shape[0] or not arr.shape[1] or not np.isfinite(arr).all():
        raise ValueError('Captured input needs a nonempty finite frame/vertex/XYZ array')
    record = dict(proof) if proof is not None else dict(schema=SCHEMA, status='unproven')
    if proof is not None and record['frame_end'] - record['frame_start'] + 1 != arr.shape[0]:
        raise ValueError('Captured input range does not match its sampled rows')
    record.update(shape=list(arr.shape), payload_sha256=_payload_digest(arr))
    record['proof_sha256'] = _proof_digest(record)
    _commit_pair(path, lambda temp: write_pc2(temp, arr), _canonical(record),
                 publish=(lambda: publish(record)) if publish else None, rollback=rollback)
    return record


def read_bound_proof(path, frames=None):
    """Read/verify only this actual input pair; never manufacture missing proof."""
    path = os.fspath(path)
    _assert_group_idle(os.path.dirname(path))
    if os.path.exists(path + PENDING_SUFFIX):
        raise ValueError('capture commit is incomplete or uncertain')
    try:
        with open(path + PROOF_SUFFIX, encoding='utf8') as handle:
            proof = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError('legacy or missing capture provenance; origin and vertex membership are unproven') from exc
    if not isinstance(proof, dict) or proof.get('schema') != SCHEMA or proof.get('status') != 'complete':
        raise ValueError('capture provenance is unproven or unsupported')
    if proof.get('proof_sha256') != _proof_digest(proof):
        raise ValueError('capture provenance was changed or damaged')
    shape = proof.get('shape')
    if not isinstance(shape, list) or len(shape) != 3 or any(type(n) is not int or n < 1 for n in shape) or shape[2] != 3:
        raise ValueError('capture provenance has an invalid shape')
    if any(type(proof.get(k)) is not int for k in ('frame_start', 'frame_end')) or proof['frame_end'] - proof['frame_start'] + 1 != shape[0]:
        raise ValueError('capture provenance has an invalid frame range')
    digest = hashlib.sha256()
    expected = shape[0] * shape[1] * 12
    with open(path, 'rb') as handle:
        header = handle.read(32)
        if len(header) != 32:
            raise ValueError('capture PC2 header is incomplete')
        magic, version, vertices, start, sampling, count = struct.unpack('<12siiffi', header)
        if (magic, version, vertices, start, sampling, count) != (b'POINTCACHE2\0', 1, shape[1], 0.0, 1.0, shape[0]):
            raise ValueError('capture PC2 header does not match its proof')
        size = 0
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk); digest.update(chunk)
        if size != expected or digest.hexdigest() != proof.get('payload_sha256'):
            raise ValueError('capture PC2 payload does not match its proof')
    if frames is not None and (list(frames.shape) != shape or not np.isfinite(frames).all() or _payload_digest(frames) != proof['payload_sha256']):
        raise ValueError('capture memory and proven disk payload differ')
    return proof


def validate_capture(scene, obj, frames, kind, vg_name=None, pin_item=None):
    from . import pc2
    from .uuid_registry import get_object_uuid
    from ..ui.dynamics.static_deform_ops import _effective_frame_range
    key = pc2.static_deform_pc2_key(obj) if kind == 'STATIC' else pc2.pin_anim_pc2_key(obj, vg_name)
    try:
        proof = read_bound_proof(pc2.get_pc2_path(key), frames)
        if bindings(scene).get(key) != proof['proof_sha256']:
            raise ValueError('this Scene cannot prove ownership of this capture generation')
        if proof.get('object_uuid') != get_object_uuid(obj) or proof.get('kind') != kind:
            raise ValueError('capture object or kind differs from the proven source')
        if kind == 'PIN' and pin_item is not None:
            from ..ui.dynamics.pin_capture_ops import _effective_pin_frame_range
            start, end = _effective_pin_frame_range(scene, obj, pin_item)[:2]
            if proof['frame_start'] != start or proof['frame_end'] < end:
                raise ValueError('capture does not cover the required Pin frame range')
        else:
            current = tuple(_effective_frame_range(scene, obj)[:2])
            recorded = (proof['frame_start'], proof['frame_end'])
            if recorded != current:
                if recorded[0] != current[0] or recorded[1] < current[1]:
                    raise ValueError('effective start/frame range changed after capture')
                legacy = tuple(_effective_frame_range(scene, obj, legacy_uncapped=True)[:2])
                if recorded != legacy:
                    raise ValueError('effective start/frame range changed after capture')
        if kind == 'PIN':
            from .encoder.pin import _get_pin_indices
            indices = proof.get('vertex_indices')
            if not isinstance(indices, list) or any(type(i) is not int for i in indices) or indices != list(_get_pin_indices(obj, vg_name)):
                raise ValueError('Pin vertex identities/order changed after capture')
        return proof
    except (OSError, ValueError, TypeError, KeyError) as exc:
        label = obj.name + (" / Pin '" + vg_name + "'" if vg_name else '')
        raise ValueError(f"Captured input '{label}' is stale or unproven: {exc}. Recapture Deformation before export/solve; existing files were retained.") from exc


def copy_pair(source, destination):
    """Save As copies a known pair verbatim; never re-stamps current settings."""
    source, destination = os.fspath(source), os.fspath(destination)
    _assert_group_idle(os.path.dirname(source))
    _assert_group_idle(os.path.dirname(destination))
    if os.path.realpath(source) == os.path.realpath(destination):
        return
    if os.path.exists(source + PENDING_SUFFIX):
        raise ValueError('Cannot migrate an uncertain capture pair; recapture required')
    proof_path = source + PROOF_SUFFIX
    proof_bytes = Path(proof_path).read_bytes() if os.path.exists(proof_path) else None
    if os.path.exists(destination) or os.path.exists(destination + PROOF_SUFFIX):
        source_proof = read_bound_proof(source)
        target_proof = read_bound_proof(destination)
        if source_proof['proof_sha256'] != target_proof['proof_sha256']:
            raise ValueError('Save As capture target already holds a different generation; target retained')
        return
    _commit_pair(destination, lambda temp: shutil.copyfile(source, temp), proof_bytes)


def validate_static_param_inputs(scene, groups):
    """Parameter-only updates must not reinterpret a stale STATIC input origin."""
    from . import pc2
    from .uuid_registry import resolve_assigned
    for group in groups:
        if group.object_type != 'STATIC':
            continue
        for assigned in group.assigned_objects:
            if not assigned.included:
                continue
            obj = resolve_assigned(assigned)
            if obj is None or obj.type != 'MESH':
                continue
            frames = pc2.get_static_deform_cache(obj)
            if frames is not None:
                validate_capture(scene, obj, frames, 'STATIC')
