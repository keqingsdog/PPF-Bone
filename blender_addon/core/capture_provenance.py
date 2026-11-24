"""Fail-closed provenance for captured solver INPUTS, never playback output.

PC2 stays unchanged. A payload-bound JSON proof and a persistent Scene binding
identify the captured facts. A pending marker rejects interrupted paired commits;
ordinary failures restore disk backups, not in-memory copies of large old files.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import math
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


_transfer_observations = ContextVar('ppf_transfer_capture_files', default=None)


def _file_identity(st):
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


@contextmanager
def observe_transfer_capture_files(observations):
    """Only record reads performed synchronously by this Transfer tick."""
    token = _transfer_observations.set(observations)
    try:
        yield
    finally:
        _transfer_observations.reset(token)


def _record_verified_file(observations, path, path_initial, descriptor_initial, handle):
    descriptor_final = _file_identity(os.fstat(handle.fileno()))
    path_final = _file_identity(os.stat(path))
    # Windows CPython can report birth time for stat(path).st_ctime_ns
    # but change time for fstat(fd).st_ctime_ns. Use the full tuple only
    # within each API, and cross-check the shared identity fields.
    if (path_initial != path_final or descriptor_initial != descriptor_final
            or path_final[:4] != descriptor_final[:4]
            or observations.setdefault(path, (path_final, descriptor_final)) != (path_final, descriptor_final)):
        raise _capture_file_changed(path)


def _capture_file_changed(path):
    from ..i18n import pgettext_iface as iface_
    return ValueError(iface_("Capture file changed during Transfer: '{file}'. Start Transfer again.").format(
        file=os.path.basename(path),
    ))


def require_transfer_capture_files_current(observations):
    """Metadata comparison only; no payload read and no persistent descriptor."""
    for path, (path_identity, _) in observations.items():
        try:
            if _file_identity(os.stat(path)) != path_identity:
                raise ValueError('recorded file identity changed')
            if not path.endswith(PROOF_SUFFIX):
                _assert_group_idle(os.path.dirname(path))
                if os.path.lexists(path + PENDING_SUFFIX):
                    raise ValueError('capture commit is incomplete or uncertain')
        except (OSError, ValueError) as exc:
            raise _capture_file_changed(path) from exc


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


def _read_capture_proof_bytes(data):
    proof = json.loads(data)
    if not isinstance(proof, dict) or proof.get('schema') != SCHEMA or proof.get('status') != 'complete':
        raise ValueError('capture provenance is unproven or unsupported')
    if proof.get('proof_sha256') != _proof_digest(proof):
        raise ValueError('capture provenance was changed or damaged')
    shape = proof.get('shape')
    if not isinstance(shape, list) or len(shape) != 3 or any(type(n) is not int or n < 1 for n in shape) or shape[2] != 3:
        raise ValueError('capture provenance has an invalid shape')
    if any(type(proof.get(k)) is not int for k in ('frame_start', 'frame_end')) or proof['frame_end'] - proof['frame_start'] + 1 != shape[0]:
        raise ValueError('capture provenance has an invalid frame range')
    return proof


def _check_capture_payload(path, proof, destination=None, *, decode=False, observed=None):
    """Verify the same bytes copied to staging, in bounded blocks."""
    from contextlib import nullcontext
    digest = hashlib.sha256() if proof is not None else None
    with open(path, 'rb') as handle:
        if observed is not None:
            path_initial = _file_identity(os.stat(path))
            descriptor_initial = _file_identity(os.fstat(handle.fileno()))
        header = handle.read(32)
        if len(header) != 32:
            raise ValueError('capture PC2 header is incomplete')
        magic, version, vertices, start, sampling, count = struct.unpack('<12siiffi', header)
        if (magic != b'POINTCACHE2\0' or version != 1 or vertices < 1 or count < 1
                or not math.isfinite(start) or not math.isfinite(sampling) or sampling <= 0):
            raise ValueError('capture PC2 header is invalid')
        if proof is not None and (vertices, start, sampling, count) != (proof['shape'][1], 0.0, 1.0, proof['shape'][0]):
            raise ValueError('capture PC2 header does not match its proof')
        expected = count * vertices * 12
        if os.fstat(handle.fileno()).st_size != 32 + expected:
            raise ValueError('capture PC2 has the wrong payload length')
        # A cold Scene decode consumes the exact chunks that feed the proof
        # digest. Publish no array until the complete stream is verified.
        if decode and (proof is None or destination is not None):
            raise ValueError('capture decode requires a proof and no copy destination')
        raw = bytearray(expected) if decode else None
        with (open(destination, 'wb') if destination is not None else nullcontext()) as output:
            if output is not None:
                output.write(header)
            left = expected
            while left:
                chunk = handle.read(min(left, 1024 * 1024))
                if not chunk:
                    raise ValueError('capture PC2 ended during validation')
                if digest is not None:
                    digest.update(chunk)
                if output is not None:
                    output.write(chunk)
                if raw is not None:
                    raw[expected-left:expected-left+len(chunk)] = chunk
                left -= len(chunk)
            if handle.read(1):
                raise ValueError('capture PC2 changed during validation')
        if observed is not None:
            _record_verified_file(observed, path, path_initial, descriptor_initial, handle)
    if digest is not None and digest.hexdigest() != proof.get('payload_sha256'):
        raise ValueError('capture PC2 payload does not match its proof')
    return np.frombuffer(raw, dtype='<f').reshape(count, vertices, 3) if raw is not None else None


def read_bound_proof(path, frames=None, *, decode=False, observed=None):
    """Read/verify only this actual input pair; never manufacture missing proof."""
    path = os.fspath(path)
    _assert_group_idle(os.path.dirname(path))
    if os.path.exists(path + PENDING_SUFFIX):
        raise ValueError('capture commit is incomplete or uncertain')
    try:
        proof_path = path + PROOF_SUFFIX
        if observed is None:
            proof = _read_capture_proof_bytes(Path(proof_path).read_bytes())
        else:
            with open(proof_path, 'rb') as handle:
                path_initial = _file_identity(os.stat(proof_path))
                descriptor_initial = _file_identity(os.fstat(handle.fileno()))
                proof = _read_capture_proof_bytes(handle.read())
                _record_verified_file(observed, proof_path, path_initial, descriptor_initial, handle)
    except FileNotFoundError as exc:
        raise ValueError('legacy or missing capture provenance; origin and vertex membership are unproven') from exc
    decoded = _check_capture_payload(path, proof, decode=decode, observed=observed)
    if decode:
        if not np.isfinite(decoded).all():
            raise ValueError("capture PC2 contains non-finite samples")
        return proof, decoded
    if frames is not None and (list(frames.shape) != proof['shape'] or not np.isfinite(frames).all()
                               or _payload_digest(frames) != proof['payload_sha256']):
        raise ValueError('capture memory and proven disk payload differ')
    return proof


def validate_capture(scene, obj, frames, kind, vg_name=None, pin_item=None, *, decode=False):
    from . import pc2
    from .uuid_registry import get_object_uuid
    from ..ui.dynamics.static_deform_ops import _effective_frame_range
    key = pc2.static_deform_pc2_key(obj) if kind == 'STATIC' else pc2.pin_anim_pc2_key(obj, vg_name, scene=scene)
    try:
        observed = _transfer_observations.get()
        verified = read_bound_proof(os.path.abspath(pc2.get_pc2_path(key)), frames,
                                    decode=decode, observed=observed)
        proof = verified[0] if decode else verified
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
        return verified
    except (OSError, ValueError, TypeError, KeyError) as exc:
        label = obj.name + (" / Pin '" + vg_name + "'" if vg_name else '')
        from ..i18n import pgettext_iface as iface_
        message = iface_("Captured input '{label}' is stale or unproven: {error}. Recapture Deformation before export or solve.")
        raise ValueError(message.format(label=label, error=iface_(str(exc)))) from exc


def copy_pair(source, destination, *, expected_proof=None):
    """Save As copies a known pair verbatim; never re-stamps current settings."""
    source, destination = os.fspath(source), os.fspath(destination)
    _assert_group_idle(os.path.dirname(source))
    _assert_group_idle(os.path.dirname(destination))
    if os.path.realpath(source) == os.path.realpath(destination):
        return
    for path in (source, destination):
        for suffix in ('', PROOF_SUFFIX, PENDING_SUFFIX):
            _regular_or_absent(path + suffix)
        if os.path.lexists(path + PENDING_SUFFIX):
            raise ValueError('Cannot migrate an uncertain capture pair; recapture required')
    proof_path = source + PROOF_SUFFIX
    proof_bytes = Path(proof_path).read_bytes() if os.path.exists(proof_path) else None
    proof = _read_capture_proof_bytes(proof_bytes) if proof_bytes is not None else None
    if expected_proof is not None and (proof is None or proof['proof_sha256'] != expected_proof):
        raise ValueError('Capture source does not match the Scene binding; recapture required')
    if os.path.lexists(destination) or os.path.lexists(destination + PROOF_SUFFIX):
        target_proof = read_bound_proof(destination)
        if proof is None or proof['proof_sha256'] != target_proof['proof_sha256']:
            raise ValueError('Save As capture target already holds a different generation; target retained')
        _check_capture_payload(source, proof)
        return
    _commit_pair(destination, lambda temp: _check_capture_payload(source, proof, temp), proof_bytes)


def load_for_export(scene, obj, kind, vg_name=None, *, required=False):
    """A recorded but missing capture must never become uncaptured motion."""
    from . import pc2
    label = f"{obj.name} / Pin '{vg_name}'" if vg_name is not None else f"{obj.name} / Static"
    try:
        if kind == 'STATIC':
            key = pc2.static_deform_pc2_key(obj)
            frames = pc2.get_static_deform_cache(obj)
        else:
            key = pc2.pin_anim_pc2_key(obj, vg_name, scene=scene)
            frames = pc2.get_pin_anim_cache(obj, vg_name, scene=scene)
        if frames is None and (required or key in bindings(scene)):
            raise ValueError("recorded PC2 file is missing")
        return frames
    except (OSError, ValueError) as exc:
        from ..i18n import pgettext_iface as iface_
        message = iface_("Captured input '{label}' cannot be used: {error}. Recapture Deformation before export.")
        raise ValueError(message.format(label=label, error=iface_(str(exc)))) from exc


def static_for_scene(scene, obj):
    """Read a cold Static array with the existing Scene proof pass.

    Param validation still rereads current on-disk proof and payload.
    """
    from . import pc2
    key = pc2.static_deform_pc2_key(obj)
    if not pc2.has_static_deform_cache_key(key) and os.path.lexists(pc2.get_pc2_path(key)):
        _proof, frames = validate_capture(scene, obj, None, 'STATIC', decode=True)
        pc2._static_deform_cache[key] = frames
        return frames
    frames = load_for_export(scene, obj, 'STATIC')
    if frames is not None:
        validate_capture(scene, obj, frames, 'STATIC')
    return frames


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
            frames = load_for_export(scene, obj, 'STATIC')
            if frames is not None:
                validate_capture(scene, obj, frames, 'STATIC')
