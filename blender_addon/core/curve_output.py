"""Curve OUTPUT locators across Save As; not captured-input provenance.

Only actual loaded output supplies identity. The pre-save locator is serialized
before copying, so failed/colliding copies retain a proven external fallback.
Hashes are evaluated at load/save/cleanup boundaries, not playback frame ticks.
"""
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager

import bpy
import numpy as np

class CurveOutputIdentityError(ValueError):
    """A valid bound output generation is no longer recoverable from its files."""
    input_preserved = False


PROPERTY = '_ppf_curve_output_locator_v1'
LEGACY_DETACHED_KEY = '_ppf_legacy_curve_detached_key_v1'
LEGACY_RETAINED_LOCATOR = '_ppf_legacy_retained_locator_v1'
LEGACY_RETAINED_PATHS = '_ppf_legacy_retained_paths_v1'
_live_paths = {}


def _pc2():
    from . import pc2
    return pc2


def _identity(array):
    arr = np.ascontiguousarray(array, dtype='<f')
    if arr.ndim != 3 or arr.shape[2] != 3 or not arr.shape[0] or not arr.shape[1] or not np.isfinite(arr).all():
        raise ValueError('Curve output needs a nonempty finite frame/CV/XYZ array')
    return {'shape': list(arr.shape), 'payload': hashlib.sha256(memoryview(arr).cast('B')).hexdigest()}


def _disk_identity(path):
    arr = _pc2()._load_pc2_array(path)
    if arr is None:
        raise ValueError('Curve output is missing or unreadable: ' + path)
    return _identity(arr)


def _record(obj):
    raw = obj.get(PROPERTY)
    if raw is None:
        return None
    try:
        record = json.loads(raw)
        if (not isinstance(record, dict) or record.get('key') != _pc2().object_pc2_key_readonly(obj)
                or record.get('status') not in {'bound', 'detached'}
                or not isinstance(record.get('source'), str)
                or not isinstance(record.get('identity'), dict)):
            raise ValueError('invalid locator fields')
        return record
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Curve '{obj.name}' output locator is damaged; existing files retained") from exc


def _matches(path, expected):
    try:
        return os.path.isfile(path) and _disk_identity(path) == expected
    except (OSError, ValueError, TypeError):
        return False


def _owned_objects():
    """Only one local Object/Scene/datablock owner may receive locator writes."""
    pc2 = _pc2()
    seen = set()
    for scene in bpy.data.scenes:
        for obj in pc2._scene_curve_objects(scene):
            if obj.library is not None or obj.data.library is not None:
                continue
            key = pc2.object_pc2_key_readonly(obj)
            if not key or key in seen:
                continue
            matches = [o for o in bpy.data.objects if pc2.object_pc2_key_readonly(o) == key]
            if len(matches) != 1:
                continue
            seen.add(key)
            yield obj


def note_loaded(key, path):
    """A producer/loader explicitly loaded this actual file into Curve memory."""
    _live_paths[key] = os.path.realpath(path)
    from .uuid_registry import get_object_by_uuid
    obj = get_object_by_uuid(key)
    if obj is not None and obj.type == 'CURVE' and PROPERTY in obj:
        record = _record(obj)
        if record and record['status'] == 'detached' and obj.get(LEGACY_DETACHED_KEY) != key:
            # Only a uniquely owned target can clear its previous detach marker.
            if any(owner == obj for owner in _owned_objects()):
                del obj[PROPERTY]


def clear_runtime():
    _live_paths.clear()


def detached(obj):
    if obj.get(LEGACY_DETACHED_KEY) == _pc2().object_pc2_key_readonly(obj) and obj.get(LEGACY_DETACHED_KEY):
        return True
    record = _record(obj)
    return bool(record and record['status'] == 'detached')


def commit_new_output(obj, key, path):
    """Clear a legacy detach only after this owner's new output is loaded."""
    if obj.get(LEGACY_DETACHED_KEY) != key or not key:
        return
    pc2 = _pc2()
    if (pc2.object_pc2_key_readonly(obj) != key
            or len([other for other in bpy.data.objects if pc2.object_pc2_key_readonly(other) == key]) != 1
            or key not in pc2._curve_cache
            or os.path.realpath(_live_paths.get(key, '')) != os.path.realpath(path)):
        raise ValueError('无法确认新曲线缓存已成功加载；旧缓存保持分离。')
    sentinel=object()
    old_locator=obj.get(PROPERTY,sentinel)
    retained=obj.get(LEGACY_RETAINED_LOCATOR,sentinel)
    if old_locator is not sentinel and retained is not sentinel and retained != old_locator:
        raise ValueError('旧曲线定位记录的保留位置已占用；未解除旧缓存保护。')
    new_locator=json.dumps(dict(status='bound',key=key,source=os.path.realpath(path),
                                identity=_identity(pc2._curve_cache[key])),sort_keys=True)
    try:
        if old_locator is not sentinel:obj[LEGACY_RETAINED_LOCATOR]=old_locator
        obj[PROPERTY]=new_locator
        del obj[LEGACY_DETACHED_KEY]
    except BaseException:
        if old_locator is sentinel:
            if PROPERTY in obj:del obj[PROPERTY]
        else:obj[PROPERTY]=old_locator
        if retained is sentinel:
            if LEGACY_RETAINED_LOCATOR in obj:del obj[LEGACY_RETAINED_LOCATOR]
        else:obj[LEGACY_RETAINED_LOCATOR]=retained
        obj[LEGACY_DETACHED_KEY]=key
        raise


@contextmanager
def legacy_new_output(obj, key, path):
    """Keep the old canonical cache while a new per-curve write commits."""
    if not key or obj.get(LEGACY_DETACHED_KEY) != key:
        yield
        return
    pc2=_pc2();path=os.path.realpath(path)
    if (pc2.object_pc2_key_readonly(obj) != key or
            len([other for other in bpy.data.objects if pc2.object_pc2_key_readonly(other)==key]) != 1):
        raise ValueError('曲线缓存归属不唯一，旧缓存与恢复记录均已保留。')
    sentinel=object()
    fields={name:obj.get(name,sentinel) for name in (PROPERTY,LEGACY_DETACHED_KEY,LEGACY_RETAINED_LOCATOR,LEGACY_RETAINED_PATHS)}
    try:
        retained=json.loads(fields[LEGACY_RETAINED_PATHS]) if fields[LEGACY_RETAINED_PATHS] is not sentinel else []
        if not isinstance(retained,list) or not all(isinstance(item,str) for item in retained):raise ValueError('invalid')
    except (TypeError,ValueError):raise ValueError('旧缓存保留路径记录损坏；未写入新输出。')
    previous_cache=pc2._curve_cache.get(key,sentinel)
    previous_live=_live_paths.get(key,sentinel)
    previous_frame=pc2._curve_last_frame
    backup=None
    backup_moved=False
    write_started=False
    def unique_path(label):
        fd,name=tempfile.mkstemp(prefix=os.path.basename(path)+label,dir=os.path.dirname(path))
        os.close(fd)
        return name
    try:
        if os.path.exists(path):
            backup=unique_path('.retained-')
            try:
                os.replace(path,backup)
                backup_moved=True
            except BaseException as move_error:
                try:os.unlink(backup)
                except Exception as cleanup_error:
                    raise RuntimeError('旧缓存未移动，占位文件清理失败，已保留：'+backup+'；'+str(cleanup_error)) from move_error
                backup=None
                raise
        # A previously loaded legacy cache cannot prove the new file loaded.
        # Keep its exact state for compensation, but require a fresh load before
        # commit_new_output may remove the persistent detach marker.
        pc2._curve_cache.pop(key,None)
        _live_paths.pop(key,None)
        pc2._curve_last_frame=-1
        write_started=True
        yield
        if obj.get(LEGACY_DETACHED_KEY)==key:
            raise ValueError('新输出尚未成功附加；旧缓存保护仍然有效。')
        if backup_moved:
            retained.append(backup)
            obj[LEGACY_RETAINED_PATHS]=json.dumps(retained,ensure_ascii=False)
    except BaseException as exc:
        errors=[]
        try:
            if write_started and os.path.exists(path):
                failed=unique_path('.failed-new-');os.replace(path,failed)
            if backup_moved:os.replace(backup,path)
        except Exception as rollback_error:errors.append('缓存文件: '+str(rollback_error))
        for name,value in fields.items():
            try:
                if value is sentinel:
                    if name in obj:del obj[name]
                else:obj[name]=value
            except Exception as rollback_error:errors.append(name+': '+str(rollback_error))
        for mapping,value in ((_live_paths,previous_live),(pc2._curve_cache,previous_cache)):
            try:
                if value is sentinel:mapping.pop(key,None)
                else:mapping[key]=value
            except Exception as rollback_error:errors.append('内存缓存: '+str(rollback_error))
        pc2._curve_last_frame=previous_frame
        if errors:raise RuntimeError('曲线新输出失败且部分恢复未完成；请保留文件和记录：'+'; '.join(errors)) from exc
        raise


def resolve_read_path(obj):
    """Prefer a matching local copy; never read a colliding foreign target."""
    pc2 = _pc2(); key = pc2.object_pc2_key_readonly(obj)
    if obj.get(LEGACY_DETACHED_KEY) == key and key:
        return None
    canonical = pc2.get_pc2_path(key)
    record = _record(obj)
    if record is None:
        return canonical  # Legacy UUID-local playback remains supported.
    if record['status'] == 'detached':
        return None
    for path in (canonical, record['source']):
        if _matches(path, record['identity']):
            return os.path.realpath(path)
    raise CurveOutputIdentityError(f"Curve '{obj.name}' output is missing or changed; no matching local copy or retained source. Existing files were not overwritten.")


def _cleanup_path_checked(obj):
    """Delete only matching current-project output; external output detaches."""
    pc2 = _pc2();key = pc2.object_pc2_key_readonly(obj)
    # A live producer can extend the cache since the last saved identity.
    live = _live_paths.get(key)
    path = live if live and key in pc2._curve_cache else resolve_read_path(obj)
    if path is None:
        return None
    if live and key in pc2._curve_cache and _record(obj) is not None:
        if not os.path.exists(live) and os.path.realpath(live) == os.path.realpath(pc2.get_pc2_path(key)):
            return live  # Existing missing-input receipt/preflight owns this case.
        try:
            same = _disk_identity(live) == _identity(pc2._curve_cache[key])
        except (OSError, ValueError) as exc:
            raise CurveOutputIdentityError(f"Curve '{obj.name}' bound output is missing or changed; sampled output must be retained") from exc
        if not same:
            raise CurveOutputIdentityError(f"Curve '{obj.name}' bound output changed outside its loaded cache; sampled output must be retained")
        path = live
    canonical = os.path.realpath(pc2.get_pc2_path(key))
    return path if os.path.realpath(path) == canonical else None


def cleanup_path(obj, receipt=None):
    try:
        return _cleanup_path_checked(obj)
    except CurveOutputIdentityError:
        if receipt is not None:
            receipt.update(phase='output_identity_uncertain', input_preserved=False)
        raise


def mark_detached(obj):
    pc2 = _pc2();key = pc2.object_pc2_key_readonly(obj)
    record = _record(obj)
    from .playback_origin import CURVE_START_FRAME
    if CURVE_START_FRAME in obj:
        del obj[CURVE_START_FRAME]
    if record is None:
        _live_paths.pop(key, None)
        return  # Legacy local deletion needs no persistent fallback tombstone.
    record['status'] = 'detached'
    obj[PROPERTY] = json.dumps(record, sort_keys=True)
    _live_paths.pop(key, None)


def preflight_write(obj):
    """A Save As conflict must not become an overwrite on a later fetch."""
    pc2 = _pc2();key = pc2.object_pc2_key_readonly(obj)
    target = os.path.realpath(pc2.get_pc2_path(key))
    if obj.get(LEGACY_DETACHED_KEY) == key and key and os.path.exists(target):
        raise ValueError('旧曲线缓存已保留；请先将原缓存移至安全位置，再写入新输出。')
    if _live_paths.get(key) == target:
        return
    record = _record(obj)
    if record is None:
        return
    if os.path.exists(target):
        if not _matches(target, record['identity']):
            raise ValueError(f"Curve '{obj.name}' output destination contains a different cache; retained source and target were not overwritten")
    elif record['status'] == 'bound' and os.path.realpath(record['source']) != target:
        raise ValueError(f"Curve '{obj.name}' output remains in the previous project; repair the Save As copy or Clear/re-run before fetching more frames")


@bpy.app.handlers.persistent
def prepare_curve_outputs_before_save(*_args):
    pc2 = _pc2()
    for obj in _owned_objects():
        try:
            record = _record(obj)
            if record and record['status'] == 'detached':
                continue
            key = pc2.object_pc2_key_readonly(obj)
            path = _live_paths.get(key) or resolve_read_path(obj)
            if not path or not os.path.exists(path):
                if record:
                    raise ValueError('previous output source is missing')
                continue
            identity = _disk_identity(path)
            if key in pc2._curve_cache and identity != _identity(pc2._curve_cache[key]):
                raise ValueError('disk output differs from the actually loaded output')
            # The disk snapshot needs both the source and its playback origin.
            # A legacy cache inherits the existing timeline, not pending settings.
            from .playback_origin import CURVE_START_FRAME, legacy_curve_start_frame
            if type(obj.get(CURVE_START_FRAME)) is not int and len(obj.users_scene) == 1:
                obj[CURVE_START_FRAME] = legacy_curve_start_frame(obj.users_scene[0])
            obj[PROPERTY] = json.dumps(dict(status='bound', key=key, source=os.path.realpath(path), identity=identity), sort_keys=True)
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            pc2._log_curve_handler_error(f"Curve '{obj.name}' output locator could not be refreshed: {exc}; original binding/files retained")


def migrate_outputs_after_save():
    pc2 = _pc2()
    for obj in _owned_objects():
        try:
            record = _record(obj)
            if not record or record['status'] == 'detached':
                continue
            source = record['source'];target = pc2.get_pc2_path(record['key'])
            if os.path.realpath(source) == os.path.realpath(target):
                continue
            if not _matches(source, record['identity']):
                raise ValueError('saved source is missing or changed')
            if os.path.exists(target):
                if not _matches(target, record['identity']):
                    raise ValueError('destination already contains a different cache')
            else:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                fd, staged = tempfile.mkstemp(prefix='.ppf-curve-output-', dir=os.path.dirname(target))
                os.close(fd)
                try:
                    shutil.copyfile(source, staged)
                    if _disk_identity(staged) != record['identity']:
                        raise ValueError('source changed while copying')
                    # Same-directory hard-link publication cannot overwrite an
                    # existing target. Unsupported filesystems retain fallback.
                    os.link(staged, target)
                finally:
                    try:os.unlink(staged)
                    except OSError:pass
            _live_paths[record['key']] = os.path.realpath(target)
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            pc2._log_curve_handler_error(f"Curve '{obj.name}' Save As output copy failed: {exc}. Saved locator retains original source; new project is not necessarily self-contained.")
