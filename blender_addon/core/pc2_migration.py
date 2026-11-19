"""Conservative Keep Copy migration; source caches are never moved/deleted."""
import ctypes
import json
import math
import os
import shutil
import sys
import tempfile


def _is_link_or_reparse(path):
    st = os.lstat(path)
    return os.path.islink(path) or bool(getattr(st, "st_file_attributes", 0) & 0x400)



def _reject_link_ancestors(path):
    current = os.path.abspath(path)
    while True:
        if os.path.lexists(current) and _is_link_or_reparse(current):
            raise ValueError('Migration refuses symbolic links or reparse points: '+current)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

def _advisory_only(path):
    if _is_link_or_reparse(path) or not os.path.isdir(path):
        return False
    names = os.listdir(path)
    if not names:
        return True
    if names != ['.manifest.json']:
        return False
    manifest = os.path.join(path, '.manifest.json')
    if _is_link_or_reparse(manifest) or not os.path.isfile(manifest) or os.path.getsize(manifest) > 65536:
        return False
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate manifest key')
            result[key] = value
        return result
    try:
        with open(manifest, encoding='utf-8') as stream:
            data = json.load(stream, object_pairs_hook=unique_object)
        return (isinstance(data, dict)
                and set(data) == {'schema_version', 'addon_version', 'last_session_id', 'last_write_utc', 'pc2_entries'}
                and type(data['schema_version']) is int and data['schema_version'] == 1
                and isinstance(data['addon_version'], str)
                and isinstance(data['last_session_id'], str)
                and type(data['last_write_utc']) in (int, float)
                and math.isfinite(data['last_write_utc'])
                and data['pc2_entries'] == [])
    except (OSError, ValueError, TypeError, OverflowError):
        return False


def _rename_noreplace(source, target):
    """Fail rather than overwrite a destination created by another writer."""
    if os.name == 'nt':
        os.rename(source, target)  # Windows rename refuses an existing target.
        return
    if not sys.platform.startswith('linux'):
        raise OSError('Safe no-replace migration is unavailable on this platform')
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, 'renameat2', None)
    if rename is None:
        raise OSError('Safe no-replace renameat2 is unavailable')
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), target)


def _no_links(path):
    if _is_link_or_reparse(path) or not os.path.isdir(path):
        raise ValueError('Migration source must be a real directory: '+path)
    for root, dirs, files in os.walk(path, followlinks=False):
        for name in dirs + files:
            entry = os.path.join(root, name)
            if _is_link_or_reparse(entry):
                raise ValueError('Migration refuses symbolic links: '+entry)


def _target_identity(path):
    if not _advisory_only(path):
        raise ValueError('Target already exists and is non-empty, resolve manually: '+path)
    st = os.stat(path, follow_symlinks=False)
    manifest = os.path.join(path, '.manifest.json')
    content = None
    if os.path.exists(manifest):
        with open(manifest, 'rb') as stream:
            content = stream.read()
    return st.st_dev, st.st_ino, content


def migrate_directory(old_dir, new_dir, keep_copy):
    """Only Keep Copy is supported by this conservative candidate.

    Source remains authoritative on any exception or process interruption.
    Destination metadata and failed staged copies are retained with explicit
    recovery paths. No source symlinks are followed, no target is overwritten.
    """
    if not keep_copy:
        raise ValueError('Move migration is disabled for safety; enable Keep Copy')
    _reject_link_ancestors(old_dir)
    _reject_link_ancestors(new_dir)
    _no_links(old_dir)
    expected = _target_identity(new_dir) if os.path.lexists(new_dir) else None
    parent = os.path.dirname(new_dir)
    stage = tempfile.mkdtemp(prefix='.'+os.path.basename(new_dir)+'.migration-', dir=parent)
    os.rmdir(stage)
    backup = None
    try:
        shutil.copytree(old_dir, stage, symlinks=True)
        _no_links(stage)
        if expected is not None:
            if not os.path.lexists(new_dir) or _target_identity(new_dir) != expected:
                raise ValueError('Migration target changed while preparing: '+new_dir)
            backup = tempfile.mkdtemp(prefix='.'+os.path.basename(new_dir)+'.manifest-backup-', dir=parent)
            os.rmdir(backup)
            _rename_noreplace(new_dir, backup)
            if _target_identity(backup) != expected:
                raise ValueError('Migration target changed while reserving: '+backup)
        _rename_noreplace(stage, new_dir)
    except Exception as exc:
        recovery = []
        if os.path.lexists(stage):
            recovery.append('source unchanged; staged copy preserved at '+stage)
        if backup and os.path.lexists(backup):
            try:
                _rename_noreplace(backup, new_dir)
            except OSError as restore_error:
                recovery.append('target restore refused ('+str(restore_error)+'); original target preserved at '+backup)
        if recovery:
            raise OSError(str(exc)+'; RECOVERY PATHS: '+'; '.join(recovery)) from exc
        raise
    if backup:
        print('[PPF migration] Preserved previous target metadata: '+backup)


def _cache_copy_identity(path):
    st = os.stat(path, follow_symlinks=False)
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


def discard_cache_copy(entry):
    """Remove our own unchanged copy on save rollback."""
    path, identity = entry
    try:
        if _cache_copy_identity(path) == identity:
            os.unlink(path)
    except FileNotFoundError:
        pass


def copy_files_noreplace(source_dir, target_dir, names):
    """Copy selected cache files into an existing directory without replacement."""
    _reject_link_ancestors(source_dir)
    _reject_link_ancestors(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    added = []
    try:
        for name in sorted(set(names)):
            if os.path.basename(name) != name or name in ('', '.', '..'):
                raise ValueError('Invalid cache filename')
            source = os.path.join(source_dir, name)
            target = os.path.join(target_dir, name)
            if _is_link_or_reparse(source) or not os.path.isfile(source):
                raise ValueError('Cache source is not a regular file: ' + source)
            if os.path.lexists(target):
                raise ValueError('Cache destination already exists: ' + target)
            fd, stage = tempfile.mkstemp(prefix='.ppf-copy-', dir=target_dir)
            try:
                with os.fdopen(fd, 'wb') as dst, open(source, 'rb') as src:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
                identity = _cache_copy_identity(stage)
                if os.name == 'nt':
                    _rename_noreplace(stage, target)
                else:
                    os.link(stage, target)
                added.append((target, identity))
            finally:
                if os.path.exists(stage):
                    os.unlink(stage)
    except Exception:
        for entry in reversed(added):
            discard_cache_copy(entry)
        raise
    return added
