"""Lossless, file-backed writer for the existing chunked CBOR pickle envelope.

The protocol is intentionally pickle's unchanged default, not HIGHEST_PROTOCOL.
Protocol 4 still memoizes ndarray reduction byte buffers during pickle.dump;
this removes whole-output buffers, not all serialization memory overhead.
"""

from __future__ import annotations

import os
import pickle
import stat
import tempfile
import uuid

import cbor2


def _write_spooled_envelope(out, spool, size, kind, payload, bridge):
    """Emit the same definite-length maps/list and byte strings as dumps."""
    if not isinstance(payload, dict) or "pickle_blob" not in payload:
        raise ValueError("pickle payload metadata must include 'pickle_blob'")
    if payload["pickle_blob"] != []:
        raise ValueError("pickle payload metadata must use an empty chunk placeholder")
    if size < 0:
        raise ValueError("pickle size must be nonnegative")
    chunk_size = bridge.PICKLE_CHUNK_BYTES
    if chunk_size <= 0:
        raise ValueError("pickle chunk size must be positive")
    encoder = cbor2.CBOREncoder(out, default=bridge._numpy_default)
    encoder.encode_length(5, 3)
    encoder.encode("version")
    encoder.encode(bridge.SCHEMA_VERSION)
    encoder.encode("kind")
    encoder.encode(kind)
    encoder.encode("payload")
    encoder.encode_length(5, len(payload))
    for key, value in payload.items():
        encoder.encode(key)
        if key != "pickle_blob":
            encoder.encode(value)
            continue
        encoder.encode_length(4, (size + chunk_size - 1) // chunk_size)
        remaining = size
        while remaining:
            wanted = min(chunk_size, remaining)
            chunk = spool.read(wanted)
            if len(chunk) != wanted:
                raise OSError("pickle spool ended before its recorded size")
            encoder.encode(chunk)
            remaining -= len(chunk)
        if spool.read(1):
            raise OSError("pickle spool grew beyond its recorded size")


def _existing_target(path):
    try:
        value = os.lstat(path)
    except FileNotFoundError:
        return None
    if (not stat.S_ISREG(value.st_mode) or value.st_nlink != 1
            or getattr(value, "st_file_attributes", 0) & 0x400):
        raise OSError("atomic pickle save requires a regular, single-link target")
    return value


def dump_pickled_envelope_file(path, kind, obj, payload_factory, bridge):
    """Serialize without a complete pickle/chunk-list/CBOR buffer in memory.

    payload_factory is evaluated after pickling, as in the original caller,
    and must return original metadata with pickle_blob=[] as a placeholder.
    All exceptions propagate. A failed serialization leaves the destination
    unchanged and removes its temporary file. New-file modes follow open(wb);
    existing owner/group/mode/xattrs are preserved or replacement is refused.
    The whole rename is atomic on the destination filesystem, but no claim
    of power-loss durability is made (the parent directory is not fsynced).
    """
    path = os.path.abspath(os.fspath(path))
    directory = os.path.dirname(path)
    temporary = None
    # TemporaryFile is disk-backed (not SpooledTemporaryFile/BytesIO).
    with tempfile.TemporaryFile(mode="w+b", dir=directory) as spool:
        pickle.dump(obj, spool)
        size = spool.tell()
        spool.seek(0)
        payload = payload_factory()
        previous = _existing_target(path)
        if previous is not None:
            # Replacing a file needs only directory write permission. Do not
            # gain write ability beyond the original open(path, 'wb').
            fd = os.open(path, os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                opened = os.fstat(fd)
                if (opened.st_dev, opened.st_ino) != (previous.st_dev, previous.st_ino):
                    raise OSError("pickle save destination changed before serialization")
            finally:
                os.close(fd)
        # Exclusive create honors 0666 & umask exactly as open(path, 'wb').
        # A random same-directory pathname is needed for atomic os.replace.
        candidate = os.path.join(
            directory, "." + os.path.basename(path) + "." + uuid.uuid4().hex + ".tmp"
        )
        try:
            with open(candidate, "xb") as out:
                temporary = candidate
                if previous is not None:
                    created = os.fstat(out.fileno())
                    if os.name != "nt":
                        if (created.st_uid, created.st_gid) != (previous.st_uid, previous.st_gid):
                            raise OSError("atomic pickle save would change target ownership")
                        os.fchmod(out.fileno(), stat.S_IMODE(previous.st_mode))
                        if hasattr(os, "listxattr"):
                            for name in os.listxattr(path):
                                os.setxattr(out.fileno(), name, os.getxattr(path, name))
                _write_spooled_envelope(out, spool, size, kind, payload, bridge)
                out.flush()
                os.fsync(out.fileno())
            current = _existing_target(path)
            before = None if previous is None else (previous.st_dev, previous.st_ino)
            after = None if current is None else (current.st_dev, current.st_ino)
            if before != after:
                raise OSError("pickle save destination changed during serialization")
            if os.name == "nt" and previous is not None:
                # ReplaceFileW preserves the replaced file's Windows ACL and
                # other metadata. Do not silently fall back to os.replace.
                import ctypes
                from ctypes import wintypes
                replace_file = ctypes.WinDLL("kernel32", use_last_error=True).ReplaceFileW
                replace_file.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR,
                                         wintypes.LPCWSTR, wintypes.DWORD,
                                         wintypes.LPVOID, wintypes.LPVOID]
                replace_file.restype = wintypes.BOOL
                if not replace_file(path, temporary, None, 0, None, None):
                    raise ctypes.WinError(ctypes.get_last_error())
            elif os.name == "nt":
                # Windows rename refuses an unexpectedly created destination;
                # os.replace would overwrite it after the identity check.
                os.rename(temporary, path)
            else:
                os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                os.unlink(temporary)
