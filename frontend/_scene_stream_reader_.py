"""Isolated, opt-in Scene reader for the matching addon's untagged containers.

No installation or monkeypatching occurs on import.  The reference adapter is
an unchanged snapshot, and owns envelope checks and ordinary rehydration.

The only representation change is a definite-length
payload[*].object[*].static_deform_animation.vert_frames array: it becomes a
C-contiguous float64 ndarray, exactly the dtype used by _populate_static.
Only one frame's Python tree exists at a time.  All input bytes stay untouched.

This reader accepts the matching producer's definite, untagged CBOR. A
structural preflight explicitly rejects tagged and indefinite items everywhere
inside the root, including nested metadata and shared/string references. It
never silently falls back to the full-file decoder. Input must be seekable.
Trailing bytes after the first item are ignored, as in the legacy loader.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import os
from typing import BinaryIO

import cbor2
import numpy as np

from . import _cbor_bridge_ as reference


class UnsupportedSceneEncoding(ValueError):
    """The opt-in fast path cannot preserve this encoding's semantics."""


@dataclass
class DecodeStats:
    streamed_fields: int = 0
    frame_count: int = 0
    largest_frame_values: int = 0
    dense_bytes: int = 0


class _RootBoundedReader:
    """Limit decoder reads to the preflighted root, excluding trailing bytes."""
    def __init__(self, stream, end):
        self.stream = stream
        self.end = end

    def read(self, n=-1):
        remaining = self.end - self.stream.tell()
        if n < 0 or n > remaining:
            n = remaining
        return self.stream.read(n)

    def readable(self):
        return True

    def seekable(self):
        return False


def _guard_root(stream):
    """Scan one CBOR item structurally, without allocating its values.

    cbor2 resets shared-reference tables between decode() calls. Reject every
    tag before decoding, including tags hidden in keys or opaque values; an
    unchecked shared-reference tag can silently change object identity.
    Indefinite containers are outside the matching producer's output domain.
    """
    if not stream.seekable():
        raise UnsupportedSceneEncoding("Scene stream must be seekable")
    start = stream.tell()
    stream.seek(0, os.SEEK_END)
    end = stream.tell()
    stream.seek(start)
    stack = [1]
    while stack:
        if stack[-1] == 0:
            stack.pop()
            continue
        stack[-1] -= 1
        initial = stream.read(1)
        if not initial:
            raise cbor2.CBORDecodeEOF("truncated Scene item")
        byte = initial[0]
        major, additional = byte >> 5, byte & 31
        if major == 6:
            raise UnsupportedSceneEncoding("CBOR tags are not supported by the streaming Scene reader")
        if additional == 31:
            raise UnsupportedSceneEncoding("indefinite CBOR is not supported by the streaming Scene reader")
        if additional >= 28:
            raise cbor2.CBORDecodeError("invalid CBOR additional information")
        if additional < 24:
            value = additional
        else:
            width = 1 << (additional - 24)
            raw = stream.read(width)
            if len(raw) != width:
                raise cbor2.CBORDecodeEOF("truncated Scene item header")
            value = int.from_bytes(raw, "big")
        if major in (2, 3):
            if value > end - stream.tell():
                raise cbor2.CBORDecodeEOF("truncated Scene byte or text string")
            stream.seek(value, os.SEEK_CUR)
        elif major in (4, 5):
            count = value if major == 4 else value * 2
            if count > end - stream.tell():
                raise cbor2.CBORDecodeEOF("declared Scene container cannot fit in input")
            if count:
                if len(stack) >= 393:
                    raise cbor2.CBORDecodeError("maximum Scene nesting depth exceeded")
                stack.append(count)
    root_end = stream.tell()
    stream.seek(start)
    return root_end


class _PrefixedReader:
    """Replay one consumed header without touching the underlying file cursor.

    Both this adapter and the parent traversal consume the SAME CBORDecoder.
    The temporary decoder uses read_size=1, so it does not consume a sibling
    while decoding a wrong-type value that must be passed to reference checks.
    """

    def __init__(self, first: bytes, decoder: cbor2.CBORDecoder):
        self.first = first
        self.decoder = decoder

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def read(self, n: int) -> bytes:
        if n < 0:
            raise ValueError("unbounded reads are not permitted")
        head, self.first = self.first[:n], self.first[n:]
        if len(head) == n:
            return head
        try:
            return head + self.decoder.read(n - len(head))
        except cbor2.CBORDecodeEOF:
            if head:
                return head
            return b""


class _Reader:
    def __init__(self, stream: BinaryIO, size: int | None, stats: DecodeStats):
        # Seven traversed containers surround a frame. Reserve that budget
        # even for shallower opaque metadata rather than accidentally accepting
        # nesting that the original whole-envelope decoder rejects at 400.
        # This is conservative (at most seven levels stricter for metadata).
        # cbor2 6.0.1's file decoder panics on float arrays with read_size>1
        # (reproduced independently on a 280 KB ordinary CBOR array). Its
        # read_size=1 path requests exact scalar/header spans and is correct;
        # the Python file object still supplies efficient OS-level buffering.
        self.decoder = cbor2.CBORDecoder(stream, read_size=1, max_depth=393)
        self.size = size
        self.stats = stats

    def _header(self, first: bytes) -> tuple[int, int | None]:
        byte = first[0]
        major, arg = byte >> 5, byte & 31
        if arg < 24:
            return major, arg
        if arg in (24, 25, 26, 27):
            width = 1 << (arg - 24)
            return major, int.from_bytes(self.decoder.read(width), "big")
        if arg == 31:
            return major, None
        raise cbor2.CBORDecodeError(f"invalid additional information {arg}")

    def _decode_prefixed(self, first: bytes):
        return cbor2.CBORDecoder(
            _PrefixedReader(first, self.decoder), read_size=1, max_depth=393,
        ).decode()

    def _container(self, expected_major: int, role: str):
        first = self.decoder.read(1)
        major = first[0] >> 5
        if major == 6:
            raise UnsupportedSceneEncoding(f"tagged {role} is not a supported fast-path container")
        if major != expected_major:
            return False, self._decode_prefixed(first)
        _, count = self._header(first)
        if count is None:
            raise UnsupportedSceneEncoding(f"indefinite {role} is not a supported fast-path container")
        return True, count

    def _map(self, role: str, routes: dict):
        is_map, count = self._container(5, role)
        if not is_map:
            return count
        result = {}
        # Match the installed cbor2 6.0.1 behavior, including its duplicate
        # map-key quirk: it stops at the declared number of UNIQUE keys.
        # A duplicate-only short map therefore raises EOF, not acceptance.
        while len(result) < count:
            # CBOR map keys are decoded in immutable mode by cbor2 itself.
            key = self.decoder.decode(immutable=True)
            route = routes.get(key) if isinstance(key, str) else None
            value = route() if route is not None else self.decoder.decode()
            result[key] = value
        return result

    def _array(self, role: str, item_reader):
        is_array, count = self._container(4, role)
        if not is_array:
            return count
        return [item_reader() for _ in range(count)]

    def envelope(self):
        return self._map("Scene envelope", {"payload": self.groups})

    def groups(self):
        return self._array("Scene payload", self.group)

    def group(self):
        return self._map("scene group", {"object": self.objects})

    def objects(self):
        return self._array("scene objects", self.object)

    def object(self):
        return self._map("scene object", {"static_deform_animation": self.static_deform})

    def static_deform(self):
        return self._map("static deformation", {"vert_frames": self.frames})

    def frames(self):
        is_array, count = self._container(4, "static vert_frames")
        if not is_array:
            return count
        self.stats.streamed_fields += 1
        self.stats.frame_count += count
        if count == 0:
            # np.ascontiguousarray([], dtype=float64) in the original decoder.
            return np.empty((0,), dtype=np.float64)
        output = None
        frame_shape = None
        for index in range(count):
            raw_frame = self.decoder.decode()
            # Scalar frames stay 0-D here so the full output has the same
            # shape as np.ascontiguousarray([scalar, scalar], dtype=float64).
            frame = np.asarray(raw_frame, dtype=np.float64)
            if output is None:
                frame_shape = frame.shape
                elements = int(frame.size) * count
                # Every scalar requires >=1 encoded byte. Reject impossible
                # declarations before a corrupt tiny file requests huge RAM.
                if self.size is not None and (count > self.size or elements > self.size):
                    raise cbor2.CBORDecodeEOF("declared frame array cannot fit in the input")
                output = np.empty((count,) + frame_shape, dtype=np.float64, order="C")
                self.stats.dense_bytes += int(output.nbytes)
            elif frame.shape != frame_shape:
                raise ValueError("setting an array element with a sequence: inconsistent vert_frames shape")
            self.stats.largest_frame_values = max(self.stats.largest_frame_values, int(frame.size))
            output[index] = frame
            # Explicitly release the temporary tree before decoding a sibling.
            del raw_frame, frame
        return output


def _finish(env):
    payload = reference._unwrap(
        env, reference.KIND_SCENE,
        hint="the payload was produced by an older addon; re-Transfer the scene from the Blender addon",
    )
    if not isinstance(payload, list):
        raise reference.CborSchemaError("scene payload must be a list of groups")
    for group in payload:
        if not isinstance(group, dict):
            raise reference.CborSchemaError("scene group must be a map")
        for obj in group.get("object", []) or []:
            if isinstance(obj, dict):
                reference._rehydrate_scene_object(obj)
    return payload


def load_scene_stream(stream: BinaryIO, *, size: int | None = None,
                      stats: DecodeStats | None = None) -> list:
    """Consume one envelope; like cbor2.loads, ignore trailing bytes.

    This does not change the original envelope/version/kind or geometry checks.
    Malformed numeric frame arrays can fail conversion earlier than the old
    reader (which defers conversion to _populate_static); they are not accepted.
    """
    if stats is None:
        stats = DecodeStats()
    start = stream.tell()
    root_end = _guard_root(stream)
    bounded = _RootBoundedReader(stream, root_end)
    return _finish(_Reader(bounded, root_end - start, stats).envelope())


def loads_scene(blob: bytes, *, stats: DecodeStats | None = None) -> list:
    """Test/convenience entry point; production candidate must use the file API."""
    return load_scene_stream(io.BytesIO(blob), size=len(blob), stats=stats)


def load_scene_file(path: str, *, stats: DecodeStats | None = None) -> list:
    """Read a matching scene without materializing its file bytes or all frames."""
    with open(path, "rb") as stream:
        first = stream.read(1)
        if not reference.is_cbor(first):
            raise reference.CborSchemaError(
                "data.pickle is a pre-envelope raw pickle written by an old addon; "
                "this build cannot read it. Re-Transfer from the Blender addon."
            )
        stream.seek(0)
        return load_scene_stream(stream, size=os.fstat(stream.fileno()).st_size, stats=stats)
