# File: blender_addon/core/encoder/cbor_encode.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
"""CBOR upload encoder.

Used by ``encode_obj`` (mesh.py) and ``encode_param`` (params.py).
Output wraps the producer dict in the schema-version envelope
``{version, kind, payload}`` defined by
``crates/ppf-cts-formats/src/envelope.rs`` and consumed by the Rust
server.

cbor2 is imported lazily so addon load doesn't fail on installs where
the wheel hasn't been pulled in yet (e.g. the test rig, which sideloads
the addon and bypasses the manifest's wheels list). Production users
get cbor2 via ``blender_manifest.toml``; the rig prepares it through
``debug/orchestrator.py``'s ``warmup_addon_install``.
"""

from __future__ import annotations

import numpy as np

SCHEMA_VERSION = 2


def _write_xyz_float_rows(encoder, value):
    """Write large capture xyz rows without Python-per-coordinate objects."""
    if (type(value) not in (np.ndarray, np.memmap) or value.ndim != 3
            or value.shape[-1] != 3 or value.dtype.kind != "f"
            or value.dtype.itemsize not in (4, 8) or not value.shape[1]
            or not hasattr(encoder, "write")):
        return False
    # This path is only used for the default noncanonical wire format.
    # Keep float encoder compatibility explicit, including the sign of zero.
    from ..module import get_cbor2
    if get_cbor2().dumps([0., -0., 1., -1.]) != bytes.fromhex(
            "84fb0000000000000000fb8000000000000000fb3ff0000000000000fbbff0000000000000"):
        return False
    encoder.encode_length(4, len(value))
    for frame in value:
        if not np.isfinite(frame).all():
            # cbor2 uses special encodings for infinities and NaN.
            encoder.encode(frame)
            continue
        encoder.encode_length(4, len(frame))
        wire_dtype = np.dtype([
            ("array", "u1"), ("float1", "u1"), ("x", ">f8"),
            ("float2", "u1"), ("y", ">f8"),
            ("float3", "u1"), ("z", ">f8"),
        ])
        for i in range(0, len(frame), 65536):
            block = frame[i:i + 65536]
            rec = np.empty(len(block), dtype=wire_dtype)
            rec["array"] = 0x83
            rec["float1"] = rec["float2"] = rec["float3"] = 0xfb
            rec["x"], rec["y"], rec["z"] = block.T
            encoder.write(rec.tobytes())
    return True


def _numpy_default(encoder, value):
    """cbor2 fallback for non-natively-encodable types.

    The producer dicts contain numpy arrays (vert / face / edge /
    transform) and numpy scalars (np.float32(...) for sim params).
    cbor2 doesn't know either, so we route through ``.tolist()`` /
    ``.item()`` to get plain Python types — which is also what the
    Rust serde schema decodes into (Vec<[f32; N]>, primitives).

    Tuples are already CBOR-encodable as arrays without help, so we
    don't override them here.
    """
    if isinstance(value, np.ndarray):
        # Preserve the original definite-length CBOR array and scalar encoding,
        # without materializing a whole captured animation as Python objects.
        # Each row goes through this same encoder; small leaves retain tolist's
        # exact dtype/scalar behavior. No samples, dimensions or proofs change.
        if value.ndim > 0 and value.nbytes > 1024 * 1024:
            if _write_xyz_float_rows(encoder, value):
                return
            encoder.encode_length(4, len(value))
            for row in value:
                encoder.encode(row)
        else:
            encoder.encode(value.tolist())
        return
    if isinstance(value, np.generic):
        encoder.encode(value.item())
        return
    raise TypeError(
        f"CBOR encoder cannot serialize {type(value).__name__} "
        f"(value: {value!r})"
    )


def dumps_envelope(kind: str, payload) -> bytes:
    """Wrap ``payload`` in the schema-version envelope and emit CBOR bytes.

    Args:
        kind: payload type tag, e.g. ``"Scene"`` or ``"Param"``. Must
            match the consumer's expected kind or it'll be rejected.
        payload: producer dict (data tree from ``_build_obj_data`` or
            ``_build_param_dict``). Numpy arrays and scalars inside
            are converted via ``_numpy_default``.

    Returns:
        CBOR-encoded bytes ready for ``upload_atomic``.
    """
    from ..module import get_cbor2

    cbor2 = get_cbor2()

    envelope = {"version": SCHEMA_VERSION, "kind": kind, "payload": payload}
    import tempfile

    print(f"[PPF encode] {kind}: disk CBOR begin", flush=True)
    # TMPDIR must name disk-backed storage in the launch environment.
    # The producer tree remains unchanged; only the output sink differs.
    with tempfile.TemporaryFile(mode="w+b", prefix="ppf-cbor-") as stream:
        cbor2.dump(envelope, stream, default=_numpy_default)
        stream.flush()
        size = stream.tell()
        print(f"[PPF encode] {kind}: disk CBOR complete ({size} bytes)", flush=True)
        stream.seek(0)
        data = stream.read()
        if len(data) != size:
            raise OSError("Incomplete staged CBOR read")
    print(f"[PPF encode] {kind}: bytes ready", flush=True)
    return data
