"""Collision-filter build acknowledgement; immutable data, no Blender I/O.

The request nonce is the existing successful atomic upload_id, not a new
parameter. Thus the original producer byte hash and Run drift check agree.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import PurePosixPath

CAPABILITY = "ppf_collision_filter_v1"
RULE = "symmetric_membership_mask_and_v1"
RECEIPT_FILE = "collision_filter_receipt.cbor"
MANIFEST_FILE = "collision_filter.cbor"


class CollisionReceiptError(ValueError):
    pass


def fail(detail):
    raise CollisionReceiptError(
        "Collision filter validation failed: " + detail
        + ". Use a frontend supporting collision filters and Transfer again."
    )


def _uuid(value):
    try:
        parsed = uuid.UUID(value)
        if str(parsed) != value or parsed.int == 0:
            fail("noncanonical scene/object UUID")
    except (TypeError, ValueError, AttributeError):
        fail("invalid scene/object UUID")
    return value


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        fail("invalid SHA256")
    return value


def token(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{12}", value):
        fail("missing or invalid atomic upload_id")
    return value


def contract_hash(config):
    if (not isinstance(config, dict) or set(config) != {"rule", "objects"}
            or config["rule"] != RULE or not isinstance(config["objects"], dict)):
        fail("invalid author contract")
    for uid, row in config["objects"].items():
        _uuid(uid)
        if (not isinstance(row, dict) or set(row) != {"enabled", "groups", "mask"}
                or type(row["enabled"]) is not bool
                or any(type(row[k]) is not int or not 0 <= row[k] <= 65535
                       for k in ("groups", "mask"))):
            fail("author contract requires boolean enabled and uint16 groups/mask")
    return hashlib.sha256(json.dumps(config, sort_keys=True,
        separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def required_contract(config):
    if config is None:
        return None
    digest = contract_hash(config)
    return digest if any(row["enabled"] for row in config["objects"].values()) else None


def _envelope(raw, kind, version=1):
    from .module import get_cbor2
    try:
        cbor = get_cbor2()
        value = cbor.loads(raw)
        # This protocol's producer uses the default cbor2.dumps wire shape.
        # Round-trip exactness rejects trailing data/duplicate keys without
        # a stream decoder (the packaged 6.0.1 decoder rejects this real seal).
        if cbor.dumps(value) != raw:
            fail("noncanonical or trailing CBOR data")
    except Exception as exc:
        if isinstance(exc, CollisionReceiptError):
            raise
        fail("missing or damaged " + kind)
    if (not isinstance(value, dict) or set(value) != {"version", "kind", "payload"}
            or type(value["version"]) is not int or value["version"] != version
            or value["kind"] != kind or not isinstance(value["payload"], dict)):
        fail("unsupported " + kind + " envelope")
    return value["payload"]


def uploaded_contract(param):
    return required_contract(_envelope(param, "Param", 2).get("collision_filter_v1"))


@dataclass(frozen=True)
class Expectation:
    author_hash: str
    data_hash: str
    param_hash: str
    upload_id: str = ""
    root: str = ""


@dataclass(frozen=True)
class VerifiedSeal:
    scene_uuid: str
    manifest_hash: str
    inputs_hash: str


def verify(receipt_raw, manifest_raw, expected, previous=None):
    receipt = _envelope(receipt_raw, "CollisionFilterReceipt")
    keys = {"capability", "request_token", "scene_uuid", "author_contract_sha256",
            "scene_inputs_sha256", "manifest_sha256"}
    if set(receipt) != keys or receipt["capability"] != CAPABILITY:
        fail("old or unsupported collision receipt capability")
    if token(receipt["request_token"]) != token(expected.upload_id):
        fail("receipt belongs to a different atomic upload")
    if _hash(receipt["author_contract_sha256"]) != expected.author_hash:
        fail("receipt does not acknowledge the uploaded author contract")
    manifest_hash = hashlib.sha256(manifest_raw).hexdigest()
    if _hash(receipt["manifest_sha256"]) != manifest_hash:
        fail("receipt does not match the downloaded sealed manifest bytes")
    payload = _envelope(manifest_raw, "CollisionFilterInput")
    if (set(payload) != {"capability", "rule", "complete", "dynamic_vertices",
            "static_vertices", "scene_uuid", "author_contract_sha256", "objects",
            "inputs", "scene_inputs_sha256"}
            or payload["capability"] != CAPABILITY or payload["rule"] != RULE
            or payload["complete"] is not True):
        fail("manifest is unsupported or build seal is incomplete")
    scene_uuid = _uuid(payload["scene_uuid"])
    if receipt["scene_uuid"] != scene_uuid:
        fail("receipt and manifest describe different scene builds")
    if payload["author_contract_sha256"] != expected.author_hash:
        fail("manifest author contract mismatch")
    for key in ("dynamic_vertices", "static_vertices"):
        if type(payload[key]) is not int or payload[key] < 0:
            fail("invalid manifest vertex count")
    if not isinstance(payload["objects"], list) or not isinstance(payload["inputs"], list):
        fail("invalid manifest inventory")
    lines = ["ppf.collision_filter/v1",
             f'{payload["dynamic_vertices"]}\t{payload["static_vertices"]}',
             scene_uuid + "\t" + expected.author_hash]
    previous_uid = ""
    for row in payload["objects"]:
        if not isinstance(row, dict) or set(row) != {"object_uuid", "pool", "packed", "vertices"}:
            fail("invalid manifest object record")
        uid = _uuid(row["object_uuid"])
        if uid <= previous_uid or row["pool"] not in ("dynamic", "static"):
            fail("duplicate, unsorted or unknown manifest object")
        previous_uid = uid
        if type(row["packed"]) is not int or not 0 <= row["packed"] <= 0xffffffff:
            fail("invalid packed collision filter")
        vertices = row["vertices"]
        count = payload[row["pool"] + "_vertices"]
        if (not isinstance(vertices, list) or not vertices
                or any(type(v) is not int or not 0 <= v < count for v in vertices)
                or len(set(vertices)) != len(vertices)):
            fail("invalid manifest vertex membership")
        lines.append(f'{uid}\t{row["pool"]}\t{row["packed"]}\t' + ",".join(map(str, vertices)))
    previous_path = ""
    for row in payload["inputs"]:
        if not isinstance(row, dict) or set(row) != {"relative_path", "byte_length", "sha256"}:
            fail("invalid manifest input record")
        path = row["relative_path"]
        if (not isinstance(path, str) or not path or "\\" in path or "\t" in path or "\n" in path
                or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts
                or path <= previous_path or type(row["byte_length"]) is not int or row["byte_length"] < 0):
            fail("invalid or unsorted manifest input path")
        previous_path = path
        lines.append(f'{path}\t{row["byte_length"]}\t{_hash(row["sha256"])}')
    digest = hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()
    if digest != _hash(payload["scene_inputs_sha256"]) or digest != _hash(receipt["scene_inputs_sha256"]):
        fail("downloaded manifest seal digest mismatch")
    seal = VerifiedSeal(scene_uuid, manifest_hash, digest)
    if previous is not None and seal != previous:
        fail("sealed scene build changed after READY validation")
    return seal
