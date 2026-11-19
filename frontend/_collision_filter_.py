"""Immutable mesh collision groups, independent of bone/joint authoring."""
from __future__ import annotations
import hashlib
import json
import os
import uuid
import numpy as np

RULE = "symmetric_membership_mask_and_v1"
CAPABILITY = "ppf_collision_filter_v1"
FILE = "collision_filter.cbor"
RECEIPT_FILE = "collision_filter_receipt.cbor"

def contract_sha256(config):
    normalize(config)
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()


def fail(detail):
    raise ValueError(f"COLLISION_FILTER_INVALID: {detail}")


def normalize(config):
    if not isinstance(config, dict) or set(config) != {"rule", "objects"} or config["rule"] != RULE or not isinstance(config["objects"], dict):
        fail("unknown rule or fields")
    packed = {}
    for uid, row in config["objects"].items():
        try:
            if str(uuid.UUID(uid)) != uid or uuid.UUID(uid).int == 0:
                fail("noncanonical object UUID")
        except (ValueError, TypeError, AttributeError):
            fail("invalid object UUID")
        if (not isinstance(row, dict) or set(row) != {"enabled", "groups", "mask"}
                or type(row["enabled"]) is not bool
                or any(type(row[k]) is not int or not 0 <= row[k] <= 65535 for k in ("groups", "mask"))):
            fail("expected boolean and two u16 values")
        packed[uid] = ((row["groups"] | row["mask"] << 16) or 0x00010000) if row["enabled"] else 0
    return packed


def prepare(scene, config, param_data):
    packed = normalize(config)
    sources = {}
    for group in param_data["group"]:
        for uid in group[2]:
            if uid in sources:
                fail("object appears in multiple physical groups")
            sources[uid] = group[0]
    if set(packed) - set(sources):
        fail("filter object absent from active physical groups")
    for uid in packed:
        scene.select(uid)
    scene._collision_filters_v1 = packed
    scene._collision_filter_author_sha256 = contract_sha256(config)


def verify(fixed, config):
    expected = normalize(config)
    built = {row["object_uuid"]: row for row in getattr(fixed, "_collision_filter_objects", [])}
    for uid, value in expected.items():
        row = built.get(uid)
        if row is None or row["packed"] != value:
            fail("settings were not installed before Scene.build")


def digest(payload):
    lines = ["ppf.collision_filter/v1", f'{payload["dynamic_vertices"]}\t{payload["static_vertices"]}']
    lines.append(f'{payload["scene_uuid"]}\t{payload["author_contract_sha256"]}')
    for row in payload["objects"]:
        lines.append(f'{row["object_uuid"]}\t{row["pool"]}\t{row["packed"]}\t' + ",".join(map(str,row["vertices"])))
    for row in payload["inputs"]:
        lines.append(f'{row["relative_path"]}\t{row["byte_length"]}\t{row["sha256"]}')
    return hashlib.sha256(("\n".join(lines)+"\n").encode()).hexdigest()


def seal(fixed, path, complete):
    dynamic = getattr(fixed, "_collision_filter", None)
    static = getattr(fixed, "_static_collision_filter", None)
    if dynamic is None and static is None:
        return
    import cbor2
    payload = {"capability": CAPABILITY, "rule": RULE, "complete": bool(complete),
               "scene_uuid":fixed._scene_uuid,"author_contract_sha256":fixed._collision_filter_author_sha256,
               "dynamic_vertices": len(fixed._vert[1]),
               "static_vertices": len(fixed._static_vert[1]) if fixed._static_vert is not None else 0,
               "objects": sorted(fixed._collision_filter_objects, key=lambda r:r["object_uuid"]),
               "inputs": fixed._r1_inventory(path), "scene_inputs_sha256": ""}
    payload["scene_inputs_sha256"] = digest(payload)
    with open(os.path.join(path,FILE),"wb") as f:
        f.write(cbor2.dumps({"version":1,"kind":"CollisionFilterInput","payload":payload}))


def freeze_upload(root):
    """Freeze the existing atomic upload nonce before ingest, clear prior acknowledgement."""
    receipt=os.path.join(root,"session",RECEIPT_FILE)
    try:os.remove(receipt)
    except FileNotFoundError:pass
    try:
        with open(os.path.join(root,"upload_id.txt"),encoding="ascii") as f:token=f.read().strip()
    except FileNotFoundError:return None
    if len(token)!=12 or any(c not in "0123456789abcdef" for c in token):fail("invalid atomic upload nonce")
    return {"root":os.path.abspath(root),"request_token":token}


def write_receipt(fixed,path):
    """Only a complete, successful sealed Session build may acknowledge authorship."""
    import cbor2
    if getattr(fixed,"_collision_filter",None) is None and getattr(fixed,"_static_collision_filter",None) is None:
        return
    with open(os.path.join(path,FILE),"rb") as f: raw=f.read()
    envelope=cbor2.loads(raw)
    payload=envelope["payload"]
    if not payload["complete"] or payload["capability"]!=CAPABILITY or payload["scene_inputs_sha256"]!=digest(payload):
        fail("receipt requires a complete successful seal")
    context=getattr(fixed,"_collision_filter_upload_context",None)
    if context is None:fail("filtered build requires an atomic upload nonce frozen before ingest")
    with open(os.path.join(context["root"],"upload_id.txt"),encoding="ascii") as f:current=f.read().strip()
    if current!=context["request_token"]:fail("atomic upload changed while this build was running")
    receipt={"version":1,"kind":"CollisionFilterReceipt","payload":{"capability":CAPABILITY,
             "request_token":context["request_token"],"scene_uuid":fixed._scene_uuid,
             "manifest_sha256":hashlib.sha256(raw).hexdigest(),
             "author_contract_sha256":fixed._collision_filter_author_sha256,
             "scene_inputs_sha256":payload["scene_inputs_sha256"]}}
    with open(os.path.join(path,RECEIPT_FILE),"wb") as f:f.write(cbor2.dumps(receipt))


def require_command(path):
    """Append a real CLI capability requirement to the existing solver line."""
    with open(path,encoding="utf-8") as f:
        lines=f.readlines()
    found=0
    for i,line in enumerate(lines):
        if "--path " in line:
            found+=1
            if "--require-collision-filter-v1" not in line:
                lines[i]=line.rstrip("\r\n")+" --require-collision-filter-v1\n"
    if found != 1:
        fail("expected one actual solver command")
    with open(path,"w",encoding="utf-8",newline="") as f:
        f.writelines(lines)
