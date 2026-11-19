"""Bounded hard authoring and exact V7 source-root union: actual native frames."""
from copy import deepcopy
import numpy as np
from . import _joint_d6_hard_v7_ as d6
from . import _angular_hard_hinge_v5_ as hinge
from ._pdrd_native_frames_ import native_body_frames, rebind_joint_payload, WORLD_UUID

def version_features(payload, base_features, fail):
    joints=payload["joints"]
    h=sum(hinge.HARD_HINGE_KEY in j for j in joints)
    d=sum(d6.D6_KEY in j for j in joints)
    if h and d or h>2 or d>1:
        fail("G2_HARD_UNSUPPORTED_DOMAIN", "hard_joint.count")
    if h or d:
        if any("angular_soft_limits_v1" in j or "angular_drives_v1" in j for j in joints):
            fail("G2_HARD_UNSUPPORTED_DOMAIN", "hard+soft_extension")
        version=5 if h else 7
        if payload.get("version",version)!=version:
            fail("G2_HARD_UNSUPPORTED_DOMAIN", "payload.version")
        return version, sorted(base_features+[hinge.FEATURE_HARD_HINGE if h else d6.D6_FEATURE])
    if "version" in payload:
        fail("G2_HARD_UNSUPPORTED_DOMAIN", "payload.version")
    return None

def bind_fixed(fixed,payload,authored_frames,fail,root_uuid=None):
    """Rebind only through actual built UUID membership/initial native fits."""
    version=payload.get("version")
    if version not in (5,7):
        return deepcopy(payload)
    root_payload=getattr(fixed,"_pdrd_root_motion_payload",None)
    if root_payload is not None:
        root_uuid=known_source_v7(payload,root_payload,getattr(fixed,"_pdrd_root_motion_version",None),fail)
    if getattr(fixed,"_pdrd_root_motion_payload_v2",None) is not None:
        fail("G2_HARD_UNSUPPORTED_DOMAIN","hard+root_batch")
    for name in ("_translation_lock","_rotation_lock"):
        table=getattr(fixed,name,None)
        if table is not None and len(table):
            fail("G2_HARD_UNSUPPORTED_DOMAIN", "hard+lock")
    if version==7:
        return d6.bind_payload_v7(fixed,payload,authored_frames,root_uuid=root_uuid,fail=fail)[0]
    frames=native_body_frames(fixed)
    bound=(rebind_joint_payload(payload,authored_frames,frames,preserve_world_joint_orientations=True)
           if authored_frames is not None else deepcopy(payload))
    rows=np.asarray(fixed._pdrd_body_rows,dtype=float).reshape(-1,23)
    if len(frames)<2 or np.any(rows[:,16]!=0.):
        fail("G2_HARD_UNSUPPORTED_DOMAIN", "pdrd_body.joint_mode")
    pairs=set(); active=[]
    for j in bound["joints"]:
        p,c=j["parent_body_uuid"],j["child_body_uuid"]
        pair=tuple(sorted((p,c)))
        if p==WORLD_UUID or p not in frames or c not in frames or p==c or pair in pairs:
            fail("G2_HARD_UNSUPPORTED_DOMAIN", "joints.graph", j["joint_uuid"])
        pairs.add(pair)
        if hinge.HARD_HINGE_KEY in j and j["enabled"]:active.append(j)
    if active:
        if len(frames)!=len(active)+1 or len(active)>2:
            fail("G2_HARD_UNSUPPORTED_DOMAIN", "joints.graph")
        parents={};children={};nodes=set()
        for j in active:
            p,c=j["parent_body_uuid"],j["child_body_uuid"]
            if c in parents or p in children:fail("G2_HARD_UNSUPPORTED_DOMAIN", "joints.graph")
            parents[c]=p;children[p]=c;nodes.update((p,c))
        roots=nodes-set(parents)
        if len(roots)!=1 or len(nodes)!=len(active)+1:fail("G2_HARD_UNSUPPORTED_DOMAIN", "joints.graph")
        walked=set();node=next(iter(roots))
        while node is not None and node not in walked:
            walked.add(node);node=children.get(node)
        if walked!=nodes:fail("G2_HARD_UNSUPPORTED_DOMAIN", "joints.graph")
        used=set()
        for node in nodes:
            row=rows[frames[node]["runtime_body_index"]];start,count=int(row[0]),int(row[1])
            used.update(map(int,fixed._pdrd_vert_list[start:start+count]))
        if any(used.intersection(map(int,pin.index)) for pin in fixed._pin):
            fail("G2_HARD_UNSUPPORTED_DOMAIN", "pin.index")
    for j in active:
        coordinate=d6._chart(frames[j["parent_body_uuid"]],frames[j["child_body_uuid"]],j)
        if coordinate is None:fail("G2_HARD_GEOMETRY_FAULT", "initial.angular_chart",j["joint_uuid"])
        if abs(coordinate[1])>5e-4 or abs(coordinate[2])>5e-4:
            fail("G2_HARD_INITIAL_INFEASIBLE", "initial.swing",j["joint_uuid"])
        twist=j[hinge.HARD_HINGE_KEY]["twist_x"]
        if twist["mode"]=="bounded" and not twist["lower_rad"]-5e-4<=coordinate[0]<=twist["upper_rad"]+5e-4:
            fail("G2_HARD_INITIAL_INFEASIBLE", "initial.twist",j["joint_uuid"])
    return bound

def known_source_v7(payload,root_payload,root_version,fail):
    joints=payload.get("joints",[]);bodies=root_payload.get("bodies",[])
    if payload.get("version")!=7 or root_version!=1 or len(joints)!=1 or len(bodies)!=1:
        fail("G2_HARD_UNSUPPORTED_DOMAIN","single_D6_one_V1_source_root")
    j=joints[0];uuid=bodies[0]["body_uuid"]
    if d6.D6_KEY not in j or hinge.HARD_HINGE_KEY in j or not j.get("enabled") or j.get("parent_body_uuid")!=uuid:
        fail("ROOT_HARD_SOURCE","root.body_uuid")
    return uuid
