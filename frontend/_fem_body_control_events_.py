"""FEM roots event producer: actual material inventory, no PDRD stand-in."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

_shared_path=Path(__file__).resolve().parents[1]/'blender_addon/core/fem_body_control_events.py'
_spec=importlib.util.spec_from_file_location('_ppf_fem_event_projection',_shared_path)
p=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(p)
FEATURE=p.FEATURE
FILE=p.FILE
RECEIPT=p.RECEIPT

CAPABILITY_DOCUMENT={'kind':'ppf.fem_body_control_events.capability','version':1,'semantic_version':2,
 'required_feature':FEATURE,'supported':True,'supported_backends':['cuda'],'backend':'cuda',
 'scope':'fem_roots_only','modes':['FREE','STATIC','KINEMATIC'],'midrun_switching':True,
 'supported_event_pose_bindings':['current_hold','current_follow','absolute'],'resume_supported':False,
 'event_clock':'accepted_physical_boundary_v1','requires_matching_native_kernel_build':True}


def capability(profile):
    import subprocess
    from . import _r14_product_profile as profiles
    result=subprocess.run([str(profile.exe_path),'--fem-body-control-events-capability-json'],
        cwd=str(profile.runtime_dir),env=profiles.solver_child_env(profile,os.environ),capture_output=True,timeout=30)
    if result.returncode!=0:p.fail('native event capability refused')
    document=json.loads(result.stdout)
    if (document!=CAPABILITY_DOCUMENT or type(document.get('version')) is not int or type(document.get('semantic_version')) is not int
            or any(type(document.get(k)) is not bool for k in ('supported','midrun_switching','resume_supported','requires_matching_native_kernel_build'))):
        p.fail('native event capability differs from paired product')
    return document


def install(fixed,document,rows,stats):
    from ._fem_body_controls_ import require_root_boundary
    from ._build_contract_ import contract_for
    from ._body_control_receipt_ import record_author
    contract=contract_for(fixed)
    if FEATURE not in contract.capabilities:p.fail('所选求解器不支持软体根控制事件，请选择匹配的新产品')
    controlled=[row for row in rows if row.get(FEATURE) is True]
    if not controlled or any('fem_root_control_v1' in row for row in rows):p.fail('FEM v1/v2 controls cannot share a scene')
    bodies=[];author=[];claimed=set()
    for row in sorted(controlled,key=lambda r:r['body_uuid']):
        require_root_boundary(row)
        uid=row['body_uuid'];actual=stats.get(uid)
        if actual is None or actual['dynamics_type']!='SOLID':p.fail('event root must own actual SOLID material')
        ids=sorted(map(int,actual['vertex_indices']))
        if not ids or len(ids)!=len(set(ids)) or claimed.intersection(ids):p.fail('invalid full FEM node inventory')
        if any(other!=uid and set(value['vertex_indices']).intersection(ids) for other,value in stats.items()):p.fail('cross-body FEM node alias')
        if any(set(ids).intersection(map(int,pin.index)) for pin in fixed._pin):p.fail('任意固定点与软体事件控制节点重叠')
        if row['root_mode']!='FREE' and any((fixed._vel[i]!=0).any() for i in ids):p.fail('prescribed FEM initial velocity unsupported')
        body=p.project(row,ids)
        if row['root_mode']!='FREE':
            # Exactly the released v1 full-node initial-position gate, without
            # creating a pin surrogate. Initial FREE retains arbitrary accepted
            # shape and initial velocity; only source identity is checked there.
            import numpy as np
            from . import _blender_bones_ as bb
            from ._pdrd_native_frames_ import _rotation
            world=np.asarray(fixed.vertex(),dtype=float)[ids]
            P=bb.S@bb.matrix(row['proxy_world_matrix_blender'],'FEM.event.proxy')
            local=(world-P[:3,3])@P[:3,:3]
            key=body['keys'][0];D=body['reference_transform']
            A0=np.eye(4);A0[:3,:3]=_rotation(key['rotation_world_wxyz']);A0[:3,3]=key['center_world_m']
            reference=np.eye(4);reference[:3,:3]=_rotation(D['rotation_local_wxyz']);reference[:3,3]=D['translation_local_m']
            target=A0@reference
            first=local@target[:3,:3].T+target[:3,3]
            if not np.allclose(first,world,rtol=0,atol=1e-7):p.fail('initial FEM event prescribed nodes differ from actual rest')
        # Capability gates apply to intermediate policies, not the legitimate
        # initial STATIC/KIN absolute shape. Unsupported requests are not dropped.
        # Midrun ABSOLUTE is admitted by the paired native whole-node gate.
        # Do not replace authored absolute policies with current capture policies.
        bodies.append(body);claimed.update(ids)
        author.append({'body_uuid':uid,'source':row['root_motion_source'],'motion':deepcopy(row['root_motion'])})
    fixed._fem_body_control_events_payload={'version':2,'kind':'ppf.fem_body_controls','scene_uuid':fixed._scene_uuid,
        'interpolation':p.INTERPOLATION,'extrapolation':p.EXTRAPOLATION,'bodies':bodies}
    fixed._fem_body_control_events_author=author
    record_author(fixed,document)


def write_sidecar(fixed,path):
    payload=getattr(fixed,'_fem_body_control_events_payload',None)
    if payload is not None:
        if getattr(fixed,'_fem_body_controls_payload',None) is not None:p.fail('FEM v1/v2 payload conflict')
        (Path(path)/FILE).write_text(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False),encoding='utf8')


def write_receipt(fixed,path):
    import cbor2
    payload=getattr(fixed,'_fem_body_control_events_payload',None)
    if payload is None:return
    upload=fixed._collision_filter_upload_context;root=Path(upload['root']);token=(root/'upload_id.txt').read_text().strip()
    if token!=upload['request_token']:p.fail('upload changed while sealing FEM event input')
    path=Path(path)
    raw={name:(path/rel).read_bytes() for name,rel in [('context','bin/pdrd_scene_context.cbor'),('joint','bin/pdrd_ball_anchor.cbor'),('sidecar',FILE)]}
    context=cbor2.loads(raw['context'])['payload'];joint=cbor2.loads(raw['joint'])['payload']
    inventory={r['relative_path']:r for r in context['inputs']}
    if (context.get('complete') is not True or FEATURE not in joint['required_features']
            or 'fem_prescribed_roots_v1' in joint['required_features'] or 'bin/fem_body_controls_v1.json' in inventory
            or inventory.get(FILE,{}).get('sha256')!=hashlib.sha256(raw['sidecar']).hexdigest()):p.fail('missing consumed FEM event seal')
    receipt={'capability':FEATURE,'request_token':token,'scene_uuid':fixed._scene_uuid,
        'author_contract_sha256':fixed._body_controls_author_contract_sha256,'scene_inputs_sha256':context['scene_inputs_sha256'],
        'controls':fixed._fem_body_control_events_author,**{name+'_sha256':hashlib.sha256(data).hexdigest() for name,data in raw.items()}}
    if (root/'upload_id.txt').read_text().strip()!=token:p.fail('upload changed before FEM event receipt publication')
    (path/RECEIPT).write_bytes(cbor2.dumps({'version':2,'kind':'FEMBodyControlReceipt','payload':receipt}))
