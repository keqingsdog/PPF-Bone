"""Initial prescribed FEM roots, bound to real topology and hard pin inputs."""
from copy import deepcopy
import hashlib,json,math
from pathlib import Path
import numpy as np
FEATURE='fem_prescribed_roots_v1'
FILE='bin/fem_body_controls_v1.json'
RECEIPT='fem_body_control_receipt.cbor'
KIND='ppf.fem_body_controls'

def fail(detail):raise ValueError('FEM_BODY_CONTROL_INVALID: '+detail)
def signature(root):
    from ._body_control_intent_ import validate_author_intent
    if (not isinstance(root,dict) or root.get('mode') not in ('static','kinematic')
            or root.get('control_events')):fail('initial STATIC/KINEMATIC only; events unsupported')
    keys=root.get('keys')
    if not isinstance(keys,list) or not keys:fail('missing trajectory')
    if root['mode']=='static' and len(keys)!=1:fail('STATIC requires exactly one key')
    if root['mode']=='kinematic' and len(keys)<2:fail('KINEMATIC requires at least two keys')
    times=[]
    for k in keys:
        if set(k)!={'time_s','center_world_m','rotation_world_wxyz'}:fail('unknown key fields')
        t=k['time_s'];q=np.asarray(k['rotation_world_wxyz'],dtype=float);p=np.asarray(k['center_world_m'],dtype=float)
        if isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t) or (times and t<=times[-1]):fail('invalid key times')
        if p.shape!=(3,) or q.shape!=(4,) or not np.isfinite(p).all() or not np.isfinite(q).all() or abs(float(q@q)-1)>1e-6:fail('invalid rigid trajectory')
        times.append(t)
    if times[0]!=0:fail('trajectory must start at zero')
    return tuple(times)

def require_root_boundary(row):
    if row.get('parent_body_uuid') is not None:
        fail('当前版本只支持软体根骨骼的静态或运动学控制；此骨骼已有物理父级或自动父锚点，请保持动态模式')

def install(fixed,document,rows,stats):
    if any('fem_root_control_events_v1' in row for row in rows):
        from ._fem_body_control_events_ import install as install_events
        return install_events(fixed,document,rows,stats)
    from . import _blender_bones_ as bb
    from ._scene_pin_ import PinData,TransformKeyframeOperation
    from ._pdrd_native_frames_ import _rotation,_matrix_quat
    from ._build_contract_ import contract_for
    controlled=[row for row in rows if 'fem_root_control_v1' in row]
    if not controlled:return
    for row in controlled:require_root_boundary(row)
    if FEATURE not in contract_for(fixed).capabilities:fail('selected product does not support '+FEATURE)
    world=np.asarray(fixed.vertex(),dtype=float);newpins=[];bodies=[];author=[]
    for row in sorted(controlled,key=lambda row:row['body_uuid']):
        uid=row['body_uuid'];s=stats.get(uid)
        if row['fem_root_control_v1'] is not True or s is None or s['dynamics_type']!='SOLID':fail('marker must own an actual SOLID object')
        if row['root_mode'] not in ('STATIC','KINEMATIC'):fail('invalid root mode')
        root=row.get('root_motion');times=signature(root)
        if root.get('body_uuid')!=uid or root['mode']!=row['root_mode'].lower():fail('root identity/mode differs')
        ids=sorted(map(int,s['vertex_indices']))
        if not ids or len(set(ids))!=len(ids):fail('invalid vertex inventory')
        if any(set(ids).intersection(map(int,p.index)) for p in fixed._pin+newpins):fail('arbitrary user pin overlap')
        dmap=np.asarray(fixed._vert[0])[ids]
        if len(set(map(int,dmap)))!=1:fail('root vertices have different displacement groups')
        displacement=np.asarray(fixed._displacement[int(dmap[0])],dtype=float)
        if np.any(np.asarray(fixed._vel)[ids]!=0):fail('prescribed FEM initial velocity unsupported')
        P=bb.S@bb.matrix(row['proxy_world_matrix_blender'],'FEM.proxy')
        initial=root['initial_frame'];A=np.eye(4);A[:3,:3]=_rotation(initial['quaternion_wxyz']);A[:3,3]=initial['translation_m']
        reference=np.linalg.inv(A)@P
        local=(world[ids]-P[:3,3])@P[:3,:3]
        operations=[]
        if row['root_mode']=='KINEMATIC':
            poses=[]
            for key in root['keys']:
                pose=np.eye(4);pose[:3,:3]=_rotation(key['rotation_world_wxyz']);pose[:3,3]=key['center_world_m'];poses.append(pose@reference)
            operations=[TransformKeyframeOperation(local_vert=local,times=list(times),translations=[m[:3,3] for m in poses],quaternions=[_matrix_quat(m[:3,:3]) for m in poses],scales=[np.ones(3) for _ in poses],segments=[{'interpolation':'LINEAR'} for _ in poses[1:]],rest_translation=displacement)]
            first=operations[0].apply(world[ids]-displacement,0)+displacement
            if not np.allclose(first,world[ids],rtol=0,atol=1e-7):fail('initial prescribed position differs from actual rest')
        pin_index=len(fixed._pin)+len(newpins)
        newpins.append(PinData(index=ids,operations=operations,pin_group_id='fem-root:'+uid,hide_in_preview=True))
        bodies.append({'body_uuid':uid,'pin_block_index':pin_index,'mode':row['root_mode'],'vertex_indices':ids})
        author.append({'body_uuid':uid,'source':row['root_motion_source'],'motion':deepcopy(root)})
    fixed._pin.extend(newpins)
    fixed._fem_body_controls_payload={'version':1,'kind':KIND,'scene_uuid':fixed._scene_uuid,'bodies':bodies}
    fixed._fem_body_controls_author=author
    from ._body_control_receipt_ import record_author
    record_author(fixed,document)

def write_sidecar(fixed,path):
    from ._fem_body_control_events_ import write_sidecar as write_events
    write_events(fixed,path)
    payload=getattr(fixed,'_fem_body_controls_payload',None)
    if payload is not None:(Path(path)/FILE).write_text(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False),encoding='utf-8')

def write_receipt(fixed,path):
    from ._fem_body_control_events_ import write_receipt as write_events
    write_events(fixed,path)
    import cbor2
    payload=getattr(fixed,'_fem_body_controls_payload',None)
    if payload is None:return
    upload=fixed._collision_filter_upload_context;root=Path(upload['root']);token=(root/'upload_id.txt').read_text().strip()
    if token!=upload['request_token']:fail('upload changed while sealing')
    path=Path(path);raw={name:(path/rel).read_bytes() for name,rel in [('context','bin/pdrd_scene_context.cbor'),('joint','bin/pdrd_ball_anchor.cbor'),('sidecar',FILE)]}
    context=cbor2.loads(raw['context'])['payload'];joint=cbor2.loads(raw['joint'])['payload'];inventory={r['relative_path']:r for r in context['inputs']}
    if not context['complete'] or FEATURE not in joint['required_features'] or inventory.get(FILE,{}).get('sha256')!=hashlib.sha256(raw['sidecar']).hexdigest():fail('missing final consumed seal')
    pinfiles={k:v for k,v in inventory.items() if any(k=='bin/pin-'+prefix+str(b['pin_block_index'])+'.bin' for b in payload['bodies'] for prefix in ('ind-','pullw-')) or any(k.startswith('bin/pin-'+str(b['pin_block_index'])+'-op-') for b in payload['bodies'])}
    receipt={'capability':FEATURE,'request_token':token,'scene_uuid':fixed._scene_uuid,'author_contract_sha256':fixed._body_controls_author_contract_sha256,'scene_inputs_sha256':context['scene_inputs_sha256'],'controls':fixed._fem_body_controls_author,'pin_inputs':list(pinfiles.values()),**{k+'_sha256':hashlib.sha256(v).hexdigest() for k,v in raw.items()}}
    if (root/'upload_id.txt').read_text().strip()!=token:fail('upload changed before publication')
    (path/RECEIPT).write_bytes(cbor2.dumps({'version':1,'kind':'FEMBodyControlReceipt','payload':receipt}))


CAPABILITY_DOCUMENT={'kind':'ppf.fem_body_controls.capability','version':1,'required_feature':FEATURE,'supported':True,'semantic_version':1,'modes':['STATIC','KINEMATIC'],'scope':'initial_only','midrun_switching':False,'requires_matching_native_kernel_build':True,'supported_backends':['cuda'],'backend':'cuda'}
def capability(profile):
    import subprocess,os
    from . import _r14_product_profile as profiles
    result=subprocess.run([str(profile.exe_path),'--fem-body-controls-capability-json'],cwd=str(profile.runtime_dir),env=profiles.solver_child_env(profile,os.environ),capture_output=True,timeout=30)
    if result.returncode!=0:fail('native capability probe refused')
    doc=json.loads(result.stdout)
    if doc!=CAPABILITY_DOCUMENT or type(doc.get('version')) is not int or type(doc.get('semantic_version')) is not int:fail('native capability mismatch')
    return doc
