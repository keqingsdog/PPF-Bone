"""Dedicated initial prescribed FEM seal; no PDRD identity substitution."""
import hashlib,json
from pathlib import PurePosixPath
from . import body_control_receipt as b
FEATURE='fem_prescribed_roots_v1'
RECEIPT_FILE='fem_body_control_receipt.cbor'
SIDECAR_FILE='bin/fem_body_controls_v1.json'

def signature(root,uid,source):
    if source not in ('POSE','OBJECT'):b.fail('invalid FEM author source')
    base=b._control_signature(root,uid)
    if root.get('control_events') or base[1] not in ('static','kinematic'):b.fail('unsupported FEM initial mode/events')
    program=tuple((k['time_s'],tuple(k['center_world_m']),tuple(k['rotation_world_wxyz'])) for k in root['keys'])
    for _,p,q in program:b._vector(list(p),3);b._rotation(list(q))
    return (base[0],base[1],program,base[3],source,'FEM')

def partition(controls):
    rigid=[];fem=[]
    for row in controls:
        if isinstance(row,tuple) and len(row)==4:rigid.append(row)
        elif isinstance(row,tuple) and len(row)==6 and row[5]=='FEM' and row[4] in ('POSE','OBJECT'):fem.append(row)
        else:b.fail('unknown control expectation family')
    return tuple(rigid),tuple(fem)

def verify(raw,expected):
    receipt=b._envelope(raw[RECEIPT_FILE],'FEMBodyControlReceipt',1)
    fields={'capability','request_token','scene_uuid','author_contract_sha256','scene_inputs_sha256','controls','pin_inputs','context_sha256','joint_sha256','sidecar_sha256'}
    if set(receipt)!=fields or receipt['capability']!=FEATURE:b.fail('invalid FEM control receipt')
    if b.token(receipt['request_token'])!=b.token(expected.upload_id) or b._hash(receipt['author_contract_sha256'])!=expected.author_hash:b.fail('stale or wrong FEM author receipt')
    names={'context':b.CONTEXT_FILE,'joint':b.JOINT_FILE,'sidecar':SIDECAR_FILE}
    hashes={key:hashlib.sha256(raw[path]).hexdigest() for key,path in names.items()}
    if any(b._hash(receipt[k+'_sha256'])!=v for k,v in hashes.items()):b.fail('FEM receipt bytes mismatch')
    ctx=b._envelope(raw[b.CONTEXT_FILE],'ppf.pdrd_scene_context',2)
    import cbor2
    envelope=cbor2.loads(raw[b.JOINT_FILE])
    if envelope.get('kind')!='ppf.pdrd_ball_anchor' or type(envelope.get('version')) is not int or envelope['version'] not in (3,9):b.fail('unsupported FEM joint envelope')
    joint=envelope['payload'];scene=b._uuid(ctx['scene_uuid'])
    if ctx.get('complete') is not True or ctx.get('context_version')!=2 or ctx.get('builder_contract_version')!=2:b.fail('incomplete FEM scene')
    if receipt['scene_uuid']!=scene or joint.get('scene_uuid')!=scene:b.fail('FEM scene identity mismatch')
    features=joint.get('required_features')
    if not isinstance(features,list) or features!=sorted(set(features)) or not {FEATURE,'fem_material_joints_v1'}.issubset(features):b.fail('FEM feature not advertised')
    lines=['ppf.pdrd_scene_context/v2',scene];last='';bodyindices=set()
    for body in ctx['bodies']:
        if set(body)!={'body_uuid','runtime_body_index','vertex_count'}:b.fail('invalid context body')
        uid=b._uuid(body['body_uuid']);idx=body['runtime_body_index'];count=body['vertex_count']
        if uid<=last or type(idx) is not int or idx<0 or idx in bodyindices or type(count) is not int or count<1:b.fail('invalid context membership')
        last=uid;bodyindices.add(idx);lines.append(f'{uid}\t{idx}\t{count}')
    if joint.get('bodies')!=[{'body_uuid':x['body_uuid'],'runtime_body_index':x['runtime_body_index']} for x in ctx['bodies']]:b.fail('context/joint membership differs')
    inv={};last=''
    for item in ctx['inputs']:
        if set(item)!={'relative_path','byte_length','sha256'}:b.fail('invalid FEM inventory')
        path=item['relative_path']
        if not isinstance(path,str) or path<=last or PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts or '\\' in path or '\t' in path or '\n' in path or type(item['byte_length']) is not int or item['byte_length']<0:b.fail('unsafe inventory')
        last=path;inv[path]=item;lines.append(f'{path}\t{item["byte_length"]}\t{b._hash(item["sha256"])}')
    absent=ctx['optional_absent']
    if absent!=sorted(set(absent)) or any(x not in {'dyn_param.txt','bin/rest_vert.bin','bin/rest_vert_anim.bin'} or x in inv for x in absent):b.fail('invalid optional inventory')
    lines.extend('optional_absent\t'+x for x in absent)
    digest=hashlib.sha256(('\n'.join(lines)+'\n').encode()).hexdigest()
    if any(b._hash(x)!=digest for x in (ctx['scene_inputs_sha256'],joint['scene_inputs_sha256'],receipt['scene_inputs_sha256'])):b.fail('FEM scene input digest mismatch')
    side=json.loads(raw[SIDECAR_FILE])
    if set(side)!={'version','kind','scene_uuid','bodies'} or type(side['version']) is not int or side['version']!=1 or side['kind']!='ppf.fem_body_controls' or side['scene_uuid']!=scene:b.fail('invalid FEM control sidecar')
    vertex_map=b._envelope(raw['map.pickle'],'VertexMap',2)
    map_item=inv.get('map.pickle')
    if map_item!={'relative_path':'map.pickle','byte_length':len(raw['map.pickle']),'sha256':hashlib.sha256(raw['map.pickle']).hexdigest()}:b.fail('vertex map differs from consumed input')
    if inv.get(SIDECAR_FILE)!={'relative_path':SIDECAR_FILE,'byte_length':len(raw[SIDECAR_FILE]),'sha256':hashes['sidecar']}:b.fail('FEM sidecar not consumed')
    rigid,wanted=partition(expected.controls);actual=[];last='';pinids=set();members=set()
    authored=receipt['controls']
    if not isinstance(authored,list):b.fail('missing FEM source declaration')
    author={x.get('body_uuid'):x for x in authored}
    if len(author)!=len(authored):b.fail('duplicate FEM author')
    for row in side['bodies']:
        if set(row)!={'body_uuid','pin_block_index','mode','vertex_indices'}:b.fail('invalid FEM controlled body')
        uid=b._uuid(row['body_uuid']);index=row['pin_block_index'];ids=row['vertex_indices']
        if uid<=last or type(index) is not int or index<0 or index in pinids or row['mode'] not in ('STATIC','KINEMATIC') or not isinstance(ids,list) or not ids or any(type(i) is not int or i<0 for i in ids) or ids!=sorted(set(ids)) or members.intersection(ids):b.fail('duplicate/invalid FEM control inventory')
        if uid not in vertex_map or sorted(vertex_map[uid])!=ids:b.fail('FEM root does not cover actual native object inventory')
        last=uid;pinids.add(index);members.update(ids)
        a=author.get(uid)
        if not isinstance(a,dict) or set(a)!={'body_uuid','source','motion'} or a['source'] not in ('POSE','OBJECT') or a['motion'].get('mode')!=row['mode'].lower():b.fail('FEM author/source/mode mismatch')
        actual.append(signature(a['motion'],uid,a['source']))
    if tuple(actual)!=wanted or set(author)!={r[0] for r in wanted}:b.fail('sealed FEM controls differ from uploaded request')
    pins=receipt['pin_inputs']
    if not isinstance(pins,list) or pins!=[inv[k] for k in sorted(inv) if any(k=='bin/pin-ind-'+str(i)+'.bin' or k.startswith('bin/pin-'+str(i)+'-op-') or k=='bin/pin-pullw-'+str(i)+'.bin' for i in pinids)]:b.fail('FEM pin inventory differs')
    for item in pins:
        path=item['relative_path']
        if path not in raw or len(raw[path])!=item['byte_length'] or hashlib.sha256(raw[path]).hexdigest()!=item['sha256']:b.fail('FEM pin input bytes differ from consumed seal')
    import struct
    for row in side['bodies']:
        path='bin/pin-ind-'+str(row['pin_block_index'])+'.bin'
        if path not in inv or inv[path]['byte_length']!=8*len(row['vertex_indices']):b.fail('missing FEM pin membership bytes')
        if list(struct.unpack('<'+'Q'*len(row['vertex_indices']),raw[path]))!=row['vertex_indices']:b.fail('pin membership differs from FEM body')
    return (scene,hashes['context'],hashes['joint'],hashes['sidecar'],digest,hashlib.sha256(raw[RECEIPT_FILE]).hexdigest())


def extra_files(raw,expected):
    if any(isinstance(row,tuple) and len(row)==8 for row in expected.controls):
        from .fem_body_control_events_receipt import partition as event_partition
        event_partition(expected.controls)
        return ()
    _,fem=partition(expected.controls)
    if not fem:return ()
    receipt=b._envelope(raw[RECEIPT_FILE],'FEMBodyControlReceipt',1)
    pins=receipt.get('pin_inputs')
    if not isinstance(pins,list) or len(pins)>200000:b.fail('invalid FEM pin inventory size')
    paths=[]
    import re
    for row in pins:
        path=row.get('relative_path') if isinstance(row,dict) else None
        if not isinstance(path,str) or not re.fullmatch(r'bin/pin-(?:ind-[0-9]+|[0-9]+-op-[0-9]+(?:-(?:time|translation|quaternion|scale|interp|handles))?)\.bin',path):b.fail('unsafe FEM pin input request')
        if path in paths:b.fail('duplicate FEM pin input request')
        paths.append(path)
    return tuple(paths)
