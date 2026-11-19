"""Exact v2 event-family seal; no mutable parallel upload state."""
import hashlib,json
from pathlib import PurePosixPath
from dataclasses import replace
from . import body_control_receipt as b
from . import fem_body_control_events as p
FEATURE=p.FEATURE
SIDECAR_FILE=p.FILE
RECEIPT_FILE=p.RECEIPT

def partition(controls):
    rigid=[];fem=[]
    for row in controls:
        if isinstance(row,tuple) and len(row)==4:rigid.append(row)
        elif (isinstance(row,tuple) and len(row)==8 and row[6:] == ('FEM_EVENTS',2)
              and row[4] in ('POSE','OBJECT')):fem.append(row)
        else:b.fail('unknown/mixed FEM event expectation family')
    if not fem:b.fail('missing FEM event expectation')
    return tuple(rigid),tuple(fem)

def required_files(expected):
    rigid,_=partition(expected.controls)
    return tuple(dict.fromkeys(([b.RECEIPT_FILE,b.CONTEXT_FILE,b.JOINT_FILE,b.SIDECAR_FILE] if rigid else [])+
           [RECEIPT_FILE,b.CONTEXT_FILE,b.JOINT_FILE,SIDECAR_FILE,'map.pickle']))

def verify(raw,expected):
    receipt=b._envelope(raw[RECEIPT_FILE],'FEMBodyControlReceipt',2)
    fields={'capability','request_token','scene_uuid','author_contract_sha256','scene_inputs_sha256','controls','context_sha256','joint_sha256','sidecar_sha256'}
    if set(receipt)!=fields or receipt['capability']!=FEATURE:b.fail('invalid FEM event receipt')
    if b.token(receipt['request_token'])!=b.token(expected.upload_id) or b._hash(receipt['author_contract_sha256'])!=expected.author_hash:b.fail('stale or wrong FEM event author receipt')
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
    if not isinstance(features,list) or features!=sorted(set(features)) or not {FEATURE,'fem_material_joints_v1'}.issubset(features) or 'fem_prescribed_roots_v1' in features:b.fail('FEM feature not advertised')
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
    def unique_object(pairs):
        result={}
        for key,value in pairs:
            if key in result:b.fail('duplicate FEM event JSON field')
            result[key]=value
        return result
    side=json.loads(raw[SIDECAR_FILE],object_pairs_hook=unique_object)
    if (not isinstance(side,dict) or set(side)!={'version','kind','scene_uuid','interpolation','extrapolation','bodies'}
            or type(side['version']) is not int or side['version']!=2 or side['kind']!='ppf.fem_body_controls'
            or side['scene_uuid']!=scene or side['interpolation']!=p.INTERPOLATION or side['extrapolation']!=p.EXTRAPOLATION):b.fail('invalid FEM event sidecar')
    vertex_map=b._envelope(raw['map.pickle'],'VertexMap',2)
    for path in ('map.pickle',SIDECAR_FILE):
        if inv.get(path)!={'relative_path':path,'byte_length':len(raw[path]),'sha256':hashlib.sha256(raw[path]).hexdigest()}:b.fail('FEM event input differs from consumed seal')
    if 'bin/fem_body_controls_v1.json' in inv:b.fail('FEM v1/v2 inventory mixing')
    _,wanted=partition(expected.controls)
    authored=receipt['controls']
    if not isinstance(authored,list) or len(authored)!=len(wanted):b.fail('missing FEM event author inventory')
    author={}
    for item in authored:
        if not isinstance(item,dict) or set(item)!={'body_uuid','source','motion'} or item['body_uuid'] in author:b.fail('invalid FEM event author record')
        author[item['body_uuid']]=item
    actual=[];members=set();last=''
    for row in side['bodies']:
        uid=b._uuid(row.get('body_uuid'));ids=row.get('vertex_indices')
        if (uid<=last or not isinstance(ids,list) or not ids or ids!=sorted(set(ids))
                or any(type(i) is not int or i<0 for i in ids) or members.intersection(ids)
                or uid not in vertex_map or sorted(vertex_map[uid])!=ids):b.fail('invalid FEM event full node ownership')
        if any(other!=uid and set(values).intersection(ids) for other,values in vertex_map.items()):b.fail('FEM event cross-body node alias')
        sig=p.wire_signature(row)
        a=author.get(uid)
        if a is None or a['source']!=row['source']:b.fail('FEM event receipt source differs')
        # The immutable uploaded projection (wanted) authenticates full keys, D,
        # source and normalized events. Receipt author vocabulary must also match.
        original=a['motion']
        if not isinstance(original,dict) or original.get('body_uuid')!=uid:b.fail('invalid FEM event receipt motion')
        base=b._control_signature(original,uid)
        program=tuple((k['time_s'],tuple(k['center_world_m']),tuple(k['rotation_world_wxyz'])) for k in original['keys'])
        normalized=[]
        for t,m,binding in base[3]:
            target=p.MODES[m]
            binding=(None if target=='FREE' else ('absolute' if binding is None else ('current_follow' if binding=='rigid_delta_v1' else binding)))
            normalized.append((t,target,binding))
        if (p.MODES[base[1]],program,tuple(normalized),a['source'])!=(sig[1],sig[2],sig[3],sig[4]):b.fail('FEM event original author differs from sidecar')
        actual.append(sig);last=uid;members.update(ids)
    if tuple(actual)!=wanted or set(author)!={r[0] for r in wanted}:b.fail('sealed FEM event controls differ from uploaded request')
    # Frozen program used by frame decoder; all constituent values are tuples.
    return (scene,hashes['context'],hashes['joint'],hashes['sidecar'],digest,hashlib.sha256(raw[RECEIPT_FILE]).hexdigest(),tuple(actual))

def verify_downloads(raw,expected,previous=None):
    rigid,_=partition(expected.controls)
    rigid_seal=b._verify_pdrd(raw[b.RECEIPT_FILE],raw[b.CONTEXT_FILE],raw[b.JOINT_FILE],raw[b.SIDECAR_FILE],replace(expected,controls=rigid)) if rigid else None
    seal=(rigid_seal,verify(raw,expected))
    if previous is not None and previous!=seal:b.fail('FEM/PDRD event seal changed after validation')
    return seal

def runtime_context(seal):
    if not isinstance(seal,tuple) or len(seal)!=2 or not isinstance(seal[1],tuple) or len(seal[1])!=7:b.fail('unverified FEM event runtime context')
    fem=seal[1]
    return {'program_sha256':fem[3],'controls':fem[6]}
