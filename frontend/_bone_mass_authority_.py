"""Explicit Bone MASS: uniform-point PDRD total mass; legacy absent is untouched."""
from copy import copy,deepcopy
from dataclasses import dataclass
import json,math,os,struct,uuid
import numpy as np

KEY='bone_mass_intent_v1'
KIND='ppf.bone_mass_intent'
POLICY='PDRD_UNIFORM_POINTS_V1'
RELATIVE='bin/bone_mass_intent_v1.json'

def fail(detail):raise ValueError('BONE_MASS_INTENT_INVALID: '+detail)
def canonical(value):
    try:checked=str(uuid.UUID(value))
    except (ValueError,TypeError,AttributeError):fail('UUID')
    if checked!=value or checked=='00000000-0000-0000-0000-000000000000':fail('canonical non-world UUID')
    return value
def positive_f32(value,label):
    if type(value) not in (float,int):fail(label)
    try:valid=math.isfinite(value) and value>0
    except (OverflowError,ValueError):valid=False
    if not valid:fail(label)
    try:result=struct.unpack('<f',struct.pack('<f',value))[0]
    except (OverflowError,struct.error):fail(label+' float32 overflow')
    if not math.isfinite(result) or result<=0:fail(label+' float32 underflow')
    return result

@dataclass(frozen=True)
class RowPlan:
    body_uuid:str
    runtime_body_index:int
    vertex_count:int
    target_mass_kg:float
    mass_per_vertex_kg:float

@dataclass(frozen=True)
class PreparedIntent:
    author_json:str
    rows:tuple[RowPlan,...]

def validate_intent(intent):
    if not isinstance(intent,dict) or set(intent)!={'kind','version','directives'} or intent['kind']!=KIND or type(intent['version']) is not int or intent['version']!=1:fail('kind/version/schema')
    if not isinstance(intent['directives'],list) or not intent['directives']:fail('empty directives')
    seen=set()
    for directive in intent['directives']:
        fields={'armature_uuid','bone_uuid','body_uuid','proxy_object_uuid','target_mass_kg','physical_model','source'}
        if not isinstance(directive,dict) or set(directive)!=fields:fail('directive fields')
        for key in ('armature_uuid','bone_uuid','body_uuid','proxy_object_uuid'):canonical(directive[key])
        if directive['body_uuid'] in seen:fail('duplicate body target')
        seen.add(directive['body_uuid'])
        if directive['physical_model']!=POLICY or directive['source']!='AUTHOR_TARGET_KG':fail('unsupported mass policy/source')
        if directive['proxy_object_uuid']!=directive['body_uuid']:fail('proxy identity')
        positive_f32(directive['target_mass_kg'],'target_mass_kg')
    return intent

def prepare(fixed,document,param_data):
    intent=validate_intent(document[KEY])
    bindings=document.get('bindings');arms=document.get('armatures')
    if not isinstance(bindings,list) or not bindings or not isinstance(arms,list):fail('author bindings/armatures')
    by_body={};bones=set();arm_ids=set()
    for arm in arms:
        aid=canonical(arm['armature_uuid'])
        if aid in arm_ids:fail('duplicate armature')
        arm_ids.add(aid)
    for binding in bindings:
        if not isinstance(binding,dict):fail('binding type')
        uid=canonical(binding['body_uuid']);bid=canonical(binding['bone_uuid']);aid=canonical(binding['armature_uuid'])
        if uid in by_body or bid in bones or aid not in arm_ids or binding['enabled'] is not True:fail('binding ownership')
        if binding.get('proxy_object_uuid',uid)!=uid:fail('binding proxy')
        by_body[uid]=binding;bones.add(bid)
    flat=np.asarray(fixed._pdrd_body_rows,dtype=np.float64)
    if flat.ndim!=1 or flat.size%23 or not np.isfinite(flat).all():fail('flat23 body rows')
    rows=flat.reshape(-1,23);runtime={};indices=set()
    for row in fixed._pdrd_body_bindings:
        uid=canonical(row['body_uuid']);index=row['runtime_body_index'];count=row['vertex_count']
        if type(index) is not int or type(count) is not int or index<0 or index>=len(rows) or count<4 or uid in runtime or index in indices:fail('runtime mapping')
        start,n=rows[index,:2]
        if not start.is_integer() or not n.is_integer() or start<0 or int(n)!=count or int(start)+count>len(fixed._pdrd_vert_list):fail('runtime start/count')
        if rows[index,15]<=0:fail('nonpositive existing body mass')
        runtime[uid]=(index,count);indices.add(index)
    if len(runtime)!=len(rows):fail('runtime row coverage')
    stats={}
    for row in fixed._statistics_objects:
        uid=row['object_uuid']
        if uid in stats:fail('duplicate statistics body')
        stats[uid]=row
    plan=[]
    for directive in intent['directives']:
        uid=directive['body_uuid'];binding=by_body.get(uid)
        if binding is None or any(binding[key]!=directive[key] for key in ('armature_uuid','bone_uuid','body_uuid')):fail('directive not owned by binding')
        materials=[group for group in param_data['group'] if uid in group[2]]
        if uid not in runtime or uid not in stats or stats[uid]['dynamics_type']!='PDRD' or stats[uid].get('closed_surface') is not True or len(materials)!=1 or materials[0][0].get('model')!='pdrd':fail('unsupported/ambiguous actual PDRD target')
        index,count=runtime[uid]
        members=fixed._pdrd_vert_list[int(rows[index,0]):int(rows[index,0])+count]
        if len(set(members))!=count or set(members)!=set(stats[uid]['vertex_indices']):fail('actual vertex ownership')
        target=float(directive['target_mass_kg'])
        per_vertex=positive_f32(target/count,'target/actualN')
        plan.append(RowPlan(uid,index,count,target,per_vertex))
    return PreparedIntent(json.dumps(intent,sort_keys=True,separators=(',',':'),allow_nan=False),tuple(plan))

def transaction(fixed,document,param_data,installer):
    if KEY not in document:return installer(fixed,document,param_data)
    if getattr(fixed,'_bone_mass_transaction_active',False):return installer(fixed,document,param_data)
    prepared=prepare(fixed,document,param_data)
    candidate=copy(fixed)
    # These are precisely the mutable installation/root payload families. Geometry
    # and material/point/FEM arrays are read-only to both bone installers.
    for name,value in fixed.__dict__.items():
        if name.startswith(('_pdrd_','_blender_bone_','_fem_bone_','_body_control_')):
            setattr(candidate,name,deepcopy(value))
    candidate._bone_mass_transaction_active=True
    candidate._bone_mass_authority_v1=prepared
    candidate._pdrd_body_rows=list(candidate._pdrd_body_rows)
    for row in prepared.rows:candidate._pdrd_body_rows[row.runtime_body_index*23+15]=row.mass_per_vertex_kg
    # Any later strict binding/frame/root/joint failure discards this private state.
    result=installer(candidate,document,param_data)
    del candidate._bone_mass_transaction_active
    fixed.__dict__.update(candidate.__dict__)
    return result

def export_intent(fixed,path):
    target=os.path.join(path,RELATIVE)
    prepared=getattr(fixed,'_bone_mass_authority_v1',None)
    if prepared is None:
        if os.path.exists(target):
            with open(target,encoding='utf-8') as stream:old=json.load(stream)
            if not isinstance(old,dict) or old.get('kind')!=KIND or type(old.get('version')) is not int or old.get('version')!=1:fail('unexpected stale owned intent file')
            os.unlink(target)
        return
    if type(prepared) is not PreparedIntent:fail('unprepared export state')
    intent=validate_intent(json.loads(prepared.author_json));resolved=[]
    for row in prepared.rows:
        bindings=[binding for binding in fixed._pdrd_body_bindings if binding['body_uuid']==row.body_uuid]
        if len(bindings)!=1 or bindings[0]['runtime_body_index']!=row.runtime_body_index or bindings[0]['vertex_count']!=row.vertex_count or fixed._pdrd_body_rows[row.runtime_body_index*23+1]!=row.vertex_count:fail('export body mapping changed')
        current=fixed._pdrd_body_rows[row.runtime_body_index*23+15]
        if positive_f32(current,'exported mass')!=row.mass_per_vertex_kg:fail('mass row changed after preparation')
        resolved.append({'body_uuid':row.body_uuid,'runtime_body_index':row.runtime_body_index,'actual_vertex_count':row.vertex_count,'target_mass_kg':row.target_mass_kg,'mass_per_vertex_kg_f32':row.mass_per_vertex_kg,'actual_total_mass_kg_f32':row.mass_per_vertex_kg*row.vertex_count})
    if len(resolved)!=len(intent['directives']):fail('export plan inventory')
    intent['resolved']=resolved
    temporary=target+'.pending'
    with open(temporary,'w',encoding='utf-8') as stream:json.dump(intent,stream,sort_keys=True,separators=(',',':'),allow_nan=False)
    os.replace(temporary,target)
