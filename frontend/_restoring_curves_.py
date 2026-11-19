"""Restoring curves V1 pure Build helpers. Not registered with product yet.

Mass helpers consume actual solver nodal masses/Jacobians, never author boxes.
Caller must seal those inputs and native must independently recompute them.
"""
import hashlib
import json
import math
import re
import struct
import uuid

def require_command(path):
    """Add the sealed requirement without changing encoding or line endings."""
    with open(path, 'rb') as handle:
        original = handle.read()
    original.decode('utf-8')  # Retain the previous UTF-8 validation, including BOM.
    lines = original.splitlines(keepends=True)
    found = 0
    for i, line in enumerate(lines):
        if b'--path ' in line:
            found += 1
            if b'--require-restoring-curves-v1' not in line:
                ending = b'\r\n' if line.endswith(b'\r\n') else (
                    line[-1:] if line.endswith((b'\r', b'\n')) else b'')
                body = line[:-len(ending)] if ending else line
                lines[i] = body + b' --require-restoring-curves-v1' + ending
    if found != 1:
        fail('expected one solver command', 'RESTORING_CAPABILITY_UNAVAILABLE')
    updated = b''.join(lines)
    if updated != original:
        with open(path, 'wb') as handle:
            handle.write(updated)

def active(author):
    validate_author(author)
    return author is not None and any(b['enabled'] for b in author['bindings'])

MAX_TABLES, MAX_BINDINGS, MAX_NODES = 4096, 65536, 262144
MAX_JSON, MAX_PACKED = 32*1024*1024, 16*1024*1024
ROW_PREFIX = b"ppf.stitch.rows/v1\\n"

class CurveError(ValueError):
    def __init__(self, code, field):
        self.code, self.field = code, field
        super().__init__(code + ": " + field)

def fail(field, code="RESTORING_SCHEMA_INVALID"):
    raise CurveError(code, field)

def keys(value, expected, field):
    if not isinstance(value, dict) or set(value) != set(expected.split()):
        fail(field)

def number(value, field, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        fail(field, "RESTORING_CURVE_INVALID")
    if positive and value <= 0 or nonnegative and value < 0:
        fail(field, "RESTORING_CURVE_INVALID")
    try:
        out = struct.unpack('<f', struct.pack('<f', value))[0]
    except (OverflowError, struct.error):
        fail(field, "RESTORING_ARITHMETIC_FAULT")
    if not math.isfinite(out) or value != 0 and out == 0:
        fail(field, "RESTORING_ARITHMETIC_FAULT")
    return out

def ident(value, field):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', value):
        fail(field)
    return value

def canonical_uuid(value, field):
    try:
        if str(uuid.UUID(value)) != value:
            fail(field)
    except (ValueError, TypeError, AttributeError):
        fail(field)
    return value

def digest(value, field):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        fail(field)
    return value

def target(value, resolved=False):
    if not isinstance(value, dict): fail('target')
    kind = value.get('kind')
    if kind == 'JOINT':
        keys(value, 'kind joint_uuid channel', 'target')
        canonical_uuid(value['joint_uuid'], 'joint_uuid')
        if value['channel'] not in ('twist_x','swing_y','swing_z','anchor'): fail('channel')
    elif kind == ('STITCH_ROW' if resolved else 'STITCH_PAIR'):
        if resolved:
            keys(value, 'kind row_index rows_sha256 epsilon_m rest_length_m', 'target')
            uint(value['row_index'], 'row_index')
            digest(value['rows_sha256'], 'rows_sha256')
            number(value['epsilon_m'], 'epsilon_m', positive=True)
            number(value['rest_length_m'], 'rest_length_m', positive=True)
        else:
            keys(value, 'kind pair_id', 'target')
            canonical_uuid(value['pair_id'], 'pair_id')
    else: fail('target.kind')
    return json.dumps(value, sort_keys=True, separators=(',',':'))

def uint(value, field):
    if type(value) is not int or not 0 <= value <= 0xffffffff: fail(field)
    return value

def table(value):
    keys(value, 'table_id domain tail nodes', 'table')
    ident(value['table_id'], 'table_id')
    domain = value['domain']
    if domain not in ('SIGNED','ODD','RADIAL') or value['tail'] not in ('LINEAR','HOLD_EFFORT'): fail('table enum')
    raw = value['nodes']
    if not isinstance(raw, list) or not 2 <= len(raw) <= 64: fail('nodes', 'RESTORING_CURVE_INVALID')
    nodes = []
    for i, n in enumerate(raw):
        if not isinstance(n, list) or len(n) != 2: fail('nodes')
        nodes.append([number(n[0], f'nodes[{i}].x'), number(n[1], f'nodes[{i}].f')])
    if nodes.count([0,0]) != 1: fail('zero node', 'RESTORING_CURVE_INVALID')
    if domain == 'SIGNED':
        if not nodes[0][0] < 0 < nodes[-1][0]: fail('SIGNED range', 'RESTORING_CURVE_INVALID')
    elif nodes[0] != [0,0]: fail('nonnegative origin', 'RESTORING_CURVE_INVALID')
    slopes, segments = [], []
    for a,b in zip(nodes,nodes[1:]):
        dx = number(b[0]-a[0], 'node spacing', positive=True)
        df = number(b[1]-a[1], 'effort spacing', nonnegative=True)
        s = number(df/dx, 'slope', nonnegative=True)
        slopes.append(s)
        area = (a[1]+b[1])*dx/2
        number(area, 'segment integral')
        segments.append(area)
    zero = next(i for i,n in enumerate(nodes) if n == [0,0])
    # Integrate independently away from zero. Subtracting a large left-side
    # prefix at zero erases legitimate small energy on the positive side.
    areas = [0.0]*len(nodes)
    for i in range(zero, len(nodes)-1):
        areas[i+1] = number(areas[i]+segments[i], 'integral')
    for i in range(zero-1, -1, -1):
        areas[i] = number(areas[i+1]-segments[i], 'integral')
    return dict(value, nodes=nodes), slopes, areas

def validate_author(doc):
    if doc is None: return None
    keys(doc, 'version tables bindings stitch_pairs', 'author')
    if type(doc['version']) is not int or doc['version'] != 1: fail('version','RESTORING_VERSION_UNSUPPORTED')
    for field in ('tables','bindings','stitch_pairs'):
        if not isinstance(doc[field], list): fail(field)
    if len(doc['tables']) > MAX_TABLES or len(doc['bindings']) > MAX_BINDINGS: fail('counts','RESTORING_BUDGET_EXCEEDED')
    tables = {}
    count = 0
    for t in doc['tables']:
        checked,_,_ = table(t)
        if t['table_id'] in tables: fail('duplicate table')
        tables[t['table_id']] = checked
        count += len(checked['nodes'])
    if count > MAX_NODES: fail('nodes','RESTORING_BUDGET_EXCEEDED')
    pairs, indices = {}, set()
    for p in doc['stitch_pairs']:
        keys(p,'pair_id cross_stitch_index','stitch_pairs')
        pid=canonical_uuid(p['pair_id'],'pair_id'); index=uint(p['cross_stitch_index'],'cross_stitch_index')
        if pid in pairs or index in indices: fail('duplicate stitch mapping')
        pairs[pid]=index; indices.add(index)
    seen, referenced = set(), set()
    for b in doc['bindings']:
        mode=b.get('mode') if isinstance(b,dict) else None
        extra = 'damping' if mode == 'ABSOLUTE_SI' else 'frequency_hz damping_ratio coordinate_scale mass_policy' if mode == 'AUTO_FREQUENCY' else ''
        if not extra: fail('mode')
        keys(b,'target enabled table_id mode '+extra,'binding')
        tid=target(b['target'])
        if type(b['enabled']) is not bool or b['table_id'] not in tables: fail('binding reference')
        if b['enabled'] and tid in seen: fail(tid,'RESTORING_CONFLICT')
        if b['enabled']: seen.add(tid)
        t=tables[b['table_id']]; tar=b['target']
        anchor=tar['kind']=='JOINT' and tar['channel']=='anchor'
        if anchor != (t['domain']=='RADIAL'): fail(tid,'RESTORING_TARGET_INVALID')
        if tar['kind']=='STITCH_PAIR': referenced.add(tar['pair_id'])
        if mode=='ABSOLUTE_SI': number(b['damping'],'damping',nonnegative=True)
        else:
            number(b['frequency_hz'],'frequency_hz',positive=True)
            number(b['damping_ratio'],'damping_ratio',nonnegative=True)
            number(b['coordinate_scale'],'coordinate_scale',positive=True)
            if b['mass_policy']!='LOCAL_REST_V1': fail('mass_policy')
    if set(pairs)!=referenced: fail('stitch mapping completeness')
    return tables

def parse_author(data):
    if len(data)>MAX_JSON: fail('json bytes','RESTORING_BUDGET_EXCEEDED')
    def unique(items):
        out={}
        for k,v in items:
            if k in out: fail('duplicate JSON key')
            out[k]=v
        return out
    try: doc=json.loads(data,object_pairs_hook=unique,parse_constant=lambda _:fail('nonfinite'))
    except (json.JSONDecodeError,UnicodeDecodeError): fail('json')
    validate_author(doc)
    return doc

def stitch_rows_hash(ind_bytes, weight_bytes, expected_count):
    if len(ind_bytes)%48 or len(weight_bytes)%24 or len(ind_bytes)//48!=len(weight_bytes)//24: fail('stitch lengths','RESTORING_TARGET_INVALID')
    n=len(ind_bytes)//48
    if n!=expected_count: fail('stitch count','RESTORING_TARGET_INVALID')
    for row in struct.iter_unpack('<6f',weight_bytes):
        if any(not math.isfinite(v) or v<0 for v in row) or any(abs(sum(row[s:s+3])-1)>1e-5 for s in (0,3)): fail('stitch weights','RESTORING_TARGET_INVALID')
    return hashlib.sha256(ROW_PREFIX+struct.pack('<Q',n)+ind_bytes+weight_bytes).hexdigest()

def expand_pairs(doc, ranges, input_count, total_rows):
    """ranges: ordered (returned_input_ordinal,before,after) Build captures."""
    validate_author(doc)
    if len(ranges)!=input_count: fail('resolved input count','RESTORING_TARGET_INVALID')
    last=None
    for ordinal,(actual,start,end) in enumerate(ranges):
        if actual!=ordinal or not 0<=start<=end<=total_rows or last is not None and start!=last: fail('resolved order/range','RESTORING_TARGET_INVALID')
        last=end
    out={}
    active_pairs={b['target']['pair_id'] for b in doc['bindings'] if b['enabled'] and b['target']['kind']=='STITCH_PAIR'}
    expansion=0
    for p in doc['stitch_pairs']:
        if p['pair_id'] not in active_pairs: continue
        i=p['cross_stitch_index']
        if i>=input_count: fail('pair ordinal','RESTORING_TARGET_INVALID')
        _,start,end=ranges[i]
        if start==end: fail('empty pair','RESTORING_TARGET_INVALID')
        expansion+=end-start
        if expansion>MAX_BINDINGS: fail('stitch expansion','RESTORING_BUDGET_EXCEEDED')
        out[p['pair_id']]=list(range(start,end))
    return out

def stitch_metric(indices, weights, positions, contact_gaps):
    """Freeze from six solver REST vertices using the legacy gathered gap."""
    if len(indices)!=6 or len(weights)!=6: fail('stitch slots','RESTORING_TARGET_INVALID')
    pts=[positions[i] for i in indices]
    if any(not math.isfinite(w) or w<0 for w in weights) or any(abs(sum(weights[s:s+3])-1)>1e-5 for s in (0,3)): fail('stitch barycentric weights','RESTORING_TARGET_INVALID')
    l0=sum(weights[i]*contact_gaps[indices[i]] for i in range(6))/2
    if not math.isfinite(l0) or l0<0 or any(not math.isfinite(v) for p in pts for v in p): fail('stitch rest gaps/positions','RESTORING_MASS_INVALID')
    diagonal=math.sqrt(sum((max(p[a] for p in pts)-min(p[a] for p in pts))**2 for a in range(3)))
    eps=max(1e-9,1e-4*l0,32*2**-23*diagonal)
    if not math.isfinite(eps) or eps>max(1e-6,.01*max(l0,diagonal)): fail('epsilon','RESTORING_MASS_INVALID')
    return number(eps,'epsilon',positive=True),number(math.hypot(l0,eps),'rest_length',positive=True)

def nodal_mobility(coefficients, masses, fixed=()):
    """Merge signed global-vertex coefficients BEFORE squaring (stitch/J axis)."""
    merged={}; locked=set(fixed)
    for vertex,coefficient in coefficients:
        if not math.isfinite(coefficient): fail('coefficient','RESTORING_MASS_INVALID')
        merged[vertex]=merged.get(vertex,0)+coefficient
    mu=0.0
    for vertex,c in merged.items():
        if vertex in locked or c==0: continue
        m=masses[vertex]
        if not math.isfinite(m) or m<=0: fail('nodal mass','RESTORING_MASS_INVALID')
        mu+=c*c/m
    if not math.isfinite(mu) or mu<=0: fail('zero mobility','RESTORING_MASS_INVALID')
    return 1/mu

def rigid_inverse_inertia(positions, masses):
    """Full tensor from solver masses, not volume/AABB inertia; local estimate."""
    import numpy as np
    p=np.asarray(positions,dtype=float); m=np.asarray(masses,dtype=float)
    if p.shape!=(len(m),3) or not np.isfinite(p).all() or not np.isfinite(m).all() or (m<=0).any(): fail('rigid nodal masses','RESTORING_MASS_INVALID')
    total=float(m.sum()); com=(p*m[:,None]).sum(axis=0)/total
    r=p-com; tensor=sum((mass*(np.dot(v,v)*np.eye(3)-np.outer(v,v)) for mass,v in zip(m,r)),np.zeros((3,3)))
    if not np.isfinite(tensor).all() or np.linalg.eigvalsh(tensor).min()<=0: fail('singular inertia','RESTORING_MASS_INVALID')
    return total,com,np.linalg.inv(tensor)

def constrained_mobility(jacobian, mass, nullspace):
    """J N (N^T M N)^-1 N^T J^T from a supplied full mass and tangent basis.

    Pure-host oracle only. Actual Build uses native DataSet evidence. N contains
    independent allowed tangent columns; it is not a Euclidean projector.
    """
    import numpy as np
    j=np.asarray(jacobian,dtype=float); m=np.asarray(mass,dtype=float); n=np.asarray(nullspace,dtype=float)
    if j.ndim!=2 or j.shape[1]==0 or m.shape!=(j.shape[1],j.shape[1]) or n.ndim!=2 or n.shape[0]!=j.shape[1] or not all(np.isfinite(a).all() for a in (j,m,n)) or not np.allclose(m,m.T,rtol=1e-12,atol=0): fail('mass/tangent shape','RESTORING_MASS_INVALID')
    if np.linalg.eigvalsh(m).min()<=0: fail('nonpositive full mass','RESTORING_MASS_INVALID')
    if n.shape[1]==0:return np.zeros((j.shape[0],j.shape[0]))
    restricted=n.T@m@n
    if np.linalg.eigvalsh(restricted).min()<=0: fail('dependent tangent basis','RESTORING_MASS_INVALID')
    jn=j@n
    try:out=jn@np.linalg.solve(restricted,jn.T)
    except np.linalg.LinAlgError:fail('singular tangent mass','RESTORING_MASS_INVALID')
    if not np.isfinite(out).all(): fail('constrained mobility','RESTORING_MASS_INVALID')
    return out

def auto_constants(binding, metric_eff):
    number(metric_eff,'metric_eff',positive=True)
    m=metric_eff
    omega=2*math.pi*binding['frequency_hz']
    return number(m*omega*omega,'K',positive=True),number(2*binding['damping_ratio']*m*omega,'C',nonnegative=True)

def resolve(doc, scene_uuid, basis_sha256, target_resolver):
    """target_resolver(target) -> list[(resolved_target, mass_evidence)].

    Evidence keys: metric_eff/quality plus anchor eigenvalues. This callback
    must derive evidence from sealed solver inputs; not an addon authority.
    Native validation still requires independent recomputation.
    """
    tables=validate_author(doc)
    if doc is None or not any(b['enabled'] for b in doc['bindings']): return None
    canonical_uuid(scene_uuid,'scene_uuid'); digest(basis_sha256,'basis_sha256')
    out=dict(kind='ppf.restoring_curves',version=1,required_feature='restoring_curves_v1',scene_uuid=scene_uuid,basis_sha256=basis_sha256,tables=[],bindings=[],mass_receipts=[])
    stored_nodes=0
    for b in doc['bindings']:
        if not b['enabled']: continue
        resolved=target_resolver(b['target'])
        if not resolved: fail(str(b['target']),'RESTORING_TARGET_INVALID')
        for tar,evidence in resolved:
            if len(out['bindings'])>=MAX_BINDINGS: fail('expanded bindings','RESTORING_BUDGET_EXCEEDED')
            target(tar,True); t=tables[b['table_id']]; auto=b['mode']=='AUTO_FREQUENCY'
            scale=b['coordinate_scale'] if auto else 1
            k,c=auto_constants(b,evidence['metric_eff']) if auto else (1,number(b['damping'],'damping',nonnegative=True))
            si=dict(t,nodes=[[number(x*scale,'SI x'),number(f*k*scale,'SI effort')] for x,f in t['nodes']])
            fingerprint=hashlib.sha256(json.dumps(si,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            si['table_id']='si_'+fingerprint[:60]
            table(si)
            if tar['kind']=='JOINT' and tar['channel']!='anchor' and any(abs(x)>=number(math.pi-.001,'chart') for x,_ in si['nodes']): fail(str(tar),'RESTORING_CURVE_INVALID')
            if not any(v['table_id']==si['table_id'] for v in out['tables']):
                if len(out['tables'])>=MAX_TABLES or stored_nodes+len(si['nodes'])>MAX_NODES: fail('resolved tables/nodes','RESTORING_BUDGET_EXCEEDED')
                out['tables'].append(si)
                stored_nodes+=len(si['nodes'])
            rid=None
            if auto:
                rid='m_'+hashlib.sha256(target(tar,True).encode()).hexdigest()[:60]
                anchor=tar['kind']=='JOINT' and tar['channel']=='anchor'; stitch=tar['kind']=='STITCH_ROW'
                eig=evidence.get('mobility_eigenvalues') if anchor else None
                if anchor and (not isinstance(eig,(list,tuple)) or len(eig)!=3 or any(not math.isfinite(v) or v<0 for v in eig) or sum(eig)<=0): fail(str(tar),'RESTORING_MASS_INVALID')
                positive=[v for v in eig if v>0] if anchor else []
                if anchor and abs(3/sum(eig)-evidence['metric_eff'])>1e-6*evidence['metric_eff']: fail('mean anchor metric','RESTORING_MASS_INVALID')
                qualities=('ANCHOR_MEAN_COMPLIANCE_ESTIMATE','FEM_LOCAL_MASS_ESTIMATE') if anchor else ('STITCH_NODAL_LOCAL_ESTIMATE',) if stitch else ('LOCAL_RIGID_ESTIMATE','FEM_LOCAL_MASS_ESTIMATE')
                if evidence['quality'] not in qualities: fail('mass quality','RESTORING_MASS_INVALID')
                receipt=dict(receipt_id=rid,target=tar,policy='LOCAL_REST_V1',quality=evidence['quality'],basis_sha256=basis_sha256,metric_eff=evidence['metric_eff'],metric_unit='kg_m2' if tar['kind']=='JOINT' and not anchor else 'kg',mobility_eigenvalues=eig,zero_direction_count=sum(v==0 for v in eig) if anchor else None,anisotropy_ratio=max(positive)/min(positive) if anchor else None,frequency_hz=b['frequency_hz'],damping_ratio=b['damping_ratio'],coordinate_scale=scale,resolved_stiffness=k,resolved_damping=c,epsilon_m=tar['epsilon_m'] if stitch else None,rest_length_m=tar['rest_length_m'] if stitch else None)
                out['mass_receipts'].append(receipt)
            out['bindings'].append(dict(target=tar,table_id=si['table_id'],damping=c,mass_receipt_id=rid))
            if len(out['bindings'])>MAX_BINDINGS: fail('expanded bindings','RESTORING_BUDGET_EXCEEDED')
    seen=set()
    for b in out['bindings']:
        tar=b['target']; key=('STITCH_ROW',tar['row_index']) if tar['kind']=='STITCH_ROW' else ('JOINT',tar['joint_uuid'],tar['channel'])
        if key in seen: fail(str(key),'RESTORING_CONFLICT')
        seen.add(key)
    nodes=sum(len(t['nodes']) for t in out['tables'])
    receipt_bytes=len(json.dumps(out['mass_receipts'],sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8'))
    if len(out['tables'])>MAX_TABLES or nodes>MAX_NODES or 32*len(out['tables'])+16*nodes+48*len(out['bindings'])+receipt_bytes>MAX_PACKED: fail('packed budget','RESTORING_BUDGET_EXCEEDED')
    return out
