"""Pure FEM event author projection. No Blender, native index or pin surrogate."""
import math

FEATURE = 'fem_root_control_events_v1'
FILE = 'bin/fem_body_controls_v2.json'
RECEIPT = 'fem_body_control_receipt_v2.cbor'
INTERPOLATION = 'linear_center_shortest_slerp_v1'
EXTRAPOLATION = 'hold'
MODES = {'dynamic': 'FREE', 'static': 'STATIC', 'kinematic': 'KINEMATIC'}


def fail(detail):
    raise ValueError('FEM_ROOT_EVENTS_INVALID: ' + detail)


def real(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        fail('finite real required')
    return float(value)


def vector(value, n, unit=False):
    if not isinstance(value, (list, tuple)) or len(value) != n:
        fail('invalid vector shape')
    result = tuple(real(v) for v in value)
    if unit and abs(sum(v*v for v in result)-1.) > 1e-6:
        fail('unit quaternion required')
    return result


def rotation(q):
    values=vector(q,4,True)
    length=math.sqrt(sum(v*v for v in values))
    w,x,y,z=(v/length for v in values)
    return ((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
            (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
            (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))


def quaternion(m):
    # Stable proper rotation conversion; canonical sign makes both projections exact.
    trace = sum(m[i][i] for i in range(3))
    if trace > 0:
        s = math.sqrt(trace+1.)*2
        q = (s/4, (m[2][1]-m[1][2])/s, (m[0][2]-m[2][0])/s, (m[1][0]-m[0][1])/s)
    else:
        i = max(range(3), key=lambda i:m[i][i]); j=(i+1)%3; k=(i+2)%3
        s = math.sqrt(1.+m[i][i]-m[j][j]-m[k][k])*2
        xyz = [0.,0.,0.]; xyz[i]=s/4; xyz[j]=(m[j][i]+m[i][j])/s; xyz[k]=(m[k][i]+m[i][k])/s
        q=((m[k][j]-m[j][k])/s, *xyz)
    norm=math.sqrt(sum(x*x for x in q)); q=tuple(x/norm for x in q)
    return tuple(-x for x in q) if q[0]<0 else q


def reference(row):
    motion=row['root_motion']; initial=motion.get('initial_frame')
    if not isinstance(initial,dict) or set(initial)!={'translation_m','quaternion_wxyz'}:
        fail('missing initial author frame')
    a=rotation(initial['quaternion_wxyz']); t=vector(initial['translation_m'],3)
    p=row.get('proxy_world_matrix_blender')
    if not isinstance(p,list) or len(p)!=4 or any(not isinstance(r,list) or len(r)!=4 for r in p):
        fail('invalid proxy matrix')
    p=[vector(r,4) for r in p]
    if any(abs(x-y)>1e-8 for x,y in zip(p[3],(0.,0.,0.,1.))):fail('proxy affine row')
    # Released Blender -> solver coordinate transform S: (x,z,-y).
    p=(p[0],p[2],tuple(-x for x in p[1]),p[3])
    r=tuple(tuple(sum(a[k][i]*p[k][j] for k in range(3)) for j in range(3)) for i in range(3))
    for i in range(3):
        for j in range(3):
            if abs(sum(r[k][i]*r[k][j] for k in range(3))-float(i==j))>1e-6:fail('proxy scale/shear unsupported')
    det=r[0][0]*(r[1][1]*r[2][2]-r[1][2]*r[2][1])-r[0][1]*(r[1][0]*r[2][2]-r[1][2]*r[2][0])+r[0][2]*(r[1][0]*r[2][1]-r[1][1]*r[2][0])
    if abs(det-1)>1e-6:fail('proxy reflection unsupported')
    d=tuple(sum(a[k][i]*(p[k][3]-t[k]) for k in range(3)) for i in range(3))
    return {'translation_local_m':list(d),'rotation_local_wxyz':list(quaternion(r))}


def reference_signature(ref):
    if not isinstance(ref,dict) or set(ref)!={'translation_local_m','rotation_local_wxyz'}:fail('invalid reference transform')
    return (vector(ref['translation_local_m'],3),vector(ref['rotation_local_wxyz'],4,True))


def project(row, vertex_indices=None):
    root=row.get('root_motion'); source=row.get('root_motion_source'); uid=row.get('body_uuid')
    if row.get(FEATURE) is not True or 'fem_root_control_v1' in row or not isinstance(root,dict):fail('strict event family marker required')
    if root.get('body_uuid')!=uid or source not in ('POSE','OBJECT') or root.get('mode') not in MODES:fail('root identity/mode/source')
    if row.get('root_mode')!=MODES[root['mode']]:fail('author mode differs')
    keys=root.get('keys')
    if not isinstance(keys,list) or not 1<=len(keys)<=4096:fail('invalid key count')
    out=[]; last=-1.
    for k in keys:
        if not isinstance(k,dict) or set(k)!={'time_s','center_world_m','rotation_world_wxyz'}:fail('invalid key schema')
        time=real(k['time_s'])
        if time<=last or (not out and time!=0):fail('invalid key times')
        out.append({'time_s':time,'center_world_m':list(vector(k['center_world_m'],3)),'rotation_world_wxyz':list(vector(k['rotation_world_wxyz'],4,True))});last=time
    initial=root.get('initial_frame')
    if not isinstance(initial,dict) or set(initial)!={'translation_m','quaternion_wxyz'}:fail('missing immutable initial frame')
    initial_t=vector(initial['translation_m'],3)
    initial_r=rotation(initial['quaternion_wxyz']);first_r=rotation(out[0]['rotation_world_wxyz'])
    if (any(abs(x-y)>1e-8 for x,y in zip(out[0]['center_world_m'],initial_t))
            or any(abs(first_r[i][j]-initial_r[i][j])>1e-6 for i in range(3) for j in range(3))):
        fail('first key differs from immutable initial author frame')
    events=root.get('control_events',[])
    if not isinstance(events,list) or len(events)>4096:fail('invalid event count')
    selected=MODES[root['mode']]; normalized=[];last=0.
    for event in events:
        if not isinstance(event,dict) or set(event) not in ({'time_s','mode'},{'time_s','mode','pose_binding'}):fail('invalid author event schema')
        time=real(event['time_s']);mode=MODES.get(event['mode'])
        if time<=last or mode is None or mode==selected:fail('invalid event order/mode')
        binding=event.get('pose_binding')
        if mode=='FREE':
            if binding is not None:fail('FREE cannot carry binding')
        elif mode=='STATIC':
            binding='absolute' if binding is None else binding
            if binding not in ('current_hold','absolute'):fail('invalid STATIC policy')
        else:
            binding={'rigid_delta_v1':'current_follow',None:'absolute'}.get(binding,binding)
            if binding not in ('current_follow','absolute'):fail('invalid KIN policy')
        normalized.append({'time_s':time,'mode':mode,'pose_binding':binding});last=time;selected=mode
    if (root['mode']=='kinematic' or any(e['mode']=='KINEMATIC' for e in normalized)) and len(out)<2:fail('KIN requires sampled trajectory')
    if root['mode']=='dynamic' and not normalized:fail('FREE without events is not controlled')
    result={'body_uuid':uid,'source':source,'mode':MODES[root['mode']],'keys':out,'reference_transform':reference(row),'control_events':normalized}
    if vertex_indices is not None:result['vertex_indices']=list(vertex_indices)
    return result


def wire_signature(row):
    required={'body_uuid','source','mode','keys','reference_transform','control_events'}
    if not isinstance(row,dict) or set(row) not in (required,required|{'vertex_indices'}):fail('invalid projected body schema')
    if row['mode'] not in ('FREE','STATIC','KINEMATIC') or row['source'] not in ('POSE','OBJECT'):fail('invalid projected mode/source')
    program=[];last=-1.
    for key in row['keys']:
        if not isinstance(key,dict) or set(key)!={'time_s','center_world_m','rotation_world_wxyz'}:fail('invalid projected key')
        t=real(key['time_s'])
        if t<=last or (not program and t!=0):fail('invalid projected key order')
        program.append((t,vector(key['center_world_m'],3),vector(key['rotation_world_wxyz'],4,True)));last=t
    if not 1<=len(program)<=4096:fail('invalid projected key count')
    events=[];last=0.;mode=row['mode']
    for e in row['control_events']:
        if not isinstance(e,dict) or set(e)!={'time_s','mode','pose_binding'}:fail('invalid projected event schema')
        t=real(e['time_s']);target=e['mode'];binding=e['pose_binding']
        if t<=last or target not in ('FREE','STATIC','KINEMATIC') or target==mode:fail('invalid projected event order')
        if not ((target=='FREE' and binding is None) or (target=='STATIC' and binding in ('current_hold','absolute')) or (target=='KINEMATIC' and binding in ('current_follow','absolute'))):fail('invalid projected policy')
        events.append((t,target,binding));last=t;mode=target
    if len(events)>4096:fail('too many projected events')
    if (row['mode']=='KINEMATIC' or any(e[1]=='KINEMATIC' for e in events)) and len(program)<2:fail('KIN projected keys')
    return (row['body_uuid'],row['mode'],tuple(program),tuple(events),row['source'],reference_signature(row['reference_transform']),'FEM_EVENTS',2)


def signature(row):
    return wire_signature(project(row))
