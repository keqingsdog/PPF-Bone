"""Author-only AUTO limits; native receives the validated SI limit wire.

Two different bases deliberately require two different authority observations:
provisional inertia resolution, then the existing final-basis restoring query.
"""
from copy import deepcopy
import json,math
from pathlib import Path
from . import _restoring_curves_ as curve

KEY='soft_limit_auto_v1'
AXES=('twist_x','swing_y','swing_z')

def fail(detail):raise ValueError('SOFT_LIMIT_AUTO_INVALID: '+detail)

def prepare_author(document):
    if not isinstance(document,dict) or KEY not in document:return document,[]
    document=deepcopy(document);intent=document.pop(KEY)
    if not isinstance(intent,dict) or set(intent)!={'version','joints'} or type(intent['version']) is not int or intent['version']!=1:fail('author version')
    if not isinstance(intent['joints'],list) or not intent['joints']:fail('empty joints')
    rows={row['joint_uuid']:row for row in document.get('bindings',[])}
    checked=[];seen=set()
    for spec in intent['joints']:
        if not isinstance(spec,dict) or set(spec) not in ({'joint_uuid','frequency_hz','strength','enabled_axes'}, {'joint_uuid','frequency_hz','strength','enabled_axes','swing_cone'}):fail('author fields')
        uid=curve.canonical_uuid(spec['joint_uuid'],'joint_uuid')
        if uid in seen or uid not in rows:fail('duplicate or unknown joint')
        seen.add(uid);row=rows[uid]
        if not row.get('parent_body_uuid'):fail('real parent joint required after carrier preparation')
        limits=row.get('physics',{}).get('angular_soft_limits_v1')
        if not limits:fail('compliant limits required')
        cone=spec.get('swing_cone',False)
        if type(cone) is not bool:fail('cone flag')
        if cone and (not isinstance(limits.get('swing_cone'),dict) or limits['swing_cone'].get('enabled') is not True):fail('cone differs')
        flags=spec['enabled_axes']
        if not isinstance(flags,list) or len(flags)!=3 or any(type(v) is not bool for v in flags) or not (any(flags) or cone) or (cone and any(flags[1:])):fail('enabled axes')
        if any(limits[axis]['enabled']!=flag for axis,flag in zip(AXES,flags)):fail('author limit flags differ')
        f=curve.number(spec['frequency_hz'],'limit_frequency',positive=True)
        strength=curve.number(spec['strength'],'limit_strength',nonnegative=True)
        entry=dict(joint_uuid=uid,frequency_hz=f,strength=strength,enabled_axes=list(flags))
        if 'swing_cone' in spec:entry['swing_cone']=cone
        checked.append(entry)
    return document,checked

def targets(intent):
    # Unique joint/axis targets; no duplicate request can mask anisotropy.
    result=[];seen=set()
    for spec in intent:
        for axis,enabled in zip(AXES,spec['enabled_axes']):
            enabled=enabled or (spec.get('swing_cone',False) and axis in AXES[1:])
            if not enabled:continue
            tar=dict(kind='JOINT',joint_uuid=spec['joint_uuid'],channel=axis)
            key=curve.target(tar)
            if key in seen:fail('duplicate authority target')
            seen.add(key);result.append(dict(target=tar,needs_mass=True))
    return result

def plan(intent,authority):
    rows={curve.target(row['target'],True):row for row in authority['targets']}
    if len(rows)!=len(authority['targets']):fail('authority duplicates')
    out=[]
    for spec in intent:
        omega=2.*math.pi*spec['frequency_hz']
        for axis,enabled in zip(AXES,spec['enabled_axes']):
            enabled=enabled or (spec.get('swing_cone',False) and axis in AXES[1:])
            if not enabled:continue
            tar=dict(kind='JOINT',joint_uuid=spec['joint_uuid'],channel=axis)
            row=rows.get(curve.target(tar))
            if row is None or row.get('needs_mass') is not True or row.get('metric_unit')!='kg_m2':fail('positive actual angular inertia required')
            if row.get('native_kind') not in (0,1) or row.get('quality')!=('LOCAL_RIGID_ESTIMATE' if row.get('native_kind')==0 else 'FEM_LOCAL_MASS_ESTIMATE'):fail('actual angular authority required')
            curve.number(row['metric_eff'],'limit_effective_inertia',positive=True)
            inertia=float(row['metric_eff'])
            stiffness=curve.number(inertia*omega*omega*spec['strength'],'limit_stiffness',nonnegative=True)
            out.append(dict(target=tar,metric_eff=inertia,metric_unit='kg_m2',quality=row['quality'],frequency_hz=spec['frequency_hz'],strength=spec['strength'],stiffness_Nm_per_rad=stiffness))
    for spec in intent:
        if not spec.get('swing_cone',False):continue
        uid=spec['joint_uuid']
        swing=[row for row in out if row['target']['joint_uuid']==uid and row['target']['channel'] in AXES[1:]]
        if len(swing)!=2:fail('cone requires both swing authorities')
        inertia=.5*sum(row['metric_eff'] for row in swing)
        k=curve.number(inertia*(2.*math.pi*spec['frequency_hz'])**2*spec['strength'],'cone_stiffness',nonnegative=True)
        out=[row for row in out if row not in swing]
        out.append(dict(target=dict(kind='JOINT',joint_uuid=uid,channel='swing_cone'),metric_eff=inertia,metric_unit='kg_m2',
            inertia_policy='mean_of_two_axis_effective_inertias',axis_authority=swing,
            quality=swing[0]['quality'],frequency_hz=spec['frequency_hz'],strength=spec['strength'],stiffness_Nm_per_rad=k))
    return out


def apply_plan(fixed,rows):
    joints={}
    for wire in (getattr(fixed,'_pdrd_ball_anchor_payload',None) or {}).get('joints',[]):
        uid=wire['joint_uuid']
        if uid in joints:fail('duplicate installed joint')
        joints[uid]=wire
    for item in (getattr(fixed,'_fem_bone_joint_payload',None) or {}).get('joints',[]):
        wire=item['wire'];uid=wire['joint_uuid']
        if uid in joints:fail('duplicate FEM/PDRD joint')
        joints[uid]=wire
    pending=[]
    for row in rows:
        tar=row['target'];wire=joints.get(tar['joint_uuid'])
        if wire is None:fail('installed joint missing')
        limits=wire.get('angular_soft_limits_v1',{});axis=tar['channel']
        if axis not in limits or limits[axis].get('enabled') is not True:fail('installed limits differ')
        pending.append((limits[axis],row['stiffness_Nm_per_rad']))
    for value,k in pending:value['stiffness_Nm_per_rad']=k

def resolve_and_finalize(fixed,path,finalizer):
    """Only called inside the existing complete Build/receipt transaction."""
    import cbor2
    from . import _run_directory
    from ._restoring_curves_build_ import query,encode
    path=Path(path);intent=fixed._soft_limit_auto_intent
    original_pdrd=deepcopy(fixed._pdrd_ball_anchor_payload)
    original_fem=deepcopy(getattr(fixed,'_fem_bone_joint_payload',None))
    try:
        fixed._soft_limit_provisional=True
        finalizer(str(path),True)
        context=cbor2.loads((path/'bin/pdrd_scene_context.cbor').read_bytes())['payload']
        if context.get('complete') is not True:fail('complete provisional Build required')
        request=dict(kind='ppf.restoring_curves.mass_request',version=1,scene_uuid=fixed._scene_uuid,basis_sha256=context['scene_inputs_sha256'],targets=targets(intent))
        exe=Path(_run_directory(contract=getattr(fixed,'_build_contract',None)))/('ppf-contact-solver.exe' if __import__('os').name=='nt' else 'ppf-contact-solver')
        authority,staging=query(exe,path,request,
                                statistics_objects=getattr(fixed,'_statistics_objects',()))
        rows=plan(intent,authority);apply_plan(fixed,rows)
        receipt=dict(kind='ppf.soft_limit_auto',version=1,policy='AUTO_INERTIA',scene_uuid=fixed._scene_uuid,provisional_basis_sha256=context['scene_inputs_sha256'],authority_staging=staging,resolved=rows)
        (path/'bin/soft_limit_auto_v1.json').write_bytes(encode(receipt))
        fixed._soft_limit_provisional=False
        # Rewrites FEM wire and context, then resolves restoring against FINAL basis.
        finalizer(str(path),True)
    except BaseException:
        fixed._pdrd_ball_anchor_payload=original_pdrd
        fixed._fem_bone_joint_payload=original_fem
        raise
    finally:
        fixed._soft_limit_provisional=False
