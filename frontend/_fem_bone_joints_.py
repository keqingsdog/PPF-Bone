from ._root_emitter_ import emit_roots
"""Typed FEM bone endpoints for real assembled topology; no solver emulation."""
from copy import deepcopy
import numpy as np
FEATURE='fem_material_joints_v1'
KIND='ppf.fem_bone_joints'

def _bary(X,p):
    beta=np.linalg.solve(np.stack([X[i]-X[0] for i in (1,2,3)],axis=1),p-X[0])
    return np.array([1-beta.sum(),*beta])

def install_mixed(fixed,document,param_data):
    if isinstance(document,dict) and 'bone_mass_intent_v1' in document:
        from ._bone_mass_authority_ import transaction
        return transaction(fixed,document,param_data,_install_mixed)
    return _install_mixed(fixed,document,param_data)

def _install_mixed(fixed,document,param_data):
    from . import _blender_bones_ as bb
    from ._pdrd_native_frames_ import native_body_frames,_matrix_quat,_rotation
    from ._scene_ import _validate_r1_payload
    bb.validate_environment(param_data)
    bb._verify_installed_collision_filters(fixed,document)
    if document.get('schema')!=bb.SCHEMA:bb.fail('schema')
    if set(document)-{'collision_filter_v1','body_controls_v1','body_controls_author_intent_v1','bone_mass_intent_v1'}!={'schema','armatures','bindings','start_frame','fps','world_scaling'}:bb.fail('document')
    if document['fps']!=param_data['scene']['fps'] or document['world_scaling']!=1:bb.fail('time/units')
    arms={a['armature_uuid']:a for a in document['armatures']}
    for a in arms.values():bb.positive_similarity_matrix(a['matrix_world_blender'],'armature.matrix')
    rows=document['bindings'];by_uuid={r['body_uuid']:r for r in rows};by_bone={r['bone_uuid']:r for r in rows}
    if not rows or len(by_uuid)!=len(rows) or len(by_bone)!=len(rows):bb.fail('bindings','duplicate/empty')
    allowed={'armature_uuid','bone_uuid','bone_name','body_uuid','proxy_object_uuid','parent_bone_uuid','parent_body_uuid','joint_uuid','enabled','physics','root_boundary','root_mode','proxy_object_name','group_uuid','shape','density','use_auto_frames','bone_world_matrix_blender','proxy_world_matrix_blender','joint_world_matrix_blender','body_to_bone_rest_blender','parent_local_frame','child_local_frame','root_motion','root_motion_source','fem_root_control_v1','fem_root_control_events_v1','author_uniform_scale'}
    for r in rows:
        if isinstance(r,dict) and 'automatic_parent_anchor_v1' in r:bb.fail('automatic_parent_anchor_v1','automatic anchors must be built as real PDRD endpoints before installation')
        if set(r)-allowed:bb.fail('binding','unknown requested fields')
        for k in ('body_uuid','bone_uuid','joint_uuid','armature_uuid'):bb.canonical_uuid(r[k],k)
        if r['armature_uuid'] not in arms or r['enabled'] is not True or r['root_mode'] not in ('FREE','KINEMATIC','STATIC'):bb.fail('binding')
        physics=r.get('physics')
        if not isinstance(physics,dict) or type(physics.get('collide_connected')) is not bool:
            bb.fail('physics.collide_connected','expected boolean')
        if r.get('proxy_object_uuid',r['body_uuid'])!=r['body_uuid']:bb.fail('proxy_object_uuid')
        boundary=r['root_boundary']
        if set(boundary)!={'is_root','nonphysical_parent_bone'} or boundary['is_root']!=(r['parent_body_uuid'] is None):bb.fail('root_boundary')
        if r['parent_body_uuid']:
            parent=by_uuid.get(r['parent_body_uuid'])
            if parent is None or r['parent_bone_uuid']!=parent['bone_uuid'] or parent['armature_uuid']!=r['armature_uuid']:bb.fail('parent_body_uuid')
        elif r['parent_bone_uuid']:bb.fail('parent_bone_uuid')
        visited={r['body_uuid']};p=r['parent_body_uuid']
        while p:
            if p in visited:bb.fail('parent_body_uuid','cycle')
            visited.add(p);p=by_uuid[p]['parent_body_uuid']
    # Installation and pose-return receipts require parents before children.
    # Sort transport only; retain the complete original author document.
    ordered=[];pending=list(rows);done=set()
    while pending:
        ready=[r for r in pending if r['parent_body_uuid'] is None or r['parent_body_uuid'] in done]
        if not ready:bb.fail('parent_body_uuid','cycle')
        for r in ready:ordered.append(r);done.add(r['body_uuid']);pending.remove(r)
    rows=ordered
    native=native_body_frames(fixed);stats={s['object_uuid']:s for s in fixed._statistics_objects}
    from ._fem_material_precision_ import native_rest_positions, material_anchor_bary
    world=fixed.vertex();native_world=native_rest_positions(fixed);tets=np.asarray(fixed._tet,dtype=np.int64)
    bodies=[];frames={};support={};receipts=[]
    for r in rows:
        uid=r['body_uuid'];source=[g for g in param_data['group'] if uid in g[2]]
        if len(source)!=1:bb.fail('body_uuid','missing material group')
        physical=r['physics'];rho=float(physical['density_kg_per_m3'])
        if float(source[0][0]['density'])!=float(np.float32(rho)):bb.fail('density_kg_per_m3')
        B=bb.S@bb.matrix(r['bone_world_matrix_blender'],'bone_world_matrix')
        if uid in native:
            if 'fem_root_control_v1' in r or 'fem_root_control_events_v1' in r:bb.fail('FEM.control','PDRD cannot declare FEM control')
            if source[0][0]['model']!='pdrd':bb.fail('body_uuid','PDRD material differs')
            T=bb.frame_matrix(native[uid]);body={'kind':'PDRD','body_uuid':uid,'runtime_body_index':native[uid]['runtime_body_index']}
        else:
            s=stats.get(uid)
            if s is None or s['dynamics_type']!='SOLID' or source[0][0]['model'] not in ('snhk','arap'):bb.fail('body_uuid','expected actual SOLID tet object')
            if r['root_mode']!='FREE' or r.get('root_motion'):
                if sum(r.get(k) is True for k in ('fem_root_control_v1','fem_root_control_events_v1'))!=1:bb.fail('root_mode','controlled FEM body requires explicit capability')
            elif 'fem_root_control_v1' in r or 'fem_root_control_events_v1' in r:bb.fail('FEM.control','dynamic FEM cannot declare uncontrolled root')
            members=set(s['vertex_indices'])
            if any(members.intersection(map(int,p.index)) for p in fixed._pin):bb.fail('FEM.pin','FEM endpoint pin overlap unsupported')
            for key in ('_translation_lock','_translation_lock_mode','_rotation_lock','_rotation_lock_mode'):
                value=getattr(fixed,key,None)
                if value is not None and np.any(value):bb.fail('FEM.aggregate_lock','FEM joint aggregate lock composition unsupported')
            T=bb.S@bb.matrix(r['proxy_world_matrix_blender'],'proxy_world_matrix')
            lo,count=s['tet_range'];choices=[]
            for ti in range(lo,lo+count):
                ids=tets[ti];X=native_world[ids];Dm=np.stack([X[i]-X[0] for i in (1,2,3)],axis=1)
                if not set(map(int,ids)).issubset(members):bb.fail('FEM.vertex_membership')
                cond=np.linalg.cond(Dm)
                detDm=float(np.linalg.det(Dm))
                # Reference winding is a convention, not a validity test: the default SOFT
                # export is consistently left-handed, and det(F)=det(Ds*Dm_inverse)<=0 is
                # guarded downstream where F is formed. Thresholds are unchanged; only the
                # sign requirement is dropped, so finite/cond/degenerate refusals remain.
                if not np.isfinite(cond) or cond>1e8 or not np.isfinite(detDm) or detDm==0.0:continue
                w=_bary(X,T[:3,3]);choices.append((float(np.linalg.norm(np.minimum(w,0))),cond,ti,ids,X,Dm,w))
            if not choices:bb.fail('FEM.rest_frame','singular/inverted or ill-conditioned support')
            _,_,ti,ids,X,Dm,w=min(choices,key=lambda q:q[:3])
            body={'kind':'FEM_TET_MATERIAL','body_uuid':uid,'tet_index':int(ti),'vertex_indices':ids.tolist(),
                'dm_inv_column_major':np.linalg.inv(Dm).T.reshape(-1).tolist(),'center_bary':w.tolist(),
                'frame_rest_column_major':T[:3,:3].T.reshape(-1).tolist()}
            support[uid]=X
        frames[uid]=T;bodies.append(body)
        author_scale=r.get('author_uniform_scale',1.)
        if type(author_scale) not in (int,float) or not np.isfinite(author_scale) or author_scale<=1.e-8:
            bb.fail('author_uniform_scale','expected finite positive static scale')
        receipt={'armature_uuid':r['armature_uuid'],'armature_name':arms[r['armature_uuid']]['object_name'],
            'bone_uuid':r['bone_uuid'],'bone_name':r['bone_name'],'body_uuid':uid,'proxy_object_uuid':uid,
            'endpoint_kind':body['kind'],'body_type':body['kind'],'parent_bone_uuid':r['parent_bone_uuid'],'bone_initial_solver':B.tolist(),
            'body_to_bone_initial_solver':(np.linalg.inv(T)@B).tolist()}
        if author_scale != 1.:receipt['author_uniform_scale']=author_scale
        if body['kind']=='PDRD':receipt['runtime_body_index']=body['runtime_body_index']
        else:receipt.update({key:deepcopy(body[key]) for key in ('tet_index','vertex_indices','dm_inv_column_major','center_bary','frame_rest_column_major')})
        if body['kind']=='FEM_TET_MATERIAL':
            from ._fem_scale_reference_ import build_reference
            lo,count=stats[uid]['tet_range']
            receipt['fem_scale_reference']=build_reference(native_world,tets[lo:lo+count],B[:3,:3].T.reshape(-1).tolist())
            # The addon owns the actual authored bone length.  Supplying the
            # material rest vertices lets it seal display head/tail supports
            # without changing the author request or native joint protocol.
            display_ids=np.unique(tets[lo:lo+count].reshape(-1))
            receipt['fem_display_rest_v1']={
                'version':1,
                'vertex_indices':display_ids.tolist(),
                'positions_m':native_world[display_ids].tolist(),
            }
        receipts.append(receipt)
    endpoint_kind={b['body_uuid']:b['kind'] for b in bodies}
    material_by_uuid={b['body_uuid']:b for b in bodies if b['kind']=='FEM_TET_MATERIAL'}
    joints=[];carrier=[];roots=[]
    for r in rows:
        uid=r['body_uuid'];parent=r['parent_body_uuid'];physics=deepcopy(r['physics'])
        allowed_phys={'density_kg_per_m3','anchor_stiffness_N_per_m','angular_control_v1','angular_soft_limits_v1','angular_drives_v1','angular_hard_hinge_v1','joint_d6_hard_v1','collide_connected'}
        if set(physics)-allowed_phys:bb.fail('physics','unsupported fields')
        physics.pop('density_kg_per_m3');physics['stiffness_N_per_m']=physics.pop('anchor_stiffness_N_per_m')
        if parent:
            if any(k in physics for k in ('angular_hard_hinge_v1','joint_d6_hard_v1')):
                bb.fail('FEM.hard_joint','hard joints are not supported by mixed installer dispatch')
            if r['use_auto_frames']:
                J=bb.S@bb.matrix(r['joint_world_matrix_blender'],'joint_world_matrix')
                pf=bb.local_frame(frames[parent],J);cf=bb.local_frame(frames[uid],J)
                # Automatic endpoint frames share one initial world frame J.
                rest_relative=[1.,0.,0.,0.]
            else:
                # Rebind manual endpoint frames through original proxy-world frames.
                parentP=bb.S@bb.matrix(by_uuid[parent]['proxy_world_matrix_blender'],'parent.proxy')
                childP=bb.S@bb.matrix(r['proxy_world_matrix_blender'],'child.proxy')
                def rebind(P,T,f):
                    J=np.eye(4);J[:3,:3]=P[:3,:3]@_rotation(f['quaternion_wxyz']);J[:3,3]=P[:3,3]+P[:3,:3]@np.array(f['translation_m'])
                    return bb.local_frame(T,J),J
                pf,PJ=rebind(parentP,frames[parent],r['parent_local_frame']);cf,CJ=rebind(childP,frames[uid],r['child_local_frame'])
                J=CJ
                rest_relative=_matrix_quat((frames[parent][:3,:3]@_rotation(pf['quaternion_wxyz'])).T@frames[uid][:3,:3]@_rotation(cf['quaternion_wxyz'])).tolist()
            wire={'joint_uuid':r['joint_uuid'],'parent_body_uuid':parent,'child_body_uuid':uid,
                'parent_local_frame':pf,'child_local_frame':cf,'enabled':True,
                'rest_relative_quaternion_wxyz':rest_relative,**physics}
            wire=_validate_r1_payload({'joints':[wire]})['joints'][0]
            item={'wire':wire}
            for side,id_ in (('parent',parent),('child',uid)):
                if id_ in support:
                    f=wire[side+'_local_frame'];material=material_by_uuid[id_]
                    item[side+'_anchor_bary']=material_anchor_bary(support[id_],material['center_bary'],material['frame_rest_column_major'],f['translation_m']).tolist()
            if len(item)>1:joints.append(item)
            else:carrier.append(wire)
        if r['root_mode']!='FREE' and uid in native:
            motion=deepcopy(r.get('root_motion'))
            if not isinstance(motion,dict) or motion.get('body_uuid')!=uid or motion.get('mode')!=r['root_mode'].lower():bb.fail('root_motion')
            roots.append(motion)
    # Material bodies also provide accepted-vertex pose authority for free
    # roots. A genuine FEM body need not have a physical parent constraint.
    # Keep the validated bodies/receipts and every requested edge intact.
    from ._fem_body_controls_ import install as install_controls
    install_controls(fixed,document,rows,stats)
    if not support:bb.fail('FEM.bodies','no actual FEM material body')
    fixed._fem_bone_joint_payload={'version':1,'kind':KIND,'scene_uuid':fixed._scene_uuid,'bodies':bodies,'joints':joints}
    fixed.set_pdrd_ball_anchor_payload({'joints':carrier})
    from ._bone_connected_collision_ import FEATURE as CONNECTED_FEATURE, required, verify_wire_pairs
    verify_wire_pairs(fixed, fixed._fem_bone_joint_payload)
    if roots:
        if document.get('body_controls_v1'):
            emit_roots(fixed, roots, True)
            from ._body_control_receipt_ import record_author
            record_author(fixed,document)
        else:emit_roots(fixed, roots)
    receipt={'schema':bb.SCHEMA,'collision_filter_v1':deepcopy(document.get('collision_filter_v1')),
        'start_frame':document['start_frame'],'fps':float(document['fps']),'world_scaling':1.,'bindings':receipts,
        'required_features':sorted([FEATURE] + ([CONNECTED_FEATURE] if required(fixed._fem_bone_joint_payload) else [])
            + (['pdrd_angular_swing_cone_v1'] if any('swing_cone' in item['wire'].get('angular_soft_limits_v1',{}) for item in joints) else []))}
    fixed._blender_bone_bindings=receipt
    return receipt
