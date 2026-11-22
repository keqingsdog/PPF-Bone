"""Transactional bone Action baking behind the original Bake All controls."""
from pathlib import Path
import math
import bpy
from mathutils import Matrix
from . import rigid_pose as rp
from .utils import get_id_fcurves
from ..i18n import pgettext_iface as iface_

class BoneBakeError(ValueError):
    code='BONE_BAKE_UNREPRESENTABLE'

def _slot(ad):
    s=getattr(ad,'action_slot',None)
    return (getattr(s,'handle',None),getattr(s,'identifier',None)) if s else None

def _assign(arm,action,slot):
    ad=arm.animation_data_create();ad.action=action
    if action is not None and slot is not None and hasattr(ad,'action_slot'):
        matches=[s for s in getattr(action,'slots',()) if s.handle==slot[0]]
        if not matches:matches=[s for s in getattr(action,'slots',()) if s.identifier==slot[1]]
        if len(matches)!=1:raise BoneBakeError('Action slot cannot be preserved')
        ad.action_slot=matches[0]

def make_entry(scene,single=None):
    key=scene.get(rp.CACHE_KEY)
    if not key:return None
    index=rp._read_index_file(key)
    if index is None:raise BoneBakeError('Bone cache index is missing or malformed; author state was not detached')
    targets=rp._bound_targets(scene,index)
    if not targets:raise BoneBakeError('Bone cache targets are unavailable')
    frames=sorted(map(int,index['frames']))
    if not frames or frames!=list(range(frames[-1]+1)):raise BoneBakeError('Bone bake requires complete contiguous cache')
    start=int(index['receipt']['start_frame']);sample_frames=[single] if single is not None else [start+n for n in frames]
    if single is not None and single-start not in frames:raise BoneBakeError('Current frame is not cached')
    arms={arm for row,arm,pb in targets}
    snapshots=[]
    for arm in arms:
        ad=arm.animation_data
        if ad and (any(not t.mute for t in ad.nla_tracks) or ad.drivers):raise BoneBakeError('Active NLA/drivers require explicit author bake before solver bone bake')
        names=[pb.name for row,a,pb in targets if a==arm]
        if any(c for name in names for c in arm.pose.bones[name].constraints if not c.mute):raise BoneBakeError('Active bone constraint cannot be baked by TRS')
        snap=dict(arm=arm,action=ad.action if ad else None,slot=_slot(ad) if ad else None,names=names,modes={n:arm.pose.bones[n].rotation_mode for n in names},basis={n:arm.pose.bones[n].matrix_basis.copy() for n in names})
        # Capture source Action and its animation graph before sampling or
        # replacing either; an Action-less arm can still have drivers or NLA.
        from .single_bake_restore import (_source_action_snapshot,
                                          _source_action_proofs,
                                          _recovery_owners)
        from . import recovery_anim_graph
        if snap['action'] is not None:
            snap['source_action_proof'], snap['source_action_refs'] = _source_action_snapshot(snap['action'])
        owners = _recovery_owners([arm])
        snap['source_data_proofs'] = _source_action_proofs(owners[1:])
        snap['source_graph_proof'] = recovery_anim_graph.capture(owners, _source_action_snapshot)
        snap['source_snapshot_schema']=2
        snapshots.append(snap)
    return dict(obj_type='BONE',obj_uuid='',obj_name='Bone Actions',n_frames=len(sample_frames),frames_done=0,start_frame=start,sample_frames=sample_frames,samples=[],scene=scene,key=key,index=index,snapshots=snapshots,entry_time=(scene.frame_current,scene.frame_subframe),created=[],committed=False,single=single is not None)

def sample(entry):
    scene=entry['scene'];frame=entry['sample_frames'][entry['frames_done']]
    n=frame-entry['start_frame']
    with rp.capture_verified_playback_for_bake(scene,entry['key'],n,entry['index']['frames'][str(n)]):
        scene.frame_set(frame)
    # Settle changes made by later frame handlers before proving reuse.
    bpy.context.view_layer.update()
    dg=bpy.context.evaluated_depsgraph_get()
    if not rp.verified_playback_matches(scene,entry['key'],n,entry['index']['frames'][str(n)],dg,entry['index']['receipt']):
        raw=rp._io_path(Path(entry['key']).parent/f'pose_{n}.cbor').read_bytes()
        if rp.hashlib.sha256(raw).hexdigest()!=entry['index']['frames'][str(n)]:raise BoneBakeError('Bone cache frame integrity mismatch')
        rp._apply(scene,entry['index']['receipt'],rp._load(raw));bpy.context.view_layer.update()
        dg=bpy.context.evaluated_depsgraph_get()
    if scene.get(rp.CACHE_KEY)!=entry['key']:raise BoneBakeError('Bone cache detached while sampling')
    row={}
    for snap in entry['snapshots']:
        arm=snap['arm'];ev=arm.evaluated_get(dg)
        row[arm.name]={n:ev.pose.bones[n].matrix.copy() for n in arm.pose.bones.keys()}
    entry['samples'].append((frame,row));entry['frames_done']+=1

def _remove_target_channels(arm,names):
    ad=arm.animation_data;action=ad.action
    layers=getattr(action,'layers',())
    bags=[]
    if layers:
        slot=getattr(ad,'action_slot',None)
        for layer in layers:
            for strip in layer.strips:
                bag=strip.channelbag(slot) if slot else None
                if bag:bags.append(bag.fcurves)
    elif hasattr(action,'fcurves'):bags=[action.fcurves]
    paths={arm.pose.bones[n].path_from_id(p) for n in names for p in ('location','rotation_quaternion','rotation_euler','rotation_axis_angle','scale')}
    for curves in bags:
        for fc in list(curves):
            if fc.data_path in paths:curves.remove(fc)

def rollback(entry):
    scene=entry['scene']
    for snap in entry['snapshots']:
        _assign(snap['arm'],snap['action'],snap['slot'])
        for n,basis in snap['basis'].items():snap['arm'].pose.bones[n].matrix_basis=basis
    # Restore real users before withdrawing this transaction's keep-alive.
    for action in entry.get('retained_source_actions',[]):
        action.use_fake_user=False
    entry['retained_source_actions']=[]
    for action in entry['created']:
        if action.users==0:bpy.data.actions.remove(action)
    entry['created'].clear()
    scene[rp.CACHE_KEY]=entry['key']
    scene.frame_set(entry['entry_time'][0],subframe=entry['entry_time'][1]);rp.bone_pose_frame_change(scene);bpy.context.view_layer.update()

def _static_similarity_parent_chain(arm, names):
    """Conservatively admit ordinary static similarity object parents."""
    from ..models.bone_bindings import similarity_frame_scale
    try:
        dg = bpy.context.evaluated_depsgraph_get()
        obj = arm
        transform_paths = {'location','rotation_euler','rotation_quaternion','rotation_axis_angle','scale',
                           'delta_location','delta_rotation_euler','delta_rotation_quaternion','delta_scale'}
        while obj.parent is not None:
            if obj.parent_type != 'OBJECT':
                return False
            obj = obj.parent
            ad = obj.animation_data
            if (any(not c.mute for c in obj.constraints) or
                    (ad and (ad.drivers or ad.nla_tracks or
                             any(not fc.mute and fc.data_path in transform_paths for fc in get_id_fcurves(obj))))):
                return False
            similarity_frame_scale(obj.evaluated_get(dg).matrix_world, obj.name)
        ev = arm.evaluated_get(dg)
        similarity_frame_scale(ev.matrix_world, arm.name)
        for name in names:
            similarity_frame_scale(ev.pose.bones[name].matrix, name)
        return True
    except (ValueError, TypeError, AttributeError, KeyError, ReferenceError):
        return False


def _static_pdrd_author_scales(entry, snap):
    """Prove a static authored basis before creating solver Action channels."""
    from .fem_pose import binding_kind
    arm = snap['arm']; ad = arm.animation_data
    if any(not c.mute for c in arm.constraints):
        return {}
    if ad and (ad.drivers or ad.nla_tracks or ad.action_blend_type != 'REPLACE'
               or ad.action_influence != 1.0 or ad.action_extrapolation != 'HOLD'
               or ad.use_tweak_mode):
        return {}
    receipt = entry.get('index', {}).get('receipt', {})
    try:
        import cbor2
        rp.decode_bindings(cbor2.dumps({'version':1,'kind':'ppf.blender_bone_bindings','payload':receipt}))
    except (ValueError, KeyError, TypeError, OverflowError):
        return {}
    rows = receipt.get('bindings', [])
    by_id = {row.get('bone_uuid'): row for row in rows if row.get('armature_uuid') == arm.ppf_armature_uuid}
    curves = list(get_id_fcurves(arm)) if ad else []
    if any(not fc.mute and fc.data_path in {'scale', 'delta_scale'} for fc in curves):
        return {}
    result = {}
    for name in snap['names']:
        pb = arm.pose.bones[name]
        chain = [pb, *pb.parent_recursive]
        try:
            if not _static_similarity_parent_chain(arm, [b.name for b in chain]):
                continue
            for b in chain:
                row = by_id.get(b.bone.ppf_bone_physics.bone_uuid)
                if row is None or binding_kind(row) != 'PDRD':
                    raise ValueError('unproven scale ancestor')
                scale = row.get('author_uniform_scale', 1.0)
                if type(scale) not in (float, int) or not math.isfinite(scale) or scale <= 1.e-8:
                    raise ValueError('invalid author scale')
                if any(not c.mute for c in b.constraints):
                    raise ValueError('constraint may change scale')
                if any(not fc.mute and fc.data_path == b.path_from_id('scale') for fc in curves):
                    raise ValueError('animated author scale')
            values = tuple(float(v) for v in pb.scale)
            if len(values) == 3 and all(math.isfinite(v) and v > 0 for v in values):
                result[name] = values
        except (ValueError, KeyError, AttributeError, TypeError, ReferenceError):
            continue
    return result


def _omit_constant_generated_scales(arm, names, created_action, frames, created_actions):
    """Omit provably constant scale curves owned by this bake Action/slot.

    An unproved channel is retained, never flattened or accepted by the encoder.
    This is a scalar-function identity check, not sampled animation inference.
    """
    ad = arm.animation_data
    if (ad is None or ad.action != created_action
            or ad.drivers or ad.nla_tracks
            or getattr(ad, 'action_blend_type', None) != 'REPLACE'
            or getattr(ad, 'action_influence', None) != 1.0
            or getattr(ad, 'action_extrapolation', None) != 'HOLD'
            or getattr(ad, 'use_tweak_mode', True)
            or any(not c.mute for c in arm.constraints)
            or created_action not in created_actions):
        return []
    if arm.parent is not None and not _static_similarity_parent_chain(arm, names):
        return []
    layers = getattr(created_action, 'layers', ())
    if layers:
        slot = getattr(ad, 'action_slot', None)
        if slot is None or sum(s == slot for s in created_action.slots) != 1 or len(layers) != 1:
            return []
        strips = layers[0].strips
        if len(strips) != 1:
            return []
        bag = strips[0].channelbag(slot)
        if bag is None:
            return []
        container = bag.fcurves
    elif hasattr(created_action, 'fcurves'):
        container = created_action.fcurves
    else:
        return []
    expected_frames = [float(frame) for frame in frames]
    if (len(expected_frames) < 2 or any(not math.isfinite(x) for x in expected_frames)
            or any(a >= b for a, b in zip(expected_frames, expected_frames[1:]))):
        return []
    plans = []
    for name in names:
        pb = arm.pose.bones[name]
        if any(not c.mute for bone in [pb, *pb.parent_recursive] for c in bone.constraints):
            continue
        path = pb.path_from_id('scale')
        curves = [fc for fc in container if fc.data_path == path]
        if len(curves) != 3 or sorted(fc.array_index for fc in curves) != [0, 1, 2]:
            continue
        values = []
        safe = True
        for fc in curves:
            keys = list(fc.keyframe_points)
            if (fc.mute or (fc.group is not None and fc.group.mute)
                    or fc.modifiers or fc.sampled_points or fc.extrapolation != 'CONSTANT'
                    or len(keys) != len(expected_frames)
                    or any(float(k.co.x) != frame for k, frame in zip(keys, expected_frames))
                    or any(k.interpolation != 'LINEAR' for k in keys)):
                safe = False
                break
            value = float(keys[0].co.y)
            if not math.isfinite(value) or value <= 0 or any(float(k.co.y) != value for k in keys):
                safe = False
                break
            values.append(value)
        if not safe or any(float(pb.scale[fc.array_index]) != float(fc.keyframe_points[0].co.y) for fc in curves):
            continue
        plans.append((pb, curves))
    for pb, curves in plans:
        for fc in curves:
            container.remove(fc)
    return [pb.name for pb, _curves in plans]


def _finish_steps(entry, *, fast):
    if entry['frames_done']!=entry['n_frames']:raise BoneBakeError('Incomplete bone samples')
    scene=entry['scene']
    # A caller must distinguish a completed rollback from a rollback that
    # itself failed; retrying the latter could destroy the remaining output.
    entry['_finish_rollback_complete'] = False
    try:
        # Restore authored graph/source before assigning the new Action. This
        # also detaches cache override without deleting any external files.
        yield iface_("Preparing bone Actions")
        warnings=rp.clear_bone_pose_cache(scene)
        if warnings:raise BoneBakeError('; '.join(warnings))
        for snap in entry['snapshots']:
            yield iface_("Preparing bone Action {name}").format(name=snap['arm'].name)
            arm=snap['arm'];old=snap['action'];static_scales=_static_pdrd_author_scales(entry,snap) if not entry['single'] else {}
            proven_static_names=set(static_scales)
            new=old.copy() if old else bpy.data.actions.new(arm.name+' Solver Bake')
            new.name=arm.name+' Solver Bake';entry['created'].append(new);_assign(arm,new,snap['slot']);_remove_target_channels(arm,snap['names'])
            previous={}
            collected={}
            # Bindings may arrive in arbitrary order; emit parent channels first.
            ordered_names=sorted(snap['names'], key=lambda name: len(arm.pose.bones[name].parent_recursive))
            for sample_index,(frame,rows) in enumerate(entry['samples']):
                yield iface_("Writing bone frame {done}/{total}").format(done=sample_index + 1, total=len(entry['samples']))
                output_frame=frame
                poses=rows[arm.name]
                if not fast:
                    scene.frame_set(output_frame);bpy.context.view_layer.update()
                for name in ordered_names:
                    pb=arm.pose.bones[name];bone=pb.bone;parent=pb.parent
                    parent_pose=(poses[parent.name] if parent and (fast or parent.name in snap['names']) else (arm.evaluated_get(bpy.context.evaluated_depsgraph_get()).pose.bones[parent.name].matrix.copy() if parent else None))
                    kwargs=dict(parent_matrix=parent_pose,parent_matrix_local=parent.bone.matrix_local) if parent else {}
                    basis=bone.convert_local_to_pose(poses[name],bone.matrix_local,invert=True,**kwargs)
                    loc,q,scale=basis.decompose(); solved_scale=scale.copy()
                    if name in static_scales:
                        from mathutils import Vector
                        scale=Vector(static_scales[name])
                    rebuilt=Matrix.LocRotScale(loc,q,scale)
                    if max(abs(basis[i][j]-rebuilt[i][j]) for i in range(4) for j in range(4))>2e-5:
                        if name in static_scales and sample_index == 0:
                            del static_scales[name]; scale=solved_scale; rebuilt=Matrix.LocRotScale(loc,q,scale)
                        if max(abs(basis[i][j]-rebuilt[i][j]) for i in range(4) for j in range(4))>2e-5:
                            raise _FastPoseMismatch('Author flags require shear not expressible as TRS: '+name) if fast else BoneBakeError('Author flags require shear not expressible as TRS: '+name)
                    pb.location=loc;pb.scale=scale;mode=snap['modes'][name]
                    if mode=='QUATERNION':
                        if name in previous and q.dot(previous[name])<0:q.negate()
                        pb.rotation_quaternion=q;previous[name]=q.copy();channel='rotation_quaternion'
                    elif mode=='AXIS_ANGLE':
                        axis,angle=q.to_axis_angle();pb.rotation_axis_angle=(angle,*axis);channel='rotation_axis_angle'
                    else:
                        e=q.to_euler(mode,previous[name]) if name in previous else q.to_euler(mode);pb.rotation_euler=e;previous[name]=e.copy();channel='rotation_euler'
                    # A single sampled scale is a static pose value. Keying it
                    # would make the later transfer reject it as animated.
                    paths = ('location', channel) if entry['single'] or name in static_scales else ('location', channel, 'scale')
                    for path in paths:
                        if fast:
                            collected.setdefault((name,path), []).append((output_frame, tuple(getattr(pb,path))))
                        if not fast or sample_index == 0:
                            if not pb.keyframe_insert(data_path=path,frame=output_frame,group=pb.name):raise BoneBakeError('Key insertion failed: '+name)
            if fast:
                from array import array
                curves=list(get_id_fcurves(arm))
                for (name,path), values in collected.items():
                    yield iface_("Finalizing bone Action {name}").format(name=arm.name)
                    data_path=arm.pose.bones[name].path_from_id(path)
                    for index in range(len(values[0][1])):
                        matches=[fc for fc in curves if fc.data_path==data_path and fc.array_index==index]
                        if len(matches)!=1 or len(matches[0].keyframe_points)!=1:
                            raise _FastPoseMismatch('Bone channel cannot be bulk written: '+name)
                        fc=matches[0]
                        if len(values)>1:fc.keyframe_points.add(len(values)-1)
                        coordinates=array('f', (component for frame, vector in values for component in (frame, vector[index])))
                        fc.keyframe_points.foreach_set('co',coordinates)
                        fc.update()
            yield iface_("Finalizing bone Action {name}").format(name=arm.name)
            for fc in get_id_fcurves(arm):
                if any(fc.data_path.startswith(arm.pose.bones[n].path_from_id()+'.') for n in snap['names']):
                    for kp in fc.keyframe_points:kp.interpolation='LINEAR'
            if not entry['single']:
                snap['static_scale_names'] = list(static_scales) + _omit_constant_generated_scales(arm, [n for n in snap['names'] if n in proven_static_names and n not in static_scales], new, [frame for frame, _rows in entry['samples']], entry['created'])
            else:
                snap['static_scale_names'] = list(snap['names'])
        if entry['single']:
            # The real depsgraph, under restored author flags and without cache,
            # is the acceptance gate; no successful-but-offset fallback.
            maximum=0.
            for sample_index,(frame,rows) in enumerate(entry['samples']):
                yield iface_("Validating bone frame {done}/{total}").format(done=sample_index + 1, total=len(entry['samples']))
                out=frame
                scene.frame_set(out);bpy.context.view_layer.update();dg=bpy.context.evaluated_depsgraph_get()
                for snap in entry['snapshots']:
                    ev=snap['arm'].evaluated_get(dg)
                    for n in snap['names']:
                        error=max(abs(ev.pose.bones[n].matrix[i][j]-rows[snap['arm'].name][n][i][j]) for i in range(4) for j in range(4));maximum=max(maximum,error)
                        if error>3e-5:
                            detail='Cache-free Action differs from mapped pose: '+n+' error='+str(error)
                            pb=ev.pose.bones[n]
                            if pb.parent is not None and pb.bone.use_connect:
                                from mathutils import Vector
                                parent_matrix=rows[snap['arm'].name][pb.parent.name]
                                target_head=rows[snap['arm'].name][n].translation
                                parent_tail=parent_matrix @ Vector((0.,pb.parent.bone.length,0.))
                                world=snap['arm'].matrix_world
                                gap=((world @ target_head)-(world @ parent_tail)).length
                                detail+='; '+snap['arm'].name+'/'+n+' at frame '+str(frame)+' has authored use_connect=True; mapped head-to-parent-tail world gap='+str(gap)+'. The connected rig may not express this solver pose; source Action and rig are preserved.'
                            raise _FastPoseMismatch(detail) if fast else BoneBakeError(detail)
        yield iface_("Completing bone Action")
        scene.frame_set(entry['entry_time'][0],subframe=entry['entry_time'][1])
        bpy.context.view_layer.update()
        # A successful bake must not discard its now-unused source Action on
        # save/reopen. Track only flags this transaction promotes so a later
        # mixed-object job abort restores the author's original retention state.
        retained=entry.setdefault('retained_source_actions',[])
        for snap in entry['snapshots']:
            old=snap['action']
            if old is not None and old.users==0 and not old.use_fake_user:
                retained.append(old);old.use_fake_user=True
        if entry['single']:
            entry['max_error']=maximum
        entry['committed']=True
    except BaseException:
        rollback(entry)
        entry['_finish_rollback_complete'] = True
        raise

class _FastPoseMismatch(BoneBakeError):
    pass


def _can_bulk_finish(entry):
    if entry['single'] or not entry['samples']:
        return False
    for snap in entry['snapshots']:
        arm=snap['arm']
        for _frame, rows in entry['samples']:
            poses=rows.get(arm.name,{})
            if any(n not in poses or (arm.pose.bones[n].parent and arm.pose.bones[n].parent.name not in poses) for n in snap['names']):
                return False
        action=snap['action']
        if action is None:
            continue
        if not hasattr(action,'layers') and not hasattr(action,'slots'):
            if not hasattr(action,'fcurves'):
                return False
            continue
        layers=getattr(action, 'layers', ())
        if len(getattr(action,'slots',()))!=1 or len(layers)!=1 or len(layers[0].strips)!=1 or layers[0].strips[0].type!='KEYFRAME':
            return False
    return True


def finish_steps(entry):
    fast=_can_bulk_finish(entry)
    try:
        yield from _finish_steps(entry, fast=fast)
    except _FastPoseMismatch as first:
        if not entry.get('_finish_rollback_complete'):
            raise
        for snap in entry['snapshots']:
            snap.pop('static_scale_names',None)
        yield iface_("Preparing bone Actions")
        try:
            yield from _finish_steps(entry, fast=False)
        except BoneBakeError as retry:
            raise BoneBakeError(str(first) + '; frame-evaluated retry: ' + str(retry)) from retry


def finish(entry):
    """Synchronously finish for callers that do not run a modal job."""
    for _status in finish_steps(entry):
        pass


def single(scene):
    entry=make_entry(scene,single=scene.frame_current)
    if entry is not None:
        try:
            sample(entry)
        except BaseException:
            rollback(entry)
            raise
        finish(entry)
    return entry



