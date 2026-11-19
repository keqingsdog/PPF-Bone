"""Reversible single-frame freeze, persisted inside the owning blend file."""
import bpy
import json
import os
from pathlib import Path
from . import pc2, bone_bake, rigid_pose
from ..models.groups import get_addon_data, iterate_active_object_groups

KEY = 'ppf_single_bake_restore_v1'


def frozen_proxy(scene, proxy):
    """Whether this object's current geometry is owned by a single bake."""
    tx = scene.get(KEY)
    return bool(tx and any(row.get('obj') == proxy for row in tx['items']))


def _invalidate_overlay():
    from ..ui import bone_overlay
    from ..models.groups import invalidate_overlays
    bone_overlay._invalidate_snapshots()
    invalidate_overlays()


def available(scene):
    from .full_bake_restore import available as full_available
    return bool(scene.get(KEY)) or full_available(scene)


def _slot(obj):
    return bone_bake._slot(obj.animation_data) if obj.animation_data else None


def _assign(obj, action, slot):
    bone_bake._assign(obj, action, slot)


def _mesh_topology(mesh):
    return (len(mesh.vertices), tuple(tuple(e.vertices) for e in mesh.edges),
            tuple(tuple(p.vertices) for p in mesh.polygons))


def _group_bindings(scene):
    """Only binding identities used to apply the saved mesh frames."""
    groups=[]
    for group in iterate_active_object_groups(scene):
        members=sorted((item.uuid, bool(item.included)) for item in group.assigned_objects)
        groups.append((group.object_type, members))
    return json.dumps(sorted(groups), ensure_ascii=False)


def _proxy_snapshot(bone):
    rows=[]
    for arm_row in bone.get('arms', []):
        arm=arm_row['arm']
        for name in arm_row['names']:
            settings=arm.data.bones[name].ppf_bone_physics
            proxy=settings.proxy
            row={'arm':arm,'bone_name':name,'has_proxy':proxy is not None}
            if proxy is not None:
                if proxy.type!='MESH':raise ValueError('物理代理不是网格；无法保存可恢复的单帧。')
                row['proxy']=proxy;row['data']=proxy.data
                row['coords']=[float(v) for vertex in proxy.data.vertices for v in vertex.co]
                row['topology']=json.dumps(_mesh_topology(proxy.data))
                row['basis']=[float(v) for matrix_row in proxy.matrix_basis for v in matrix_row]
                row['parent_inverse']=[float(v) for matrix_row in proxy.matrix_parent_inverse for v in matrix_row]
                row['parent_type']=proxy.parent_type
                row['parent_bone']=proxy.parent_bone
                if proxy.parent is not None:row['parent']=proxy.parent
            rows.append(row)
    return rows


def _validate_proxy_snapshot(rows):
    for row in rows:
        arm=row['arm'];name=row['bone_name']
        if not arm or name not in arm.data.bones:raise ValueError('物理代理的骨骼已改变；未修改场景。')
        proxy=arm.data.bones[name].ppf_bone_physics.proxy
        if bool(proxy is not None)!=bool(row['has_proxy']):raise ValueError('物理代理绑定已改变；未修改场景。')
        if proxy is None:continue
        if proxy!=row.get('proxy') or proxy.type!='MESH' or proxy.data!=row.get('data'):
            raise ValueError('物理代理已替换；未修改场景。')
        if (proxy.parent!=row.get('parent') or proxy.parent_type!=row['parent_type']
                or proxy.parent_bone!=row['parent_bone']):
            raise ValueError('物理代理的父级已改变；未修改场景。')
        if json.dumps(_mesh_topology(proxy.data))!=row['topology']:
            raise ValueError('物理代理拓扑已改变；未修改场景。')
        values=[float(v) for vertex in proxy.data.vertices for v in vertex.co]
        basis=[float(v) for matrix_row in proxy.matrix_basis for v in matrix_row]
        parent_inverse=[float(v) for matrix_row in proxy.matrix_parent_inverse for v in matrix_row]
        if (len(values)!=len(row['coords']) or any(abs(a-b)>1.e-7 for a,b in zip(values,row['coords']))
                or any(abs(a-b)>1.e-7 for a,b in zip(basis,row['basis']))
                or any(abs(a-b)>1.e-7 for a,b in zip(parent_inverse,row['parent_inverse']))):
            raise ValueError('物理代理几何或变换已改变；未修改场景。')


def _curve_signatures(curves, *, strict=False):
    rows=[]
    for fc in curves:
        points=[]
        for p in fc.keyframe_points:
            points.append([list(p.co),list(p.handle_left),list(p.handle_right),p.interpolation,p.easing,p.handle_left_type,p.handle_right_type,p.amplitude,p.back,p.period])
        modifiers=[]
        for mod in fc.modifiers:
            props={}
            for prop in mod.bl_rna.properties:
                if prop.identifier=='rna_type' or prop.is_readonly:continue
                try:
                    value=getattr(mod,prop.identifier)
                    if isinstance(value,(str,int,float,bool)):props[prop.identifier]=value
                    elif prop.is_array:props[prop.identifier]=list(value)
                    elif strict and isinstance(value,set):props[prop.identifier]=sorted(value)
                    elif strict:raise ValueError('原始Action含未支持的修改器属性。')
                except (AttributeError,TypeError,ValueError):
                    if strict:raise ValueError('原始Action修改器属性无法读取；请保留源动画。')
            if mod.type == 'ENVELOPE':
                props['control_points']=[[p.frame,p.min,p.max] for p in mod.control_points]
            modifiers.append([mod.type,props])
        rows.append([fc.data_path,fc.array_index,fc.extrapolation,fc.mute,points,[list(p.co) for p in fc.sampled_points],modifiers,
                     [fc.group.name,fc.group.mute] if fc.group else None])
    return rows


def _action_signature(obj):
    from .utils import get_id_fcurves
    return json.dumps(_curve_signatures(get_id_fcurves(obj)),sort_keys=True)


def _source_action_snapshot(action):
    """Snapshot detached Action itself, including all slots and layers."""
    if action is None:
        return '', []
    import math
    refs = []
    def custom(value):
        if isinstance(value, bpy.types.ID):
            index=len(refs); refs.append({'id':value}); return ['idref',index]
        if value is None:return ['none']
        if isinstance(value,bool):return ['bool',value]
        if isinstance(value,int):return ['int',value]
        if isinstance(value,float):
            if not math.isfinite(value):raise ValueError('原Action自定义数值不是有限值。')
            return ['float',value]
        if isinstance(value,str):return ['str',value]
        if hasattr(value,'typecode') and hasattr(value,'to_list'):
            return ['array',value.typecode,[custom(x) for x in value.to_list()]]
        if hasattr(value,'keys'):
            return ['group',[[key,custom(value[key])] for key in sorted(value.keys())]]
        if isinstance(value,(list,tuple)):
            return ['list',[custom(x) for x in value]]
        raise ValueError('原Action自定义属性类型尚不支持；未开始烘焙。')
    def source_curves(curves):
        curves=list(curves)
        extra=[]
        for fc in curves:
            def scalar(item, structural=()):
                out={}
                for prop in item.bl_rna.properties:
                    name=prop.identifier
                    if name=='rna_type' or prop.is_readonly or name in structural:continue
                    if prop.type in {'POINTER','COLLECTION'}:raise ValueError('原Action曲线含未知嵌套属性。')
                    value=getattr(item,name)
                    if getattr(prop,'is_array',False):out[name]=list(value)
                    elif isinstance(value,set):out[name]=sorted(value)
                    elif isinstance(value,(str,int,float,bool)):out[name]=value
                    else:raise ValueError('原Action曲线属性无法确认。')
                return out
            extra.append([scalar(fc,{'group','driver','keyframe_points','sampled_points','modifiers'}),
                          scalar(fc.group,{'channels'}) if fc.group else None,
                          [scalar(k) for k in fc.keyframe_points], [scalar(k) for k in fc.sampled_points]])
        return [_curve_signatures(curves,strict=True),extra]
    def values(item):
        out={}
        for prop in item.bl_rna.properties:
            if prop.identifier in {'rna_type','use_fake_user','use_extra_user'} or prop.is_readonly or prop.type in {'COLLECTION','POINTER'}:
                continue
            value=getattr(item,prop.identifier)
            if isinstance(value,(str,int,float,bool)):out[prop.identifier]=value
            elif prop.is_array:out[prop.identifier]=list(value)
            elif isinstance(value,set):out[prop.identifier]=sorted(value)
            else:raise ValueError('原动画包含无法确认的属性；请保留烘焙并备份源Action。')
        return out
    slots=[values(slot) for slot in getattr(action,'slots',())]
    layers=[]
    for layer in getattr(action,'layers',()):
        strips=[]
        for strip in layer.strips:
            bags=[]
            for bag in strip.channelbags:
                bags.append([values(bag),getattr(bag,'slot_handle',None),source_curves(bag.fcurves)])
            strips.append([values(strip),bags])
        layers.append([values(layer),strips])
    legacy=source_curves(getattr(action,'fcurves',()) or []) if not layers else []
    payload=[values(action),slots,layers,legacy,['group',[[key,custom(action[key])] for key in sorted(action.keys())]]]
    return json.dumps(payload,sort_keys=True,allow_nan=False), refs


def _source_action_signature(action):
    return _source_action_snapshot(action)[0]


def _source_refs_equal(actual, expected):
    return len(actual)==len(expected) and all(a.get('id') is not None and a.get('id')==b.get('id') for a,b in zip(actual,expected))


def _preflight_source_owners(objects):
    from . import recovery_anim_graph
    owners=_recovery_owners(objects)
    return owners, _source_action_proofs(owners), recovery_anim_graph.capture(owners,_source_action_snapshot)


def _recheck_source_owners(owners, proofs, graph):
    from . import recovery_anim_graph
    actual=_source_action_proofs(owners)
    if len(actual)!=len(proofs) or any(a['obj']!=b['obj'] or a['action']!=b['action'] or a['signature']!=b['signature'] or not _source_refs_equal(a['source_refs'],b['source_refs']) for a,b in zip(actual,proofs)):
        raise ValueError('源动画在预验后改变；未继续烘焙。')
    recovery_anim_graph.validate(graph,_source_action_snapshot)


def _bone_snapshot_rows(snap):
    rows=[]
    if snap.get('action') is not None:
        rows.append(dict(obj=snap['arm'],action=snap['action'],signature=snap['source_action_proof'],source_refs=snap['source_action_refs']))
    rows.extend(snap.get('source_data_proofs',[]))
    return rows


def _validate_bone_commit_snapshot(tx, entry):
    if entry is None:return
    for snap in entry.get('snapshots',[]):
        if snap.get('source_snapshot_schema')!=2:raise ValueError('骨骼源动画证明版本不匹配。')
        owners=_recovery_owners([snap['arm']])
        expected=[row for row in tx['source_action_proofs'] if row['obj'] in owners]
        actual=_bone_snapshot_rows(snap)
        # Full queue may contain the same already verified bone proof twice.
        for rows in (expected,actual):
            unique=[]
            for row in rows:
                if not any(row['obj']==x['obj'] for x in unique):unique.append(row)
            rows[:]=unique
        if len(actual)!=len(expected) or any(not any(a['obj']==b['obj'] and a['action']==b['action'] and a['signature']==b['signature'] and _source_refs_equal(a['source_refs'],b['source_refs']) for b in expected) for a in actual):
            raise ValueError('骨骼源动画与最初预验不一致；未提交混合恢复记录。')
        retained=[row for proof in tx['source_graph_proofs'] for row in proof['owners'] if row['owner'] in owners]
        current=snap['source_graph_proof']['owners']
        from .recovery_anim_graph import _same_ref_rows
        if any(not any(a['owner']==b['owner'] and a['state']==b['state'] and _same_ref_rows(a['refs'],b['refs']) for b in retained) for a in current) or any(not any(a['owner']==b['owner'] for a in current) for b in retained):
            raise ValueError('骨骼动画依赖与最初预验不一致；未提交混合恢复记录。')


def _validate_bone_source_snapshots(entries):
    from . import recovery_anim_graph
    arms=[]
    for entry in entries:
        for snap in entry.get('snapshots',[]):
            if snap.get('source_snapshot_schema')!=2:raise ValueError('骨骼源动画证明版本不匹配；未开始烘焙。')
            arm=snap['arm'];action=arm.animation_data.action if arm.animation_data else None
            if action!=snap.get('action'):raise ValueError('骨骼源动画绑定已改变。')
            if action is not None:
                signature,refs=_source_action_snapshot(action)
                if signature!=snap.get('source_action_proof') or not _source_refs_equal(refs,snap.get('source_action_refs',[])):
                    raise ValueError('骨骼源动画在预验后改变。')
            _recheck_source_owners(_recovery_owners([arm]),_bone_snapshot_rows(snap),snap['source_graph_proof'])
            arms.append(arm)
    return arms


def _source_action_proofs(objects):
    rows=[]
    for obj in objects:
        action=obj.animation_data.action if obj.animation_data else None
        if action is not None:
            signature,refs=_source_action_snapshot(action)
            rows.append(dict(obj=obj,action=action,signature=signature,source_refs=refs))
    return rows


def _recovery_owners(objects):
    owners=[]
    for obj in objects:
        if obj is None:continue
        owners.append(obj)
        data=getattr(obj,'data',None)
        if data is not None:
            owners.append(data)
            key=getattr(data,'shape_keys',None)
            if key is not None:owners.append(key)
    return owners


def _validate_source_action_proofs(tx, *, permitted_bindings=()):
    expected=[row.get('action') for row in tx.get('items',[]) if row.get('action')]
    expected += [row.get('action') for row in tx.get('bone',{}).get('arms',[]) if row.get('action')]
    proofs=tx.get('source_action_proofs',[])
    if tx.get('source_action_proof_schema') != 2:
        raise ValueError('旧记录缺少原始Action证明；请保留烘焙并备份源动画。')
    if any(not any(p.get('action')==action for p in proofs) for action in expected):
        raise ValueError('原始Action缺少烘焙前证明；请保留烘焙并备份源Action。')
    for row in proofs:
        action=row.get('action')
        if action is None:
            raise ValueError('原始Action已删除；未恢复或消费记录。')
        signature,refs=_source_action_snapshot(action)
        if signature!=row.get('signature') or not _source_refs_equal(refs,row.get('source_refs',[])):
            raise ValueError('原始Action已修改或不可用；未恢复或消费记录。请保留烘焙并备份源Action。')
        owner=row.get('obj')
        if owner is not None and not isinstance(owner,bpy.types.Object):
            ad=owner.animation_data
            if ad is None or ad.action!=action:
                allowed=[item for item in permitted_bindings if item['owner']==owner and item['source']==action]
                if len(allowed)!=1 or ad is None or ad.action!=allowed[0]['working'] or _slot(owner)!=allowed[0]['slot']:
                    raise ValueError('原数据或形态键的Action绑定已改变；未恢复。')


def _validate_source_graph_proofs(tx):
    from . import recovery_anim_graph
    if tx.get('source_graph_proof_schema') != 2:
        raise ValueError('旧记录缺少动画依赖证明；请保留烘焙与源动画。')
    proofs=tx.get('source_graph_proofs',[])
    owners=[]
    for row in tx.get('items',[]):
        obj=row.get('obj')
        if obj is None:continue
        owners.append(obj)
        data=getattr(obj,'data',None)
        if data is not None:owners.append(data)
        # Full baking can create a new Key; only an author Key existing in the
        # pre-bake backup belongs to the source dependency proof.
        backup=row.get('backup')
        if backup is not None and getattr(backup,'shape_keys',None) is not None:
            key=getattr(data,'shape_keys',None)
            if key is not None:owners.append(key)
    owners.extend(_recovery_owners([r.get('arm') for r in tx.get('bone',{}).get('arms',[])]))
    if any(not any(row.get('owner') == owner for proof in proofs for row in proof.get('owners',[])) for owner in owners):
        raise ValueError('恢复目标动画依赖证明不完整；请保留烘焙与源动画。')
    for proof in proofs:
        recovery_anim_graph.validate(proof, _source_action_snapshot)


def _release_backup_safe(scene, tx, backup, own_key=KEY):
    """Only release a simple snapshot when current ownership is provable."""
    if backup is None or getattr(backup,'shape_keys',None) is not None:
        return False
    if (tx.get('working_action_signature_schema') != 2
            or tx.get('source_action_proof_schema') != 2
            or tx.get('source_graph_proof_schema') != 2):
        return False
    try:
        _validate_source_action_proofs(tx)
        _validate_source_graph_proofs(tx)
        users=bpy.data.user_map(subset={backup}).get(backup,set())
        if not all(user == scene for user in users):return False
        def references(value):
            if isinstance(value,bpy.types.ID):return value == backup
            if hasattr(value,'keys'):
                return any(references(value[key]) for key in value.keys())
            if isinstance(value,(list,tuple)) or hasattr(value,'to_list'):
                return any(references(item) for item in value)
            return False
        return not any(references(scene[key]) for key in scene.keys() if key != own_key)
    except Exception:
        return False


def _validate_working_action_schema(tx):
    if tx.get('working_action_signature_schema') != 2:
        raise ValueError('旧恢复记录没有新版工作Action签名，无法确认作者动画未改；未修改场景或清空记录。请先另存保留当前Action的工程，再显式选择Keep保留当前烘焙；Keep会消耗恢复记录。')


def _modifier_signature(mod):
    props={}
    for prop in mod.bl_rna.properties:
        if prop.identifier=='rna_type' or prop.is_readonly:continue
        try:
            value=getattr(mod,prop.identifier)
            if isinstance(value,(str,int,float,bool)):props[prop.identifier]=value
            elif isinstance(value,bpy.types.ID):props[prop.identifier]=[value.bl_rna.identifier,value.name_full]
            elif prop.is_array:props[prop.identifier]=list(value)
        except (AttributeError,TypeError,ValueError):pass
    return json.dumps(props,sort_keys=True)


def _modifier_matches(mod, signature, saved_cache_path=None):
    current = json.loads(_modifier_signature(mod))
    saved = json.loads(signature)
    if mod.type == 'MESH_CACHE' and saved.get('filepath') and current.get('filepath'):
        if saved_cache_path and saved_cache_path.startswith('//'):
            return False  # An old relative record has no saved base directory.
        # Old recovery records stored the original modifier spelling. The
        # separate captured cache path is its authority after Save As.
        saved['filepath'] = _canonical_cache_path(saved_cache_path or saved['filepath'], mod.id_data)
        current['filepath'] = _canonical_cache_path(current['filepath'], mod.id_data)
    return current == saved


def _valid_action_slot(action, row):
    if action is None or 'slot_handle' not in row:return True
    slots=getattr(action,'slots',())
    return any(slot.handle==row['slot_handle'] or slot.identifier==row['slot_identifier'] for slot in slots)


def source_action_flags(objects):
    """Snapshot aliases before promoting any Action's fake user."""
    return {o.animation_data.action.as_pointer():bool(o.animation_data.action.use_fake_user)
            for o in objects if o is not None and o.animation_data and o.animation_data.action}


def begin(scene, captured, *, object_only=False):
    if available(scene):
        raise ValueError('已有可恢复的单帧烘焙，请先恢复模拟。')
    from .uuid_registry import get_object_by_uuid, resolve_assigned
    if object_only:
        if rigid_pose.has_bone_pose_cache(scene):
            raise ValueError('存在骨骼模拟缓存，请使用全部单帧烘焙以保留同步。')
        for group in iterate_active_object_groups(scene):
            for assigned in group.assigned_objects:
                other = resolve_assigned(assigned)
                if other and assigned.uuid not in captured and pc2.has_mesh_cache(other):
                    raise ValueError('存在其他物体模拟缓存，请使用全部单帧烘焙以保留同步。')
    bone_arms=[]
    if not object_only:
        from . import bone_bake
        bone_preview=bone_bake.make_entry(scene,single=scene.frame_current)
        if bone_preview:bone_arms=_validate_bone_source_snapshots([bone_preview])
    owners,proofs,graph=_preflight_source_owners([get_object_by_uuid(uid) for uid in captured]+bone_arms)
    flags=source_action_flags(get_object_by_uuid(uid) for uid in captured)
    items=[]
    try:
        for uid, (kind, _pose) in captured.items():
            if kind != 'MESH':
                raise ValueError('曲线单帧可恢复烘焙尚不支持；原数据保持不变。')
            obj=get_object_by_uuid(uid)
            if obj is None or obj.modifiers.get(pc2.MODIFIER_NAME) is None:
                raise ValueError('单帧缓存目标不可用；原数据保持不变。')
            names=pc2.modifiers_above_cache(obj)+[pc2.MODIFIER_NAME]
            # Animated visibility after freezing would reactivate a modifier.
            ad=obj.animation_data
            if ad and any(not track.mute for track in ad.nla_tracks):
                raise ValueError('物体存在活动NLA轨道，无法安全冻结缓存；请先处理动画。')
            from .utils import get_id_fcurves
            for name in names[:-1]:
                modifier=obj.modifiers[name]
                if any(fc.data_path in (modifier.path_from_id('show_viewport'), modifier.path_from_id('show_render')) for fc in get_id_fcurves(obj)):
                    raise ValueError('上游修改器的可见性已动画化，请先处理动画。')
            if ad and any(any(fc.data_path.startswith(m.path_from_id()+'.') for m in (obj.modifiers[n] for n in names)) for fc in ad.drivers):
                raise ValueError('缓存或上游修改器含驱动，请先处理驱动。')
            _recheck_source_owners(owners,proofs,graph)
            backup=obj.data.copy();backup.use_fake_user=True
            action=ad.action if ad else None
            row=dict(obj=obj, original_data=obj.data, backup=backup, uid=uid, cache_path=bpy.path.abspath(obj.modifiers[pc2.MODIFIER_NAME].filepath), mods=[dict(name=n, type=obj.modifiers[n].type, view=obj.modifiers[n].show_viewport, render=obj.modifiers[n].show_render) for n in names], action_was_fake=flags.get(action.as_pointer(),False) if action else False)
            items.append(row)
            from . import full_bake_restore
            row['cache_path']=_canonical_cache_path(row['cache_path'],obj)
            row['cache_sha256']=full_bake_restore._cache_digest(row['cache_path'])
            row['source_signature']=full_bake_restore.data_signature(backup)
            if action:
                row['action']=action; slot=_slot(obj)
                if slot: row['slot_handle']=slot[0];row['slot_identifier']=slot[1]
                action.use_fake_user=True
        from .facade import runner
        with runner._anim_lock:
            runner_fetched=list(runner._fetched)
        from . import recovery_anim_graph
        _recheck_source_owners(owners,proofs,graph)
        return dict(working_action_signature_schema=2, source_action_proof_schema=2, source_action_proofs=proofs, source_graph_proof_schema=2, source_graph_proofs=[graph], items=items, fetched=get_addon_data(scene).state.convert_fetched_frames_to_list(), runner_fetched=runner_fetched, frame=scene.frame_current, subframe=scene.frame_subframe, prior_stale=bool(scene.get('ppf_single_bake_session_stale')))
    except BaseException:
        discard_uncommitted({'items':items})
        raise


def _slot_row(row):
    return (row['slot_handle'],row['slot_identifier']) if 'slot_handle' in row else None


def _bone_row(entry):
    if entry is None:return {}
    result={'key':entry['key'],'frame':entry['entry_time'][0],'subframe':entry['entry_time'][1],'arms':[]}
    for snap in entry['snapshots']:
        row={'arm':snap['arm'],'names':list(snap['names']),'basis':{n:[float(v) for r in snap['basis'][n] for v in r] for n in snap['names']},'bone_ids':{n:snap['arm'].data.bones[n].ppf_bone_physics.bone_uuid for n in snap['names']}}
        if snap['action']:
            row['action']=snap['action']
            row['source_fake_promoted']=snap['action'] in entry.get('retained_source_actions',[])
        if snap['slot']:row['slot_handle']=snap['slot'][0];row['slot_identifier']=snap['slot'][1]
        if snap['arm'].animation_data and snap['arm'].animation_data.action:row['baked_action']=snap['arm'].animation_data.action
        row['baked_signature']=_action_signature(snap['arm'])
        row['baked_static_scales']={n:[float(v) for v in snap['arm'].pose.bones[n].scale] for n in snap.get('static_scale_names',[])}
        result['arms'].append(row)
    return result


def _validate_static_scales(row):
    import math
    arm=row['arm']
    # Blender stores pose channels as float32 and can re-evaluate them on load.
    # Permit storage roundoff without accepting authored scale changes.
    tolerance=8 * 2**-23
    for name,values in row.get('baked_static_scales',{}).items():
        if (name not in arm.pose.bones or len(values)!=3
                or any(not math.isfinite(float(v)) for v in values)
                or any(not math.isfinite(float(a))
                       or abs(float(a)-float(b)) > tolerance * max(1.0,abs(float(a)),abs(float(b)))
                       for a,b in zip(arm.pose.bones[name].scale,values))):
            raise ValueError('烘焙后骨骼静态缩放已修改；未覆盖作者修改。')


def commit(scene, tx, bone_entry):
    try:
        _validate_bone_commit_snapshot(tx,bone_entry)
        _validate_source_action_proofs(tx)
        _validate_source_graph_proofs(tx)
        for row in tx['items']:
            obj=row['obj']; old=row.get('action')
            if old:
                working=old.copy();_assign(obj,working,_slot_row(row));row['working_action']=working
                pc2.remove_cache_visibility_keys(obj)
            for item in row['mods']:
                mod=obj.modifiers.get(item['name'])
                mod.show_viewport=False;mod.show_render=False
            row['baked_coords']=[float(v) for vert in obj.data.vertices for v in vert.co]
            row['action_signature']=_action_signature(obj)
            for item in row['mods']:item['signature']=_modifier_signature(obj.modifiers[item['name']])
        tx['bone']=_bone_row(bone_entry)
        for snap in bone_entry.get('snapshots',[]) if bone_entry else []:
            if 'source_graph_proof' not in snap:raise ValueError('骨骼动画依赖证明缺失；未提交烘焙。')
            if 'source_graph_proof' in snap:tx['source_graph_proofs'].append(snap['source_graph_proof'])
        relevant=_recovery_owners([r['obj'] for r in tx['items']]+[r['arm'] for r in tx['bone'].get('arms',[])])
        tx['source_action_proofs']=[p for p in tx['source_action_proofs'] if p['obj'] in relevant]
        tx['groups']=_group_bindings(scene)
        tx['proxies']=_proxy_snapshot(tx['bone'])
        tx['invalidated']=False
        scene[KEY]=tx
        _invalidate_overlay()
        get_addon_data(scene).state.clear_fetched_frames()
        from .facade import runner
        from .effects import DoClearAnimation
        runner.execute(DoClearAnimation())
    except BaseException:
        if KEY in scene:
            del scene[KEY]
        state=get_addon_data(scene).state
        state.clear_fetched_frames();state.fetch_frames_from_list(tx['fetched'])
        from .facade import runner
        runner.set_fetched_frames(tx['runner_fetched'])
        for row in tx['items']:
            obj=row['obj']
            if row.get('working_action'):_assign(obj,row.get('action'),_slot_row(row))
            for item in row['mods']:
                mod=obj.modifiers.get(item['name'])
                if mod:mod.show_viewport=item['view'];mod.show_render=item['render']
        raise


def discard_uncommitted(tx):
    for row in tx['items']:
        release_backup(row.get('backup'))
        if row.get('action'):row['action'].use_fake_user=row['action_was_fake']


def release_backup(backup):
    if backup is None:return
    backup.use_fake_user=False
    if backup.users==0:
        if isinstance(backup,bpy.types.Mesh):bpy.data.meshes.remove(backup)
        elif isinstance(backup,bpy.types.Curve):bpy.data.curves.remove(backup)


def invalidate(scene):
    from . import full_bake_restore
    if full_bake_restore.available(scene):full_bake_restore.invalidate(scene)
    if scene.get(KEY):scene[KEY]['invalidated']=True


def suspended_meshes_cover_scene(scene):
    tx=scene.get(KEY)
    if not tx:return False
    covered={row['uid'] for row in tx['items']}
    from .uuid_registry import resolve_assigned
    for group in iterate_active_object_groups(scene):
        for assigned in group.assigned_objects:
            obj=resolve_assigned(assigned)
            if obj and pc2.has_mesh_cache(obj):
                if assigned.uuid not in covered:return False
                mod=obj.modifiers.get(pc2.MODIFIER_NAME)
                if not mod or mod.show_viewport:return False
    return True




def _canonical_cache_path(path, owner=None):
    import os
    if not path:
        return ''
    library = owner.library if owner is not None else None
    return os.path.normcase(os.path.realpath(bpy.path.abspath(path, library=library)))

def _validate_restore_ownership(scene, tx):
    """Admission is repeated after Bake: later sharing is not author permission."""
    for row in tx.get('items', []):
        backup = row.get('backup')
        if backup is not None and any(getattr(owner, 'data', None) == backup for owner in bpy.data.objects):
            raise ValueError('Recovery backup data is now used by an author object; backup and recovery retained. Back up before separating ownership.')
    objects = [row.get('obj') for row in tx.get('items', [])]
    objects += [row.get('arm') for row in tx.get('bone', {}).get('arms', [])]
    for obj in objects:
        if obj is None:
            raise ValueError('Recovery target is missing; baked data and record retained.')
        data = getattr(obj, 'data', None)
        if (obj.library is not None or obj.override_library is not None
                or (data is not None and (data.library is not None or data.override_library is not None))):
            raise ValueError('Recovery target is linked or overridden; back up and make independent local ownership before retrying.')
        if len(obj.users_scene) != 1 or obj.users_scene[0] != scene:
            raise ValueError('Recovery target has changed Scene ownership; baked data and record retained. Back up before making ownership independent.')
        if data is not None and any(other != obj and getattr(other, 'data', None) == data for other in bpy.data.objects):
            raise ValueError('Recovery target data is shared by another object; baked data and record retained. Back up before separating ownership.')
        for other_scene in bpy.data.scenes:
            if other_scene == scene:
                continue
            for key in ('ppf_single_bake_restore_v1', 'ppf_full_bake_restore_v1'):
                other_tx = other_scene.get(key)
                if other_tx and (any(row.get('obj') == obj for row in other_tx.get('items', [])) or any(row.get('arm') == obj for row in other_tx.get('bone', {}).get('arms', []))):
                    raise ValueError('Recovery target is referenced by another Scene recovery record; records retained.')

def restore(scene):
    from . import full_bake_restore
    if full_bake_restore.available(scene):return full_bake_restore.restore(scene)
    tx=scene.get(KEY)
    if not tx:raise ValueError('没有可恢复的单帧烘焙。')
    _validate_restore_ownership(scene, tx)
    for row in tx.get('items', []):
        obj = row.get('obj')
        if obj is not None and obj.type == 'MESH' and obj.data.shape_keys is not None:
            raise ValueError('单帧烘焙后新增了形态键，未恢复或消费记录。请先另存备份；可明确选择保留当前烘焙。')
    _validate_working_action_schema(tx)
    _validate_source_action_proofs(tx)
    _validate_source_graph_proofs(tx)
    # Preserve 1.0.26 static-scale rejection before compensation snapshots.
    for row in tx.get('bone',{}).get('arms',[]):
        if row.get('arm') is not None:_validate_static_scales(row)
    saved_tx=tx.to_dict()
    frame,subframe=scene.frame_current,scene.frame_subframe
    from .facade import runner
    state=get_addon_data(scene).state
    fetched=state.convert_fetched_frames_to_list();runner_fetched=list(runner._fetched)
    prior_stale=scene.get('ppf_single_bake_session_stale')
    bone_key=scene.get(rigid_pose.CACHE_KEY)
    snapshots=[];arms=[];flags=[]
    for row in tx['items']:
        obj=row.get('obj')
        if obj is not None:
            snapshots.append((obj,obj.data,[v.co.copy() for v in obj.data.vertices],obj.animation_data.action if obj.animation_data else None,_slot(obj),obj.animation_data is not None,[(m,m.show_viewport,m.show_render) for m in obj.modifiers]))
        for key in ('backup','action'):
            value=row.get(key)
            if value is not None:flags.append((value,value.use_fake_user))
    for row in tx.get('bone',{}).get('arms',[]):
        arm=row.get('arm')
        if arm is not None:arms.append((arm,arm.animation_data.action if arm.animation_data else None,_slot(arm),arm.animation_data is not None,{pb.name:pb.matrix_basis.copy() for pb in arm.pose.bones}))
        if row.get('action') is not None:flags.append((row['action'],row['action'].use_fake_user))
    try:return _restore(scene)
    except BaseException as exc:
        errors=[]
        for obj,data,coords,action,slot,had_anim_data,mods in snapshots:
            try:
                obj.data=data
                for vertex,co in zip(data.vertices,coords):vertex.co=co
            except Exception as rollback_error:errors.append(str(rollback_error))
            try:
                if had_anim_data:_assign(obj,action,slot)
                elif obj.animation_data is not None:obj.animation_data_clear()
            except Exception as rollback_error:errors.append(str(rollback_error))
            for mod,view,render in mods:
                try:mod.show_viewport=view;mod.show_render=render
                except Exception as rollback_error:errors.append(str(rollback_error))
        for arm,action,slot,had_anim_data,bases in arms:
            try:
                if had_anim_data:_assign(arm,action,slot)
                elif arm.animation_data is not None:arm.animation_data_clear()
            except Exception as rollback_error:errors.append(str(rollback_error))
            for name,basis in bases.items():
                try:arm.pose.bones[name].matrix_basis=basis
                except Exception as rollback_error:errors.append(str(rollback_error))
        try:
            if KEY not in scene:scene[KEY]=saved_tx
            for value,fake in flags:value.use_fake_user=fake
            if bone_key:scene[rigid_pose.CACHE_KEY]=bone_key
            elif rigid_pose.CACHE_KEY in scene:del scene[rigid_pose.CACHE_KEY]
            if prior_stale is not None:scene['ppf_single_bake_session_stale']=prior_stale
            elif 'ppf_single_bake_session_stale' in scene:del scene['ppf_single_bake_session_stale']
            state.clear_fetched_frames();state.fetch_frames_from_list(fetched);runner.set_fetched_frames(runner_fetched)
            scene.frame_set(frame,subframe=subframe)
        except Exception as rollback_error:errors.append(str(rollback_error))
        if errors:raise ValueError('单帧恢复中断，部分回滚未完成；保留恢复记录：'+'; '.join(errors)) from exc
        raise ValueError('单帧恢复未提交，原烘焙与恢复记录已保留：'+str(exc)) from exc


def _restore(scene):
    from . import full_bake_restore
    if full_bake_restore.available(scene):return full_bake_restore.restore(scene)
    from mathutils import Matrix
    tx=scene.get(KEY)
    if not tx:raise ValueError('没有可恢复的单帧烘焙。')
    if tx.get('invalidated'):raise ValueError('代理几何已改变，不能将旧模拟缓存恢复到新物理体。')
    if _group_bindings(scene)!=tx.get('groups'):raise ValueError('物理组绑定已改变；未套用旧模拟缓存。')
    _validate_proxy_snapshot(tx.get('proxies', []))
    for row in tx['items']:
        obj=row['obj']; backup=row['backup']
        if row['cache_path'].startswith('//'):
            raise ValueError('旧版恢复记录只保存了相对缓存路径，缺少原工程目录，无法安全定位缓存。请返回原工程并重新烘焙，或结束此恢复记录。')
        if not obj or not backup or obj.type!='MESH' or obj.data!=row['original_data'] or len(obj.data.vertices)!=len(backup.vertices):
            raise ValueError('恢复目标或原网格不可用；未修改场景。')
        if not row.get('source_signature') or not row.get('cache_sha256'):raise ValueError('旧单帧恢复记录缺少内容指纹，无法安全恢复；请先备份并选择保留烘焙。')
        if full_bake_restore.data_signature(backup)!=row['source_signature']:raise ValueError('原始备份数据已改变；未覆盖当前工程。')
        if _mesh_topology(obj.data)!=_mesh_topology(backup):raise ValueError('烘焙后的网格拓扑已改变；未套用旧缓存。')
        if any(obj.modifiers.get(m['name']) is None for m in row['mods']):raise ValueError('恢复所需修改器已改变；未修改场景。')
        cache=obj.modifiers.get(pc2.MODIFIER_NAME)
        # Save As may rebase the modifier path to // while leaving the ID
        # property's recorded spelling intact. Compare resolved paths instead.
        current_path = _canonical_cache_path(cache.filepath, obj) if cache else ''
        saved_path = _canonical_cache_path(row['cache_path'], obj)
        if (not current_path or not saved_path
                or os.path.normcase(os.path.normpath(current_path)) != os.path.normcase(os.path.normpath(saved_path))
                or not os.path.isfile(current_path)):
            raise ValueError('原模拟缓存不可用或已更改；未修改场景。')
        if full_bake_restore._cache_digest(saved_path)!=row['cache_sha256']:raise ValueError('原模拟缓存内容已改变；未覆盖当前工程。')
        if 'action' in row and row.get('action') is None:raise ValueError('原物体动画已删除，未修改场景。')
        if row.get('working_action') and (not obj.animation_data or obj.animation_data.action!=row['working_action']):raise ValueError('物体动画已改变；未覆盖作者动画。')
        if _action_signature(obj)!=row['action_signature'] or not _valid_action_slot(row.get('action'),row):raise ValueError('物体动画已改变；未覆盖作者动画。')
        if any(not _modifier_matches(obj.modifiers[m['name']],m['signature'],row['cache_path'] if m['name']==pc2.MODIFIER_NAME else None) for m in row['mods']):raise ValueError('冻结修改器已改变；未覆盖作者修改。')
        values=[float(v) for vert in obj.data.vertices for v in vert.co]
        if any(abs(a-b)>1.e-7 for a,b in zip(values,row['baked_coords'])):raise ValueError('烘焙后的网格已编辑；未覆盖作者网格。')
    bone=tx.get('bone')
    if bone:
        if not rigid_pose._io_path(bone['key']).is_file():raise ValueError('原骨骼缓存不可用；未修改场景。')
        index=rigid_pose._read_index_file(bone['key'])
        if not index or any(not rigid_pose._io_path(Path(bone['key']).parent / ('pose_'+n+'.cbor')).is_file() for n in index['frames']):raise ValueError('原骨骼缓存帧不可用；未修改场景。')
        rigid_pose._targets(scene,index['receipt'])
        for row in bone['arms']:
            arm=row['arm']
            if not arm or not arm.animation_data or arm.animation_data.action!=row.get('baked_action'):raise ValueError('骨骼动画已改变；未覆盖作者动画。')
            if _action_signature(arm)!=row['baked_signature'] or not _valid_action_slot(row.get('action'),row):raise ValueError('骨骼动画已改变或原动画槽不可用；未修改场景。')
            for name,uid in row['bone_ids'].items():
                if name not in arm.pose.bones or arm.data.bones[name].ppf_bone_physics.bone_uuid!=uid:raise ValueError('恢复目标骨骼已改变；未修改场景。')
            _validate_static_scales(row)
    for row in tx['items']:
        obj=row['obj']
        for v,old in zip(obj.data.vertices,row['backup'].vertices):v.co=old.co
        if row.get('working_action'):_assign(obj,row.get('action'),_slot_row(row))
        for item in row['mods']:
            mod=obj.modifiers[item['name']];mod.show_viewport=item['view'];mod.show_render=item['render']
        row['backup'].use_fake_user=False
        if row.get('action'):row['action'].use_fake_user=row['action_was_fake']
    if bone:
        for row in bone['arms']:
            arm=row['arm'];_assign(arm,row.get('action'),_slot_row(row))
            if row.get('source_fake_promoted') and row.get('action'):row['action'].use_fake_user=False
            for name,values in row['basis'].items():arm.pose.bones[name].matrix_basis=Matrix([values[i:i+4] for i in range(0,16,4)])
        scene[rigid_pose.CACHE_KEY]=bone['key']
    state=get_addon_data(scene).state
    state.clear_fetched_frames();state.fetch_frames_from_list(list(tx['fetched']))
    from .facade import runner
    runner.set_fetched_frames(list(tx['runner_fetched']))
    frame,subframe=tx['frame'],tx['subframe']
    has_bone=bool(bone)
    if tx.get('prior_stale'):scene['ppf_single_bake_session_stale']=True
    elif 'ppf_single_bake_session_stale' in scene:del scene['ppf_single_bake_session_stale']
    del scene[KEY]
    _invalidate_overlay()
    scene.frame_set(frame,subframe=subframe)
    if has_bone:rigid_pose.bone_pose_frame_change(scene)
    bpy.context.view_layer.update()



def discard(scene):
    from . import full_bake_restore
    if full_bake_restore.available(scene):return full_bake_restore.discard(scene)
    """Keep the baked author result and retire only the in-blend snapshot.

    Cache files and displaced source Actions remain on disk/in the blend.
    """
    tx=scene.get(KEY)
    if not tx:raise ValueError('没有可结束的单帧恢复记录。')
    _validate_restore_ownership(scene, tx)
    # Discard makes the frozen mesh the new author result. Rebase only the
    # original, still-bound proxies before dropping their recovery record.
    from . import bone_author_follow as follow
    from .facade import communicator
    if communicator.busy() or rigid_pose.has_bone_pose_cache(scene):
        raise ValueError('模拟仍在运行，无法结束单帧恢复记录。')
    bpy.context.view_layer.update()
    items = {row['obj'].as_pointer(): row for row in tx['items'] if row.get('obj')}
    removals = []
    backups = []
    for row in tx['items']:
        obj = row.get('obj'); backup = row.get('backup')
        cache_path = row['cache_path']
        if obj and not obj.is_editable:
            raise ValueError('冻结物体不可编辑，无法结束单帧恢复记录。')
        if backup and not backup.is_editable:
            raise ValueError('恢复备份不可编辑，无法结束单帧恢复记录。')
        if obj:
            for item in reversed(row['mods']):
                name, kind, signature = item['name'], item['type'], item['signature']
                mod = obj.modifiers.get(name)
                if (mod and mod.type == kind and not mod.show_viewport
                        and _modifier_matches(mod, signature, cache_path if name == pc2.MODIFIER_NAME else None)):
                    removals.append((obj, mod))
        if backup: backups.append(backup)
    targets = []
    for row in tx.get('proxies', []):
        arm = row.get('arm'); proxy = row.get('proxy'); name = row['bone_name']
        if not arm or not proxy or proxy.as_pointer() not in items or name not in arm.data.bones:
            continue
        bone = arm.data.bones[name]; settings = bone.ppf_bone_physics
        if settings.proxy != proxy or settings.body_uuid != items[proxy.as_pointer()]['uid']:
            continue
        pose = arm.pose.bones.get(name)
        if (pose is None or not follow._finite_invertible(arm.matrix_world @ pose.matrix)
                or not follow._finite_invertible(proxy.matrix_world)):
            raise ValueError('代理或骨骼变换无效，无法结束单帧恢复记录。')
        targets.append((arm, bone, proxy))
    records = [(proxy, follow._AUTHOR_RECORD in proxy, proxy.get(follow._AUTHOR_RECORD)) for _, _, proxy in targets]
    states = (follow._seen.copy(), follow._rest_seen.copy(), follow._pose_preview.copy(),
              follow._unproven.copy(), follow._author_signatures.copy())
    try:
        for arm, bone, proxy in targets:
            follow.baseline_proxy(scene, arm, bone)
    except BaseException:
        for proxy, had_record, record in records:
            if had_record: proxy[follow._AUTHOR_RECORD] = record
            elif follow._AUTHOR_RECORD in proxy: del proxy[follow._AUTHOR_RECORD]
        for current, saved in zip((follow._seen, follow._rest_seen, follow._pose_preview,
                                   follow._unproven, follow._author_signatures), states):
            current.clear(); current.update(saved)
        raise
    for obj, mod in removals:
        obj.modifiers.remove(mod)
    for backup in backups:
        backup.use_fake_user=not _release_backup_safe(scene,tx,backup)
    del scene[KEY]
    _invalidate_overlay()
