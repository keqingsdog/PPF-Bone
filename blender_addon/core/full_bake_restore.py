"""Explicit full-animation recovery; disk inputs remain until user cleanup."""
import json
import bpy
from . import single_bake_restore as single, rigid_pose, pc2
from ..models.groups import get_addon_data
KEY = 'ppf_full_bake_restore_v1'


def _cache_digest(path):
    """Bind recovery to the retained bytes, not merely a reusable filename."""
    import hashlib
    import os
    digest = hashlib.sha256()
    path_before=os.stat(path) if os.name == 'nt' else None
    with open(path, 'rb') as stream:
        before = os.fstat(stream.fileno())
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
        path_after=os.stat(path)
    signature = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    if os.name == 'nt':
        same = (signature(before) == signature(after)
                and signature(path_before) == signature(path_after)
                and signature(after)[:4] == signature(path_after)[:4])
    else:
        same = signature(before) == signature(after) == signature(path_after)
    if not same:
        raise ValueError('模拟缓存正在改变，未提交恢复。')
    return digest.hexdigest()


def available(scene):
    return bool(scene.get(KEY))


def _validate_full_source_graph_proofs(tx):
    """Only a proven Full Key/Curve writer may create this exact default graph.

    Original persisted graph proofs are never rewritten. The expected view is
    derived from the fixed transition contract, not from the current graph.
    """
    empty = json.dumps([{}, [], []], sort_keys=True, ensure_ascii=False)
    created = []
    for item in tx.get('items', []):
        binding = item.get('key_source_binding')
        if binding is None or 'created_animdata' not in binding:continue
        if (binding['created_animdata'] is not True or binding.get('had_animation') is not False
                or binding.get('source_present') is not False or binding.get('source') is not None
                or 'working' not in binding or binding['backup_key'].animation_data is not None):
            raise ValueError('Mesh Key created AnimData receipt is invalid')
        key = binding['key']
        if key in created:raise ValueError('Duplicate created AnimData receipt')
        created.append(key)
    for item in tx.get('items', []):
        binding = item.get('curve_source_binding')
        if binding is None or 'created_animdata' not in binding:continue
        if (binding['created_animdata'] is not True or binding.get('had_animation') is not False
                or binding.get('source_present') is not False or binding.get('source') is not None
                or 'working' not in binding or binding['backup'].animation_data is not None
                or not _key_slot_matches(binding['backup'], binding['source_slot'])):
            raise ValueError('Curve created AnimData receipt is invalid')
        data = binding['data']
        if data in created:raise ValueError('Duplicate created AnimData receipt')
        created.append(data)
    proofs = []
    matched = []
    for proof in tx.get('source_graph_proofs', []):
        rows = []
        for original in proof.get('owners', []):
            row = dict(original.items())
            owner = row.get('owner')
            if owner in created:
                if row.get('state')!=empty or len(row.get('refs', []))!=0 or owner in matched:
                    raise ValueError('Mesh Key original graph does not prove absent AnimData')
                ad = owner.animation_data
                if ad is None or ad.use_nla is not True:
                    raise ValueError('Created AnimData is not in its permitted state')
                prop = ad.bl_rna.properties.get('use_nla')
                settings = {} if prop is not None and prop.default is True else {'use_nla': True}
                matched.append(owner)
                row['state'] = json.dumps([settings, [], []], sort_keys=True, ensure_ascii=False)
            rows.append(row)
        copy = dict(proof.items());copy['owners'] = rows;proofs.append(copy)
    if len(matched)!=len(created):raise ValueError('Mesh Key original graph proof is missing')
    view = dict(tx.items());view['source_graph_proofs'] = proofs
    single._validate_source_graph_proofs(view, original_meshes=_original_meshes(tx))


def _original_meshes(tx):
    """Bind detached originals to the exact pre-bake Mesh and Key IDs."""
    originals = {}
    for row in tx.get('items', []):
        if not row.get('retains_original_mesh'):
            continue
        obj, backup = row.get('obj'), row.get('backup')
        key = row.get('original_mesh_key')
        if (obj is None or not isinstance(backup, bpy.types.Mesh)
                or key is None or backup.shape_keys != key
                or obj.data != row.get('data') or obj.data == backup
                or obj in originals):
            raise ValueError('Original mesh or its working bake binding changed; recovery retained')
        originals[obj] = backup
    return originals


def prepare_working_mesh(obj, entry):
    """Create the bake output without touching the original mesh or its Key."""
    row = entry['original_mesh_row']
    source = row['backup']
    if obj != row['obj'] or obj.data != source or source.shape_keys != row['original_mesh_key']:
        raise ValueError('Original mesh changed before preparing the bake output')
    working = source.copy()
    if working.shape_keys is None or working.shape_keys == source.shape_keys:
        if working.users == 0:
            bpy.data.meshes.remove(working)
        raise ValueError('Could not create an independent bake mesh')
    try:
        temporary = bpy.data.objects.new("PPF bake mesh preparation", working)
        try:
            temporary.shape_key_clear()
        finally:
            bpy.data.objects.remove(temporary)
        if working.shape_keys is not None:
            raise ValueError('Could not clear the copied mesh shape keys')
        entry['working_base_signature'] = data_signature(working)
    except BaseException:
        if working.users == 0:
            bpy.data.meshes.remove(working)
        raise
    row['data'] = working
    obj.data = working


def working_mesh_unchanged(entry):
    row = entry['original_mesh_row']
    working = row['data']
    obj = row.get('obj')
    return (obj is not None and obj.data == working and working != row['backup']
            and data_signature(working, include_shape_keys=False) == entry.get('working_base_signature'))


def rollback_working_mesh(row):
    """Rebind only while the transaction still owns the object binding."""
    source, obj = row['backup'], row.get('obj')
    if obj is not None and obj.data == row['data'] and obj.data != source:
        obj.data = source
    if obj is not None and obj.data == source:
        source.use_fake_user = row['original_mesh_fake_user']


def begin(scene, queue):
    rows=[]
    try:return _begin(scene,queue,rows)
    except BaseException:
        discard_pending({'items':rows})
        raise


def _key_slot_record(owner):
    slot=single._slot(owner)
    return dict(present=slot is not None,handle=slot[0],identifier=slot[1]) if slot else dict(present=False)


def _key_slot_matches(owner, record):
    if not hasattr(record,'keys') or not isinstance(record.get('present'),bool):return False
    expected={'present','handle','identifier'} if record['present'] else {'present'}
    if set(record.keys())!=expected:return False
    slot=single._slot(owner)
    if not record['present']:return slot is None
    if not isinstance(record['handle'],int) or isinstance(record['handle'],bool) or not isinstance(record['identifier'],str) or not record['identifier']:return False
    return slot is not None and slot==(record['handle'],record['identifier'])


def _capture_source_key_binding(obj, backup):
    if obj.type!='MESH' or obj.data.shape_keys is None:return None
    key=obj.data.shape_keys;backup_key=backup.shape_keys
    if backup_key is None or backup_key==key:raise ValueError('Mesh Key backup identity is unavailable')
    action=key.animation_data.action if key.animation_data else None
    backup_action=backup_key.animation_data.action if backup_key.animation_data else None
    slot=_key_slot_record(key)
    if backup_action!=action or not _key_slot_matches(backup_key,slot):raise ValueError('Mesh Key backup Action binding differs from source')
    return dict(schema=1,obj=obj,data=obj.data,key=key,backup=backup,backup_key=backup_key,
                source_present=action is not None,source=action,source_slot=slot,had_animation=key.animation_data is not None)


def _validate_key_source_identity(row, binding):
    if type(binding.get('schema')) is not int or binding['schema']!=1:raise ValueError('Mesh Key source binding schema is unavailable')
    obj=row.get('obj');data=row.get('data');backup=row.get('backup');key=binding.get('key');backup_key=binding.get('backup_key')
    if (obj is None or data is None or backup is None or key is None or backup_key is None
            or obj!=binding.get('obj') or data!=binding.get('data') or backup!=binding.get('backup')
            or obj.type!='MESH' or obj.data!=data or data.shape_keys!=key or backup.shape_keys!=backup_key
            or key==backup_key):
        raise ValueError('Mesh Key or backup identity changed; recovery retained')
    source=binding.get('source')
    if type(binding.get('source_present')) is not bool or binding['source_present']!=(source is not None):raise ValueError('Original Key Action is missing')
    backup_action=backup_key.animation_data.action if backup_key.animation_data else None
    if backup_action!=source or not _key_slot_matches(backup_key,binding['source_slot']):
        raise ValueError('Backup Key Action identity or slot changed; recovery retained')
    return key,source


def _plan_key_bindings(tx, bindings):
    plans=[]
    for row in tx['items']:
        binding=row.get('key_source_binding')
        if binding is None:continue
        key,source=_validate_key_source_identity(row,binding)
        matches=[item for item in bindings if item['owner']==key]
        if len(matches)>1:raise ValueError('Ambiguous Mesh Key writer binding')
        if matches:
            item=matches[0]
            source_slot=binding['source_slot']
            expected_slot=(source_slot.get('handle'),source_slot.get('identifier')) if source_slot.get('present') else None
            if item['source']!=source or item.get('source_slot')!=expected_slot:raise ValueError('Mesh Key writer source differs from begin snapshot')
            ad=key.animation_data
            if ad is None or item.get('working') is None or ad.action!=item['working'] or single._slot(key)!=item.get('slot'):
                raise ValueError('Mesh Key current working Action or slot differs from validated writer')
            updates=dict(working=item['working'],working_slot=_key_slot_record(key),working_signature=single._action_signature(key))
            if 'had_animation' in binding:
                if type(binding['had_animation']) is not bool or type(item.get('had_animation')) is not bool:
                    raise ValueError('Mesh Key AnimData provenance is unavailable')
                if binding['had_animation']!=item['had_animation']:
                    raise ValueError('Mesh Key AnimData changed before its writer')
                if not binding['had_animation']:
                    if source is not None or expected_slot is not None or binding['backup_key'].animation_data is not None:
                        raise ValueError('Mesh Key absent AnimData provenance changed')
                    updates['created_animdata']=True
            plans.append((binding,updates))
        else:
            current=key.animation_data.action if key.animation_data else None
            if current!=source or not _key_slot_matches(key,binding['source_slot']):raise ValueError('Unproven Mesh Key Action replacement')
    return plans



def _capture_curve_source_binding(obj, backup):
    if obj.type!='CURVE':return None
    data=obj.data
    action=data.animation_data.action if data.animation_data else None
    if backup==data or (backup.animation_data is not None)!=(data.animation_data is not None):
        raise ValueError('Curve source backup identity or AnimData is unavailable')
    slot=_key_slot_record(data)
    backup_action=backup.animation_data.action if backup.animation_data else None
    if backup_action!=action or not _key_slot_matches(backup,slot):
        raise ValueError('Curve backup source Action binding differs')
    return dict(schema=1,obj=obj,data=data,backup=backup,source_present=action is not None,
                source=action,source_slot=slot,had_animation=data.animation_data is not None)


def _validate_curve_source_identity(row, binding):
    if type(binding.get('schema')) is not int or binding['schema']!=1:
        raise ValueError('Curve source binding schema is unavailable')
    obj=row.get('obj');data=row.get('data');backup=row.get('backup')
    if (obj is None or data is None or backup is None or obj.type!='CURVE'
            or obj!=binding.get('obj') or data!=binding.get('data')
            or backup!=binding.get('backup') or obj.data!=data or backup==data):
        raise ValueError('Curve source or backup data identity changed; recovery retained')
    source=binding.get('source')
    if (type(binding.get('source_present')) is not bool or binding['source_present']!=(source is not None)
            or type(binding.get('had_animation')) is not bool
            or (source is not None and not binding['had_animation'])):
        raise ValueError('Original Curve Action or AnimData provenance is unavailable')
    ad=backup.animation_data
    backup_action=ad.action if ad else None
    if ((ad is not None)!=binding['had_animation'] or backup_action!=source
            or not _key_slot_matches(backup,binding['source_slot'])):
        raise ValueError('Backup Curve source Action identity or slot changed')
    return data,source


def _plan_curve_bindings(tx, bindings):
    plans=[];seen=[]
    for row in tx.get('items',[]):
        binding=row.get('curve_source_binding')
        if binding is None:continue
        data,source=_validate_curve_source_identity(row,binding)
        if data in seen:raise ValueError('Duplicate Curve source mapping')
        seen.append(data)
        if ((source is not None and not any(p.get('obj')==data and p.get('action')==source for p in tx.get('source_action_proofs',[])))
                or (source is None and any(p.get('obj')==data for p in tx.get('source_action_proofs',[])))):
            raise ValueError('Curve source proof differs from its begin mapping')
        matches=[item for item in bindings if item['owner']==data]
        if len(matches)>1:raise ValueError('Ambiguous Curve writer binding')
        if not matches:
            current=data.animation_data.action if data.animation_data else None
            if ((data.animation_data is not None)!=binding['had_animation']
                    or current!=source or not _key_slot_matches(data,binding['source_slot'])):
                raise ValueError('Unproven Curve Action replacement')
            continue
        item=matches[0];source_slot=binding['source_slot']
        expected_slot=(source_slot.get('handle'),source_slot.get('identifier')) if source_slot.get('present') else None
        if (item['source']!=source or item.get('source_slot')!=expected_slot
                or type(item.get('had_animation')) is not bool
                or item['had_animation']!=binding['had_animation']):
            raise ValueError('Curve writer source differs from begin snapshot')
        ad=data.animation_data
        if ad is None or item.get('working') is None or ad.action!=item['working'] or single._slot(data)!=item.get('slot'):
            raise ValueError('Curve current working Action or slot differs from validated writer')
        updates=dict(working=item['working'],working_slot=_key_slot_record(data),
                     working_signature=single._action_signature(data))
        if not binding['had_animation']:
            if source is not None or expected_slot is not None or binding['backup'].animation_data is not None:
                raise ValueError('Curve absent AnimData provenance changed')
            updates['created_animdata']=True
        plans.append((binding,updates))
    return plans


def _commit_source_bindings(tx, bindings):
    # Validate every Key and Curve row before publishing any mapping update.
    plans=_plan_key_bindings(tx,bindings)+_plan_curve_bindings(tx,bindings)
    for binding,updates in plans:
        binding.update(updates)


def _validated_restore_curve_bindings(tx):
    bindings=[];seen=[]
    for row in tx.get('items',[]):
        binding=row.get('curve_source_binding')
        if binding is None:continue  # Historical records gain no new exception.
        data,source=_validate_curve_source_identity(row,binding)
        if data in seen:raise ValueError('Duplicate Curve recovery mapping')
        seen.append(data)
        if ((source is not None and not any(p.get('obj')==data and p.get('action')==source for p in tx.get('source_action_proofs',[])))
                or (source is None and any(p.get('obj')==data for p in tx.get('source_action_proofs',[])))):
            raise ValueError('Curve source proof does not match its original binding')
        current=data.animation_data.action if data.animation_data else None
        if 'working' not in binding:
            if ((data.animation_data is not None)!=binding['had_animation']
                    or current!=source or not _key_slot_matches(data,binding['source_slot'])):
                raise ValueError('Unproven Curve Action binding')
            continue
        working=binding.get('working')
        if (working is None or current!=working or not _key_slot_matches(data,binding['working_slot'])
                or single._action_signature(data)!=binding.get('working_signature')):
            raise ValueError('Working Curve Action identity, slot or content changed')
        bindings.append(dict(owner=data,source=source,working=working,slot=single._slot(data)))
    return bindings


def _validated_restore_key_bindings(tx):
    bindings=[];seen=[]
    for row in tx.get('items',[]):
        binding=row.get('key_source_binding')
        if binding is None:continue  # Old records gain no new binding exception.
        key,source=_validate_key_source_identity(row,binding)
        if key in seen:raise ValueError('Duplicate Mesh Key recovery mapping')
        seen.append(key)
        if source is not None and not any(p.get('obj')==key and p.get('action')==source for p in tx.get('source_action_proofs',[])):
            raise ValueError('Mesh Key source proof does not match its original binding')
        current=key.animation_data.action if key.animation_data else None
        if 'working' not in binding:
            if current!=source or not _key_slot_matches(key,binding['source_slot']):raise ValueError('Unproven Mesh Key Action binding')
            continue
        working=binding.get('working')
        if (working is None or current!=working or not _key_slot_matches(key,binding['working_slot'])
                or single._action_signature(key)!=binding.get('working_signature')):
            raise ValueError('Working Key Action identity, slot or content changed; recovery retained')
        bindings.append(dict(owner=key,source=source,working=working,slot=single._slot(key)))
    return bindings


def _begin(scene, queue, rows):
    from .uuid_registry import get_object_by_uuid
    if available(scene):raise ValueError('请先恢复或保留上一次完整烘焙。')
    bone_arms=single._validate_bone_source_snapshots([e for e in queue if e['obj_type']=='BONE'])
    owners,proofs,graph=single._preflight_source_owners([get_object_by_uuid(e['obj_uuid']) for e in queue if e['obj_type'] in {'MESH','CURVE'}]+bone_arms)
    actions=single.source_action_flags(get_object_by_uuid(e['obj_uuid']) for e in queue if e['obj_type'] in {'MESH','CURVE'})
    for entry in queue:
        if entry['obj_type'] not in {'MESH','CURVE'}:continue
        obj=get_object_by_uuid(entry['obj_uuid'])
        if obj is None:raise ValueError('烘焙对象已删除。')
        single._recheck_source_owners(owners,proofs,graph)
        retain_original = bool(entry.get('use_working_mesh'))
        backup = obj.data if retain_original else obj.data.copy()
        original_fake_user = backup.use_fake_user
        backup.use_fake_user=True
        action=obj.animation_data.action if obj.animation_data else None
        was_fake=actions.get(action.as_pointer(),False) if action else False
        row=dict(obj=obj,data=obj.data,backup=backup,mods=[dict(name=m.name,view=m.show_viewport,render=m.show_render) for m in obj.modifiers])
        if action:row['action']=action;row['action_was_fake']=was_fake
        rows.append(row)
        if retain_original:
            row['retains_original_mesh'] = True
            row['original_mesh_key'] = backup.shape_keys
            row['original_mesh_fake_user'] = original_fake_user
            entry['original_mesh_row'] = row
        key_binding=None if retain_original else _capture_source_key_binding(obj,backup)
        if key_binding is not None:row['key_source_binding']=key_binding
        curve_binding=_capture_curve_source_binding(obj,backup)
        if curve_binding is not None:row['curve_source_binding']=curve_binding
        if action:action.use_fake_user=True
        cache=obj.modifiers.get(pc2.MODIFIER_NAME)
        if cache and cache.filepath:row['cache_path']=bpy.path.abspath(cache.filepath)
        if obj.type=='CURVE':
            from .curve_output import resolve_read_path
            path=resolve_read_path(obj)
            if path:row['cache_path']=path
        if row.get('cache_path'):row['cache_sha256']=_cache_digest(row['cache_path'])
        if action:row['action']=action;row['action_was_fake']=was_fake
        slot=single._slot(obj)
        if slot:row['slot_handle']=slot[0];row['slot_identifier']=slot[1]
        row['source_signature']=_source_mesh_signature(row)
    from .facade import runner
    from . import recovery_anim_graph
    graphs=[graph]
    for entry in queue:
        if entry['obj_type']!='BONE':continue
        for snap in entry.get('snapshots',[]):
            if 'source_graph_proof' not in snap:raise ValueError('骨骼动画依赖证明缺失；未提交烘焙。')
            graphs.append(snap['source_graph_proof'])
    single._recheck_source_owners(owners,proofs,graph)
    return dict(working_action_signature_schema=2, source_action_proof_schema=2, source_action_proofs=proofs, source_graph_proof_schema=2, source_graph_proofs=graphs, items=rows, groups=single._group_bindings(scene),
                fetched=get_addon_data(scene).state.convert_fetched_frames_to_list(),
                runner_fetched=list(runner._fetched), frame=scene.frame_current,
                subframe=scene.frame_subframe,
                bone_key=scene.get(rigid_pose.CACHE_KEY,''))


def _rna_values(value):
    result={}
    for prop in value.bl_rna.properties:
        key=prop.identifier
        if key=='rna_type' or prop.is_readonly:continue
        item=getattr(value,key)
        if isinstance(item,(str,int,float,bool)):result[key]=item
        elif getattr(prop,'is_array',False):result[key]=list(item)
        elif isinstance(item,set):result[key]=sorted(item)
        elif isinstance(item,bpy.types.ID):result[key]=[item.bl_rna.identifier,item.name_full,item.library.filepath if item.library else '']
    return result


def _plain_value(value):
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if isinstance(value,bpy.types.ID):return [value.bl_rna.identifier,value.name_full,value.library.filepath if value.library else '']
    if hasattr(value,'to_dict'):return {k:_plain_value(v) for k,v in value.to_dict().items()}
    if hasattr(value,'to_list'):return [_plain_value(v) for v in value.to_list()]
    if isinstance(value,dict):return {k:_plain_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_plain_value(v) for v in value]
    raise ValueError('作者自定义属性无法安全保存恢复签名。')


def _custom_values(data):
    try:keys=data.keys()
    except TypeError:return {}  # KeyBlock on Blender versions without IDProperties.
    return {k:_plain_value(data[k]) for k in keys}


def _author_layers(data):
    """Only evaluated at bake/restore boundaries, never in UI/timer polling."""
    result={'properties':_rna_values(data),'materials':[_plain_value(m) for m in data.materials],
            'custom':{k:_plain_value(data[k]) for k in data.keys()}}
    if isinstance(data,bpy.types.Mesh):
        result['attributes']=[(a.name,a.data_type,a.domain,[_rna_values(v) for v in a.data]) for a in data.attributes]
        result['uv_layers']=[(u.name,u.active_render,u.active_clone) for u in data.uv_layers]
        result['polygons']=[(p.material_index,p.use_smooth) for p in data.polygons]
        result['edges']=[e.use_seam for e in data.edges]
        result['weights']=[[(g.group,g.weight) for g in v.groups] for v in data.vertices]
    else:
        result['splines']=[[_rna_values(s),[_rna_values(v) for v in (s.bezier_points if s.type=='BEZIER' else s.points)]] for s in data.splines]
    return result


def _animated_shape_paths(key):
    if key.animation_data is None:
        return set()
    from ..ui.dynamics.bake_ops import _single_animation_channels
    return {curve.data_path for _, curve in _single_animation_channels(key) if not curve.mute} | {
        curve.data_path for curve in key.animation_data.drivers if not curve.mute
    }


def _shape_settings(key, animated):
    properties = _rna_values(key)
    if 'eval_time' in animated:
        properties.pop('eval_time', None)
    blocks = []
    for block in key.key_blocks:
        settings = _rna_values(block)
        if block.path_from_id('value') in animated:
            settings.pop('value', None)
        blocks.append({'properties': settings, 'relative': block.relative_key.name if block.relative_key else None,
                       'custom': _custom_values(block)})
    return {'properties': properties, 'custom': {k: _plain_value(key[k]) for k in key.keys()}, 'blocks': blocks}


def _source_mesh_signature(row):
    backup = row['backup']
    return data_signature(backup, source_shape_graph=bool(row.get('retains_original_mesh')))


def data_signature(data, *, source_shape_graph=False, include_shape_keys=True):
    if isinstance(data,bpy.types.Mesh):
        shape=data.shape_keys if include_shape_keys else None
        animated = _animated_shape_paths(shape) if shape and source_shape_graph else ()
        return json.dumps([single._mesh_topology(data),[list(v.co) for v in data.vertices],
            [[k.name,[list(v.co) for v in k.data]] for k in shape.key_blocks] if shape else [],
            single._action_signature(shape) if shape else '',
            _shape_settings(shape, animated) if shape else {},_author_layers(data)],sort_keys=True)
    rows=[]
    for s in data.splines:
        rows.append([s.type,[[list(v.co),list(v.handle_left),list(v.handle_right),v.handle_left_type,v.handle_right_type] for v in s.bezier_points]
                     if s.type=='BEZIER' else [list(v.co) for v in s.points]])
    return json.dumps([rows,single._action_signature(data),_author_layers(data)],sort_keys=True)


def _unkeyed_bone_scales(row):
    """Read only unkeyed target scale axes; these persist outside the Action."""
    import math
    from .utils import get_id_fcurves
    arm = row.get('arm')
    if arm is None:raise ValueError('目标骨骼已删除，未修改场景。')
    curves = list(get_id_fcurves(arm))
    values = {}
    for name in row['names']:
        pb = arm.pose.bones.get(name)
        if pb is None:raise ValueError('目标骨骼已删除，未修改场景。')
        keyed = {fc.array_index for fc in curves if fc.data_path == pb.path_from_id('scale')}
        axes = {}
        for axis in range(3):
            if axis not in keyed:
                value = float(pb.scale[axis])
                if not math.isfinite(value):raise ValueError('骨骼静态缩放无效，未修改场景。')
                axes[str(axis)] = value
        if axes:values[name] = axes
    return values


def _validate_static_bone_scales(tx):
    # Run before frame_set/snapshots: a missing Action curve cannot prove that
    # its static pose value is still the value accepted at Bake completion.
    for row in tx.get('bone', {}).get('arms', []):
        actual = _unkeyed_bone_scales(row)
        version = row.get('static_scale_proof_schema')
        if version is None:
            if actual:raise ValueError('旧记录缺少骨骼静态缩放指纹；已保留烘焙与记录，请先另存备份，可选择保留烘焙。')
        elif version != 1 or _plain_value(row.get('static_scale_proof', {})) != actual:
            raise ValueError('烘焙后骨骼静态缩放已修改或指纹不可用，未覆盖作者修改。')


def _validated_commit_bindings(queue):
    """Permit only the current Full writer's proven data/Key Action replacement."""
    from ..ui.dynamics import bake_ops
    from .uuid_registry import get_object_by_uuid
    bindings=[]
    for entry in queue:
        kind=entry['obj_type']
        if kind not in {'MESH','CURVE'}:continue
        obj=get_object_by_uuid(entry['obj_uuid'])
        if kind=='MESH':
            receipt=entry.get('mesh_action_transaction')
            if receipt is None:continue
            bake_ops._validate_mesh_action(obj,entry)
            owner=receipt['key']
        else:
            receipt=entry.get('curve_transaction')
            if receipt is None:continue
            bake_ops._validate_curve_transaction(obj,entry)
            owner=receipt['data']
        if owner is None or receipt.get('working') is None or any(row['owner']==owner for row in bindings):
            raise ValueError('Full Bake writer Action receipt is missing or ambiguous')
        bindings.append(dict(owner=owner,source=receipt['action'],working=receipt['working'],slot=receipt['working_slot'],source_slot=receipt['slot'],had_animation=receipt['had_animation']))
    return bindings


def preflight_sources(tx, queue):
    """Read-only source admission before any target is finalized.

    Plan permitted writer bindings on independent row dictionaries only. Keep
    the original pending proofs and mappings untouched; commit must recheck.
    """
    bones=[entry for entry in queue if entry['obj_type']=='BONE']
    for entry in bones:single._validate_bone_commit_snapshot(tx,entry)
    single._validate_bone_source_snapshots(bones)
    bindings=_validated_commit_bindings(queue)
    view=dict(tx.items());rows=[]
    for original in tx.get('items',[]):
        row=dict(original.items())
        if 'key_source_binding' in original:
            row['key_source_binding']=dict(original['key_source_binding'].items())
        if 'curve_source_binding' in original:
            row['curve_source_binding']=dict(original['curve_source_binding'].items())
        rows.append(row)
    view['items']=rows
    _commit_source_bindings(view,bindings)
    single._validate_source_action_proofs(view,permitted_bindings=bindings)
    _validate_full_source_graph_proofs(view)


def _curve_point_count(obj):
    return sum(3 * len(s.bezier_points) if s.type == 'BEZIER' else len(s.points) for s in obj.data.splines)


def _is_zero_curve_cache(path, count):
    import struct
    try:
        with open(path, 'rb') as stream:
            header = stream.read(33)
        if len(header) != 32 or header[:12] != pc2.PC2_HEADER:
            return False
        version, points, start, sampling, frames = struct.unpack('<iiffi', header[12:])
        import math
        return (version == pc2.PC2_VERSION and points == count and frames == 0
                and math.isfinite(start) and math.isfinite(sampling) and sampling > 0)
    except (OSError, TypeError, ValueError, struct.error):
        return False


def commit(scene, tx, queue):
    single._validate_bone_commit_snapshot(tx,next((e for e in queue if e['obj_type']=='BONE'),None))
    bindings=_validated_commit_bindings(queue)
    _commit_source_bindings(tx,bindings)
    single._validate_source_action_proofs(tx,permitted_bindings=bindings)
    _validate_full_source_graph_proofs(tx)
    scene.frame_set(scene.frame_current,subframe=scene.frame_subframe)
    bpy.context.view_layer.update()
    bone=next((e for e in queue if e['obj_type']=='BONE'),None)
    tx['bone']=single._bone_row(bone) if bone else {}
    relevant=single._recovery_owners([r['obj'] for r in tx['items']]+[r['arm'] for r in tx['bone'].get('arms',[])])
    for backup in _original_meshes(tx).values():
        relevant.extend((backup, backup.shape_keys))
    tx['source_action_proofs']=[p for p in tx['source_action_proofs'] if p['obj'] in relevant]
    for row in tx['bone'].get('arms', []):
        row['static_scale_proof_schema']=1
        row['static_scale_proof']=_unkeyed_bone_scales(row)
    tx['proxies']=single._proxy_snapshot(tx['bone'])
    for row in tx['items']:
        obj=row['obj']
        entries = [e for e in queue if e.get('obj_type') == 'CURVE' and e.get('obj_uuid') == obj.get('_solver_uuid')]
        if (obj.type == 'CURVE' and len(entries) == 1 and entries[0].get('n_frames') == 0
                and entries[0].get('frames_done') == 0 and row.get('curve_source_binding')
                and _is_zero_curve_cache(row.get('cache_path'), _curve_point_count(obj))):
            row['curve_no_output_schema'] = 1
            row['curve_no_output_points'] = _curve_point_count(obj)
        row['signature']=data_signature(obj.data)
        row['object_action']=single._action_signature(obj)
        if obj.animation_data and obj.animation_data.action:row['current_action']=obj.animation_data.action
        row['mod_signatures']=[single._modifier_signature(m) for m in obj.modifiers]
    tx['validation_frame']=scene.frame_current
    tx['validation_subframe']=scene.frame_subframe
    scene[KEY]=tx
    get_addon_data(scene).state.clear_fetched_frames()
    from .facade import runner
    from .effects import DoClearAnimation
    runner.execute(DoClearAnimation())


def invalidate(scene):
    if available(scene):scene[KEY]['invalidated']=True


def _validate_current_key_graph(tx):
    """Only the default current Key Action evaluation is certified here."""
    for row in tx.get('items', []):
        obj = row.get('obj')
        data = getattr(obj, 'data', None) if obj is not None else None
        key = getattr(data, 'shape_keys', None) if data is not None else None
        ad = key.animation_data if key is not None else None
        if ad is None:
            continue
        if (len(ad.drivers) or len(ad.nla_tracks)
                or ad.action_blend_type != 'REPLACE'
                or ad.action_influence != 1.0
                or ad.action_extrapolation != 'HOLD'
                or ad.use_tweak_mode):
            raise ValueError('当前形态键含未认证的驱动、NLA或非默认动画混合；未恢复或消费记录。请先另存备份，可明确选择保留当前烘焙。')

def _validate_backup_key_graph(tx):
    """Do not restore or release an uncertified backup Key animation graph."""
    originals = _original_meshes(tx)
    for row in tx.get('items', []):
        if row.get('obj') in originals:
            # An exact retained original is verified against the source graph
            # and is never released as a generated backup.
            continue
        backup = row.get('backup')
        key = getattr(backup, 'shape_keys', None) if backup is not None else None
        ad = key.animation_data if key is not None else None
        if ad is None:
            continue
        if (len(ad.drivers) or len(ad.nla_tracks)
                or ad.action_blend_type != 'REPLACE'
                or ad.action_influence != 1.0
                or ad.action_extrapolation != 'HOLD'
                or ad.use_tweak_mode):
            raise ValueError('备份形态键含未认证的动画图，未恢复、释放备份或消费记录。请先另存保留备份图；确认处理备份图后重试。')


def _validate_backup_source_signatures(tx):
    """Read every retained backup before any Restore snapshot or frame change."""
    for row in tx.get('items',[]):
        backup=row.get('backup');expected=row.get('source_signature')
        if backup is None or not isinstance(expected,str) or not expected:
            raise ValueError('原始备份或数据指纹缺失；未修改场景。')
        if _source_mesh_signature(row)!=expected:
            raise ValueError('原始备份数据或作者动画已改变；未进入恢复或切换帧。')


def _validation_backup_signatures(tx):
    """Predict only simple, proven Curve custom-property Action evaluation."""
    from .utils import get_id_fcurves
    result = []
    for row in tx['items']:
        expected = json.loads(row['source_signature'])
        backup = row['backup']
        binding = row.get('curve_source_binding')
        ad = getattr(backup, 'animation_data', None)
        layers = getattr(ad.action, 'layers', None) if ad and ad.action else None
        if (binding and ad and ad.action and ad.action == binding.get('source')
                and ad.action_blend_type == 'REPLACE' and ad.action_extrapolation == 'HOLD'
                and ad.action_influence == 1.0 and not ad.drivers and not ad.nla_tracks
                and not getattr(ad, 'use_tweak_mode', True)
                and layers is not None and len(layers) == 1 and len(layers[0].strips) == 1
                and layers[0].strips[0].type == 'KEYFRAME'):
            curves = list(get_id_fcurves(backup))
            parsed = []
            for fc in curves:
                path = fc.data_path
                try:
                    key = json.loads(path[1:-1]) if path.startswith('["') and path.endswith('"]') else None
                except (ValueError, TypeError):
                    key = None
                if (not isinstance(key, str) or key not in expected[2]['custom']
                        or type(expected[2]['custom'][key]) is not float or fc.array_index != 0
                        or fc.mute or fc.modifiers or any(k == key for k, _ in parsed)):
                    parsed = []
                    break
                parsed.append((key, fc))
            for key, fc in parsed:
                expected[2]['custom'][key] = fc.evaluate(tx['validation_frame'] + tx['validation_subframe'])
        result.append(json.dumps(expected, sort_keys=True))
    return result


def restore(scene):
    from . import bone_bake
    frame,sub=scene.frame_current,scene.frame_subframe
    tx=scene.get(KEY)
    if not tx:raise ValueError('没有可恢复的完整烘焙。')
    single._validate_restore_ownership(scene, tx)
    _validate_backup_key_graph(tx)
    _validate_current_key_graph(tx)
    single._validate_working_action_schema(tx)
    single._validate_source_action_proofs(tx,permitted_bindings=_validated_restore_key_bindings(tx)+_validated_restore_curve_bindings(tx))
    _validate_full_source_graph_proofs(tx)
    _validate_static_bone_scales(tx)
    # Reject an edited static scale before rollback snapshots are reassigned;
    # decomposing matrix_basis itself can perturb an untouched author value.
    for row in tx.get('bone',{}).get('arms',[]):
        if row.get('arm') is not None:single._validate_static_scales(row)
    _validate_backup_source_signatures(tx)
    validation_signatures = _validation_backup_signatures(tx)
    saved_tx=tx.to_dict()
    objects=[]
    arms=[]
    source_flags=[(row['action'],bool(row['action'].use_fake_user)) for row in tx['items'] if row.get('action')]
    for row in tx['items']:
        obj=row.get('obj')
        if obj is not None:
            objects.append((obj,obj.data,obj.animation_data.action if obj.animation_data else None,
                            single._slot(obj),obj.animation_data is not None,[(m,m.show_viewport,m.show_render) for m in obj.modifiers]))
    for row in tx.get('bone',{}).get('arms',[]):
        arm=row.get('arm')
        if arm is not None:
            arms.append((arm,arm.animation_data.action if arm.animation_data else None,
                         single._slot(arm),arm.animation_data is not None,{pb.name:pb.matrix_basis.copy() for pb in arm.pose.bones}))
    from .facade import runner
    state=get_addon_data(scene).state
    fetched=state.convert_fetched_frames_to_list();runner_fetched=list(runner._fetched)
    bone_key=scene.get(rigid_pose.CACHE_KEY)
    try:return _restore(scene, validation_signatures)
    except BaseException as exc:
        errors=[]
        for obj,data,action,slot,had_anim_data,mods in objects:
            try:
                obj.data=data
                if had_anim_data:bone_bake._assign(obj,action,slot)
                elif obj.animation_data is not None:obj.animation_data_clear()
                for mod,view,render in mods:mod.show_viewport=view;mod.show_render=render
            except Exception as rollback_error:errors.append(str(rollback_error))
        for arm,action,slot,had_anim_data,bases in arms:
            try:
                if had_anim_data:bone_bake._assign(arm,action,slot)
                elif arm.animation_data is not None:arm.animation_data_clear()
                for name,basis in bases.items():
                    if name in arm.pose.bones:arm.pose.bones[name].matrix_basis=basis
            except Exception as rollback_error:errors.append(str(rollback_error))
        if bone_key:scene[rigid_pose.CACHE_KEY]=bone_key
        elif rigid_pose.CACHE_KEY in scene:del scene[rigid_pose.CACHE_KEY]
        state.clear_fetched_frames();state.fetch_frames_from_list(fetched)
        runner.set_fetched_frames(runner_fetched)
        if KEY not in scene:scene[KEY]=saved_tx
        for action,fake in source_flags:action.use_fake_user=fake
        for row in scene[KEY]['items']:
            if row.get('backup'):row['backup'].use_fake_user=True
        scene.frame_set(frame,subframe=sub)
        if errors:raise ValueError('恢复中断，部分回滚未完成；保留恢复记录：'+'; '.join(errors)) from exc
        raise ValueError('恢复未提交，原烘焙与恢复记录已保留：'+str(exc)) from exc


def _restore(scene, validation_signatures):
    tx=scene.get(KEY)
    if not tx:raise ValueError('没有可恢复的完整烘焙。')
    if tx.get('invalidated'):raise ValueError('代理几何已改变，不能恢复旧缓存。')
    scene.frame_set(tx['validation_frame'],subframe=tx['validation_subframe'])
    bpy.context.view_layer.update()
    single._validate_source_action_proofs(tx,permitted_bindings=_validated_restore_key_bindings(tx)+_validated_restore_curve_bindings(tx))
    _validate_full_source_graph_proofs(tx)
    if single._group_bindings(scene)!=tx['groups']:raise ValueError('物理组已改绑，未覆盖当前工程。')
    single._validate_proxy_snapshot(tx.get('proxies',[]))
    for index,row in enumerate(tx['items']):
        obj=row.get('obj');backup=row.get('backup')
        if obj is None or backup is None or obj.data!=row['data']:raise ValueError('烘焙对象或数据已替换，未覆盖当前工程。')
        if 'action' in row and row.get('action') is None:raise ValueError('原物体动画已删除，未修改场景。')
        if not single._valid_action_slot(row.get('action'),row):raise ValueError('原物体动画槽已删除，未修改场景。')
        if _source_mesh_signature(row)!=validation_signatures[index]:raise ValueError('原始数据或作者动画已改变，未修改场景。')
        if row.get('cache_path') and not __import__('os').path.isfile(row['cache_path']):raise ValueError('原模拟缓存已被删除，未修改当前工程。')
        if row.get('cache_path'):
            expected=row.get('cache_sha256')
            if not expected:raise ValueError('旧恢复记录没有缓存内容指纹，无法安全恢复；原烘焙与记录已保留。')
            if _cache_digest(row['cache_path'])!=expected:raise ValueError('原模拟缓存内容已改变，未覆盖当前工程。')
        if data_signature(obj.data)!=row['signature'] or single._action_signature(obj)!=row['object_action']:
            raise ValueError('烘焙后对象或动画已编辑，未覆盖作者修改。')
        action=obj.animation_data.action if obj.animation_data else None
        if action!=row.get('current_action'):raise ValueError('烘焙后动画已改绑，未覆盖作者修改。')
        signatures=list(row['mod_signatures'])
        if len(obj.modifiers)!=len(signatures) or any(not single._modifier_matches(m,sig,row.get('cache_path') if m.type=='MESH_CACHE' and m.name==pc2.MODIFIER_NAME else None) for m,sig in zip(obj.modifiers,signatures)):raise ValueError('烘焙后修改器已修改，未覆盖作者修改。')
    bone=tx.get('bone',{})
    if tx['bone_key']:
        from pathlib import Path
        index=rigid_pose._read_index_file(tx['bone_key'])
        if index is None or any(not rigid_pose._io_path(Path(tx['bone_key']).parent / ('pose_'+n+'.cbor')).is_file() for n in index['frames']):
            raise ValueError('原骨骼缓存帧不可用，未修改场景。')
        rigid_pose._targets(scene,index['receipt'])
    for row in bone.get('arms',[]):
        arm=row.get('arm')
        if arm is None or any(name not in arm.pose.bones or arm.data.bones[name].ppf_bone_physics.bone_uuid!=uid for name,uid in row['bone_ids'].items()):raise ValueError('目标骨骼已删除或改绑，未修改场景。')
        if 'action' in row and row.get('action') is None:raise ValueError('原骨骼动画已删除，未修改场景。')
        if not single._valid_action_slot(row.get('action'),row):raise ValueError('原骨骼动画槽不可用，未修改场景。')
        if arm is None or not arm.animation_data or arm.animation_data.action!=row.get('baked_action') or single._action_signature(arm)!=row['baked_signature']:
            raise ValueError('烘焙后骨骼动画已修改，未覆盖作者修改。')
        single._validate_static_scales(row)
    # Everything is admitted before any source data is reassigned.
    for row in tx['items']:
        obj=row['obj'];obj.data=row['backup']
        row['backup'].use_fake_user=bool(row.get('original_mesh_fake_user', False))
        if row.get('action') is not None or obj.animation_data is not None:
            single._assign(obj,row.get('action'),single._slot_row(row))
        if row.get('action'):row['action'].use_fake_user=row['action_was_fake']
        for state in row['mods']:
            mod=obj.modifiers[state['name']];mod.show_viewport=state['view'];mod.show_render=state['render']
    from mathutils import Matrix
    for row in bone.get('arms',[]):
        arm=row['arm'];single._assign(arm,row.get('action'),single._slot_row(row))
        for name,values in row['basis'].items():
            if name not in arm.pose.bones:raise ValueError('目标骨骼已删除。')
            arm.pose.bones[name].matrix_basis=Matrix([values[i:i+4] for i in range(0,16,4)])
    if tx['bone_key']:scene[rigid_pose.CACHE_KEY]=tx['bone_key']
    state=get_addon_data(scene).state;state.clear_fetched_frames();state.fetch_frames_from_list(list(tx['fetched']))
    from .facade import runner
    runner.set_fetched_frames(list(tx['runner_fetched']))
    frame,sub=tx['frame'],tx['subframe'];del scene[KEY]
    scene.frame_set(frame,subframe=sub);bpy.context.view_layer.update()
    if scene.get(rigid_pose.CACHE_KEY):rigid_pose.bone_pose_frame_change(scene)


def _keep_curve_outputs(scene, tx):
    """Keep current author output, not old signatures, and stop only owned playback.

    Curve-containing Keep deliberately retains groups, disabled Mesh modifiers,
    and backups. No frame evaluation or author geometry/Action write is needed.
    """
    from pathlib import Path
    from . import curve_output
    plans=[]
    for row in tx.get('items', []):
        obj=row.get('obj')
        if obj is None or obj.type != 'CURVE':continue
        if obj.data != row.get('data'):
            raise ValueError('曲线数据已替换，无法确认Keep缓存归属；场景与恢复记录保留。')
        key=pc2.object_pc2_key_readonly(obj)
        if len([other for other in bpy.data.objects if pc2.object_pc2_key_readonly(other) == key]) != 1:
            raise ValueError('曲线Keep缓存key另有对象引用，未修改任何播放或恢复记录。')
        locator=curve_output._record(obj)
        path=row.get('cache_path')
        digest=row.get('cache_sha256')
        binding=row.get('curve_source_binding')
        if type(row.get('curve_no_output_schema')) is int and row['curve_no_output_schema'] == 1:
            if (not binding or any(k in binding for k in ('working','working_slot','working_signature','created_animdata'))
                    or not key or not path or Path(path).stem != key or locator is not None
                    or key in pc2._curve_cache or key in curve_output._live_paths
                    or type(row.get('curve_no_output_points')) is not int or row['curve_no_output_points'] < 1
                    or not _is_zero_curve_cache(path,row['curve_no_output_points'])):
                raise ValueError('零帧曲线缓存身份不明确；场景与恢复记录保留。')
            _validate_curve_source_identity(row,binding)
            if any(plan[1] == key for plan in plans):
                raise ValueError('曲线Keep缓存key重复，未修改场景。')
            plans.append((obj,key,None))
            continue
        legacy=tx.get('working_action_signature_schema') is None and not digest
        if legacy:
            if not key or not path or Path(path).stem != key:
                raise ValueError('旧曲线缓存归属不明确；未修改播放或恢复记录。')
            if any(plan[1] == key for plan in plans):
                raise ValueError('曲线Keep缓存key重复，未修改场景。')
            plans.append((obj,key,True))
            continue
        if (not key or not locator or not path or not digest
                or Path(path).stem != key):
            raise ValueError('曲线Keep缺少可验证缓存身份；请先备份，场景与恢复记录保留。')
        if _cache_digest(path) != digest or not curve_output._matches(path, locator['identity']):
            raise ValueError('曲线Keep原缓存内容已改变，未清除播放或恢复记录。')
        memory=pc2._curve_cache.get(key)
        if memory is not None and curve_output._identity(memory) != locator['identity']:
            raise ValueError('曲线Keep内存缓存不是保留的输出，未修改场景。')
        live=curve_output._live_paths.get(key)
        if live is not None and _cache_digest(live) != digest:
            raise ValueError('曲线Keep当前加载路径已改变，未修改场景。')
        if locator['status'] != 'detached':
            current=curve_output.resolve_read_path(obj)
            if not current or _cache_digest(current) != digest:
                raise ValueError('曲线Keep当前缓存不是保留的输出，未修改场景。')
        if any(plan[1] == key for plan in plans):
            raise ValueError('曲线Keep缓存key重复，未修改场景。')
        plans.append((obj,key,False))
    sentinel=object()
    saved=[]
    for obj,key,legacy in plans:
        saved.append((obj,key,obj.get(curve_output.PROPERTY,sentinel),
                      curve_output._live_paths.get(key,sentinel),
                      pc2._curve_cache.get(key,sentinel),
                      pc2._curve_pending_author_types.get(key,sentinel),
                      key in pc2._handles_freed,obj.get(curve_output.LEGACY_DETACHED_KEY,sentinel)))
    old_last=pc2._curve_last_frame
    old_stale=scene.get('ppf_single_bake_session_stale',sentinel)
    old_tx=tx.to_dict()
    try:
        for obj,key,legacy in plans:
            if legacy is None:continue
            if legacy:obj[curve_output.LEGACY_DETACHED_KEY]=key
            else:curve_output.mark_detached(obj)
            pc2.unload_curve_cache(key)
        for source in _original_meshes(tx).values():
            source.use_fake_user = True
        scene['ppf_single_bake_session_stale']=True
        del scene[KEY]
    except BaseException as exc:
        errors=[]
        def restore_field(owner,key,value):
            if value is sentinel:
                if key in owner:del owner[key]
            else:owner[key]=value
        for obj,key,locator,live,cache,pending,freed,legacy_marker in saved:
            for label,owner,name,value in (
                    ('locator',obj,curve_output.PROPERTY,locator),
                    ('legacy marker',obj,curve_output.LEGACY_DETACHED_KEY,legacy_marker),
                    ('live path',curve_output._live_paths,key,live),
                    ('cache',pc2._curve_cache,key,cache),
                    ('pending types',pc2._curve_pending_author_types,key,pending)):
                try:restore_field(owner,name,value)
                except Exception as recovery_error:errors.append(label+': '+str(recovery_error))
            try:
                if freed:pc2._handles_freed.add(key)
                else:pc2._handles_freed.discard(key)
            except Exception as recovery_error:errors.append('handles: '+str(recovery_error))
        pc2._curve_last_frame=old_last
        for label,owner,key,value in (
                ('stale flag',scene,'ppf_single_bake_session_stale',old_stale),
                ('record',scene,KEY,old_tx)):
            try:restore_field(owner,key,value)
            except Exception as recovery_error:errors.append(label+': '+str(recovery_error))
        if errors:raise RuntimeError('曲线Keep失败且部分补偿失败；请保留现场和缓存备份：'+ '; '.join(errors)) from exc
        raise


def discard(scene):
    tx=scene.get(KEY)
    if tx:single._validate_restore_ownership(scene, tx)
    if tx:_validate_backup_key_graph(tx)
    if tx and any(row.get('obj') is not None and row['obj'].type == 'CURVE' for row in tx.get('items', [])):
        return _keep_curve_outputs(scene, tx)
    frame,sub=scene.frame_current,scene.frame_subframe
    try:return _discard(scene)
    finally:scene.frame_set(frame,subframe=sub)


def _discard(scene):
    tx=scene.get(KEY)
    if not tx:raise ValueError('没有可结束的完整烘焙恢复。')
    scene.frame_set(tx['validation_frame'],subframe=tx['validation_subframe'])
    bpy.context.view_layer.update()
    # Deliberately leave source files available to Undo/recovery. Explicit Clear
    # performs ordinary ownership-checked cleanup when the user requests it.
    from ..models.groups import iterate_active_object_groups
    targets={row['obj'].as_pointer() for row in tx['items'] if row.get('obj') and len(row['obj'].modifiers)==len(row['mod_signatures']) and all(single._modifier_matches(m,sig,row.get('cache_path') if m.type=='MESH_CACHE' and m.name==pc2.MODIFIER_NAME else None) for m,sig in zip(row['obj'].modifiers,row['mod_signatures']))} if single._group_bindings(scene)==tx['groups'] else set()
    for group in iterate_active_object_groups(scene):
        for i in range(len(group.assigned_objects)-1,-1,-1):
            from .uuid_registry import resolve_assigned
            obj=resolve_assigned(group.assigned_objects[i])
            if obj is not None and obj.as_pointer() in targets:group.assigned_objects.remove(i)
    for row in tx['items']:
        if row.get('backup'):
            if row.get('retains_original_mesh'):
                row['backup'].use_fake_user = True
            else:
                row['backup'].use_fake_user=not single._release_backup_safe(scene,tx,row['backup'],own_key=KEY)
        obj=row.get('obj')
        if obj is None:continue
        unchanged=(obj.data==row['data'] and data_signature(obj.data)==row['signature'] and single._action_signature(obj)==row['object_action'])
        if obj.type=='CURVE':
            if unchanged:
                from .curve_output import mark_detached
                mark_detached(obj)
        else:
            mod=obj.modifiers.get(pc2.MODIFIER_NAME)
            expected=next((signature for signature in row['mod_signatures'] if json.loads(signature).get('name')==pc2.MODIFIER_NAME),None)
            if mod is not None and not mod.show_viewport and expected is not None and single._modifier_matches(mod,expected,row.get('cache_path')):obj.modifiers.remove(mod)
    scene['ppf_single_bake_session_stale']=True
    del scene[KEY]


def suspend_object(obj):
    action=obj.animation_data.action if obj.animation_data else None
    if action:
        slot=single._slot(obj)
        single._assign(obj,action.copy(),slot)
    pc2.remove_cache_visibility_keys(obj)


def discard_pending(tx, *, restore_source_flags=True):
    for row in tx.get('items',[]):
        if row.get('retains_original_mesh'):
            if row.get('obj') is not None and row['obj'].data == row['backup']:
                row['backup'].use_fake_user = row['original_mesh_fake_user']
        else:
            single.release_backup(row.get('backup'))
        if restore_source_flags and row.get('action'):row['action'].use_fake_user=row['action_was_fake']
