"""Validated, transactional cross-chain distance-spring authoring in SI units.

No native indices or body-local frames are guessed here. The encoder emits
world anchors alongside the bone contract; the frontend owns native frames.
"""
import json
import math
import struct
import uuid
import bpy
from pathlib import Path

from mathutils import Matrix, Vector

SCHEMA = 'PPF_BONE_CROSS_LINKS_V1'
PRESET_KIND = 'ppf.bone_cross_links_preset'
MAX_LINKS = 10000
F32_MAX = 3.4028234663852886e38
REFERENCE_SCALE_MIN = struct.unpack('<f',struct.pack('<f',1.e-8))[0]
FIELDS = ('link_uuid','enabled','bone_a_uuid','bone_b_uuid','bone_a_name','bone_b_name',
          'anchor_a_bone','anchor_b_bone','rest_length_m','reference_armature_scale',
          'stiffness_N_per_m','damping_Ns_per_m','source','topology',
          'tension_enabled','compression_enabled','tension_stiffness_N_per_m','compression_stiffness_N_per_m','exclude_collision')
AUTHOR_FIELDS = FIELDS
FIELDS = AUTHOR_FIELDS + ('preview_color_custom', 'preview_color')
LEGACY_FIELDS = AUTHOR_FIELDS[:-5]
DIRECTION_FIELDS = AUTHOR_FIELDS[:-1]


def _f32(value, label):
    """RNA stores binary32: reject clamp/zero-underflow before any batch write."""
    if type(value) not in (int,float):raise ValueError(label+'必须是数值')
    try:value=float(value)
    except OverflowError:raise ValueError(label+'超出有限 float32 范围') from None
    if not math.isfinite(value) or abs(value)>F32_MAX:
        raise ValueError(label+'超出有限 float32 范围')
    rounded=struct.unpack('<f',struct.pack('<f',value))[0]
    if value!=0 and rounded==0:
        raise ValueError(label+'过小，float32 存储会下溢为零')
    return float(value)


def _number(value, label, positive=False):
    value=_f32(value,label)
    if value < 0 or positive and value <= 0:
        raise ValueError(label+'必须是有限'+('正数' if positive else '非负数'))
    return value


def _point(value):
    if not isinstance(value, (list,tuple)) or len(value)!=3:
        raise ValueError('锚点必须是三个有限数值')
    return [_f32(v,'锚点坐标') for v in value]


def _coefficients(stiffness,damping):
    k=_number(stiffness,'刚度');c=_number(damping,'阻尼')
    if k==0 and c==0:raise ValueError('跨链刚度与阻尼不能同时为零；请停用或删除连接')
    return k,c


def _preflight_records(records):
    """One final boundary covers explicit values, presets and derived defaults."""
    for row in records:
        _point(row['anchor_a_bone']);_point(row['anchor_b_bone'])
        _number(row['rest_length_m'],'作者参考距离',True)
        scale=_number(row['reference_armature_scale'],'参考骨架缩放',True)
        if scale<REFERENCE_SCALE_MIN:raise ValueError('参考骨架缩放小于RNA存储下限')
        _number(row['damping_Ns_per_m'],'阻尼')
        if type(row.get('preview_color_custom',False)) is not bool:raise ValueError('颜色自定义开关必须是布尔值')
        color=row.get('preview_color',(0.2,0.85,1.,1.))
        if (not isinstance(color,(list,tuple)) or len(color)!=4
                or any(type(v) not in (int,float) or not 0<=v<=1 for v in color)):
            raise ValueError('颜色必须是0至1的RGBA数值')
        if type(row['exclude_collision']) is not bool:raise ValueError('碰撞排除必须是布尔值')
        for direction in ('tension','compression'):
            if type(row[direction+'_enabled']) is not bool:raise ValueError('方向开关必须是布尔值')
            _number(row[direction+'_stiffness_N_per_m'],'方向刚度')


def armature_scale(arm):
    from ..models.bone_bindings import similarity_frame_scale
    return similarity_frame_scale(arm.matrix_world, arm.name)[1]


def require_ready(context, arm, writable=True, allow_edit=False):
    from . import bone_author_follow
    from ..ui.dynamics.group_ops import cleanup_identity_rebuilding
    if bone_author_follow._identity_rebuilding or cleanup_identity_rebuilding():
        raise ValueError('Undo/Redo 正在重建身份，请稍后操作跨链连接')
    if arm is None or arm.type!='ARMATURE' or context.scene.objects.get(arm.name)!=arm:
        raise ValueError('请选择当前场景的一副骨架')
    if float(context.scene.unit_settings.scale_length)!=1.:
        raise ValueError('跨链连接使用 SI 单位，场景单位缩放必须为1')
    if arm.mode=='EDIT' and not allow_edit:
        raise ValueError('请退出骨骼编辑模式后操作跨链连接')
    if not arm.ppf_armature_uuid or sum(o.type=='ARMATURE' and o.ppf_armature_uuid==arm.ppf_armature_uuid for o in context.scene.objects)!=1:
        raise ValueError('骨架身份缺失或重复，请先修复骨骼物理绑定')
    armature_scale(arm)
    if writable:
        if arm.library or arm.data.library or arm.data.users!=1:
            raise ValueError('请使用本地、独立骨架数据编辑跨链连接')
        from .facade import communicator
        from ..ops.bone_proxy import stale_simulation_output
        communicator.assert_bone_body_switch_idle()
        if stale_simulation_output(context):
            raise ValueError('请先 Clear 已有模拟帧，再修改跨链连接')


def _resolve_bone(arm, bone_uuid, bone_index=None):
    matches=(bone_index.get(bone_uuid, []) if bone_index is not None else [b for b in arm.data.bones if b.ppf_bone_physics.bone_uuid==bone_uuid and bone_uuid])
    if len(matches)!=1:raise ValueError('端点骨骼已删除或 UUID 不唯一')
    return matches[0]


def _link_bones(arm, bone_a_uuid, bone_b_uuid, bone_index=None):
    if bone_a_uuid==bone_b_uuid:raise ValueError('跨链连接不能连接同一骨骼')
    return (_resolve_bone(arm,bone_a_uuid,bone_index),
            _resolve_bone(arm,bone_b_uuid,bone_index))


def _link_paused(bones):
    return any(not bone.ppf_bone_physics.enabled for bone in bones)


def endpoint(context, arm, bone_uuid, bone_index=None, proxy_counts=None, *, validate_pose=True):
    from ..models.bone_bindings import validate_proxy_owner, _owned_group, bone_world_frame_scale
    b=_resolve_bone(arm,bone_uuid,bone_index);s=b.ppf_bone_physics
    if not s.enabled:raise ValueError(b.name+'：端点未参与物理')
    if proxy_counts is None:
        validate_proxy_owner(arm,b)
    else:
        # Equivalent owner checks, with the duplicate-proxy scan shared by UI rows.
        from ..models.bone_bindings import _error
        if arm.library or arm.data.library:raise _error('Make the armature local before authoring bone physics')
        if len(arm.users_scene)>1:raise _error('骨架被多个场景共享；请使用独立的场景副本来编辑骨骼物理')
        if arm.data.users>1:raise _error('Duplicate armature identity; duplicated rigs need independent bindings')
        proxy=s.proxy
        if proxy is None:
            if s.body_uuid and any(o.get('_solver_uuid')==s.body_uuid for o in bpy.data.objects):raise _error('Duplicate bone or body identity; do not share a proxy between bones')
        else:
            if len(proxy.users_scene)>1:raise _error('物理代理被多个场景共享；请先创建独立副本')
            if (proxy.type!='MESH' or not proxy.get('ppf_bone_proxy_owned') or proxy.get('ppf_bone_uuid')!=s.bone_uuid or proxy.get('ppf_armature_uuid')!=arm.ppf_armature_uuid or proxy.get('_solver_uuid')!=s.body_uuid):raise _error('{name}: proxy UUID changed; update its binding',name=b.name)
            from ..models.bone_bindings import proxy_parent_armature
            owner=proxy_parent_armature(proxy)
            if owner is not None and owner!=arm:raise _error('Duplicate armature identity; duplicated rigs need independent bindings')
            if owner is None and any(o!=arm and o.type=='ARMATURE' and o.ppf_armature_uuid==arm.ppf_armature_uuid for o in bpy.data.objects):raise _error('Duplicate armature identity; duplicated rigs need independent bindings')
            if proxy_counts.get(proxy.as_pointer(),0)>1:raise _error('Duplicate bone or body identity; do not share a proxy between bones')
    if validate_pose:bone_world_frame_scale(arm,b)
    if s.proxy is None or context.scene.objects.get(s.proxy.name)!=s.proxy:
        raise ValueError(b.name+'：缺少当前场景的物理代理')
    group=_owned_group(s,'SOLID' if s.physical_model=='SOFT' else 'PDRD')
    if group is None or not group.active or not any(a.uuid==s.body_uuid and a.included for a in group.assigned_objects):
        raise ValueError(b.name+'：端点材质或参与绑定失效')
    return b,group


def status(context, arm, row, bone_index=None, proxy_counts=None):
    """Read-only diagnostic; no cleanup or binding/name mutation from UI draw."""
    try:
        a,b=_link_bones(arm,row.bone_a_uuid,row.bone_b_uuid,bone_index)
        if _link_paused((a,b)):
            return a.name+' ↔ '+b.name+'（端点物理已停用）', ''
        require_ready(context,arm,writable=False,allow_edit=True)
        a,_=endpoint(context,arm,row.bone_a_uuid,bone_index,proxy_counts,validate_pose=False);b,_=endpoint(context,arm,row.bone_b_uuid,bone_index,proxy_counts,validate_pose=False)
        return a.name+' ↔ '+b.name, ''
    except ValueError as exc:return row.bone_a_name+' ↔ '+row.bone_b_name, str(exc)


def _mass(context, arm, bone, group):
    from .bone_mass import _measure
    from .bone_mass_math import resolve_mass
    s=bone.ppf_bone_physics
    volume=_measure(context.scene,s)['volume_m3']
    density=group.solid_density if group.object_type=='SOLID' else group.pdrd_density
    return resolve_mass(s.mass_mode,float(density),float(s.target_mass_kg),volume)[1]


def _fixed(bone):
    from ..models.bone_bindings import has_physical_joint_parent
    s=bone.ppf_bone_physics
    # A non-root's root_mode is dormant, not evidence that it is constrained.
    return not has_physical_joint_parent(bone) and s.root_mode in {'STATIC','KINEMATIC'}


def defaults(context, arm, a, b, masses):
    def mass(bone):
        uid=bone.ppf_bone_physics.bone_uuid
        if uid not in masses:
            resolved,group=endpoint(context,arm,uid);masses[uid]=_mass(context,arm,resolved,group)
        return masses[uid]
    ma,mb=mass(a),mass(b)
    fixed_a,fixed_b=_fixed(a),_fixed(b)
    # Preserve fixed-to-fixed edges (common at pinned cloth tops). Their
    # finite reduced-mass coefficients remain meaningful if later unlocked.
    effective=(mb if fixed_a else ma) if fixed_a!=fixed_b else ma*mb/(ma+mb)
    omega=2*math.pi*2.
    return effective*omega*omega, 2*.7*effective*omega


def _chains(arm, names, mode):
    names=set(names);roots=[b for b in arm.data.bones if b.name in names and (b.parent is None or b.parent.name not in names)]
    if any(other in root.parent_recursive for root in roots for other in roots if other!=root):
        raise ValueError('自动连接需要连续骨链；请勿跳过同一链的中间骨骼')
    chains=[]
    for root in roots:
        chain=[];b=root
        while b is not None:
            chain.append(b)
            children=[c for c in b.children if c.name in names]
            if len(children)>1:raise ValueError('自动跨链连接不接受分叉；请选择独立无分叉骨链')
            b=children[0] if children else None
        chains.append(chain)
    if sum(map(len,chains))!=len(names):raise ValueError('选中骨链结构不完整')
    if len(chains)<(2 if mode=='PLANE' else 3):raise ValueError('平面至少两链，筒形至少三链')
    if len({len(c) for c in chains})!=1:raise ValueError('自动连接要求各链选中骨数相同，避免静默丢失端点')
    import numpy as np
    centers=np.asarray([list(sum((arm.matrix_world@arm.pose.bones[b.name].center for b in c),Vector())/len(c)) for c in chains])
    centered=centers-centers.mean(axis=0);_u,sv,vt=np.linalg.svd(centered,full_matrices=True)
    if not len(sv) or sv[0]<1.e-8:raise ValueError('骨链中心重合，无法确定相邻链顺序')
    axis=vt[0];axis=axis if axis[np.argmax(abs(axis))]>=0 else -axis
    if mode=='PLANE':keys=centered@axis
    else:
        if len(sv)<2 or sv[1]<1.e-8:raise ValueError('筒形链中心共线，无法确定环绕顺序')
        second=vt[1];second=second if second[np.argmax(abs(second))]>=0 else -second
        keys=np.arctan2(centered@second,centered@axis)
    return [chains[i] for i in sorted(range(len(chains)),key=lambda i:(float(keys[i]),chains[i][0].name))]


def _records(arm):
    records=[]
    for row in arm.ppf_cross_links:
        record={key:list(getattr(row,key)) if (key.startswith('anchor_') or key=='preview_color') else getattr(row,key) for key in FIELDS}
        # Existing .blend records have no directional values until explicitly saved.
        for direction in ('tension','compression'):
            key=direction+'_stiffness_N_per_m'
            if not row.is_property_set(key):record[key]=row.stiffness_N_per_m
        records.append(record)
    return records


def _replace(arm, records):
    """All records were validated before this short commit; rollback on RNA error."""
    _preflight_records(records)
    old=_records(arm)
    selected={r.link_uuid:r.selected for r in arm.ppf_cross_links}
    active=active_link(arm)
    active_uuid=active.link_uuid if active else None
    if not old and active_uuid is None and records:
        active_uuid=records[0]['link_uuid']
    def write(rows):
        arm.ppf_cross_links.clear()
        for record in rows:
            record=dict(record,preview_color_custom=record.get('preview_color_custom',False),preview_color=list(record.get('preview_color',(.2,.85,1.,1.))))
            row=arm.ppf_cross_links.add()
            for key in FIELDS:setattr(row,key,record[key])
            row.selected=selected.get(row.link_uuid,False)
        matches=[i for i,row in enumerate(arm.ppf_cross_links) if active_uuid and row.link_uuid==active_uuid]
        arm.ppf_cross_link_draft.active_index=matches[0] if len(matches)==1 else -1
    try:write(records)
    except Exception:write(old);raise


def create(context, arm, bone_names, mode='PAIR', diagonals=False, use_mass_defaults=True, stiffness_N_per_m=100., damping_Ns_per_m=1., tension_enabled=True, compression_enabled=True, tension_stiffness_N_per_m=None, compression_stiffness_N_per_m=None, exclude_collision=False, preview_color_custom=False, preview_color=(.2,.85,1.,1.), exclude_direct_only=False):
    require_ready(context,arm)
    if any(type(v) is not bool for v in (diagonals,use_mass_defaults,tension_enabled,compression_enabled,exclude_collision,exclude_direct_only)):raise ValueError('连接开关必须是布尔值')
    if mode not in {'PAIR','PLANE','TUBE'}:raise ValueError('连接类型必须为 PAIR/PLANE/TUBE')
    if not isinstance(bone_names,list) or any(not isinstance(n,str) or not n for n in bone_names) or len(set(bone_names))!=len(bone_names):raise ValueError('骨骼名称列表必须唯一')
    bones=[]
    for name in bone_names:
        b=arm.data.bones.get(name)
        if b is None:raise ValueError('未知骨骼：'+str(name))
        endpoint(context,arm,b.ppf_bone_physics.bone_uuid);bones.append(b)
    if mode=='PAIR':
        if len(bones)!=2:raise ValueError('PAIR 必须选中恰好两个物理骨骼')
        pairs=[(bones[0],bones[1],False)]
    else:
        chains=_chains(arm,bone_names,mode);neighbors=list(zip(chains,chains[1:]))
        if mode=='TUBE':neighbors.append((chains[-1],chains[0]))
        pairs=[]
        for a,b in neighbors:
            pairs.extend((x,y,False) for x,y in zip(a,b))
        if diagonals:
            for a,b in neighbors:
                pairs.extend((x,y,True) for x,y in zip(a[:-1],b[1:]))
                pairs.extend((x,y,True) for x,y in zip(a[1:],b[:-1]))
    old=_records(arm);known={tuple(sorted((r['bone_a_uuid'],r['bone_b_uuid']))) for r in old};new=[];masses={};scale=armature_scale(arm)
    k=_number(stiffness_N_per_m,'刚度');c=_number(damping_Ns_per_m,'阻尼')
    for a,b,is_diagonal in pairs:
        pair=tuple(sorted((a.ppf_bone_physics.bone_uuid,b.ppf_bone_physics.bone_uuid)))
        if pair in known:continue # Idempotent creation never resets manually tuned existing links.
        pa,pb=Vector((0,a.length*.5,0)),Vector((0,b.length*.5,0))
        wa=arm.matrix_world@arm.pose.bones[a.name].matrix@pa;wb=arm.matrix_world@arm.pose.bones[b.name].matrix@pb
        rest=_number((wb-wa).length,'端点初始距离',True)
        stiffness,damping=defaults(context,arm,a,b,masses) if use_mass_defaults else (k,c)
        tk=stiffness if use_mass_defaults or tension_stiffness_N_per_m is None else _number(tension_stiffness_N_per_m,'拉伸刚度')
        ck=stiffness if use_mass_defaults or compression_stiffness_N_per_m is None else _number(compression_stiffness_N_per_m,'压缩刚度')
        new.append(dict(link_uuid=str(uuid.uuid4()),enabled=True,bone_a_uuid=a.ppf_bone_physics.bone_uuid,bone_b_uuid=b.ppf_bone_physics.bone_uuid,bone_a_name=a.name,bone_b_name=b.name,anchor_a_bone=list(pa),anchor_b_bone=list(pb),rest_length_m=rest,reference_armature_scale=scale,stiffness_N_per_m=stiffness,damping_Ns_per_m=damping,source='MASS_DEFAULT_2HZ' if use_mass_defaults else 'MANUAL',topology=mode,tension_enabled=tension_enabled,compression_enabled=compression_enabled,tension_stiffness_N_per_m=tk,compression_stiffness_N_per_m=ck,exclude_collision=exclude_collision and not (exclude_direct_only and is_diagonal),preview_color_custom=preview_color_custom,preview_color=list(preview_color)))
        known.add(pair)
    if len(old)+len(new)>MAX_LINKS:raise ValueError('跨链连接数量超过10000')
    _replace(arm,old+new)
    return [r['link_uuid'] for r in new]


def edit(context, arm, link_uuids, action='SET', **parameters):
    display_only=action=='SET' and bool(parameters) and set(parameters)<={'preview_color_custom','preview_color'}
    require_display_ready(context,arm) if display_only else require_ready(context,arm)
    if not isinstance(link_uuids,list) or not link_uuids or any(not isinstance(n,str) for n in link_uuids) or len(set(link_uuids))!=len(link_uuids):raise ValueError('请明确选择连接 UUID')
    if action not in {'SET','DELETE','RECAPTURE','DEFAULTS'}:raise ValueError('未知跨链操作')
    old=_records(arm);wanted=set(link_uuids)
    if wanted-{r['link_uuid'] for r in old}:raise ValueError('连接 UUID 不存在，整批操作已取消')
    if set(parameters)-{'enabled','stiffness_N_per_m','damping_Ns_per_m','tension_enabled','compression_enabled','tension_stiffness_N_per_m','compression_stiffness_N_per_m','exclude_collision','preview_color_custom','preview_color'}:raise ValueError('不支持的跨链参数')
    for key,v in parameters.items():
        if key in {'enabled','tension_enabled','compression_enabled','exclude_collision','preview_color_custom'}:
            if type(v) is not bool:raise ValueError('enabled必须是布尔值')
        elif key=='preview_color':
            if len(v)!=4 or any(not math.isfinite(float(x)) or not 0<=x<=1 for x in v):raise ValueError('颜色必须是0至1的RGBA')
            parameters[key]=list(v)
        else:_number(v,key)
    records=[];masses={}
    for row in old:
        if row['link_uuid'] not in wanted:records.append(row);continue
        if action=='DELETE':continue
        if action=='SET':
            row.update(parameters)
            if 'stiffness_N_per_m' in parameters:
                for direction in ('tension','compression'):
                    key=direction+'_stiffness_N_per_m'
                    if key not in parameters:row[key]=parameters['stiffness_N_per_m']
            if set(parameters)-{'enabled','preview_color_custom','preview_color'}:row['source']='MANUAL'
            if row['enabled'] or display_only:
                bones=_link_bones(arm,row['bone_a_uuid'],row['bone_b_uuid'])
                if not _link_paused(bones):
                    endpoint(context,arm,row['bone_a_uuid'],validate_pose=not display_only);endpoint(context,arm,row['bone_b_uuid'],validate_pose=not display_only)
        else:
            a,_=endpoint(context,arm,row['bone_a_uuid']);b,_=endpoint(context,arm,row['bone_b_uuid'])
            if action=='DEFAULTS':
                row['stiffness_N_per_m'],row['damping_Ns_per_m']=defaults(context,arm,a,b,masses);row['source']='MASS_DEFAULT_2HZ'
                row['tension_stiffness_N_per_m']=row['compression_stiffness_N_per_m']=row['stiffness_N_per_m']
                row['tension_enabled']=row['compression_enabled']=True
            else:
                wa=arm.matrix_world@arm.pose.bones[a.name].matrix@Vector(row['anchor_a_bone']);wb=arm.matrix_world@arm.pose.bones[b.name].matrix@Vector(row['anchor_b_bone'])
                row['rest_length_m']=_number((wb-wa).length,'端点初始距离',True);row['reference_armature_scale']=armature_scale(arm)
        records.append(row)
    _replace(arm,records)
    return len(wanted)


def describe(context,arm):
    return [dict(record,**dict(zip(('label','error'),status(context,arm,row)))) for record,row in zip(_records(arm),arm.ppf_cross_links)]


def encode(context, bindings):
    rows=[];by_bone={(r['armature_uuid'],r['bone_uuid']):r for r in bindings};seen=set()
    for arm in context.scene.objects:
        if arm.type!='ARMATURE' or not any(r.enabled for r in arm.ppf_cross_links):continue
        ready=False
        for row in arm.ppf_cross_links:
            if not row.enabled:continue
            if not row.link_uuid or row.link_uuid in seen:raise ValueError('跨链连接 UUID 缺失或重复')
            seen.add(row.link_uuid);ends=[]
            bones=_link_bones(arm,row.bone_a_uuid,row.bone_b_uuid)
            if _link_paused(bones):continue
            if not ready:
                require_ready(context,arm,writable=False)
                ready=True
            for uid,point in ((row.bone_a_uuid,row.anchor_a_bone),(row.bone_b_uuid,row.anchor_b_bone)):
                bone,_=endpoint(context,arm,uid);binding=by_bone.get((arm.ppf_armature_uuid,uid))
                if binding is None:raise ValueError(bone.name+'：跨链端点未包含在本次导出中')
                from ..models.bone_bindings import bone_world_frame_scale
                _frame,total_scale=bone_world_frame_scale(arm,bone)
                world=Matrix(binding['bone_world_matrix_blender'])@(Vector(_point(list(point)))*total_scale)
                ends.append(dict(body_uuid=binding['body_uuid'],anchor_world_blender=list(world)))
            # Solver zero-force reference is this transfer's initial pose, not
            # an old author pose. Persisted reference remains preset metadata.
            _number(row.rest_length_m,'作者参考距离',True)
            _number(row.reference_armature_scale,'参考骨架缩放',True)
            rest=_number((Vector(ends[1]['anchor_world_blender'])-Vector(ends[0]['anchor_world_blender'])).length,'模拟初始距离',True)
            record=next(r for r in _records(arm) if r['link_uuid']==row.link_uuid)
            if not record['tension_enabled'] and not record['compression_enabled']:continue
            k=_number(record['stiffness_N_per_m'],'刚度');c=_number(record['damping_Ns_per_m'],'阻尼')
            tk=_number(record['tension_stiffness_N_per_m'],'拉伸刚度')
            ck=_number(record['compression_stiffness_N_per_m'],'压缩刚度')
            if c==0 and (not record['tension_enabled'] or tk==0) and (not record['compression_enabled'] or ck==0):continue
            rows.append(dict(link_uuid=row.link_uuid,model='AXIAL_DISTANCE',endpoint_a=ends[0],endpoint_b=ends[1],rest_length_m=rest,stiffness_N_per_m=k,damping_Ns_per_m=c,tension_stiffness_N_per_m=tk,compression_stiffness_N_per_m=ck,tension_enabled=record['tension_enabled'],compression_enabled=record['compression_enabled']))
    return dict(schema=SCHEMA,links=rows) if rows else None


def collision_pairs(context, bindings):
    """Enabled authored links only; return canonical UUID pairs, no groups."""
    by_bone={(r['armature_uuid'],r['bone_uuid']):r for r in bindings}
    pairs=set()
    for arm in context.scene.objects:
        if arm.type!='ARMATURE':continue
        for row in arm.ppf_cross_links:
            if not row.enabled or not row.exclude_collision:continue
            bones=_link_bones(arm,row.bone_a_uuid,row.bone_b_uuid)
            if _link_paused(bones):continue
            ends=[]
            for uid in (row.bone_a_uuid,row.bone_b_uuid):
                endpoint(context,arm,uid)
                binding=by_bone.get((arm.ppf_armature_uuid,uid))
                if binding is None:raise ValueError('跨链碰撞排除端点未包含在本次导出中')
                ends.append(binding['body_uuid'])
            if ends[0]==ends[1]:raise ValueError('跨链碰撞排除端点不能相同')
            pairs.add(tuple(sorted(ends)))
    return [list(pair) for pair in sorted(pairs)]


def preset(context,arm):
    require_ready(context,arm,writable=False)
    records=_records(arm)
    # Capture current labels for explicit cross-rig mapping; UUIDs still own
    # same-rig resolution. Reading a preset never rewrites the live network.
    for row in records:
        for suffix in ('a','b'):
            matches=[b for b in arm.data.bones if b.ppf_bone_physics.bone_uuid==row['bone_'+suffix+'_uuid']]
            if len(matches)==1:row['bone_'+suffix+'_name']=matches[0].name
    return dict(kind=PRESET_KIND,version=1,armature_uuid=arm.ppf_armature_uuid,links=records)


def apply_preset(context,arm,document,bone_map=None):
    require_ready(context,arm)
    if (not isinstance(document,dict) or document.get('kind')!=PRESET_KIND
            or isinstance(document.get('version'),bool) or document.get('version')!=1):
        raise ValueError('不支持的跨链预设')
    records=document.get('links')
    if not isinstance(records,list) or len(records)>MAX_LINKS:raise ValueError('预设连接数量无效')
    mapping={} if bone_map is None else bone_map
    if not isinstance(mapping,dict):raise ValueError('跨骨架预设需要明确的源骨名到目标骨名映射')
    result=[];ids=set();pairs=set()
    for source in records:
        if not isinstance(source,dict) or set(source) not in (set(FIELDS),set(LEGACY_FIELDS),set(DIRECTION_FIELDS),set(AUTHOR_FIELDS)):raise ValueError('预设连接字段不完整或不受支持')
        row=dict(source)
        row.setdefault('exclude_collision',False)
        row.setdefault('preview_color_custom',False)
        row.setdefault('preview_color',[.2,.85,1.,1.])
        if set(source)==set(LEGACY_FIELDS):
            for direction in ('tension','compression'):
                row[direction+'_enabled']=True
                row[direction+'_stiffness_N_per_m']=row['stiffness_N_per_m']
        if type(row['enabled']) is not bool:raise ValueError('预设enabled必须为布尔值')
        if any(not isinstance(row[k],str) for k in ('source','topology','bone_a_uuid','bone_b_uuid','bone_a_name','bone_b_name')):raise ValueError('预设身份字段必须为文字')
        try:uuid.UUID(row['link_uuid'])
        except (ValueError,TypeError,AttributeError):raise ValueError('预设连接 UUID 无效')
        if row['link_uuid'] in ids:raise ValueError('预设连接 UUID 重复')
        ids.add(row['link_uuid'])
        for suffix in ('a','b'):
            row['anchor_'+suffix+'_bone']=_point(row['anchor_'+suffix+'_bone'])
            if document.get('armature_uuid')!=arm.ppf_armature_uuid:
                target=mapping.get(row['bone_'+suffix+'_name'])
                bone=arm.data.bones.get(target) if isinstance(target,str) else None
                if bone is None:raise ValueError('跨骨架预设缺少明确映射：'+str(row['bone_'+suffix+'_name']))
                row['bone_'+suffix+'_uuid']=bone.ppf_bone_physics.bone_uuid;row['bone_'+suffix+'_name']=bone.name
        if row['enabled']:
            bones=_link_bones(arm,row['bone_a_uuid'],row['bone_b_uuid'])
            if not _link_paused(bones):
                endpoint(context,arm,row['bone_a_uuid']);endpoint(context,arm,row['bone_b_uuid'])
        pair=tuple(sorted((row['bone_a_uuid'],row['bone_b_uuid'])))
        if pair[0]==pair[1] or pair in pairs:raise ValueError('预设端点相同或骨对重复')
        pairs.add(pair)
        for key in ('stiffness_N_per_m','damping_Ns_per_m'):_number(row[key],key)
        for key in ('rest_length_m','reference_armature_scale'):_number(row[key],key,True)
        if document.get('armature_uuid')!=arm.ppf_armature_uuid:
            row['link_uuid']=str(uuid.uuid4()) # A new rig is a new physical link, not an alias.
        result.append(row)
    _replace(arm,result)
    return len(result)


def save_preset(context,arm,path):
    document=preset(context,arm)
    Path(path).write_text(json.dumps(document,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


def load_preset(context,arm,path,bone_map=None):
    p=Path(path)
    if p.stat().st_size>8*1024*1024:raise ValueError('跨链预设超过8MB')
    return apply_preset(context,arm,json.loads(p.read_text(encoding='utf8')),bone_map)


def visible_link_indices(arm):
    """Match the list search exactly; selection and visibility are distinct."""
    query=arm.ppf_cross_link_draft.search_text.strip().casefold()
    if not query:return list(range(len(arm.ppf_cross_links)))
    names={b.ppf_bone_physics.bone_uuid:b.name for b in arm.data.bones if b.ppf_bone_physics.bone_uuid}
    return [i for i,row in enumerate(arm.ppf_cross_links)
            if query in (names.get(row.bone_a_uuid,row.bone_a_name)+' '+names.get(row.bone_b_uuid,row.bone_b_name)).casefold()]


def active_link(arm):
    index=arm.ppf_cross_link_draft.active_index
    if index<0 or index>=len(arm.ppf_cross_links):return None
    return arm.ppf_cross_links[index]


def selected_link_ids(arm):
    """Visible checked links win; hidden checks never fall back to another row."""
    rows=arm.ppf_cross_links
    visible=visible_link_indices(arm)
    checked=any(r.selected for r in rows)
    if checked:return [rows[i].link_uuid for i in visible if rows[i].selected]
    index=arm.ppf_cross_link_draft.active_index
    return [rows[index].link_uuid] if index in visible and 0<=index<len(rows) else []


def display_link(arm):
    """Choose a row for read-only display without expanding the write scope."""
    ids=selected_link_ids(arm)
    if ids:
        return next((r for r in arm.ppf_cross_links if r.link_uuid==ids[0]),None)
    return next((r for r in arm.ppf_cross_links if r.selected),None) or active_link(arm)


def actual_value(arm,field):
    row=display_link(arm)
    if row is None:return False if field.endswith('enabled') or field in {'exclude_collision','preview_color_custom'} else (0.,0.,0.,1.) if field=='preview_color' else 0.
    if field in {'tension_stiffness_N_per_m','compression_stiffness_N_per_m'} and not row.is_property_set(field):return row.stiffness_N_per_m
    return getattr(row,field)


def direct_set(context,arm,field,value):
    require_display_ready(context,arm) if field in {'preview_color_custom','preview_color'} else require_ready(context,arm)
    from ..models.bone_bindings import active_armature
    if active_armature(context)!=arm:raise ValueError('活动骨架已切换，请重新选择当前连接')
    ids=selected_link_ids(arm)
    if field=='enabled' and not value:
        return edit(context,arm,ids,'SET',enabled=False)
    # Validate every endpoint before mutation, including disabled links.
    for row in arm.ppf_cross_links:
        if row.link_uuid in ids:
            validate_pose=field not in {'preview_color_custom','preview_color'}
            bones=_link_bones(arm,row.bone_a_uuid,row.bone_b_uuid)
            if not _link_paused(bones):
                endpoint(context,arm,row.bone_a_uuid,validate_pose=validate_pose);endpoint(context,arm,row.bone_b_uuid,validate_pose=validate_pose)
    return edit(context,arm,ids,'SET',**{field:value})


def mixed_fields(arm,fields):
    wanted=set(selected_link_ids(arm));rows=[r for r in _records(arm) if r['link_uuid'] in wanted]
    return [f for f in fields if rows and any(r[f]!=rows[0][f] for r in rows[1:])]


def require_display_ready(context,arm):
    """Author display writes do not invalidate simulation/cache state."""
    require_ready(context,arm,writable=False)
    if arm.library or arm.data.library or arm.data.users!=1:
        raise ValueError('请使用本地、独立骨架数据编辑连接显示')
