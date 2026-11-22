"""Explicit physical-unit drafts over the existing restoring author contract."""
import copy,json,uuid,math
import bpy
from ..core.bone_physical_units import CHANNELS,to_display,to_si,linear_constants,resolved_channels,trusted_build
from ..core.bone_restoring_preview import angular_constants, angular_metrics, angular_input_signature, angular_mass_pending
from ..models import restoring_curves as curves
from .restoring_curve_operators import read_state,write_state

# UI refresh is local author data only. Never run Build provenance/scene hashing
# from a property callback, panel draw, or the selection timer.
_refresh_depth=0
_refresh_key=None
_AUTO_PREVIEW_LABEL='代理惯量预览'
_AUTO_MASS_PENDING='等待有效代理惯量'


def _author_key(context,bone):
    s=bone.ppf_bone_physics;author=context.scene.ppf_restoring_curves
    return (context.scene.as_pointer(),bone.as_pointer(),s.bone_uuid,s.joint_uuid,
            author.mode,author.document_json,tuple(s.angular_stiffness),tuple(s.angular_damping),
            s.anchor_stiffness,s.angular_restoring_mode,s.mmd_auto_epoch,curves.mmd_automatic_angular_channels(s))


def _mmd_auto_channels(context, bone, state):
    if state['mode']=='LEGACY' and state['document'] is not None:
        return ()
    channels=curves.mmd_automatic_angular_channels(bone.ppf_bone_physics)
    if not channels:
        return ()
    from ..models.bone_bindings import active_armature, has_physical_joint_parent
    arm=active_armature(context)
    return channels if arm is not None and has_physical_joint_parent(bone, arm=arm) else ()


def _preview_metrics(context, bone):
    from ..models.bone_bindings import active_armature
    return angular_metrics(context, active_armature(context), bone)


def refresh(context,bone):
    return _refresh(context, bone, _preview_metrics(context, bone))


def _refresh(context,bone,metrics):
    global _refresh_depth
    _refresh_depth+=1
    try:
        owner,state=_initialize(context,bone)
        s=bone.ppf_bone_physics; available=[]; labels=[]; custom=[]
        auto_channels=_mmd_auto_channels(context, bone, state)
        for index,channel in enumerate(CHANNELS):
            saved=_channel_state(state,s.joint_uuid,channel);value=None
            implicit=saved is None and channel in auto_channels
            if implicit:
                saved=curves.mmd_automatic_angular_curve(s.joint_uuid, channel, frequency_hz=curves.mmd_automatic_frequency(s))
            if not implicit and (state['mode']=='LEGACY' or saved is None or not saved['binding']['enabled']):
                value=(s.anchor_stiffness,0.) if channel=='anchor' else (s.angular_stiffness[index],s.angular_damping[index])
                label='固定 SI（旧线性）'
            elif saved['binding']['mode']=='ABSOLUTE_SI':
                value=linear_constants(saved['table'],saved['binding']['damping'])
                label='固定 SI' if value is not None else '自定义曲线'
            elif index < 3:
                value=angular_constants(metrics[index] if metrics is not None else None,saved)
                label=(_AUTO_PREVIEW_LABEL if value is not None else
                       '自动自定义曲线' if linear_constants(saved['table'],0.) is None else _AUTO_MASS_PENDING)
            else:
                from ..core.bone_physical_units import display_auto_value
                value=display_auto_value(context,s.joint_uuid,channel,saved)
                label='当前构建自动值' if value is not None else '按质量自动适配；待构建计算'
            custom.append(bool(saved and state['mode']!='LEGACY' and saved['binding']['enabled'] and saved['binding']['mode']=='ABSOLUTE_SI'))
            available.append(value is not None);labels.append(label)
            if value is not None:_set_value(owner,index,to_display(value[0],channel),to_display(value[1],channel))
        owner.physical_custom=tuple(custom[:3])
        owner.physical_available=tuple(available)
        owner.physical_channel_labels=json.dumps(labels,ensure_ascii=False)
        owner.physical_loaded_values=json.dumps(_values(owner))
        owner.physical_live=True
        owner.physical_status=''
        return owner
    finally:_refresh_depth-=1


def refresh_active():
    """Refresh author controls when the selected joint or cached mass changes."""
    global _refresh_key
    from ..core import encode_progress
    context = bpy.context
    if encode_progress.is_active() or (context.screen is not None and context.screen.is_animation_playing):
        return .3
    try:
        from ..models.bone_bindings import active_armature,active_bone
        from .bone_chain_presets import _joint_parent_status
        arm=active_armature(context);bone=active_bone(context)
        if arm is None or bone is None or context.mode not in {'OBJECT','POSE'}:
            _refresh_key=None;return .3
        s=bone.ppf_bone_physics
        if not s.enabled or not s.proxy or s.joint_mode in {'HINGE','D6'} or not _joint_parent_status(bone, arm)[0]:
            _refresh_key=None;return .3
        from ..core.bone_physical_units import display_build_key
        from ..models.bone_bindings import has_physical_joint_parent
        author_key=_author_key(context,bone)
        if (_refresh_key is not None and _refresh_key[0] == author_key
                and bone.ppf_bone_restoring_controls.physical_live
                and angular_mass_pending(context,arm,bone)):
            return .3
        key=(author_key,display_build_key(context),has_physical_joint_parent(bone, arm=arm),angular_input_signature(context,arm,bone))
        if key!=_refresh_key:
            owner=bone.ppf_bone_restoring_controls
            if (not owner.physical_live and owner.physical_draft_loaded
                    and owner.loaded_joint_uuid==s.joint_uuid and owner.loaded_bone_uuid==s.bone_uuid
                    and (any(owner.physical_force_replace)
                         or (owner.physical_loaded_values
                             and _values(owner)!=json.loads(owner.physical_loaded_values)))):
                return .3
            _refresh(context,bone,angular_metrics(context,arm,bone));_refresh_key=key
    except (ValueError,RuntimeError,ReferenceError,AttributeError,KeyError,TypeError):
        _refresh_key=None
    return .3


def direct_commit(context,bone,channel,k,c,*,replace=False,component=None):
    """Single-channel entry point used by the explicit custom mode switch."""
    count,skipped,_state=_direct_commit_many(context,bone,{channel:(k,c,component)},replace=replace)
    return count,skipped


def _direct_commit_many(context,bone,edits,*,replace=False):
    """Plan every axis for every scoped joint before one document write."""
    from .bone_chain_presets import _active_physical_bone,_targets_for_bone
    from .bone_proxy import selected_bones,_owner_animation_paths
    from .bone_proxy_editor import editor_active,editor_scope
    from . import bone_proxy as proxy
    from . import restoring_curves as author
    from . import restoring_curve_presets as presets
    arm,active=_active_physical_bone(context)
    if active!=bone:raise ValueError('活动骨骼已改变，恢复参数未写入')
    if not context.scene.is_editable or not arm.data.is_editable:
        raise ValueError('场景或骨架不可编辑')
    proxy._facade_communicator().assert_bone_body_switch_idle()
    if proxy.stale_simulation_output(context):raise ValueError('请先 Clear 已有模拟帧，再修改恢复力')
    converted={ch:(to_si(k,ch),to_si(c,ch),component) for ch,(k,c,component) in edits.items()}
    scope=editor_scope(context) if editor_active(context) else 'SELECTED'
    _arm,bones=selected_bones(context,scope=scope)
    if bone not in bones:raise ValueError('活动骨骼不在当前操作范围')
    state=read_state(context.scene)
    if state['mode']=='LEGACY' and state.get('document') is not None:
        raise ValueError('保留中的旧自定义曲线尚未启用，不能自动替换')
    targets=[];skipped=[];tables=[];bindings=[];joint_ids=set()
    paths=[p for block,label in ((context.scene,'scene'),(arm.data,'armature data'),(arm,'armature object')) for p in _owner_animation_paths(block,label)]
    for target_bone in bones:
        rows=_targets_for_bone(arm,target_bone)
        if not rows:skipped.append(target_bone.name);continue
        settings=target_bone.ppf_bone_physics
        if not settings.enabled or not settings.proxy or context.scene.objects.get(settings.proxy.name)!=settings.proxy:
            raise ValueError(target_bone.name+': 请先创建并启用物理代理')
        from ..models.bone_bindings import validate_proxy_owner
        validate_proxy_owner(arm,target_bone)
        prefix=target_bone.path_from_id()+'.ppf_bone_restoring_controls.'
        if any(p.startswith(prefix) for p in paths):raise ValueError(target_bone.name+': 恢复参数有动画，不能直接改写')
        matched=[t for t in rows if t['channel'] in converted]
        if not matched:
            skipped.append(target_bone.name)
            continue
        if len(matched)!=len(converted):
            raise ValueError(target_bone.name+': 当前关节缺少选中的恢复通道')
        for target in matched:
            channel=target['channel']
            k,c,component=converted[channel]
            saved=_channel_state(state,target['joint_uuid'],channel)
            if saved and saved['binding']['enabled'] and state['mode']!='LEGACY':
                if not replace and (saved['binding']['mode']!='ABSOLUTE_SI'
                                    or linear_constants(saved['table'],saved['binding']['damping']) is None):
                    raise ValueError(target_bone.name+': 此通道仍为自适应或自定义曲线，请先显式选择固定 SI')
            if component is not None:
                original=(settings.anchor_stiffness,0.) if channel=='anchor' else (settings.angular_stiffness[CHANNELS.index(channel)],settings.angular_damping[CHANNELS.index(channel)])
                if saved and saved['binding']['enabled'] and state['mode']!='LEGACY':
                    original=linear_constants(saved['table'],saved['binding']['damping'])
                if component=='k':c=original[1]
                else:k=original[0]
            tid='physical_'+uuid.uuid4().hex
            tables.append(dict(table_id=tid,domain='RADIAL' if channel=='anchor' else 'ODD',tail='LINEAR',nodes=[[0.,0.],[1.,k]]))
            bindings.append(dict(target=target,enabled=True,table_id=tid,mode='ABSOLUTE_SI',damping=c))
            joint_ids.add(target['joint_uuid'])
            targets.append(target)
    if not targets:raise ValueError('当前范围无可写父关节')
    doc=curves.validate_author_map(dict(version=1,tables=tables,bindings=bindings,stitch_pairs=[]))
    author.apply_preset(state,presets.named_preset('物理刚度',doc),[dict(source=t,destination=t) for t in targets],targets,[])
    owner=context.scene.ppf_restoring_curves
    raw=(owner.document_json,owner.mode,owner.requested_mode)
    try:write_state(context.scene,state)
    except Exception:
        owner.document_json,owner.mode,owner.requested_mode=raw
        raise
    return len(joint_ids),skipped,state


def field_changed(owner,context):
    global _refresh_depth,_refresh_key
    if _refresh_depth or not owner.physical_live or not owner.physical_loaded_values:return
    old=json.loads(owner.physical_loaded_values);values=_values(owner)
    if values==old:return
    _refresh_depth+=1
    try:
        from .bone_chain_presets import _active_physical_bone
        _arm,bone=_active_physical_bone(context)
        if bone.ppf_bone_restoring_controls!=owner or owner.loaded_joint_uuid!=bone.ppf_bone_physics.joint_uuid:
            raise ValueError('活动关节已改变，未写入恢复力')
        changed=[i for i in range(4) if values[i]!=old[i]]
        snapshot=json.loads(owner.physical_channel_snapshot or '{}')
        current=read_state(context.scene)
        if current['mode']!=snapshot.get('__author_mode__',current['mode']) or any(
                _channel_state(current,owner.loaded_joint_uuid,CHANNELS[i])!=snapshot.get(CHANNELS[i])
                for i in changed):
            raise ValueError('作者通道已改变，已恢复当前值')
        settings=bone.ppf_bone_physics
        if current['mode']=='LEGACY' and snapshot.get('__legacy_values__')!=[list(settings.angular_stiffness),list(settings.angular_damping),settings.anchor_stiffness]:
            raise ValueError('旧线性作者参数已改变，请等待刷新')
        edits={}
        for i in changed:
            if not owner.physical_available[i] or (i<3 and not owner.physical_custom[i]):
                raise ValueError('请先开启该轴自定义恢复力')
            k,c=values[i];old_k,old_c=old[i]
            edits[CHANNELS[i]]=(k,c,'k' if c==old_c else 'c' if k==old_k else None)
        # Blender's native multi-number gesture stages values before RNA update callbacks.
        # Commit all changed components once, without rewriting the vector inside a callback.
        count,skipped,state=_direct_commit_many(context,bone,edits)
        for channel in edits:
            snapshot[channel]=_channel_state(state,owner.loaded_joint_uuid,channel)
        snapshot['__author_mode__']=state['mode']
        owner.physical_channel_snapshot=json.dumps(snapshot,sort_keys=True)
        owner.physical_loaded_values=json.dumps(values)
        owner.physical_status=f'已修改 {count} 个父关节'+('；跳过无父关节：'+', '.join(skipped) if skipped else '')
        _refresh_key=None
    except Exception as error:
        for i,(k,c) in enumerate(old):_set_value(owner,i,k,c)
        owner.physical_status='未写入：'+str(error)
    finally:_refresh_depth-=1


def _channel_state(state,joint,channel):
    doc=state.get('document')
    if doc is None:return None
    key=curves.target_key(dict(kind='JOINT',joint_uuid=joint,channel=channel))
    rows=[b for b in doc['bindings'] if curves.target_key(b['target'])==key]
    if len(rows)>1:raise ValueError('当前恢复通道重复，不能载入物理刚度')
    if not rows:return None
    b=rows[0];t=next(t for t in doc['tables'] if t['table_id']==b['table_id'])
    return dict(binding=copy.deepcopy(b),table=copy.deepcopy(t))


def _values(owner):
    return [[float(owner.angular_stiffness_degree[i]),float(owner.angular_damping_degree[i])] for i in range(3)]+[[float(owner.linear_stiffness_cm),float(owner.linear_damping_cm)]]


def _set_value(owner,index,k,c):
    if index==3:owner.linear_stiffness_cm=k;owner.linear_damping_cm=c
    else:owner.angular_stiffness_degree[index]=k;owner.angular_damping_degree[index]=c


def _last_build(context):
    # One authoritative pair of sealed raw files per explicit action.
    return trusted_build(context)


def _initialize(context,bone):
    owner=bone.ppf_bone_restoring_controls;settings=bone.ppf_bone_physics
    owner.physical_live=False
    state=read_state(context.scene)
    owner.loaded_bone_uuid=settings.bone_uuid;owner.loaded_joint_uuid=settings.joint_uuid
    owner.draft_loaded=True;owner.physical_draft_loaded=True
    owner.physical_available=(False,)*4;owner.physical_force_replace=(False,)*4
    owner.apply_anchor=False
    snapshots={ch:_channel_state(state,settings.joint_uuid,ch) for ch in CHANNELS}
    snapshots['__author_mode__']=state['mode']
    snapshots['__legacy_values__']=[list(settings.angular_stiffness),list(settings.angular_damping),settings.anchor_stiffness]
    owner.physical_channel_snapshot=json.dumps(snapshots,sort_keys=True)
    return owner,state


def load(context,bone):
    # A failed provenance check must preserve existing unapplied author drafts.
    from .restoring_curve_operators import snapshot_rna,restore_rna
    owner=bone.ppf_bone_restoring_controls;snapshot=snapshot_rna(owner)
    old_values=_values(owner)
    old_loaded=owner.physical_loaded_values
    s=bone.ppf_bone_physics
    dirty=[]
    if owner.physical_draft_loaded and owner.loaded_bone_uuid==s.bone_uuid and owner.loaded_joint_uuid==s.joint_uuid:
        baseline=json.loads(old_loaded) if old_loaded else old_values
        dirty=[i for i in range(4) if owner.physical_force_replace[i] or old_values[i]!=baseline[i]]
    try:
        result=_load(context,bone)
        if any(not result.physical_available[i] for i in dirty):
            restore_rna(owner,snapshot)
            owner.physical_status='部分数值不可读取；保留未应用草稿'
        return owner
    except Exception:
        restore_rna(owner,snapshot)
        raise


def _load(context,bone):
    owner,state=_initialize(context,bone);s=bone.ppf_bone_physics
    last=None;snapshot=None;source=None;sources={};preview_sources={};attempted=False;available=[];has_custom=False;used_build=False;pending_auto=False
    auto_channels=_mmd_auto_channels(context, bone, state)
    metrics=_preview_metrics(context,bone)
    for index,channel in enumerate(CHANNELS):
        saved=_channel_state(state,s.joint_uuid,channel)
        value=None
        implicit=saved is None and channel in auto_channels
        if implicit:
            saved=curves.mmd_automatic_angular_curve(s.joint_uuid, channel, frequency_hz=curves.mmd_automatic_frequency(s))
        if not implicit and (state['mode']=='LEGACY' or saved is None or not saved['binding']['enabled']):
            value=(s.anchor_stiffness,0.) if channel=='anchor' else (s.angular_stiffness[index],s.angular_damping[index])
        elif saved['binding']['mode']=='ABSOLUTE_SI':
            value=linear_constants(saved['table'],saved['binding']['damping'])
            has_custom=has_custom or value is None
        else:
            # Enabled exact-zero AUTO curves are already known without a Build.
            if all(float(f)==0. for _x,f in saved['table']['nodes']) and saved['binding']['damping_ratio']==0:
                value=(0.,0.)
            elif linear_constants(saved['table'],0.) is None:
                has_custom=True
            elif index < 3 and metrics is not None:
                value=angular_constants(metrics[index],saved)
                preview_sources[channel]=metrics[index]
            else:
                if not attempted:
                    attempted=True
                    try:
                        snapshot,source=_last_build(context)
                        last=resolved_channels(snapshot,s.joint_uuid)
                    except (ImportError,ValueError,OSError,TypeError,KeyError):
                        last=None
                if implicit:
                    from ..core.bone_physical_units import resolved_auto_value
                    value=resolved_auto_value(snapshot,s.joint_uuid,channel,saved)
                else:
                    value=(last or {}).get(channel)
                if value is not None:sources[channel]=source
                used_build=used_build or value is not None
                pending_auto=pending_auto or value is None
        available.append(value is not None)
        if value is not None:_set_value(owner,index,to_display(value[0],channel),to_display(value[1],channel))
    snapshots=json.loads(owner.physical_channel_snapshot)
    snapshots['__auto_resolved_sources__']=sources
    snapshots['__auto_preview_sources__']=preview_sources
    owner.physical_channel_snapshot=json.dumps(snapshots,sort_keys=True)
    owner.physical_available=tuple(available)
    owner.physical_channel_labels='[]'
    owner.physical_loaded_values=json.dumps(_values(owner))
    owner.physical_status=('已载入代理惯量预览' if preview_sources else '部分自动值等待有效质量或构建' if pending_auto else '保留自定义曲线' if has_custom else '上次构建实际值' if used_build else '已载入物理参数')
    return owner


def new_draft(context,bone,channel):
    owner=bone.ppf_bone_restoring_controls;s=bone.ppf_bone_physics
    valid=owner.physical_draft_loaded and owner.loaded_bone_uuid==s.bone_uuid and owner.loaded_joint_uuid==s.joint_uuid
    if not valid:owner,_state=_initialize(context,bone)
    owner.physical_live=False
    baseline=json.loads(owner.physical_loaded_values) if valid and owner.physical_loaded_values else _values(owner)
    state=read_state(context.scene);index=CHANNELS.index(channel)
    snapshots=json.loads(owner.physical_channel_snapshot or '{}')
    snapshots[channel]=_channel_state(state,s.joint_uuid,channel)
    snapshots.get('__auto_resolved_sources__',{}).pop(channel,None)
    snapshots.get('__auto_preview_sources__',{}).pop(channel,None)
    owner.physical_channel_snapshot=json.dumps(snapshots,sort_keys=True)
    # Explicit replacement draft, not a claim about active AUTO/custom values.
    k,c=(s.anchor_stiffness,0.) if channel=='anchor' else (s.angular_stiffness[index],s.angular_damping[index])
    _set_value(owner,index,to_display(k,channel),to_display(c,channel))
    owner.physical_available[index]=True;owner.physical_force_replace[index]=True
    # Preserve unapplied edits and baselines on every other axis.
    baseline[index]=_values(owner)[index]
    owner.physical_loaded_values=json.dumps(baseline)
    owner.physical_status='固定物理刚度草稿'
    return owner


def apply(context,bone):
    owner=bone.ppf_bone_restoring_controls;s=bone.ppf_bone_physics
    if not (owner.physical_draft_loaded and owner.loaded_bone_uuid==s.bone_uuid and owner.loaded_joint_uuid==s.joint_uuid):
        raise ValueError('先载入数值或为所选轴创建固定刚度草稿')
    values=_values(owner);loaded=json.loads(owner.physical_loaded_values)
    snapshots=json.loads(owner.physical_channel_snapshot);state=read_state(context.scene)
    before_mode=state['mode']
    channel_modes={ch:snapshots.get('__channel_author_modes__',{}).get(ch,snapshots.get('__author_mode__')) for ch in CHANNELS}
    changed=[ch for i,ch in enumerate(CHANNELS) if owner.physical_available[i] and (i!=3 or owner.apply_anchor)
             and (owner.physical_force_replace[i] or values[i]!=loaded[i])]
    if not changed:return False
    # Draft Apply commits the same author curves as direct fixed controls.
    # Reject busy or retained-output state before any author/draft mutation.
    from . import bone_proxy as proxy
    proxy._facade_communicator().assert_bone_body_switch_idle()
    if proxy.stale_simulation_output(context):raise ValueError('请先 Clear 已有模拟帧，再修改恢复力')
    if any(channel_modes[ch] not in {'LEGACY','COMMON','ADVANCED'} for ch in changed):
        raise ValueError('恢复力草稿缺少有效模式来源，请重新载入')
    for channel in changed:
        saved=_channel_state(state,s.joint_uuid,channel)
        if saved!=snapshots.get(channel):
            raise ValueError('该通道在载入后已改变，请重新载入')
        index=CHANNELS.index(channel)
        if owner.physical_force_replace[index]:continue
        if state['mode']!=channel_modes[channel]:
            raise ValueError('恢复力模式在载入后已改变，请重新载入')
        if state['mode']=='LEGACY' or saved is None or not saved['binding']['enabled']:
            # A missing/disabled curve still derives both draft components
            # from legacy author values. Never publish a stale other half.
            legacy=snapshots.get('__legacy_values__')
            current=(s.anchor_stiffness,0.) if channel=='anchor' else (s.angular_stiffness[index],s.angular_damping[index])
            previous=(legacy[2],0.) if legacy and channel=='anchor' else (legacy[0][index],legacy[1][index]) if legacy else None
            if current!=previous:
                raise ValueError('该通道的旧线性作者参数已改变，请重新载入')
    sources=snapshots.get('__auto_resolved_sources__',{})
    preview_sources=snapshots.get('__auto_preview_sources__',{})
    current_metrics=_preview_metrics(context,bone) if preview_sources else None
    derived=[]
    auto_channels=_mmd_auto_channels(context, bone, state)
    for channel in changed:
        if owner.physical_force_replace[CHANNELS.index(channel)]:continue
        saved=_channel_state(state,s.joint_uuid,channel)
        dependent=(state['mode']!='LEGACY' and saved is not None and saved['binding']['enabled']
                   and saved['binding']['mode']=='AUTO_FREQUENCY'
                   and not (all(float(f)==0. for _x,f in saved['table']['nodes'])
                            and saved['binding']['damping_ratio']==0))
        if dependent or (saved is None and channel in auto_channels):
            if channel in preview_sources:
                if current_metrics is None or not math.isclose(
                        current_metrics[CHANNELS.index(channel)],preview_sources[channel],rel_tol=1e-6):
                    raise ValueError('代理惯量在载入后已改变，请重新载入数值')
                continue
            if channel not in sources:
                raise ValueError('旧恢复力草稿缺少当前构建来源，请重新载入数值')
            derived.append(channel)
    if derived:
        try:
            document,current=_last_build(context)
            if any(sources[channel]!=current for channel in derived):
                raise ValueError('恢复力数值来源已改变，请重新构建并载入数值')
        except (ValueError,OSError,TypeError,KeyError):
            owner.physical_status='数值已过期，请重新传输、构建并载入'
            raise
    if state['mode']=='LEGACY' and state.get('document') is not None:
        raise ValueError('保留中的旧自定义曲线尚未启用，不能自动激活或替换')
    from . import restoring_curves as author
    from . import restoring_curve_presets as presets
    tables=[];bindings=[];targets=[]
    for channel in changed:
        i=CHANNELS.index(channel);k,c=values[i];scale=.01 if channel=='anchor' else math.pi/180.
        target=dict(kind='JOINT',joint_uuid=s.joint_uuid,channel=channel);table_id='physical_'+uuid.uuid4().hex
        # x is real metres/radians; at1cm/1degree the node effort is exactly
        # the displayed stiffness (angular N·mm converts /1000). Zero remains enabled, avoiding legacy fallback.
        tables.append(dict(table_id=table_id,domain='RADIAL' if channel=='anchor' else 'ODD',tail='LINEAR',nodes=[[0.,0.],[scale,float(k) if channel=='anchor' else float(k)/1000.]]))
        bindings.append(dict(target=target,enabled=True,table_id=table_id,mode='ABSOLUTE_SI',damping=to_si(c,channel)))
        targets.append(target)
    doc=curves.validate_author_map(dict(version=1,tables=tables,bindings=bindings,stitch_pairs=[]))
    preset=presets.named_preset('物理刚度',doc)
    author.apply_preset(state,preset,[dict(source=t,destination=t) for t in targets],targets,[])
    author_owner=context.scene.ppf_restoring_curves
    raw=(author_owner.document_json,author_owner.mode,author_owner.requested_mode)
    try:write_state(context.scene,state)
    except Exception:
        author_owner.document_json,author_owner.mode,author_owner.requested_mode=raw
        raise
    # No post-commit file/network IO: an unreadable old Build must never turn
    # a successful author write into an operator cancellation.
    for channel in changed:
        index=CHANNELS.index(channel)
        loaded[index]=values[index]
        snapshots[channel]=_channel_state(state,s.joint_uuid,channel)
        owner.physical_force_replace[index]=False
        sources.pop(channel,None)
        preview_sources.pop(channel,None)
    # A forced replacement after an external mode change refreshes only its
    # own channels. Do not legitimize untouched drafts from the old mode.
    for channel in CHANNELS:
        if channel in changed or (before_mode!=state['mode'] and channel_modes[channel]==before_mode):
            channel_modes[channel]=state['mode']
    snapshots['__channel_author_modes__']=channel_modes
    if snapshots.get('__author_mode__',before_mode)==before_mode:
        snapshots['__author_mode__']=state['mode']
    owner.physical_loaded_values=json.dumps(loaded)
    owner.physical_channel_snapshot=json.dumps(snapshots,sort_keys=True)
    owner.physical_status='已应用改动通道'
    if 'anchor' in changed:owner.apply_anchor=False
    return True


def custom_changed(owner,context):
    """Explicit mode change; callbacks report failures without RNA exceptions."""
    global _refresh_depth,_refresh_key
    if _refresh_depth or not owner.physical_live:return
    from .bone_chain_presets import _active_physical_bone,_targets_for_bone,_apply_targets,strength_to_frequency
    try:
        arm,bone=_active_physical_bone(context)
        state=read_state(context.scene)
        actual=[bool((v:=_channel_state(state,bone.ppf_bone_physics.joint_uuid,ch)) and state['mode']!='LEGACY' and v['binding']['enabled'] and v['binding']['mode']=='ABSOLUTE_SI') for ch in CHANNELS[:3]]
        changed=[i for i in range(3) if owner.physical_custom[i]!=actual[i]]
        if not changed:return
        if len(changed)!=1:raise ValueError('请一次切换一个恢复通道')
        i=changed[0];channel=CHANNELS[i]
        if owner.physical_custom[i]:
            # Store author intent, not a resolved inertia-dependent number.
            from .bone_proxy import selected_bones
            from .bone_proxy_editor import editor_scope,editor_active
            _arm,scoped=selected_bones(context,scope=editor_scope(context) if editor_active(context) else 'SELECTED')
            backups=[]
            for target_bone in scoped:
                if not _targets_for_bone(arm,target_bone):continue
                control=target_bone.ppf_bone_restoring_controls
                backup=json.loads(control.physical_auto_backup or '{}')
                backup[channel]=_channel_state(state,target_bone.ppf_bone_physics.joint_uuid,channel)
                backups.append((control,json.dumps(backup)))
            k,c=_values(owner)[i] if owner.physical_available[i] else (0.,0.)
            direct_commit(context,bone,channel,k,c,replace=True)
            for control,backup in backups:control.physical_auto_backup=backup
        else:
            from .bone_proxy import selected_bones
            from .bone_proxy_editor import editor_scope,editor_active
            from . import bone_proxy as proxy
            proxy._facade_communicator().assert_bone_body_switch_idle()
            if proxy.stale_simulation_output(context):raise ValueError('请先清除已有模拟帧，再修改恢复力')
            _arm,bones=selected_bones(context,scope=editor_scope(context) if editor_active(context) else 'SELECTED')
            targets=[]
            for b in bones:targets.extend(t for t in _targets_for_bone(arm,b) if t['channel']==channel)
            if not targets:raise ValueError('当前范围无可写父关节')
            # Re-enable the authored AUTO curve where available. For a channel
            # originally authored as fixed SI, use its retained normalized control.
            from . import restoring_curves as author
            from . import restoring_curve_presets as presets
            tables=[];bindings=[]
            for b in bones:
                rows=[t for t in _targets_for_bone(arm,b) if t['channel']==channel]
                if not rows:continue
                control=b.ppf_bone_restoring_controls
                saved=json.loads(control.physical_auto_backup or '{}').get(channel)
                if saved and saved['binding']['mode']=='AUTO_FREQUENCY':
                    table=copy.deepcopy(saved['table']);binding=copy.deepcopy(saved['binding'])
                    table['table_id']='auto_'+uuid.uuid4().hex;binding['table_id']=table['table_id'];binding['target']=rows[0]
                    tables.append(table);bindings.append(binding)
                else:
                    from .bone_chain_presets import _linear_frequency_document
                    field='twist_strength' if i==0 else 'bend_strength'
                    strength=getattr(control,field)
                    settings=b.ppf_bone_physics
                    if (settings.joint_control_mode == 'NATIVE' and settings.proxy
                            and not settings.proxy.get('ppf_mmd_source')
                            and not control.is_property_set(field)):
                        from .restoring_curves import NEW_NATIVE_ANGULAR_HZ
                        strength=(NEW_NATIVE_ANGULAR_HZ[channel]/10.)**2
                    doc=_linear_frequency_document(rows,{channel:dict(frequency_hz=strength_to_frequency(strength),damping_ratio=control.damping_ratio,enabled=True,zero_effort=strength==0)})
                    tables.extend(doc['tables']);bindings.extend(doc['bindings'])
            doc=curves.validate_author_map(dict(version=1,tables=tables,bindings=bindings,stitch_pairs=[]))
            author.apply_preset(state,presets.named_preset('自动恢复力',doc),[dict(source=t,destination=t) for t in targets],targets,[])
            write_state(context.scene,state)
        refresh(context,bone);_refresh_key=None
    except Exception as error:
        _refresh_depth+=1
        try:
            if 'actual' in locals():owner.physical_custom=actual
            owner.physical_status='未写入：'+str(error)
        finally:_refresh_depth-=1


def _format_auto_angular(value):
    """Expose small nonzero estimates that a fixed precision RNA widget hides."""
    return format(float(value), '.6g')


def draw_inline(layout, context, bone, owner, physical_loaded, *, show_swing_mode=True):
    """Draw restoring parameters in their original locations."""
    settings = bone.ppf_bone_physics
    if show_swing_mode:
        layout.prop(settings, 'angular_restoring_mode', text='摆动恢复')
    cone = show_swing_mode and settings.angular_restoring_mode == 'CONE'
    axes = ('X扭转',) if cone else ('X扭转', 'Y弯曲', 'Z弯曲')
    labels=json.loads(owner.physical_channel_labels or '[]') if physical_loaded else []
    def unavailable(index):
        if not physical_loaded:
            return '等待选择父关节'
        return labels[index] if index < len(labels) else _AUTO_MASS_PENDING

    for index, label in enumerate(axes):
        row=layout.row(align=True);row.label(text=label)
        switch=row.row(align=True);switch.enabled=physical_loaded
        switch.prop(owner,'physical_custom',index=index,text='自定义')
        value=row.row(align=True);value.enabled=physical_loaded and owner.physical_custom[index] and owner.physical_available[index]
        if physical_loaded and owner.physical_available[index]:
            if index < len(labels) and labels[index] == _AUTO_PREVIEW_LABEL:
                value.enabled=True;value.label(text=_format_auto_angular(owner.angular_stiffness_degree[index])+' N·mm/°')
            else:
                value.prop(owner,'angular_stiffness_degree',index=index,text='')
        else:value.label(text=unavailable(index))
    if cone:
        layout.prop(settings, 'swing_restoring_stiffness_degree', text='圆锥刚度 N·mm/°')
    header,body=layout.panel('ppf_bone_physical_damping',default_closed=True)
    header.label(text='角向阻尼 N·mm·s/°')
    if body is not None:
        for index, label in enumerate(axes):
            row=body.row(align=True);row.enabled=physical_loaded and owner.physical_custom[index] and owner.physical_available[index]
            if physical_loaded and owner.physical_available[index]:
                if index < len(labels) and labels[index] == _AUTO_PREVIEW_LABEL:
                    row.enabled=True;row.label(text=label+'：'+_format_auto_angular(owner.angular_damping_degree[index]))
                else:
                    row.prop(owner,'angular_damping_degree',index=index,text=label)
            else:row.label(text=label+'：'+unavailable(index))
        if cone:
            body.prop(settings, 'swing_restoring_damping_degree', text='圆锥阻尼')
