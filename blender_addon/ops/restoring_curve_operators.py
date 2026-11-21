"""Explicit Undoable RNA author/editor/preset operations, separate from Build."""
from .. import i18n as _ppf_i18n
import copy
import hashlib
import json
from functools import lru_cache
import bpy
from ..i18n import pgettext_iface as iface
from ..i18n.props import BoolProperty,EnumProperty
from ..models import restoring_curves as model
from . import restoring_curves as author
from . import restoring_curve_presets as presets

PANEL_SCOPES=[('AUTO','Direct API','Use the explicit raw channel'),('BONE','Bone Panel','Current joint and remembered joint channel'),('STITCH','Stitch Panel','Current merge pair only')]


def _read_document(raw):
    return model.validate_author_map(json.loads(raw, object_pairs_hook=presets._object_pairs))


def read_state(scene):
    owner = scene.ppf_restoring_curves
    raw = owner.document_json
    return {'mode': owner.mode, 'document': _read_document(raw) if raw else None}


@lru_cache(maxsize=4)
def _anchor_curve_index(raw):
    enabled = set()
    seen = set()
    ambiguous = set()
    for binding in _read_document(raw)['bindings']:
        target = binding['target']
        if target['kind'] != 'JOINT' or target['channel'] != 'anchor':
            continue
        joint_uuid = target['joint_uuid']
        if joint_uuid in seen:
            ambiguous.add(joint_uuid)
        seen.add(joint_uuid)
        if binding['enabled']:
            enabled.add(joint_uuid)
    return frozenset(enabled), frozenset(ambiguous)


def anchor_curve_index(scene):
    """Read immutable display data keyed by the current document contents."""
    owner = scene.ppf_restoring_curves
    if owner.mode == 'LEGACY':
        return None
    raw = owner.document_json
    return _anchor_curve_index(raw) if raw else None


def write_state(scene,state):
    owner=scene.ppf_restoring_curves
    raw=model._json_bytes(state['document']).decode('utf-8') if state['document'] is not None else ''
    # All validation precedes these two RNA writes. Blender operator Undo owns
    # both fields; no property update callback can emit partially applied data.
    owner.document_json=raw;owner.mode=state['mode'];owner.requested_mode=state['mode']


def retire_joint_or_pair(state, *, joint_ids=(), pair_ids=()):
    """Explicit deletion candidate; no export/load cleanup."""
    if state['document'] is None:return state
    doc=model.validate_author_map(state['document'])
    doc['bindings']=[b for b in doc['bindings'] if not (
        b['target']['kind']=='JOINT' and b['target']['joint_uuid'] in joint_ids or
        b['target']['kind']=='STITCH_PAIR' and b['target']['pair_id'] in pair_ids)]
    used={b['table_id'] for b in doc['bindings']}
    pairs={b['target']['pair_id'] for b in doc['bindings'] if b['target']['kind']=='STITCH_PAIR'}
    doc['tables']=[t for t in doc['tables'] if t['table_id'] in used]
    doc['stitch_pairs']=[p for p in doc['stitch_pairs'] if p['pair_id'] in pairs]
    return {'mode':state['mode'],'document':model.validate_author_map(doc)}


def prepare_bone_copy(context,active,bones):
    """Validate/remap complete curve candidate before the existing batch writes."""
    from ..models.bone_bindings import validate_proxy_owner
    arm=context.object
    state=read_state(context.scene)
    if state['document'] is None:return None
    source_id=active.ppf_bone_physics.joint_uuid
    bindings=[b for b in state['document']['bindings'] if b['target']['kind']=='JOINT' and b['target']['joint_uuid']==source_id]
    if not bindings:return None
    for bone in [active,*bones]:validate_proxy_owner(arm,bone)
    original_mode=state['mode']
    used={b['table_id'] for b in bindings}
    document={'version':1,'bindings':bindings,'tables':[t for t in state['document']['tables'] if t['table_id'] in used],'stitch_pairs':[]}
    preset=presets.named_preset('Copied bone settings',document)
    for bone in bones:
        s=bone.ppf_bone_physics
        if (not s.enabled or not s.proxy or not bone.parent
                or not (bone.parent.ppf_bone_physics.enabled
                        or getattr(s, 'auto_follow_parent', True) and s.root_mode == 'FREE')):
            raise ValueError(iface('{name}: create its body and parent joint before copying curves').format(name=bone.name))
        if s.joint_mode in {'HINGE','D6'}:raise ValueError(iface('{name}: curves require a compliant joint').format(name=bone.name))
        model.canonical_uuid(s.joint_uuid,'joint_uuid')
        state=retire_joint_or_pair(state,joint_ids=[s.joint_uuid])
        remap=[];selected=[]
        for binding in bindings:
            old=binding['target'];new=dict(old,joint_uuid=s.joint_uuid)
            remap.append({'source':old,'destination':new});selected.append(new)
        author.apply_preset(state,preset,remap,selected)
    if original_mode in ('LEGACY','ADVANCED'):state['mode']=original_mode
    return state


def snapshot_rna(owner):
    result={}
    for prop in owner.bl_rna.properties:
        key=prop.identifier
        # Collection properties cannot be assigned, but their mutable items
        # must be included in draft/creation rollback snapshots.
        if key in ('rna_type','soft_stiffness_degree','swing_cone_stiffness_degree','swing_restoring_stiffness_degree','swing_restoring_damping_degree','solid_young_mpa') or (prop.is_readonly and prop.type!='COLLECTION'):continue
        if key=='pin_vertex_group_items':
            from ..core.pin_picker_identity import snapshot_metadata
            result[key]=(key in owner,('PIN_PICKER_METADATA',snapshot_metadata(owner)))
            continue
        value=getattr(owner,key)
        if prop.type=='COLLECTION':value=('COLLECTION',[snapshot_rna(item) for item in value])
        elif prop.type=='POINTER' and isinstance(value,bpy.types.PropertyGroup):value=('GROUP',snapshot_rna(value))
        else:value=('VALUE',tuple(value) if getattr(prop,'is_array',False) else value)
        result[key]=(owner.is_property_set(key),value)
    return result


def restore_rna(owner,snapshot):
    from ..models.restoring_curve_rna import PPF_PG_RestoringCurves
    items = snapshot.items()
    if (isinstance(owner, PPF_PG_RestoringCurves)
            and 'ui_tier' in snapshot and 'advanced_ui' in snapshot):
        # Restoring ui_tier invokes its compatibility callback. Restore the
        # saved display flag last, including its original explicit/unset state.
        items = [(key, value) for key, value in items if key != 'advanced_ui']
        items.append(('advanced_ui', snapshot['advanced_ui']))
    for key,(was_set,(kind,value)) in items:
        if kind=='PIN_PICKER_METADATA':
            from ..core.pin_picker_identity import restore_metadata
            restore_metadata(owner,value)
        elif kind=='COLLECTION':
            collection=getattr(owner,key);collection.clear()
            for item in value:restore_rna(collection.add(),item)
        elif kind=='GROUP':restore_rna(getattr(owner,key),value)
        else:
            setattr(owner,key,value)
            if not was_set:owner.property_unset(key)


def live_target(context, *, create_pair_id=False, scope='AUTO'):
    from ..models.bone_bindings import active_armature,active_bone,validate_proxy_owner
    from ..models.groups import get_addon_data
    owner=context.scene.ppf_restoring_curves
    if scope not in ('AUTO','BONE','STITCH'):raise ValueError('Invalid restoring panel scope')
    channel='stitch' if scope=='STITCH' else owner.joint_channel if scope=='BONE' else owner.channel
    if channel=='stitch':
        state=get_addon_data(context.scene).state;index=state.merge_pairs_index
        if not 0<=index<len(state.merge_pairs):raise ValueError(iface('Select a merge pair first'))
        pair=state.merge_pairs[index]
        if not pair.pair_id:
            if not create_pair_id:raise ValueError(iface('Load the current pair into the curve editor first'))
            pair.pair_id=model.pair_identity()
        model.canonical_uuid(pair.pair_id,'pair_id')
        return {'kind':'STITCH_PAIR','pair_id':pair.pair_id}
    arm=active_armature(context);bone=active_bone(context)
    if not arm or not bone:raise ValueError(iface('Select a physical bone first'))
    s=bone.ppf_bone_physics
    if (not s.enabled or not s.proxy or not bone.parent
            or not (bone.parent.ppf_bone_physics.enabled
                    or getattr(s, 'auto_follow_parent', True) and s.root_mode == 'FREE')):
        raise ValueError(iface('This curve needs a created body and physical parent joint'))
    validate_proxy_owner(arm,bone)
    if s.joint_mode in {'HINGE','D6'}:raise ValueError(iface('Restoring curves require a compliant joint'))
    model.canonical_uuid(s.joint_uuid,'joint_uuid')
    return {'kind':'JOINT','joint_uuid':s.joint_uuid,'channel':channel}


def draft_matches(context,scope):
    """Read-only panel check; draw never changes selection, drafts or author."""
    owner=context.scene.ppf_restoring_curves
    try:return owner.editor_target_json==model._json_bytes(live_target(context,scope=scope)).decode('utf-8')
    except ValueError:return False


def draft_label(context):
    owner=context.scene.ppf_restoring_curves
    try:target=json.loads(owner.editor_target_json)
    except (ValueError,TypeError):return iface('No current curve draft')
    if not isinstance(target,dict):return iface('Curve draft target is no longer available')
    if target.get('kind')=='JOINT':
        captions={'twist_x':'Twist X','swing_y':'Swing Y','swing_z':'Swing Z','anchor':'Bone Anchor'}
        if target.get('channel') not in captions:return iface('Curve draft target is no longer available')
        for arm in context.scene.objects:
            if arm.type!='ARMATURE':continue
            for bone in arm.data.bones:
                if bone.ppf_bone_physics.joint_uuid==target.get('joint_uuid'):
                    return iface('Bone: {armature} / {bone} / {channel}').format(armature=arm.name,bone=bone.name,channel=iface(captions[target['channel']]))
    elif target.get('kind')=='STITCH_PAIR':
        from ..models.groups import get_addon_data
        for pair in get_addon_data(context.scene).state.merge_pairs:
            if pair.pair_id==target.get('pair_id'):
                return iface('Stitch: {a} / {b}').format(a=pair.object_a,b=pair.object_b)
    return iface('Curve draft target is no longer available')


def pair_map(scene,targets):
    from ..models.groups import get_addon_data
    required=[t['pair_id'] for t in targets if t['kind']=='STITCH_PAIR']
    ids=[p.pair_id for p in get_addon_data(scene).state.merge_pairs]
    return model.stitch_mapping(ids,required) if required else []


def editor_table(owner):
    return {'table_id':'editor','domain':owner.domain,'tail':owner.tail,
            'nodes':[[node.coordinate,node.effort] for node in owner.nodes]}


def fill_editor(owner,target,document):
    key=model.target_key(target)
    binding=next((b for b in document['bindings'] if model.target_key(b['target'])==key),None)
    if binding is None:
        raise ValueError('No binding in editor source')
    table=next(t for t in document['tables'] if t['table_id']==binding['table_id'])
    owner.editor_target_json=model._json_bytes(target).decode('utf-8')
    owner.channel='stitch' if target['kind']=='STITCH_PAIR' else target['channel']
    if target['kind']=='JOINT':owner.joint_channel=target['channel']
    owner.enabled=binding['enabled'];owner.policy=binding['mode']
    owner.domain=table['domain'];owner.tail=table['tail'];owner.nodes.clear()
    for x,f in table['nodes']:
        node=owner.nodes.add();node.coordinate=x;node.effort=f
    if binding['mode']=='AUTO_FREQUENCY':
        owner.frequency_hz=binding['frequency_hz'];owner.damping_ratio=binding['damping_ratio'];owner.coordinate_scale=binding['coordinate_scale']
    else:owner.damping=binding['damping']


def editor_document(context,target):
    owner=context.scene.ppf_restoring_curves
    table=editor_table(owner)
    binding={'target':target,'enabled':owner.enabled,'table_id':'editor','mode':owner.policy}
    if owner.policy=='AUTO_FREQUENCY':binding.update(frequency_hz=owner.frequency_hz,damping_ratio=owner.damping_ratio,coordinate_scale=owner.coordinate_scale,mass_policy='LOCAL_REST_V1')
    else:binding['damping']=owner.damping
    return model.validate_author_map({'version':1,'tables':[table],'bindings':[binding],
                                     'stitch_pairs':pair_map(context.scene,[target])})


class CurveOperator:
    bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        try:self.perform(context)
        except (ValueError,OSError,TypeError) as exc:
            self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}


class PPF_OT_RestoringEditBegin(CurveOperator,bpy.types.Operator):
    bl_idname='ppf.restoring_edit_begin';bl_label='Load Current Curve'
    stitch:BoolProperty(default=False,options={'HIDDEN'})
    scope:EnumProperty(items=PANEL_SCOPES,default='AUTO',options={'HIDDEN'})
    def perform(self,context):
        owner=context.scene.ppf_restoring_curves
        state=read_state(context.scene);document=state['document']
        target=live_target(context,create_pair_id=True,scope='STITCH' if self.stitch and self.scope=='AUTO' else self.scope)
        if document is None or not any(model.target_key(b['target'])==model.target_key(target) for b in document['bindings']):
            document=author.common_document([target],pair_map(context.scene,[target]))
        fill_editor(owner,target,document)


class PPF_OT_RestoringEditApply(CurveOperator,bpy.types.Operator):
    bl_idname='ppf.restoring_edit_apply';bl_label='Apply Edited Curve'
    scope:EnumProperty(items=PANEL_SCOPES,default='AUTO',options={'HIDDEN'})
    def perform(self,context):
        owner=context.scene.ppf_restoring_curves;target=live_target(context,scope=self.scope)
        if model._json_bytes(target).decode('utf-8')!=owner.editor_target_json:
            raise ValueError(iface('Selection or channel changed; load its curve before Apply'))
        document=editor_document(context,target)
        state=read_state(context.scene)
        preset=presets.named_preset('Edited curve',document)
        author.apply_preset(state,preset,[{'source':target,'destination':target}],[target],document['stitch_pairs'])
        if owner.mode=='ADVANCED':state['mode']='ADVANCED'
        write_state(context.scene,state)
        self.report({'INFO'},iface('Curve authored; a compatible fresh Build is required'))


class PPF_OT_RestoringModeSet(CurveOperator,bpy.types.Operator):
    bl_idname='ppf.restoring_mode_set';bl_label='Apply Restoring Mode'
    stitch:BoolProperty(default=False,options={'HIDDEN'})
    scope:EnumProperty(items=PANEL_SCOPES,default='AUTO',options={'HIDDEN'})
    def perform(self,context):
        state=read_state(context.scene);owner=context.scene.ppf_restoring_curves
        scope='STITCH' if self.stitch and self.scope=='AUTO' else self.scope
        if scope!='AUTO':live_target(context,scope=scope)
        if state['document'] is None and owner.requested_mode!='LEGACY':
            target=live_target(context,create_pair_id=True,scope=scope)
            state['document']=author.common_document([target],pair_map(context.scene,[target]))
        author.set_mode(state,owner.requested_mode);write_state(context.scene,state)
        if self.stitch and self.scope=='AUTO':owner.channel='stitch'


class PPF_OT_RestoringReset(CurveOperator,bpy.types.Operator):
    bl_idname='ppf.restoring_reset';bl_label='Reset Current Curve'
    scope:EnumProperty(items=PANEL_SCOPES,default='AUTO',options={'HIDDEN'})
    def perform(self,context):
        state=read_state(context.scene)
        target=live_target(context,create_pair_id=True,scope=self.scope)
        document=author.common_document([target],pair_map(context.scene,[target]))
        preset=presets.named_preset('Common defaults',document)
        author.apply_preset(state,preset,[{'source':target,'destination':target}],[target],document['stitch_pairs'])
        if context.scene.ppf_restoring_curves.mode=='ADVANCED':state['mode']='ADVANCED'
        write_state(context.scene,state);fill_editor(context.scene.ppf_restoring_curves,target,state['document'])


class PPF_OT_RestoringNode(CurveOperator,bpy.types.Operator):
    bl_idname='ppf.restoring_node';bl_label='Edit Curve Nodes'
    action:EnumProperty(items=[('ADD','Add',''),('REMOVE','Remove','')],default='ADD',options=set())
    scope:EnumProperty(items=PANEL_SCOPES,default='AUTO',options={'HIDDEN'})
    def perform(self,context):
        owner=context.scene.ppf_restoring_curves
        if self.scope!='AUTO' and not draft_matches(context,self.scope):raise ValueError(iface('Selection or channel changed; load its curve before Apply'))
        if self.action=='ADD':
            if len(owner.nodes)>=64:raise ValueError(iface('At most 64 curve nodes'))
            last=owner.nodes[-1] if owner.nodes else None
            x=last.coordinate+1 if last else 0;f=last.effort+1 if last else 0
            node=owner.nodes.add();node.coordinate=x;node.effort=f
        elif 0<=owner.node_index<len(owner.nodes):owner.nodes.remove(owner.node_index)
        owner.node_index=max(0,min(owner.node_index,len(owner.nodes)-1))


class PPF_OT_RestoringPreset(CurveOperator,bpy.types.Operator):
    bl_idname='ppf.restoring_preset';bl_label='Restoring Preset'
    action:EnumProperty(items=[('LOAD','Load','Select file without applying'),('SAVE','Save','Save current target author preset'),('APPLY','Apply','Apply selected preset to current target')],default='LOAD',options=set())
    scope:EnumProperty(items=PANEL_SCOPES,default='AUTO',options={'HIDDEN'})
    def perform(self,context):
        owner=context.scene.ppf_restoring_curves
        if self.scope!='AUTO':live_target(context,scope=self.scope)
        if self.action=='LOAD':
            preset=presets.load_preset(bpy.path.abspath(owner.preset_path))
            raw=model._json_bytes(preset).decode('utf-8')
            owner.preset_json=raw;owner.preset_name=preset['name']
            return
        target=live_target(context,scope=self.scope)
        state=read_state(context.scene)
        if self.action=='SAVE':
            if state['document'] is None:raise ValueError(iface('Apply a curve before saving a preset'))
            selected=[b for b in state['document']['bindings'] if model.target_key(b['target'])==model.target_key(target)]
            if not selected:raise ValueError(iface('The current target has no applied curve'))
            table_ids={b['table_id'] for b in selected}
            document={'version':1,'tables':[t for t in state['document']['tables'] if t['table_id'] in table_ids],
                      'bindings':selected,'stitch_pairs':pair_map(context.scene,[target])}
            presets.save_preset(bpy.path.abspath(owner.preset_path),presets.named_preset(owner.preset_name,document))
        else:
            preset=presets.decode_preset(owner.preset_json)
            candidates=[b['target'] for b in preset['restoring_curves_v1']['bindings']]
            keys={model.target_key(t) for t in candidates}
            if len(keys)!=1:raise ValueError(iface('This editor applies single-channel presets; select a single-channel file'))
            source=candidates[0]
            source_key=model.target_key(source);target_key=model.target_key(target)
            if source_key[0]!=target_key[0] or (source_key[0]=='JOINT' and source_key[2]!=target_key[2]):
                raise ValueError(iface('The selected preset does not match the current channel. Switch to its matching channel, or load a preset for the current channel.'))
            author.apply_preset(state,preset,[{'source':source,'destination':target}],[target],pair_map(context.scene,[target]))
            if owner.mode=='ADVANCED':state['mode']='ADVANCED'
            write_state(context.scene,state);fill_editor(owner,target,state['document'])


def _chain_target_for_bone(arm,bone):
    from ..models.bone_bindings import validate_proxy_owner
    settings=bone.ppf_bone_physics
    if not settings.enabled or not settings.proxy or bone.parent is None:
        return None
    parent=bone.parent.ppf_bone_physics
    if not (parent.enabled or getattr(settings,'auto_follow_parent',True) and settings.root_mode=='FREE'):
        return None
    validate_proxy_owner(arm,bone)
    if settings.joint_mode in {'HINGE','D6'}:
        raise ValueError(f'{bone.name}: custom restoring presets require a compliant SOFT joint')
    return {'kind':'JOINT','joint_uuid':model.canonical_uuid(settings.joint_uuid,'joint_uuid')}


def _chain_bones(context,scope):
    from .bone_proxy import selected_bones
    if scope not in ('SELECTED','CHAIN','ALL'):
        raise ValueError('Invalid chain preset scope')
    arm,bones=selected_bones(context,scope=scope)
    targets=[]
    for bone in bones:
        target=_chain_target_for_bone(arm,bone)
        if target is not None: targets.append((bone,target))
    if not targets:
        raise ValueError('No eligible physical SOFT joint in the selected range')
    ids=[target['joint_uuid'] for _bone,target in targets]
    if len(ids)!=len(set(ids)):
        raise ValueError('Selected range contains duplicate joint UUIDs')
    return targets


def _chain_preset_channels(state,joint_uuid):
    from . import restoring_curve_presets as chain_presets
    doc=state['document'];by_table={t['table_id']:t for t in doc['tables']} if doc else {}
    output={channel:None for channel in chain_presets.CHAIN_CHANNELS}
    matches={channel:[] for channel in chain_presets.CHAIN_CHANNELS}
    for binding in (doc or {}).get('bindings',[]):
        target=binding['target']
        if target['kind']=='JOINT' and target['joint_uuid']==joint_uuid and target['channel'] in matches:
            matches[target['channel']].append(binding)
    for channel,rows in matches.items():
        if len(rows)>1:
            raise ValueError(f'{channel}: duplicate authored bindings cannot be bundled unambiguously')
        if not rows: continue
        binding=rows[0]
        record={key:value for key,value in binding.items() if key not in ('target','table_id')}
        output[channel]={'binding':record,'table':copy.deepcopy(by_table[binding['table_id']])}
    return output


def _apply_chain_document(context,preset,targets):
    from . import restoring_curve_presets as chain_presets
    checked=chain_presets.validate_chain_preset(preset)
    state=read_state(context.scene)
    if state['mode']=='LEGACY' and state['document'] is not None:
        raise ValueError('Dormant legacy curves exist; explicitly convert the restoring mode before applying a chain preset')
    doc=copy.deepcopy(state['document']) if state['document'] is not None else {
        'version':1,'tables':[],'bindings':[],'stitch_pairs':[]}
    selected_ids={target['joint_uuid'] for _bone,target in targets}
    # A null preset channel means "not included". Keep that channel's
    # destination binding and table intact; enabled=False is still a
    # configured record and therefore an explicit replacement.
    replaced_channels={channel for channel,record in checked['channels'].items()
                       if record is not None}
    doc['bindings']=[binding for binding in doc['bindings'] if not (
        binding['target']['kind']=='JOINT'
        and binding['target']['joint_uuid'] in selected_ids
        and binding['target']['channel'] in replaced_channels)]
    referenced={binding['table_id'] for binding in doc['bindings']}
    doc['tables']=[table for table in doc['tables'] if table['table_id'] in referenced]
    table_by_id={table['table_id']:table for table in doc['tables']}
    for _bone,target in targets:
        for channel in chain_presets.CHAIN_CHANNELS:
            record=checked['channels'][channel]
            if record is None: continue
            table=copy.deepcopy(record['table'])
            binding=copy.deepcopy(record['binding'])
            seed=target['joint_uuid']+'|'+channel+'|'+model._json_bytes(table).decode('utf-8')
            table_id='chain_'+hashlib.sha256(seed.encode('utf-8')).hexdigest()[:24]
            table['table_id']=table_id
            existing=table_by_id.get(table_id)
            if existing is not None and existing!=table:
                raise ValueError(f'{channel}: deterministic table ID collision')
            if existing is None:
                table_by_id[table_id]=table
            else:
                table_id=existing['table_id']
            binding.update(target={'kind':'JOINT','joint_uuid':target['joint_uuid'],'channel':channel},
                           table_id=table_id)
            doc['bindings'].append(binding)
    doc['tables']=list(table_by_id.values())
    checked_doc=model.validate_author_map(doc)
    mode='COMMON' if all(binding['mode']=='AUTO_FREQUENCY' for binding in checked_doc['bindings']) else 'ADVANCED'
    candidate={'mode':mode,'document':checked_doc}
    # Do not mutate RNA until every selected target and the whole merged map validate.
    model.export_author(candidate)
    write_state(context.scene,candidate)
    return candidate


class PPF_OT_RestoringChainPreset(CurveOperator,bpy.types.Operator):
    bl_idname='ppf.restoring_chain_preset';bl_label='Restoring Chain Preset'
    action:EnumProperty(items=[('SAVE','Save','Save all configured restoring channels for the active physical joint'),('LOAD','Load','Select a chain preset without applying'),('APPLY','Apply','Apply the selected channels to the current physical-joint range')],default='LOAD',options=set())
    def perform(self,context):
        from . import restoring_curve_presets as chain_presets
        owner=context.scene.ppf_restoring_curves
        if self.action=='LOAD':
            checked=chain_presets.load_chain_preset(bpy.path.abspath(owner.preset_path))
            owner.preset_json=model._json_bytes(checked).decode('utf-8')
            owner.preset_name=checked['name']
            return
        if self.action=='SAVE':
            target=live_target(context,scope='BONE')
            state=read_state(context.scene)
            channels=_chain_preset_channels(state,target['joint_uuid'])
            preset=chain_presets.named_chain_preset(owner.preset_name,target['joint_uuid'],channels)
            chain_presets.save_chain_preset(bpy.path.abspath(owner.preset_path),preset)
            owner.preset_json=model._json_bytes(preset).decode('utf-8')
            return
        preset=chain_presets.decode_chain_preset(owner.preset_json)
        targets=_chain_bones(context,owner.bone_chain_scope)
        _apply_chain_document(context,preset,targets)


CLASSES=(PPF_OT_RestoringEditBegin,PPF_OT_RestoringEditApply,PPF_OT_RestoringModeSet,PPF_OT_RestoringReset,PPF_OT_RestoringNode,PPF_OT_RestoringPreset,PPF_OT_RestoringChainPreset)
def register():
    for cls in CLASSES:_ppf_i18n.register_class(cls)
def unregister():
    for cls in reversed(CLASSES):bpy.utils.unregister_class(cls)
