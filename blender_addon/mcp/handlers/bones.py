"""Named, context-preserving MCP adapters over the production bone tools."""
import math
import bpy
from ..decorators import mcp_handler,MCPError
from ._bone_context import (bone_context,named_armature,names_argument,require_idle,
                            finished,push_undo,preset_operator_state,SCOPES,bone_undo_transaction,mark_bone_undo_commit)

SIMPLE_FIELDS=('mass_mode','target_mass_kg','density','solid_young_modulus','solid_deformation_damping',
    'soft_limit_mode','swing_cone_half_angle','swing_cone_stiffness','soft_stiffness',
    'soft_limits_enabled','soft_lower','soft_upper','soft_limit_policy','soft_limit_strength',
    'auto_follow_parent','collide_connected','collision_filter_enabled','collision_group_layers','collision_mask_layers')
PHYSICAL_FIELDS=('angular_stiffness_degree','angular_damping_degree')
WRITE=dict(readOnlyHint=False,destructiveHint=True,idempotentHint=False,openWorldHint=False)

def _plain(v):
    if isinstance(v,(str,int,float,bool)) or v is None:return v
    return list(v)

def _row(arm,bone):
    from ...models.bone_bindings import validate_proxy_owner,effective_preview_color
    from ...models.proxy_generation import effective_proxy_quality,effective_proxy_face_mode
    s=bone.ppf_bone_physics;error=''
    try:validate_proxy_owner(arm,bone)
    except ValueError as exc:error=str(exc)
    return dict(name=bone.name,parent=bone.parent.name if bone.parent else None,enabled=bool(s.enabled),
        proxy=s.proxy.name if s.proxy else None,physical_model=s.physical_model,new_binding_physical_model=s.new_binding_physical_model,
        body_uuid=s.body_uuid,bone_uuid=s.bone_uuid,joint_uuid=s.joint_uuid,group_uuid=s.group_uuid,
        ownership_valid=not bool(error),ownership_error=error,shape=s.shape,
        proxy_quality=effective_proxy_quality(s),proxy_face_mode=effective_proxy_face_mode(s),
        custom_object=s.custom_object.name if s.custom_object else None,
        preview_color_custom=bool(s.preview_color_custom),preview_color=list(s.preview_color),effective_preview_color=list(effective_preview_color(s,s.physical_model)),
        simple_settings={k:_plain(getattr(s,k)) for k in SIMPLE_FIELDS},
        restoring_note='Use set_bone_physics physical degree fields for explicit SI curve edits; AUTO effective coefficients require a real Build')

@mcp_handler
def get_bone_physics(armature_name:str='',bone_names:list[str]|None=None):
    """List armatures or query explicit bone proxy/ownership/color/simple settings.

    Empty armature_name lists current-scene armatures. With an armature, empty
    bone_names queries all its bones. This never creates a binding or UUID.

    Args:
        armature_name: Exact current-scene armature object name, or empty to list rigs
        bone_names: Explicit bone names, or empty/omitted for all bones
    """
    if not armature_name:
        if bone_names:raise MCPError('bone_names requires armature_name')
        return dict(armatures=[dict(name=o.name,bone_count=len(o.data.edit_bones if o.mode=='EDIT' else o.data.bones)) for o in bpy.context.scene.objects if o.type=='ARMATURE'])
    with bone_context(armature_name,bone_names,'SELECTED' if bone_names else 'ALL',writable=False) as (arm,bones):
        return dict(armature=arm.name,bones=[_row(arm,b) for b in bones])

@mcp_handler(annotations=WRITE)
@bone_undo_transaction(may_delete_proxy=True)
def edit_bone_proxies(armature_name:str,bone_names:list[str]|None=None,scope:str='SELECTED',action:str='CREATE_UPDATE',rebuild:bool=False,source_object_name:str='',shape:str='',proxy_quality:str='',proxy_face_mode:str=''):
    """Batch create/update, apply a custom mesh or delete owned proxies via Undo operators.

    New bones use the existing author new-binding defaults. Normal update keeps
    edited geometry; rebuild=True explicitly replaces it. DELETE preserves IDs.
    Requires idle/Clear and preserves the caller's object/bone selection and mode.

    Args:
        armature_name: Exact armature object name
        bone_names: Explicit selected names; CHAIN needs one root; ALL requires empty
        scope: SELECTED, CHAIN (root plus descendants), or ALL
        action: CREATE_UPDATE, CUSTOM, DELETE or GENERATE (replaces generated geometry)
        rebuild: Explicitly replace existing edited geometry for CREATE_UPDATE or CUSTOM
        source_object_name: Exact current-scene source mesh name, required for CUSTOM only
        shape: GENERATE only: CAPSULE, SPHERE or BOX; empty preserves each bone's shape
        proxy_quality: GENERATE only: AUTO, VERY_LOW, LOW, MEDIUM, HIGH or VERY_HIGH
        proxy_face_mode: GENERATE only: QUAD or TRI
    """
    if action not in ('CREATE_UPDATE','CUSTOM','DELETE','GENERATE'):raise MCPError('action must be CREATE_UPDATE, CUSTOM, DELETE or GENERATE')
    if action=='GENERATE':
        if shape not in ('','CAPSULE','SPHERE','BOX') or proxy_quality not in ('','AUTO','VERY_LOW','LOW','MEDIUM','HIGH','VERY_HIGH') or proxy_face_mode not in ('','QUAD','TRI'):
            raise MCPError('Invalid generation shape, LOD or topology')
        if not any((shape,proxy_quality,proxy_face_mode)):raise MCPError('GENERATE requires explicit shape, proxy_quality or proxy_face_mode')
    elif any((shape,proxy_quality,proxy_face_mode)):raise MCPError('Generation fields require action GENERATE')
    if type(rebuild) is not bool or action=='DELETE' and rebuild:raise MCPError('rebuild is a boolean for CREATE_UPDATE or CUSTOM only')
    if not isinstance(source_object_name,str):raise MCPError('source_object_name must be a string')
    if action=='CUSTOM':
        source=bpy.context.scene.objects.get(source_object_name) if source_object_name else None
        if source is None or source.type!='MESH' or source.get('ppf_bone_proxy_owned'):
            raise MCPError('CUSTOM requires an explicit current-scene source mesh, not an owned proxy')
    elif source_object_name:raise MCPError('source_object_name is only used for CUSTOM')
    require_idle()
    with bone_context(armature_name,bone_names,scope) as (arm,bones):
        names=[b.name for b in bones]
        if action=='GENERATE':
            arm.data.bones.active=bones[0]
            finished(bpy.ops.ppf.bone_proxy_generate('EXEC_DEFAULT',True,scope=scope,copy_settings=False,
                shape=shape or 'ACTIVE',quality=proxy_quality or 'ACTIVE',face_mode=proxy_face_mode or 'ACTIVE'),'Generate bone proxies')
            mark_bone_undo_commit()
            return dict(armature=arm.name,action=action,scope=scope,bones=[_row(arm,arm.data.bones[n]) for n in names],undoable=True)
        op={'CREATE_UPDATE':bpy.ops.ppf.bone_proxy_update,'CUSTOM':bpy.ops.ppf.bone_custom_apply,'DELETE':bpy.ops.ppf.bone_proxy_delete}[action]
        kw={'scope':scope}
        if action in ('CREATE_UPDATE','CUSTOM'):kw['rebuild']=rebuild
        if action=='CUSTOM':kw['source_name']=source_object_name
        finished(op('EXEC_DEFAULT',True,**kw),action)
        mark_bone_undo_commit()
        return dict(armature=arm.name,action=action,scope=scope,bones=[_row(arm,arm.data.bones[n]) for n in names],undoable=True)

@mcp_handler(annotations=WRITE)
@bone_undo_transaction
def copy_bone_physics(armature_name:str,source_bone_name:str,bone_names:list[str]|None=None,scope:str='SELECTED',category:str='PARAMETERS'):
    """Apply active-source settings to a batch through the unified editor's Undo operator.

    PARAMETERS copies common material/mass/restoring/limit parameters while preserving
    target geometry, collision, participation, controls and identities. SHAPE copies
    creation settings without rebuilding edited meshes. COLLISION copies collision
    settings only. ALL retains the existing full-settings copy behavior.

    Args:
        armature_name: Exact current-scene armature object name
        source_bone_name: Exact source bone, also selected for SELECTED scope
        bone_names: SELECTED includes source and targets; CHAIN empty or source only; ALL empty
        scope: SELECTED, CHAIN rooted at source, or ALL
        category: PARAMETERS, SHAPE, COLLISION or ALL
    """
    if category not in ('PARAMETERS','SHAPE','COLLISION','ALL'):raise MCPError('Unknown bone settings category')
    arm=named_armature(armature_name)
    if not isinstance(source_bone_name,str) or not source_bone_name or arm.data.bones.get(source_bone_name) is None:
        raise MCPError('source_bone_name must explicitly name a bone in the armature')
    if scope=='CHAIN':
        if bone_names not in (None,[],[source_bone_name]):raise MCPError('CHAIN is rooted at source_bone_name')
        names=[source_bone_name]
    else:
        names=names_argument(bone_names,scope)
        if scope=='SELECTED':
            if source_bone_name not in names:raise MCPError('SELECTED bone_names must include the source bone')
            names=[source_bone_name,*[name for name in names if name!=source_bone_name]]
    require_idle()
    with bone_context(armature_name,names,scope) as (arm,bones):
        arm.data.bones.active=arm.data.bones[source_bone_name]
        target_names=[bone.name for bone in bones if bone.name!=source_bone_name]
        finished(bpy.ops.ppf.bone_settings_copy('EXEC_DEFAULT',True,scope=scope,category=category),'Batch bone settings')
        mark_bone_undo_commit()
        return dict(armature=arm.name,source_bone=source_bone_name,scope=scope,category=category,
                    bones=[_row(arm,arm.data.bones[name]) for name in target_names],
                    geometry_rebuilt=False,undoable=True,requires_transfer_build=True)

@mcp_handler(annotations=WRITE)
@bone_undo_transaction
def set_bone_participation(armature_name:str,enabled:bool,bone_names:list[str]|None=None,scope:str='SELECTED'):
    """Enable/disable existing owned proxies, using the production transactional Undo operator.

    Bones without proxies are skipped and reported. Does not create/delete a proxy.

    Args:
        armature_name: Exact armature object name
        enabled: True to participate, False to retain but disable existing proxies
        bone_names: Explicit names; CHAIN exactly one root; ALL empty
        scope: SELECTED, CHAIN or ALL
    """
    if type(enabled) is not bool:raise MCPError('enabled must be a JSON boolean')
    require_idle()
    with bone_context(armature_name,bone_names,scope) as (arm,bones):
        names=[b.name for b in bones];skipped=[b.name for b in bones if b.ppf_bone_physics.proxy is None]
        finished(bpy.ops.ppf.bone_participation('EXEC_DEFAULT',True,scope=scope,enabled=enabled),'Participation')
        mark_bone_undo_commit()
        return dict(armature=arm.name,bones=[_row(arm,arm.data.bones[n]) for n in names],skipped_without_proxy=skipped,undoable=True)

@mcp_handler(annotations=dict(readOnlyHint=False,destructiveHint=False,idempotentHint=True,openWorldHint=False))
@bone_undo_transaction
def set_bone_preview_color(armature_name:str,custom:bool,color:list[float]|None=None,bone_names:list[str]|None=None,scope:str='SELECTED'):
    """Set per-bone GPU preview auto/custom RGB only; never changes physics or Build state.

    Auto retains stored custom RGB unless color is supplied. New fields are also
    included by the full user preset core. This is one undoable display-only edit.

    Args:
        armature_name: Exact armature object name
        custom: True for author RGB, False for automatic material color
        color: Optional three finite RGB numbers in [0,1]; required when custom=True
        bone_names: Explicit names; CHAIN one root; ALL empty
        scope: SELECTED, CHAIN or ALL
    """
    if type(custom) is not bool:raise MCPError('custom must be a JSON boolean')
    if color is None and custom:raise MCPError('custom=True requires color=[r,g,b]')
    if color is not None and (not isinstance(color,list) or len(color)!=3 or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in color)):raise MCPError('color must contain three finite RGB numbers in [0,1]')
    require_idle(stale=False)
    with bone_context(armature_name,bone_names,scope) as (arm,bones):
        from ...models.bone_bindings import validate_proxy_owner
        for b in bones:validate_proxy_owner(arm,b)
        if not bpy.ops.ed.undo_push.poll():raise MCPError('Undo is unavailable in this context')
        before=[(b.ppf_bone_physics,bool(b.ppf_bone_physics.preview_color_custom),tuple(b.ppf_bone_physics.preview_color),
                 b.ppf_bone_physics.is_property_set('preview_color_custom'),b.ppf_bone_physics.is_property_set('preview_color')) for b in bones]
        try:
            for s,_custom,_color,_custom_set,_color_set in before:
                if color is not None:s.preview_color=color
                s.preview_color_custom=custom
            push_undo('PPF bone preview color')
        except Exception:
            for s,old_custom,old_color,custom_set,color_set in reversed(before):
                if color_set:s.preview_color=old_color
                else:s.property_unset('preview_color')
                if custom_set:s.preview_color_custom=old_custom
                else:s.property_unset('preview_color_custom')
            raise
        return dict(armature=arm.name,bones=[_row(arm,b) for b in bones],display_only=True,requires_build=False,undoable=True)

@mcp_handler(annotations=WRITE)
def bone_user_presets(action:str='LIST',preset_id:str='',name:str='',armature_name:str='',bone_name:str=''):
    """List/save/rename/delete the complete cross-project user bone preset library.

    SAVE captures exactly one explicitly named physical bone via the production
    ownership/idle/closed-mesh core. RENAME/DELETE require a library preset ID.
    These are persistent library file operations, not Blender Undo actions.

    Args:
        action: LIST, SAVE, RENAME or DELETE
        preset_id: Exact returned preset ID for RENAME/DELETE, empty for SAVE
        name: New unique user name for SAVE/RENAME
        armature_name: Exact source armature name for SAVE only
        bone_name: Exact source bone name for SAVE only
    """
    from ...core import bone_preset_library as lib
    if action=='LIST':return dict(presets=lib.list_presets(),directory=str(lib.preset_directory()))
    if action=='SAVE':
        if preset_id:raise MCPError('SAVE creates a new preset; preset_id must be empty')
        require_idle()
        with bone_context(armature_name,[bone_name]) as (_arm,_bones):document=lib.capture_preset(bpy.context,name)
        path=lib.save_preset(name,document)
    elif action=='RENAME':path=lib.rename_preset(preset_id,name)
    elif action=='DELETE':path=lib.delete_preset(preset_id)
    else:raise MCPError('action must be LIST, SAVE, RENAME or DELETE')
    return dict(action=action,preset_id=path.stem,name=name,presets=lib.list_presets(),file_operation_undoable=False)

@mcp_handler(annotations=WRITE)
@bone_undo_transaction
def apply_bone_user_preset(preset_id:str,armature_name:str,bone_names:list[str]|None=None,scope:str='SELECTED',size_mode:str='FIT_BONE'):
    """Apply a complete user preset through the production Undo operator and atomic core.

    Geometry, material, curves, participation and preview RGB are included. No
    arbitrary internal property paths are accepted. The existing capability and
    ownership gates remain authoritative. Caller context/library selection restore.

    Args:
        preset_id: Exact ID returned by bone_user_presets LIST/SAVE
        armature_name: Exact target armature object name
        bone_names: Explicit targets; CHAIN one root; ALL empty
        scope: SELECTED, CHAIN or ALL
        size_mode: FIT_BONE scales by bone length; KEEP_SIZE retains saved size
    """
    if size_mode not in ('FIT_BONE','KEEP_SIZE'):raise MCPError('size_mode must be FIT_BONE or KEEP_SIZE')
    require_idle()
    with bone_context(armature_name,bone_names,scope) as (arm,bones):
        names=[b.name for b in bones]
        with preset_operator_state(preset_id,scope,size_mode):
            finished(bpy.ops.ppf.bone_user_preset_apply('EXEC_DEFAULT',True),'Preset Apply')
            mark_bone_undo_commit()
        return dict(armature=arm.name,preset_id=preset_id,scope=scope,size_mode=size_mode,bones=[_row(arm,arm.data.bones[n]) for n in names],undoable=True)

@mcp_handler(annotations=WRITE)
@bone_undo_transaction
def set_bone_physics(armature_name:str,bone_name:str,changes:dict):
    """Set one bone's common panel parameters through existing checked production paths.

    Exactly one category per call: physical_model alone uses the body-switch Undo
    operator; angular_stiffness_degree/angular_damping_degree use the simple panel's
    explicit physical draft/apply (N mm/degree and N mm s/degree); remaining supported
    simple fields use full-preset capture/apply KEEP_SIZE as the existing atomic
    validation/rollback path, preserving unspecified settings/geometry values.
    No arbitrary setattr, advanced drives/HARD fields or unverified physical formula.
    If Undo recording fails AFTER common core Apply commits, the response remains
    success with committed=True, undoable=False and a warning. It is not an
    unapplied atomic failure and must not be retried as one.

    Args:
        armature_name: Exact armature object name
        bone_name: Exactly one target bone; use full preset Apply for transactional batches
        changes: Whitelisted common panel fields; query tool schema properties for exact names
    """
    if not isinstance(changes,dict) or not changes:raise MCPError('changes must be a nonempty object')
    allowed=set(SIMPLE_FIELDS+PHYSICAL_FIELDS+('physical_model',))
    unknown=set(changes)-allowed
    if unknown:raise MCPError('Unsupported SET fields: '+', '.join(sorted(unknown))+'. Use edit_bone_proxies, participation, preview-color or complete preset tools as appropriate')
    require_idle()
    with bone_context(armature_name,[bone_name]) as (arm,bones):
        b=bones[0];s=b.ppf_bone_physics
        if 'physical_model' in changes:
            if len(changes)!=1 or changes['physical_model'] not in ('SOFT','RIGID'):raise MCPError('physical_model must be SOFT or RIGID and is a separate transaction')
            finished(bpy.ops.ppf.bone_physical_body_change('EXEC_DEFAULT',True,scope='SELECTED',physical_body=changes['physical_model']),'Physical body switch')
            mark_bone_undo_commit()
        elif any(k in changes for k in PHYSICAL_FIELDS):
            if set(changes)-set(PHYSICAL_FIELDS):raise MCPError('Physical restoring K/C and other SET fields require separate transactions')
            for key,v in changes.items():
                if not isinstance(v,list) or len(v)!=3 or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in v):raise MCPError(key+' requires three finite nonnegative physical values')
            from ...ops.restoring_curve_operators import snapshot_rna,restore_rna
            from ...ops.bone_physical_controls import new_draft
            owner=b.ppf_bone_restoring_controls;old=snapshot_rna(owner)
            try:
                finished(bpy.ops.ppf.bone_restoring_controls_load('EXEC_DEFAULT'),'Load physical draft')
                available=tuple(owner.physical_available);previous={k:list(getattr(owner,k)) for k in PHYSICAL_FIELDS}
                if len(changes)!=2 and not all(available[:3]):raise MCPError('AUTO/custom coefficients are unresolved. Supply both K and C arrays for explicit replacement, or Build/load resolved channels first')
                for axis in ('twist_x','swing_y','swing_z'):new_draft(bpy.context,b,axis)
                for key in PHYSICAL_FIELDS:setattr(owner,key,changes.get(key,previous[key]))
                finished(bpy.ops.ppf.bone_restoring_controls_apply('EXEC_DEFAULT',True),'Apply physical draft')
                mark_bone_undo_commit()
            finally:restore_rna(owner,old)
        else:
            from ...core import bone_preset_library as lib
            from ...core.bone_preset_apply import _validate_rna
            for key,value in changes.items():_validate_rna(s,key,value)
            if not bpy.ops.ed.undo_push.poll():raise MCPError('Undo is unavailable in this context')
            document=lib.capture_preset(bpy.context,'MCP common field transaction')
            document['settings']['bone'].update(changes)
            lib.apply_preset(bpy.context,document,scope='SELECTED',size_mode='KEEP_SIZE')
            mark_bone_undo_commit()
            try:
                push_undo('PPF bone common physics')
            except Exception as exc:
                # Core Apply has committed and may have released the old mesh.
                # Do not disguise this as an atomic failure safe to retry.
                return dict(status='success',armature=arm.name,bone=_row(arm,b),
                    changed_fields=sorted(changes),committed=True,undoable=False,
                    warning='Settings were applied, but Blender could not record Undo: '+str(exc)+'. Do not retry this request as a failed transaction.',
                    requires_transfer_build=True)
        return dict(armature=arm.name,bone=_row(arm,b),changed_fields=sorted(changes),committed=True,undoable=True,requires_transfer_build=True)

# Extend the actual registered schemas (not a parallel tool list). Runtime gates
# above validate the same enums and strict arrays even for direct dispatcher calls.
def _enum(tool,key,values):tool._schema['inputSchema']['properties'][key]['enum']=list(values)
for tool in (edit_bone_proxies,copy_bone_physics,set_bone_participation,set_bone_preview_color,apply_bone_user_preset):_enum(tool,'scope',SCOPES)
_enum(edit_bone_proxies,'action',('CREATE_UPDATE','CUSTOM','DELETE','GENERATE'))
_enum(edit_bone_proxies,'shape',('','CAPSULE','SPHERE','BOX'))
_enum(edit_bone_proxies,'proxy_quality',('','AUTO','VERY_LOW','LOW','MEDIUM','HIGH','VERY_HIGH'))
_enum(edit_bone_proxies,'proxy_face_mode',('','QUAD','TRI'))
_enum(copy_bone_physics,'category',('PARAMETERS','SHAPE','COLLISION','ALL'))
_enum(bone_user_presets,'action',('LIST','SAVE','RENAME','DELETE'))
_enum(apply_bone_user_preset,'size_mode',('FIT_BONE','KEEP_SIZE'))
set_bone_preview_color._schema['inputSchema']['properties']['color'].update(minItems=3,maxItems=3,items=dict(type='number',minimum=0,maximum=1))
_fields={k:dict(type='number') for k in ('target_mass_kg','density','solid_young_modulus','solid_deformation_damping','soft_limit_strength','swing_cone_half_angle','swing_cone_stiffness')}
_fields.update({k:dict(type='boolean') for k in ('auto_follow_parent','collide_connected','collision_filter_enabled')})
_fields.update({k:dict(type='array',items=dict(type='boolean')) for k in ('soft_limits_enabled','collision_group_layers','collision_mask_layers')})
_fields.update({k:dict(type='array',minItems=3,maxItems=3,items=dict(type='number')) for k in ('soft_lower','soft_upper','soft_stiffness')+PHYSICAL_FIELDS})
_fields['soft_limit_mode']=dict(type='string',enum=['AXES','CONE'])
_fields['swing_cone_half_angle'].update(exclusiveMinimum=0,exclusiveMaximum=math.pi-1.e-3,description='Cone half angle in radians; UI displays scene angle units')
_fields['swing_cone_stiffness'].update(minimum=0,description='Explicit SI Nm/rad; active for ABSOLUTE_SI policy')
_fields.update(mass_mode=dict(type='string',enum=['MASS','DENSITY']),physical_model=dict(type='string',enum=['SOFT','RIGID']),soft_limit_policy=dict(type='string',enum=['AUTO_INERTIA','ABSOLUTE_SI']))
set_bone_physics._schema['inputSchema']['properties']['changes'].update(properties=_fields,additionalProperties=False,minProperties=1)
