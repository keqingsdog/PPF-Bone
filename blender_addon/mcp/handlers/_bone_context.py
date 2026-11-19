"""Explicit named-bone context for MCP adapters; business logic stays in core/ops."""
from contextlib import contextmanager
from functools import wraps
from inspect import signature
import bpy
from ...core.compat import pose_bone_selected,set_pose_bone_selected
from ..decorators import MCPError

SCOPES=('SELECTED','CHAIN','ALL')

# Main-thread, single-call bridge to a native Undo operator. Never a timer batch.
_pending_undo_call=None
_undo_execution_depth=0


class PPF_OT_MCPBoneTransaction(bpy.types.Operator):
    bl_idname='ppf.mcp_bone_transaction'
    bl_label='MCP Bone Edit'
    bl_options={'INTERNAL','UNDO'}

    @classmethod
    def poll(cls,context):
        return _pending_undo_call is not None

    def execute(self,context):
        global _undo_execution_depth
        call=_pending_undo_call
        if call is None:return {'CANCELLED'}
        _undo_execution_depth+=1
        try:
            call['result']=call['callback']()
            call['completed']=True
            if isinstance(call['result'],dict) and call['result'].get('status')=='error':return {'CANCELLED'}
            return {'FINISHED'}
        except Exception as exc:
            call['error']=exc
            return {'FINISHED'} if call['committed'] else {'CANCELLED'}
        finally:
            _undo_execution_depth-=1


def mark_bone_undo_commit():
    if _undo_execution_depth and _pending_undo_call is not None:
        _pending_undo_call['committed']=True


def register_undo_operator():
    bpy.utils.register_class(PPF_OT_MCPBoneTransaction)


def unregister_undo_operator():
    if getattr(PPF_OT_MCPBoneTransaction,'is_registered',False):
        bpy.utils.unregister_class(PPF_OT_MCPBoneTransaction)


def _invoke_undo_operator():
    return bpy.ops.ppf.mcp_bone_transaction('EXEC_DEFAULT',True)


def bone_undo_transaction(func=None,*,may_delete_proxy=False):
    """Record one successful write only after its caller context is restored."""
    if func is None:
        return lambda callback:bone_undo_transaction(callback,may_delete_proxy=may_delete_proxy)
    parameters=signature(func) if may_delete_proxy else None
    @wraps(func)
    def invoke(*args,**kwargs):
        global _pending_undo_call
        if _undo_execution_depth:return func(*args,**kwargs)
        if _pending_undo_call is not None:raise MCPError('Another bone Undo transaction is pending')
        # Native history records only the active Edit type/cohort. Other Edit
        # members are lost on Undo/Redo, even when immediate restoration works.
        edits=[obj for obj in bpy.data.objects if obj.mode=='EDIT']
        if edits:
            active=bpy.context.view_layer.objects.active
            if active not in edits or active.type!='ARMATURE' or any(obj.type!=active.type or obj.name not in bpy.context.view_layer.objects for obj in edits):
                raise MCPError('Exit non-armature, mixed or non-active Edit modes before using undoable bone tools')
            # Deleting proxy IDs with Edit Undo can restore invalid ID pointers
            # and crash later Undo/Redo. Refuse before entering any operator.
            if may_delete_proxy and parameters.bind(*args,**kwargs).arguments.get('action')=='DELETE':
                raise MCPError('Exit Edit Mode before deleting bone proxies so Blender can safely record Undo/Redo')
        if not bpy.context.preferences.edit.use_global_undo:
            raise MCPError('Enable Global Undo before using undoable bone tools')
        if not bpy.ops.ed.undo_push.poll():raise MCPError('Undo is unavailable in this Blender context')
        call={'callback':lambda:func(*args,**kwargs),'completed':False,'committed':False,'result':None,'error':None}
        _pending_undo_call=call;record_error=None;outcome=None
        try:
            try:outcome=_invoke_undo_operator()
            except Exception as exc:record_error=exc
        finally:
            _pending_undo_call=None
        if call['error'] is not None and not call['committed']:raise call['error']
        if call['completed'] or call['committed']:
            # FINISHED alone is not proof: a headless session may have no
            # initialized Undo history. Never create an extra baseline step.
            try:undo_available=bpy.ops.ed.undo.poll()
            except Exception:undo_available=False
            record_confirmed=record_error is None and outcome is not None and 'FINISHED' in outcome and undo_available
            result=call['result']
            if call['error'] is not None:
                result=dict(status='success',operation=func.__name__,committed=True,undoable=record_confirmed,
                            warning='Bone changes were committed, but finishing the request failed: '+str(call['error'])+'. Inspect the current scene before further edits; do not retry as an unapplied transaction.')
            if isinstance(result,dict) and result.get('status')=='error':return result
            if not record_confirmed:
                # The body already returned after context restoration. Never
                # pretend this is an unapplied failure safe to retry.
                result=dict(result) if isinstance(result,dict) else {'result':result}
                result.update(status='success',committed=True,undoable=False)
                warning='Bone edit completed, but Blender did not confirm the final Undo recording: '+str(record_error or (outcome if outcome is None or 'FINISHED' not in outcome else 'Undo is unavailable after the completed call'))+'. Inspect the current scene; do not retry as an unapplied transaction.'
                result['warning']=(str(result.get('warning',''))+' '+warning).strip()
            return result
        if record_error is not None:raise record_error
        raise MCPError('Bone Undo transaction was not executed')
    return invoke


def require_idle(stale=True):
    from ...core.facade import communicator
    communicator.assert_bone_body_switch_idle()
    if stale:
        from ...ops.bone_proxy import stale_simulation_output
        reasons=stale_simulation_output(bpy.context)
        if reasons:raise MCPError('Clear existing simulation frames before editing bone physics: '+', '.join(reasons))

def named_armature(name):
    if not isinstance(name,str) or not name:raise MCPError('armature_name must explicitly name one armature')
    arm=bpy.context.scene.objects.get(name)
    if arm is None or arm.type!='ARMATURE':raise MCPError(f'Armature {name!r} is not in the current scene')
    if arm.name not in bpy.context.view_layer.objects:raise MCPError('Armature is excluded from the current view layer')
    return arm

def names_argument(names,scope):
    if scope not in SCOPES:raise MCPError('scope must be SELECTED, CHAIN or ALL')
    if names is None:names=[]
    if not isinstance(names,list) or any(not isinstance(n,str) or not n for n in names) or len(set(names))!=len(names):raise MCPError('bone_names must be a list of distinct nonempty bone names')
    if scope=='ALL' and names:raise MCPError('ALL requires empty bone_names; the whole named armature is used')
    if scope=='CHAIN' and len(names)!=1:raise MCPError('CHAIN requires exactly one explicit root bone name')
    if scope=='SELECTED' and not names:raise MCPError('SELECTED requires explicit bone_names; current selection is never inferred')
    return names

def _bone_selection(arm):
    if arm.mode=='EDIT':
        return ('EDIT',[(b.name,b.select,b.select_head,b.select_tail) for b in arm.data.edit_bones],arm.data.edit_bones.active.name if arm.data.edit_bones.active else None)
    return ('POSE',[(p.name,pose_bone_selected(p)) for p in arm.pose.bones],arm.data.bones.active.name if arm.data.bones.active else None)

def _restore_bones(arm,saved):
    kind,rows,active=saved
    if arm.mode=='EDIT':
        # Assigning active can select an edit bone; restore exact flags afterward.
        arm.data.edit_bones.active=arm.data.edit_bones.get(active) if active else None
        for row in rows:
            b=arm.data.edit_bones.get(row[0])
            if b:
                b.select=row[1]
                if kind=='EDIT':b.select_head=row[2];b.select_tail=row[3]
    else:
        arm.data.bones.active=arm.data.bones.get(active) if active else None
        for row in rows:
            p=arm.pose.bones.get(row[0])
            if p:set_pose_bone_selected(p,row[1])

def _restore_mode_members(view,saved):
    """Restore Pose/Edit membership, not whatever objects happen to be selected."""
    def groups(items):
        result={}
        for obj,mode in items:
            if mode not in ('POSE','EDIT'):continue
            try:
                if obj.name in view.objects:result.setdefault((mode,obj.type),set()).add(obj)
            except ReferenceError:pass # An explicit proxy DELETE can remove an Edit object.
        return result
    expected=groups(saved);current=groups((obj,obj.mode) for obj in view.objects if obj is not None)
    changed=[key for key in current.keys()|expected.keys() if current.get(key,set())!=expected.get(key,set())]
    # A mode exit affects the entire cohort, including deselected members.
    for key in changed:
        members=current.get(key,set())
        if members:
            view.objects.active=next(iter(members))
            finished(bpy.ops.object.mode_set(mode='OBJECT'),'Leave temporary bone-tool mode')
    for key in changed:
        members=expected.get(key,set())
        if members:
            for obj in view.objects:
                if obj is not None:obj.select_set(obj in members,view_layer=view)
            view.objects.active=next(iter(members))
            finished(bpy.ops.object.mode_set(mode=key[0]),'Restore original bone-tool mode')
    if groups((obj,obj.mode) for obj in view.objects if obj is not None)!=expected:
        raise MCPError('Could not restore original Pose/Edit object membership')

@contextmanager
def bone_context(armature_name,bone_names,scope='SELECTED',writable=True):
    names=names_argument(bone_names,scope);context=bpy.context;view=context.view_layer
    arm=named_armature(armature_name)
    if writable and (arm.library or arm.data.library):raise MCPError('Make the armature and its data local before editing bone physics')
    if arm.hide_get(view_layer=view) or arm.hide_select:raise MCPError('Unhide the armature and allow object selection before using bone tools')
    active=view.objects.active;mode=active.mode if active else 'OBJECT'
    objects=[(obj,obj.select_get(view_layer=view)) for obj in view.objects]
    object_modes=[(obj,obj.mode) for obj in view.objects]
    # Some existing mode members cannot be re-entered while hidden or locked.
    # Use Blender's effective selectability, including collection restrictions.
    selectable=set(context.selectable_objects or ())
    blocked=[obj.name for obj,saved_mode in object_modes if saved_mode in ('POSE','EDIT') and obj not in selectable]
    if blocked:
        raise MCPError('Cannot safely restore Pose/Edit context for hidden or unselectable objects: '
                       +', '.join(sorted(blocked))+'. Finish their Pose/Edit modes or make them visible and selectable before retrying; '
                       'no bone operation was executed')
    bone_selection=[(obj,_bone_selection(obj)) for obj in view.objects if obj.type=='ARMATURE']
    try:
        if mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
        # Deselecting an object does not remove its bones from multi-Pose context.
        # Exit existing Pose cohorts and commit a non-active target's Edit buffer.
        for obj in tuple(view.objects):
            if obj is not None and (obj.mode=='POSE' or obj==arm and obj.mode=='EDIT'):
                view.objects.active=obj
                finished(bpy.ops.object.mode_set(mode='OBJECT'),'Isolate named armature')
        missing=[name for name in names if arm.data.bones.get(name) is None]
        if missing:raise MCPError('Unknown bone_names: '+', '.join(missing))
        for obj,_selected in objects:obj.select_set(False,view_layer=view)
        arm.select_set(True,view_layer=view);view.objects.active=arm
        bpy.ops.object.mode_set(mode='POSE')
        for p in arm.pose.bones:set_pose_bone_selected(p,p.name in names or scope=='ALL')
        arm.data.bones.active=arm.data.bones.get(names[0]) if names else next(iter(arm.data.bones),None)
        from ...ops.bone_proxy import selected_bones
        _arm,bones=selected_bones(context,scope=scope)
        if scope=='SELECTED' and {b.name for b in bones}!=set(names):
            # Pose context filters hidden bones/collections. Never apply only the
            # visible subset of an explicitly named batch. Leave visibility alone.
            raise MCPError('Requested bones are unavailable in the current pose selection. '
                           'Unhide the requested bones and their bone collections, then retry; '
                           'no bone operation was executed')
        yield arm,bones
    finally:
        try:
            current=view.objects.active
            if current is not None and current.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
            for obj,saved in bone_selection:_restore_bones(obj,saved)
            _restore_mode_members(view,object_modes)
            for obj,selected in objects:
                try:obj.select_set(selected,view_layer=view)
                except ReferenceError:pass # Explicit DELETE may remove an originally selected proxy.
            try:view.objects.active=active
            except ReferenceError:view.objects.active=None
            if view.objects.active is not None and mode not in ('OBJECT','POSE','EDIT'):bpy.ops.object.mode_set(mode=mode)
            for obj,saved in bone_selection:_restore_bones(obj,saved)
        except (RuntimeError,ReferenceError) as exc:
            raise MCPError('Could not restore original object/bone context: '+str(exc)) from exc

def finished(result,operation):
    if 'FINISHED' not in result:raise MCPError(operation+' was cancelled; resolve the production operator error before retrying')

def push_undo(label):
    if not bpy.ops.ed.undo_push.poll():raise MCPError('Undo is unavailable in this Blender context')
    # The outer native operator records its final context and data together.
    if _undo_execution_depth:
        mark_bone_undo_commit()
        return
    finished(bpy.ops.ed.undo_push(message=label),label)

@contextmanager
def preset_operator_state(preset_id,scope,size_mode):
    """Populate the existing UI operator's RNA input, then restore its UI cache."""
    from ...core import bone_preset_library as lib
    document=lib.load_preset(preset_id)
    state=bpy.context.window_manager.ppf_bone_preset_library
    old=dict(entries=[(r.name,r.preset_id,r.valid,r.error) for r in state.entries],active_index=state.active_index,scope=state.scope,size_mode=state.size_mode,status=state.status)
    try:
        state.entries.clear();item=state.entries.add();item.name=document['name'];item.preset_id=preset_id;item.valid=True
        state.active_index=0;state.scope=scope;state.size_mode=size_mode
        yield document
    finally:
        state.entries.clear()
        for name,identifier,valid,error in old['entries']:
            item=state.entries.add();item.name=name;item.preset_id=identifier;item.valid=valid;item.error=error
        for key in ('active_index','scope','size_mode','status'):setattr(state,key,old[key])
