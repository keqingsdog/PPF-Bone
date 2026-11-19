"""Named MCP access; all validation/transactions belong to the author module."""
import bpy
from typing import Optional
from ..decorators import mcp_handler,MCPError
from ._bone_context import named_armature,push_undo
from ...core import cross_links as author

WRITE=dict(readOnlyHint=False,destructiveHint=True,idempotentHint=False,openWorldHint=False)


def _undo_available():
    if not bpy.ops.ed.undo_push.poll():raise MCPError('Undo is unavailable; no cross-link data was changed')


@mcp_handler
def get_bone_cross_links(armature_name:str):
    """List stable link UUIDs, SI coefficients, author references and endpoint errors."""
    return dict(armature=armature_name,links=author.describe(bpy.context,named_armature(armature_name)))


@mcp_handler(annotations=WRITE)
def create_bone_cross_links(armature_name:str,bone_names:list[str],mode:str='PAIR',diagonals:bool=False,use_mass_defaults:bool=True,stiffness_N_per_m:float=100.,damping_Ns_per_m:float=1.,tension_enabled:bool=True,compression_enabled:bool=True,tension_stiffness_N_per_m:Optional[float]=None,compression_stiffness_N_per_m:Optional[float]=None,exclude_collision:bool=False):
    """Create PAIR or equal-length unbranched PLANE/TUBE networks (>=2/3 chains).

    Bone names explicitly select participating endpoints in the named rig.
    Defaults measure closed author proxies; later edits are not auto-overwritten.
    """
    try:
        _undo_available()
        ids=author.create(bpy.context,named_armature(armature_name),bone_names,mode,diagonals,use_mass_defaults,stiffness_N_per_m,damping_Ns_per_m,tension_enabled,compression_enabled,tension_stiffness_N_per_m,compression_stiffness_N_per_m,exclude_collision)
        push_undo('Create bone cross links')
        return dict(created=ids)
    except ValueError as exc:raise MCPError(str(exc)) from exc


@mcp_handler(annotations=WRITE)
def edit_bone_cross_links(armature_name:str,link_uuids:list[str],action:str='SET',parameters:Optional[dict]=None):
    """SET/DELETE/RECAPTURE/DEFAULTS explicit link UUIDs atomically.

    SET parameters: enabled, tension_enabled, compression_enabled, tension_stiffness_N_per_m, compression_stiffness_N_per_m, damping_Ns_per_m, exclude_collision. Legacy stiffness_N_per_m sets both directions.
    RECAPTURE changes author preset metadata, not simulation pretension.
    """
    try:
        _undo_available()
        count=author.edit(bpy.context,named_armature(armature_name),link_uuids,action,**(parameters or {}));push_undo('Edit bone cross links')
        return dict(edited=count)
    except ValueError as exc:raise MCPError(str(exc)) from exc


@mcp_handler(annotations=WRITE)
def bone_cross_links_preset(armature_name:str,action:str='CAPTURE',document:Optional[dict]=None,bone_map:Optional[dict]=None):
    """CAPTURE returns the full portable network; APPLY replaces it atomically.

    Different rigs require an explicit source-bone-name -> target-name map.
    Same rig resolves UUIDs, preserving renamed bones and author references.
    """
    try:
        arm=named_armature(armature_name)
        if action=='CAPTURE':return author.preset(bpy.context,arm)
        if action!='APPLY':raise ValueError('action must be CAPTURE or APPLY')
        _undo_available()
        count=author.apply_preset(bpy.context,arm,document,bone_map);push_undo('Apply cross link preset');return dict(links=count)
    except ValueError as exc:raise MCPError(str(exc)) from exc
