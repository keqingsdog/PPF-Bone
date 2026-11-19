# Fork modification: changed from the upstream baseline.
# File: group_ops.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

from ...i18n import props as _ppf_props
from ... import i18n as _ppf_i18n
import bpy  # pyright: ignore

from bpy.types import Operator  # pyright: ignore
from ...i18n import pgettext_iface as iface_, pgettext_tip as tip_

from ...models.collection_utils import safe_update_index
from ...models.groups import get_addon_data
from ..state import (
    N_MAX_GROUPS,
    ObjectGroup,
    assign_display_indices,
    find_available_group_slot,
    iterate_active_object_groups,
)
from .utils import cleanup_group_references_for_object, get_group_from_index, reset_object_display


class OBJECT_OT_CreateGroup(Operator):
    """Create a new dynamics group"""

    bl_idname = "object.create_group"
    bl_label = "Create Group"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        slot_index = find_available_group_slot(scene)
        if slot_index is None:
            self.report({"WARNING"}, iface_("Maximum number of groups reached"))
            return {"CANCELLED"}

        prop_name = f"object_group_{slot_index}"
        group: ObjectGroup = getattr(get_addon_data(scene), prop_name)
        group.reset_to_defaults()
        group.active = True
        group.ensure_uuid()
        # Don't set name - leave it empty so UI shows "Group N" dynamically

        # Publish which group this call created. Callers cannot recover it by
        # scanning: the slot allocated is the lowest free one, so once any
        # group has been deleted it is no longer the highest active slot, and
        # a scan would hand back some other group to be renamed and configured.
        get_addon_data(scene).state.current_group_uuid = group.uuid

        assign_display_indices(scene)

        # Update overlays when creating a new group
        from .overlay import apply_object_overlays

        apply_object_overlays()

        return {"FINISHED"}


class OBJECT_OT_DeleteGroup(Operator):
    """Delete the active group"""

    bl_idname = "object.delete_group"
    bl_label = "Delete Group"
    bl_options = {"REGISTER", "UNDO"}

    group_index: _ppf_props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    @staticmethod
    def _target_identity(scene, group):
        # Scalars only: retaining an RNA wrapper across Undo is unsafe.
        return (scene.as_pointer(), bpy.data.filepath, group.as_pointer(),
                group.uuid, int(group.get("_lifecycle_generation", 0)),
                _cleanup_generation)

    def invoke(self, context, event):
        group = get_group_from_index(context.scene, self.group_index)
        if group is not None:
            self._delete_confirmation_target = self._target_identity(context.scene, group)
        else:
            self._delete_confirmation_target = None
        if group and len(group.assigned_objects) > 0:
            return context.window_manager.invoke_confirm(self, event)
        return self.execute(context)

    def execute(self, context):
        from .overlay import apply_object_overlays

        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            self.report({"ERROR"}, iface_("Group not found or not active"))
            return {"CANCELLED"}

        confirmed = getattr(self, "_delete_confirmation_target", None)
        if confirmed is not None and confirmed != self._target_identity(scene, group):
            self.report({"WARNING"}, iface_("Scene or group changed; confirm deletion again"))
            return {"CANCELLED"}

        # Reset object display properties
        for obj_ref in group.assigned_objects:
            from ...core.uuid_registry import resolve_assigned
            obj = resolve_assigned(obj_ref)
            if obj:
                reset_object_display(obj)

        # Delete the group by resetting to defaults (which now sets active=False)
        group.reset_to_defaults()
        assign_display_indices(scene)
        apply_object_overlays()

        return {"FINISHED"}


class OBJECT_OT_DuplicateGroup(Operator):
    """Duplicate a dynamics group (material params only, no objects or pins)"""

    bl_idname = "object.duplicate_group"
    bl_label = "Duplicate Group"
    bl_options = {"REGISTER", "UNDO"}

    group_index: _ppf_props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    @staticmethod
    def _next_name(name: str) -> str:
        """Generate incremented name: foo -> foo-1, foo-1 -> foo-2."""
        import re

        m = re.match(r"^(.*)-(\d+)$", name)
        if m:
            return f"{m.group(1)}-{int(m.group(2)) + 1}"
        return f"{name}-1" if name else ""

    def execute(self, context):
        from ...core.profile import apply_material_profile, read_material_profile
        from .overlay import apply_object_overlays

        scene = context.scene
        src = get_group_from_index(scene, self.group_index)
        if src is None:
            self.report({"ERROR"}, iface_("Source group not found"))
            return {"CANCELLED"}

        slot_index = find_available_group_slot(scene)
        if slot_index is None:
            self.report({"WARNING"}, iface_("Maximum number of groups reached"))
            return {"CANCELLED"}

        addon_data = get_addon_data(scene)
        dst = getattr(addon_data, f"object_group_{slot_index}")
        dst.reset_to_defaults()
        dst.active = True
        dst.ensure_uuid()
        dst.name = self._next_name(src.name)

        # Copy material params only
        mat_data = read_material_profile(src)
        apply_material_profile(mat_data, dst)

        assign_display_indices(scene)
        apply_object_overlays()
        return {"FINISHED"}


class OBJECT_OT_DeleteAllGroups(Operator):
    """Delete all groups"""

    bl_idname = "object.delete_all_groups"
    bl_label = "Delete All Groups"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        # iterate_active_object_groups already yields only active groups,
        # so poll is True iff it yields at least one.
        return next(iterate_active_object_groups(context.scene), None) is not None

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        from .overlay import apply_object_overlays

        scene = context.scene

        for i in range(N_MAX_GROUPS):
            prop_name = f"object_group_{i}"
            group: ObjectGroup | None = getattr(get_addon_data(scene), prop_name, None)
            if group and group.active:
                for obj_ref in group.assigned_objects:
                    from ...core.uuid_registry import resolve_assigned
                    obj = resolve_assigned(obj_ref)
                    if obj:
                        reset_object_display(obj)

                group.reset_to_defaults()

        get_addon_data(scene).state.current_group_uuid = ""
        apply_object_overlays()

        return {"FINISHED"}


class OBJECT_OT_AddObjectsToGroup(Operator):
    """Add selected objects to the active group"""

    bl_idname = "object.add_objects_to_group"
    bl_label = "Add Selected Objects"
    bl_options = {"REGISTER", "UNDO"}

    group_index: _ppf_props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    @classmethod
    def poll(cls, context):
        return len(context.selected_objects) > 0

    def execute(self, context):
        from .overlay import apply_object_overlays

        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is not None:
            from ...core.uuid_registry import get_or_create_object_uuid, UUIDOwnershipError
            # Resolve the entire explicit selection before changing membership
            # or display. A late identity refusal must not commit a prefix.
            # New local UUID allocation is allowed; it is not an author reset.
            try:
                selection_plan = [
                    (obj, get_or_create_object_uuid(obj))
                    for obj in list(context.selected_objects)
                ]
            except UUIDOwnershipError as exc:
                self.report({"ERROR"}, str(exc))
                return {"CANCELLED"}
            existing_uuids = {a.uuid for a in group.assigned_objects if a.uuid}
            group_uuid = group.ensure_uuid()

            from ...core.utils import (
                count_duplicate_faces,
                find_linked_duplicate_siblings,
            )
            for obj, obj_uid in selection_plan:
                is_acceptable = obj.type == "MESH" or (
                    obj.type == "CURVE" and group.object_type == "ROD"
                )
                if not obj_uid:
                    self.report(
                        {"WARNING"},
                        iface_("Object '{name}' is library-linked and cannot be assigned").format(name=obj.name),
                    )
                    continue
                if not is_acceptable or obj_uid in existing_uuids:
                    continue

                # A SAND group accepts ONLY a committed particle mesh: a
                # faceless cloud of loose vertices (grain centers) with the
                # ``particle_mesh`` stamp, no polygons and no edges. A
                # raw solid mesh must be converted first (Convert To Solid
                # Particle Mesh in the group panel).
                if group.object_type == "SAND":
                    is_particle_mesh = (
                        obj.type == "MESH"
                        and obj.get("particle_mesh")
                        and len(obj.data.polygons) == 0
                        and len(obj.data.edges) == 0
                    )
                    if not is_particle_mesh:
                        self.report(
                            {"ERROR"},
                            iface_("Object '{name}' is not a particle mesh. Convert to particle mesh first.").format(name=obj.name),
                        )
                        continue

                # Reject shallow-copied (Alt-D / Linked Duplicate) objects
                # before they can be assigned. Shared mesh data would let
                # the encoder ship inconsistent vertex coords across two
                # "different" assigned objects and silently corrupt PC2
                # playback at fetch time; the only safe fix is for the
                # user to make the duplicate single-user via Object > Make
                # Single User > Object & Data.
                siblings = find_linked_duplicate_siblings(obj)
                if siblings:
                    self.report(
                        {"ERROR"},
                        iface_("Object '{name}' is a Linked Duplicate (shares mesh data with {sibling!r}). Make it single-user (Object > Relations > Make Single User > Object & Data) before assigning.").format(name=obj.name, sibling=siblings[0]),
                    )
                    continue

                # Reject doubled geometry (two triangles sharing the same
                # three vertices). The solver builds a degenerate bending
                # element from coincident faces and aborts at startup;
                # this is the airbag/inflate "Merge by Distance leftover"
                # case. Refuse it here rather than silently dropping faces
                # the user may have placed intentionally.
                dup_count = count_duplicate_faces(obj)
                if dup_count > 0:
                    self.report(
                        {"ERROR"},
                        iface_("Object '{name}' has {count} duplicate face(s) (two triangles sharing the same three vertices), usually from doubled geometry. Select the mesh in Edit Mode and run Mesh > Merge > By Distance (or delete the duplicate faces) before assigning.").format(name=obj.name, count=dup_count),
                    )
                    continue

                in_other = False
                for other_group in iterate_active_object_groups(scene):
                    if other_group.ensure_uuid() != group_uuid:
                        other_uuids = {a.uuid for a in other_group.assigned_objects if a.uuid}
                        if obj_uid in other_uuids:
                            self.report(
                                {"WARNING"},
                                iface_("Object '{name}' is already in another group").format(name=obj.name),
                            )
                            in_other = True
                            break
                if in_other:
                    continue

                item = group.assigned_objects.add()
                item.name = obj.name
                item.uuid = obj_uid
                if obj.type == "MESH":
                    obj.show_wire = True
                    obj.show_all_edges = True
                    # Wireframe requires faces: on an edge-only ROD it
                    # produces an empty evaluated mesh. Preserve authored
                    # modifiers; only gate this automatic visualization.
                    if (group.object_type == "ROD" and obj.data.polygons
                            and not any(m.type == "WIREFRAME" for m in obj.modifiers)):
                        wire = obj.modifiers.new(name="Wireframe", type="WIREFRAME")
                        # Even-offset compensates sharp dihedrals by extruding
                        # further; rod simulations fold/crease faces freely
                        # (only edge length is constrained), and that turns
                        # into wild visual jumps in the wireframe geometry.
                        # Uniform thickness keeps the visualization stable.
                        wire.use_even_offset = False
                        new_idx = len(obj.modifiers) - 1
                        if new_idx > 0:
                            obj.modifiers.move(new_idx, 0)
                # Set object color to group color if overlay is enabled
                if group.show_overlay_color:
                    obj.color = group.color

            apply_object_overlays()

            # The per-object overlay tint just written above only
            # renders when the Solid shading sub-mode is OBJECT. The
            # user signed up for the tint by clicking Add, so flip
            # every Solid VIEW_3D's color sub-mode to OBJECT as a
            # one-shot here. We do not touch non-SOLID modes
            # (Material / Rendered previews use materials anyway),
            # and we never re-flip on every redraw / undo — that was
            # the prior bug fixed in overlay.py. See
            # feedback_narrow_user_owned_state_writes for the rule:
            # writes to user-owned viewport state are allowed inside
            # an explicit user-invoked operator's execute().
            if group.show_overlay_color:
                for win in bpy.context.window_manager.windows:
                    for area in win.screen.areas:
                        if area.type != "VIEW_3D":
                            continue
                        for sp in area.spaces:
                            if sp.type != "VIEW_3D":
                                continue
                            if sp.shading.type == "SOLID":
                                sp.shading.color_type = "OBJECT"

            return {"FINISHED"}

        self.report({"ERROR"}, iface_("Group not found"))
        return {"CANCELLED"}


class OBJECT_OT_RemoveObjectFromGroup(Operator):
    """Remove the selected object from the group"""

    bl_idname = "object.remove_object_from_group"
    bl_label = "Remove Object"
    bl_options = {"REGISTER", "UNDO"}

    group_index: _ppf_props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    def execute(self, context):
        from .overlay import apply_object_overlays
        from ...core.pc2 import cleanup_mesh_cache

        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is not None:
            index = group.assigned_objects_index
            if 0 <= index < len(group.assigned_objects):
                assigned = group.assigned_objects[index]
                from ...core.uuid_registry import resolve_assigned
                obj = resolve_assigned(assigned)
                obj_uuid = assigned.uuid
                if obj:
                    reset_object_display(obj)
                    cleanup_mesh_cache(obj)

                group.assigned_objects.remove(index)
                cleanup_group_references_for_object(group, obj_uuid)
                group.assigned_objects_index = safe_update_index(
                    index, len(group.assigned_objects)
                )

            apply_object_overlays()
            return {"FINISHED"}

        self.report({"ERROR"}, iface_("Group not found"))
        return {"CANCELLED"}


_cleanup_scheduled = False
_cleanup_registered = False
_cleanup_generation = 0
_cleanup_rebuilding = False
_cleanup_timer = None
_cleanup_identity_warning = None


def cleanup_identity_rebuilding():
    return _cleanup_rebuilding


def _cancel_cleanup_timer():
    global _cleanup_scheduled, _cleanup_timer
    if _cleanup_timer is not None and bpy.app.timers.is_registered(_cleanup_timer):
        bpy.app.timers.unregister(_cleanup_timer)
    _cleanup_timer = None
    _cleanup_scheduled = False


def _schedule_cleanup(*, delay=0.0, resume=False):
    global _cleanup_scheduled, _cleanup_timer
    if not _cleanup_registered or _cleanup_scheduled:
        return
    generation = _cleanup_generation
    # Capture only an integer ticket, never a scene/object/RNA wrapper across Undo.
    def tick():
        global _cleanup_rebuilding
        if generation != _cleanup_generation or not _cleanup_registered:
            return None
        if resume:
            _cleanup_rebuilding = False
        return _apply_cleanup(generation)
    _cleanup_scheduled = True
    _cleanup_timer = tick
    bpy.app.timers.register(tick, first_interval=delay)


@bpy.app.handlers.persistent
def _cleanup_before_identity_rebuild(*_args):
    global _cleanup_generation, _cleanup_rebuilding, _cleanup_identity_warning
    _cleanup_identity_warning = None
    _cleanup_generation += 1
    _cleanup_rebuilding = True
    _cancel_cleanup_timer()


@bpy.app.handlers.persistent
def _cleanup_after_identity_rebuild(*_args):
    # The post handler does not inspect RNA or force dependency-graph evaluation.
    _schedule_cleanup(delay=0.1, resume=True)


def _apply_cleanup(generation=None):
    """Deferred cleanup body — runs from a timer, outside depsgraph context.

    Blender 5.x forbids ID writes (collection mutation, obj.color) from
    depsgraph_update_post. This timer tick fires after the depsgraph
    evaluation completes, so the writes are allowed.
    """
    global _cleanup_scheduled, _cleanup_timer, _cleanup_identity_warning
    if (not _cleanup_registered or _cleanup_rebuilding
            or generation is not None and generation != _cleanup_generation):
        return None

    from ...core.bone_author_follow import _transform_active
    if _transform_active(bpy.context.scene):
        return 0.1
    _cleanup_scheduled = False
    _cleanup_timer = None

    from .overlay import apply_object_overlays
    from ...core.uuid_registry import resolve_assigned

    scene = bpy.context.scene
    if scene is None:
        return None
    # No reconciliation, name synchronization or group removal may precede
    # the complete identity admission. Ambiguity is not a missing object.
    from ...core.uuid_registry import get_object_by_uuid, UUIDOwnershipError
    try:
        for group in iterate_active_object_groups(scene):
            for assigned in group.assigned_objects:
                get_object_by_uuid(assigned.uuid)
    except UUIDOwnershipError as exc:
        warning = (int(scene.session_uid), str(exc))
        if warning != _cleanup_identity_warning:
            print(f"PPF group cleanup paused: {exc}")
            _cleanup_identity_warning = warning
        # The Groups panel shows the same conflict without binding an owner.
        # Do not schedule an endless retry or continue cleaning a valid prefix.
        return None
    _cleanup_identity_warning = None
    from ...ops.bone_proxy import isolate_copied_armature_bindings, reconcile_deleted_proxies
    isolate_copied_armature_bindings(scene)
    reconcile_deleted_proxies(scene, strict=False)
    changed = False

    for group in iterate_active_object_groups(scene):
        selected_index = group.assigned_objects_index
        indices_to_remove = []
        for i in range(len(group.assigned_objects)):
            assigned = group.assigned_objects[i]
            # resolve_assigned() also calls _sync_all_names() when
            # assigned.name drifts from obj.name, so every downstream
            # reference (pin_items, merge_pairs, saved keyframes) is kept
            # coherent through the single reconciler in core/uuid_registry.
            # Do NOT write assigned.name directly here: it would skip siblings.
            if resolve_assigned(assigned) is None:
                indices_to_remove.append(i)
        selected_survives = (
            0 <= selected_index < len(group.assigned_objects)
            and selected_index not in indices_to_remove
        )
        for i in reversed(indices_to_remove):
            obj_uuid = group.assigned_objects[i].uuid
            group.assigned_objects.remove(i)
            cleanup_group_references_for_object(group, obj_uuid)
            changed = True
        if indices_to_remove:
            if selected_survives:
                group.assigned_objects_index = selected_index - sum(
                    i < selected_index for i in indices_to_remove
                )
            else:
                group.assigned_objects_index = safe_update_index(
                    selected_index, len(group.assigned_objects)
                )

    # ``obj.color`` is NOT touched here. The add-on writes a display color
    # only on an object that is in a group, and clears it while that object
    # is still a member, at the point membership ends: every operator that
    # disowns an object calls ``reset_object_display`` before dropping the
    # assignment. An object that is not in a group is the user's, so a
    # scene-wide pass has nothing it is allowed to write. Scanning for a
    # stray tint instead of clearing it at the source is what made the
    # Object Properties color field unsettable on every mesh in the file,
    # since a non-member and a never-tinted object are indistinguishable
    # from here.
    #
    # A copy made with Duplicate is the one case left uncovered: it
    # inherits the color of the member it was copied from and belongs to no
    # group, so it keeps that color until the user changes it. That is a
    # stale color on an object the add-on does not own, which is the side
    # to err on.

    # Merge pairs are not scanned here. A pair ends with its object's
    # membership, which the loop above ends through
    # ``cleanup_group_references_for_object`` for a deleted object. A pair
    # whose object sits in a DEACTIVATED group is the artist's, kept for when
    # the group is switched back on, and Transfer refuses it by name meanwhile
    # (``merge_pair_problem``).

    if changed:
        apply_object_overlays()
    return None


@bpy.app.handlers.persistent
def _cleanup_deleted_objects(scene, depsgraph):
    """Schedule deferred cleanup of stale group references.

    Blender 5.x disallows ID writes from depsgraph handlers, so we only
    mark that cleanup is needed and let a one-shot timer run the actual
    mutations from a permissive context.
    """
    if not _cleanup_registered or _cleanup_scheduled or _cleanup_rebuilding:
        return
    _schedule_cleanup()


classes = (
    OBJECT_OT_CreateGroup,
    OBJECT_OT_DeleteGroup,
    OBJECT_OT_DuplicateGroup,
    OBJECT_OT_DeleteAllGroups,
    OBJECT_OT_AddObjectsToGroup,
    OBJECT_OT_RemoveObjectFromGroup,
)


@bpy.app.handlers.persistent
def _reset_cleanup_flag_on_load(*_args):
    """Clear the single-shot guard when a new .blend loads.

    If a file is saved mid-cycle (flag True), then reopened, the stale
    flag would suppress the next legitimate cleanup scheduling.
    """
    global _cleanup_generation, _cleanup_rebuilding
    _cleanup_generation += 1
    _cleanup_rebuilding = False
    _cancel_cleanup_timer()


def register():
    global _cleanup_registered, _cleanup_rebuilding, _cleanup_generation
    for cls in classes:
        _ppf_i18n.register_class(cls)
    _cleanup_registered = True
    _cleanup_rebuilding = False
    _cleanup_generation += 1
    _cancel_cleanup_timer()
    bpy.app.handlers.depsgraph_update_post.append(_cleanup_deleted_objects)
    for h in list(bpy.app.handlers.load_post):
        if getattr(h, "__name__", "") == "_reset_cleanup_flag_on_load":
            bpy.app.handlers.load_post.remove(h)
    bpy.app.handlers.load_post.append(_reset_cleanup_flag_on_load)
    for handlers, fn in ((bpy.app.handlers.undo_pre, _cleanup_before_identity_rebuild),
                         (bpy.app.handlers.redo_pre, _cleanup_before_identity_rebuild),
                         (bpy.app.handlers.undo_post, _cleanup_after_identity_rebuild),
                         (bpy.app.handlers.redo_post, _cleanup_after_identity_rebuild)):
        for old in list(handlers):
            if getattr(old, '__name__', '') == fn.__name__:
                handlers.remove(old)
        handlers.append(fn)


def unregister():
    global _cleanup_registered, _cleanup_scheduled
    _cleanup_registered = False
    _cancel_cleanup_timer()
    if bpy.app.timers.is_registered(_apply_cleanup):
        bpy.app.timers.unregister(_apply_cleanup)
    _cleanup_scheduled = False
    # Remove all instances (handles reload where function identity changes)
    handlers = bpy.app.handlers.depsgraph_update_post
    for h in list(handlers):
        if getattr(h, "__name__", "") == "_cleanup_deleted_objects":
            handlers.remove(h)
    for h in list(bpy.app.handlers.load_post):
        if getattr(h, "__name__", "") == "_reset_cleanup_flag_on_load":
            bpy.app.handlers.load_post.remove(h)
    for handlers in (bpy.app.handlers.undo_pre, bpy.app.handlers.redo_pre,
                     bpy.app.handlers.undo_post, bpy.app.handlers.redo_post):
        for fn in list(handlers):
            if getattr(fn, '__name__', '') in ('_cleanup_before_identity_rebuild',
                                              '_cleanup_after_identity_rebuild'):
                handlers.remove(fn)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
