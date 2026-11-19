from ...core.pin_reference import decode_stored_pin_identifier, PinReferenceError, encode_pin_pair, validate_pin_name_for_write, validate_stored_pin_items
# Fork modification: changed from the upstream baseline.
# File: pin_ops.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

from ...i18n import props as _ppf_props
from ... import i18n as _ppf_i18n
import json

import bpy  # pyright: ignore

from bpy.types import Operator  # pyright: ignore

from ...i18n import pgettext_iface as iface_, pgettext_tip as tip_

from ...core.utils import get_vertices_in_group, redraw_all_areas
from ...models.collection_utils import safe_update_index
from ...models.enum_props import EnumProperty
from ..state import iterate_active_object_groups
from .utils import get_group_from_index


def pin_already_exists(group, obj_uuid: str, vg_name: str) -> bool:
    """Return True when *group* already holds a pin for (obj_uuid, vg_name).

    Identity is keyed on the object UUID plus the decoded vertex-group
    name, never on the raw encoded identifier. The identifier embeds the
    object's display name, so an object (or content-hashed vertex group)
    rename changes the stored string while the logical pin is unchanged.
    Keying on UUID + decoded name keeps this consistent with
    _raw_create_pin and the MCP handler add_pin_vertex_group.
    """
    from ...models.groups import decode_vertex_group_identifier

    for item in group.pin_vertex_groups:
        if item.object_uuid != obj_uuid:
            continue
        _, item_vg = decode_vertex_group_identifier(item.name)
        if item_vg == vg_name:
            return True
    return False


class OBJECT_OT_CreatePinVertexGroup(Operator):
    """Create a vertex group from selected vertices and add it to the pin list"""

    bl_idname = "object.create_pin_vertex_group"
    bl_label = "Create"
    bl_options = {"REGISTER", "UNDO"}

    group_index: _ppf_props.IntProperty(options={'HIDDEN'})  # pyright: ignore
    vg_name: _ppf_props.StringProperty(  # pyright: ignore
        name="Vertex Group Name",
        default="pin",
        description="Name for the new vertex group",
    )

    @classmethod
    def poll(cls, context):
        if context.mode == "EDIT_MESH":
            obj = context.edit_object
            if obj is None or obj.type != "MESH":
                return False
            import bmesh
            bm = bmesh.from_edit_mesh(obj.data)
            if not any(v.select for v in bm.verts):
                return False
        elif context.mode == "EDIT_CURVE":
            obj = context.edit_object
            if obj is None or obj.type != "CURVE":
                return False
            # Check if any control points are selected
            has_selected = False
            for s in obj.data.splines:
                if s.type == "BEZIER":
                    if any(bp.select_control_point for bp in s.bezier_points):
                        has_selected = True
                        break
                elif s.type in ("NURBS", "POLY"):
                    if any(p.select for p in s.points):
                        has_selected = True
                        break
            if not has_selected:
                return False
        else:
            return False
        obj = context.edit_object
        # Check the object belongs to at least one active group
        from ...core.uuid_registry import get_object_uuid
        obj_uuid = get_object_uuid(obj)
        if not obj_uuid:
            return False
        scene = context.scene
        for group in iterate_active_object_groups(scene):
            if group.object_type == "STATIC":
                continue
            for assigned in group.assigned_objects:
                if assigned.uuid == obj_uuid:
                    return True
        return False

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        self.layout.prop(self, "vg_name")

    def execute(self, context):
        from .overlay import apply_object_overlays

        obj = context.edit_object
        if obj is None:
            self.report({"ERROR"}, iface_("No active edit object"))
            return {"CANCELLED"}

        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            self.report({"ERROR"}, iface_("Group not found"))
            return {"CANCELLED"}

        # Check this object is in the group (UUID comparison)
        from ...core.uuid_registry import get_object_uuid as _get_uuid
        _obj_uid = _get_uuid(obj)
        if not _obj_uid or not any(a.uuid == _obj_uid for a in group.assigned_objects):
            self.report({"ERROR"}, iface_("'{name}' is not in this group").format(name=obj.name))
            return {"CANCELLED"}

        name = self.vg_name.strip()
        if not name:
            self.report({"ERROR"}, iface_("Name cannot be empty"))
            return {"CANCELLED"}

        try:
            validate_pin_name_for_write(obj, name)
            validate_stored_pin_items(group.pin_vertex_groups)
            from ...core.uuid_registry import get_object_by_uuid
            if get_object_by_uuid(_obj_uid) is not obj:
                raise PinReferenceError("Pin owner UUID is not proven")
        except PinReferenceError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        # Get selected indices
        if obj.type == "MESH":
            import bmesh
            bm = bmesh.from_edit_mesh(obj.data)
            selected = [v.index for v in bm.verts if v.select]
        elif obj.type == "CURVE":
            selected = []
            idx = 0
            for s in obj.data.splines:
                if s.type == "BEZIER":
                    for bp in s.bezier_points:
                        if bp.select_control_point:
                            selected.append(idx)
                        idx += 1
                elif s.type in ("NURBS", "POLY"):
                    for p in s.points:
                        if p.select:
                            selected.append(idx)
                        idx += 1
        else:
            selected = []

        if not selected:
            self.report({"ERROR"}, iface_("No vertices selected"))
            return {"CANCELLED"}

        # Switch to object mode
        bpy.ops.object.mode_set(mode="OBJECT")

        if obj.type == "MESH":
            # Use Blender vertex groups for meshes
            vg = obj.vertex_groups.get(name)
            if vg is None:
                vg = obj.vertex_groups.new(name=name)
            else:
                # vg.add(..., "REPLACE") only overwrites weights for the
                # listed indices, so clear stale membership first.
                vg.remove(list(range(len(obj.data.vertices))))
            if vg.name != name:
                raise PinReferenceError("Blender did not preserve the prevalidated Pin name; no Pin published")
            name = vg.name
            vg.add(selected, 1.0, "REPLACE")
        elif obj.type == "CURVE":
            # Store as custom property for curves (no native vertex group support)
            key = f"_pin_{name}"
            obj[key] = json.dumps(selected)

        # Add to pin list
        from ...core.uuid_registry import get_or_create_object_uuid, compute_vg_hash
        from ...models.groups import encode_vertex_group_identifier
        obj_uuid = get_or_create_object_uuid(obj)
        if not obj_uuid:
            self.report({"ERROR"}, iface_("'{name}' is not writable (library-linked)").format(name=obj.name))
            return {"CANCELLED"}
        new_hash = str(compute_vg_hash(obj, name))
        # Duplicate check by UUID + vg_name — consistent with _raw_create_pin
        # and MCP handler add_pin_vertex_group.
        if not pin_already_exists(group, obj_uuid, name):
            identifier = encode_vertex_group_identifier(obj.name, name)
            item = group.pin_vertex_groups.add()
            item.name = identifier
            item.object_uuid = obj_uuid
            item.vg_hash = new_hash

        bpy.ops.object.mode_set(mode="EDIT")
        apply_object_overlays()
        self.report({"INFO"}, iface_("Created pin '{name}' with {count} points").format(name=name, count=len(selected)))
        return {"FINISHED"}


class OBJECT_OT_AddPinVertexGroup(Operator):
    """Add selected vertex group to pin list"""

    bl_idname = "object.add_pin_vertex_group"
    bl_label = "Add"
    bl_options = {"REGISTER", "UNDO"}

    group_index: _ppf_props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    def execute(self, context):
        from .overlay import apply_object_overlays

        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            self.report({"ERROR"}, iface_("Group not found"))
            return {"CANCELLED"}

        from ...core.pin_picker_identity import resolve as resolve_selection
        from ...models.groups import encode_vertex_group_identifier
        from ...core.uuid_registry import get_or_create_object_uuid, compute_vg_hash

        try:
            validate_stored_pin_items(group.pin_vertex_groups)
        except PinReferenceError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        selected = resolve_selection(group)
        if selected is None:
            self.report({"ERROR"}, iface_("Pin selection is unavailable or unproven; select the vertex group again"))
            return {"CANCELLED"}
        obj, vg_name = selected

        try:
            validate_pin_name_for_write(obj, vg_name)
        except PinReferenceError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        resolved_uuid = get_or_create_object_uuid(obj)
        if not resolved_uuid:
            self.report({"ERROR"}, iface_("'{name}' is not writable (library-linked)").format(name=obj.name))
            return {"CANCELLED"}

        # Duplicate check by (resolved UUID, vg_name) — consistent with
        # CreatePinVertexGroup, _raw_create_pin, and the MCP handler. The
        # stored identifier embeds the display object name, so a raw
        # string compare would let a rename slip the same logical pin past
        # this guard and add it twice.
        if not pin_already_exists(group, resolved_uuid, vg_name):
            item = group.pin_vertex_groups.add()
            item.name = encode_vertex_group_identifier(obj.name, vg_name)
            item.object_uuid = resolved_uuid
            if vg_name:
                item.vg_hash = str(compute_vg_hash(obj, vg_name))

        apply_object_overlays()
        return {"FINISHED"}


class OBJECT_OT_RemovePinVertexGroup(Operator):
    """Remove selected vertex group from pin list"""

    bl_idname = "object.remove_pin_vertex_group"
    bl_label = "Remove"
    bl_options = {"REGISTER", "UNDO"}

    group_index: _ppf_props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    def execute(self, context):
        from .overlay import apply_object_overlays

        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            self.report({"ERROR"}, iface_("Group not found"))
            return {"CANCELLED"}

        index = group.pin_vertex_groups_index
        if 0 <= index < len(group.pin_vertex_groups):
            group.pin_vertex_groups.remove(index)
            group.pin_vertex_groups_index = safe_update_index(index, len(group.pin_vertex_groups))

        apply_object_overlays()
        return {"FINISHED"}


class OBJECT_OT_MovePinVertexGroup(Operator):
    """Move the selected pin up or down in the list.

    Pin order matters: when pins overlap (share vertices), the pin lower
    in the list takes precedence for the shared vertices.
    """

    bl_idname = "object.move_pin_vertex_group"
    bl_label = "Move Pin"
    bl_options = {"REGISTER", "UNDO"}

    group_index: _ppf_props.IntProperty(options={'HIDDEN'})  # pyright: ignore
    direction: _ppf_props.IntProperty()  # pyright: ignore  # -1 = up, 1 = down

    def execute(self, context):
        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            return {"CANCELLED"}
        idx = group.pin_vertex_groups_index
        n = len(group.pin_vertex_groups)
        new_idx = idx + self.direction
        if idx < 0 or idx >= n or new_idx < 0 or new_idx >= n:
            return {"CANCELLED"}
        group.pin_vertex_groups.move(idx, new_idx)
        group.pin_vertex_groups_index = new_idx
        return {"FINISHED"}


class OBJECT_OT_RenamePinVertexGroup(Operator):
    """Rename a pin vertex group"""

    bl_idname = "object.rename_pin_vertex_group"
    bl_label = "Rename Pin"
    bl_options = {"REGISTER", "UNDO"}

    group_index: _ppf_props.IntProperty(options={'HIDDEN'})  # pyright: ignore
    new_name: _ppf_props.StringProperty(name="New Name")  # pyright: ignore

    def invoke(self, context, event):
        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            return {"CANCELLED"}
        idx = group.pin_vertex_groups_index
        if idx < 0 or idx >= len(group.pin_vertex_groups):
            return {"CANCELLED"}
        from ...core.uuid_registry import resolve_pin
        pin_item = group.pin_vertex_groups[idx]
        try:
            resolve_pin(pin_item)
        except PinReferenceError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        from ...models.groups import decode_vertex_group_identifier
        _, vg_name = decode_stored_pin_identifier(pin_item.name)
        self.new_name = vg_name or ""
        self._pin_rename_origin = (
            scene.as_pointer(), group.as_pointer(), group.uuid,
            pin_item.as_pointer(), pin_item.object_uuid, vg_name,
        )
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        from .overlay import apply_object_overlays
        from ...models.groups import decode_vertex_group_identifier
        from ...core.uuid_registry import resolve_pin, compute_vg_hash

        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            self.report({"ERROR"}, iface_("Group not found"))
            return {"CANCELLED"}

        idx = group.pin_vertex_groups_index
        if idx < 0 or idx >= len(group.pin_vertex_groups):
            self.report({"WARNING"}, iface_("No pin vertex group selected"))
            return {"CANCELLED"}

        pin_item = group.pin_vertex_groups[idx]
        new_vg = self.new_name.strip()
        if not new_vg:
            return {"CANCELLED"}
        try:
            from ...core.uuid_registry import plan_pin_reference
            plan = plan_pin_reference(pin_item)
            if plan is None:
                raise PinReferenceError("Cannot resolve Pin owner")
            planned_obj, _, planned_old, _ = plan
            validate_pin_name_for_write(planned_obj, new_vg)
            if new_vg != planned_old:
                exists = (planned_obj.vertex_groups.get(new_vg) is not None
                          if planned_obj.type == "MESH" else f"_pin_{new_vg}" in planned_obj)
                if exists:
                    raise PinReferenceError("Pin target name already exists; choose another name")
        except PinReferenceError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        origin = getattr(self, "_pin_rename_origin", None)
        try:
            _, current_vg = decode_stored_pin_identifier(pin_item.name)
        except PinReferenceError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        if origin is not None and origin != (
            scene.as_pointer(), group.as_pointer(), group.uuid,
            pin_item.as_pointer(), pin_item.object_uuid, current_vg,
        ):
            self.report({"WARNING"}, iface_("Pin rename target changed while the dialog was open; reopen Rename"))
            return {"CANCELLED"}
        # Captures are keyed by object UUID AND vertex-group name. Refuse
        # before resolve_pin can synchronize author RNA; never orphan a capture
        # by publishing a new name without a proven cache/proof migration.
        if new_vg != planned_old:
            try:
                from ...core.pc2 import has_pin_anim_pc2
                captured = bool(pin_item.has_captured_anim)
                if not captured:
                    # Both names come from this Pin's read-only identity plan,
                    # not from scanning or guessing historical cache filenames.
                    captured = any(has_pin_anim_pc2(planned_obj, name)
                                   for name in dict.fromkeys((plan[1], planned_old)))
            except Exception:
                captured = True  # Uncertain capture ownership must fail closed.
            if captured:
                self.report({"WARNING"}, iface_("Pin capture state is present or uncertain; rename was not applied. Keep its current name and review the capture."))
                return {"CANCELLED"}
        try:
            obj = resolve_pin(pin_item)
        except PinReferenceError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        obj_name, old_vg = decode_stored_pin_identifier(pin_item.name)
        if origin is not None and old_vg != origin[-1]:
            self.report({"WARNING"}, iface_("Pin vertex group changed while the dialog was open; reopen Rename"))
            return {"CANCELLED"}
        if not obj or not old_vg:
            self.report({"ERROR"}, iface_("Cannot resolve pin"))
            return {"CANCELLED"}

        new_vg = self.new_name.strip()
        if not new_vg or new_vg == old_vg:
            return {"CANCELLED"}

        # Rename the actual vertex group or curve custom property
        if obj.type == "MESH":
            vg = obj.vertex_groups.get(old_vg)
            if not vg:
                self.report({"ERROR"}, iface_("Vertex group '{name}' not found").format(name=old_vg))
                return {"CANCELLED"}
            if obj.vertex_groups.get(new_vg) is not None:
                self.report({"ERROR"}, iface_("Vertex group '{name}' already exists; choose a different name").format(name=new_vg))
                return {"CANCELLED"}
            capacity = bpy.types.VertexGroup.bl_rna.properties["name"].length_max - 1
            if capacity > 0 and len(new_vg.encode("utf-8")) > capacity:
                self.report({"ERROR"}, iface_("Vertex group name is too long ({limit} UTF-8 bytes maximum)").format(limit=capacity))
                return {"CANCELLED"}
            vg.name = new_vg
            if vg.name != new_vg:
                raise PinReferenceError("Blender changed the prevalidated Rename target; no Pin published")
            new_vg = vg.name  # Store Blender's actual name, never a stale request.
        elif obj.type == "CURVE":
            key_old = f"_pin_{old_vg}"
            key_new = f"_pin_{new_vg}"
            if key_old not in obj:
                self.report({"ERROR"}, iface_("Curve pin '{name}' is missing").format(name=old_vg))
                return {"CANCELLED"}
            if key_new in obj:
                self.report({"ERROR"}, iface_("Curve pin '{name}' already exists; choose a different name").format(name=new_vg))
                return {"CANCELLED"}
            try:
                obj[key_new] = obj[key_old]
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                self.report({"ERROR"}, str(exc))
                return {"CANCELLED"}
            del obj[key_old]
        else:
            self.report({"ERROR"}, iface_("Pin rename requires a mesh or curve object"))
            return {"CANCELLED"}

        # Update the uncaptured Pin identifier and hash. Captured Pins were
        # refused above because their PC2/proof lookup includes the group name.
        from ...models.groups import encode_vertex_group_identifier
        pin_item.name = encode_vertex_group_identifier(obj.name, new_vg)
        pin_item.vg_hash = str(compute_vg_hash(obj, new_vg))

        apply_object_overlays()
        self.report({"INFO"}, iface_("Renamed '{old_name}' to '{new_name}'").format(old_name=old_vg, new_name=new_vg))
        return {"FINISHED"}


def _pin_select(operator, context, select):
    """Select or deselect pin vertices in edit mode (like Blender's built-in)."""
    import bmesh

    from ...models.groups import decode_vertex_group_identifier

    scene = context.scene
    group = get_group_from_index(scene, operator.group_index)
    if group is None:
        operator.report({"ERROR"}, iface_("Group not found"))
        return {"CANCELLED"}

    idx = group.pin_vertex_groups_index
    if idx < 0 or idx >= len(group.pin_vertex_groups):
        operator.report({"WARNING"}, iface_("No pin vertex group selected"))
        return {"CANCELLED"}

    from ...core.uuid_registry import resolve_pin, get_object_uuid, get_object_by_uuid
    pin_item = group.pin_vertex_groups[idx]
    try:
        resolve_pin(pin_item)
    except PinReferenceError as exc:
        operator.report({"ERROR"}, str(exc))
        return {"CANCELLED"}
    _, vg_name = decode_stored_pin_identifier(pin_item.name)
    if not pin_item.object_uuid or not vg_name:
        operator.report({"ERROR"}, iface_("Invalid pin identifier"))
        return {"CANCELLED"}

    obj = context.edit_object
    if obj is None or get_object_uuid(obj) != pin_item.object_uuid:
        expected = get_object_by_uuid(pin_item.object_uuid)
        expected_name = expected.name if expected else "<unknown>"
        operator.report({"ERROR"}, iface_("Edit object must be '{name}'").format(name=expected_name))
        return {"CANCELLED"}

    # Get pin indices
    if obj.type == "CURVE":
        key = f"_pin_{vg_name}"
        raw = obj.get(key)
        pin_indices = set(json.loads(raw)) if raw else set()
    else:
        vg = obj.vertex_groups.get(vg_name)
        if not vg:
            operator.report({"ERROR"}, iface_("Vertex group '{name}' not found").format(name=vg_name))
            return {"CANCELLED"}
        pin_indices = set(get_vertices_in_group(obj, vg))

    if not pin_indices:
        operator.report({"WARNING"}, iface_("No pin vertices found"))
        return {"CANCELLED"}

    count = 0
    if obj.type == "CURVE":
        idx = 0
        for s in obj.data.splines:
            if s.type == "BEZIER":
                for bp in s.bezier_points:
                    if idx in pin_indices:
                        bp.select_control_point = select
                        count += 1
                    idx += 1
            elif s.type in ("NURBS", "POLY"):
                for p in s.points:
                    if idx in pin_indices:
                        p.select = select
                        count += 1
                    idx += 1
    else:
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        for idx in pin_indices:
            if idx < len(bm.verts):
                bm.verts[idx].select = select
                count += 1
        bm.select_flush_mode()
        bmesh.update_edit_mesh(obj.data)

    action = iface_("Selected") if select else iface_("Deselected")
    operator.report({"INFO"}, iface_("{action} {count} points").format(action=action, count=count))
    return {"FINISHED"}


class OBJECT_OT_SelectPinVertices(Operator):
    """Select vertices in the active pin vertex group"""

    bl_idname = "object.select_pin_vertices"
    bl_label = "Select"

    group_index: _ppf_props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    @classmethod
    def poll(cls, context):
        return context.mode in ("EDIT_MESH", "EDIT_CURVE")

    def execute(self, context):
        return _pin_select(self, context, select=True)


class OBJECT_OT_DeselectPinVertices(Operator):
    """Deselect vertices in the active pin vertex group"""

    bl_idname = "object.deselect_pin_vertices"
    bl_label = "Deselect"

    group_index: _ppf_props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    @classmethod
    def poll(cls, context):
        return context.mode in ("EDIT_MESH", "EDIT_CURVE")

    def execute(self, context):
        return _pin_select(self, context, select=False)


class OBJECT_OT_MakePinKeyframe(Operator):
    """Insert a positional keyframe for vertices in the selected pin vertex group"""

    bl_idname = "object.make_pin_keyframe"
    bl_label = "Make Keyframe"
    bl_options = {"UNDO"}

    group_index: _ppf_props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    def execute(self, context):
        from ...models.groups import decode_vertex_group_identifier

        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            self.report({"ERROR"}, iface_("Group not found"))
            return {"CANCELLED"}

        idx = group.pin_vertex_groups_index
        if idx < 0 or idx >= len(group.pin_vertex_groups):
            self.report({"WARNING"}, iface_("No pin vertex group selected"))
            return {"CANCELLED"}

        from ...core.uuid_registry import resolve_pin, get_object_by_uuid
        pin_item = group.pin_vertex_groups[idx]
        try:
            resolve_pin(pin_item)
        except PinReferenceError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        _, vg_name = decode_stored_pin_identifier(pin_item.name)
        if not pin_item.object_uuid or not vg_name:
            self.report({"ERROR"}, iface_("Invalid pin identifier"))
            return {"CANCELLED"}

        obj = get_object_by_uuid(pin_item.object_uuid)
        if not obj or obj.type not in ("MESH", "CURVE"):
            self.report({"ERROR"}, iface_("Pin object not found"))
            return {"CANCELLED"}

        if obj.type == "CURVE":
            self.report({"INFO"}, iface_("Keyframing not supported for curve pins"))
            return {"FINISHED"}

        vg = obj.vertex_groups.get(vg_name)
        if not vg:
            self.report({"ERROR"}, iface_("Vertex group '{name}' not found").format(name=vg_name))
            return {"CANCELLED"}

        # Reject if any non-EMBEDDED_MOVE op exists. EMBEDDED_MOVE is
        # the sentinel we add ourselves, so re-pressing Make Keyframe
        # at another frame finds it there and must NOT bail.
        non_embedded = [
            op for op in pin_item.operations
            if op.op_type != "EMBEDDED_MOVE"
        ]
        if non_embedded:
            self.report(
                {"ERROR"},
                iface_(
                    "Pin has Move/Spin/Scale/Torque operations; keyframed "
                    "animation cannot be combined with them"
                ),
            )
            return {"CANCELLED"}

        # Reject if this pin currently owns a captured-deformation
        # cache. PC2 wins over fcurves in the encoder, so manual
        # keyframes added on top would be silently ignored; force
        # the user to Clear the capture first to avoid that confusion.
        # Native Undo can restore the RNA flag while leaving external PC2
        # (and its in-memory copy) intact. Probe the same source the encoder
        # uses before accepting keys; do not erase either authoring source.
        from ...core.pc2 import has_pin_anim_pc2
        if getattr(pin_item, "has_captured_anim", False) or has_pin_anim_pc2(obj, vg_name):
            self.report(
                {"ERROR"},
                iface_(
                    "Pin is captured from the depsgraph; press Clear "
                    "Deformation Cache first before adding manual keyframes"
                ),
            )
            return {"CANCELLED"}

        # Flush edit-mode edits to mesh data before sampling positions.
        was_edit = bool(context.active_object and context.active_object.mode == "EDIT")

        # Pin keyframes are stored as native vertex-co fcurves on the
        # mesh's action. The DopeSheet / Graph Editor surface them as
        # standard diamonds, drag to retime, X to delete, all owned by
        # Blender. Our role is just inserting the keys; playback flows
        # through Blender's animation system on each frame change.
        # Capture Deformation PC2 is a separate motion source and is
        # not written here.
        #
        # Call shape mirrors the pre-PC2 pub-main pattern:
        # ``vertex.keyframe_insert(data_path="co")`` (no ``index``, no
        # ``frame``) inserts on all three axes at the current scene
        # frame in a single call. The per-axis form crashes Blender
        # 5.x's animation evaluator on subsequent frame changes,
        # apparently because of a slot-binding edge case when fcurves
        # are added one array-index at a time.
        scene_frame = scene.frame_current
        # Use the shared membership helper so the keyframed vertex set
        # matches the set the solver pins, the overlay highlights, and
        # the encoder reads. A vertex group attached via Add Pin (rather
        # than Create Pin) can contain weight-0 members; filtering by
        # weight here would silently drop those boundary vertices from
        # the keyframed animation while the encoder still pins them.
        from . import pin_transaction as transaction
        stage = None
        snapshot = None
        changed_ops = False
        inserted = 0
        try:
            if was_edit and bpy.ops.object.mode_set(mode='OBJECT') != {'FINISHED'}:
                raise RuntimeError('Could not leave edit mode')
            pin_indices = sorted(get_vertices_in_group(obj, vg))
            if not pin_indices:
                raise ValueError('The selected pin vertex group has no vertices')
            transaction.check_overlapping_pins(obj, pin_item, pin_indices)
            snapshot = transaction.snapshot_pin_operations(pin_item)
            stage = transaction.StagedMeshAction(obj)
            for vi in pin_indices:
                if not obj.data.vertices[vi].keyframe_insert(data_path="co"):
                    raise ValueError('Could not insert a pin vertex keyframe')
                inserted += 1
            transaction.set_linear(obj.data, pin_indices)
            changed_ops = True
            _ensure_embedded_move_op(pin_item)
            from ...models.groups import invalidate_overlays
            invalidate_overlays()
            if was_edit:
                if bpy.ops.object.mode_set(mode="EDIT") != {'FINISHED'}:
                    raise RuntimeError('Could not restore edit mode')
            stage.commit()
        except Exception as exc:
            failures = transaction.rollback_pin_edit(context, stage, pin_item, snapshot, changed_ops, was_edit)
            message = str(exc)
            if failures:
                message += '; rollback incomplete: ' + '; '.join(failures)
            self.report({'ERROR'}, message)
            return {'CANCELLED'}

        self.report(
            {"INFO"},
            iface_("Inserted keyframes on {count} pin vertices at frame {frame}").format(
                count=inserted, frame=scene_frame
            ),
        )
        return {"FINISHED"}


class OBJECT_OT_DeletePinKeyframes(Operator):
    """Delete all positional keyframes for vertices in the selected pin vertex group"""

    bl_idname = "object.delete_pin_keyframes"
    bl_label = "Delete All Keyframes"
    bl_options = {"UNDO"}

    group_index: _ppf_props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    def execute(self, context):
        from ...models.groups import decode_vertex_group_identifier

        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            self.report({"ERROR"}, iface_("Group not found"))
            return {"CANCELLED"}

        idx = group.pin_vertex_groups_index
        if idx < 0 or idx >= len(group.pin_vertex_groups):
            self.report({"WARNING"}, iface_("No pin vertex group selected"))
            return {"CANCELLED"}

        from ...core.uuid_registry import resolve_pin, get_object_by_uuid
        pin_item = group.pin_vertex_groups[idx]
        try:
            resolve_pin(pin_item)
        except PinReferenceError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        _, vg_name = decode_stored_pin_identifier(pin_item.name)
        if not pin_item.object_uuid or not vg_name:
            self.report({"ERROR"}, iface_("Invalid pin identifier"))
            return {"CANCELLED"}

        obj = get_object_by_uuid(pin_item.object_uuid)
        if not obj or obj.type not in ("MESH", "CURVE"):
            self.report({"ERROR"}, iface_("Pin object not found"))
            return {"CANCELLED"}

        if obj.type == "CURVE":
            self.report({"INFO"}, iface_("Keyframe deletion not applicable for curve pins"))
            return {"FINISHED"}

        vg = obj.vertex_groups.get(vg_name)
        if vg is None:
            self.report({'ERROR'}, iface_("Vertex group '{name}' not found").format(name=vg_name))
            return {'CANCELLED'}
        was_edit = bool(context.active_object and context.active_object.mode == 'EDIT')
        from . import pin_transaction as transaction
        stage = None
        snapshot = None
        changed_ops = False
        try:
            if was_edit and bpy.ops.object.mode_set(mode='OBJECT') != {'FINISHED'}:
                raise RuntimeError('Could not leave edit mode')
            pin_indices = sorted(get_vertices_in_group(obj, vg))
            if not pin_indices:
                raise ValueError('The selected pin vertex group has no vertices')
            transaction.check_overlapping_pins(obj, pin_item, pin_indices)
            snapshot = transaction.snapshot_pin_operations(pin_item)
            stage = transaction.StagedMeshAction(obj)
            removed = transaction.remove_curves(obj.data, pin_indices)
            changed_ops = True
            ops_removed = _remove_embedded_move_ops(pin_item)
            from ...models.groups import invalidate_overlays
            invalidate_overlays()
            if was_edit:
                if bpy.ops.object.mode_set(mode='EDIT') != {'FINISHED'}:
                    raise RuntimeError('Could not restore edit mode')
            stage.commit()
        except Exception as exc:
            failures = transaction.rollback_pin_edit(context, stage, pin_item, snapshot, changed_ops, was_edit)
            message = str(exc)
            if failures:
                message += '; rollback incomplete: ' + '; '.join(failures)
            self.report({'ERROR'}, message)
            return {'CANCELLED'}

        self.report({"INFO"},
                    iface_("Removed {removed} pin-vertex fcurves, "
                           "{ops_removed} EMBEDDED_MOVE ops").format(
                        removed=removed, ops_removed=ops_removed))
        return {"FINISHED"}


def _ensure_embedded_move_op(pin_item) -> None:
    """Ensure exactly one ``EMBEDDED_MOVE`` op exists at index 0 in
    ``pin_item.operations``. Adds one if absent, moves the first
    existing one to the head, and drops any duplicates.

    The encoder treats ``EMBEDDED_MOVE``'s index in this list as the
    insertion point for the per-vertex ``pin_anim`` track via
    ``embedded_move_index``. Pinning it to slot 0 keeps the splice
    deterministic and matches pub-main.
    """
    embedded_positions = [
        i for i, op in enumerate(pin_item.operations)
        if op.op_type == "EMBEDDED_MOVE"
    ]
    if not embedded_positions:
        op = pin_item.operations.add()
        op.op_type = "EMBEDDED_MOVE"
        # Newly added entries land at the end; move to head.
        last = len(pin_item.operations) - 1
        if last > 0:
            pin_item.operations.move(last, 0)
        return
    # Keep the first EMBEDDED_MOVE; drop the rest. Walk in reverse
    # so earlier indices remain valid as later entries are removed.
    keep = embedded_positions[0]
    for i in reversed(embedded_positions[1:]):
        pin_item.operations.remove(i)
    if keep != 0:
        pin_item.operations.move(keep, 0)


def _remove_embedded_move_ops(pin_item) -> int:
    """Drop every ``EMBEDDED_MOVE`` op from ``pin_item.operations``
    and return the count removed. Used by Delete All Keyframes so the
    sentinel doesn't outlive the fcurves it represents.
    """
    embedded_positions = [
        i for i, op in enumerate(pin_item.operations)
        if op.op_type == "EMBEDDED_MOVE"
    ]
    for i in reversed(embedded_positions):
        pin_item.operations.remove(i)
    if pin_item.operations_index >= len(pin_item.operations):
        pin_item.operations_index = max(-1, len(pin_item.operations) - 1)
    return len(embedded_positions)


class OBJECT_OT_AddPinOperation(Operator):
    """Add an operation to the selected pin"""
    bl_idname = "object.add_pin_operation"
    bl_label = "Add Operation"
    bl_options = {"REGISTER", "UNDO"}

    group_index: _ppf_props.IntProperty(options={'HIDDEN'})  # pyright: ignore
    op_type: EnumProperty(
        items=[
            ("MOVE_BY", "Move By", "Move by delta"),
            ("SPIN", "Spin", "Rotate around axis"),
            ("SCALE", "Scale", "Scale from center"),
            ("TORQUE", "Torque", "Apply rotational force"),
        ],
    )  # pyright: ignore

    def execute(self, context):
        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            return {"CANCELLED"}
        idx = group.pin_vertex_groups_index
        if idx < 0 or idx >= len(group.pin_vertex_groups):
            return {"CANCELLED"}
        pin_item = group.pin_vertex_groups[idx]
        # Reject if the pin already carries an EMBEDDED_MOVE sentinel
        # (Make Keyframe authored fcurve animation); the kinematic
        # ops would need to compose with that and pub-main's contract
        # is one-or-the-other on a pin.
        if any(op.op_type == "EMBEDDED_MOVE" for op in pin_item.operations):
            self.report(
                {"ERROR"},
                iface_(
                    "Pin is keyframed; remove its keyframed animation before "
                    "adding Move/Spin/Scale/Torque operations"
                ),
            )
            return {"CANCELLED"}
        # Torque cannot be mixed with other operation types
        existing_types = {o.op_type for o in pin_item.operations}
        if self.op_type == "TORQUE" and existing_types - {"TORQUE"}:
            self.report({"ERROR"}, iface_("Torque cannot be mixed with Move/Spin/Scale operations"))
            return {"CANCELLED"}
        if self.op_type != "TORQUE" and "TORQUE" in existing_types:
            self.report({"ERROR"}, iface_("Cannot add Move/Spin/Scale to a pin that has Torque"))
            return {"CANCELLED"}
        op = pin_item.operations.add()
        op.op_type = self.op_type
        from ...core.encoder import seed_window_at_start
        from ...models.groups import get_addon_data
        seed_window_at_start(op, get_addon_data(context.scene).state)
        # Default center mode to CENTROID for scale and spin
        # (the solver computes the center from vertex positions at runtime)
        if self.op_type == "SCALE":
            op.scale_center_mode = "CENTROID"
        elif self.op_type == "SPIN":
            op.spin_center_mode = "CENTROID"
        elif self.op_type == "TORQUE":
            op.torque_axis_component = "PC3"
        # Insert at head
        pin_item.operations.move(len(pin_item.operations) - 1, 0)
        pin_item.operations_index = 0
        redraw_all_areas(context)
        return {"FINISHED"}


class OBJECT_OT_RemovePinOperation(Operator):
    """Remove the selected operation from the pin"""
    bl_idname = "object.remove_pin_operation"
    bl_label = "Remove Operation"
    bl_options = {"REGISTER", "UNDO"}

    group_index: _ppf_props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    def execute(self, context):
        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            return {"CANCELLED"}
        idx = group.pin_vertex_groups_index
        if idx < 0 or idx >= len(group.pin_vertex_groups):
            return {"CANCELLED"}
        pin_item = group.pin_vertex_groups[idx]
        op_idx = pin_item.operations_index
        if op_idx < 0 or op_idx >= len(pin_item.operations):
            return {"CANCELLED"}
        pin_item.operations.remove(op_idx)
        pin_item.operations_index = safe_update_index(op_idx, len(pin_item.operations))
        from ...models.groups import invalidate_overlays
        invalidate_overlays()
        redraw_all_areas(context)
        return {"FINISHED"}


class OBJECT_OT_MovePinOperation(Operator):
    """Move the selected operation up or down"""
    bl_idname = "object.move_pin_operation"
    bl_label = "Move Operation"
    bl_options = {"REGISTER", "UNDO"}

    group_index: _ppf_props.IntProperty(options={'HIDDEN'})  # pyright: ignore
    direction: _ppf_props.IntProperty()  # pyright: ignore  # -1 = up, 1 = down

    def execute(self, context):
        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            return {"CANCELLED"}
        idx = group.pin_vertex_groups_index
        if idx < 0 or idx >= len(group.pin_vertex_groups):
            return {"CANCELLED"}
        pin_item = group.pin_vertex_groups[idx]
        op_idx = pin_item.operations_index
        new_idx = op_idx + self.direction
        if op_idx < 0 or op_idx >= len(pin_item.operations):
            return {"CANCELLED"}
        if new_idx < 0 or new_idx >= len(pin_item.operations):
            return {"CANCELLED"}
        pin_item.operations.move(op_idx, new_idx)
        pin_item.operations_index = new_idx
        return {"FINISHED"}


def _commit_selected_pin_center(operator, context, vertex):
    """Selection-derived Scene write, committed with Object-mode global Undo."""
    import bmesh
    import math
    obj = context.edit_object
    if context.mode != "EDIT_MESH" or obj is None or obj.type != "MESH":
        operator.report({"ERROR"}, "Pick Center requires Mesh Edit Mode")
        return {"CANCELLED"}
    if len(context.objects_in_mode) != 1:
        operator.report({"ERROR"}, "Pick Center requires a single edit object")
        return {"CANCELLED"}
    if operator.target not in {"spin", "scale"}:
        operator.report({"ERROR"}, "Invalid center target")
        return {"CANCELLED"}
    if not context.preferences.edit.use_global_undo:
        operator.report({"ERROR"}, "Pick Center requires Global Undo; no changes were made")
        return {"CANCELLED"}
    if not context.scene.is_editable or not obj.is_editable or not obj.data.is_editable:
        operator.report({"ERROR"}, "Pick Center requires editable scene, object and mesh")
        return {"CANCELLED"}
    group = get_group_from_index(context.scene, operator.group_index)
    if group is None or not 0 <= group.pin_vertex_groups_index < len(group.pin_vertex_groups):
        return {"CANCELLED"}
    pin = group.pin_vertex_groups[group.pin_vertex_groups_index]
    if not 0 <= pin.operations_index < len(pin.operations):
        return {"CANCELLED"}
    op = pin.operations[pin.operations_index]
    if op.op_type != {"spin": "SPIN", "scale": "SCALE"}[operator.target]:
        operator.report({"ERROR"}, "Selected operation does not match center target")
        return {"CANCELLED"}
    if vertex:
        from ...core.uuid_registry import get_object_uuid
        obj_uuid = get_object_uuid(obj)
        if not obj_uuid or obj_uuid != pin.object_uuid:
            operator.report({"ERROR"}, "Pick Vertex requires the pin object's mesh")
            return {"CANCELLED"}
        from ...core.uuid_registry import get_object_by_uuid, UUIDOwnershipError
        try:
            owner = get_object_by_uuid(pin.object_uuid)
        except UUIDOwnershipError as exc:
            operator.report({"ERROR"}, "Pick Vertex refused: " + str(exc))
            return {"CANCELLED"}
        if owner is not obj:
            operator.report({"ERROR"}, "Pick Vertex requires the proven pin object owner")
            return {"CANCELLED"}
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.index_update()
    selected = [v for v in bm.verts if v.select]
    if not selected or (vertex and len(selected) != 1):
        operator.report({"WARNING"}, "Select exactly one vertex" if vertex else "No vertices selected")
        return {"CANCELLED"}
    if vertex:
        value = selected[0].index
    else:
        points = [obj.matrix_world @ v.co for v in selected]
        value = tuple(sum(p[i] for p in points) / len(points) for i in range(3))
        if not all(math.isfinite(x) for x in value):
            operator.report({"ERROR"}, "Selected center must be finite")
            return {"CANCELLED"}
    field = operator.target + ("_center_vertex" if vertex else "_center")
    old_value = getattr(op, field)
    old_value = old_value if vertex else tuple(old_value)

    def restore_edit():
        if obj.mode == "EDIT":
            return None
        try:
            if bpy.ops.object.mode_set(mode="EDIT") == {"FINISHED"} and obj.mode == "EDIT":
                return None
            return "could not restore Edit Mode"
        except Exception as exc:
            return "restore Edit Mode: " + str(exc)

    try:
        if bpy.ops.object.mode_set(mode="OBJECT") != {"FINISHED"} or obj.mode != "OBJECT":
            raise RuntimeError("Could not enter Object Mode")
    except Exception as exc:
        recovery = restore_edit()
        operator.report({"ERROR"}, "Pick Center cancelled before author changes: " + str(exc)
                        + ("; " + recovery if recovery else ""))
        return {"CANCELLED"}
    try:
        setattr(op, field, value)
    except Exception as exc:
        recovery_errors = []
        try:
            setattr(op, field, old_value)
            restored = getattr(op, field)
            if (restored if vertex else tuple(restored)) != old_value:
                recovery_errors.append("center restoration mismatch")
        except Exception as restore_exc:
            recovery_errors.append("center restoration: " + str(restore_exc))
        if not recovery_errors:
            recovery = restore_edit()
            if recovery:
                recovery_errors.append(recovery)
        if recovery_errors:
            operator.report({"ERROR"}, "Pick Center failed; recovery incomplete: "
                            + "; ".join(recovery_errors) + "; original error: " + str(exc))
            return {"FINISHED"}  # Retain global Undo if compensation was incomplete.
        operator.report({"ERROR"}, "Pick Center failed; author value restored: " + str(exc))
        return {"CANCELLED"}
    operator.report({"INFO"}, "Center set; Object Mode retained for complete Undo")
    return {"FINISHED"}


class OBJECT_OT_PickCenterFromSelected(Operator):
    """Set the absolute center from the centroid of selected vertices"""

    bl_idname = "object.pick_center_from_selected"
    bl_label = "Pick from Selected"
    bl_options = {"UNDO"}

    group_index: _ppf_props.IntProperty(options={'HIDDEN'})  # pyright: ignore
    target: _ppf_props.StringProperty()  # pyright: ignore  # "spin" or "scale"

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
            obj is not None
            and obj.type == "MESH"
            and obj.mode == "EDIT"
        )

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        self.layout.label(text=iface_("Success stays in Object Mode."))
        self.layout.label(text=iface_("Undo restores the previous Edit state."))
        self.layout.label(text=iface_("Cancel leaves your selection unchanged."))

    def execute(self, context):
        return _commit_selected_pin_center(self, context, vertex=False)


class OBJECT_OT_PickVertexCenter(Operator):
    """Set the center vertex from a single selected vertex in Edit Mode"""

    bl_idname = "object.pick_vertex_center"
    bl_label = "Pick Vertex"
    bl_options = {"UNDO"}

    group_index: _ppf_props.IntProperty(options={'HIDDEN'})  # pyright: ignore
    target: _ppf_props.StringProperty()  # pyright: ignore  # "spin" or "scale"

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
            obj is not None
            and obj.type == "MESH"
            and obj.mode == "EDIT"
        )

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        self.layout.label(text=iface_("Success stays in Object Mode."))
        self.layout.label(text=iface_("Undo restores the previous Edit state."))
        self.layout.label(text=iface_("Cancel leaves your selection unchanged."))

    def execute(self, context):
        return _commit_selected_pin_center(self, context, vertex=True)

classes = (
    OBJECT_OT_CreatePinVertexGroup,
    OBJECT_OT_AddPinVertexGroup,
    OBJECT_OT_RemovePinVertexGroup,
    OBJECT_OT_MovePinVertexGroup,
    OBJECT_OT_RenamePinVertexGroup,
    OBJECT_OT_SelectPinVertices,
    OBJECT_OT_DeselectPinVertices,
    OBJECT_OT_MakePinKeyframe,
    OBJECT_OT_DeletePinKeyframes,
    OBJECT_OT_AddPinOperation,
    OBJECT_OT_RemovePinOperation,
    OBJECT_OT_MovePinOperation,
    OBJECT_OT_PickCenterFromSelected,
    OBJECT_OT_PickVertexCenter,
)


def register():
    for cls in classes:
        _ppf_i18n.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
