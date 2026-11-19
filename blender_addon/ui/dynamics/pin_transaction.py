from ...core.pin_reference import decode_stored_pin_identifier, PinReferenceError
"""Primitives for staging pin vertex-curve edits on the owning Action slot.

This module does not commit pin operations itself. Callers must coordinate the
staged datablocks with the pin operation collection and roll back on failure.
"""
import re
import copy

import bpy

_VERTEX_CO = re.compile(r"vertices\[(\d+)\]\.co\Z")


def vertex_curves(mesh, indices):
    """Yield (container, curve) for selected vertices in the active slot only."""
    animation = mesh.animation_data
    action = animation.action if animation else None
    if action is None:
        return
    indices = set(indices)
    if getattr(action, 'layers', None):
        slot = getattr(animation, 'action_slot', None)
        if slot is None:
            raise ValueError('Mesh Action has no active slot')
        containers = []
        for layer in action.layers:
            for strip in layer.strips:
                bag = strip.channelbag(slot)
                if bag is not None:
                    containers.append(bag.fcurves)
    else:
        containers = [action.fcurves]
    matches = []
    seen = set()
    for container in containers:
        for curve in list(container):
            match = _VERTEX_CO.fullmatch(curve.data_path)
            if match and int(match.group(1)) in indices:
                identity = (curve.data_path, curve.array_index)
                if identity in seen:
                    raise ValueError('Ambiguous duplicate vertex curves in the active Action slot')
                seen.add(identity)
                matches.append((container, curve))
    yield from matches


def set_linear(mesh, indices):
    for _, curve in vertex_curves(mesh, indices):
        for key in curve.keyframe_points:
            key.interpolation = 'LINEAR'


def remove_curves(mesh, indices):
    matches = list(vertex_curves(mesh, indices))
    for container, curve in matches:
        container.remove(curve)
    return len(matches)


def check_overlapping_pins(obj, selected_pin, indices):
    """Refuse edits that would also change another live pin on this object."""
    from ...models.groups import iterate_active_object_groups, decode_vertex_group_identifier
    from ...core.utils import get_vertices_in_group
    wanted = set(indices)
    for scene in bpy.data.scenes:
        for group in iterate_active_object_groups(scene):
            for pin in group.pin_vertex_groups:
                if pin.as_pointer() == selected_pin.as_pointer():
                    continue
                if pin.object_uuid != selected_pin.object_uuid:
                    continue
                _, name = decode_stored_pin_identifier(pin.name)
                vertex_group = obj.vertex_groups.get(name)
                if vertex_group is None:
                    raise ValueError('Another pin on this object has an unresolved vertex group')
                if wanted.intersection(get_vertices_in_group(obj, vertex_group)):
                    raise ValueError('Another pin on this object overlaps the selected vertices')


def snapshot_pin_operations(pin):
    """Capture complete writable PinOperation RNA plus custom properties."""
    rows = []
    for item in pin.operations:
        row = {}
        for prop in item.bl_rna.properties:
            name = prop.identifier
            if name == 'rna_type' or prop.is_readonly:
                continue
            if prop.type in {'COLLECTION', 'POINTER'}:
                raise ValueError('Unsupported nested pin operation property: ' + name)
            value = getattr(item, name)
            row[name] = tuple(value) if getattr(prop, 'is_array', False) else value
        extra = {key: copy.deepcopy(item[key]) for key in item.keys()
                 if key not in item.bl_rna.properties}
        rows.append((row, extra))
    return rows, pin.operations_index


def restore_pin_operations(pin, snapshot):
    rows, index = snapshot
    pin.operations.clear()
    for row, extra in rows:
        item = pin.operations.add()
        for name, value in row.items():
            setattr(item, name, value)
        for name, value in extra.items():
            item[name] = value
    pin.operations_index = index


def rollback_pin_edit(context, stage, pin, snapshot, changed_ops, was_edit):
    """Attempt all rollback steps, even when an earlier restoration fails."""
    failures = []
    if was_edit and stage is not None and context.active_object and context.active_object.mode == 'EDIT':
        try:
            if bpy.ops.object.mode_set(mode='OBJECT') != {'FINISHED'}:
                raise RuntimeError('Could not leave edit mode for rollback')
        except Exception as exc:
            failures.append('object mode: ' + str(exc))
    if stage is not None:
        try:
            stage.rollback()
        except Exception as exc:
            failures.append('mesh/action: ' + str(exc))
    if changed_ops:
        try:
            restore_pin_operations(pin, snapshot)
        except Exception as exc:
            failures.append('pin operations: ' + str(exc))
    if was_edit and context.active_object and context.active_object.mode != 'EDIT':
        try:
            if bpy.ops.object.mode_set(mode='EDIT') != {'FINISHED'}:
                raise RuntimeError('Could not restore edit mode')
        except Exception as exc:
            failures.append('edit mode: ' + str(exc))
    return failures


def _remap_copied_self_drivers(old_mesh, new_mesh):
    """Keep explicit Mesh/Key self-targets self-relative after Mesh.copy()."""
    pairs = [(old_mesh, new_mesh)]
    if old_mesh.shape_keys is not None:
        if new_mesh.shape_keys is None or new_mesh.shape_keys == old_mesh.shape_keys:
            raise ValueError('Mesh copy did not create an independent Key datablock')
        pairs.append((old_mesh.shape_keys, new_mesh.shape_keys))
    for old_id, new_id in pairs:
        animation = new_id.animation_data
        if animation is None:
            continue
        for curve in animation.drivers:
            for variable in curve.driver.variables:
                for target in variable.targets:
                    for source, replacement in pairs:
                        if target.id == source:
                            target.id = replacement


class StagedMeshAction:
    """Stage a mesh Action without modifying other objects or Action users.

    The caller must also protect its pin-operation collection, restore edit
    mode, and invoke either commit() or rollback().
    """

    def __init__(self, obj):
        self.obj = obj
        self.old_mesh = obj.data
        self.mesh = self.old_mesh
        self.mesh_copy = None
        self.action_copy = None
        self.had_animation = self.old_mesh.animation_data is not None
        self.old_action = None
        self.old_slot = None
        self.old_fake_user = None
        try:
            # Data.users includes non-object references; only an actual other
            # object sharing this Mesh requires single-user authoring.
            if any(other != obj and other.data == self.old_mesh
                   for other in bpy.data.objects):
                self.mesh_copy = self.old_mesh.copy()
                _remap_copied_self_drivers(self.old_mesh, self.mesh_copy)
                obj.data = self.mesh_copy
                self.mesh = self.mesh_copy
            animation = self.mesh.animation_data
            self.old_action = animation.action if animation else None
            self.old_slot = getattr(animation, 'action_slot', None) if animation else None
            if self.old_action is not None:
                self.old_fake_user = self.old_action.use_fake_user
                self.action_copy = self.old_action.copy()
                slot_identifier = self.old_slot.identifier if self.old_slot else None
                animation.action = self.action_copy
                if slot_identifier is not None:
                    matches = [slot for slot in self.action_copy.slots
                               if slot.identifier == slot_identifier]
                    if len(matches) != 1:
                        raise ValueError('Action slot cannot be identified after copy')
                    animation.action_slot = matches[0]
        except BaseException:
            self.rollback()
            raise

    def rollback(self):
        animation = self.mesh.animation_data
        current_action = animation.action if animation else None
        if self.mesh_copy is None:
            if animation:
                animation.action = self.old_action
                if self.old_action is not None and self.old_slot is not None:
                    animation.action_slot = self.old_slot
                if not self.had_animation:
                    self.mesh.animation_data_clear()
        else:
            self.obj.data = self.old_mesh
            key = self.mesh_copy.shape_keys
            # Keep the temporary ID if a callback established a real external
            # reference to either the Mesh or its copied Key datablock.
            if self.mesh_copy.users == 0 and (key is None or key.users <= 1):
                bpy.data.meshes.remove(self.mesh_copy)
        if self.old_action is not None and self.old_fake_user is not None:
            self.old_action.use_fake_user = self.old_fake_user
        for action in (current_action, self.action_copy):
            if action is not None:
                try:
                    if action == self.old_action:
                        continue
                    if action.users == 0:
                        bpy.data.actions.remove(action)
                except ReferenceError:
                    pass

    def commit(self):
        # A previously assigned authored Action must survive saving even if
        # this transaction moved its last current user to the staged copy.
        if self.old_action is not None and self.old_action.users == 0:
            self.old_action.use_fake_user = True
