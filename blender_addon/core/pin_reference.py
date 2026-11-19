"""Read-only syntax admission for existing persisted Pin references."""
class PinReferenceError(ValueError):
    pass

def decode_stored_pin_identifier(raw):
    if (not isinstance(raw, str) or not raw.startswith("[")
            or not raw.endswith("]") or raw.count("][") != 1):
        raise PinReferenceError("Stored Pin reference is malformed or ambiguous; preserve it and repair the reference explicitly: " + repr(raw))
    owner, vertex_group = raw[1:-1].split("][", 1)
    if not owner or not vertex_group:
        raise PinReferenceError("Stored Pin reference has an empty component: " + repr(raw))
    return owner, vertex_group

def validate_stored_pin_items(items):
    for item in items:
        decode_stored_pin_identifier(item.name)

def validate_export_pin_references(context):
    """Read-only admission covering all rows visited by active non-static exporters."""
    from ..models.groups import iterate_active_object_groups
    for group in iterate_active_object_groups(context.scene):
        if group.object_type == "STATIC":
            continue
        # Existing consumers resolve before their inclusion filter. Refuse the
        # whole visited set here rather than allowing an excluded row to fail late.
        validate_stored_pin_items(group.pin_vertex_groups)
        from .uuid_registry import plan_pin_reference
        for pin in group.pin_vertex_groups:
            if plan_pin_reference(pin) is None:
                raise PinReferenceError("Pin object is unavailable; cannot prove its current name pair")
            if sum(1 for row in group.assigned_objects if row.uuid == pin.object_uuid) != 1:
                raise PinReferenceError("Pin group ownership is missing or ambiguous")

def encode_pin_pair(object_name, vertex_group_name):
    """Validate an actual or fully planned pair without author writes."""
    for name in (object_name, vertex_group_name):
        if not isinstance(name, str) or not name or "\x00" in name:
            raise PinReferenceError("Pin names must be nonempty strings without NUL")
        try:
            name.encode("utf-8")
        except UnicodeError as exc:
            raise PinReferenceError("Pin name is not representable as UTF-8") from exc
    encoded = f"[{object_name}][{vertex_group_name}]"
    if decode_stored_pin_identifier(encoded) != (object_name, vertex_group_name):
        raise PinReferenceError("Pin name pair cannot round-trip without ambiguity")
    return encoded

def validate_pin_name_for_write(obj, name):
    """No truncation/suffix discovery by mutating author data."""
    encoded = encode_pin_pair(obj.name, name)
    if obj.type == "MESH":
        import bpy
        capacity = bpy.types.VertexGroup.bl_rna.properties["name"].length_max - 1
        if capacity <= 0 or len(name.encode("utf-8")) > capacity:
            raise PinReferenceError("Pin vertex group name exceeds the supported UTF-8 byte capacity")
    elif obj.type == "CURVE":
        # IDProperty name limit on the supported Blender runtime; includes prefix.
        if len(("_pin_" + name).encode("utf-8")) > 63:
            raise PinReferenceError("Curve Pin key exceeds the supported 63 UTF-8 bytes")
    else:
        raise PinReferenceError("Pin owner must be a mesh or curve")
    return encoded
