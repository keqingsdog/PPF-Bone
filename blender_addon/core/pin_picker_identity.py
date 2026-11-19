"""Explicit UUID + exact-VG picker identity; never infer legacy ordinals.

This identifies a current named VG, not a persistent VG lifetime. Native numeric events use non-reused process tokens rather than current row
positions. Invalid events clear only this picker selection, not stored Pins.
"""
FIELD = 'pin_vertex_group_items'
UUID_KEY = '_ppf_pin_vertex_group_object_uuid_v1'
VG_KEY = '_ppf_pin_vertex_group_name_v1'
METADATA_KEYS = (FIELD, UUID_KEY, VG_KEY)


def eligible(group):
    """Read-only eligible rows; excluded memberships keep the existing contract.

    Duplicate membership rows and any globally duplicated Object UUID refuse
    that identity rather than trusting a warm registry/name first match.
    """
    import bpy
    from .uuid_registry import _OBJ_UUID_KEY
    from .pin_reference import encode_pin_pair, PinReferenceError
    memberships = {}
    for row in group.assigned_objects:
        uid = row.uuid
        if uid:
            memberships[uid] = memberships.get(uid, 0) + 1
    owners = {}
    for obj in bpy.data.objects:
        uid = obj.get(_OBJ_UUID_KEY, '')
        if isinstance(uid, str) and uid:
            owners.setdefault(uid, []).append(obj)
    result = []
    for row in group.assigned_objects:
        uid = row.uuid
        matches = owners.get(uid, ())
        if not uid or memberships.get(uid) != 1 or len(matches) != 1:
            continue
        obj = matches[0]
        if obj.type != 'MESH':
            continue
        for vg in obj.vertex_groups:
            try:
                identifier = encode_pin_pair(obj.name, vg.name)
            except PinReferenceError:
                continue
            result.append((obj, uid, vg.name, identifier))
    return result


def items(group):
    from . import pin_picker_tokens as tokens
    try:
        return tokens.enum_items(group, eligible(group))
    except Exception:
        return [tokens.none_item()]


def selected_identity(group):
    uid, name = group.get(UUID_KEY), group.get(VG_KEY)
    if not isinstance(uid, str) or not uid or not isinstance(name, str) or not name:
        return None
    return uid, name


def get_selection(group):
    from . import pin_picker_tokens as tokens
    try:
        return tokens.number_for(group, eligible(group), selected_identity(group))
    except Exception:
        return 0


def set_selection(group, value):
    from . import pin_picker_tokens as tokens
    if type(value) is int and value == 0:
        clear(group)
        return
    try:
        uid, name = tokens.authenticate(group, eligible(group), value)
    except Exception as exc:
        # RNA can swallow setter exceptions: clear stale intent BEFORE reporting.
        try:
            clear(group)
        except Exception as clear_exc:
            tokens.block('Pin picker rejection could not clear selection: ' + str(clear_exc))
            print('Pin picker rejection clear failed:', clear_exc)
        print('Pin picker selection rejected:', exc)
        raise ValueError('Pin selection is stale or unavailable; select it again') from exc
    try:
        group[UUID_KEY] = uid
        group[VG_KEY] = name
        if FIELD in group:
            del group[FIELD]
    except Exception:
        tokens.block('Pin picker selection commit failed')
        raise


def resolve(group):
    from . import pin_picker_tokens as tokens
    try:
        identity = selected_identity(group)
        rows = eligible(group)
        number = tokens.number_for(group, rows, identity)
        if not number:
            return None
        tokens.authenticate(group, rows, number)
        for obj, uid, name, _identifier in rows:
            if identity == (uid, name):
                return obj, name
    except Exception:
        return None
    return None


def clear(group):
    from . import pin_picker_tokens as tokens
    try:
        tokens.retire_group(group)
        for key in METADATA_KEYS:
            if key in group:
                del group[key]
    except Exception:
        tokens.block('Pin picker selection clear failed')
        raise


def snapshot_metadata(group):
    """Only this picker's raw presence/values; never call the enum getter."""
    from copy import deepcopy
    return {key: deepcopy(group[key]) for key in METADATA_KEYS if key in group}


def restore_metadata(group, snapshot):
    """Restore author metadata, never revive old native event tokens."""
    from copy import deepcopy
    if not isinstance(snapshot, dict) or any(key not in METADATA_KEYS for key in snapshot):
        raise ValueError('Invalid Pin picker metadata snapshot')
    values = deepcopy(snapshot)
    clear(group)
    try:
        for key, value in values.items():
            group[key] = value
    except Exception:
        from . import pin_picker_tokens as tokens
        tokens.block('Pin picker metadata restore failed')
        raise
