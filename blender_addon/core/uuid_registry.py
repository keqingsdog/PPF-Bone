from .pin_reference import decode_stored_pin_identifier, PinReferenceError, encode_pin_pair
# Fork modification: changed from the upstream baseline.
# File: core/uuid_registry.py
# License: Apache v2.0
#
# Stable UUID-based identification for Blender objects.
# UUIDs are stored as custom properties on objects and persist across renames.
# Vertex groups are identified by a content hash of their vertex indices,
# enabling recovery after renames.

import hashlib
import struct
import uuid as _uuid

import bpy  # pyright: ignore

_OBJ_UUID_KEY = "_solver_uuid"


def _is_writable(obj: bpy.types.Object) -> bool:
    """True when a custom property can be safely written to obj.

    Library-linked objects (obj.library is not None) are read-only;
    attempting to write raises `RuntimeError: Writing to ID classes in
    this context is not allowed`.  Library overrides (obj.override_library)
    DO allow custom-prop writes because overrides have local storage for
    ID-properties, so we allow those.
    """
    return obj.library is None


# Runtime-only ownership: Blender names and Python RNA wrappers are not identity.
# Never persist these session IDs into the .blend or trust them across load/Undo.
_owner_session_uids: dict[str, int] = {}


def _remember_uuid_owner(uid, obj):
    _owner_session_uids[uid] = int(obj.session_uid)
    _cache[uid] = obj.name


def resolve_uuid_owner(
    objs: "list[bpy.types.Object]", prev_name: "str | None"
) -> "bpy.types.Object | None":
    """Return a proven current owner, never guess ownership from a name.

    A live, previously observed session ID survives rename/copy. With no hot
    memory, only a unique UUID establishes ownership. An absent former owner
    or cold collision is unresolved, including a copy using the former name.
    ``prev_name`` remains a display-cache argument, not ownership evidence.
    """
    if not objs:
        return None
    uid = str(objs[0].get(_OBJ_UUID_KEY, ""))
    known = _owner_session_uids.get(uid)
    if known is not None:
        return next((o for o in objs if int(o.session_uid) == known), None)
    return objs[0] if len(objs) == 1 else None


class UUIDOwnershipError(ValueError):
    """An object UUID has no safely established owner; never treat as missing."""


def _unresolved_uuid(uid):
    return UUIDOwnershipError(
        f"Solver UUID '{uid}' has ambiguous or unavailable object ownership. "
        "The group was not reassigned by object name; preserve the objects "
        "and resolve the duplicate identity before continuing."
    )


def get_or_create_object_uuid(obj: bpy.types.Object) -> str:
    """Get or assign a stable UUID for a Blender object.

    If another object shares the same UUID, a proven live session owner
    keeps it; a writable copy gets a fresh one. Cold ambiguity is refused.

    For library-linked objects we cannot write custom props — we return
    an ownership-verified existing UUID but never create one. Callers receiving
    an empty string MUST treat the object as unidentifiable.
    """
    uid = obj.get(_OBJ_UUID_KEY)
    if uid:
        uid = str(uid)
        # Keep the common lookup O(1), but only for the same observed Blender
        # ID. A pending duplicate cannot replace that owner by taking its name.
        # Cold/missed cache ownership is established by the scan below.
        if (_owner_session_uids.get(uid) == int(obj.session_uid)
                and _cache.get(uid) == obj.name
                and bpy.data.objects.get(obj.name) is obj):
            return uid
        # Gather every object currently carrying this UUID. A duplicate
        # (e.g. produced by duplicate-obj) copies the custom property, so
        # more than one object may share it; handle >2-way collisions too.
        objs = [
            o for o in bpy.data.objects
            if str(o.get(_OBJ_UUID_KEY, "")) == uid
        ]
        owner = resolve_uuid_owner(objs, _prev_names.get(uid))
        # Read-only does not mean identity-safe: returning another object's
        # UUID would silently redirect the caller's new group/pair reference.
        if owner is None or (not _is_writable(obj) and owner is not obj):
            raise _unresolved_uuid(uid)
        _remember_uuid_owner(uid, owner)
        if not _is_writable(obj):
            # A proven linked owner may be referenced, but this read must not
            # opportunistically rewrite any writable copies around it.
            return uid
        if len(objs) <= 1:
            return uid
        for other in objs:
            if other is owner or not _is_writable(other):
                # Unwritable non-owners (library-linked) cannot be fixed
                # here; the duplicate UUID persists harmlessly until a
                # writable copy is made.
                continue
            new_uid = str(_uuid.uuid4())
            other[_OBJ_UUID_KEY] = new_uid
            _remember_uuid_owner(new_uid, other)
            # If the object we were handed is itself a non-owner, return
            # its NEW UUID rather than the original.
            if other is obj:
                uid = new_uid
        return uid
    if not _is_writable(obj):
        return ""
    uid = str(_uuid.uuid4())
    obj[_OBJ_UUID_KEY] = uid
    _remember_uuid_owner(uid, obj)
    return uid


def get_object_uuid(obj: bpy.types.Object) -> str:
    """Get UUID of an object, or empty string if none assigned."""
    uid = obj.get(_OBJ_UUID_KEY)
    return str(uid) if uid else ""


_cache: dict[str, str] = {}  # uuid -> object name (lazy, warm)


def get_object_by_uuid(uid: str) -> "bpy.types.Object | None":
    """Find a Blender object by its UUID. O(1) when cached."""
    if not uid:
        raise ValueError("get_object_by_uuid called with empty/None UUID")
    # A fast cached lookup is safe only while the same Blender ID still owns
    # the UUID. A copy/replacement taking the old name cannot satisfy this.
    name = _cache.get(uid)
    if name:
        obj = bpy.data.objects.get(name)
        if (obj and obj.get(_OBJ_UUID_KEY) == uid
                and _owner_session_uids.get(uid) == int(obj.session_uid)):
            return obj
    matches = [obj for obj in bpy.data.objects
               if str(obj.get(_OBJ_UUID_KEY, "")) == uid]
    if not matches:
        return None
    owner = resolve_uuid_owner(matches, _prev_names.get(uid))
    if owner is None:
        raise _unresolved_uuid(uid)
    _remember_uuid_owner(uid, owner)
    return owner


def resolve_assigned(assigned) -> "bpy.types.Object | None":
    """Resolve an AssignedObject by UUID.

    Syncs display-cache names if the object was renamed.
    Auto-migration at load_post (core.migrate.migrate_legacy_data) populates
    UUIDs on every AssignedObject before this is reached — a missing UUID
    here means the record was never migrated (e.g. programmatic insertion
    without going through the proper API), and we return None to surface
    the failure rather than silently pick by name.
    """
    uid = getattr(assigned, "uuid", "")
    if not uid:
        raise ValueError(
            f"resolve_assigned: AssignedObject has empty UUID"
            f" (name={getattr(assigned, 'name', '?')})"
        )
    obj = get_object_by_uuid(uid)
    if obj is None:
        return None
    if hasattr(assigned, "name") and assigned.name != obj.name:
        _sync_all_names(uid, obj.name)
    return obj


def _sync_all_names(uid: str, new_name: str):
    """Sync every stored display name that references *uid* across the addon.

    Identity-carrying strings that get reconciled:
      - AssignedObject.name                  (group membership display)
      - PinVertexGroupItem.name              (composite [obj][vg])
      - MergePairItem.object_a / object_b    (snap-pair display)

    Not synced here: vertex-group renames.  Those don't change an object's
    name; ``resolve_pin`` handles VG-rename reconciliation via content hash.
    cross_stitch_json is UUID-only since the latest migration.
    """
    try:
        from ..models.groups import (
            decode_vertex_group_identifier,
            encode_vertex_group_identifier,
            get_addon_data,
            iterate_active_object_groups,
        )
        scene = bpy.context.scene
        if not scene:
            return
        for group in iterate_active_object_groups(scene):
            for assigned in group.assigned_objects:
                if assigned.uuid == uid and assigned.name != new_name:
                    assigned.name = new_name
            for pin_item in group.pin_vertex_groups:
                if pin_item.object_uuid == uid:
                    try:
                        plan = plan_pin_reference(pin_item)
                    except PinReferenceError:
                        continue  # preserve raw/hash; the external author rename remains
                    if plan is not None and pin_item.name != plan[3]:
                        pin_item.name = plan[3]
        root = get_addon_data(scene)
        if hasattr(root, "state"):
            for pair in root.state.merge_pairs:
                if pair.object_a_uuid == uid and pair.object_a != new_name:
                    pair.object_a = new_name
                if pair.object_b_uuid == uid and pair.object_b != new_name:
                    pair.object_b = new_name
    except Exception:
        pass


def plan_pin_reference(pin_item, obj=None):
    """Read current identity/name candidates without changing author RNA."""
    _, old_vg = decode_stored_pin_identifier(pin_item.name)
    uid = getattr(pin_item, "object_uuid", "")
    if not uid:
        raise PinReferenceError("Pin has no object UUID; current pair is unproven")
    # A previously warm UUID registry can predate a later duplicated ID.
    # Prove uniqueness from current raw IDs before trusting any cached owner.
    matches = [candidate for candidate in bpy.data.objects
               if candidate.get(_OBJ_UUID_KEY, "") == uid]
    if not matches:
        return None
    if len(matches) != 1:
        raise PinReferenceError("Pin object UUID is duplicated; current owner is unproven")
    if obj is None:
        obj = matches[0]
    elif obj != matches[0]:
        raise PinReferenceError("Pin plan owner does not match its unique UUID owner")
    if get_object_uuid(obj) != uid:
        raise PinReferenceError("Pin plan owner does not match its UUID")
    current_vg = old_vg
    if not _vg_slot_exists(obj, current_vg):
        try:
            stored_hash = int(pin_item.vg_hash) if pin_item.vg_hash else 0
        except (ValueError, TypeError) as exc:
            raise PinReferenceError("Pin membership hash is invalid") from exc
        if stored_hash:
            current_vg = resolve_vg_name(obj, old_vg, stored_hash)
        if not _vg_slot_exists(obj, current_vg):
            raise PinReferenceError("Pin vertex group is missing or has no unique current match")
    return obj, old_vg, current_vg, encode_pin_pair(obj.name, current_vg)


def resolve_pin(pin_item) -> "bpy.types.Object | None":
    """Resolve a PinVertexGroupItem's object by UUID, syncing stale names."""
    plan = plan_pin_reference(pin_item)
    if plan is None:
        return None
    obj, old_vg_name, vg_name, encoded = plan
    if pin_item.name != encoded:
        pin_item.name = encoded
        # Existing curve-key migration only after the complete pair is admitted.
        if old_vg_name != vg_name:
            old_key = f"_pin_{old_vg_name}"
            new_key = f"_pin_{vg_name}"
            if old_key in obj and new_key not in obj:
                obj[new_key] = obj[old_key]
                del obj[old_key]
    return obj


# ---------------------------------------------------------------------------
# Vertex group identification by content hash
# ---------------------------------------------------------------------------

def _get_vg_indices(obj: bpy.types.Object, vg_name: str) -> list[int]:
    """Get sorted vertex indices belonging to a vertex group."""
    vg = obj.vertex_groups.get(vg_name)
    if not vg:
        return []
    idx = vg.index
    indices = []
    for v in obj.data.vertices:
        for g in v.groups:
            if g.group == idx:
                indices.append(v.index)
                break
    return sorted(indices)


def _hash_indices(indices: list[int]) -> int:
    """Compute a 64-bit blake2b hash from sorted vertex indices."""
    if not indices:
        return 0
    data = struct.pack(f"<{len(indices)}I", *sorted(indices))
    digest = hashlib.blake2b(data, digest_size=8).digest()
    return struct.unpack("<Q", digest)[0]


def _get_curve_pin_indices(obj: bpy.types.Object, vg_name: str) -> list[int]:
    """Get pin indices from a curve's custom property."""
    import json as _json
    raw = obj.get(f"_pin_{vg_name}")
    if raw:
        return sorted(_json.loads(raw))
    return []


def compute_vg_hash(obj: bpy.types.Object, vg_name: str) -> int:
    """Compute a 64-bit content hash from vertex/pin indices."""
    if obj.type == "CURVE":
        return _hash_indices(_get_curve_pin_indices(obj, vg_name))
    return _hash_indices(_get_vg_indices(obj, vg_name))


def _iter_vg_candidates(obj: bpy.types.Object):
    """Yield (name, hash) for all vertex group / curve pin slots on obj."""
    if obj.type == "MESH":
        for vg in obj.vertex_groups:
            yield vg.name, _hash_indices(_get_vg_indices(obj, vg.name))
    elif obj.type == "CURVE":
        for key in obj.keys():
            if key.startswith("_pin_"):
                name = key[5:]
                yield name, _hash_indices(_get_curve_pin_indices(obj, name))


def _vg_slot_exists(obj: bpy.types.Object, vg_name: str) -> bool:
    """True when *obj* still carries a vertex group / curve pin slot *vg_name*.

    O(1)-ish name lookup, with no vertex walk, unlike ``compute_vg_hash``.
    """
    if obj.type == "MESH":
        return obj.vertex_groups.get(vg_name) is not None
    if obj.type == "CURVE":
        return obj.get(f"_pin_{vg_name}") is not None
    return False


def resolve_vg_name(obj: bpy.types.Object, stored_name: str, stored_hash: int) -> str:
    """Resolve the current vertex group name, following a RENAME if one happened.

    The hash exists to survive a rename: same contents, different name. A
    rename is therefore the only thing it can recover, and a rename always
    leaves ``stored_name`` absent. So when a slot named ``stored_name`` is
    still present this returns it immediately, hashing nothing.

    That order matters because the scan is expensive and must stay off the
    common path. ``_iter_vg_candidates`` calls ``_get_vg_indices`` once per
    vertex group, and each call walks every vertex and its group memberships,
    so a rigged character (70 vertex groups over 6.4k vertices) costs ~450k
    iterations per scan. ``resolve_pin`` reaches this function, and the Groups
    panel calls ``resolve_pin`` several times per redraw, so a scan on the
    common path is a per-redraw cost measured in hundreds of milliseconds.

    A membership change must not reach the scan, and does not. Welding or
    deleting vertices rewrites a pin group's index list, so its content hash
    stops matching while its name stays valid; the scan would then search for
    a rename that never happened, find nothing (no slot carries the old hash),
    and repair nothing, so the same fruitless scan would repeat on every
    redraw for as long as the mismatch lives in the .blend. Checking the name
    first is what keeps a content change from being read as a rename.

    Deliberate trade-off: if a group is renamed A -> B AND a NEW group named A
    is created, the stored name A resolves, so the pin stays on the new A
    rather than following the old contents to B. Distinguishing those two
    requires the full scan on every lookup, at the cost above. A plain rename,
    where the old name is gone, is followed exactly.
    """
    if not obj:
        return stored_name
    # The slot is still there: not a rename. Its contents may well have changed
    # (a weight edit, or a merge), which is not this function's business.
    if _vg_slot_exists(obj, stored_name):
        return stored_name
    # The stored name is gone, so a rename is the only thing the hash can
    # still recover. This is the one path that pays for the scan. Two groups
    # with the same contents (a Copy Vertex Group) cannot say which one was
    # renamed, so only a unique match is followed; otherwise the stored name
    # stays, finds no group, and the encoder refuses the pin by name.
    matches = [name for name, h in _iter_vg_candidates(obj) if h == stored_hash]
    return matches[0] if len(matches) == 1 else stored_name


# ---------------------------------------------------------------------------
# Depsgraph handler for rename detection
# ---------------------------------------------------------------------------

_prev_names: dict[str, str] = {}  # uuid -> last known name
_duplicate_detected: list[bool] = [False]  # set by handler, consumed after loop


@bpy.app.handlers.persistent
def _on_depsgraph_update(scene, depsgraph):
    """Detect object renames and keep display names in sync.

    Also strips duplicate UUIDs from copied objects immediately.
    Duplicate detection scans by UUID (never by name) — names are
    unstable across renames and can collide independently of identity.

    The UUID index is built once per invocation — O(M) — so that
    per-update duplicate checks are O(1) instead of O(M), giving
    O(N+M) total instead of the previous O(N*M).
    """
    # During animation playback objects are never renamed or duplicated,
    # yet a frame change reports every animated object as updated — which
    # would run the full O(N+M) rename/duplicate scan on every frame for
    # no possible result. Skip it while playing; renames/duplicates are
    # editing actions handled on the next non-playback update.
    screen = getattr(bpy.context, "screen", None)
    if screen is not None and screen.is_animation_playing:
        return
    # Fast path: skip the O(M) index build when no object was updated.
    has_object_update = False
    for update in depsgraph.updates:
        if isinstance(update.id, bpy.types.Object):
            has_object_update = True
            break
    if not has_object_update:
        return

    # Build a UUID -> owner index once: O(M). Copies carry custom properties,
    # but names and collection iteration order cannot identify their source.
    # Retain a previously observed live Blender ID; leave cold ambiguity intact.
    by_uuid: dict[str, list[bpy.types.Object]] = {}
    for obj in bpy.data.objects:
        uid = obj.get(_OBJ_UUID_KEY)
        if uid:
            by_uuid.setdefault(str(uid), []).append(obj)

    uuid_index: dict[str, bpy.types.Object | None] = {}
    for uid, objs in by_uuid.items():
        owner = resolve_uuid_owner(objs, _prev_names.get(uid))
        uuid_index[uid] = owner
        if owner is None:
            # Cold ambiguity: do not mutate either UUID or displayed membership.
            # Public consumers will refuse instead of choosing a shorter name.
            continue
        _remember_uuid_owner(uid, owner)
        if len(objs) == 1:
            continue
        # Strip UUID from every non-owner now — the copy may not
        # appear in depsgraph.updates (only the new object's update
        # fires for a duplicate; the original is unchanged), so the
        # per-update loop alone would miss it.
        for o in objs:
            if o is owner:
                continue
            if _is_writable(o):
                del o[_OBJ_UUID_KEY]
                _duplicate_detected[0] = True

    # Prune _prev_names for deleted objects.  The index already contains
    # every live UUID, so anything absent is stale.
    stale = [u for u in (_prev_names.keys() | _owner_session_uids.keys())
             if u not in uuid_index]
    for u in stale:
        _prev_names.pop(u, None)
        _cache.pop(u, None)
        _owner_session_uids.pop(u, None)

    renamed = False
    for update in depsgraph.updates:
        if not isinstance(update.id, bpy.types.Object):
            continue
        obj = update.id
        uid = obj.get(_OBJ_UUID_KEY)
        if not uid:
            continue
        uid = str(uid)
        # Duplicate check: O(1) lookup.  uuid_index's owner was
        # already resolved above via a live session ID; anything else is a
        # copy whose UUID the collision pass should have stripped.
        # This branch catches the residual case where _is_writable
        # refused the strip (library-linked).
        owner = uuid_index.get(uid)
        if owner is None:
            continue
        if owner is not obj:
            if _is_writable(obj):
                del obj[_OBJ_UUID_KEY]
                _duplicate_detected[0] = True
            continue
        # Genuine rename or first encounter
        prev = _prev_names.get(uid)
        if prev != obj.name:
            _cache[uid] = obj.name
            _sync_all_names(uid, obj.name)
            if prev is not None:
                renamed = True
        _prev_names[uid] = obj.name

    if _duplicate_detected[0]:
        _duplicate_detected[0] = False
        try:
            from ..ui.dynamics.overlay import apply_object_overlays
            apply_object_overlays()
        except Exception:
            pass
    elif renamed:
        # A pure rename updated name fields in group-membership /
        # pin-list / merge-pair PropertyGroups. Those panels otherwise
        # stay stuck on the old name until the next unrelated redraw.
        try:
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    area.tag_redraw()
        except Exception:
            pass


@bpy.app.handlers.persistent
def _clear_uuid_runtime_identity(*_args):
    """RNA IDs may be replaced by load/Undo/Redo; never carry hot authority over."""
    _prev_names.clear()
    _cache.clear()
    _owner_session_uids.clear()
    _duplicate_detected[0] = False


def _identity_boundary_handlers():
    return (bpy.app.handlers.load_pre, bpy.app.handlers.undo_pre,
            bpy.app.handlers.redo_pre)


def register():
    """Register the rename detection handler."""
    handlers = bpy.app.handlers.depsgraph_update_post
    # Drop any stale copies left by a previous reload (function identity
    # changes across reloads, so `in` alone wouldn't find them).
    for h in list(handlers):
        if getattr(h, "__name__", "") == "_on_depsgraph_update":
            handlers.remove(h)
    handlers.append(_on_depsgraph_update)
    _clear_uuid_runtime_identity()
    for boundary in _identity_boundary_handlers():
        for h in list(boundary):
            if getattr(h, "__name__", "") == "_clear_uuid_runtime_identity":
                boundary.remove(h)
        boundary.append(_clear_uuid_runtime_identity)


def unregister():
    """Unregister the rename detection handler."""
    handlers = bpy.app.handlers.depsgraph_update_post
    for h in list(handlers):
        if getattr(h, "__name__", "") == "_on_depsgraph_update":
            handlers.remove(h)
    for boundary in _identity_boundary_handlers():
        for h in list(boundary):
            if getattr(h, "__name__", "") == "_clear_uuid_runtime_identity":
                boundary.remove(h)
    _clear_uuid_runtime_identity()
