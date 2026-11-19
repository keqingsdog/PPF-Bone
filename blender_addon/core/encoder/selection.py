"""Scene-scoped object-group selection for solver payloads.

Bone material groups are owned by the bone authoring system. A stale,
unlinked proxy can remain in one of those groups after repeated proxy rebuilds;
because the normal UUID resolver is intentionally global, it must be screened
here before either geometry or parameters are encoded.
"""
from __future__ import annotations

import bpy

from ...models.groups import (
    is_bone_material_group,
    iterate_object_groups,
)
from ..uuid_registry import resolve_assigned


_reported_orphans: set[tuple[int, str, str]] = set()


def _bone_rna_references(obj, group_uuid: str, body_uuid: str):
    """Return real Bone RNA references by pointer or identity, across all rigs."""
    refs = []
    for arm in bpy.data.objects:
        if arm.type != "ARMATURE":
            continue
        for bone in arm.data.bones:
            settings = bone.ppf_bone_physics
            if (settings.proxy == obj or settings.body_uuid == body_uuid
                    or settings.group_uuid == group_uuid):
                refs.append(f"{arm.name}/{bone.name}")
    return refs


def _stale_dedicated_bone_proxy(scene, group, assigned, obj) -> bool:
    """Recognize only a provable, unlinked orphan in a dedicated bone group."""
    if not is_bone_material_group(scene, group.uuid):
        return False
    if group.object_type not in {"PDRD", "SOLID"} or obj.type != "MESH":
        return False
    if len(group.assigned_objects) != 1 or not assigned.included:
        return False
    body_uuid = assigned.uuid
    if (not body_uuid or assigned.uuid != obj.get("_solver_uuid")
            or not obj.get("ppf_bone_proxy_owned")
            or not obj.get("ppf_bone_uuid")
            or not obj.get("ppf_armature_uuid")):
        return False
    # A detached object may still be kept in a collection and later linked
    # back into a Scene. Treat that as live authoring data, not an orphan.
    if obj.users_scene or obj.users_collection:
        return False
    return not _bone_rna_references(obj, group.uuid, body_uuid)


def _resolve_unique_assignment(assigned, objects_by_uuid):
    """Resolve only after proving the UUID identifies exactly one datablock.

    The global UUID registry intentionally caches a first match for speed.
    Export cannot rely on that cache when a damaged file contains duplicates:
    it could select a different object (including one from another Scene).
    Refuse ambiguity locally without changing registry or scene data.
    """
    uid = str(getattr(assigned, "uuid", "") or "")
    if not uid:
        raise ValueError(
            f"Included object '{assigned.name}' in an active group has an empty UUID."
        )
    matches = objects_by_uuid.get(uid, ())
    if not matches:
        raise ValueError(
            f"Included object '{assigned.name}' has no object for UUID '{uid}'."
        )
    if len(matches) != 1:
        locations = []
        for obj in matches:
            scenes = sorted(s.name for s in obj.users_scene)
            collections = sorted(c.name for c in obj.users_collection)
            locations.append(
                f"{obj.name} (Scenes={scenes}, Collections={collections})"
            )
        raise ValueError(
            f"UUID '{uid}' assigned to '{assigned.name}' is ambiguous across "
            f"{len(matches)} Blender objects: {'; '.join(locations)}. "
            "Repair duplicate solver UUIDs before export."
        )
    obj = matches[0]
    resolved = resolve_assigned(assigned)
    if resolved is not obj:
        raise ValueError(
            f"UUID registry resolved '{assigned.name}' ({uid}) to a different "
            "object than the unique datablock scan."
        )
    return obj


def iter_export_groups(scene):
    """Yield active groups with one scene-consistent membership set.

    Hidden objects remain members because Blender keeps them in scene.objects.
    An unlinked record is ignored only for a dedicated bone material group
    whose owned mesh proxy identity is intact, belongs to a supported bone
    body group, has no Scene or Collection users, and has no Bone RNA pointer
    or identity reference. All other unresolved/out-of-scene included
    assignments are authoring errors and fail with the offending
    group/object/UUID instead of leaking globally-resolved geometry.
    """
    omitted = []
    selected = []
    # One local snapshot per export call keeps duplicate detection O(objects +
    # assignments), while naturally refreshing across Undo and Scene changes.
    objects_by_uuid = {}
    for obj in bpy.data.objects:
        uid = str(obj.get("_solver_uuid", "") or "")
        if uid:
            objects_by_uuid.setdefault(uid, []).append(obj)
    for group in iterate_object_groups(scene):
        if not group.active:
            continue
        orphan_records = []
        for assigned in group.assigned_objects:
            if not assigned.included:
                continue
            obj = _resolve_unique_assignment(assigned, objects_by_uuid)
            if scene.objects.get(obj.name) is obj:
                continue
            if _stale_dedicated_bone_proxy(scene, group, assigned, obj):
                orphan_records.append((assigned, obj))
                continue
            refs = _bone_rna_references(obj, group.uuid, assigned.uuid)
            owner = f"; Bone RNA references: {', '.join(refs)}" if refs else ""
            raise ValueError(
                f"Included object '{obj.name}' (UUID {assigned.uuid}) in group "
                f"'{group.name}' is not linked to the current Scene '{scene.name}'"
                f"{owner}. Link it to this Scene or remove the stale assignment."
            )
        if orphan_records:
            # A dedicated group can own exactly one body. The predicate above
            # enforces this, so dropping the group also drops every group-level
            # parameter and side channel in both payloads.
            omitted.extend(
                (group.name, assigned.name, assigned.uuid)
                for assigned, _obj in orphan_records
            )
            continue
        selected.append(group)
    if omitted:
        key = (scene.as_pointer(), scene.name, ";".join(uid for _, _, uid in omitted))
        if key not in _reported_orphans:
            _reported_orphans.add(key)
            details = ", ".join(
                f"{group}/{name} [{uid}]" for group, name, uid in omitted
            )
            print(
                "PPF encoder omitted unlinked, unreferenced dedicated bone "
                f"proxy records from this build: {details}"
            )
    return iter(selected)
