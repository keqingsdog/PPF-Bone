# Fork modification: changed from the upstream baseline.
# File: utils.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

from ...models.groups import get_addon_data
from ..state import ObjectGroup


def get_group_from_index(scene, group_index: int) -> ObjectGroup | None:
    """Return the group at *group_index* if it exists and is active, else None."""
    prop_name = f"object_group_{group_index}"
    group: ObjectGroup | None = getattr(get_addon_data(scene), prop_name, None)
    if group is None or not group.active:
        return None
    return group


def get_assigned_by_selection_uuid(group: ObjectGroup, selection_attr: str):
    """Return the AssignedObject whose uuid matches ``group.<selection_attr>``.

    Used by the Velocity/Collision-Window sub-panels which store the active
    object as a UUID string on the group rather than as a positional index.
    Returns None if nothing is selected or the uuid is stale.
    """
    sel_uuid = getattr(group, selection_attr, "") or ""
    if not sel_uuid or sel_uuid == "NONE":
        return None
    for assigned in group.assigned_objects:
        if assigned.uuid == sel_uuid:
            return assigned
    return None


def reset_object_display(obj):
    """Reset an object's display color and wireframe overlays to defaults."""
    obj.color = (1.0, 1.0, 1.0, 1.0)
    obj.show_wire = False
    obj.show_all_edges = False


def cleanup_group_references_for_object(group: ObjectGroup, object_uuid: str):
    """Drop everything on *group* that names *object_uuid*.

    Called at the moment an object's membership ends, by every path that
    disowns one, so no list on the group outlives the assignment it was
    written against. A reference left behind would be shown in a panel and
    would silently start meaning a different object once the group's slot
    was reused.
    """
    cleanup_pin_vertex_groups_for_object(group, object_uuid)
    cleanup_intersection_allowances_for_object(group, object_uuid)
    cleanup_merge_pairs_for_object(group.id_data, object_uuid)


def cleanup_merge_pairs_for_object(scene, object_uuid: str):
    """Remove every merge pair naming *object_uuid* from *scene*.

    A pair stitches two group members, so it ends with either membership.
    Removed here, when the membership ends, rather than found stale at the
    next Transfer, which refuses a pair naming an object outside every group.
    """
    from ...models.collection_utils import safe_update_index
    from ...models.groups import get_addon_data

    state = get_addon_data(scene).state
    selected_index = state.merge_pairs_index
    removed_indices = [
        index for index, pair in enumerate(state.merge_pairs)
        if object_uuid in (pair.object_a_uuid, pair.object_b_uuid)
    ]
    selected_survives = (
        0 <= selected_index < len(state.merge_pairs)
        and selected_index not in removed_indices
    )
    for index in reversed(removed_indices):
        state.merge_pairs.remove(index)
    if selected_survives:
        state.merge_pairs_index = selected_index - sum(
            index < selected_index for index in removed_indices
        )
    else:
        state.merge_pairs_index = safe_update_index(
            selected_index, len(state.merge_pairs)
        )


def cleanup_intersection_allowances_for_object(group: ObjectGroup, object_uuid: str):
    """Remove *object_uuid* from every intersection-allowance subset."""
    from ...models.intersection_allowances import (
        INTERSECTION_ALLOWANCES,
        allowance_objects,
    )
    from ...models.collection_utils import safe_update_index

    for spec in INTERSECTION_ALLOWANCES:
        collection = allowance_objects(group, spec)
        selected_index = getattr(group, spec.index_prop)
        selected_uuid = (
            collection[selected_index].uuid
            if 0 <= selected_index < len(collection) else ""
        )
        removed = False
        for index in range(len(collection) - 1, -1, -1):
            if collection[index].uuid == object_uuid:
                collection.remove(index)
                removed = True
        if removed:
            # Removing another member may shift the selected row. Keep its
            # identity when it survives; only clamp if that selection is gone.
            surviving_index = next(
                (index for index, item in enumerate(collection)
                 if selected_uuid and item.uuid == selected_uuid),
                None,
            )
            setattr(
                group,
                spec.index_prop,
                surviving_index if surviving_index is not None
                else safe_update_index(selected_index, len(collection)),
            )


def cleanup_pin_vertex_groups_for_object(group: ObjectGroup, object_uuid: str):
    """Remove pin vertex groups that reference the specified object (by UUID)."""
    selected_index = group.pin_vertex_groups_index
    pin_indices_to_remove = []
    for pin_index in range(len(group.pin_vertex_groups)):
        pin_item = group.pin_vertex_groups[pin_index]
        if pin_item.object_uuid == object_uuid:
            pin_indices_to_remove.append(pin_index)

    selected_survives = (
        0 <= selected_index < len(group.pin_vertex_groups)
        and selected_index not in pin_indices_to_remove
    )
    for pin_index in reversed(pin_indices_to_remove):
        group.pin_vertex_groups.remove(pin_index)

    if selected_survives:
        # Keep the exact surviving row, including separate pins on one object.
        group.pin_vertex_groups_index = selected_index - sum(
            index < selected_index for index in pin_indices_to_remove
        )
    elif selected_index >= len(group.pin_vertex_groups):
        group.pin_vertex_groups_index = max(0, len(group.pin_vertex_groups) - 1)
