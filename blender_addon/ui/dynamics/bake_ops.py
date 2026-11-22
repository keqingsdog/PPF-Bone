# File: bake_ops.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
# Fork modifications: bone and mesh bake recovery, author-data protection and playback improvements.

from ...i18n import props as _ppf_props
from ... import i18n as _ppf_i18n
import os
import re
import struct
import time
from types import SimpleNamespace

import bpy  # pyright: ignore
import numpy  # pyright: ignore

from bpy.app.handlers import persistent
from bpy.types import Operator  # pyright: ignore
from ...i18n import pgettext_iface as iface_, pgettext_tip as tip_  # pyright: ignore

from ...core.client import communicator as com
from ...core.curve_rod import curve_data_external_users, curve_key_bake_conflict


def _bake_geometry_users(obj):
    if obj.type == 'CURVE':
        return curve_data_external_users(obj)
    return obj.data.users - int(obj.data.use_fake_user)

from ...core.derived import is_server_busy_from_response as is_running
from ...core.pc2 import (
    _curve_cache,
    cleanup_mesh_cache,
    preflight_cache_cleanup,
    get_pc2_path,
    has_mesh_cache,
    load_curve_cache,
    modifiers_above_cache,
    object_pc2_key,
    read_pc2_frame,
    read_pc2_frame_count,
    read_pc2_n_verts,
    strip_modifiers_above_cache,
)
from ...core.utils import get_id_fcurves, redraw_all_areas, set_linear_interpolation
from ...models.collection_utils import safe_update_index
from ..state import iterate_active_object_groups
from .utils import cleanup_group_references_for_object, get_group_from_index, reset_object_display


# ---------------------------------------------------------------------------
# Job state (module-level singleton — read by UI panels, written by modal ops)
# ---------------------------------------------------------------------------

_bake_modal_owner = None

_bake_job: dict = {
    "active": False,
    "aborted": False,
    "kind": "",            # "single" (per-object) or "all"
    "objects": [],         # list of per-object entry dicts
    "object_cursor": 0,    # index into objects of the in-flight entry
    "total_frames_processed": 0,
    "total_frames": 0,
    "status": "",
    "finalize_error": None,
    "finalize_result": None,
    "scene": None,
}


def _reset_job() -> None:
    _bake_job.update({
        "active": False,
        "aborted": False,
        "kind": "",
        "objects": [],
        "object_cursor": 0,
        "total_frames_processed": 0,
        "total_frames": 0,
        "status": "",
        "finalize_error": None,
        "finalize_result": None,
        "scene": None,
        "material_archive": None,
        "source_archive": None,
    })


@persistent
def cancel_bake_before_file_change(*_args):
    """Never serialize partial bake work or retain RNA across file replacement."""
    global _bake_modal_owner
    if not _bake_job["active"]:
        return
    owner = _bake_modal_owner
    try:
        _abort_job()
    finally:
        try:
            if owner is not None:
                owner._externally_cancelled = True
                owner._detach_timer()
        finally:
            _bake_modal_owner = None
            _reset_job()


def is_bake_running() -> bool:
    """True while a modal Bake Animation job is still processing frames."""
    return bool(_bake_job.get("active"))


def bake_progress_snapshot() -> tuple[int, int, str, int]:
    """(done_frames, total_frames, status_line, object_count) for the UI."""
    return (
        int(_bake_job.get("total_frames_processed", 0)),
        int(_bake_job.get("total_frames", 0)),
        str(_bake_job.get("status", "")),
        len(_bake_job.get("objects", [])),
    )


# ---------------------------------------------------------------------------
# Bakeability detection
# ---------------------------------------------------------------------------


def _is_bakeable(obj) -> bool:
    """True when *obj* has PC2 simulation animation that can be baked."""
    return obj is not None and obj.type in ("MESH", "CURVE") and has_mesh_cache(obj)


def _has_animated_objects(context) -> bool:
    """Poll helper: True when at least one dynamic object has animation.

    Runs on every panel redraw. After the live server-state guards it uses
    the stateless ``scene_has_solver_cache`` scan, which reports a
    ContactSolverCache (or an in-memory curve cache) only on objects assigned
    to an active group -- the same objects the bake below walks, so a stray
    modifier on an unassigned copy neither enables nor disables the button.
    Recomputed every call, so it never goes stale.
    """
    if com.busy() or com.animation.frame:
        return False
    response = com.info.response
    if is_running(response):
        return False
    from ...core.pc2 import scene_has_solver_cache
    from ...core.rigid_pose import has_bone_pose_cache
    return scene_has_solver_cache(context.scene) or has_bone_pose_cache(context.scene)


def _has_unfetched_frames(scene) -> bool:
    """True when the connected run produced animation frames that have
    not been fetched into local PC2 yet. Baking now would commit an
    incomplete animation, so the Bake Animation operators refuse and
    ask the user to fetch first. Mirrors the render-time and panel
    warnings (see ``warn_missing_frames_on_render`` in ``core/pc2.py``).
    """
    try:
        response = com.info.response
        remote_frames = int(response.get("frame", 0)) if response else 0
        if remote_frames <= 0 or is_running(response):
            return False
        from ...models.groups import get_addon_data
        fetched = get_addon_data(scene).state.convert_fetched_frames_to_list()
        return (remote_frames - len(fetched)) > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Curve pose capture/apply (Bake Single Frame path)
# ---------------------------------------------------------------------------


def _capture_curve_pose(curve_data):
    """Snapshot current CV positions for a CURVE's data-block."""
    snapshot = []
    for spline in curve_data.splines:
        if spline.type == "BEZIER":
            snapshot.append((
                "BEZIER",
                [
                    (bp.co.copy(), bp.handle_left.copy(), bp.handle_right.copy())
                    for bp in spline.bezier_points
                ],
            ))
        else:
            snapshot.append((
                "POLY",
                [tuple(pt.co) for pt in spline.points],
            ))
    return snapshot


def _apply_curve_pose(curve_data, snapshot):
    """Write a :func:`_capture_curve_pose` result back onto *curve_data*."""
    for spline, (kind, points) in zip(curve_data.splines, snapshot):
        if kind == "BEZIER" and spline.type == "BEZIER":
            for bp, (co, hl, hr) in zip(spline.bezier_points, points):
                bp.handle_left_type = "FREE"
                bp.handle_right_type = "FREE"
                bp.co = co
                bp.handle_left = hl
                bp.handle_right = hr
        elif kind == "POLY" and spline.type != "BEZIER":
            for pt, co in zip(spline.points, points):
                pt.co[0] = co[0]
                pt.co[1] = co[1]
                pt.co[2] = co[2]


# ---------------------------------------------------------------------------
# Pre-bake snapshots for safe abort
# ---------------------------------------------------------------------------


def _snapshot_fcurves(obj) -> set:
    """Set of ``(data_path, array_index)`` for fcurves that exist on
    ``obj.data`` before the bake — used to discriminate newly-inserted
    fcurves during abort."""
    keys = set()
    if obj.data.animation_data and obj.data.animation_data.action:
        for fc in get_id_fcurves(obj.data):
            keys.add((fc.data_path, fc.array_index))
    return keys


def _snapshot_bezier_handles(obj):
    """Per-spline list of ``(handle_left_type, handle_right_type)`` tuples
    for every bezier point. Non-bezier splines yield ``None``. Returns
    ``None`` when *obj* isn't a CURVE."""
    if obj.type != "CURVE":
        return None
    snapshot = []
    for spline in obj.data.splines:
        if spline.type == "BEZIER":
            snapshot.append([
                (bp.handle_left_type, bp.handle_right_type)
                for bp in spline.bezier_points
            ])
        else:
            snapshot.append(None)
    return snapshot


def _restore_bezier_handles(obj, snapshot, *, data=None) -> None:
    if snapshot is None or (data is None and obj.type != "CURVE"):
        return
    data = data if data is not None else obj.data
    for spline, per_spline in zip(data.splines, snapshot):
        if spline.type != "BEZIER" or per_spline is None:
            continue
        for bp, (left, right) in zip(spline.bezier_points, per_spline):
            bp.handle_left_type = left
            bp.handle_right_type = right


def _force_bezier_free(obj) -> None:
    """Pin all bezier handle types to FREE so keyframe_insert captures
    the values we write directly (AUTO handles would be recomputed)."""
    if obj.type != "CURVE":
        return
    for spline in obj.data.splines:
        if spline.type == "BEZIER":
            for bp in spline.bezier_points:
                bp.handle_left_type = "FREE"
                bp.handle_right_type = "FREE"


# ---------------------------------------------------------------------------
# Queue builders
# ---------------------------------------------------------------------------


def _resolve_bake_start_frame() -> int:
    """Blender frame that PC2 index 0 was displayed at, so a bake reproduces
    the timeline placement the artist was scrubbing rather than snapping the
    result back to frame 1."""
    from ...core.playback_origin import cache_start_frame
    return cache_start_frame(bpy.context.scene)


def _make_entry(obj, group_index: int, assigned_uuid: str, assigned_name: str):
    """Build one bake-queue entry.

    Missing/empty caches retain the terminal stale-cache cleanup behavior.
    Mesh vertex-count mismatches also retain that legacy behavior. A present
    Curve animation must match its topology and contain finite coordinates;
    malformed data raises before any authored keyframe is touched.
    """
    from ...core.uuid_registry import get_or_create_object_uuid

    key = object_pc2_key(obj)
    path = get_pc2_path(key)
    if obj.type == "CURVE":
        from ...core.curve_output import resolve_read_path
        path = resolve_read_path(obj)

    entry = {
        "obj_uuid": get_or_create_object_uuid(obj),
        "obj_type": obj.type,
        "obj_name": assigned_name or obj.name,
        "key": key,
        "path": path,
        "n_frames": 0,
        "group_index": group_index,
        "assigned_uuid": assigned_uuid,
        "frames_done": 0,
        # Resolved once per bake so every frame of it keys onto the same
        # timeline placement: PC2 index f bakes onto scene frame
        # f + start_frame, matching where MESH_CACHE displayed it.
        "start_frame": _resolve_bake_start_frame(),
    }

    pc2_exists = bool(path) and os.path.exists(path)
    pc2_n_frames = read_pc2_frame_count(path) if pc2_exists else 0

    if obj.type == "MESH":
        sk_block = obj.data.shape_keys
        entry["pre_sk_block_existed"] = sk_block is not None
        entry["pre_sk_names"] = (
            {kb.name for kb in sk_block.key_blocks} if sk_block is not None else set()
        )
        entry["added_sk_names"] = []
        entry["sk_prefix"] = "ContactSolverBake"
        if pc2_n_frames > 0:
            n_verts = read_pc2_n_verts(path)
            if len(obj.data.vertices) == n_verts:
                entry["n_verts"] = n_verts
                entry["n_frames"] = pc2_n_frames
    else:
        entry["pre_fcurves"] = _snapshot_fcurves(obj)
        entry["handle_snapshot"] = _snapshot_bezier_handles(obj)
        if pc2_n_frames > 0:
            if key not in _curve_cache:
                load_curve_cache(key, filepath=path)
            cache = _curve_cache.get(key)
            if cache is not None and cache.shape[0] > 0:
                _validate_curve_bake_cache(obj, cache)
                entry["cache"] = cache
                entry["n_frames"] = int(cache.shape[0])
    return entry


def _validate_curve_bake_cache(obj, cache):
    """Reject a bad animation before any authored CV/keyframe is touched."""
    expected = sum(3 * len(s.bezier_points) if s.type == "BEZIER"
                   else len(s.points) for s in obj.data.splines)
    if cache.ndim != 3 or cache.shape[0] < 1 or cache.shape[1:] != (expected, 3):
        raise ValueError(f"Curve '{obj.name}' cache does not match {expected} CVs")
    if not numpy.isfinite(cache).all():
        raise ValueError(f"Curve '{obj.name}' cache has non-finite coordinates")


def _soft_static_exclusions(context, group_index=None):
    """Find soft STATIC meshes without resolving or changing group references."""
    from ...core.uuid_registry import get_object_by_uuid
    from ...models.groups import get_addon_data
    from ..state import N_MAX_GROUPS
    data = get_addon_data(context.scene)
    indices = range(N_MAX_GROUPS) if group_index is None else (group_index,)
    result = {}
    for gi in indices:
        group = getattr(data, f"object_group_{gi}", None)
        if (group is None or not group.active or group.object_type != "STATIC"
                or not group.enable_soft_constraint):
            continue
        for assigned in group.assigned_objects:
            obj = get_object_by_uuid(assigned.uuid)
            if obj is not None and obj.type == "MESH":
                result[assigned.uuid] = obj.name
    return result


def _soft_static_candidates(excluded):
    """Return readable soft-static outputs and their confirmation signatures."""
    from ...core.uuid_registry import get_object_by_uuid
    from ...core.pc2 import object_pc2_key_readonly
    result = {}
    for uid in excluded:
        obj = get_object_by_uuid(uid)
        if obj is None:
            continue
        key = object_pc2_key_readonly(obj)
        path = get_pc2_path(key) if key else ""
        if not path or not os.path.exists(path):
            path = get_pc2_path(obj.name)
        try:
            frames = read_pc2_frame_count(path)
            verts = read_pc2_n_verts(path)
            stat = os.stat(path)
        except (OSError, ValueError, struct.error):
            continue
        if frames > 0 and verts == len(obj.data.vertices) and stat.st_size >= 32 + frames * verts * 12:
            result[uid] = (path, frames, verts, stat.st_size, stat.st_mtime_ns,
                           tuple(modifiers_above_cache(obj)))
    return result


def _bake_group_signature(context, group_index=None):
    from ...models.groups import get_addon_data
    from ..state import N_MAX_GROUPS
    data = get_addon_data(context.scene)
    indices = range(N_MAX_GROUPS) if group_index is None else (group_index,)
    groups = []
    for gi in indices:
        group = getattr(data, f"object_group_{gi}", None)
        if group is not None and group.active:
            groups.append((gi, group.object_type, group.enable_soft_constraint,
                           group.assigned_objects_index if group_index is not None else None,
                           tuple(a.uuid for a in group.assigned_objects)))
    return tuple(groups)


def _build_queue(context, *, group_index: int | None = None, skipped=None, include_soft=False) -> list:
    """Return job entries for every assigned MESH/CURVE in scope. The
    "all groups" case (``group_index is None``) queues every assigned
    object in every active group except soft STATIC meshes, so finalize can purge them from their groups
    as a terminal Bake step. Entries without a readable cache get
    ``n_frames == 0`` and skip keyframing but still reach finalize for
    modifier + group cleanup."""
    from ...core.uuid_registry import resolve_assigned
    queue: list = []
    excluded = _soft_static_exclusions(context, group_index)
    admitted = _soft_static_candidates(excluded) if include_soft else {}

    def append_entry(obj, gi, assigned):
        entry = _make_entry(obj, gi, assigned.uuid, assigned.name)
        keys = obj.data.shape_keys if obj.type == "MESH" else None
        if assigned.uuid in admitted and keys is not None:
            entry["use_working_mesh"] = True
        queue.append(entry)

    if group_index is not None:
        group = get_group_from_index(context.scene, group_index)
        if group is None:
            return queue
        idx = group.assigned_objects_index
        if 0 <= idx < len(group.assigned_objects):
            assigned = group.assigned_objects[idx]
            if (assigned.uuid in excluded and assigned.uuid not in admitted) or (skipped and assigned.uuid in skipped):
                return queue
            obj = resolve_assigned(assigned)
            if obj is not None and obj.type in ("MESH", "CURVE"):
                append_entry(obj, group_index, assigned)
        return queue

    from ...models.groups import get_addon_data
    from ..state import N_MAX_GROUPS
    addon_data = get_addon_data(context.scene)
    for gi in range(N_MAX_GROUPS):
        prop_name = f"object_group_{gi}"
        group = getattr(addon_data, prop_name, None)
        if group is None or not group.active:
            continue
        for assigned in group.assigned_objects:
            if (assigned.uuid in excluded and assigned.uuid not in admitted) or (skipped and assigned.uuid in skipped):
                continue
            obj = resolve_assigned(assigned)
            if obj is None or obj.type not in ("MESH", "CURVE"):
                continue
            append_entry(obj, gi, assigned)
    from ...core import bone_bake
    bones=bone_bake.make_entry(context.scene)
    if bones is not None:queue.insert(0,bones)
    return queue


# ---------------------------------------------------------------------------
# Pre-bake validation
# ---------------------------------------------------------------------------

def _preview_all(context):
    """Read mesh cache headers without changing objects or migrating caches."""
    from ...core.uuid_registry import get_object_by_uuid
    from ...core.pc2 import object_pc2_key_readonly
    from ...models.groups import get_addon_data
    from ..state import N_MAX_GROUPS
    skipped, eligible, seen = {}, [], set()
    excluded = _soft_static_exclusions(context)
    for gi in range(N_MAX_GROUPS):
        group = getattr(get_addon_data(context.scene), f"object_group_{gi}", None)
        if group is None or not group.active:
            continue
        for assigned in group.assigned_objects:
            if assigned.uuid in excluded:
                continue
            obj = get_object_by_uuid(assigned.uuid)
            if obj is None or obj.type not in ("MESH", "CURVE"):
                continue
            pointer = obj.as_pointer()
            if pointer in seen:
                continue
            seen.add(pointer)
            keys = obj.data.shape_keys if obj.type == "MESH" else None
            conflict = False
            if keys and len(keys.key_blocks) > 1:
                uid = object_pc2_key_readonly(obj)
                path = get_pc2_path(uid) if uid else ""
                if not path or not os.path.exists(path):
                    path = get_pc2_path(obj.name)
                if os.path.exists(path):
                    conflict = (read_pc2_frame_count(path) > 0 and
                                read_pc2_n_verts(path) == len(obj.data.vertices))
            if conflict:
                names = [kb.name for kb in list(keys.key_blocks)[1:]]
                preview = ", ".join(names[:3])
                if len(names) > 3:
                    preview += iface_(" (+{count} more)").format(count=len(names) - 3)
                skipped[assigned.uuid] = f"{obj.name} [{preview}]"
            else:
                eligible.append(obj)
    return skipped, eligible, excluded


def _preview_signature(skipped, eligible, excluded):
    return (tuple(sorted(skipped.items())),
            tuple((obj.as_pointer(), tuple(modifiers_above_cache(obj)))
                  for obj in eligible), tuple(sorted(excluded.items())))


def _single_soft_preview(context, group_index):
    group = get_group_from_index(context.scene, group_index)
    if group is None or not (0 <= group.assigned_objects_index < len(group.assigned_objects)):
        return {}, {}
    uid = group.assigned_objects[group.assigned_objects_index].uuid
    excluded = _soft_static_exclusions(context, group_index)
    excluded = {uid: excluded[uid]} if uid in excluded else {}
    return excluded, _soft_static_candidates(excluded)


def _check_mesh_shape_keys(queue: list) -> list[str]:
    """Return blocker lines for MESH entries that already carry non-Basis
    shape keys. Baking absolute poses into additional shape keys would
    double-blend with the user's existing keys, so refuse the bake."""
    from ...core.uuid_registry import get_object_by_uuid
    blockers: list[str] = []
    for entry in queue:
        if entry.get("use_working_mesh"):
            continue
        if entry["obj_type"] != "MESH":
            continue
        # Entries with n_frames == 0 only need a modifier sweep — we
        # aren't adding shape keys for them, so existing non-Basis keys
        # can't double-blend against anything.
        if entry.get("n_frames", 0) <= 0:
            continue
        obj = get_object_by_uuid(entry["obj_uuid"])
        if obj is None:
            continue
        sk_block = obj.data.shape_keys
        if sk_block is None:
            continue
        # Index 0 is always the Basis (reference_key) in Blender.
        extras = [kb.name for kb in list(sk_block.key_blocks)[1:]]
        if extras:
            preview = ", ".join(extras[:3])
            if len(extras) > 3:
                preview += iface_(" (+{count} more)").format(count=len(extras) - 3)
            blockers.append(f"{entry['obj_name']} [{preview}]")
    return blockers


# ---------------------------------------------------------------------------
# Job lifecycle: start, tick, finalize, abort
# ---------------------------------------------------------------------------


def _single_channel_muted(curve):
    """A layered Action group's mute flag does not prove evaluation is muted."""
    return (curve.mute or (curve.group is not None and curve.group.mute
                           and not getattr(curve.id_data, "is_action_layered", False)))


def _single_animation_channels(id_block):
    """Read active Action/NLA bindings without changing author data or slots."""
    ad = id_block.animation_data
    channels = [("Action", curve) for curve in get_id_fcurves(id_block)]
    if ad.use_nla:
        tracks = [track for track in ad.nla_tracks if not track.mute]
        solo = [track for track in tracks if track.is_solo]
        if solo:
            tracks = solo
            # The active Action can still reset channels under a solo NLA track.
        def strip_channels(strips):
            for strip in strips:
                # A muted strip can still cause NLA to reset its channels
                # to default values; only a muted whole track is excluded.
                if strip.type == 'META':
                    yield from strip_channels(strip.strips)
                    continue
                action = getattr(strip, "action", None)
                if action is None:
                    continue
                # NLA strips expose the same action/action_slot binding as
                # AnimData. Reuse the project's slot-aware curve lookup.
                yield from get_id_fcurves(SimpleNamespace(animation_data=strip))
        for track in tracks:
            channels.extend(("NLA", curve) for curve in strip_channels(track.strips))
    return channels


def _curve_bake_driver_conflict(obj, *, include_fcurve_modifiers=False,
                                include_coordinate_actions=False):
    """Curve Animation guard; Single also checks Mesh coordinate channels.

    Single writes base coordinates whereas Animation creates coordinate keys.
    Neither option rejects unrelated data channels or Object transforms.
    """
    conflict = curve_key_bake_conflict(obj)
    if conflict:
        return conflict
    if (obj.data.animation_data is None
            or (obj.type != "CURVE" and not (obj.type == "MESH" and include_coordinate_actions))):
        return None
    label = obj.type.title()
    paths = set()
    for spline in obj.data.splines if obj.type == "CURVE" else ():
        if spline.type == "BEZIER":
            for point in spline.bezier_points:
                paths.update(point.path_from_id(prop)
                             for prop in ("co", "handle_left", "handle_right"))
        else:
            paths.update(point.path_from_id("co") for point in spline.points)
    def coordinate_path(path):
        if obj.type == "MESH":
            return bool(re.fullmatch(r"vertices\[\d+\]\.co", path))
        return path in paths

    for driver in obj.data.animation_data.drivers:
        if not driver.mute and coordinate_path(driver.data_path):
            return (f"{label} '{obj.name}' has a coordinate driver at "
                    f"{driver.data_path}[{driver.array_index}]; "
                    "Bake cannot replace its evaluated result. Mute the driver explicitly first.")
    if include_coordinate_actions:
        # Single Frame writes base coordinates, not animation keys. Existing
        # coordinate animation would reapply on the next depsgraph evaluation
        # after the external cache has already been deleted.
        channels = _single_animation_channels(obj.data)
        for owner, curve in channels:
            if not coordinate_path(curve.data_path) or _single_channel_muted(curve):
                continue
            if not (curve.keyframe_points or curve.sampled_points
                    or any(not modifier.mute for modifier in curve.modifiers)):
                continue
            return (f"{label} '{obj.name}' has coordinate {owner} animation at "
                    f"{curve.data_path}[{curve.array_index}]; Single Frame Bake "
                    "cannot replace its evaluated result. Handle this animation explicitly first.")
    if include_fcurve_modifiers:
        for curve in get_id_fcurves(obj.data):
            if not coordinate_path(curve.data_path):
                continue
            modifiers = [modifier.type for modifier in curve.modifiers if not modifier.mute]
            if modifiers:
                return (f"Curve '{obj.name}' has coordinate F-curve modifiers "
                        f"({', '.join(modifiers)}) at {curve.data_path}[{curve.array_index}]; "
                        "handle these modifiers explicitly before animation Bake.")
    return None



def _curve_shape_key_single_conflict(obj):
    """Curve shape keys deform final geometry, not evaluated spline RNA."""
    if obj.type != 'CURVE':
        return None
    key = obj.data.shape_keys
    if key is None or key.animation_data is None:
        return None
    active_values = {block.path_from_id('value') for block in list(key.key_blocks)[1:]
                     if not block.mute}
    mute_paths = {block.path_from_id('mute') for block in list(key.key_blocks)[1:]}
    paths = active_values | mute_paths
    if not key.use_relative and len(key.key_blocks) > 1:
        paths.add('eval_time')
    channels = _single_animation_channels(key)
    channels.extend(('driver', curve) for curve in key.animation_data.drivers)
    for owner, curve in channels:
        if curve.data_path not in paths or _single_channel_muted(curve):
            continue
        if owner != 'driver' and not (curve.keyframe_points or curve.sampled_points
                or any(not modifier.mute for modifier in curve.modifiers)):
            continue
        return (f"Curve '{obj.name}' has active shape-key {owner} animation at "
                f"{curve.data_path}; Single Frame Bake cannot freeze its evaluated "
                "geometry without changing authored shape keys.")
    return None


def _capture_single_bake_pose(obj, depsgraph):
    """Read-only Single admission and frozen pose, before any cache/author writes.

    Single writes the existing base datablock. Unlike Animation Bake it cannot
    compensate authored shape keys, coordinate drivers or downstream deformers.
    A batch must call this for every target before committing bone baking too.
    """
    if obj.type not in {"MESH", "CURVE"}:
        raise ValueError(f"Object '{obj.name}' is not a Mesh or Curve")
    if not any(owner == obj for owner in bpy.context.scene.objects):
        raise ValueError(f"Single Frame Bake target '{obj.name}' is not in this Scene")
    if obj.library is not None or obj.data.library is not None:
        raise ValueError(f"Object '{obj.name}' or its data is linked from a library; "
                         "make an independent local copy before Single Frame Bake.")
    if obj.data.override_library is not None:
        raise ValueError(f"Object '{obj.name}' geometry is a library override; "
                         "Single Frame base-coordinate writes do not persist on that data.")
    if (_bake_geometry_users(obj) > 1
            or len(obj.users_scene) > 1):
        raise ValueError(f"Object '{obj.name}' has shared data or Scene ownership; "
                         "make its ownership explicit before Single Frame Bake.")
    conflict = _curve_bake_driver_conflict(obj, include_coordinate_actions=True)
    if conflict:
        raise ValueError(conflict)
    conflict = _curve_shape_key_single_conflict(obj)
    if conflict:
        raise ValueError(conflict)
    if obj.type == "MESH":
        if obj.data.shape_keys is not None:
            raise ValueError(f"Mesh '{obj.name}' has authored shape keys; Single Frame "
                             "Bake cannot replace their evaluated result without changing them.")
        from ...core.pc2 import MODIFIER_NAME
        after_cache = False
        for modifier in obj.modifiers:
            if modifier.name == MODIFIER_NAME and modifier.type == 'MESH_CACHE':
                after_cache = True
            elif after_cache:
                animated_visibility = False
                if obj.animation_data is not None:
                    path = modifier.path_from_id('show_viewport')
                    channels = _single_animation_channels(obj)
                    channels.extend(('driver', curve) for curve in obj.animation_data.drivers)
                    animated_visibility = any(
                        curve.data_path == path and not _single_channel_muted(curve)
                        and (owner == 'driver' or curve.keyframe_points or curve.sampled_points
                             or any(not item.mute for item in curve.modifiers))
                        for owner, curve in channels)
                if not modifier.show_viewport and not animated_visibility:
                    continue
                raise ValueError(f"Mesh '{obj.name}' has an enabled or visibility-animated modifier after "
                                 f"the solver cache ('{modifier.name}'); Single Frame "
                                 "Bake would apply its evaluated effect again.")
    evaluated = obj.evaluated_get(depsgraph)
    if obj.type == "MESH":
        positions = [v.co.copy() for v in evaluated.data.vertices]
        if len(positions) != len(obj.data.vertices):
            raise ValueError(f"Mesh '{obj.name}' evaluated vertex count "
                             f"{len(positions)} differs from base {len(obj.data.vertices)}; "
                             "Single Frame Bake requires unchanged topology.")
        if not numpy.isfinite(numpy.asarray(positions)).all():
            raise ValueError(f"Mesh '{obj.name}' has non-finite evaluated coordinates")
        return "MESH", positions
    if not hasattr(evaluated.data, 'splines'):
        raise ValueError(f"Curve '{obj.name}' no longer evaluates to Curve data")
    def topology(data):
        return [(s.type, len(s.bezier_points) if s.type == 'BEZIER' else len(s.points))
                for s in data.splines]
    if topology(evaluated.data) != topology(obj.data):
        raise ValueError(f"Curve '{obj.name}' evaluated control-point topology differs")
    pose = _capture_curve_pose(evaluated.data)
    if any(not numpy.isfinite(numpy.asarray(points)).all() for _kind, points in pose):
        raise ValueError(f"Curve '{obj.name}' has non-finite evaluated coordinates")
    return "CURVE", pose

def _apply_single_mesh_positions(obj, positions):
    for vertex, pos in zip(obj.data.vertices, positions):
        vertex.co = pos


def _stage_single_meshes(scene, captured):
    """Write all base coordinates before any bone or cache teardown.

    If a write fails, restore every base mesh while the original caches and
    membership still exist. Cleanup remains a separate commit boundary.
    """
    from ...core.uuid_registry import get_object_by_uuid
    saved = []
    try:
        for uid, (kind, positions) in captured.items():
            if kind != "MESH":
                continue
            obj = get_object_by_uuid(uid)
            if obj is None or obj.type != "MESH":
                raise ValueError("Single Frame Bake target disappeared before commit")
            saved.append((obj, [v.co.copy() for v in obj.data.vertices]))
            _apply_single_mesh_positions(obj, positions)
    except BaseException:
        for obj, positions in reversed(saved):
            for vertex, pos in zip(obj.data.vertices, positions):
                vertex.co = pos
        raise
    return saved


def _restore_single_meshes(saved):
    for obj, positions in reversed(saved):
        for vertex, pos in zip(obj.data.vertices, positions):
            vertex.co = pos


def _animation_bake_ownership_error(obj, scene):
    """Read-only ordinary ownership check against the captured originating Scene."""
    if not any(owner == obj for owner in scene.objects):
        return f"Animation Bake target '{obj.name}' is not in its originating Scene"
    if obj.library is not None or obj.data.library is not None:
        return (f"Object '{obj.name}' or its data is linked from a library; "
                "Animation Bake requires independently local object data.")
    if obj.type == "MESH" and obj.data.override_library is not None:
        return (f"Mesh '{obj.name}' geometry is a library override; "
                "Animation Bake shape keys do not persist reliably on that data.")
    if _bake_geometry_users(obj) > 1 or len(obj.users_scene) > 1:
        return f"Object '{obj.name}' has shared data or Scene ownership"
    return None


def _validate_live_bake_ownership(entry, obj):
    scene = entry.get('_owner_scene')
    if scene is None:
        raise ValueError("Animation Bake originating Scene is unavailable")
    if bpy.context.scene != scene:
        raise ValueError("Active Scene changed during Animation Bake; original inputs retained")
    conflict = _animation_bake_ownership_error(obj, scene)
    if conflict:
        raise ValueError(conflict)


def _start_job(queue: list, kind: str) -> tuple[bool, str]:
    if _bake_job["active"]:
        return False, iface_("Another bake job is in progress")
    if not queue:
        return False, iface_("No bakeable objects")

    from ...core.uuid_registry import get_object_by_uuid
    # Legacy/externally edited group memberships may list the same Curve twice.
    # Refuse before creating nested working Actions on the same data-block.
    curve_data = set()
    for entry in queue:
        if entry["obj_type"] in {"CURVE", "MESH"} and entry["n_frames"] > 0:
            obj = get_object_by_uuid(entry["obj_uuid"])
            if obj is not None:
                conflict = _animation_bake_ownership_error(obj, bpy.context.scene)
                if conflict:
                    return False, conflict
                if obj.type == 'CURVE' and obj.data.shape_keys is not None:
                    return False, (f"Curve '{obj.name}' has authored shape keys; "
                                   "Animation Bake cannot replace their final evaluated geometry.")
                conflict = _curve_bake_driver_conflict(obj, include_fcurve_modifiers=True)
                if conflict:
                    return False, conflict
                pointer = obj.data.as_pointer()
                if pointer in curve_data:
                    return False, "Object data is assigned more than once in the bake queue"
                curve_data.add(pointer)
    origin_scene = bpy.context.scene
    for entry in queue:
        entry['_owner_scene'] = origin_scene
    from ...core import full_bake_restore
    try:
        recovery = full_bake_restore.begin(origin_scene, queue)
    except (ValueError, RuntimeError, OSError) as exc:
        return False, str(exc)
    from ...core import source_materials
    source_context = dict(scene=origin_scene, objects=queue, recovery=recovery, source_archive=None)
    try:
        source_materials.retain(source_context)
    except Exception as exc:
        full_bake_restore.discard_pending(recovery)
        return False, str(exc)
    prepared = []
    try:
        for entry in queue:
            if entry["obj_type"] == "CURVE" and entry["n_frames"] > 0:
                obj = get_object_by_uuid(entry["obj_uuid"])
                if obj is None:
                    raise ValueError("Curve bake target disappeared")
                if _bake_geometry_users(obj) > 1 or len(obj.users_scene) > 1:
                    raise ValueError(f"Curve '{obj.name}' has shared data or Scene ownership")
                prepared.append((obj, entry))
                _begin_curve_bake(obj, entry)
            elif entry["obj_type"] == "MESH" and entry["n_frames"] > 0:
                obj = get_object_by_uuid(entry["obj_uuid"])
                if obj is None:
                    raise ValueError("Mesh bake target disappeared")
                if _bake_geometry_users(obj) > 1 or len(obj.users_scene) > 1:
                    raise ValueError("Mesh has shared data or Scene ownership")
                if entry.get("use_working_mesh"):
                    full_bake_restore.prepare_working_mesh(obj, entry)
                if obj.data.shape_keys is not None:
                    prepared.append((obj, entry))
                    _begin_mesh_action(obj, entry)
                else:
                    prepared.append((obj, entry))
                    _begin_created_mesh_keys(obj, entry)
    except Exception as exc:
        source_holder_error = ''
        try:
            source_materials.verify(source_context)
        except Exception as source_exc:
            source_holder_error = str(source_exc)
        from ...core import prepared_materials
        failed_context = dict(source_context, prepared=prepared,
                              finalize_error=str(exc), source_holder_error=source_holder_error)
        for entry in queue:
            if entry.get("original_mesh_row") is not None:
                _rollback_staged_mesh(entry)
        try:
            archive = prepared_materials.retain(failed_context)
        except Exception as retain_exc:
            return False, ('RETENTION_FAILED: prepared output was not rolled back; materials are not verified. '
                           + str(exc) + '; source holder: ' + source_holder_error + '; ' + str(retain_exc))
        return False, ('PARTIAL_RETAINED: prepared output was not rolled back. '
                       'Not a Restore record; inspect Text ' + archive.name + '. ' + str(exc))

    _bake_job.update({
        "active": True,
        "aborted": False,
        "kind": kind,
        "objects": queue,
        "object_cursor": 0,
        "total_frames_processed": 0,
        "total_frames": sum(e["n_frames"] for e in queue),
        "status": "Baking...",
        "finalize_error": None,
        "finalize_result": None,
        "scene": origin_scene,
        "recovery": recovery,
        "source_archive": source_context["source_archive"],
        "material_archive": None,
    })
    return True, ""


def _sk_value_fcurve(shape_keys, sk_name):
    """Find the fcurve driving ``key_blocks["<sk_name>"].value`` on the
    shape-keys action, or return ``None``."""
    if shape_keys is None or shape_keys.animation_data is None:
        return None
    action = shape_keys.animation_data.action
    if action is None:
        return None
    target = f'key_blocks["{sk_name}"].value'
    for fc in get_id_fcurves(shape_keys):
        if fc.data_path == target:
            return fc
    return None


def _process_one_frame_mesh(entry, obj) -> None:
    """MESH: one shape key per PC2 frame. Positions via foreach_set
    (native C loop — orders of magnitude faster than per-vertex
    keyframe_insert). Value is keyed 0→1→0 with CONSTANT interpolation
    so each shape key is "on" for exactly its frame."""
    _validate_mesh_action(obj, entry)
    f = entry["frames_done"]
    scene_frame = f + entry["start_frame"]
    n_verts = entry["n_verts"]
    positions = read_pc2_frame(entry["path"], f, n_verts)
    if not numpy.isfinite(positions).all():
        raise ValueError(f"Mesh '{obj.name}' cache has non-finite coordinates")
    flat = numpy.ascontiguousarray(positions, dtype=numpy.float32).ravel()

    sk_name = f"{entry['sk_prefix']}_{scene_frame:05d}"
    existing_tx = entry.get("mesh_action_transaction")
    if existing_tx is not None:
        prior_curve = _sk_value_fcurve(existing_tx["key"], sk_name)
        prior = _mesh_fcurve_snapshot(prior_curve) if prior_curve is not None else None
        expected = existing_tx["expected_curves"].get((f'key_blocks["{sk_name}"].value', 0))
        if prior != expected:
            existing_tx["preserve_working"] = True
            raise ValueError("Author changed a Shape Key curve before Bake could write it")
    kb = obj.shape_key_add(name=sk_name, from_mix=False)
    entry["added_sk_names"].append(kb.name)
    key_transaction = entry.get("mesh_created_key_transaction") or entry.get("mesh_action_transaction")
    if key_transaction is not None:
        key_transaction["keys"].append((kb, kb.name, _mesh_key_digest(flat),
             (kb.slider_min, kb.slider_max, kb.vertex_group, kb.interpolation)))
    if existing_tx is not None and kb.name != sk_name:
        actual_curve = _sk_value_fcurve(existing_tx["key"], kb.name)
        actual_snapshot = _mesh_fcurve_snapshot(actual_curve) if actual_curve is not None else None
        actual_expected = existing_tx["expected_curves"].get((f'key_blocks["{kb.name}"].value', 0))
        if actual_snapshot != actual_expected:
            existing_tx["preserve_working"] = True
            obj.shape_key_remove(kb)
            entry["added_sk_names"].pop()
            key_transaction["keys"].pop()
            raise ValueError("Author changed a disambiguated Shape Key curve before Bake")
    kb.data.foreach_set("co", flat)

    created = entry.get("mesh_created_key_transaction")
    if created is not None and created.get("encoding") == "ABSOLUTE_STEP_V1":
        kb.interpolation = 'KEY_LINEAR'
        created["keys"][-1] = (kb, kb.name, _mesh_key_digest(flat),
                              (kb.slider_min, kb.slider_max, kb.vertex_group, kb.interpolation))
        created["sample_frames"].append(float(kb.frame))
        key = created["key"]
        if f == 0:
            key.eval_time = key.reference_key.frame
            key.keyframe_insert("eval_time", frame=scene_frame - 1)
        key.eval_time = kb.frame
        key.keyframe_insert("eval_time", frame=scene_frame)
        key.eval_time = key.reference_key.frame
        key.keyframe_insert("eval_time", frame=scene_frame + 1)
        created["action"] = key.animation_data.action
        created["slot"] = _curve_action_slot(key)
        curves = list(get_id_fcurves(key))
        if len(curves) != 1 or curves[0].data_path != 'eval_time':
            raise ValueError("Unexpected Shape Key animation during Bake")
        curves[0].extrapolation = 'CONSTANT'
        for point in curves[0].keyframe_points:
            point.interpolation = 'CONSTANT'
        created["expected_curves"] = _mesh_action_snapshot(key)
        key.eval_time = curves[0].evaluate(bpy.context.scene.frame_current_final)
        return

    kb.value = 0.0
    kb.keyframe_insert("value", frame=scene_frame - 1)
    kb.value = 1.0
    kb.keyframe_insert("value", frame=scene_frame)
    kb.value = 0.0
    kb.keyframe_insert("value", frame=scene_frame + 1)

    created = entry.get("mesh_created_key_transaction")
    if created is not None:
        created["action"] = obj.data.shape_keys.animation_data.action
        created["slot"] = _curve_action_slot(obj.data.shape_keys)
    tx = entry.get("mesh_action_transaction")
    if tx is not None and entry["frames_done"] == 0 and tx["working_slot"] is None:
        tx["working_slot"] = _curve_action_slot(tx["key"])
    fc = _sk_value_fcurve(obj.data.shape_keys, kb.name)
    if fc is not None:
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT"
        if tx is not None:
            tx["expected_curves"][(fc.data_path, fc.array_index)] = _mesh_fcurve_snapshot(fc)


def _process_one_frame_curve(entry, obj) -> None:
    """CURVE: per-CV keyframes on bezier_points/points. CV counts are
    small so the per-point keyframe_insert cost is tolerable."""
    f = entry["frames_done"]
    frame = f + entry["start_frame"]
    frame_data = entry["cache"][f]
    cv_i = 0
    for spline in obj.data.splines:
        if spline.type == "BEZIER":
            for bp in spline.bezier_points:
                bp.handle_left = frame_data[cv_i]
                cv_i += 1
                bp.co = frame_data[cv_i]
                cv_i += 1
                bp.handle_right = frame_data[cv_i]
                cv_i += 1
                bp.keyframe_insert("co", frame=frame)
                bp.keyframe_insert("handle_left", frame=frame)
                bp.keyframe_insert("handle_right", frame=frame)
        else:
            for pt in spline.points:
                c = frame_data[cv_i]
                cv_i += 1
                pt.co[0] = float(c[0])
                pt.co[1] = float(c[1])
                pt.co[2] = float(c[2])
                pt.keyframe_insert("co", frame=frame)


def _process_one_frame(entry) -> bool:
    """Process one PC2 frame of *entry*'s object. Returns ``False`` if
    the object is no longer resolvable (skip it)."""
    if bpy.context.scene != entry.get('_owner_scene'):
        raise ValueError("Active Scene changed during Animation Bake; original inputs retained")
    if entry["obj_type"] == "BONE":
        from ...core import bone_bake
        bone_bake.sample(entry)
        return True
    from ...core.uuid_registry import get_object_by_uuid
    obj = get_object_by_uuid(entry["obj_uuid"])
    if obj is None:
        return False
    _validate_live_bake_ownership(entry, obj)
    if entry["obj_type"] == "MESH":
        _process_one_frame_mesh(entry, obj)
    else:
        _validate_curve_transaction(obj, entry)
        _process_one_frame_curve(entry, obj)
        tx = entry.get("curve_transaction")
        if tx is not None:
            tx["expected_curves"] = _mesh_action_snapshot(obj.data)
            tx["expected_pose"] = _capture_curve_pose(obj.data)
            tx["expected_handles"] = _snapshot_bezier_handles(obj)
        if tx is not None and entry["frames_done"] == 0 and tx["working_slot"] is None:
            # Blender 4.4+ may create the first slot lazily on key insertion.
            # Accept our first write only, never refresh over an external rebind.
            tx["working_slot"] = _curve_action_slot(obj.data)
    entry["frames_done"] += 1
    return True


def _tick_job(*, budget_ms: int = 40) -> bool:
    """Run work until the per-tick budget expires. Returns ``True`` if
    more frames remain after the tick, ``False`` when keyframing is
    complete (or aborted)."""
    deadline = time.perf_counter() + budget_ms / 1000.0
    objects = _bake_job["objects"]
    while time.perf_counter() < deadline:
        if _bake_job["aborted"]:
            return False
        idx = _bake_job["object_cursor"]
        if idx >= len(objects):
            return False
        entry = objects[idx]
        if entry["frames_done"] >= entry["n_frames"]:
            _bake_job["object_cursor"] += 1
            continue
        _bake_job["status"] = iface_("Baking {name} [{done}/{total}]").format(
            name=entry["obj_name"],
            done=entry["frames_done"] + 1,
            total=entry["n_frames"],
        )
        if not _process_one_frame(entry):
            _bake_job["object_cursor"] += 1
            continue
        _bake_job["total_frames_processed"] += 1
    return _bake_job["object_cursor"] < len(_bake_job["objects"])


def _retain_bake_source_action(entry):
    """Keep displaced authors when their new output cannot safely be aborted."""
    if entry['obj_type'] == 'BONE':
        for snap in entry['snapshots']:
            try:
                action = snap['action']
                if action is not None and action.users == 0 and not action.use_fake_user:
                    action.use_fake_user = True
            except ReferenceError:
                pass
        return
    tx = entry.get("curve_transaction") or entry.get("mesh_action_transaction")
    if tx is None:
        return
    try:
        action = tx["action"]
        if action is not None and action.users == 0 and not action.use_fake_user:
            action.use_fake_user = True
            tx["retained_source_action"] = True
    except ReferenceError:
        pass  # Respect an author explicitly deleting the source Action.


def _remove_finalized_members(context, uuids_per_group):
    from ...models.groups import get_addon_data
    addon_data = get_addon_data(_bake_job["scene"])
    for gi, uuids in uuids_per_group.items():
        group = getattr(addon_data, f"object_group_{gi}", None)
        if group is None:
            continue
        for i in range(len(group.assigned_objects) - 1, -1, -1):
            if group.assigned_objects[i].uuid in uuids:
                cleanup_group_references_for_object(group, group.assigned_objects[i].uuid)
                group.assigned_objects.remove(i)
        group.assigned_objects_index = safe_update_index(
            group.assigned_objects_index, len(group.assigned_objects))


def _finalize_job(context) -> tuple[int, int]:
    """Finalize with truthful partial-commit boundaries, never blind rollback.

    All known cleanup refusals are checked before any finalizer writes. A later
    filesystem/RNA failure preserves already committed (or uncertain) output;
    the modal error path aborts only entries whose original inputs remain safe.
    This is not an atomic filesystem transaction or native Undo mechanism.
    """
    from ...core.uuid_registry import get_object_by_uuid
    if _bake_job.get("finalize_error"):
        raise RuntimeError("Bake finalization already failed; abort/reset the remaining "
                           "job before retrying. " + _bake_job["finalize_error"])
    if _bake_job.get("finalize_result") is not None:
        return _bake_job["finalize_result"]
    uuids_per_group = {}
    objects_baked = 0
    current = None
    groups_attempted = False
    try:
        if context.scene != _bake_job.get('scene'):
            raise ValueError("Active Scene changed during Animation Bake; original inputs retained")
        # Resolve, validate and admit the entire batch before even bone commit.
        for entry in _bake_job["objects"]:
            if entry["obj_type"] == "BONE":
                continue
            obj = get_object_by_uuid(entry["obj_uuid"])
            if obj is not None:
                _validate_live_bake_ownership(entry, obj)
            if entry["obj_type"] == "CURVE":
                _validate_curve_transaction(obj, entry)
            elif entry["obj_type"] == "MESH":
                _validate_mesh_action(obj, entry)
            if obj is None:
                raise ValueError(f"Bake target disappeared: {entry['obj_name']}")
            from ...core.curve_output import CurveOutputIdentityError
            try:
                plan = preflight_cache_cleanup(obj)
            except CurveOutputIdentityError:
                # The original bound generation cannot be recovered; an Abort
                # must not discard this target's only sampled output.
                entry['_finalize_state'] = 'output_preserved'
                raise
            if (entry['frames_done'] > 0 and plan['delete_path'] is not None
                    and not plan.get('present')):
                # Missing input proves neither successful cleanup nor rollback
                # safety. Keep this target's sampled output for recovery.
                entry['_finalize_state'] = 'output_preserved'
                raise FileNotFoundError(f"Bake input disappeared before cleanup: {plan['delete_path']}")
            entry['_cleanup_plan'] = plan
        from ...core import full_bake_restore
        from ...core import source_materials
        source_materials.verify(_bake_job)
        full_bake_restore.preflight_sources(_bake_job['recovery'], _bake_job['objects'])
        from ...core import bone_bake
        for entry in _bake_job["objects"]:
            if entry["obj_type"] == "BONE":
                current = entry
                entry['_finalize_state'] = 'attempted'
                bone_bake.finish(entry)
                entry['_finalize_state'] = 'committed'
        for entry in _bake_job["objects"]:
            if entry["obj_type"] == "BONE":
                continue
            current = entry
            obj = get_object_by_uuid(entry["obj_uuid"])
            upstream = modifiers_above_cache(obj) if entry['obj_type'] == 'MESH' else []
            receipt = {'require_existing': entry['frames_done'] > 0}
            entry['_finalize_receipt'] = receipt
            entry['_finalize_state'] = 'attempted'
            # Actual unlink comes before modifier removal / Curve memory unload.
            # The receipt marks disk removal immediately, even if later RNA fails.
            from ...core.full_bake_restore import suspend_object
            suspend_object(obj)
            # Retain the exact input cache/modifiers for explicit recovery.
            # Disable their display after sampling; never unlink Undo inputs.
            for name in upstream + ["ContactSolverCache"]:
                modifier=obj.modifiers.get(name)
                if modifier is not None:
                    modifier.show_viewport=False;modifier.show_render=False
            if entry['obj_type']=='CURVE':
                from ...core.pc2 import _curve_cache, object_pc2_key_readonly
                _curve_cache.pop(object_pc2_key_readonly(obj),None)
            entry['_finalize_state'] = 'output_preserved'
            reset_object_display(obj)
            if entry['obj_type'] == 'CURVE' and entry['frames_done'] > 0:
                paths = set()
                for spline in obj.data.splines:
                    if spline.type == 'BEZIER':
                        for point in spline.bezier_points:
                            paths.update(point.path_from_id(p) for p in ('co', 'handle_left', 'handle_right'))
                    else:
                        paths.update(point.path_from_id('co') for point in spline.points)
                start = entry['start_frame']
                end = start + entry['frames_done'] - 1
                for fc in get_id_fcurves(obj.data):
                    if fc.data_path in paths:
                        for kp in fc.keyframe_points:
                            if start <= kp.co.x <= end:
                                kp.interpolation = 'LINEAR'
                tx = entry.get("curve_transaction")
                if tx is not None:
                    tx["expected_curves"] = _mesh_action_snapshot(obj.data)
            entry['_finalize_state'] = 'committed'
            _retain_bake_source_action(entry)
            if entry['frames_done'] > 0:
                objects_baked += 1
            uuids_per_group.setdefault(entry['group_index'], set()).add(entry['assigned_uuid'])
        groups_attempted = True
        from ...core import full_bake_restore
        full_bake_restore.commit(context.scene,_bake_job['recovery'],_bake_job['objects'])
        result = (objects_baked, _bake_job['total_frames_processed'])
        _bake_job['finalize_result'] = result
        return result
    except Exception as exc:
        if current is not None and current.get('_finalize_state') == 'attempted':
            receipt = current.get('_finalize_receipt', {})
            if current['obj_type'] == 'BONE':
                safe = current.get('_finish_rollback_complete', False)
            else:
                safe = (not receipt.get('disk_removed') and receipt.get('input_preserved')
                        and receipt.get('phase') in {'preflight', 'unlink'})
            current['_finalize_state'] = 'pending' if safe else 'output_preserved'
        errors = [str(exc)]
        # Retain a recoverable snapshot even when finalization only partly
        # committed. Abort must not subsequently mutate its admitted outputs.
        recovery=_bake_job.get('recovery')
        if recovery is not None and any(
                entry.get('_finalize_state') == 'committed' or
                (entry.get('_finalize_state') == 'output_preserved' and
                 (entry.get('_finalize_receipt') is not None or entry['obj_type'] == 'BONE'))
                for entry in _bake_job['objects']):
            try:
                from ...core import full_bake_restore
                if not full_bake_restore.available(context.scene):
                    full_bake_restore.commit(context.scene,recovery,_bake_job['objects'])
            except Exception as recovery_error:
                errors.append('Recovery snapshot failed: '+str(recovery_error))
        preserved = []
        for entry in _bake_job['objects']:
            if entry.get('_finalize_state') in {'committed', 'output_preserved'}:
                preserved.append(entry['obj_name'])
                try:
                    _retain_bake_source_action(entry)
                except Exception as retention_error:
                    errors.append('Source Action retention failed: ' + str(retention_error))
        detail = '; '.join(errors)
        if preserved:
            detail += '. Committed or uncertain output preserved for: ' + ', '.join(preserved)
        else:
            detail += '. No target was committed; original inputs remain for abort/retry'
        _bake_job['finalize_error'] = detail
        raise RuntimeError('Bake finalization failed: ' + detail) from exc


def _rollback_staged_mesh(entry):
    from ...core import full_bake_restore
    if not full_bake_restore.working_mesh_unchanged(entry):
        return
    row = entry["original_mesh_row"]
    obj = row["obj"]
    if entry.get("mesh_created_key_transaction") is not None:
        if not _abort_mesh_entry(obj, entry):
            return
    elif obj.data.shape_keys is not None:
        return
    full_bake_restore.rollback_working_mesh(row)


def _abort_job() -> None:
    """Roll back uncommitted entries; retain terminal or uncertain baked output.

    For uncommitted work, leave PC2 files, MESH_CACHE
    modifiers, and group membership intact so the pre-bake state is
    fully recoverable. For MESH: remove shape keys we added (plus Basis
    if we created the block) and their value fcurves. For CURVE: remove
    inserted per-CV fcurves and restore bezier handle types."""
    from ...core import full_bake_restore
    scene=_bake_job.get('scene')
    if scene is not None and full_bake_restore.available(scene):return
    from ...core import source_materials
    try:
        source_materials.verify(_bake_job)
    except Exception as exc:
        detail = 'Original source holder is invalid; independent failed materials required: ' + str(exc)
        _bake_job['finalize_error'] = str(_bake_job.get('finalize_error') or '') + '\n' + detail
        print(detail)
    from ...core import abort_materials
    abort_materials.retain(_bake_job)
    from ...core.uuid_registry import get_object_by_uuid
    for entry in _bake_job["objects"]:
        if entry.get('_finalize_state') in {'committed', 'output_preserved'}:
            # Its cache may be gone: abort must not delete the only baked output.
            _retain_bake_source_action(entry)
            continue
        if entry["obj_type"] == "BONE":
            from ...core import bone_bake
            bone_bake.rollback(entry)
            continue
        obj = get_object_by_uuid(entry["obj_uuid"])
        if obj is None:
            if (entry.get("mesh_action_transaction") is not None
                    or entry.get("mesh_created_key_transaction") is not None):
                _abort_mesh_entry(None, entry)
            if entry.get("curve_transaction") is not None:
                _abort_curve_entry(None, entry)
            continue
        if entry["obj_type"] == "MESH":
            if entry.get("original_mesh_row") is not None:
                _rollback_staged_mesh(entry)
            else:
                _abort_mesh_entry(obj, entry)
        else:
            _abort_curve_entry(obj, entry)

    recovery=_bake_job.get('recovery')
    if recovery is not None:full_bake_restore.discard_pending(recovery, restore_source_flags=False)


def _id_fcurve_collections(id_block):
    """Editable collections belonging to this ID, never another Action slot."""
    ad = getattr(id_block, "animation_data", None)
    action = getattr(ad, "action", None) if ad is not None else None
    if action is None:
        return
    layers = getattr(action, "layers", ())
    if layers:
        slot = getattr(ad, "action_slot", None)
        if slot is not None:
            for layer in layers:
                for strip in layer.strips:
                    bag = strip.channelbag(slot)
                    if bag is not None:
                        yield bag.fcurves
        return
    curves = getattr(action, "fcurves", None)
    if curves is not None:
        yield curves


def _mesh_key_digest(coordinates):
    import hashlib
    return hashlib.blake2b(numpy.asarray(coordinates, dtype=numpy.float32).tobytes(),
                           digest_size=16).digest()


def _begin_created_mesh_keys(obj, entry):
    from ...core.full_bake_restore import _author_layers
    tx = dict(data=obj.data, key=None, basis=None, action=None, slot=None, keys=[])
    tx["author_layers"] = _author_layers(obj.data)
    entry["mesh_created_key_transaction"] = tx
    basis = obj.shape_key_add(name="Basis", from_mix=False)
    rest = numpy.empty(len(basis.data) * 3, dtype=numpy.float32)
    basis.data.foreach_get("co", rest)
    key = obj.data.shape_keys
    # Blender may otherwise reuse the object's Action when inserting the first
    # Shape Key keyframe, adding a KEY slot to the protected source Action.
    # Give the newly created Key its own Action before any insertion.
    working = bpy.data.actions.new(obj.name + " Bake Shape Keys")
    tx["action"] = working
    _assign_curve_action(key, working, None)
    tx["slot"] = _curve_action_slot(key)
    key.use_relative = False
    tx["encoding"] = "ABSOLUTE_STEP_V1"
    tx["expected_curves"] = {}
    tx["sample_frames"] = []
    tx.update(key=key, basis=basis, basis_name=basis.name,
              basis_digest=_mesh_key_digest(rest), key_settings=(key.use_relative, key.eval_time),
              basis_settings=(basis.value, basis.mute, basis.slider_min, basis.slider_max,
                              basis.vertex_group, basis.interpolation))


def _abort_created_mesh_keys(obj, entry, tx):
    temporary = None
    try:
        if tx["key"] is None or tx["data"].shape_keys != tx["key"]:
            return  # Author cleared/replaced our Key; never revive it.
        if obj is None or obj.data != tx["data"]:
            temporary = bpy.data.objects.new("PPF rollback temporary", tx["data"])
            obj = temporary
        key = tx["key"]
        ad = key.animation_data
        author_action = ad.action if ad else None
        author_slot = _curve_action_slot(key)
        owned_action = tx["action"]
        try:
            if owned_action is not None and owned_action.as_pointer() == 0:
                owned_action = None
        except ReferenceError:
            owned_action = None
        if tx.get("encoding") == "ABSOLUTE_STEP_V1":
            if not _absolute_mesh_unchanged(key, tx):
                warning = "Bake rollback preserved author-edited Shape Keys; review remaining keys"
                print(warning)
                if _bake_modal_owner is not None:
                    _bake_modal_owner.report({"WARNING"}, warning)
                return
            for block in reversed(list(key.key_blocks)):
                obj.shape_key_remove(block)
            return obj.data.shape_keys is None
        # Public KeyBlock RNA can reuse addresses after deletion. If any
        # author-visible state diverged, preserve the whole uncertain Key.
        unchanged = (author_action == owned_action and author_slot == tx["slot"]
                     and len(key.key_blocks) == len(tx["keys"]) + 1
                     and key.reference_key == tx["basis"])
        if unchanged:
            rest = numpy.empty(len(key.reference_key.data) * 3, dtype=numpy.float32)
            key.reference_key.data.foreach_get("co", rest)
            unchanged = (key.reference_key.name == tx["basis_name"]
                         and _mesh_key_digest(rest) == tx["basis_digest"]
                         and (key.use_relative, key.eval_time) == tx["key_settings"]
                         and (key.reference_key.value, key.reference_key.mute,
                              key.reference_key.slider_min, key.reference_key.slider_max,
                              key.reference_key.vertex_group, key.reference_key.interpolation)
                             == tx["basis_settings"])
            paths = [fc.data_path for curves in _id_fcurve_collections(key) for fc in curves]
            expected_paths = [f'key_blocks["{name}"].value' for _block, name, _digest, _meta in tx["keys"]]
            unchanged = (unchanged and sorted(paths) == sorted(expected_paths)
                         and (ad is None or (not ad.drivers and not ad.nla_tracks)))
        if unchanged:
            for index, (block, name, expected, metadata) in enumerate(tx["keys"]):
                try:
                    actual = numpy.empty(len(block.data) * 3, dtype=numpy.float32)
                    block.data.foreach_get("co", actual)
                    unchanged = (block.name == name
                                 and not block.mute and block.relative_key == tx["basis"]
                                 and (block.slider_min, block.slider_max, block.vertex_group,
                                      block.interpolation) == metadata
                                 and _mesh_key_digest(actual) == expected)
                    curve = _sk_value_fcurve(key, name)
                    frame = entry["start_frame"] + index
                    expected_points = [(float(frame - 1), 0.0, "CONSTANT"),
                                       (float(frame), 1.0, "CONSTANT"),
                                       (float(frame + 1), 0.0, "CONSTANT")]
                    evaluated_value = (curve.evaluate(bpy.context.scene.frame_current_final)
                                       if curve is not None else None)
                    unchanged = (unchanged and curve is not None
                                 and not curve.modifiers and not curve.mute and not curve.lock
                                 and curve.extrapolation == "CONSTANT"
                                 and block.value in (0.0, evaluated_value)
                                 and [(float(p.co.x), float(p.co.y), p.interpolation)
                                      for p in curve.keyframe_points] == expected_points)
                except (ReferenceError, UnicodeError):
                    unchanged = False
                if not unchanged:
                    break
        if not unchanged:
            warning = "Bake rollback preserved author-edited Shape Keys; review remaining keys"
            print(warning)
            if _bake_modal_owner is not None:
                _bake_modal_owner.report({"WARNING"}, warning)
            return
        _assign_curve_action(key, owned_action, tx["slot"])
        # Reuse exact added-path removal, never the legacy remove-all branch.
        owned_names = []
        for block, _name, expected, _metadata in tx["keys"]:
            try:
                if any(item == block for item in key.key_blocks):
                    actual = numpy.empty(len(block.data) * 3, dtype=numpy.float32)
                    block.data.foreach_get("co", actual)
                    if _mesh_key_digest(actual) == expected:
                        owned_names.append(block.name)
            except (ReferenceError, UnicodeError):
                pass  # Deleted/reconstructed RNA is not proof of ownership.
        if key.reference_key is not None and key.reference_key.name in owned_names:
            owned_names.remove(key.reference_key.name)
        # Author keys can depend on generated keys through relative_key.
        # Preserve their transitive support, rather than silently changing shape.
        changed = True
        while changed:
            changed = False
            for block in key.key_blocks:
                if block.name not in owned_names and block.relative_key is not None:
                    support = block.relative_key.name
                    if support in owned_names:
                        owned_names.remove(support)
                        changed = True
        protected = dict(entry, pre_sk_block_existed=True, added_sk_names=owned_names,
                         preserve_shared_action=owned_action is not None and owned_action.users > 1)
        try:
            _remove_baked_mesh_keys(obj, protected)
        finally:
            if obj.data.shape_keys is key:
                _assign_curve_action(key, author_action, author_slot)
        remaining = list(key.key_blocks)
        if len(remaining) == 1 and remaining[0] == tx["basis"]:
            obj.shape_key_remove(remaining[0])
        elif remaining:
            warning = "Bake rollback preserved author-edited Shape Keys; review remaining keys"
            print(warning)
            owner = _bake_modal_owner
            if owner is not None:
                owner.report({"WARNING"}, warning)
    except ReferenceError:
        pass  # Original data or original Key was explicitly removed.
    finally:
        if temporary is not None:
            bpy.data.objects.remove(temporary)
        try:
            action = tx["action"]
            if action is not None and action.users == 0:
                bpy.data.actions.remove(action)
        except ReferenceError:
            pass


def _mesh_fcurve_snapshot(curve):
    """Value snapshot, never compare against the mutable shared source Action."""
    modifiers = []
    for modifier in curve.modifiers:
        values = []
        for prop in modifier.bl_rna.properties:
            name = prop.identifier
            if name == "rna_type":
                continue
            value = getattr(modifier, name, None)
            if isinstance(value, (str, int, float, bool)):
                values.append((name, value))
            elif getattr(prop, "is_array", False):
                values.append((name, tuple(value)))
        if hasattr(modifier, "control_points"):
            values.append(("control_points", tuple((point.frame, point.min, point.max)
                                                   for point in modifier.control_points)))
        modifiers.append((modifier.type, tuple(values)))
    points = tuple((tuple(p.co), tuple(p.handle_left), tuple(p.handle_right),
                    p.handle_left_type, p.handle_right_type, p.interpolation,
                    p.easing, p.type, p.amplitude, p.back, p.period)
                   for p in curve.keyframe_points)
    return (curve.extrapolation, curve.mute, curve.lock, curve.color_mode,
            tuple(curve.color), getattr(curve, "auto_smoothing", None),
            (curve.group.name, curve.group.mute, curve.group.lock) if curve.group else None, points,
            tuple(tuple(p.co) for p in curve.sampled_points), tuple(modifiers))


def _mesh_action_snapshot(key):
    return {(fc.data_path, fc.array_index): _mesh_fcurve_snapshot(fc)
            for curves in _id_fcurve_collections(key) for fc in curves}


def _begin_mesh_action(obj, entry):
    from ...core.full_bake_restore import _author_layers
    key = obj.data.shape_keys
    ad = key.animation_data
    old = ad.action if ad else None
    slot = _curve_action_slot(key)
    tx = dict(data=obj.data, key=key, action=old, slot=slot,
              author_layers=_author_layers(obj.data),
              had_animation=ad is not None, working=None,
              expected_curves=_mesh_action_snapshot(key), keys=[],
              initial_keys=tuple(block.name for block in key.key_blocks),
              initial_paths=tuple(fc.data_path for curves in _id_fcurve_collections(key) for fc in curves))
    entry["mesh_action_transaction"] = tx
    working = old.copy() if old else bpy.data.actions.new(obj.name + " Key Bake Working")
    tx["working"] = working
    working.use_fake_user = False
    _assign_curve_action(key, working, slot)
    tx["working_slot"] = _curve_action_slot(key)


def _absolute_mesh_unchanged(key, tx):
    """Prove ownership before overwriting or removing a generated Key."""
    try:
        ad = key.animation_data
        if (key.use_relative or key.reference_key != tx["basis"]
                or len(key.key_blocks) != len(tx["keys"]) + 1
                or (ad.action if ad else None) != tx["action"]
                or _curve_action_slot(key) != tx["slot"]
                or (ad and (ad.drivers or ad.nla_tracks))
                or _mesh_action_snapshot(key) != tx["expected_curves"]):
            return False
        basis = key.reference_key
        coords = numpy.empty(len(basis.data) * 3, dtype=numpy.float32)
        basis.data.foreach_get("co", coords)
        if (basis.name != tx["basis_name"] or _mesh_key_digest(coords) != tx["basis_digest"]
                or (basis.value, basis.mute, basis.slider_min, basis.slider_max,
                    basis.vertex_group, basis.interpolation) != tx["basis_settings"]):
            return False
        for index, (block, name, digest, metadata) in enumerate(tx["keys"]):
            coords = numpy.empty(len(block.data) * 3, dtype=numpy.float32)
            block.data.foreach_get("co", coords)
            if (key.key_blocks[index + 1] != block or block.name != name or block.mute
                    or float(block.frame) != tx["sample_frames"][index]
                    or (block.slider_min, block.slider_max, block.vertex_group,
                        block.interpolation) != metadata or _mesh_key_digest(coords) != digest):
                return False
        curves = list(get_id_fcurves(key)) if ad and ad.action else []
        if not curves and abs(key.eval_time - tx["key_settings"][1]) > 1e-4:
            return False
        if curves:
            if len(curves) != 1 or curves[0].data_path != "eval_time":
                return False
            expected = curves[0].evaluate(bpy.context.scene.frame_current_final)
            if abs(key.eval_time - expected) > 1e-4:
                return False
        return True
    except (ReferenceError, RuntimeError, IndexError):
        return False


def _validate_mesh_action(obj, entry):
    created = entry.get("mesh_created_key_transaction")
    if created is not None:
        if obj is None or obj.data != created["data"] or obj.data.shape_keys != created["key"]:
            raise ValueError("Mesh Shape Key data changed during Bake")
        from ...core.full_bake_restore import _author_layers
        if _author_layers(obj.data) != created["author_layers"]:
            raise ValueError("Mesh author data changed during Bake")
        if len(created["key"].key_blocks) != len(created["keys"]) + 1:
            raise ValueError("Mesh Shape Key set changed during Bake")
        if created.get("encoding") == "ABSOLUTE_STEP_V1" and not _absolute_mesh_unchanged(created["key"], created):
            raise ValueError("Mesh Shape Keys or animation changed during Bake")
        ad = created["key"].animation_data
        if ((ad.action if ad else None) != created["action"]
                or _curve_action_slot(created["key"]) != created["slot"]):
            raise ValueError("Mesh Shape Key Action changed during Bake")
        return
    tx = entry.get("mesh_action_transaction")
    if tx is None:
        return
    if obj is None or obj.data != tx["data"] or obj.data.shape_keys != tx["key"]:
        raise ValueError("Mesh Shape Key data changed during Bake")
    from ...core.full_bake_restore import _author_layers
    if _author_layers(obj.data) != tx["author_layers"]:
        raise ValueError("Mesh author data changed during Bake")
    ad = tx["key"].animation_data
    if ad is None or ad.action != tx["working"] or _curve_action_slot(tx["key"]) != tx["working_slot"]:
        raise ValueError("Mesh Shape Key Action changed during Bake")


def _abort_mesh_entry(obj, entry):
    created = entry.pop("mesh_created_key_transaction", None)
    if created is not None:
        return _abort_created_mesh_keys(obj, entry, created)
    tx = entry.pop("mesh_action_transaction", None)
    if tx is None:
        if obj is not None:
            _remove_baked_mesh_keys(obj, entry)
        return
    temporary = None
    try:
        try:
            key_survives = tx["data"].shape_keys == tx["key"] and tx["key"].as_pointer() != 0
        except ReferenceError:
            key_survives = False
        if not key_survives:
            return  # Respect an author deleting the original Key; do not revive it.
        # A new data binding must never receive rollback meant for the old Mesh.
        if obj is None or obj.data != tx["data"]:
            temporary = bpy.data.objects.new("PPF rollback temporary", tx["data"])
            obj = temporary
        try:
            working_survives = tx["working"] is not None and tx["working"].as_pointer() != 0
        except ReferenceError:
            working_survives = False
        key = tx["key"]
        ad = key.animation_data
        current_action = ad.action if ad else None
        expected_keys = set(tx["initial_keys"]) | set(entry["added_sk_names"])
        names_unchanged = (len(key.key_blocks) == len(expected_keys)
                           and {block.name for block in key.key_blocks} == expected_keys)
        expected_paths = set(tx["initial_paths"]) | {
            f'key_blocks["{name}"].value' for name in entry["added_sk_names"]}
        paths = {fc.data_path for curves in _id_fcurve_collections(key) for fc in curves}
        author_action_changed = ((working_survives and
                                  (current_action != tx["working"]
                                   or _curve_action_slot(key) != tx["working_slot"]))
                                 or (not working_survives and current_action is not None))
        generated_unchanged = True
        for _reference, name, digest, metadata in tx["keys"]:
            block = key.key_blocks.get(name)
            if block is None:
                generated_unchanged = False
                break
            coordinates = numpy.empty(len(block.data) * 3, dtype=numpy.float32)
            block.data.foreach_get("co", coordinates)
            curve = _sk_value_fcurve(key, name)
            evaluated = curve.evaluate(bpy.context.scene.frame_current_final) if curve else None
            if (_mesh_key_digest(coordinates) != digest or block.mute
                    or block.relative_key != key.reference_key
                    or (working_survives and block.value not in (0.0, evaluated))
                    or (block.slider_min, block.slider_max, block.vertex_group,
                        block.interpolation) != metadata):
                generated_unchanged = False
                break
        curves_unchanged = (not working_survives or author_action_changed
                            or _mesh_action_snapshot(key) == tx["expected_curves"])
        if (not names_unchanged or author_action_changed or not paths.issubset(expected_paths)
                or not curves_unchanged or not generated_unchanged
                or (ad is not None and (ad.drivers or ad.nla_tracks))):
            tx["preserve_working"] = True
            # Respect the new author state without orphaning the displaced source.
            original = tx["action"]
            try:
                if original is not None and original.users == 0 and not original.use_fake_user:
                    original.use_fake_user = True
            except ReferenceError:
                pass  # An author-deleted source must stay deleted.
            warning = "Bake rollback preserved author-edited Shape Keys or Action; review remaining keys"
            print(warning)
            if _bake_modal_owner is not None:
                _bake_modal_owner.report({"WARNING"}, warning)
            return
        _assign_curve_action(key, tx["working"] if working_survives else None,
                             tx.get("working_slot") if working_survives else None)
        protected = dict(entry, preserve_shared_action=working_survives and tx["working"].users > 1)
        _remove_baked_mesh_keys(obj, protected)
        original = tx["action"]
        try:
            if original is not None and original.as_pointer() == 0:
                original = None
        except ReferenceError:
            original = None  # Respect author deletion of the original Action.
        _assign_curve_action(tx["key"], original, tx["slot"] if original is not None else None)
        if tx.get("retained_source_action") and original is not None:
            original.use_fake_user = False
            tx["retained_source_action"] = False
        if not tx["had_animation"]:
            tx["key"].animation_data_clear()
    finally:
        if temporary is not None:
            bpy.data.objects.remove(temporary)
        working = tx["working"]
        try:
            if working is not None and working.users == 0:
                if tx.get("preserve_working", False):
                    working.use_fake_user = True
                else:
                    bpy.data.actions.remove(working)
        except ReferenceError:
            pass  # The author may have removed the working Action too.


def _remove_baked_mesh_keys(obj, entry) -> None:
    sk_block = obj.data.shape_keys
    if sk_block is None:
        return
    # Remove the value-channel fcurves we added, matching by data_path
    # (per-shape-key, so we know exactly which to strip).
    added_paths = {
        f'key_blocks["{name}"].value' for name in entry.get("added_sk_names", [])
    }
    if added_paths and not entry.get("preserve_shared_action", False):
        for curves in _id_fcurve_collections(sk_block):
            for fc in [fc for fc in curves if fc.data_path in added_paths]:
                curves.remove(fc)
    # Remove the shape keys themselves. Non-Basis first so Basis can be
    # removed cleanly afterwards if we were the ones who created it.
    for name in list(entry.get("added_sk_names", [])):
        kb = sk_block.key_blocks.get(name)
        if kb is not None:
            obj.shape_key_remove(kb)
    if not entry.get("pre_sk_block_existed", False) and obj.data.shape_keys is not None:
        remaining = [kb for kb in obj.data.shape_keys.key_blocks]
        for kb in remaining:
            obj.shape_key_remove(kb)


def _curve_action_slot(data):
    slot = getattr(data.animation_data, "action_slot", None)
    return (slot.handle, slot.identifier) if slot else None


def _validate_curve_transaction(obj, entry):
    tx = entry.get("curve_transaction")
    if tx is None:
        return
    if obj is None or obj.data != tx["data"]:
        raise ValueError("Curve data changed during Bake")
    ad = obj.data.animation_data
    if ad is None or ad.action != tx["working"] or _curve_action_slot(obj.data) != tx["working_slot"]:
        raise ValueError("Curve Action changed during Bake")
    if _mesh_action_snapshot(obj.data) != tx["expected_curves"]:
        raise ValueError("Curve keyframes changed during Bake")


def _assign_curve_action(data, action, slot):
    ad = data.animation_data_create()
    ad.action = action
    if action is not None and slot is not None and hasattr(ad, "action_slot"):
        matches = [s for s in getattr(action, "slots", ()) if s.handle == slot[0]]
        if not matches:
            matches = [s for s in getattr(action, "slots", ()) if s.identifier == slot[1]]
        if len(matches) != 1:
            raise ValueError("Curve Action slot cannot be preserved")
        ad.action_slot = matches[0]


def _begin_curve_bake(obj, entry):
    """Copy-on-write keeps shared author Actions pristine, including handles."""
    data = obj.data
    ad = data.animation_data
    slot = getattr(ad, "action_slot", None) if ad else None
    old = ad.action if ad else None
    entry["curve_transaction"] = dict(
        data=data, action=old, slot=(slot.handle, slot.identifier) if slot else None,
        had_animation=ad is not None, pose=_capture_curve_pose(data),
        handles=_snapshot_bezier_handles(obj), working=None)
    tx = entry["curve_transaction"]
    working = old.copy() if old else bpy.data.actions.new(obj.name + " Bake Working")
    tx["working"] = working
    working.use_fake_user = False
    _assign_curve_action(data, working, tx["slot"])
    tx["working_slot"] = _curve_action_slot(data)
    _force_bezier_free(obj)
    tx["expected_curves"] = _mesh_action_snapshot(data)
    tx["expected_pose"] = _capture_curve_pose(data)
    tx["expected_handles"] = _snapshot_bezier_handles(obj)


def _abort_curve_entry(obj, entry) -> None:
    tx = entry.pop("curve_transaction", None)
    if tx is not None:
        data = tx["data"]
        try:
            ad = data.animation_data
            current = ad.action if ad else None
            unchanged = (current == tx["working"]
                         and _curve_action_slot(data) == tx.get("working_slot")
                         and _mesh_action_snapshot(data) == tx.get("expected_curves")
                         and _capture_curve_pose(data) == tx.get("expected_pose")
                         and (ad is None or (not ad.drivers and not ad.nla_tracks)))
            # Handle edits also belong to the author, even without keyframes.
            if obj is not None and obj.data == data:
                unchanged = unchanged and _snapshot_bezier_handles(obj) == tx.get("expected_handles")
        except ReferenceError:
            return  # The author removed the original Curve or its Action.
        if not unchanged:
            # Preserve the author's current rebind/clear and its pose. Also keep
            # the displaced source; retaining the working Action conservatively
            # preserves this branch's stronger edited-Action rollback protection.
            source = tx["action"]
            try:
                if source is not None and source.users == 0 and not source.use_fake_user:
                    source.use_fake_user = True
            except ReferenceError:
                pass  # Explicitly deleted Actions must not be recreated.
            working = tx["working"]
            try:
                if working is not None and working.users == 0:
                    working.use_fake_user = True
            except ReferenceError:
                pass
            warning = "Bake rollback preserved author-edited Curve or Action; review remaining keys"
            print(warning)
            if _bake_modal_owner is not None:
                _bake_modal_owner.report({"WARNING"}, warning)
            return
        original = tx["action"]
        try:
            if original is not None and original.as_pointer() == 0:
                original = None
        except ReferenceError:
            original = None
        _assign_curve_action(data, original, tx["slot"] if original is not None else None)
        if tx.get("retained_source_action") and original is not None:
            original.use_fake_user = False
            tx["retained_source_action"] = False
        _apply_curve_pose(data, tx["pose"])
        _restore_bezier_handles(obj, tx["handles"], data=data)
        if not tx["had_animation"]:
            data.animation_data_clear()
        working = tx["working"]
        if working is not None and working.users == 0:
            bpy.data.actions.remove(working)
        return
    _restore_bezier_handles(obj, entry.get("handle_snapshot"))
    pre = entry.get("pre_fcurves", set())
    for curves in _id_fcurve_collections(obj.data):
        to_remove = [fc for fc in curves
                     if (fc.data_path, fc.array_index) not in pre]
        for fc in to_remove:
            curves.remove(fc)


# ---------------------------------------------------------------------------
# Bake-confirm dialog: list the deforming modifiers bake will strip
# ---------------------------------------------------------------------------


def _removable_modifiers_for_objs(objs) -> list:
    """For each object that has at least one modifier above
    ContactSolverCache in its stack, return ``(obj_name, [mod_names])``
    in stack order. Objects with no upstream modifiers (or no
    ContactSolverCache, or non-MESH) are skipped. The bake-confirm
    dialog renders this list so the user sees which modifiers the
    finalize step will delete (so PC2 isn't re-deformed at playback)
    before clicking OK."""
    result = []
    for obj in objs:
        if obj is None:
            continue
        mods = modifiers_above_cache(obj)
        if mods:
            result.append((obj.name, mods))
    return result


def _draw_bake_dialog(layout, prompt: str, removable: list) -> None:
    """Render the shared bake-confirm dialog body."""
    layout.label(text=prompt, icon="QUESTION")
    layout.label(text="Cache modifiers on baked objects will be deactivated.")
    if removable:
        layout.separator()
        box = layout.box()
        box.label(
            text="These modifiers will be deactivated for baked objects:",
            icon="ERROR",
        )
        for obj_name, mod_names in removable:
            box.label(text=obj_name, icon="OBJECT_DATA")
            for mod_name in mod_names:
                box.label(text="     " + mod_name, icon="MODIFIER")
        box.separator()
        box.label(
            text="They fed the captured-deformation cache; leaving them in",
        )
        box.label(
            text="would re-deform the baked shape keys at playback.",
        )


def _draw_soft_static_options(layout, operator):
    excluded = getattr(operator, "_soft_excluded", {})
    candidates = getattr(operator, "_soft_candidates", {})
    if not excluded:
        return
    box = layout.box()
    if candidates:
        box.prop(operator, "bake_soft_static")
        box.label(text="Baking uses a result mesh; original shape-key animation is inactive.", icon="INFO")
        box.label(text="Original shape keys, keyframes and drivers are preserved for restore.")
    for uid, name in excluded.items():
        box.label(text=name if uid in candidates else iface_("No readable animation cache: {name}").format(name=name))


# ---------------------------------------------------------------------------
# Modal operator mixin (shared between per-object and all-object bakes)
# ---------------------------------------------------------------------------


class _ModalBakeBase:
    """Shared modal loop. Subclasses provide ``_kind`` and
    ``_build_queue_for``; everything else here is reused."""

    _kind = ""

    def _build_queue_for(self, context) -> list:
        raise NotImplementedError

    def execute(self, context):
        global _bake_modal_owner
        from ...core.single_bake_restore import available
        if available(context.scene):
            self.report({"ERROR"}, "请先点击恢复单帧烘焙前的模拟，再烘焙完整动画。")
            return {"CANCELLED"}
        try:
            if self._kind == "all":
                skipped, eligible, excluded = _preview_all(context)
                confirmed = getattr(self, "_confirmed_skipped", None)
                candidates = _soft_static_candidates(excluded)
                signature = (_preview_signature(skipped, eligible, excluded), tuple(sorted(candidates.items())), _bake_group_signature(context))
                if (skipped and confirmed is None) or (confirmed is not None and getattr(self, "_confirmed_signature", None) != signature):
                    self.report({"ERROR"}, iface_("Bake targets or conflicts changed; reopen Bake All to review."))
                    return {"CANCELLED"}
                include_soft = bool(confirmed is not None and self.bake_soft_static)
                queue = _build_queue(context, skipped=skipped, include_soft=include_soft)
                if (skipped or excluded) and not any(e.get("n_frames", 0) > 0 for e in queue):
                    self.report({"INFO"}, iface_("No eligible animation remains to bake."))
                    return {"CANCELLED"}
            else:
                excluded_single, candidates_single = _single_soft_preview(context, self.group_index)
                confirmed_single = getattr(self, "_confirmed_soft_signature", None)
                signature_single = (tuple(sorted(excluded_single.items())), tuple(sorted(candidates_single.items())), _bake_group_signature(context, self.group_index))
                if confirmed_single is not None and confirmed_single != signature_single:
                    self.report({"ERROR"}, iface_("Bake targets or conflicts changed; reopen Bake All to review."))
                    return {"CANCELLED"}
                include_soft = bool(confirmed_single is not None and self.bake_soft_static)
                queue = _build_queue(context, group_index=self.group_index, include_soft=include_soft)
                if not queue:
                    self.report({"INFO"}, iface_("No eligible animation to bake; soft STATIC meshes are skipped."))
                    return {"CANCELLED"}
        except (OSError, ValueError, struct.error) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        blockers = _check_mesh_shape_keys(queue)
        if blockers:
            joined = "; ".join(blockers)
            self.report(
                {"ERROR"},
                iface_(
                    "Existing shape keys conflict with animation bake on: {conflicts}"
                ).format(conflicts=joined),
            )
            return {"CANCELLED"}
        ok, err = _start_job(queue, kind=self._kind)
        if not ok:
            self.report({"WARNING"}, err)
            return {"CANCELLED"}
        if self._kind == "all" and skipped:
            self.report({"WARNING"}, iface_("Skipped shape-key conflicts: {conflicts}").format(conflicts="; ".join(skipped.values())))
        if self._kind == "all" and excluded:
            remaining = {uid: name for uid, name in excluded.items() if not include_soft or uid not in candidates}
            if remaining:
                self.report({"INFO"}, iface_("Skipped soft STATIC meshes: {objects}").format(objects="; ".join(remaining.values())))
        wm = context.window_manager
        self._wm = wm
        self._externally_cancelled = False
        _bake_modal_owner = self
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        redraw_all_areas(context)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if getattr(self, "_externally_cancelled", False) or _bake_modal_owner is not self:
            self._detach_timer()
            return {"CANCELLED"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if _bake_job["aborted"]:
            _abort_job()
            self._teardown(context)
            redraw_all_areas(context)
            self.report({"INFO"}, iface_("Bake aborted"))
            return {"CANCELLED"}
        try:
            more = _tick_job()
        except Exception as exc:
            _abort_job();self._teardown(context);self.report({"ERROR"}, str(exc));return {"CANCELLED"}
        redraw_all_areas(context)
        if not more and not _bake_job["aborted"]:
            try:
                n_objs, n_frames = _finalize_job(context)
            except Exception as exc:
                _abort_job();self._teardown(context);self.report({"ERROR"}, str(exc));return {"CANCELLED"}
            self._teardown(context)
            from .overlay import apply_object_overlays
            apply_object_overlays()
            redraw_all_areas(context)
            self.report(
                {"INFO"},
                iface_("Baked {frame_count} frame(s) for {object_count} object(s)").format(
                    frame_count=n_frames, object_count=n_objs
                ),
            )
            return {"FINISHED"}
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        if _bake_modal_owner is self and _bake_job["active"]:
            _abort_job()
        self._teardown(context)
        redraw_all_areas(context)

    def _detach_timer(self):
        timer = getattr(self, "_timer", None)
        self._timer = None
        if timer is not None:
            try:
                self._wm.event_timer_remove(timer)
            except (ReferenceError, RuntimeError):
                # Blender may already dispose WM timers during file replacement.
                pass

    def _teardown(self, context):
        global _bake_modal_owner
        self._detach_timer()
        if _bake_modal_owner is self:
            _bake_modal_owner = None
            _reset_job()


# ---------------------------------------------------------------------------
# Per-object Bake Animation (modal with progress)
# ---------------------------------------------------------------------------


class OBJECT_OT_BakeAnimation(_ModalBakeBase, Operator):
    """Bake animation for the selected object and remove it from the group"""

    bl_idname = "object.bake_animation"
    bl_label = "Bake Animation"
    bl_options = {"UNDO"}

    group_index: _ppf_props.IntProperty(options={'HIDDEN'})  # pyright: ignore
    bake_soft_static: _ppf_props.BoolProperty(name="Bake soft STATIC meshes", default=False, options={'SKIP_SAVE'})  # pyright: ignore
    _kind = "single"

    @classmethod
    def poll(cls, context):
        return not _bake_job["active"]

    def invoke(self, context, event):
        if _has_unfetched_frames(context.scene):
            self.report(
                {"ERROR"},
                iface_(
                    "Unfetched animation frames exist. "
                    "Fetch all animation frames first."
                ),
            )
            return {"CANCELLED"}
        from ...core.uuid_registry import get_object_by_uuid
        try:
            excluded, candidates = _single_soft_preview(context, self.group_index)
            self._confirmed_soft_signature = (tuple(sorted(excluded.items())), tuple(sorted(candidates.items())), _bake_group_signature(context, self.group_index))
            self._soft_candidates = candidates
            self._soft_excluded = excluded
            self.bake_soft_static = False
            if excluded:
                if not candidates:
                    self.report({"INFO"}, iface_("No eligible animation to bake; soft STATIC meshes are skipped."))
                    return {"CANCELLED"}
                self._removable = _removable_modifiers_for_objs([get_object_by_uuid(uid) for uid in candidates])
                return context.window_manager.invoke_props_dialog(self, width=520, title="Bake Animation", confirm_text="Bake")
            queue = self._build_queue_for(context)
        except (OSError, ValueError, struct.error) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        if not queue:
            self.report({"INFO"}, iface_("No eligible animation to bake; soft STATIC meshes are skipped."))
            return {"CANCELLED"}
        objs = [
            get_object_by_uuid(e["obj_uuid"])
            for e in queue if e["obj_type"] == "MESH"
        ]
        self._removable = _removable_modifiers_for_objs(objs)
        return context.window_manager.invoke_props_dialog(
            self, width=420, title="Bake Animation", confirm_text="Bake",
        )

    def draw(self, context):
        _draw_bake_dialog(
            self.layout,
            "Bake animation for the selected object?",
            getattr(self, "_removable", []) if not getattr(self, "_soft_candidates", {}) or self.bake_soft_static else [],
        )
        _draw_soft_static_options(self.layout, self)

    def _build_queue_for(self, context) -> list:
        return _build_queue(context, group_index=self.group_index)


# ---------------------------------------------------------------------------
# Per-object Bake Single Frame (stays synchronous — cheap, no hang risk)
# ---------------------------------------------------------------------------


class OBJECT_OT_BakeSingleFrame(Operator):
    """Bake current frame as Frame 1 and remove object from the group"""

    bl_idname = "object.bake_single_frame"
    bl_label = "Bake Single Frame"
    bl_options = {"UNDO"}

    group_index: _ppf_props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    @classmethod
    def poll(cls, context):
        return not _bake_job["active"]

    def invoke(self, context, event):
        obj = self._resolve_target_obj(context)
        self._has_curve = bool(obj and obj.type == "CURVE")
        self._removable = _removable_modifiers_for_objs([obj] if obj else [])
        return context.window_manager.invoke_props_dialog(
            self, width=420, title="Bake Single Frame", confirm_text="Bake",
        )

    def draw(self, context):
        _draw_bake_dialog(
            self.layout,
            ("曲线使用原有单帧流程，不能用恢复模拟按钮撤销。" if getattr(self, "_has_curve", False) else "Bake the current frame as Frame 1 for the selected object?"),
            getattr(self, "_removable", []),
        )

    def _resolve_target_obj(self, context):
        from ...core.uuid_registry import resolve_assigned
        group = get_group_from_index(context.scene, self.group_index)
        if group is None:
            return None
        idx = group.assigned_objects_index
        if not (0 <= idx < len(group.assigned_objects)):
            return None
        return resolve_assigned(group.assigned_objects[idx])

    def execute(self, context):
        from .overlay import apply_object_overlays

        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            self.report({"ERROR"}, iface_("Group not found"))
            return {"CANCELLED"}

        index = group.assigned_objects_index
        if not (0 <= index < len(group.assigned_objects)):
            self.report({"ERROR"}, iface_("No object selected"))
            return {"CANCELLED"}

        assigned = group.assigned_objects[index]
        from ...core.uuid_registry import resolve_assigned
        obj = resolve_assigned(assigned)
        obj_name = assigned.name
        if not obj or obj.type not in ("MESH", "CURVE"):
            self.report({"ERROR"}, iface_("Object '{name}' not found").format(name=obj_name))
            return {"CANCELLED"}

        if not _is_bakeable(obj):
            self.report({"WARNING"}, iface_("No animation data on '{name}'").format(name=obj_name))
            return {"CANCELLED"}

        try:
            _kind, captured_pose = _capture_single_bake_pose(
                obj, context.evaluated_depsgraph_get())
        except (ValueError, RuntimeError, AttributeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        from ...core import single_bake_restore
        if _kind == "CURVE":
            if single_bake_restore.available(scene):
                self.report({"ERROR"}, "请先恢复或结束已有单帧恢复记录。")
                return {"CANCELLED"}
            try:
                scene["ppf_single_bake_session_stale"] = True
                cleanup_mesh_cache(obj, keep_baked_pose=True)
                scene.frame_set(_resolve_bake_start_frame())
                _apply_curve_pose(obj.data, captured_pose)
                reset_object_display(obj)
                uid=assigned.uuid;group.assigned_objects.remove(index)
                cleanup_group_references_for_object(group, uid)
                group.assigned_objects_index=safe_update_index(index,len(group.assigned_objects))
            except Exception as exc:
                self.report({"ERROR"},str(exc));return {"CANCELLED"}
            apply_object_overlays()
            self.report({"WARNING"}, "曲线单帧使用原有流程，不能用恢复模拟按钮撤销。")
            return {"FINISHED"}
        try:
            tx = single_bake_restore.begin(scene, {assigned.uuid: (_kind, captured_pose)}, object_only=True)
            staged = _stage_single_meshes(scene, {assigned.uuid: (_kind, captured_pose)})
            try:
                single_bake_restore.commit(scene, tx, None)
            except BaseException:
                _restore_single_meshes(staged)
                raise
        except (ValueError, RuntimeError, AttributeError) as exc:
            if 'tx' in locals(): single_bake_restore.discard_uncommitted(tx)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        scene["ppf_single_bake_session_stale"] = True

        apply_object_overlays()
        self.report({"INFO"}, iface_("Baked frame {frame} as Frame 1 for '{name}'").format(frame=scene.frame_current, name=obj_name))
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# All-object Bake Animation (modal with progress)
# ---------------------------------------------------------------------------


class SOLVER_OT_BakeAllAnimation(_ModalBakeBase, Operator):
    """Bake animation for all objects and remove them from their groups"""

    bl_idname = "solver.bake_all_animation"
    bl_label = "Bake Animation"
    bl_options = {"UNDO"}
    bake_soft_static: _ppf_props.BoolProperty(name="Bake soft STATIC meshes", default=False, options={'SKIP_SAVE'})  # pyright: ignore
    _kind = "all"

    @classmethod
    def poll(cls, context):
        if _bake_job["active"]:
            return False
        return _has_animated_objects(context)

    def invoke(self, context, event):
        if _has_unfetched_frames(context.scene):
            self.report(
                {"ERROR"},
                iface_(
                    "Unfetched animation frames exist. "
                    "Fetch all animation frames first."
                ),
            )
            return {"CANCELLED"}
        try:
            skipped, objs, excluded = _preview_all(context)
            candidates = _soft_static_candidates(excluded)
        except (OSError, ValueError, struct.error) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self._confirmed_skipped = skipped
        self._confirmed_excluded = excluded
        self._confirmed_signature = (_preview_signature(skipped, objs, excluded), tuple(sorted(candidates.items())), _bake_group_signature(context))
        self._soft_candidates = candidates
        self._soft_excluded = excluded
        self.bake_soft_static = False
        self._removable = _removable_modifiers_for_objs(objs)
        from ...core.uuid_registry import get_object_by_uuid
        self._soft_removable = _removable_modifiers_for_objs([get_object_by_uuid(uid) for uid in candidates])
        return context.window_manager.invoke_props_dialog(
            self, width=520, title="Bake Animation",
            confirm_text="Bake" if candidates else ("Bake Remaining" if skipped or excluded else "Bake All"),
        )

    def draw(self, context):
        _draw_bake_dialog(
            self.layout,
            "Bake animation for all objects in active groups?",
            getattr(self, "_removable", []) + (getattr(self, "_soft_removable", []) if self.bake_soft_static else []),
        )
        skipped = getattr(self, "_confirmed_skipped", {})
        if skipped:
            box = self.layout.box()
            box.label(text="These objects have shape-key conflicts and will be skipped:", icon="ERROR")
            for line in skipped.values():
                box.label(text=line)
        _draw_soft_static_options(self.layout, self)

    def _build_queue_for(self, context) -> list:
        return _build_queue(context, group_index=None)


# ---------------------------------------------------------------------------
# All-object Bake Single Frame (synchronous — cheap)
# ---------------------------------------------------------------------------


def _single_all_with_curves_legacy(op, context, captured):
    """Preserve the original Curve path; this path has no restore snapshot."""
    from ...core import bone_bake
    from ...core.uuid_registry import resolve_assigned
    from .overlay import apply_object_overlays
    scene=context.scene
    try:
        staged=_stage_single_meshes(scene,captured)
        try: bone_entry=bone_bake.single(scene)
        except BaseException:
            _restore_single_meshes(staged)
            raise
    except (ValueError,RuntimeError,AttributeError) as exc:
        op.report({"ERROR"},str(exc));return {"CANCELLED"}
    if bone_entry is not None or captured:scene['ppf_single_bake_session_stale']=True
    try:
        scene.frame_set(_resolve_bake_start_frame());count=0
        for group in iterate_active_object_groups(scene):
            for i in range(len(group.assigned_objects)-1,-1,-1):
                item=group.assigned_objects[i];obj=resolve_assigned(item);uid=item.uuid
                if not obj or uid not in captured:continue
                kind,pose=captured[uid]
                if kind=='MESH':
                    strip_modifiers_above_cache(obj);cleanup_mesh_cache(obj)
                else:
                    cleanup_mesh_cache(obj,keep_baked_pose=True);_apply_curve_pose(obj.data,pose)
                reset_object_display(obj);group.assigned_objects.remove(i)
                cleanup_group_references_for_object(group,uid);count+=1
            group.assigned_objects_index=safe_update_index(group.assigned_objects_index,len(group.assigned_objects))
    except Exception as exc:
        op.report({"ERROR"},iface_("Single Frame Bake stopped during cache cleanup; some changes may have been retained and were not rolled back: {error}").format(error=exc))
        return {"CANCELLED"}
    if bone_entry is not None:
        scene.frame_set(bone_entry['entry_time'][0],subframe=bone_entry['entry_time'][1]);context.view_layer.update()
    apply_object_overlays()
    op.report({"WARNING"},"曲线单帧使用原有流程，不能用恢复模拟按钮撤销。")
    return {"FINISHED"}


class SOLVER_OT_BakeAllSingleFrame(Operator):
    """Bake meshes to first frame; bone poses key on the current frame, preserving parent animation"""

    bl_idname = "solver.bake_all_single_frame"
    bl_label = "Bake Single Frame"
    bl_options = {"UNDO"}

    @classmethod
    def poll(cls, context):
        if _bake_job["active"]:
            return False
        return _has_animated_objects(context)

    def invoke(self, context, event):
        from ...core.uuid_registry import resolve_assigned
        objs = []
        for group in iterate_active_object_groups(context.scene):
            for assigned in group.assigned_objects:
                obj = resolve_assigned(assigned)
                if obj is not None and _is_bakeable(obj):
                    objs.append(obj)
        self._has_curve = any(obj.type == "CURVE" for obj in objs)
        self._removable = _removable_modifiers_for_objs(objs)
        return context.window_manager.invoke_props_dialog(
            self, width=420, title="Bake Single Frame",
            confirm_text="Bake All",
        )

    def draw(self, context):
        _draw_bake_dialog(
            self.layout,
            ("包含曲线：将使用原有单帧流程，不能用恢复模拟按钮撤销。" if getattr(self, "_has_curve", False) else "网格烘焙到首帧；骨骼姿态写入当前帧，保留父骨动画。"),
            getattr(self, "_removable", []),
        )

    def execute(self, context):
        from .overlay import apply_object_overlays

        scene = context.scene
        # Resolve the whole batch before bone baking or irreversible PC2 teardown.
        from ...core.uuid_registry import resolve_assigned
        missing = []
        conflicts = []
        captured = {}
        depsgraph = context.evaluated_depsgraph_get()
        for group in iterate_active_object_groups(scene):
            for assigned in group.assigned_objects:
                obj = resolve_assigned(assigned) if assigned.uuid else None
                if not assigned.uuid or obj is None:
                    missing.append(assigned.name or assigned.uuid or "<unknown>")
                elif _is_bakeable(obj) and assigned.uuid not in captured:
                    try:
                        captured[assigned.uuid] = _capture_single_bake_pose(obj, depsgraph)
                    except (ValueError, RuntimeError, AttributeError) as exc:
                        conflicts.append(str(exc))
        if missing:
            self.report({"ERROR"}, iface_("UUID resolve failed for: {names}").format(names=', '.join(missing)))
            return {"CANCELLED"}
        if conflicts:
            self.report({"ERROR"}, "; ".join(conflicts))
            return {"CANCELLED"}
        from ...core import bone_bake, single_bake_restore
        if any(kind == "CURVE" for kind, _pose in captured.values()):
            if single_bake_restore.available(scene):
                self.report({"ERROR"}, "请先恢复或结束已有单帧恢复记录。")
                return {"CANCELLED"}
            return _single_all_with_curves_legacy(self, context, captured)
        try:
            restore_tx = single_bake_restore.begin(scene, captured)
        except (ValueError, RuntimeError, AttributeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        # Stage mesh base writes as one unit while every cache is still bound.
        # Bone bake samples the original current frame after this succeeds.
        try:
            staged_meshes = _stage_single_meshes(scene, captured)
            try:
                bone_entry = bone_bake.single(scene)
            except BaseException:
                _restore_single_meshes(staged_meshes)
                raise
        except (ValueError, RuntimeError, AttributeError) as exc:
            single_bake_restore.discard_uncommitted(restore_tx)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        try:
            single_bake_restore.commit(scene, restore_tx, bone_entry)
        except Exception as exc:
            if bone_entry is not None: bone_bake.rollback(bone_entry)
            _restore_single_meshes(staged_meshes)
            single_bake_restore.discard_uncommitted(restore_tx)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        if bone_entry is not None or captured:
            scene["ppf_single_bake_session_stale"] = True
        # Every ordinary pose was validated and frozen before bone baking.

        # Keep group membership and source caches intact; the reversible
        # freeze disables their modifiers without unlinking any cache file.
        count = len(captured)

        if bone_entry is not None:
            scene.frame_set(bone_entry["entry_time"][0],subframe=bone_entry["entry_time"][1])
            context.view_layer.update()
        apply_object_overlays()
        self.report({"INFO"}, iface_("Baked single frame for {count} object(s)").format(count=count))
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Abort button (reads/writes the module-level job flag)
# ---------------------------------------------------------------------------


class SOLVER_OT_BakeAbort(Operator):
    """Abort the running bake and revert any partially-inserted keyframes"""

    bl_idname = "solver.bake_abort"
    bl_label = "Abort"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return bool(_bake_job.get("active")) and not _bake_job.get("aborted")

    def execute(self, context):
        _bake_job["aborted"] = True
        _bake_job["status"] = "Aborting..."
        redraw_all_areas(context)
        return {"FINISHED"}


classes = (
    OBJECT_OT_BakeAnimation,
    OBJECT_OT_BakeSingleFrame,
    SOLVER_OT_BakeAllAnimation,
    SOLVER_OT_BakeAllSingleFrame,
    SOLVER_OT_BakeAbort,
)


def register():
    for cls in classes:
        _ppf_i18n.register_class(cls)
    for handlers in (bpy.app.handlers.save_pre, bpy.app.handlers.load_pre):
        if cancel_bake_before_file_change not in handlers:
            handlers.append(cancel_bake_before_file_change)


def unregister():
    cancel_bake_before_file_change()
    for handlers in (bpy.app.handlers.save_pre, bpy.app.handlers.load_pre):
        for handler in list(handlers):
            if getattr(handler, "__name__", "") == "cancel_bake_before_file_change":
                handlers.remove(handler)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
