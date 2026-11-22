# Fork modification: changed from the upstream baseline.
# File: client.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Re-exports the event-driven ``CommunicatorFacade`` as the ``communicator``
# singleton, plus the helpers operators and UI code call directly. State
# transitions go through the pure ``transition()`` function in
# ``core/transitions.py``.
#
# The ``communicator`` re-export here is a backward-compatible surface for
# legacy UI and MCP call sites; new core code should import the singleton
# directly from ``core/facade.py`` (its definition site, which is import-cycle
# free) rather than going through this module.

import collections
import os
import threading

import bpy  # pyright: ignore
import numpy

from ..models.console import console
from ..models.groups import get_addon_data, iterate_active_object_groups
from .facade import communicator, engine
from .pc2 import (
    append_pc2_frame,
    cache_placement_is_stale,
    clear_gap_tracking,
    create_pc2_file,
    ensure_curve_handler,
    fill_gap_frames,
    get_pc2_path,
    has_static_deform_animation,
    get_static_deform_cache,
    load_curve_cache,
    mark_real_frame,
    needs_cache_visibility_keys,
    object_pc2_key,
    object_pc2_key_readonly,
    overwrite_pc2_frame,
    read_pc2_frame_count,
    read_pc2_n_verts,
    setup_mesh_cache_modifier,
)
from .status import (
    CommunicatorInfo,
    ConnectionInfo,
    RemoteStatus,
)
from .transform import inv_world_matrix


# Per-target apply state. ``needs_per_frame`` is True for meshes of any
# object type whose matrix_world can vary across frames (fcurves, NLA,
# drivers, parent chain, constraints). For those, ``constant_inv`` is
# None and the apply loop refreshes the inverse from a per-tick
# ``scene.frame_set`` before each write. For everyone else
# ``constant_inv`` carries a pre-snapshotted ``inv_world_matrix(obj)``
# numpy (4,4) and the loop skips the per-frame depsgraph eval.
_ApplyTarget = collections.namedtuple(
    "_ApplyTarget",
    ["uuid", "obj", "map", "object_type", "needs_per_frame", "constant_inv"],
)

_applying_animation = False


def _needs_per_frame_matrix(obj):
    """True if an object's matrix_world can vary across frames.

    Conservative: any frame-dependent source of motion makes us pay the
    extra per-frame ``scene.frame_set`` cost. We use Blender's own
    depsgraph eval (rather than re-implementing fcurve/NLA/driver math)
    so this also covers parent chains, constraints, and drivers.
    """
    from .utils import get_id_fcurves
    ad = obj.animation_data
    if ad:
        if get_id_fcurves(obj):
            return True
        if list(ad.nla_tracks):
            return True
        if list(ad.drivers):
            return True
    if obj.parent is not None:
        return True
    if obj.constraints:
        return True
    return False


def _apply_matrix_np(mat, positions):
    """Apply a 4x4 mathutils.Matrix to an (N,3) numpy array. Returns (N,3) float64."""
    m = numpy.array(mat, dtype=numpy.float64)
    return positions @ m[:3, :3].T + m[:3, 3]


def _needs_after_deformers(object_type, obj):
    """True when the object's MESH_CACHE must sit after upstream deformers.

    STATIC colliders always need it (captured-deformation or fcurve
    animation flows into the simulator). SHELL/SOLID/ROD cloths inherit
    the same need whenever they carry a deforming modifier stack
    (Armature, Lattice, ...) that would otherwise re-deform the solver's
    PC2 output on top of itself.
    """
    from .utils import has_deforming_modifier_stack
    return object_type == "STATIC" or has_deforming_modifier_stack(obj)


# ---------------------------------------------------------------------------
# Animation helpers (main-thread only)
# ---------------------------------------------------------------------------

def _group_type_of(obj):
    """The object type of the active group *obj* is assigned to, or None."""
    from .uuid_registry import get_object_uuid
    uid = get_object_uuid(obj)
    if not uid:
        return None
    for group in iterate_active_object_groups(bpy.context.scene):
        if any(a.uuid == uid for a in group.assigned_objects):
            return group.object_type
    return None


def _apply_post_snap_closure(context, world_by_uuid):
    """Snap every stitched source vertex exactly onto its target, in world
    space, across the per-object reconstructed position arrays.

    ``world_by_uuid`` maps object uuid -> ndarray (N, 3) of that object's
    deformed positions for the frame, indexed by the object's local vertex
    (mesh) or sample (rod) index, which is the same index space the stitch
    data uses. Arrays are mutated in place. Handles both the cross-object
    snap/merge stitches (every supported type pair, solids reconstructed via
    surface embedding upstream) and the intra-object loose-edge stitches.
    The caller gates this on the global ``post_snap_exactly`` toggle.
    """
    try:
        import json

        # Snapshot so chained stitches read original (pre-move) targets.
        snap = {uid: arr.copy() for uid, arr in world_by_uuid.items()}

        # Cross-object stitches (snap/merge pairs, all supported type pairs).
        state = get_addon_data(context.scene).state
        for pair in state.merge_pairs:
            if not pair.cross_stitch_json:
                continue
            try:
                data = json.loads(pair.cross_stitch_json)
            except Exception:
                continue
            src_uid = data.get("source_uuid")
            tgt_uid = data.get("target_uuid")
            if src_uid not in world_by_uuid or tgt_uid not in snap:
                continue
            src_arr = world_by_uuid[src_uid]
            tgt = snap[tgt_uid]
            n_src, n_tgt = len(src_arr), len(tgt)
            for row, wt in zip(data.get("ind", []), data.get("w", [])):
                # The snap-time JSON keeps the source as a single Blender
                # vertex (the SOLID-source tet re-projection is solver-only),
                # so this Blender-space closure snaps that source vertex onto
                # the target barycentric point. New rows are 6-wide
                # [s, s, s, t0, t1, t2] / [1, 0, 0, a, b, c] (target bary in
                # slots 3..5); legacy rows are 4-wide [s, t0, t1, t2] /
                # [1, a, b, c]. Accept both so pre-migration scenes still
                # snap until they are re-snapped.
                if len(row) >= 6 and len(wt) >= 6:
                    si = int(row[0])
                    t0, t1, t2 = int(row[3]), int(row[4]), int(row[5])
                    a, b, c = float(wt[3]), float(wt[4]), float(wt[5])
                elif len(row) >= 4 and len(wt) >= 4:
                    si = int(row[0])
                    t0, t1, t2 = int(row[1]), int(row[2]), int(row[3])
                    a, b, c = float(wt[1]), float(wt[2]), float(wt[3])
                else:
                    continue
                if si >= n_src or t0 >= n_tgt or t1 >= n_tgt or t2 >= n_tgt:
                    continue
                src_arr[si] = a * tgt[t0] + b * tgt[t1] + c * tgt[t2]

        # Intra-object loose-edge stitches. Skip ROD: every rod edge is
        # "loose", so averaging endpoints would corrupt the rod.
        from .encoder import detect_stitch_edges
        from .encoder.params import group_contact_lengths
        from .uuid_registry import get_object_uuid, resolve_assigned
        for group in iterate_active_object_groups(context.scene):
            if group.object_type == "ROD":
                continue
            gap, offset = group_contact_lengths(group)
            thr = 2 * gap + offset
            for obj_ref in group.assigned_objects:
                if not obj_ref.included:
                    continue
                obj = resolve_assigned(obj_ref)
                if not obj or obj.type != "MESH":
                    continue
                uid = get_object_uuid(obj)
                if uid is None or uid not in world_by_uuid:
                    continue
                sd = detect_stitch_edges(obj.data)
                if not sd:
                    continue
                arr = world_by_uuid[uid]
                base = snap[uid]
                Ind, _ = sd
                v1 = numpy.asarray(Ind[:, 0], dtype=numpy.intp)
                v2 = numpy.asarray(Ind[:, 1], dtype=numpy.intp)
                ok = (v1 < len(arr)) & (v2 < len(arr))
                v1, v2 = v1[ok], v2[ok]
                close = numpy.linalg.norm(base[v1] - base[v2], axis=1) < thr
                v1, v2 = v1[close], v2[close]
                mid = 0.5 * (base[v1] + base[v2])
                arr[v1] = mid
                arr[v2] = mid

    except Exception as e:
        console.write(f"post-snap closure failed: {e}")


def _validate_curve_mapping(display_name, map_indices, vert, spline_meta):
    # display_name is used ONLY for human-readable error text. Internal
    # routing is UUID-based via the tuples constructed in apply_animation.
    if len(map_indices) == 0:
        raise ValueError(f"Missing animation mapping for curve '{display_name}'.")
    mi = numpy.asarray(map_indices)
    if numpy.any((mi < 0) | (mi >= len(vert))):
        raise ValueError(f"Curve mapping out of range for '{display_name}'.")
    required = 0
    for meta in spline_meta:
        required += len(meta.get("params", []))
    if len(map_indices) < required:
        raise ValueError(
            f"Curve mapping too short for '{display_name}': need {required}, got {len(map_indices)}."
        )


def _validate_pc2_positions(display_name, positions, expected_count):
    """Admission only: PC2 must contain finite, exactly shaped float32 XYZ.

    No sign/range constraint on valid embedding coefficients or geometry.
    Validate before publication; finite float64 can still overflow PC2 float32.
    """
    try:
        values = numpy.asarray(positions)
        if values.shape != (expected_count, 3) or not numpy.isfinite(values).all():
            raise ValueError("non-finite or incorrectly shaped XYZ")
        with numpy.errstate(over="ignore", invalid="ignore"):
            stored = values.astype(numpy.float32)
        if not numpy.isfinite(stored).all():
            raise ValueError("XYZ exceeds finite PC2 float32 range")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid animation positions for '{display_name}': {exc}") from exc


def _validate_mesh_mapping(display_name, obj, map_indices, surface_map, vert):
    # display_name is used ONLY for human-readable error text.
    n_verts = len(obj.data.vertices)
    mi = numpy.asarray(map_indices)
    if surface_map is None:
        if len(mi) < n_verts:
            raise ValueError(
                f"Mesh mapping too short for '{display_name}': need {n_verts}, got {len(mi)}."
            )
        used = mi[:n_verts]
        if numpy.any((used < 0) | (used >= len(vert))):
            raise ValueError(f"Mesh mapping out of range for '{display_name}'.")
        return

    tri_indices, coefs, surf_tri = surface_map
    coefficients = numpy.asarray(coefs)
    if (coefficients.ndim != 2 or coefficients.shape[1] != 3
            or not numpy.isfinite(coefficients).all()):
        raise ValueError(f"Surface coefficients must be finite XYZ triples for '{display_name}'.")
    if len(tri_indices) < n_verts or len(coefs) < n_verts:
        raise ValueError(
            f"Surface mapping too short for '{display_name}': need {n_verts} entries."
        )
    ti_arr = numpy.asarray(tri_indices[:n_verts])
    if numpy.any((ti_arr < 0) | (ti_arr >= len(surf_tri))):
        raise ValueError(f"Surface triangle index out of range for '{display_name}'.")
    st_arr = numpy.asarray(surf_tri)
    tris_used = st_arr[ti_arr]
    if tris_used.shape[1] != 3:
        raise ValueError(f"Surface triangle shape is invalid for '{display_name}'.")
    if numpy.any((tris_used < 0) | (tris_used >= len(mi))):
        raise ValueError(f"Surface map index out of range for '{display_name}'.")
    mapped = mi[tris_used]
    if numpy.any((mapped < 0) | (mapped >= len(vert))):
        raise ValueError(f"Surface vertex mapping out of range for '{display_name}'.")


def _gap_fill_poses(obj, n_verts):
    """Compute the per-frame poses used to fill frames before the first
    real simulation frame arrives.

    Returns ``(rest_co, sd_cache_world)``:

    - ``rest_co``: ``(n_verts, 3)`` fallback pose, in object-LOCAL space. The
      object's rest cage, replaced by the deform-evaluated pose (Geometry
      Nodes, Armature, ...) at the CURRENT frame when the stack deforms
      vertices, and by the curve rest CVs for a CURVE. The MESH_CACHE
      modifier is excluded from the eval so we don't read prior solver
      output back in. The caller re-evaluates the deformed pose at each gap
      frame's own frame; this snapshot is only its last-resort fallback.
    - ``sd_cache_world``: ``(n_frames, n_verts, 3)`` for a Case-3 STATIC
      collider that carries a captured-deformation cache, else ``None``. When
      present, gap frame ``i`` should use ``sd_cache_world[i]`` (the
      depsgraph-baked pose) in preference to ``rest_co``.

    The cache is returned in WORLD space, deliberately un-projected. MESH_CACHE
    reads object-LOCAL positions and Blender re-applies the object's own
    animated transform on top, so each gap frame has to be divided by the
    matrix_world of ITS OWN frame. Projecting the whole cache here with a single
    matrix would silently bake in whichever frame the scene happened to be
    parked on; the caller does the per-frame projection.
    """
    if obj.type == "CURVE":
        from .curve_rod import get_rod_rest_cvs
        return get_rod_rest_cvs(obj, bpy.context), None

    rest_co = numpy.empty(n_verts * 3, dtype=numpy.float64)
    obj.data.vertices.foreach_get("co", rest_co)
    rest_co = rest_co.reshape(n_verts, 3)
    # A deform-only modifier stack (Geometry Nodes, Armature, ...) makes
    # the visible pose diverge from the rest cage. Gap-fill from the
    # deform-evaluated pose so frames before the first sim arrival (notably
    # PC2 frame 0 / scene frame 1) match the shape the solver started from,
    # instead of showing the flat rest mesh for one frame.
    from .utils import (
        eval_deform_local_positions,
        has_deforming_modifier_stack,
    )
    if has_deforming_modifier_stack(obj):
        from .pc2 import display_only_modifier_names
        deform_co = eval_deform_local_positions(
            obj, exclude_modifier_names=display_only_modifier_names(
                obj, _group_type_of(obj),
            ),
        )
        if deform_co is not None and len(deform_co) == n_verts:
            rest_co = deform_co.astype(numpy.float64)
    # Case 3 STATIC: gap-fill from the captured-deformation cache instead
    # of the undeformed rest mesh, so frames before the first sim arrival
    # still show the depsgraph-baked pose at each gap frame.
    sd_cache_world = None
    if has_static_deform_animation(obj):
        sd_cache = get_static_deform_cache(obj)
        if sd_cache is not None and sd_cache.shape[1] == n_verts:
            sd_cache_world = sd_cache.astype(numpy.float64)
    return rest_co, sd_cache_world


def _append_leading_gap_frames(pc2_path, frame_idx, n_verts, obj, start_frame):
    """Append gap frames covering PC2 indices ``[0, frame_idx)`` so the
    next appended real frame lands at PC2 index ``frame_idx``.

    Uses the Case-3 captured-deformation pose per frame when available,
    falling back to the deform-evaluated pose re-read at each gap frame.
    Shared by the file-create path and the recovery path that finds an
    existing but 0-frame (header-only) PC2 file on disk.

    Every gap frame must be built AT ITS OWN FRAME (PC2 index ``i`` is
    Blender frame ``i + start_frame``), the same way the real-frame path
    does, in both branches:

    - The captured cache is in world space, so each gap frame is divided
      by the matrix_world of its own frame. Using one matrix for every gap
      frame instead offsets the frame by exactly how far the object travels
      between the frame the scene is parked on and the gap frame, which
      surfaced as a moving STATIC being visibly out of place on the first
      frame while every later frame landed on its captured pose exactly.
    - The fallback pose is object-LOCAL, so no matrix is involved, but the
      evaluated pose itself (armature, shape keys, Geometry Nodes, ...) is
      frame-dependent and must be re-read with the scene set to each gap
      frame. A single snapshot bakes in the parked frame's pose: fetch
      never downloads solver frame 0 (the initial state), so PC2 index 0
      always comes from this gap-fill, and an armature-driven SOLID
      collider showed the parked frame's pose on Blender frame 1 while
      every later frame was correct.
    """
    if frame_idx <= 0:
        return
    rest_co, sd_cache_world = _gap_fill_poses(obj, n_verts)
    if sd_cache_world is None:
        if obj.type != "MESH":
            # CURVE gap frames use the rest CVs; rod playback has no
            # depsgraph-evaluated per-frame local pose to draw from.
            for _ in range(frame_idx):
                append_pc2_frame(pc2_path, rest_co, n_verts)
            return
        from .pc2 import display_only_modifier_names
        from .utils import eval_deform_local_positions
        hidden = display_only_modifier_names(obj, _group_type_of(obj))
        scene = bpy.context.scene
        saved_frame = scene.frame_current
        try:
            for gap_i in range(frame_idx):
                scene.frame_set(gap_i + start_frame)
                co = eval_deform_local_positions(
                    obj, exclude_modifier_names=hidden,
                )
                if co is None or len(co) != n_verts:
                    co = rest_co
                append_pc2_frame(
                    pc2_path, numpy.asarray(co, dtype=numpy.float64), n_verts,
                )
        finally:
            if scene.frame_current != saved_frame:
                scene.frame_set(saved_frame)
        return

    # A collider whose matrix_world is the same on every frame (no animation,
    # parent, constraint or driver) needs the matrix read only once.
    per_frame = _needs_per_frame_matrix(obj)
    scene = bpy.context.scene
    saved_frame = scene.frame_current
    constant_inv = (
        None if per_frame
        else numpy.array(inv_world_matrix(obj), dtype=numpy.float64)
    )
    try:
        for gap_i in range(frame_idx):
            if gap_i >= sd_cache_world.shape[0]:
                append_pc2_frame(pc2_path, rest_co, n_verts)
                continue
            if per_frame:
                scene.frame_set(gap_i + start_frame)
                inv = numpy.array(inv_world_matrix(obj), dtype=numpy.float64)
            else:
                inv = constant_inv
            local = sd_cache_world[gap_i] @ inv[:3, :3].T + inv[:3, 3]
            append_pc2_frame(pc2_path, local, n_verts)
    finally:
        if per_frame and scene.frame_current != saved_frame:
            scene.frame_set(saved_frame)


def _write_mesh_frame_to_pc2(obj, map_vert, blender_frame, start_frame,
                              n_verts_override=None,
                              place_after_deformers=False):
    if obj.type == 'CURVE':
        from .curve_output import legacy_new_output
        key=object_pc2_key(obj)
        with legacy_new_output(obj,key,get_pc2_path(key)):
            return _write_mesh_frame_to_pc2_impl(obj,map_vert,blender_frame,start_frame,
                                                  n_verts_override,place_after_deformers)
    return _write_mesh_frame_to_pc2_impl(obj,map_vert,blender_frame,start_frame,
                                         n_verts_override,place_after_deformers)


def _write_mesh_frame_to_pc2_impl(obj, map_vert, blender_frame, start_frame,
                                   n_verts_override=None,
                                   place_after_deformers=False):
    """Write a single frame to the object's PC2 file.

    Works for both mesh objects (vertex positions) and curve objects
    (CV positions in Blender's modifier layout).

    The PC2 file is keyed on the object's UUID so it survives renames;
    ``object_pc2_key`` migrates a legacy ``{obj.name}.pc2`` to the new
    location the first time it's accessed.

    Creates the file + MESH_CACHE modifier on the first arriving frame
    (regardless of frame index).  Fills gap frames when frames are
    skipped during live simulation preview. For STATIC meshes with a
    captured-deformation cache, the gap-fill draws from the cache so
    pre-real-frame display matches the depsgraph-baked pose at each
    gap frame, not the undeformed rest mesh.
    """
    n_verts = n_verts_override if n_verts_override is not None else len(obj.data.vertices)
    _validate_pc2_positions(obj.name, map_vert, n_verts)
    key = object_pc2_key(obj)
    pc2_path = get_pc2_path(key)
    if obj.type == "CURVE":
        from .curve_output import preflight_write
        preflight_write(obj)
        from .pc2 import preflight_curve_handle_restore
        preflight_curve_handle_restore(obj)
        if not os.path.exists(pc2_path):
            from .pc2 import note_new_curve_output_author
            note_new_curve_output_author(obj)
    # PC2 index 0 is the solve's first frame, which plays back at Blender frame
    # ``start_frame``; the MESH_CACHE modifier carries the same offset so the
    # index and the timeline stay in step.
    frame_idx = blender_frame - start_frame

    # If the cache on disk was recorded with a different vertex count
    # (e.g. the user edited the mesh and re-transferred), discard it and
    # start fresh — appending/overwriting against a stale header produces
    # a corrupt PC2 that MESH_CACHE silently ignores.
    if os.path.exists(pc2_path):
        try:
            cached_n_verts = read_pc2_n_verts(pc2_path)
        except Exception:
            cached_n_verts = -1
        if cached_n_verts != n_verts:
            try:
                os.remove(pc2_path)
            except OSError:
                pass
            clear_gap_tracking(key)

    # An existing file whose header reports 0 frames (header-only: an
    # interrupted write between create and the first append, a disk-full
    # error mid-sequence, or external truncation) has no frame 0 to
    # duplicate, so it cannot go through the fill_gap_frames path below.
    # Treat it like a missing file and re-run the create/gap-fill flow so
    # the appended real frame still lands at PC2 index ``frame_idx``.
    file_present = os.path.exists(pc2_path)
    if file_present:
        try:
            file_present = read_pc2_frame_count(pc2_path) >= 1
        except Exception:
            file_present = False

    if not file_present:
        # First frame to arrive (or a 0-frame file we discard): create
        # PC2, gap-fill up to this frame, and write the real data.
        os.makedirs(os.path.dirname(pc2_path), exist_ok=True)
        create_pc2_file(pc2_path, n_verts, start=0.0, sampling=1.0)
        # Fill gap frames [0, frame_idx).
        _append_leading_gap_frames(pc2_path, frame_idx, n_verts, obj, start_frame)
        # Write the actual simulation frame
        append_pc2_frame(pc2_path, map_vert, n_verts)
        mark_real_frame(key, frame_idx)
        if obj.type == "CURVE":
            # Curves use persistent handlers (MESH_CACHE doesn't work with curves)
            load_curve_cache(key)
            ensure_curve_handler()
            from .curve_output import commit_new_output
            commit_new_output(obj, key, pc2_path)
        else:
            # Defensive try/except: if ID-write is momentarily denied,
            # the next PPF_OT_FramePump tick heals the modifier.
            try:
                setup_mesh_cache_modifier(
                    obj, pc2_path, frame_start=float(start_frame),
                    place_after_deformers=place_after_deformers,
                )
            except Exception:
                pass
    else:
        current_frame_count = read_pc2_frame_count(pc2_path)
        if frame_idx >= current_frame_count:
            if frame_idx > current_frame_count:
                fill_gap_frames(
                    pc2_path, current_frame_count - 1, frame_idx, n_verts,
                    obj_key=key,
                )
            append_pc2_frame(pc2_path, map_vert, n_verts)
        else:
            overwrite_pc2_frame(pc2_path, frame_idx, map_vert, n_verts)
        if obj.type == "CURVE":
            load_curve_cache(key)
            ensure_curve_handler()
            from .curve_output import commit_new_output
            commit_new_output(obj, key, pc2_path)
        mark_real_frame(key, frame_idx)

    # PC2 is external data: rewriting it at the current frame does not
    # automatically dirty Blender's evaluated mesh or the overlay snapshot.
    # Target only this object; the overlay producer observes the depsgraph update.
    if obj.type == "CURVE":
        from .playback_origin import CURVE_START_FRAME
        if obj.get(CURVE_START_FRAME) != start_frame:
            obj[CURVE_START_FRAME] = int(start_frame)
    elif obj.type == "MESH":
        obj.update_tag(refresh={"DATA"})


# ---------------------------------------------------------------------------
# apply_animation — called from main-thread timer
# ---------------------------------------------------------------------------

# Per-tick cap on apply work during a "Fetch All" batch. The FramePump
# modal fires every 0.1s; processing one frame per tick lets the main
# thread yield between frames so tag_redraw can paint and the progress
# bar advances visibly. Without a cap the worker thread refills the
# queue as fast as the consumer drains it and the loop never exits,
# blocking redraws for the entire fetch. Live uses a larger bounded batch.
_BATCH_FRAMES_PER_TICK = 1
_LIVE_FRAMES_PER_TICK = 8
_LIVE_APPLY_BUDGET_S = 0.008


def _get_curve_cv_count(obj):
    """Return total CV count for a curve in Blender's modifier layout."""
    total = 0
    for spline in obj.data.splines:
        if spline.type == "BEZIER":
            total += 3 * len(spline.bezier_points)
        else:
            total += len(spline.points)
    return total


def _display_pins_by_uuid(display_pin_map, display_pin_frame, n):
    """Group solver frame ``n``'s display-pin positions by object.

    Returns ``{uuid: [(blender_index, world_positions), ...]}`` holding only
    the blocks still pinned at that frame, or ``{}`` when the session writes
    no display pins. ``display_pin_frame`` is ``(active, positions)`` as the
    effect runner parsed it from ``display_pin_<n>.bin``.
    """
    if not display_pin_map:
        return {}
    if display_pin_frame is None:
        raise ValueError(
            f"client: display pins are missing for solver frame {n}"
        )
    active, positions = display_pin_frame
    blocks = display_pin_map["blocks"]
    if len(active) != len(blocks) or len(positions) != display_pin_map["n_total"]:
        raise ValueError(
            f"client: display pins for solver frame {n} hold {len(active)} "
            f"blocks and {len(positions)} positions, but the session's map "
            f"has {len(blocks)} blocks and {display_pin_map['n_total']} positions"
        )
    grouped = {}
    for (uuid, blender_index, offset), is_active in zip(blocks, active, strict=True):
        if not is_active:
            continue
        rows = positions[offset:offset + blender_index.size]
        grouped.setdefault(uuid, []).append((blender_index, rows))
    return grouped


def _apply_single_frame(context, n, vert, map_by_uuid, surface_map_by_uuid,
                        target_objects, world_inv_by_uuid, start_frame,
                        curve_fit_cache=None, *, display_pins_by_uuid):
    """Process one simulation frame: write vertex/CV data to PC2 files.

    Both mesh and curve objects are written to PC2 without calling
    frame_set() here — the caller has already done so for frame N when
    any target needs a per-frame matrix. World-to-local conversion picks
    the per-frame inverse from ``world_inv_by_uuid`` for targets with an
    animated matrix_world, falling back to the target's pre-snapshotted
    ``constant_inv`` for everyone else.

    ``curve_fit_cache`` is an optional ``{uuid: (cache_list, params_data)}``
    dict pre-populated by :func:`apply_animation` so each curve's
    pseudo-inverse is computed once per fetch session instead of once per
    frame. The fit matrix only depends on the spline parameterisation
    (segment indices, ``t``, weights, cyclic flag), which is frame-
    invariant.
    """
    # Solver frame 0 is the pose at the starting frame, so solver frame n
    # displays on Blender frame n + start_frame.
    blender_frame = n + start_frame
    vert = numpy.asarray(vert)
    if vert.ndim != 2 or vert.shape[1] != 3 or not numpy.isfinite(vert).all():
        raise ValueError(f"Animation frame {blender_frame} must contain finite XYZ vertices.")

    from .curve_rod import apply_fit_cached, build_fit_cache

    # Phase 1: reconstruct every object's deformed positions for this frame,
    # carried in WORLD space so the post-snap closure can move stitched
    # vertices across objects in one common frame. Solid surfaces are
    # reconstructed in LOCAL space first (their embedding normal coef is in
    # local units) then converted to world; the world->local matrix is
    # reapplied at write time.
    world_by_uuid = {}
    records = []
    for target in target_objects:
        uid = target.uuid
        obj = target.obj
        map = target.map
        if target.needs_per_frame:
            mat = world_inv_by_uuid[uid]
        else:
            mat = target.constant_inv

        if obj.type == "CURVE":
            cached = curve_fit_cache.get(uid) if curve_fit_cache is not None else None
            if cached is None:
                cached = build_fit_cache(obj)
                if curve_fit_cache is not None:
                    curve_fit_cache[uid] = cached
            cache_list, params_data = cached
            if not params_data.get("splines"):
                continue
            _validate_curve_mapping(uid, map, vert, params_data["splines"])
            # Rod samples are already world-space solver output.
            world_sim = numpy.array(vert[map], dtype=numpy.float64)
            world_by_uuid[uid] = world_sim
            records.append({
                "kind": "curve", "obj": obj, "mat": mat, "world": world_sim,
                "cache_list": cache_list, "params": params_data,
            })
            continue

        # STATIC colliders (and any cloth with a deforming modifier stack)
        # need their MESH_CACHE after the upstream deformers so PC2 wins.
        place_after_deformers = _needs_after_deformers(target.object_type, obj)

        # --- Mesh path ---
        surface_map = surface_map_by_uuid.get(uid)
        _validate_mesh_mapping(uid, obj, map, surface_map, vert)
        if surface_map is not None:
            # Frame-embedding reconstruction p' = x0' + c1*b1' + c2*b2' + c3*n.
            # Done in LOCAL space (corners converted via mat first) because c3
            # carries local-space length units; mixing it with world-space
            # corners would inject a scale-dependent rest-pose offset.
            tri_indices_arr, coefs_arr, surf_tri_arr = surface_map
            n_verts = len(obj.data.vertices)
            ti = numpy.asarray(tri_indices_arr[:n_verts])
            c = numpy.asarray(coefs_arr[:n_verts], dtype=numpy.float64)
            tris = numpy.asarray(surf_tri_arr)[ti]
            v0 = _apply_matrix_np(mat, vert[map[tris[:, 0]]])
            v1 = _apply_matrix_np(mat, vert[map[tris[:, 1]]])
            v2 = _apply_matrix_np(mat, vert[map[tris[:, 2]]])
            b1 = v1 - v0
            b2 = v2 - v0
            nrm = numpy.cross(b1, b2)
            n_sq = numpy.einsum("ij,ij->i", nrm, nrm)
            # Guard against deformed degenerate triangles: drop the normal
            # term instead of dividing by ~0. Matches the kernel's fallback.
            safe = n_sq > 1e-20
            inv_nlen = numpy.zeros_like(n_sq)
            inv_nlen[safe] = 1.0 / numpy.sqrt(n_sq[safe])
            n_hat = nrm * inv_nlen[:, None]
            map_vert_local = (
                v0 + c[:, 0:1] * b1 + c[:, 1:2] * b2 + c[:, 2:3] * n_hat
            )
            mat_to_world = numpy.linalg.inv(numpy.array(mat, dtype=numpy.float64))
            world_vert = _apply_matrix_np(mat_to_world, map_vert_local)
        else:
            n_verts = len(obj.data.vertices)
            world_vert = numpy.array(vert[map[:n_verts]], dtype=numpy.float64)
        # A vertex an exact pin holds goes where its script puts it. The
        # solver evaluated that script at this frame's time
        # (display_pin_<N>.bin), so the vertex is not reconstructed from a
        # tetrahedral surface that cannot represent it exactly.
        for blender_index, rows in display_pins_by_uuid.get(uid, ()):
            if blender_index.size and int(blender_index.max()) >= len(world_vert):
                raise ValueError(
                    f"client: display pins for '{obj.name}' reach vertex "
                    f"{int(blender_index.max())}, but it has {len(world_vert)}"
                )
            world_vert[blender_index] = rows
        world_by_uuid[uid] = world_vert
        records.append({
            "kind": "mesh", "obj": obj, "mat": mat, "world": world_vert,
            "after": place_after_deformers,
        })

    # Phase 2: snap every stitched source vertex exactly onto its target
    # (world space), across all object types, when the global toggle is on.
    if get_addon_data(context.scene).state.post_snap_exactly:
        _apply_post_snap_closure(context, world_by_uuid)

    # Phase 3: stage and validate ALL local outputs before publishing any PC2.
    # This protects a valid prefix when later reconstruction/float32 conversion fails.
    pending_writes = []
    for rec in records:
        obj = rec["obj"]
        mat = rec["mat"]
        if rec["kind"] == "curve":
            sim_pos = _apply_matrix_np(mat, rec["world"])
            params_data = rec["params"]
            cache_list = rec["cache_list"]
            all_cvs = []
            vi_offset = 0
            for si, spline_meta in enumerate(params_data["splines"]):
                if si >= len(obj.data.splines):
                    continue
                n_sp_verts = len(spline_meta["params"])
                sp_sim = sim_pos[vi_offset : vi_offset + n_sp_verts]
                cvs = apply_fit_cached(sp_sim, cache_list[si])
                all_cvs.append(cvs)
                vi_offset += n_sp_verts
            if all_cvs:
                curve_cvs = numpy.concatenate(all_cvs, axis=0)
                n_cvs = _get_curve_cv_count(obj)
                positions = curve_cvs[:n_cvs]
                _validate_pc2_positions(obj.name, positions, n_cvs)
                pending_writes.append((obj, positions, {"n_verts_override": n_cvs}))
        else:
            map_vert = _apply_matrix_np(mat, rec["world"])
            _validate_pc2_positions(obj.name, map_vert, len(obj.data.vertices))
            pending_writes.append((obj, map_vert, {"place_after_deformers": rec["after"]}))

    # Phase 4: numeric admission is complete. IO errors are still propagated;
    # this is not a general filesystem transaction or rollback guarantee.
    for obj, positions, options in pending_writes:
        _write_mesh_frame_to_pc2(obj, positions, blender_frame, start_frame, **options)

    return blender_frame, [obj for obj, _positions, _options in pending_writes]


_heal_logged: set = set()


def heal_mesh_caches_if_stale():
    """For every assigned object in an active group, guarantee that a
    correctly-configured `ContactSolverCache` MESH_CACHE modifier exists
    whenever the current RNA owns playback and a matching PC2 file is on disk:

      * no modifier    → recreate only for a current fetched/bone-cache session
      * broken modifier (cache_format != PC2 or empty filepath) → rebind
      * modifier placed above a deformer it has to follow → re-place
        (see :func:`cache_placement_is_stale`)
      * vertex count mismatch between live mesh and PC2 → skip (the next
        fresh frame arrival will delete the stale PC2 and start over)

    Runs from the main-thread timer where ID writes are permitted. Each
    rebind is wrapped in try/except because certain Blender states briefly
    deny ID writes — the next tick will retry. Fast early-out when
    everything is already fine.
    """
    try:
        scene = bpy.context.scene
        if scene is None:
            return
        from ..ui.dynamics.group_ops import cleanup_identity_rebuilding
        from .bone_author_follow import _identity_rebuilding
        if _identity_rebuilding or cleanup_identity_rebuilding():return
        from .single_bake_restore import available
        if available(scene):return
        from .uuid_registry import resolve_assigned
        from ..models.groups import iterate_active_object_groups
        from .playback_origin import cache_start_frame
        # Hoisted: this timer fires every 0.1s, so resolve the offset once
        # per tick rather than once per object. Defaulted rather than raised:
        # the timer can tick during addon register, and a bare raise here
        # would be swallowed by the outer except and silently stop healing.
        heal_frame_start = float(cache_start_frame(scene))
        for g in iterate_active_object_groups(scene):
            for assigned in g.assigned_objects:
                if not assigned.included:
                    continue
                obj = resolve_assigned(assigned)
                if obj is None or obj.type != "MESH":
                    continue
                # Read-only key: this timer fires every 0.1s, and the
                # side-effecting variant runs an O(N) duplicate scan per
                # object (O(N^2) across the scene). An object that owns a
                # cache already has a UUID, so an empty key means "no
                # cache" and there is nothing to heal.
                key = object_pc2_key_readonly(obj)
                if not key:
                    continue
                pc2 = get_pc2_path(key)
                if not os.path.exists(pc2):
                    continue
                # Decide whether the modifier needs (re)binding BEFORE
                # touching the PC2 file. In steady state every modifier is
                # already healthy, so this skips the per-object PC2 header
                # read (1.6k file reads/tick on a large scene) that only
                # exists to validate vertex count before a rebind.
                mod = obj.modifiers.get("ContactSolverCache")
                if mod is None:
                    # Undo restores RNA, but cannot remove external PC2 files.
                    # A file alone must not resurrect an undone playback session
                    # and permanently block ordinary author proxy following.
                    from .rigid_pose import CACHE_KEY
                    if (not scene.get(CACHE_KEY)
                            and not get_addon_data(scene).state.convert_fetched_frames_to_list()):
                        continue
                needs_setup = (
                    mod is None
                    or mod.cache_format != "PC2"
                    or not mod.filepath
                    # Existing fetched data keeps its own origin. Pending
                    # start-frame edits only take effect after a new Transfer.
                    or mod.frame_start != heal_frame_start
                    # A cache bound before the visibility keying existed, or
                    # whose keys the user deleted. Gated on a late start inside
                    # the helper, so the common frame-1 scene never pays the
                    # fcurve walk.
                    or needs_cache_visibility_keys(obj, heal_frame_start)
                    # The cache sits at whatever index it was BOUND at, and
                    # a bind only happens when no PC2 exists yet, so a stack
                    # that gains a deformer after a solve keeps the cache
                    # above it and re-deforms the solver output on top of
                    # itself. Nothing else re-places it: a re-run appends to
                    # the PC2 already on disk instead of rebinding. Gated
                    # on the stack holding more than the cache itself: a
                    # lone cache has only one possible index, so the common
                    # case pays no deformer scan.
                    or (
                        len(obj.modifiers) > 1
                        and cache_placement_is_stale(
                            obj, _needs_after_deformers(g.object_type, obj)
                        )
                    )
                )
                if not needs_setup:
                    continue
                try:
                    if read_pc2_n_verts(pc2) != len(obj.data.vertices):
                        continue
                except Exception as e:
                    # Log once per (pc2, error-kind) so a broken file doesn't
                    # spam the console on every heal tick.
                    log_key = (pc2, type(e).__name__)
                    if log_key not in _heal_logged:
                        _heal_logged.add(log_key)
                        from ..models.console import console
                        console.write(
                            f"[heal_mesh_caches] skipping {obj.name}: could "
                            f"not read {pc2}: {e}"
                        )
                    continue
                try:
                    setup_mesh_cache_modifier(
                        obj, pc2, frame_start=heal_frame_start,
                        place_after_deformers=_needs_after_deformers(
                            g.object_type, obj
                        ),
                    )
                except Exception:
                    # ID-write may be briefly blocked; next tick retries.
                    pass
    except Exception:
        pass


def _capture_paused_bone_display(scene, receipt):
    """Keep the visible rig at a time the pose cache cannot replay exactly."""
    from .rigid_pose import _targets
    arms = {arm for _row, arm, _pb in _targets(scene, receipt)}
    return [(arm, [(pb.name, pb.matrix_basis.copy(), bool(pb.bone.use_connect),
                    pb.bone.inherit_scale) for pb in arm.pose.bones]) for arm in arms]


def _restore_paused_bone_display(display):
    from .rigid_pose import _write_connections
    connections = {}
    for arm, bones in display:
        for name, _basis, connected, _inherit in bones:
            pb = arm.pose.bones.get(name)
            if pb is not None and bool(pb.bone.use_connect) != connected:
                connections.setdefault(arm, {})[name] = connected
    _write_connections(connections)
    # Edit-mode connection writes can invalidate PoseBone references.
    for arm, bones in display:
        for name, basis, _connected, inherit in bones:
            pb = arm.pose.bones.get(name)
            if pb is not None:
                pb.bone.inherit_scale = inherit
                pb.matrix_basis = basis
        arm.update_tag()


def _apply_animation_impl(context=None):
    """Drain queued animation frames, writing vertex/CV data to PC2.

    One call handles both live simulation (``total==0``) and batch fetch
    (``total>0``):

    - During batch fetch: apply at most ``_BATCH_FRAMES_PER_TICK`` frames
      then yield, so the FramePump modal can return PASS_THROUGH and the
      progress bar repaint between ticks. Without this cap the worker
      thread refills the queue as fast as the consumer drains it and
      the main thread stays inside this function for the entire fetch,
      blocking redraws. Dispatches
      ``ProgressUpdated(0.5 + 0.5*applied/total)`` so the bar fills
      0.5 → 1.0 across ticks, and ``AllFramesApplied`` on full drain.
    - During live sim: apply up to eight frames per call. A paused scrub
      leaves frames in the bounded producer queue; resuming catches up over
      multiple modal ticks without monopolizing Blender's event loop.

    Must be driven from PPF_OT_FramePump.modal (a modal-operator timer
    context) — Blender 5.x denies the State / modifier / scene.frame_*
    ID writes this performs when called from bpy.app.timers callbacks.
    """
    context = context if context is not None else bpy.context
    producer_scene = context.scene
    # Retire the old generation, including in-flight producers and backlog,
    # before they can write to the newly baked author scene.
    if context.scene.get("ppf_single_bake_session_stale"):
        from .state import Activity
        runner = communicator._runner
        pending = (runner.has_pending_animation_frames()
                   or runner.has_live_backlog_work())
        if pending or engine.state.activity in (Activity.FETCHING, Activity.APPLYING):
            from .effects import DoClearAnimation
            runner.execute(DoClearAnimation())
            if engine.state.activity in (Activity.FETCHING, Activity.APPLYING):
                from .events import FetchFailed
                from .facade import tick
                from ..i18n import pgettext_iface as iface_
                engine.dispatch(FetchFailed(reason=iface_("Baked results retired the previous animation session")))
                tick()
        return
    com = communicator
    from . import live_timeline
    from . import playback
    epoch = com._runner.animation_epoch()
    # A native drag holds Blender's own frame-change state. Do not consume
    # packets or change RNA while it owns the playhead. Keep the queue intact.
    if playback.is_scrubbing(context):
        live_timeline.pause(context.scene, epoch)
        return
    state = get_addon_data(context.scene).state
    live_timeline.sync_epoch(epoch)
    if playback.is_playing(context):
        live_timeline.pause(context.scene, epoch)
    # The transport continues appending under its own lock. Even an apparently
    # static mesh can seek in gap-fill to sample its evaluated rest pose, and
    # bone packets can seek while capturing the author's basis. Defer all
    # application while the artist owns the playhead.
    paused_follow = not live_timeline.is_following(context.scene)
    # Keep receiving/installing while the artist owns the playhead. Restore
    # that frame after this bounded batch; never resume follow implicitly.
    # Mesh and bone playback share the fetched generation's time origin.
    # Editing pending transfer settings must not retime only the mesh cache.
    with com._runner._anim_lock:
        start_frame = com._runner._anim_playback_origin
    if start_frame is None:
        return
    # Snapshotted so a mid-fetch error can return the user to their
    # original frame instead of stranding the playhead wherever the
    # exception landed. Success leaves frame_current at the latest
    # written frame (existing intentional behavior for live preview).
    original_frame = context.scene.frame_current
    original_subframe = context.scene.frame_subframe
    paused_bone_display = None

    try:
        target_objects = None
        # Per-fetch-session cache of {uuid: (per_spline_cache_list, params)}.
        # build_fit_cache runs once per curve and is reused across every
        # frame in the loop below.
        curve_fit_cache = {}
        max_blender_frame = 0
        last_applied = 0
        last_total = 0
        # Snapshot before we start filling the set: if it was empty,
        # this is the first apply call in the current run and we should
        # OVERWRITE scene.frame_end (which still carries Blender's 250
        # default). On subsequent calls we clamp non-decreasing so an
        # out-of-order chunk (lower max_blender_frame than a previous
        # call) cannot make the timeline go backward, which surfaced as
        # ``advanced_first10: [3, 2, 4, ...]`` on macOS CI.
        first_apply_in_run = len(state.fetched_frame) == 0
        applied_this_tick = 0
        any_per_frame = False
        import time
        apply_started = time.perf_counter()

        while True:
            (frame, map_by_uuid, surface_map_by_uuid, display_pin_map,
             statistics_manifest, applied, total) = com.take_one_animation_frame()
            last_applied, last_total = applied, total
            if frame is None:
                break

            n, vert, statistics_blobs, display_pin_frame = frame[:4]
            # The producer identity travels with the packet; a newly connected
            # session must not claim a packet queued by an earlier connection.
            packet_sid = frame[5] if len(frame) > 5 else None
            packet_generation = frame[6] if len(frame) > 6 else None
            if packet_generation is not None and (
                    packet_generation != com._runner.animation_epoch()[0]
                    or packet_sid != com.session_id):
                raise RuntimeError("Animation producer session changed before frame application")
            from .animation import validate_fetch_scene_membership
            validate_fetch_scene_membership(context.scene, map_by_uuid)
            if total == 0:
                live_timeline.begin(context.scene, epoch)
            bone_packet = frame[4] if len(frame) > 4 else None
            if n < 0:
                continue
            from .rigid_pose import accept_frame, has_enabled_bones
            if bone_packet is not None:
                if paused_follow and paused_bone_display is None:
                    paused_bone_display = _capture_paused_bone_display(
                        context.scene, bone_packet['receipt'])
                with live_timeline.own_frame_change():
                    accept_frame(context.scene, bone_packet)
            elif has_enabled_bones(context.scene):
                raise ValueError("Bone physics frames require the current build binding receipt; transfer and build this scene again")

            if statistics_manifest:
                from .statistics_cache import install_manifest, write_frame_blob

                manifest = install_manifest(statistics_manifest)
                if not statistics_blobs:
                    raise ValueError(
                        f"client: statistics are missing for solver frame {n}"
                    )
                # The vertex frame index bounds the statistics: each blob
                # belongs to a frame already fetched, and the last one is the
                # frame this iteration carries.
                statistics_frames = [
                    write_frame_blob(blob, manifest, max_solver_frame=n)
                    for blob in statistics_blobs
                ]
                statistics_frame = statistics_frames[-1]
                if statistics_frame != n:
                    raise ValueError(
                        "client: statistics frame mismatch "
                        f"(vertices={n}, statistics={statistics_frame})"
                    )

            if target_objects is None:
                target_objects = []
                for group in iterate_active_object_groups(context.scene):
                    for assigned in group.assigned_objects:
                        if not assigned.uuid:
                            raise ValueError(
                                f"client: assigned object has empty UUID"
                                f" (name={getattr(assigned, 'name', '?')})"
                            )
                        if assigned.uuid in map_by_uuid:
                            from .uuid_registry import get_object_by_uuid
                            obj = get_object_by_uuid(assigned.uuid)
                            if obj is None or obj.type not in ("MESH", "CURVE"):
                                continue
                            # ROD sims fold faces freely, so the Wireframe
                            # modifier's even-offset compensation produces
                            # huge frame-to-frame extrusion jumps. Force it
                            # off on every fetch so existing scenes (and any
                            # user-added Wireframe modifier) stay stable.
                            if (
                                group.object_type == "ROD"
                                and obj.type == "MESH"
                            ):
                                for m in obj.modifiers:
                                    if m.type == "WIREFRAME" and m.use_even_offset:
                                        m.use_even_offset = False
                            # Any mesh whose matrix_world varies across
                            # frames needs the per-frame inverse, whatever
                            # its object type. MESH_CACHE re-applies the
                            # object's animated transform at display time,
                            # so localizing frame N with any other frame's
                            # matrix displaces the displayed mesh by the
                            # object's travel between the two frames. A
                            # fully pinned SOLID collider parented to an
                            # animated armature shipped this bug: the cloth
                            # appeared to penetrate it by exactly one frame
                            # of root motion during live fetch.
                            needs_per_frame = (
                                obj.type == "MESH"
                                and _needs_per_frame_matrix(obj)
                            )
                            constant_inv = (
                                None if needs_per_frame
                                else numpy.array(
                                    inv_world_matrix(obj), dtype=numpy.float64,
                                )
                            )
                            target_objects.append(_ApplyTarget(
                                uuid=assigned.uuid,
                                obj=obj,
                                map=map_by_uuid[assigned.uuid],
                                object_type=group.object_type,
                                needs_per_frame=needs_per_frame,
                                constant_inv=constant_inv,
                            ))
                any_per_frame = any(t.needs_per_frame for t in target_objects)

            # For targets whose matrix_world varies per frame, we must
            # evaluate Blender's depsgraph at frame N to read the true
            # playback matrix_world before computing PC2_local. This is
            # the cost of letting PC2 + MESH_CACHE display the solver's
            # world-space output against the user's preserved
            # fcurves/parents/constraints without drift.
            world_inv_by_uuid = {}
            if any_per_frame:
                with live_timeline.own_frame_change():
                    context.scene.frame_set(n + start_frame)
                for t in target_objects:
                    if t.needs_per_frame:
                        world_inv_by_uuid[t.uuid] = numpy.array(
                            inv_world_matrix(t.obj), dtype=numpy.float64,
                        )

            if (context.scene != producer_scene or bpy.context.scene != producer_scene
                    or (packet_generation is not None and (
                        packet_generation != com._runner.animation_epoch()[0]
                        or packet_sid != com.session_id))):
                raise RuntimeError("Animation scene or producer session changed before frame application")
            with live_timeline.own_frame_change():
                bf, written_objects = _apply_single_frame(
                    context, n, vert, map_by_uuid, surface_map_by_uuid,
                    target_objects, world_inv_by_uuid, start_frame,
                    curve_fit_cache=curve_fit_cache,
                    display_pins_by_uuid=_display_pins_by_uuid(
                        display_pin_map, display_pin_frame, n,
                    ),
                )
            if (packet_sid and packet_sid == com.session_id
                    and packet_generation == com._runner.animation_epoch()[0]
                    and context.scene == producer_scene and bpy.context.scene == producer_scene):
                from . import pc2
                for obj in written_objects:
                    key = pc2.object_pc2_key_readonly(obj)
                    if (any(other != obj and pc2.object_pc2_key_readonly(other) == key
                            for other in bpy.data.objects)
                            or any(other != context.scene and other.objects.get(obj.name) == obj
                                   for other in bpy.data.scenes)):
                        continue
                    owned = obj.type == "MESH" and pc2.mesh_output_matches_session(
                        context.scene, obj, packet_sid)
                    if obj.type == "CURVE":
                        from . import curve_output
                        from .curve_rod import curve_data_external_users
                        owned = (not obj.library and not obj.data.library
                                 and obj.is_editable and curve_data_external_users(obj) == 1
                                 and key in pc2.assigned_uuids_in_active_groups(context.scene)
                                 and key in pc2._curve_cache
                                 and os.path.realpath(curve_output._live_paths.get(key, ""))
                                 == os.path.realpath(pc2.get_pc2_path(key)))
                    if owned:
                        state.last_session_id = packet_sid
                        break
            max_blender_frame = max(max_blender_frame, bf)
            state.add_fetched_frame(n)
            applied_this_tick += 1
            # Yield between batches, including a live backlog accumulated
            # while the artist scrubbed, so resume does not lock the UI.
            budget = _BATCH_FRAMES_PER_TICK if total > 0 else _LIVE_FRAMES_PER_TICK
            if (applied_this_tick >= budget
                    or time.perf_counter() - apply_started >= _LIVE_APPLY_BUDGET_S):
                break

        if max_blender_frame > 0:
            context.scene.frame_start = start_frame
            # Track the latest fetched frame; first apply in a run
            # overwrites Blender's 250 default, later calls clamp
            # non-decreasing so async out-of-order chunks can't
            # walk the timeline backward.
            if first_apply_in_run:
                context.scene.frame_end = max_blender_frame
            else:
                context.scene.frame_end = max(
                    context.scene.frame_end, max_blender_frame
                )
            if not paused_follow:
                with live_timeline.own_frame_change():
                    context.scene.frame_set(max_blender_frame)
            # Rods play back via a frame_change_post handler, not MESH_CACHE.
            # A per-frame frame_set earlier in this tick (STATIC per-frame
            # matrices) can run that handler before this tick's curve cache
            # was loaded, leaving the rod a frame behind the meshes during
            # live fetch. Force a fresh curve apply at the final frame so
            # rods stay in lockstep.
            from .pc2 import refresh_curves_at_current_frame
            if not paused_follow:
                refresh_curves_at_current_frame(context.scene)
            else:
                with live_timeline.own_frame_change():
                    context.scene.frame_set(original_frame, subframe=original_subframe)
                if paused_bone_display is not None:
                    from . import rigid_pose
                    index = rigid_pose._cache.get(context.scene.get(rigid_pose.CACHE_KEY), {})
                    selected = original_frame - index.get('receipt', {}).get('start_frame', original_frame)
                    exact_pose = (original_subframe == 0.0
                                  and str(selected) in index.get('frames', {}))
                    if not exact_pose:
                        _restore_paused_bone_display(paused_bone_display)
                refresh_curves_at_current_frame(context.scene)

        if last_total > 0:
            if last_applied >= last_total:
                from .events import AllFramesApplied
                engine.dispatch(AllFramesApplied())
            else:
                # Each phase (FETCHING, APPLYING) owns its own 0→1 bar.
                # The apply side must only drive progress while activity
                # is APPLYING — dispatching during FETCHING would fight
                # the download-side value and flicker the bar.
                from .state import Activity
                if engine.state.activity == Activity.APPLYING:
                    from .events import ProgressUpdated
                    engine.dispatch(ProgressUpdated(
                        progress=last_applied / last_total
                    ))
    except Exception as e:
        # Restore the playhead so the user isn't stranded on whatever
        # frame the per-frame scene.frame_set last set before the error.
        try:
            if (context.scene.frame_current != original_frame
                    or context.scene.frame_subframe != original_subframe):
                with live_timeline.own_frame_change():
                    context.scene.frame_set(original_frame, subframe=original_subframe)
            if paused_follow and paused_bone_display is not None:
                _restore_paused_bone_display(paused_bone_display)
        except Exception:
            pass
        console.write(f"Error applying animation frame: {e}")
        from .events import FetchFailed
        engine.dispatch(FetchFailed(reason=f"apply: {e}"))


def apply_animation(context=None):
    """Apply on Blender's main thread, ignoring nested pump invocations."""
    global _applying_animation
    from .live_timeline import assert_main_thread
    assert_main_thread()
    if _applying_animation:
        return
    _applying_animation = True
    try:
        _apply_animation_impl(context)
    finally:
        _applying_animation = False
