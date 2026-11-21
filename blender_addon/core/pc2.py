# Fork modification: changed from the upstream baseline.
# File: pc2.py
# License: Apache v2.0
#
# PC2 (PointCache2) file writer and Blender MESH_CACHE modifier management.
# Used to store per-frame vertex positions for simulation animation.

import os
import errno
import stat
import json
import struct
import tempfile

import numpy

try:
    import bpy  # pyright: ignore
    from ..i18n import pgettext_iface as iface_, pgettext_tip as tip_
except ImportError:
    bpy = None

PC2_HEADER = b"POINTCACHE2\0"
PC2_VERSION = 1
PC2_HEADER_SIZE = 32
FRAME_VERTEX_SIZE = 12  # 3 * sizeof(float32)
# Byte offsets into the PC2 header (see write_pc2): 12-byte magic +
# version(i) + n_verts(i) + start(f) + sampling(f) + n_frames(i).
PC2_NVERTS_OFFSET = 16
PC2_FRAMECOUNT_OFFSET = 28
MODIFIER_NAME = "ContactSolverCache"
# The geometry-nodes modifier Convert To Particle Mesh gives a SAND body.
PARTICLE_MESH_MODIFIER_NAME = "Particle Mesh"


def display_only_modifier_names(obj, object_type) -> set:
    """Names of the modifiers on *obj* that exist only to DRAW it.

    Every read of the pose a simulated object starts from hides these: the
    encoder's starting-frame evaluation and Capture Deformation both, so the
    two agree by construction. The ContactSolverCache replays a previous solve,
    a ROD's Wireframe turns an edge-only mesh into zero vertices (and a faced
    one into different vertices), and a SAND body's Particle Mesh turns it into
    a point cloud with no mesh vertices at all.
    """
    names = {MODIFIER_NAME}
    if object_type == "ROD":
        names.update(m.name for m in obj.modifiers if m.type == "WIREFRAME")
    elif object_type == "SAND":
        names.add(PARTICLE_MESH_MODIFIER_NAME)
    return names
# Cache namespace suffixes embedded in PC2 keys / on-disk filenames.
STATIC_DEFORM_SUFFIX = "_staticdeform"
PIN_DEFORM_SUFFIX = "__pindeform"
PIN_DEFORM_V2_SUFFIX = "__pindeform_v2"

# Gap tracking: {obj_uuid: set of real (non-gap-filled) frame indices}
_real_frames: dict[str, set[int]] = {}


# ---------------------------------------------------------------------------
# PC2 file I/O
# ---------------------------------------------------------------------------


def write_pc2(filepath, frames, start=0.0, sampling=1.0):
    """Write a complete PC2 file from a list of per-frame vertex positions.

    Args:
        filepath: Output .pc2 file path.
        frames: List of numpy arrays, each shape (n_verts, 3).
        start: Start frame (float).
        sampling: Frame sampling rate (float).
    """
    n_frames = len(frames)
    n_verts = frames[0].shape[0]
    with open(filepath, "wb") as f:
        f.write(PC2_HEADER)
        f.write(struct.pack("<i", PC2_VERSION))
        f.write(struct.pack("<i", n_verts))
        f.write(struct.pack("<f", start))
        f.write(struct.pack("<f", sampling))
        f.write(struct.pack("<i", n_frames))
        for positions in frames:
            f.write(numpy.asarray(positions, dtype="<f").tobytes())


def create_pc2_file(filepath, n_verts, start=0.0, sampling=1.0):
    """Create an empty PC2 file with header only (0 frames)."""
    with open(filepath, "wb") as f:
        f.write(PC2_HEADER)
        f.write(struct.pack("<i", PC2_VERSION))
        f.write(struct.pack("<i", n_verts))
        f.write(struct.pack("<f", start))
        f.write(struct.pack("<f", sampling))
        f.write(struct.pack("<i", 0))


def append_pc2_frame(filepath, positions, n_verts):
    """Append a single frame to an existing PC2 file and update the header.

    Ordering is important: we write the frame data at EOF and fsync
    before bumping the header's frame count.  If MESH_CACHE or another
    reader observes the file mid-write, it sees either the old count
    with the old data (stale but consistent) or the new count with the
    new data on disk — never "new count, data missing".
    """
    with open(filepath, "r+b") as f:
        f.seek(PC2_FRAMECOUNT_OFFSET)
        frame_tot = struct.unpack("<i", f.read(4))[0]
        # Data first: append to EOF and flush to disk.
        f.seek(0, 2)
        f.write(numpy.asarray(positions, dtype="<f").tobytes())
        f.flush()
        try:
            os.fsync(f.fileno())
        except (OSError, AttributeError):
            pass
        # Header update publishes the new frame to readers.
        f.seek(PC2_FRAMECOUNT_OFFSET)
        f.write(struct.pack("<i", frame_tot + 1))
        f.flush()
        try:
            os.fsync(f.fileno())
        except (OSError, AttributeError):
            pass


def read_pc2_frame_count(filepath):
    """Read the frame_tot from a PC2 file header."""
    with open(filepath, "rb") as f:
        f.seek(PC2_FRAMECOUNT_OFFSET)
        return struct.unpack("<i", f.read(4))[0]


def read_pc2_n_verts(filepath):
    """Read the vertex count from a PC2 file header."""
    with open(filepath, "rb") as f:
        f.seek(PC2_NVERTS_OFFSET)
        return struct.unpack("<i", f.read(4))[0]


def read_pc2_frame(filepath, frame_idx, n_verts):
    """Read a specific frame from a PC2 file."""
    if frame_idx < 0:
        raise ValueError(
            f"read_pc2_frame: negative frame index {frame_idx} for {filepath}"
        )
    offset = PC2_HEADER_SIZE + frame_idx * n_verts * FRAME_VERTEX_SIZE
    with open(filepath, "rb") as f:
        f.seek(offset)
        data = f.read(n_verts * FRAME_VERTEX_SIZE)
        return numpy.frombuffer(data, dtype="<f").reshape(n_verts, 3).copy()


def overwrite_pc2_frame(filepath, frame_idx, positions, n_verts):
    """Overwrite a specific frame in an existing PC2 file."""
    offset = PC2_HEADER_SIZE + frame_idx * n_verts * FRAME_VERTEX_SIZE
    with open(filepath, "r+b") as f:
        f.seek(offset)
        f.write(numpy.asarray(positions, dtype="<f").tobytes())


def _load_pc2_array(filepath):
    """Read a whole PC2 file into an ``(n_frames, n_verts, 3)`` array.

    Returns ``None`` (and leaves no side effect) when the file is
    missing or carries an empty (0-frame / 0-vert) header. Reads the
    header and the float body from a single open handle so the bulk
    payload streams without reopening the file.
    """
    if not os.path.exists(filepath):
        return None
    with open(filepath, "rb") as f:
        f.seek(PC2_NVERTS_OFFSET)
        n_verts = struct.unpack("<i", f.read(4))[0]
        f.seek(PC2_FRAMECOUNT_OFFSET)
        n_frames = struct.unpack("<i", f.read(4))[0]
        if n_frames < 1 or n_verts < 1:
            return None
        data = numpy.frombuffer(f.read(), dtype="<f")
    return data.reshape(n_frames, n_verts, 3).copy()


def fill_gap_frames(filepath, from_frame_idx, to_frame_idx, n_verts, obj_key=None):
    """Fill gap frames by duplicating the nearest real frame across (from, to).

    If ``obj_key`` (a UUID string) is supplied and the object has any real
    frames tracked in ``_real_frames``, the largest real frame <=
    ``from_frame_idx`` is used as the source. This avoids cascading a
    previously-gap-filled pose into new gaps. When no real frame is available
    yet, the function falls back to reading ``from_frame_idx`` (old behavior)
    so the first-ever gap-fill still produces a plausible pose.

    Raises ``ValueError`` when ``from_frame_idx < 0``: there is no source
    frame to duplicate (a 0-frame header has no frame 0 either), so the
    caller must instead seed the leading gap from a known pose (rest /
    captured-deformation), as the file-create path does.
    """
    if from_frame_idx < 0:
        raise ValueError(
            f"fill_gap_frames: no source frame for from_frame_idx="
            f"{from_frame_idx} ({filepath})"
        )
    src_idx = from_frame_idx
    if obj_key is not None:
        reals = _real_frames.get(obj_key, set())
        candidates = [i for i in reals if i <= from_frame_idx]
        if candidates:
            src_idx = max(candidates)
    src_data = read_pc2_frame(filepath, src_idx, n_verts)
    src_bytes = numpy.asarray(src_data, dtype="<f").tobytes()
    with open(filepath, "r+b") as f:
        # Append gap frames at end of file
        f.seek(0, 2)
        for _ in range(from_frame_idx + 1, to_frame_idx):
            f.write(src_bytes)
        # Update frame_tot
        f.seek(PC2_FRAMECOUNT_OFFSET)
        old_tot = struct.unpack("<i", f.read(4))[0]
        new_tot = max(old_tot, to_frame_idx)
        f.seek(PC2_FRAMECOUNT_OFFSET)
        f.write(struct.pack("<i", new_tot))


# ---------------------------------------------------------------------------
# Gap tracking
# ---------------------------------------------------------------------------


def mark_real_frame(obj_key, frame_idx):
    """Record that a frame contains real (non-gap-filled) data."""
    _real_frames.setdefault(obj_key, set()).add(frame_idx)


def get_gap_frame_indices(obj_key, total_frames):
    """Return frame indices that are gap-filled (not real)."""
    real = _real_frames.get(obj_key, set())
    return [i for i in range(total_frames) if i not in real]


def clear_gap_tracking(obj_key=None):
    """Clear gap tracking for one or all objects."""
    if obj_key is None:
        _real_frames.clear()
    else:
        _real_frames.pop(obj_key, None)


# ---------------------------------------------------------------------------
# File path helpers
# ---------------------------------------------------------------------------


def get_pc2_dir():
    """Get directory for PC2 cache files.

    When the .blend is saved, files are stored under a per-blend subfolder so
    multiple .blend files sharing the same directory don't collide:
        <blend_dir>/data/<blend_basename_noext>/<name>.pc2
    Before first save, a flat temp dir is used; migrate_pc2_on_save moves
    those files into the per-blend layout on save_post.
    """
    blend_path = bpy.data.filepath if bpy else ""
    if blend_path:
        basename = os.path.splitext(os.path.basename(blend_path))[0]
        return os.path.join(os.path.dirname(blend_path), "data", basename)
    return os.path.join(tempfile.gettempdir(), "data")


def get_pc2_path(key):
    """Get the PC2 file path for a given stable key string.

    The key is normally an object UUID (see ``object_pc2_key``). For
    legacy data that was written under the object's Blender name, the
    same sanitisation still produces the old filename so
    ``object_pc2_key`` can detect and migrate it.
    """
    safe = key.replace(" ", "_").replace("/", "_")
    return os.path.join(get_pc2_dir(), f"{safe}.pc2")


def object_pc2_key(obj) -> str:
    """Return the UUID-backed PC2 key for ``obj``.

    Ensures the object carries a UUID, and migrates a legacy
    ``<name>.pc2`` file in place so the UUID-keyed path is the only
    one on disk afterwards. Return value is safe to pass to
    ``get_pc2_path`` and to use as a ``_real_frames`` / ``_curve_cache``
    dict key.
    """
    from .uuid_registry import get_or_create_object_uuid
    uid = get_or_create_object_uuid(obj)
    target = get_pc2_path(uid)
    if not os.path.exists(target):
        legacy = get_pc2_path(obj.name)
        if legacy != target and os.path.exists(legacy):
            try:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                os.rename(legacy, target)
            except OSError:
                pass
    return uid


def object_pc2_key_readonly(obj) -> str:
    """Return the existing PC2 key for ``obj`` with no side effects.

    Unlike :func:`object_pc2_key`, this never assigns a UUID, never
    reassigns another object's UUID on duplicate detection, and never
    migrates a legacy ``<name>.pc2`` file. It returns the empty string
    when ``obj`` has no UUID yet. Use it from read-only contexts (draw,
    operator poll, frame_change handlers): any object that actually owns
    a cache was assigned a UUID by the write-context capture path or by
    load_post migration, so an empty key here is behaviorally "no cache".
    """
    from .uuid_registry import get_object_uuid
    return get_object_uuid(obj)


# ---------------------------------------------------------------------------
# MESH_CACHE modifier management
# ---------------------------------------------------------------------------


# Modifier types that change vertex count or topology. MESH_CACHE
# must come BEFORE any of these in the stack so its per-frame data
# matches the vertex count baked into the PC2 file. Deform-only
# modifiers (Armature, Lattice, Hook, Shape Keys, ...) preserve
# vertex count and can sit above MESH_CACHE; their output is then
# overridden by the PC2 data on display. Geometry Nodes can do
# anything topology-wise, so we treat them as generative.
_GENERATIVE_MODIFIER_TYPES = frozenset({
    "ARRAY", "BEVEL", "BOOLEAN", "BUILD", "DECIMATE", "EDGE_SPLIT",
    "MASK", "MIRROR", "MULTIRES", "REMESH", "SCREW", "SKIN",
    "SOLIDIFY", "SUBSURF", "TRIANGULATE", "WELD", "WIREFRAME",
    "VOLUME_TO_MESH", "MESH_TO_VOLUME",
    "EXPLODE", "PARTICLE_INSTANCE",
    "NODES",
})


def _stack_preserves_vertex_count(obj) -> bool:
    """True if *obj*'s evaluated mesh keeps the base vertex count.

    Used to disambiguate a Geometry Nodes modifier, which lives in both
    the deforming and the generative type sets. When the whole stack
    preserves the vertex count, no modifier changes topology, so any GN
    present only displaces existing vertices (deform-only) and the
    cache must sit AFTER it. When the count differs, something is
    generative and we keep the conservative before-the-boundary
    placement. Returns False if evaluation isn't available, which keeps
    the safe (treat GN as generative) behavior.
    """
    try:
        import bpy
        base = len(obj.data.vertices)
        deps = bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(deps)
        mesh = eval_obj.to_mesh()
        try:
            return len(mesh.vertices) == base
        finally:
            eval_obj.to_mesh_clear()
    except Exception:
        return False


def _cache_insertion_index(obj) -> int:
    """Index where ContactSolverCache should sit so it lands AFTER every
    position-preserving deformer and BEFORE the first topology-changing
    modifier. Computed over the stack with the cache excluded, so the
    returned value is the cache's desired final index and can be passed
    straight to ``obj.modifiers.move(current_idx, here)``.

    A NODES (Geometry Nodes) modifier is treated as a topology boundary
    only when it actually changes the vertex count; a deform-only GN
    (e.g. a Set Position wave) is treated as a deformer so PC2 wins on
    display instead of the GN re-deforming the cache output on top of
    itself.

    That vertex-count question costs a depsgraph ``to_mesh()``, so it is
    asked only once a NODES modifier is actually reached.
    :func:`cache_placement_is_stale` calls this on every frame-pump tick,
    and a stack carrying no Geometry Nodes has no use for the answer.
    """
    nodes_are_deformers = None
    pos = 0
    for m in obj.modifiers:
        if m.name == MODIFIER_NAME:
            continue
        is_boundary = m.type in _GENERATIVE_MODIFIER_TYPES
        if is_boundary and m.type == "NODES":
            if nodes_are_deformers is None:
                nodes_are_deformers = _stack_preserves_vertex_count(obj)
            is_boundary = not nodes_are_deformers
        if is_boundary:
            return pos
        pos += 1
    return pos


def modifiers_after_cache_boundary(obj) -> list:
    """The modifiers ContactSolverCache sits in front of when it is placed
    after the deformers: the first topology-changing modifier and every one
    after it, as :func:`_cache_insertion_index` finds them. The cache itself
    is not listed.

    On display these run on top of the solver output, while the cache
    replaces everything in front of it. So the pose the solver starts from is
    the stack evaluated WITHOUT them, which is how
    ``utils.eval_deform_local_positions`` reads a stack that changes the
    vertex count.
    """
    boundary = _cache_insertion_index(obj)
    after = []
    pos = 0
    for m in obj.modifiers:
        if m.name == MODIFIER_NAME:
            continue
        if pos >= boundary:
            after.append(m)
        pos += 1
    return after


def cache_placement_is_stale(obj, place_after_deformers) -> bool:
    """True when *obj* carries a ContactSolverCache sitting at an index
    other than the one *place_after_deformers* asks for.

    Placement is decided when the cache is BOUND, and a bind happens only
    on the first frame written to a PC2 that is not already on disk. A
    stack that gains a deformer after that (an Armature added, or an
    existing one un-muted, once a solve has run) keeps its cache above the
    deformer, and re-running does not correct it: the PC2 file is still
    there, so the frame write takes the append path and never rebinds.
    MESH_CACHE replays in OVERWRITE mode, so the deformer then re-applies
    its deformation on top of the solver output that already carries it,
    and every driven vertex moves twice. ``heal_mesh_caches_if_stale``
    tests this so the next frame-pump tick re-places the modifier.

    The verdict must match what :func:`setup_mesh_cache_modifier` would do
    with the same flag, across BOTH of its branches: index 0 when the
    caller wants the cache first, ``_cache_insertion_index`` otherwise.
    """
    if obj is None or getattr(obj, "type", None) != "MESH":
        return False
    mods = getattr(obj, "modifiers", None)
    if mods is None or len(mods) < 2:
        # A stack holding nothing but the cache has one arrangement, and
        # it is the one both branches ask for.
        return False
    idx = mods.find(MODIFIER_NAME)
    if idx < 0:
        return False
    target = _cache_insertion_index(obj) if place_after_deformers else 0
    return idx != target


def _cache_visibility_paths():
    """RNA data paths (object-relative) for the cache modifier's two
    visibility toggles. Both are keyed together so a render cannot disagree
    with the viewport about whether the solve has started."""
    return (
        f'modifiers["{MODIFIER_NAME}"].show_viewport',
        f'modifiers["{MODIFIER_NAME}"].show_render',
    )


def _iter_cache_visibility_fcurves(action):
    """Yield ``(container, fcurve)`` for every cache-visibility fcurve in
    *action*, across ALL slots of a Blender 5.x layered action (and the legacy
    flat layout).

    ``container`` is the ``fcurves`` collection the curve lives in, so a caller
    can remove it. The all-slots walk is load-bearing: when the object already
    owns an action (an armature-driven cloth, a shape-key-animated mesh), a
    fresh ``keyframe_insert`` on the modifier lands in the object's OWN slot,
    which is often not the first channelbag. ``utils._get_fcurves`` returns only
    the first non-empty channelbag, so reading through it silently misses these
    keys, and the reader concludes they are absent even though Blender is
    evaluating them. That false "missing" is what drove the heal pass to rewrite
    the keys on every tick.
    """
    paths = set(_cache_visibility_paths())
    if action is None:
        return
    if hasattr(action, "layers") and len(action.layers) > 0:
        for layer in action.layers:
            for strip in layer.strips:
                for slot in action.slots:
                    cb = strip.channelbag(slot)
                    if cb is None:
                        continue
                    for fc in list(cb.fcurves):
                        if fc.data_path in paths:
                            yield cb.fcurves, fc
    elif hasattr(action, "fcurves"):
        for fc in list(action.fcurves):
            if fc.data_path in paths:
                yield action.fcurves, fc


def remove_cache_visibility_keys(obj) -> int:
    """Drop the modifier-visibility fcurves this module writes. Returns the
    number removed."""
    ad = getattr(obj, "animation_data", None)
    action = getattr(ad, "action", None) if ad else None
    if action is None:
        return 0
    removed = 0
    for container, fc in _iter_cache_visibility_fcurves(action):
        try:
            container.remove(fc)
            removed += 1
        except Exception:
            pass
    return removed


def needs_cache_visibility_keys(obj, frame_start) -> bool:
    """True when *obj* is missing the visibility keys its Starting Frame needs.

    Lets the heal pass adopt a cache that was bound before this keying existed,
    without waiting for a re-fetch. Callers must gate on ``frame_start > 1``:
    at frame 1 there are no keys to look for, and this walks fcurves, which the
    heal timer cannot afford to do per object per tick on a large scene.
    """
    if frame_start <= 1:
        return False
    ad = getattr(obj, "animation_data", None)
    action = getattr(ad, "action", None) if ad else None
    if action is None:
        return True
    paths = set(_cache_visibility_paths())
    found = {fc.data_path for _, fc in _iter_cache_visibility_fcurves(action)}
    return found != paths


def sync_cache_visibility_keys(obj, frame_start) -> None:
    """Keep the cache modifier switched off until the solve's first frame.

    MESH_CACHE clamps a cache index below zero to row 0 and its OVERWRITE
    deform mode then stamps that row onto the mesh, so a modifier left enabled
    through the lead-in replaces whatever the artist animated there with the
    simulation's starting pose. That is not a cosmetic problem: the lead-in is
    the entire reason a solve would start late.

    Two CONSTANT keys (off at ``frame_start - 1``, on at ``frame_start``) hand
    the mesh back to its own deformers before the solve and to the cache from
    the solve onward. CONSTANT because this is a switch, not a blend: a default
    BEZIER handle would put fractional visibility on the frames in between.

    A solve that starts at frame 1 has no lead-in to protect, so the keys are
    removed and the modifier goes back to plain always-on. That also restores a
    scene whose Starting Frame was moved back to 1.
    """
    removed = remove_cache_visibility_keys(obj)
    mod = obj.modifiers.get(MODIFIER_NAME)
    if mod is None:
        return
    if frame_start <= 1:
        # No lead-in. Leave the modifier visible; the removal above already
        # cleared any keys a higher Starting Frame had written.
        if removed:
            mod.show_viewport = True
            mod.show_render = True
        return

    for path, value_off, value_on in (
        (_cache_visibility_paths()[0], False, True),
        (_cache_visibility_paths()[1], False, True),
    ):
        attr = path.rsplit(".", 1)[1]
        setattr(mod, attr, value_off)
        obj.keyframe_insert(data_path=path, frame=frame_start - 1)
        setattr(mod, attr, value_on)
        obj.keyframe_insert(data_path=path, frame=frame_start)

    # Walk every slot, not just the first channelbag: when the object already
    # owns an action the freshly inserted keys land in its own slot, which
    # utils._get_fcurves would skip, leaving them BEZIER.
    action = obj.animation_data.action if obj.animation_data else None
    for _, fc in _iter_cache_visibility_fcurves(action):
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT"


def setup_mesh_cache_modifier(obj, pc2_path, frame_start=0.0,
                              place_after_deformers=False):
    """Add or update a MESH_CACHE modifier on *obj* pointing to *pc2_path*.

    Also stamps the current session id on the object as ``_solver_session``
    so reconcile-on-reconnect can tell whether the modifier's cached data
    was produced by the currently-connected run.

    ``place_after_deformers``: when True, position the new modifier so it
    sits AFTER all position-preserving deformers (Armature, Lattice,
    Hook, Shape Keys, ...) and BEFORE the first topology-changing
    modifier (Subsurf, Mirror, Solidify, ...). Used for STATIC colliders
    whose captured-deformation cache feeds into the simulator: the
    simulator-projected positions in PC2 must win over the deformers
    that produced the input, while the user's downstream visual
    decorators (Subsurf, etc.) keep applying on top. The default
    (False) puts the modifier first, which is the right semantics for
    dynamic objects (sim output is the base; decorators run after).
    """
    mod = obj.modifiers.get(MODIFIER_NAME)
    if mod is None:
        mod = obj.modifiers.new(name=MODIFIER_NAME, type="MESH_CACHE")
    # Enforce placement for both freshly-created AND already-present
    # caches: a stale order saved by an older session (or a file made
    # before a deformer was added) is corrected on the next setup,
    # rather than leaving the cache wherever it happened to land.
    idx = obj.modifiers.find(MODIFIER_NAME)
    if place_after_deformers:
        target = _cache_insertion_index(obj)
        if target != idx:
            obj.modifiers.move(idx, target)
    else:
        if idx > 0:
            obj.modifiers.move(idx, 0)
    mod.cache_format = "PC2"
    mod.filepath = bpy.path.relpath(pc2_path) if bpy.data.filepath else pc2_path
    mod.interpolation = "LINEAR"
    mod.deform_mode = "OVERWRITE"
    mod.play_mode = "SCENE"
    mod.time_mode = "FRAME"
    mod.frame_start = frame_start
    mod.frame_scale = 1.0
    # Must follow the frame_start write: the keys are derived from it, and a
    # rebind after the Starting Frame moved has to re-place them.
    sync_cache_visibility_keys(obj, int(frame_start))
    mod.factor = 1.0
    mod.forward_axis = "POS_Y"
    mod.up_axis = "POS_Z"
    # Session stamp: record which connected run bound this cache so
    # a later reopen can detect stale bindings.
    try:
        from .facade import communicator as _com
        sid = _com.session_id or ""
        if sid:
            obj["_solver_session"] = sid
    except Exception:
        pass
    return mod


def remove_mesh_cache_modifier(obj):
    """Remove the MESH_CACHE modifier from *obj* if present."""
    mod = obj.modifiers.get(MODIFIER_NAME)
    if mod is not None:
        obj.modifiers.remove(mod)


def suspend_mesh_cache_display(obj):
    """Disable the ContactSolverCache modifier's viewport evaluation and
    return its prior ``show_viewport`` flag (``None`` if absent).

    Capture Deformation records the pure deformer (Armature, Lattice,
    Mesh Deform, shape-key, ...) output by sampling
    ``obj.evaluated_get(depsgraph).to_mesh()``. The ContactSolverCache
    MESH_CACHE modifier runs with ``deform_mode='OVERWRITE'``, so while it
    is enabled the evaluated mesh is the solver's *previous* output, not
    the deformer result. Re-capturing after a run would then record stale
    (often gap-filled) solver positions and feed them back as the next
    input. Capture operators suspend it for the duration of the job and
    restore it via :func:`resume_mesh_cache_display`. The flag is the only
    state touched, and the modifier is addon-owned, so this stays within
    the addon's own state.
    """
    if obj is None:
        return None
    mod = obj.modifiers.get(MODIFIER_NAME)
    if mod is None:
        return None
    prior = bool(mod.show_viewport)
    try:
        mod.show_viewport = False
    except (AttributeError, RuntimeError):
        # Restricted context (e.g. a UI draw() handler): Blender forbids
        # writing to ID data. Callers that might run there gate the heavy
        # path off separately; returning None here means "nothing to
        # restore" so a stray call can't crash.
        return None
    return prior


def resume_mesh_cache_display(obj, prior):
    """Restore the ContactSolverCache ``show_viewport`` flag saved by
    :func:`suspend_mesh_cache_display`. No-op when *prior* is ``None``
    (modifier was absent) or the modifier has since been removed."""
    if prior is None or obj is None:
        return
    mod = obj.modifiers.get(MODIFIER_NAME)
    if mod is not None:
        mod.show_viewport = bool(prior)


def modifiers_above_cache(obj) -> list:
    """Names of modifiers sitting above ContactSolverCache in *obj*'s
    stack. This is exactly the set :func:`strip_modifiers_above_cache`
    would remove on bake finalize; the bake-confirm dialog renders
    this list so the user can see what they'd lose before clicking
    OK. Empty when the cache modifier is at index 0, absent, or
    *obj* is not a MESH (curves and other types are never struck).
    """
    if obj is None or getattr(obj, "type", None) != "MESH":
        return []
    names: list[str] = []
    for m in obj.modifiers:
        if m.name == MODIFIER_NAME:
            return names
        names.append(m.name)
    return []


def strip_modifiers_above_cache(obj):
    """Remove every modifier that sits above ContactSolverCache in the
    stack. Used by Bake operators on STATIC objects whose Case-3
    deformation feeders (Armature, Lattice, MeshDeform, Shape Keys
    via a Shape Keys modifier, ...) would otherwise re-deform the
    baked shape-key data once ContactSolverCache itself is removed.

    No-op when the cache modifier is absent (e.g. already removed)
    or sits at index 0 with nothing above it (dynamics, Case 1, and
    Case 2 stacks all land here, so the helper is safe to call
    unconditionally for any baked MESH).
    """
    for name in modifiers_above_cache(obj):
        mod = obj.modifiers.get(name)
        if mod is not None:
            obj.modifiers.remove(mod)


def has_mesh_cache(obj):
    """Check if *obj* has simulation animation (modifier, curve cache, or PC2 file).

    Reached from read-only contexts (panel draw, operator poll), so it
    uses the side-effect-free :func:`object_pc2_key_readonly`. An object
    that owns a cache always carries a UUID (assigned by the capture path
    or load_post migration), so an empty key means no cache.
    """
    if obj.modifiers.get(MODIFIER_NAME) is not None:
        return True
    key = object_pc2_key_readonly(obj)
    if not key:
        return False
    if obj.type == "CURVE":
        from .curve_output import PROPERTY, detached
        try:
            if detached(obj):
                return False
        except ValueError:
            return True  # Keep damaged references actionable, not silently absent.
        if key in _curve_cache or PROPERTY in obj:
            return True
    # Also check for PC2 file on disk (survives addon reload)
    if os.path.exists(get_pc2_path(key)):
        return True
    return False


def assigned_uuids_in_active_groups(scene) -> "set[str]":
    """UUIDs of every object assigned to an ACTIVE object group.

    This is exactly the set ``core.animation.clear_animation_data`` walks,
    including entries whose ``included`` flag is off, which that function
    clears as well. Reads the stored UUID strings directly instead of
    resolving each entry to its object: ``resolve_assigned`` reconciles
    display names on a rename, which writes RNA, and the callers here run
    from ``poll()`` and panel draw where writes are not permitted.
    """
    from ..models.groups import iterate_active_object_groups

    return _assigned_uuids(scene, iterate_active_object_groups)


def assigned_uuids_in_any_group(scene) -> "set[str]":
    """UUIDs assigned to ANY object group, active or not.

    The complement of this set is what :func:`find_orphan_solver_caches`
    calls an orphan. Deactivating a group must not turn its objects into
    orphans: the user is holding a second setup aside, not abandoning it.
    """
    from ..models.groups import iterate_object_groups

    return _assigned_uuids(scene, iterate_object_groups)


def _assigned_uuids(scene, iterate_groups) -> "set[str]":
    uids: set[str] = set()
    if scene is None:
        return uids
    for group in iterate_groups(scene):
        for assigned in group.assigned_objects:
            uid = getattr(assigned, "uuid", "")
            if uid:
                uids.add(str(uid))
    return uids


def find_orphan_solver_caches(scene=None) -> list:
    """Objects carrying a ``ContactSolverCache`` that no group claims.

    Duplicating a simulated object copies the modifier, and the depsgraph
    handler strips the copy's inherited UUID, so the copy ends up owning a
    cache while belonging to no group at all. Every gate in the addon
    ignores it, by design (see :func:`scene_has_solver_cache`): it must
    never disable Run, and Clear Local Animation must not reach across
    into it. But ignoring it silently leaves an addon-created modifier in
    the user's scene, usually with a filepath that resolves to nothing,
    which Blender itself reports as a modifier error. The panel names what
    this finds and offers to remove it, so the state is visible and
    fixable rather than invisible and permanent.

    Membership is tested against EVERY group, not the active ones, so a
    deactivated setup's caches are never called orphans. An object with no
    UUID at all cannot be assigned to anything, so it is an orphan as soon
    as it carries the modifier.

    Read-only, and called from panel draw: no UUID is stamped, no file is
    touched, and the group set is built only once a modifier is found.
    """
    if scene is None:
        scene = bpy.context.scene
    uids = None
    orphans = []
    for obj in bpy.data.objects:
        if obj.modifiers.get(MODIFIER_NAME) is None:
            continue
        if uids is None:
            uids = assigned_uuids_in_any_group(scene)
        if object_pc2_key_readonly(obj) not in uids:
            orphans.append(obj)
    return orphans


def remove_orphan_solver_cache(obj) -> bool:
    """Strip an orphaned ``ContactSolverCache`` from *obj*, keeping its PC2.

    Returns True when a modifier was actually removed. The visibility
    keyframes go first, as in :func:`cleanup_mesh_cache`, since they name
    the modifier by data path and would linger as broken channels.

    **No PC2 file is deleted, and that is the difference from
    ``cleanup_mesh_cache``.** An orphan is normally a copy of a simulated
    object, so its modifier points at the ORIGINAL's cache: deleting that
    file would take the original's animation with it. An orphan that
    happens to own an unreferenced PC2 leaves it behind on disk instead,
    which costs a file and destroys nothing.
    """
    if obj is None or obj.modifiers.get(MODIFIER_NAME) is None:
        return False
    remove_cache_visibility_keys(obj)
    remove_mesh_cache_modifier(obj)
    return True


def scene_has_solver_cache(scene=None) -> bool:
    """Stateless: True when a solver-managed object carries animation.

    Solver-managed means the set Clear Local Animation actually clears:
    objects assigned to an ACTIVE object group. The scope is the whole
    point. A ``ContactSolverCache`` MESH_CACHE modifier can outlive its
    group membership, because duplicating a simulated object copies the
    modifier onto a copy whose inherited UUID the depsgraph handler then
    strips, leaving the copy in no group at all. Counting that copy would
    report animation that Clear Local Animation cannot reach, and since
    ``SOLVER_OT_Run.poll`` is disabled while this returns True, Run would
    stay greyed out with nothing the user can press to re-enable it.
    Objects in inactive groups are skipped for the same reason: a parallel
    setup the user is not simulating right now must not block this one.

    Curves keep their cache in the in-memory ``_curve_cache`` instead of a
    modifier, keyed by the same UUID, so both paths take the same scope.

    Panel ``poll()`` methods call this on every UI redraw, so the modifier
    scan stays a C-level collection lookup per object and the assigned-UUID
    set is built lazily, only once a candidate modifier is actually found.
    It recomputes every call, so it never goes stale the way a memoized
    result can.
    """
    if scene is None:
        scene = bpy.context.scene
    from .rigid_pose import has_bone_pose_cache
    if has_bone_pose_cache(scene):
        return True
    uids = None
    for obj in bpy.data.objects:
        if obj.modifiers.get(MODIFIER_NAME) is None:
            continue
        if uids is None:
            uids = assigned_uuids_in_active_groups(scene)
            if not uids:
                return False
        if object_pc2_key_readonly(obj) in uids:
            return True
    if not _curve_cache:
        return False
    if uids is None:
        uids = assigned_uuids_in_active_groups(scene)
    return any(key in uids for key in _curve_cache)


def scene_has_static_deform_cache(scene=None) -> bool:
    """Only captures with exact ownership in this scene enable its Clear button."""
    # Skip walking scene objects when the project has no static capture files.
    if not any(key.endswith(STATIC_DEFORM_SUFFIX) for key in _static_deform_cache):
        try:
            with os.scandir(get_pc2_dir()) as entries:
                possible = any(entry.name.casefold().endswith((STATIC_DEFORM_SUFFIX + '.pc2').casefold())
                               for entry in entries)
        except FileNotFoundError:
            return False
        except OSError:
            possible = True  # Inaccessible directories retain the original gate.
        if not possible:
            return False
    return bool(_owned_capture_keys(scene, static_cache_only=True))


def remove_pc2_file(key):
    """Delete the PC2 file for the given UUID key if it exists."""
    path = get_pc2_path(key)
    if os.path.exists(path):
        os.remove(path)
    clear_gap_tracking(key)


def _owned_mesh_output_path(obj, key, mod):
    """Resolve only this object's UUID-named PPF output, including Save As."""
    if not key or any(part in key for part in ("/", "\\")) or getattr(obj, "library", None):
        return None
    if mod is not None and (mod.type != "MESH_CACHE" or mod.cache_format != "PC2"):
        return None
    reference = getattr(mod, "filepath", "") if mod is not None else ""
    path = bpy.path.abspath(reference) if reference else get_pc2_path(key)
    path = os.path.normcase(os.path.realpath(path))
    if os.path.basename(path) != os.path.basename(get_pc2_path(key)):
        return None  # A pin cache or user file is never an output-cache fallback.
    for root in (os.path.realpath(get_pc2_dir()),):
        root = os.path.normcase(root)
        try:
            if os.path.commonpath((root, path)) == root:
                return path
        except ValueError:  # Different Windows drives cannot share an owner root.
            continue
    return None


def _mesh_output_path_shared(path, owner, owned_modifier):
    """Keep files referenced by any other mesh-cache modifier, even user ones."""
    for other in bpy.data.objects:
        for mod in other.modifiers:
            if other == owner and mod == owned_modifier:
                continue
            if mod.type != "MESH_CACHE" or not mod.filepath:
                continue
            reference = bpy.path.abspath(mod.filepath, library=getattr(other, "library", None))
            if os.path.normcase(os.path.realpath(reference)) == path:
                return True
    return False


def mesh_output_matches_session(scene, obj, session_id, *, key=None, mod=None):
    """Read only: prove a local PPF mesh display came from this explicit session."""
    import uuid
    from ..models.groups import has_addon_data

    if key is None:
        key = object_pc2_key_readonly(obj)
    if mod is None:
        mod = obj.modifiers.get(MODIFIER_NAME)
    if (scene is None or obj.type != "MESH" or obj.library or obj.data.library
            or not getattr(obj, "is_editable", True)
            or scene.objects.get(obj.name) != obj
            or mod is None or mod.name != MODIFIER_NAME
            or mod.type != "MESH_CACHE" or mod.cache_format != "PC2"):
        return False
    try:
        if str(uuid.UUID(key)) != key:
            return False
    except (ValueError, TypeError, AttributeError):
        return False
    if any(other != obj and object_pc2_key_readonly(other) == key
           for other in scene.objects):
        return False
    reference = mod.filepath
    if (not has_addon_data(scene) or not reference
            or os.path.basename(bpy.path.abspath(reference)) != key + ".pc2"
            or key not in assigned_uuids_in_active_groups(scene)):
        return False
    if not session_id or obj.get("_solver_session") != session_id:
        return False
    # Bone proxies additionally retain their exact author-side pointer/UUID proof.
    if obj.get("ppf_bone_proxy_owned") or obj.get("ppf_bone_uuid"):
        from ..models.bone_bindings import validate_proxy_owner
        bindings = [(arm, bone) for arm in scene.objects if arm.type == "ARMATURE"
                    for bone in arm.data.bones
                    if bone.ppf_bone_physics.proxy == obj
                    and bone.ppf_bone_physics.body_uuid == key]
        if len(bindings) != 1:
            return False
        try:
            validate_proxy_owner(*bindings[0])
        except ValueError:
            return False
    return True


def _can_detach_external_mesh_output(obj, key, mod):
    """Use this scene's already-saved producer identity; never infer a new one."""
    from ..models.groups import get_addon_data, has_addon_data

    scene = bpy.context.scene
    if scene is None or not has_addon_data(scene):
        return False
    return mesh_output_matches_session(
        scene, obj, get_addon_data(scene).state.last_session_id, key=key, mod=mod,
    )


def _cache_cleanup_plan(obj, *, allow_unproven_mesh=False, _receipt=None):
    """Read-only ownership plan; shared or proven external files are retained."""
    key = object_pc2_key_readonly(obj)
    if obj.type == 'MESH':
        modifier = obj.modifiers.get(MODIFIER_NAME)
        path = _owned_mesh_output_path(obj, key, modifier)
        if path is None:
            if modifier is None or _can_detach_external_mesh_output(obj, key, modifier):
                return {'delete_path': None}
            if allow_unproven_mesh:
                # Default Clear leaves this unproven modifier and file intact.
                return {'delete_path': None}
            raise ValueError(f"Mesh '{obj.name}' cache ownership cannot be proven; cleanup refused")
        return {'delete_path': None if _mesh_output_path_shared(path, obj, modifier) else path}
    if obj.type == 'CURVE':
        if not key or obj.library is not None:
            raise ValueError(f"Curve '{obj.name}' cache ownership cannot be proven; cleanup refused")
        from .curve_output import cleanup_path
        return {'delete_path': cleanup_path(obj, _receipt)}
    return {'delete_path': None}


def preflight_cache_cleanup(obj, *, allow_unproven_mesh=False):
    """Check known unlink refusals without changing files or Blender data.

    This is admission, not a guarantee against later filesystem races. Actual
    unlink still runs and propagates its real error during cleanup.
    """
    plan = _cache_cleanup_plan(obj, allow_unproven_mesh=allow_unproven_mesh)
    path = plan['delete_path']
    if path is None:
        return plan
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        plan['present'] = False
        return plan
    plan['present'] = True
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Cache cleanup requires an ordinary file: {path}")
    parent = os.path.dirname(path)
    if not os.access(parent, os.W_OK | os.X_OK):
        raise PermissionError(errno.EACCES, 'Cache directory does not permit unlink', path)
    return plan


def _unlink_cache_path(path, receipt=None):
    """Unlink first; report the exact irreversible boundary to a Bake caller."""
    try:
        before = os.stat(path)
    except FileNotFoundError:
        if receipt is not None and receipt.get('require_existing'):
            receipt.update(phase='missing_input', input_preserved=False)
            raise FileNotFoundError(errno.ENOENT, 'Bake input disappeared before cleanup', path)
        return False
    if receipt is not None:
        receipt.update(phase='unlink', path=path, input_preserved=False)
    try:
        os.remove(path)
    except PermissionError:
        # Confirm a known denied unlink retained the same input. Other/uncertain
        # outcomes are never treated as a successful bake or a safe rollback.
        if receipt is not None:
            try:
                after = os.stat(path)
                fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
                receipt['input_preserved'] = all(getattr(before, f) == getattr(after, f) for f in fields)
            except OSError:
                pass
        raise
    if receipt is not None:
        receipt.update(disk_removed=True, phase='disk_removed')
    return True


def cleanup_mesh_cache(obj, *, keep_baked_pose: bool = False, strict=False, _receipt=None):
    """Remove modifier/cache and delete PC2 file for *obj*.

    For curves, restores rest-pose CVs (frame 0 from PC2) before deleting,
    since the handler modifies curve data directly. Pass
    ``keep_baked_pose=True`` when the caller has already written baked
    fcurves/positions and wants neither the rest-pose restore nor the
    bezier handle_type restore (AUTO handles would silently overwrite
    the baked handle positions).
    """
    if _receipt is not None:
        _receipt.update(phase='preflight', disk_removed=False, input_preserved=True)
    if obj.type == "CURVE" and not keep_baked_pose:
        preflight_curve_handle_restore(obj)
    if strict:
        _cache_cleanup_plan(obj, _receipt=_receipt)  # Ownership only; do not hide a late unlink error.
    if obj.type == "MESH":
        # Capture the real modifier target before removing it. Save As may keep
        # a valid reference in the previous basename's data folder.
        key = object_pc2_key_readonly(obj)
        mod = obj.modifiers.get(MODIFIER_NAME)
        path = _owned_mesh_output_path(obj, key, mod)
        if path is None:
            # Save As can leave a proven local display bound to the old directory.
            # Detach it locally, while never deleting that external source file.
            if _can_detach_external_mesh_output(obj, key, mod):
                if _receipt is not None:
                    _receipt['phase'] = 'detach'
                remove_cache_visibility_keys(obj)
                remove_mesh_cache_modifier(obj)
                clear_gap_tracking(key)
            return  # Unproven/user modifiers and external files stay intact.
        shared = _mesh_output_path_shared(path, obj, mod)
        if not shared:
            # Preserve the reference on PermissionError so a later Clear can retry;
            # let the existing operator error path report it, never fake success.
            _unlink_cache_path(path, _receipt)
        if _receipt is not None:
            _receipt['phase'] = 'detach'
        remove_cache_visibility_keys(obj)
        remove_mesh_cache_modifier(obj)
        clear_gap_tracking(key)
        return
    key = object_pc2_key_readonly(obj) if strict else object_pc2_key(obj)
    if obj.type == "CURVE":
        from .curve_output import cleanup_path
        path = cleanup_path(obj, _receipt)
        if path is not None:
            _unlink_cache_path(path, _receipt)
    else:
        _unlink_cache_path(get_pc2_path(key), _receipt)
    if _receipt is not None:
        _receipt['phase'] = 'detach'
    # Before the modifier goes: these keys name it by data path, so they would
    # otherwise linger as broken channels on the object.
    remove_cache_visibility_keys(obj)
    remove_mesh_cache_modifier(obj)
    if obj.type == "CURVE":
        if not keep_baked_pose:
            cache = _curve_cache.get(key)
            if cache is not None and len(cache) > 0:
                try:
                    _apply_curve_cvs(obj, 0)
                except CurveCacheValidationError as exc:
                    # A malformed cache must remain removable without applying it.
                    _log_curve_handler_error(str(exc))
            snapshot = preflight_curve_handle_restore(obj)
            if snapshot is None:
                snapshot = _saved_handle_types.get(key)
            _saved_handle_types.pop(key, None)
            _handles_freed.discard(key)
            if snapshot is not None:
                for spline, per_spline in zip(obj.data.splines, snapshot):
                    if spline.type != "BEZIER":
                        continue
                    for bp, (left, right) in zip(spline.bezier_points, per_spline):
                        bp.handle_left_type = left
                        bp.handle_right_type = right
        else:
            _saved_handle_types.pop(key, None)
            _handles_freed.discard(key)
        if _CURVE_HANDLE_TYPES in obj:
            del obj[_CURVE_HANDLE_TYPES]
        from .curve_rod import restore_key_author
        restore_key_author(obj, keep_baked_pose=keep_baked_pose)
        unload_curve_cache(key)
        from .curve_output import mark_detached
        mark_detached(obj)
    clear_gap_tracking(key)


# ---------------------------------------------------------------------------
# Curve CV playback via persistent handlers
# ---------------------------------------------------------------------------

# In-memory cache keyed by object UUID: {uuid: numpy array shape (n_frames, n_cvs, 3)}
_curve_cache: dict[str, numpy.ndarray] = {}


def load_curve_cache(key, *, filepath=None):
    """Load a curve's PC2 file into the in-memory cache (keyed by UUID)."""
    global _curve_last_frame
    path = filepath if filepath is not None else get_pc2_path(key)
    arr = _load_pc2_array(path)
    if arr is not None:
        _curve_cache[key] = arr
        from .curve_output import note_loaded
        note_loaded(key, path)
        # Invalidate the frame-dedup guard. A live append grows this
        # cache mid-tick; if an earlier per-frame scene.frame_set this
        # tick (STATIC per-frame matrices, client.apply_animation) already
        # ran the playback handler against the shorter cache, it stamped
        # _curve_last_frame and clamped the rod a frame back. Clearing the
        # guard lets the tick's final frame_set re-apply with the just-
        # loaded frame so rods stay in lockstep with the meshes' MESH_CACHE.
        _curve_last_frame = -1


def unload_curve_cache(key=None):
    """Remove one or all curves from the in-memory cache."""
    global _curve_last_frame
    if key is None:
        _curve_cache.clear()
        _curve_pending_author_types.clear()
        _handles_freed.clear()
        # Reset the log-once dedup so "log once per message" resets
        # between playback sessions rather than per process.
        _curve_handler_error_reported.clear()
    else:
        _curve_cache.pop(key, None)
        _curve_pending_author_types.pop(key, None)
        _handles_freed.discard(key)
    _curve_last_frame = -1


def has_curve_cache(key):
    """Check if a curve UUID has cached animation data."""
    return key in _curve_cache


# Tracks objects whose handle types have been set to FREE and remembers
# the original (left_type, right_type) per spline→point so
# ``cleanup_mesh_cache`` can restore the user's intent when playback
# ends. Keyed by object UUID.
_handles_freed: set[str] = set()
_saved_handle_types: dict[str, list] = {}


class CurveCacheValidationError(ValueError):
    """Malformed frame rejected before any curve/handle changes."""


_CURVE_HANDLE_TYPES = "_ppf_curve_handle_types_v1"
_HANDLE_TYPES = {"FREE", "VECTOR", "ALIGNED", "AUTO", "AUTO_CLAMPED"}
_curve_pending_author_types = {}


def note_new_curve_output_author(obj):
    """Only an explicit new producer, never cold legacy playback, owns this intent."""
    key = object_pc2_key_readonly(obj)
    if obj.get(_CURVE_HANDLE_TYPES) is not None or key in _curve_cache:
        return
    from .curve_output import PROPERTY
    raw = obj.get(PROPERTY)
    if raw is not None:
        try:
            if json.loads(raw).get("status") != "detached":
                return  # Missing old bound output is not a fresh author cycle.
        except (TypeError, ValueError, AttributeError):
            return
    _curve_pending_author_types[key] = {
        "data": obj.data.as_pointer(), "topology": _curve_handle_topology(obj),
        "types": _curve_current_handle_types(obj),
    }


def _curve_current_handle_types(obj):
    return [[(bp.handle_left_type, bp.handle_right_type) for bp in sp.bezier_points]
            if sp.type == "BEZIER" else [] for sp in obj.data.splines]


def _validate_pending_curve_author(obj):
    pending = _curve_pending_author_types.get(object_pc2_key_readonly(obj))
    if pending is not None and (
            pending["data"] != obj.data.as_pointer()
            or pending["topology"] != _curve_handle_topology(obj)
            or pending["types"] != _curve_current_handle_types(obj)):
        raise CurveCacheValidationError(
            f"Curve '{obj.name}' author data/topology/handle types changed before "
            "first playback; cache and edited intent are retained. Resolve the "
            "edit before playback or Clear.")
    return pending



def _curve_handle_topology(obj):
    return [[s.type, len(s.bezier_points) if s.type == "BEZIER" else len(s.points)]
            for s in obj.data.splines]


def preflight_curve_handle_restore(obj):
    """Validate saved type intent and output-owned FREE state without mutation.

    No metadata means legacy behavior, not proof of the original author types.
    Deliberate type/topology edits are never replaced by an older snapshot.
    """
    from .curve_rod import author_curve_data
    author_curve_data(obj)
    raw = obj.get(_CURVE_HANDLE_TYPES)
    if raw is None:
        _validate_pending_curve_author(obj)
        return None
    try:
        saved = json.loads(raw)
        if (saved.get("version") != 1
                or saved.get("key") != object_pc2_key_readonly(obj)
                or saved.get("topology") != _curve_handle_topology(obj)):
            raise ValueError("identity or topology changed")
        types = saved["types"]
        if len(types) != len(obj.data.splines):
            raise ValueError("spline count changed")
        for spline, pairs in zip(obj.data.splines, types):
            points = spline.bezier_points if spline.type == "BEZIER" else []
            if len(pairs) != len(points):
                raise ValueError("point count changed")
            for bp, pair in zip(points, pairs):
                if (not isinstance(pair, list) or len(pair) != 2
                        or any(t not in _HANDLE_TYPES for t in pair)):
                    raise ValueError("invalid handle type")
                if bp.handle_left_type != "FREE" or bp.handle_right_type != "FREE":
                    raise ValueError("handle types were edited after playback")
        return types
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        raise CurveCacheValidationError(
            f"Curve '{obj.name}' saved handle-type state is incompatible: {exc}; "
            "preserve the cache and resolve the author edit before Clear/playback") from exc


def _apply_curve_cvs(obj, frame_idx, key=None):
    """Set curve CVs from cached data for the given frame index.

    When ``key`` is given (the playback handler already resolved it via
    the read-only accessor), reuse it. When ``None`` (the
    ``cleanup_mesh_cache`` operator path), fall back to the side-effecting
    :func:`object_pc2_key` so the legacy ``<name>.pc2`` migration still
    runs in that write-permitted context.
    """
    if key is None:
        key = object_pc2_key(obj)
    cache = _curve_cache.get(key)
    if cache is None:
        return
    expected = sum(3 * len(spline.bezier_points) if spline.type == "BEZIER"
                   else len(spline.points) for spline in obj.data.splines)
    if cache.ndim != 3 or cache.shape[0] < 1 or cache.shape[1:] != (expected, 3):
        raise CurveCacheValidationError(
            f"Curve '{obj.name}' cache shape {cache.shape} does not match {expected} CVs")
    n_frames = cache.shape[0]
    idx = max(0, min(frame_idx, n_frames - 1))
    cv_data = cache[idx]
    if not numpy.isfinite(cv_data).all():
        raise CurveCacheValidationError(f"Curve '{obj.name}' cache has non-finite coordinates")
    from .curve_rod import bind_key_free_output
    preflight_curve_handle_restore(obj)
    bind_key_free_output(obj)
    pending = _curve_pending_author_types.get(key)
    if pending is not None:
        pending["data"] = obj.data.as_pointer()
    # Persist original type intent before the first destructive FREE conversion.
    # Existing metadata survives save/reopen; never recapture from played RNA.
    persistent = preflight_curve_handle_restore(obj)
    if key not in _handles_freed:
        snapshot = persistent
        if snapshot is None:
            snapshot = [[(bp.handle_left_type, bp.handle_right_type)
                         for bp in spline.bezier_points]
                        if spline.type == "BEZIER" else []
                        for spline in obj.data.splines]
            known = _curve_pending_author_types.pop(key, None)
            if known is not None and any(known["types"]):
                snapshot = known["types"]
                obj[_CURVE_HANDLE_TYPES] = json.dumps({
                    "version": 1, "key": key,
                    "topology": _curve_handle_topology(obj), "types": snapshot,
                }, separators=(",", ":"))
        for spline in obj.data.splines:
            if spline.type == "BEZIER":
                for bp in spline.bezier_points:
                    bp.handle_left_type = "FREE"
                    bp.handle_right_type = "FREE"
        _saved_handle_types[key] = snapshot
        _handles_freed.add(key)
    cv_i = 0
    for spline in obj.data.splines:
        if spline.type == "BEZIER":
            for bp in spline.bezier_points:
                bp.handle_left = cv_data[cv_i]
                cv_i += 1
                bp.co = cv_data[cv_i]
                cv_i += 1
                bp.handle_right = cv_data[cv_i]
                cv_i += 1
        else:
            for pt in spline.points:
                c = cv_data[cv_i]
                pt.co[0] = c[0]
                pt.co[1] = c[1]
                pt.co[2] = c[2]
                cv_i += 1


_curve_last_frame = -1  # Last (Scene pointer, frame, start frame), or invalid sentinel.


def _scene_curve_objects(scene):
    """Playback writes datablocks: refuse objects/data shared outside this owner."""
    from .utils import find_linked_duplicate_siblings
    for obj in scene.objects:
        if obj.type != "CURVE":
            continue
        if len(obj.users_scene) > 1 or find_linked_duplicate_siblings(obj):
            _log_curve_handler_error(f"Curve '{obj.name}' playback refused: shared Scene or curve data")
            continue
        yield obj


def _ensure_curve_caches(scene=None):
    """Lazy-load curve caches from PC2 files on disk.

    Handles file reopen / addon reload: the PC2 files persist but the
    in-memory cache is lost.  Scans curve objects and loads any PC2
    files that aren't already cached.
    """
    scene = scene or bpy.context.scene
    for obj in _scene_curve_objects(scene):
        key = object_pc2_key_readonly(obj)
        if not key or key in _curve_cache:
            continue
        try:
            from .curve_output import resolve_read_path
            path = resolve_read_path(obj)
            if path and os.path.exists(path):
                load_curve_cache(key, filepath=path)
        except (OSError, ValueError, struct.error) as exc:
            _log_curve_handler_error(f"Curve '{obj.name}' cache could not be loaded: {exc}")
            continue


def _apply_curves_at_current_frame(scene=None):
    """Apply cached curve CVs for the current scene frame.

    Lazy-loads caches from PC2 files if needed (file reopen / addon reload).
    Tracks last-applied frame to avoid redundant work.
    """
    global _curve_last_frame
    try:
        scene = scene or bpy.context.scene
        from .full_bake_restore import KEY as full_restore_key
        full_targets={row['obj'].as_pointer() for row in scene.get(full_restore_key,{}).get('items',[]) if row.get('obj') is not None}
        _ensure_curve_caches(scene)
        if not _curve_cache:
            return
        current = scene.frame_current
        from .playback_origin import curve_start_frame, legacy_curve_start_frame
        legacy_origin = None
        signature = (scene.as_pointer(), current)
        if signature == _curve_last_frame:
            return
        _curve_last_frame = signature
        for obj in _scene_curve_objects(scene):
            if obj.as_pointer() in full_targets:continue
            from .curve_output import detached
            if detached(obj):continue
            # Read-only key: this runs from frame_change_post, so it must
            # not stamp a UUID on an uncached curve or migrate any file.
            key = object_pc2_key_readonly(obj)
            if key and key in _curve_cache:
                try:
                    origin = curve_start_frame(obj, legacy_origin)
                    if origin is None:
                        legacy_origin = origin = legacy_curve_start_frame(scene)
                    _apply_curve_cvs(obj, current - origin, key=key)
                except CurveCacheValidationError as exc:
                    _log_curve_handler_error(str(exc))
                    continue  # One bad curve must not freeze valid siblings.
    except Exception as e:
        # frame_change handlers run VERY frequently — log once per
        # message rather than every tick, but don't silence entirely.
        _log_curve_handler_error(str(e))


_curve_handler_error_reported: set[str] = set()


def _log_curve_handler_error(msg: str) -> None:
    if msg in _curve_handler_error_reported:
        return
    _curve_handler_error_reported.add(msg)
    try:
        from ..models.console import console
        console.write(f"[curve playback] {msg}")
    except Exception:
        pass


def refresh_curves_at_current_frame(scene=None):
    """Force curve CV playback to the current frame, bypassing the
    per-frame dedup guard.

    Called at the end of an apply tick (``client.apply_animation``): the
    frame_change handler may have already run earlier in the tick (e.g. a
    per-frame ``scene.frame_set`` for STATIC matrices) against a curve
    cache that was still a frame short, and its ``_curve_last_frame``
    guard would otherwise suppress the corrected re-apply when the tick's
    final ``frame_set`` lands on the same frame. This guarantees rods sit
    on the same frame as the meshes' MESH_CACHE.
    """
    global _curve_last_frame
    _curve_last_frame = -1
    _apply_curves_at_current_frame(scene)


@bpy.app.handlers.persistent
def curve_frame_change_handler(scene=None, *_args):
    """``frame_change_post`` handler for the Scene Blender actually evaluated."""
    _apply_curves_at_current_frame(scene)


@bpy.app.handlers.persistent
def clear_pc2_caches_before_load(*_args):
    """UUIDs can repeat across files; PC2 memory and handle snapshots cannot."""
    unload_curve_cache()
    from .curve_output import clear_runtime
    clear_runtime()
    _saved_handle_types.clear()
    _curve_pending_author_types.clear()
    unload_static_deform_cache()
    unload_pin_anim_cache()
    _real_frames.clear()


@bpy.app.handlers.persistent
def migrate_curve_caches_on_load(*_args):
    """Migrate name-keyed legacy files only in a write-permitted load context."""
    global _capture_project_dir
    _capture_project_dir = get_pc2_dir()
    base = os.path.normcase(os.path.realpath(os.path.dirname(get_pc2_path("_curve_probe"))))
    for scene in bpy.data.scenes:
        for obj in _scene_curve_objects(scene):
            legacy = get_pc2_path(obj.name)
            parent = os.path.normcase(os.path.realpath(os.path.dirname(legacy)))
            if parent != base or not os.path.isfile(legacy):
                continue
            try:
                object_pc2_key(obj)
            except (OSError, ValueError, RuntimeError) as exc:
                _log_curve_handler_error(f"Curve '{obj.name}' legacy cache migration failed: {exc}")


def _migrate_curve_caches_after_register():
    # Extension registration exposes _RestrictData; wait for ordinary RNA access.
    migrate_curve_caches_on_load()
    return None


def ensure_curve_handler():
    """Register frame_change_post handler for curve CV playback.

    Detects stale handlers by __name__ (not identity) so a reloaded
    module generation isn't double-added next to the old function
    object from the previous generation.
    """
    from .curve_output import prepare_curve_outputs_before_save
    if not any(getattr(h, "__name__", "") == "prepare_curve_outputs_before_save"
               for h in bpy.app.handlers.save_pre):
        bpy.app.handlers.save_pre.append(prepare_curve_outputs_before_save)
    fc = bpy.app.handlers.frame_change_post
    if not any(
        getattr(h, "__name__", "") == "curve_frame_change_handler"
        for h in fc
    ):
        fc.append(curve_frame_change_handler)
        clear_pc2_caches_before_load()
        if not bpy.app.timers.is_registered(_migrate_curve_caches_after_register):
            bpy.app.timers.register(_migrate_curve_caches_after_register, first_interval=0.0)
    pre = bpy.app.handlers.load_pre
    if not any(getattr(h, "__name__", "") == "clear_pc2_caches_before_load" for h in pre):
        pre.append(clear_pc2_caches_before_load)
    lp = bpy.app.handlers.load_post
    if not any(getattr(h, "__name__", "") == "migrate_curve_caches_on_load" for h in lp):
        lp.append(migrate_curve_caches_on_load)


def remove_curve_handler():
    """Unregister every handler named ``curve_frame_change_handler``
    from ``frame_change_post``, regardless of module generation."""
    fc = bpy.app.handlers.frame_change_post
    for h in list(fc):
        if getattr(h, "__name__", "") == "curve_frame_change_handler":
            fc.remove(h)
    for h in list(bpy.app.handlers.load_pre):
        if getattr(h, "__name__", "") == "clear_pc2_caches_before_load":
            bpy.app.handlers.load_pre.remove(h)
    for h in list(bpy.app.handlers.load_post):
        if getattr(h, "__name__", "") == "migrate_curve_caches_on_load":
            bpy.app.handlers.load_post.remove(h)
    for h in list(bpy.app.handlers.save_pre):
        if getattr(h, "__name__", "") == "prepare_curve_outputs_before_save":
            bpy.app.handlers.save_pre.remove(h)
    if bpy.app.timers.is_registered(_migrate_curve_caches_after_register):
        bpy.app.timers.unregister(_migrate_curve_caches_after_register)
    # Runs on addon reload (see __init__.py), giving a clean per-reload
    # reset of the curve-handler log-once dedup.
    _curve_handler_error_reported.clear()



# ---------------------------------------------------------------------------
# Static-deform mesh cache (deforming STATIC collider input)
#
# A STATIC mesh whose modifier stack deforms vertices (Armature,
# MeshDeform, Lattice, shape keys, ...) carries a per-frame absolute
# vertex buffer captured from Blender's depsgraph. Cache layout matches
# the pin-input PC2 (PointCache2 binary, dict[key]->ndarray), but the
# data is solver-input only: the depsgraph already paints the deformed
# mesh in the viewport, so this namespace registers NO frame_change
# handler. Suffix ``_staticdeform`` keeps these keys isolated from
# pin-input PC2 (``_pininput``) so an object moving between groups
# can't inherit a stale animation from the wrong subsystem.
# ---------------------------------------------------------------------------

_static_deform_cache: dict[str, numpy.ndarray] = {}
# Extension import runs with _RestrictData. Origin is established only by
# deferred registration/load or a completed capture, never guessed on import.
_capture_project_dir = None


def _write_captured_input(obj, key, frames, memory, proof=None, scene=None):
    """Commit captured bytes/proof before exposing the new memory generation."""
    from . import capture_provenance as provenance
    scene = scene or bpy.context.scene
    arr = numpy.ascontiguousarray(frames, dtype="<f")
    old_memory = memory.get(key)
    owners = [scene]
    if len(obj.users_scene) == 1 and obj.users_scene[0] != scene:
        owners.append(obj.users_scene[0])
    snapshots = [(owner, name, name in owner, owner.get(name))
                 for owner in owners for name in (provenance.BINDINGS, _CAPTURE_OWNERS)]
    def publish(record):
        provenance.bind(scene, key, record)
        _record_capture_owner(obj, key)
        memory[key] = arr.copy()
    def rollback():
        for owner, name, present, value in snapshots:
            if present:
                owner[name] = value
            elif name in owner:
                del owner[name]
        if old_memory is None:
            memory.pop(key, None)
        else:
            memory[key] = old_memory
    provenance.commit_capture(get_pc2_path(key), arr, proof, publish=publish, rollback=rollback)
    global _capture_project_dir
    _capture_project_dir = get_pc2_dir()



def static_deform_pc2_key(obj) -> str:
    """Return the UUID-backed key for an object's static-deform PC2 file."""
    from .uuid_registry import get_or_create_object_uuid
    return get_or_create_object_uuid(obj) + STATIC_DEFORM_SUFFIX


def write_static_deform_pc2(obj, frames, *, proof=None, scene=None):
    """Write an object's static-deform cache to PC2 and stash it in memory.

    Args:
        obj: The mesh object.
        frames: ndarray ``(n_frames, n_verts, 3)`` of solver-world-space
            positions; index 0 maps to ``scene.frame_start``.
    """
    key = static_deform_pc2_key(obj)
    _write_captured_input(obj, key, frames, _static_deform_cache, proof, scene)


def load_static_deform_cache(key):
    """Load a static-deform PC2 file into the in-memory cache."""
    arr = _load_pc2_array(get_pc2_path(key))
    if arr is not None:
        _static_deform_cache[key] = arr


def unload_static_deform_cache(key=None):
    """Remove one or all objects from the static-deform cache."""
    if key is None:
        _static_deform_cache.clear()
    else:
        _static_deform_cache.pop(key, None)


def has_static_deform_cache_key(key):
    """Check if a static-deform key has cached data in memory."""
    return key in _static_deform_cache


def has_static_deform_animation(obj):
    """True if *obj* has a static-deform PC2 (in memory or on disk)."""
    if obj is None or getattr(obj, "type", None) != "MESH":
        return False
    key = static_deform_pc2_key(obj)
    if key in _static_deform_cache:
        return True
    return os.path.exists(get_pc2_path(key))


def get_static_deform_cache(obj):
    """Return the cached ``(n_frames, n_verts, 3)`` array for *obj*.

    Lazy-loads from the PC2 file on disk (file reopen / addon reload).
    Returns ``None`` when the object has no static-deform cache.
    """
    key = static_deform_pc2_key(obj)
    if key not in _static_deform_cache:
        load_static_deform_cache(key)
    return _static_deform_cache.get(key)


_CAPTURE_OWNERS = 'ppf_deform_capture_owners_v1'
_CAPTURE_SUFFIXES = (STATIC_DEFORM_SUFFIX, PIN_DEFORM_SUFFIX, PIN_DEFORM_V2_SUFFIX)


def _capture_scene(scene=None):
    return scene if scene is not None else (bpy.context.scene if bpy is not None else None)


def _safe_capture_key(key):
    return (isinstance(key, str) and bool(key) and key not in ('.', '..')
            and not any(char in key for char in ('/', '\\', ':', '\0'))
            and key.endswith(_CAPTURE_SUFFIXES)
            and os.path.basename(get_pc2_path(key)) == key + '.pc2')


def _capture_records(scene, *, static_only=False):
    if scene is None:
        return {}
    try:
        value = json.loads(scene.get(_CAPTURE_OWNERS, '{}'))
    except (TypeError, ValueError):
        return {}
    return {key: uid for key, uid in value.items()
            if (not static_only or key.endswith(STATIC_DEFORM_SUFFIX))
            and _safe_capture_key(key) and isinstance(uid, str) and uid} if isinstance(value, dict) else {}


def _live_capture_keys(obj, *, static_only=False):
    """Read-only full keys from the real UUID and actual vertex-group names."""
    from .uuid_registry import get_object_uuid
    uid = get_object_uuid(obj)
    if not uid or obj.type != 'MESH':
        return {}
    keys = {uid + STATIC_DEFORM_SUFFIX: uid}
    if static_only:
        return keys
    legacy_names = [vg.name.replace(' ', '_').replace('/', '_') for vg in obj.vertex_groups]
    # Call-local counts preserve case-insensitive legacy ambiguity rejection.
    from collections import Counter
    legacy_counts = Counter(name.casefold() for name in legacy_names)
    for vg, legacy in zip(obj.vertex_groups, legacy_names):
        keys[f'{uid}__{_safe_vg_for_key(vg.name)}{PIN_DEFORM_V2_SUFFIX}'] = uid
        if legacy_counts[legacy.casefold()] == 1:
            key = f'{uid}__{legacy}{PIN_DEFORM_SUFFIX}'
            if _safe_capture_key(key):
                keys[key] = uid
    return keys


def _exclusive_capture_object(scene, obj, uid):
    from .uuid_registry import get_object_uuid
    return (obj is not None and scene.objects.get(obj.name) == obj
            and len(obj.users_scene) == 1 and not obj.library
            and not getattr(obj.data, 'library', None) and getattr(obj, 'is_editable', True)
            and sum(get_object_uuid(other) == uid for other in bpy.data.objects) == 1)


def _record_capture_owner(obj, key):
    if bpy is None or not _safe_capture_key(key) or len(obj.users_scene) != 1:
        return
    from .uuid_registry import get_object_uuid
    scene = obj.users_scene[0]
    uid = get_object_uuid(obj)
    if not _exclusive_capture_object(scene, obj, uid):
        return
    records = _capture_records(scene)
    records[key] = uid
    scene[_CAPTURE_OWNERS] = json.dumps(records, sort_keys=True)


def _owned_capture_keys(scene=None, *, static_cache_only=False):
    """Same authority for saved/unsaved projects and disk/memory cleanup.

    New captures retain their exact owner on the scene, including after object
    deletion. Old live captures resolve from full real UUID/VG keys. Unknown
    orphans, copied scene records, shared objects and duplicate UUIDs stay put.
    """
    scene = _capture_scene(scene)
    if scene is None or bpy is None:
        return set()
    from .uuid_registry import get_object_uuid
    candidates = _capture_records(scene, static_only=static_cache_only)
    for obj in scene.objects:
        for key, uid in _live_capture_keys(obj, static_only=static_cache_only).items():
            if key in candidates and candidates[key] != uid:
                candidates[key] = None
            else:
                candidates[key] = uid
    if static_cache_only:
        # This runs from an operator poll during redraw. Restrict the ownership
        # checks to present static files or arrays before resolving paths.
        candidates = {
            key: uid for key, uid in candidates.items()
            if uid and _safe_capture_key(key)
            and (key in _static_deform_cache or os.path.exists(get_pc2_path(key)))
        }
        if not candidates:
            return set()
    foreign = set()
    for other_scene in bpy.data.scenes:
        if other_scene == scene:
            continue
        foreign.update(_capture_records(other_scene, static_only=static_cache_only))
        for obj in other_scene.objects:
            foreign.update(_live_capture_keys(obj, static_only=static_cache_only))
    # Rebuild all indexes for THIS synchronous read. No cross-draw cache and no
    # skipped ownership gate: duplicate UUIDs, linked/shared objects, foreign
    # scenes, recorded orphans, and every mesh-cache path retain their meaning.
    candidate_uids = set(candidates.values())
    owners_by_uid = {}
    referenced_paths = set()
    for obj in bpy.data.objects:
        uid = get_object_uuid(obj)
        if uid in candidate_uids:
            owners_by_uid.setdefault(uid, []).append(obj)
        for mod in obj.modifiers:
            # Original call is _mesh_output_path_shared(path, owner, None):
            # no real modifier can be the exempt None, including owner's own.
            if mod.type == "MESH_CACHE" and mod.filepath:
                reference = bpy.path.abspath(mod.filepath, library=getattr(obj, "library", None))
                referenced_paths.add(os.path.normcase(os.path.realpath(reference)))
    eligible = {}
    for uid, owners in owners_by_uid.items():
        obj = owners[0]
        eligible[uid] = (len(owners) == 1
                        and scene.objects.get(obj.name) == obj
                        and len(obj.users_scene) == 1 and not obj.library
                        and not getattr(obj.data, 'library', None)
                        and getattr(obj, 'is_editable', True))
    owned = set()
    for key, uid in candidates.items():
        if not uid or not _safe_capture_key(key) or key in foreign:
            continue
        owners = owners_by_uid.get(uid, ())
        if owners and not eligible[uid]:
            continue
        path = os.path.normcase(os.path.realpath(get_pc2_path(key)))
        if path in referenced_paths:
            continue
        owned.add(key)
    return owned


def _clear_capture_keys(keys, scene=None, *, include_missing=False):
    scene = _capture_scene(scene)
    allowed = _owned_capture_keys(scene)
    records = _capture_records(scene)
    original_records = dict(records)
    cleared = []
    for key in set(keys) & allowed:
        path = get_pc2_path(key)
        existed = os.path.exists(path) or key in _static_deform_cache or key in _pin_anim_cache
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            continue  # Keep memory and provenance when the source could not be removed.
        from . import capture_provenance as provenance
        try:
            for suffix in (provenance.PROOF_SUFFIX, provenance.PENDING_SUFFIX):
                if os.path.exists(path + suffix):
                    os.remove(path + suffix)
        except OSError as exc:
            raise RuntimeError("Capture data removed but proof cleanup failed; retained metadata needs inspection") from exc
        unload_static_deform_cache(key)
        unload_pin_anim_cache(key)
        provenance.bind(scene, key, None)
        records.pop(key, None)
        if existed or include_missing:
            cleared.append(key)
    if scene is not None and records != original_records:
        scene[_CAPTURE_OWNERS] = json.dumps(records, sort_keys=True)
    return cleared


def remove_static_deform_pc2(obj, scene=None):
    from .uuid_registry import get_object_uuid
    uid = get_object_uuid(obj)
    return bool(uid and _clear_capture_keys([uid + STATIC_DEFORM_SUFFIX], scene))


def _clear_all_by_suffix(file_suffix, scene=None):
    suffixes = (file_suffix,) if isinstance(file_suffix, str) else file_suffix
    keys = [key for key in _owned_capture_keys(scene)
            if (key + '.pc2').endswith(suffixes)]
    return _clear_capture_keys(keys, scene)


def _capture_file_belongs_to_current_project(filename):
    """Compatibility predicate: a full key is required even in a saved blend."""
    return (filename.endswith('.pc2') and filename[:-4] in _owned_capture_keys())


def clear_all_static_deform_animation(scene=None):
    return len(_clear_all_by_suffix(STATIC_DEFORM_SUFFIX + '.pc2', scene))


def count_deform_cache_files(scene=None) -> int:
    return sum(os.path.exists(get_pc2_path(key)) or key in _static_deform_cache
               or key in _pin_anim_cache for key in _owned_capture_keys(scene))


def clear_orphan_deform_caches(scene=None, *, exclude_keys=()) -> int:
    """Clear only this scene's proven captures, including memory-only/orphaned ones."""
    return len(_clear_capture_keys(_owned_capture_keys(scene) - set(exclude_keys), scene))


# ---------------------------------------------------------------------------
# Pin-deform animation cache (Capture Deformation on SHELL/SOLID/ROD pins)
#
# A pin whose vertices move because of a deformer on the cloth mesh
# (Armature, Lattice, MeshDeform, Hook, shape keys, drivers, ...)
# carries a per-frame absolute vertex buffer captured from the
# depsgraph. The encoder reads this in preference to the manual
# vertex-co fcurves the Make Keyframe button writes (implicit
# PC2-wins). Suffix ``_pindeform`` keeps these keys isolated from
# the STATIC ``_staticdeform`` namespace.
#
# Keyed per (object, vertex-group) so two pins on the same mesh
# capture independently. The cache stores ONLY the pin's vertices,
# in the same order as ``_get_pin_indices`` returns at capture
# time, in solver world space (z2y @ matrix_world @ co_local). The
# decoder consumes consecutive frames as MoveBy deltas; absolute
# space cancels out, but world space matches what the cloth mesh's
# initial-upload coords use, so frame-0 alignment is exact.
# ---------------------------------------------------------------------------

_pin_anim_cache: dict[str, numpy.ndarray] = {}


def _safe_vg_for_key(vg_name: str) -> str:
    """Make a vertex-group name filesystem-safe for use in a PC2 key.

    UTF-8 hex preserves spaces, separators, punctuation and letter case even
    on case-insensitive filesystems. The v2 prefix separates this namespace
    from legacy lossy names; pin_anim_pc2_key may still read an unambiguous
    existing legacy capture without moving its source.
    """
    # Encode every UTF-8 byte: Windows filenames are case-insensitive, so
    # percent-encoding only punctuation would still merge pins A and a.
    return 'vg2-' + vg_name.encode('utf-8').hex()


def pin_anim_pc2_key(obj, vg_name: str) -> str:
    """Return the UUID-backed key for a pin's depsgraph-captured cache."""
    from .uuid_registry import get_or_create_object_uuid
    uid = get_or_create_object_uuid(obj)
    key = f"{uid}__{_safe_vg_for_key(vg_name)}{PIN_DEFORM_V2_SUFFIX}"
    if os.path.exists(get_pc2_path(key)):
        return key
    legacy_name = vg_name.replace(' ', '_').replace('/', '_')
    legacy = f"{uid}__{legacy_name}{PIN_DEFORM_SUFFIX}"
    # Read existing unambiguous v1 captures without renaming/deleting a source.
    # Colliding old names cannot prove which pin produced that file; recapture
    # writes each pin's independent v2 key and leaves the unknown source alone.
    same_names = sum(group.name.replace(' ', '_').replace('/', '_').casefold() == legacy_name.casefold()
                     for group in obj.vertex_groups)
    if ('\\' not in legacy_name and same_names == 1
            and (legacy in _pin_anim_cache or os.path.exists(get_pc2_path(legacy)))):
        return legacy
    return key


def write_pin_anim_pc2(obj, vg_name: str, frames, *, proof=None, scene=None):
    """Write a pin's captured per-frame positions to PC2 and the in-memory cache.

    Args:
        obj: The mesh object the pin belongs to.
        vg_name: The vertex group name identifying the pin.
        frames: ndarray ``(n_frames, n_pin_verts, 3)`` of solver-world-space
            positions; index 0 maps to ``scene.frame_start``. Vertex
            order matches ``_get_pin_indices(obj, vg_name)`` at capture
            time.
    """
    key = pin_anim_pc2_key(obj, vg_name)
    _write_captured_input(obj, key, frames, _pin_anim_cache, proof, scene)


def load_pin_anim_cache(key):
    """Load a pin-deform PC2 file into the in-memory cache (no-op if missing)."""
    arr = _load_pc2_array(get_pc2_path(key))
    if arr is not None:
        _pin_anim_cache[key] = arr


def unload_pin_anim_cache(key=None):
    """Drop one or all pin-deform entries from the in-memory cache."""
    if key is None:
        _pin_anim_cache.clear()
    else:
        _pin_anim_cache.pop(key, None)


def has_pin_anim_pc2(obj, vg_name: str) -> bool:
    """True if a pin-deform PC2 exists (in memory or on disk)."""
    if obj is None or getattr(obj, "type", None) != "MESH":
        return False
    key = pin_anim_pc2_key(obj, vg_name)
    if key in _pin_anim_cache:
        return True
    return os.path.exists(get_pc2_path(key))


def get_pin_anim_cache(obj, vg_name: str):
    """Return ``(n_frames, n_pin_verts, 3)`` for a pin, lazy-loading from disk."""
    key = pin_anim_pc2_key(obj, vg_name)
    if key not in _pin_anim_cache:
        load_pin_anim_cache(key)
    return _pin_anim_cache.get(key)


def peek_pin_anim_frame_count(obj, vg_name: str) -> int:
    """Read the displayed count without loading the pin's vertex samples."""
    key = pin_anim_pc2_key(obj, vg_name)
    cache = _pin_anim_cache.get(key)
    if cache is not None:
        return int(cache.shape[0])
    try:
        with open(get_pc2_path(key), "rb") as handle:
            header = handle.read(PC2_HEADER_SIZE)
            if len(header) != PC2_HEADER_SIZE:
                return 0
            magic, version, vertices, _, _, frames = struct.unpack("<12siiffi", header)
            if magic != PC2_HEADER or version != PC2_VERSION or vertices < 1 or frames < 1:
                return 0
            # Match the array loader's expected payload length. Stat the open
            # file so a concurrent replacement cannot supply its size.
            expected_size = PC2_HEADER_SIZE + frames * vertices * FRAME_VERTEX_SIZE
            return frames if os.fstat(handle.fileno()).st_size == expected_size else 0
    except OSError:
        return 0


def remove_pin_anim_pc2(obj, vg_name: str, scene=None, *, allow_missing=False):
    scene = _capture_scene(scene)
    from .uuid_registry import get_object_uuid
    uid = get_object_uuid(obj)
    if scene is None or not uid or not _exclusive_capture_object(scene, obj, uid):
        return False
    # Metadata-only cleanup is successful only after the same full ownership guard.
    return bool(_clear_capture_keys([pin_anim_pc2_key(obj, vg_name)], scene,
                                    include_missing=allow_missing))


def clear_all_pin_anim_animation(scene=None):
    return len(_clear_all_by_suffix((PIN_DEFORM_SUFFIX + '.pc2',
                                     PIN_DEFORM_V2_SUFFIX + '.pc2'), scene))


# ---------------------------------------------------------------------------
# Save As / Save Incremental processing and first-save migration
# ---------------------------------------------------------------------------


_mesh_cache_save = None


def _report_cache_save_error(exc):
    from ..models.console import console

    console.write(f"[PPF cache] Could not fork mesh animation; original cache kept: {exc}")


@bpy.app.handlers.persistent
def prepare_mesh_cache_save(filepath, *_args):
    """Copy cache files and install their references before .blend serialization."""
    global _mesh_cache_save
    previous = bpy.data.filepath
    if not previous or not filepath or os.path.normcase(previous) == os.path.normcase(filepath):
        return
    from .pc2_migration import copy_files_noreplace
    from .single_bake_restore import KEY as single_bake_key
    from .full_bake_restore import KEY as full_bake_key

    previous_dir = os.path.dirname(previous)
    next_dir = os.path.dirname(filepath)
    target = os.path.join(next_dir, "data", os.path.splitext(os.path.basename(filepath))[0])
    same_directory = os.path.normcase(previous_dir) == os.path.normcase(next_dir)
    source_data = os.path.normcase(os.path.realpath(os.path.join(previous_dir, "data")))
    recovery_objects = {
        row["obj"].as_pointer()
        for scene in bpy.data.scenes
        for key in (single_bake_key, full_bake_key)
        for row in scene.get(key, {}).get("items", ())
        if row.get("obj") is not None
    }
    paths = []
    copies = {}
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.library or obj.override_library:
            continue
        mod = obj.modifiers.get(MODIFIER_NAME)
        if mod is None or mod.type != "MESH_CACHE" or mod.cache_format != "PC2" or not mod.filepath:
            continue
        original = mod.filepath
        source = os.path.abspath(bpy.path.abspath(original))
        owned_name = os.path.basename(get_pc2_path(object_pc2_key_readonly(obj)))
        if source.lower().endswith(".pc2") and os.path.basename(source) == owned_name:
            under_data = os.path.normcase(os.path.realpath(source)).startswith(source_data + os.sep)
            if under_data and os.path.isfile(source) and obj.as_pointer() not in recovery_objects:
                destination = os.path.join(target, owned_name)
                if os.path.normcase(source) != os.path.normcase(destination):
                    copies.setdefault(os.path.dirname(source), set()).add(owned_name)
                new_path = bpy.path.relpath(destination) if same_directory else destination
                paths.append((mod, original, source, new_path))
                continue
        # A recovery record or an external reference still belongs to its owner.
        # Write a stable path when changing directories, even with remapping off.
        if not same_directory and original.startswith("//"):
            paths.append((mod, original, source, source))

    if not paths:
        return

    added = []
    _mesh_cache_save = (filepath, paths, added)
    try:
        for source_dir, names in copies.items():
            added.extend(copy_files_noreplace(source_dir, target, names))
        for mod, _original, _source, new_path in paths:
            mod.filepath = new_path
    except (OSError, ValueError, RuntimeError) as exc:
        _rollback_mesh_cache_files(added)
        for mod, original, source, _new_path in paths:
            mod.filepath = original if same_directory else source
        _report_cache_save_error(exc)


def _rollback_mesh_cache_files(added):
    from .pc2_migration import discard_cache_copy

    while added:
        try:
            discard_cache_copy(added.pop())
        except OSError as exc:
            _report_cache_save_error(exc)


@bpy.app.handlers.persistent
def rollback_mesh_cache_save(*_args):
    global _mesh_cache_save
    transaction, _mesh_cache_save = _mesh_cache_save, None
    if transaction is not None:
        _path, paths, added = transaction
        for mod, original, _source, _new_path in paths:
            mod.filepath = original
        _rollback_mesh_cache_files(added)


@bpy.app.handlers.persistent
def migrate_pc2_on_save(*_args):
    """Move PC2 files from temp dir to 'data/' next to .blend after first save.

    Called from a ``bpy.app.handlers.save_post`` handler.
    """
    import shutil
    from .statistics_cache import migrate_statistics_on_save

    from .uuid_registry import get_object_uuid

    blend_path = bpy.data.filepath
    global _mesh_cache_save
    transaction, _mesh_cache_save = _mesh_cache_save, None
    saved_path = _args[0] if _args and isinstance(_args[0], str) else blend_path
    if saved_path and os.path.normcase(saved_path) != os.path.normcase(blend_path):
        if transaction is not None:
            _path, paths, _added = transaction
            for mod, original, _source, _new_path in paths:
                mod.filepath = original
        return  # Save Copy: leave the current project's cache state in place.
    if not blend_path:
        return
    migrate_statistics_on_save()
    target_dir = get_pc2_dir()
    global _capture_project_dir
    previous_dir = _capture_project_dir
    from .capture_provenance import copy_pair
    if previous_dir is not None and os.path.realpath(previous_dir) != os.path.realpath(target_dir):
        for scene in bpy.data.scenes:
            for key in _owned_capture_keys(scene):
                name = os.path.basename(get_pc2_path(key))
                source, destination = os.path.join(previous_dir, name), os.path.join(target_dir, name)
                if os.path.exists(source):
                    try:
                        copy_pair(source, destination)
                    except (OSError, ValueError, RuntimeError) as exc:
                        _log_curve_handler_error(f"Capture '{key}' migration refused; source retained: {exc}")
    _capture_project_dir = target_dir
    from .curve_output import migrate_outputs_after_save
    migrate_outputs_after_save()
    tmp_dir = os.path.realpath(os.path.join(tempfile.gettempdir(), "data"))
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            mod = obj.modifiers.get(MODIFIER_NAME)
            if mod is None:
                continue
            abs_path = os.path.realpath(bpy.path.abspath(mod.filepath))
            if not abs_path.startswith(tmp_dir + os.sep) and abs_path != tmp_dir:
                continue
            if not os.path.exists(abs_path):
                continue
            os.makedirs(target_dir, exist_ok=True)
            filename = os.path.basename(abs_path)
            new_path = os.path.join(target_dir, filename)
            shutil.move(abs_path, new_path)
            mod.filepath = bpy.path.relpath(new_path)



# ---------------------------------------------------------------------------
# Render-animation warning (gentle, non-blocking)
#
# Blender doesn't pass an ``is_animation`` flag to render handlers, and
# ``wm.operators`` doesn't record RENDER_OT_render, so we detect
# animation renders by counting ``render_pre`` calls: a single-image
# render reaches frame 1 only; an animation reaches frame 2+.
# ---------------------------------------------------------------------------


_render_job: dict[str, object] = {"frame_count": 0, "warned": False}


@bpy.app.handlers.persistent
def reset_render_counter(_scene, *_args):
    """``render_init`` handler: reset per-job state."""
    _render_job["frame_count"] = 0
    _render_job["warned"] = False


@bpy.app.handlers.persistent
def warn_missing_frames_on_render(scene, *_args):
    """``render_pre`` handler: on the 2nd frame of a render job we know
    it's an animation. If remote frames are still unfetched, show a
    one-shot non-blocking popup."""
    try:
        _render_job["frame_count"] = int(_render_job["frame_count"]) + 1
        if _render_job["frame_count"] < 2 or _render_job["warned"]:
            return

        from .facade import communicator as com
        from .derived import is_server_busy_from_response as is_running
        from ..models.groups import get_addon_data

        response = com.info.response
        remote_frames = int(response.get("frame", 0)) if response else 0
        if remote_frames <= 0 or is_running(response):
            _render_job["warned"] = True
            return
        fetched = get_addon_data(scene).state.convert_fetched_frames_to_list()
        n_missing = remote_frames - len(fetched)
        _render_job["warned"] = True
        if n_missing <= 0:
            return

        msg = iface_(
            "{count} frames unfetched — rendered animation may be "
            "incomplete. Press \"Fetch All Animation\" first."
        ).format(count=n_missing)

        def _popup():
            def _draw(self, _ctx):
                self.layout.label(text=msg, icon="ERROR")
            bpy.context.window_manager.popup_menu(
                _draw, title="Unfetched Frames", icon="ERROR")
            return None
        bpy.app.timers.register(_popup, first_interval=0.0)
    except Exception:
        pass
