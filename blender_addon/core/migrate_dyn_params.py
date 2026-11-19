# Fork modification: changed from the upstream baseline.
# File: migrate_dyn_params.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# One-shot conversion of the addon's own scene-parameter keyframe list into
# real Blender F-curves.
#
# The list was a second place to author keyframes, with its own UI, its own
# hold semantics and no presence on the timeline. Everything it expressed is
# expressible as an F-curve on the slider it drove, so the list is converted
# and cleared: after this the artist keyframes the slider, the same gesture
# that drives a material parameter.
#
# Conversion is lossless for what the list could say. A HOLD keyframe repeats
# the previous value, which is a CONSTANT interpolation segment, so it becomes
# one rather than being flattened into a linear ramp.

import bpy  # pyright: ignore


# Legacy param_type -> the state property that now carries its curve, and the
# keyframe field the legacy entry stored it in.
_LEGACY = {
    "GRAVITY": ("gravity_3d", "gravity_value", 3),
    "WIND": ("wind_direction", "wind_direction_value", 3),
    "AIR_DENSITY": ("air_density", "scalar_value", 0),
    "AIR_FRICTION": ("air_friction", "scalar_value", 0),
    "VERTEX_AIR_DAMP": ("vertex_air_damp", "scalar_value", 0),
}


def count_legacy_dyn_params(scene) -> int:
    """How many legacy entries are still waiting to be converted."""
    root = getattr(scene, "zozo_contact_solver", None)
    state = getattr(root, "state", None)
    items = getattr(state, "dyn_params", None)
    if not items:
        return 0
    return sum(1 for item in items if len(item.keyframes) >= 2)


def convert_legacy_dyn_params(scene) -> str:
    """Convert every legacy entry to F-curves and remove it from the list.

    The legacy first entry is the slider's own value in effect at the starting
    frame (``resolve_start_frame``), whatever frame it stores, which is how the
    encoder reads it, so its key lands on the starting frame. An entry with a
    later keyframe at or before the starting frame has no faithful F-curve (the
    key would sort ahead of the one standing for time zero), so it is left in
    the list, where the encoder refuses it by name at Transfer.

    Returns a human-readable summary, empty when there was nothing to do.
    """
    from .encoder import resolve_start_frame

    root = getattr(scene, "zozo_contact_solver", None)
    state = getattr(root, "state", None)
    items = getattr(state, "dyn_params", None)
    if not items:
        return ""
    start_frame = resolve_start_frame(state)

    # Never overwrite visible author animation during an automatic migration.
    # A shared Action belongs to other IDs as well; leave legacy data intact
    # rather than modifying that Action or silently choosing a copy for them.
    from .utils import get_id_fcurves
    animation = getattr(scene, "animation_data", None)
    action = getattr(animation, "action", None)
    existing_paths = {curve.data_path for curve in get_id_fcurves(scene)}
    existing_paths.update(curve.data_path for curve in getattr(animation, "drivers", ()))
    unsafe_action = bool(
        (action is not None and (action.users > 1 or action.library is not None))
        or list(getattr(animation, "nla_tracks", ()))
    )
    converted = []
    preserved = []
    retained_wind = []
    kept = []
    kept_indices = set()
    for item_index, item in enumerate(items):
        spec = _LEGACY.get(item.param_type)
        if spec is None or len(item.keyframes) < 2:
            continue
        frames = [start_frame] + [int(kf.frame) for kf in list(item.keyframes)[1:]]
        if any(right <= left for left, right in zip(frames, frames[1:])):
            kept.append(item.param_type)
            kept_indices.add(item_index)
            continue
        prop, value_field, length = spec
        if item.param_type == "WIND" and any(
            not kf.use_hold
            and tuple(kf.wind_direction_value) != tuple(state.wind_direction)
            for kf in list(item.keyframes)[1:]
        ):
            # Legacy interpolates Cartesian velocity endpoints. Interpolating
            # direction then normalizing it changes the trajectory and speed.
            # Keep the exact old schedule instead of an approximate conversion.
            retained_wind.append(item.param_type)
            kept_indices.add(item_index)
            continue
        paths = {state.path_from_id(prop)}
        if item.param_type == "WIND":
            paths.add(state.path_from_id("wind_strength"))
        if unsafe_action or paths.intersection(existing_paths):
            preserved.append(item.param_type)
            kept_indices.add(item_index)
            continue
        for index, kf in enumerate(item.keyframes):
            frame = start_frame if index == 0 else int(kf.frame)
            if index == 0:
                # The legacy first entry means "the slider's current value at
                # the starting frame", so key the slider as it stands.
                pass
            elif kf.use_hold:
                # Hold repeats the previous value. Setting the slider back to
                # what the previous key holds and marking that key CONSTANT is
                # the F-curve that means the same thing.
                pass
            else:
                if length:
                    setattr(state, prop, tuple(getattr(kf, value_field)))
                else:
                    setattr(state, prop, float(kf.scalar_value))
            state.keyframe_insert(data_path=prop, frame=frame)

        # WIND kept its strength in a separate legacy field, so it needs its
        # own curve or the direction would animate against a frozen strength.
        if item.param_type == "WIND":
            for index, kf in enumerate(item.keyframes):
                if index and not kf.use_hold:
                    state.wind_strength = float(kf.wind_strength_value)
                state.keyframe_insert(
                    data_path="wind_strength",
                    frame=start_frame if index == 0 else int(kf.frame),
                )

        _apply_hold_interpolation(scene, state, item, prop)
        converted.append(item.param_type)

    if converted:
        # Everything but a kept entry goes: an entry with fewer than two
        # keyframes never reached the solver, so it carries nothing to keep.
        for item_index in reversed(range(len(items))):
            if item_index not in kept_indices:
                items.remove(item_index)
        state.dyn_params_index = -1
    summary = []
    if converted:
        summary.append(
            "converted %d scene keyframe list(s) to F-curves: %s"
            % (len(converted), ", ".join(converted))
        )
    if retained_wind:
        summary.append("left WIND keyframe list intact to preserve its changing-direction vector trajectory")
    if preserved:
        summary.append(
            "left %d scene keyframe list(s) intact to preserve existing or shared "
            "author animation: %s" % (len(preserved), ", ".join(preserved))
        )
    if kept:
        summary.append(
            "left %d scene keyframe list(s) unconverted: frames are not strictly "
            "increasing after the starting frame %d: %s"
            % (len(kept), start_frame, ", ".join(kept))
        )
    return "; ".join(summary)


def _apply_hold_interpolation(scene, state, item, prop):
    """Mark the keys a legacy HOLD produced as CONSTANT.

    A hold said "keep the previous value until here". On an F-curve that is a
    constant-interpolation segment; leaving it linear would ramp through values
    the legacy schedule never produced.
    """
    animation = getattr(scene, "animation_data", None)
    if animation is None or animation.action is None:
        return
    paths = {state.path_from_id(prop)}
    if item.param_type == "WIND":
        paths.add(state.path_from_id("wind_strength"))
    holds = {int(kf.frame) for i, kf in enumerate(item.keyframes)
             if i and kf.use_hold}
    from .utils import get_id_fcurves
    for curve in get_id_fcurves(scene):
        if curve.data_path not in paths:
            continue
        # These paths were checked for existing author animation before insert.
        # Legacy ordinary segments are linear, not Blender's default Bezier.
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"
        points = sorted(curve.keyframe_points, key=lambda k: k.co[0])
        for hold_frame in holds:
            # Interpolation belongs to the key a segment STARTS at, so
            # the key that must go constant is the one BEFORE the hold,
            # not the hold's own. Marking the hold instead would flatten
            # the segment AFTER it, freezing a value the legacy
            # schedule meant to move.
            previous = None
            for point in points:
                if point.co[0] < hold_frame:
                    previous = point
                else:
                    break
            if previous is not None:
                previous.interpolation = "CONSTANT"
