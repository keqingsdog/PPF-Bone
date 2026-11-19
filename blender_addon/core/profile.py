from .pin_reference import decode_stored_pin_identifier, PinReferenceError
# Fork modification: changed from the upstream baseline.
# File: profile.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Connection profile loading from TOML files.

import os
import io
import tempfile
import stat
import sys
import tomllib

# TOML "type" string → SSHState.server_type enum value
PROFILE_TYPE_MAP = {
    "SSH": "CUSTOM",
    "SSH Command": "COMMAND",
    "Docker": "DOCKER",
    "Docker over SSH": "DOCKER_SSH",
    "Docker over SSH Command": "DOCKER_SSH_COMMAND",
    "Windows Native": "WIN_NATIVE",
    "macOS Native": "MAC_NATIVE",
    "Linux Native": "LINUX_NATIVE",
}

# Types a profile file may still name that the add-on no longer offers, mapped
# to what they became. A profile is a file the USER wrote and keeps, so a name
# that was documented has to keep working: `apply_profile` answers False for an
# unknown type and the caller reports the profile as invalid, which for a
# retired name would tell the artist their own file is wrong rather than that
# the type moved. "Local" became the platform's native connection, the same
# place `core.migrate_renames.migrate_retired_connection` sends a saved .blend.
_RETIRED_PROFILE_TYPES = {
    "Local": {
        "win32": "WIN_NATIVE",
        "darwin": "MAC_NATIVE",
    },
}
_RETIRED_PROFILE_DEFAULT = {"Local": "LINUX_NATIVE"}

# TOML keys a profile file may still carry for a field that no longer exists,
# mapped to the field that replaced it, per platform.
_RETIRED_PROFILE_KEYS = {
    "local_path": {
        "win32": "win_native_path",
        "darwin": "mac_native_path",
    },
}
_RETIRED_PROFILE_KEY_DEFAULT = {"local_path": "linux_native_path"}

# TOML key → SSHState property name (keys not listed here are ignored)
_SSH_STATE_FIELDS = {
    "host": "host",
    "port": "port",
    "username": "username",
    "key_path": "key_path",
    "proxy_jump": "proxy_jump",
    "command": "command",
    "container": "container",
    "remote_path": "ssh_remote_path",
    "docker_path": "docker_path",
    "win_native_path": "win_native_path",
    "mac_native_path": "mac_native_path",
    "linux_native_path": "linux_native_path",
    "solver_gpu": "solver_gpu_index",
    "solver_gpu_uuid": "solver_gpu_uuid",
    "docker_port": "docker_port",
}


CONNECTION_PROFILE_NAME_KEY = "_ppf_connection_profile_name_v1"


def validate_connection_profile_name(name: str) -> str:
    """Validate file names before Blender's enum/string boundary can truncate them."""
    if not isinstance(name, str) or name in {"", "NONE"}:
        raise ValueError("Connection profile name is empty or reserved: NONE")
    if "\x00" in name:
        raise ValueError("Connection profile name contains an unsupported NUL character")
    try:
        name.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("Connection profile name is not valid Unicode") from exc
    return name


def connection_profile_selection_name(ssh_state) -> str:
    """Persisted named identity, never inferred from a changing enum ordinal."""
    name = ssh_state.get(CONNECTION_PROFILE_NAME_KEY, "")
    return name if isinstance(name, str) and name not in {"", "NONE"} else ""


def connection_profile_selection_is_legacy(ssh_state) -> bool:
    return (CONNECTION_PROFILE_NAME_KEY not in ssh_state
            and "profile_selection" in ssh_state and bool(ssh_state.profile_path))


def load_profiles(path: str, *, strict: bool = False) -> dict[str, dict]:
    """Parse a TOML profile file and return all profiles.

    Args:
        path: Absolute path to the .toml file.
        strict: Propagate read/parse failures instead of treating them as empty.

    Returns:
        dict mapping profile name → profile data dict.
        Empty dict on errors unless strict=True. A readable empty file is valid.
    """
    try:
        path = os.path.expanduser(path)
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception:
        if strict:
            raise
        return {}


def get_profile_names(path: str) -> list[str]:
    """Return sorted list of profile names from a TOML file."""
    profiles = load_profiles(path)
    return sorted(profiles.keys())


def validate_connection_profile(profile: dict, ssh_state) -> dict:
    """Normalize and validate only fields this connection profile will apply.

    Unknown metadata stays compatible. Current path keys shadow retired aliases;
    no property setters or connection actions are executed during validation.
    """
    from .profile_validation import validate_fields
    if not isinstance(profile, dict):
        raise ValueError("Connection profile must be a table")
    type_str = profile.get("type", "")
    if not isinstance(type_str, str):
        raise ValueError("Connection profile type must be a string")
    server_type = PROFILE_TYPE_MAP.get(type_str) or _retired_type(type_str)
    if server_type is None:
        raise ValueError(f"Unknown connection profile type: {type_str!r}")
    values = {"server_type": server_type}
    if "solver_gpu" in profile and "solver_gpu_uuid" not in profile:
        values["solver_gpu_uuid"] = ""
    for toml_key, prop_name in _SSH_STATE_FIELDS.items():
        if toml_key in profile:
            values[prop_name] = profile[toml_key]
    for toml_key, platforms in _RETIRED_PROFILE_KEYS.items():
        if toml_key not in profile:
            continue
        prop_name = platforms.get(sys.platform, _RETIRED_PROFILE_KEY_DEFAULT[toml_key])
        if prop_name not in profile:
            values[prop_name] = profile[toml_key]
    validate_fields(values, ssh_state.bl_rna, path="connection profile")
    # RNA string setters silently truncate NUL rather than raising. Validate
    # only normalized fields, preserving ignored metadata/alias precedence.
    for name, value in values.items():
        if isinstance(value, str) and "\x00" in value:
            raise ValueError(f"Connection profile {name} contains an unsupported NUL character")
    return values


def apply_profile(profile: dict, ssh_state) -> bool:
    """Apply validated scalar properties, restoring them if a setter fails.

    Preserve the legacy False result for unknown string types; UI callers must
    report that refusal rather than treating it as successful application.
    """
    if isinstance(profile, dict):
        type_str = profile.get("type", "")
        if isinstance(type_str, str) and not (PROFILE_TYPE_MAP.get(type_str) or _retired_type(type_str)):
            return False
    values = validate_connection_profile(profile, ssh_state)
    before = {name: getattr(ssh_state, name) for name in values}
    try:
        for name, value in values.items():
            setattr(ssh_state, name, value)
    except Exception:
        for name, value in reversed(list(before.items())):
            setattr(ssh_state, name, value)
        raise
    return True


def _retired_type(type_str: str):
    """The current type a retired profile type became, or ``None``."""
    platforms = _RETIRED_PROFILE_TYPES.get(type_str)
    if platforms is None:
        return None
    return platforms.get(sys.platform, _RETIRED_PROFILE_DEFAULT[type_str])


# Reverse map: server_type enum → TOML type string
_REVERSE_TYPE_MAP = {v: k for k, v in PROFILE_TYPE_MAP.items()}


def read_connection_profile(ssh_state) -> dict:
    """Read current SSHState properties into a profile dict."""
    result = {}
    type_str = _REVERSE_TYPE_MAP.get(ssh_state.server_type)
    if type_str:
        result["type"] = type_str
    for toml_key, prop_name in _SSH_STATE_FIELDS.items():
        val = getattr(ssh_state, prop_name)
        if val is not None:
            result[toml_key] = val
    return result


# TOML key → State property name for scene params
_SCENE_PARAM_FIELDS = {
    "step_size": "step_size",
    "min_newton_steps": "min_newton_steps",
    "air_density": "air_density",
    "air_friction": "air_friction",
    "friction_mode": "friction_mode",
    "gravity": "gravity_3d",
    "frame_count": "frame_count",
    "frame_start": "frame_start",
    "use_scene_frame_start": "use_scene_frame_start",
    "frame_rate": "frame_rate",
    "use_scene_fps": "use_scene_fps",
    "time_scale": "time_scale",
    "inactive_momentum_frames": "inactive_momentum_frames",
    "wind_direction": "wind_direction",
    "wind_strength": "wind_strength",
    "contact_nnz": "contact_nnz",
    "vertex_air_damp": "vertex_air_damp",
    "auto_save": "auto_save",
    "auto_save_interval": "auto_save_interval",
    "line_search_max_t": "line_search_max_t",
    "constraint_ghat": "constraint_ghat",
    "cg_max_iter": "cg_max_iter",
    "cg_tol": "cg_tol",
    "precond": "precond",
    "schwarz_levels": "schwarz_levels",
    "include_face_mass": "include_face_mass",
    "disable_contact": "disable_contact",
}

# TOML key → ObjectGroup property name for material params.
# Everything here is group-level. Per-assigned-object state (velocity
# overrides, collision windows, pins) lives on `AssignedObject` and is
# therefore already outside this set — do NOT mirror those into the
# profile: they are tied to a specific assigned object's identity and
# don't make sense to copy across groups.
_MATERIAL_PARAM_FIELDS = {
    "object_type": "object_type",
    "color": "color",
    "solid_model": "solid_model",
    "shell_model": "shell_model",
    "rod_model": "rod_model",
    "solid_density": "solid_density",
    "shell_density": "shell_density",
    "rod_density": "rod_density",
    "pdrd_density": "pdrd_density",
    "sand_grain_radius": "sand_grain_radius",
    "sand_particle_mass": "sand_particle_mass",
    "sand_friction": "sand_friction",
    "force_field_weight": "force_field_weight",
    "solid_young_modulus": "solid_young_modulus",
    "shell_young_modulus": "shell_young_modulus",
    "rod_young_modulus": "rod_young_modulus",
    "young_mod_density_normalized": "young_mod_density_normalized",
    "solid_poisson_ratio": "solid_poisson_ratio",
    "shell_poisson_ratio": "shell_poisson_ratio",
    "friction": "friction",
    "contact_gap": "contact_gap",
    "contact_offset": "contact_offset",
    "use_group_bounding_box_diagonal": "use_group_bounding_box_diagonal",
    "contact_gap_rat": "contact_gap_rat",
    "contact_offset_rat": "contact_offset_rat",
    "bend": "bend",
    "deformation_damping": "deformation_damping",
    "bending_damping": "bending_damping",
    "bend_warp": "bend_warp",
    "bend_weft": "bend_weft",
    "length_factor": "length_factor",
    "shrink": "shrink",
    "shrink_x": "shrink_x",
    "shrink_y": "shrink_y",
    "enable_strain_limit": "enable_strain_limit",
    "strain_limit_percent": "strain_limit_percent",
    "enable_inflate": "enable_inflate",
    "inflate_pressure": "inflate_pressure",
    "enable_soft_constraint": "enable_soft_constraint",
    "soft_constraint_stiffness": "soft_constraint_stiffness",
    "stitch_stiffness": "stitch_stiffness",
    "enable_plasticity": "enable_plasticity",
    "plasticity": "plasticity",
    "plasticity_threshold": "plasticity_threshold",
    "enable_bend_plasticity": "enable_bend_plasticity",
    "bend_plasticity": "bend_plasticity",
    "bend_plasticity_threshold": "bend_plasticity_threshold",
    "bend_rest_angle_source": "bend_rest_angle_source",
    "bend_rest_from_reference": "bend_rest_from_reference",
    "use_collision_windows": "use_collision_windows",
}

_VECTOR_PROPERTIES = {"gravity_3d", "wind_direction", "color"}

# Retired TOML key -> its replacement. A profile written by an older build still
# carries the old key; without this the value is dropped and the profile quietly
# loads with a different simulation time base than it was saved with.
_LEGACY_SCENE_PARAM_KEYS = {
    "use_frame_rate_in_output": "use_scene_fps",
}


def validate_scene_profile(profile: dict, state) -> dict:
    """Read-only preflight; return a copy with legacy keys normalized."""
    if not isinstance(profile, dict):
        raise ValueError("Scene profile must be a table")
    profile = dict(profile)
    # Profiles from the experimental FP16 release remain loadable.
    # Current scene export always uses FP32.
    profile.pop("pcg_precision", None)
    for old_key, new_key in _LEGACY_SCENE_PARAM_KEYS.items():
        if old_key in profile and new_key not in profile:
            profile[new_key] = profile.pop(old_key)
    from .profile_validation import validate_fields
    validate_fields(profile, state.bl_rna,
                    {**_SCENE_PARAM_FIELDS, "dyn_params": "dyn_params", "colliders": "invisible_colliders"})
    for entry in profile.get("dyn_params", []):
        if "param_type" not in entry:
            raise ValueError("Dynamic parameter profile is missing param_type")
    return profile


def apply_scene_profile(profile: dict, state) -> bool:
    """Apply a scene param profile dict after a complete input preflight."""
    profile = validate_scene_profile(profile, state)
    for toml_key, prop_name in _SCENE_PARAM_FIELDS.items():
        if toml_key in profile:
            value = profile[toml_key]
            if prop_name in _VECTOR_PROPERTIES and isinstance(value, list):
                value = tuple(value)
            setattr(state, prop_name, value)
    # Restore dynamic parameters
    if "dyn_params" in profile:
        state.dyn_params.clear()
        for param_entry in profile["dyn_params"]:
            param_type = param_entry.get("param_type", "")
            keyframes = param_entry.get("keyframes", [])
            dyn_item = state.dyn_params.add()
            dyn_item.param_type = param_type
            for i, kf_entry in enumerate(keyframes):
                kf = dyn_item.keyframes.add()
                kf.frame = kf_entry.get("frame", 1)
                if i > 0:
                    kf.use_hold = kf_entry.get("use_hold", False)
                    if not kf.use_hold:
                        if param_type == "GRAVITY" and "gravity_value" in kf_entry:
                            kf.gravity_value = tuple(kf_entry["gravity_value"])
                        elif param_type == "WIND":
                            if "wind_direction_value" in kf_entry:
                                kf.wind_direction_value = tuple(kf_entry["wind_direction_value"])
                            if "wind_strength_value" in kf_entry:
                                kf.wind_strength_value = kf_entry["wind_strength_value"]
                        elif "scalar_value" in kf_entry:
                            kf.scalar_value = kf_entry["scalar_value"]
        state.dyn_params_index = 0 if len(state.dyn_params) > 0 else -1
    # Restore invisible colliders
    if "colliders" in profile:
        apply_collider_profile(profile, state)
    return True


def validate_material_profile(profile: dict, object_group):
    """Read-only preflight of all fields and targeted Pins before any writes."""
    from .profile_validation import validate_fields
    validate_fields(profile, object_group.bl_rna, _MATERIAL_PARAM_FIELDS)
    return _validate_pin_profile_batch(profile, object_group,
        object_type=profile.get("object_type", object_group.object_type))


def apply_material_profile(profile: dict, object_group) -> bool:
    """Apply a material param profile dict after a complete input preflight."""
    targets = validate_material_profile(profile, object_group)
    for toml_key, prop_name in _MATERIAL_PARAM_FIELDS.items():
        if toml_key in profile:
            value = profile[toml_key]
            if prop_name in _VECTOR_PROPERTIES and isinstance(value, list):
                value = tuple(value)
            setattr(object_group, prop_name, value)
    # Restore embedded pin profiles when present (UUID-keyed apply).
    for target, pin_data in targets:
        _write_pin_operations(pin_data, target)
    return True


_TOML_STRING_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def _toml_escape_string(s: str) -> str:
    """Escape a Python string for a basic TOML string literal.

    TOML basic strings use backslash escapes for: backslash, double-quote,
    \\b \\t \\n \\f \\r, plus \\uXXXX for other control characters. Anything
    else is emitted as-is (TOML is UTF-8, so non-ASCII text passes through).
    """
    out = []
    for ch in s:
        esc = _TOML_STRING_ESCAPES.get(ch)
        if esc is not None:
            out.append(esc)
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return "".join(out)


def _toml_is_bare_key(s: str) -> bool:
    """TOML bare keys: [A-Za-z0-9_-]+."""
    if not s:
        return False
    return all(c.isascii() and (c.isalnum() or c in "_-") for c in s)


def _toml_key(s: str) -> str:
    """Format a TOML key: bare if simple, quoted otherwise."""
    if _toml_is_bare_key(s):
        return s
    return f'"{_toml_escape_string(s)}"'


def _toml_value(val) -> str:
    """Format a Python value as a TOML value string."""
    if val is None:
        # TOML has no null; emit an empty string so loader round-trips produce
        # an empty value the caller's coerce step can handle, instead of the
        # bare literal "None" which would make the whole file unparsable.
        return '""'
    if isinstance(val, bool):
        return "true" if val else "false"
    elif isinstance(val, (list, tuple)):
        items = ", ".join(_toml_value(x) for x in val)
        return f"[{items}]"
    elif isinstance(val, float):
        return str(val)
    elif isinstance(val, int):
        return str(val)
    elif isinstance(val, str):
        return f'"{_toml_escape_string(val)}"'
    return str(val)


def read_scene_profile(state) -> dict:
    """Read current State properties into a profile dict."""
    result = {}
    for toml_key, prop_name in _SCENE_PARAM_FIELDS.items():
        val = getattr(state, prop_name)
        if prop_name in _VECTOR_PROPERTIES:
            val = list(val)
        result[toml_key] = val
    # Dynamic parameters: grouped by param_type with nested keyframes
    dyn_entries = []
    for dyn_item in state.dyn_params:
        param_entry = {"param_type": dyn_item.param_type, "keyframes": []}
        for i, kf in enumerate(dyn_item.keyframes):
            kf_entry = {"frame": kf.frame}
            if i > 0:
                kf_entry["use_hold"] = kf.use_hold
                if not kf.use_hold:
                    if dyn_item.param_type == "GRAVITY":
                        kf_entry["gravity_value"] = list(kf.gravity_value)
                    elif dyn_item.param_type == "WIND":
                        kf_entry["wind_direction_value"] = list(kf.wind_direction_value)
                        kf_entry["wind_strength_value"] = kf.wind_strength_value
                    else:
                        kf_entry["scalar_value"] = kf.scalar_value
            param_entry["keyframes"].append(kf_entry)
        dyn_entries.append(param_entry)
    if dyn_entries:
        result["dyn_params"] = dyn_entries
    # Invisible colliders
    collider_data = read_collider_profile(state)
    if "colliders" in collider_data:
        result["colliders"] = collider_data["colliders"]
    return result


def read_material_profile(object_group, include_pins=False) -> dict:
    """Read current ObjectGroup properties into a profile dict."""
    result = {}
    for toml_key, prop_name in _MATERIAL_PARAM_FIELDS.items():
        val = getattr(object_group, prop_name)
        if prop_name in _VECTOR_PROPERTIES:
            val = list(val)
        result[toml_key] = val
    if include_pins:
        pin_data = read_pin_profiles(object_group)
        result.update(pin_data)
    return result


_OP_FIELDS = {
    "op_type": "op_type",
    "delta": "delta",
    "spin_axis": "spin_axis",
    "spin_angular_velocity": "spin_angular_velocity",
    "spin_flip": "spin_flip",
    "spin_center": "spin_center",
    "spin_center_mode": "spin_center_mode",
    "spin_center_vertex": "spin_center_vertex",
    "spin_center_direction": "spin_center_direction",
    "scale_factor": "scale_factor",
    "scale_center": "scale_center",
    "scale_center_mode": "scale_center_mode",
    "scale_center_vertex": "scale_center_vertex",
    "scale_center_direction": "scale_center_direction",
    "torque_axis_component": "torque_axis_component",
    "torque_magnitude": "torque_magnitude",
    "torque_flip": "torque_flip",
    "frame_start": "frame_start",
    "frame_end": "frame_end",
    "transition": "transition",
}

_OP_VECTOR_FIELDS = {"delta", "spin_axis", "spin_center", "scale_center", "spin_center_direction", "scale_center_direction"}


def read_collider_profile(state) -> dict:
    """Read current invisible colliders into a profile dict."""
    colliders = []
    for item in state.invisible_colliders:
        entry = {
            "collider_type": item.collider_type,
            "name": item.name,
            "position": list(item.position),
            "contact_gap": item.contact_gap,
            "friction": item.friction,
            "thickness": item.thickness,
            "enable_active_duration": item.enable_active_duration,
            "active_duration": item.active_duration,
        }
        if item.collider_type == "WALL":
            entry["normal"] = list(item.normal)
        else:
            entry["radius"] = item.radius
            entry["hemisphere"] = item.hemisphere
            entry["invert"] = item.invert
        kf_list = []
        for i, kf in enumerate(item.keyframes):
            kf_entry = {"frame": kf.frame}
            if i > 0:
                kf_entry["use_hold"] = kf.use_hold
                if not kf.use_hold:
                    kf_entry["position"] = list(kf.position)
                    if item.collider_type == "SPHERE":
                        kf_entry["radius"] = kf.radius
            kf_list.append(kf_entry)
        entry["keyframes"] = kf_list
        colliders.append(entry)
    return {"colliders": colliders}


def apply_collider_profile(profile: dict, state) -> bool:
    """Apply a collider profile dict to State invisible_colliders."""
    if "colliders" not in profile:
        return True
    from .profile_validation import validate_fields
    validate_fields(profile, state.bl_rna, {"colliders": "invisible_colliders"})
    state.invisible_colliders.clear()
    for entry in profile["colliders"]:
        item = state.invisible_colliders.add()
        item.collider_type = entry.get("collider_type", "WALL")
        item.name = entry.get("name", "")
        if "position" in entry:
            item.position = tuple(entry["position"])
        if item.collider_type == "WALL":
            if "normal" in entry:
                item.normal = tuple(entry["normal"])
        else:
            if "radius" in entry:
                item.radius = entry["radius"]
            item.hemisphere = entry.get("hemisphere", False)
            item.invert = entry.get("invert", False)
        item.contact_gap = entry.get("contact_gap", 0.001)
        item.friction = entry.get("friction", 0.0)
        item.thickness = entry.get("thickness", 1.0)
        item.enable_active_duration = entry.get("enable_active_duration", False)
        item.active_duration = entry.get("active_duration", 60)
        for i, kf_entry in enumerate(entry.get("keyframes", [])):
            kf = item.keyframes.add()
            kf.frame = kf_entry.get("frame", 1)
            if i > 0:
                kf.use_hold = kf_entry.get("use_hold", False)
                if not kf.use_hold:
                    if "position" in kf_entry:
                        kf.position = tuple(kf_entry["position"])
                    if "radius" in kf_entry:
                        kf.radius = kf_entry["radius"]
    state.invisible_colliders_index = 0 if len(state.invisible_colliders) > 0 else -1
    return True


def read_pin_operations(pin_item) -> dict:
    """Read operations from a PinVertexGroupItem into a profile dict."""
    ops = []
    for op in pin_item.operations:
        op_dict = {}
        for toml_key, prop_name in _OP_FIELDS.items():
            val = getattr(op, prop_name)
            if prop_name in _OP_VECTOR_FIELDS:
                val = list(val)
            op_dict[toml_key] = val
        ops.append(op_dict)
    return {"operations": ops}


def validate_pin_operations(profile: dict, pin_rna):
    """Read-only Pin operation preflight using the registered item RNA type."""
    from .profile_validation import validate_fields
    validate_fields(profile, pin_rna, {"operations": "operations"})


def validate_pin_source_replacement(profile, pin_item, *, had_marker=None, object_type=None):
    """Read-only admission for a replacement that removes a live source marker.

    Captured tracks remain independent of the marker. Only the unflagged,
    plain-pin cache case needs it to reach the encoder's existing cache path.
    Unknown manual animation containers cannot prove a marker obsolete.
    """
    if "operations" not in profile:
        return
    if had_marker is None:
        had_marker = any(op.op_type == "EMBEDDED_MOVE" for op in pin_item.operations)
    incoming = profile["operations"]
    if not had_marker or any(op.get("op_type") == "EMBEDDED_MOVE" for op in incoming):
        return
    reason = "the current animation source cannot be proven safe to replace"
    try:
        from .uuid_registry import plan_pin_reference
        from .encoder.pin import _collect_pin_vertex_fcurve_frames, _get_pin_indices
        plan = plan_pin_reference(pin_item)
        if plan is not None:
            obj, _old_vg, vg_name, _encoded = plan
            if obj.type in {"MESH", "CURVE"} and _get_pin_indices(obj, vg_name):
                # A valid Curve cannot use the Mesh-only embedded source.
                # Allow replacing its obsolete marker with a valid operation.
                if obj.type == "CURVE":
                    return
                animation = obj.data.animation_data
                uncertain = animation is not None and (
                    bool(animation.drivers) or bool(animation.nla_tracks)
                    or bool(getattr(animation.action, "layers", ())))
                if not uncertain:
                    frames, curves = _collect_pin_vertex_fcurve_frames(obj, vg_name)
                    if frames or curves:
                        reason = "Mesh vertex animation data is present; removing its embedded animation operation requires review"
                    else:
                        # Do not load a cache or enable historical unflagged data.
                        # Identity above proves an existing, unique UUID first.
                        from .pc2 import has_pin_anim_pc2
                        risky_plain = (not incoming
                            and not pin_item.has_captured_anim
                            and not pin_item.use_pin_duration
                            and not pin_item.use_pull
                            and not getattr(pin_item, "allow_intersection", False))
                        if risky_plain and has_pin_anim_pc2(obj, vg_name):
                            effective_type = object_type
                            if effective_type is None:
                                from ..models.groups import iterate_object_groups
                                owners = [group for group in iterate_object_groups(pin_item.id_data)
                                    if any(item.as_pointer() == pin_item.as_pointer()
                                        for item in group.pin_vertex_groups)]
                                if len(owners) == 1:
                                    effective_type = owners[0].object_type
                            if effective_type == "SOLID":
                                return
                            reason = "a capture cache entry or path is present, and this empty replacement would skip checking it"
                        else:
                            return
    except (PinReferenceError, OSError, ValueError, TypeError, AttributeError, ReferenceError):
        pass
    raise ValueError("Pin operations were not replaced: " + reason
        + ". Keep the embedded animation operation or review the animation source before replacing it.")


def apply_pin_operations(profile: dict, pin_item):
    """Validate the complete replacement before writing operations."""
    validate_pin_operations(profile, pin_item.bl_rna)
    validate_pin_source_replacement(profile, pin_item)
    _write_pin_operations(profile, pin_item)


def _write_pin_operations(profile: dict, pin_item):
    """Write an already-admitted payload; batch callers preflight all aliases."""
    if "operations" not in profile:
        return
    pin_item.operations.clear()
    for op_data in profile["operations"]:
        op = pin_item.operations.add()
        for toml_key, prop_name in _OP_FIELDS.items():
            if toml_key in op_data:
                val = op_data[toml_key]
                if prop_name in _OP_VECTOR_FIELDS and isinstance(val, list):
                    val = tuple(val)
                setattr(op, prop_name, val)


def read_pin_profiles(object_group) -> dict:
    """Read all pin vertex groups on *object_group* into a profile dict.

    Per project-wide UUID-only identity rule, internal identity is
    ``(object_uuid, vg_name)``. The on-disk TOML table key uses the
    display identifier ``[obj_name][vg_name]`` purely for user
    readability; ``object_uuid`` and the vertex group name are embedded
    in the entry itself so :func:`apply_pin_profiles` can match by UUID
    even after object or VG renames.
    """
    from ..models.groups import decode_vertex_group_identifier

    result = {}
    for pin_item in object_group.pin_vertex_groups:
        entry = read_pin_operations(pin_item)
        # Embed stable identity (UUID + VG name) so apply-back is
        # robust to object/VG renames. Readable display key stays for
        # the TOML table name.
        entry["object_uuid"] = pin_item.object_uuid
        _, vg_name = decode_stored_pin_identifier(pin_item.name)
        if vg_name:
            entry["vg_name"] = vg_name
        result[pin_item.name] = entry
    return {"pins": result} if result else {}


def _pin_profile_targets(profile: dict, object_group):
    """Resolve targeted Pin/profile pairs without writing author data.

    Match by ``(object_uuid, vg_name)`` — NEVER by display name —
    so renames never break profile apply. Falls back to display-name
    match only when the profile entry lacks a stored UUID (legacy
    profiles written before UUID embedding).
    """
    from ..models.groups import decode_vertex_group_identifier

    pins = profile.get("pins")
    if pins is None:
        return []
    if not isinstance(pins, dict):
        raise ValueError("profile.pins must be a table")
    targets = []
    # Build UUID-keyed index: (object_uuid, vg_name) -> pin_item.
    # vg_name is decoded from the display identifier; object_uuid is
    # the stable identity we trust.
    uuid_index = {}
    name_index = {}
    for item in object_group.pin_vertex_groups:
        _, vg_name = decode_stored_pin_identifier(item.name)
        if item.object_uuid and vg_name:
            uuid_index[(item.object_uuid, vg_name)] = item
        name_index[item.name] = item
    for pin_key, pin_data in pins.items():
        if not isinstance(pin_data, dict):
            raise ValueError(f"profile.pins.{pin_key} must be a table")
        target = None
        obj_uuid = pin_data.get("object_uuid", "")
        vg_name = pin_data.get("vg_name", "")
        if not isinstance(obj_uuid, str) or not isinstance(vg_name, str):
            raise ValueError(f"profile.pins.{pin_key} object_uuid and vg_name must be strings")
        if obj_uuid and vg_name:
            target = uuid_index.get((obj_uuid, vg_name))
        # Only UUID-less legacy entries may use a display-name fallback.
        # A supplied UUID that does not match must never redirect to another Pin.
        if target is None and not obj_uuid:
            target = name_index.get(pin_key)
        if target is None:
            continue
        targets.append((target, pin_data))
    return targets


def _validate_pin_profile_batch(profile: dict, object_group, *, object_type=None):
    """Validate schemas first, then simulate marker state in actual write order."""
    if object_type is None:
        object_type = object_group.object_type
    targets = _pin_profile_targets(profile, object_group)
    for target, pin_data in targets:
        validate_pin_operations(pin_data, target.bl_rna)
    markers = {}
    for target, pin_data in targets:
        identity = target.as_pointer()
        previous = markers.setdefault(identity,
            any(op.op_type == "EMBEDDED_MOVE" for op in target.operations))
        validate_pin_source_replacement(pin_data, target, had_marker=previous, object_type=object_type)
        if "operations" in pin_data:
            markers[identity] = any(op.get("op_type") == "EMBEDDED_MOVE"
                for op in pin_data["operations"])
    return targets


def apply_pin_profiles(profile: dict, object_group):
    """Admit the entire ordered batch before clearing any Pin operations."""
    targets = _validate_pin_profile_batch(profile, object_group)
    for target, pin_data in targets:
        _write_pin_operations(pin_data, target)


def save_profile_entry(path: str, entry_name: str, data: dict):
    """Save a profile entry to a TOML file, preserving other entries.

    If the file exists, updates/adds the entry. Otherwise creates a new file.
    """
    path = os.path.expanduser(path)
    # UI loaders are forgiving; saves must never interpret unreadable data as
    # an empty document and overwrite the user's other entries.
    path = os.path.realpath(path)
    existing_mode = None
    if os.path.exists(path):
        existing_mode = stat.S_IMODE(os.stat(path).st_mode)
        if not existing_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise PermissionError("Profile file is read-only; original file was not changed")
        # Match the old writer's effective access check without truncating data.
        descriptor = os.open(path, os.O_WRONLY)
        os.close(descriptor)
        with open(path, "rb") as source:
            profiles = tomllib.load(source)
    else:
        profiles = {}
    if not isinstance(data, dict) or any(not isinstance(v, dict) for v in profiles.values()):
        raise ValueError("Profile file must contain named profile tables; original file was not changed")
    profiles[entry_name] = data

    with io.StringIO() as f:
        for name, profile in profiles.items():
            name_tok = _toml_key(name)
            f.write(f"[{name_tok}]\n")
            for key, val in profile.items():
                if isinstance(val, list) and val and isinstance(val[0], dict):
                    continue  # Write arrays of tables after scalar fields
                f.write(f"{_toml_key(key)} = {_toml_value(val)}\n")
            # Write arrays of tables (e.g., operations, dyn_params)
            for key, val in profile.items():
                if isinstance(val, list) and val and isinstance(val[0], dict):
                    key_tok = _toml_key(key)
                    for item in val:
                        f.write(f"\n[[{name_tok}.{key_tok}]]\n")
                        for k, v in item.items():
                            if isinstance(v, list) and v and isinstance(v[0], dict):
                                continue  # Write nested arrays after scalar fields
                            f.write(f"{_toml_key(k)} = {_toml_value(v)}\n")
                        # Write nested arrays of tables (e.g., keyframes)
                        for k, v in item.items():
                            if isinstance(v, list) and v and isinstance(v[0], dict):
                                k_tok = _toml_key(k)
                                for sub_item in v:
                                    f.write(f"\n[[{name_tok}.{key_tok}.{k_tok}]]\n")
                                    for sk, sv in sub_item.items():
                                        f.write(f"{_toml_key(sk)} = {_toml_value(sv)}\n")
            f.write("\n")

        document = f.getvalue()
    # Reject unsupported/nonnormalized values before opening a destination.
    tomllib.loads(document)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                dir=os.path.dirname(path), prefix=".ppf-profile-", suffix=".tmp",
                delete=False) as target:
            temporary_path = target.name
            target.write(document)
            target.flush()
            os.fsync(target.fileno())
        if existing_mode is not None:
            os.chmod(temporary_path, existing_mode)  # Preserve standard permission bits; see documented metadata limits.
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            os.unlink(temporary_path)
