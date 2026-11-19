"""User defaults for per-scene PPF state.

Preferences belong to Blender's user configuration; scene choices remain in
the .blend.  Import this module only after the add-on's RNA is registered.
"""
import os
import sys

import bpy
from bpy.app.handlers import persistent

from ..models.groups import _ADDON_NAMESPACE, get_addon_data
from .utils import resolve_local_path

_MARKER = '_ppf_user_defaults_v1'
_FIELDS = {'win32': ('WIN_NATIVE', 'win_native_path'),
           'darwin': ('MAC_NATIVE', 'mac_native_path')}
_seen_scenes = set()
_normalizing = False
_initial_scan = True


def _platform_native():
    if sys.platform.startswith('linux'):
        return 'LINUX_NATIVE', 'linux_native_path'
    return _FIELDS.get(sys.platform, (None, None))


def _preferences(context=None):
    from ..ui.layout import preferences
    context = context or bpy.context
    try:
        return preferences(context)
    except (AttributeError, ReferenceError):
        return None


def _preferred_path(context=None):
    prefs = _preferences(context)
    return getattr(prefs, 'native_solver_path', '') if prefs else ''


def effective_native_path(scene, kind):
    """Resolve the same local path for UI refusal and actual connect."""
    platform_kind, field = _platform_native()
    if kind != platform_kind or not field:
        from ..ui.connection_ops import NATIVE_PATH_FIELDS
        field = NATIVE_PATH_FIELDS.get(kind)
        if not field:
            return ''
        return resolve_local_path(getattr(get_addon_data(scene).ssh_state, field))
    stored = getattr(get_addon_data(scene).ssh_state, field)
    return resolve_local_path(stored if stored.strip() else _preferred_path())


def initialize_native_default(scene):
    """Fill only this host's empty native path; never change connection type."""
    _, field = _platform_native()
    if not field:
        return
    props = get_addon_data(scene).ssh_state
    chosen = _preferred_path()
    if not getattr(props, field).strip() and chosen.strip():
        setattr(props, field, resolve_local_path(chosen))


def initialize_scene(scene, legacy_loaded=False):
    # Inspect raw ownership before any pointer-property access can create it.
    from .migrate_renames import migrate_renamed_state_props, migrate_retired_connection
    root = get_addon_data(scene)
    state = root.state
    initialized = _MARKER in state.keys()
    has_current = 'use_scene_fps' in state.keys()
    has_legacy = 'use_frame_rate_in_output' in state.keys()
    if has_current and has_legacy:
        # The newer explicit scene setting wins over a stranded older key.
        del state["use_frame_rate_in_output"]
    else:
        migrate_renamed_state_props(scene)
    migrate_retired_connection(scene)
    if not initialized:
        if legacy_loaded and not has_current and not has_legacy:
            state.use_scene_fps = False
        state[_MARKER] = True
    initialize_native_default(scene)


def preference_changed(prefs, context):
    global _normalizing
    if _normalizing:
        return
    _normalizing = True
    try:
        value = prefs.native_solver_path
        if value.strip():
            normalized = os.path.normpath(bpy.path.abspath(value.strip()))
            if normalized != value:
                prefs.native_solver_path = normalized
        for scene in bpy.data.scenes:
            initialize_native_default(scene)
    finally:
        _normalizing = False


@persistent
def _defaults_load_post(_unused):
    _seen_scenes.clear()
    _initialize_new_scenes(legacy_loaded=True)


def _initialize_new_scenes(legacy_loaded=False):
    present = set()
    for scene in bpy.data.scenes:
        key = scene.as_pointer()
        present.add(key)
        if key not in _seen_scenes:
            initialize_scene(scene, legacy_loaded=legacy_loaded)
            _seen_scenes.add(key)
    _seen_scenes.intersection_update(present)


def _defaults_tick():
    global _initial_scan
    _initialize_new_scenes(legacy_loaded=_initial_scan and bool(bpy.data.filepath))
    _initial_scan = False
    return 1.0


def register():
    global _initial_scan
    _initial_scan = True
    _seen_scenes.clear()
    for old in list(bpy.app.handlers.load_post):
        if getattr(old, '__name__', '') == '_defaults_load_post':
            bpy.app.handlers.load_post.remove(old)
    bpy.app.handlers.load_post.append(_defaults_load_post)
    # Add-on registration runs under _RestrictData; defer scene access.
    if not bpy.app.timers.is_registered(_defaults_tick):
        bpy.app.timers.register(_defaults_tick, first_interval=0.0, persistent=True)


def unregister():
    if bpy.app.timers.is_registered(_defaults_tick):
        bpy.app.timers.unregister(_defaults_tick)
    for old in list(bpy.app.handlers.load_post):
        if getattr(old, '__name__', '') == '_defaults_load_post':
            bpy.app.handlers.load_post.remove(old)
    _seen_scenes.clear()

