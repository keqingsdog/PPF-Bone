"""Stable named identity for local scene/material/Pin profile selections.
Ordinal-only historical state requires reselection; no identity is guessed.
"""
import os
import bpy
from .profile import load_profiles, save_profile_entry

_KINDS = frozenset(('scene', 'material', 'pin'))

def _fields(kind):
    if kind not in _KINDS:
        raise ValueError('Unknown profile selection kind')
    return kind + '_profile_path', kind + '_profile_selection', '_ppf_' + kind + '_profile_name_v1'

def validate_name(name):
    if not isinstance(name, str) or name in ('', 'NONE') or '\x00' in name:
        raise ValueError('Profile name is empty, reserved (NONE), or contains NUL')
    try:
        name.encode('utf-8')
    except UnicodeError as exc:
        raise ValueError('Profile name is not valid Unicode') from exc
    return name

def selection_name(owner, kind):
    name = owner.get(_fields(kind)[2], '')
    return name if isinstance(name, str) and name not in ('', 'NONE') else ''

def is_legacy(owner, kind):
    path, selection, key = _fields(kind)
    return key not in owner and selection in owner and bool(getattr(owner, path))

def items(owner, kind):
    path, selection, key = _fields(kind)
    raw_path = getattr(owner, path)
    profiles = load_profiles(bpy.path.abspath(raw_path)) if raw_path else {}
    names = []
    for name in sorted(profiles):
        try:
            validate_name(name)
        except ValueError:
            continue
        names.append(name)
    label = '(Legacy selection - reselect profile)' if is_legacy(owner, kind) else '(No Profile)'
    result = [('NONE', label, '', 0)]
    result.extend((name, name, kind.title() + ' profile: ' + name, i) for i, name in enumerate(names, 1))
    selected = selection_name(owner, kind)
    if selected and selected not in names:
        result.append((selected, selected + ' (missing - reselect)', 'No other profile is substituted', len(result)))
    return result

def get_selection(owner, kind):
    selected = selection_name(owner, kind)
    return next((number for name, _label, _tip, number in items(owner, kind) if name == selected), 0)

def set_selection(owner, value, kind):
    for name, _label, _tip, number in items(owner, kind):
        if number == value:
            owner[_fields(kind)[2]] = '' if name == 'NONE' else name
            return
    raise ValueError('Profile selection is no longer available')

def selected_profile(owner, kind):
    path, _selection, _key = _fields(kind)
    name = selection_name(owner, kind)
    if not name:
        raise ValueError('Select a named profile; legacy ordinal selections require explicit reselection')
    profiles = load_profiles(bpy.path.abspath(getattr(owner, path)), strict=True)
    if name not in profiles:
        raise ValueError('Selected profile is missing; explicitly reselect it')
    return profiles[name]

def save_mode(operator):
    mode = operator.save_mode
    if mode == 'AUTO':
        return 'PATH' if operator.properties.is_property_set('filepath') else 'BOUND'
    return mode

def bind_saved(owner, kind, path, name):
    path_field, key = ('profile_path', '_ppf_connection_profile_name_v1') if kind == 'connection' else (_fields(kind)[0], _fields(kind)[2])
    previous_path = getattr(owner, path_field)
    had_key, previous_name = key in owner, owner.get(key)
    try:
        setattr(owner, path_field, path)
        owner[key] = name
    except Exception:
        setattr(owner, path_field, previous_path)
        if had_key:
            owner[key] = previous_name
        elif key in owner:
            del owner[key]
        raise

def save_selected(owner, kind, data, filepath, entry_name, mode='BOUND'):
    path, _selection, _key = _fields(kind)
    selected = selection_name(owner, kind)
    bound = bpy.path.abspath(getattr(owner, path)) if getattr(owner, path) else ''
    explicit = bpy.path.abspath(filepath) if filepath else ''
    use_bound = mode == 'BOUND' and bool(bound and selected)
    same_destination = bool(bound and explicit and os.path.realpath(bound) == os.path.realpath(explicit))
    if use_bound:
        try:
            selected_profile(owner, kind)
        except (OSError, ValueError, TypeError, AttributeError):
            raise
    if use_bound:
        destination, name = bound, selected
    else:
        if mode == 'BOUND':
            raise ValueError('Select a named profile before saving')
        if not explicit:
            raise ValueError('Reselect a profile or choose an explicit Save As destination')
        if is_legacy(owner, kind) and same_destination:
            raise ValueError('Legacy profile identity is ambiguous; explicitly reselect it')
        destination, name = explicit, validate_name(entry_name)
    validate_name(name)
    save_profile_entry(destination, name, data)
    return name, use_bound
