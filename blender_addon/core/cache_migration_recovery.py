"""Narrow recovery-path rebasing for an already byte-verified Keep Copy."""
import json
import os
import bpy
from . import pc2, single_bake_restore as single, full_bake_restore as full


def _same_bytes(left, right):
    with open(left, 'rb') as a, open(right, 'rb') as b:
        while True:
            x, y = a.read(1024 * 1024), b.read(1024 * 1024)
            if x != y:
                return False
            if not x:
                return True


def _absolute_anchor(path):
    return isinstance(path, str) and os.path.isabs(path) and not path.startswith('//')


def validate_rewrites(rewrites, scene):
    """Reject global/linked ownership changes before any destination is copied."""
    for mod, _filename in rewrites:
        obj = mod.id_data
        if (obj.library is not None or obj.override_library is not None
                or obj.data.library is not None or obj.data.override_library is not None):
            raise ValueError('Migration affects a linked or overridden object; no copy or references changed. Make an independent local project first.')
        if len(obj.users_scene) != 1 or obj.users_scene[0] != scene:
            raise ValueError('Migration affects an object outside this Scene or shared across Scenes; no copy or references changed. Separate the project first.')
        for other_scene in bpy.data.scenes:
            if other_scene == scene:
                continue
            for key in (single.KEY, full.KEY):
                tx = other_scene.get(key)
                if tx and any(row.get('obj') == obj for row in tx.get('items', [])):
                    raise ValueError('Migration affects another Scene recovery record; references and records retained. Separate the project first.')


def apply_paths(rewrites, new_dir):
    """Change only proved cache locations; never re-sign author state.

    Unknown/legacy recovery anchors remain unchanged and continue to refuse.
    An author change to geometry, action, modifier settings or a different
    source filepath is never blessed by refreshing its fingerprint here.
    """
    def path_key(path):
        return os.path.normcase(os.path.realpath(bpy.path.abspath(path)))

    def signature_change(signature, mod, anchor, new_raw):
        try:
            props = json.loads(signature)
        except (TypeError, ValueError):
            return None
        if (not isinstance(props, dict) or props.get('name') != mod.name
                or not _absolute_anchor(anchor)
                or not single._modifier_matches(mod, signature, anchor)):
            return None
        props['filepath'] = new_raw
        return json.dumps(props, sort_keys=True)

    validate_rewrites(rewrites, bpy.context.scene)
    mods = []
    changes = []
    for mod, filename in rewrites:
        old_raw = mod.filepath
        old_abs = path_key(old_raw)
        new_abs = path_key(os.path.join(new_dir, filename))
        new_raw = bpy.path.relpath(new_abs)
        if mod.type != 'MESH_CACHE' or not _same_bytes(old_abs, new_abs):
            raise ValueError('Migration cache bytes differ; references and recovery were not changed')
        mods.append((mod, old_raw, new_raw))
        obj = mod.id_data
        digest = full._cache_digest(old_abs)
        if digest != full._cache_digest(new_abs):
            raise ValueError('Migration cache changed during verification; records retained')
        for scene in bpy.data.scenes:
            for kind, key in (('full', full.KEY), ('single', single.KEY)):
                tx = scene.get(key)
                if not tx:
                    continue
                for row in tx.get('items', []):
                    anchor = row.get('cache_path')
                    if (row.get('obj') != obj or row.get('cache_sha256') != digest
                            or not row.get('source_signature')):
                        continue
                    if kind == 'single':
                        if not _absolute_anchor(anchor) or path_key(anchor) != old_abs:
                            continue
                        matches = []
                        for item in row.get('mods', []):
                            if item.get('name') == mod.name and item.get('type') == 'MESH_CACHE':
                                old_sig = item.get('signature', '')
                                new_sig = signature_change(old_sig, mod, anchor, new_raw)
                                if new_sig is not None:
                                    matches.append((item, old_sig, new_sig))
                        if len(matches) != 1:
                            continue
                        item, old_sig, new_sig = matches[0]
                        changes.extend(((row, 'cache_path', anchor, new_abs), (item, 'signature', old_sig, new_sig)))
                    else:
                        if not _absolute_anchor(anchor) or path_key(anchor) != old_abs:
                            continue
                        old_sigs = list(row.get('mod_signatures', []))
                        matches = [(i, signature_change(sig, mod, anchor, new_raw)) for i, sig in enumerate(old_sigs)]
                        matches = [(i, sig) for i, sig in matches if sig is not None]
                        if len(matches) != 1:
                            continue
                        new_sigs = old_sigs.copy()
                        new_sigs[matches[0][0]] = matches[0][1]
                        changes.extend(((row, 'cache_path', anchor, new_abs), (row, 'mod_signatures', old_sigs, new_sigs)))
    applied_mods = []
    applied_rows = []
    try:
        for mod, old, new in mods:
            applied_mods.append((mod, old))
            mod.filepath = new
        for owner, key, old, new in changes:
            applied_rows.append((owner, key, old))
            owner[key] = new
    except Exception as exc:
        errors = []
        for owner, key, old in reversed(applied_rows):
            try:
                owner[key] = old
            except Exception as error:
                errors.append(str(error))
        for mod, old in reversed(applied_mods):
            try:
                mod.filepath = old
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError(str(exc)+'; path rollback incomplete: '+'; '.join(errors)) from exc
        raise
    return len(mods)
