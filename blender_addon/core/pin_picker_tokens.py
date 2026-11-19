"""Process-local native Enum event tokens. Not persistent Pin/VG identity.

Numbers and retained strings are never recycled within this Blender process.
Unsupported external mutation of the private anchor is not a supported lifecycle.
"""
import builtins
import threading

_ANCHOR = '_ppf_native_pin_enum_tokens_v1'
_SCHEMA = 1
MAX_TOKENS = 16384
MAX_STRINGS = 65536
MAX_STRING_BYTES = 8 * 1024 * 1024
MAX_NUMBER = 2147483647
NONE = ('NONE', 'None', 'No proven selection; select a vertex group', 0)
_HANDLER_TAG = '_ppf_pin_tokens_handler_v1'


def _anchor():
    if threading.current_thread() is not threading.main_thread():
        raise ValueError('Pin picker requires the Blender main thread')
    a = getattr(builtins, _ANCHOR, None)
    if a is None:
        a = {'schema': _SCHEMA, 'next': 1, 'epoch': 1, 'records': {},
             'groups': [], 'strings': {}, 'string_bytes': 0, 'blocked': '',
             'enabled': False, 'none': NONE}
        setattr(builtins, _ANCHOR, a)
    if not isinstance(a, dict) or a.get('schema') != _SCHEMA:
        raise ValueError('Pin picker process registry is incompatible')
    return a


def none_item():
    try:
        return _anchor()['none']
    except Exception:
        return NONE


def _available(a):
    if not a['enabled'] or a['blocked']:
        raise ValueError(a['blocked'] or 'Pin picker is not registered')


def block(reason):
    try:
        _anchor()['blocked'] = str(reason)
    except Exception:
        pass


def invalidate_all(reason='lifecycle'):
    a = _anchor()
    a['epoch'] += 1
    a['records'].clear()
    a['groups'].clear()
    # Counter, strings, resource budgets and a safety failure are not reset.


def _same_live(a, b):
    try:
        return bool(a.as_pointer() and a.as_pointer() == b.as_pointer() and a == b)
    except (ReferenceError, RuntimeError, TypeError):
        return False


def _group_key(group):
    import bpy
    from ..models.groups import iterate_object_groups
    scene = group.id_data
    if not any(_same_live(scene, s) for s in bpy.data.scenes):
        raise ValueError('Pin group Scene is unavailable')
    uid = group.uuid
    generation = group.get('_lifecycle_generation', 0)
    if not uid or type(generation) is not int:
        raise ValueError('Pin group lifetime is unavailable')
    matches = [g for g in iterate_object_groups(scene) if g.uuid == uid]
    if len(matches) != 1 or not _same_live(matches[0], group):
        raise ValueError('Pin group lifetime is ambiguous or stale')
    return scene, uid, generation


def retire_group(group):
    a = _anchor()
    keep = []
    for entry in a['groups']:
        if _same_live(entry['group'], group):
            for number in entry['tokens'].values():
                a['records'].pop(number, None)
        else:
            keep.append(entry)
    a['groups'] = keep


def _current(group, rows):
    a = _anchor(); _available(a)
    scene, uid, generation = _group_key(group)
    entry = None
    keep = []
    for old in a['groups']:
        if _same_live(old['group'], group):
            if (old['uid'] == uid and old['generation'] == generation
                    and _same_live(old['scene'], scene) and old['epoch'] == a['epoch']):
                entry = old; keep.append(old)
            else:
                for number in old['tokens'].values():
                    a['records'].pop(number, None)
        else:
            keep.append(old)
    a['groups'] = keep
    if entry is None:
        if len(a['groups']) >= MAX_TOKENS:
            a['blocked'] = 'Pin picker session group capacity exhausted'
            raise ValueError(a['blocked'])
        entry = {'group': group, 'scene': scene, 'uid': uid, 'generation': generation,
                 'epoch': a['epoch'], 'tokens': {}}
        a['groups'].append(entry)
    live = {(uid, name): obj for obj, uid, name, label in rows}
    for identity, number in list(entry['tokens'].items()):
        record = a['records'].get(number)
        if (identity not in live or record is None
                or not _same_live(record['object'], live[identity])):
            a['records'].pop(number, None); del entry['tokens'][identity]
    return a, entry


def _prepare(group, rows):
    a, entry = _current(group, rows)
    new = [(obj, uid, name, label) for obj, uid, name, label in rows
           if (uid, name) not in entry['tokens']]
    text = set()
    for obj, uid, name, label in rows:
        text.update((label, f'Vertex group {name} of {obj.name}'))
    missing = text.difference(a['strings'])
    size = sum(len(s.encode('utf8')) for s in missing)
    if (a['next'] + len(new) - 1 > min(MAX_TOKENS, MAX_NUMBER)
            or len(a['strings']) + len(missing) > MAX_STRINGS
            or a['string_bytes'] + size > MAX_STRING_BYTES):
        a['blocked'] = 'Pin picker session capacity exhausted; use a new Blender process'
        raise ValueError(a['blocked'])
    for s in missing:
        a['strings'][s] = s
    a['string_bytes'] += size
    for obj, uid, name, label in new:
        number = a['next']; a['next'] += 1
        a['records'][number] = {'entry': entry, 'object': obj, 'identity': (uid, name)}
        entry['tokens'][(uid, name)] = number
    return a, entry


def enum_items(group, rows):
    a, entry = _prepare(group, rows)
    out = [a['none']]
    for obj, uid, name, label in rows:
        label = a['strings'][label]
        description = a['strings'][f'Vertex group {name} of {obj.name}']
        out.append((label, label, description, entry['tokens'][(uid, name)]))
    return out


def number_for(group, rows, identity):
    a, entry = _prepare(group, rows)
    return entry['tokens'].get(identity, 0)


def authenticate(group, rows, number):
    if type(number) is not int or not 1 <= number <= MAX_NUMBER:
        raise ValueError('Pin picker event number is invalid')
    a, entry = _current(group, rows)
    record = a['records'].get(number)
    if record is None or record['entry'] is not entry:
        raise ValueError('Pin picker selection is stale; select the vertex group again')
    uid, name = record['identity']
    found = [(obj, vg) for obj, row_uid, vg, label in rows
             if (row_uid, vg) == (uid, name) and _same_live(record['object'], obj)]
    if len(found) != 1:
        raise ValueError('Pin picker Object or vertex group is unavailable')
    return uid, name


def begin_registration():
    a = _anchor()
    a['enabled'] = False
    invalidate_all('register entry')


def register():
    import bpy
    from bpy.app.handlers import persistent
    a = _anchor()
    invalidate_all('register')
    lists = (bpy.app.handlers.load_pre, bpy.app.handlers.load_post,
             bpy.app.handlers.undo_pre, bpy.app.handlers.undo_post,
             bpy.app.handlers.redo_pre, bpy.app.handlers.redo_post)
    for callbacks in lists:
        for fn in list(callbacks):
            if getattr(fn, _HANDLER_TAG, False):
                callbacks.remove(fn)
    @persistent
    def retire(*args):
        try:
            invalidate_all('native lifecycle')
        except Exception as exc:
            block('Pin picker lifecycle invalidation failed: ' + str(exc))
            print('Pin picker lifecycle invalidation failed:', exc)
    setattr(retire, _HANDLER_TAG, True)
    for callbacks in lists:
        callbacks.insert(0, retire)
    a['enabled'] = True


def unregister():
    import bpy
    a = _anchor(); a['enabled'] = False
    invalidate_all('unregister')
    for callbacks in (bpy.app.handlers.load_pre, bpy.app.handlers.load_post,
                      bpy.app.handlers.undo_pre, bpy.app.handlers.undo_post,
                      bpy.app.handlers.redo_pre, bpy.app.handlers.redo_post):
        for fn in list(callbacks):
            if getattr(fn, _HANDLER_TAG, False):
                callbacks.remove(fn)
