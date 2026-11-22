"""Proof of the animation dependencies owned by a recovery participant."""
import json
import bpy


def _snapshot(owner, action_signature, expected_refs=()):
    refs = []

    def properties(item, skip=()):
        out = {}
        for prop in item.bl_rna.properties:
            name = prop.identifier
            if name in {'rna_type', 'action_suitable_slots'} or name in skip:
                continue
            if prop.type == 'COLLECTION':
                raise ValueError('动画依赖含未支持的集合 %s；请保留原动画。' % name)
            if prop.is_readonly:
                continue
            value = getattr(item, name)
            if prop.type == 'POINTER':
                if value is None:
                    out[name] = None
                elif isinstance(value, bpy.types.ID):
                    expected = expected_refs[len(refs)].get('action_signature') if len(refs) < len(expected_refs) else None
                    signature, action_refs = action_signature(value, expected=expected) if isinstance(value, bpy.types.Action) else ('', [])
                    out[name] = len(refs)
                    refs.append({'id': value, 'action_signature': signature, 'action_refs': action_refs})
                elif name == 'action_slot' and hasattr(value, 'handle') and hasattr(value, 'identifier'):
                    out[name] = {'handle': value.handle, 'identifier': value.identifier}
                else:
                    raise ValueError('动画依赖含未支持的引用；请保留原动画。')
            elif getattr(prop, 'is_array', False):
                out[name] = list(value)
            elif isinstance(value, set):
                out[name] = sorted(value)
            elif isinstance(value, (bool, int, float, str)):
                out[name] = value
            else:
                raise ValueError('动画依赖属性无法确认；请保留原动画。')
        return out

    def curves(items):
        from .single_bake_restore import _curve_signatures
        return _curve_signatures(items, strict=True)

    def strip_data(strip, ancestors=()):
        identity = strip.as_pointer() if hasattr(strip, 'as_pointer') else id(strip)
        if identity in ancestors:
            raise ValueError('NLA嵌套包含循环；请保留原动画。')
        if len(ancestors) >= 256:
            raise ValueError('NLA嵌套过深；请保留原动画。')
        modifiers = []
        for mod in strip.modifiers:
            row = properties(mod, {'control_points'})
            if mod.type == 'ENVELOPE':
                row['control_points'] = [[p.frame, p.min, p.max] for p in mod.control_points]
            modifiers.append([mod.type, row])
        children = [strip_data(child, ancestors + (identity,)) for child in getattr(strip, 'strips', ())]
        return [properties(strip, {'strips', 'fcurves', 'modifiers', 'select'}), curves(strip.fcurves), modifiers, children]

    ad = owner.animation_data
    if ad is None:
        return json.dumps([{}, [], []], sort_keys=True, ensure_ascii=False), refs
    drivers = []
    for fc in ad.drivers:
        driver = fc.driver
        variables = []
        for variable in driver.variables:
            variables.append([properties(variable, {'targets'}),
                              [properties(target) for target in variable.targets]])
        drivers.append([curves([fc]), properties(driver, {'variables'}), variables])
    tracks = [[properties(track, {'strips', 'select'}), [strip_data(strip) for strip in track.strips]]
              for track in ad.nla_tracks]
    # Current action is replaced as part of baking; it is proven separately.
    settings = properties(ad, {'action', 'action_slot', 'action_slot_handle', 'last_slot_identifier', 'nla_tracks', 'drivers'})
    # Creating AnimData solely to attach the working Action is expected. Treat
    # its default settings like absent AnimData, while retaining nondefaults.
    for name in list(settings):
        prop = ad.bl_rna.properties.get(name)
        if settings[name] is None or (prop is not None and hasattr(prop, 'default') and settings[name] == prop.default):
            del settings[name]
    state = [settings, drivers, tracks]
    return json.dumps(state, sort_keys=True, ensure_ascii=False), refs


def _memoized_action_signature(action_signature):
    cache = {}
    def snapshot(action, *, expected=None):
        key=(action.as_pointer(), bool(expected and not expected.startswith('action-v3:')))
        if key not in cache:
            cache[key]=action_signature(action, expected=expected)
        return cache[key]
    return snapshot


def capture(owners, action_signature):
    action_signature = _memoized_action_signature(action_signature)
    rows = []
    for owner in owners:
        if owner is None or any(row['owner'] == owner for row in rows):
            continue
        state, refs = _snapshot(owner, action_signature)
        rows.append({'owner': owner, 'state': state, 'refs': refs})
    return {'schema': 2, 'owners': rows}


def _same_ref_rows(actual, expected):
    from .single_bake_restore import _source_refs_equal
    return len(actual)==len(expected) and all(a.get('id') is not None and a.get('id')==b.get('id') and a.get('action_signature')==b.get('action_signature') and _source_refs_equal(a.get('action_refs',[]),b.get('action_refs',[])) for a,b in zip(actual,expected))


def validate(proof, action_signature):
    action_signature = _memoized_action_signature(action_signature)
    if proof.get('schema') != 2:
        raise ValueError('恢复记录缺少动画依赖证明；请保留原动画。')
    for row in proof.get('owners', []):
        owner = row.get('owner')
        if owner is None:
            raise ValueError('恢复动画依赖对象已删除。')
        old_refs = row.get('refs', [])
        state, refs = _snapshot(owner, action_signature, old_refs)
        if state != row.get('state') or len(refs) != len(old_refs):
            raise ValueError('恢复动画驱动或NLA已改变；未恢复。')
        from .single_bake_restore import _source_refs_equal
        if any(a['id'] != b.get('id') or a['action_signature'] != b.get('action_signature') or not _source_refs_equal(a['action_refs'],b.get('action_refs',[]))
               for a, b in zip(refs, old_refs)):
            raise ValueError('恢复动画依赖引用已改变；未恢复。')
