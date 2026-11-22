"""Reversible viewport authority handoff for verified baked rigid proxies.

The mesh and transform are untouched. Ambiguous or animated modifier authority
is rejected before any state is written.
"""
from . import pc2


def _resolve(scene, row):
    arm = row['arm']; proxy = row['proxy']
    if (arm is None or proxy is None or scene.objects.get(arm.name) != arm
            or scene.objects.get(proxy.name) != proxy or arm.type != 'ARMATURE'):
        raise ValueError('Baked proxy owner is no longer available')
    bone = arm.data.bones.get(row['bone_name'])
    if bone is None:
        raise ValueError('Baked proxy bone is no longer available')
    settings = bone.ppf_bone_physics
    if (not settings.enabled or settings.physical_model != 'RIGID'
            or settings.proxy != proxy or settings.bone_uuid != row['bone_uuid']
            or settings.body_uuid != row['body_uuid']
            or arm.ppf_armature_uuid != row['arm_uuid']
            or proxy.get('ppf_bone_proxy_owned') != True
            or proxy.get('ppf_bone_uuid') != row['bone_uuid']
            or proxy.get('ppf_armature_uuid') != row['arm_uuid']
            or proxy.get('_solver_uuid') != row['body_uuid']):
        raise ValueError('Baked rigid proxy binding changed')
    mod = proxy.modifiers.get(pc2.MODIFIER_NAME)
    if mod is None or mod.type != 'MESH_CACHE' or mod.cache_format != 'PC2':
        raise ValueError('Baked rigid proxy cache changed')
    if mod.filepath != row['filepath'] or pc2._owned_mesh_output_path(proxy, pc2.object_pc2_key_readonly(proxy), mod) != row['owned_path']:
        raise ValueError('Baked rigid proxy cache path changed')
    return proxy, mod


def suspend(scene, bone_entry):
    if not bone_entry or bone_entry.get('single'):
        return []
    rows = []
    from collections import Counter
    counts = Counter(b.ppf_bone_physics.proxy.as_pointer() for obj in scene.objects
                     if obj.type == 'ARMATURE' for b in obj.data.bones
                     if b.ppf_bone_physics.proxy)
    receipt = {(r['armature_uuid'], r['bone_uuid'], r['body_uuid'])
               for r in bone_entry['index']['receipt']['bindings']}
    for snap in bone_entry['snapshots']:
        arm = snap['arm']
        for name in snap['names']:
            settings = arm.data.bones[name].ppf_bone_physics
            proxy = settings.proxy
            if settings.physical_model != 'RIGID' or proxy is None:
                continue
            key = (arm.ppf_armature_uuid, settings.bone_uuid, settings.body_uuid)
            if key not in receipt:
                continue
            mod = proxy.modifiers.get(pc2.MODIFIER_NAME)
            if mod is None or not mod.show_viewport or mod.type != 'MESH_CACHE' or mod.cache_format != 'PC2':
                continue
            owned_path = pc2._owned_mesh_output_path(proxy, pc2.object_pc2_key_readonly(proxy), mod)
            if not owned_path:
                continue
            row = dict(arm=arm, proxy=proxy, bone_name=name, arm_uuid=key[0],
                       bone_uuid=key[1], body_uuid=key[2],
                       view=bool(mod.show_viewport), render=bool(mod.show_render),
                       filepath=mod.filepath, owned_path=owned_path)
            try:
                _resolve(scene, row)
            except (ValueError, AttributeError, ReferenceError):
                continue
            if counts[proxy.as_pointer()] != 1:
                continue
            if any(not c.mute for c in proxy.constraints):
                continue
            if any(m != mod and m.type == 'MESH_CACHE' and m.show_viewport for m in proxy.modifiers):
                continue
            ad = proxy.animation_data
            if ad and (ad.action or ad.drivers or ad.nla_tracks):
                continue
            rows.append(row)
    changed = []
    try:
        for row in rows:
            _, mod = _resolve(scene, row)
            mod.show_viewport = False
            mod.show_render = False
            changed.append(row)
    except BaseException:
        for row in reversed(changed):
            _, mod = _resolve(scene, row)
            mod.show_viewport = row['view']; mod.show_render = row['render']
        raise
    return rows


def validate(scene, rows):
    for row in rows:
        _, mod = _resolve(scene, row)
        if mod.show_viewport or mod.show_render:
            raise ValueError('Baked rigid proxy cache visibility changed')


def restore(scene, rows):
    validate(scene, rows)
    for row in rows:
        _, mod = _resolve(scene, row)
        mod.show_viewport = row['view']; mod.show_render = row['render']


def snapshot(scene, rows):
    validate(scene, rows)
    return [(row['proxy'], row['filepath'], False, False) for row in rows]


