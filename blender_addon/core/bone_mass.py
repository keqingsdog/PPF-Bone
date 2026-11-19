"""Author-volume mass refresh and transactional producer/material gates.

Handlers only mark plain identifiers. Timer resolves live RNA anew, measures
base geometry (Edit BMesh copy included), then writes only uniquely owned groups.
"""
import math
import time
from contextlib import contextmanager
import bpy
from bpy.app.handlers import persistent
from .bone_mass_cache import DirtyMassCache
from .bone_mass_math import resolve_mass, valid_density, valid_authored_density
from .bone_proxy_volume import measure_volume, matrix_determinant, author_geometry_bytes

CACHE = DirtyMassCache()
_indexes = {}
_snapshots = {}
_writing = 0
_inventory_dirty = True
_producer_depth = 0
_identity_rebuilding = False
_modes = None
_settle_until = 0.
DIAGNOSTIC_FIELDS = {'mass_volume_m3', 'mass_measured_kg', 'mass_valid', 'mass_status'}
TIMER_KEY = '_ppf_bone_mass_timer'


@contextmanager
def material_write_guard():
    global _writing
    _writing += 1
    try: yield
    finally: _writing -= 1


def writing_material(): return bool(_writing)


def cached_status(scene, arm, bone):
    """Scalar cache-only panel query; no evaluation, geometry scan or writes."""
    if arm is None or bone is None: return False, '请先选择骨骼'
    if _identity_rebuilding or arm.mode == 'EDIT' or time.monotonic() < _settle_until:
        return False, '等待骨架编辑完成后刷新质量'
    s = bone.ppf_bone_physics
    if not CACHE.clean(_key(scene, arm, bone)): return False, '等待作者体积刷新'
    return bool(s.mass_valid), s.mass_status


def _key(scene, arm, bone):
    if _identity_rebuilding or arm.mode == 'EDIT':
        raise RuntimeError('质量读取等待骨架数据重建完成')
    s = bone.ppf_bone_physics; p = s.proxy
    def token(block): return int(block.as_pointer()) if block is not None else 0
    return (scene.name, arm.name, arm.data.name, bone.name, s.bone_uuid,
            s.body_uuid, s.group_uuid, p.name if p else '', p.data.name if p else '',
            token(scene), token(arm), token(arm.data), token(p), token(p.data if p else None))


def _metadata_snapshot(settings):
    p = settings.proxy
    return (settings.enabled, settings.mass_mode, float(settings.target_mass_kg),
            float(settings.density), settings.physical_model,
            tuple(float(v) for row in p.matrix_world for v in row))


def _resolve(key):
    scene = bpy.data.scenes.get(key[0]); arm = bpy.data.objects.get(key[1])
    if scene is None or arm is None or arm.type != 'ARMATURE' or arm.mode == 'EDIT' or scene.objects.get(arm.name) != arm: return None
    bone = arm.data.bones.get(key[3])
    return bone if bone is not None and _key(scene, arm, bone) == key else None


def _live():
    rows = {}
    for scene in bpy.data.scenes:
        for arm in scene.objects:
            if arm.type != 'ARMATURE' or arm.mode == 'EDIT': continue
            for bone in arm.data.bones:
                s = bone.ppf_bone_physics
                if s.proxy is not None: rows[_key(scene, arm, bone)] = (scene, arm, bone)
    return rows


def invalidate_all():
    global _inventory_dirty
    _inventory_dirty = True
    now = time.monotonic()
    for key in CACHE.entries: CACHE.dirty(key, now)


def mark_settings(settings, _context=None):
    global _inventory_dirty
    if _writing: return
    _inventory_dirty = True
    now = time.monotonic(); name = settings.id_data.name
    for key in _indexes.get(('ARMATURE', name), ()): CACHE.dirty(key, now, geometry=False)


@persistent
def mass_depsgraph_updated(_scene, depsgraph):
    global _inventory_dirty
    if _writing or not _stable_mass_context(): return
    now = time.monotonic()
    for update in depsgraph.updates:
        item = update.id.original
        kind = 'OBJECT' if isinstance(item, bpy.types.Object) else 'MESH' if isinstance(item, bpy.types.Mesh) else 'ARMATURE' if isinstance(item, bpy.types.Armature) else 'SCENE' if isinstance(item, bpy.types.Scene) else None
        if kind is None: continue
        for key in _indexes.get((kind, item.name), ()):
            if kind == 'MESH':
                CACHE.dirty(key, now, geometry=True)
            elif kind == 'OBJECT' and item.type == 'MESH':
                CACHE.dirty(key, now, geometry=bool(update.is_updated_geometry))
            elif kind == 'SCENE':
                CACHE.dirty(key, now, geometry=False)
            else:
                # Bone diagnostic RNA and playback can tag the armature too.
                # Compare only scalar author inputs + proxy transform, so our
                # own metric publication cannot re-trigger a geometry scan.
                bone = _resolve(key)
                if bone is None or _metadata_snapshot(bone.ppf_bone_physics) != _snapshots.get(key): CACHE.dirty(key, now, geometry=False)
        if kind in ('OBJECT', 'ARMATURE'): _inventory_dirty = True


@persistent
def mass_context_reset(*_args):
    global _inventory_dirty, _identity_rebuilding, _modes, _settle_until
    CACHE.entries.clear(); _indexes.clear(); _snapshots.clear(); _inventory_dirty = True
    _identity_rebuilding = False
    _modes = None
    _settle_until = time.monotonic() + .2


@persistent
def mass_before_identity_rebuild(*_args):
    """Do not dereference Bone property pointers while Undo/load replaces RNA."""
    global _identity_rebuilding
    _identity_rebuilding = True
    CACHE.entries.clear(); _indexes.clear(); _snapshots.clear()


def _stable_mass_context():
    global _modes, _settle_until
    if _identity_rebuilding: return False
    # Object/Armature IDs survive ordinary mode transitions; Bone RNA does not.
    # Observe only IDs/modes first, and reacquire Bone/PG handles on a later tick.
    modes = tuple((arm.as_pointer(), arm.data.as_pointer(), arm.mode)
                  for arm in bpy.data.objects if arm.type == 'ARMATURE')
    now = time.monotonic()
    if modes != _modes:
        _modes = modes
        _settle_until = now + .2
        CACHE.entries.clear(); _indexes.clear(); _snapshots.clear()
        return False
    return now >= _settle_until and not any(row[2] == 'EDIT' for row in modes)


def _owned(scene, arm, bone):
    from ..models.bone_bindings import _owned_group, validate_proxy_owner
    from ..models.groups import iterate_active_object_groups
    from ..ops.bone_proxy import _owner_animation_paths
    s = bone.ppf_bone_physics
    validate_proxy_owner(arm, bone)
    if s.proxy is None or scene.objects.get(s.proxy.name) != s.proxy:
        raise ValueError(f'{bone.name}：请先创建本场景的物理代理')
    kind = 'SOLID' if s.physical_model == 'SOFT' else 'PDRD'
    group = _owned_group(s, kind)
    if group is None or sum(g.uuid == s.group_uuid for g in iterate_active_object_groups(scene)) != 1:
        raise ValueError(f'{bone.name}：代理或材质被共享、失配；请使用独立副本')
    prefix = group.path_from_id()+'.'
    paths = [path for block, label in ((scene, '场景'), (arm.data, '骨架数据'), (arm, '骨架对象'))
             for path in _owner_animation_paths(block, label)]
    bone_prefix = bone.path_from_id()+'.ppf_bone_physics.'
    if any(path.startswith(prefix) or path.startswith(bone_prefix) for path in paths):
        raise ValueError(f'{bone.name}：带动画或驱动的材质不能自动同步质量')
    return group


def _measure(scene, settings, mesh=None, matrix=None):
    if float(scene.unit_settings.scale_length) != 1.:
        raise ValueError('骨骼质量使用 SI 单位；场景单位缩放必须为 1')
    proxy = settings.proxy
    if mesh is None and proxy is None: raise ValueError('请先创建物理代理，再计算质量')
    if mesh is not None: return measure_volume(mesh, matrix, edit_mesh=False)
    return measure_volume(proxy.data, proxy.matrix_world, edit_mesh=proxy.mode == 'EDIT')


def _reuse_geometry(scene, settings, geometry):
    """Positive affine transforms preserve shell intersections and orientation.

    Only a clean geometry generation can use this proof. Matrix validity is
    checked anew; zero/reflected/nonfinite transforms still refuse immediately.
    """
    from .bone_mass_math import positive_finite
    started = time.perf_counter()
    if float(scene.unit_settings.scale_length) != 1.:
        raise ValueError('骨骼质量使用 SI 单位；场景单位缩放必须为 1')
    determinant = matrix_determinant(settings.proxy.matrix_world)
    result = dict(geometry)
    result.update(volume_m3=positive_finite(geometry['local_volume_m3']*determinant, '缩放后体积（m³）'),
                  world_determinant=determinant, reused_validated_local_geometry=True,
                  elapsed_seconds=time.perf_counter()-started)
    return result


def _diagnostics(settings, result=None, error=None):
    with material_write_guard():
        if settings.mass_valid != (error is None): settings.mass_valid = error is None
        if error is None:
            for key, value in (('mass_volume_m3', result['volume_m3']), ('mass_measured_kg', result['mass_kg'])):
                if not math.isclose(float(getattr(settings, key)), value, rel_tol=2e-7, abs_tol=1e-30): setattr(settings, key, value)
            status = '质量已更新；作者基础网格体积，含对象缩放'
            if settings.mass_status != status: settings.mass_status = status
        else:
            # Last valid numbers are retained for diagnostics, but the panel
            # hides them whenever mass_valid is false. No invented mass.
            if settings.mass_status != str(error): settings.mass_status = str(error)


def _density_plan(settings, group, density, *, authored=False):
    """Preserve true E and the stored convention; never resolve stale authors."""
    from .bone_preset_apply import _validate_rna
    from ..models.bone_bindings import effective_solid_material
    density = valid_authored_density(density) if authored else valid_density(density)
    _validate_rna(settings, 'density', density)
    key = 'solid_density' if group.object_type == 'SOLID' else 'pdrd_density'
    _validate_rna(group, key, density)
    # Compute the stored E/rho from the density RNA will actually retain.
    import struct
    density = struct.unpack('<f', struct.pack('<f', density))[0]
    writes = {key: density}; author = {}
    if group.object_type == 'SOLID':
        effective = effective_solid_material(settings, group)
        for field, actual in effective.items():
            if settings.is_property_set(field):
                value = getattr(settings, field)
                equal = math.isclose(value, actual, rel_tol=2e-6, abs_tol=1e-9) if isinstance(actual, float) else value == actual
                if not equal: raise ValueError('旧作者值与真实材质不一致；请明确处理后重试')
        young = effective['solid_young_modulus']
        _validate_rna(settings, 'solid_young_modulus', young)
        stored = young / density if group.young_mod_density_normalized else young
        _validate_rna(group, 'solid_young_modulus', stored)
        writes['solid_young_modulus'] = stored
        author = effective
    return writes, author


def _apply_density(settings, group, density, *, authored=False):
    writes, author = _density_plan(settings, group, density, authored=authored)
    old_s = _property_snapshot(settings, ('density', *author))
    old_g = _property_snapshot(group, writes)
    with material_write_guard():
        try:
            settings.density = next(iter(writes.values()))
            for key, value in writes.items(): setattr(group, key, value)
            # RNA stores binary32. Synchronize authors from the committed values.
            from ..models.bone_bindings import effective_solid_material
            if author:
                for key, value in effective_solid_material(settings, group).items(): setattr(settings, key, value)
        except BaseException:
            _restore_properties(settings, old_s); _restore_properties(group, old_g)
            raise


def mass_timer():
    global _indexes, _inventory_dirty
    try:
        if not _stable_mass_context(): return .1
        rows = _live(); now = time.monotonic(); CACHE.retain(rows)
        for key in set(_snapshots)-set(rows): del _snapshots[key]
        indexes = {}
        for key, (scene, arm, bone) in rows.items():
            s = bone.ppf_bone_physics; proxy = s.proxy
            for ident in (('SCENE', scene.name), ('OBJECT', arm.name), ('OBJECT', proxy.name), ('MESH', proxy.data.name), ('ARMATURE', arm.data.name)):
                indexes.setdefault(ident, set()).add(key)
            if key not in CACHE.entries: CACHE.dirty(key, now)
        _indexes = indexes; _inventory_dirty = False
        # One complete body per tick bounds multi-body authoring work. A failed
        # generation is cached too, and won't rescan until another edit/reset.
        for key in CACHE.ready(now)[:1]:
            scene, arm, bone = rows[key]; s = bone.ppf_bone_physics
            entry = CACHE.entries[key]; measured_geometry = None
            signature = None
            try:
                group = _owned(scene, arm, bone)
                # Evaluated modifier/PC2 updates can mark an Object's geometry
                # dirty while its author base mesh remains byte-identical.
                # Check exact bounded base bytes in the timer, never the handler
                # or draw. True Mesh/Edit changes cannot reuse the prior proof.
                signature = author_geometry_bytes(s.proxy.data, edit_mesh=s.proxy.mode == 'EDIT')
                if entry['geometry'] is not None and signature == entry['geometry_signature']:
                    entry['geometry_dirty'] = False
                    result = _reuse_geometry(scene, s, entry['geometry'])
                else:
                    result = _measure(scene, s)
                    if signature != author_geometry_bytes(s.proxy.data, edit_mesh=s.proxy.mode == 'EDIT'):
                        raise ValueError('作者网格在测量期间改变；请等待刷新')
                    measured_geometry = dict(result)
                density, mass = resolve_mass(s.mass_mode, s.density, s.target_mass_kg, result['volume_m3'])
                if s.mass_mode == 'MASS': _apply_density(s, group, density)
                result['mass_kg'] = mass; _diagnostics(s, result)
                CACHE.store(key, dict(result), geometry=measured_geometry, geometry_signature=signature)
            except (ValueError, RuntimeError, ReferenceError) as error:
                _diagnostics(s, error=error); CACHE.store(key, {'error': str(error)}, geometry=measured_geometry, geometry_signature=signature)
            _snapshots[key] = _metadata_snapshot(s)
    except (ReferenceError, RuntimeError):
        mass_context_reset()
    return .1


def prepare_mass_batch(scene, overrides=()):
    """Read-only FULL MASS batch. Overrides describe disposable planned meshes.

    Rows are (arm, bone, mesh, matrix, planned_density), including newly created
    selected proxies. Ephemeral RNA exists only within this synchronous call.
    """
    overrides = {(arm.name, bone.name): (arm, bone, mesh, matrix, density)
                 for arm, bone, mesh, matrix, density in overrides}
    rows = {(arm.name, bone.name): (arm, bone, None, None, None)
            for arm in scene.objects if arm.type == 'ARMATURE'
            for bone in arm.data.bones if bone.ppf_bone_physics.enabled}
    rows.update(overrides); prepared = []
    if rows and float(scene.unit_settings.scale_length) != 1.:
        raise ValueError('骨骼质量使用 SI 单位；场景单位缩放必须为 1')
    for arm, bone, mesh, matrix, planned_density in rows.values():
        s = bone.ppf_bone_physics
        density = valid_authored_density(float(s.density)) if planned_density is None else valid_density(planned_density)
        if s.mass_mode == 'DENSITY': continue
        if s.mass_mode != 'MASS': raise ValueError(f'{bone.name}：未知的质量模式')
        # Existing bindings must own one material. New bodies are validated by
        # the proxy transaction and receive their group only after this batch.
        group = _owned(scene, arm, bone) if s.proxy else None
        result = _measure(scene, s, mesh, matrix)
        density, mass = resolve_mass(s.mass_mode, density, s.target_mass_kg, result['volume_m3'])
        if group is not None: _density_plan(s, group, density)
        prepared.append((arm, bone, group, density, dict(result, mass_kg=mass)))
    return prepared


def apply_mass_batch(prepared, scene):
    # Resolve ownership again before the first write, so stale/shared targets
    # can never cause a partially applied batch.
    resolved = [(bone.ppf_bone_physics, _owned(scene, arm, bone), rho, result)
                for arm, bone, _group, rho, result in prepared]
    for settings, group, rho, result in resolved: _density_plan(settings, group, rho)
    snapshots = [(settings, group, _property_snapshot(settings, _TRANSACTION_FIELDS),
                  _property_snapshot(group, ('solid_density', 'solid_young_modulus') if group.object_type == 'SOLID' else ('pdrd_density',)))
                 for settings, group, _rho, _result in resolved]
    try:
        for settings, group, rho, result in resolved:
            _apply_density(settings, group, rho); _diagnostics(settings, result)
    except BaseException:
        with material_write_guard():
            for settings, group, authors, material in snapshots:
                _restore_properties(settings, authors); _restore_properties(group, material)
        raise


def producer_preflight(context, *, commit=False):
    if _producer_depth: return
    prepared = prepare_mass_batch(context.scene)
    if commit: apply_mass_batch(prepared, context.scene)
    else:
        # Hash/read-only paths refuse a stale derived density. The timer or an
        # explicit producer commit will synchronize it; hash polling never writes.
        for _arm, bone, group, density, _result in prepared:
            s = bone.ppf_bone_physics
            actual = group.solid_density if group.object_type == 'SOLID' else group.pdrd_density
            if not math.isclose(float(s.density), density, rel_tol=2e-7) or actual != s.density:
                raise ValueError(f'{bone.name}：质量密度尚未同步，请等待刷新或重新提交')


_TRANSACTION_FIELDS = ('solid_young_modulus', 'solid_model', 'solid_poisson_ratio', 'solid_deformation_damping', 'density', 'mass_volume_m3', 'mass_measured_kg', 'mass_valid', 'mass_status')


def _property_snapshot(owner, names):
    return [(name, getattr(owner, name), owner.is_property_set(name)) for name in names]


def _restore_properties(owner, values):
    for name, value, was_set in values:
        if was_set: setattr(owner, name, value)
        else: owner.property_unset(name)


def _mass_transaction_snapshot(scene, prepared):
    rows=[]
    for arm, bone, group, _rho, _result in prepared:
        settings=bone.ppf_bone_physics
        density_field='solid_density' if group.object_type=='SOLID' else 'pdrd_density'
        rows.append((scene.name, arm.name, arm.ppf_armature_uuid, bone.name,
                     settings.bone_uuid, settings.body_uuid, settings.group_uuid,
                     _property_snapshot(settings, _TRANSACTION_FIELDS),
                     _property_snapshot(group, (density_field, 'solid_young_modulus') if group.object_type=='SOLID' else (density_field,))))
    return rows


def _rollback_mass_transaction(rows):
    # Encoders may enter Edit mode. Store identifiers/values, then reacquire
    # live Bone/PG handles instead of retaining stale RNA across the yield.
    resolved=[]
    for scene_name, arm_name, arm_uuid, bone_name, bone_uuid, body_uuid, group_uuid, settings_values, group_values in rows:
        scene=bpy.data.scenes.get(scene_name)
        arm=scene.objects.get(arm_name) if scene else None
        bone=arm.data.bones.get(bone_name) if arm and arm.type=='ARMATURE' else None
        if bone is None or arm.ppf_armature_uuid!=arm_uuid:
            raise RuntimeError('质量事务回滚无法确认原骨架')
        settings=bone.ppf_bone_physics
        if (settings.bone_uuid, settings.body_uuid, settings.group_uuid)!=(bone_uuid, body_uuid, group_uuid):
            raise RuntimeError('质量事务回滚无法确认原绑定身份')
        group=_owned(scene, arm, bone)
        resolved.append((settings, group, settings_values, group_values))
    with material_write_guard():
        for settings, group, settings_values, group_values in resolved:
            _restore_properties(settings, settings_values)
            _restore_properties(group, group_values)


@contextmanager
def producer_mass_transaction(context):
    global _producer_depth
    if _producer_depth:
        yield
        return
    prepared=prepare_mass_batch(context.scene)
    snapshot=_mass_transaction_snapshot(context.scene, prepared)
    _producer_depth += 1
    try:
        apply_mass_batch(prepared, context.scene)
        del prepared
        yield
    except BaseException:
        _rollback_mass_transaction(snapshot)
        raise
    finally:
        _producer_depth -= 1


def register():
    mass_context_reset()
    old = bpy.app.driver_namespace.get(TIMER_KEY)
    if old is not None and bpy.app.timers.is_registered(old): bpy.app.timers.unregister(old)
    handlers = ((bpy.app.handlers.depsgraph_update_post, mass_depsgraph_updated),
                (bpy.app.handlers.load_pre, mass_before_identity_rebuild),
                (bpy.app.handlers.undo_pre, mass_before_identity_rebuild),
                (bpy.app.handlers.redo_pre, mass_before_identity_rebuild),
                (bpy.app.handlers.load_post, mass_context_reset),
                (bpy.app.handlers.undo_post, mass_context_reset),
                (bpy.app.handlers.redo_post, mass_context_reset))
    for collection, callback in handlers:
        for old in list(collection):
            if old.__name__ == callback.__name__: collection.remove(old)
        collection.append(callback)
    if not bpy.app.timers.is_registered(mass_timer): bpy.app.timers.register(mass_timer, first_interval=.1, persistent=True)
    bpy.app.driver_namespace[TIMER_KEY] = mass_timer


def unregister():
    if bpy.app.timers.is_registered(mass_timer): bpy.app.timers.unregister(mass_timer)
    old = bpy.app.driver_namespace.pop(TIMER_KEY, None)
    if old is not None and old is not mass_timer and bpy.app.timers.is_registered(old): bpy.app.timers.unregister(old)
    for collection in (bpy.app.handlers.depsgraph_update_post, bpy.app.handlers.load_pre, bpy.app.handlers.undo_pre, bpy.app.handlers.redo_pre, bpy.app.handlers.load_post, bpy.app.handlers.undo_post, bpy.app.handlers.redo_post):
        for callback in list(collection):
            if callback.__name__ in ('mass_depsgraph_updated', 'mass_context_reset', 'mass_before_identity_rebuild'): collection.remove(callback)
    mass_context_reset()
