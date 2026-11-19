"""Explicit, transactional duplication of an authored rig and its owned bodies.

Blender Object Duplicate deliberately does not perform this ownership transfer.
This operation copies author data only; solver output must first be cleared.
"""
from copy import deepcopy
import json
import uuid

import bpy
from .. import i18n


def _uuid():
    return str(uuid.uuid4())


def _author_material_plan(settings, group):
    """Preflight the actual group values before allocating a copied rig."""
    import struct
    from ..core.bone_preset_apply import _validate_rna
    from ..models.bone_bindings import effective_solid_material

    values = {'density': float(group.solid_density if group.object_type == 'SOLID' else group.pdrd_density)}
    if group.object_type == 'SOLID':
        values.update(effective_solid_material(settings, group))
    for key, value in values.items():
        try:
            if settings.bl_rna.properties[key].type == 'FLOAT':
                # RNA stores these fields as float32. Accept normal rounding
                # at an endpoint, but never a value requiring range clamping.
                value = struct.unpack('f', struct.pack('f', value))[0]
            _validate_rna(settings, key, value)
        except (ValueError, OverflowError) as exc:
            raise ValueError('实际材质无法由骨骼作者字段表示：' + key + '；整架未复制') from exc
        values[key] = value
    return values


def _remap_snapshot(snapshot, ids, objects):
    """Keep explicit/unset RNA state, including all nested author collections."""
    result = {}
    for key, (was_set, (kind, value)) in snapshot.items():
        if kind == 'GROUP':
            value = _remap_snapshot(value, ids, objects)
        elif kind == 'COLLECTION':
            value = [_remap_snapshot(row, ids, objects) for row in value]
        elif isinstance(value, str):
            value = ids.get(value, value)
            # Pin labels are display caches, but must also name the copied
            # object when the existing pin decoder consults them.
            if value.startswith('['):
                from ..models.groups import decode_vertex_group_identifier, encode_vertex_group_identifier
                name, vertex_group = decode_vertex_group_identifier(value)
                copied = next((dst for src, dst in objects.items() if src.name == name), None)
                if copied is not None:
                    value = encode_vertex_group_identifier(copied.name, vertex_group)
        elif isinstance(value, bpy.types.ID):
            value = objects.get(value, value)
        result[key] = (was_set, (kind, value))
    return result


def _pin_copy_group_snapshot(snapshot, plans, objects, ids):
    """Remap verified Pin owners without interpreting arbitrary author strings."""
    from ..core.pin_reference import encode_pin_pair, PinReferenceError
    from ..core.pin_picker_identity import FIELD, METADATA_KEYS
    source = dict(snapshot)
    picker = source.pop(FIELD, None)
    if picker is not None:
        try:
            _was_set, (kind, metadata) = picker
        except (TypeError, ValueError) as exc:
            raise PinReferenceError('Invalid copied picker snapshot structure') from exc
        if kind != 'PIN_PICKER_METADATA' or not isinstance(metadata, dict) or set(metadata) - set(METADATA_KEYS):
            raise PinReferenceError('Invalid copied picker metadata schema')
    try:
        was_set, (kind, rows) = source.pop('pin_vertex_groups')
    except (KeyError, TypeError, ValueError) as exc:
        raise PinReferenceError('Invalid copied Pin collection snapshot') from exc
    if kind != 'COLLECTION' or not isinstance(rows, list) or len(rows) != len(plans):
        raise PinReferenceError('Copied Pin plan no longer matches snapshot')
    copied_rows = []
    for row, (owner, owner_uuid, vg_name) in zip(rows, plans):
        if not isinstance(row, dict) or owner not in objects or owner_uuid not in ids:
            raise PinReferenceError('Copied Pin owner is unavailable')
        isolated = dict(row)
        for key in ('name', 'object_uuid'):
            try:
                _set, (_kind, value) = isolated.pop(key)
            except (KeyError, TypeError, ValueError) as exc:
                raise PinReferenceError('Invalid stored Pin identity snapshot') from exc
            if _kind != 'VALUE' or not isinstance(value, str):
                raise PinReferenceError('Invalid stored Pin identity value')
        target = objects[owner]
        if target.type != 'MESH' or target.vertex_groups.get(vg_name) is None:
            raise PinReferenceError('Copied Pin vertex group is missing')
        copied = _remap_snapshot(isolated, ids, objects)
        copied['name'] = (True, ('VALUE', encode_pin_pair(target.name, vg_name)))
        copied['object_uuid'] = (True, ('VALUE', ids[owner_uuid]))
        copied_rows.append(copied)
    result = _remap_snapshot(source, ids, objects)
    result['pin_vertex_groups'] = (was_set, ('COLLECTION', copied_rows))
    return result


def _driver_data_ids(obj):
    if obj.data is None:
        return ()
    ids = [obj.data]
    if obj.type == 'MESH' and obj.data.shape_keys is not None:
        ids.append(obj.data.shape_keys)
    return tuple(ids)


def _has_shape_key_animdata(obj):
    keys = getattr(obj.data, 'shape_keys', None)
    return keys is not None and keys.animation_data is not None


def _validate_driver_data_targets(arm, sources, independently_copied):
    """Reject only referenced data with more than one copied destination."""
    data_users = {}
    for source in sources:
        shared = source.type == 'MESH' and source not in independently_copied
        for data in _driver_data_ids(source):
            data_users.setdefault(data, set()).add(None if shared else source)
    ambiguous = {data for data, users in data_users.items() if len(users) > 1}
    if not ambiguous:
        return
    # These are the AnimData owners remapped by this operation. Pose drivers
    # live on the armature Object, not on individual PoseBone records.
    for owner in [*sources, arm.data]:
        animation = getattr(owner, 'animation_data', None)
        for curve in getattr(animation, 'drivers', ()):
            for variable in curve.driver.variables:
                for target in variable.targets:
                    if target.id in ambiguous:
                        raise ValueError('驱动引用的数据将产生多个副本，无法确定目标：' + target.id.name + '；整架未复制')


def _copied_data_targets(objects):
    destinations = {}
    for source, target in objects.items():
        source_ids, target_ids = _driver_data_ids(source), _driver_data_ids(target)
        if len(source_ids) != len(target_ids):
            raise ValueError('复制数据或形态键引用缺失；整架已撤回')
        for source_id, target_id in zip(source_ids, target_ids):
            destinations.setdefault(source_id, set()).add(target_id)
    return {source: next(iter(targets)) if len(targets) == 1 else None
            for source, targets in destinations.items()}


def _remap_targets(owner, objects, data_targets):
    """Remap copied self/proxy constraint, modifier and driver object targets.

    Actions remain shared author assets and are never edited here. Drivers are
    part of the copied AnimData and their target records belong to the copy.
    """
    if isinstance(owner, bpy.types.PoseBone) and owner.custom_shape in objects:
        owner.custom_shape = objects[owner.custom_shape]
    for item in list(getattr(owner, 'constraints', ())) + list(getattr(owner, 'modifiers', ())):
        for prop in item.bl_rna.properties:
            if prop.type == 'POINTER' and not prop.is_readonly:
                value = getattr(item, prop.identifier)
                if isinstance(value, bpy.types.Object) and value in objects:
                    setattr(item, prop.identifier, objects[value])
        # Geometry Nodes interface inputs are modifier IDProperties, not RNA
        # pointer properties. Only remap actual per-instance Object inputs;
        # shared group defaults and arbitrary author metadata stay untouched.
        if isinstance(item, bpy.types.NodesModifier) and item.node_group:
            for socket in item.node_group.interface.items_tree:
                if (socket.item_type == 'SOCKET' and socket.in_out == 'INPUT'
                        and socket.socket_type == 'NodeSocketObject'):
                    value = item.get(socket.identifier)
                    if isinstance(value, bpy.types.Object) and value in objects:
                        item[socket.identifier] = objects[value]
        if isinstance(item, bpy.types.UVProjectModifier):
            for projector in item.projectors:
                if projector.object in objects:
                    projector.object = objects[projector.object]
        for target in getattr(item, 'targets', ()):
            if target.target in objects:
                target.target = objects[target.target]
    animation = getattr(owner, 'animation_data', None)
    if animation:
        for curve in animation.drivers:
            for variable in curve.driver.variables:
                for target in variable.targets:
                    if target.id in objects:
                        target.id = objects[target.id]
                    elif target.id in data_targets:
                        copied = data_targets[target.id]
                        if copied is None:
                            raise ValueError('复制驱动数据目标不明确；整架已撤回')
                        target.id = copied


def _duplicate_curves(state, joints):
    candidate = deepcopy(state)
    doc = candidate['document']
    if doc is None:
        return candidate
    tables = {row['table_id']: row for row in doc['tables']}
    table_ids = {}
    for binding in list(doc['bindings']):
        target = binding['target']
        if target.get('kind') != 'JOINT' or target.get('joint_uuid') not in joints:
            continue
        copied = deepcopy(binding)
        old = binding['table_id']
        if old not in table_ids:
            table_ids[old] = 'rig_copy_' + uuid.uuid4().hex
            table = deepcopy(tables[old])
            table['table_id'] = table_ids[old]
            doc['tables'].append(table)
        copied['table_id'] = table_ids[old]
        copied['target']['joint_uuid'] = joints[target['joint_uuid']]
        doc['bindings'].append(copied)
    from ..models.restoring_curves import validate_author_map
    candidate['document'] = validate_author_map(doc)
    return candidate


def _commit_duplicate_curves(scene, before_raw, before_set, original, candidate, joint_ids):
    """Publish only new joint bindings; never rewrite unrelated RNA fields."""
    from ..models import restoring_curves as model
    owner = scene.ppf_restoring_curves
    if candidate['document'] == original['document']:
        return None
    if owner.document_json != before_raw or owner.is_property_set('document_json') != before_set:
        raise ValueError('作者已修改恢复力文档；未覆盖作者修改')
    before_doc = original['document']
    after_doc = candidate['document']
    before_tables = {x['table_id'] for x in before_doc['tables']}
    new_tables = [deepcopy(x) for x in after_doc['tables'] if x['table_id'] not in before_tables]
    new_bindings = [deepcopy(x) for x in after_doc['bindings']
                    if x['target'].get('kind') == 'JOINT' and x['target'].get('joint_uuid') in joint_ids]
    after_raw = model._json_bytes(after_doc).decode('utf-8')
    owner.document_json = after_raw
    return {'before_raw':before_raw, 'before_set':before_set, 'after_raw':after_raw,
            'bindings':new_bindings, 'tables':new_tables, 'joints':set(joint_ids)}


def _rollback_duplicate_curves(scene, journal):
    """Remove only unmodified additions; return False when author edits conflict."""
    if journal is None:
        return True
    from ..models import restoring_curves as model
    from . import restoring_curve_presets as presets
    owner = scene.ppf_restoring_curves
    raw = owner.document_json
    if raw == journal['after_raw']:
        if journal['before_set']:
            owner.document_json = journal['before_raw']
        else:
            owner.property_unset('document_json')
        return True
    try:
        doc = model.validate_author_map(json.loads(raw, object_pairs_hook=presets._object_pairs))
        additions = journal['bindings']
        for binding in doc['bindings']:
            target = binding.get('target', {})
            if target.get('kind') == 'JOINT' and target.get('joint_uuid') in journal['joints'] and binding not in additions:
                return False
        old_tables = {table['table_id']:table for table in journal['tables']}
        for table in doc['tables']:
            if table['table_id'] in old_tables and table != old_tables[table['table_id']]:
                return False
        doc['bindings'] = [binding for binding in doc['bindings'] if binding not in additions]
        used = {binding['table_id'] for binding in doc['bindings']}
        doc['tables'] = [table for table in doc['tables'] if table['table_id'] not in old_tables or table['table_id'] in used]
        owner.document_json = model._json_bytes(model.validate_author_map(doc)).decode('utf-8')
        return True
    except (ValueError, TypeError, KeyError):
        return False


def duplicate_rig(context, report=None):
    from . import bone_proxy as ops
    from .restoring_curve_operators import snapshot_rna, restore_rna, read_state, write_state
    from ..models.bone_bindings import active_armature, validate_proxy_owner, validate_final_binding_ids
    from ..models.groups import get_addon_data, invalidate_overlays
    from ..core.bone_mass import material_write_guard, mark_settings
    from ..models.bone_collision import collision_edit_batch
    from ..core.bone_proxy_visibility import MANAGED, manage_proxy, cleanup_new_collections
    from ..core.bone_author_follow import baseline_proxy

    arm = active_armature(context)
    if arm is None or context.mode not in {'OBJECT', 'POSE'}:
        raise ValueError('请在对象或姿态模式选中骨架')
    if arm.library or arm.data.library or arm.override_library or arm.data.override_library:
        raise ValueError('请先将骨架及骨架数据转为本地数据')
    com = ops._facade_communicator()
    com.assert_bone_body_switch_idle()
    if (context.screen and context.screen.is_animation_playing) or ops.stale_simulation_output(context):
        raise ValueError('请停止播放并 Clear 模拟输出、恢复作者状态后再复制骨架')
    validate_final_binding_ids(context.scene, None, [])
    bones = list(arm.data.bones)
    bone_ids = {b.ppf_bone_physics.bone_uuid for b in bones if b.ppf_bone_physics.bone_uuid}
    # Never quietly reconnect an external cross-rig edge to the source rig.
    for owner in (o for o in context.scene.objects if o.type == 'ARMATURE'):
        for link in owner.ppf_cross_links:
            endpoints = {link.bone_a_uuid, link.bone_b_uuid}
            if (owner == arm and not endpoints <= bone_ids) or (owner != arm and endpoints & bone_ids):
                raise ValueError('骨架含跨骨架外部连接；请明确解除外部连接后复制')
    groups = {}
    pin_plans = {}
    author_materials = {}
    for bone in bones:
        s = bone.ppf_bone_physics
        if s.enabled and not s.proxy:
            raise ValueError(bone.name + '：启用的骨骼缺少代理')
        if s.proxy:
            validate_proxy_owner(arm, bone)
            group = ops._owned_bone_material(context.scene, arm, bone)
            if group is None:
                raise ValueError(bone.name + '：代理的独占材质组失配')
            # Match the existing author-material write gate: scene/group and
            # bone-material animation cannot be copied as a static new group.
            from ..core.bone_mass import _owned
            _owned(context.scene, arm, bone)
            try:
                author_materials[bone.name] = _author_material_plan(s, group)
            except ValueError as exc:
                raise ValueError(bone.name + '：' + str(exc)) from exc
            if s.proxy.mode != 'OBJECT' or any(m.type == 'MESH_CACHE' for m in s.proxy.modifiers):
                raise ValueError(bone.name + '：请退出代理编辑并移除缓存播放后复制')
            if s.proxy.animation_data or s.proxy.data.animation_data or _has_shape_key_animdata(s.proxy):
                raise ValueError(bone.name + '：代理含动画或驱动，须先明确作者静态代理后复制')
            if any(getattr(p, 'has_captured_anim', False) for p in group.pin_vertex_groups):
                raise ValueError(bone.name + '：代理含已捕获的 Pin 缓存，请先清除后复制')
            if any(p.object_uuid != s.body_uuid for p in group.pin_vertex_groups):
                raise ValueError(bone.name + '：材质组含外部 Pin 引用，请先明确解除后复制')
            from ..core.uuid_registry import plan_pin_reference, get_object_uuid
            from ..core.pin_reference import PinReferenceError
            planned = []
            for pin in group.pin_vertex_groups:
                plan = plan_pin_reference(pin, s.proxy)
                if plan is None or plan[0] != s.proxy or pin.object_uuid != s.body_uuid or get_object_uuid(plan[0]) != s.body_uuid:
                    raise PinReferenceError('Bone Pin owner is not the verified proxy')
                planned.append((plan[0], pin.object_uuid, plan[2]))
            pin_plans[bone.name] = tuple(planned)
            groups[bone.name] = snapshot_rna(group)
        custom = s.custom_object
        if custom and (custom.type != 'MESH' or custom.animation_data or custom.data.animation_data
                       or _has_shape_key_animdata(custom)
                       or any(m.type == 'MESH_CACHE' for m in custom.modifiers)):
            raise ValueError(bone.name + '：自定义生成源含动画或缓存，请先使用静态作者网格')
    # Copy only the internal Empty ancestry actually required by owned proxies.
    # External parents remain author references and are not cloned.
    internal_parents = []
    unsupported_parents = set()
    for bone in bones:
        proxy = bone.ppf_bone_physics.proxy
        parent = proxy.parent if proxy else None
        chain, seen = [], set()
        while parent is not None and parent != arm:
            if parent in seen:
                raise ValueError('代理父级存在循环，无法安全复制')
            seen.add(parent)
            chain.append(parent)
            parent = parent.parent
        if parent == arm:
            if any(obj.type not in {'EMPTY', 'MESH'} or obj.library or obj.override_library for obj in chain):
                unsupported_parents.add(bone.name)
            else:
                for obj in reversed(chain):
                    if obj not in internal_parents:
                        internal_parents.append(obj)
    if unsupported_parents and report is not None:
        report({'WARNING'}, '兼容旧版复制：部分代理父级类型暂不支持精确复制，可能产生偏移；请检查副本')
    independently_copied = {obj for bone in bones for obj in
                            (bone.ppf_bone_physics.proxy, bone.ppf_bone_physics.custom_object)
                            if obj is not None}
    _validate_driver_data_targets(arm, {arm} | independently_copied | set(internal_parents),
                                  independently_copied)
    root = get_addon_data(context.scene)
    curve_owner = context.scene.ppf_restoring_curves
    curve_raw = curve_owner.document_json
    curve_was_set = curve_owner.is_property_set('document_json')
    curve_state = read_state(context.scene)
    curve_journal = None
    old_selection = tuple(context.selected_objects)
    old_active = context.view_layer.objects.active
    old_mode = arm.mode
    created_objects, created_data, created_groups, created_collections = [], [], [], []
    group_token = _uuid()
    group_marker = '_ppf_rig_duplicate_transaction'
    objects, ids = {}, {}
    for value in [arm.ppf_armature_uuid, *(getattr(b.ppf_bone_physics, key)
                  for b in bones for key in ('bone_uuid', 'body_uuid', 'joint_uuid', 'group_uuid'))]:
        if value:
            ids[value] = _uuid()
    new_arm = None

    def copy_object(source, *, share_data=False):
        if source in objects:
            return objects[source]
        obj = source.copy()
        created_objects.append(obj)
        if source.data is not None and not share_data:
            obj.data = source.data.copy()
            created_data.append(obj.data)
        objects[source] = obj
        # Object.copy retains author actions only. There are no source caches
        # at this point, and cached session/author-baseline tags never migrate.
        for key in list(obj.keys()):
            if key.startswith('_solver') or key.startswith('_ppf') or key.startswith('ppf_proxy_edit') or key == 'ppf_bone_author_follow_v1':
                del obj[key]
        return obj

    with collision_edit_batch():
        try:
            with material_write_guard():
                new_arm = copy_object(arm)
                new_arm.name = arm.name + ' 副本'
                new_arm.ppf_armature_uuid = ids.get(arm.ppf_armature_uuid, _uuid())
                for collection in arm.users_collection:
                    if collection in tuple(context.scene.collection.children_recursive) or collection == context.scene.collection:
                        collection.objects.link(new_arm)
                if not new_arm.users_collection:
                    context.scene.collection.objects.link(new_arm)
                # Allocate all objects before writing references; no callback or
                # deferred timer can observe borrowed source proxy identities.
                for source_parent in internal_parents:
                    scaffold_only = source_parent.type == 'MESH' and source_parent not in independently_copied
                    copied_parent = copy_object(source_parent, share_data=scaffold_only)
                    if scaffold_only:
                        for key in ('ppf_bone_proxy_owned', 'ppf_bone_uuid', 'ppf_armature_uuid', 'ppf_bone_proxy_object_container'):
                            copied_parent.pop(key, None)
                    if source_parent.get('ppf_bone_proxy_object_container') == arm.ppf_armature_uuid:
                        copied_parent['ppf_bone_proxy_object_container'] = new_arm.ppf_armature_uuid
                    else:
                        context.scene.collection.objects.link(copied_parent)
                for bone in bones:
                    s = bone.ppf_bone_physics
                    if s.proxy:
                        proxy = copy_object(s.proxy)
                        # This temporary root link belongs to the new transaction.
                        proxy.pop(MANAGED, None)
                        if proxy.name not in context.scene.collection.objects:
                            context.scene.collection.objects.link(proxy)
                    if s.custom_object:
                        custom = copy_object(s.custom_object)
                        if not custom.users_collection:
                            context.scene.collection.objects.link(custom)
                            custom.hide_set(True)
                            custom.hide_render = True
                data_targets = _copied_data_targets(objects)
                copied_group_snapshots = {
                    name: _pin_copy_group_snapshot(snapshot, pin_plans[name], objects, ids)
                    for name, snapshot in groups.items()
                }
                for bone in bones:
                    src = bone.ppf_bone_physics
                    dst = new_arm.data.bones[bone.name].ppf_bone_physics
                    # data.copy preserved settings, but before callbacks run all
                    # owner identifiers and pointers must become independent.
                    for key in ('bone_uuid', 'body_uuid', 'joint_uuid', 'group_uuid'):
                        value = getattr(src, key)
                        setattr(dst, key, ids.get(value, '') if value else '')
                    dst.proxy = objects.get(src.proxy)
                    dst.custom_object = objects.get(src.custom_object)
                    if src.recreate_geometry_intent_json:
                        token = json.loads(src.recreate_geometry_intent_json)
                        token['owner'] = {key: ids.get(value, value) for key, value in token['owner'].items()}
                        dst.recreate_geometry_intent_json = json.dumps(token, sort_keys=True)
                    if not src.proxy:
                        continue
                    proxy = dst.proxy
                    source_parent = src.proxy.parent
                    inverse = src.proxy.matrix_parent_inverse.copy()
                    basis = src.proxy.matrix_basis.copy()
                    legacy_world = src.proxy.matrix_world.copy() if bone.name in unsupported_parents else None
                    proxy.parent = new_arm if bone.name in unsupported_parents else objects.get(source_parent, source_parent)
                    proxy['_solver_uuid'] = dst.body_uuid
                    proxy['ppf_bone_proxy_owned'] = True
                    proxy['ppf_bone_uuid'] = dst.bone_uuid
                    proxy['ppf_armature_uuid'] = new_arm.ppf_armature_uuid
                    if legacy_world is None:
                        proxy.matrix_parent_inverse = inverse
                        proxy.matrix_basis = basis
                    else:
                        proxy.matrix_world = legacy_world
                    group = root.bone_material_groups.add()
                    # Register the actual newly allocated slot before writing
                    # anything. Later CollectionProperty.add() can relocate
                    # prior items, so raw RNA pointers are not stable across
                    # this transaction. A temporary per-item token survives
                    # relocation and does not depend on UUID restoration.
                    created_groups.append(len(root.bone_material_groups) - 1)
                    group[group_marker] = group_token
                    restore_rna(group, copied_group_snapshots[bone.name])
                    from ..core.pin_picker_identity import clear as clear_pin_picker
                    clear_pin_picker(group)
                    group.uuid = dst.group_uuid
                    group.assigned_objects[0].name = proxy.name
                    # The group keeps its original unit representation. Only
                    # publish the author values already proven representable.
                    for key, value in author_materials[bone.name].items():
                        setattr(dst, key, value)
                        if getattr(dst, key) != value:
                            raise ValueError(bone.name + '：复制作者材质值被限制；整架已撤回')
                # Object.copy already copied links; replace their complete RNA
                # rather than dropping author fields or external endpoints.
                new_arm.ppf_cross_links.clear()
                for link in arm.ppf_cross_links:
                    copied = new_arm.ppf_cross_links.add()
                    restore_rna(copied, _remap_snapshot(snapshot_rna(link), ids, objects))
                    copied.link_uuid = _uuid()
                for source, obj in objects.items():
                    if source != arm and source.parent in objects and not source.get('ppf_bone_proxy_owned'):
                        inverse = source.matrix_parent_inverse.copy()
                        basis = source.matrix_basis.copy()
                        obj.parent = objects[source.parent]
                        obj.matrix_parent_inverse = inverse
                        obj.matrix_basis = basis
                    _remap_targets(obj, objects, data_targets)
                    if obj.type == 'ARMATURE':
                        _remap_targets(obj.data, objects, data_targets)
                        for pose in obj.pose.bones:
                            _remap_targets(pose, objects, data_targets)
                joints = {b.ppf_bone_physics.joint_uuid: new_arm.data.bones[b.name].ppf_bone_physics.joint_uuid
                          for b in bones if b.ppf_bone_physics.joint_uuid}
                curve_journal = _commit_duplicate_curves(context.scene, curve_raw, curve_was_set,
                                                        curve_state, _duplicate_curves(curve_state, joints),
                                                        set(joints.values()))
                for bone in new_arm.data.bones:
                    if bone.ppf_bone_physics.proxy:
                        source_parent = arm.data.bones[bone.name].ppf_bone_physics.proxy.parent
                        # The proxy may have author-owned intermediate Empty
                        # parents between it and our managed container. Reuse
                        # the copied, explicitly owned container at any point
                        # in that chain so manage_proxy does not create a
                        # second container and leave the copied one orphaned.
                        copied_container = None
                        source_ancestor = source_parent
                        seen_ancestors = set()
                        while source_ancestor is not None and source_ancestor != arm:
                            if source_ancestor in seen_ancestors:
                                raise ValueError('代理父级存在循环，无法安全复制')
                            seen_ancestors.add(source_ancestor)
                            if (source_ancestor.type == 'EMPTY'
                                    and source_ancestor.get('ppf_bone_proxy_object_container') == arm.ppf_armature_uuid):
                                copied_container = objects.get(source_ancestor)
                                break
                            source_ancestor = source_ancestor.parent
                        manage_proxy(context.scene, new_arm, bone, bone.ppf_bone_physics.proxy,
                                     preserve_local=bone.name not in unsupported_parents, created_objects=created_objects,
                                     created_collections=created_collections,
                                     owned_container=copied_container)
                context.view_layer.update()
                for bone in new_arm.data.bones:
                    if bone.ppf_bone_physics.proxy:
                        validate_proxy_owner(new_arm, bone)
                validate_final_binding_ids(context.scene, None, [])
            if old_mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            for obj in context.selected_objects:
                obj.select_set(False)
            new_arm.select_set(True)
            context.view_layer.objects.active = new_arm
            if old_mode == 'POSE':
                bpy.ops.object.mode_set(mode='POSE')
            for bone in new_arm.data.bones:
                if bone.ppf_bone_physics.proxy:
                    baseline_proxy(context.scene, new_arm, bone)
                    mark_settings(bone.ppf_bone_physics, context)
            com.bone_physical_body_changed()
            for group in root.bone_material_groups:
                if group.get(group_marker) == group_token:
                    del group[group_marker]
        except BaseException as cause:
            with material_write_guard():
                curve_rollback_ok = _rollback_duplicate_curves(context.scene, curve_journal)
                for index in reversed(range(len(root.bone_material_groups))):
                    if root.bone_material_groups[index].get(group_marker) == group_token:
                        if curve_rollback_ok:
                            root.bone_material_groups.remove(index)
                        else:
                            del root.bone_material_groups[index][group_marker]
                if context.object and context.object.mode != 'OBJECT':
                    bpy.ops.object.mode_set(mode='OBJECT')
                if curve_rollback_ok:
                    for obj in reversed(created_objects):
                        bpy.data.objects.remove(obj, do_unlink=True)
                    for data in reversed(created_data):
                        if data.users == 0:
                            if isinstance(data, bpy.types.Armature):
                                bpy.data.armatures.remove(data)
                            else:
                                bpy.data.meshes.remove(data)
                    for collection in reversed(created_collections):
                        if collection.name in bpy.data.collections and not collection.objects and not collection.children:
                            bpy.data.collections.remove(collection)
                for obj in context.selected_objects:
                    obj.select_set(False)
                for obj in old_selection:
                    obj.select_set(True)
                context.view_layer.objects.active = old_active
                if old_mode == 'POSE':
                    bpy.ops.object.mode_set(mode='POSE')
            if not curve_rollback_ok:
                raise RuntimeError('作者修改冲突，未自动完整撤销；已保留复制数据和恢复力文档') from cause
            raise
    invalidate_overlays()
    return new_arm


class PPF_OT_BoneRigDuplicate(bpy.types.Operator):
    bl_idname = 'ppf.bone_rig_duplicate'
    bl_label = '复制骨架与代理'
    bl_description = '复制完整作者骨架、独立代理网格、材质、关节和内部连接；先 Clear 模拟输出'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        from ..models.bone_bindings import active_armature
        return context.mode in {'OBJECT', 'POSE'} and active_armature(context) is not None

    def execute(self, context):
        try:
            new_arm = duplicate_rig(context, self.report)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, '已创建独立骨架：' + new_arm.name)
        return {'FINISHED'}


def register():
    i18n.register_class(PPF_OT_BoneRigDuplicate)


def unregister():
    bpy.utils.unregister_class(PPF_OT_BoneRigDuplicate)
