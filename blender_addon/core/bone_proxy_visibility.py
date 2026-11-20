"""Presentation of UUID-owned collision meshes; export inclusion stays author-owned.

Object hide_set hides the real mesh without disabling dependency-graph geometry.
Collections are never excluded/unhidden on the user's behalf.
"""
import bpy

MANAGED = 'ppf_bone_proxy_managed_visibility'
COLLECTION = 'ppf_bone_proxy_collection_owner'
ROOT_COLLECTION = 'ppf_bone_proxy_collection_root'
EDITING = 'ppf_bone_proxy_visibility_editing'


def owned_proxy(scene, arm, bone, proxy):
    settings = bone.ppf_bone_physics
    if (not proxy or proxy.type != 'MESH' or settings.proxy != proxy
            or proxy.name not in scene.objects or arm.name not in scene.objects
            or not proxy.get('ppf_bone_proxy_owned')
            or not settings.body_uuid or not settings.bone_uuid or not arm.ppf_armature_uuid
            or proxy.get('_solver_uuid') != settings.body_uuid
            or proxy.get('ppf_bone_uuid') != settings.bone_uuid
            or proxy.get('ppf_armature_uuid') != arm.ppf_armature_uuid):
        return False
    bodies = sum(obj.get('_solver_uuid') == settings.body_uuid for obj in scene.objects)
    refs = sum(b.ppf_bone_physics.proxy == proxy for a in scene.objects
               if a.type == 'ARMATURE' for b in a.data.bones)
    owners = sum(a.type == 'ARMATURE' and getattr(a, 'ppf_armature_uuid', '') == arm.ppf_armature_uuid
                 for a in scene.objects)
    return bodies == 1 and refs == 1 and owners == 1


def _scene_collections(root):
    result, pending = set(), [root]
    while pending:
        collection = pending.pop()
        if collection not in result:
            result.add(collection)
            pending.extend(collection.children)
    return result


def _owned_collection(collection):
    return bool(collection.get(COLLECTION) or collection.get(ROOT_COLLECTION))


def _collection_parents(collection):
    parents = [scene.collection for scene in bpy.data.scenes]
    parents.extend(bpy.data.collections)
    return tuple(parent for parent in parents if collection.name in parent.children)


def _layer_nodes(node, path=()):
    path = path + (node.collection,)
    yield path, node
    for child in node.children:
        yield from _layer_nodes(child, path)


def _layer_node(layer, path):
    node = layer.layer_collection
    if not path or node.collection != path[0]:
        return None
    for collection in path[1:]:
        node = next((child for child in node.children if child.collection == collection), None)
        if node is None:
            return None
    return node


def _snapshot_layer_collections(scene):
    """Capture flags by collection path, including descendants of owned nodes."""
    return tuple((layer, tuple((path, node.exclude, node.hide_viewport)
                              for path, node in _layer_nodes(layer.layer_collection)
                              if any(_owned_collection(c) for c in path[1:])))
                 for layer in scene.view_layers)


def _restore_layer_collections(states, old_prefix=(), new_prefix=()):
    for layer, rows in states:
        layer.update()
        # Parent-first restores descendants after any exclude propagation.
        for path, excluded, hidden in rows:
            if old_prefix and path[:len(old_prefix)] == old_prefix:
                path = new_prefix + path[len(old_prefix):]
            node = _layer_node(layer, path)
            if node is not None:
                if node.exclude != excluded:
                    node.exclude = excluded
                node = _layer_node(layer, path)
                if node is not None and node.hide_viewport != hidden:
                    node.hide_viewport = hidden
        layer.update()


def _proxy_collection(scene, arm, *, created_collections=None):
    collections = _scene_collections(scene.collection)
    roots = [c for c in collections if c != scene.collection and c.get(ROOT_COLLECTION)]
    root = next((c for c in scene.collection.children if c.get(ROOT_COLLECTION)), None)
    if root is None:
        root = roots[0] if roots else None
    children = [c for c in collections if c.get(COLLECTION) == arm.ppf_armature_uuid]
    collection = next((c for c in root.children if c.get(COLLECTION) == arm.ppf_armature_uuid), None) if root else None
    if collection is None:
        collection = children[0] if children else None
    legacy = collection is not None and collection.name in scene.collection.children
    # Capture before links are changed: LayerCollection RNA belongs to a path,
    # and moving the Collection datablock creates a fresh node in every layer.
    layer_states = _snapshot_layer_collections(scene) if legacy else ()
    if root is None:
        root = bpy.data.collections.new('PPF 骨骼代理')
        if created_collections is not None:
            created_collections.append(root)
        root[ROOT_COLLECTION] = True
    if root.name not in scene.collection.children:
        scene.collection.children.link(root)
    if collection is None:
        collection = bpy.data.collections.new('骨架 · ' + arm.name)
        if created_collections is not None:
            created_collections.append(collection)
        collection[COLLECTION] = arm.ppf_armature_uuid
    if collection.name not in root.children:
        root.children.link(collection)
    # Preserve all user extra paths. Only the legacy scene-root path moves.
    if legacy:
        scene.collection.children.unlink(collection)
        _restore_layer_collections(layer_states, (scene.collection, collection),
                                   (scene.collection, root, collection))
    return collection


def manage_proxy(scene, arm, bone, proxy, *, preserve_local=False, created_objects=None, created_collections=None, owned_container=None, _defer_layers=False):
    """Group owned proxies under one scene root; hide only their real mesh."""
    if not owned_proxy(scene, arm, bone, proxy):
        raise ValueError('BONE_PROXY_OWNER_INVALID')
    was_managed = bool(proxy.get(MANAGED))
    collection = _proxy_collection(scene, arm, created_collections=created_collections)
    # A single collapsible object container avoids hundreds of proxy children
    # directly under the armature in the Outliner. Keep its armature parenting
    # so inherited object transforms retain their previous evaluation semantics.
    container = owned_container
    if container is not None:
        if (container.type != 'EMPTY' or container.parent != arm
                or container.get('ppf_bone_proxy_object_container') != arm.ppf_armature_uuid):
            raise ValueError('BONE_PROXY_CONTAINER_OWNER_INVALID')
        if container.name not in collection.objects:
            collection.objects.link(container)
    else:
        container = next((obj for obj in collection.objects
                          if obj.type == 'EMPTY' and obj.parent == arm
                          and obj.get('ppf_bone_proxy_object_container') == arm.ppf_armature_uuid), None)
    if container is None:
        container = bpy.data.objects.new('代理 · ' + arm.name, None)
        if created_objects is not None:
            created_objects.append(container)
        container['ppf_bone_proxy_object_container'] = arm.ppf_armature_uuid
        collection.objects.link(container)
        container.parent = arm
        container.matrix_parent_inverse.identity()
        container.matrix_basis.identity()
    # Object.copy does not preserve per-view-layer hide_set state.  This also
    # covers a copied owned_container and newly added view layers without
    # disabling evaluation of its children via hide_viewport/exclude.
    if not container.hide_render: container.hide_render = True
    if not _defer_layers:
        for layer in scene.view_layers:
            layer.update()
            if container.name in layer.objects and not container.hide_get(view_layer=layer):
                container.hide_set(True, view_layer=layer)
    if proxy.parent == arm and proxy.parent_type == 'OBJECT':
        if _defer_layers and not preserve_local:
            from mathutils import Matrix
            identity = Matrix.Identity(4)
            if (container.parent_type == 'OBJECT' and
                    not any(not constraint.mute and constraint.influence for constraint in container.constraints)
                    and container.matrix_parent_inverse == identity and container.matrix_basis == identity):
                # An identity intermediate parent has exactly the same local
                # space as the armature, even before depsgraph evaluation.
                preserve_local = True
            else:
                # Existing authored container transforms need a fresh world
                # matrix before Blender can preserve the proxy's world pose.
                for layer in scene.view_layers:
                    layer.update()
        world = proxy.matrix_world.copy() if not preserve_local else None
        inverse = proxy.matrix_parent_inverse.copy() if preserve_local else None
        basis = proxy.matrix_basis.copy() if preserve_local else None
        proxy.parent = container
        if preserve_local:
            proxy.matrix_parent_inverse = inverse
            proxy.matrix_basis = basis
        else:
            proxy.matrix_world = world
    if proxy.name not in collection.objects:
        collection.objects.link(proxy)
    scene_collections = _scene_collections(scene.collection)
    other_collections = set()
    for other in bpy.data.scenes:
        if other != scene:
            other_collections.update(_scene_collections(other.collection))
    for old in tuple(proxy.users_collection):
        if old == collection:
            continue
        # Fresh creation initially links to Scene Collection. Additional user
        # collections are intentional and must survive updates and migration.
        if ((old == scene.collection and not was_managed)
                or (old.get(COLLECTION) == arm.ppf_armature_uuid
                    and old in scene_collections and old not in other_collections)):
            old.objects.unlink(proxy)
    if not _defer_layers:
        for layer in scene.view_layers:
            layer.update()
    if not proxy.get(MANAGED): proxy[MANAGED] = True
    if not proxy.hide_render: proxy.hide_render = True
    if not _edit_layer(proxy) and not proxy.hide_select:
        proxy.hide_select = True
    if not _defer_layers and not proxy.get(EDITING):
        for layer in scene.view_layers:
            if proxy.name in layer.objects and not proxy.hide_get(view_layer=layer):
                proxy.hide_set(True, view_layer=layer)
    from .bone_proxy_outliner import request_root_collapse
    request_root_collapse(scene, collection)
    return collection


def cleanup_collections(scene):
    """Remove empty owned containers only when no extra user/scene link exists."""
    collections = _scene_collections(scene.collection)
    for collection in tuple(collections):
        if not collection.get(COLLECTION):
            continue
        for obj in tuple(collection.objects):
            if (obj.type == 'EMPTY'
                    and obj.get('ppf_bone_proxy_object_container') == collection.get(COLLECTION)
                    and not obj.children and len(obj.users_scene) == 1
                    and len(obj.users_collection) == 1):
                bpy.data.objects.remove(obj, do_unlink=True)
    roots = {c for c in collections if c.get(ROOT_COLLECTION)}
    for collection in tuple(collections):
        if (collection != scene.collection and collection.get(COLLECTION)
                and not collection.objects and not collection.children
                and set(_collection_parents(collection)) <= roots | {scene.collection}):
            # Shared roots may expose this empty collection to another scene.
            if any(collection in _scene_collections(other.collection)
                   for other in bpy.data.scenes if other != scene):
                continue
            bpy.data.collections.remove(collection)
    for root in roots:
        if (not root.objects and not root.children
                and set(_collection_parents(root)) == {scene.collection}):
            bpy.data.collections.remove(root)


def begin_mesh_edit(scene, arm, bone, proxy, view_layer=None):
    if not owned_proxy(scene, arm, bone, proxy) or not proxy.get(MANAGED):
        raise ValueError('BONE_PROXY_OWNER_INVALID')
    if proxy.hide_viewport:
        raise ValueError('BONE_PROXY_VIEWPORT_DISABLED')
    layer = view_layer or bpy.context.view_layer
    if proxy.name not in layer.objects:
        raise ValueError('BONE_PROXY_VIEW_LAYER_EXCLUDED')
    proxy.hide_select = False
    proxy.hide_set(False, view_layer=layer)
    proxy[EDITING] = True


def end_mesh_edit(scene, arm, bone, proxy, view_layer=None):
    if not owned_proxy(scene, arm, bone, proxy) or not proxy.get(MANAGED):
        raise ValueError('BONE_PROXY_OWNER_INVALID')
    proxy.pop(EDITING, None)
    proxy.hide_select = True
    for layer in scene.view_layers:
        if proxy.name in layer.objects:
            proxy.hide_set(True, view_layer=layer)


def _collection_visible(node, collections, parent_visible=True):
    visible = parent_visible and not node.exclude and not node.hide_viewport and not node.collection.hide_viewport
    if visible and node.collection in collections:
        return True
    return any(_collection_visible(child, collections, visible) for child in node.children)


def overlay_visible(context, proxy, *, collection_visibility=None):
    """Bypass only our object hide_set; retain collection and viewport disables."""
    if not proxy.get(MANAGED):
        return proxy.visible_get(view_layer=context.view_layer, viewport=context.space_data)
    if proxy.hide_viewport:
        return False
    collections = tuple(proxy.users_collection)
    if collection_visibility is None:
        return _collection_visible(context.view_layer.layer_collection, collections)
    if collections not in collection_visibility:
        collection_visibility[collections] = _collection_visible(context.view_layer.layer_collection, collections)
    return collection_visibility[collections]


def _edit_layer(proxy):
    # The lease is process-local and stores the actual scene root and layer.
    from . import bone_geometry_edit
    lease = bone_geometry_edit._lease
    if lease:
        for state in lease.get('presentation', ()):
            if state['proxy'] == proxy:
                return state['root'], state['view_layer']
    return None


def protect_scene_layers(scene):
    """Hide managed native objects after a view-layer lifecycle change."""
    for arm in tuple(scene.objects):
        if arm.type != 'ARMATURE':
            continue
        for bone in arm.data.bones:
            proxy = bone.ppf_bone_physics.proxy
            if not owned_proxy(scene, arm, bone, proxy) or not proxy.get(MANAGED):
                continue
            edit = _edit_layer(proxy)
            if not edit:
                proxy.hide_select = True
            for layer in scene.view_layers:
                if proxy.name in layer.objects and (not edit or edit != (scene.collection, layer)):
                    if not proxy.hide_get(view_layer=layer):
                        proxy.hide_set(True, view_layer=layer)
            # Author-owned intermediate parents may lie between proxy and the
            # managed container.  Hide only the container carrying our owner
            # marker; never change those intermediate objects.
            container = proxy.parent
            seen = set()
            while container is not None and container != arm and container not in seen:
                seen.add(container)
                if (container.type == 'EMPTY' and container.parent == arm
                        and container.get('ppf_bone_proxy_object_container') == arm.ppf_armature_uuid
                        and any(c.get(COLLECTION) == arm.ppf_armature_uuid for c in container.users_collection)):
                    for layer in scene.view_layers:
                        if container.name in layer.objects and not container.hide_get(view_layer=layer):
                            container.hide_set(True, view_layer=layer)
                    break
                container = container.parent


def sync_scene(scene):
    """Explicit startup/load migration; never called from drawing or a depsgraph loop."""
    managed = []
    for arm in tuple(scene.objects):
        if arm.type != 'ARMATURE':
            continue
        for bone in arm.data.bones:
            proxy = bone.ppf_bone_physics.proxy
            if owned_proxy(scene, arm, bone, proxy):
                collection = manage_proxy(scene, arm, bone, proxy, _defer_layers=True)
                container = next((obj for obj in collection.objects if obj.type == 'EMPTY'
                                  and obj.parent == arm and obj.get('ppf_bone_proxy_object_container') == arm.ppf_armature_uuid), None)
                managed.append((container, proxy))
    if managed:
        for layer in scene.view_layers:
            layer.update()
        for layer in scene.view_layers:
            for container, proxy in managed:
                if container and container.name in layer.objects and not container.hide_get(view_layer=layer):
                    container.hide_set(True, view_layer=layer)
                if not proxy.get(EDITING) and proxy.name in layer.objects and not proxy.hide_get(view_layer=layer):
                    proxy.hide_set(True, view_layer=layer)


def snapshot_proxy(proxy, scene=None):
    scene = scene or bpy.context.scene
    for layer in scene.view_layers:
        layer.update()
    return {
        'collections': tuple(proxy.users_collection),
        'collection_pool': tuple(bpy.data.collections),
        'collection_links': tuple((c, _collection_parents(c))
                                  for c in _scene_collections(scene.collection)
                                  if c != scene.collection and _owned_collection(c)),
        'layer_collections': _snapshot_layer_collections(scene),
        'hide_viewport': proxy.hide_viewport,
        'hide_select': proxy.hide_select,
        'hide_render': proxy.hide_render,
        'flags': {key: (key in proxy, proxy.get(key)) for key in (MANAGED, EDITING)},
        'layers': tuple((layer, proxy.hide_get(view_layer=layer)) for layer in scene.view_layers
                        if proxy.name in layer.objects),
    }


def restore_proxy(proxy, snapshot):
    """Restore only this object's presentation and newly created empty owned collections."""
    current = tuple(proxy.users_collection)
    for collection in snapshot['collections']:
        if proxy.name not in collection.objects:
            collection.objects.link(proxy)
    for collection in current:
        if collection not in snapshot['collections']:
            collection.objects.unlink(proxy)
    proxy.hide_viewport = snapshot['hide_viewport']
    proxy.hide_select = snapshot['hide_select']
    proxy.hide_render = snapshot['hide_render']
    for key, (present, value) in snapshot['flags'].items():
        if present:
            proxy[key] = value
        else:
            proxy.pop(key, None)
    for collection, parents in snapshot.get('collection_links', ()):
        for parent in parents:
            if collection.name not in parent.children:
                parent.children.link(collection)
        for parent in _collection_parents(collection):
            if parent not in parents:
                parent.children.unlink(collection)
    cleanup_new_collections({c.as_pointer() for c in snapshot['collection_pool']})
    _restore_layer_collections(snapshot.get('layer_collections', ()))
    for layer, hidden in snapshot['layers']:
        layer.update()
        if proxy.name in layer.objects:
            proxy.hide_set(hidden, view_layer=layer)


def cleanup_new_collections(existing_pointers):
    """Rollback only newly created empty owned containers, children first."""
    pending = [c for c in bpy.data.collections if c.as_pointer() not in existing_pointers
               and _owned_collection(c)]
    while pending:
        empty = [c for c in pending if not c.objects and not c.children]
        if not empty:
            break
        for collection in empty:
            pending.remove(collection)
            bpy.data.collections.remove(collection)
