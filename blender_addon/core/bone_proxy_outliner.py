"""Collapse owned scene-root collections without changing unrelated tree nodes.

Blender exposes no collection.expanded RNA. Its Outliner UI operators act on
the *filtered runtime tree*, not the complete TreeStore. Build an exact,
collections-only tree, perform the UI operation, then restore the filters in
one callback without swapping the temporary image to the window.

Owned Empty containers parented to armatures use an exact filtered object tree
to close their occurrences, with conservative ownership and ambiguity guards.
"""
import bpy

_pending = set()
_FILTERS = ('filter_text', 'use_filter_complete',
            'use_filter_case_sensitive', 'use_filter_object',
            'use_filter_collection', 'use_filter_view_layers')
_ROOT = 'ppf_bone_proxy_collection_root'


def _literal_pattern(name):
    # Blender uses shell-style wildcards, including bracket expressions.
    return ''.join({'*': '[*]', '?': '[?]', '[': '[[]'}.get(c, c) for c in name)


def collapse_root(window, area, root):
    """Collapse one direct scene child in one VIEW_LAYER Outliner.

    Selection, active IDs, hierarchy and scene data are never modified. Only
    UI flags for the matched root change. The Scene Collection ancestor is
    always reopened by Blender's ViewLayer tree builder on filter restoration.
    """
    space = area.spaces.active
    if area.type != 'OUTLINER' or space.display_mode != 'VIEW_LAYER':
        return False
    if any(not hasattr(space, key) for key in _FILTERS):
        return False
    if root.name not in window.scene.collection.children or not root.get(_ROOT):
        return False
    region = next((r for r in area.regions if r.type == 'WINDOW'), None)
    if region is None:
        return False
    saved = {key: getattr(space, key) for key in _FILTERS}
    with bpy.context.temp_override(window=window, area=area, region=region):
        try:
            pattern = _literal_pattern(root.name)
            space.filter_text = pattern
            # A version's RNA string length may truncate an escaped long name.
            # Never operate on a shortened/broader search pattern.
            if space.filter_text != pattern:
                return False
            space.use_filter_complete = True
            space.use_filter_case_sensitive = True
            space.use_filter_object = False
            space.use_filter_collection = True
            space.use_filter_view_layers = False
            # DRAW rebuilds the native filtered tree without a window swap.
            bpy.ops.wm.redraw_timer(type='DRAW', iterations=1)
            # The first op makes the following toggle deterministic when the
            # root was already closed. Neither UI operator pushes Undo.
            bpy.ops.outliner.show_one_level(open=True)
            bpy.ops.outliner.expanded_toggle()
        finally:
            for key, value in saved.items():
                setattr(space, key, value)
            bpy.ops.wm.redraw_timer(type='DRAW', iterations=1)
            area.tag_redraw()
    return True



def collapse_container(window, area, container):
    """Close both visible occurrences of a safely identified owned Empty."""
    space = area.spaces.active
    fields = ('filter_text', 'use_filter_complete', 'use_filter_case_sensitive',
              'use_filter_object', 'use_filter_children', 'use_filter_collection',
              'use_filter_object_content', 'use_filter_object_armature')
    if area.type != 'OUTLINER' or space.display_mode != 'VIEW_LAYER' or any(not hasattr(space, k) for k in fields):
        return False
    arm = container.parent
    uid = getattr(arm, 'ppf_armature_uuid', '') if arm else ''
    if (container.type != 'EMPTY' or not arm or arm.type != 'ARMATURE' or not uid
            or container.get('ppf_bone_proxy_object_container') != uid):
        return False
    ancestor = arm.parent
    while ancestor:
        if ancestor.type != 'ARMATURE':
            return False
        ancestor = ancestor.parent
    name = container.name
    if any(c in name for c in '*?[') or any(o != container and o.name == name for o in bpy.data.objects):
        return False
    if any(c.name == name for c in bpy.data.collections):
        return False
    region = next ( (r for r in area.regions if r.type == 'WINDOW'), None)
    if region is None:
        return False
    saved = {k: getattr(space, k) for k in fields}
    with bpy.context.temp_override(window=window, area=area, region=region):
        try:
            space.filter_text = name
            if space.filter_text != name:
                return False
            space.use_filter_complete = True
            space.use_filter_case_sensitive = True
            space.use_filter_object = True
            space.use_filter_children = True
            space.use_filter_collection = True
            space.use_filter_object_armature = False
            space.use_filter_object_content = False
            bpy.ops.wm.redraw_timer(type='DRAW', iterations=1)
            bpy.ops.outliner.show_hierarchy()
        finally:
            for key, value in saved.items():
                setattr(space, key, value)
            bpy.ops.wm.redraw_timer(type='DRAW', iterations=1)
            area.tag_redraw()
    return True

def _flush():
    pending = tuple(_pending)
    _pending.clear()
    # Only primitive names survive the deferred callback. Undo/load can replace
    # every RNA object between creation and this callback.
    for scene_name, root_name, collection_name in pending:
        scene = bpy.data.scenes.get(scene_name)
        root = bpy.data.collections.get(root_name)
        if not scene or not root or not root.get(_ROOT):
            continue
        for window in bpy.context.window_manager.windows:
            if window.scene != scene:
                continue
            for area in window.screen.areas:
                if area.type == 'OUTLINER':
                    try:
                        collapse_root(window, area, root)
                        collection = bpy.data.collections.get(collection_name)
                        if collection:
                            for container in collection.objects:
                                if container.get('ppf_bone_proxy_object_container'):
                                    collapse_container(window, area, container)
                    except (ReferenceError, RuntimeError) as error:
                        # Presentation failure must not invalidate created
                        # geometry or leave an exception in the creation op.
                        print('PPF Outliner collapse deferred:', error)
    return None


def request_root_collapse(scene, collection):
    """Coalesce a whole creation batch into one small presentation callback."""
    if bpy.app.background:
        return
    root = next((c for c in scene.collection.children
                 if c.get(_ROOT) and collection.name in c.children), None)
    if root is None:
        return
    if not bpy.app.timers.is_registered(_flush):
        _pending.clear()
        bpy.app.timers.register(_flush, first_interval=0.05)
    _pending.add((scene.name, root.name, collection.name))


def cancel_pending():
    """Called before add-on teardown; no delayed UI work after disable."""
    _pending.clear()
    if bpy.app.timers.is_registered(_flush):
        bpy.app.timers.unregister(_flush)
