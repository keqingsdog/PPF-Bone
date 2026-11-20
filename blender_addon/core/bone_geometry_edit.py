"""Short-lived mutual exclusion for editing owned bone collision geometry.

No lease is stored in a blend file. Preview/cancel never invalidates a build;
only an accepted geometry change retires the previous uploaded body.
"""

_lease = None


def _communicator():
    from ..ops.bone_proxy import _facade_communicator
    return _facade_communicator()


def assert_solver_available():
    if _lease is not None:
        raise ValueError('请先确认或取消代理缩放／网格编辑，再传输、构建、计算或获取帧')


def preflight(context):
    """Refuse before any authoring, cache, visibility or mode mutation."""
    assert_solver_available()
    communicator = _communicator()
    communicator.assert_bone_body_switch_idle()
    from ..ops.bone_proxy import stale_simulation_output
    reasons = stale_simulation_output(context)
    from .single_bake_restore import suspended_meshes_cover_scene
    if suspended_meshes_cover_scene(context.scene):
        reasons = [reason for reason in reasons if reason != "solver mesh cache"]
    if reasons:
        raise ValueError('请先清除已有模拟帧再编辑代理几何：' + ', '.join(reasons))
    return communicator


def _geometry(proxy):
    try:
        mesh = proxy.data
        return (tuple(tuple(vertex.co) for vertex in mesh.vertices),
                tuple(tuple(edge.vertices) for edge in mesh.edges),
                tuple(tuple(face.vertices) for face in mesh.polygons),
                tuple(tuple(row) for row in proxy.matrix_world))
    except ReferenceError:
        return None


def _edit_presentation(context,proxy):
    """Expose only this real mesh; leave hidden/excluded owner collections alone."""
    fields=('display_type','show_in_front','show_wire','show_all_edges','hide_viewport','hide_select')
    state=dict(proxy=proxy,view_layer=context.view_layer,active=context.object,mode=context.mode,fields={key:getattr(proxy,key) for key in fields},
               editing_flag=('ppf_bone_proxy_visibility_editing' in proxy,proxy.get('ppf_bone_proxy_visibility_editing')),
               root=context.scene.collection,root_linked=proxy.name in context.scene.collection.objects,
               layers=[(layer,proxy.hide_get(view_layer=layer)) for layer in context.scene.view_layers
                       if proxy.name in layer.objects])
    try:
        if not state['root_linked']:context.scene.collection.objects.link(proxy)
        proxy.hide_viewport=False;proxy.hide_select=False
        proxy.display_type='SOLID';proxy.show_in_front=True
        proxy.show_wire=True;proxy.show_all_edges=True
        for layer in context.scene.view_layers:
            layer.update()
            if proxy.name in layer.objects:proxy.hide_set(layer!=context.view_layer,view_layer=layer)
        return state
    except Exception:
        _restore_presentation(state)
        raise


def _restore_presentation(state):
    try:
        import bpy
        proxy=state['proxy'];layer=state['view_layer']
        return_owner=(bpy.context.view_layer==layer and layer.objects.active==proxy)
        if return_owner and proxy.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
        for key,value in state['fields'].items():setattr(proxy,key,value)
        present,value=state['editing_flag']
        if present:proxy['ppf_bone_proxy_visibility_editing']=value
        else:proxy.pop('ppf_bone_proxy_visibility_editing',None)
        if not state['root_linked'] and proxy.name in state['root'].objects:
            state['root'].objects.unlink(proxy)
        for layer,hidden in state['layers']:
            layer.update()
            if proxy.name in layer.objects:proxy.hide_set(hidden,view_layer=layer)
        owner=state['active']
        if return_owner and owner is not None and owner.name in state['view_layer'].objects:
            owner.select_set(True,view_layer=state['view_layer']);state['view_layer'].objects.active=owner
            mode='EDIT' if state['mode']=='EDIT_ARMATURE' else state['mode']
            if mode!='OBJECT':bpy.ops.object.mode_set(mode=mode)
    except (ReferenceError,RuntimeError):
        pass  # A completed Blender identity replacement owns the new datablocks.


def _restore_presentations(lease):
    for state in reversed(lease.get('presentation',())):_restore_presentation(state)


def acquire(context, proxies, kind='MESH'):
    global _lease
    preflight(context)
    proxies = tuple(proxies)
    if not proxies:
        raise ValueError('请选择已有代理的骨骼')
    snapshots = tuple((proxy, _geometry(proxy)) for proxy in proxies)
    if any(geometry is None for _proxy, geometry in snapshots):
        raise ValueError('代理已经失效，请重新选择骨骼')
    token = object()
    presentation=[]
    try:
        if kind=='MESH':
            for proxy in proxies:presentation.append(_edit_presentation(context,proxy))
    except Exception:
        for state in reversed(presentation):_restore_presentation(state)
        raise
    _lease = {'token': token, 'kind': kind, 'snapshots': snapshots,'presentation':presentation}
    return token


def token_for(proxy, kind=None):
    if _lease is None or (kind is not None and _lease['kind'] != kind):
        return None
    for candidate, _before in _lease['snapshots']:
        try:
            if candidate == proxy:
                return _lease['token']
        except ReferenceError:
            continue
    return None


def leased_proxies(kind=None):
    """Return the current lease's proxy references without exposing the lease."""
    if _lease is None or (kind is not None and _lease['kind'] != kind):
        return ()
    return tuple(proxy for proxy, _before in _lease['snapshots'])


def finish(token, commit=False):
    """Commit a real change, or release unchanged/cancelled preview.

    A failed commit retains the token, so its owner can restore its geometry
    and call finish(commit=False). A lifecycle reset makes late tokens no-ops.
    """
    global _lease
    if _lease is None or token is not _lease['token']:
        return False
    changed = any(_geometry(proxy) != before for proxy, before in _lease['snapshots'])
    if commit and changed:
        communicator = _communicator()
        communicator.assert_bone_body_switch_idle()
        import bpy
        from .single_bake_restore import invalidate
        invalidate(bpy.context.scene)
        communicator.bone_physical_body_changed()
    _restore_presentations(_lease)
    _lease = None
    return changed


def release_all():
    global _lease
    if _lease is not None:_restore_presentations(_lease)
    _lease = None


def active():
    return _lease is not None


def _after_state_change(*_args):
    release_all()


def register():
    import bpy
    release_all()
    bpy.app.handlers.persistent(_after_state_change)
    for handlers in (bpy.app.handlers.load_post, bpy.app.handlers.undo_post, bpy.app.handlers.redo_post):
        if _after_state_change not in handlers:
            handlers.append(_after_state_change)


def unregister():
    import bpy
    release_all()
    for handlers in (bpy.app.handlers.load_post, bpy.app.handlers.undo_post, bpy.app.handlers.redo_post):
        if _after_state_change in handlers:
            handlers.remove(_after_state_change)
