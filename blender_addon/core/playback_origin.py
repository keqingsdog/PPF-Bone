"""Origins of installed animation, independent of the next Transfer."""

CURVE_START_FRAME = "_ppf_curve_start_frame_v1"


def _mesh_and_bone_origins(scene):
    from . import rigid_pose

    key = scene.get(rigid_pose.CACHE_KEY)
    index = (rigid_pose._cache.get(key) or rigid_pose._read_index_file(key)) if key else None
    origins = {int(index["receipt"]["start_frame"])} if index is not None else set()
    for obj in scene.objects:
        for mod in obj.modifiers:
            if mod.type == "MESH_CACHE" and mod.name == "ContactSolverCache" and mod.filepath:
                origins.add(int(mod.frame_start))
    return origins


def legacy_curve_start_frame(scene):
    """Old curve files have no origin; use the installed result's timeline."""
    origins = _mesh_and_bone_origins(scene)
    return next(iter(origins)) if len(origins) == 1 else int(scene.frame_start)


def curve_start_frame(obj, legacy_origin):
    saved = obj.get(CURVE_START_FRAME)
    return saved if type(saved) is int else legacy_origin


def cache_start_frame(scene):
    from . import pc2

    origins = _mesh_and_bone_origins(scene)
    legacy_origin = next(iter(origins)) if len(origins) == 1 else int(scene.frame_start)
    for obj in pc2._scene_curve_objects(scene):
        key = pc2.object_pc2_key_readonly(obj)
        if key and (pc2.has_curve_cache(key) or type(obj.get(CURVE_START_FRAME)) is int):
            origins.add(curve_start_frame(obj, legacy_origin))
    if len(origins) == 1:
        return next(iter(origins))
    if len(origins) > 1:
        raise ValueError("模拟缓存的起始帧不一致，请清除本地动画后重新获取。")
    from .encoder import resolve_start_frame_or_default

    return resolve_start_frame_or_default(scene)
