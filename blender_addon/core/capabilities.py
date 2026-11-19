"""Read-only Blender capability probes, independent of version numbers.

These report available APIs, not successful rendering or simulation. In
particular, Sand uses a mesh of grain centres and a Geometry Nodes display;
it does not require writable PointCloud data blocks.
"""

from functools import lru_cache
import bpy


def _rna_has(type_name, member, collection="properties"):
    cls = getattr(bpy.types, type_name, None)
    rna = getattr(cls, "bl_rna", None)
    return rna is not None and member in getattr(rna, collection, ())


def _node_available(name):
    node_type = getattr(bpy.types, name, None)
    return (node_type is not None and
            callable(getattr(node_type, "is_registered_node_type", None)) and
            node_type.is_registered_node_type())


def _operator_available(module, name):
    try:
        getattr(getattr(bpy.ops, module), name).get_rna_type()
        return True
    except (AttributeError, RuntimeError, KeyError):
        return False


def export_capability(kind, *, build_options=None, operator_available=None):
    """Keep working operators enabled if a build flag is unavailable."""
    operator = {"usd": "usd_export", "alembic": "alembic_export"}[kind]
    available = _operator_available if operator_available is None else operator_available
    options = bpy.app.build_options if build_options is None else build_options
    if not available("wm", operator):
        return False, "当前 Blender 未提供此导出器"
    if getattr(options, kind, None) is False:
        return False, "当前 Blender 构建未包含此导出器"
    return True, ""


@lru_cache(maxsize=1)
def api_capabilities():
    """Static RNA/API facts only; no datablocks, shaders or UI state writes."""
    import gpu
    required_nodes = ("NodeGroupInput", "NodeGroupOutput",
                      "GeometryNodeMeshToPoints", "GeometryNodeSetMaterial")
    missing = tuple(name for name in required_nodes if not _node_available(name))
    geometry_interface = _rna_has("NodeTreeInterface", "new_socket", "functions")
    mesh_centres = callable(getattr(bpy.types.Mesh, "from_pydata", None))
    sand_ready = not missing and geometry_interface and mesh_centres
    pose_route = ("POSE_BONE" if _rna_has("PoseBone", "select") else
                  "BONE" if _rna_has("Bone", "select") else "UNAVAILABLE")
    return {
        "sand_mesh_centres": mesh_centres,
        "sand_nodes_missing": missing,
        "sand_geometry_interface": geometry_interface,
        "sand_ready": sand_ready,
        "pointcloud_add": _rna_has("PointCloudPoints", "add", "functions"),
        "pose_selection_route": pose_route,
        "action_layers": _rna_has("Action", "layers"),
        "action_legacy_fcurves": _rna_has("Action", "fcurves"),
        "mesh_cache": getattr(bpy.types, "MeshCacheModifier", None) is not None,
        # Actual shaders are chosen lazily in ui.gpu_compat in a GPU context.
        # Missing POINT_UNIFORM_COLOR is handled there, never a UI veto here.
        "gpu_shader_create_info": hasattr(gpu.types, "GPUShaderCreateInfo"),
        "gpu_builtin_factory": callable(getattr(gpu.shader, "from_builtin", None)),
    }


def sand_creation_ui(caps=None):
    caps = api_capabilities() if caps is None else caps
    if not caps["sand_ready"]:
        if not caps["sand_mesh_centres"]:
            reason = "当前 Blender 缺少网格创建接口"
        elif not caps["sand_geometry_interface"]:
            reason = "当前 Blender 缺少几何节点接口"
        else:
            reason = "当前 Blender 缺少沙粒显示节点"
        return False, reason, "请使用支持此功能的 Blender 构建"
    if not caps["pointcloud_add"]:
        return True, "兼容方案：网格顶点生成沙粒", "显示使用几何节点，沙粒物理不变"
    return True, "从闭合网格生成沙粒", "显示使用几何节点，沙粒物理不变"
