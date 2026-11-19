"""Point-shader capability adapter; retains the builtin when it exists."""
import bpy
import gpu

_point_shaders = {}
_custom_ids = set()
_routes = {}


def _context_key():
    # OffScreen uses an FBO in the current Window GPU context. Cache per
    # Window/backend so another Window never borrows a context-owned shader.
    window = bpy.context.window
    return (gpu.platform.backend_type_get(), window.as_pointer() if window else 0)


def point_uniform_shader():
    key = _context_key()
    if key in _point_shaders:
        return _point_shaders[key]
    try:
        shader = gpu.shader.from_builtin('POINT_UNIFORM_COLOR')
        route = 'BUILTIN_POINT_UNIFORM_COLOR'
    except ValueError as exc:
        if 'POINT_UNIFORM_COLOR' not in str(exc):
            raise
        info = gpu.types.GPUShaderCreateInfo()
        info.vertex_in(0, 'VEC3', 'pos')
        info.push_constant('MAT4', 'ModelViewProjectionMatrix')
        info.push_constant('VEC4', 'color')
        info.push_constant('FLOAT', 'size')
        info.vertex_source('void main() { gl_Position = ModelViewProjectionMatrix * vec4(pos, 1.0); gl_PointSize = size; }')
        info.fragment_out(0, 'VEC4', 'fragColor')
        info.fragment_source('void main() { fragColor = color; }')
        shader = gpu.shader.create_from_info(info)
        _custom_ids.add(id(shader))
        route = 'CUSTOM_POINT_SIZE_MVP'
    _point_shaders[key] = shader
    _routes[key] = route
    return shader


def bind_shader(shader):
    shader.bind()
    if id(shader) in _custom_ids:
        shader.uniform_float('ModelViewProjectionMatrix',
            gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix())


def point_shader_routes():
    return [{'backend': key[0], 'window': key[1], 'route': route}
            for key, route in _routes.items()]


def clear_point_shaders():
    _point_shaders.clear()
    _custom_ids.clear()
    _routes.clear()
