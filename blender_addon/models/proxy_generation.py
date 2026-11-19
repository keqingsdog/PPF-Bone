"""Bounded proxy presets and read-only first-creation resolution."""
import math

PROXY_QUALITY_ITEMS = [
    ('AUTO', '自动（刚体低 / 软体高）', '首次创建随物理模型选择精度', 5),
    ('VERY_LOW', '极低', '最轻量的封闭代理', 2),
    ('LOW', '低', '原刚体默认分辨率', 0),
    ('MEDIUM', '中', '原软体默认分辨率', 3),
    ('HIGH', '高', '保持原高精度分辨率；Box每面2x2细分', 1),
    ('VERY_HIGH', '极高', '更密的曲面；Box每面4x4细分，长链请谨慎使用', 4),
]
PROXY_FACE_ITEMS = [('QUAD', '四边面', '全部四边面，包含调整后的极点连接；求解导出仍三角化'),
                    ('TRI', '三角面', '显式三角化生成网格')]
# segments, hemisphere layers, shaft intermediate rings. LOW and MEDIUM
# retain the previous RIGID/SOFT default vertex positions exactly.
PROXY_RESOLUTIONS = {'VERY_LOW': (8, 2, 0), 'LOW': (12, 3, 0),
                     'MEDIUM': (16, 4, 3), 'HIGH': (32, 5, 10),
                     'VERY_HIGH': (40, 6, 12)}


def effective_proxy_quality(settings):
    from .new_binding_defaults import uses_new_geometry_defaults, uses_new_binding_preset
    value = settings.proxy_quality
    if value != 'AUTO':
        return value
    if not uses_new_geometry_defaults(settings):
        if not settings.is_property_set('proxy_quality'):
            return 'HIGH'  # legacy unset HIGH fallback
        return 'MEDIUM' if settings.physical_model == 'SOFT' else 'LOW'
    value = settings.new_proxy_quality
    if value != 'AUTO':
        return value
    model = settings.new_binding_physical_model if uses_new_binding_preset(settings) else settings.physical_model
    return 'HIGH' if model == 'SOFT' else 'LOW'


def effective_proxy_face_mode(settings):
    return getattr(settings, 'proxy_face_mode', 'QUAD')


def triangulate_faces(faces, vertices=None):
    if vertices is not None:
        # Exactly the same canonical loop tessellation consumed by the native
        # encoder, including the adjusted nonplanar pole quads.
        import bpy
        mesh = bpy.data.meshes.new('PPF_Temporary_Tessellation')
        try:
            mesh.from_pydata(vertices, [], faces)
            mesh.calc_loop_triangles()
            return [tuple(tri.vertices) for tri in mesh.loop_triangles]
        finally:
            bpy.data.meshes.remove(mesh)
    # Match the historical (a,b,c),(b,d,c) split for ring quads (a,b,d,c).
    result = []
    for face in faces:
        if len(face) == 4:
            a, b, d, c = face
            result.extend(((a, b, c), (b, d, c)))
        else:
            result.append(tuple(face))
    return result


def round_geometry(length, radius, quality, face_mode='QUAD', sphere=False, legacy=False):
    if quality not in PROXY_RESOLUTIONS or face_mode not in {'QUAD', 'TRI'}:
        raise ValueError('Unknown proxy resolution or face mode')
    if not math.isfinite(length) or not math.isfinite(radius) or radius <= 0 or (not sphere and length <= 2 * radius):
        raise ValueError('Capsule radius must be less than half proxy length')
    segments, layers, straight = PROXY_RESOLUTIONS[quality]
    half = 0. if sphere else length / 2 - radius
    rings = [(-half-radius*math.cos(j*math.pi/(2*layers)), radius*math.sin(j*math.pi/(2*layers))) for j in range(1, layers+1)]
    if not sphere:
        rings += [(-half+2*half*j/(straight+1), radius) for j in range(1, straight+1)]
    # Sphere has one equator, never two coincident rings.
    start = 1 if sphere else 0
    rings += [(half+radius*math.sin(j*math.pi/(2*layers)), radius*math.cos(j*math.pi/(2*layers))) for j in range(start, layers)]
    extent = radius if sphere else length/2
    vertices = [(-extent, 0., 0.)] + [(x, r*math.cos(i*2*math.pi/segments), r*math.sin(i*2*math.pi/segments)) for x, r in rings for i in range(segments)] + [(extent, 0., 0.)]
    faces = ([(0, 1+(i+1)%segments, 1+i) for i in range(segments)] if legacy else
             [(0, 1+(i+2)%segments, 1+(i+1)%segments, 1+i) for i in range(0, segments, 2)])
    for j in range(len(rings)-1):
        for i in range(segments):
            a=1+j*segments+i; b=1+j*segments+(i+1)%segments; c=a+segments; d=b+segments
            faces.append((a,b,d,c))
    base=1+(len(rings)-1)*segments
    faces += ([(len(vertices)-1, base+i, base+(i+1)%segments) for i in range(segments)] if legacy else
              [(len(vertices)-1, base+i, base+(i+1)%segments, base+(i+2)%segments) for i in range(0, segments, 2)])
    return vertices, triangulate_faces(faces, None if legacy else vertices) if face_mode == 'TRI' else faces
