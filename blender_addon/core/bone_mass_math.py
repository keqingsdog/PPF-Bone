"""Read-only SI mass algebra and strict triangle-shell volume, without Blender.

Intersection contacts are allowed only on an actual shared topological edge or
vertex. Unsupported numerical/budget cases refuse rather than estimate volume.
"""
import math
import struct
import time
from collections import defaultdict

DENSITY_MIN = .01
DENSITY_MAX = 10000.
RNA_DENSITY_MIN = struct.unpack('<f', struct.pack('<f', DENSITY_MIN))[0]
MAX_TRIANGLES = 40000
MAX_VERTICES = MAX_TRIANGLES//2+2
MAX_INTERSECTION_PAIRS = 250000
MAX_BROAD_PHASE_CHECKS = 500000


def validate_mesh_counts(vertex_count, face_count, loop_count):
    """Cheap refusal before a disposable BMesh can allocate author geometry.

    A connected outward closed triangulated manifold obeys V=F/2+2-2g,
    so F<=40000 implies V<=20002. Polygon triangulation yields loops-2*faces
    triangles without adding vertices. Counts do not certify valid geometry.
    """
    if vertex_count > MAX_VERTICES or face_count > MAX_TRIANGLES or loop_count > 3*MAX_TRIANGLES:
        raise ValueError('代理顶点、面或环数超过质量换算的资源预算；请简化代理')
    if vertex_count < 4 or face_count < 4 or loop_count < 3*face_count:
        raise ValueError('代理必须是闭合三维网格')
    if loop_count-2*face_count > MAX_TRIANGLES:
        raise ValueError('代理三角化后的面数超过质量换算预算；请简化代理')


def positive_finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{label}必须是有限正值')
    return float(value)


def valid_density(value):
    value = positive_finite(value, '密度（kg/m³）')
    if not DENSITY_MIN <= value <= DENSITY_MAX:
        raise ValueError('密度超出 0.01–10000 kg/m³；请调整目标质量或代理体积')
    return value


def valid_authored_density(value):
    """Accept the exact legacy RNA binary32 minimum without changing it.

    This is only for reading an authored FloatProperty. Derived densities must
    still pass valid_density's exact SI interval before any RNA/material write.
    No tolerance band admits other values below 0.01 and nothing is clamped.
    """
    value = positive_finite(value, '密度（kg/m³）')
    return value if value == RNA_DENSITY_MIN else valid_density(value)


def density_for_mass(mass_kg, volume_m3):
    return valid_density(positive_finite(mass_kg, '目标质量（kg）') /
                         positive_finite(volume_m3, '闭合体积（m³）'))


def mass_for_density(density, volume_m3):
    mass = valid_authored_density(density) * positive_finite(volume_m3, '闭合体积（m³）')
    return positive_finite(mass, '推导质量（kg）')


def resolve_mass(mode, density, target_mass_kg, volume_m3):
    if mode == 'MASS':
        return density_for_mass(target_mass_kg, volume_m3), positive_finite(target_mass_kg, '目标质量（kg）')
    if mode == 'DENSITY':
        return valid_authored_density(density), mass_for_density(density, volume_m3)
    raise ValueError('未知的质量输入模式')


def _sub(a, b): return tuple(x-y for x, y in zip(a, b))
def _add(a, b): return tuple(x+y for x, y in zip(a, b))
def _mul(a, s): return tuple(x*s for x in a)
def _dot(a, b): return sum(x*y for x, y in zip(a, b))
def _cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def _norm(a): return math.sqrt(_dot(a, a))


def _inside(point, tri, normal, tol):
    return all(_dot(normal, _cross(_sub(tri[(i+1)%3], tri[i]), _sub(point, tri[i])))
               >= -tol*_norm(_sub(tri[(i+1)%3], tri[i])) for i in range(3))


def _coplanar_polygon(a, b, normal, tol):
    polygon = list(a)
    for i in range(3):
        start, end = b[i], b[(i+1)%3]
        edge = _sub(end, start)
        def signed(p): return _dot(normal, _cross(edge, _sub(p, start)))
        if not polygon: break
        previous = polygon[-1]; previous_d = signed(previous); out = []
        for current in polygon:
            current_d = signed(current)
            previous_in = previous_d >= -tol*_norm(edge)
            current_in = current_d >= -tol*_norm(edge)
            if previous_in != current_in:
                denominator = previous_d-current_d
                if denominator:
                    out.append(_add(previous, _mul(_sub(current, previous), previous_d/denominator)))
            if current_in: out.append(current)
            previous, previous_d = current, current_d
        polygon = out
    return polygon


def _coplanar_segment(start, end, tri, normal, tol):
    """Clip an in-plane segment, including endpoints outside the triangle."""
    low, high = 0., 1.
    for i in range(3):
        edge = _sub(tri[(i+1)%3], tri[i]); bound = tol*_norm(edge)
        ds = _dot(normal, _cross(edge, _sub(start, tri[i])))
        de = _dot(normal, _cross(edge, _sub(end, tri[i])))
        if ds < -bound and de < -bound: return []
        if ds < -bound: low = max(low, ds/(ds-de))
        if de < -bound: high = min(high, ds/(ds-de))
    if low > high: return []
    return [_add(start, _mul(_sub(end, start), f)) for f in (low, high)]


def _intersections(a, b, na, nb, tol):
    if _norm(_cross(na, nb)) <= 1e-10 and all(abs(_dot(nb, _sub(p, b[0]))) <= tol for p in a):
        polygon = _coplanar_polygon(a, b, nb, tol)
        area = sum(_norm(_cross(_sub(polygon[i], polygon[0]), _sub(polygon[i+1], polygon[0])))
                   for i in range(1, len(polygon)-1))/2 if polygon else 0.
        return polygon, area > tol*tol
    points = []
    for tri, other, normal in ((a, b, nb), (b, a, na)):
        for i in range(3):
            start, end = tri[i], tri[(i+1)%3]
            ds, de = (_dot(normal, _sub(p, other[0])) for p in (start, end))
            if abs(ds) <= tol and abs(de) <= tol:
                points.extend(_coplanar_segment(start, end, other, normal, tol))
            else:
                for p, d in ((start, ds), (end, de)):
                    if abs(d) <= tol and _inside(p, other, normal, tol): points.append(p)
                if ds*de < 0:
                    p = _add(start, _mul(_sub(end, start), ds/(ds-de)))
                    if _inside(p, other, normal, tol): points.append(p)
    return points, False


def _allowed_contact(point, shared, vertices, tol):
    if len(shared) == 1: return _norm(_sub(point, vertices[shared[0]])) <= tol
    if len(shared) == 2:
        a, b = (vertices[i] for i in shared); edge = _sub(b, a); length = _norm(edge)
        fraction = _dot(_sub(point, a), edge)/_dot(edge, edge)
        return (-tol/length <= fraction <= 1+tol/length and
                _norm(_sub(point, _add(a, _mul(edge, fraction)))) <= tol)
    return False


def _candidate_pairs(boxes, axis, tol):
    active = []
    checks = 0
    for i in sorted(range(len(boxes)), key=lambda i: boxes[i][0][axis]):
        low, high = boxes[i]
        active = [j for j in active if boxes[j][1][axis] >= low[axis]-tol]
        for j in active:
            checks += 1
            if checks > MAX_BROAD_PHASE_CHECKS:
                raise ValueError('代理宽相交检查超过验证预算；请简化代理')
            if all(boxes[j][1][k] >= low[k]-tol and high[k] >= boxes[j][0][k]-tol for k in range(3)):
                yield j, i
        active.append(i)


def strict_volume(vertices, triangles, world_determinant=1.):
    """Measure the author base shell. Never recalculate winding or take abs(V).

    A bounded streaming sweep covers every triangle AABB contact; adjacent
    triangles are included. No external candidate list can omit intersections.
    """
    started = time.perf_counter()
    determinant = positive_finite(world_determinant, '对象变换行列式；镜像和零缩放不支持，')
    vertices = [tuple(float(c) for c in p) for p in vertices]
    triangles = [tuple(t) for t in triangles]
    if len(vertices) < 4 or not triangles: raise ValueError('代理必须是闭合三维网格')
    if len(triangles) > MAX_TRIANGLES: raise ValueError('代理面数超过质量换算的验证预算；请简化代理')
    if any(len(p) != 3 or any(not math.isfinite(c) for c in p) for p in vertices):
        raise ValueError('代理顶点包含非有限坐标')
    edges = defaultdict(list); incident = defaultdict(set)
    for i, tri in enumerate(triangles):
        if len(tri) != 3 or len(set(tri)) != 3 or any(type(v) is not int or not 0 <= v < len(vertices) for v in tri):
            raise ValueError('代理包含无效或退化三角面')
        for v in tri: incident[v].add(i)
        for a, b in zip(tri, tri[1:]+tri[:1]): edges[tuple(sorted((a, b)))].append((i, a, b))
    if len(incident) != len(vertices): raise ValueError('代理包含孤立顶点')
    if any(len(pair) != 2 for pair in edges.values()): raise ValueError('代理有开口或非流形边；质量换算要求闭合网格')
    if any(pair[0][1:] != pair[1][1:][::-1] for pair in edges.values()): raise ValueError('代理面方向不一致；请检查法线')
    adjacent = defaultdict(set)
    for pair in edges.values():
        a, b = pair[0][0], pair[1][0]; adjacent[a].add(b); adjacent[b].add(a)
    def reachable(first, allowed):
        seen = {first}; pending = [first]
        while pending:
            for nxt in adjacent[pending.pop()] & allowed - seen: seen.add(nxt); pending.append(nxt)
        return seen
    if len(reachable(0, set(range(len(triangles))))) != len(triangles):
        raise ValueError('质量换算只支持单一闭合连通壳；请拆分或连接代理')
    if any(reachable(next(iter(faces)), faces) != faces for faces in incident.values()):
        raise ValueError('代理含非流形顶点；不能计算有效质量')
    low = tuple(min(p[k] for p in vertices) for k in range(3)); high = tuple(max(p[k] for p in vertices) for k in range(3))
    diagonal = _norm(_sub(high, low)); tol = max(diagonal*1e-9, 1e-12)
    center = _mul(_add(low, high), .5)
    normals = []; boxes = []
    for tri in triangles:
        points = [vertices[i] for i in tri]; normal = _cross(_sub(points[1], points[0]), _sub(points[2], points[0])); length = _norm(normal)
        if length <= tol*tol: raise ValueError('代理包含退化三角面')
        normals.append(_mul(normal, 1/length))
        boxes.append((tuple(min(p[k] for p in points) for k in range(3)), tuple(max(p[k] for p in points) for k in range(3))))
    volume = math.fsum(_dot(_sub(vertices[a], center), _cross(_sub(vertices[b], center), _sub(vertices[c], center)))/6 for a, b, c in triangles)
    if not math.isfinite(volume) or volume <= max(diagonal**3*1e-14, 1e-24):
        raise ValueError('代理有向体积非正或接近零；请检查外向法线，不能使用绝对体积')
    candidate_pairs = _candidate_pairs(boxes, max(range(3), key=lambda k: high[k]-low[k]), tol)
    tested = 0; seen = set()
    for a, b in candidate_pairs:
        if a == b: continue
        pair = (min(a, b), max(a, b))
        if pair in seen: continue
        seen.add(pair); tested += 1
        if tested > MAX_INTERSECTION_PAIRS: raise ValueError('代理自交检查超过验证预算；请简化代理')
        ta, tb = triangles[a], triangles[b]
        shared = tuple(sorted(set(ta) & set(tb)))
        # A triangle lying strictly to one side of the other's plane except
        # for the actual common edge/vertex can touch only that common feature.
        # This is a geometric separation proof, not an adjacency exemption:
        # coplanar overlap and folded incident triangles still use narrow phase.
        separated = False
        for first, second, normal in ((ta, tb, normals[a]), (tb, ta, normals[b])):
            distances = [_dot(normal, _sub(vertices[i], vertices[first[0]])) for i in second if i not in shared]
            if distances and (all(d > tol for d in distances) or all(d < -tol for d in distances)):
                separated = True; break
        if separated: continue
        points, overlap = _intersections([vertices[i] for i in ta], [vertices[i] for i in tb], normals[a], normals[b], tol)
        if overlap or any(not _allowed_contact(p, shared, vertices, tol*4) for p in points):
            raise ValueError('代理表面自交、接触或重复重叠；不能计算有效质量')
    volume = positive_finite(volume*determinant, '缩放后体积（m³）')
    return {'volume_m3': volume, 'vertices': len(vertices), 'triangles': len(triangles),
            'intersection_pairs_tested': tested, 'elapsed_seconds': time.perf_counter()-started,
            'world_determinant': determinant, 'object_scale_included': True}
