"""Exact draft polyline preview with bounded Blender icon resources."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import zlib

_LIMIT = 16
_icons = None
_directory = None
_cache = OrderedDict()


def polyline(table, validate_table, preview_table):
    """Draw actual knots and exact endpoint tails; values retain table units."""
    table = validate_table(table)
    points = [tuple(node) for node in table['nodes']]
    if table['domain'] == 'ODD':
        points = [(-x, -f) for x, f in reversed(points[1:])] + points
    span = points[-1][0] - points[0][0]
    end_x = points[-1][0] + span * .12
    points.append((end_x, preview_table(table, end_x)['gradient']))
    if table['domain'] != 'RADIAL':
        start_x = points[0][0] - span * .12
        points.insert(0, (start_x, preview_table(table, start_x)['gradient']))
    return points


def encode_plot(points, width=320, height=200):
    """Straight segments, zero axes and knots; no smoothing or force changes."""
    pixels = bytearray(bytes((32, 34, 38)) * width * height)
    xmin, xmax = points[0][0], points[-1][0]
    low, high = min(0, *(f for _, f in points)), max(0, *(f for _, f in points))
    if high == low:
        low, high = -1, 1
    margin = (high - low) * .12
    low, high = low - margin, high + margin

    def xy(x, f):
        return (round(18 + (x - xmin) / (xmax - xmin) * (width - 36)),
                round(height - 18 - (f - low) / (high - low) * (height - 36)))

    def put(x, y, color):
        if 0 <= x < width and 0 <= y < height:
            i = 3 * (y * width + x)
            pixels[i:i + 3] = bytes(color)

    def line(a, b, color):
        x0, y0 = a
        x1, y1 = b
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for i in range(steps + 1):
            put(round(x0 + (x1 - x0) * i / steps),
                round(y0 + (y1 - y0) * i / steps), color)

    line(xy(xmin, 0), xy(xmax, 0), (108, 111, 119))
    if xmin <= 0 <= xmax:
        line(xy(0, low), xy(0, high), (108, 111, 119))
    for a, b in zip(points, points[1:]):
        line(xy(*a), xy(*b), (92, 192, 245))
    for x, f in points:
        px, py = xy(x, f)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                put(px + dx, py + dy, (233, 241, 247))

    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)

    raw = b''.join(b'\0' + pixels[y * width * 3:(y + 1) * width * 3] for y in range(height))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


def preview_icon(table, validate_table, preview_table):
    """Read-only author preview. Nothing is stored in the scene or presets."""
    global _icons, _directory
    table = validate_table(table)
    shape = {key: table[key] for key in ('domain', 'tail', 'nodes')}
    key = hashlib.sha256(json.dumps(shape, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('ascii')).hexdigest()
    if key in _cache:
        _cache.move_to_end(key)
        # Blender may allocate icon IDs after the first load. Never latch an
        # initial zero ID, which would leave the displayed graph blank forever.
        return _icons[key].icon_id
    points = polyline(table, validate_table, preview_table)
    png = encode_plot(points)
    import bpy.utils.previews
    if _icons is None:
        _icons = bpy.utils.previews.new()
    if _directory is None:
        _directory = Path(tempfile.mkdtemp(prefix='ppf_restoring_preview_')).resolve()
    path = _directory / (key + '.png')
    path.write_bytes(png)
    try:
        icon = _icons.load(key, str(path), 'IMAGE')
    except Exception:
        path.unlink(missing_ok=True)
        raise
    _cache[key] = path
    while len(_cache) > _LIMIT:
        old_key, old_path = _cache.popitem(last=False)
        del _icons[old_key]
        old_path.unlink(missing_ok=True)
    return icon.icon_id


def clear():
    """Unload icons before deleting only the files created by this cache."""
    global _icons, _directory
    if _icons is not None:
        import bpy.utils.previews
        bpy.utils.previews.remove(_icons)
        _icons = None
    for path in _cache.values():
        path.unlink(missing_ok=True)
    _cache.clear()
    if _directory is not None:
        _directory.rmdir()
        _directory = None
