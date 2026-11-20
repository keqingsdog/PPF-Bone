"""Dormant V1 restoring-curve author model. No bpy, registration or solver edits.

The public transport API is validate_author_map() and export_author(). Native
mass resolution, capability and force application are deliberately separate.
"""
import copy
import json
import math
import re
import struct
import uuid

VERSION = 1
MODES = {'LEGACY', 'COMMON', 'ADVANCED'}
CHANNELS = {'twist_x', 'swing_y', 'swing_z', 'anchor'}
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_TABLES = 4096
MAX_BINDINGS = 65536
MAX_NODES = 262144
ID_PATTERN = re.compile(r'[A-Za-z0-9_-]{1,64}\Z')


class CurveError(ValueError):
    def __init__(self, code, field, detail):
        self.code, self.field, self.detail = code, field, detail
        super().__init__(f'{code}: {field}: {detail}')


def fail(code, field, detail):
    raise CurveError(code, field, detail)


def exact(value, keys, field):
    if type(value) is not dict or set(value) != set(keys):
        fail('RESTORING_SCHEMA_INVALID', field, f'expected exact keys {sorted(keys)}')


def number(value, field, *, positive=False, nonnegative=False):
    if type(value) not in (int, float):
        fail('RESTORING_SCHEMA_INVALID', field, 'expected a finite number, not bool')
    try:
        value = float(value)
    except (ValueError, OverflowError):
        fail('RESTORING_CURVE_INVALID', field, 'unrepresentable number')
    if not math.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        fail('RESTORING_CURVE_INVALID', field, 'invalid finite range')
    return value


def f32(value, field):
    value = number(value, field)
    try:
        narrowed = struct.unpack('<f', struct.pack('<f', value))[0]
    except (OverflowError, struct.error):
        fail('RESTORING_CURVE_INVALID', field, 'float32 overflow')
    if not math.isfinite(narrowed) or (value != 0 and narrowed == 0):
        fail('RESTORING_CURVE_INVALID', field, 'float32 overflow or nonzero underflow')
    return narrowed


def canonical_uuid(value, field):
    if type(value) is not str:
        fail('RESTORING_TARGET_INVALID', field, 'expected canonical non-world UUID')
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        fail('RESTORING_TARGET_INVALID', field, 'expected canonical non-world UUID')
    if str(parsed) != value or parsed.int == 0:
        fail('RESTORING_TARGET_INVALID', field, 'expected canonical non-world UUID')
    return value


def target_key(target):
    if type(target) is not dict:
        fail('RESTORING_TARGET_INVALID', 'target', 'expected target map')
    if target.get('kind') == 'JOINT':
        exact(target, {'kind', 'joint_uuid', 'channel'}, 'target')
        canonical_uuid(target['joint_uuid'], 'target.joint_uuid')
        if type(target['channel']) is not str or target['channel'] not in CHANNELS:
            fail('RESTORING_TARGET_INVALID', 'target.channel', 'unsupported channel')
        return ('JOINT', target['joint_uuid'], target['channel'])
    if target.get('kind') == 'STITCH_PAIR':
        exact(target, {'kind', 'pair_id'}, 'target')
        canonical_uuid(target['pair_id'], 'target.pair_id')
        return ('STITCH_PAIR', target['pair_id'])
    fail('RESTORING_TARGET_INVALID', 'target.kind', 'unsupported target')


def _areas(nodes):
    """Integrals from zero, accumulated independently on each side."""
    zero = next(i for i, node in enumerate(nodes) if node == (0.0, 0.0))
    areas = [0.0] * len(nodes)
    for i in range(zero + 1, len(nodes)):
        dx = nodes[i][0] - nodes[i-1][0]
        areas[i] = areas[i-1] + (nodes[i][1] + nodes[i-1][1]) * dx / 2
    for i in range(zero - 1, -1, -1):
        dx = nodes[i+1][0] - nodes[i][0]
        areas[i] = areas[i+1] - (nodes[i][1] + nodes[i+1][1]) * dx / 2
    return areas


def validate_table(table):
    exact(table, {'table_id', 'domain', 'tail', 'nodes'}, 'table')
    if type(table['table_id']) is not str or not ID_PATTERN.fullmatch(table['table_id']):
        fail('RESTORING_SCHEMA_INVALID', 'table_id', 'invalid ASCII identifier')
    if table['domain'] not in ('SIGNED', 'ODD', 'RADIAL') or table['tail'] not in ('LINEAR', 'HOLD_EFFORT'):
        fail('RESTORING_SCHEMA_INVALID', table['table_id'], 'invalid domain or tail')
    if type(table['nodes']) is not list or not 2 <= len(table['nodes']) <= 64:
        fail('RESTORING_CURVE_INVALID', table['table_id'], 'expected 2..64 stored nodes')
    nodes = []
    for i, node in enumerate(table['nodes']):
        if type(node) is not list or len(node) != 2:
            fail('RESTORING_SCHEMA_INVALID', f'{table["table_id"]}.nodes[{i}]', 'expected [x,f]')
        nodes.append(tuple(f32(v, f'{table["table_id"]}.nodes[{i}]') for v in node))
    if nodes.count((0.0, 0.0)) != 1:
        fail('RESTORING_CURVE_INVALID', table['table_id'], 'expected exactly [0,0]')
    if table['domain'] == 'SIGNED':
        if not nodes[0][0] < 0 < nodes[-1][0]:
            fail('RESTORING_CURVE_INVALID', table['table_id'], 'SIGNED requires both sides of zero')
    elif nodes[0] != (0.0, 0.0) or any(x < 0 for x, _ in nodes):
        fail('RESTORING_CURVE_INVALID', table['table_id'], 'nonnegative domain starts at [0,0]')
    for i, ((x0, f0), (x1, f1)) in enumerate(zip(nodes, nodes[1:])):
        if x1 <= x0 or f1 < f0:
            fail('RESTORING_CURVE_INVALID', table['table_id'], 'coordinates collapse or effort decreases after float32 narrowing')
        dx=f32(x1-x0, f'{table["table_id"]}.dx[{i}]')
        df=f32(f1-f0, f'{table["table_id"]}.df[{i}]')
        f32(df/dx, f'{table["table_id"]}.slope[{i}]')
    for area in _areas(nodes):
        f32(area, f'{table["table_id"]}.integral')
    return copy.deepcopy(table)


def _json_bytes(value):
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(',', ':'), sort_keys=True).encode('utf-8')
    except (ValueError, TypeError, OverflowError, RecursionError):
        fail('RESTORING_SCHEMA_INVALID', 'document', 'expected finite JSON values')


def validate_author_map(document):
    exact(document, {'version', 'tables', 'bindings', 'stitch_pairs'}, 'author')
    if type(document['version']) is not int or document['version'] != VERSION:
        fail('RESTORING_VERSION_UNSUPPORTED', 'version', 'supported version is 1')
    for key in ('tables', 'bindings', 'stitch_pairs'):
        if type(document[key]) is not list:
            fail('RESTORING_SCHEMA_INVALID', key, 'expected array')
    if len(document['tables']) > MAX_TABLES or len(document['bindings']) > MAX_BINDINGS:
        fail('RESTORING_BUDGET_EXCEEDED', 'author', 'table or binding budget exceeded')
    tables = {}
    total_nodes = 0
    for table in document['tables']:
        checked = validate_table(table)
        key = checked['table_id']
        if key in tables:
            fail('RESTORING_SCHEMA_INVALID', key, 'duplicate table_id')
        tables[key] = checked
        total_nodes += len(checked['nodes'])
    if total_nodes > MAX_NODES or len(_json_bytes(document)) > MAX_JSON_BYTES:
        fail('RESTORING_BUDGET_EXCEEDED', 'author', 'node or JSON budget exceeded')
    enabled = set(); pair_targets = set()
    for binding in document['bindings']:
        if type(binding) is not dict:
            fail('RESTORING_SCHEMA_INVALID', 'binding', 'expected binding map')
        mode = binding.get('mode')
        extra = {'damping'} if mode == 'ABSOLUTE_SI' else {
            'frequency_hz', 'damping_ratio', 'coordinate_scale', 'mass_policy'} if mode == 'AUTO_FREQUENCY' else None
        if extra is None:
            fail('RESTORING_SCHEMA_INVALID', 'binding.mode', 'unsupported binding mode')
        exact(binding, {'target', 'enabled', 'table_id', 'mode'} | extra, 'binding')
        key = target_key(binding['target'])
        if type(binding['enabled']) is not bool:
            fail('RESTORING_SCHEMA_INVALID', 'binding.enabled', 'expected bool')
        if type(binding['table_id']) is not str or binding['table_id'] not in tables:
            fail('RESTORING_TARGET_INVALID', str(key), 'unknown table_id')
        table = tables[binding['table_id']]
        is_anchor = key[0] == 'JOINT' and key[2] == 'anchor'
        if (is_anchor and table['domain'] != 'RADIAL') or (not is_anchor and table['domain'] == 'RADIAL'):
            fail('RESTORING_TARGET_INVALID', str(key), 'table domain incompatible with channel')
        if mode == 'ABSOLUTE_SI':
            f32(number(binding['damping'], 'damping', nonnegative=True), 'damping')
            scale = 1.0
        else:
            for field, positive in (('frequency_hz', True), ('damping_ratio', False), ('coordinate_scale', True)):
                f32(number(binding[field], field, positive=positive, nonnegative=not positive), field)
            if binding['mass_policy'] != 'LOCAL_REST_V1':
                fail('RESTORING_SCHEMA_INVALID', 'mass_policy', 'unsupported policy')
            scale = binding['coordinate_scale']
        if key[0] == 'JOINT' and not is_anchor:
            for x, _ in table['nodes']:
                if abs(f32(number(x, 'node.x') * scale, 'SI angle')) >= math.pi - .001:
                    fail('RESTORING_CURVE_INVALID', str(key), 'angular node outside chart domain')
        if binding['enabled']:
            if key in enabled:
                fail('RESTORING_CONFLICT', str(key), 'duplicate enabled target')
            enabled.add(key)
        if key[0] == 'STITCH_PAIR': pair_targets.add(key[1])
    pair_ids = set(); ordinals = set()
    for mapping in document['stitch_pairs']:
        exact(mapping, {'pair_id', 'cross_stitch_index'}, 'stitch_pairs')
        pair_id = canonical_uuid(mapping['pair_id'], 'pair_id')
        ordinal = mapping['cross_stitch_index']
        if type(ordinal) is not int or not 0 <= ordinal <= 0xffffffff:
            fail('RESTORING_TARGET_INVALID', pair_id, 'invalid input ordinal')
        if pair_id in pair_ids or ordinal in ordinals:
            fail('RESTORING_CONFLICT', pair_id, 'duplicate pair or input ordinal')
        pair_ids.add(pair_id); ordinals.add(ordinal)
    if pair_ids != pair_targets:
        fail('RESTORING_TARGET_INVALID', 'stitch_pairs', 'exact mapping required for every referenced pair')
    return copy.deepcopy(document)


def new_state():
    """Stored scenes begin in LEGACY. Creation into COMMON is an explicit op."""
    return {'mode': 'LEGACY', 'document': None}


def pair_identity(existing=None, *, duplicate=False):
    """Explicit creation/actual duplication helper; never called on export."""
    if duplicate or existing in (None, ''):
        return str(uuid.uuid4())
    return canonical_uuid(existing, 'pair_id')


def stitch_mapping(ordered_pair_ids, referenced_pair_ids):
    """Read-only mapping from the encoder's actual cross_stitch loop order.

    The caller provides IDs in precisely that loop's order, including legacy
    pairs with empty IDs. Only curve-referenced pairs are mapped; no identity
    is assigned while producing a hash or encoding a snapshot.
    """
    required={canonical_uuid(value,'referenced_pair_id') for value in referenced_pair_ids}
    seen=set();result=[]
    for ordinal,value in enumerate(ordered_pair_ids):
        if value in (None,''):continue
        value=canonical_uuid(value,'pair_id')
        if value in seen:fail('RESTORING_CONFLICT',value,'duplicate pair identity')
        seen.add(value)
        if value in required:result.append({'pair_id':value,'cross_stitch_index':ordinal})
    if {entry['pair_id'] for entry in result}!=required:
        fail('RESTORING_TARGET_INVALID','stitch_pairs','referenced pair absent from consumed input order')
    return result


def export_author(state=None):
    """B's exact schema export. None means omit the optional transport key.

    Explicitly present nonlegacy data is fully validated even if disabled.
    This function never resolves mass, repairs IDs or writes the author state.
    """
    if state is not None and type(state) is not dict:
        fail('RESTORING_SCHEMA_INVALID', 'state', 'expected author state map')
    if state is None or state.get('mode', 'LEGACY') == 'LEGACY':
        return None
    if type(state.get('mode')) is not str or state.get('mode') not in MODES:
        fail('RESTORING_SCHEMA_INVALID', 'mode', 'unknown author mode')
    document = validate_author_map(state.get('document'))
    if not any(binding['enabled'] for binding in document['bindings']):
        return None
    return document


def state_from_scene(scene):
    """Read persisted RNA without mutation; no migration of an old scene."""
    owner=getattr(scene,'ppf_restoring_curves',None)
    if owner is None or owner.mode=='LEGACY':return new_state()
    def pairs(items):
        value={}
        for key,item in items:
            if key in value:fail('RESTORING_SCHEMA_INVALID',key,'duplicate JSON key')
            value[key]=item
        return value
    raw=owner.document_json.encode('utf-8')
    if len(raw)>MAX_JSON_BYTES:fail('RESTORING_BUDGET_EXCEEDED','document_json','JSON budget exceeded')
    try:
        document=json.loads(raw,object_pairs_hook=pairs,
                            parse_constant=lambda value:fail('RESTORING_SCHEMA_INVALID',value,'nonfinite JSON constant'))
    except (ValueError,RecursionError) as exc:
        if isinstance(exc,CurveError):raise
        fail('RESTORING_SCHEMA_INVALID','document_json','invalid author JSON')
    return {'mode':owner.mode,'document':document}


def export_scene(scene, ordered_pair_ids=None, active_joint_ids=None, *,
                 cone_restoring_joint_ids=()):
    """Export the author library entries consumed by this payload only.

    The scene document is an authoring library and can retain records for
    retired joints or stitch pairs. Validate that complete library before
    projecting it onto the actual owners emitted by this encode operation.
    The projection is read-only: it never retires or rewrites author records.
    """
    state=state_from_scene(scene)
    if state['mode']=='LEGACY':return None
    # Strictly validate every authored record, including inactive library
    # entries, before selecting the entries consumed by this payload.
    document=validate_author_map(state['document'])
    joint_ids={canonical_uuid(value,'active_joint_id') for value in (active_joint_ids or ())}
    if ordered_pair_ids is None:
        from .groups import get_addon_data
        ordered_pair_ids=[pair.pair_id for pair in get_addon_data(scene).state.merge_pairs]
    active_pair_ids={canonical_uuid(value,'active_pair_id') for value in ordered_pair_ids if value not in (None,'')}
    bindings=[]
    for binding in document['bindings']:
        target=binding['target']
        if (target['kind'] == 'JOINT' and target['joint_uuid'] in cone_restoring_joint_ids
                and target['channel'] in ('swing_y', 'swing_z')):
            continue  # Circular swing uses the native linear fallback on both axes.
        if (target['kind']=='JOINT' and target['joint_uuid'] in joint_ids
                or target['kind']=='STITCH_PAIR' and target['pair_id'] in active_pair_ids):
            bindings.append(copy.deepcopy(binding))
    used_tables={binding['table_id'] for binding in bindings}
    referenced_pairs={binding['target']['pair_id'] for binding in bindings
                      if binding['target']['kind']=='STITCH_PAIR'}
    projected={
        'version':document['version'],
        'tables':[copy.deepcopy(table) for table in document['tables'] if table['table_id'] in used_tables],
        'bindings':bindings,
        'stitch_pairs':stitch_mapping(ordered_pair_ids,referenced_pairs),
    }
    state['document']=validate_author_map(projected)
    return export_author(state)


def preview_table(table, coordinate):
    """Author preview only: E, positive gradient and curvature in table units."""
    table = validate_table(table)
    x = number(coordinate, 'preview.coordinate')
    odd = table['domain'] == 'ODD'
    sign = -1 if odd and x < 0 else 1
    if odd: x = abs(x)
    if table['domain'] == 'RADIAL' and x < 0:
        fail('RESTORING_CURVE_INVALID', 'preview.coordinate', 'radial coordinate is nonnegative')
    nodes = [tuple(float(v) for v in node) for node in table['nodes']]
    areas = _areas(nodes)
    if x < nodes[0][0]:
        i=0; boundary=0; exterior=True
    elif x >= nodes[-1][0]:
        i=len(nodes)-2; boundary=len(nodes)-1; exterior=True
    else:
        i=next(i for i in range(len(nodes)-1) if nodes[i][0] <= x < nodes[i+1][0])
        boundary=i; exterior=False
    slope=(nodes[i+1][1]-nodes[i][1])/(nodes[i+1][0]-nodes[i][0])
    if exterior and table['tail']=='HOLD_EFFORT': slope=0.0
    dx=x-nodes[boundary][0]; f=nodes[boundary][1]
    energy=areas[boundary]+f*dx+slope*dx*dx/2
    gradient=sign*(f+slope*dx)
    if not all(math.isfinite(v) for v in (energy,gradient,slope)):
        fail('RESTORING_ARITHMETIC_FAULT', 'preview', 'nonfinite sample')
    return {'energy':energy,'gradient':gradient,'curvature':slope}
