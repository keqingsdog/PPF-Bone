"""Dormant author transactions; RNA/Undo operators are connected in a later gate."""
import copy
import hashlib
from ..models import restoring_curves as curves
from ..models.restoring_curves import COMMON_ANGULAR_FREQUENCY_HZ, COMMON_DAMPING_RATIO

COMMON_FREQUENCY_HZ = 10.0


def _commit(state, mode, document):
    if type(mode) is not str or mode not in curves.MODES:
        curves.fail('RESTORING_SCHEMA_INVALID', 'mode', 'unknown author mode')
    checked = curves.validate_author_map(document) if document is not None else None
    if mode != 'LEGACY' and checked is None:
        curves.fail('RESTORING_SCHEMA_INVALID', 'document', 'nonlegacy mode requires a document')
    if mode == 'COMMON' and any(b['mode'] != 'AUTO_FREQUENCY' for b in checked['bindings']):
        curves.fail('RESTORING_CONFLICT', 'mode', 'COMMON requires explicit frequency policy')
    # One commit after all validation; return an independent previous snapshot
    # for the future RNA Undo transaction, never modify it in-place.
    previous = copy.deepcopy(state)
    state.clear(); state.update({'mode': mode, 'document': checked})
    return previous


def set_mode(state, mode):
    """Explicit mode change. Does not silently convert SI to frequency units."""
    if mode == 'COMMON' and state.get('document') is not None:
        if any(b['mode'] != 'AUTO_FREQUENCY' for b in curves.validate_author_map(state['document'])['bindings']):
            curves.fail('RESTORING_CONFLICT', 'mode', 'COMMON requires explicit frequency settings, not SI conversion')
    return _commit(state, mode, state.get('document'))


def replace_document(state, document, mode='ADVANCED'):
    """Editor transaction: caller provides the complete edited author map."""
    return _commit(state, mode, document)


def common_document(targets, stitch_pairs=(), *, frequency_hz=None, damping_ratio=None, coordinate_scale=None, enable_anchor=False):
    """Linear frequency authoring with separate angular/positional defaults.

    All five channels have distinct tables so subsequent axis edits are local.
    Effective mass and actual K/C belong exclusively to fresh Build resolution.
    Positional anchors retain their authored N/m connection unless the author
    explicitly enables an anchor curve; angular defaults never soften it.
    """
    tables=[]; bindings=[]
    for target in targets:
        key=curves.target_key(target)
        angular=key[0]=='JOINT' and key[2]!='anchor'
        channel=key[2] if key[0]=='JOINT' else 'stitch'
        identifier=f'{channel}_{key[1].replace("-", "")}'
        domain='ODD' if angular else 'RADIAL' if channel=='anchor' else 'SIGNED'
        tables.append({'table_id':identifier,'domain':domain,'tail':'LINEAR',
                       'nodes':[[0,0],[1,1]] if domain!='SIGNED' else [[-1,-1],[0,0],[1,1]]})
        bindings.append({'target':copy.deepcopy(target),'enabled':bool(enable_anchor) if channel=='anchor' else True,'table_id':identifier,
                         'mode':'AUTO_FREQUENCY','frequency_hz':frequency_hz if frequency_hz is not None else (COMMON_ANGULAR_FREQUENCY_HZ if angular else COMMON_FREQUENCY_HZ),
                         'damping_ratio':damping_ratio if damping_ratio is not None else COMMON_DAMPING_RATIO,
                         'coordinate_scale':coordinate_scale if coordinate_scale is not None else 1 if angular else .001,
                         'mass_policy':'LOCAL_REST_V1'})
    return curves.validate_author_map({'version':1,'tables':tables,'bindings':bindings,
                                      'stitch_pairs':copy.deepcopy(list(stitch_pairs))})


def _merge_author_pair_mappings(current, incoming):
    """Merge validated author mappings without reindexing retired library data.

    These ordinals identify the author document, not the current live transport
    order. Existing identities retain their positions; export resolves live IDs
    separately. Reserve nonconflicting incoming positions before filling holes.
    """
    merged = copy.deepcopy(list(current))
    existing = {item['pair_id'] for item in merged}
    used = {item['cross_stitch_index'] for item in merged}
    pending = []
    for item in incoming:
        if item['pair_id'] in existing:
            continue
        added = copy.deepcopy(item)
        merged.append(added)
        existing.add(added['pair_id'])
        if added['cross_stitch_index'] in used:
            pending.append(added)
        else:
            used.add(added['cross_stitch_index'])
    free = 0
    for item in pending:
        while free in used:
            free += 1
        if free > 0xffffffff:
            curves.fail('RESTORING_CONFLICT', 'stitch_pairs', 'no free author ordinal')
        item['cross_stitch_index'] = free
        used.add(free)
    return merged


def add_new_common(state, targets, ordered_pair_ids=(), *, has_draft=False):
    """Creation-only candidate. Dormant legacy data is never activated."""
    candidate=copy.deepcopy(state)
    if state['mode']=='LEGACY' and (state.get('document') is not None or has_draft):
        return candidate
    document=curves.validate_author_map(state['document']) if state.get('document') is not None else None
    existing={curves.target_key(b['target']) for b in document['bindings']} if document else set()
    missing=[target for target in targets if curves.target_key(target) not in existing]
    if not missing:return candidate
    required=[t['pair_id'] for t in missing if t['kind']=='STITCH_PAIR']
    new=common_document(missing,curves.stitch_mapping(ordered_pair_ids,required))
    if document is None:document=new
    else:
        document['tables'].extend(new['tables']);document['bindings'].extend(new['bindings'])
        document['stitch_pairs']=_merge_author_pair_mappings(document['stitch_pairs'],new['stitch_pairs'])
    _commit(candidate,'COMMON' if state['mode']=='LEGACY' else state['mode'],document)
    return candidate


def restore_common_defaults(state, targets, stitch_pairs=()):
    """Explicit reset replaces the document; it never migrates on registration."""
    return _commit(state, 'COMMON', common_document(targets, stitch_pairs))


def apply_preset(state, preset, remap, selected_targets, stitch_pairs=()):
    """Apply only to explicitly supplied current targets, atomically.

    remap is a list of exact {source:<target>,destination:<target>} maps.
    selected_targets is the current live selection supplied by the RNA adapter.
    Preset ordinals are discarded. Current stitch_pairs validate live targets;
    existing author ordinals remain stable when merging the validated document.
    Existing unrelated target settings remain intact. Selection alone changes
    no author data; the later UI invokes this only from its explicit Apply op.
    """
    from .restoring_curve_presets import validate_preset
    source=validate_preset(preset)['restoring_curves_v1']
    selected={curves.target_key(t) for t in selected_targets}
    if len(selected)!=len(selected_targets):
        curves.fail('RESTORING_CONFLICT','selection','duplicate selected target')
    mapping={};destinations=set()
    for entry in remap:
        curves.exact(entry, {'source','destination'}, 'remap')
        src=curves.target_key(entry['source']);dst=curves.target_key(entry['destination'])
        if src in mapping or dst in destinations or dst not in selected:
            curves.fail('RESTORING_TARGET_INVALID','remap','duplicate or unselected target')
        if src[0]!=dst[0] or (src[0]=='JOINT' and src[2]!=dst[2]):
            curves.fail('RESTORING_TARGET_INVALID','remap','preserve target kind and channel')
        mapping[src]=copy.deepcopy(entry['destination']);destinations.add(dst)
    required={curves.target_key(b['target']) for b in source['bindings']}
    if set(mapping)!=required or destinations!=selected:
        curves.fail('RESTORING_TARGET_INVALID','remap','explicit exact source and current-selection mapping required')
    current=state.get('document')
    current=curves.validate_author_map(current) if current is not None else {'version':1,'tables':[],'bindings':[],'stitch_pairs':[]}
    added=copy.deepcopy(source)
    for binding in added['bindings']:
        binding['target']=mapping[curves.target_key(binding['target'])]
    # Discard stale preset ordinals unconditionally, even on the same model.
    added['stitch_pairs']=copy.deepcopy(list(stitch_pairs))
    curves.validate_author_map(added)
    existing_tables={table['table_id']:table for table in current['tables']}
    id_remap={}
    for table in added['tables']:
        old_id=table['table_id'];new_id=old_id
        if new_id in existing_tables and table!=existing_tables[new_id]:
            digest=hashlib.sha256(curves._json_bytes(table)).hexdigest()[:16]
            new_id=old_id[:47]+'_'+digest
            changed=copy.deepcopy(table);changed['table_id']=new_id
            if new_id in existing_tables and changed!=existing_tables[new_id]:
                curves.fail('RESTORING_CONFLICT',new_id,'table identity collision')
        id_remap[old_id]=new_id;table['table_id']=new_id;existing_tables[new_id]=table
    for binding in added['bindings']:binding['table_id']=id_remap[binding['table_id']]
    retained=[b for b in current['bindings'] if curves.target_key(b['target']) not in destinations]
    final_bindings=retained+added['bindings']
    final={'version':1,'tables':list(existing_tables.values()),'bindings':final_bindings,
           'stitch_pairs':_merge_author_pair_mappings(current['stitch_pairs'],added['stitch_pairs'])}
    # Applying a preset preserves its declared SI/frequency policy. COMMON is
    # valid only if every resulting binding uses AUTO_FREQUENCY.
    mode='COMMON' if all(b['mode']=='AUTO_FREQUENCY' for b in final_bindings) else 'ADVANCED'
    return _commit(state,mode,final)
