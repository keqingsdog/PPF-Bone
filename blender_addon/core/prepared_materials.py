"""Independent materials for synchronous prepared-start failure, not Restore.

New Text IDs are retained with Fake User; source flags/bindings are untouched.
The event is only in the current Blender Main until the user saves the blend.
"""
import datetime
import hashlib
import json
import uuid
import bpy

KEY = 'ppf_prepared_materials_v1'
PREFIX = 'PPF Prepared Materials '
SCHEMA = 1


class RetentionError(RuntimeError):
    pass


def _label(value):
    return {'type': value.bl_rna.identifier, 'name': value.name_full,
            'library': value.library.filepath if value.library else ''}


def _collect(job):
    refs, missing = [], []
    def add(value, role, required=False):
        if value is None:
            if required: missing.append(role)
            return
        try:
            if not isinstance(value, bpy.types.ID):
                raise RetentionError('Non-ID material at ' + role)
            if not value.as_pointer(): raise ReferenceError('deleted ID')
            label = _label(value)
        except ReferenceError:
            missing.append(role)
            return
        for row in refs:
            if row['id'] == value:
                if role not in row['roles']: row['roles'].append(role)
                return
        refs.append({'id': value, 'roles': [role], 'label': label,
                     'fake_at_archive': bool(value.use_fake_user)})
    def owner(value, role):
        add(value, role, required=True)
        if value is None: return
        try:
            ad = getattr(value, 'animation_data', None)
            if ad is not None:
                add(ad.action, role + '/current_action')
                for fi, curve in enumerate(ad.drivers):
                    for vi, variable in enumerate(curve.driver.variables):
                        for ti, target in enumerate(variable.targets):
                            add(target.id, '%s/driver/%d/%d/%d' % (role, fi, vi, ti))
            key = getattr(value, 'shape_keys', None)
            if key is not None: owner(key, role + '/key')
        except ReferenceError:
            missing.append(role + '/deleted_during_read')
    tx = job.get('recovery') or {}
    for i, row in enumerate(tx.get('items', [])):
        owner(row.get('data'), 'item/%d/original_data' % i)
        owner(row.get('backup'), 'item/%d/backup_data' % i)
        add(row.get('action'), 'item/%d/source_action' % i)
        for key in ('key_source_binding', 'curve_source_binding'):
            for field in ('source', 'working', 'key', 'backup_key', 'data', 'backup'):
                add((row.get(key) or {}).get(field), 'item/%d/%s/%s' % (i,key,field))
    for i, proof in enumerate(tx.get('source_action_proofs', [])):
        add(proof.get('action'), 'proof/%d/source_action' % i)
        for j, ref in enumerate(proof.get('source_refs', [])):
            add(ref.get('id'), 'proof/%d/ref/%d' % (i,j), required=True)
    for i, graph in enumerate(tx.get('source_graph_proofs', [])):
        for j, item in enumerate(graph.get('owners', [])):
            for k, ref in enumerate(item.get('refs', [])):
                add(ref.get('id'), 'graph/%d/%d/ref/%d' % (i,j,k), required=True)
    for i, entry in enumerate(job.get('objects', [])):
        for kind in ('mesh_action_transaction', 'curve_transaction', 'mesh_created_key_transaction'):
            receipt = entry.get(kind) or {}
            for field in ('data', 'key'):
                if field in receipt: owner(receipt[field], 'writer/%d/%s/%s' % (i,kind,field))
            for field in ('action', 'working'):
                add(receipt.get(field), 'writer/%d/%s/%s' % (i,kind,field))
        for j, snap in enumerate(entry.get('snapshots', [])):
            add(snap.get('action'), 'bone/%d/%d/source_action' % (i,j))
            for k, proof in enumerate(snap.get('source_data_proofs', [])):
                add(proof.get('action'), 'bone/%d/%d/data/%d' % (i,j,k))
    for i, (obj, entry) in enumerate(job.get('prepared', [])):
        try:
            owner(obj, 'prepared/%d/current_object' % i)
            owner(obj.data, 'prepared/%d/current_data' % i)
        except ReferenceError:
            missing.append('prepared/%d/deleted_object' % i)
    return refs, missing


def _metadata(job, event, refs, missing):
    scene = job.get('scene')
    scene_label = _label(scene) if scene is not None else None
    return {
        'schema': SCHEMA, 'event': event,
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'kind': 'FAILED_PREPARED_MATERIALS_NOT_RESTORE',
        'scene': scene_label,  # Metadata only; do not retain an entire Scene.
        'source_holder_error': str(job.get('source_holder_error') or ''),
        'phase': 'FAILED_PREPARED_MATERIALS',
        'reason': str(job.get('finalize_error') or ('abort requested' if job.get('aborted') else job.get('status') or 'Abort')),
        'targets': [{'name': str(e.get('obj_name', '')), 'uuid': str(e.get('obj_uuid', '')),
                     'type': str(e.get('obj_type', '')), 'phase': str(e.get('_finalize_state', 'pending')),
                     'sampled_frames': int(e.get('frames_done', 0))} for e in job.get('objects', [])],
        'refs': [{'roles': r['roles'], 'label': r['label'], 'fake_at_archive': r['fake_at_archive']} for r in refs],
        'deleted_sources_not_recreated': missing,
    }


def _body(metadata):
    return ('FAILED PREPARED BAKE MATERIALS — NOT A RESTORE RECORD\n'
            'No automatic restoration is provided. Original bindings/flags were not changed by this archive.\n'
            'This Text holds real source/working/current/backup IDs and adds users/storage. Partial preparation is intentionally not rolled back.\n'
            'Save the current .blend for persistence; this is not a crash-safe external backup.\n'
            'Deleting this Text releases its references; explicitly deleted sources cannot be revived.\n\n'
            + json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2))


def _verify(text, event, required, *, sealed, expected_metadata):
    if text is None or not text.as_pointer() or bpy.data.texts.get(text.name) != text:
        raise RetentionError('Bake material archive Text was removed')
    record = text.get(KEY)
    if record is None or type(record.get('schema')) is not int or record['schema'] != SCHEMA or record.get('event') != event:
        raise RetentionError('Bake material archive identity/schema is invalid')
    if record.get('phase') != ('PARTIAL_RETAINED' if sealed else 'WRITING') or text.use_fake_user is not True:
        raise RetentionError('Bake material archive is incomplete or not retained')
    metadata = record.get('metadata')
    if not isinstance(metadata, str):raise RetentionError('Bake material archive metadata is missing')
    parsed = json.loads(metadata)
    if parsed.get('event') != event or type(parsed.get('schema')) is not int or parsed['schema'] != SCHEMA:
        raise RetentionError('Bake material archive metadata differs')
    if parsed != expected_metadata or parsed.get('kind') != 'FAILED_PREPARED_MATERIALS_NOT_RESTORE':
        raise RetentionError('Prepared material purpose/metadata differs')
    expected_body = _body(parsed)
    if text.as_string() != expected_body or record.get('body_sha256') != hashlib.sha256(expected_body.encode()).hexdigest():
        raise RetentionError('Bake material archive body differs')
    rows = record.get('refs', [])
    if len(rows) != len(parsed.get('refs', [])):
        raise RetentionError('Bake material archive reference count differs')
    for row, descriptor in zip(rows, parsed['refs']):
        value = row.get('id')
        if value is None or not isinstance(value, bpy.types.ID) or not value.as_pointer():
            raise RetentionError('Bake material archive reference was removed')
        if row.get('roles') != json.dumps(descriptor['roles'], ensure_ascii=False):
            raise RetentionError('Bake material archive reference roles differ')
    if len(rows) != len(required) or any(row.get('id') != expected['id'] or row.get('roles') != json.dumps(expected['roles'], ensure_ascii=False) for row, expected in zip(rows, required)):
        raise RetentionError('Bake material archive does not hold every currently required source/backup')
    return text


def retain(job):
    """Verify retention before any caller pops writer receipts/releases backups."""
    refs, missing = _collect(job)
    event = str(uuid.uuid4())
    text = bpy.data.texts.new(PREFIX + event)
    job['prepared_archive'] = {'text': text, 'event': event}
    try:
        text.use_fake_user = True
        metadata = _metadata(job, event, refs, missing)
        body = _body(metadata)
        text[KEY] = {'schema': SCHEMA, 'event': event, 'phase': 'WRITING',
                     'metadata': json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                     'body_sha256': hashlib.sha256(body.encode()).hexdigest(),
                     'refs': [{'id': row['id'], 'roles': json.dumps(row['roles'], ensure_ascii=False)} for row in refs]}
        text.write(body)
        _verify(text, event, refs, sealed=False, expected_metadata=metadata)
        text[KEY]['phase'] = 'PARTIAL_RETAINED'
        return _verify(text, event, refs, sealed=True, expected_metadata=metadata)
    except Exception as exc:
        try:text[KEY]['phase'] = 'INCOMPLETE'
        except Exception:pass
        raise RetentionError('Bake material retention is not confirmed; cleanup stopped. Inspect Text ' + text.name + ': ' + str(exc)) from exc


def events():
    """Programmatic discovery; entries are materials, never Restore authority."""
    return [text for text in bpy.data.texts if KEY in text]
