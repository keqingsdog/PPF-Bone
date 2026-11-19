"""Discoverable source materials for a failed/cancelled Full Bake, not Restore.

New Text IDs are retained with Fake User; source flags/bindings are untouched.
The event is only in the current Blender Main until the user saves the blend.
"""
import datetime
import hashlib
import json
import uuid
import bpy

KEY = 'ppf_abort_materials_v1'
PREFIX = 'PPF Bake Materials '
SCHEMA = 1


class RetentionError(RuntimeError):
    pass


def _label(value):
    return {'type': value.bl_rna.identifier, 'name': value.name_full,
            'library': value.library.filepath if value.library else ''}


def _collect(job):
    refs = []
    missing = []

    def add(value, role, required=False):
        if value is None:
            if required:missing.append(role)
            return
        try:
            if not isinstance(value, bpy.types.ID) or not value.as_pointer():
                raise ReferenceError('missing ID')
            label = _label(value)
        except ReferenceError:
            missing.append(role)
            return  # An explicitly deleted source is not recreated.
        for row in refs:
            if row['id'] == value:
                row['roles'].append(role)
                return
        refs.append({'id': value, 'roles': [role], 'label': label,
                     'fake_at_archive': bool(value.use_fake_user)})

    tx = job.get('recovery') or {}
    for index, row in enumerate(tx.get('items', [])):
        add(row.get('action'), 'item/%d/source_action' % index)
        add(row.get('backup'), 'item/%d/backup_data' % index)
        add(row.get('key_source_binding', {}).get('source'), 'item/%d/key_source_action' % index)
    for index, proof in enumerate(tx.get('source_action_proofs', [])):
        add(proof.get('action'), 'proof/%d/source_action' % index, required=True)
    for proof_index, graph in enumerate(tx.get('source_graph_proofs', [])):
        for owner_index, owner in enumerate(graph.get('owners', [])):
            for ref_index, reference in enumerate(owner.get('refs', [])):
                value=reference.get('id')
                if isinstance(value,bpy.types.Action):
                    add(value,'graph/%d/%d/%d/action' % (proof_index,owner_index,ref_index),required=True)
    for index, entry in enumerate(job.get('objects', [])):
        for kind in ('mesh_action_transaction', 'curve_transaction'):
            add((entry.get(kind) or {}).get('action'), 'writer/%d/%s/source_action' % (index, kind))
        for arm_index, snap in enumerate(entry.get('snapshots', [])):
            add(snap.get('action'), 'bone/%d/%d/source_action' % (index, arm_index))
            for proof_index, proof in enumerate(snap.get('source_data_proofs', [])):
                add(proof.get('action'), 'bone/%d/%d/data/%d' % (index, arm_index, proof_index))
    return refs, missing


def _metadata(job, event, refs, missing):
    scene = job.get('scene')
    scene_label = _label(scene) if scene is not None else None
    return {
        'schema': SCHEMA, 'event': event,
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'kind': 'FAILED_OR_CANCELLED_FULL_BAKE_MATERIALS_NOT_RESTORE',
        'scene': scene_label,  # Metadata only; do not retain an entire Scene.
        'reason': str(job.get('finalize_error') or ('abort requested' if job.get('aborted') else job.get('status') or 'Abort')),
        'targets': [{'name': str(e.get('obj_name', '')), 'uuid': str(e.get('obj_uuid', '')),
                     'type': str(e.get('obj_type', '')), 'phase': str(e.get('_finalize_state', 'pending')),
                     'sampled_frames': int(e.get('frames_done', 0))} for e in job.get('objects', [])],
        'refs': [{'roles': r['roles'], 'label': r['label'], 'fake_at_archive': r['fake_at_archive']} for r in refs],
        'deleted_sources_not_recreated': missing,
    }


def _body(metadata):
    return ('FAILED/CANCELLED BAKE SOURCE MATERIALS — NOT A RESTORE RECORD\n'
            'No automatic restoration is provided. Original bindings/flags were not changed by this archive.\n'
            'This Text holds real source/backup IDs and adds users/storage. Normal Cancel also retains materials.\n'
            'Save the current .blend for persistence; this is not a crash-safe external backup.\n'
            'Deleting this Text releases its references; explicitly deleted sources cannot be revived.\n\n'
            + json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2))


def _verify(text, event, required, *, sealed):
    if text is None or not text.as_pointer() or bpy.data.texts.get(text.name) != text:
        raise RetentionError('Bake material archive Text was removed')
    record = text.get(KEY)
    if record is None or type(record.get('schema')) is not int or record['schema'] != SCHEMA or record.get('event') != event:
        raise RetentionError('Bake material archive identity/schema is invalid')
    if record.get('phase') != ('RETAINED_IN_MEMORY' if sealed else 'WRITING') or text.use_fake_user is not True:
        raise RetentionError('Bake material archive is incomplete or not retained')
    metadata = record.get('metadata')
    if not isinstance(metadata, str):raise RetentionError('Bake material archive metadata is missing')
    parsed = json.loads(metadata)
    if parsed.get('event') != event or type(parsed.get('schema')) is not int or parsed['schema'] != SCHEMA:
        raise RetentionError('Bake material archive metadata differs')
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
    if any(not any(row.get('id') == expected['id'] for row in rows) for expected in required):
        raise RetentionError('Bake material archive does not hold every currently required source/backup')
    return text


def retain(job):
    """Verify retention before any caller pops writer receipts/releases backups."""
    refs, missing = _collect(job)
    previous = job.get('material_archive')
    if previous is not None:
        try:return _verify(previous.get('text'), previous.get('event'), refs, sealed=True)
        except Exception as exc:
            raise RetentionError('Bake material retention is not confirmed; cleanup stopped: ' + str(exc)) from exc
    event = str(uuid.uuid4())
    text = bpy.data.texts.new(PREFIX + event)
    job['material_archive'] = {'text': text, 'event': event}
    try:
        text.use_fake_user = True
        metadata = _metadata(job, event, refs, missing)
        body = _body(metadata)
        text[KEY] = {'schema': SCHEMA, 'event': event, 'phase': 'WRITING',
                     'metadata': json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                     'body_sha256': hashlib.sha256(body.encode()).hexdigest(),
                     'refs': [{'id': row['id'], 'roles': json.dumps(row['roles'], ensure_ascii=False)} for row in refs]}
        text.write(body)
        _verify(text, event, refs, sealed=False)
        text[KEY]['phase'] = 'RETAINED_IN_MEMORY'
        return _verify(text, event, refs, sealed=True)
    except Exception as exc:
        try:text[KEY]['phase'] = 'INCOMPLETE'
        except Exception:pass
        raise RetentionError('Bake material retention is not confirmed; cleanup stopped. Inspect Text ' + text.name + ': ' + str(exc)) from exc


def events():
    """Programmatic discovery; entries are materials, never Restore authority."""
    return [text for text in bpy.data.texts if KEY in text]
