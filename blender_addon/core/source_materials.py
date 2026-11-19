"""Live original Action holders for Full Bake, not snapshots or Restore authority."""
import hashlib
import json
import uuid
import bpy

KEY = 'ppf_source_materials_v1'
PREFIX = 'PPF Source Materials '
SCHEMA = 1

class RetentionError(RuntimeError):
    pass

def _sources(job):
    """Only captured originals; never writer working/generated IDs or backup data."""
    rows = []
    def add(value, role):
        if value is None:
            return
        if not isinstance(value, bpy.types.Action) or not value.as_pointer():
            raise RetentionError('Original source Action is unavailable')
        for row in rows:
            if row['id'] == value:
                row['roles'].append(role)
                return
        rows.append({'id': value, 'roles': [role]})
    tx = job.get('recovery') or {}
    for i, proof in enumerate(tx.get('source_action_proofs', [])):
        add(proof.get('action'), 'source_proof/%d' % i)
    for i, graph in enumerate(tx.get('source_graph_proofs', [])):
        for j, owner in enumerate(graph.get('owners', [])):
            for k, ref in enumerate(owner.get('refs', [])):
                value = ref.get('id')
                if isinstance(value, bpy.types.Action):
                    add(value, 'graph/%d/%d/%d' % (i, j, k))
    for i, entry in enumerate(job.get('objects', [])):
        for j, snap in enumerate(entry.get('snapshots', [])):
            add(snap.get('action'), 'bone/%d/%d' % (i, j))
            for k, proof in enumerate(snap.get('source_data_proofs', [])):
                add(proof.get('action'), 'bone_data/%d/%d/%d' % (i, j, k))
    return rows

def _body(metadata):
    return ('FULL BAKE ORIGINAL ACTION MATERIALS — NOT A RESTORE RECORD\n'
            'Live original Action references; not immutable content or all dependencies.\n'
            'Retained on success, Cancel, failure, Keep and Restore; adds users/storage.\n'
            'Save this blend for persistence. Unsaved Load, deletion and crashes are not protected.\n'
            'No backup Mesh/Curve or generated working Action is held by this event.\n\n'
            + json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2))

def verify(job):
    memo = job.get('source_archive')
    if not isinstance(memo, dict):
        raise RetentionError('Original source material event is missing; cleanup stopped')
    try:
        text = memo['text']
        if text is None or not text.as_pointer() or bpy.data.texts.get(text.name) != text:
            raise RetentionError('Original source material Text was removed')
        record = text.get(KEY)
        if (record is None or type(record.get('schema')) is not int
                or record['schema'] != SCHEMA or record.get('event') != memo['event']
                or record.get('phase') != 'RETAINED_IN_MEMORY' or text.use_fake_user is not True):
            raise RetentionError('Original source material event is incomplete or changed')
        if record.get('metadata') != memo['metadata']:
            raise RetentionError('Original source material metadata changed')
        body = _body(json.loads(memo['metadata']))
        if (text.as_string() != body or record.get('body_sha256') != memo['body_sha256']
                or hashlib.sha256(body.encode()).hexdigest() != memo['body_sha256']):
            raise RetentionError('Original source material body changed')
        actual = record.get('refs', [])
        expected = memo['refs']
        if len(actual) != len(expected):
            raise RetentionError('Original source material reference count changed')
        for row, plan in zip(actual, expected):
            value = row.get('id')
            if (value is None or not isinstance(value, bpy.types.Action) or not value.as_pointer()
                    or value != plan['id'] or row.get('roles') != plan['roles']):
                raise RetentionError('Original source material reference changed or was deleted')
        for required in _sources(job):
            if not any(row['id'] == required['id'] for row in expected):
                raise RetentionError('A source is not held by the original event; no re-signing')
        return text
    except RetentionError:
        raise
    except Exception as exc:
        raise RetentionError('Original source material verification failed; cleanup stopped: ' + str(exc)) from exc

def retain(job):
    if job.get('source_archive') is not None:
        return verify(job)
    refs = _sources(job)
    event = str(uuid.uuid4())
    text = bpy.data.texts.new(PREFIX + event)
    job['source_archive'] = {'text': text, 'event': event}
    try:
        text.use_fake_user = True
        scene = job.get('scene')
        metadata = {'schema': SCHEMA, 'event': event, 'purpose': 'FULL_BAKE_ORIGINAL_ACTION_MATERIALS_NOT_RESTORE',
                    'scene_name': scene.name_full if scene is not None else '',
                    'sources': [{'name': row['id'].name_full, 'roles': row['roles'],
                                 'fake_at_capture': bool(row['id'].use_fake_user)} for row in refs]}
        metadata_raw = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        body = _body(metadata)
        plans = [{'id': row['id'], 'roles': json.dumps(row['roles'], ensure_ascii=False)} for row in refs]
        digest = hashlib.sha256(body.encode()).hexdigest()
        job['source_archive'].update(metadata=metadata_raw, body_sha256=digest, refs=plans)
        text[KEY] = {'schema': SCHEMA, 'event': event, 'phase': 'WRITING',
                     'metadata': metadata_raw, 'body_sha256': digest, 'refs': plans}
        text.write(body)
        text[KEY]['phase'] = 'RETAINED_IN_MEMORY'
        return verify(job)
    except Exception as exc:
        try:
            text[KEY]['phase'] = 'INCOMPLETE'
        except Exception:
            pass
        raise RetentionError('Original source retention is not confirmed; no prepared writer started. Inspect Text '
                             + text.name + ': ' + str(exc)) from exc

def events():
    return [text for text in bpy.data.texts if KEY in text]
