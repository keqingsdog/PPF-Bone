"""Strict versioned preset I/O. No auto import, registration or RNA mutation."""
import copy
import json
import os
from pathlib import Path
import tempfile
from ..models import restoring_curves as curves

CHAIN_CHANNELS=('twist_x','swing_y','swing_z','anchor')
_CHAIN_KIND='ppf.restoring_chain.preset'
_VALIDATION_JOINT='00000000-0000-0000-0000-000000000001'


def validate_preset(preset):
    curves.exact(preset,{'kind','version','name','restoring_curves_v1'},'preset')
    if preset['kind']!='ppf.restoring_curves.preset':
        curves.fail('RESTORING_SCHEMA_INVALID','kind','not a restoring curve preset')
    if type(preset['version']) is not int or preset['version']!=1:
        curves.fail('RESTORING_VERSION_UNSUPPORTED','preset.version','supported version is 1')
    name=preset['name']
    if type(name) is not str or not name.strip() or name!=name.strip() or len(name)>128 or any(ord(c)<32 for c in name):
        curves.fail('RESTORING_SCHEMA_INVALID','preset.name','expected nonempty trimmed name, at most 128 characters')
    curves.validate_author_map(preset['restoring_curves_v1'])
    if len(curves._json_bytes(preset))>curves.MAX_JSON_BYTES:
        curves.fail('RESTORING_BUDGET_EXCEEDED','preset','JSON budget exceeded')
    return copy.deepcopy(preset)


def named_preset(name, document):
    return validate_preset({'kind':'ppf.restoring_curves.preset','version':1,
                            'name':name,'restoring_curves_v1':document})


def rename_preset(preset, name):
    changed=validate_preset(preset);changed['name']=name
    return validate_preset(changed)


def _object_pairs(pairs):
    result={}
    for key,value in pairs:
        if key in result:curves.fail('RESTORING_SCHEMA_INVALID',key,'duplicate JSON key')
        result[key]=value
    return result


def decode_preset(raw):
    if type(raw) not in (bytes,str):
        curves.fail('RESTORING_SCHEMA_INVALID','preset','expected UTF-8 bytes or text')
    if isinstance(raw,str):raw=raw.encode('utf-8')
    if len(raw)>curves.MAX_JSON_BYTES:
        curves.fail('RESTORING_BUDGET_EXCEEDED','preset','JSON budget exceeded')
    try:
        value=json.loads(raw.decode('utf-8'),object_pairs_hook=_object_pairs,
                         parse_constant=lambda value:curves.fail('RESTORING_SCHEMA_INVALID',value,'nonfinite JSON constant'))
    except (UnicodeError,json.JSONDecodeError,RecursionError):
        curves.fail('RESTORING_SCHEMA_INVALID','preset','invalid UTF-8 JSON')
    return validate_preset(value)


def load_preset(path):
    """Loading/selecting returns a detached candidate; it never applies it."""
    with Path(path).open('rb') as handle:
        return decode_preset(handle.read(curves.MAX_JSON_BYTES+1))


def save_preset(path, preset):
    """Validate then atomically replace one file; failures preserve old bytes."""
    checked=validate_preset(preset)
    raw=curves._json_bytes(checked)+b'\n'
    if len(raw)>curves.MAX_JSON_BYTES:
        curves.fail('RESTORING_BUDGET_EXCEEDED','preset','JSON budget exceeded')
    destination=Path(path)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='wb',dir=destination.parent,
                                         prefix='.restoring-preset-',suffix='.tmp',delete=False) as handle:
            temporary=Path(handle.name);handle.write(raw);handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,destination)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()
    return checked


def validate_chain_preset(preset):
    """Validate a UUID-independent bundle of up to four authored channels."""
    curves.exact(preset,{'kind','version','name','source_joint_uuid','channels'},'chain_preset')
    if preset['kind']!=_CHAIN_KIND:
        curves.fail('RESTORING_SCHEMA_INVALID','kind','not a restoring chain preset')
    if type(preset['version']) is not int or preset['version']!=1:
        curves.fail('RESTORING_VERSION_UNSUPPORTED','chain_preset.version','supported version is 1')
    name=preset['name']
    if type(name) is not str or not name.strip() or name!=name.strip() or len(name)>128 or any(ord(c)<32 for c in name):
        curves.fail('RESTORING_SCHEMA_INVALID','chain_preset.name','expected nonempty trimmed name, at most 128 characters')
    curves.canonical_uuid(preset['source_joint_uuid'],'source_joint_uuid')
    channels=preset['channels']
    curves.exact(channels,set(CHAIN_CHANNELS),'chain_preset.channels')
    configured=0
    for channel in CHAIN_CHANNELS:
        record=channels[channel]
        if record is None: continue
        configured+=1
        curves.exact(record,{'binding','table'},f'channels.{channel}')
        table=curves.validate_table(record['table'])
        binding=record['binding']
        mode=binding.get('mode') if type(binding) is dict else None
        extra={'damping'} if mode=='ABSOLUTE_SI' else {
            'frequency_hz','damping_ratio','coordinate_scale','mass_policy'} if mode=='AUTO_FREQUENCY' else None
        if extra is None: curves.fail('RESTORING_SCHEMA_INVALID',f'channels.{channel}.binding.mode','unsupported mode')
        curves.exact(binding,{'enabled','mode'}|extra,f'channels.{channel}.binding')
        target={'kind':'JOINT','joint_uuid':_VALIDATION_JOINT,'channel':channel}
        curves.validate_author_map({'version':1,'tables':[table],
            'bindings':[dict(binding,target=target,table_id=table['table_id'])],
            'stitch_pairs':[]})
    if not configured:
        curves.fail('RESTORING_SCHEMA_INVALID','chain_preset.channels','at least one channel must be configured')
    checked=copy.deepcopy(preset)
    if len(curves._json_bytes(checked))>curves.MAX_JSON_BYTES:
        curves.fail('RESTORING_BUDGET_EXCEEDED','chain_preset','JSON budget exceeded')
    return checked


def named_chain_preset(name, source_joint_uuid, channels):
    return validate_chain_preset({'kind':_CHAIN_KIND,'version':1,'name':name,
                                   'source_joint_uuid':source_joint_uuid,'channels':channels})


def decode_chain_preset(raw):
    if type(raw) not in (bytes,str):
        curves.fail('RESTORING_SCHEMA_INVALID','chain_preset','expected UTF-8 bytes or text')
    if isinstance(raw,str): raw=raw.encode('utf-8')
    if len(raw)>curves.MAX_JSON_BYTES:
        curves.fail('RESTORING_BUDGET_EXCEEDED','chain_preset','JSON budget exceeded')
    try:
        value=json.loads(raw.decode('utf-8'),object_pairs_hook=_object_pairs,
                         parse_constant=lambda value:curves.fail('RESTORING_SCHEMA_INVALID',value,'nonfinite JSON constant'))
    except (UnicodeError,json.JSONDecodeError,RecursionError):
        curves.fail('RESTORING_SCHEMA_INVALID','chain_preset','invalid UTF-8 JSON')
    return validate_chain_preset(value)


def load_chain_preset(path):
    """Load and validate a chain preset without changing author state."""
    with Path(path).open('rb') as handle:
        return decode_chain_preset(handle.read(curves.MAX_JSON_BYTES+1))


def save_chain_preset(path, preset):
    """Atomically save a validated chain preset without touching prior bytes on error."""
    checked=validate_chain_preset(preset)
    raw=curves._json_bytes(checked)+b'\n'
    if len(raw)>curves.MAX_JSON_BYTES:
        curves.fail('RESTORING_BUDGET_EXCEEDED','chain_preset','JSON budget exceeded')
    destination=Path(path);temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='wb',dir=destination.parent,
                prefix='.restoring-chain-preset-',suffix='.tmp',delete=False) as handle:
            temporary=Path(handle.name);handle.write(raw);handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,destination)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()
    return checked
