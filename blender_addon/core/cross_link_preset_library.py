"""Portable spring settings, deliberately independent of network/rig identity."""
from copy import deepcopy
import json
import os
from pathlib import Path
import secrets
import tempfile

from . import bone_preset_library as storage

KIND = 'ppf.cross_link_settings_preset'
VERSION = 1
MAX_BYTES = 65536
PHYSICS_FIELDS = ('tension_enabled', 'compression_enabled',
    'tension_stiffness_N_per_m', 'compression_stiffness_N_per_m', 'damping_Ns_per_m')
COLLISION_FIELDS = ('exclude_collision',)
DISPLAY_FIELDS = ('preview_color_custom', 'preview_color')
GROUPS = {'physics': PHYSICS_FIELDS, 'collision': COLLISION_FIELDS, 'display': DISPLAY_FIELDS}


def preset_directory():
    import bpy
    path = bpy.utils.user_resource('SCRIPTS', path='presets/ppf_cross_link_settings', create=True)
    if not path:
        raise OSError('无法创建 Blender 用户预设目录')
    return Path(path)


def _root(directory=None, create=False):
    return storage._root(preset_directory() if directory is None else directory, create=create)


def _path(identifier, directory=None):
    return storage._path(identifier, _root(directory))


def validate_preset(document):
    from .cross_links import _number
    try:
        value = deepcopy(document)
        storage._finite(value)
        if not isinstance(value, dict) or set(value) != {'kind', 'version', 'id', 'name', *GROUPS}:
            raise ValueError('链间弹簧预设字段不完整或含未知字段')
        if value['kind'] != KIND or type(value['version']) is not int or value['version'] != VERSION:
            raise ValueError('不支持的链间弹簧预设版本')
        if value['id'] is not None and (not isinstance(value['id'], str) or not storage._ID.fullmatch(value['id'])):
            raise ValueError('无效的预设 ID')
        if storage.clean_name(value['name']) != value['name']:
            raise ValueError('预设名称未规范化')
        for group, fields in GROUPS.items():
            record = value[group]
            if not isinstance(record, dict) or set(record) != set(fields):
                raise ValueError('链间弹簧预设类别含未知字段')
            for field, item in record.items():
                if field == 'preview_color':
                    if not isinstance(item, list) or len(item) != 4 or any(type(c) not in (int, float) or not 0 <= c <= 1 for c in item):
                        raise ValueError('显示颜色必须是 0 到 1 的 RGBA')
                elif field.endswith('_enabled') or field in ('exclude_collision', 'preview_color_custom'):
                    if type(item) is not bool:
                        raise ValueError('预设开关必须是布尔值')
                else:
                    _number(item, field)
        if len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf8')) > MAX_BYTES:
            raise ValueError('预设文件过大')
        return value
    except (TypeError, KeyError, OverflowError, RecursionError) as error:
        raise ValueError('链间弹簧预设数据类型或深度无效') from error


def _decode(raw):
    if len(raw) > MAX_BYTES:
        raise ValueError('预设文件过大')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('预设包含重复 JSON 字段')
            result[key] = value
        return result
    def bad(value):
        raise ValueError('预设包含非有限 JSON 常量')
    try:
        return validate_preset(json.loads(raw.decode('utf8'), object_pairs_hook=pairs, parse_constant=bad))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError('预设不是有效 UTF8 JSON') from error


def load_preset(identifier, *, directory=None):
    with _path(identifier, directory).open('rb') as handle:
        document = _decode(handle.read(MAX_BYTES + 1))
    if document['id'] != identifier:
        raise ValueError('预设文件 ID 与内容不符')
    return document


def list_presets(*, directory=None):
    rows = []
    for path in _root(directory).glob('*.json'):
        if not storage._ID.fullmatch(path.stem):
            continue
        try:
            document = load_preset(path.stem, directory=directory)
            rows.append(dict(id=path.stem, name=document['name'], valid=True))
        except (ValueError, OSError) as error:
            rows.append(dict(id=path.stem, name=path.stem, valid=False, error=str(error)))
    return sorted(rows, key=lambda row: (row['name'].casefold(), row['id']))


def _write(path, document):
    raw = (json.dumps(validate_preset(document), ensure_ascii=False, allow_nan=False, sort_keys=True) + '\n').encode('utf8')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.ppf-spring-', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def save_preset(name, document, overwrite=False, *, preset_id=None, directory=None):
    name = storage.clean_name(name)
    checked = validate_preset(document)
    with storage._LOCK:
        root = _root(directory, create=True)
        if preset_id is None:
            if overwrite:
                raise ValueError('覆盖必须明确选择预设 ID')
            identifier = secrets.token_hex(16)
        else:
            if not overwrite:
                raise ValueError('覆盖已有预设必须明确确认')
            load_preset(preset_id, directory=root)
            identifier = preset_id
        if any(row['valid'] and row['id'] != identifier and row['name'] == name for row in list_presets(directory=root)):
            raise FileExistsError('同名预设已存在，请显式覆盖所选项')
        checked.update(id=identifier, name=name)
        path = _path(identifier, root)
        _write(path, checked)
        return path


def rename_preset(identifier, new_name, *, directory=None):
    with storage._LOCK:
        return save_preset(new_name, load_preset(identifier, directory=directory), True, preset_id=identifier, directory=directory)


def delete_preset(identifier, *, directory=None):
    with storage._LOCK:
        load_preset(identifier, directory=directory)
        path = _path(identifier, directory)
        path.unlink()
        return path


def capture_preset(context, arm, name, source='ACTIVE'):
    from . import cross_links as author
    author.require_ready(context, arm, writable=False)
    if source == 'ACTIVE':
        index = arm.ppf_cross_link_draft.active_index
        if not 0 <= index < len(arm.ppf_cross_links):
            raise ValueError('请选择一个当前连接作为保存来源')
        target = arm.ppf_cross_links[index]
        author.endpoint(context, arm, target.bone_a_uuid)
        author.endpoint(context, arm, target.bone_b_uuid)
        matches = [row for row in author._records(arm) if row['link_uuid'] == target.link_uuid]
        if not target.link_uuid or len(matches) != 1:
            raise ValueError('当前连接身份缺失或重复，请先修复连接')
        values = matches[0]
    elif source == 'DRAFT':
        target = arm.ppf_cross_link_draft
        if target.use_mass_defaults:
            raise ValueError('按质量初始化时灰色数值不是创建参数；请先切换为手动参数')
        values = {field: list(getattr(target, field)) if field == 'preview_color' else getattr(target, field)
                  for fields in GROUPS.values() for field in fields}
    else:
        raise ValueError('未知预设保存来源')
    return validate_preset(dict(kind=KIND, version=VERSION, id=None, name=storage.clean_name(name),
        **{group: {field: list(values[field]) if field == 'preview_color' else values[field] for field in fields}
           for group, fields in GROUPS.items()}))


def apply_preset(context, arm, document, target='SELECTED', mask=None):
    from . import cross_links as author
    checked = validate_preset(document)
    mask = {group: True for group in GROUPS} if mask is None else dict(mask)
    if set(mask) != set(GROUPS) or any(type(value) is not bool for value in mask.values()):
        raise ValueError('无效的应用类别')
    if not any(mask.values()):
        raise ValueError('请至少选择一个应用类别')
    parameters = {field: value for group in GROUPS if mask[group] for field, value in checked[group].items()}
    display_only = mask['display'] and not mask['physics'] and not mask['collision']
    if display_only:
        author.require_display_ready(context, arm)
    else:
        author.require_ready(context, arm)
    if target == 'SELECTED':
        identifiers = author.selected_link_ids(arm)
        for row in arm.ppf_cross_links:
            if row.link_uuid in identifiers:
                author.endpoint(context, arm, row.bone_a_uuid, validate_pose=not display_only)
                author.endpoint(context, arm, row.bone_b_uuid, validate_pose=not display_only)
        return author.edit(context, arm, identifiers, 'SET', **parameters)
    if target != 'DRAFT':
        raise ValueError('未知预设应用目标')
    draft = arm.ppf_cross_link_draft
    if mask['physics']:
        parameters['use_mass_defaults'] = False
    old = {field: list(getattr(draft, field)) if field == 'preview_color' else getattr(draft, field) for field in parameters}
    try:
        for field, value in parameters.items():
            setattr(draft, field, value)
    except Exception:
        for field, value in old.items():
            setattr(draft, field, value)
        raise
    return 1
