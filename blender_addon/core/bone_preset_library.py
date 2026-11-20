"""Portable, versioned bone-author preset library. No engine or UI IO in draw."""
from copy import deepcopy
import json,math,os,re,secrets,tempfile,threading,unicodedata
from pathlib import Path

KIND='ppf.bone_physics_preset'
VERSION=1
MAX_BYTES=8*1024*1024
MAX_VERTICES=20002
MAX_FACES=40000
CHANNELS=('twist_x','swing_y','swing_z','anchor')
BONE_FIELDS=('enabled','physical_model','shape','proxy_quality','proxy_face_mode','radius','length_ratio','box_width','box_depth','density','mass_mode','target_mass_kg','material_preset','solid_model','solid_young_modulus','solid_poisson_ratio','solid_deformation_damping','fem_scale_enabled','anchor_stiffness','angular_stiffness','angular_damping','soft_limits_enabled','soft_lower','soft_upper','soft_stiffness','soft_limit_mode','swing_cone_half_angle','swing_cone_stiffness','soft_limit_policy','soft_limit_frequency_hz','soft_limit_strength','collide_connected','collision_filter_enabled','collision_group_layers','collision_mask_layers','root_mode','root_motion_source','auto_follow_parent','joint_mode','hinge_twist_mode','hinge_lower','hinge_upper','use_auto_frames','joint_axis_euler')
CONE_FIELDS={'soft_limit_mode','swing_cone_half_angle','swing_cone_stiffness'}
RESTORING_DEFAULTS={'angular_restoring_mode':'AXES','swing_restoring_stiffness':.04,'swing_restoring_damping':.001}
BONE_FIELDS += tuple(RESTORING_DEFAULTS)
GROUP_FIELDS=('friction','shrink','enable_plasticity','plasticity','plasticity_threshold','use_group_bounding_box_diagonal','contact_gap_rat','contact_offset_rat','contact_gap','contact_offset')
EVENT_FIELDS=('frame','mode','pose_binding')
DRIVE_FIELDS=('enabled','gain','max_effort')
DISPLAY_FIELDS=('preview_color_custom','preview_color')
_LOCK=threading.RLock()
_ID=re.compile(r'^[0-9a-f]{32}$')

def clean_name(name):
    if not isinstance(name,str):raise ValueError('预设名称必须为文字')
    name=unicodedata.normalize('NFKC',name)
    name=''.join(c for c in name if ord(c)>=32 and ord(c)!=127).strip()
    if not name or len(name)>80:raise ValueError('预设名称需为1到80个字符')
    return name

def preset_directory():
    import bpy
    path=bpy.utils.user_resource('SCRIPTS',path='presets/ppf_bone_physics',create=True)
    if not path:raise OSError('无法创建Blender用户预设目录')
    return Path(path)

def _root(directory=None,create=False):
    path=Path(directory) if directory is not None else preset_directory()
    if path.is_symlink():raise ValueError('预设目录不可为符号链接')
    if create:path.mkdir(parents=True,exist_ok=True)
    if not path.is_dir():raise OSError('预设目录不存在或不可用')
    return path

def _path(identifier,directory=None):
    if not isinstance(identifier,str) or not _ID.fullmatch(identifier):raise ValueError('无效的预设ID')
    path=_root(directory)/(identifier+'.json')
    if path.is_symlink():raise ValueError('预设文件不可为符号链接')
    return path

def _finite(value):
    if isinstance(value,bool) or value is None or isinstance(value,str):return
    if isinstance(value,(float,int)):
        if not math.isfinite(value):raise ValueError('预设不接受非有限数值')
    elif isinstance(value,list):
        for item in value:_finite(item)
    elif isinstance(value,dict):
        for key,item in value.items():
            if not isinstance(key,str):raise ValueError('JSON字段必须为文字')
            _finite(item)
    else:raise ValueError('预设包含非JSON数据')

def validate_preset(document):
    try:return _validate_preset(document)
    except (TypeError,KeyError,OverflowError,RecursionError) as error:raise ValueError('骨骼预设数据类型或深度无效') from error

def _validate_preset(document):
    value=deepcopy(document);_finite(value)
    required={'kind','version','id','name','source_bone_length','settings','material','display','geometry','frames','curves'}
    if not isinstance(value,dict) or set(value)!=required or value.get('kind')!=KIND:raise ValueError('骨骼预设格式不支持')
    if type(value['version']) is not int or value['version']!=VERSION:raise ValueError('骨骼预设版本不支持')
    if value['id'] is not None and (not isinstance(value['id'],str) or not _ID.fullmatch(value['id'])):raise ValueError('预设ID无效')
    if clean_name(value['name'])!=value['name']:raise ValueError('预设名称未规范化')
    if isinstance(value['source_bone_length'],bool) or not isinstance(value['source_bone_length'],(int,float)) or value['source_bone_length']<=0:raise ValueError('原骨长无效')
    settings=value['settings']
    if not isinstance(settings,dict) or set(settings)!={'bone','events','drives','d6'}:raise ValueError('作者设置格式不支持')
    if isinstance(settings['bone'],dict) and RESTORING_DEFAULTS.keys().isdisjoint(settings['bone']):
        settings['bone'].update(RESTORING_DEFAULTS)  # Version-1 presets saved before circular restoring.
    if isinstance(settings['bone'],dict) and set(settings['bone']) in (set(BONE_FIELDS)-{'proxy_face_mode'},set(BONE_FIELDS)-CONE_FIELDS-{'proxy_face_mode'}):
        settings['bone']['proxy_face_mode']='TRI'  # Version-1 libraries authored before topology choice.
    if not isinstance(settings['bone'],dict) or set(settings['bone']) not in (set(BONE_FIELDS),set(BONE_FIELDS)-CONE_FIELDS):raise ValueError('作者字段不完整或包含工程引用')
    if not isinstance(settings['events'],list) or len(settings['events'])>4096:raise ValueError('根事件数量超限')
    for event in settings['events']:
        if not isinstance(event,dict) or set(event)!=set(EVENT_FIELDS):raise ValueError('根事件字段无效')
    if not isinstance(settings['drives'],dict) or set(settings['drives'])!=set(CHANNELS[:3]):raise ValueError('驱动轴字段无效')
    for axes in settings['drives'].values():
        if not isinstance(axes,dict) or set(axes)!={'position','velocity','torque'}:raise ValueError('驱动类型字段无效')
        for channel in axes.values():
            if not isinstance(channel,dict) or set(channel)!=set(DRIVE_FIELDS)|{'targets'} or not isinstance(channel['targets'],list) or len(channel['targets'])>4096:raise ValueError('驱动数据无效')
            if any(not isinstance(target,dict) or set(target)!={'frame','value'} for target in channel['targets']):raise ValueError('驱动关键帧无效')
    if not isinstance(settings['d6'],dict) or set(settings['d6'])!={k+'_'+a for k in ('translation','rotation') for a in 'xyz'}:raise ValueError('D6字段无效')
    if any(not isinstance(axis,dict) or set(axis)!={'mode','lower','upper','value'} for axis in settings['d6'].values()):raise ValueError('D6数据无效')
    if not isinstance(value['material'],dict) or set(value['material'])!=set(GROUP_FIELDS):raise ValueError('材质字段无效')
    if not isinstance(value['display'],dict) or set(value['display'])!=set(DISPLAY_FIELDS) or type(value['display']['preview_color_custom']) is not bool:raise ValueError('预览显示字段无效')
    color=value['display']['preview_color']
    if not isinstance(color,list) or len(color)!=3 or any(type(x) not in (int,float) or not 0<=x<=1 for x in color):raise ValueError('预览颜色范围无效')
    geometry=value['geometry']
    if not isinstance(geometry,dict) or set(geometry)!={'vertices','faces','bone_local_from_mesh'}:raise ValueError('代理几何字段无效')
    vertices=geometry['vertices'];faces=geometry['faces']
    if not isinstance(vertices,list) or not 4<=len(vertices)<=MAX_VERTICES or any(not isinstance(v,list) or len(v)!=3 or any(type(x) not in (int,float) for x in v) for v in vertices):raise ValueError('代理顶点无效或超限')
    if not isinstance(faces,list) or not 4<=len(faces)<=MAX_FACES:raise ValueError('代理面无效或超限')
    for face in faces:
        if not isinstance(face,list) or len(face)<3 or any(type(i) is not int or not 0<=i<len(vertices) for i in face) or len(set(face))!=len(face):raise ValueError('代理面索引无效')
    if sum(map(len,faces))>3*MAX_FACES:raise ValueError('代理面环数超限')
    def matrix(m):
        if not isinstance(m,list) or len(m)!=4 or any(not isinstance(row,list) or len(row)!=4 or any(type(x) not in (int,float) for x in row) for row in m):raise ValueError('代理局部变换无效')
        if m[3]!=[0,0,0,1]:raise ValueError('代理局部变换非仿射')
    matrix(geometry['bone_local_from_mesh'])
    if not isinstance(value['frames'],dict) or set(value['frames'])!={'parent_joint_in_bone','child_joint_in_bone'}:raise ValueError('关节帧字段无效')
    for m in value['frames'].values():matrix(m)
    curves=value['curves']
    if not isinstance(curves,dict) or set(curves)!={'mode','channels'} or curves['mode'] not in ('LEGACY','ACTIVE') or not isinstance(curves['channels'],dict) or set(curves['channels'])!=set(CHANNELS):raise ValueError('恢复曲线字段无效')
    for channel in curves['channels'].values():
        if channel is not None and (not isinstance(channel,dict) or set(channel)!={'binding','table'} or not isinstance(channel['binding'],dict) or not isinstance(channel['table'],dict)):raise ValueError('恢复曲线格式无效')
        if channel is not None and (set(channel['binding'])&{'target','table_id'} or 'table_id' in channel['table']):raise ValueError('恢复曲线不得保存工程目标')
    from ..models.restoring_curves import validate_author_map
    for name,record in curves['channels'].items():
        if record is not None:
            validate_author_map(dict(version=1,stitch_pairs=[],tables=[dict(table_id='portable',**record['table'])],bindings=[dict(target=dict(kind='JOINT',joint_uuid='11111111-1111-4111-8111-111111111111',channel=name),table_id='portable',**record['binding'])]))
    if len(json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf-8'))>MAX_BYTES:raise ValueError('预设文件过大')
    return value

def _decode(raw):
    if len(raw)>MAX_BYTES:raise ValueError('预设文件过大')
    def pairs(items):
        out={}
        for key,value in items:
            if key in out:raise ValueError('预设包含重复JSON字段')
            out[key]=value
        return out
    def bad(value):raise ValueError('预设不接受非有限JSON常量')
    try:value=json.loads(raw.decode('utf-8'),object_pairs_hook=pairs,parse_constant=bad)
    except (UnicodeError,json.JSONDecodeError,RecursionError) as error:raise ValueError('预设不是有效UTF8 JSON') from error
    return validate_preset(value)

def load_preset(identifier,*,directory=None):
    path=_path(identifier,directory)
    with path.open('rb') as handle:document=_decode(handle.read(MAX_BYTES+1))
    if document['id']!=identifier:raise ValueError('预设文件ID与内容不符')
    return document

def list_presets(*,directory=None):
    rows=[]
    for path in sorted(_root(directory).glob('*.json')):
        if not _ID.fullmatch(path.stem):continue
        try:document=load_preset(path.stem,directory=directory);rows.append(dict(id=path.stem,name=document['name'],path=str(path),valid=True))
        except (ValueError,OSError) as error:rows.append(dict(id=path.stem,name=path.stem,path=str(path),valid=False,error=str(error)))
    return sorted(rows,key=lambda row:(row['name'].casefold(),row['id']))

def _write(path,document):
    raw=(json.dumps(validate_preset(document),ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(',',':'))+'\n').encode('utf-8')
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.ppf-preset-',suffix='.tmp',mode='wb',delete=False) as handle:
            temporary=Path(handle.name);handle.write(raw);handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()

def save_preset(name,document,overwrite=False,*,preset_id=None,directory=None):
    name=clean_name(name);checked=validate_preset(document)
    with _LOCK:
        root=_root(directory,create=True)
        if preset_id is None:
            if overwrite:raise ValueError('覆盖必须明确选择预设ID')
            if any(row['valid'] and row['name']==name for row in list_presets(directory=root)):raise FileExistsError('同名预设已存在，请显式覆盖所选项')
            identifier=secrets.token_hex(16)
        else:
            if not overwrite:raise FileExistsError('指定已有ID必须显式确认覆盖')
            load_preset(preset_id,directory=root);identifier=preset_id
            if any(row['valid'] and row['id']!=identifier and row['name']==name for row in list_presets(directory=root)):raise FileExistsError('同名预设已存在')
        checked['id']=identifier;checked['name']=name
        path=_path(identifier,root);_write(path,checked);return path

def rename_preset(identifier,new_name,*,directory=None):
    name=clean_name(new_name)
    with _LOCK:
        checked=load_preset(identifier,directory=directory)
        if any(row['valid'] and row['id']!=identifier and row['name']==name for row in list_presets(directory=directory)):raise FileExistsError('同名预设已存在')
        checked['name']=name;path=_path(identifier,directory);_write(path,checked);return path

def delete_preset(identifier,*,directory=None):
    with _LOCK:
        load_preset(identifier,directory=directory);path=_path(identifier,directory);path.unlink();return path

def capture_preset(context,name):
    from .bone_preset_apply import capture
    return validate_preset(capture(context,clean_name(name)))

def apply_preset(context,document,scope='SELECTED',size_mode='FIT_BONE',mask=None):
    from .bone_preset_apply import apply
    return apply(context,validate_preset(document),scope,size_mode,mask)
