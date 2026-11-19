"""Physical control display units; no mass/inertia estimates or scene writes."""
import math

CHANNELS=('twist_x','swing_y','swing_z','anchor')
ANGULAR_FACTOR=1000.0*math.pi/180.0
LINEAR_FACTOR=.01

def nonnegative(value):
    value=float(value)
    if not math.isfinite(value) or value<0:
        raise ValueError('物理刚度和阻尼必须为有限非负数')
    return value

def to_display(value,channel):
    return nonnegative(value)*(LINEAR_FACTOR if channel=='anchor' else ANGULAR_FACTOR)

def to_si(value,channel):
    return nonnegative(value)/(LINEAR_FACTOR if channel=='anchor' else ANGULAR_FACTOR)

def linear_constants(table,damping):
    """A through-origin SI line, including an enabled exact-zero line."""
    nodes=table['nodes']
    if table.get('tail')!='LINEAR' or not nodes or any(len(p)!=2 for p in nodes):
        return None
    nonzero=[(float(x),float(f)) for x,f in nodes if float(x)!=0.]
    if not nonzero or any(float(f)!=0. for x,f in nodes if float(x)==0.):return None
    k=nonnegative(nonzero[0][1]/nonzero[0][0]);c=nonnegative(damping)
    if any(not math.isclose(float(f),k*float(x),rel_tol=1e-6,abs_tol=1e-12) for x,f in nodes):return None
    return k,c

def resolved_channels(document,joint_uuid):
    """Read actual Build SI tables, validating each target/receipt relationship."""
    if document.get('kind')!='ppf.restoring_curves' or document.get('version')!=1:
        raise ValueError('上次构建恢复参数格式不支持')
    tables={t['table_id']:t for t in document['tables']}
    receipts={r['receipt_id']:r for r in document.get('mass_receipts',[])}
    result={}
    for binding in document['bindings']:
        target=binding['target']
        if target.get('kind')!='JOINT' or target.get('joint_uuid')!=joint_uuid:continue
        channel=target['channel']
        if channel not in CHANNELS:continue
        if channel in result:raise ValueError('上次构建有重复恢复通道')
        rid=binding.get('mass_receipt_id')
        if rid:
            receipt=receipts.get(rid)
            if receipt is None or receipt['target']!=target:
                raise ValueError('上次构建恢复参数与有效质量凭据不匹配')
            if receipt.get('metric_unit')!=('kg' if channel=='anchor' else 'kg_m2'):
                raise ValueError('上次构建有效质量或惯量单位不匹配')
            if nonnegative(receipt['metric_eff'])<=0:raise ValueError('上次构建有效质量或惯量无效')
            if receipt.get('basis_sha256')!=document.get('basis_sha256'):
                raise ValueError('上次构建恢复参数不属于同一构建')
            nonnegative(receipt['resolved_stiffness']);nonnegative(receipt['resolved_damping'])
            if not math.isclose(float(binding['damping']),float(receipt['resolved_damping']),rel_tol=1e-6,abs_tol=1e-12):
                raise ValueError('上次构建阻尼与质量凭据不匹配')
        value=linear_constants(tables[binding['table_id']],binding['damping'])
        if value is not None:result[channel]=value
    return result


def producer_signature(context):
    """Explicit Load/Apply current-input check through existing Run producers.

    This is a read-only author-input identity, not a new inertia calculation.
    Never call from draw or through prepare_upload's committing transaction.
    """
    from .rigid_pose import suspend_playback
    from .encoder.mesh import compute_data_hash
    from .encoder.params import compute_param_hash
    with suspend_playback():
        return compute_data_hash(context),compute_param_hash(context)


def _seal_contexts(seal):
    if seal is None:return []
    if hasattr(seal,'scene_uuid') and hasattr(seal,'inputs_hash'):
        return [(seal.scene_uuid,seal.inputs_hash,seal.context_hash)]
    if isinstance(seal,tuple):
        # Production FEM verified-seal tuple, or its mixed rigid/FEM pair.
        if len(seal)>=5 and isinstance(seal[0],str):return [(seal[0],seal[4],seal[1])]
        return [item for part in seal for item in _seal_contexts(part)]
    raise ValueError('当前构建物理凭据格式不支持，请重新构建')


def trusted_build(context):
    """Read current acknowledged Build proof; never query or validate remotely.

    Reuses the body receipt already verified by the normal Build path, including
    its request_token/upload binding. Checks only the two fixed sealed protocol
    artifacts against existing proof; no inventory scan or inertia estimation.
    """
    from pathlib import Path
    import cbor2,json,hashlib
    from .facade import communicator,engine,runner
    from ..models.groups import get_addon_data
    failure='恢复力数值已过期或缺少当前构建凭据，请重新传输、构建并载入数值'
    response=communicator.info.response or {}
    project=get_addon_data(context.scene).state.project_name
    with runner._collision_lock:
        expected=runner._body_expected;seal=runner._body_seal
        verified=runner._body_verified;dirty=runner._body_retransfer
    if (not verified or dirty or expected is None or seal is None
            or communicator.project_name!=project):raise ValueError(failure)
    identity={key:getattr(expected,key,'') for key in ('root','upload_id','data_hash','param_hash')}
    if (not all(isinstance(v,str) and v for v in identity.values())
            or any(response.get(key)!=value for key,value in identity.items())
            or str(engine.state.remote_root)!=identity['root']):raise ValueError(failure)
    if get_addon_data(context.scene).ssh_state.server_type!='WIN_NATIVE':raise ValueError(failure)
    path=Path(identity['root'])/'session/bin/pdrd_scene_context.cbor'
    context_raw=path.read_bytes()
    proofs=_seal_contexts(seal)
    context_identity=hashlib.sha256(context_raw).hexdigest()
    if not proofs or any(item[2]!=context_identity for item in proofs):raise ValueError(failure)
    envelope=cbor2.loads(context_raw)
    if (not isinstance(envelope,dict) or envelope.get('kind')!='ppf.pdrd_scene_context'
            or envelope.get('version')!=2):raise ValueError(failure)
    sealed=envelope.get('payload') or {};scene=sealed.get('scene_uuid');basis=sealed.get('scene_inputs_sha256')
    marker=sealed.get('restoring_curves_v1') or {}
    sidecar_raw=(path.parent/'restoring_curves_v1.json').read_bytes()
    if (marker.get('sidecar_byte_length')!=len(sidecar_raw)
            or marker.get('sidecar_sha256')!=hashlib.sha256(sidecar_raw).hexdigest()):raise ValueError(failure)
    document=json.loads(sidecar_raw.decode('utf8'))
    if (sealed.get('complete') is not True or not scene or not basis or not proofs
            or any(item[:2]!=(scene,basis) for item in proofs)
            or marker.get('required_feature')!='restoring_curves_v1'
            or not marker.get('sidecar_sha256')
            or document.get('scene_uuid')!=scene or document.get('basis_sha256')!=basis):raise ValueError(failure)
    current=producer_signature(context)
    if current!=(identity['data_hash'],identity['param_hash']):raise ValueError(failure)
    return document,dict(project=project,**identity,scene_uuid=scene,base_inputs=basis,
                restoring_sidecar=marker['sidecar_sha256'])


_display_key=None
_display_document=None


def display_build_key(context):
    """Only existing acknowledged in-memory Build state; no hashes or IO."""
    from .facade import communicator,engine,runner
    from ..models.groups import get_addon_data
    with runner._collision_lock:
        expected=runner._body_expected;seal=runner._body_seal
        if not runner._body_verified or runner._body_retransfer or expected is None or seal is None:return None
    project=get_addon_data(context.scene).state.project_name
    response=communicator.info.response or {}
    identity=tuple(getattr(expected,k,'') for k in ('root','upload_id','data_hash','param_hash'))
    if not all(identity) or project!=communicator.project_name:return None
    if any(response.get(k)!=v for k,v in zip(('root','upload_id','data_hash','param_hash'),identity)):return None
    if str(engine.state.remote_root)!=identity[0]:return None
    if get_addon_data(context.scene).ssh_state.server_type!='WIN_NATIVE':return None
    return (context.scene.as_pointer(),project,*identity,tuple(_seal_contexts(seal)))


def display_build_document(context):
    """Read a verified Build's sidecar once per generation for display only.

    Never author from this display cache. Normal Transfer/Run guards retain
    their full verification. Failed reads are cached too, avoiding redraw IO.
    """
    global _display_key,_display_document
    key=display_build_key(context)
    if key is None:
        _display_key=None;_display_document=None;return None
    if key==_display_key:return _display_document
    _display_key=key;_display_document=None
    try:
        import json
        from pathlib import Path
        document=json.loads((Path(key[2])/'session/bin/restoring_curves_v1.json').read_text(encoding='utf8'))
        proofs=key[-1]
        if not proofs or any((document.get('scene_uuid'),document.get('basis_sha256'))!=proof[:2] for proof in proofs):return None
        if document.get('kind')!='ppf.restoring_curves' or document.get('version')!=1:return None
        _display_document=document
    except (OSError,ValueError,TypeError,KeyError):pass
    return _display_document


def display_auto_value(context,joint,channel,saved):
    document=display_build_document(context)
    if document is None or linear_constants(saved['table'],0.) is None:return None
    receipts=[r for r in document.get('mass_receipts',[]) if r.get('target',{}).get('joint_uuid')==joint and r.get('target',{}).get('channel')==channel]
    if len(receipts)!=1:return None
    receipt=receipts[0];binding=saved['binding']
    if any(receipt.get(k)!=binding.get(k) for k in ('frequency_hz','damping_ratio','coordinate_scale')):return None
    # Compare the authored line to the resolved line; mass scaling must be the
    # only transformation, so nonlinear or replaced author curves stay pending.
    expected_k=nonnegative(receipt['resolved_stiffness'])*linear_constants(saved['table'],0.)[0]
    value=resolved_channels(document,joint).get(channel)
    if value is None or not math.isclose(value[0],expected_k,rel_tol=1e-6,abs_tol=1e-12):return None
    return value
