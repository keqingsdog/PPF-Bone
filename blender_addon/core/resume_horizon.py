"""Same-seal horizon-only request; never Build or mutate author RNA."""
from copy import deepcopy
import hashlib

FEATURE='bone_checkpoint_resume_v1'
TWIST_MODE='continuous_from_current'
CONTINUOUS_MODEL='parent_local_swing_twist_x_continuous_v2'
MATERIAL_SPIN_MODEL='material_spin_path_v3'
LEGACY_MODEL='parent_local_swing_twist_x_v1'
MODEL_FIELDS=('angular_control_v1','angular_soft_limits_v1','angular_drives_v1')

def validate_contract(response, contract, project_name):
    if not isinstance(response, dict) or not isinstance(contract, dict) or not project_name:
        raise ValueError('TWIST_RESUME_CAPABILITY_UNAVAILABLE')
    if contract.get('name') != project_name:
        raise ValueError('PROJECT_CONTEXT_CHANGED')
    for key in ('root', 'upload_id', 'param_hash', 'data_hash', 'solver_backend'):
        if not response.get(key) or contract.get(key) != response.get(key):
            raise ValueError('PROJECT_CONTEXT_CHANGED')
    if type(contract.get('version')) is not int or contract['version'] != 1 or not isinstance(contract.get('supported_modes'), list) or TWIST_MODE not in contract['supported_modes']:
        raise ValueError('TWIST_RESUME_CAPABILITY_UNAVAILABLE')
    return contract

def project_sealed_models(tree, contract):
    """Project only server-attested eligible fields before the existing Param guard."""
    if (not isinstance(contract,dict) or type(contract.get('version')) is not int or contract['version'] != 1
            or not isinstance(contract.get('supported_modes'),list)
            or TWIST_MODE not in contract['supported_modes']):
        raise ValueError('TWIST_RESUME_CAPABILITY_UNAVAILABLE')
    rows=contract.get('sealed_models')
    if not isinstance(rows,list) or not rows:
        raise ValueError('TWIST_RESUME_CAPABILITY_UNAVAILABLE')
    indexed={}
    for row in rows:
        if not isinstance(row,dict) or not isinstance(row.get('joint_uuid'),str) or row['joint_uuid'] in indexed or not isinstance(row.get('models'),dict):
            raise ValueError('TWIST_RESUME_CAPABILITY_UNAVAILABLE')
        if 'angular_control_v1' not in row['models'] or any(k not in MODEL_FIELDS or v != LEGACY_MODEL for k,v in row['models'].items()):
            raise ValueError('TWIST_RESUME_CAPABILITY_UNAVAILABLE')
        indexed[row['joint_uuid']]=row['models']
    result=deepcopy(tree)
    seen=set()
    for row in result.get('blender_bones_v1',{}).get('bindings',[]):
        jid=row.get('joint_uuid')
        if jid not in indexed:continue
        if jid in seen:raise ValueError('RESUME_INPUT_DRIFT')
        seen.add(jid)
        physics=row.get('physics')
        if not isinstance(physics,dict):raise ValueError('RESUME_INPUT_DRIFT')
        for field,old_model in indexed[jid].items():
            item=physics.get(field)
            allowed=(old_model,CONTINUOUS_MODEL,MATERIAL_SPIN_MODEL)
            if not isinstance(item,dict) or item.get('coordinate_model') not in allowed:
                raise ValueError('RESUME_INPUT_DRIFT')
            item['coordinate_model']=old_model
    if seen != set(indexed):raise ValueError('RESUME_INPUT_DRIFT')
    return result

def _legacy_policy_candidate(tree):
    """Try the prior encoding policy without changing Blender authoring data.

    This candidate still has to pass the entire existing sealed Param guard;
    it does not opt into continuous mode or authorize any other modification.
    """
    candidate = deepcopy(tree)
    for row in candidate.get('blender_bones_v1', {}).get('bindings', []):
        physics = row.get('physics', {})
        if not isinstance(physics, dict):
            continue
        for field in MODEL_FIELDS:
            item = physics.get(field)
            if isinstance(item, dict) and item.get('coordinate_model') in (CONTINUOUS_MODEL,MATERIAL_SPIN_MODEL):
                item['coordinate_model'] = LEGACY_MODEL
    return candidate

def _matches_sealed_param(tree, sealed_hash, dumps_envelope, allow_legacy_policy):
    if not sealed_hash:
        return False
    def digest(value):
        return hashlib.sha256(dumps_envelope('Param', value)).hexdigest()
    def matches(value):
        if digest(value) == sealed_hash:
            return True
        if value.get('solver_mode') == 'strict':
            # Older clients sealed the default strict mode without a field.
            legacy = {key: item for key, item in value.items() if key != 'solver_mode'}
            return digest(legacy) == sealed_hash
        return False
    if matches(tree):
        return True
    # A new export may describe v3, while an already sealed session is still
    # v2.  This projection is used only for the existing sealed-input guard;
    # it does not mutate the session or replace its native material model.
    candidate=deepcopy(tree)
    for row in candidate.get('blender_bones_v1',{}).get('bindings',[]):
        for field in MODEL_FIELDS:
            item=row.get('physics',{}).get(field)
            if isinstance(item,dict) and item.get('coordinate_model')==MATERIAL_SPIN_MODEL:
                item['coordinate_model']=CONTINUOUS_MODEL
    if matches(candidate):
        return True
    return allow_legacy_policy and matches(_legacy_policy_candidate(tree))

def normalized_param_tree(tree, sealed_target):
    """Allow only the count and redundant constant bone-root tail samples.

    Everything else stays byte-identical under the existing Param encoder.
    In particular this does not manufacture coverage for changed animation,
    force fields, sampled materials, transforms, events or a new generation.
    Native independently decides whether its sealed tail contract covers it.
    """
    tree=deepcopy(tree)
    fps=float(tree['scene']['fps'])
    end=sealed_target/fps
    tree['scene']['frames']=sealed_target
    def root_tail(root):
        if not isinstance(root,dict) or not root.get('keys'):return
        keys=root['keys'];past=[k for k in keys if float(k['time_s'])<=end+1e-12]
        future=[k for k in keys if float(k['time_s'])>end+1e-12]
        if not future:return
        if not past:raise ValueError('RESUME_CONTROL_COVERAGE_INSUFFICIENT: no sealed root sample')
        value=lambda k:{a:v for a,v in k.items() if a!='time_s'}
        if any(value(k)!=value(past[-1]) for k in future):
            raise ValueError('RESUME_CONTROL_COVERAGE_INSUFFICIENT: new author root trajectory requires fresh generation')
        root['keys']=past
    for row in tree.get('blender_bones_v1',{}).get('bindings',[]):
        root_tail(row.get('root_motion'))
        root_tail(row.get('automatic_parent_anchor_v1',{}).get('root_motion'))
    return tree

def prepare_request(context, response, from_frame, twist_resume_mode=None):
    from .encoder.params import _build_param_dict
    from .encoder.cbor_encode import dumps_envelope
    from .bone_mass import producer_preflight
    from ..models.groups import get_addon_data
    target=int(get_addon_data(context.scene).state.frame_count)-1
    sealed=response.get('sealed_target_frame')
    if type(sealed) is not int or sealed<0:
        raise ValueError('RESUME_CAPABILITY_UNAVAILABLE: server does not report sealed horizon')
    if type(from_frame) is not int or from_frame<=0 or target<=from_frame:
        raise ValueError('RESUME_TARGET_INVALID: select nonzero checkpoint earlier than target')
    producer_preflight(context,commit=False)
    pre_mmd_default=[]
    tree=_build_param_dict(context,pre_mmd_default=pre_mmd_default)
    if 'solver_mode' in tree:
        # A changed panel selection applies to the next Transfer. Resume
        # validates the exact mode that owns the current checkpoints.
        sealed_mode = response.get('solver_mode')
        tree['solver_mode'] = sealed_mode if sealed_mode in ('strict', 'animation') else 'strict'
    if twist_resume_mode is not None:
        if twist_resume_mode != TWIST_MODE:raise ValueError('TWIST_RESUME_MODE_INVALID')
        tree=project_sealed_models(tree,response.get('twist_resume_contract'))
    if target!=sealed:
        if target<sealed:raise ValueError('RESUME_TARGET_INVALID: horizon may only increase')
        tree=normalized_param_tree(tree,sealed)
    seal_hash=response.get('param_hash')
    allow_legacy=twist_resume_mode is None
    if _matches_sealed_param(tree, seal_hash, dumps_envelope, allow_legacy):
        return target if target>sealed else None
    # Previously sealed generations keep their original restoring values.
    # Only remove newly synthesized defaults and require the full old seal.
    if pre_mmd_default:
        if pre_mmd_default[0] is None:
            tree.pop('restoring_curves_v1')
        else:
            tree['restoring_curves_v1']=pre_mmd_default[0]
        if _matches_sealed_param(tree, seal_hash, dumps_envelope, allow_legacy):
            return target if target>sealed else None
    raise ValueError('RESUME_INPUT_DRIFT: only horizon and identical constant root tail are allowed; do not Update Params/Build for same-seal resume')
