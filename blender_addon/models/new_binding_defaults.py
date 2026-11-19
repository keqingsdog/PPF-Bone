"""First-creation defaults only; never reinterpret legacy RNA fallbacks.

This module performs read-only planning. The Add / Update transaction applies
its result only after the complete batch has passed ownership/shape/graph gates.
The material numbers are a bounded calibration candidate, not a rigidity claim.
"""
NEW_SOFT_YOUNG_PA = 5.0e8
NEW_SOFT_DAMPING_SECONDS = 0.002
NEW_SOFT_ANCHOR_N_M = 2.0e5
NEW_GENERATED_SOFT_TET_BACKEND = 'TETGEN'
AUTO_GEOMETRY_FIELDS = ('radius', 'length_ratio', 'box_width', 'box_depth')


def geometry_values_match(settings, values):
    return (isinstance(values, dict) and set(values) == set(AUTO_GEOMETRY_FIELDS)
            and all(float(getattr(settings, key)) == values[key] for key in AUTO_GEOMETRY_FIELDS))


def geometry_snapshot_matches(settings):
    import json
    try:
        return geometry_values_match(settings, json.loads(settings.auto_geometry_snapshot_json))
    except (AttributeError, ValueError, TypeError):
        return False


def recreate_geometry_intent(settings, armature_uuid=None):
    """Validate explicit Delete intent, never infer historical AUTO source."""
    import json
    if settings.proxy or settings.enabled:
        return None
    try:
        token = json.loads(settings.recreate_geometry_intent_json)
        owner = token['owner']
        if (token.get('version') != 1 or token.get('mode') not in ('DEFAULTS', 'PRESERVE')
                or token.get('shape') != settings.shape
                or not settings.bone_uuid or not settings.body_uuid
                or any(owner.get(key) != getattr(settings, key)
                       for key in ('bone_uuid', 'body_uuid', 'joint_uuid'))
                or not owner.get('armature_uuid')
                or armature_uuid is not None and owner['armature_uuid'] != armature_uuid
                or not geometry_values_match(settings, token['dimensions'])):
            return None
        return token['mode']
    except (AttributeError, ValueError, TypeError, KeyError):
        return None


def deletion_geometry_intent(settings, armature_uuid, mode):
    import json
    return json.dumps(dict(version=1, mode=mode, shape=settings.shape,
        owner=dict(armature_uuid=armature_uuid, bone_uuid=settings.bone_uuid,
                   body_uuid=settings.body_uuid, joint_uuid=settings.joint_uuid),
        dimensions={key:float(getattr(settings,key)) for key in AUTO_GEOMETRY_FIELDS}), sort_keys=True)


def preserve_recreate_after_edit(settings):
    """Cancel default regeneration without falling back into an old AUTO mark."""
    import json
    try:
        token=json.loads(settings.recreate_geometry_intent_json)
        owner=token['owner']
        if (not settings.proxy and not settings.enabled and owner.get('armature_uuid')
                and all(owner.get(key)==getattr(settings,key)
                        for key in ('bone_uuid','body_uuid','joint_uuid'))):
            settings.recreate_geometry_intent_json=deletion_geometry_intent(settings,owner['armature_uuid'],'PRESERVE')
            return
    except (AttributeError,ValueError,TypeError,KeyError):
        pass
    settings.recreate_geometry_intent_json=''


def auto_geometry_recreatable(settings):
    """Only an absent proxy with unchanged proven automatic dimensions."""
    if settings.proxy or settings.enabled:
        return False
    if recreate_geometry_intent(settings) == 'PRESERVE':
        return False
    return geometry_snapshot_matches(settings)


def record_auto_geometry(settings):
    import json
    settings.auto_geometry_snapshot_json = json.dumps(
        {key: float(getattr(settings, key)) for key in AUTO_GEOMETRY_FIELDS}, sort_keys=True)


def _default_geometry_size(bone_length):
    import math
    if bone_length is None or not math.isfinite(bone_length) or bone_length <= 0:
        raise ValueError('骨骼长度必须大于零')
    return dict(length_ratio=1., radius=bone_length*.2,
                box_width=bone_length*.3, box_depth=bone_length*.3,
                _auto_geometry_defaults=True)


def connected_clearance_ratios(records, edges):
    """Plan real symmetric shortening; records/edges contain no mutable RNA.

    Only an eligible fresh endpoint moves. Contact gaps are a conservative
    initial clearance budget, not a guarantee for bent or overlapping rigs.
    """
    import math
    insets = {name: 0. for name in records}
    for parent, child in edges:
        a, b = records[parent], records[child]
        eligible = [name for name in (parent, child) if records[name]['eligible']]
        if not eligible:
            continue
        values = [a['offset'], b['offset'], a['gap'], b['gap'],
                  a['length'], b['length'], a['coordinate_scale'], b['coordinate_scale']]
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('Invalid new proxy contact clearance')
        rounding = 32. * 2.**-23 * max(a['coordinate_scale'], b['coordinate_scale'], a['length'], b['length'])
        distance = a['offset'] + b['offset'] + max(a['gap'], b['gap']) + rounding
        for name in eligible:
            insets[name] = max(insets[name], distance / len(eligible))
    result = {}
    for name, inset in insets.items():
        if not inset:
            continue
        record = records[name]
        length = record['length'] - 2. * inset
        minimum = 2. * record['radius'] if record['shape'] == 'CAPSULE' else 0.
        if not math.isfinite(length) or length <= minimum:
            raise ValueError('New connected proxy is too short for its radius and contact clearance: '+name)
        result[name] = length / record['length']
    return result


def uses_new_geometry_defaults(settings):
    """No existing binding or authored geometry is reinterpreted as new."""
    if settings.enabled or settings.proxy:
        return False
    if any(getattr(settings, key) for key in
           ('bone_uuid', 'body_uuid', 'joint_uuid', 'group_uuid')):
        return False
    return True


def uses_new_binding_preset(settings):
    """A pristine binding with no explicit legacy material selection."""
    if not uses_new_geometry_defaults(settings):
        return False
    # is_property_set distinguishes an explicit RIGID choice from the legacy
    # RIGID fallback. Fail closed if introspection is unavailable.
    try:
        return not settings.is_property_set('physical_model')
    except (AttributeError, RuntimeError, TypeError):
        return False


def new_binding_plan(settings, bone_length=None, reset_default_size=False, armature_uuid=None):
    """Return only fields the creation transaction may initialize."""
    intent = recreate_geometry_intent(settings, armature_uuid)
    if intent == 'PRESERVE' and not reset_default_size:
        return {}
    if reset_default_size or (settings.shape != 'CUSTOM' and
                             (intent == 'DEFAULTS' or auto_geometry_recreatable(settings))):
        if settings.shape == 'CUSTOM':
            raise ValueError('自定义网格不支持重算默认尺寸')
        if bone_length is None and not reset_default_size:
            # Read-only UI/material queries need no new size calculation.
            return {}
        return _default_geometry_size(bone_length)
    if not uses_new_geometry_defaults(settings):
        return {}
    preset = uses_new_binding_preset(settings)
    model = settings.new_binding_physical_model if preset else settings.physical_model
    planned = {'physical_model': model} if preset else {}
    from .proxy_generation import effective_proxy_quality, effective_proxy_face_mode
    values = {'density': 1., 'shape': 'CAPSULE', 'proxy_quality': effective_proxy_quality(settings),
              'proxy_face_mode': effective_proxy_face_mode(settings)}
    if bone_length is not None:
        import math
        if not math.isfinite(bone_length) or bone_length <= 0:
            raise ValueError('骨骼长度必须大于零')
        # BoneX CapsuleShape/BoxShape dimensions are relative to bone length.
        # Only unset first-creation fields acquire these defaults.
        values.update(length_ratio=1.,radius=bone_length*.2,
                      box_width=bone_length*.3,box_depth=bone_length*.3)
    if preset and model == 'SOFT':
        values.update(solid_model='STABLE_NEOHOOKEAN',
                      solid_young_modulus=NEW_SOFT_YOUNG_PA,
                      solid_poisson_ratio=0.3,
                      anchor_stiffness=NEW_SOFT_ANCHOR_N_M)
    # Explicit SOFT selection alone is not an authored modulus. Keep every
    # other material/anchor value on that path; initialize only unset E.
    if model == 'SOFT' and not settings.is_property_set('solid_young_modulus'):
        values['solid_young_modulus'] = NEW_SOFT_YOUNG_PA
    for key, value in values.items():
        if not settings.is_property_set(key):
            planned[key] = value
    if bone_length is not None and all(not settings.is_property_set(key) for key in AUTO_GEOMETRY_FIELDS):
        planned['_auto_geometry_defaults'] = True
    return planned


# Finite world capsules only; these helpers never modify author RNA.
import math
def _vsub(a,b):return tuple(x-y for x,y in zip(a,b))
def _vadd(a,b):return tuple(x+y for x,y in zip(a,b))
def _vmul(a,t):return tuple(x*t for x in a)
def _vdot(a,b):return sum(x*y for x,y in zip(a,b))
def _clamp01(v):return min(1.,max(0.,v))
def capsule_segment_distance(p1,q1,p2,q2):
    d1=_vsub(q1,p1);d2=_vsub(q2,p2);r=_vsub(p1,p2)
    a=_vdot(d1,d1);e=_vdot(d2,d2);f=_vdot(d2,r)
    if a==0 and e==0:return math.dist(p1,p2)
    if a==0:s=0.;t=_clamp01(f/e)
    else:
        c=_vdot(d1,r)
        if e==0:t=0.;s=_clamp01(-c/a)
        else:
            b=_vdot(d1,d2);denom=a*e-b*b
            s=_clamp01((b*f-c*e)/denom) if denom>0 else 0.
            t=(b*s+f)/e
            if t<0:t=0.;s=_clamp01(-c/a)
            elif t>1:t=1.;s=_clamp01((b-c)/a)
    return math.dist(_vadd(p1,_vmul(d1,s)),_vadd(p2,_vmul(d2,t)))
def _capsule_axis(record,inset):
    radius=record['radius'];u=record['direction']
    return _vadd(record['head'],_vmul(u,radius+inset)),_vsub(record['tail'],_vmul(u,radius+inset))
def angle_clearance_insets(records,pairs):
    insets={name:0. for name in records};proof=[]
    for a,b in pairs:
        ra,rb=records[a],records[b]
        capacities={name:(records[name]['length']/2-records[name]['radius']) if records[name]['eligible'] else 0. for name in (a,b)}
        if not any(capacities.values()):continue
        rounding=32.*2.**-23*max(abs(v) for r in (ra,rb) for p in (r['head'],r['tail']) for v in (*p,r['length']))
        required=ra['radius']+rb['radius']+ra['offset']+rb['offset']+max(ra['gap'],rb['gap'])+rounding
        def distance(t):return capsule_segment_distance(*_capsule_axis(ra,t*capacities[a]),*_capsule_axis(rb,t*capacities[b]))
        lo=0.;hi=1.-32.*2.**-23
        if distance(hi)<required:raise ValueError('Cannot fit new capsule clearance: '+a+'/'+b)
        if distance(0)>=required:hi=0.
        else:
            for _ in range(64):
                mid=(lo+hi)/2
                if distance(mid)>=required:hi=mid
                else:lo=mid
        for name in (a,b):insets[name]=max(insets[name],hi*capacities[name])
        proof.append((a,b,required))
    for a,b,required in proof:
        actual=capsule_segment_distance(*_capsule_axis(records[a],insets[a]),*_capsule_axis(records[b],insets[b]))
        if actual<required:raise ValueError('Final capsule clearance failed: '+a+'/'+b)
    return insets,proof


def default_capsule_clearance_eligible(settings, plan, shape):
    """Whether automatic angle shortening may change this new capsule.

    Explicit body-model selection does not make untouched shape dimensions
    authored. Conversely, an explicitly authored radius/length ratio is never
    changed by the clearance planner.
    """
    if shape != 'CAPSULE':
        return False
    if plan.get('_auto_geometry_defaults'):
        return True
    if not uses_new_geometry_defaults(settings):
        return False
    model=plan.get('physical_model',settings.physical_model)
    if model not in ('RIGID','SOFT'):
        return False
    try:
        return not settings.is_property_set('radius') and not settings.is_property_set('length_ratio')
    except (AttributeError, RuntimeError, TypeError):
        return False


def _planned_contact(record, inset):
    """Ideal world capsule AABB bounds every generated polygonal capsule."""
    relative = record.get('contact_relative')
    if relative is None:
        return record['gap'], record['offset']
    length = record['length'] - 2. * inset
    straight = max(0., length - 2. * record['radius'])
    diagonal = math.sqrt(sum((abs(u) * straight + 2. * record['radius']) ** 2
                             for u in record['direction']))
    return diagonal * relative[0], diagonal * relative[1]


def _pair_margin(a, b, inset_a, inset_b):
    ga, oa = _planned_contact(a, inset_a)
    gb, ob = _planned_contact(b, inset_b)
    rounding = 32. * 2. ** -23 * max(
        a['coordinate_scale'], b['coordinate_scale'], a['length'], b['length'])
    return oa + ob + max(ga, gb) + rounding


def _inset_capacity(record):
    if not record['eligible'] and not record.get('length_adjustable'):
        return 0.
    minimum = max(record['minimum_length'],
                  2. * record['radius'] * (1. + 64. * 2. ** -23))
    return max(0., (record['length'] - minimum) / 2.)


def adaptive_capsule_clearance(records, pairs):
    """Fit only untouched new capsules; all writes remain in the transaction.

    First fit radii at the shortest representable lengths. Then find the
    least shortening for each pair and verify every pair using final dimensions.
    Existing or explicitly authored geometry is a fixed obstacle throughout.
    """
    fitted = {name: dict(record) for name, record in records.items()}
    active_pairs = [(a, b) for a, b in pairs
                    if fitted[a]['eligible'] or fitted[b]['eligible']
                    or fitted[a].get('new_geometry') or fitted[b].get('new_geometry')]

    def clearance(a, b, inset_a, inset_b):
        return (capsule_segment_distance(*_capsule_axis(a, inset_a),
                                         *_capsule_axis(b, inset_b))
                - a['radius'] - b['radius'] - _pair_margin(a, b, inset_a, inset_b))

    for first, second in active_pairs:
        a, b = fitted[first], fitted[second]
        if clearance(a, b, _inset_capacity(a), _inset_capacity(b)) >= 0.:
            continue
        original = {name: fitted[name]['radius'] for name in (first, second)}

        def scaled(t):
            candidates = []
            for name in (first, second):
                value = dict(fitted[name])
                if value['eligible']:
                    value['radius'] = max(value['minimum_radius'], original[name] * t)
                candidates.append(value)
            return candidates

        lo, hi = 0., 1.
        smallest_a, smallest_b = scaled(0.)
        if clearance(smallest_a, smallest_b, _inset_capacity(smallest_a),
                     _inset_capacity(smallest_b)) < 0.:
            raise ValueError(f'{first}/{second}：保守包围界无法证明新代理的碰撞间距，或可用空间不足；请调整初始姿态或代理尺寸后重试')
        for _ in range(40):
            mid = (lo + hi) / 2.
            candidate_a, candidate_b = scaled(mid)
            if clearance(candidate_a, candidate_b, _inset_capacity(candidate_a),
                         _inset_capacity(candidate_b)) >= 0.:
                lo = mid
            else:
                hi = mid
        # Leave room for float32 RNA storage rather than fitting at tangency.
        for name, value in zip((first, second), scaled(lo * .98)):
            if value['eligible']:
                fitted[name]['radius'] = value['radius']

    insets = dict.fromkeys(fitted, 0.)
    for first, second in active_pairs:
        a, b = fitted[first], fitted[second]
        ca, cb = _inset_capacity(a), _inset_capacity(b)
        lo, hi = 0., 1.
        if clearance(a, b, 0., 0.) >= 0.:
            hi = 0.
        else:
            if clearance(a, b, ca, cb) < 0.:
                raise ValueError(f'{first}/{second}：无法为新代理保留有效碰撞间隙')
            for _ in range(48):
                mid = (lo + hi) / 2.
                if clearance(a, b, mid * ca, mid * cb) >= 0.:
                    hi = mid
                else:
                    lo = mid
        insets[first] = max(insets[first], hi * ca)
        insets[second] = max(insets[second], hi * cb)

    # Only the explicit shape-conversion path needs this small extra reserve
    # for storing its new length ratio in float32 RNA. Radius stays authored.
    for name,record in fitted.items():
        if record.get('length_adjustable') and insets[name]>0.:
            reserve=32.*2.**-23*max(record['coordinate_scale'],record['length'])
            insets[name]=min(_inset_capacity(record),insets[name]+reserve)
    for first, second in active_pairs:
        if clearance(fitted[first], fitted[second], insets[first], insets[second]) < -1.e-12:
            raise ValueError(f'{first}/{second}：最终代理碰撞间隙检查失败')
    return fitted, insets
