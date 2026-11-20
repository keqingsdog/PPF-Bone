"""Exact new bone author intent correspondence to the encoded control program."""
import math
import uuid

def validate_author_intent(document, fail):
    """No Blender sampling: validate real uploaded author/control fields only."""
    required_document = {'schema', 'armatures', 'bindings', 'start_frame', 'fps', 'world_scaling',
                         'body_controls_v1', 'body_controls_author_intent_v1'}
    if (not isinstance(document, dict) or not required_document.issubset(document)
            or set(document) - required_document - {'collision_filter_v1', 'bone_mass_intent_v1', 'soft_limit_auto_v1', 'cross_links_v1', 'cross_link_collision_pairs_v1', 'pmx_joints_v1'}
            or document['body_controls_v1'] is not True or document['schema'] != 'PPF_BLENDER_BONES_V1'
            or not isinstance(document['bindings'], list)):
        fail('unsupported encoded body author document')
    if 'cross_links_v1' in document:
        from .cross_link_author_contract import validate_author
        validate_author(document['cross_links_v1'], document['bindings'], fail)
    if 'cross_link_collision_pairs_v1' in document:
        pairs = document['cross_link_collision_pairs_v1']
        if not isinstance(pairs, list) or not pairs or len(pairs) > 65536:
            fail('cross-link collision pairs must be a nonempty bounded list')
        bodies = {}
        for row in document['bindings']:
            if not isinstance(row, dict) or row.get('enabled') is not True:
                continue
            body = row.get('body_uuid')
            try:
                parsed = uuid.UUID(body)
                if str(parsed) != body or not parsed.int or body in bodies:
                    fail('invalid or duplicate cross-link collision body')
            except (TypeError, ValueError, AttributeError):
                fail('invalid cross-link collision body')
            bodies[body] = row.get('armature_uuid')
        normalized = []
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                fail('invalid cross-link collision pair')
            a, b = pair
            if (not isinstance(a, str) or not isinstance(b, str) or a >= b
                    or a not in bodies or b not in bodies or bodies[a] != bodies[b]):
                fail('cross-link collision pair must name ordered enabled bodies in one armature')
            normalized.append((a, b))
        if normalized != sorted(set(normalized)):
            fail('cross-link collision pairs must be sorted and unique')
    def real(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            fail('body intent requires finite real values')
        return float(value)
    def canonical(value):
        try:
            parsed = uuid.UUID(value)
            if str(parsed) != value or parsed.int == 0: fail('invalid canonical body UUID')
        except (ValueError, TypeError, AttributeError): fail('invalid canonical body UUID')
        return value
    if 'soft_limit_auto_v1' in document:
        limits_intent=document['soft_limit_auto_v1']
        if (not isinstance(limits_intent,dict) or set(limits_intent)!={'version','joints'}
                or type(limits_intent['version']) is not int or limits_intent['version']!=1
                or not isinstance(limits_intent['joints'],list) or not limits_intent['joints']):
            fail('invalid automatic soft-limit author shape')
        seen_limits=set();axes=('twist_x','swing_y','swing_z')
        for spec in limits_intent['joints']:
            if not isinstance(spec,dict) or set(spec) not in ({'joint_uuid','frequency_hz','strength','enabled_axes'}, {'joint_uuid','frequency_hz','strength','enabled_axes','swing_cone'}):
                fail('invalid automatic soft-limit author fields')
            joint=canonical(spec['joint_uuid'])
            matches=[row for row in document['bindings'] if isinstance(row,dict) and row.get('joint_uuid')==joint]
            if joint in seen_limits or len(matches)!=1:fail('duplicate or unknown automatic soft-limit joint')
            seen_limits.add(joint);row=matches[0];physics=row.get('physics',{})
            limits=physics.get('angular_soft_limits_v1') if isinstance(physics,dict) else None
            if (row.get('enabled') is not True
                    or not (row.get('parent_body_uuid') or row.get('automatic_parent_anchor_v1'))
                    or not isinstance(limits,dict)
                    or 'angular_hard_hinge_v1' in physics or 'joint_d6_hard_v1' in physics):
                fail('automatic soft-limit requires an enabled compliant parent joint')
            cone=spec.get('swing_cone',False)
            if type(cone) is not bool:fail('invalid automatic cone flag')
            if cone and (not isinstance(limits.get('swing_cone'),dict) or limits['swing_cone'].get('enabled') is not True):fail('automatic cone differs from joint limits')
            flags=spec['enabled_axes']
            if (not isinstance(flags,list) or len(flags)!=3 or any(type(flag) is not bool for flag in flags)
                    or not (any(flags) or cone) or (cone and any(flags[1:])) or any(not isinstance(limits.get(axis),dict)
                    or limits[axis].get('enabled') is not flag for axis,flag in zip(axes,flags))):
                fail('automatic soft-limit axes differ from joint limits')
            if real(spec['frequency_hz'])<=0 or real(spec['strength'])<0:
                fail('automatic soft-limit frequency/strength invalid')
    intent = document.get('body_controls_author_intent_v1')
    if (not isinstance(intent, dict) or set(intent) != {'body_controls_v1', 'controls'}
            or intent['body_controls_v1'] is not True or not isinstance(intent['controls'], list)
            or not intent['controls']):
        fail('missing or invalid encoded body author intent')
    start = document.get('start_frame'); fps = real(document.get('fps'))
    if type(start) is not int or fps <= 0: fail('invalid body author time base')
    rows = {}
    for row in document.get('bindings', []):
        if not isinstance(row, dict): fail('invalid encoded binding shape')
        if 'fem_root_control_v1' in row:
            if (row['fem_root_control_v1'] is not True or 'root_motion' not in row
                    or row.get('root_mode') not in ('STATIC','KINEMATIC')
                    or row['root_motion'].get('control_events')):
                fail('invalid FEM root control family marker')
        if 'fem_root_control_events_v1' in row:
            if (row['fem_root_control_events_v1'] is not True or 'fem_root_control_v1' in row
                    or not isinstance(row.get('root_motion'),dict) or row.get('root_mode') not in ('FREE','STATIC','KINEMATIC')
                    or (row['root_mode']=='FREE' and not row['root_motion'].get('control_events'))):
                fail('invalid FEM event family marker')
        if 'root_motion' in row:
            uid = canonical(row.get('body_uuid'))
            if uid in rows or row.get('enabled') is not True: fail('duplicate or disabled encoded control')
            rows[uid] = row
    last, seen = '', set()
    modes = {'FREE': 'dynamic', 'KINEMATIC': 'kinematic', 'STATIC': 'static'}
    for control in intent['controls']:
        if not isinstance(control, dict) or set(control) != {'body_uuid', 'mode', 'source', 'events'}:
            fail('invalid encoded author control shape')
        uid = canonical(control['body_uuid'])
        if uid <= last or uid not in rows: fail('unsorted, duplicate or unmatched encoded author control')
        last = uid; seen.add(uid); row = rows[uid]
        mode, source = control['mode'], control['source']
        motion = row['root_motion']
        if (not isinstance(mode, str) or not isinstance(source, str) or mode not in modes or source not in ('POSE', 'OBJECT')
                or row.get('root_mode') != mode or row.get('root_motion_source') != source
                or not isinstance(motion, dict) or motion.get('body_uuid') != uid
                or motion.get('mode') != modes[mode] or motion.get('authoring') != 'object_world_trs'):
            fail('encoded mode/source/body differs from immutable author intent')
        events, wire_events = control['events'], motion.get('control_events', [])
        if not isinstance(events, list) or not isinstance(wire_events, list) or len(events) != len(wire_events):
            fail('encoded event count differs from author intent')
        previous = float(start); previous_mode = mode
        for event, wire in zip(events, wire_events):
            if (not isinstance(event, dict) or set(event) != {'frame', 'mode', 'pose_binding'}
                    or not isinstance(wire, dict) or set(wire) not in (
                        {'time_s', 'mode'}, {'time_s', 'mode', 'pose_binding'})):
                fail('invalid encoded author event shape')
            frame = real(event['frame']); selected = event['mode']; binding = event['pose_binding']
            if (frame <= previous or not isinstance(selected, str) or not isinstance(binding, str)
                    or selected not in modes or selected == previous_mode or binding not in ('CURRENT', 'ABSOLUTE')):
                fail('invalid encoded author event order/mode/binding')
            if selected == 'FREE' and binding != 'ABSOLUTE': fail('dynamic events cannot carry pose binding')
            expected_wire = {'time_s': (frame - start) / fps, 'mode': modes[selected]}
            if binding == 'CURRENT':
                expected_wire['pose_binding'] = 'current_hold' if selected == 'STATIC' else 'rigid_delta_v1'
            elif selected in ('STATIC', 'KINEMATIC'):
                expected_wire['pose_binding'] = 'absolute'
            real(wire.get('time_s'))
            if wire != expected_wire: fail('encoded event program differs from author frames/mode/policy')
            previous, previous_mode = frame, selected
        # Check the source's initial transform using the released rigid-frame
        # tolerance (1e-6 rotation, 1e-8 affine/translation). Full Action sample
        # authentication remains the original encoder/Param drift gate.
        matrix = row.get('bone_world_matrix_blender' if source == 'POSE' else 'proxy_world_matrix_blender')
        if not isinstance(matrix, list) or len(matrix) != 4 or any(not isinstance(r, list) or len(r) != 4 for r in matrix):
            fail('invalid encoded initial source matrix')
        a = [[real(v) for v in r] for r in matrix]
        if any(abs(v - t) > 1e-8 for v, t in zip(a[3], (0., 0., 0., 1.))): fail('invalid initial source affine matrix')
        for i in range(3):
            for j in range(3):
                if abs(sum(a[k][i] * a[k][j] for k in range(3)) - float(i == j)) > 1e-6:
                    fail('initial source scale/shear unsupported')
        determinant = (a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
                       - a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
                       + a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]))
        if abs(determinant - 1.) > 1e-6: fail('initial source reflection unsupported')
        initial = motion.get('initial_frame')
        if not isinstance(initial, dict) or set(initial) != {'translation_m', 'quaternion_wxyz'}:
            fail('invalid initial control frame')
        xyz, quaternion = initial['translation_m'], initial['quaternion_wxyz']
        if not isinstance(xyz, list) or len(xyz) != 3 or not isinstance(quaternion, list) or len(quaternion) != 4:
            fail('invalid initial control frame components')
        xyz = list(map(real, xyz)); q = list(map(real, quaternion)); norm = math.sqrt(sum(v*v for v in q))
        if norm <= 0 or abs(norm - 1.) > 1e-6: fail('invalid initial source quaternion')
        w, x, y, z = [v/norm for v in q]
        rotation = ((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                    (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                    (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))
        basis = (a[0][:3], a[2][:3], [-v for v in a[1][:3]])
        translation = (a[0][3], a[2][3], -a[1][3])
        if any(abs(v-t) > 1e-8 for v,t in zip(xyz, translation)) or any(
                abs(rotation[i][j] - basis[i][j]) > 1e-6 for i in range(3) for j in range(3)):
            fail('encoded initial frame differs from chosen POSE/OBJECT source')
        keys = motion.get('keys')
        if not isinstance(keys, list) or not keys: fail('missing encoded trajectory keys')
        first = keys[0]
        if isinstance(first, dict): real(first.get('time_s'))
        if (not isinstance(first, dict) or first.get('time_s') != 0. or
                first.get('center_world_m', first.get('translation_m')) != initial['translation_m'] or
                first.get('rotation_world_wxyz', first.get('quaternion_wxyz')) != initial['quaternion_wxyz']):
            fail('encoded first key differs from initial author source')
    if seen != set(rows): fail('encoded enabled control set differs from author intent')
    return intent
