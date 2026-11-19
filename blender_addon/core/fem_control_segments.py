"""Validate event output identities against the already verified upload seal."""
import math
from .fem_body_control_events import FEATURE


def _tuple(value):
    return tuple(_tuple(x) for x in value) if isinstance(value,(tuple,list)) else value


def validate(pose, context):
    def fail(message):raise ValueError('FEM_CONTROL_SEGMENTS_INVALID: '+message)
    if not isinstance(context,dict) or set(context)!={'program_sha256','controls'}:fail('missing sealed runtime context')
    if pose.get('control_program_sha256')!=context['program_sha256']:fail('control program generation changed')
    controls=_tuple(context['controls'])
    if not controls or any(not isinstance(r,tuple) or len(r)!=8 or r[6:]!=('FEM_EVENTS',2) for r in controls):fail('unknown control family')
    expected={r[0]:r for r in controls}
    rows=pose.get('control_segments')
    if not isinstance(rows,list) or len(rows)!=len(expected):fail('missing control segments')
    fields={'body_uuid','segment_mode','segment_pose_binding','segment_epoch','segment_event_cursor',
            'accepted_end_mode','accepted_end_pose_binding','accepted_end_epoch','accepted_end_event_cursor'}
    a=pose['accepted_bracket_start_seconds'];b=pose['accepted_solver_time_seconds'];seen=set()
    def state(program,time):
        mode=program[1];binding=None if mode=='FREE' else 'absolute';cursor=0
        for event in program[3]:
            if event[0]>time:break
            cursor+=1;mode=event[1];binding=event[2]
        return (mode,binding,cursor,cursor)
    last=''
    for row in rows:
        if not isinstance(row,dict) or set(row)!=fields:fail('invalid segment schema')
        uid=row['body_uuid']
        if uid not in expected or uid in seen or uid<=last:fail('segment identity/order differs')
        seen.add(uid);last=uid
        if any(a<event[0]<b for event in expected[uid][3]):fail('accepted bracket crosses an uncut control event')
        for prefix,time in (('segment',a),('accepted_end',b)):
            epoch=row[prefix+'_epoch'];cursor=row[prefix+'_event_cursor']
            if type(epoch) is not int or type(cursor) is not int or epoch<0 or cursor<0:fail('invalid segment cursor/epoch')
            actual=(row[prefix+'_mode'],row[prefix+'_pose_binding'],epoch,cursor)
            if actual!=state(expected[uid],time):fail('segment mode/policy/epoch differs from accepted event program')
    if seen!=set(expected):fail('segment root inventory differs')
