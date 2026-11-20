"""Read PMX joint records owned by participating armatures."""
import json


def encode(context, bindings):
    owners = {row['body_uuid']: row['armature_uuid'] for row in bindings}
    active_arms = set(owners.values())
    joints = []
    identities = set()
    for arm in context.scene.objects:
        if arm.type != 'ARMATURE' or arm.ppf_armature_uuid not in active_arms:
            continue
        raw = arm.get('ppf_mmd_joints_v1')
        if raw is None:
            continue
        try:
            rows = json.loads(raw)
            if not isinstance(rows, list):
                raise ValueError('expected a list')
            for row in rows:
                for endpoint in ('body_a_uuid', 'body_b_uuid'):
                    if owners.get(row[endpoint]) != arm.ppf_armature_uuid:
                        raise ValueError(f"joint {row['joint_uuid']} has an inactive or missing rigid body")
                if row['joint_uuid'] in identities:
                    raise ValueError(f"duplicate joint {row['joint_uuid']}")
                identities.add(row['joint_uuid'])
                joints.append(row)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'{arm.name}: invalid PMX joint records ({error})') from error
    return sorted(joints, key=lambda row: row['joint_uuid'])
