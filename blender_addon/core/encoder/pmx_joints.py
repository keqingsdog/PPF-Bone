"""Read PMX joint records owned by participating armatures."""
import json
from functools import lru_cache


def _body_pair(row):
    return frozenset((row['body_a_uuid'], row['body_b_uuid']))


def _read_rows(raw):
    rows = json.loads(raw)
    if not isinstance(rows, list):
        raise ValueError('expected a list')
    return rows


@lru_cache(maxsize=16)
def _index_body_pairs(raw):
    return frozenset(_body_pair(row) for row in _read_rows(raw))


def parent_joint_pairs(arm):
    """Index the stored endpoints without keeping any Blender RNA alive."""
    raw = arm.get('ppf_mmd_joints_v1')
    if isinstance(raw, str):
        try:
            return _index_body_pairs(raw)
        except (KeyError, TypeError, ValueError):
            pass  # Export reports invalid native records with armature context.
    return frozenset()


def parent_joint_pair(bone, pairs):
    """Return the native endpoint pair for this exact hierarchy edge, if any."""
    if not pairs or bone is None or bone.parent is None:
        return None
    child = bone.ppf_bone_physics.body_uuid
    parent = bone.parent.ppf_bone_physics.body_uuid
    if child and parent and child != parent:
        pair = frozenset((child, parent))
        if pair in pairs:
            return pair
    return None


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
            rows = _read_rows(raw)
            disabled_pairs = set()
            pairs = parent_joint_pairs(arm)
            if pairs:
                for bone in arm.data.bones:
                    if not bone.ppf_bone_physics.pmx_parent_joint_enabled:
                        pair = parent_joint_pair(bone, pairs)
                        if pair is not None:
                            disabled_pairs.add(pair)
            for row in rows:
                for endpoint in ('body_a_uuid', 'body_b_uuid'):
                    if owners.get(row[endpoint]) != arm.ppf_armature_uuid:
                        raise ValueError(f"joint {row['joint_uuid']} has an inactive or missing rigid body")
                if row['joint_uuid'] in identities:
                    raise ValueError(f"duplicate joint {row['joint_uuid']}")
                identities.add(row['joint_uuid'])
                if _body_pair(row) not in disabled_pairs:
                    joints.append(row)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'{arm.name}: invalid PMX joint records ({error})') from error
    return sorted(joints, key=lambda row: row['joint_uuid'])
