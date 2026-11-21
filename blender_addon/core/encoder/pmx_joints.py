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
            pass  # Export reports invalid records when the armature uses MMD.
    return frozenset()


def parent_joint_pair(bone, pairs):
    """Return the stored endpoint pair for this exact hierarchy edge, if any."""
    if not pairs or bone is None or bone.parent is None:
        return None
    child = bone.ppf_bone_physics.body_uuid
    parent = bone.parent.ppf_bone_physics.body_uuid
    if child and parent and child != parent:
        pair = frozenset((child, parent))
        if pair in pairs:
            return pair
    return None


def _mmd_endpoint(bone, enabled_names=None):
    settings = bone.ppf_bone_physics
    enabled = settings.enabled if enabled_names is None else bone.name in enabled_names
    return bool(enabled and settings.proxy
                and getattr(settings, 'joint_control_mode', 'NATIVE') == 'MMD')


def effective_parent_joint_pair(bone, pairs, *, enabled_names=None):
    """Return the PMX-owned hierarchy edge, including a manually disabled edge.

    Disabling the imported connection leaves the edge disconnected until the
    author switches modes or re-enables the connection.
    """
    pair = parent_joint_pair(bone, pairs)
    if pair is not None and _mmd_endpoint(bone, enabled_names) and _mmd_endpoint(bone.parent, enabled_names):
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
        endpoints = {
            bone.ppf_bone_physics.body_uuid: bone
            for bone in arm.data.bones if _mmd_endpoint(bone)
            and owners.get(bone.ppf_bone_physics.body_uuid) == arm.ppf_armature_uuid
        }
        if len(endpoints) < 2:
            continue
        try:
            rows = _read_rows(raw)
            pairs = parent_joint_pairs(arm)
            disabled_pairs = set()
            for bone in endpoints.values():
                if not bone.ppf_bone_physics.pmx_parent_joint_enabled:
                    pair = effective_parent_joint_pair(bone, pairs)
                    if pair is not None:
                        disabled_pairs.add(pair)
            for row in rows:
                pair = _body_pair(row)
                if not pair.issubset(endpoints) or pair in disabled_pairs:
                    continue
                if len(pair) != 2:
                    raise ValueError(f"joint {row['joint_uuid']} requires two distinct bodies")
                if any(endpoints[body].ppf_bone_physics.physical_model != 'RIGID' for body in pair):
                    raise ValueError(f"joint {row['joint_uuid']} requires two rigid proxy bodies")
                if row['joint_uuid'] in identities:
                    raise ValueError(f"duplicate joint {row['joint_uuid']}")
                identities.add(row['joint_uuid'])
                joints.append(row)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'{arm.name}: invalid PMX joint records ({error})') from error
    return sorted(joints, key=lambda row: row['joint_uuid'])
