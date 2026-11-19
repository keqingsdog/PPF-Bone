"""Static disabled-parent bridges before the legacy CPU root producer.

The bridge is a joint endpoint transformation, not a root-motion feature.
Keep the original document for the visible receipt and run the existing
bridge transform on a private copy. Ordinary root v2 admission and the
ordinary joint/native-frame validators remain authoritative.
"""
from copy import deepcopy

import numpy as np

from . import _blender_bones_ as bb

META_KEYS = {
    'version', 'anchor_body_uuid', 'parent_bone_uuid', 'parent_bone_name',
    'parent_world_matrix_blender', 'anchor_proxy_world_matrix_blender',
    'physical_ancestor_bridge_v1',
}
BRIDGE_KEYS = {
    'version', 'ancestor_bone_uuid', 'ancestor_body_uuid',
    'ancestor_bone_name', 'ancestor_to_parent_blender', 'chain',
}
CHAIN_KEYS = {
    'name', 'parent_name', 'matrix_basis', 'matrix_local',
    'inherit_scale', 'use_inherit_rotation',
}


def has_bridges(param_data):
    document = param_data.get('blender_bones_v1')
    return isinstance(document, dict) and any(
        isinstance(row, dict)
        and isinstance(row.get('automatic_parent_anchor_v1'), dict)
        and 'physical_ancestor_bridge_v1' in row['automatic_parent_anchor_v1']
        for row in document.get('bindings', []))


def _name(value, field):
    if not isinstance(value, str) or not value:
        bb.fail(field, 'expected nonempty name')


def _frozen_matrix(value, field):
    # Blender FK accepts frozen basis/rest matrices, not only rigid ones.
    # Do not normalize or alter them: the original FK consumer owns them.
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        bb.fail(field, 'expected finite affine 4x4 matrix')
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1.e-8, rtol=0)):
        bb.fail(field, 'expected finite affine 4x4 matrix')


def validate_bridges(rows):
    """Reject malformed intent before a bridge can consume its metadata."""
    if not isinstance(rows, list):
        bb.fail('bindings', 'expected list')
    by_body, by_bone = {}, {}
    for row in rows:
        if not isinstance(row, dict):
            bb.fail('binding', 'expected object')
        body = bb.canonical_uuid(row.get('body_uuid'), 'body_uuid')
        bone = bb.canonical_uuid(row.get('bone_uuid'), 'bone_uuid')
        if body in by_body or bone in by_bone:
            bb.fail('binding', 'duplicate bone/body UUID')
        by_body[body], by_bone[bone] = row, row
    anchor_intents = {}
    for row in rows:
        meta = row.get('automatic_parent_anchor_v1')
        if not isinstance(meta, dict) or 'physical_ancestor_bridge_v1' not in meta:
            continue  # The unchanged legacy validator owns world anchors.
        if set(meta) != META_KEYS or type(meta['version']) is not int or meta['version'] != 1:
            bb.fail('automatic_parent_anchor_v1', 'invalid bridge metadata')
        anchor = bb.canonical_uuid(meta['anchor_body_uuid'], 'automatic_parent_anchor_v1.anchor_body_uuid')
        parent_bone = bb.canonical_uuid(meta['parent_bone_uuid'], 'automatic_parent_anchor_v1.parent_bone_uuid')
        if anchor in by_body or parent_bone in by_bone:
            bb.fail('automatic_parent_anchor_v1', 'internal endpoint identity collides')
        identity = (row.get('armature_uuid'), meta)
        if anchor in anchor_intents and anchor_intents[anchor] != identity:
            bb.fail('automatic_parent_anchor_v1', 'shared parent samples differ')
        anchor_intents[anchor] = identity
        _name(meta['parent_bone_name'], 'automatic_parent_anchor_v1.parent_bone_name')
        if (row.get('enabled') is not True or row.get('root_mode') != 'FREE'
                or row.get('parent_body_uuid') is not None
                or row.get('parent_bone_uuid') is not None
                or row.get('root_motion') is not None
                or row.get('root_boundary') != {'is_root': True,
                    'nonphysical_parent_bone': meta['parent_bone_name']}):
            bb.fail('automatic_parent_anchor_v1', 'parent/root intent differs')
        parent = bb.matrix(meta['parent_world_matrix_blender'], 'automatic parent pose')
        proxy = bb.matrix(meta['anchor_proxy_world_matrix_blender'], 'automatic parent proxy')
        if not np.allclose(parent, proxy, atol=1.e-8, rtol=0):
            bb.fail('automatic_parent_anchor_v1', 'anchor frame differs from parent pose')
        bridge = meta['physical_ancestor_bridge_v1']
        if (not isinstance(bridge, dict) or set(bridge) != BRIDGE_KEYS
                or type(bridge['version']) is not int or bridge['version'] != 1):
            bb.fail('physical_ancestor_bridge_v1', 'invalid static bridge')
        body = bb.canonical_uuid(bridge['ancestor_body_uuid'], 'bridge.ancestor_body_uuid')
        bone = bb.canonical_uuid(bridge['ancestor_bone_uuid'], 'bridge.ancestor_bone_uuid')
        _name(bridge['ancestor_bone_name'], 'bridge.ancestor_bone_name')
        ancestor = by_body.get(body)
        if (ancestor is None or ancestor['bone_uuid'] != bone
                or ancestor.get('enabled') is not True
                or ancestor.get('bone_name') != bridge['ancestor_bone_name']
                or ancestor.get('armature_uuid') != row.get('armature_uuid')
                or ancestor is row):
            bb.fail('physical_ancestor_bridge_v1', 'physical ancestor differs')
        chain = bridge['chain']
        if not isinstance(chain, list) or not chain:
            bb.fail('physical_ancestor_bridge_v1', 'expected nonempty frozen chain')
        previous, seen = bridge['ancestor_bone_name'], {bridge['ancestor_bone_name']}
        for item in chain:
            if not isinstance(item, dict) or set(item) != CHAIN_KEYS:
                bb.fail('physical_ancestor_bridge_v1.chain', 'invalid frozen chain record')
            _name(item['name'], 'bridge.chain.name')
            if (item['name'] in seen or item['parent_name'] != previous
                    or item['inherit_scale'] != 'FULL'
                    or item['use_inherit_rotation'] is not True):
                bb.fail('physical_ancestor_bridge_v1.chain', 'parent/identity/inheritance differs')
            _frozen_matrix(item['matrix_basis'], 'bridge.chain.matrix_basis')
            _frozen_matrix(item['matrix_local'], 'bridge.chain.matrix_local')
            previous = item['name']
            seen.add(previous)
        if previous != meta['parent_bone_name']:
            bb.fail('physical_ancestor_bridge_v1.chain', 'terminal parent differs')
        # The existing transformer additionally validates both rigid frames
        # and initial ancestor @ relation == parent, then rebinds manual J.


def prepare(scene, param_data, contract):
    from ._automatic_parent_anchor_build_ import _prepare_static_bridges
    from ._automatic_parent_anchor_legacy_ import prepare as legacy_prepare
    from ._bone_connected_collision_ import authored_pairs
    from ._build_contract_ import validate_requests

    prepared = deepcopy(param_data)
    document = prepared['blender_bones_v1']
    validate_bridges(document['bindings'])
    _prepare_static_bridges(document['bindings'])
    # New real edges must pass the unchanged selected-backend gates too.
    validate_requests(contract, prepared)
    prepared = legacy_prepare(scene, prepared)
    # Legacy does not rebuild this table. Preserve explicit cross-link pairs
    # as well as the newly materialized physical ancestor edges.
    scene._connected_no_collision_uuid_pairs = authored_pairs(prepared['blender_bones_v1'])
    return prepared
