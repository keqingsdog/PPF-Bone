# File: frontend/_angular_hard_hinge_v5_.py
# License: Apache v2.0
"""G2 joint-envelope V5 authoring validation: the hard parent-local hinge.

Pure, side-effect-free validator for the optional version-5 joint key
``angular_hard_hinge_v1`` frozen by
``coordination/checkpoints/G2/hard-constraints/design-20260927.md``.

The key describes a **hard** relative hinge in the V3
``parent_local_swing_twist_x_v1`` chart: the two swing coordinates are locked
to zero and the twist coordinate is either free or strictly bounded by a
finite signed interval.  It is authored intent only.  This module contains no
solver, native ABI writer, renderer, Blender, or file-writing code, and nothing
here solves, enforces, or proves a hard constraint.

Numeric and domain policy is imported from :mod:`frontend._angular_extensions_`
so the released V4 rules stay byte-identical; only the V5 key rules live here.
Like that module this file does not import :mod:`frontend._scene_`; the caller
passes its stable error raiser so the two modules cannot form an import cycle
and the same ``JOINT_*`` codes reach the Rust boundary.
"""
from __future__ import annotations

import struct

from ._angular_extensions_ import (
    CODE_INVALID,
    CODE_INTERVAL,
    CODE_OUT_OF_DOMAIN,
    COORDINATE_DOMAIN_RAD,
    COORDINATE_MODEL,
    _exact_map,
    _real,
)

HARD_HINGE_KEY = "angular_hard_hinge_v1"
FEATURE_HARD_HINGE = "pdrd_angular_hard_hinge_v1"

# The V5 object names exactly two coordinate groups: the locked swing pair is
# implicit (always zero) and only the twist axis is authorable.
HARD_HINGE_KEYS = ("coordinate_model", "twist_x")
TWIST_MODES = ("free", "bounded")
FREE_TWIST_KEYS = ("mode",)
BOUNDED_TWIST_KEYS = ("mode", "lower_rad", "upper_rad")

# Stable codes.  ``CODE_INVALID`` / ``CODE_INTERVAL`` / ``CODE_OUT_OF_DOMAIN``
# are shared with the V4 module so an existing export receipt stays greppable.
# ``CODE_HARD_CONFLICT`` is new for the V5-only refusal of a hard joint that
# also carries soft-limit or drive channels.  That refusal is a property of the
# whole joint record, so :mod:`frontend._scene_` raises it where both keys are
# visible; this module only owns the code and the single-key shape rules.
CODE_HARD_CONFLICT = "JOINT_ANGULAR_HARD_EXTENSION_CONFLICT"


def validate_hard_hinge_v1(value, joint_uuid, fail):
    """Validate ``angular_hard_hinge_v1`` and return the normalized object.

    The object must be an exact map with exactly ``coordinate_model`` and
    ``twist_x``.  ``twist_x`` is exactly ``{"mode": "free"}`` (no bound keys)
    or exactly ``{"mode": "bounded", "lower_rad": r, "upper_rad": r}``; neither
    mode tolerates the other's keys, so ``{"mode":"free","lower_rad":…}`` cannot
    smuggle an ignored bound.

    Bounded endpoints are finite ``f32``-representable radians with
    ``lower_rad < upper_rad`` strictly after f32 narrowing, and both strictly
    inside the V3 principal chart domain.  A ``free`` twist reports no bound key at all, so a
    reader can never mistake "no bounds" for "bounds at zero".  The author's
    input mapping is never mutated.
    """
    field = HARD_HINGE_KEY
    hinge = _exact_map(value, HARD_HINGE_KEYS, fail, field, joint_uuid)
    if hinge["coordinate_model"] != COORDINATE_MODEL:
        fail(CODE_INVALID, f"{field}.coordinate_model", joint_uuid)
    twist = hinge["twist_x"]
    if not isinstance(twist, dict) or "mode" not in twist:
        fail(CODE_INVALID, f"{field}.twist_x", joint_uuid)
    mode = twist["mode"]
    if mode not in TWIST_MODES:
        fail(CODE_INVALID, f"{field}.twist_x.mode", joint_uuid)
    if mode == "free":
        # Free mode must not carry a bound at all: a present bound would be
        # silently ignored by native ingest, which is exactly the authoring
        # failure this strict shape check prevents.
        _exact_map(twist, FREE_TWIST_KEYS, fail, f"{field}.twist_x", joint_uuid)
        result_twist = {"mode": "free"}
    else:
        bound = _exact_map(twist, BOUNDED_TWIST_KEYS, fail,
                           f"{field}.twist_x", joint_uuid)
        lower = _real(bound["lower_rad"], fail, f"{field}.twist_x.lower_rad",
                      joint_uuid)
        upper = _real(bound["upper_rad"], fail, f"{field}.twist_x.upper_rad",
                      joint_uuid)
        # ``COORDINATE_DOMAIN_RAD`` is already the chart's conservative inner
        # domain (``pi - 1e-3``); its own boundary is not strictly inside, so the
        # comparison is inclusive there, matching the released V4 bound rule.
        if abs(lower) >= COORDINATE_DOMAIN_RAD:
            fail(CODE_OUT_OF_DOMAIN, f"{field}.twist_x.lower_rad", joint_uuid)
        if abs(upper) >= COORDINATE_DOMAIN_RAD:
            fail(CODE_OUT_OF_DOMAIN, f"{field}.twist_x.upper_rad", joint_uuid)
        # The native 24-byte POD carries f32 bounds. Distinct Python floats
        # can become the same endpoint there; reject that interval at export.
        lower_f32 = struct.unpack("<f", struct.pack("<f", lower))[0]
        upper_f32 = struct.unpack("<f", struct.pack("<f", upper))[0]
        if not lower_f32 < upper_f32:
            # A degenerate (equal) interval has no strict two-sided bound, and a
            # reversed interval is not a bound at all; both are refused rather
            # than repaired.
            fail(CODE_INTERVAL, f"{field}.twist_x.lower_rad", joint_uuid)
        result_twist = {"mode": "bounded", "lower_rad": lower, "upper_rad": upper}
    return {"coordinate_model": COORDINATE_MODEL, "twist_x": result_twist}


def required_features_v5(joints, base_features):
    """Exact lexicographically sorted V5 feature set.

    ``base_features`` is the already-sorted V3/V4 feature list for the same
    joints (``_angular_extensions_.required_features``); the hard-hinge feature
    is added only when at least one joint carries the key, and the union is
    re-sorted so ordering is wire-stable.
    """
    features = list(base_features)
    if any(HARD_HINGE_KEY in joint for joint in joints):
        features.append(FEATURE_HARD_HINGE)
    return sorted(features)


__all__ = [
    "CODE_HARD_CONFLICT",
    "COORDINATE_DOMAIN_RAD",
    "COORDINATE_MODEL",
    "FEATURE_HARD_HINGE",
    "HARD_HINGE_KEY",
    "TWIST_MODES",
    "required_features_v5",
    "validate_hard_hinge_v1",
]
