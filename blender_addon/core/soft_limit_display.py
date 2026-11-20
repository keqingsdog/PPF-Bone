"""Display the base angular stiffness from the most recent local Build."""
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path

from .bone_physical_units import ANGULAR_FACTOR, local_build_key


_RECEIPT = "bin/soft_limit_auto_v1.json"
_display_key = None
_resolved = {}


def _read_base_values(root, upload_id):
    import cbor2

    root = Path(root)
    if (root / "upload_id.txt").read_text(encoding="utf-8").strip() != upload_id:
        return {}
    session = root / "session"
    envelope = cbor2.loads((session / "bin/pdrd_scene_context.cbor").read_bytes())
    context = envelope["payload"]
    if envelope["kind"] != "ppf.pdrd_scene_context" or envelope["version"] != 2 or context["complete"] is not True:
        return {}
    records = [item for item in context["inputs"] if item["relative_path"] == _RECEIPT]
    if len(records) != 1:
        return {}
    raw = (session / _RECEIPT).read_bytes()
    # Match the one receipt against the completed Build's existing seal.
    record = records[0]
    if record["byte_length"] != len(raw) or record["sha256"] != sha256(raw).hexdigest():
        return {}
    if (root / "upload_id.txt").read_text(encoding="utf-8").strip() != upload_id:
        return {}
    document = json.loads(raw)
    if (document["kind"] != "ppf.soft_limit_auto" or document["version"] != 1
            or document["policy"] != "AUTO_INERTIA" or document["scene_uuid"] != context["scene_uuid"]):
        return {}
    resolved = defaultdict(dict)
    for row in document["resolved"]:
        target = row["target"]
        channel = target["channel"]
        joint = target["joint_uuid"]
        inertia, frequency = float(row["metric_eff"]), float(row["frequency_hz"])
        if (target["kind"] != "JOINT" or row["metric_unit"] != "kg_m2" or channel in resolved[joint]
                or not math.isfinite(inertia) or not math.isfinite(frequency) or min(inertia, frequency) <= 0
                or channel == "swing_cone" and row.get("inertia_policy") != "mean_of_two_axis_effective_inertias"):
            return {}
        value = inertia * (2 * math.pi * frequency) ** 2 * ANGULAR_FACTOR
        if not math.isfinite(value):
            return {}
        resolved[joint][channel] = (frequency, value)
    return dict(resolved)


def display_values(context, joint_uuid, frequency_hz, channels):
    """Return the Build's 1.0 multiplier values in N·mm/degree for these channels."""
    global _display_key, _resolved
    key = local_build_key(context)
    if key != _display_key:
        _display_key = key
        _resolved = {}
        if key is not None:
            try:
                _resolved = _read_base_values(key[1], key[2])
            except (ImportError, OSError, ValueError, TypeError, KeyError, OverflowError):
                pass
    values = _resolved.get(joint_uuid, {})
    if not channels or set(values) != set(channels):
        return None
    if any(frequency != frequency_hz for frequency, _ in values.values()):
        return None
    return {channel: value for channel, (_, value) in values.items()}
