"""Changing author intent must change the envelope actually exported to disk."""
import copy

import cbor2

from frontend._scene_ import FixedScene


def test_set_payload_reseals_current_contract_in_both_directions(tmp_path):
    (tmp_path / "bin").mkdir()
    (tmp_path / "param.toml").write_text("[param]\nworld_scaling=1.0\n", encoding="utf-8")
    parent = "22222222-2222-4222-8222-222222222222"
    child = "33333333-3333-4333-8333-333333333333"
    base = {"joints": [{
        "joint_uuid": "44444444-4444-4444-8444-444444444444",
        "parent_body_uuid": parent,
        "child_body_uuid": child,
        "parent_local_frame": {"translation_m": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]},
        "child_local_frame": {"translation_m": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]},
        "stiffness_N_per_m": 20.0,
        "enabled": True,
        "collide_connected": False,
    }]}
    controlled = copy.deepcopy(base)
    controlled["joints"][0]["angular_control_v1"] = {
        "coordinate_model": "parent_local_swing_twist_x_v1",
        "stiffness_Nm_per_rad": {"twist_x": 0, "swing_y": 0, "swing_z": 10},
        "damping_Nms_per_rad": {"twist_x": 0, "swing_y": 0, "swing_z": 0},
    }
    fixed = FixedScene.__new__(FixedScene)
    fixed._scene_uuid = "11111111-1111-4111-8111-111111111111"
    fixed._pdrd_body_rows = [0.0] * 46
    fixed._pdrd_body_bindings = [
        {"body_uuid": parent, "runtime_body_index": 0, "vertex_count": 4},
        {"body_uuid": child, "runtime_body_index": 1, "vertex_count": 4},
    ]
    for authored, version in [(base, 2), (controlled, 3), (base, 2)]:
        fixed.set_pdrd_ball_anchor_payload(authored)
        fixed.seal_pdrd_scene_context(str(tmp_path), session_complete=True)
        envelope = cbor2.loads((tmp_path / "bin/pdrd_ball_anchor.cbor").read_bytes())
        assert envelope["version"] == version
        assert envelope["payload"]["joints"] == fixed._pdrd_ball_anchor_payload["joints"]
        if version == 3:
            assert envelope["payload"]["required_features"] == ["pdrd_angular_controls_v1"]
        else:
            assert "required_features" not in envelope["payload"]
