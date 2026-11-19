from __future__ import annotations

import pytest

from frontend._scene_ import (
    FixedScene,
    ValidationError,
    _angular_control_contract,
    _validate_r1_payload,
)


def payload():
    return {"joints": [{
        "joint_uuid": "44444444-4444-4444-4444-444444444444",
        "parent_body_uuid": "22222222-2222-2222-2222-222222222222",
        "child_body_uuid": "33333333-3333-3333-3333-333333333333",
        "parent_local_frame": {"translation_m": [0.0, 0.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
        "child_local_frame": {"translation_m": [0.0, 0.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
        "stiffness_N_per_m": 20.0,
        "enabled": True,
        "collide_connected": True,
    }]}


def directional():
    """Exactly the canonical V3 record the native gate and the case configs use."""
    item = payload()["joints"][0]
    item["angular_control_v1"] = {
        "coordinate_model": "parent_local_swing_twist_x_v1",
        "stiffness_Nm_per_rad": {"twist_x": 0.0, "swing_y": 0.0, "swing_z": 10.0},
        "damping_Nms_per_rad": {"twist_x": 0.0, "swing_y": 0.0, "swing_z": 0.0},
    }
    return {"joints": [item]}


def test_old_payload_gets_zero_angular_terms():
    joint = _validate_r1_payload(payload())["joints"][0]
    assert joint["rest_relative_quaternion_wxyz"] == [1.0, 0.0, 0.0, 0.0]
    assert joint["angle_stiffness_Nm_per_rad"] == 0.0
    assert joint["angle_damping_Nms_per_rad"] == 0.0
    assert joint["max_angle_rad"] == 0.0


def test_angular_coefficients_and_rest_pose_are_preserved():
    item = payload()["joints"][0]
    item.update({
        "rest_relative_quaternion_wxyz": [0.9238795325, 0.0, 0.0, 0.3826834324],
        "angle_stiffness_Nm_per_rad": 46.0,
        "angle_damping_Nms_per_rad": 1.5,
        "max_angle_rad": 0.9,
    })
    joint = _validate_r1_payload({"joints": [item]})["joints"][0]
    assert joint["rest_relative_quaternion_wxyz"] == item["rest_relative_quaternion_wxyz"]
    assert joint["angle_stiffness_Nm_per_rad"] == 46.0
    assert joint["angle_damping_Nms_per_rad"] == 1.5
    assert joint["max_angle_rad"] == 0.9


def test_connected_collision_setting_is_preserved():
    item = payload()["joints"][0]
    item["collide_connected"] = False
    joint = _validate_r1_payload({"joints": [item]})["joints"][0]
    assert joint["collide_connected"] is False


@pytest.mark.parametrize(("field", "value", "code"), [
    ("angle_stiffness_Nm_per_rad", -1.0, "JOINT_ANGULAR_INVALID_STIFFNESS"),
    ("angle_damping_Nms_per_rad", float("inf"), "JOINT_ANGULAR_INVALID_DAMPING"),
    ("max_angle_rad", 3.2, "JOINT_ANGULAR_INVALID_LIMIT"),
    ("rest_relative_quaternion_wxyz", [2.0, 0.0, 0.0, 0.0], "JOINT_ANGULAR_INVALID_REST_ROTATION"),
])
def test_invalid_angular_data_is_refused(field, value, code):
    item = payload()["joints"][0]
    item[field] = value
    with pytest.raises(ValidationError) as exc:
        _validate_r1_payload({"joints": [item]})
    assert exc.value.violations[0]["stable_code"] == code


def test_directional_record_is_preserved_key_for_key():
    joint = _validate_r1_payload(directional())["joints"][0]
    assert joint["angular_control_v1"] == {
        "coordinate_model": "parent_local_swing_twist_x_v1",
        "stiffness_Nm_per_rad": {"twist_x": 0.0, "swing_y": 0.0, "swing_z": 10.0},
        "damping_Nms_per_rad": {"twist_x": 0.0, "swing_y": 0.0, "swing_z": 0.0},
    }
    # The joint-level rest orientation is not discarded by the directional form.
    assert joint["rest_relative_quaternion_wxyz"] == [1.0, 0.0, 0.0, 0.0]


def test_contract_is_v2_without_a_record_and_v3_with_one():
    assert _angular_control_contract(_validate_r1_payload(payload())) == (2, [])
    version, features = _angular_control_contract(_validate_r1_payload(directional()))
    assert version == 3
    assert features == ["pdrd_angular_controls_v1"]


def test_directional_record_conflicts_with_legacy_isotropic_terms():
    item = directional()["joints"][0]
    item["angle_stiffness_Nm_per_rad"] = 1.0
    with pytest.raises(ValidationError) as exc:
        _validate_r1_payload({"joints": [item]})
    assert exc.value.violations[0]["stable_code"] == "JOINT_G2_LEGACY_CONFLICT"


def test_incomplete_soft_limits_are_refused_by_name():
    item = directional()["joints"][0]
    item["angular_soft_limits_v1"] = {"coordinate_model": "parent_local_swing_twist_x_v1"}
    with pytest.raises(ValidationError) as exc:
        _validate_r1_payload({"joints": [item]})
    violations = exc.value.violations[0]
    assert violations["stable_code"] == "JOINT_ANGULAR_EXTENSION_INVALID"
    assert violations["field"] == "angular_soft_limits_v1"


def test_valid_soft_limits_and_position_drive_keep_v4_features():
    item = directional()["joints"][0]
    axis = lambda: {"enabled": True, "lower_rad": -0.5, "upper_rad": 0.5, "stiffness_Nm_per_rad": 1.0}
    item["angular_soft_limits_v1"] = {"coordinate_model": "parent_local_swing_twist_x_v1", **{a: axis() for a in ("twist_x", "swing_y", "swing_z")}}
    item["angular_drives_v1"] = {"coordinate_model": "parent_local_swing_twist_x_v1", "twist_x": {"position": {"enabled": True, "gain_Nm_per_rad": 1.0, "max_effort_Nm": 2.0, "interpolation": "linear_hold", "target_keys": [{"time_s": 0.0, "value_rad": 0.1}]}}}
    result = _validate_r1_payload({"joints": [item]})
    assert result["joints"][0]["angular_drives_v1"] == item["angular_drives_v1"]
    version, features = _angular_control_contract(result)
    assert version == 4
    assert "pdrd_angular_soft_limits_v1" in features and "pdrd_angular_drives_v1" in features


def test_continuous_model_twist_target_and_model_mismatch():
    item = directional()["joints"][0]
    model = "parent_local_swing_twist_x_continuous_v2"
    item["angular_control_v1"]["coordinate_model"] = model
    item["angular_drives_v1"] = {"coordinate_model": model, "twist_x": {"position": {"enabled": True, "gain_Nm_per_rad": 1.0, "max_effort_Nm": 2.0, "interpolation": "linear_hold", "target_keys": [{"time_s": 0.0, "value_rad": 12.566370614359172}]}}}
    result = _validate_r1_payload({"joints": [item]})
    assert result["joints"][0]["angular_drives_v1"]["twist_x"]["position"]["target_keys"][0]["value_rad"] > 12.0
    item["angular_control_v1"]["coordinate_model"] = "parent_local_swing_twist_x_v1"
    with pytest.raises(ValidationError) as exc:
        _validate_r1_payload({"joints": [item]})
    assert exc.value.violations[0]["stable_code"] == "JOINT_ANGULAR_CONTROL_INVALID"
    assert exc.value.violations[0]["field"] == "coordinate_model mismatch"


def test_material_spin_restoring_and_drive_share_v3_chart():
    item=directional()['joints'][0]
    item['angular_control_v1']['coordinate_model']='material_spin_path_v3'
    control=item['angular_control_v1'].copy()
    item['angular_drives_v1']={'coordinate_model':'material_spin_path_v3', 'swing_y': {'position': {'enabled':True,'gain_Nm_per_rad':1.0,'max_effort_Nm':2.0,'interpolation':'linear_hold','target_keys':[{'time_s':0.0,'value_rad':12.0}]}}}
    joint=_validate_r1_payload({'joints':[item]})['joints'][0]
    assert joint['angular_control_v1']==control
    assert joint['angular_drives_v1']==item['angular_drives_v1']
    item['angular_drives_v1']['coordinate_model']='parent_local_swing_twist_x_continuous_v2'
    with pytest.raises(ValidationError):_validate_r1_payload({'joints':[item]})


@pytest.mark.parametrize(("mutation", "code"), [
    ("coordinate_model", "JOINT_ANGULAR_CONTROL_INVALID"),
    ("flat_axes", "JOINT_ANGULAR_CONTROL_INVALID"),
    ("missing_axis", "JOINT_ANGULAR_CONTROL_INVALID"),
    ("extra_axis", "JOINT_ANGULAR_CONTROL_INVALID"),
    ("negative", "JOINT_ANGULAR_CONTROL_INVALID"),
    ("nonfinite", "JOINT_ANGULAR_CONTROL_INVALID"),
    ("overflow_f32", "JOINT_ANGULAR_CONTROL_INVALID"),
    ("underflow_f32", "JOINT_ANGULAR_CONTROL_INVALID"),
    ("bool", "JOINT_ANGULAR_CONTROL_INVALID"),
    ("flat_list", "JOINT_ANGULAR_CONTROL_INVALID"),
])
def test_malformed_directional_records_are_refused(mutation, code):
    item = directional()["joints"][0]
    control = item["angular_control_v1"]
    if mutation == "coordinate_model":
        control["coordinate_model"] = "parent_local_other"
    elif mutation == "flat_axes":
        control["stiffness_Nm_per_rad"] = {"swing_z": 1.0}
    elif mutation == "missing_axis":
        del control["damping_Nms_per_rad"]["swing_y"]
    elif mutation == "extra_axis":
        control["stiffness_Nm_per_rad"]["extra"] = 0.0
    elif mutation == "negative":
        control["stiffness_Nm_per_rad"]["swing_z"] = -1.0
    elif mutation == "nonfinite":
        control["damping_Nms_per_rad"]["twist_x"] = float("nan")
    elif mutation == "overflow_f32":
        control["stiffness_Nm_per_rad"]["swing_z"] = 1e300
    elif mutation == "underflow_f32":
        control["damping_Nms_per_rad"]["swing_y"] = 1e-300
    elif mutation == "bool":
        control["stiffness_Nm_per_rad"]["swing_z"] = True
    elif mutation == "flat_list":
        control["stiffness_Nm_per_rad"] = [0.0, 0.0, 10.0]
    with pytest.raises(ValidationError) as exc:
        _validate_r1_payload({"joints": [item]})
    assert exc.value.violations[0]["stable_code"] == code


def test_v2_export_writes_no_required_features_and_v3_does(tmp_path, monkeypatch):
    """The envelope version and the feature list are decided by the records."""
    import cbor2

    import frontend._cbor_bridge_ as bridge

    def envelope_version_and_features(payload):
        version, features = _angular_control_contract(_validate_r1_payload(payload))
        body = {"required_capability": "pdrd_ball_anchor_v1", "scene_uuid": "1" * 8 + "-1111-1111-1111-111111111111",
                "scene_inputs_sha256": "0" * 64, "bodies": [], "joints": payload["joints"]}
        if version != 2:
            body["required_features"] = features
        blob = bridge.dumps_envelope_with_version(version, "ppf.pdrd_ball_anchor", body)
        envelope = cbor2.loads(blob)
        return envelope["version"], envelope["payload"].get("required_features")

    assert envelope_version_and_features(payload()) == (2, None)
    assert envelope_version_and_features(directional()) == (3, ["pdrd_angular_controls_v1"])

