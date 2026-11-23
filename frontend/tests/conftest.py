import pytest


@pytest.fixture
def joint_test_scene(tmp_path):
    import numpy as np
    from frontend._asset_ import AssetManager
    from frontend._mesh_ import MeshManager
    from frontend._scene_ import Scene

    parent = "22222222-2222-4222-8222-222222222222"
    child = "33333333-3333-4333-8333-333333333333"
    asset = AssetManager()
    mesh = MeshManager(str(tmp_path / "cache"))
    vertices, faces = mesh.box(.25, .25, .25)
    asset.add.tri("joint-box", vertices, faces)
    scene = Scene("joint-fixture", None, asset)
    uv = [np.array([[0., 0.], [1., 0.], [0., 1.]], dtype=np.float32)
          for _ in range(len(faces))]
    scene.add("joint-box", parent).as_pdrd().at(0., 1., 0.).set_uv(uv)
    scene.add("joint-box", child).as_pdrd().at(0., 1.4, 0.).set_uv(uv)
    scene._connected_no_collision_uuid_pairs = [(parent, child)]
    fixed = scene.build(quiet=True)
    fixed.set_pdrd_ball_anchor_payload({"joints": [{
        "joint_uuid": "44444444-4444-4444-8444-444444444444",
        "parent_body_uuid": parent,
        "child_body_uuid": child,
        "parent_local_frame": {"translation_m": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]},
        "child_local_frame": {"translation_m": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]},
        "stiffness_N_per_m": 20.0,
        "enabled": True,
        "collide_connected": False,
    }]})
    session = tmp_path / "session"
    fixed.export_fixed(str(session), delete_exist=False)
    (session / "param.toml").write_text("[param]\nworld_scaling=1.0\n", encoding="utf-8")
    return fixed, session, parent, child
