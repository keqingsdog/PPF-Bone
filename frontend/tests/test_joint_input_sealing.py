"""Latest official force-field inputs must participate in joint context sealing."""
import cbor2
from frontend._scene_ import FixedScene


def test_force_field_script_and_manifest_change_the_sealed_context(tmp_path):
    field = tmp_path / "bin/force_field"
    field.mkdir(parents=True)
    (tmp_path / "param.toml").write_text("[param]\nworld_scaling=1.0\n")
    manifest = field / "force_field.toml"
    script = field / "script-0.py"
    manifest.write_text("strength=1.0\n")
    script.write_text("return 1.0\n")
    fixed = FixedScene.__new__(FixedScene)
    fixed._pdrd_body_bindings = []
    fixed._pdrd_body_rows = []
    fixed._scene_uuid = "11111111-1111-4111-8111-111111111111"
    fixed._pdrd_ball_anchor_payload = {"joints": []}

    def seal():
        fixed.seal_pdrd_scene_context(str(tmp_path), session_complete=True)
        return cbor2.loads((tmp_path / "bin/pdrd_scene_context.cbor").read_bytes())["payload"]

    first = seal()
    assert first["complete"]
    assert {p["relative_path"] for p in first["inputs"]} == {
        "param.toml", "bin/force_field/force_field.toml", "bin/force_field/script-0.py"}
    script.write_text("return 2.0\n")
    second = seal()
    assert second["scene_inputs_sha256"] != first["scene_inputs_sha256"]
    manifest.write_text("strength=2.0\n")
    third = seal()
    assert third["scene_inputs_sha256"] != second["scene_inputs_sha256"]
    (tmp_path / "output").mkdir()
    (tmp_path / "output/live-frame.bin").write_bytes(b"mutable output")
    assert seal()["scene_inputs_sha256"] == third["scene_inputs_sha256"]
