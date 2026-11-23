"""Latest official force-field inputs must participate in joint context sealing."""
import cbor2
def test_force_field_script_and_manifest_change_the_sealed_context(joint_test_scene):
    fixed, tmp_path, _parent, _child = joint_test_scene
    fixed.seal_pdrd_scene_context(str(tmp_path), session_complete=True)
    baseline = cbor2.loads((tmp_path / "bin/pdrd_scene_context.cbor").read_bytes())["payload"]
    field = tmp_path / "bin/force_field"
    field.mkdir(parents=True)
    manifest = field / "force_field.toml"
    script = field / "script-0.py"
    manifest.write_text("strength=1.0\n")
    script.write_text("return 1.0\n")
    def seal():
        fixed.seal_pdrd_scene_context(str(tmp_path), session_complete=True)
        return cbor2.loads((tmp_path / "bin/pdrd_scene_context.cbor").read_bytes())["payload"]

    first = seal()
    assert first["complete"]
    assert {p["relative_path"] for p in first["inputs"]} == {
        p["relative_path"] for p in baseline["inputs"]} | {
        "bin/force_field/force_field.toml", "bin/force_field/script-0.py"}
    script.write_text("return 2.0\n")
    second = seal()
    assert second["scene_inputs_sha256"] != first["scene_inputs_sha256"]
    manifest.write_text("strength=2.0\n")
    third = seal()
    assert third["scene_inputs_sha256"] != second["scene_inputs_sha256"]
    (tmp_path / "output").mkdir()
    (tmp_path / "output/live-frame.bin").write_bytes(b"mutable output")
    assert seal()["scene_inputs_sha256"] == third["scene_inputs_sha256"]
