"""Run with Blender --background --factory-startup --python-exit-code 1 --python."""

import struct
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from blender_addon.core import pc2
from blender_addon.core.pin_reference import encode_pin_pair
from blender_addon.core.uuid_registry import get_or_create_object_uuid
from blender_addon.ui.dynamics.pin_capture_ops import pin_captured_frame_count


def test_pin_cache_frame_count():
    mesh = bpy.data.meshes.new("Pin cache mesh")
    mesh.from_pydata([(0, 0, 0), (0, 1, 0)], [], [])
    obj = bpy.data.objects.new("Pin cache owner", mesh)
    bpy.context.scene.collection.objects.link(obj)
    group = obj.vertex_groups.new(name="Pin A")
    uid = get_or_create_object_uuid(obj)
    pin = SimpleNamespace(
        object_uuid=uid, name=encode_pin_pair(obj.name, group.name), vg_hash=""
    )
    samples = np.arange(24, dtype=np.float32).reshape(4, 2, 3)

    with TemporaryDirectory() as directory, patch.object(pc2, "get_pc2_dir", return_value=directory):
        key = pc2.pin_anim_pc2_key(obj, group.name)
        current_path = Path(pc2.get_pc2_path(key))
        legacy_key = f"{uid}__Pin_A{pc2.PIN_DEFORM_SUFFIX}"
        legacy_path = Path(pc2.get_pc2_path(legacy_key))

        with patch.object(pc2, "_load_pc2_array", side_effect=AssertionError("count loaded samples")):
            assert pin_captured_frame_count(pin) == 0
            pc2.write_pc2(legacy_path, samples[:3])
            assert pin_captured_frame_count(pin) == 3
            ambiguous_group = obj.vertex_groups.new(name="Pin_A")
            assert pin_captured_frame_count(pin) == 0
            obj.vertex_groups.remove(ambiguous_group)
            pc2.write_pc2(current_path, samples)
            assert pin_captured_frame_count(pin) == 4
            assert key not in pc2._pin_anim_cache and legacy_key not in pc2._pin_anim_cache

            valid = current_path.read_bytes()
            truncated = valid[:-1]
            wrong_magic = b"?" + valid[1:]
            wrong_count = valid[:28] + struct.pack("<i", 5) + valid[32:]
            for malformed in (b"short", truncated, wrong_magic, wrong_count):
                current_path.write_bytes(malformed)
                assert pin_captured_frame_count(pin) == 0
            current_path.write_bytes(valid)

        try:
            assert np.array_equal(pc2.get_pin_anim_cache(obj, group.name), samples)
            legacy_path.unlink()
            current_path.unlink()
            assert pin_captured_frame_count(pin) == 4
        finally:
            pc2.unload_pin_anim_cache(key)
        assert pin_captured_frame_count(pin) == 0


if __name__ == "__main__":
    test_pin_cache_frame_count()
    print("Pin cache display count passed")
