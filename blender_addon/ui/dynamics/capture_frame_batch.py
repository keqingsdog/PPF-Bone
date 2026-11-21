"""Opt-in, frame-first sampling for the existing Recapture All phase jobs.

The coordinator still owns start, cancellation, publish, and cleanup. This
module never publishes a partial frame or a cache, never changes the authored
timeline settings, and never disables Blender handlers. Use frame scheduling
only when explicitly requested; evaluation semantics can vary in stateful scenes.
"""

import time

import bpy

from ...core import capture_origin as origin
from ...i18n import pgettext_iface as iface_


def _callback_stamp():
    # Preserve the installed callbacks as-is. A registration/unregistration
    # during a live phase changes evaluation and is an integrity error.
    return tuple(
        tuple(getattr(bpy.app.handlers, name))
        for name in ("frame_change_pre", "frame_change_post",
                     "depsgraph_update_pre", "depsgraph_update_post")
    )


def _same_callbacks(frozen):
    live=_callback_stamp()
    return all(len(now)==len(old) and all(a is b for a,b in zip(now,old))
               for now,old in zip(live,frozen))


def _phase(module):
    from . import pin_capture_ops as pin, static_deform_ops as static

    if module is pin:
        return "entries", True
    if module is static:
        return "objects", False
    raise ValueError("unknown capture phase")


def _world_stamp(scene):
    world = scene.rigidbody_world
    if world is None:
        return None
    return (
        world.as_pointer(), bool(world.enabled),
        world.collection.as_pointer() if world.collection is not None else 0,
        world.constraints.as_pointer() if world.constraints is not None else 0,
    )


def admit(context, module):
    """Start frame scheduling for any normal nonempty phase; never pick legacy."""
    job = module._capture_job
    key, _ = _phase(module)
    entries = job.get(key, ())
    if not job.get("active") or not job.get("token") or not entries:
        raise RuntimeError(iface_("No valid capture work to start"))
    origin.require_origin(job, context)
    targets=origin.sampling_targets(job, entries)
    if not targets:
        raise RuntimeError(iface_("No valid capture work to start"))
    job["_frame_batch"] = (
        job["token"], context.view_layer.as_pointer(), _world_stamp(context.scene),
        _callback_stamp(),
    )
    job["_frame_batch_seeded"] = None
    return True


def _require_supported(context, job, token, layer, world_stamp, callbacks):
    origin.require_capture_token(job, token)
    origin.require_origin(job, context)
    if context.view_layer.as_pointer() != layer:
        raise RuntimeError(iface_("View layer changed during frame batch"))
    if _world_stamp(context.scene) != world_stamp:
        raise RuntimeError(iface_("Rigid body world settings changed during batch capture"))
    if not _same_callbacks(callbacks):
        raise RuntimeError(iface_("Frame or depsgraph callbacks changed during frame batch"))


def _pending_frame(entries):
    return min((row["frame_start"] + row["frames_done"] for row in entries
                if row["frames_done"] < row["frames"].shape[0]), default=None)


def tick(context, module, *, budget_ms=40):
    """Advance an admitted phase; same return contract as the legacy primitives."""
    job = module._capture_job
    key, is_pin = _phase(module)
    if job.get("aborted"):
        return False, True, str(job.get("error", ""))
    token, layer, world_stamp, callbacks = job["_frame_batch"]
    _require_supported(context, job, token, layer, world_stamp, callbacks)
    entries = job[key]
    scene = context.scene
    if is_pin:
        module._require_capture_range_settings(scene, entries)
    else:
        origin.require_capture_ranges(scene, entries)
    deadline = time.perf_counter() + budget_ms / 1000
    with origin.sampling_cache_scope(origin.sampling_targets(job, entries)):
        while time.perf_counter() < deadline:
            if job.get("aborted"):
                return False, True, str(job.get("error", ""))
            frame = _pending_frame(entries)
            if frame is None:
                break
            # Only an integer timeline stamp survives a timer yield, never an
            # evaluated object/depsgraph. Reacquire the current graph below.
            seeded = job["_frame_batch_seeded"]
            if seeded is not None and (scene.frame_current != seeded or scene.frame_subframe):
                raise RuntimeError(iface_("Capture sampling frame changed; capture cancelled"))
            if seeded != frame:
                scene.frame_set(frame)
                _require_supported(context, job, token, layer, world_stamp, callbacks)
                if scene.frame_current != frame or scene.frame_subframe:
                    raise RuntimeError(iface_("Capture sampling frame changed; capture cancelled"))
                job["_frame_batch_seeded"] = frame
            with origin.sampling_cache_scope(origin.sampling_targets(job, entries)):
                depsgraph = context.evaluated_depsgraph_get()
                _require_supported(context, job, token, layer, world_stamp, callbacks)
                if scene.frame_current != frame or scene.frame_subframe:
                    raise RuntimeError(iface_("Capture sampling frame changed; capture cancelled"))
                # Keep Blender's same-frame source and guard the entire pending
                # queue. Check deadline before each row, not after an entire frame.
                for row in entries:
                    if (row["frames_done"] >= row["frames"].shape[0]
                            or row["frame_start"] + row["frames_done"] != frame):
                        continue
                    if job.get("aborted"):
                        return False, True, str(job.get("error", ""))
                    origin.sampling_targets(job, entries)
                    if is_pin:
                        pins = module._resolve_capture_pins(scene, entries)
                        module._require_capture_range_settings(scene, entries, pins)
                        ok, why = module._process_one_frame(scene, depsgraph, row, frame)
                    else:
                        origin.require_capture_ranges(scene, entries)
                        ok, why = module._process_one_frame(context, depsgraph, row, frame)
                    _require_supported(context, job, token, layer, world_stamp, callbacks)
                    if scene.frame_current != frame or scene.frame_subframe:
                        raise RuntimeError(iface_("Capture sampling frame changed; capture cancelled"))
                    if not ok:
                        job["aborted"] = True
                        job["error"] = why
                        return False, True, why
                    job["total_frames_processed"] += 1
                    if is_pin:
                        status = iface_("Capturing pin '{pin}' on '{object}' [{done}/{total}] ({source})")
                        job["status"] = status.format(
                            pin=row["vg_name"], object=row["obj_name"], done=row["frames_done"],
                            total=row["frames"].shape[0], source=row["range_source"],
                        )
                    else:
                        job["status"] = iface_("Capturing {name} [{done}/{total}] ({source})").format(
                            name=row["obj_name"], done=row["frames_done"],
                            total=row["frames"].shape[0], source=row["range_source"],
                        )
                    if time.perf_counter() >= deadline:
                        break
    if is_pin:
        module._require_capture_range_settings(scene, entries)
    else:
        origin.require_capture_ranges(scene, entries)
    if job.get("aborted"):
        return False, True, str(job.get("error", ""))
    return _pending_frame(entries) is not None, False, ""
