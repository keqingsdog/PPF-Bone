"""Private shared-frame coordinator; samples only, does not publish."""
import time
import numpy as np

from ...core import capture_origin as origin
from ...core.uuid_registry import get_object_by_uuid
from . import capture_frame_batch as frame, pin_capture_ops as pin, static_deform_ops as static


class JointCapture:
    _combined_capture_owner = True
    _capture_cancelled = False

    def __init__(self, context, static_objects, pin_specs):
        self.context = context
        self.scene = context.scene
        self.saved_frame = int(self.scene.frame_current)
        self.layer = context.view_layer.as_pointer()
        self.world = frame._world_stamp(self.scene)
        self.callbacks = frame._callback_stamp()
        self._capture_sampling_targets = None
        self.jobs = []
        self.closed = self.restored = self.complete = False
        self.seeded = None
        self.frame_sets = 0
        if static.is_capture_running() or pin.is_pin_capture_running():
            raise RuntimeError("Another deformation capture is in progress")
        errors = []
        static_rows = static._build_entries(self.scene, static_objects, errors)
        pins = [pin._build_entry(self.scene, *spec, errors) for spec in pin_specs]
        if errors or len(static_rows) != len(static_objects) or any(row is None for row in pins):
            raise RuntimeError(errors[0] if errors else "A queued deformation target is no longer valid")
        self._capture_sampling_targets = origin.freeze_sampling_targets(
            self.scene, [get_object_by_uuid(row['obj_uuid']) for row in static_rows + pins])
        if not self._capture_sampling_targets:
            raise RuntimeError("No valid capture work to start")
        try:
            for kind, module, rows in (("STATIC", static, static_rows), ("PIN", pin, pins)):
                if not rows:
                    continue
                old_token = module._capture_job.get('token')
                try:
                    ok, why = module._start_job(context, rows, _joint_owner=self)
                finally:
                    job = module._capture_job
                    if job.get('owner') is self and job.get('token') is not old_token:
                        self.jobs.append((kind, module, job, job['token'], rows))
                if not ok:
                    raise RuntimeError(why)
            self._guard()
        except BaseException:
            self.close()
            raise

    def _guard(self):
        if self.closed or self._capture_cancelled:
            raise RuntimeError("Combined deformation capture cancelled")
        for _, _, job, token, rows in self.jobs:
            if job.get('aborted'):
                raise RuntimeError(job.get('error') or "Combined deformation capture cancelled")
            frame._require_supported(self.context, job, token, self.layer, self.world, self.callbacks)
        if self.jobs:
            origin.sampling_targets(self.jobs[0][2], ())
        if self.restored:
            if self.jobs and (self.scene.frame_current != self.saved_frame
                              or self.scene.frame_subframe != self.jobs[0][2]['saved_subframe']):
                raise RuntimeError("Capture frame changed after restoring author state")
        elif self.seeded is not None and (
                self.scene.frame_current != self.seeded or self.scene.frame_subframe):
            raise RuntimeError("Capture sampling frame changed; capture cancelled")

    def _ranges(self):
        for kind, mod, _, _, rows in self.jobs:
            if kind == 'STATIC':
                origin.require_capture_ranges(self.scene, rows)
            else:
                mod._require_capture_range_settings(self.scene, rows)

    def _pending(self):
        return min((f for _, _, _, _, rows in self.jobs
                    if (f := frame._pending_frame(rows)) is not None), default=None)

    def tick(self, context=None, *, budget_ms=40):
        """Sample union frames; return whether more samples remain."""
        if context is not None:
            self.context = context
        if self.restored:
            raise RuntimeError("Combined capture frame was already restored")
        try:
            return self._tick(budget_ms)
        except BaseException:
            self.close()
            raise

    def _tick(self, budget_ms):
        if budget_ms <= 0:
            raise ValueError("Capture budget must be positive")
        self._guard()
        self._ranges()
        deadline = time.perf_counter() + budget_ms / 1000
        with origin.sampling_cache_scope(self._capture_sampling_targets):
            # Always process at least one row, even if entering the visibility scope
            # consumes the entire nominal time slice.
            while True:
                f = self._pending()
                if f is None:
                    break
                if self.seeded != f:
                    self.scene.frame_set(f)
                    self.frame_sets += 1
                    self.seeded = f
                    self._guard()
                # Same hide-after-frame-set rule as the standalone production path.
                with origin.sampling_cache_scope(self._capture_sampling_targets):
                    graph = self.context.evaluated_depsgraph_get()
                    self._guard()
                    expired = False
                    for kind, mod, job, _, rows in self.jobs:
                        for row in rows:
                            if row['frames_done'] >= len(row['frames']) or row['frame_start'] + row['frames_done'] != f:
                                continue
                            if kind == 'PIN':
                                pins = mod._resolve_capture_pins(self.scene, rows)
                                mod._require_capture_range_settings(self.scene, rows, pins)
                                ok, why = mod._process_one_frame(self.scene, graph, row, f)
                            else:
                                origin.require_capture_ranges(self.scene, rows)
                                ok, why = mod._process_one_frame(self.context, graph, row, f)
                            self._guard()
                            if not ok:
                                raise RuntimeError(why or "Cannot sample deformation")
                            job['total_frames_processed'] += 1
                            job['status'] = f"{kind} | {row['obj_name']} | frame {f}"
                            if time.perf_counter() >= deadline:
                                expired = True
                                break
                        if expired:
                            break
                if expired or time.perf_counter() >= deadline:
                    break
        self._guard()
        self._ranges()
        self.complete = self._pending() is None
        return not self.complete

    def _restore_once(self):
        if not self.restored:
            self.restored = True
            if self.jobs:
                # An unrelated job may have reopened or replaced the scene.
                job, token = self.jobs[0][2:4]
                if origin.owns_capture_job(job, token):
                    origin.restore_frame(job)

    def tasks(self, context=None):
        """Return live writer inputs; also usable as prepublication recheck.

        The visible frame is restored before returning any handles. Restoration
        can evaluate callbacks, so revalidate tokens, world, ranges, and Pin
        operation/reference constraints *after* restoration, never after a write.
        Call synchronously, with no UI turn between this and the transaction.
        """
        if context is not None:
            self.context = context
        if not self.complete:
            raise RuntimeError("Combined capture is incomplete")
        self._guard()
        self._restore_once()
        self._guard()
        self._ranges()
        tasks = []
        for kind, mod, _, _, rows in self.jobs:
            if kind == 'PIN':
                # The hot range guard only checks explicit settings. Full
                # provenance validation is mandatory before publication.
                origin.require_capture_ranges(self.scene, rows)
                pins = mod._resolve_capture_pins(self.scene, rows)
            else:
                pins = [None] * len(rows)
            for row, item in zip(rows, pins):
                obj = get_object_by_uuid(row['obj_uuid'])
                if obj is None or row['frames_done'] != len(row['frames']) or not np.isfinite(row['frames']).all():
                    raise RuntimeError("Combined capture target has incomplete or invalid samples")
                if item is not None and (any(op.op_type == 'TORQUE' for op in item.operations)
                                         or mod._pin_has_vertex_co_fcurves(item)):
                    raise RuntimeError("Pin author motion changed during capture")
                tasks.append((kind, obj, row, item))
        self._guard()
        return tasks

    def abort_for_save(self):
        # Called through either native job while Blender saves the project.
        # Never save a partial capture or let the next timer resume.
        self._capture_cancelled = True
        for _, _, job, token, _ in self.jobs:
            if origin.owns_capture_job(job, token):
                job['aborted'] = True
                job['error'] = 'Capture cancelled before saving the project'
        self._restore_once()

    def close(self):
        if self.closed:
            return
        try:
            self._restore_once()
        finally:
            for _, mod, job, token, _ in self.jobs:
                if origin.owns_capture_job(job, token):
                    mod._reset_job()
            self.closed = True
            self._capture_sampling_targets = None
