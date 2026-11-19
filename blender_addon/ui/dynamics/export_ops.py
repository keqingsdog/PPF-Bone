# Fork modification: changed from the upstream baseline.
# File: ui/dynamics/export_ops.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
"""Export the simulated mesh sequence to USD / Alembic caches.

A lightweight alternative to the shape-key bake (``solver.bake_all_animation``).
The bake freezes every fetched PC2 frame into a per-frame shape key, which is
heavy to manage on the timeline and slow for Blender to play back. Instead,
these operators drive Blender's built-in USD / Alembic exporters over the
simulated frame range and let them sample the ``ContactSolverCache`` MESH_CACHE
deformed mesh at each frame. The result is a standard point cache that Blender
and other DCC tools can play back and retime efficiently.

Design notes (verified against Blender 5.1.2, live):
  * The MESH_CACHE modifier is set up with ``frame_start`` at the solve's
    starting frame. Export reads that installed cache origin, never the
    pending settings for a future solve, so PC2 frame
    ``f`` shows at scene frame ``f + start``; ``n`` cached frames occupy scene
    frames ``[start, start + n - 1]``. Exporting that inclusive range samples the deform at
    each PC2 frame (USD stores sparse time samples, so a run of identical poses,
    e.g. a static tail, is written once, spanning the full range via the stage's
    start/end time codes).
  * ``evaluation_mode='VIEWPORT'`` reproduces what the user sees scrubbing the
    timeline. Under VIEWPORT eval the deform is only captured when the cache
    modifier's ``show_viewport`` is on and the object is not hidden, so we force
    both (saved/restored) for the exact target set.
  * ``bpy.ops.wm.usd_export('EXEC_DEFAULT', ...)`` runs synchronously (verified:
    the file is complete on return), so the finally-block state restore is safe.
    Alembic keeps ``as_background_job=False`` for the same reason; do NOT flip it
    on, or the restore would race the still-running export and corrupt the tail.
  * Rods (CURVE objects) deform through a frame-change handler rather than a
    MESH_CACHE modifier, and the built-in exporters do not fire it reliably, so
    they are excluded and reported as a WARNING rather than silently dropped.
"""

from __future__ import annotations

from ... import i18n as _ppf_i18n
import os
import struct
import contextlib
import tempfile
import math

import bpy  # pyright: ignore
from ...i18n.props import StringProperty  # pyright: ignore
from bpy.types import Operator  # pyright: ignore
from bpy_extras.io_utils import ExportHelper  # pyright: ignore
from ...i18n import pgettext_iface as iface_, pgettext_tip as tip_

from ...core.client import communicator as com
from ...core.derived import is_server_busy_from_response as is_running
from ...core.pc2 import (
    MODIFIER_NAME,
    get_pc2_path,
    has_mesh_cache,
    object_pc2_key_readonly,
    read_pc2_frame_count,
)
from .bake_ops import _has_unfetched_frames, is_bake_running
from .pin_capture_ops import is_pin_capture_running
from .static_deform_ops import is_capture_running


# ---------------------------------------------------------------------------
# Export-set discovery
# ---------------------------------------------------------------------------


def _busy(context) -> bool:
    """True when another solver activity must lock out the export (draw-safe,
    disk-free).

    Mirrors the cache-mutating modal ops (bake / capture / pin capture) so an
    export cannot race a cache rewrite, and refuses while the server is actively
    simulating (the on-disk cache is still growing). The finer "frames were
    produced but not fetched yet" case is caught by ``_has_unfetched_frames`` in
    ``_preflight``. We deliberately do NOT gate on ``com.busy()``: it stays set
    through transient upload / apply phases and is not reliably cleared outside
    the modal loop, which would grey the button right after a run + fetch,
    exactly when the user wants to export.
    """
    if is_bake_running() or is_capture_running() or is_pin_capture_running():
        return True
    return is_running(com.info.response)


def _has_exportable_mesh(context) -> bool:
    """Cheap, disk-free predicate for ``poll``: is there at least one MESH in the
    view layer carrying a live ContactSolverCache modifier? The exact
    (disk-touching) check lives in ``execute`` / ``_exportable_meshes``."""
    for obj in context.view_layer.objects:
        if obj.type == "MESH" and obj.modifiers.get(MODIFIER_NAME) is not None:
            return True
    return False


def _exportable_meshes(context):
    """``[(obj, n_frames), ...]`` for every MESH in the view layer that has a
    live ContactSolverCache modifier AND a readable PC2 reporting > 0 frames.

    Keyed on the live modifier (not ``has_mesh_cache``, which also returns True
    for a disk-only PC2 with no deformer, or for a modifier whose file has since
    been deleted): the modifier is what actually moves the vertices the exporter
    samples. Invalid or unreadable paths reject the entire set. Does disk I/O; never call from
    ``poll``.
    """
    out = []
    for obj in context.view_layer.objects:
        if obj.type != "MESH":
            continue
        mod = obj.modifiers.get(MODIFIER_NAME)
        if mod is None:
            continue
        if mod.type != "MESH_CACHE" or mod.cache_format != "PC2":
            raise ValueError(iface_("The solver export requires a PC2 Mesh Cache modifier: {name}").format(name=obj.name))
        # Read the file the exporter will actually sample (the modifier's own
        # filepath), not the recomputed canonical UUID path: after a project
        # rename or a data/ folder copy the two can diverge, and the modifier
        # path is what drives the deform. A canonical UUID file cannot repair
        # an empty or broken live modifier path by merely being read here.
        path = bpy.path.abspath(mod.filepath) if mod.filepath else ""
        if not path or not os.path.isfile(path):
            raise ValueError("Export cache path is missing or invalid for {}: {}".format(obj.name, mod.filepath or "<empty>"))
        try:
            n_frames = read_pc2_frame_count(path)
        except (OSError, struct.error) as exc:
            raise ValueError("Export cache cannot be read for {}: {}".format(obj.name, path)) from exc
        if n_frames <= 0:
            raise ValueError("Export cache has no frames for {}: {}".format(obj.name, path))
        out.append((obj, n_frames))
    return out


def _cache_frame_range(scene, pairs):
    """Only export the installed PPF frame-based, unretimed cache layout.

    Different origins cannot share a single sequence range with the existing
    particle exporter. Refuse rather than shifting one object's sample indices.
    """
    origins = set()
    for obj, _ in pairs:
        mod = obj.modifiers[MODIFIER_NAME]
        start = float(mod.frame_start)
        if (mod.play_mode != "SCENE" or mod.time_mode != "FRAME"
                or float(mod.frame_scale) != 1.0
                or not math.isfinite(start) or not start.is_integer()):
            raise ValueError(iface_("Restore the original frame timing before exporting the solver cache: {name}").format(name=obj.name))
        origins.add(int(start))
    if len(origins) != 1:
        raise ValueError(iface_("Solver cache start frames differ. Export matching caches from one simulation at a time."))
    start = next(iter(origins))
    end = start + max(n for _, n in pairs) - 1
    if (start < scene.bl_rna.properties["frame_start"].hard_min
            or end > scene.bl_rna.properties["frame_end"].hard_max):
        raise ValueError(iface_("The simulation cache range cannot be represented on the Blender timeline."))
    return start, end


def _excluded_sim_curves(context):
    """Names of CURVE (rod) objects in the view layer that carry a solver cache.

    Rods deform via a frame-change handler the built-in exporters do not fire,
    so they cannot be exported faithfully through this path; we list them so the
    user is warned rather than silently losing them.
    """
    names = []
    for obj in context.view_layer.objects:
        if obj.type == "CURVE" and has_mesh_cache(obj):
            names.append(obj.name)
    return names


# ---------------------------------------------------------------------------
# Shared operator scaffolding
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _particle_export_meshes(context, pairs, start, end):
    """Export Sand as temporary low-poly spheres driven by its real PC2.

    Point-cloud GN is display-only and older built-in exporters otherwise write
    an empty mesh. Only temporary objects, geometry and PC2 are created here;
    source mesh, nodes, radius and simulation samples remain untouched.
    """
    import bmesh
    import numpy as np
    from ...core.pc2 import (read_pc2_n_verts, read_pc2_frame,
                             create_pc2_file, append_pc2_frame,
                             setup_mesh_cache_modifier)
    made = []
    meshes = []
    work_owner = tempfile.TemporaryDirectory(prefix="ppf_particle_export_")
    work = work_owner.name
    try:
        sphere = bmesh.new()
        try:
            bmesh.ops.create_icosphere(sphere, subdivisions=1, radius=1.)
            sphere.verts.ensure_lookup_table()
            unit = np.asarray([tuple(v.co) for v in sphere.verts], dtype=np.float64)
            sphere.verts.index_update()
            triangles = [tuple(v.index for v in face.verts) for face in sphere.faces]
        finally:
            sphere.free()
        for source, frame_count in pairs:
            if not source.get("particle_mesh") or not source.visible_get():
                continue
            radius = float(source.get("grain_radius", 0.))
            if not math.isfinite(radius) or radius <= 0:
                raise ValueError("Particle export requires a finite positive grain radius: " + source.name)
            mod = source.modifiers.get(MODIFIER_NAME)
            path = bpy.path.abspath(mod.filepath) if mod and mod.filepath else get_pc2_path(object_pc2_key_readonly(source))
            count = read_pc2_n_verts(path)
            if count != len(source.data.vertices) or count <= 0:
                raise ValueError("Particle export cache count does not match original particle indices: " + source.name)
            n_unit = len(unit)
            faces = [tuple(i*n_unit+j for j in face) for i in range(count) for face in triangles]
            cache = os.path.join(work, str(len(made))+".pc2")
            create_pc2_file(cache, count*n_unit)
            first = None
            for frame in range(start, end+1):
                context.scene.frame_set(frame)
                context.view_layer.update()
                # The source transform maps PC2-local centers into world;
                # inverse vectors retain the stamped WORLD-space grain radius
                # even when its object has non-uniform scale.
                inv = np.asarray(source.matrix_world.to_3x3().inverted(), dtype=np.float64)
                offsets = (unit*radius) @ inv.T
                centers = read_pc2_frame(path, min(frame-start, frame_count-1), count)
                points = (centers[:,None,:]+offsets[None,:,:]).reshape(-1,3).astype(np.float32)
                if not np.isfinite(points).all():
                    raise ValueError("Particle export contains non-finite real cache coordinates: " + source.name)
                if first is None:
                    first = points
                append_pc2_frame(cache, points, count*n_unit)
            mesh = bpy.data.meshes.new(source.name+"_ExportParticles")
            meshes.append(mesh)
            mesh.from_pydata(first.tolist(), [], faces)
            mesh.update()
            # The original GN's Set Material is the authoring source of truth.
            materials = list(source.data.materials)
            display = source.modifiers.get("Particle Mesh")
            if display and display.type == 'NODES' and display.node_group:
                for node in display.node_group.nodes:
                    if node.bl_idname == 'GeometryNodeSetMaterial':
                        socket = node.inputs.get('Material')
                        material = socket.default_value if socket else None
                        if material:
                            # Set Material overrides every incoming face/point;
                            # stale source mesh slots must not take precedence.
                            materials = [material]
            for material in materials:
                mesh.materials.append(material)
            obj = bpy.data.objects.new(source.name+"_Particles", mesh)
            made.append(obj)
            context.scene.collection.objects.link(obj)
            # Preserve the source object hierarchy and sampled transform.
            # A selected child exports its unselected parent as a transform,
            # without exporting the parent's empty point-cloud mesh.
            obj.parent = source
            setup_mesh_cache_modifier(obj, cache, frame_start=float(start))
            source.select_set(False)
            obj.select_set(True)
        yield tuple(made)
    finally:
        actions = []
        for obj in made:
            if obj.name in bpy.data.objects:
                animation = obj.animation_data
                action = animation.action if animation else None
                if action is not None and action not in actions:
                    actions.append(action)
                bpy.data.objects.remove(obj, do_unlink=True)
        # setup_mesh_cache_modifier can key visibility for a non-first-frame
        # start. These actions belong only to our new temporary objects.
        for action in actions:
            if action.users == 0:
                bpy.data.actions.remove(action)
        for mesh in meshes:
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        work_owner.cleanup()


def preflight_error(context):
    """Return a user-facing error string when export can't proceed, else None.

    Mirrors the Bake operators' guards so a truncated or malformed cache is
    never written with a success toast. Module level rather than a method
    because the MCP export tools run the same checks before dispatching, and
    bpy.ops hands a caller only a CANCELLED status, never the reason.
    """
    if _busy(context):
        return iface_("Another solver activity is in progress")
    if context.mode != "OBJECT":
        return iface_("Exit Edit/Sculpt mode before exporting")
    if _has_unfetched_frames(context.scene):
        return iface_(
            "Unfetched animation frames exist. "
            "Fetch all animation frames first."
        )
    try:
        pairs = _exportable_meshes(context)
    except ValueError as exc:
        return str(exc)
    if not pairs:
        return iface_("No simulated mesh sequence to export")
    try:
        _cache_frame_range(context.scene, pairs)
    except ValueError as exc:
        return str(exc)
    return None


class _ExportSimCacheBase(ExportHelper):
    """Select / frame-range / visibility save-restore shared by both exporters.

    Subclasses set ``filename_ext`` + ``filter_glob`` and implement
    ``_run_exporter(context, start, end)`` returning the exporter's result set.
    """

    @classmethod
    def poll(cls, context):
        if _busy(context):
            return False
        return _has_exportable_mesh(context)

    def _preflight(self, context):
        return preflight_error(context)

    def invoke(self, context, event):
        # Fail fast before opening the file browser so the user does not pick a
        # filename only to have the export refuse.
        err = self._preflight(context)
        if err is not None:
            self.report({"ERROR"}, err)
            return {"CANCELLED"}
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        err = self._preflight(context)
        if err is not None:
            self.report({"ERROR"}, err)
            return {"CANCELLED"}

        try:
            pairs = _exportable_meshes(context)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        if not pairs:
            self.report({"ERROR"}, iface_("No simulated mesh sequence to export"))
            return {"CANCELLED"}
        try:
            export_start, export_end = _cache_frame_range(context.scene, pairs)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        objs = [obj for obj, _ in pairs]
        counts = [n for _, n in pairs]
        excluded_curves = _excluded_sim_curves(context)

        scene = context.scene
        vl = context.view_layer
        window = context.window

        # --- snapshot every piece of state we are about to touch ---
        saved_start = scene.frame_start
        saved_end = scene.frame_end
        saved_current = scene.frame_current
        saved_step = scene.frame_step
        saved_sel = [(obj, obj.select_get()) for obj in vl.objects]
        saved_active = vl.objects.active
        saved_vis = {}
        for obj in objs:
            mod = obj.modifiers.get(MODIFIER_NAME)
            saved_vis[obj] = (
                obj.hide_viewport,
                obj.hide_get(),
                mod.show_viewport if mod is not None else True,
            )

        if window is not None:
            window.cursor_set("WAIT")

        result = {"CANCELLED"}
        export_error = None
        invisible = []
        try:
            for obj in vl.objects:
                try:
                    obj.select_set(False)
                except RuntimeError:
                    pass
            for obj in objs:
                try:
                    obj.hide_viewport = False
                    obj.hide_set(False)
                    mod = obj.modifiers.get(MODIFIER_NAME)
                    if mod is not None:
                        mod.show_viewport = True
                    obj.select_set(True)
                except RuntimeError:
                    pass
            try:
                vl.objects.active = objs[0]
            except (RuntimeError, ReferenceError):
                pass
            # Under VIEWPORT eval a target still hidden by its collection
            # ("Disable in Viewports", which object-level flags can't override)
            # is dropped by the exporter. Forcing object flags above handles the
            # common case; flag whatever remains invisible so the omission is
            # reported rather than silent.
            invisible = [obj.name for obj in objs if not obj.visible_get()]
            scene.frame_start = export_start
            scene.frame_end = export_end
            scene.frame_step = 1
            try:
                with _particle_export_meshes(context, pairs, export_start, export_end):
                    result = self._run_exporter(context, export_start, export_end)
            except Exception as exc:  # noqa: BLE001 - surfaced as a clean report
                export_error = str(exc)
        finally:
            if window is not None:
                window.cursor_set("DEFAULT")
            scene.frame_start = saved_start
            scene.frame_end = saved_end
            scene.frame_step = saved_step
            # Removing temporary particle objects invalidates cached view-layer bases.
            vl.update()
            # Restore selection while the targets are still visible, then
            # re-apply their original hide flags.
            for obj in vl.objects:
                if obj is None:
                    continue
                try:
                    obj.select_set(False)
                except RuntimeError:
                    pass
            for obj, sel in saved_sel:
                try:
                    obj.select_set(sel)
                except (RuntimeError, ReferenceError):
                    pass
            try:
                vl.objects.active = saved_active
            except (RuntimeError, ReferenceError):
                pass
            for obj, (hide_vp, hide_set, show_vp) in saved_vis.items():
                try:
                    obj.hide_viewport = hide_vp
                    obj.hide_set(hide_set)
                    mod = obj.modifiers.get(MODIFIER_NAME)
                    if mod is not None:
                        mod.show_viewport = show_vp
                except (RuntimeError, ReferenceError):
                    pass
            scene.frame_set(saved_current)

        if export_error is not None:
            self.report({"ERROR"}, iface_("Export failed: {error}").format(error=export_error))
            return {"CANCELLED"}
        if "FINISHED" not in result:
            self.report({"ERROR"}, iface_("Export wrote nothing"))
            return {"CANCELLED"}

        n_exported = len(objs) - len(invisible)
        msg = iface_("Exported {count} mesh(es), frames {start}-{end} to {path}").format(
            count=n_exported, start=export_start, end=export_end,
            path=bpy.path.abspath(self.filepath),
        )
        if len(set(counts)) > 1:
            msg += iface_(" (shorter caches hold their final pose past their length)")
        self.report({"INFO"}, msg)
        particle_names = [obj.name for obj in objs if obj.get("particle_mesh")]
        if particle_names:
            self.report({"INFO"}, "Sand exported as animated low-poly particle meshes: " + ", ".join(particle_names))
        if invisible:
            self.report(
                {"WARNING"},
                iface_(
                    "{count} object(s) hidden in the viewport were not "
                    "exported: {names}"
                ).format(count=len(invisible), names=', '.join(invisible)),
            )
        if excluded_curves:
            self.report(
                {"WARNING"},
                iface_(
                    "{count} rod/curve object(s) not exported "
                    "(unsupported by this cache export): {names}"
                ).format(count=len(excluded_curves), names=', '.join(excluded_curves)),
            )
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Concrete operators
# ---------------------------------------------------------------------------

_USD_EXTS = (".usdc", ".usda", ".usd", ".usdz")


class SOLVER_OT_ExportUSD(Operator, _ExportSimCacheBase):
    """Export the simulated mesh sequence as a USD cache.

    A lighter alternative to baking shape keys: the deformation is sampled
    per frame straight from the solver cache into a USD file that Blender and
    other DCC tools can play back and retime.
    """

    bl_idname = "solver.export_usd"
    bl_label = "Export USD"
    bl_options = {"REGISTER"}

    filename_ext = ".usdc"
    filter_glob: StringProperty(
        default="*.usd;*.usdc;*.usda;*.usdz", options={"HIDDEN"}
    )

    def check(self, context):
        # ExportHelper.check would coerce any suffix to ``.usdc`` (the USD
        # exporter picks crate/ascii/zip by extension, so that would make
        # .usda/.usdz unreachable). Keep any of the four known USD suffixes;
        # for anything else, replace the extension with the default (mirroring
        # ExportHelper.check's splitext + empty-basename guard, since
        # bpy.path.ensure_ext only appends and would otherwise double it).
        fp = self.filepath
        if not os.path.basename(fp):
            return False
        if fp.lower().endswith(_USD_EXTS):
            new_fp = fp
        else:
            new_fp = bpy.path.ensure_ext(
                os.path.splitext(fp)[0], self.filename_ext
            )
        if new_fp != self.filepath:
            self.filepath = new_fp
            return True
        return False

    def _run_exporter(self, context, start, end):
        # USD has no start/end kwargs; execute() has set scene.frame_start/end.
        return bpy.ops.wm.usd_export(
            "EXEC_DEFAULT",
            filepath=self.filepath,
            selected_objects_only=True,
            export_animation=True,
            evaluation_mode="VIEWPORT",
            check_existing=False,
            export_materials=False,
            relative_paths=False,
        )


class SOLVER_OT_ExportAlembic(Operator, _ExportSimCacheBase):
    """Export the simulated mesh sequence as an Alembic (ABC) cache.

    A lighter alternative to baking shape keys: the deformation is sampled
    per frame straight from the solver cache into an Alembic file that Blender
    and other DCC tools can play back and retime.
    """

    bl_idname = "solver.export_alembic"
    bl_label = "Export Alembic (ABC)"
    bl_options = {"REGISTER"}

    filename_ext = ".abc"
    filter_glob: StringProperty(default="*.abc", options={"HIDDEN"})

    def _run_exporter(self, context, start, end):
        # as_background_job MUST stay False: the finally-block state restore runs
        # right after this returns and would otherwise race a background job.
        return bpy.ops.wm.alembic_export(
            "EXEC_DEFAULT",
            filepath=self.filepath,
            selected=True,
            start=start,
            end=end,
            evaluation_mode="VIEWPORT",
            as_background_job=False,
            check_existing=False,
        )


classes = (
    SOLVER_OT_ExportUSD,
    SOLVER_OT_ExportAlembic,
)


def register():
    for cls in classes:
        _ppf_i18n.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

