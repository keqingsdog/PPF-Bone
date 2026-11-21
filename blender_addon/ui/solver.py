from ..core.pin_reference import decode_stored_pin_identifier, PinReferenceError
# Fork modification: changed from the upstream baseline.
# File: solver.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

from .. import i18n as _ppf_i18n
import os
import textwrap

import bpy  # pyright: ignore

from ..i18n.props import BoolProperty  # pyright: ignore
from bpy.types import (  # pyright: ignore
    Operator,
    Panel,
)
from ..i18n import (  # pyright: ignore
    pgettext_iface as iface_,
    pgettext_tip as tip_,
)

from ..core.animation import (
    clear_animation_data,
    prepare_animation_targets,
)
from ..core import encode_progress
from ..core.async_op import AsyncOperator, StageAbort
from ..core.playback import stop_for_edit, TimelineBusyError
from ..core.client import RemoteStatus
from ..core.client import communicator as com
from ..core.derived import is_server_busy_from_response as is_running
from ..core.encoder import prepare_upload, prepare_authoring_for_upload
from ..core.module import Cbor2NotInstalledError, cbor2_available
from ..core.encoder.mesh import compute_data_hash, encode_obj_with_hash
from ..core.encoder.params import compute_param_hash, encode_param_with_hash
from ..core.pc2 import (
    MODIFIER_NAME,
    assigned_uuids_in_active_groups,
    find_orphan_solver_caches,
    get_pc2_dir,
    get_pc2_path,
    has_mesh_cache,
    object_pc2_key_readonly,
    remove_orphan_solver_cache,
)
from ..core.uuid_registry import get_object_uuid
from ..core.utils import (
    get_category_name,
    redraw_all_areas,
)
from ..models.groups import (
    get_addon_data,
    has_addon_data,
    has_simulatable_dynamics,
    iterate_active_object_groups,
)
from .geometry_cleanup_ops import popup_transfer_error
from .dynamics.bake_ops import bake_progress_snapshot, is_bake_running
from .dynamics.pin_capture_ops import (
    is_pin_capture_running,
    pin_capture_progress_snapshot,
    pin_capture_stop_save_failed,
)
from .dynamics.static_deform_ops import (
    capture_progress_snapshot,
    is_capture_running,
)

# Shared wire payload and user-facing message for a remote-data delete.
# Referenced by both TransferRequestMixin.request_delete and
# SOLVER_OT_DeleteRemoteData.execute so the literals stay in one place.
_DELETE_QUERY = {"request": "delete"}
_DELETE_MESSAGE = "Deleting Remote Data..."

# Character width of the ASCII progress-bar fallback used when the Blender
# build lacks the native ``UILayout.progress`` widget. Shared by the bake,
# capture, and pin-capture progress blocks via ``_ascii_bar``.
_ASCII_BAR_WIDTH = 20


def _draw_export_buttons(box, capability=None):
    from ..core.capabilities import export_capability
    check = export_capability if capability is None else capability
    row = box.row()
    for kind, operator in (("usd", "solver.export_usd"),
                           ("alembic", "solver.export_alembic")):
        available, reason = check(kind)
        export_col = row.column()
        export_col.enabled = available
        export_col.operator(operator, icon="EXPORT")
        if not available:
            box.label(text=("USD：" if kind == "usd" else "Alembic：") + reason,
                      icon="INFO")


def _geometry_edit_blocks(cls):
    from ..core.bone_geometry_edit import active
    if not active():
        return False
    cls.poll_message_set('请先确认或取消代理缩放／网格编辑')
    return True


def _ascii_bar(factor: float) -> str:
    """Render a ``factor`` in ``[0, 1]`` as a fixed-width ASCII progress bar
    like ``[#####...............]`` for builds without ``progress``."""
    filled = int(round(factor * _ASCII_BAR_WIDTH))
    return "[" + "#" * filled + "." * (_ASCII_BAR_WIDTH - filled) + "]"


def _draw_progress(layout, done, total, status_text, icon, abort_op, *, capture_samples=False):
    """Draw one progress block (status label, native ``progress`` bar with an
    ASCII fallback, and an abort operator) into a fresh sub-box of ``layout``.

    Shared by the bake, capture, and pin-capture sites so the divide-by-zero
    guard and the ``progress``-widget fallback live in one place.
    """
    factor = (done / total) if total > 0 else 0.0
    prog_box = layout.box()
    prog_box.label(text=status_text, icon=icon)
    label_text = (iface_("Samples: {done}/{total}").format(done=done, total=total)
                  if capture_samples else f"{int(factor * 100)}% ({done}/{total})")
    try:
        prog_box.progress(factor=factor, type="BAR", text=label_text)
    except (AttributeError, TypeError):
        bar = _ascii_bar(factor)
        prog_box.label(text=f"{bar} {label_text}")
    prog_box.operator(abort_op, icon="X")


# Session-only, single most recent bulk run (no data saved in the .blend).
_bulk_capture_display = {}


def _bulk_capture_scene_key(context):
    scene = context.scene
    return (scene.session_uid, bpy.data.filepath)


def _bulk_capture_phase(context, kind):
    if _bulk_capture_display.get("scene") == _bulk_capture_scene_key(context):
        return _bulk_capture_display["phases"].get(kind)
    return None


def _draw_bulk_capture_modes(box, context):
    if _bulk_capture_display.get("scene") != _bulk_capture_scene_key(context):
        return
    phases = _bulk_capture_display["phases"]
    if not phases:
        return
    col = box.column(align=True)
    running = any(phase["active"] for phase in phases.values())
    col.label(text=iface_("Bulk capture methods") if running else iface_("Last bulk capture"), icon="INFO")
    region_width = getattr(getattr(context, "region", None), "width", 260)
    ui_scale = bpy.context.preferences.system.ui_scale or 1.0
    reason_width = max(16, int((region_width / ui_scale - 45) / 7))
    if _bulk_capture_display.get("joint") and len(phases) > 1:
        col.label(text=iface_("Capture modes share one timeline pass"), translate=False)
    for kind, phase in phases.items():
        title = iface_("Static deformation") if kind == "STATIC" else iface_("Pin capture")
        mode = (iface_("Frame batch") if phase["batch"] else iface_("Sequential"))
        col.label(text=f"{title}: {mode}", translate=False)
        count = iface_("Objects: {count}").format(count=phase["objects"])
        if kind == "PIN":
            count += " | " + iface_("Pin groups: {count}").format(count=phase["targets"])
        col.label(text=count, translate=False)
        if phase["reason"]:
            reason = iface_("Stopped: {reason}").format(reason=phase["reason"])
            # Fullwidth/CJK labels consume roughly twice the width of Latin.
            width = max(10, reason_width // 2) if any(ord(c) > 0x2E80 for c in reason) else reason_width
            for line in textwrap.wrap(reason, width=width):
                col.label(text=line, translate=False)


def _bulk_capture_frame_label(phase, *, is_pin):
    if phase["frame"] is None:
        return iface_("Preparing Pin capture...") if is_pin else iface_("Preparing Static capture...")
    if phase["batch"]:
        return iface_("Scene frame {frame} | {count} objects").format(
            frame=phase["frame"], count=phase["frame_objects"],
        )
    return iface_("Scene frame {frame}").format(frame=phase["frame"])


def _find_missing_pc2_paths(context) -> list[str]:
    """Every referenced-but-missing PC2 path, in deterministic order.

    Two kinds of reference, both scoped to objects assigned to an ACTIVE
    object group:
      * MESH with ``ContactSolverCache``: ``mod.filepath`` resolved via
        ``bpy.path.abspath`` — the file Blender plays back.
      * CURVE: the basename-derived ``get_pc2_path(key)`` — curves have no
        modifier, playback reads this path directly.

    The scope is load-bearing, and it is the same scope
    ``scene_has_solver_cache`` takes. A ``ContactSolverCache`` outlives its
    group membership: duplicating a simulated object copies the modifier
    onto a copy that belongs to no group, and pointing that copy at a file
    that does not exist is enough to raise this warning. Clear Local
    Animation only clears assigned objects, so an unscoped warning names a
    path that nothing in the UI can act on, and it stays on screen for the
    life of the file.

    Gated on ``bpy.data.filepath`` (path resolution is meaningless for
    an unsaved scene) and on at least one assigned mesh carrying the
    modifier (proxy for "a sim has been run"), otherwise absent PC2
    files are the expected state for a fresh project.
    """
    if not bpy.data.filepath:
        return []

    # Fast common-case gate: this warning exists for the "data folder
    # moved/gone" case (a rename or OS-copy of the .blend leaves the current
    # project's PC2 directory empty). So if that directory holds ANY cache
    # file, the data is present and nothing is missing -- return immediately
    # without touching a single object. ``os.scandir`` + ``any`` early-exits
    # on the first ``.pc2`` entry, so this is ~one cheap syscall instead of
    # the O(objects) scan below, which now runs ONLY in the rare folder-gone
    # state (exactly when the enumerated list is actually wanted for display).
    try:
        with os.scandir(get_pc2_dir()) as _it:
            if any(_e.name.endswith(".pc2") for _e in _it):
                return []
    except OSError:
        pass  # directory absent -> fall through and enumerate what's missing

    # Collect every referenced PC2 path in deterministic order first, then
    # resolve existence with ONE directory listing per referenced directory
    # rather than an ``os.path.exists`` per file.
    candidates: list[str] = []
    saw_modifier = False
    active_uuids = assigned_uuids_in_active_groups(context.scene)
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        mod = obj.modifiers.get(MODIFIER_NAME)
        if mod is None or not mod.filepath:
            continue
        # Read-only key: draw must never mutate, and the side-effecting
        # variant runs an O(N) duplicate scan per object (O(N^2) overall,
        # ~0.6s on a 1.6k-object scene). An object that owns a cache was
        # already assigned a UUID at capture time, so an empty key here
        # means "no cache", and it is not in the assigned set either.
        key = object_pc2_key_readonly(obj)
        if key not in active_uuids:
            continue
        saw_modifier = True
        # Modifier's stored filepath — what MESH_CACHE plays back from.
        candidates.append(bpy.path.abspath(mod.filepath, library=mod.id_data.library))
        # Basename-derived canonical path — what the overlay/heal/bake
        # code paths read. Diverges from mod.filepath after a rename or
        # OS-copy of the .blend (mod.filepath still points at the old
        # data/<basename>/ folder, canonical now targets the new one).
        candidates.append(get_pc2_path(key))

    if not saw_modifier:
        return []

    for obj in bpy.data.objects:
        if obj.type != "CURVE":
            continue
        uid = get_object_uuid(obj)
        if not uid or uid not in active_uuids:
            continue
        candidates.append(get_pc2_path(uid))

    # Resolve existence per directory: one os.listdir() per unique dir, then
    # membership in that set (an absent dir lists as empty -> all missing,
    # matching the prior os.path.exists semantics). Order/dedup preserved so
    # ``missing[0]`` stays the first referenced path.
    listings: dict[str, set] = {}
    missing: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        directory = os.path.dirname(path)
        names = listings.get(directory)
        if names is None:
            try:
                names = set(os.listdir(directory))
            except OSError:
                names = set()
            listings[directory] = names
        if os.path.basename(path) not in names:
            missing.append(path)

    return missing


def _detect_pc2_basename_mismatch(context):
    """``(old_name, new_name)`` when every ``ContactSolverCache`` modifier
    points into a single ``data/<old_name>/`` folder that differs from
    the current .blend basename, else ``None``.

    Fires only on the unambiguous "rename / OS-copy of the .blend"
    shape: one old prefix across all modifiers, current basename is
    the new one, old folder still on disk (so a migration can actually
    move or copy it). Ambiguous states (mixed prefixes, modifiers
    pointing outside ``<blend_dir>/data/``) return ``None``.

    Reads the same assigned-in-an-active-group scope as
    :func:`_find_missing_pc2_paths`, whose warning this button appears
    under: a modifier on an unassigned copy is not this project's data, and
    letting one vote here would either invent a second prefix and suppress
    the button, or offer to migrate a folder on the strength of a cache
    nothing is simulating.
    """
    if not bpy.data.filepath:
        return None
    blend_dir = os.path.dirname(bpy.data.filepath)
    current = os.path.splitext(os.path.basename(bpy.data.filepath))[0]
    expected_root = os.path.realpath(os.path.join(blend_dir, "data"))
    active_uuids = assigned_uuids_in_active_groups(context.scene)

    old_names: set[str] = set()
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        mod = obj.modifiers.get(MODIFIER_NAME)
        if mod is None or not mod.filepath:
            continue
        if object_pc2_key_readonly(obj) not in active_uuids:
            continue
        abs_path = os.path.realpath(bpy.path.abspath(mod.filepath, library=mod.id_data.library))
        parent = os.path.dirname(abs_path)
        if os.path.dirname(parent) != expected_root:
            continue
        name = os.path.basename(parent)
        if name:
            old_names.add(name)

    if len(old_names) != 1:
        return None
    (old_name,) = old_names
    if old_name == current:
        return None
    if not os.path.isdir(os.path.join(blend_dir, "data", old_name)):
        return None
    return (old_name, current)


def _warn_if_mesh_topology_stale(op, context) -> None:
    """Report a WARNING (and log to the Console) when the mesh hash
    stored at last transfer no longer matches the current scene.

    Doesn't block the operation — transfers must remain in the user's
    control — but makes the divergence visible instead of silently
    running the sim on stale geometry."""
    try:
        state = get_addon_data(context.scene).state
        msg = state.validate_mesh_hash(context)
    except Exception:
        return
    if msg:
        op.report({"WARNING"}, msg)
        try:
            from ..models.console import console
            console.write(f"[Mesh hash] {msg}")
        except Exception:
            pass


def _staged_encode_active() -> bool:
    """True while a Transfer/Run staged encode is running its modal stages.

    ``AsyncOperator.start_stages`` publishes an encode-progress run
    synchronously inside ``execute`` but only dispatches to the engine on its
    final stage, so ``engine.state.busy`` (which the action polls otherwise
    rely on to disable each other) stays False for the whole encode window.
    The action operators add ``not _staged_encode_active()`` to their poll so
    a second Transfer / Run / Update Params / Resume cannot start while an
    encode is still staging.
    """
    return encode_progress.is_active()


def _check_project_name_sync(context) -> str:
    """Detect project-name drift between UI and the active connection.

    The runner captures the project name at Connect. Later writes to
    ``state.project_name`` (e.g. MCP ``set_scene_parameters``) do not resync,
    so a transfer would land under the stale name. Returns an error string,
    or empty string if the two are in sync.
    """
    state = get_addon_data(context.scene).state
    ui_name = (state.project_name or "").strip()
    runner_name = (com.project_name or "").strip()
    if ui_name == runner_name:
        return ""
    return (
        iface_(
            "Project name out of sync: UI='{ui_name}' but active session='{runner_name}'. "
            "Disconnect and reconnect to resync."
        ).format(ui_name=ui_name, runner_name=runner_name)
    )


def _check_uuid_consistency(context) -> str:
    """Check UUID consistency for all assigned objects.

    Returns an error message string, or empty string if all OK.
    """
    from ..core.uuid_registry import get_object_uuid, resolve_assigned

    missing_uuid = []
    stale_uuid = []
    for group in iterate_active_object_groups(context.scene):
        for assigned in group.assigned_objects:
            if not assigned.included:
                continue
            obj = resolve_assigned(assigned)
            if obj is None:
                continue
            obj_uuid = get_object_uuid(obj)
            if not obj_uuid:
                missing_uuid.append(obj.name)
            elif assigned.uuid != obj_uuid:
                stale_uuid.append(obj.name)
    if missing_uuid:
        names = ", ".join(missing_uuid[:3])
        more = iface_(" (+{count} more)").format(count=len(missing_uuid) - 3) if len(missing_uuid) > 3 else ""
        return iface_("Objects missing UUID: {names}{more}. Run UUID Migration first.").format(names=names, more=more)
    if stale_uuid:
        names = ", ".join(stale_uuid[:3])
        more = iface_(" (+{count} more)").format(count=len(stale_uuid) - 3) if len(stale_uuid) > 3 else ""
        return iface_("Stale UUID references: {names}{more}. Run UUID Migration first.").format(names=names, more=more)
    # Check if broader migration is needed (pins, pairs, keyframes)
    from ..core.migrate import needs_migration
    return needs_migration()


def _check_merge_pairs_stitch(context) -> str:
    """Reject a transfer when any merge pair's stitch cannot reach the solver.

    Asks ``merge_pair_problem``, the same function the encoder refuses with,
    so the check and the encode cannot disagree about a pair. Returns an
    error message naming each such pair and its reason, or an empty string.
    Nothing is written to a pair: a stale one is reported, not cleared.
    """
    from ..mesh_ops.merge_ops import _member_types, merge_pair_problem, pair_label

    scene = context.scene
    state = get_addon_data(scene).state
    members = _member_types(scene)
    problems = []
    for pair in state.merge_pairs:
        problem = merge_pair_problem(scene, pair, members)
        if problem is not None:
            problems.append(f"{pair_label(pair)}: {problem}")
    if not problems:
        return ""
    more = (
        iface_(" (+{count} more)").format(count=len(problems) - 3)
        if len(problems) > 3 else ""
    )
    return iface_(
        "Merge pair(s) that cannot stitch: {problems}{more}. Fix each under "
        "Merge Pairs before transferring."
    ).format(problems="; ".join(problems[:3]), more=more)


def report_build_outcome(op) -> None:
    """Report a finished build's outcome through ``op.report``.

    Every operator that starts a remote build finishes as soon as the
    status leaves ``BUILDING``, and a failed build leaves it exactly as a
    successful one does, so the outcome is read here before anything is
    reported about it. This is the single reporting path for all of them,
    so the status test, the headline extraction and the wording are
    defined once and cannot differ from one build button to the next.

    ``com.server_error`` carries the build worker's headline on its first
    line and its traceback below it. The two surfaces this reaches, the
    operator status bar and the panel label, each render a single line,
    so only the headline is reported and the traceback stays in the
    Console panel.

    Success is claimed only for the statuses that mean the scene is built
    and runnable. Leaving ``BUILDING`` does not imply that on its own: a
    lost connection, a stopped server and a protocol mismatch all leave
    it too, and each ends the build with no scene to run. Those report
    the status they landed on, so the message never claims more than the
    status supports.
    """
    status = com.info.status
    if status == RemoteStatus.SIMULATION_FAILED:
        headline = com.server_error.partition("\n")[0]
        op.report(
            {"ERROR"},
            iface_("Build failed: {error}").format(error=iface_(headline)),
        )
    elif status in (RemoteStatus.READY, RemoteStatus.RESUMABLE):
        op.report({"INFO"}, iface_("Build completed successfully."))
    else:
        op.report(
            {"WARNING"},
            iface_("Build ended without a result: {status}").format(
                status=status.value
            ),
        )


class TransferRequestMixin:
    """Shared request methods for transfer operators.

    Data and param ship together through a single ``upload_atomic``
    transaction so the server cannot observe a mismatched (data, param)
    pair mid-build. Upload sites call ``core.encoder.prepare_upload`` for
    ``(data, param, data_hash, param_hash)`` (which also stamps the local
    cache), then dispatch via ``com.build_pipeline`` or ``com.upload_only``.
    """

    def request_delete(self):
        self._mode = "delete"
        com.query(_DELETE_QUERY, _DELETE_MESSAGE)


def _scene_has_enabled_bone_dynamics(scene):
    from ..core.rigid_pose import has_enabled_bones
    return has_enabled_bones(scene)


class SOLVER_OT_RecoverBoneConnections(Operator):
    bl_idname = "ppf.recover_pending_connections"
    bl_label = "恢复骨骼连接"
    bl_description = "恢复失败的骨骼连接事务；如骨骼已被修改则不会覆盖"

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if not obj or obj.type != 'ARMATURE' or not obj.select_get():
            return False
        from ..core.rigid_pose import pending_connection_status
        return pending_connection_status(obj)[1]

    def execute(self, context):
        from ..core.rigid_pose import recover_pending_connections
        try:
            result = recover_pending_connections(context.active_object)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, "骨骼连接已恢复" if result is True else "骨骼连接已经恢复")
        return {'FINISHED'}


class SOLVER_PT_SolverPanel(Panel):
    @classmethod
    def poll(cls, context):
        from .layout import classic_visible
        return classic_visible(context) and cls.content_poll(context)

    bl_label = "Solver"
    bl_idname = "SSH_PT_SolverPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = get_category_name()

    @classmethod
    def content_poll(cls, context):
        return has_addon_data(context.scene)

    def draw(self, context):
        root = get_addon_data(context.scene)
        state = root.state
        layout = self.layout
        from ..core.rigid_pose import pending_connection_status
        pending_names, pending_active = pending_connection_status(context.active_object if context.active_object and context.active_object.type == 'ARMATURE' else None)
        if pending_names:
            box = layout.box()
            box.alert = True
            box.label(text="骨骼连接恢复待处理", icon='ERROR')
            box.label(text=", ".join(pending_names))
            box.label(text="选择原骨架并恢复；修改冲突不会被覆盖")
            if pending_active:
                box.operator("ppf.recover_pending_connections", icon='LOOP_BACK')
        warning = context.scene.get("ppf_bone_pose_save_warning")
        if warning:
            box = layout.box()
            box.alert = True
            box.label(text="Bone cache was NOT migrated", icon='ERROR')
            box.label(text="Keep the old cache; Save As to a new filename.")
            box.label(text=str(warning))
        has_dynamic = has_simulatable_dynamics(context.scene) or _scene_has_enabled_bone_dynamics(context.scene)
        box = layout.box()
        # Every button in this box encodes the scene with cbor2, so an install
        # whose bundled wheel never landed can do nothing here. Saying so on
        # THIS panel is what makes the condition findable: the recovery button
        # lives on the Backend Communicator panel, which a user who only wants
        # to press Transfer has no reason to open, and without this line their
        # first sign of trouble is a failure from a button that looked ready.
        if not cbor2_available():
            row = box.row()
            row.label(text=iface_("cbor2 is missing; Transfer and Run cannot encode"), icon="ERROR")
            row = box.row()
            row.label(text=iface_("Install it from the Backend Communicator panel"))
        row = box.row()
        row.operator(SOLVER_OT_Transfer.bl_idname, icon="EXPORT")
        row.operator(SOLVER_OT_UpdateParams.bl_idname, icon="OPTIONS")
        row = box.row()
        row.operator(SOLVER_OT_Run.bl_idname, icon="PLAY")
        row.operator(SOLVER_OT_ResumeFrom.bl_idname, icon="PREV_KEYFRAME", text=iface_("续算 / 延长动画"))
        if com.info.response.get("resume_block_reason") == "CHECKPOINTS_FROM_PREVIOUS_BUILD":
            box.label(text=iface_("参数已更新，旧断点只保留供查看。请重新运行。"), icon="INFO")
        elif com.is_connected() and com.info.status in (RemoteStatus.READY, RemoteStatus.RESUMABLE, RemoteStatus.SIMULATION_FAILED) and not any(f > 0 for f in com.saved_state_frames()):
            box.label(text=iface_("无可续算状态：启用完成时保存状态后重新运行"), icon="INFO")

        row = box.row()
        row.operator(SOLVER_OT_FetchData.bl_idname, icon="IMPORT")
        row.operator(SOLVER_OT_DeleteRemoteData.bl_idname, icon="TRASH")

        row = box.row()
        row.operator("solver.bake_all_animation", icon="ACTION")
        row.operator("solver.bake_all_single_frame", icon="KEYFRAME")

        row = box.row()
        row.operator(SOLVER_OT_ClearAnimation.bl_idname, icon="OBJECT_ORIGIN")
        row.operator(SOLVER_OT_RestoreSingleBake.bl_idname, icon="LOOP_BACK", text=iface_("撤销本地烘焙 / 恢复模拟"))
        row = box.row()
        row.operator(SOLVER_OT_DiscardSingleBakeRecovery.bl_idname, icon="CHECKMARK", text=iface_("保留烘焙并结束恢复"))

        box.label(text=iface_("延长动画：增加总帧数后点击续算；不要修改起始帧。"), icon="INFO")

        from ..core.live_timeline import is_paused
        from ..core.facade import runner
        fetch_paused = runner.live_fetch_paused(context.scene)
        if is_paused(context.scene) or fetch_paused:
            row = box.row()
            row.label(text=iface_("自动取帧暂停，请重试" if fetch_paused
                                 else "已暂停实时跟随，可查看历史帧"), icon="PAUSE")
            row = box.row()
            row.operator("ppf.resume_live_follow", icon="PLAY", text=iface_("恢复实时跟随"))

        # Surface missing PC2 files so the user can copy data/<basename>/
        # into place after a rename/duplicate. File-level check (not dir)
        # because an empty data/ directory can still exist.
        missing_pc2 = _find_missing_pc2_paths(context)
        if missing_pc2:
            row = box.row()
            row.label(
                text=iface_("Data path: {path} does not exist.").format(path=missing_pc2[0]),
                icon="ERROR",
            )
            mismatch = _detect_pc2_basename_mismatch(context)
            if mismatch:
                old_name, new_name = mismatch
                row = box.row()
                row.operator(
                    SOLVER_OT_MigratePC2Folder.bl_idname,
                    icon="FILE_REFRESH",
                    text=iface_("Migrate data/{old_name}/ \u2192 data/{new_name}/").format(old_name=old_name, new_name=new_name),
                )
            list_box = box.box()
            list_box.label(text=iface_("Missing files ({count}):").format(count=len(missing_pc2)))
            col = list_box.column(align=True)
            for path in missing_pc2:
                col.label(text=path)

        # A cache modifier on an object no group claims. Every gate above
        # ignores it on purpose, so name it here: otherwise it is an
        # addon-created modifier, usually with a filepath that resolves to
        # nothing, that the addon will neither act on nor admit exists.
        orphans = find_orphan_solver_caches(context.scene)
        if orphans:
            names = ", ".join(repr(obj.name) for obj in orphans[:3])
            if len(orphans) > 3:
                names += iface_(", +{count} more").format(count=len(orphans) - 3)
            col = box.column(align=True)
            col.label(
                text=iface_("Stale cache on {names}").format(names=names),
                icon="ERROR",
            )
            col.label(text=iface_("(not in any group)"))
            col.operator(SOLVER_OT_RemoveStaleCaches.bl_idname, icon="TRASH")

        if is_bake_running():
            done, total, status, _ = bake_progress_snapshot()
            _draw_progress(
                box, done, total, status or "Baking...", "TIME", "solver.bake_abort"
            )

        joint = (_bulk_capture_display.get("joint") if
                 _bulk_capture_display.get("scene") == _bulk_capture_scene_key(context) else None)
        joint_active = bool(joint and joint.get("active"))
        if joint_active:
            frame = joint["frame"]
            title = (iface_("Preparing combined capture...") if frame is None else
                     iface_("Scene frame {frame} | shared pass").format(frame=frame))
            _draw_progress(box, joint["done"], joint["total"], title, "REC",
                           "solver.capture_abort" if joint["st_total"] else "solver.pin_capture_abort",
                           capture_samples=True)
            if joint["st_total"]:
                box.label(text=iface_("Static samples: {done}/{total}").format(
                    done=joint["st_done"], total=joint["st_total"]), translate=False)
            if joint["pin_total"]:
                box.label(text=iface_("Pin samples: {done}/{total}").format(
                    done=joint["pin_done"], total=joint["pin_total"]), translate=False)

        if is_capture_running() and not joint_active:
            done, total, status, _ = capture_progress_snapshot()
            phase = _bulk_capture_phase(context, "STATIC")
            active = bool(phase and phase["active"])
            if active:
                status = _bulk_capture_frame_label(phase, is_pin=False)
            _draw_progress(
                box,
                done,
                total,
                status or "Capturing...",
                "REC",
                "solver.capture_abort",
                capture_samples=active,
            )

        if is_pin_capture_running() and not joint_active:
            done, total, status, _ = pin_capture_progress_snapshot()
            phase = _bulk_capture_phase(context, "PIN")
            active = bool(phase and phase["active"])
            if active:
                status = _bulk_capture_frame_label(phase, is_pin=True)
            _draw_progress(
                box,
                done,
                total,
                status or "Capturing pin...",
                "REC",
                "solver.pin_capture_abort",
                capture_samples=active,
            )
            if pin_capture_stop_save_failed():
                box.operator("solver.pin_capture_abort", text="丢弃未完成进度", icon="X").save_partial = False

        # Show missing frames warning (not during simulation)
        response = com.info.response
        remote_frames = int(response.get("frame", 0)) if response else 0
        if remote_frames > 0 and not is_running(response):
            fetched = state.convert_fetched_frames_to_list()
            n_missing = remote_frames - len(fetched)
            if n_missing > 0:
                row = box.row()
                row.label(text=iface_("{count} frames unfetched. Press \"Fetch All Animation\".").format(count=n_missing), icon="ERROR")

        if not has_dynamic:
            layout.label(text="No dynamics to simulate", icon="ERROR")

        if com.info.status == RemoteStatus.WAITING_FOR_DATA and has_dynamic and SOLVER_OT_Transfer.poll(context):
            row = box.row()
            row.label(text="Click \"Transfer\" to upload data", icon="INFO")

        # Warn when local animation is the only blocker for Run
        if SOLVER_OT_ClearAnimation.poll(context):
            status = com.info.status
            response = com.info.response
            run_ready = (
                not com.busy()
                and status in (
                    RemoteStatus.READY,
                    RemoteStatus.RESUMABLE,
                    RemoteStatus.SIMULATION_FAILED,
                )
                and com.is_connected()
                and com.is_server_running()
                and status.ready()
                and not is_running(response)
            )
            if run_ready:
                row = box.row()
                row.label(text="Clear local animation before running", icon="INFO")

        # Deformation caches: batch re-capture / clear of every STATIC-collider
        # deform cache and every animated-pin capture. Plain (always-expanded)
        # box so both actions are always visible; each button's poll() greys it
        # out when there is nothing to capture / clear.
        box = layout.box()
        box.label(text="Deformations", icon="MOD_SIMPLEDEFORM")
        batch_control = box.row()
        batch_control.enabled = not (is_capture_running() or is_pin_capture_running())
        batch_control.prop(context.window_manager, "ppf_capture_frame_batch")
        box.operator(
            SOLVER_OT_RecaptureAllDeformations.bl_idname, icon="FILE_REFRESH"
        )
        box.operator(SOLVER_OT_ClearAllDeformations.bl_idname, icon="TRASH")
        _draw_bulk_capture_modes(box, context)

        # Export the simulated mesh sequence to a USD / Alembic cache: a
        # lighter alternative to the shape-key bake for long/heavy animations.
        # Each button's poll() greys it out when there is nothing to export
        # (no simulated mesh in the view layer, or a solver activity in flight).
        box = layout.box()
        box.label(text="Export", icon="EXPORT")
        _draw_export_buttons(box)

        # JupyterLab expandable box
        box = layout.box()
        row = box.row(align=True)
        row.alignment = 'LEFT'
        row.prop(state, "show_jupyter", icon="TRIA_DOWN" if state.show_jupyter else "TRIA_RIGHT", emboss=False, text="")
        row.label(text="JupyterLab", icon="FILE_SCRIPT")
        if state.show_jupyter:
            if not com.is_connected():
                box.label(text="Connect to server to export/delete", icon="INFO")
            row = box.row(align=True)
            row.operator("solver.jupyter_export", icon="EXPORT")
            row.operator("solver.jupyter_open", icon="URL")
            row.operator("solver.jupyter_delete", icon="TRASH")
            row = box.row(align=True)
            row.operator("debug.transfer_without_build", icon="ARROW_LEFTRIGHT", text="Transfer without Build")
            row.operator("debug.build", icon="MOD_BUILD")
            row = box.row()
            row.prop(state, "jupyter_port", text="Port")

        # MCP Server section
        from ..mcp.mcp_server import is_mcp_running
        box = layout.box()
        row = box.row(align=True)
        row.alignment = 'LEFT'
        row.prop(
            state, "show_mcp",
            icon="TRIA_DOWN" if state.show_mcp else "TRIA_RIGHT",
            emboss=False, text="",
        )
        mcp_label = iface_("MCP Server (Running :{port})").format(port=state.mcp_port) if is_mcp_running() else "MCP Server (Stopped)"
        row.label(text=mcp_label, icon="NETWORK_DRIVE")
        if state.show_mcp:
            from .mcp_ops import MCP_OT_StartServer, MCP_OT_StopServer
            row = box.row(align=True)
            mcp_running = is_mcp_running()
            if mcp_running:
                row.operator(MCP_OT_StopServer.bl_idname, icon="PAUSE")
            else:
                row.operator(MCP_OT_StartServer.bl_idname, icon="PLAY")
            sub = row.row()
            sub.enabled = not mcp_running
            sub.prop(state, "mcp_port", text="Port")


def _remove_transfer_upload_listener(op):
    listener = getattr(op, "_transfer_upload_listener", None)
    if listener is not None:
        com._engine.remove_listener(listener)
        op._transfer_upload_listener = None


class SOLVER_OT_Transfer(AsyncOperator):
    """Upload the scene to the solver, overwriting data.pickle and
    param.pickle without touching cached artifacts (tetrahedralization,
    BVH, mesh caches). Use ``Delete Remote Data`` to wipe everything."""

    bl_idname = "solver.transfer"
    bl_label = "Transfer"

    _mode = None
    timeout: float = 300.0
    auto_redraw: bool = True

    @classmethod
    def poll(cls, context):
        if _geometry_edit_blocks(cls):
            return False
        response = com.info.response
        has_dynamic = has_simulatable_dynamics(context.scene) or _scene_has_enabled_bone_dynamics(context.scene)
        # ``status.ready()`` is a protocol-version check only; it
        # returns True for every active operation (BUILDING, FETCHING,
        # STARTING_SOLVER, ...). ``status.in_progress()`` is the
        # explicit "server is doing something the user shouldn't
        # interrupt" set. ``not is_running(response)`` only inspects
        # the cached server response, which lags by one PollTick after
        # a click, so STARTING_SOLVER right after Run would otherwise
        # let Transfer stay clickable for a window.
        return (
            has_dynamic
            and not com.busy()
            and not _staged_encode_active()
            and com.is_connected()
            and com.is_server_running()
            and com.info.status.ready()
            and not com.info.status.in_progress()
            and not is_running(response)
            and com.connection.remote_root
        )

    def execute(self, context):
        from ..core.single_bake_restore import available
        if available(context.scene):
            self.report({"ERROR"}, "请先选择恢复单帧烘焙前的模拟，或保留单帧结果并结束恢复。")
            return {"CANCELLED"}
        error = _check_project_name_sync(context)
        if error:
            self.report({"ERROR"}, error)
            return {"CANCELLED"}
        error = _check_uuid_consistency(context)
        if error:
            self.report({"ERROR"}, error)
            return {"CANCELLED"}
        error = _check_merge_pairs_stitch(context)
        if error:
            self.report({"ERROR"}, error)
            return {"CANCELLED"}
        # Transfer is upload-only: it overwrites data.pickle and
        # param.pickle atomically and lets the server's build pipeline
        # decide what cached artifacts (tetrahedralization, BVH, mesh
        # caches) remain valid for the new payload. To wipe everything,
        # the user has a dedicated "Delete Remote Data" button.
        #
        # The scene encode runs on the main thread and can take seconds on
        # heavy scenes, so split it across staged modal ticks: the panel
        # shows a labeled progress bar from the moment of the click instead
        # of a frozen cursor, and the geometry / parameter / upload phases
        # are each named as they run. See AsyncOperator.start_stages.
        self._stage_communicator = com
        self._mode = "pipeline"
        self._payload = {}
        self._transfer_frame = (context.scene.frame_current, context.scene.frame_subframe)
        self.start_stages(context, [
            ("Encoding scene and parameters...", self._stage_encode_geometry),
            ("Uploading scene...", self._stage_upload),
        ])
        return {"RUNNING_MODAL"}

    def _stage_encode_geometry(self, context):
        try:
            if (context.scene.frame_current, context.scene.frame_subframe) != self._transfer_frame:
                raise ValueError(iface_("Timeline changed before Transfer encoding; start Transfer again"))
            data, param, data_hash, param_hash = prepare_upload(context)
        except (Cbor2NotInstalledError, ValueError) as e:
            com.set_error(str(e))
            # A degenerate tessellation is the one refusal that both costs the
            # artist a Transfer and has a one-click repair, so it gets a dialog
            # carrying that repair rather than a status-bar line they have to
            # read sideways. Every other error keeps the reported line alone.
            popup_transfer_error(e)
            raise StageAbort(str(e))
        self._payload["data"] = data
        self._payload["data_hash"] = data_hash
        self._payload["param"] = param
        self._payload["param_hash"] = param_hash

    def _stage_upload(self, context):
        # The topology stamp lands HERE, not in the encoder: every stage that
        # can refuse the Transfer has run by now, so a scene that never reached
        # the wire does not get recorded as transferred and its stale-topology
        # warning still fires.
        from ..core.async_op import _origin_cancel_message
        if self._origin_changed(context):
            raise StageAbort(_origin_cancel_message(self), 'WARNING')
        if (context.scene.frame_current, context.scene.frame_subframe) != self._transfer_frame:
            raise StageAbort(iface_("Timeline changed during Transfer; start Transfer again"), 'WARNING')
        state = get_addon_data(context.scene).state
        state.commit_pending_mesh_hash()
        if self._origin_changed(context):
            raise StageAbort(_origin_cancel_message(self), 'WARNING')
        # Local fetched animation is keyed by the previous upload's data;
        # new data invalidates it, so drop it before kicking the pipeline
        # (matches the prior NO_DATA fast-path behavior).
        com.animation.clear()
        if self._origin_changed(context):
            raise StageAbort(_origin_cancel_message(self), 'WARNING')
        # Pin the server generation before this full-scene upload.  An old
        # READY status alone cannot prove this request reached the server.
        self._pre_transfer_upload_id = com._engine.state.server_upload_id
        # Observe this pipeline's upload-complete transition on the main thread.
        # READY alone may still describe the old server scene.
        from ..core.state import Activity
        self._transfer_upload_acknowledged = False
        engine = com._engine
        def observe_upload(old, new):
            if old.activity != Activity.SENDING or not old.pending_build:
                return
            if new.activity == Activity.SENDING:
                return
            self._transfer_upload_acknowledged = new.activity == Activity.BUILDING
            engine.remove_listener(observe_upload)
            self._transfer_upload_listener = None
        self._transfer_upload_listener = observe_upload
        engine.on_change(observe_upload)
        from ..core.effect_runner import ConnectionOriginChanged
        try:
            com.build_pipeline(
                data=self._payload["data"], param=self._payload["param"],
                data_hash=self._payload["data_hash"],
                param_hash=self._payload["param_hash"],
                message="Uploading scene...",
                expected_connection_ticket=getattr(self, '_stage_connection_ticket', None),
            )
        except ConnectionOriginChanged as exc:
            engine.remove_listener(observe_upload)
            self._transfer_upload_listener = None
            raise StageAbort(str(exc), 'WARNING') from exc
        except BaseException:
            engine.remove_listener(observe_upload)
            self._transfer_upload_listener = None
            raise
        self.mark_remote_requested()
        # Submission has transferred ownership to the engine/worker queue.
        # Build completion needs only receipts, never the original bytes.
        self._payload = {}

    def cleanup_modal(self, context):
        # Also release locally retained bytes after refusal/cancel/timeout.
        # Queued or in-flight upload commands keep their own references.
        self._payload = {}
        _remove_transfer_upload_listener(self)
        super().cleanup_modal(context)

    def is_complete(self) -> bool:
        if self._mode != "pipeline":
            return False
        if com.busy():
            return False
        return com.info.status != RemoteStatus.BUILDING

    def is_cancelled(self) -> bool:
        return com.is_aborting()

    def on_complete(self, context):
        from ..models.console import console as _console
        _console.write(f"[Transfer] complete: {com.info.status.value}")
        upload_id = com._engine.state.server_upload_id
        previous = getattr(self, "_pre_transfer_upload_id", None)
        listener = getattr(self, "_transfer_upload_listener", None)
        if listener is not None:
            com._engine.remove_listener(listener)
            self._transfer_upload_listener = None
        if (getattr(self, "_transfer_upload_acknowledged", False)
                and com.info.status in (RemoteStatus.READY, RemoteStatus.RESUMABLE)
                and (not isinstance(previous, str) or not isinstance(upload_id, str)
                     or not upload_id or upload_id != previous)
                and "ppf_single_bake_session_stale" in context.scene):
            del context.scene["ppf_single_bake_session_stale"]
        report_build_outcome(self)


class SOLVER_OT_Run(AsyncOperator):
    """Run the solver."""

    bl_idname = "solver.run"
    bl_label = "Run"

    timeout: float = 86400.0  # 24 hours for long simulations
    auto_redraw: bool = True

    @classmethod
    def poll(cls, context):
        from ..core.single_bake_restore import available
        if available(context.scene):
            return False
        if _geometry_edit_blocks(cls):
            return False
        if is_bake_running():
            return False
        status = com.info.status
        response = com.info.response
        base = (
            not com.busy()
            and not _staged_encode_active()
            and status in (
                RemoteStatus.READY,
                RemoteStatus.RESUMABLE,
                RemoteStatus.SIMULATION_FAILED,
            )
            and com.is_connected()
            and com.is_server_running()
            and com.info.status.ready()
            and not is_running(response)
            and not SOLVER_OT_ClearAnimation.poll(context)
        )
        return base

    def execute(self, context):
        try:
            stop_for_edit(context)
        except TimelineBusyError as exc:
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}
        error = _check_project_name_sync(context)
        if error:
            self.report({"ERROR"}, error)
            return {"CANCELLED"}
        _warn_if_mesh_topology_stale(self, context)
        # The click-time drift check re-encodes the scene (compute_data_hash),
        # which is the same heavy main-thread work as Transfer, so stage it
        # behind a labeled progress bar instead of freezing on the click. The
        # final stage starts the solver; after it the modal waits on the sim.
        self.start_stages(context, [
            ("Checking scene geometry...", self._stage_check_geometry),
            ("Checking parameters...", self._stage_check_params),
            ("Starting solver...", self._stage_start_run),
        ])
        return {"RUNNING_MODAL"}

    def _stage_check_geometry(self, context):
        try:
            stop_for_edit(context)
        except TimelineBusyError as exc:
            raise StageAbort(str(exc))
        # Refuse the run when the live encoded data no longer matches what
        # the server echoed on its last status response. ``poll`` deliberately
        # ignores hashes so the button is always actionable; the user lands
        # here when their geometry edits haven't been pushed yet.
        try:
            local_data = compute_data_hash(context)
        except (Cbor2NotInstalledError, ValueError) as e:
            # The panel's repair branch reads `com.error`, so a dialog the
            # artist dismisses would otherwise take the repair with it.
            com.set_error(str(e))
            popup_transfer_error(e)
            raise StageAbort(str(e))
        from ..core.facade import engine
        server_data = engine.state.server_data_hash
        if server_data and local_data != server_data:
            raise StageAbort(
                iface_(
                    "Geometry has changed since the last transfer. "
                    "Click \"Transfer\" to re-upload before running."
                )
            )

    def _stage_check_params(self, context):
        try:
            stop_for_edit(context)
        except TimelineBusyError as exc:
            raise StageAbort(str(exc))
        try:
            local_param = compute_param_hash(context)
        except (Cbor2NotInstalledError, ValueError) as e:
            # Same treatment as the geometry check above: a refusal here is
            # the artist's to act on, so it sets the panel error rather than
            # reaching the generic encode-failure fallback.
            com.set_error(str(e))
            raise StageAbort(str(e))
        from ..core.facade import engine
        server_param = engine.state.server_param_hash
        if server_param and local_param != server_param:
            raise StageAbort(
                iface_(
                    "Parameters have changed since the last transfer. "
                    "Click \"Update Params\" before running."
                )
            )

    def _stage_start_run(self, context):
        try:
            stop_for_edit(context)
        except TimelineBusyError as exc:
            raise StageAbort(str(exc))
        prepare_animation_targets(context, clear_existing=True)
        com.run(context)
        self.mark_remote_requested()

    def is_complete(self) -> bool:
        status = com.info.status
        return status not in (
            RemoteStatus.SIMULATION_IN_PROGRESS,
            RemoteStatus.STARTING_SOLVER,
        )


def _has_bone_checkpoint_authoring(context):
    """Include retained bindings and accepted playback after proxy deletion."""
    from ..core.rigid_pose import has_bone_pose_cache
    if has_bone_pose_cache(context.scene):
        return True
    return any(
        bool(settings.enabled) or settings.proxy is not None
        or any(getattr(settings, name, "")
               for name in ("bone_uuid", "body_uuid", "joint_uuid", "group_uuid"))
        for obj in context.scene.objects if obj.type == "ARMATURE"
        for bone in obj.data.bones
        for settings in [getattr(bone, "ppf_bone_physics", None)]
        if settings is not None
    )


def _validate_checkpoint_geometry(context):
    """Use the existing complete Data encoder gate, not a topology/AABB hint."""
    from ..core.facade import engine
    server_data = engine.state.server_data_hash
    if not server_data:
        raise ValueError("RESUME_DATA_IDENTITY_UNAVAILABLE: reconnect to the sealed session")
    if compute_data_hash(context) != server_data:
        raise ValueError(
            "RESUME_GEOMETRY_DRIFT: geometry changed; Transfer/Run a fresh generation")


def _latest_nonzero_checkpoint():
    frames = [frame for frame in com.saved_state_frames()
              if type(frame) is int and frame > 0]
    if not frames:
        raise ValueError("RESUME_TARGET_INVALID: nonzero checkpoint required")
    return max(frames)


class SOLVER_OT_Resume(AsyncOperator):
    """Resume the solver."""

    bl_idname = "solver.resume"
    bl_label = "Resume"

    timeout: float = 86400.0  # 24 hours for long simulations
    auto_redraw: bool = True

    @classmethod
    def poll(cls, _):
        if _geometry_edit_blocks(cls):
            return False
        if is_bake_running():
            return False
        status = com.info.status
        response = com.info.response
        # Resume stays available not only in the normal RESUMABLE state but also
        # after a failed simulation, as long as the server still holds at least
        # one saved checkpoint to load from. saved_state_frames() reads the
        # cached status response (no network), so this is cheap for poll().
        return (
            not com.busy()
            and not _staged_encode_active()
            and status in (RemoteStatus.RESUMABLE, RemoteStatus.SIMULATION_FAILED)
            and not response.get("resume_block_reason")
            and com.is_connected()
            and com.is_server_running()
            and com.info.status.ready()
            and not is_running(response)
            and any(f > 0 for f in com.saved_state_frames())
        )

    def execute(self, context):
        try:
            stop_for_edit(context)
        except TimelineBusyError as exc:
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}
        error = _check_project_name_sync(context)
        if error:
            self.report({"ERROR"}, error)
            return {"CANCELLED"}
        _warn_if_mesh_topology_stale(self, context)
        resume_kwargs = {}
        if _has_bone_checkpoint_authoring(context):
            from ..core.resume_horizon import prepare_request
            try:
                from_frame = _latest_nonzero_checkpoint()
                _validate_checkpoint_geometry(context)
                target_frame = prepare_request(context, com.info.response, from_frame)
            except (Cbor2NotInstalledError, ValueError) as e:
                com.set_error(str(e))
                self.report({"ERROR"}, str(e))
                return {"CANCELLED"}
            resume_kwargs = {"from_frame": from_frame, "target_frame": target_frame}
        com.resume(context, **resume_kwargs)
        self.setup_modal(context)
        self.mark_remote_requested()
        return {"RUNNING_MODAL"}

    def is_complete(self) -> bool:
        status = com.info.status
        return status not in (
            RemoteStatus.SIMULATION_IN_PROGRESS,
            RemoteStatus.STARTING_SOLVER,
        )


# A temporary read-only result bound to project generation; no file is rewritten.
_twist_resume_contract = None

class SOLVER_OT_TwistUpgradeQuery(AsyncOperator):
    """Check whether the selected solver can upgrade a saved legacy state."""
    bl_idname = "solver.twist_upgrade_query"
    bl_label = "连续累计扭转（旧缓存升级）"
    timeout = 30.0
    auto_redraw = True
    _origin_tracks_remote_root = False

    @classmethod
    def poll(cls, context):
        return SOLVER_OT_ResumeFrom.poll(context)

    def execute(self, context):
        self._stage_communicator = com
        try:
            self._query_id = com.query({'request':'twist_resume_contract'}, "检查连续扭转续算能力…")
            self.setup_modal(context)
        except (ValueError, RuntimeError) as exc:
            if getattr(self, '_query_id', None) is not None:
                com.cancel_query(self._query_id)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def is_complete(self):
        return com.query_outcome(self._query_id) is not None

    def is_cancelled(self):
        outcome = com.query_outcome(self._query_id)
        if outcome and outcome['error']:
            self.report({'ERROR'}, outcome['error'])
            return True
        return False

    def cleanup_modal(self, context):
        if getattr(self, '_query_id', None) is not None:
            com.cancel_query(self._query_id)
        AsyncOperator.cleanup_modal(self, context)

    def on_timeout(self, context):
        com.cancel_query(self._query_id)
        self.report({'ERROR'}, "能力查询超时，未开始续算")

    def on_complete(self, context):
        global _twist_resume_contract
        outcome = com.query_outcome(self._query_id)
        response = outcome.get('response', {}) if outcome else {}
        contract = response.get('twist_resume_contract')
        if response.get('twist_resume_error'):
            self.report({'ERROR'}, str(response['twist_resume_error']))
            return
        from ..core.resume_horizon import validate_contract
        try:
            validate_contract(response, contract, get_addon_data(context.scene).state.project_name)
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            return
        _twist_resume_contract = contract
        bpy.ops.solver.resume_from('INVOKE_DEFAULT', twist_upgrade=True)


class SOLVER_OT_ResumeFrom(AsyncOperator):
    """Resume the simulation from a chosen saved checkpoint.

    Opens a dialog listing the server's saved checkpoint frames. On
    confirm it guards against geometry/topology drift (a fresh run is
    required when geometry changed), warns when only materials changed
    (those will not apply on resume), re-encodes the parameters with a
    preserve-output build that keeps the already-simulated frames, and
    once the build completes resumes the solver from the chosen frame.
    Frames before the chosen point are kept; the rest are overwritten.
    """

    bl_idname = "solver.resume_from"
    bl_label = "Resume"
    twist_upgrade: BoolProperty(name="连续累计扭转", default=False, options={'HIDDEN', 'SKIP_SAVE'})

    timeout: float = 120.0
    auto_redraw: bool = True

    @classmethod
    def poll(cls, _):
        if _geometry_edit_blocks(cls):
            return False
        if is_bake_running():
            return False
        status = com.info.status
        response = com.info.response
        # Resume stays available not only in the normal RESUMABLE state but also
        # after a failed simulation, as long as the server still holds at least
        # one saved checkpoint to load from. saved_state_frames() reads the
        # cached status response (no network), so this is cheap for poll().
        return (
            not com.busy()
            and not _staged_encode_active()
            and status in (RemoteStatus.RESUMABLE, RemoteStatus.SIMULATION_FAILED)
            and not response.get("resume_block_reason")
            and com.is_connected()
            and com.is_server_running()
            and com.info.status.ready()
            and not is_running(response)
            and any(f > 0 for f in com.saved_state_frames())
        )

    def invoke(self, context, event):
        try:
            stop_for_edit(context)
        except TimelineBusyError as exc:
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}
        # Run the drift guards BEFORE opening the checkpoint picker, so the user
        # is told to Transfer / Update Params up front instead of after choosing
        # a checkpoint. Resume does NOT re-upload or rebuild; it continues the
        # simulation already on the server from the chosen checkpoint (a rebuild
        # happens only on "Transfer" or "Update Params"). So, like Run, refuse
        # when the live encoding has drifted from what the server last echoed: a
        # geometry change invalidates the cached state entirely (a fresh
        # Transfer + Run is required). A param change requires restoring the
        # original settings to resume, or updating them and running fresh.
        error = _check_project_name_sync(context)
        if error:
            self.report({"ERROR"}, error)
            return {"CANCELLED"}
        _warn_if_mesh_topology_stale(self, context)
        # The drift check re-encodes the scene on the main thread (the same
        # heavy work as Transfer). It runs before the checkpoint picker dialog
        # opens, so the panel's encode bar is not reachable from here; drive
        # the cursor's progress indicator instead so the click is not a silent
        # freeze. The encode is the fast path now (see the encoder perf fixes),
        # so this is brief.
        wm = context.window_manager
        wm.progress_begin(0.0, 2.0)
        try:
            wm.progress_update(0.0)
            try:
                local_data = compute_data_hash(context)
            except (Cbor2NotInstalledError, ValueError) as e:
                # See `_stage_check_geometry`: the panel's repair branch reads
                # `com.error`.
                com.set_error(str(e))
                popup_transfer_error(e)
                self.report({"ERROR"}, str(e))
                return {"CANCELLED"}
            wm.progress_update(1.0)
            try:
                from ..core.resume_horizon import prepare_request
                candidates=[f for f in com.saved_state_frames() if f>0]
                if not candidates:raise ValueError("RESUME_TARGET_INVALID: nonzero checkpoint required")
                check_response = dict(com.info.response)
                mode = None
                if self.twist_upgrade:
                    from ..core.resume_horizon import validate_contract, TWIST_MODE
                    validate_contract(check_response, _twist_resume_contract, get_addon_data(context.scene).state.project_name)
                    check_response['twist_resume_contract'] = _twist_resume_contract
                    mode = TWIST_MODE
                prepare_request(context,check_response,min(candidates),mode)
                local_param=check_response["param_hash"]
            except (Cbor2NotInstalledError, ValueError) as e:
                # Same treatment as the geometry refusal above: a parameter
                # refusal is just as actionable and must not surface as a
                # traceback with no panel error behind it.
                com.set_error(str(e))
                popup_transfer_error(e)
                self.report({"ERROR"}, str(e))
                return {"CANCELLED"}
            wm.progress_update(2.0)
        finally:
            wm.progress_end()
        from ..core.facade import engine
        server_data = engine.state.server_data_hash
        server_param = engine.state.server_param_hash
        if server_data and local_data != server_data:
            self.report({"ERROR"},
                        iface_("Geometry has changed; resume is not possible. Click "
                               "\"Transfer\" and \"Run\" for a fresh simulation."))
            return {"CANCELLED"}
        if server_param and local_param != server_param:
            self.report({"ERROR"},
                        iface_("Parameters have changed since the last transfer. Restore "
                               "the original parameters to resume, or click \"Update Params\" "
                               "and run a fresh simulation."))
            return {"CANCELLED"}
        state = get_addon_data(context.scene).state
        state.checkpoint_frames.clear()
        for frame in com.saved_state_frames():
            item = state.checkpoint_frames.add()
            item.frame = int(frame)
        state.checkpoint_frames_index = (
            len(state.checkpoint_frames) - 1 if len(state.checkpoint_frames) else -1
        )
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        state = get_addon_data(context.scene).state
        layout = self.layout
        if self.twist_upgrade:
            layout.label(text="从保存姿态开始计数，此前圈数无法恢复", icon="INFO")
        if len(state.checkpoint_frames):
            layout.label(text="Resume from saved checkpoint:")
            layout.template_list(
                "SOLVER_UL_CheckpointFrames", "",
                state, "checkpoint_frames",
                state, "checkpoint_frames_index",
                rows=4,
            )
        else:
            layout.label(text="No saved checkpoints available.", icon="ERROR")

    def execute(self, context):
        # Drift guards already ran in invoke() (before the picker opened); the
        # modal dialog blocks scene edits, so nothing can drift in between. Here
        # we only resolve the chosen checkpoint and resume.
        try:
            stop_for_edit(context)
        except TimelineBusyError as exc:
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}
        state = get_addon_data(context.scene).state
        frames = state.convert_checkpoint_frames_to_list()
        index = int(state.checkpoint_frames_index)
        if not frames or index < 0 or index >= len(frames):
            self.report({"ERROR"}, iface_("Select a checkpoint frame to resume from."))
            return {"CANCELLED"}
        from_frame = int(frames[index])
        from ..core.resume_horizon import prepare_request
        try:
            check_response = dict(com.info.response)
            mode = None
            if self.twist_upgrade:
                from ..core.resume_horizon import validate_contract, TWIST_MODE
                validate_contract(check_response, _twist_resume_contract, get_addon_data(context.scene).state.project_name)
                check_response['twist_resume_contract'] = _twist_resume_contract
                mode = TWIST_MODE
            target_frame=prepare_request(context,check_response,from_frame,mode)
        except ValueError as exc:
            self.report({"ERROR"},str(exc));return {"CANCELLED"}
        com.resume(context, from_frame=from_frame,target_frame=target_frame,twist_resume_mode=mode,
                   twist_resume_contract=_twist_resume_contract if mode else None)
        self.setup_modal(context)
        self.mark_remote_requested()
        return {"RUNNING_MODAL"}

    def is_complete(self) -> bool:
        status = com.info.status
        return status not in (
            RemoteStatus.SIMULATION_IN_PROGRESS,
            RemoteStatus.STARTING_SOLVER,
        )

    def is_cancelled(self) -> bool:
        return com.is_aborting()


class SOLVER_OT_UpdateParams(AsyncOperator):
    """Update the parameters of the solver."""

    bl_idname = "solver.update_params"
    bl_label = "Update Params on Remote"

    _mode = None
    timeout: float = 120.0
    auto_redraw: bool = True

    @classmethod
    def poll(cls, context):
        if _geometry_edit_blocks(cls):
            return False
        status = com.info.status
        response = com.info.response
        base = (
            (
                status == RemoteStatus.WAITING_FOR_BUILD
                or status
                in (
                    RemoteStatus.READY,
                    RemoteStatus.RESUMABLE,
                    RemoteStatus.SIMULATION_FAILED,
                )
            )
            and not _staged_encode_active()
            and com.is_connected()
            and com.is_server_running()
            and com.info.status.ready()
            and not is_running(response)
        )
        # Validation stays in execute; poll must not encode the scene.
        return base

    def execute(self, context):
        error = _check_project_name_sync(context)
        if error:
            self.report({"ERROR"}, error)
            return {"CANCELLED"}
        remote_root = com.normalized_remote_root()
        if not remote_root:
            self.report({"ERROR"}, iface_("Not connected; cannot send parameters"))
            return {"CANCELLED"}
        if _has_bone_checkpoint_authoring(context) and com.saved_state_frames():
            # A checkpoint owns its original sealed parameters and generation.
            # A horizon request is validated locally and sent only by Resume.
            from ..core.resume_horizon import prepare_request
            try:
                from_frame = _latest_nonzero_checkpoint()
                _validate_checkpoint_geometry(context)
                target_frame = prepare_request(context, com.info.response, from_frame)
                if target_frame is None:
                    raise ValueError(
                        "CHECKPOINT_PARAMETERS_IMMUTABLE: no horizon extension; "
                        "use Resume without Update Params, or Transfer/Run a fresh generation")
            except (Cbor2NotInstalledError, ValueError) as e:
                com.set_error(str(e))
                self.report({"ERROR"}, str(e))
                return {"CANCELLED"}
            self.report({"INFO"},
                        f"Resume target {target_frame + 1} samples validated locally; "
                        "click Resume. Remote sealed parameters remain unchanged.")
            return {"FINISHED"}
        try:
            _, param_data, _, param_hash = prepare_upload(
                context, want_data=False, want_param=True,
            )
        except (Cbor2NotInstalledError, ValueError) as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        # An identical update would replace the upload ID and drop the
        # native launch status needed to resume existing checkpoints.
        from ..core.facade import engine
        remote = engine.state
        if (com.info.status in (RemoteStatus.READY, RemoteStatus.RESUMABLE)
                and remote.remote_root.rstrip("/") == remote_root
                and remote.server_upload_id
                and param_hash
                and param_hash == remote.server_param_hash):
            self.report({"INFO"}, "参数未更改，保留当前解算结果。")
            return {"FINISHED"}
        # Upload param-only atomically + auto-build. data_size=0 tells
        # the server to keep the existing data.pickle untouched while a
        # fresh upload_id is minted against the new param. The param
        # hash rides along so the server's next status response echoes
        # the new fingerprint and ``poll`` re-disables this button. We
        # leave ``data_hash`` empty here: param-only uploads don't
        # touch the server's data fingerprint, so omitting it
        # preserves whatever's already on disk.
        com.build_pipeline(
            data=b"", param=param_data,
            param_hash=param_hash,
            preserve_output=True,
            message="Updating parameters...",
        )
        self._mode = "pipeline"
        self.setup_modal(context)
        self.mark_remote_requested()
        return {"RUNNING_MODAL"}

    def is_complete(self) -> bool:
        if self._mode != "pipeline":
            return False
        if com.busy():
            return False
        return com.info.status != RemoteStatus.BUILDING

    def is_cancelled(self) -> bool:
        return com.is_aborting()

    def on_complete(self, context):
        report_build_outcome(self)
        if (com.info.status == RemoteStatus.READY
                and com.info.response.get("resume_block_reason") == "CHECKPOINTS_FROM_PREVIOUS_BUILD"):
            self.report({"INFO"}, iface_("参数已更新，旧断点只保留供查看。请重新运行。"))


class SOLVER_OT_RestoreSingleBake(Operator):
    """Restore the source mesh, bone Actions and solver cache of a single bake."""
    bl_idname = "solver.restore_single_bake"
    bl_label = "Restore Simulation Before Single Bake"

    @classmethod
    def poll(cls, context):
        from ..core.single_bake_restore import available
        return available(context.scene) and not com.busy() and not is_bake_running()

    def execute(self, context):
        from ..core.single_bake_restore import restore
        from ..core.full_bake_restore import available as full_available
        try:
            was_full = full_available(context.scene)
            restore(context.scene)
        except (ValueError, RuntimeError, AttributeError, ReferenceError, KeyError, OSError) as exc:
            self.report({"ERROR"}, iface_(str(exc)))
            return {"CANCELLED"}
        self.report({"INFO"}, iface_("Simulation restored from before Full Animation Bake") if was_full else iface_("Simulation restored from before Single Frame Bake"))
        return {"FINISHED"}


class SOLVER_OT_DiscardSingleBakeRecovery(Operator):
    """Keep the baked result and stop offering the previous simulation restore."""
    bl_idname = "solver.discard_single_bake_recovery"
    bl_label = "Keep Single Bake and End Recovery"

    @classmethod
    def poll(cls, context):
        from ..core.single_bake_restore import available
        return available(context.scene) and not com.busy() and not is_bake_running()

    def execute(self, context):
        from ..core.single_bake_restore import discard
        from ..core.full_bake_restore import available as full_available
        try:
            was_full = full_available(context.scene)
            discard(context.scene)
        except (ValueError, RuntimeError, AttributeError, ReferenceError, KeyError, OSError) as exc:
            self.report({"ERROR"}, iface_(str(exc)))
            return {"CANCELLED"}
        self.report({"INFO"}, iface_("Full Animation Bake retained; recovery ended") if was_full else iface_("Single Bake retained; recovery ended"))
        return {"FINISHED"}


class SOLVER_OT_ClearAnimation(Operator):
    """Reset the position of the object."""

    bl_idname = "solver.clear_animation"
    bl_label = "Clear Local Animation"

    @classmethod
    def poll(cls, context):
        from ..core.single_bake_restore import available
        if available(context.scene):
            return False
        if is_bake_running():
            return False
        if com.busy() or com.animation.frame:
            return False
        response = com.info.response
        if is_running(response):
            return False
        # Recomputed on every redraw (and again via SOLVER_OT_Run.poll, which
        # gates on this result) -- stateless, so it can never go stale the way
        # a memoized flag can. scene_has_solver_cache answers for the same set
        # of objects execute() clears below, so a cache this button cannot
        # reach never keeps Run disabled.
        from ..core.pc2 import scene_has_solver_cache
        return scene_has_solver_cache(context.scene)

    def execute(self, context):
        try:
            from ..core.animation import preflight_clear_animation_ownership
            preflight_clear_animation_ownership(context)
            stop_for_edit(context)
        except (TimelineBusyError, ValueError, OSError) as exc:
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}
        # Retire queued frames and receipt verification before restoring author
        # channels. This original effect is local and retains the upload/seal.
        from ..core.effects import DoClearAnimation
        from ..core.facade import runner
        runner.execute(DoClearAnimation())
        from ..core.encoder import resolve_start_frame
        context.scene.frame_set(
            resolve_start_frame(get_addon_data(context.scene).state)
        )
        clear_animation_data(context)
        return {"FINISHED"}


class SOLVER_OT_RemoveStaleCaches(Operator):
    """Remove ContactSolverCache modifiers left on objects that belong to no
    group, which is what duplicating a simulated object leaves behind. The
    PC2 files stay on disk: a copy's modifier normally points at the
    original's cache, and deleting that would take the original's animation
    with it"""

    bl_idname = "solver.remove_stale_caches"
    bl_label = "Remove Stale Cache"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(find_orphan_solver_caches(context.scene))

    def execute(self, context):
        removed = [
            obj.name
            for obj in find_orphan_solver_caches(context.scene)
            if remove_orphan_solver_cache(obj)
        ]
        if not removed:
            self.report({"INFO"}, iface_("No stale caches to remove."))
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            iface_("Removed stale cache from {names}.").format(
                names=", ".join(removed)
            ),
        )
        redraw_all_areas(context)
        return {"FINISHED"}


def _recapture_pin_queue_identity(scene, specs):
    """Snapshot author identity, not only mutable collection indices."""
    from .dynamics.utils import get_group_from_index
    from ..models.groups import decode_vertex_group_identifier
    result = []
    for gi, pi in specs:
        group = get_group_from_index(scene, gi)
        if group is None or not (0 <= pi < len(group.pin_vertex_groups)):
            raise ValueError("Queued capture Pin no longer exists")
        pin = group.pin_vertex_groups[pi]
        result.append((gi, pi, group.as_pointer(), group.uuid, group.object_type,
                       pin.as_pointer(), pin.object_uuid, decode_stored_pin_identifier(pin.name)[1]))
    return tuple(result)


class SOLVER_OT_RecaptureAllDeformations(Operator):
    """Re-capture deformation for every deforming STATIC collider and every
    animated pin with one action, so you don't have to capture each one by hand
    before Transfer. When batching is enabled, one timeline pass samples both
    Static and Pin. Shows the actual capture mode, progress and an Abort button."""

    bl_idname = "solver.recapture_all_deformations"
    bl_label = "Re-capture All Deformations"

    _PHASE_STATIC = "STATIC"
    _PHASE_PIN = "PIN"

    @classmethod
    def poll(cls, context):
        # Disabled while any capture/bake is already in flight, and (the
        # "no capturable objects" case) when nothing across the active
        # groups needs a deformation capture. Cheap, draw-safe predicates
        # only: this runs on every panel redraw.
        if is_bake_running() or is_capture_running() or is_pin_capture_running():
            return False
        # Fast stateless poll: early-returns on the first capturable static
        # via the cached uuid resolver (no per-object name reconciliation),
        # instead of building the full list with resolve_assigned per redraw.
        from .dynamics.static_deform_ops import scene_has_capturable_static
        from .dynamics.pin_capture_ops import collect_capturable_pins
        return bool(
            scene_has_capturable_static(context)
            or collect_capturable_pins(context, cheap=True)
        )

    def execute(self, context):
        self._capture_cancelled = False
        self._capture_sampling_targets = None
        self._capture_ranges = None
        # Snapshot the UI choice once, so it cannot change an active capture.
        self._capture_frame_batch_requested = bool(
            getattr(context.window_manager, "ppf_capture_frame_batch", False)
        )
        self._capture_frame_batch_active = False
        self._joint_capture = None
        if self._capture_frame_batch_requested:
            if not bpy.data.filepath:
                self.report({"ERROR"}, iface_("Save the project before using mixed capture"))
                return {"CANCELLED"}
            try:
                from ..core.capture_provenance import _assert_group_idle
                from ..core.pc2 import get_pc2_dir
                _assert_group_idle(get_pc2_dir())
            except (OSError, RuntimeError, ValueError) as exc:
                self.report({"ERROR"}, str(exc))
                return {"CANCELLED"}
        from .dynamics.static_deform_ops import collect_capturable_static_objects
        from .dynamics.pin_capture_ops import collect_capturable_pins, validate_pin_capture_specs

        from ..core.capture_origin import remember_origin, require_origin
        self._capture_origin = {}
        remember_origin(self._capture_origin, context.scene)
        try:
            pin_specs = collect_capturable_pins(context)
            validate_pin_capture_specs(context, pin_specs)
        except (ReferenceError, RuntimeError, ValueError) as exc:
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}
        static_objs = collect_capturable_static_objects(context, allow_eval=True)
        try:
            require_origin(self._capture_origin, context)
        except RuntimeError as exc:
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}
        try:
            self._queued_pin_identity = _recapture_pin_queue_identity(context.scene, pin_specs)
        except (ReferenceError, RuntimeError, ValueError, IndexError) as exc:
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}
        from ..core.capture_origin import freeze_sampling_targets, freeze_capture_ranges
        from ..core.uuid_registry import get_object_by_uuid
        try:
            self._capture_sampling_targets = freeze_sampling_targets(
                context.scene, list(static_objs) + [
                    get_object_by_uuid(identity[6]) for identity in self._queued_pin_identity
                ])
            self._capture_ranges = freeze_capture_ranges(context.scene, [
                get_object_by_uuid(item['obj_uuid']) for item in self._capture_sampling_targets
            ])
        except RuntimeError as exc:
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}
        self._queue = []
        if static_objs:
            self._queue.append((self._PHASE_STATIC, static_objs))
        if pin_specs:
            self._queue.append((self._PHASE_PIN, pin_specs))
        if not self._queue:
            self._capture_sampling_targets = None
            self.report({"WARNING"}, iface_("No deformations to capture"))
            return {"CANCELLED"}
        self._phase = None
        self._summary = {"objs": 0, "pins": 0}
        self._bulk_display = {"scene": _bulk_capture_scene_key(context), "phases": {}}
        _bulk_capture_display.clear()
        _bulk_capture_display.update(self._bulk_display)
        if self._capture_frame_batch_requested:
            try:
                from .dynamics.capture_joint_schedule import JointCapture
                self._joint_capture = JointCapture(context, static_objs, pin_specs)
                self._joint_capture._timer_owner = self
                self._queue = []
                self._capture_frame_batch_active = True
                self._phase = "JOINT"
                for kind, _, _, _, rows in self._joint_capture.jobs:
                    self._bulk_display["phases"][kind] = {
                        "active": True, "batch": True, "reason": "", "targets": len(rows),
                        "objects": len({row["obj_uuid"] for row in rows}),
                        "frame": None, "frame_objects": 0,
                    }
                self._update_joint_capture_display()
            except Exception as exc:
                self._cleanup_active(context)
                self.report({"ERROR"}, iface_("Frame batch could not start: {error}").format(error=exc))
                return self._end_timer(context, {"CANCELLED"})
        elif not self._start_next(context):
            self._capture_sampling_targets = None
            if not self._capture_cancelled:
                self.report({"WARNING"}, iface_("Nothing to capture"))
            return {"CANCELLED"}
        wm = context.window_manager
        self._capture_wm = wm
        try:
            self._timer = wm.event_timer_add(0.01, window=context.window)
            wm.modal_handler_add(self)
        except Exception as exc:
            self._cleanup_active(context)
            self.report({"ERROR"}, iface_(str(exc)))
            return self._end_timer(context, {"CANCELLED"})
        redraw_all_areas(context)
        return {"RUNNING_MODAL"}

    def _start_next(self, context) -> bool:
        """Pop the next queued phase and start its job. Returns True once a
        job is running, False when the queue is exhausted (or the remaining
        phases all failed to start, each reported as a warning)."""
        from .dynamics import pin_capture_ops as pc
        from .dynamics import static_deform_ops as sd

        if getattr(self, '_capture_cancelled', False):
            self._queue = []
            self._phase = None
            return False
        from ..core.capture_origin import require_origin, require_capture_ranges
        try:
            require_origin(getattr(self, '_capture_origin', {}), context)
            require_capture_ranges(context.scene, getattr(self, '_capture_ranges', None) or ())
        except RuntimeError as exc:
            self._capture_cancelled = True
            self._queue = []
            self._phase = None
            self.report({"WARNING"}, str(exc))
            return False
        while self._queue:
            kind, payload = self._queue.pop(0)
            if kind == self._PHASE_PIN:
                try:
                    eligible = set(pc.collect_capturable_pins(context))
                    identity = _recapture_pin_queue_identity(context.scene, payload)
                    unchanged = (identity == self._queued_pin_identity
                                 and all(spec in eligible for spec in payload))
                except (ReferenceError, RuntimeError, ValueError, IndexError):
                    unchanged = False
                if not unchanged:
                    self._capture_cancelled = True
                    self._queue = []
                    self._phase = None
                    self.report({"WARNING"}, "Queued capture Pin changed; capture cancelled")
                    return False
            module = sd if kind == self._PHASE_STATIC else pc
            old_token = module._capture_job.get("token")
            try:
                if kind == self._PHASE_STATIC:
                    ok, err = sd.start_capture_for_objects(context, payload)
                else:
                    ok, err = pc.start_capture_for_pins(context, payload, strict=True)
            except Exception as exc:
                if not self._capture_frame_batch_requested:
                    raise
                self._capture_cancelled = True
                self._queue = []
                job = module._capture_job
                if job.get("active") and job.get("token") is not old_token and job.get("owner") is None:
                    self._capture_module = module
                    self._capture_token = job["token"]
                    job["owner"] = self
                    self._phase = kind
                    self._cleanup_active(context)
                self._phase = None
                self._bulk_display["phases"][kind] = {
                    "active": False, "batch": True, "reason": str(exc), "objects": 0,
                    "targets": 0, "frame": None, "frame_objects": 0,
                }
                self.report({"ERROR"}, iface_("Frame batch could not start: {error}").format(error=exc))
                return False
            if ok:
                self._capture_module = module
                self._capture_token = self._capture_module._capture_job["token"]
                self._capture_module._capture_job["owner"] = self
                self._phase = kind
                self._capture_frame_batch_active = False
                job = self._capture_module._capture_job
                entries = job["entries" if kind == self._PHASE_PIN else "objects"]
                phase = self._bulk_display["phases"][kind] = {
                    "active": True, "batch": self._capture_frame_batch_requested,
                    "reason": "", "targets": len(entries),
                    "objects": len({row["obj_uuid"] for row in entries}),
                    "frame": None, "frame_objects": 0,
                }
                if self._capture_frame_batch_requested:
                    try:
                        from .dynamics import capture_frame_batch
                        if not capture_frame_batch.admit(context, self._capture_module):
                            raise RuntimeError(iface_("No valid capture work to start"))
                        self._capture_frame_batch_active = True
                    except Exception as exc:
                        phase["reason"] = str(exc)
                        phase["active"] = False
                        self._capture_cancelled = True
                        self._queue = []
                        self._cleanup_active(context)
                        self.report({"ERROR"}, iface_("Frame batch could not start: {error}").format(error=exc))
                        return False
                return True
            if err:
                self.report({"WARNING"}, err)
            if kind == self._PHASE_PIN or self._capture_frame_batch_requested:
                self._capture_cancelled = True
                self._queue = []
                self._phase = None
                return False
        self._phase = None
        return False

    def _advance_phase(self, context, module, legacy_advance):
        job = module._capture_job
        before = job["total_frames_processed"]
        if self._capture_frame_batch_requested:
            if not self._capture_frame_batch_active:
                raise RuntimeError(iface_("Frame batch lost its active job"))
            from .dynamics import capture_frame_batch
            result = capture_frame_batch.tick(context, module)
        else:
            result = legacy_advance(context)
        if job["total_frames_processed"] > before:
            phase = self._bulk_display["phases"][self._phase]
            frame = context.scene.frame_current
            phase["frame"] = frame
            if phase["batch"]:
                entries = job["entries" if self._phase == self._PHASE_PIN else "objects"]
                phase["frame_objects"] = len({row["obj_uuid"] for row in entries
                                              if row["frame_start"] <= frame <= row["frame_end"]})
        if not result[0] or result[1]:
            self._bulk_display["phases"][self._phase]["active"] = False
        return result

    def _update_joint_capture_display(self):
        joint = self._joint_capture
        counters = {"STATIC": (0, 0), "PIN": (0, 0)}
        for kind, _, job, _, _ in joint.jobs:
            counters[kind] = (job["total_frames_processed"], job["total_frames"])
        sd, st = counters["STATIC"]
        pd, pt = counters["PIN"]
        self._bulk_display["joint"] = {
            "active": True, "frame": joint.seeded, "done": sd + pd, "total": st + pt,
            "st_done": sd, "st_total": st, "pin_done": pd, "pin_total": pt,
        }
        _bulk_capture_display["joint"] = self._bulk_display["joint"]

    def _modal_joint_capture(self, context, event):
        if event.type != "TIMER" or getattr(event, "timer", self._timer) != self._timer:
            return {"PASS_THROUGH"}
        joint = self._joint_capture
        try:
            more = joint.tick(context)
            self._update_joint_capture_display()
            if more:
                redraw_all_areas(context)
                return {"RUNNING_MODAL"}
            from ..core.capture_batch_commit import commit_batch_capture
            tasks = joint.tasks(context)
            commit_batch_capture(context.scene, tasks, recheck=lambda: joint.tasks(context))
            self._summary["objs"] = sum(kind == "STATIC" for kind, *_ in tasks)
            self._summary["pins"] = sum(kind == "PIN" for kind, *_ in tasks)
            self._cleanup_active(context)
            return self._finish(context)
        except Exception as exc:
            for phase in self._bulk_display["phases"].values():
                phase["reason"] = str(exc)
            try:
                self._cleanup_active(context)
            finally:
                self.report({"ERROR"}, iface_("Re-capture failed: {error}").format(error=exc))
                self._end_timer(context, {"CANCELLED"})
            redraw_all_areas(context)
            return {"CANCELLED"}

    def modal(self, context, event):
        joint = getattr(self, '_joint_capture', None)
        if joint is not None:
            return self._modal_joint_capture(context, event)
        from ..core.capture_origin import owns_capture_job
        module = getattr(self, '_capture_module', None)
        if module is None or not owns_capture_job(module._capture_job, getattr(self, '_capture_token', None)):
            self._phase = None
            self._queue = []
            return self._end_timer(context, {"CANCELLED"})
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        from .dynamics import pin_capture_ops as pc
        from .dynamics import static_deform_ops as sd

        try:
            if self._phase == self._PHASE_STATIC:
                more, aborted, err = self._advance_phase(context, sd, sd.advance_capture)
                if aborted:
                    return self._abort(context, err)
                if not more:
                    n_objs, _ = sd.finalize_capture(context)
                    self._summary["objs"] += n_objs
                    sd.cleanup_capture(context)
                    if not self._start_next(context):
                        return self._finish(context)
                redraw_all_areas(context)
                return {"RUNNING_MODAL"}
            elif self._phase == self._PHASE_PIN:
                more, aborted, err = self._advance_phase(context, pc, pc.advance_pin_capture)
                if aborted:
                    return self._abort(context, err)
                if not more:
                    n_pins, _ = pc.finalize_pin_capture(context)
                    self._summary["pins"] += n_pins
                    pc.cleanup_pin_capture(context)
                    if not self._start_next(context):
                        return self._finish(context)
                redraw_all_areas(context)
                return {"RUNNING_MODAL"}
            return self._finish(context)
        except Exception as exc:  # noqa: BLE001 — must restore state
            phase = self._bulk_display["phases"].get(self._phase)
            if phase:
                phase["reason"] = str(exc)
            self._cleanup_active(context)
            redraw_all_areas(context)
            self.report({"ERROR"}, iface_("Re-capture failed: {error}").format(error=exc))
            return self._end_timer(context, {"CANCELLED"})

    def _cleanup_active(self, context):
        joint = getattr(self, '_joint_capture', None)
        if joint is not None:
            self._joint_capture = None
            self._phase = None
            joint.close()
            return
        from ..core.capture_origin import owns_capture_job
        module = getattr(self, '_capture_module', None)
        if module is None or not owns_capture_job(module._capture_job, getattr(self, '_capture_token', None)):
            self._phase = None
            return
        from .dynamics import pin_capture_ops as pc
        from .dynamics import static_deform_ops as sd

        if self._phase == self._PHASE_STATIC:
            sd.cleanup_capture(context)
        elif self._phase == self._PHASE_PIN:
            pc.cleanup_pin_capture(context)
        self._phase = None

    def _abort(self, context, err):
        phase = self._bulk_display["phases"].get(self._phase)
        if phase and err:
            phase["reason"] = str(err)
        self._cleanup_active(context)
        redraw_all_areas(context)
        if err:
            self.report({"ERROR"}, err)
        else:
            self.report({"INFO"}, iface_("Re-capture aborted"))
        return self._end_timer(context, {"CANCELLED"})

    def _finish(self, context):
        if getattr(self, '_capture_cancelled', False):
            return self._end_timer(context, {"CANCELLED"})
        redraw_all_areas(context)
        parts = []
        if self._summary["objs"]:
            parts.append(iface_("{count} object(s)").format(count=self._summary["objs"]))
        if self._summary["pins"]:
            parts.append(iface_("{count} pin(s)").format(count=self._summary["pins"]))
        target = ", ".join(parts) if parts else iface_("nothing")
        self.report({"INFO"}, iface_("Re-captured deformation for {target}").format(target=target))
        return self._end_timer(context, {"FINISHED"})

    def _end_timer(self, context, retval):
        from ..core.capture_origin import detach_capture_timer
        self._capture_sampling_targets = None
        self._capture_ranges = None
        for phase in self._bulk_display["phases"].values():
            phase["active"] = False
        if self._bulk_display.get("joint"):
            self._bulk_display["joint"]["active"] = False
        detach_capture_timer(self, context)
        return retval

    def cancel(self, context):
        self._cleanup_active(context)
        self._end_timer(context, {"CANCELLED"})


class SOLVER_OT_ClearAllDeformations(Operator):
    """Clear this Scene's owned STATIC and Pin deformation input caches.

    Shared or unproven caches are retained. Authored geometry and keyframes
    remain intact; Re-capture All Deformations can rebuild the inputs."""

    bl_idname = "solver.clear_all_deformations"
    bl_label = "Clear All Deformations"
    bl_options = {"UNDO"}

    def invoke(self, context, event):
        # Capture completion can move this button under a late Abort click.
        # Confirm interactive invocations without changing EXEC_DEFAULT.
        self._clear_confirmation_origin = (context.scene.as_pointer(), bpy.data.filepath)
        return context.window_manager.invoke_confirm(
            self, event,
            title=iface_("Clear Captured Input Caches?"),
            message=iface_(
                "Delete this Scene's owned STATIC/Pin input caches? "
                "Shared or unproven caches are kept. "
                "Authored geometry and keyframes are kept."
            ),
            confirm_text=iface_("Clear Input Caches"),
            icon="WARNING",
        )

    @classmethod
    def poll(cls, context):
        # Disabled while a capture/bake is in flight, and (the "nothing to
        # clear" case) when no cache exists across the active groups.
        if is_bake_running() or is_capture_running() or is_pin_capture_running():
            return False
        # The static predicate checks ownership only for existing PC2 files
        # or in-memory arrays. Clear revalidates the caches when it runs.
        from ..core.pc2 import scene_has_static_deform_cache
        from .dynamics.pin_capture_ops import collect_pins_with_captured_anim
        return bool(
            scene_has_static_deform_cache(context.scene)
            or collect_pins_with_captured_anim(context)
        )

    def execute(self, context):
        origin = getattr(self, "_clear_confirmation_origin", None)
        if origin is not None and origin != (context.scene.as_pointer(), bpy.data.filepath):
            self.report({"WARNING"}, iface_("Scene or project changed; confirm clearing again"))
            return {"CANCELLED"}
        from ..core.pc2 import clear_orphan_deform_caches, remove_static_deform_pc2, static_deform_pc2_key
        from .dynamics.static_deform_ops import collect_objects_with_deform_cache
        from .dynamics.pin_capture_ops import (
            clear_captured_pin,
            collect_pins_with_captured_anim,
        )

        n_objs = 0
        n_refused = 0
        attempted_keys = set()
        for obj in collect_objects_with_deform_cache(context):
            attempted_keys.add(static_deform_pc2_key(obj))
            if remove_static_deform_pc2(obj, context.scene):
                n_objs += 1
            else:
                n_refused += 1
        # Snapshot the pin specs before clearing (clearing only flips flags /
        # removes files, so indices stay valid, but snapshotting is clearer).
        n_pins = 0
        for gi, pi in collect_pins_with_captured_anim(context):
            if clear_captured_pin(context, gi, pi, attempted_keys=attempted_keys):
                n_pins += 1
            else:
                n_refused += 1
        # Whatever is left in the cache directory belongs to no object this
        # pass could reach: deleted objects, or objects taken out of a group.
        # poll() gates on that directory, so leaving these would keep the
        # button enabled forever while every click cleared nothing.
        # Do not retry a refused live cache as an orphan behind its author state.
        # A later explicit Clear can retry; unrelated proven orphans still clear.
        n_orphans = clear_orphan_deform_caches(context.scene, exclude_keys=attempted_keys)
        if n_refused and not (n_objs or n_pins or n_orphans):
            self.report({"WARNING"}, iface_("Nothing cleared; {count} item(s) refused, remaining caches retained").format(count=n_refused))
            return {"CANCELLED"}

        # One overlay refresh + invalidation after the whole batch.
        from .dynamics.overlay import apply_object_overlays
        from ..models.groups import invalidate_overlays
        apply_object_overlays()
        invalidate_overlays()
        redraw_all_areas(context)

        parts = []
        if n_objs:
            parts.append(iface_("{count} object(s)").format(count=n_objs))
        if n_pins:
            parts.append(iface_("{count} pin(s)").format(count=n_pins))
        if n_orphans:
            parts.append(iface_("{count} orphaned cache(s)").format(count=n_orphans))
        target = ", ".join(parts) if parts else iface_("nothing")
        if n_refused:
            self.report({"WARNING"}, iface_("Cleared deformation cache for {target}; {count} item(s) retained because clearing was refused").format(target=target, count=n_refused))
        else:
            self.report({"INFO"}, iface_("Cleared deformation cache for {target}").format(target=target))
        return {"FINISHED"}


class SOLVER_OT_MigratePC2Folder(Operator):
    """Rename or copy data/<old>/ to match the current .blend basename
    and rewrite every ContactSolverCache modifier filepath in lockstep,
    so playback and overlay both read from one consistent location."""

    bl_idname = "solver.migrate_pc2_folder"
    bl_label = "Migrate PC2 Cache"
    bl_options = {"REGISTER", "UNDO"}

    keep_copy: BoolProperty(  # pyright: ignore
        name="Keep Copy",
        description="Preserve the old data/<name>/ folder as a backup "
        "(copy instead of rename)",
        default=True,
    )

    @classmethod
    def poll(cls, context):
        return _detect_pc2_basename_mismatch(context) is not None

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        layout = self.layout
        mismatch = _detect_pc2_basename_mismatch(context)
        if mismatch:
            old_name, new_name = mismatch
            col = layout.column(align=True)
            col.label(text=iface_("From:  data/{old_name}/").format(old_name=old_name), icon="FILE_FOLDER")
            col.label(text=iface_("To:    data/{new_name}/").format(new_name=new_name), icon="FORWARD")
        layout.prop(self, "keep_copy")

    def execute(self, context):
        mismatch = _detect_pc2_basename_mismatch(context)
        if mismatch is None:
            self.report({"ERROR"}, iface_("No migration target detected."))
            return {"CANCELLED"}
        old_name, new_name = mismatch
        blend_dir = os.path.dirname(bpy.data.filepath)
        old_dir = os.path.join(blend_dir, "data", old_name)
        new_dir = os.path.join(blend_dir, "data", new_name)

        # Capture exact source references before rename can change path resolution.
        rewrites = []
        source_parent = os.path.normcase(os.path.realpath(old_dir))
        for obj in bpy.data.objects:
            if obj.type != "MESH":
                continue
            mod = obj.modifiers.get(MODIFIER_NAME)
            if mod is None or not mod.filepath:
                continue
            abs_path = os.path.realpath(bpy.path.abspath(mod.filepath, library=mod.id_data.library))
            parent = os.path.dirname(abs_path)
            if os.path.normcase(parent) != source_parent:
                continue
            filename = os.path.basename(abs_path)
            rewrites.append((mod, filename))

        from ..core.pc2_migration import migrate_directory, copy_files_noreplace
        from ..core.cache_migration_recovery import apply_paths, validate_rewrites
        try:
            validate_rewrites(rewrites, context.scene)
            if os.path.isdir(new_dir) and os.listdir(new_dir):
                if not self.keep_copy:
                    raise ValueError('Move migration is disabled for safety; enable Keep Copy')
                if not rewrites:
                    raise ValueError('No mesh cache references to migrate')
                copy_files_noreplace(old_dir, new_dir, [name for _mod, name in rewrites])
            else:
                migrate_directory(old_dir, new_dir, self.keep_copy)
        except (OSError, ValueError) as exc:
            self.report({"ERROR"}, iface_("Migration failed: {error}").format(error=exc))
            return {"CANCELLED"}
        try:
            n_rewritten = apply_paths(rewrites, new_dir)
        except Exception as exc:
            self.report({"ERROR"}, iface_("Migration failed: {error}; copied data retained at {path}").format(error=exc, path=new_dir))
            return {"CANCELLED"}

        try:
            from ..models.groups import invalidate_overlays
            invalidate_overlays()
        except Exception:
            pass

        action = iface_("copied") if self.keep_copy else iface_("moved")
        self.report(
            {"INFO"},
            iface_("Migrated {count} modifier path(s); "
                   "data/{old_name}/ {action} to data/{new_name}/").format(
                count=n_rewritten, old_name=old_name, action=action, new_name=new_name),
        )
        return {"FINISHED"}


class SOLVER_OT_FetchData(AsyncOperator):
    """Fetch data on the remote server."""

    bl_idname = "solver.fetch_remote_data"
    bl_label = "Fetch All Animation"

    timeout: float = 600.0
    auto_redraw: bool = True

    @classmethod
    def poll(cls, _):
        if _geometry_edit_blocks(cls):
            return False
        if is_bake_running():
            return False
        if com.animation.frame:
            return False
        response = com.info.response
        frame = response.get("frame", -1)
        frame = int(frame) if frame is not None else -1
        status = com.info.status
        return (
            not com.busy()
            and frame > 0
            and status in (RemoteStatus.READY, RemoteStatus.RESUMABLE, RemoteStatus.SIMULATION_FAILED)
            and com.is_connected()
            and com.is_server_running()
            and com.info.status.ready()
            and not is_running(response)
        )

    def execute(self, context):
        if context.scene.get("ppf_single_bake_session_stale"):
            self.report({"ERROR"}, iface_("Single Frame Bake committed the author scene; transfer it again before fetching animation from a new session."))
            return {"CANCELLED"}
        try:
            stop_for_edit(context)
        except TimelineBusyError as exc:
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}
        _warn_if_mesh_topology_stale(self, context)
        from ..core.animation import validate_fetch_scene_membership
        try:
            validate_fetch_scene_membership(context.scene)
        except (ValueError, RuntimeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        com.fetch(context)
        from ..core.facade import engine
        self._fetch_request_id = engine.state.fetch_request_id
        self.setup_modal(context)
        return {"RUNNING_MODAL"}

    def is_complete(self) -> bool:
        return not com.busy() and len(com.animation.frame) == 0

    def on_complete(self, context):
        redraw_all_areas(context)
        from ..core.facade import engine
        state_result = engine.state
        if (state_result.fetch_request_id != getattr(self, "_fetch_request_id", -1)
                or state_result.fetch_outcome != "SUCCESS"):
            if state_result.fetch_outcome == "FAILED":
                self.report({"ERROR"}, state_result.error or iface_("Animation data fetch failed."))
            return
        from ..core.playback import is_scrubbing
        if is_scrubbing(context):
            self.report({"INFO"}, iface_("Animation data fetch finished."))
            return
        state = get_addon_data(context.scene).state
        with com._runner._anim_lock:
            start_frame = com._runner._anim_playback_origin
        if start_frame is None:
            self.report({"ERROR"}, iface_("The result has no verifiable playback origin."))
            return
        fetched = state.convert_fetched_frames_to_list()
        frame_count = max(fetched) if fetched else 0
        if frame_count > 0:
            context.scene.frame_end = frame_count + start_frame
            context.scene.frame_start = start_frame
        self.report({"INFO"}, iface_("Animation data fetch finished."))

        # Fetch installs playable frames; it must not restart animation or
        # move a playhead that the artist deliberately paused or scrubbed.


_delete_retry_scope = None


class SOLVER_OT_DeleteRemoteData(AsyncOperator):
    """Delete data on the remote server."""

    bl_idname = "solver.delete_remote_data"
    bl_label = "Delete Remote Data"

    timeout: float = 60.0
    auto_redraw: bool = True
    # Delete intentionally retires the server's project root.  Retain the
    # connection, project and session guards while allowing that transition.
    _origin_tracks_remote_root = False

    @classmethod
    def poll(cls, _):
        response = com.info.response
        if response is None:
            return False
        data_stat = response.get("data", None)
        # A partial filesystem failure can remove scene inputs but leave other
        # data. Only its confirmed failure permits NO_DATA retry in the same
        # connection, project, root and session.
        retry = _delete_retry_scope
        can_retry = bool(
            retry and retry[0] is com._runner
            and com._runner.connection_ticket_is_current(retry[1])
            and com.connection.remote_root == retry[2]
            and com.session_id == retry[3]
        )
        # status.in_progress() covers the local-side
        # window right after Run/Resume/Fetch is clicked; is_running
        # only flips once the remote echoes BUSY, which can lag a tick.
        return (
            not com.busy()
            and (data_stat == "READY" or (data_stat == "NO_DATA" and can_retry))
            and com.is_connected()
            and com.is_server_running()
            and com.info.status.ready()
            and not com.info.status.in_progress()
            and not is_running(response)
            and bool(com.connection.remote_root)
        )

    def execute(self, context):
        global _delete_retry_scope
        error = _check_project_name_sync(context)
        if error:
            self.report({"ERROR"}, error)
            return {"CANCELLED"}
        self._stage_communicator = com
        self._delete_root = com.connection.remote_root
        if not self._delete_root:
            self.report({"ERROR"}, "Remote project location is unavailable; refresh its status before deleting.")
            return {"CANCELLED"}
        self._delete_phase = "delete"
        self._delete_error = ""
        self._delete_retry_allowed = False
        self._delete_ticket = com._runner.capture_connection_ticket()
        self._delete_session = com.session_id
        try:
            self._delete_query_id = com.query(_DELETE_QUERY, _DELETE_MESSAGE)
        except (RuntimeError, ValueError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        _delete_retry_scope = None
        try:
            self.setup_modal(context)
        except Exception as exc:
            com.cancel_query(self._delete_query_id)
            self.report({"ERROR"}, "Remote delete was submitted, but its local wait failed: " + str(exc))
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}

    def is_complete(self) -> bool:
        global _delete_retry_scope
        observation = com.query_status(self._delete_query_id)
        if observation["state"] == "pending":
            return False
        if not observation["response_confirmed"] or not observation["context_current"]:
            error = observation.get("error") or "Remote delete result is unconfirmed."
            if (self._delete_phase == "delete" and observation["context_current"]
                    and error.startswith("DELETE_IO_FAILED:")):
                outcome = com.query_outcome(self._delete_query_id)
                self._delete_retry_allowed = bool(
                    outcome and outcome["response"].get("delete_confirmed") is False
                )
                self._delete_error = error
                try:
                    # A partial delete can retire the server's root even on
                    # failure. Re-select the project for a possible explicit
                    # retry, without submitting another delete.
                    self._delete_query_id = com.query({}, "Refreshing project after deletion...")
                except (RuntimeError, ValueError) as exc:
                    self._delete_error += " Project refresh failed: " + str(exc)
                    return True
                self._delete_phase = "refresh"
                return False
            self._delete_error = error
            return True
        outcome = com.query_outcome(self._delete_query_id)
        if outcome is None:
            return False
        if outcome["error"]:
            self._delete_error = outcome["error"]
            return True
        response = outcome["response"]
        if self._delete_phase == "delete":
            if response.get("delete_confirmed") is not True:
                self._delete_error = (
                    "The server did not confirm filesystem deletion. "
                    "Update the solver and check the project before retrying."
                )
                try:
                    self._delete_query_id = com.query({}, "Refreshing project after deletion...")
                except (RuntimeError, ValueError) as exc:
                    self._delete_error += " Project refresh failed: " + str(exc)
                    return True
                self._delete_phase = "refresh"
                return False
            if (response.get("root") != "" or response.get("data") != "NO_DATA"
                    or response.get("upload_id") != ""):
                self._delete_error = "Remote delete was not confirmed. Check the server before retrying."
                return True
            try:
                # The server creates a fresh project context on a normal
                # status query.  Never reuse the retired root from the ACK.
                self._delete_query_id = com.query({}, "Refreshing project after deletion...")
            except (RuntimeError, ValueError) as exc:
                self._delete_error = "Remote delete was acknowledged, but project refresh failed: " + str(exc)
                return True
            self._delete_phase = "refresh"
            return False
        if self._delete_error:
            if (self._delete_retry_allowed and response.get("root") == self._delete_root
                    and response.get("data") == "NO_DATA"
                    and com._runner.connection_ticket_is_current(self._delete_ticket)
                    and com.session_id == self._delete_session):
                _delete_retry_scope = (
                    com._runner, self._delete_ticket, self._delete_root, self._delete_session
                )
            return True
        if (response.get("root") != self._delete_root
                or response.get("data") != "NO_DATA"
                or response.get("upload_id") != ""):
            self._delete_error = (
                "Remote deletion could not be confirmed after project refresh. "
                "Check the server data before retrying."
            )
        return True

    def on_complete(self, context):
        redraw_all_areas(context)
        if self._delete_error:
            self.report({"ERROR"}, self._delete_error)
            return
        self.report({"INFO"}, iface_("Remote data deleted successfully."))

    def cleanup_modal(self, context):
        query_id = getattr(self, "_delete_query_id", None)
        if query_id is not None:
            observation = com.query_status(query_id)
            if observation["state"] == "pending" and observation["context_current"]:
                com.cancel_query(query_id)
        super().cleanup_modal(context)


classes = (
    SOLVER_PT_SolverPanel,
    SOLVER_OT_RecoverBoneConnections,
    SOLVER_OT_Transfer,
    SOLVER_OT_Run,
    SOLVER_OT_Resume,
    SOLVER_OT_ResumeFrom,
    SOLVER_OT_TwistUpgradeQuery,
    SOLVER_OT_ClearAnimation,
    SOLVER_OT_RestoreSingleBake,
    SOLVER_OT_DiscardSingleBakeRecovery,
    SOLVER_OT_RemoveStaleCaches,
    SOLVER_OT_RecaptureAllDeformations,
    SOLVER_OT_ClearAllDeformations,
    SOLVER_OT_MigratePC2Folder,
    SOLVER_OT_UpdateParams,
    SOLVER_OT_DeleteRemoteData,
    SOLVER_OT_FetchData,
)


def register():
    bpy.types.WindowManager.ppf_capture_frame_batch = BoolProperty(
        name="Batch capture by frame",
        description=(
            "On: sample Static and Pin in one timeline pass; errors stop capture. "
            "Off: sample sequentially. Stateful scenes may differ; keep a project copy"
        ),
        default=False,
        options={"SKIP_SAVE"},
    )
    for cls in classes:
        _ppf_i18n.register_class(cls)


def unregister():
    _bulk_capture_display.clear()
    del bpy.types.WindowManager.ppf_capture_frame_batch
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
