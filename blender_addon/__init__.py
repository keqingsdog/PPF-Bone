# File: __init__.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
# Fork modifications: register bone physics, author-follow and related extension features.

def _bootstrap_cbor2():
    """Load the bundled dependency without retaining a global search path."""
    import importlib
    import platform
    import sys
    import sysconfig
    from pathlib import Path

    vendor = Path(__file__).resolve().parent / '_vendor'
    if not vendor.is_dir():
        return  # Source and single-ABI packages use their existing dependency path.
    abi = 'cp%d%d' % sys.version_info[:2]
    if sys.platform == 'win32':
        # Select the running Python's platform even under Windows emulation.
        tag = sysconfig.get_platform().replace('-', '_')
    else:
        arch = platform.machine().lower()
        tag = {
            ('linux', 'x86_64'): 'manylinux_2_28_x86_64',
            ('linux', 'aarch64'): 'manylinux_2_28_aarch64',
            ('darwin', 'arm64'): 'macosx_11_0_arm64',
        }.get((sys.platform, arch))
    dependency = vendor / abi / (tag or 'unsupported')
    if not dependency.is_dir():
        raise ImportError(f'PPF has no bundled cbor2 for {abi} on {sys.platform}/{tag or platform.machine()}')
    cbor2 = sys.modules.get('cbor2')
    if cbor2 is None:
        if sys.platform == 'win32':
            # Native DLLs loaded from the extension itself prevent replacing it.
            import bpy
            import os
            import shutil
            import tempfile

            user_root = bpy.utils.extension_path_user(__package__, path='dependencies', create=True)
            if not user_root:
                raise ImportError('PPF cannot create its extension dependency directory')
            root = Path(user_root)
            cached = root / f'cbor2-6.0.1-{abi}-{tag}'
            if not cached.is_dir():
                temporary = Path(tempfile.mkdtemp(prefix='.cbor2-', dir=root))
                try:
                    shutil.copytree(dependency / 'cbor2', temporary / 'cbor2')
                    shutil.copytree(dependency / 'cbor2-6.0.1.dist-info', temporary / 'cbor2-6.0.1.dist-info')
                    try:
                        os.rename(temporary, cached)
                    except OSError:
                        if not cached.is_dir():
                            raise
                finally:
                    if temporary.exists():
                        shutil.rmtree(temporary)
            dependency = cached
        path = str(dependency)
        sys.path.insert(0, path)
        try:
            cbor2 = importlib.import_module('cbor2')
        finally:
            sys.path.remove(path)
    try:
        if cbor2.loads(cbor2.dumps({'ppf': 1})) != {'ppf': 1}:
            raise ValueError('roundtrip mismatch')
    except Exception as exc:
        raise ImportError('Existing cbor2 is incompatible with PPF; restart Blender and try again') from exc

_bootstrap_cbor2()

from .core.pin_reference import decode_stored_pin_identifier, PinReferenceError

from bpy.app.handlers import persistent  # pyright: ignore

from .ui import main_panel, console, solver, state
from .ui import dynamics  # Import the dynamics package
from . import mesh_ops  # Import the mesh_ops package (snap + merge only)
from . import i18n  # Import the localization package (bpy.app.translations catalogs)
from .ops import zozo_contact_solver

reload_server = None

# Captured while Blender RNA is alive; the exit hook never imports or dispatches.
_exit_detach = None

# One-shot timers scheduled during register() that we must be able to
# deregister on unregister() so they can't fire into a torn-down
# module namespace after a reload that happens inside their delay.
_deferred_timers: list = []


@persistent
def _abort_capture_before_save(*_args):
    """Save stable author state rather than a capture job's temporary state."""
    from .core.encoder import cancel_upload_steps
    cancel_upload_steps()
    from .core.encoder.transfer_bone_stage import cancel as cancel_sliced
    cancel_sliced()
    from .core.capture_origin import abort_before_save
    from .ui.dynamics.bone_capture import cancel_active
    cancel_active()
    from .ui.dynamics import static_deform_ops, pin_capture_ops
    for module in (static_deform_ops, pin_capture_ops):
        abort_before_save(module._capture_job)


@persistent
def _cancel_captures_before_change(*_args):
    from .core.encoder import cancel_upload_steps
    cancel_upload_steps()
    from .core.capture_origin import cancel_capture_jobs
    cancel_capture_jobs()
    from .core.encoder.transfer_bone_stage import cancel as cancel_sliced
    cancel_sliced()


@persistent
def _cancel_captures_before_undo(*_args):
    from .core.encoder import cancel_upload_steps
    cancel_upload_steps()
    from .core.capture_origin import cancel_capture_jobs
    cancel_capture_jobs(restore=False)
    from .core.encoder.transfer_bone_stage import cancel as cancel_sliced
    cancel_sliced(restore=False)


@persistent
def _persist_session_id(*_args):
    """Preserve source sessions offline; stamp only proven producer scenes online.

    The saved id identifies the session that produced a scene's mesh caches,
    not an unrelated connection that happened to be active during Save As.
    """
    import bpy  # pyright: ignore
    try:
        from .core import facade
        from .models.groups import get_addon_data, has_addon_data
        sid = facade.engine.state.session_id or ""
        if not sid:
            return  # Offline Save As must preserve the cached producer identity.
        from .core.pc2 import mesh_output_matches_session
        for scene in bpy.data.scenes:
            if not has_addon_data(scene):
                continue
            # A connection alone does not claim every scene's existing caches.
            if not any(mesh_output_matches_session(scene, obj, sid)
                       for obj in scene.objects if obj.type == "MESH"):
                continue
            root = get_addon_data(scene)
            if hasattr(root, "state") and root.state.last_session_id != sid:
                root.state.last_session_id = sid
    except Exception:
        pass


@persistent
def _save_manifest_post(*_args):
    """save_post hook: write the project manifest in the PPF data folder.

    Runs AFTER migrate_pc2_on_save so the PC2 inventory reflects the
    final on-disk file set. Manifest failure is advisory, but reported.
    """
    try:
        from .core.manifest import write_manifest_now
        write_manifest_now()
    except Exception as exc:
        print(f"[PPF manifest] Could not write project manifest: {exc}")


def _safe_disconnect():
    """Drop any active server connection, swallowing all errors.

    Used by load_pre while Blender RNA is alive. Interpreter shutdown
    must use the separate Python-only detach path.
    """
    try:
        from .core.facade import communicator
        communicator.disconnect()
    except Exception:
        pass


@persistent
def _disconnect_on_load(*_args):
    """load_pre hook: drop any active server connection before the new
    .blend takes over.

    The communicator and engine are module-level singletons that
    survive ``wm.read_homefile`` / ``wm.open_mainfile``; without this
    hook the new scene inherits the previous project's live socket and
    ONLINE phase, even though its own scene-level state defaults to
    OFFLINE. The result is an inconsistency where the UI thinks it's
    still connected to a server bound to a different project.
    Dispatching DisconnectRequested here clears engine.state and tears
    down ``runner._backend`` so the new scene starts with a clean
    slate.

    Marked ``@persistent`` so the handler survives file loads: Blender
    wipes non-persistent handlers each time a new file is opened, which
    would silently disable this disconnect on every load after the
    first one.
    """
    _safe_disconnect()


def _disconnect_at_exit():
    """Detach local Python I/O only; Blender may already have freed its RNA.

    No communicator/engine dispatch, author snapshot, new import or remote
    termination belongs here. register() captured the live runner beforehand.
    """
    callback = _exit_detach
    if callback is not None:
        try:
            callback()
        except Exception:
            pass


@persistent
def _reconcile_manifest_on_load(*_args):
    """load_post hook: auto-migrate legacy data, then reconcile manifest.

    Auto-migration runs BEFORE the first depsgraph update so code paths
    that now require UUIDs (merge_ops cleanup, encoder, overlay) don't
    silently drop un-migrated data.
    """
    # Renamed properties first: a stranded legacy key (e.g. the pre-rename fps
    # flag) otherwise leaves the scene reading a default that silently changes
    # its simulation time base. The retired Local connection is the same kind of
    # stranding through an enum item number rather than a key.
    try:
        from .core.user_defaults import initialize_scene
        import bpy
        for scene in bpy.data.scenes:
            initialize_scene(scene, legacy_loaded=True)
    except Exception:
        pass
    try:
        from .core.migrate import needs_migration, migrate_legacy_data
        if needs_migration():
            result = migrate_legacy_data()
            from .models.console import console
            console.write(f"[auto-migrate] {result}")
    except Exception:
        pass
    try:
        # The addon's own scene-parameter keyframe list is superseded by real
        # F-curves on the sliders. Convert on load so a saved scene keeps
        # working and the artist sees the curves on the timeline where every
        # other keyframe lives.
        import bpy  # pyright: ignore
        from .core.migrate_dyn_params import (
            convert_legacy_dyn_params,
            count_legacy_dyn_params,
        )
        if count_legacy_dyn_params(bpy.context.scene):
            summary = convert_legacy_dyn_params(bpy.context.scene)
            if summary:
                from .models.console import console
                console.write(f"[auto-migrate] {summary}")
    except Exception:
        pass
    try:
        from .core.manifest import reconcile_on_load
        reconcile_on_load()
    except Exception:
        pass


@persistent
def _request_cache_heal_on_load(*_args):
    """load_post hook: owe the frame pump one MESH_CACHE heal pass.

    Blender cancels every modal operator when a file is loaded, and the
    pump is not resident, so nothing would otherwise notice that the
    freshly opened scene may carry a ContactSolverCache to rebind. The
    request only sets a flag; the pump starts on the next persistent
    tick and does the work in the modal-operator context those ID writes
    require.
    """
    try:
        from .core.frame_pump import request_heal
        request_heal()
    except Exception:
        pass


@persistent
def _reconcile_pin_captured_anim_on_load(*_args):
    """load_post hook: sync ``pin_item.has_captured_anim`` flags with
    on-disk ``_pindeform.pc2`` files.

    The bool is the source of truth at runtime (cheap to read in
    panel polls and UIList rows), but it can desync across machines:
    a .blend saved with the cache present may open on a host where
    the cache file was never copied. This handler walks every pin in
    the scene, flips ``has_captured_anim`` to match disk presence,
    and drops the EMBEDDED_MOVE sentinel if the cache vanished and no
    manual fcurves remain.
    """
    import bpy  # pyright: ignore
    try:
        from .core.pc2 import has_pin_anim_pc2
        from .core.uuid_registry import resolve_pin
        from .models.groups import (
            decode_vertex_group_identifier,
            iterate_active_object_groups,
        )
        from .ui.dynamics.pin_capture_ops import (
            _pin_has_vertex_co_fcurves,
        )
        from .ui.dynamics.pin_ops import _remove_embedded_move_ops

        for scene in bpy.data.scenes:
            for group in iterate_active_object_groups(scene):
                if group.object_type == "STATIC":
                    continue
                for pin_item in group.pin_vertex_groups:
                    try:
                        obj = resolve_pin(pin_item)
                        _, vg_name = decode_stored_pin_identifier(pin_item.name)
                        manual_has = _pin_has_vertex_co_fcurves(pin_item)
                    except ValueError:
                        continue
                    if obj is None or obj.type != "MESH":
                        continue
                    if not vg_name:
                        continue
                    on_disk = has_pin_anim_pc2(obj, vg_name)
                    pin_item.has_captured_anim = bool(on_disk)
                    if not on_disk and not manual_has:
                        # Cache vanished AND no manual fcurves to claim
                        # the EMBEDDED_MOVE sentinel; drop it so the
                        # encoder doesn't trip "more than one source"
                        # checks on a stale marker.
                        _remove_embedded_move_ops(pin_item)
    except Exception:
        pass


def _register_deferred_timer(fn, first_interval: float) -> None:
    import bpy  # pyright: ignore
    bpy.app.timers.register(fn, first_interval=first_interval)
    _deferred_timers.append(fn)


def _clear_deferred_timers() -> None:
    import bpy  # pyright: ignore
    for fn in _deferred_timers:
        if bpy.app.timers.is_registered(fn):
            try:
                bpy.app.timers.unregister(fn)
            except ValueError:
                pass
    _deferred_timers.clear()


def register():
    global reload_server
    # A refused controlled reload leaves the existing owner retired. A plain
    # enable must not revive it or partially register classes before failing.
    from .core.facade import runner
    runner.assert_reload_available()
    try:
        from .core import pin_picker_tokens
        pin_picker_tokens.begin_registration()
        _register_body()
        pin_picker_tokens.register()
        # Install last so admission reopens only after our load reconciliation.
        from .mcp.task_system import register_context_handlers
        register_context_handlers()
    except Exception:
        # Partial registration would leave Blender with half-set-up
        # classes and scheduled timers that Blender's addon framework
        # won't tear down. Clean up on our own before re-raising so
        # the user gets a working "disabled" state.
        try:
            unregister()
        except Exception as e:
            print(f"Rollback unregister failed: {e}")
        raise


def _warn_if_cbor2_missing() -> None:
    """Print a console warning when the bundled cbor2 wheel is not installed.

    Blender installs the wheels named in ``blender_manifest.toml`` when it
    installs the extension, and it does NOT refuse to enable one whose wheel
    files are absent: an add-on directory placed under
    ``extensions/user_default`` by hand, or copied from a checkout whose
    gitignored ``wheels/*.whl`` were never fetched, enables normally and
    registers every operator. The first sign of trouble then arrives much
    later, from a Transfer that cannot encode, with nothing connecting it to
    the install. Saying it here puts the diagnosis at the moment the
    condition is created.

    This warns rather than raising. Raising would leave the add-on disabled,
    and the recovery the user needs, the "Install cbor2 to Add-on Directory"
    button, lives in the add-on's own panel.
    """
    from .core.module import CBOR2_MISSING_MESSAGE, cbor2_available
    if cbor2_available():
        return
    print(f"ZOZO's Contact Solver: {CBOR2_MISSING_MESSAGE}")


def _register_body():
    global reload_server
    # Register translation catalogs first so the very first UI draw is
    # already localized when the user has interface translation enabled.
    i18n.register()
    state.register()
    from .ui import layout
    layout.register()
    main_panel.register()
    solver.register()
    dynamics.register()
    from .models import bone_bindings
    from .ops import bone_proxy
    from .ui import bone_physics
    from .core import rigid_pose
    bone_bindings.register()
    bone_proxy.register()
    from .ops import bone_rig_duplicate
    bone_rig_duplicate.register()
    bone_physics.register()
    from .models import cross_links as cross_link_model
    from .ops import cross_links as cross_link_ops
    from .ops import cross_link_presets
    cross_link_model.register()
    cross_link_presets.register()
    cross_link_ops.register()
    from .ops import mmd_physics_import, bone_selection
    mmd_physics_import.register()
    bone_selection.register()
    from .models import restoring_curve_rna
    from .ops import restoring_curve_operators
    from .ui import restoring_curves
    restoring_curve_rna.register()
    restoring_curve_operators.register()
    from .models import bone_chain_presets as bone_chain_preset_model
    from .ops import bone_chain_presets as bone_chain_preset_ops
    bone_chain_preset_model.register()
    bone_chain_preset_ops.register()
    from .ops import bone_presets
    bone_presets.register()
    restoring_curves.register()
    from .ui import bone_collision_ui
    bone_collision_ui.register()
    from .ui import group_collision
    group_collision.register()
    rigid_pose.register()
    from .core import bone_author_follow
    bone_author_follow.register()
    mesh_ops.register()
    from .mcp.handlers import _bone_context
    _bone_context.register_undo_operator()
    console.register()
    zozo_contact_solver.register()

    _warn_if_cbor2_missing()

    # Start the persistent engine timer
    from .core.facade import ensure_engine_timer
    ensure_engine_timer()

    # Migrate existing groups to UUID system
    import bpy  # pyright: ignore

    # Clean up any existing groups with messy names on startup
    def cleanup_group_names():
        mutated = False
        try:
            if hasattr(bpy.context, "scene") and bpy.context.scene:
                from .models.groups import iterate_active_object_groups

                scene = bpy.context.scene

                # Clean up active groups using display indices (1, 2, 3, 4, 5)
                for display_index, group in enumerate(
                    iterate_active_object_groups(scene)
                ):
                    # Ensure UUID exists
                    if not group.uuid:
                        group.ensure_uuid()
                    # Clean up messy names using display index
                    if not group.name or "Group Group" in group.name:
                        group.name = f"Group {display_index + 1}"
                        mutated = True
                        from .models.console import console
                        console.write(
                            f"Cleaned up group name to: {group.name}"
                        )
        except Exception as e:
            print(f"Group cleanup failed: {e}")
        if mutated:
            try:
                from .core.utils import redraw_all_areas
                redraw_all_areas(bpy.context)
            except Exception:
                pass
        return None  # Remove timer

    _register_deferred_timer(cleanup_group_names, 1.0)

    # Register UUID rename detection handler
    from .core.uuid_registry import register as uuid_register
    uuid_register()

    # Register save_post handler for PC2 temp→data migration. Compare
    # existing handlers by __name__ rather than identity so any stale
    # handler from an earlier module generation (pre-reload) gets
    # detected and isn't double-added.
    from .core.pc2 import (
        ensure_curve_handler,
        migrate_pc2_on_save,
        prepare_mesh_cache_save,
        reset_render_counter,
        rollback_mesh_cache_save,
        warn_missing_frames_on_render,
    )

    if not any(
        getattr(h, "__name__", "") == "migrate_pc2_on_save"
        for h in bpy.app.handlers.save_post
    ):
        bpy.app.handlers.save_post.append(migrate_pc2_on_save)

    if not any(
        getattr(h, "__name__", "") == "reset_render_counter"
        for h in bpy.app.handlers.render_init
    ):
        bpy.app.handlers.render_init.append(reset_render_counter)
    if not any(
        getattr(h, "__name__", "") == "warn_missing_frames_on_render"
        for h in bpy.app.handlers.render_pre
    ):
        bpy.app.handlers.render_pre.append(warn_missing_frames_on_render)

    if not any(getattr(h, "__name__", "") == "_abort_capture_before_save"
               for h in bpy.app.handlers.save_pre):
        bpy.app.handlers.save_pre.append(_abort_capture_before_save)
    from .core.encoder import captured_bone_motion
    captured_bone_motion.register()
    for handlers, callback in (
        (bpy.app.handlers.save_pre, prepare_mesh_cache_save),
        (bpy.app.handlers.save_post_fail, rollback_mesh_cache_save),
    ):
        if not any(getattr(h, "__name__", "") == callback.__name__ for h in handlers):
            handlers.append(callback)
    for handlers, callback in ((bpy.app.handlers.load_pre, _cancel_captures_before_change),
                               (bpy.app.handlers.undo_pre, _cancel_captures_before_undo),
                               (bpy.app.handlers.redo_pre, _cancel_captures_before_undo)):
        if not any(getattr(h, "__name__", "") == callback.__name__ for h in handlers):
            handlers.append(callback)

    # Session-id persistence: write the live engine's session id into the
    # scene PropertyGroup just before the user saves, so the id survives
    # close/reopen and can be compared against the remote server to
    # detect orphaned sims on reconnect.
    if not any(
        getattr(h, "__name__", "") == "_persist_session_id"
        for h in bpy.app.handlers.save_pre
    ):
        bpy.app.handlers.save_pre.append(_persist_session_id)

    # Manifest lifecycle: save_post writes into data/<blend-name> only
    # after actual PPF data exists; load_post reads legacy adjacent manifests
    # without migrating them, then reconciles against the current version.
    if not any(
        getattr(h, "__name__", "") == "_save_manifest_post"
        for h in bpy.app.handlers.save_post
    ):
        bpy.app.handlers.save_post.append(_save_manifest_post)
    if not any(
        getattr(h, "__name__", "") == "_reconcile_manifest_on_load"
        for h in bpy.app.handlers.load_post
    ):
        bpy.app.handlers.load_post.append(_reconcile_manifest_on_load)
    if not any(
        getattr(h, "__name__", "") == "_reconcile_pin_captured_anim_on_load"
        for h in bpy.app.handlers.load_post
    ):
        bpy.app.handlers.load_post.append(_reconcile_pin_captured_anim_on_load)
    if not any(
        getattr(h, "__name__", "") == "_request_cache_heal_on_load"
        for h in bpy.app.handlers.load_post
    ):
        bpy.app.handlers.load_post.append(_request_cache_heal_on_load)
    if not any(
        getattr(h, "__name__", "") == "_disconnect_on_load"
        for h in bpy.app.handlers.load_pre
    ):
        bpy.app.handlers.load_pre.append(_disconnect_on_load)

    # atexit: cover Blender shutdown (no quit_pre handler exists). Idempotent
    # under register/unregister churn — atexit dedupes on the function object.
    import atexit as _atexit
    from functools import partial
    from ._shutdown import detach_runner
    from .core.facade import runner as _exit_runner
    global _exit_detach
    _exit_detach = partial(detach_runner, _exit_runner)
    _atexit.unregister(_disconnect_at_exit)
    _atexit.register(_disconnect_at_exit)

    # Register curve playback handler (persistent — survives file loads).
    # No pin handler: pin animation is driven by native vertex-co
    # fcurves through Blender's own animation system; an extra
    # frame_change_post handler that touches obj.data.vertices during
    # playback corrupts the heap in Blender 5.x.
    ensure_curve_handler()

    # Note: MESH_CACHE self-heal is driven by the PPF_OT_FramePump modal
    # (registered further down), not from here — ID writes from addon
    # register context are unreliable on Blender 5.x when the reload
    # originates from the debug-server TCP handler.

    # Restart servers that were running before a manual reload
    def _restart_servers_after_reload():
        from .core.reload_server import get_restart_server_state, ReloadServer, get_reload_server_status
        state = get_restart_server_state()
        if state is None:
            return None  # fresh launch, not a reload
        global reload_server
        if state.get("debug") and not get_reload_server_status():
            reload_server = ReloadServer()
            reload_server.start()
        if state.get("mcp"):
            try:
                from .mcp.mcp_server import start_mcp_server, is_mcp_running
                from .models.defaults import DEFAULT_MCP_PORT
                if not is_mcp_running():
                    start_mcp_server(DEFAULT_MCP_PORT)
            except Exception as e:
                print(f"MCP restart after reload failed: {e}")
        # Kick the 3D-view panels to redraw so the Start/Stop button
        # reflects the just-restarted server without a manual hover.
        try:
            from .core.utils import redraw_all_windows
            redraw_all_windows("VIEW_3D")
        except Exception:
            pass
        return None

    _register_deferred_timer(_restart_servers_after_reload, 1.0)

    # Frame-pump modal: drives apply_animation + MESH_CACHE self-heal
    # from a modal-operator timer context (the only context where
    # Blender 5.x permits the ID writes those involve).
    from .core import frame_pump
    frame_pump.register()

    from .core import user_defaults
    user_defaults.register()

    # Everything above succeeded — tell the persistent tick it's safe to
    # read/modify PropertyGroup state. Must be the final line of register().
    try:
        from .core.facade import mark_addon_ready
        mark_addon_ready(True)
    except Exception as e:
        print(f"mark_addon_ready failed: {e}")


def unregister():
    from .core.encoder import cancel_upload_steps
    cancel_upload_steps()
    from .core import rigid_pose as _rigid_pose_recovery
    _rigid_pose_recovery.export_pending_connection_recoveries()
    import bpy
    global reload_server

    from .mcp.task_system import unregister_context_handlers
    unregister_context_handlers()

    try:
        from .core import pin_picker_tokens
        pin_picker_tokens.unregister()
    except Exception as exc:
        print("Pin picker token unregister failed; continuing addon cleanup:", exc)

    from .core import user_defaults
    user_defaults.unregister()
    from .core.bone_proxy_outliner import cancel_pending
    cancel_pending()

    # Mark the addon unsafe for any in-flight timer tick before anything
    # else. This must happen first so the tick gate flips immediately,
    # preventing a mid-unregister tick from touching partially-freed
    # PropertyGroup state (has triggered Blender segfaults on reload).
    try:
        from .core.facade import mark_addon_ready
        mark_addon_ready(False)
    except Exception as e:
        print(f"mark_addon_ready(False) failed: {e}")

    # Roll back Bake while bone bindings/rigid-pose state and their RNA still
    # exist. dynamics.unregister keeps an idempotent fallback for direct use.
    try:
        from .ui.dynamics.bake_ops import cancel_bake_before_file_change
        cancel_bake_before_file_change()
    except Exception as e:
        print(f"cancel_bake_before_file_change failed: {e}")

    try:
        _cancel_captures_before_change()
    except Exception as e:
        print(f"cancel_capture_jobs failed: {e}")
    from .core.encoder import captured_bone_motion
    captured_bone_motion.unregister()
    for handlers in (bpy.app.handlers.load_pre, bpy.app.handlers.undo_pre, bpy.app.handlers.redo_pre):
        for handler in list(handlers):
            if getattr(handler, "__name__", "") in ("_cancel_captures_before_change", "_cancel_captures_before_undo"):
                handlers.remove(handler)

    # Deregister any one-shot timers scheduled at register-time that
    # haven't fired yet. After reload they would call into a freed
    # module's globals.
    try:
        _clear_deferred_timers()
    except Exception as e:
        print(f"_clear_deferred_timers failed: {e}")

    # Tell the frame-pump modal to exit on its next TIMER tick. Must
    # happen before unregister_class() so the modal instance detaches
    # cleanly (Blender will otherwise crash if we unregister a class
    # whose modal is still alive).
    try:
        from .core import frame_pump
        frame_pump.unregister()
    except Exception as e:
        print(f"frame_pump.unregister failed: {e}")

    # Stop engine worker thread and timer
    try:
        from .core.facade import cleanup
        cleanup()
    except Exception as e:
        print(f"Error during engine cleanup: {e}")

    # Stop reload server. Do NOT clear the restart-state sentinel in
    # driver_namespace here: _reload_phase1_disable writes it just
    # before calling addon_disable (which runs this function), so
    # popping here would drop the handoff the next register() needs to
    # rebuild the servers.
    if reload_server:
        reload_server.stop()
        reload_server = None

    # Remove handlers. Match by __name__ rather than object identity so
    # stale handlers from a previous module generation (before reload)
    # get removed even though the function object they point at no
    # longer equals the one we just imported.
    try:
        import bpy  # pyright: ignore
        from .core.pc2 import remove_curve_handler
        remove_curve_handler()
        for handler_list in (bpy.app.handlers.frame_change_pre,
                             bpy.app.handlers.frame_change_post,
                             bpy.app.handlers.depsgraph_update_post):
            for h in list(handler_list):
                if getattr(h, "__name__", "") in (
                    # Dead handlers whose names stay in the cleanup list so any
                    # leftover instance from an older addon generation is swept
                    # on this reload. ``pin_frame_change_handler`` is the retired
                    # _pininput.pc2 playback handler; ``invalidate_full_pin_cache``
                    # was a short-lived full-pin cache invalidator (a
                    # depsgraph_update_post callback, removed as discouraged).
                    "curve_frame_change_handler", "pin_frame_change_handler",
                    "invalidate_full_pin_cache",
                ):
                    handler_list.remove(h)
        for h in list(bpy.app.handlers.save_post):
            if getattr(h, "__name__", "") == "migrate_pc2_on_save":
                bpy.app.handlers.save_post.remove(h)
        for h in list(bpy.app.handlers.save_pre):
            if getattr(h, "__name__", "") in (
                "_persist_session_id", "_abort_capture_before_save", "prepare_mesh_cache_save",
            ):
                bpy.app.handlers.save_pre.remove(h)
        for h in list(bpy.app.handlers.save_post_fail):
            if getattr(h, "__name__", "") == "rollback_mesh_cache_save":
                bpy.app.handlers.save_post_fail.remove(h)
        for h in list(bpy.app.handlers.save_post):
            if getattr(h, "__name__", "") == "_save_manifest_post":
                bpy.app.handlers.save_post.remove(h)
        for h in list(bpy.app.handlers.load_post):
            if getattr(h, "__name__", "") in (
                "_reconcile_manifest_on_load",
                "_reconcile_pin_captured_anim_on_load",
                "_request_cache_heal_on_load",
            ):
                bpy.app.handlers.load_post.remove(h)
        for h in list(bpy.app.handlers.load_pre):
            if getattr(h, "__name__", "") == "_disconnect_on_load":
                bpy.app.handlers.load_pre.remove(h)
        import atexit as _atexit
        _atexit.unregister(_disconnect_at_exit)
        for h in list(bpy.app.handlers.render_init):
            if getattr(h, "__name__", "") == "reset_render_counter":
                bpy.app.handlers.render_init.remove(h)
        for h in list(bpy.app.handlers.render_pre):
            if getattr(h, "__name__", "") == "warn_missing_frames_on_render":
                bpy.app.handlers.render_pre.remove(h)
    except Exception:
        pass

    # Unregister UUID handlers
    try:
        from .core.uuid_registry import unregister as uuid_unregister
        uuid_unregister()
    except Exception:
        pass

    # Reset heal-log dedup set so a broken PC2 from the prior session
    # doesn't silently suppress logging after re-enable.
    try:
        from .core import client as _client_mod
        _client_mod._heal_logged.clear()
    except Exception:
        pass

    # Cleanup MCP server before unregistering other components
    try:
        from .mcp.mcp_server import cleanup_mcp_server

        cleanup_mcp_server()
    except ImportError:
        pass  # MCP module not available
    except Exception as e:
        print(f"Error during MCP cleanup: {e}")

    zozo_contact_solver.unregister()
    from .models import bone_bindings
    from .ops import bone_proxy
    from .ui import bone_physics
    from .core import rigid_pose
    rigid_pose.unregister(export_pending=False)
    from .core import bone_author_follow
    bone_author_follow.unregister()
    from .models import restoring_curve_rna
    from .ops import restoring_curve_operators
    from .ui import restoring_curves
    from .ui import bone_collision_ui
    from .ui import group_collision, layout
    layout.unregister()
    group_collision.unregister()
    bone_collision_ui.unregister()
    restoring_curves.unregister()
    from .models import bone_chain_presets as bone_chain_preset_model
    from .ops import bone_chain_presets as bone_chain_preset_ops
    from .ops import bone_presets
    bone_presets.unregister()
    bone_chain_preset_ops.unregister()
    bone_chain_preset_model.unregister()
    restoring_curve_operators.unregister()
    restoring_curve_rna.unregister()
    from .ops import cross_links as cross_link_ops
    from .models import cross_links as cross_link_model
    from .ops import cross_link_presets
    from .ops import mmd_physics_import, bone_selection
    bone_selection.unregister()
    mmd_physics_import.unregister()
    cross_link_ops.unregister()
    cross_link_presets.unregister()
    cross_link_model.unregister()
    bone_physics.unregister()
    from .ops import bone_rig_duplicate
    bone_rig_duplicate.unregister()
    bone_proxy.unregister()
    bone_bindings.unregister()
    from .mcp.handlers import _bone_context
    _bone_context.unregister_undo_operator()
    console.unregister()
    mesh_ops.unregister()
    dynamics.unregister()
    solver.unregister()
    main_panel.unregister()
    state.unregister()
    i18n.unregister()
