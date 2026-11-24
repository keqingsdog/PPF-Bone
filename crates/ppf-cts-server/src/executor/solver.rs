// Fork modification: changed from the upstream baseline.
// File: crates/ppf-cts-server/src/executor/solver.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Solver-launch plumbing for `DoLaunchSolver`. Mirrors
// `frontend/_session_.py` session.start: build a shell command from
// the per-project run script, run it detached with cwd pinned to the
// project root, redirect stdout/stderr to log files inside the
// session dir.

use std::path::PathBuf;

use ppf_cts_core::events::Event;
use ppf_cts_formats::files::{
    session_dir as session_dir_for, session_output_dir, CRASH_SIGNAL, ERROR_LOG, FINISHED,
    HELD, HOLD_AT_FRAME, INPUTS_UPDATED, SAVE_AND_QUIT, STATUS_RECORD, STDOUT_LOG,
    TERMINATE_REQUEST,
};

use super::EffectExecutor;
use crate::engine::ServerEngine;

/// The reaped exit status of the launcher process, kept for the monitor.
///
/// The direct child is `bash` (or `cmd`), not the solver, so this is
/// corroborating evidence and never the lifecycle source: the monitor's
/// reading of `status.cbor` stays authoritative. What it adds is the ONLY
/// witness to an uncatchable kill. A `SIGKILL`ed solver writes nothing, and
/// its launcher reports `128 + 9`, which is the whole reason the status is
/// retained rather than dropped with the `Child`.
pub(crate) mod exit_watch {
    use std::process::ExitStatus;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;
    use std::time::SystemTime;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct SolverExit {
        pub code: Option<i32>,
        pub signal: Option<i32>,
        /// When the launcher's status was observed.
        ///
        /// `SystemTime` rather than a monotonic `Instant` because the only
        /// thing this is ever compared against is the modification time of
        /// the solver's own `status.cbor`, which is a `SystemTime` from the
        /// same host clock. A backward clock step between the two makes a
        /// contemporaneous status read as older than the record, and the
        /// monitor then declines to quote it, which is the safe direction.
        pub reaped_at: SystemTime,
    }

    /// Identity of ONE launch: the project root plus a monotonic generation.
    ///
    /// The root alone cannot identify a launch, because every launch of a
    /// project carries the same one and the server holds one project at a
    /// time. Terminate-then-Run produces two launches under that same root
    /// with their waits overlapping, so a root-keyed write lets the first
    /// launcher's status land in the second launch's slot. The generation is
    /// what [`record`] and [`release`] check, so a stale launcher can only
    /// write to a slot that is still its own.
    ///
    /// Named for the launcher process, not for the solver's own `launch_id`
    /// (the 12-hex stamp in `RunStatus`), which is a different identity with
    /// a different lifetime.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct LaunchToken {
        pub root: String,
        generation: u64,
    }

    struct Slot {
        launch: LaunchToken,
        exit: Option<SolverExit>,
    }

    /// Holds at most one entry: [`arm`] claims the slot for the launch about
    /// to be spawned, which clears whatever the previous launch left, and
    /// [`disarm`] / [`release`] free it when there is no launcher of ours
    /// left to wait for. Without that second half the slot outlives its
    /// launch, and either a later externally-started run reads the previous
    /// run's exit status as its own, or the slot stays armed with nothing to
    /// record and pins the monitor's liveness witness at "still starting".
    static SLOT: Mutex<Option<Slot>> = Mutex::new(None);

    /// Never reused within a process, so a token from a launch that has been
    /// superseded can never match the slot again.
    static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);

    fn slot() -> std::sync::MutexGuard<'static, Option<Slot>> {
        SLOT.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub(crate) fn start_if_idle<T>(root:&str, f:impl FnOnce()->Result<T,String>)->Result<(LaunchToken,T),String> {
        let mut g=slot();
        if g.as_ref().is_some_and(|s|s.exit.is_none()) {return Err("SOLVER_LAUNCH_PENDING".into());}
        let launch=LaunchToken {root:root.into(),generation:NEXT_GENERATION.fetch_add(1,Ordering::Relaxed)};
        *g=Some(Slot {launch:launch.clone(),exit:None});
        match f() {Ok(v)=>Ok((launch,v)),Err(e)=>{*g=None;Err(e)}}
    }

    /// Claim the slot immediately before spawning, so a stale status from an
    /// earlier launch is never attributed to this one. The returned token is
    /// this launch's identity; the task that reaps its launcher must carry it.
    pub(crate) fn arm(root: &str) -> LaunchToken {
        let launch = LaunchToken {
            root: root.to_string(),
            generation: NEXT_GENERATION.fetch_add(1, Ordering::Relaxed),
        };
        *slot() = Some(Slot {
            launch: launch.clone(),
            exit: None,
        });
        launch
    }

    /// Release the slot for `root`, whichever launch holds it. The monitor
    /// calls this when it adopts a run this server did not spawn: whatever
    /// status the slot holds belongs to an earlier launch of that same
    /// project, and it must not be offered as evidence about the run now in
    /// progress.
    pub(crate) fn disarm(root: &str) {
        let mut g = slot();
        if g.as_ref().is_some_and(|s| s.launch.root == root) {
            *g = None;
        }
    }

    /// Release the slot held by `launch`, and only by `launch`.
    ///
    /// Called when a launch will never produce a status: its spawn failed, or
    /// its launcher could not be reaped. Leaving it armed keeps [`snapshot`]
    /// answering "armed, nothing recorded", which the monitor reads as a
    /// solver that is still starting, for the rest of the process's life.
    /// Generation-scoped, so a launch that has already been superseded cannot
    /// release its successor's slot.
    pub(crate) fn release(launch: &LaunchToken) {
        let mut g = slot();
        if g.as_ref().is_some_and(|s| s.launch == *launch) {
            *g = None;
        }
    }

    /// Record the launcher's reaped status, if the slot still belongs to this
    /// exact launch. A relaunch during the wait re-armed it for a later one.
    pub(crate) fn record(launch: &LaunchToken, status: ExitStatus) {
        let mut g = slot();
        if let Some(s) = g.as_mut() {
            if s.launch == *launch {
                #[cfg(unix)]
                let signal = {
                    use std::os::unix::process::ExitStatusExt as _;
                    status.signal()
                };
                #[cfg(not(unix))]
                let signal = None;
                s.exit = Some(SolverExit {
                    code: status.code(),
                    signal,
                    reaped_at: SystemTime::now(),
                });
            }
        }
    }

    /// The slot's whole state for `root` from ONE read: `(status, armed)`.
    ///
    /// Reading without consuming, so a monitor tick that runs twice before the
    /// state settles sees the same evidence.
    ///
    /// A caller that needs both must take them together. Read separately, a
    /// status landing between the two calls reports a launcher that has been
    /// reaped alongside the `None` that says it has not, and the monitor then
    /// opens a verdict on a witness it cannot quote.
    ///
    /// `armed` is what separates the two readings of a `None` status. Armed
    /// means this server spawned a launcher for that project and it has not
    /// been reaped yet, so the run is still going; unarmed means there is no
    /// launcher of ours to wait for, and liveness has to come from somewhere
    /// else.
    pub(crate) fn snapshot(root: &str) -> (Option<SolverExit>, bool) {
        match slot().as_ref() {
            Some(s) if s.launch.root == root => (s.exit.clone(), true),
            _ => (None, false),
        }
    }

    #[test]
    fn pending_slot_refuses_replacement_and_old_token_cannot_release() {
        let (a,())=start_if_idle("private-test-a",||Ok(())).unwrap();
        assert!(start_if_idle("private-test-b",||Ok(())).is_err());
        release(&a);
        let (b,())=start_if_idle("private-test-b",||Ok(())).unwrap();
        release(&a);assert!(snapshot("private-test-b").1);release(&b);
    }
}

/// Spawn the solver subprocess. Build a shell command from the
/// per-project `command.sh` (or `command.bat` on Windows), run it
/// detached with cwd pinned to the project root, redirect
/// stdout/stderr to log files inside the session dir.
///
/// `resume_from` maps to the `--load N` argument:
///   * None        -> --load 0       (fresh start, scrub export dir)
///   * Some(-1)    -> --load -1      (resume from latest checkpoint)
///   * Some(n)     -> --load n       (resume from specific frame)
pub(super) async fn launch_solver(engine: &ServerEngine, resume_from: Option<i32>, generation: u64) {
    launch_solver_horizon(engine,resume_from,None,generation).await
}
pub(super) async fn launch_solver_twist_upgrade(engine: &ServerEngine, from_frame: i32, target_frame: Option<i32>, generation: u64) {
    let Some(state) = engine.pending_run(generation) else { return; };
    match super::sealed_twist_contract::query(engine).await {
        Ok(contract) => launch_solver_inner(engine, state, Some(from_frame), target_frame, Some(contract)).await,
        Err(error) => {
            let effects = engine.fail_launch(&state, Event::ErrorOccurred { error:error.into() });
            drain(engine,effects).await;
        }
    }
}
pub(super) async fn launch_solver_horizon(engine: &ServerEngine, resume_from: Option<i32>, target_frame: Option<i32>, generation: u64) {
    let Some(state) = engine.pending_run(generation) else { return; };
    launch_solver_inner(engine,state,resume_from,target_frame,None).await
}
async fn drain(engine: &ServerEngine, effects: Vec<ppf_cts_core::effects::Effect>) {
    let exec=engine.executor().unwrap_or_else(|| std::sync::Arc::new(super::DefaultExecutor::new()));
    for fx in effects { exec.execute(fx,engine).await; }
}
async fn launch_solver_inner(engine: &ServerEngine, state: ppf_cts_core::state::ServerState, resume_from: Option<i32>, target_frame: Option<i32>, contract: Option<serde_json::Value>) {
    let root = state.root.clone();
    let root_path = PathBuf::from(&root);
    let session_dir = session_dir_for(&root_path);
    let output_dir = session_output_dir(&root_path);
    let (started,effects) = engine.launch_matching_context(&state, || {
        if root.is_empty() { return Err("no project root set".into()); }
        if let Some(expected)=contract.as_ref() {
            if expected["name"].as_str()!=Some(state.name.as_str()) || expected["root"].as_str()!=Some(state.root.as_str())
                || expected["upload_id"].as_str()!=Some(state.upload_id.as_str()) || expected["param_hash"].as_str()!=Some(state.param_hash.as_str())
                || expected["data_hash"].as_str()!=Some(state.data_hash.as_str())
                || expected["solver_backend"].as_str()!=Some(engine.config().solver_backend.as_str()) {
                return Err("PROJECT_CONTEXT_CHANGED".into());
            }
        }
        let mut args=super::solver_args::arguments(resume_from.unwrap_or(0),target_frame).map_err(str::to_owned)?;
        if resume_from.is_some_and(|frame| frame != 0) && super::retained_checkpoints(&root_path) {
            return Err("CHECKPOINTS_FROM_PREVIOUS_BUILD: run a fresh simulation for updated parameters".into());
        }
        if contract.is_some() {args.push("--twist-resume-mode=continuous_from_current".into());}
        std::fs::create_dir_all(&session_dir).map_err(|e|format!("session dir create failed: {e}"))?;
        #[cfg(windows)]
        let (program,cmd_path)=("cmd", session_dir.join(ppf_cts_formats::files::COMMAND_BAT));
        #[cfg(not(windows))]
        let (program,cmd_path)=("bash", session_dir.join(ppf_cts_formats::files::COMMAND_SH));
        if !cmd_path.exists() {return Err(format!("solver run script not found: {}",cmd_path.display()));}
        #[cfg(windows)]
        if engine.config().solver_mode_v1 && resume_from.is_some_and(|frame| frame != 0) {
            crate::response::check_legacy_r14_animation_resume(&cmd_path, &session_dir.join(STDOUT_LOG))?;
        }
        let stdout=std::fs::File::create(session_dir.join(STDOUT_LOG)).map_err(|e|format!("stdout open: {e}"))?;
        let stderr=std::fs::File::create(session_dir.join(ERROR_LOG)).map_err(|e|format!("stderr open: {e}"))?;
        let mut cmd=tokio::process::Command::new(program);
        #[cfg(windows)]
        cmd.arg("/C");
        cmd.arg(&cmd_path).args(&args).current_dir(&root).stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::from(stdout)).stderr(std::process::Stdio::from(stderr));
        #[cfg(unix)]
        cmd.process_group(0);
        exit_watch::start_if_idle(&root, || {
        std::fs::create_dir_all(&output_dir).map_err(|e|format!("output directory: {e}"))?;
        let lease=ppf_cts_formats::status::lock::acquire(&output_dir).map_err(|e|format!("status lease: {e}"))?;
        if crate::upload::read_upload_id(&root_path).ok().as_deref()!=Some(state.upload_id.as_str()) {return Err("PROJECT_CONTEXT_CHANGED".into());}
        let status_path=output_dir.join(STATUS_RECORD);
        match std::fs::symlink_metadata(&status_path) {
            Ok(m) if m.is_file() && !m.file_type().is_symlink()=>{
                #[cfg(windows)]
                {use std::os::windows::fs::MetadataExt; if m.file_attributes() & 0x400 != 0 {return Err("invalid status reparse point".into());}}
                let old=ppf_cts_formats::status::read(&output_dir).map_err(|e|format!("invalid status: {e}"))?.ok_or("status vanished")?;
                let baseline=super::resume_baseline::Baseline::new(&state,&old.launch_id).map_err(str::to_owned)?;
                super::resume_baseline::write(&output_dir,&baseline).map_err(str::to_owned)?;
            }
            Err(e) if e.kind()==std::io::ErrorKind::NotFound=>{
                let p=output_dir.join("status.resume-baseline.json");
                match std::fs::symlink_metadata(p) {Ok(_)=>{super::resume_baseline::read(&output_dir,&state).map_err(str::to_owned)?;},Err(e) if e.kind()==std::io::ErrorKind::NotFound=>(),Err(e)=>return Err(format!("baseline metadata: {e}"))}
            }
            _=>return Err("invalid status file".into()),
        }
        for sentinel in [SAVE_AND_QUIT,FINISHED,STATUS_RECORD,TERMINATE_REQUEST,CRASH_SIGNAL,HOLD_AT_FRAME,HELD,INPUTS_UPDATED] {
            match std::fs::remove_file(output_dir.join(sentinel)) {Ok(())=>(),Err(e) if e.kind()==std::io::ErrorKind::NotFound=>(),Err(e)=>return Err(format!("scrub {sentinel}: {e}"))}
        }
        drop(lease);
        cmd.spawn().map_err(|e|format!("spawn: {e}"))
        })
    });
    if let Some(exec)=engine.executor() {for fx in effects {exec.execute(fx,engine).await;}}
    else {let exec=super::DefaultExecutor::new();for fx in effects {exec.execute(fx,engine).await;}}
    match started {
        Some(Ok((launch,mut child))) => {
            log::info!(target: "ppf::solver", "DoLaunchSolver: solver pid={:?}", child.id());
            // Reap the launcher in the background and keep its exit status.
            // The monitor stays the authoritative lifecycle source (it reads
            // the solver-authored status.cbor); this status is corroborating
            // evidence, and the only witness left when the solver is killed by
            // a signal it cannot catch. Awaiting does not kill the child
            // (`kill_on_drop` is false) and `process_group(0)` above already
            // provides the signal isolation the detach is for, so the solver
            // still outlives a server shutdown.
            tokio::spawn(async move {
                let root = &launch.root;
                match child.wait().await {
                    Ok(status) => {
                        log::info!(
                            target: "ppf::solver",
                            "solver launcher for {root} exited: {status}"
                        );
                        exit_watch::record(&launch, status);
                    }
                    Err(e) => {
                        log::warn!(
                            target: "ppf::solver",
                            "could not reap the solver launcher for {root}: {e}"
                        );
                        // No status will ever arrive for this launch. Hand the
                        // slot back so the monitor's liveness witness falls
                        // through to the process scan; an armed slot with
                        // nothing to record reads as "still starting" forever.
                        exit_watch::release(&launch);
                    }
                }
            });
        }
        other => {
            // Same reason as the unreapable-launcher arm above: nothing was
            // spawned, so nothing will ever be recorded against this slot.
            let e=match other {Some(Err(e))=>e,None=>"PROJECT_CONTEXT_CHANGED".into(),_=>unreachable!()};
            log::error!(target: "ppf::solver", "DoLaunchSolver: spawn failed: {e}");
        }
    }
}
