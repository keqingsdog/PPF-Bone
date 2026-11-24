// File: crates/ppf-cts-server/src/engine.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// `ServerEngine`: tokio-friendly state holder + dispatcher.
// Replaces server/engine.py's `ServerEngine` class.
//
// Concurrency model:
//   * `state` is held under a `parking_lot::RwLock` so multi-reader
//     queries (status responses) don't block each other.
//   * `dispatch(event)` acquires the write lock for the duration of
//     the pure transition, replaces the snapshot, releases the lock,
//     and returns the produced effects to the caller.
//   * Effect *execution* is the executor's responsibility, NOT the
//     engine's. This keeps the engine cancellation-token-free and
//     makes it trivially testable in pure Rust.
//   * `cancel_token` holds the cooperative-cancellation handle for an
//     in-flight build. `DoSpawnBuild` mints + stores a fresh token;
//     `DoCancelBuild` calls `.cancel()` on it. Replaces the
//     ctypes.PyThreadState_SetAsyncExc thread-kill in the Python
//     reference (see CancelHandle in ppf-cts-core::cancel).

use std::sync::{Arc, OnceLock, Weak};

use parking_lot::{Mutex, RwLock};
use ppf_cts_core::cancel::CancelHandle;
use ppf_cts_core::effects::Effect;
use ppf_cts_core::events::Event;
use ppf_cts_core::state::ServerState;
use ppf_cts_core::transitions::transition;

use crate::config::EngineConfig;
use crate::executor::EffectExecutor;

#[derive(Clone)]
pub struct ServerEngine {
    inner: Arc<EngineInner>,
}

struct EngineInner {
    state: RwLock<ServerState>,
    config: EngineConfig,
    /// Cancellation control for the in-flight build, kept under a
    /// single mutex so install/cancel/clear are atomic with respect
    /// to each other. The `pending` flag latches a cancel that
    /// arrives in the gap between `BuildRequested` dispatch and the
    /// executor's call to `install_cancel_handle`, so the next build
    /// observes the cancel instead of silently ignoring it.
    cancel: Mutex<CancelState>,
    /// Weak handle to the active executor, set once at startup by
    /// `attach_executor`. Re-entrant effect handlers (build pipeline,
    /// solver launch, save/quit) upgrade this to dispatch follow-up
    /// events through the same executor that processed the original
    /// effect, instead of minting a stateless `DefaultExecutor`. Held
    /// as `Weak` to avoid the engine→executor→... cycle when an
    /// executor implementation captures an `Arc<ServerEngine>`
    /// internally; the embedder owns the `Arc<dyn EffectExecutor>`
    /// for the process lifetime so the upgrade is expected to
    /// succeed during normal operation.
    executor: OnceLock<Weak<dyn EffectExecutor>>,
}

#[derive(Default)]
struct CancelState {
    /// Active build cancellation token (if any).
    handle: Option<CancelHandle>,
    /// Sticky "cancel pending" latch: set when `cancel_active_build`
    /// runs while no handle is installed yet. The next
    /// `install_cancel_handle` consumes the latch and returns an
    /// already-cancelled handle.
    pending: bool,
}

impl ServerEngine {
    pub fn new(config: EngineConfig) -> Self {
        Self {
            inner: Arc::new(EngineInner {
                state: RwLock::new(ServerState::default()),
                config,
                cancel: Mutex::new(CancelState::default()),
                executor: OnceLock::new(),
            }),
        }
    }

    /// Bind the executor used for re-entrant effect dispatch (build
    /// pipeline emitting `BuildCompleted`, solver helpers emitting
    /// `ErrorOccurred`, etc.). Idempotent: subsequent calls are
    /// ignored so tests can leave it unset and fall back to a fresh
    /// `DefaultExecutor` via `executor_or_default`. Stored as `Weak`
    /// so the embedder keeps ownership and can drop the executor at
    /// shutdown without the engine pinning it alive.
    pub fn attach_executor(&self, executor: &Arc<dyn EffectExecutor>) {
        let _ = self.inner.executor.set(Arc::downgrade(executor));
    }

    /// Upgrade the attached executor to a concrete `Arc`, or return
    /// `None` if no executor has been attached or the embedder has
    /// already dropped it. Re-entrant handlers use this to thread
    /// effects back through the user's executor instead of building
    /// a fresh `DefaultExecutor` each time.
    pub fn executor(&self) -> Option<Arc<dyn EffectExecutor>> {
        self.inner.executor.get().and_then(Weak::upgrade)
    }

    pub fn config(&self) -> &EngineConfig {
        &self.inner.config
    }

    /// Snapshot the current state. Cheap (read lock + clone of a
    /// ~14-field POD).
    pub fn state(&self) -> ServerState {
        self.inner.state.read().clone()
    }

    pub(crate) fn with_matching_context<T>(&self, expected:&ServerState, f:impl FnOnce()->T)->Option<T> {
        let current=self.inner.state.read();
        if current.project_generation!=expected.project_generation || current.name!=expected.name || current.root!=expected.root || current.upload_id!=expected.upload_id || current.param_hash!=expected.param_hash || current.data_hash!=expected.data_hash {return None;}
        Some(f())
    }
    pub(crate) fn launch_matching_context<T>(&self, expected:&ServerState, f:impl FnOnce()->Result<T,String>)->(Option<Result<T,String>>,Vec<Effect>) {
        let mut current=self.inner.state.write();
        if !Self::same_run(&current,expected) || !current.launch_pending {return (None,Vec::new());}
        let result=f();
        current.launch_pending=false;
        let effects=match &result {Err(e)=>{let (next,fx)=transition(current.clone(),Event::SolverCrashed{error:format!("spawn: {e}"),kind_tag:"launch_failed".into(),violations:vec![]});*current=next;fx},_=>Vec::new()};
        (Some(result),effects)
    }

    fn same_run(current: &ServerState, expected: &ServerState) -> bool {
        current.project_generation == expected.project_generation
            && current.run_generation == expected.run_generation
            && current.name == expected.name && current.root == expected.root
            && current.upload_id == expected.upload_id && current.param_hash == expected.param_hash
            && current.data_hash == expected.data_hash
    }

    /// Keep the lock through disk/exit-watch writes, blocking a successor launch.
    pub(crate) fn with_matching_run<T>(&self, expected: &ServerState, f: impl FnOnce() -> T) -> Option<T> {
        let current = self.inner.state.read();
        if !Self::same_run(&current, expected) || current.launch_pending { return None; }
        Some(f())
    }

    pub(crate) fn dispatch_matching_run(&self, expected: &ServerState, event: Event) -> Option<Vec<Effect>> {
        let mut current = self.inner.state.write();
        if !Self::same_run(&current, expected) || current.launch_pending { return None; }
        let (next, effects) = transition(current.clone(), event);
        *current = next;
        Some(effects)
    }

    pub(crate) fn with_generation<T>(&self, generation: u64, f: impl FnOnce(&ServerState) -> T) -> Option<T> {
        let current = self.inner.state.read();
        if current.run_generation != generation || current.launch_pending { return None; }
        Some(f(&current))
    }

    pub(crate) fn pending_run(&self, generation: u64) -> Option<ServerState> {
        let current = self.inner.state.read();
        (current.run_generation == generation && current.launch_pending).then(|| current.clone())
    }

    pub(crate) fn fail_launch(&self, expected: &ServerState, event: Event) -> Vec<Effect> {
        let mut current = self.inner.state.write();
        if !Self::same_run(&current, expected) || !current.launch_pending { return vec![]; }
        current.launch_pending = false;
        let error = match event { Event::ErrorOccurred { error } => error, _ => unreachable!("launch failure requires error") };
        let (next, effects) = transition(current.clone(), Event::SolverCrashed { error, kind_tag:"launch_failed".into(), violations:vec![] });
        *current = next;
        effects
    }

    /// Run the pure transition under the write lock and return the
    /// produced effects. The caller forwards these effects to an
    /// `EffectExecutor`. Re-entrance from inside an effect (e.g. a
    /// build task dispatching `BuildCompleted` back) is handled
    /// because the lock is dropped before this returns.
    pub fn dispatch(&self, event: Event) -> Vec<Effect> {
        let mut guard = self.inner.state.write();
        let snapshot = guard.clone();
        let (new_state, effects) = transition(snapshot, event);
        *guard = new_state;
        effects
    }

    /// Capture the exact post-transition context before another client can
    /// select a different project.
    pub(crate) fn dispatch_with_snapshot(&self, event: Event) -> (ServerState, Vec<Effect>) {
        let mut guard = self.inner.state.write();
        let (next, effects) = transition(guard.clone(), event);
        *guard = next;
        (guard.clone(), effects)
    }

    pub(crate) fn same_project_context(current: &ServerState, expected: &ServerState) -> bool {
        current.project_generation == expected.project_generation
            && current.name == expected.name && current.root == expected.root
            && current.upload_id == expected.upload_id
            && current.param_hash == expected.param_hash && current.data_hash == expected.data_hash
    }

    pub(crate) fn dispatch_matching_context(&self, expected:&ServerState, event:Event)->Option<Vec<Effect>> {
        let mut current=self.inner.state.write();
        if !Self::same_project_context(&current, expected) {return None;}
        let (next,effects)=transition(current.clone(),event);*current=next;Some(effects)
    }

    pub(crate) fn begin_build(
        &self,
        expected: &ppf_cts_core::effects::BuildContext,
        prepare: impl FnOnce() -> std::io::Result<()>,
    ) -> Option<(ServerState, CancelHandle, std::io::Result<()>)> {
        let current = self.inner.state.read();
        (expected.matches(&current) && current.build == ppf_cts_core::state::Build::Building)
            .then(|| (current.clone(), self.install_cancel_handle(), prepare()))
    }

    pub(crate) fn finish_build(
        &self, expected: &ppf_cts_core::effects::BuildContext, event: Option<Event>,
    ) -> Vec<Effect> {
        let mut current = self.inner.state.write();
        if current.project_generation != expected.generation
            || current.build != ppf_cts_core::state::Build::Building
        {
            return vec![];
        }
        let event = if expected.matches(&current) {
            event
        } else {
            Some(Event::BuildFailed {
                error: "BUILD_INPUTS_CHANGED: build again with the latest transfer".into(),
                violations: vec![],
            })
        };
        let Some(event) = event else { return vec![]; };
        self.clear_cancel_handle();
        let (next, effects) = transition(current.clone(), event);
        *current = next;
        effects
    }

    pub(crate) fn dispatch_project_upload(&self, name: &str, root: &str, event: Event) -> Vec<Effect> {
        let mut current = self.inner.state.write();
        let context = if current.name == name && current.root == root {
            current.clone()
        } else {
            let project_root = std::path::Path::new(root);
            let mut context = ServerState {
                project_generation: current.project_generation,
                run_generation: current.run_generation,
                data_hash: crate::upload::read_data_hash(project_root),
                param_hash: crate::upload::read_param_hash(project_root),
                ..ServerState::default()
            };
            Self::select_context(&mut context, name, root);
            context
        };
        let (next, effects) = transition(context, event);
        *current = next;
        effects
    }

    /// Replace the project name + root without going through a full
    /// `ProjectSelected` transition. Used by request handlers that
    /// need to set context before dispatching guarded events.
    pub fn set_project_context(&self, name: &str, root: &str) {
        let mut guard = self.inner.state.write();
        Self::select_context(&mut guard, name, root);
    }

    fn select_context(current: &mut ServerState, name: &str, root: &str) {
        if current.name != name || current.root != root {
            current.run_generation = current.run_generation.checked_add(1).expect("run generation exhausted");
            current.launch_pending = false;
            current.project_generation = current.project_generation.checked_add(1)
                .expect("project generation exhausted");
        }
        current.name = name.to_string();
        current.root = root.to_string();
    }

    // --- Cancellation token plumbing --------------------------------

    /// Install a cancel handle for an in-flight build. Returns a
    /// clone the build task can use to poll `is_cancelled()`.
    /// Cooperative cancellation only; no ctypes thread-kill.
    ///
    /// If a cancel arrived between `BuildRequested` dispatch and this
    /// call (e.g. another connection raced a `cancel_build` against
    /// the just-issued `build`), the returned handle is already
    /// cancelled so the build observes it on the first checkpoint
    /// instead of running to completion.
    pub fn install_cancel_handle(&self) -> CancelHandle {
        let h = CancelHandle::new();
        let mut guard = self.inner.cancel.lock();
        if guard.pending {
            h.cancel();
            guard.pending = false;
        }
        guard.handle = Some(h.clone());
        h
    }

    /// Trip the active cancel token if there is one. If no build is
    /// in flight yet, latches the cancel so the next
    /// `install_cancel_handle` returns an already-cancelled handle.
    pub fn cancel_active_build(&self) {
        let mut guard = self.inner.cancel.lock();
        match guard.handle.as_ref() {
            Some(h) => h.cancel(),
            None => guard.pending = true,
        }
    }

    /// Drop the cancel handle when the build finishes (success,
    /// failure, or cancellation observed). Also clears any latched
    /// pending-cancel: by the time a build has run to a terminal
    /// state, any earlier latched cancel has been resolved (either
    /// observed via the cancelled handle, or the build completed
    /// before the next `install_cancel_handle` could consume it; in
    /// the latter case the latch was meant for *that* build, not a
    /// future one, so we drop it here rather than letting it bleed
    /// into the next launch).
    pub fn clear_cancel_handle(&self) {
        let mut guard = self.inner.cancel.lock();
        guard.handle = None;
        guard.pending = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ppf_cts_core::events::Event;
    use ppf_cts_core::state::{Build, Data};

    #[test]
    fn dispatch_advances_state_and_emits_effects() {
        let engine = ServerEngine::new(EngineConfig::default());
        engine.set_project_context("p", "/tmp/p");
        // Mark data uploaded so BuildRequested isn't rejected.
        engine.dispatch(Event::upload_landed("uid"));

        let effects = engine.dispatch(Event::BuildRequested { preserve_output: false });
        let s = engine.state();
        assert_eq!(s.build, Build::Building);
        assert_eq!(s.data, Data::Uploaded);
        assert!(effects.iter().any(|e| matches!(e, Effect::DoSpawnBuild { .. })));
    }

    #[test]
    fn cancel_handle_lifecycle() {
        let engine = ServerEngine::new(EngineConfig::default());
        let h = engine.install_cancel_handle();
        assert!(!h.is_cancelled());

        engine.cancel_active_build();
        assert!(h.is_cancelled(), "cancel propagates to held handle");

        engine.clear_cancel_handle();
        // Subsequent cancel without an installed handle latches a
        // pending cancel; clear_cancel_handle wipes it so the next
        // build starts fresh.
        engine.cancel_active_build();
        engine.clear_cancel_handle();
        let h2 = engine.install_cancel_handle();
        assert!(!h2.is_cancelled(), "cleared latch should not bleed forward");
    }

    #[test]
    fn cancel_before_install_latches_into_next_handle() {
        // Simulates the TOCTOU window the audit called out: a cancel
        // arrives between `BuildRequested` dispatch and the executor's
        // call to `install_cancel_handle`. The next install must
        // observe the cancel rather than silently dropping it.
        let engine = ServerEngine::new(EngineConfig::default());
        engine.cancel_active_build();
        let h = engine.install_cancel_handle();
        assert!(
            h.is_cancelled(),
            "cancel latched before install must apply to the new handle"
        );
    }

    #[test]
    fn set_project_context_persists() {
        let engine = ServerEngine::new(EngineConfig::default());
        engine.set_project_context("demo", "/tmp/demo");
        let s = engine.state();
        assert_eq!(s.name, "demo");
        assert_eq!(s.root, "/tmp/demo");
    }
}
