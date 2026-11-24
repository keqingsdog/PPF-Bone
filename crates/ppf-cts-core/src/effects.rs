// Fork modification: changed from the upstream baseline.
// File: crates/ppf-cts-core/src/effects.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Typed effect enum for the server state machine. Historically ported
// from the now-removed server/effects.py (Rust is now authoritative).
// Effects describe side-effects to perform; they carry no behavior.
// The executor (in ppf-cts-server) is the only place they're actually
// run.

use crate::state::ServerState;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildContext {
    pub generation: u64,
    pub name: String,
    pub root: String,
    pub upload_id: String,
    pub param_hash: String,
    pub data_hash: String,
}

impl BuildContext {
    pub(crate) fn capture(state: &ServerState) -> Self {
        Self {
            generation: state.project_generation,
            name: state.name.clone(),
            root: state.root.clone(),
            upload_id: state.upload_id.clone(),
            param_hash: state.param_hash.clone(),
            data_hash: state.data_hash.clone(),
        }
    }

    pub fn matches(&self, state: &ServerState) -> bool {
        self.generation == state.project_generation
            && self.name == state.name && self.root == state.root
            && self.upload_id == state.upload_id
            && self.param_hash == state.param_hash && self.data_hash == state.data_hash
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Effect {
    // ---- Build effects ----
    /// Spawn the background build task. `preserve_output` flows to the
    /// build worker as `--preserve-output`, keeping `session/output/`
    /// checkpoints in place for a resume-rebuild.
    DoSpawnBuild { preserve_output: bool, context: BuildContext },
    /// Signal the build task to stop (via CancellationToken).
    DoCancelBuild,

    // ---- Solver effects ----
    /// Start the solver subprocess.
    /// `resume_from = None` for fresh start, `Some(-1)` for latest
    /// checkpoint (matching the Python `int | None` semantics).
    DoLaunchSolver { generation: u64, resume_from: Option<i32> },
    DoLaunchResumeHorizon { generation: u64, from_frame: i32, target_frame: i32 },
    DoLaunchTwistUpgrade { generation: u64, from_frame: i32, target_frame: Option<i32> },
    /// Terminate the solver subprocess.
    DoKillSolver { generation: u64 },
    /// Create the `save_and_quit` flag file the solver polls for.
    DoRequestSaveAndQuit,

    // ---- Disk I/O effects ----
    /// Ensure this selected project directory exists without changing context.
    DoLoadApp { name: String, root: String, generation: u64 },
    /// Delete the entire project directory.
    DoDeleteProjectData { root: String, name: String, generation: u64 },

    // ---- Logging ----
    /// Write a message to the server log.
    DoLog { message: String },
}
