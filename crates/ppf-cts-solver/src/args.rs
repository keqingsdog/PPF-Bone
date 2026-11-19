// Fork modification: changed from the upstream baseline.
// File: args.rs
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use clap::Parser;

#[derive(Parser, Debug, Clone, serde::Serialize, serde::Deserialize)]
#[clap(author, version, about, long_about = None)]
pub struct ProgramArgs {
    /// Explicit migration from a legacy checkpoint into continuous twist;
    /// the sealed scene data remains unchanged.
    #[clap(long, value_parser = ["continuous_from_current"])]
    #[serde(default)]
    pub twist_resume_mode: Option<String>,
    #[clap(long)]
    #[serde(default)]
    pub target_frame:Option<i32>,
    #[clap(long,default_value_t=false)]
    #[serde(default)]
    pub resume_capability_json:bool,
    #[clap(long)]
    #[serde(default)]
    pub restoring_curves_mass_query:Option<String>,
    #[clap(long,default_value_t=false)]
    #[serde(default)]
    pub require_restoring_curves_v1:bool,
    #[clap(long, default_value = "")]
    pub path: String,

    #[clap(long, default_value = "output")]
    pub output: String,

    /// Publish accepted frame snapshots on a bounded CPU writer thread.
    #[clap(long, default_value_t = false)]
    #[serde(default)]
    pub async_output: bool,

    #[clap(long, default_value_t = 0)]
    pub load: i32,
    #[clap(long, default_value_t = false)]
    #[serde(default)]
    pub require_collision_filter_v1: bool,
    #[clap(long, default_value_t = false)]
    #[serde(default)]
    pub require_pcg_precision_v1: bool,
}

/// Storage precision of the reduced nodal block-Jacobi preconditioner.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PcgPrecision {
    #[default]
    #[serde(rename = "fp32")]
    Fp32,
    #[serde(rename = "mixed-fp16-nodal-v1")]
    MixedFp16Nodal,
    #[serde(rename = "mixed-fp16-arithmetic-v2")]
    MixedFp16Arithmetic,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SimArgs {
    pub disable_contact: bool,
    pub keep_states: i32,
    pub keep_verts: i32,
    pub dt: f32,
    pub inactive_momentum: bool,
    pub playback: f32,
    pub min_newton_steps: u32,
    pub target_toi: f32,
    pub air_friction: f32,
    pub line_search_max_t: f32,
    pub constraint_ghat: f32,
    pub constraint_tol: f32,
    pub fps: f64,
    pub cg_max_iter: u32,
    pub cg_tol: f32,
    #[serde(default)]
    pub pcg_precision: PcgPrecision,
    pub ccd_eps: f32,
    pub max_dx: f32,
    pub eiganalysis_eps: f32,
    pub friction_eps: f32,
    pub csrmat_max_nnz: u32,
    pub frames: i32,
    pub auto_save: i32,
    #[serde(default)]
    pub save_state_on_finish: bool,
    /// Comma-separated solver frame indices at which to write a resumable
    /// state, independent of `auto_save`. Empty string disables explicit
    /// checkpoints. `#[serde(default)]` keeps older `param.toml` files
    /// (written before this field existed) loadable for resume.
    #[serde(default)]
    pub checkpoints: String,
    pub barrier: String,
    pub friction_mode: String,
    pub stitch_length_factor: f32,
    pub air_density: f32,
    pub isotropic_air_friction: f32,
    pub gravity: [f32; 3],
    pub wind: [f32; 3],
    pub include_face_mass: bool,
    pub fix_xz: f32,
    pub fake_crash_frame: i32,
    /// Linear-solve preconditioner: "block-jacobi" (default) or "schwarz".
    /// Block-Jacobi is the default since the PCG inner loop became
    /// device-resident: its many cheap iterations beat Schwarz's few expensive
    /// ones, and it does not OOM on heavy-contact scenes. Parsed in `make_param`
    /// (mirrors the `barrier` / `friction_mode` idiom). `#[serde(default)]` keeps
    /// older `param.toml` files (written before this field existed) loadable;
    /// they default to the block-Jacobi preconditioner.
    #[serde(default = "default_precond")]
    pub precond: String,
    /// Number of additive levels for the Schwarz preconditioner (1 =
    /// single-level smoother, 2 = two-level coarse correction). Only used
    /// when `precond` is "schwarz". `#[serde(default)]` keeps older
    /// `param.toml` files (written before this field existed) loadable;
    /// they default to the two-level coarse correction.
    #[serde(default = "default_schwarz_levels")]
    pub schwarz_levels: u32,
    /// Uniform spatial scale applied to all input geometry on ingest; output
    /// positions are divided back by it on write. 1.0 is a no-op. Lets a mesh
    /// authored at the wrong scale (e.g. a 15 m human) simulate at the right
    /// physical size (1.5 m) and still be written back at the authored scale.
    /// `#[serde(default = "default_world_scaling")]` keeps older `param.toml`
    /// files (written before this field existed) loadable for resume; they
    /// default to 1.0 (no scaling).
    #[serde(default = "default_world_scaling")]
    pub world_scaling: f32,
    /// Upper bound on Newton iterations per substep. The loop is otherwise
    /// unbounded, so an over-constrained configuration spins forever instead of
    /// failing (see `ParamSet::max_newton_steps`). 0 disables the bound.
    /// `#[serde(default)]` keeps older `param.toml` files loadable for resume.
    #[serde(default = "default_max_newton_steps")]
    pub max_newton_steps: u32,
}

fn default_precond() -> String {
    "block-jacobi".to_string()
}

fn default_max_newton_steps() -> u32 {
    2048
}

fn default_schwarz_levels() -> u32 {
    2
}

fn default_world_scaling() -> f32 {
    1.0
}

/// Resolve explicit migration's latest sentinel without accepting zero or
/// unrelated negative values. Caller supplies only the read-only directory scan.
pub(crate) fn resolve_twist_resume_load(load: i32, latest: Option<u64>) -> Result<i32, &'static str> {
    let resolved = if load == -1 {
        i32::try_from(latest.ok_or("TWIST_RESUME_NO_CHECKPOINT")?).map_err(|_| "TWIST_RESUME_INVALID_FRAME")?
    } else { load };
    if resolved <= 0 { return Err("TWIST_RESUME_INVALID_FRAME") }
    Ok(resolved)
}

#[cfg(test)]
mod twist_args_tests {
    use super::*;
    #[test]
    fn twist_resume_latest_resolution() {
        assert_eq!(resolve_twist_resume_load(4,None),Ok(4));
        assert_eq!(resolve_twist_resume_load(-1,Some(8)),Ok(8));
        assert!(resolve_twist_resume_load(-1,None).is_err());
        assert!(resolve_twist_resume_load(-1,Some(0)).is_err());
        assert!(resolve_twist_resume_load(0,Some(8)).is_err());
        assert!(resolve_twist_resume_load(-2,Some(8)).is_err());
        assert!(resolve_twist_resume_load(-1,Some(u64::MAX)).is_err());
    }
    #[test]
    fn twist_resume_cli_strict_enum() {
        assert_eq!(ProgramArgs::try_parse_from(["solver", "--load=1", "--twist-resume-mode=continuous_from_current"]).unwrap().twist_resume_mode.as_deref(), Some("continuous_from_current"));
        assert!(ProgramArgs::try_parse_from(["solver", "--twist-resume-mode=invalid"]).is_err());
        assert!(ProgramArgs::try_parse_from(["solver", "--twist-resume-mode"]).is_err());
        assert!(ProgramArgs::try_parse_from(["solver", "--twist-resume-mode=continuous_from_current", "--twist-resume-mode=continuous_from_current"]).is_err());
    }
}
