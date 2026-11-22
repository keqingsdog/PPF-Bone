// Fork modification: changed from the upstream baseline.
// File: crates/ppf-cts-solver/src/driver/step.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The Newton driver: one `advance()`.
//!
//! One substep of the implicit integrator, expressed as a Rust loop over the
//! shared neutral kernel bodies. The `// STEP` markers below number the stages
//! in the order they run, and that order is load-bearing rather than cosmetic:
//! moving a stage past another changes an fp32 fold order, or moves the
//! `tmp_fixed` snapshot boundary, and neither shows up as a failure.
//!
//! # What this driver covers, and what it refuses
//!
//! Tets, the shell membrane and its bending, the two rod terms, both strain
//! limiters, the cross-stitch, and self-contact with its barrier, its friction,
//! its ACCD line search and its intersection gate. Everything else is refused
//! BY NAME at `initialize()` (`super::refusal`), and this function asserts the
//! refusals held rather than assuming them: a driver that accepted a scene it
//! cannot step would write a full run of plausible frames, and this backend has
//! no hardware fault to catch that.
//!
//! # Three traps kept deliberately
//!
//! 1. `target` is CLOBBERED at B20 to hold the iterate's pre-step positions,
//!    and every CONTINUING iteration rebuilds it with `compute_target` at the
//!    bottom of the loop. The line search reads the old positions out of it, so
//!    the reuse is load-bearing rather than a saved allocation.
//! 2. `tmp_fixed` is a REAL second buffer here. Metal splits the assembly into
//!    an evaluate and a scatter because it has no second matrix to spare;
//!    memory is cheap on the host, so this backend keeps the whole snapshot and
//!    assembles in one pass.
//! 3. The bounded `max-newton-steps` escape ships WITH the loop. It is the only
//!    exit from an over-constrained configuration: `toi <= FLT_EPSILON` does
//!    not cover a prescribed vertex driven into geometry that cannot yield,
//!    because the line search clamps the SHARED toi just above zero and the
//!    loop then re-assembles a bit-identical system forever. Without the bound
//!    such a scene HANGS instead of reporting `NewtonStall`.
//!
//! # The host scalars, and why they are not kernels moved to the host
//!
//! Eleven float expressions in this module and one in `pcg` are not routed
//! through a shared body. EVERY ONE OPERATES ON A SCALAR THAT HAS ALREADY BEEN
//! REDUCED OUT OF THE MESH, so it is host arithmetic by nature rather than a
//! kernel rewritten on the host. The list is exhaustive on purpose: it is
//! what a rule-8 allowlist would have to name, and a twelfth site appearing
//! without an entry here is the thing to catch.
//!
//! - `max_u = max(speed^2).sqrt()`, the square root taken after the fold.
//! - `dt = param.dt * playback`.
//! - `back = 1 - toi_advanced`, seeding the kinematic rewind.
//! - `dt = (double(dt) * toi_advanced) as f32`. `dt` is fp32 and `toi_advanced`
//!   is double, so the product is formed in double and rounded ONCE; forming it
//!   in fp32 rounds twice and gives a different step size, which is the clock
//!   the whole trailing iteration integrates against.
//! - `toi_recale = min(1, max_dx_param / max_dx)`.
//! - `toi = min(1, smallest / line_search_max_t)`, `smallest` being the folded
//!   per-primitive contact time of impact.
//! - `shell_toi = smallest / line_search_max_t`, then `toi = min(toi,
//!   shell_toi)`, and the same for the rod.
//! - `sl_toi = min(1, shell_toi, rod_toi)`.
//! - `toi_advanced += max(0, 1 - toi_advanced) * (toi_recale * toi)`, in double.
//! - `time += dt / playback`, in double.
//! - `reresid = sum|r| / err0` (`pcg`), the ratio taken on the host once both
//!   terms exist.
//!
//! Every PER-ELEMENT value comes from a body, including the ones that look like
//! they could not: the per-tet mass scale and the preconditioner's diagonal sum
//! both go through `vec_add_scaled`, because a matrix times a scalar is that
//! scalar applied to each of its stored floats.
//!
//! # THE FOLDS THAT PRODUCE THOSE SCALARS RUN ON THE HOST, AND THAT IS A
//! RECORDED DIVERGENCE
//!
//! Read the list above for what it says and not for what it looks like it
//! says. Each entry is true of the ARITHMETIC once the scalar exists; none of
//! them is a claim about how the scalar was reduced out of the mesh, and the
//! reduction is where this file diverges.
//!
//! Each of those scalars should fall out of a DEVICE reduction, a warp-shuffle
//! tree into a block tree into a strided fold, ending in a read of four bytes.
//! This module instead calls `ReadbackBuffer::download` on the whole array and
//! folds it in `super::reduce` over rayon, everywhere it needs one: `max_u` and
//! `reach` at the start of the step, `max_sigma` over the shell faces and the
//! rods, the count of removed Dirichlet rows, `max_dx` inside the Newton loop,
//! and both strain-limit times of impact. Four bytes should cross the seam
//! where `vertices`, `faces` or `rods` floats cross today, and the arithmetic
//! on top of the scalar is the same either way, so nothing in the list above
//! shows the difference.
//!
//! **THIS IS A RECORDED DIVERGENCE, NOT A TRANSPORT PREFERENCE**, and it takes
//! the same shape the PCG scalars do: a pass whose work is proportional to the
//! mesh belongs on the device, and running it on the host is a different solver
//! rather than a slower one. It is not closable from this file:
//! `kernels/primitives/reduce.kernel.cpp` declares `warp_reduce` and
//! `block_reduce` as `[[seam::device_fn]]` and declares NO `[[seam::entry]]`, so
//! there is no reduction for `driver/launch.rs` to bind and no verb on `Device`
//! to dispatch. Adding one is what closes every one of those folds at once, and
//! converting any of them any other way would be inventing a second design.

use crate::data::{DataSet, ParamSet, StepResult};

use super::contact::Windows;
use super::fixedcsr::FixedCsr;
use super::operator::{DynamicView, Operator};
use super::scene::{Fatal, FatalResult};
use super::state::SolverState;
use super::{assemble, dirichlet, log, pcg};

#[path = "b32_shadow.rs"]
mod b32_shadow;

use ppf_cts_compute::{Device, EncoderExt, Fault, Handle};

// Compiled into Rust test harnesses only; no production environment variable,
// command-line flag or operator can inject a solver failure.
#[cfg(test)]
static PRIVATE_TEST_COMMIT_STAGE:std::sync::atomic::AtomicU8=std::sync::atomic::AtomicU8::new(0);
#[cfg(all(test,feature="cpu"))]
static PRIVATE_TEST_B3_PCG:std::sync::atomic::AtomicU8=std::sync::atomic::AtomicU8::new(0);
#[cfg(all(test,feature="cpu"))]
static PRIVATE_TEST_B3_DT:std::sync::atomic::AtomicU32=std::sync::atomic::AtomicU32::new(0);
#[cfg(all(test,feature="cpu"))]
pub(super) fn arm_private_b3_pcg(){PRIVATE_TEST_B3_DT.store(0,std::sync::atomic::Ordering::SeqCst);PRIVATE_TEST_B3_PCG.store(1,std::sync::atomic::Ordering::SeqCst);}
#[cfg(all(test,feature="cpu"))]
pub(super) fn private_b3_dt()->f32{f32::from_bits(PRIVATE_TEST_B3_DT.load(std::sync::atomic::Ordering::SeqCst))}
#[cfg(test)]
pub(super) fn set_private_test_commit_stage(stage:u8){PRIVATE_TEST_COMMIT_STAGE.store(stage,std::sync::atomic::Ordering::SeqCst);}
#[cfg(test)]
fn private_test_commit_stage(stage:u8)->bool{PRIVATE_TEST_COMMIT_STAGE.compare_exchange(stage,0,std::sync::atomic::Ordering::SeqCst,std::sync::atomic::Ordering::SeqCst).is_ok()}
#[cfg(test)]
static PRIVATE_TEST_FEM_INJECTIONS:std::sync::atomic::AtomicU32=std::sync::atomic::AtomicU32::new(0);
#[cfg(test)]
pub(super) fn set_private_test_fem_injections(count:u32){PRIVATE_TEST_FEM_INJECTIONS.store(count,std::sync::atomic::Ordering::SeqCst);}
#[cfg(test)]
fn private_test_fem_injection()->bool {PRIVATE_TEST_FEM_INJECTIONS.fetch_update(std::sync::atomic::Ordering::SeqCst,std::sync::atomic::Ordering::SeqCst,|v|v.checked_sub(1)).is_ok()}
#[cfg(all(test,feature="cpu"))]
static PRIVATE_TEST_FEM_ROW2_INJECTIONS:std::sync::atomic::AtomicU32=std::sync::atomic::AtomicU32::new(0);
#[cfg(all(test,feature="cpu"))]
pub(super) fn set_private_test_fem_row2_injections(count:u32){PRIVATE_TEST_FEM_ROW2_INJECTIONS.store(count,std::sync::atomic::Ordering::SeqCst);}
#[cfg(all(test,feature="cpu"))]
fn private_test_fem_row2_injection()->bool {PRIVATE_TEST_FEM_ROW2_INJECTIONS.fetch_update(std::sync::atomic::Ordering::SeqCst,std::sync::atomic::Ordering::SeqCst,|v|v.checked_sub(1)).is_ok()}
use super::kernels::{
    SandGrainCondenseRowArgs, SandGrainIntegrateRowArgs, SandGrainRecoverRowArgs,
    BlockJacobiInvertRowArgs, ComputeTargetSeedArgs, VecFillArgs, DxMagnitudeArgs, DxSeedArgs, FixXzDragArgs, PositionAcceptArgs,
    PositionStepArgs, RewindFixArgs, VelocityTermsArgs,
};

extern "C" {
    // NOT A DISPATCH, and the only shim this module still names. It reports the
    // largest absolute coordinate the position representation covers, a
    // compile-time constant of the shared header rather than work over a range,
    // so it has no extent and no argument record and the seam has nothing to
    // carry it.
    fn position_domain_abi() -> f32;
}

/// Whether the DOF-removal count has been reported for this process.
///
/// Reported once: the mask is rebuilt every step but its SIZE is a
/// property of the scene, so repeating the line every step would bury the log
/// under a constant.
static DOF_REPORTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The parameters one step reads, taken once at its start.
///
/// `ParamSet` is neither `Copy` nor `Clone` (it is a `repr(C)` mirror of a C++
/// record and copying it wholesale would copy container handles too), so the
/// snapshot names its fields. That is the better shape here anyway: it lists,
/// in one place, exactly what this driver's answer depends on.
struct StepParams {
    time: f64,
    dt: f32,
    playback: f32,
    prev_dt: f32,
    gravity: [f32; 3],
    inactive_momentum: bool,
    cg_tol: f32,
    cg_max_iter: u32,
    max_dx: f32,
    max_newton_steps: u32,
    min_newton_steps: u32,
    target_toi: f32,
    line_search_max_t: f32,
    eiganalysis_eps: f32,
    /// The factor the cross-stitch length cap scales its rest length by. One
    /// scene-wide scalar, read by the stitch layer and by nothing else.
    stitch_length_factor: f32,
    /// The barrier shape, as the `repr(C)` discriminant the shared bodies
    /// dispatch over. Carried as an integer because that is what crosses the
    /// FFI boundary; `Barrier` itself is the Rust mirror of the C++ enum.
    barrier: u32,
    disable_contact: bool,
    disable_pin_dof_removal: bool,
    /// The `fix-xz` drag's threshold, and the switch that decides whether the
    /// drag runs at all: at zero the step dispatches neither half. Both halves
    /// ship, so a scene setting it is accepted rather than refused, which
    /// `refusal.rs` asserts as a difference against the same scene with the
    /// threshold off.
    fix_xz: f32,
}

impl StepParams {
    /// # Safety
    /// `param` must address a live `ParamSet`.
    unsafe fn snapshot(param: *const ParamSet) -> Self {
        let p = &*param;
        Self {
            time: p.time,
            dt: p.dt,
            playback: p.playback,
            prev_dt: p.prev_dt,
            gravity: [p.gravity[0], p.gravity[1], p.gravity[2]],
            inactive_momentum: p.inactive_momentum,
            cg_tol: p.cg_tol,
            cg_max_iter: p.cg_max_iter,
            max_dx: p.max_dx,
            max_newton_steps: p.max_newton_steps,
            min_newton_steps: p.min_newton_steps,
            target_toi: p.target_toi,
            line_search_max_t: p.line_search_max_t,
            eiganalysis_eps: p.eiganalysis_eps,
            stitch_length_factor: p.stitch_length_factor,
            barrier: p.barrier as u32,
            disable_contact: p.disable_contact,
            disable_pin_dof_removal: p.disable_pin_dof_removal,
            fix_xz: p.fix_xz,
        }
    }
}

/// `fix_index` for every vertex, which `compute_target` and the seed both read.
///
/// Gathered into its own array because the shared body takes the index list
/// rather than the whole `VertexProp`: the same body serves a backend whose
/// props live in a different layout. Written into a buffer the driver already
/// owns, so a step allocates nothing.
///
/// # Safety
/// `data` must address a live `DataSet` whose prop array is `out.len()` long.
/// The fraction of a grain's spin angular momentum fed back into its
/// translation, read from `PPF_SAND_SPIN_COUPLE` and defaulting to 0.5.
///
/// 0.5 IS NEAR-TEXTBOOK ROLLING WHILE STAYING BOUNDED. Fully included, the spin
/// propels the translation which feeds the spin, which is a runaway rather than
/// a stiff case; zero leaves rolling to decay to about a third of the textbook
/// rate.
pub(super) fn sand_spin_couple() -> f32 {
    std::env::var("PPF_SAND_SPIN_COUPLE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.5)
}

/// The under-roll fraction the grain spin integrate caps omega at, read from
/// `PPF_SAND_ROLL_RESIST` and defaulting to 0.05.
///
/// ZERO IS NOT THE NEUTRAL VALUE HERE, WHICH IS WHY THIS IS NOT A LITERAL.
/// The cap is `(1 - roll_resist)` times the no-slip rate `|v_t| / (dt * r)`.
/// Capping at EXACTLY no-slip lets the lagged omega over-roll on an oscillating
/// contact and PROPEL the grain, which is the energy-pumping ratchet the bound
/// exists to prevent; capping slightly below it leaves a small residual forward
/// slip every step, so friction always opposes motion and travel stays bounded
/// while the grain still visibly rolls. `sand_rigid.kernel.cpp` carries the same
/// reasoning at the clamp itself.
///
/// The spin integrate takes `c_roll` as a literal `0.0` and `roll_resist` as
/// THIS, so the two trailing arguments of that dispatch do not agree and must
/// not be written as if they did.
pub(super) fn sand_roll_resist() -> f32 {
    std::env::var("PPF_SAND_ROLL_RESIST")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.05)
}

/// Whether `PPF_SAND_NO_ROLL` pins every grain's omega at zero so grains slide
/// instead of roll, which is the lever for separating a rolling artifact from a
/// sliding one. Presence is the signal; the value is not read.
pub(super) fn sand_no_roll() -> bool {
    std::env::var_os("PPF_SAND_NO_ROLL").is_some()
}

/// A per-step delay in milliseconds, for a caller that needs to OBSERVE a run
/// while it is in progress. Zero, and off, unless `PPF_STEP_DELAY_MS` says
/// otherwise.
///
/// WHY IT EXISTS. Seven rig scenarios watch a solve as it runs: a progress bar
/// advancing, a statistics panel filling, a frame counter tracking, a transfer
/// refused mid-run. A real solve on a scenario-sized scene finishes faster than
/// the observer can sample it, so a run that is correct is also unobservable,
/// and the scenario cannot tell "the state never appeared" from "the state was
/// never sampled".
///
/// IT DEFAULTS TO ZERO, AND IS A DELAY RATHER THAN A FREEZE. Defaulting to
/// anything else would slow every run to serve a handful of tests. Freezing
/// would let a scenario pass while the solver never reached the state it claims
/// to observe, which is the failure a pacing hook is most able to hide.
///
/// IT IS NOT REACHABLE FROM A SCENE. Read from the process environment and
/// never from `ParamSet`: a scene parameter that slows the solver is a way to
/// make a slow solver look correct, and a build default would do the same to
/// every user. It is read ONCE, so a run cannot change pace halfway and leave
/// two halves that are not comparable.
fn step_delay() -> std::time::Duration {
    static DELAY: std::sync::OnceLock<std::time::Duration> = std::sync::OnceLock::new();
    *DELAY.get_or_init(|| parse_step_delay(std::env::var("PPF_STEP_DELAY_MS").ok().as_deref()))
}

/// The parse, split out so it can be tested without the process environment.
///
/// UNPARSABLE READS AS ZERO RATHER THAN FAILING, which is the one place this
/// knob differs from the project's fail-loud default and is deliberate. A
/// pacing hook has no effect on what is computed, so a typo that silently
/// leaves the run at full speed costs an observer a retry, while a panic would
/// take down a solve over a debugging aid.
fn parse_step_delay(value: Option<&str>) -> std::time::Duration {
    std::time::Duration::from_millis(
        value
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(0),
    )
}

/// One `advance()`.
///
/// # Safety
/// `data` must address a live `DataSet`, `param` a live `ParamSet` the host
/// keeps for the whole run, and `state` must have been allocated for this scene.
/// Bring the Schwarz preconditioner up to date for this Newton step.
///
/// EXTRACTED SO BOTH SOLVE PATHS CAN INSTALL IT. The hierarchy is set on the
/// operator BEFORE the caller branches on `aggregate_locked`, so the locked
/// solve gets the preconditioner the scene asked for. A locked scene that took
/// block-Jacobi regardless would answer `precond = schwarz` with a label rather
/// than a preconditioner.
///
/// THE PARTITION IS REUSED AND THE FACTORIZATION IS NOT, and the split follows
/// from what each depends on: the partition depends on the operator's STRUCTURE
/// and the factorization on its VALUES, and the values change at every Newton
/// step.
/// The hierarchy is rebuilt with the factorization rather than with the
/// partition, every level's operator being a Galerkin product of the fine one.
///
/// # Safety
/// The operator's handles must be live for this step and `vertices` must be its
/// row count.
unsafe fn install_schwarz<D: Device>(
    device: &mut D,
    schwarz: &mut super::schwarz::State,
    operator: &Operator,
    vertices: usize,
    param: *const ParamSet,
) -> FatalResult<()> {
    // THE STAND-IN OFFSETS COME FIRST, because the operator view names them for
    // every span this scene does not have.
    let empty = schwarz.empty_offsets(device, vertices as u32)?;
    let rows_view = operator.rows_view(empty);
    if schwarz.rows != vertices as u32 {
        super::schwarz::partition(device, schwarz, rows_view, vertices as u32)?;
    }
    super::schwarz::refactor(device, schwarz, rows_view)?;
    let levels = if (*param).schwarz_levels > 0 {
        (*param).schwarz_levels
    } else {
        2
    };
    super::schwarz::build_hierarchy(device, schwarz, rows_view, vertices as u32, levels)?;
    Ok(())
}

/// Bounded optional root-curve subdivision. Everything accepted stays in
/// curr/prev/prev_dt; the gate returns before C3/C5/C6/C7. Trial caches mutated
/// before then are restored, and A1..B33 recompute all other scratch on retry.
fn root_final_budget_exhausted(step:u32,maximum:u32,final_step:bool,active_roots:bool)->bool {
    (!final_step||active_roots)&&maximum>0&&step>=maximum
}
fn initialize_final_clock(final_step:bool,initialized:bool)->bool {final_step&&!initialized}
fn root_boundary_ready(preserve_boundary:bool,full_sweep_accepted:bool,step:u32,maximum:u32)->bool {
    // The final iteration uses the same index and still runs its own
    // shortened solve and CCD. Leave that slot available.
    !preserve_boundary || full_sweep_accepted || step.saturating_add(1)>=maximum
}
// Experimental routing into the existing bounded, fully restored subdivision.
// A rounded candidate refusal still refuses the entire uncommitted trial.
fn root_acceptance_retry_enabled()->bool {
    std::env::var_os("PPF_ROOT_ACCEPTANCE_RETRY_EXPERIMENT").is_some_and(|v|v=="1")
}
fn root_trial_needs_subdivision(outcome:&FatalResult<StepResult>)->bool {
    match outcome {
        // These are precommit numerical step failures. Physical intersection,
        // invalid prescribed paths, overlapping starts and PCG faults retain
        // their own refusals and never become accepted states through retry.
        Ok(r)=>r.pcg_success && r.intersection_free && r.pin_feasible && r.contact_separated &&
            (!r.newton_progress || !r.ccd_success),
        Err(error)=>error.code==ppf_cts_formats::status::error_code::SOLVER_INVARIANT &&
            matches!(error.detail.as_str(),
                "ANALYTIC_ACCEPT_UNRESOLVED: no representable free candidate progress" |
                "ANALYTIC_ACCEPT_UNRESOLVED: bounded conservative candidate search exhausted"),
    }
}

pub unsafe fn advance<D:Device>(device:&mut D,data:&DataSet,param:*mut ParamSet,state:&mut SolverState)->FatalResult<StepResult> {
    advance_impl(device,data,param,state,false,None,false,false,false,false,None,None,None,false)
}

pub unsafe fn advance_retry<D:Device>(device:&mut D,data:&DataSet,param:*mut ParamSet,state:&mut SolverState,
    control_boundary_already_applied:bool,rejection_budget:u32)->FatalResult<StepResult> {
    advance_impl(device,data,param,state,control_boundary_already_applied,Some(rejection_budget),true,false,false,false,None,None,None,false)
}

pub(crate) struct FemHostRefusal {
    pub token:super::soft_joints::MaterialFrameCandidateRejection,
    pub original:Fatal,
    pub native_rejections:u32,
}
pub(crate) struct SwingHostRefusal {
    pub token:super::joints::SampleSwingTrial,
    pub original:Fatal,
    pub native_rejections:u32,
}
pub(crate) struct ChartHostRefusal {
    pub token:super::joints::CandidateFaultToken,
    pub original:Fatal,
    pub native_rejections:u32,
}
pub(crate) struct MaterialSpinHostRefusal {pub failed_dt:f32,pub generation:u64,pub start_bits:u64,pub original:Fatal,pub native_rejections:u32}
pub(crate) struct NewtonCapHostRefusal {pub failed_dt:f32,pub generation:u64,pub start_bits:u64,pub native_rejections:u32,pub step:u32,pub last_toi:f32,pub original:Fatal}
pub(crate) struct KinematicsRefusal { pub actual_dt: f32, pub original: Fatal, pub native_rejections: u32 }
pub(super) enum HostDomainRefusal { Kinematics(KinematicsRefusal), Fem(FemHostRefusal), Swing(SwingHostRefusal), Chart(ChartHostRefusal), MaterialSpin(MaterialSpinHostRefusal), NewtonCap(NewtonCapHostRefusal) }
impl HostDomainRefusal {
    fn original(&self)->&Fatal {match self {Self::Kinematics(v)=>&v.original,Self::Fem(v)=>&v.original,Self::Swing(v)=>&v.original,Self::Chart(v)=>&v.original,Self::MaterialSpin(v)=>&v.original,Self::NewtonCap(v)=>&v.original}}
}
pub(super) unsafe fn advance_host<D:Device>(device:&mut D,data:&DataSet,param:*mut ParamSet,state:&mut SolverState,
    control_boundary_already_applied:bool,rejection_budget:u32,allow_cg:bool,allow_fem:bool,allow_swing:bool,allow_newton:bool,prepared:Option<crate::joint_root_motion_state::NativeStep>,allow_kinematics:bool)
    ->FatalResult<(StepResult,Option<HostDomainRefusal>,Option<f32>)> {
    let mut refusal=None;
    let mut pcg_failed_dt=None;
    let route_domain=allow_kinematics||allow_fem||allow_swing||state.joints.material_spin_active()||state.joints.soft.material_spin_active();
    let result=advance_impl(device,data,param,state,control_boundary_already_applied,Some(rejection_budget),allow_cg,allow_fem,allow_swing,allow_newton,
        if route_domain{Some(&mut refusal)}else{None},prepared,Some(&mut pcg_failed_dt),allow_kinematics)?;
    Ok((result,refusal,pcg_failed_dt))
}

unsafe fn advance_impl<D:Device>(device:&mut D,data:&DataSet,param:*mut ParamSet,state:&mut SolverState,
    control_boundary_already_applied:bool,retry_budget:Option<u32>,allow_cg:bool,allow_fem:bool,allow_swing:bool,allow_newton:bool,
    mut domain_sink:Option<&mut Option<HostDomainRefusal>>,prepared:Option<crate::joint_root_motion_state::NativeStep>,mut pcg_sink:Option<&mut Option<f32>>,allow_kinematics:bool)->FatalResult<StepResult> {
    if !control_boundary_already_applied {
        super::roots::control_boundary(device,data,state,param)?;
    }
    let adaptive_retry=allow_cg;
    let max_rejections=retry_budget.unwrap_or(super::adaptive_retry::MAX_REJECTIONS);
    let rotating_root=state.roots.rotating();
    let acceptance_retry_root=root_acceptance_retry_enabled() && state.roots.active();
    // Ordinary no-angular-control/no-rotating-root scenes keep the direct path.
    if !rotating_root && !acceptance_retry_root && !state.joints.has_control() && !state.joints.soft.active() && !adaptive_retry && !allow_kinematics {let mut committed=false;let outcome=advance_trial(device,data,param,state,false,false,None,&mut committed,&mut None,&mut None);if !outcome.as_ref().is_ok_and(|r|r.success()){state.fem_roots.abort();}return outcome;}
    use ppf_cts_compute::{Buffer,AllocLabel};
    // Buffer has no releasing Drop. Keep every handle reachable even when a
    // size/copy/restore returns early, then explicitly attempt all four frees.
    let mut fix=Buffer::default();
    let mut rigid=Buffer::default();
    let mut rotation=Buffer::default();
    let mut omega=Buffer::default();
    let mut accepted_positions=Buffer::<f32>::default();
    let mut accepted_previous=Buffer::<f32>::default();
    let outcome=(||->FatalResult<StepResult> {
        fix.size(device,state.fix.len(),AllocLabel("root.retry.fix"))?;fix.copy_from(device,&state.fix)?;
        rigid.size(device,state.rigid_state.len(),AllocLabel("root.retry.rigid"))?;rigid.copy_from(device,&state.rigid_state)?;
        rotation.size(device,state.rigid_running_rotation.len(),AllocLabel("root.retry.rotation"))?;rotation.copy_from(device,&state.rigid_running_rotation)?;
        omega.size(device,state.grain_omega.len(),AllocLabel("root.retry.omega"))?;omega.copy_from(device,&state.grain_omega)?;
        let seeded=state.rigid_rotation_seeded;
        let history=((*param).time,(*param).prev_dt,(*param).dt);
        let continuous=state.joints.twist_selection().is_some();
        let material_spin=state.joints.material_spin_active();
        let fem_material_spin=state.joints.soft.material_spin_active();
        let needs_full_trial_snapshot=continuous||material_spin||state.joints.soft.active()||allow_kinematics;
        let before_clock=((*param).time,(*param).time_f32,(*param).prev_dt,(*param).dt);
        let before_root=state.roots.transaction.clone();
        let before_fem=state.fem_roots.runtime.clone();
        let before_fem_attempt=state.fem_roots.attempt.clone();
        let before_fem_steps=state.fem_roots.accepted_steps;
        let before_fem_end=state.fem_roots.prepared_end;
        let before_restoring=(state.restoring_commit_count,state.restoring_commit_time);
        let before_curves=if needs_full_trial_snapshot {
            accepted_positions.size(device,state.positions.len(),AllocLabel("twist.rollback.positions"))?;
            device.copy(accepted_positions.handle(),0,state.positions.handle(),0,state.positions.len()*std::mem::size_of::<f32>())?;
            accepted_previous.size(device,state.positions_prev.len(),AllocLabel("twist.rollback.previous"))?;
            device.copy(accepted_previous.handle(),0,state.positions_prev.handle(),0,state.positions_prev.len()*std::mem::size_of::<f32>())?;
            state.joints.curves.snapshot_trial(device)?
        } else {None};
        let mut requested=None;
        let mut rejections=0u32;
        for retry in 0..=max_rejections {
            state.roots.retry_step(requested);
            state.joints.reset_candidate_retry()?;
            if material_spin {
                state.joints.begin_material_spin_trial()?;
                if let Err(error)=state.joints.upload_material_spin_context(device) {state.joints.abort_material_spin_trial();return Err(Fatal::from(error));}
            }
            if fem_material_spin {
                state.joints.soft.begin_material_spin_trial()?;
                if let Err(error)=state.joints.soft.upload_material_spin_context(device){state.joints.soft.abort_material_spin_trial();return Err(Fatal::from(error));}
            }
            if let Some(selected)=state.joints.twist_selection().map(|x|x.to_vec()) {
                state.joints.begin_twist_trial()?;
                if let Err(error)=state.joints.upload_certified_twist_context(device,&selected) {
                    state.joints.abort_twist_trial();return Err(Fatal::from(error));
                }
            }
            let swing_scope_supported=domain_sink.is_some() && allow_swing && !allow_fem && (*param).playback==1.0 && continuous &&
                !state.hard.active() && state.roots.transaction.is_none() && state.fem_roots.runtime.is_none() && state.joints.swing_retry_scope_supported();
            let mut committed=false;
            let newton_scope=allow_newton && domain_sink.is_some() && super::newton_root_scope(state,(*param).playback);
            let mut newton_candidate=None;
            let mut kinematics_shortened=if allow_kinematics {Some(0.0)} else {None};
            let outcome=advance_trial(device,data,param,state,adaptive_retry,swing_scope_supported,prepared,&mut committed,&mut newton_candidate,&mut kinematics_shortened);
            let root_pcg=newton_scope && !committed && outcome.as_ref().is_ok_and(|r|r.retry_reason==crate::data::RETRY_REASON_PCG_MAX_ITERATIONS);
            let root_pcg_dt=if root_pcg {state.roots.trial_dt().filter(|h|h.is_finite() && *h>0.)} else {None};
            if !committed && outcome.is_err() && (material_spin||fem_material_spin) {
                let faults=state.joints.curves.snapshot_faults(device)?;
                if (!faults.is_empty()||state.joints.soft.material_spin_base_fault()) && faults.iter().all(|(family,_,code)|matches!(*family,"angular"|"fem") && *code==8) && state.joints.material_spin_control_retry_compatible(device)? {
                    state.joints.mark_material_spin_retry((*param).dt);
                }
            }
            // Once C5 has accepted positions and twist history, no failure is
            // a trial failure: preserve the original outcome and never route it
            // through root/FEM abort or any automatic retry path.
            if needs_full_trial_snapshot && committed {
                state.roots.retry_step(None);
                return outcome.map(|mut result|{result.retry_rejections=rejections;result});
            }
            if !outcome.as_ref().is_ok_and(|r|r.success()) {state.joints.abort_twist_trial();state.joints.abort_material_spin_trial();state.joints.soft.abort_material_spin_trial();}
            // A fatal before C5 may have changed the provisional clock, root
            // transaction and curve history. Restore only trusted in-memory
            // snapshots, never the fault banks/token that explain the failure.
            // Only the typed host entry can ask the backend to resample all
            // authored inputs. The old C ABI always retains the original fatal.
            let fem_scope_supported=domain_sink.is_some() && allow_fem && (*param).playback==1.0 && !continuous && !state.hard.active() && !state.roots.rotating() && !state.roots.has_control_events() && state.fem_roots.runtime.is_none() && state.joints.soft.retry_scope_supported();
            if !fem_scope_supported && state.joints.soft.candidate_rejection().is_some() {
                ::log::info!("FEM_MATERIAL_RETRY_UNSUPPORTED_SCOPE: nonunit playback, continuous, HARD, rotating roots, FEM controls or angular drives require separate rollback proof");
            }
            let fem_candidate=if !committed && outcome.is_err() && fem_scope_supported {
                state.joints.material_candidate_retry(device).ok().flatten()
            }else{None};
            let swing_candidate=if !committed && outcome.is_err() && swing_scope_supported {
                state.joints.swing_candidate_retry(device).ok().flatten()
            }else{None};
            let chart_requested=!committed && outcome.is_err() && swing_scope_supported && state.joints.candidate_retry_dt().is_some();
            let chart_candidate=if chart_requested {state.joints.host_chart_candidate(device).ok().flatten()}else{None};
            if needs_full_trial_snapshot && !outcome.as_ref().is_ok_and(|r|r.success()) && !committed {
                let rollback=(||->FatalResult<()> {
                    device.copy(state.positions.handle(),0,accepted_positions.handle(),0,state.positions.len()*std::mem::size_of::<f32>())?;
                    device.copy(state.positions_prev.handle(),0,accepted_previous.handle(),0,state.positions_prev.len()*std::mem::size_of::<f32>())?;
                    state.fix.copy_from(device,&fix)?;state.rigid_state.copy_from(device,&rigid)?;
                    state.rigid_running_rotation.copy_from(device,&rotation)?;state.rigid_rotation_seeded=seeded;
                    state.grain_omega.copy_from(device,&omega)?;state.eval_x.copy_from(device,&state.positions)?;
                    match (&mut state.roots.transaction,&before_root) {
                        (Some(current),Some(saved))=>current.restore_trial_snapshot(saved.clone()).map_err(Fatal::invariant)?,
                        (None,None)=>{},_=>return Err(Fatal::invariant("continuous rollback root identity changed")),
                    }
                    match (&mut state.fem_roots.runtime,&before_fem) {
                        (Some(current),Some(saved))=>current.restore_trial_snapshot(saved.clone()).map_err(Fatal::invariant)?,
                        (None,None)=>{},_=>return Err(Fatal::invariant("continuous rollback FEM identity changed")),
                    }
                    state.fem_roots.attempt=before_fem_attempt.clone();state.fem_roots.accepted_steps=before_fem_steps;state.fem_roots.prepared_end=before_fem_end;
                    state.joints.curves.restore_trial(device,&before_curves)?;
                    state.joints.soft.invalidate_trial_cache();
                    state.restoring_commit_count=before_restoring.0;state.restoring_commit_time=before_restoring.1;
                    (*param).time=before_clock.0;(*param).time_f32=before_clock.1;(*param).prev_dt=before_clock.2;(*param).dt=before_clock.3;
                    Ok(())
                })();
                if let Err(error)=rollback {
                    ::log::error!("continuous pre-C5 rollback failed; original failure retained: {:?}",error);
                    return match outcome {Err(mut original)=>{original.detail.push_str(&format!("; trial rollback failed: {}",error.detail));Err(original)},Ok(_)=>Err(error)};
                }
            }
            if allow_kinematics && !committed {
                if let Some(actual_dt) = kinematics_shortened.filter(|h| *h > 0.0) {
                    let sink = domain_sink.as_deref_mut()
                        .ok_or_else(|| Fatal::invariant("KINEMATICS_RETRY: missing host sink"))?;
                    *sink = Some(HostDomainRefusal::Kinematics(KinematicsRefusal {
                        actual_dt,
                        original: Fatal::invariant("KINEMATICS_RETRY: integration step shortened"),
                        native_rejections: rejections,
                    }));
                    state.roots.retry_step(None);
                    let mut refused = StepResult::default();
                    refused.newton_progress = false;
                    return Ok(refused);
                }
            }            if newton_scope && !committed && !outcome.as_ref().is_ok_and(|r|r.success()) {
                state.roots.invalidate_trial_targets(device).map_err(Fatal::from)?;
            }
            if root_pcg {
                let failed=root_pcg_dt.ok_or_else(||Fatal::invariant("HOST_NEWTON_PCG: missing actual trial dt after rollback"))?;
                *pcg_sink.as_deref_mut().ok_or_else(||Fatal::invariant("HOST_NEWTON_PCG: missing host sink after rollback"))?=Some(failed);
            }
            if newton_scope && !committed {
                if let (Some((failed_dt,step,last_toi)),Ok(result))=(newton_candidate,outcome.as_ref()) {
                    if result.pcg_success && result.intersection_free && result.pin_feasible && result.contact_separated && !result.newton_progress {
                        let generation=state.joints.accepted_material_spin().ok_or_else(||Fatal::invariant("NEWTON_CAP: missing material generation"))?.generation;
                        let sink=domain_sink.as_deref_mut().ok_or_else(||Fatal::invariant("NEWTON_CAP: missing host sink"))?;
                        *sink=Some(HostDomainRefusal::NewtonCap(NewtonCapHostRefusal{failed_dt,generation,start_bits:history.0.to_bits(),native_rejections:rejections,step,last_toi,original:Fatal::invariant("NEWTON_CAP: iteration limit reached")}));
                        let mut refused=StepResult::default();refused.newton_progress=false;refused.retry_rejections=rejections;return Ok(refused);
                    }
                }
            }
            if let Some(token)=&fem_candidate {
                if state.joints.material_candidate_retry(device).ok().flatten().as_ref()!=Some(token) {
                    return match outcome {Err(mut original)=>{original.detail.push_str("; FEM rollback verification failed: fault bank or token changed");Err(original)},other=>other};
                }
                if let Err(cleanup)=state.joints.soft.consume_candidate_rejection(device,token) {
                    return match outcome {Err(mut original)=>{original.detail.push_str(&format!("; FEM rollback verification failed: {cleanup:?}"));Err(original)},other=>other};
                }
                let original=match outcome {Err(original)=>original,Ok(_)=>return Err(Fatal::invariant("FEM candidate token without original fault"))};
                let Some(sink)=domain_sink.as_deref_mut() else{return Err(original)};
                *sink=Some(HostDomainRefusal::Fem(FemHostRefusal{token:token.clone(),original,native_rejections:rejections}));
                state.roots.retry_step(None);
                let mut result=StepResult::default();result.newton_progress=false;result.retry_rejections=rejections;
                return Ok(result);
            }
            if let Some(token)=&swing_candidate {
                if state.joints.swing_candidate_retry(device).ok().flatten().as_ref()!=Some(token) {
                    return match outcome {Err(mut original)=>{original.detail.push_str("; swing rollback verification failed: fault bank or token changed");Err(original)},other=>other};
                }
                if let Err(cleanup)=state.joints.consume_swing_candidate(device,token) {
                    return match outcome {Err(mut original)=>{original.detail.push_str(&format!("; swing rollback verification failed: {cleanup:?}"));Err(original)},other=>other};
                }
                let original=match outcome {Err(original)=>original,Ok(_)=>return Err(Fatal::invariant("swing candidate token without original fault"))};
                let Some(sink)=domain_sink.as_deref_mut() else{return Err(original)};
                *sink=Some(HostDomainRefusal::Swing(SwingHostRefusal{token:token.clone(),original,native_rejections:rejections}));
                state.roots.retry_step(None);
                let mut result=StepResult::default();result.newton_progress=false;result.retry_rejections=rejections;
                return Ok(result);
            }
            if chart_requested {
                let Some(token)=chart_candidate else {
                    return match outcome {Err(mut original)=>{original.detail.push_str("; chart host qualification failed");Err(original)},other=>other};
                };
                if state.joints.host_chart_candidate(device).ok().flatten().as_ref()!=Some(&token) {
                    return match outcome {Err(mut original)=>{original.detail.push_str("; chart host rollback verification failed");Err(original)},other=>other};
                }
                if let Err(cleanup)=state.joints.consume_candidate_faults(device) {
                    return match outcome {Err(mut original)=>{original.detail.push_str(&format!("; chart host cleanup failed: {cleanup:?}"));Err(original)},other=>other};
                }
                let original=match outcome {Err(original)=>original,Ok(_)=>return Err(Fatal::invariant("chart token without original fault"))};
                let Some(sink)=domain_sink.as_deref_mut() else{return Err(original)};
                *sink=Some(HostDomainRefusal::Chart(ChartHostRefusal{token,original,native_rejections:rejections}));
                state.roots.retry_step(None);
                let mut result=StepResult::default();result.newton_progress=false;result.retry_rejections=rejections;
                return Ok(result);
            }
            if acceptance_retry_root && root_trial_needs_subdivision(&outcome) {
                let failed_dt=state.roots.trial_dt().filter(|dt|dt.is_finite() && *dt>0.0)
                    .ok_or_else(||Fatal::invariant("ROOT_MOTION_COMMIT_MISMATCH: rejected native trial has no duration"))?;
                log::message!("ROOT_TRIAL_SUBDIVISION: rejected uncommitted trial dt={failed_dt:?} restored_subdivision={retry}");
                state.roots.request_curve_retry(failed_dt);
            }
            let chart_failed_dt=state.joints.candidate_retry_dt();
            let material_failed_dt=if !committed && outcome.is_err(){state.joints.material_spin_retry_dt()}else{None};
            let wants_retry=state.roots.curve_retry || chart_failed_dt.is_some() || material_failed_dt.is_some();
            if let (Some(failed_dt),Some(sink))=(material_failed_dt,domain_sink.as_deref_mut()) {
                state.joints.consume_material_spin_faults(device)?;
                if state.joints.soft.material_spin_base_fault(){state.joints.soft.consume_material_spin_base_fault(device)?;}
                let generation=state.joints.accepted_material_spin().or_else(||state.joints.soft.accepted_material_spin()).map(|v|v.generation).ok_or_else(||Fatal::invariant("material spin retry missing accepted generation"))?;
                let original=match outcome {Err(v)=>v,Ok(_)=>return Err(Fatal::invariant("material retry missing fault"))};
                *sink=Some(HostDomainRefusal::MaterialSpin(MaterialSpinHostRefusal{failed_dt,generation,start_bits:history.0.to_bits(),original,native_rejections:rejections}));
                state.roots.retry_step(None);
                let mut result=StepResult::default();result.newton_progress=false;result.retry_rejections=rejections;return Ok(result);
            }
            if prepared.is_some() && wants_retry {
                // Such an inner subdivision would need another host resample.
                // It is outside this first material retry scope: retain the
                // original failure instead of using a stale sampled endpoint.
                state.roots.retry_step(None);return outcome;
            }
            if state.joints.candidate_has_fatal_fault() {state.fem_roots.abort();state.roots.retry_step(None);return outcome;}
            if outcome.as_ref().is_ok_and(|r|r.success()) {
                state.roots.retry_step(None);
                return outcome.map(|mut result|{result.retry_rejections=rejections;result});
            }
            state.fem_roots.abort();
            let adaptive_cap=adaptive_retry&&outcome.as_ref().is_ok_and(|r|r.retry_reason==crate::data::RETRY_REASON_PCG_MAX_ITERATIONS);
            // A new non-root angular path must preserve unrelated/postcommit
            // native failures. Only the precommit domain marker requests rollback.
            if !rotating_root && !wants_retry && !adaptive_cap {state.roots.retry_step(None);return outcome.map(|mut result|{result.retry_rejections=rejections;result});}
            // No accepted pose/history or plastic creep can have run: the curve
            // gate is before C3. Check that explicitly before restoring trial data.
            if ((*param).time,(*param).prev_dt,(*param).dt)!=history {
                return Err(Fatal::invariant("ROOT_MOTION_COMMIT_MISMATCH: failed curve trial changed accepted physical history"));
            }
            state.fix.copy_from(device,&fix)?;state.rigid_state.copy_from(device,&rigid)?;
            state.rigid_running_rotation.copy_from(device,&rotation)?;state.rigid_rotation_seeded=seeded;
            state.grain_omega.copy_from(device,&omega)?;
            state.eval_x.copy_from(device,&state.positions)?;
            if let Some(tx)=state.roots.transaction.as_mut(){tx.abort_trial();}
            if adaptive_cap {
                state.roots.retry_step(None);
                return outcome.map(|mut result|{result.retry_rejections=rejections;result});
            }
            if !wants_retry {state.roots.retry_step(None);return outcome;}
            let root_request=if state.roots.curve_retry {state.roots.retry_request()}else{None};
            let chart_request=chart_failed_dt.or(material_failed_dt).and_then(|dt|super::joints::chart_retry_dt(dt,history.0));
            requested=match (root_request,chart_request) {
                (Some(a),Some(b))=>Some(a.min(b)),(Some(a),None)=>Some(a),(None,Some(b))=>Some(b),(None,None)=>None,
            };
            if retry==max_rejections || requested.is_none_or(|dt|!dt.is_finite()||dt<=0.0)
                || ((chart_failed_dt.is_some() || material_failed_dt.is_some()) && chart_request.is_none()) {
                state.roots.retry_step(None);
                if chart_failed_dt.is_some() || material_failed_dt.is_some() {
                    return Err(Fatal::invariant(format!("JOINT_CHART_FEASIBILITY_UNRESOLVED: candidate domain subdivision exhausted or dt underflow after {retry} restored subdivisions; original={:?}",outcome.as_ref().err())));
                }
                return Err(Fatal::invariant(format!("ROOT_MOTION_SWEEP_UNRESOLVED: curved CCD exhausted {max_rejections} restored physical subdivisions")));
            }
            if chart_failed_dt.is_some() {state.joints.consume_candidate_faults(device)?;}
            if material_failed_dt.is_some() {state.joints.consume_material_spin_faults(device)?;}
            rejections+=1;
            ::log::info!("native trial: rejected uncommitted {}; restored physical state and retry dt {:?}, subdivision {}",
                if chart_failed_dt.is_some(){"finite current chart domain"}else{"root curve"},requested,retry+1);
        }
        unreachable!()
    })();
    // Evaluate every free before propagating any error. The original native
    // failure has precedence; cleanup must not hide the solver's diagnosis.
    let mut cleanup_error=None;
    for freed in [fix.free(device),rigid.free(device),rotation.free(device),omega.free(device),accepted_positions.free(device),accepted_previous.free(device)] {
        if let Err(error)=freed {
            if cleanup_error.is_none() {cleanup_error=Some(Fatal::from(error));}
        }
    }
    match (outcome,cleanup_error) {
        (Err(original),Some(cleanup))=>{
            ::log::error!("retry snapshot cleanup failed; native failure retained: {:?}",cleanup);
            Err(original)
        },
        (Err(original),None)=>Err(original),
        (Ok(_),Some(cleanup))=>{
            if let Some(original)=domain_sink.as_deref().and_then(|sink|sink.as_ref()).map(|refusal|refusal.original().clone()) {
                let mut original=original;original.detail.push_str(&format!("; retry snapshot cleanup failed: {}",cleanup.detail));Err(original)
            } else {Err(cleanup)}
        },
        (Ok(result),None)=>Ok(result),
    }
}

unsafe fn advance_trial<D: Device>(
    device: &mut D,
    data: &DataSet,
    param: *mut ParamSet,
    state: &mut SolverState,
    adaptive_retry: bool,
    allow_swing_retry:bool,
    prepared:Option<crate::joint_root_motion_state::NativeStep>,
    c5_committed:&mut bool,
    newton_candidate:&mut Option<(f32,u32,f32)>,
    kinematics_shortened:&mut Option<f32>,
) -> FatalResult<StepResult> {
    // Name: Time Per Simulation Step
    // Format: list[(time, ms)]
    // Map: time_per_step
    // Description:
    // Wall-clock time in milliseconds spent inside a single advance call
    // (one simulation step). Note that a step does not advance by a fixed
    // dt: the actual step size is reduced by the accumulated time of
    // impact found during the inner Newton loop, so these values also
    // reflect how hard the solver had to work to progress the step.
    let _section = log::Section::new("advance");
    // THE DOCSTRING ABOVE IS LOAD-BEARING AND ITS PLACEMENT IS TOO.
    // `parsers.rs` registers a channel only when a `Name:` field precedes the
    // call, with everything between `Description:` and the call becoming the
    // text the addon shows. So no note of ours may sit inside that run, which
    // is why this one is below the call rather than above it.
    //
    // The section is the step's own log scope: opening it prints
    // `====== advance ======`, dropping it prints
    // `===== advance: N msec =====` and writes the step's total to
    // `advance.out`.
    //
    // DECLARED FIRST SO IT DROPS LAST. Rust drops locals in reverse
    // declaration order, and every early return below drops it on the way out,
    // so the footer closes a failed step as well as a clean one.
    // PACING, BEFORE ANY WORK. A zero delay is the default and costs a branch
    // on a `Duration` that is already resolved; see `step_delay`. It sits here
    // rather than at the end so a caller sampling between steps observes a run
    // IN PROGRESS rather than one between steps, which is the state the seven
    // observation scenarios assert on.
    let delay = step_delay();
    if !delay.is_zero() {
        std::thread::sleep(delay);
    }

    // STEP A0. Step entry: the result opens optimistic and every failure path
    // below clears exactly the flag it failed on, which is what lets
    // `backend.rs` derive a crash sub-kind without parsing a log.
    let mut result = StepResult {
        time: 0.0,
        ccd_success: true,
        pcg_success: true,
        intersection_free: true,
        newton_progress: true,
        pin_feasible: true,
        contact_separated: true,
        retry_reason: crate::data::RETRY_REASON_NONE,
        retry_iterations: 0,
        retry_residual: 0.0,
        retry_rejections: 0,
    };

    // `time_f32` is written from the double clock BEFORE the record is
    // snapshotted, and the wind ramp reads it, so the order matters.
    (*param).time_f32 = (*param).time as f32;
    // THE SNAPSHOT. It matters
    // that this is a copy and not a borrow: STEP C3 writes `prev_dt` and `time`
    // back through the same pointer, and every field read below must be the one
    // the step OPENED with. Only the fields the driver reads are taken, so the
    // record need not be `Clone` and a field that arrives later cannot be read
    // by accident from a stale copy.
    let prm = StepParams::snapshot(param);
    log::set_time(prm.time);

    let vertices = state.sizes.vertices;
    // THE AGGREGATE LOCK'S GROUPS, read once per step. Zero is the ordinary
    // case and every projector method returns early on it, so an unlocked scene
    // pays one comparison.
    let locked_groups = data.translation_lock.size as usize;
    // How many PDRD rigid bodies. A body's vertices own no per-vertex degrees
    // of freedom, so one body routes the WHOLE system through the reduction.
    let pdrd_bodies = state.rigid.n_bodies;
    // How many SAND grains the scene carries, taken from the SCENE and not from
    // the contact layer.
    //
    // THE DISTINCTION IS NOT ACADEMIC: `disable-contact` leaves
    // `state.contact` at `None`, so reading the count off it reports zero for a
    // scene that plainly has grains, and the three sites below would then
    // condense, recover and integrate nothing. The analytic colliders are
    // assembled OUTSIDE that gate, so a grain resting on a floor still needs
    // every one of them. The count is derived at setup, scanning the rolling
    // inertia once.
    let grains = state.sizes.grains;
    // Safety: the scene is live and holds `locked_groups` records.
    let locked_records: &[crate::data::TranslationLock] =
        super::scene::slice(&data.translation_lock);
    if vertices == 0 {
        result.time = prm.time;
        return Ok(result);
    }
    if data.vertex.curr.size as usize != vertices || data.vertex.prev.size as usize != vertices {
        return Err(Fatal::invariant(format!(
            "solver driver: the scene was allocated for {vertices} vertices and now reports \
             {} current and {} previous positions",
            data.vertex.curr.size, data.vertex.prev.size
        )));
    }

    // THE REFUSALS ARE ASSERTED, NOT ASSUMED. Each of these is refused by name
    // at `initialize()`; reaching one here means the gate and the driver
    // disagree, which is the one failure this backend must never answer with a
    // plausible frame.
    assert_unsupported_absent(data, param)?;
    // A principal coupled root inverse is not implemented by this bounded
    // hard adapter. Refuse before any target/pose/operator/clock write.
    // Exact root+singleD6 domain is sealed and checked by hard install/freeze.
    // Additional roots/hinges/projectors remain typed refusals below.


    let curr = state.positions.handle();
    // THE MESH TOPOLOGY, taken once. `mesh_face` and `mesh_edge` are staged
    // buffers the state seeds at allocate and the mesh does not move after the
    // build, so the contact subsystem takes the two handles rather than its own
    // copies. `StagedBuffer::handle` asserts the device copy is current, so a
    // missed upload traps here rather than dispatching stale topology.
    let mesh_refs = state.refs();
    let prev = state.positions_prev.handle();

    // The collision-window masks, taken ONCE and held for the whole step. They
    // are read by every broad-phase pass and by the intersection scan, and the
    // host may install a new table only between steps, so one borrow is both
    // cheaper and the only reading that cannot see a table change mid-step.
    let mut window_guard = super::lock_collision_windows();
    let windows = Windows {
        vertex: window_guard.as_mut().and_then(|table| table.vertex_active()),
        edge: window_guard.as_mut().and_then(|table| table.edge_active()),
        face: window_guard.as_mut().and_then(|table| table.face_active()),
    };

    let _pre_newton = std::time::Instant::now();
    // STEP A1. The initial scene build owns the primitive topology for this
    // Contact lifetime. Refit all leaves, window masks and internal bounds to
    // the current pose on every step; no old bounds or contacts are reused.
    // A full rebuild remains available for diagnostics and comparison.
    if !prm.disable_contact {
        let Some(contact) = state.contact.as_mut() else {
            return Err(Fatal::invariant(
                "solver driver: this scene wants contact and the contact subsystem was never \
                 allocated, so the step would run with no broad phase, no barrier, no CCD \
                 filter and no intersection gate. That is a run without the penetration \
                 guarantee, which this backend does not perform",
            ));
        };
        let rebuild = std::env::var_os("PPF_BVH_REBUILD_EACH_STEP").is_some_and(|v| v == "1");
        let _phase = super::phase::start(if rebuild { "contact.rebuild_trees" } else { "contact.refit_trees" });
        let _lbvh = std::time::Instant::now();
        if rebuild {
            contact.rebuild_trees(device, data, mesh_refs, curr, windows)?;
        } else {
            contact.refresh_leaves(device, data, mesh_refs, curr, curr, 1.0, windows)?;
        }
        drop(_phase);
        // Name: LBVH Build Time
        // Format: list[(time, ms)]
        // Map: lbvh_build
        // Description:
        // Wall-clock time to refit (or diagnostically rebuild) the LBVH
        // over faces, edges, and vertices at the start of each step.
        log::mark("advance", "lbvh_build", _lbvh.elapsed());
    }

    // STEP A3. The start-of-step velocity, the largest free-vertex speed, and
    // the domain check. One pass over the two position arrays for all three.
    let velocity_args = VelocityTermsArgs {
        current: curr,
        previous: prev,
        prop: state.prop_vertex.handle(),
        previous_dt: prm.prev_dt,
        velocity: state.velocity.handle(),
        speed_squared: state.scalar.handle(),
        reach: state.scalar_b.handle(),
        count: vertices as u32,
        seam_arena_count: 0,
    };
    device.launch("step.velocity", &velocity_args, vertices as u32)?;
    // THE TWO FOLDS READ THE MIRRORS, so both come back across the seam first;
    // each went stale at its own `handle()` above.
    //
    // THE VERDICTS ARE SCALARS AND ONLY THE SCALARS MOVE. `max_u` and `reach`
    // are one float each, so folding them on the host would move `vertices`
    // floats twice across the seam. Both folds run on the device and only the
    // four bytes of each verdict come back.
    let (max_u, reach) = {
        let a = state.scalar.handle();
        let b = state.scalar_b.handle();
        // Safety: both buffers outlive the calls and name `vertices` floats.
        let u = unsafe { state.fold.max(device, "step.max_u", a, vertices as u32, 0.0) }?;
        let r = unsafe { state.fold.max(device, "step.reach", b, vertices as u32, 0.0) }?;
        (u.sqrt(), r)
    };
    let domain = position_domain_abi();
    if !(reach < domain) {
        // LEAVING THE DOMAIN IS NOT CAUGHT ANYWHERE ELSE: the overflow guards
        // in the shared coordinate header are behind a debug switch that is
        // never defined, so a position past the end of the representable range
        // is carried on silently and the run continues on nonsense. Ingest is
        // bounded at scene build, so what this catches is a position that MOVED
        // out during the run.
        return Err(Fatal::invariant(format!(
            "solver driver: a vertex has moved to {reach:.6} from the origin, outside the \
             +/-{domain:.6} domain the position representation covers. Nothing past that \
             end is a meaningful position, so the run cannot continue. The scene either drifts \
             or has blown up; check max_u ({max_u:.3e} here) and the frames before this one"
        )));
    }
    // Name: Max Vertex Velocity
    // Format: list[(time, m/s)]
    // Map: max_velocity
    // Description:
    // Maximum speed (in meters per second) among all non-pinned vertices,
    // measured from the previous to the current positions at the start of
    // the step. Pinned (fixed) vertices are excluded. Useful for spotting
    // explosions or abrupt motion in the simulation.
    log::mark("advance", "max_u", max_u as f64);

    // STEP A5. The step size this advance aims at. Host arithmetic on two
    // scalars the scene already carries.
    super::roots::gate_runtime(data,state,prm.playback)?;
    let authored_root_plan = state.roots.plan_step(prm.time,prm.dt * prm.playback)?;
    let root_plan = state.fem_roots.plan(prm.time,prm.dt*prm.playback,authored_root_plan)?;
    let root_plan = if kinematics_shortened.is_some() {
        prepared.ok_or_else(|| Fatal::invariant("KINEMATICS_PLAN: missing prepared endpoint"))?
    } else {
        if prepared.is_some_and(|expected| expected != root_plan) {
            return Err(Fatal::invariant("HOST_RETRY_PLAN: native endpoint changed after host sampling"));
        }
        root_plan
    };
    if state.fem_roots.runtime.is_some()&&prm.playback!=1. {return Err(Fatal::invariant("FEM event physical clock requires unit playback"));}
    let mut dt = root_plan.integration_dt_s;
    let mut root_attempt=if state.fem_roots.runtime.is_some(){state.roots.begin_shared(device,root_plan)?}else{state.roots.begin(device,root_plan)?};
    // Preserve an exact authored root boundary within the existing bounded
    // Newton loop. B3 otherwise shortens every accepted prefix and can leave
    // a geometric series of ever smaller trials before the same key.
    let preserve_root_boundary=prm.max_newton_steps>0 && authored_root_plan.lands_on_boundary
        && root_plan.end_s==authored_root_plan.end_s && root_attempt.is_some();
    state.fem_roots.begin(root_plan)?;
    state.fem_roots.gate_shared_clock(root_attempt.as_ref())?;
    if state.fem_roots.prepared_end!=Some(root_plan.end_s){state.fem_roots.retarget(device,&mut state.fix)?;}
    // Name: Target Step Size
    // Format: list[(time, seconds)]
    // Description:
    // Target integration step size in seconds at the start of this
    // simulation step, computed as the configured dt scaled by the current
    // playback rate. The actually advanced step size can be smaller (see
    // the Final Step Size channel) if the line search reduces it.
    log::mark("advance", "dt", dt as f64);
    // Name: Playback Speed
    // Format: list[(time, ratio)]
    // Description:
    // Playback rate applied this step, as a multiplier on the configured
    // dt. A value of 1.0 means real-time playback, below 1.0 slows motion
    // down, and above 1.0 speeds it up. The value can change between
    // steps when the scene scripts playback over time.
    log::mark("advance", "playback", prm.playback as f64);

    // STEP A6. `max_sigma`, the largest shell / rod stretch ratio, measured at
    // the start-of-step pose. Telemetry rather than a term of the solve: nothing
    // below reads it and a scene runs the same trajectory with and without it.
    // It is emitted because it is one of the three indicator streams a lost PCG
    // residual denominator is read off, and a collapsing `SL_toi` is only
    // diagnostic beside it.
    //
    // GATED ON THE TWO ELEMENT COUNTS, so a tet-only scene pays nothing and
    // emits nothing. The reduction is here rather than in the walk because this
    // backend's determinism rests on every fold having one stated shape;
    // `reduce::max`'s floor argument is the identity the fold opens with, which
    // is how the shell and rod folds chain into one verdict.
    if state.sizes.shell_faces > 0 || state.sizes.rods > 0 {
        assemble::stretch_indicator(device, data, state)?;
        let mut max_sigma = 0.0f32;
        {
            let faces = state.sizes.shell_faces as u32;
            let rods = state.sizes.rods as u32;
            let shell = state.stretch.ratio.handle();
            let rod = state.stretch.rod_ratio.handle();
            // Safety: both buffers outlive the calls and name their counts.
            max_sigma =
                unsafe { state.fold.max(device, "step.max_sigma", shell, faces, max_sigma) }?;
            max_sigma =
                unsafe { state.fold.max(device, "step.max_sigma_rod", rod, rods, max_sigma) }?;
        }
        // Name: Max Stretch Ratio
        // Format: list[(time, ratio)]
        // Description:
        // Maximum stretch ratio among all shell faces and rod edges in the
        // scene, measured at the start of the step before the Newton loop.
        // For shells this is the largest singular value of the deformation
        // gradient (scaled by the shrink factor), for rods it is the current
        // edge length divided by its rest length. A value of 1.02 means a
        // 2 percent stretch. Useful for diagnosing strain-limit tightness.
        log::mark("advance", "max_sigma", max_sigma as f64);
    }

    // STEP A7. The implicit target, and the iterate seeded at the current pose.
    // THE PIN INDEX, GATHERED ON THE DEVICE off the record that already holds
    // it. A host pass would be O(vertices) plus one upload per advance for a
    // field `prop_vertex` already carries.
    {
        let count = state.sizes.vertices as u32;
        let args = crate::driver::kernels::VertexFixIndexFromRecordsArgs {
            prop: state.prop_vertex.handle(),
            fix_index: state.fix_index.handle(),
            count,
            seam_arena_count: 0,
        };
        // Safety: both arrays are borrowed for the whole call.
        unsafe { device.launch("step.fix_index", &args, count) }?;
    }
    // TAKES THE THREE ARRAYS RATHER THAN `&mut SolverState`, so it can be
    // called while other fields of the same state are borrowed, and so no
    // buffer has to be moved out and put back. The put-back version had a real
    // defect: an early return from a failed step never reached the restore,
    // and the NEXT step then gathered the pin indices into an empty slice and
    // handed the shared body a pointer to nothing.
    //
    // THE EXTERNAL FORCE FIELD IS EVALUATED FIRST, ONCE PER STEP, at the
    // committed positions and the step's opening clock. Every target this step
    // builds (here, after a dt cut, and on each continuing iteration) reads
    // that one evaluation, which is what keeps the term out of the Newton
    // system's Hessian. A scene with no field dispatches nothing here.
    state
        .field
        .evaluate(device, curr, state.prop_vertex.handle(), prm.time as f32)?;
    compute_target(
        device,
        &prm,
        curr,
        prev,
        vertices,
        state.target.handle(),
        state.fix.handle(),
        dt,
        state.fix_index.handle(),
        state.field.acceleration(),
    )?;
    state.roots.target(device,state.rigid_staged.vert_list.handle(),state.rigid_staged.rest_centered.handle(),state.target.handle(),data.pdrd_vert_list.size,vertices as u32)?;
    // The iterate starts at the committed pose. DEVICE WORK now that both are
    // device buffers, rather than a round trip through the host.
    state.eval_x.copy_from(device, &state.positions)?;
    // This first extension milestone uses physical seconds. The upstream
    // animation clock advances by dt/playback, so nonunit playback requires a
    // separate clock contract and must not silently sample the wrong target.
    if state.joints.has_extensions() && prm.playback != 1.0 {
        return Err(Fatal::invariant("JOINT_ANGULAR_EXTENSION_CLOCK_UNSUPPORTED: playback must be 1 for physical-time drive keys"));
    }
    state.joints.set_step_start_time(prm.time).map_err(Fatal::from)?;
    if let Some(attempt)=root_attempt.as_ref(){state.joints.set_step_end_time(attempt.end_s,dt)?;}
            else if let Some(attempt)=state.fem_roots.attempt.as_ref(){state.joints.set_step_end_time(attempt.end,dt)?;}
    super::joints::begin_step(device, data, state, dt).map_err(|error|{
        let _=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||eprintln!("FEM_PREPARE_CONTEXT source=substep_start parameter_time={:.17e} newton_step=unavailable adaptive_retry_token={} dt_bits={:08x} accepted_commit_clock=unavailable",prm.time,state.adaptive_retry_token,dt.to_bits())));error
    })?;
    super::restoring::begin_step(device,state,prm.time,dt)?;
    // Re-measure the real substep-start/resume pose. Empty hard scenes do nothing.
    if state.hard.active() {
        let twist_context=state.joints.twist_context();
        let SolverState {hard,rigid,rigid_staged,prop_vertex,eval_x,..}=&mut *state;
        let mut reduction=super::pdrd::Reduction {map:rigid,staged:rigid_staged};
        hard.gate_candidate(device,&mut reduction,prop_vertex.handle(),eval_x.handle(),
            data.pdrd_vert_list.size,true,twist_context)?;
    }


    // STEP A8. Which vertices leave the system as exact Dirichlet rows. A fix
    // pin inside a PDRD body keeps its barrier instead (it owns no per-vertex
    // DOF), and the A/B lever reverts every pin to the barrier; PDRD is refused
    // here, so the mask is `fix_index > 0` unless the lever is set.
    //
    let disable_dof_removal = prm.disable_pin_dof_removal;
    // ON THE DEVICE, BOTH HALVES. A dispatch over the vertices writes the mask
    // off the record (`vertex_dof_removal_mask`), and the fold below reduces it
    // to one word. Walking every vertex on the host, uploading the mask and
    // summing the host copy would put a per-vertex pass and a reduction on the
    // host, which is the divergence this shape exists to avoid.
    //
    // THE FOLD IS A `u32` ONE, NOT THE FLOAT FORM. `driver::reduce` is the CPU
    // backend's fold arm over host slices, and `vec_block_sum_u32` sits beside
    // the float form for this count: a count folded through `float` is exact
    // only under 2^24, and a scene past sixteen million marked vertices would
    // round its own tally rather than refuse.
    {
        let count = vertices as u32;
        let mask_args = crate::driver::kernels::VertexDofRemovalMaskArgs {
            prop: state.prop_vertex.handle(),
            disable: u32::from(disable_dof_removal),
            mask: state.dof_mask.handle(),
            count,
            seam_arena_count: 0,
        };
        // Safety: both arrays are borrowed for the whole call.
        unsafe { device.launch("step.dof_mask", &mask_args, count) }?;
    }
    // LEVELS UNTIL ONE WORD, alternating into the scratch and ending in
    // `dof_total`, which is the only thing read back.
    let removed: u32 = {
        let width = crate::driver::state::DOF_FOLD_WIDTH;
        let mut input = state.dof_mask.handle();
        let mut remaining = vertices;
        let mut cursor = 0usize;
        loop {
            let blocks = remaining.div_ceil(width);
            let destination = if blocks == 1 {
                state.dof_total.handle()
            } else {
                state.dof_fold.span(cursor, blocks)
            };
            let args = crate::driver::kernels::VecBlockSumU32Args {
                source: input,
                length: remaining as u32,
                width: width as u32,
                total: destination,
                count: blocks as u32,
                seam_arena_count: 0,
            };
            // Safety: every handle names a live allocation for the call.
            unsafe { device.launch("step.dof_fold", &args, blocks as u32) }?;
            if blocks == 1 {
                break;
            }
            input = state.dof_fold.span(cursor, blocks);
            cursor += blocks;
            remaining = blocks;
        }
        state.dof_total.download(device)?;
        state.dof_total.host()[0]
    };
    // One-time visibility, and silent in a scene with no pins at all. It is
    // not decoration: a driver that skipped the elimination would still hold
    // the pins approximately, so this line and the tracking error are the two
    // things that separate an exact Dirichlet pin from a penalty.
    //
    // A LINE THIS DRIVER EMITS ON ITS OWN ACCOUNT, AND SO IT KEEPS ITS
    // `::log::` PREFIX. The DOF-removal count is not part of the solver's own
    // transcript, so it takes the timestamped Rust convention; only a line
    // belonging to that transcript goes out bare (see `super::log`).
    if !DOF_REPORTED.swap(true, std::sync::atomic::Ordering::SeqCst)
        && (removed > 0 || disable_dof_removal)
    {
        ::log::info!(
            "dof-removal: {removed} pinned vertices eliminated as Dirichlet BCs ({})",
            if disable_dof_removal {
                "DISABLED via env"
            } else {
                "enabled"
            }
        );
    }
    if disable_dof_removal && removed == 0 && !state.fix.is_empty() {
        // The lever leaves every fix pin on the barrier path, which this
        // driver does not assemble. Rather than run a scene whose pins are held
        // by nothing, say so.
        return Err(Fatal::invariant(
            "solver driver: PPF_DISABLE_PIN_DOF_REMOVAL reverts every fix pin to the contact \
             barrier, which this backend does not assemble, so the pins would be held by \
             nothing at all. Unset it to run this scene here",
        ));
    }

    let mut analytic_assembly_ledger=AnalyticAssemblyLedger::new();
    let mut frozen_newton_witness=pcg::FrozenNewtonWitness::default();
    let frozen_joint_tangent = std::env::var_os("PPF_JOINT_FROZEN_TANGENT")
        .is_some_and(|value| value == "1");
    let mut analytic_gap_trace=AnalyticGapTrace::new(prm.time,vertices,(*param).ccd_eps);
    analytic_gap_trace.capture_props(device,state.prop_vertex.handle());
    if analytic_gap_trace.enabled() {
        let list=super::scene::slice(&data.pdrd_vert_list);
        let rest=super::scene::slice(&data.pdrd_rest_centered);
        let props=super::scene::slice(&data.prop.pdrd_body);
        analytic_gap_trace.root_layout=Some(serde_json::json!({
            "pdrd_vertex_list":list,
            "rest_centered_gpu_source_bits":rest.iter().map(|p|std::array::from_fn::<_,3,_>(|a|p[a].to_bits())).collect::<Vec<_>>(),
            "body_spans":props.iter().enumerate().map(|(body,p)|serde_json::json!({"body":body,"start":p.vertex_start,"count":p.vertex_count})).collect::<Vec<_>>(),
            "controlled_bodies":root_attempt.as_ref().map(|a|a.body_indices.as_slice()),
            "reduced_body_base":state.rigid.body_base,"reduced_dimension":state.rigid.dim}));
    }
    if analytic_gap_trace.enabled() { analytic_gap_trace.capture(device,"accepted-current",0,false,dt,0.0,1.0,1.0,state.positions.handle(),state.positions.handle(),None,state.floor.host(),root_attempt.as_ref()); }
    let mut sand_offset_trace=SandOffsetTrace::new(prm.time,vertices);
    if sand_offset_trace.enabled {sand_offset_trace.capture(device,"accepted-current",0,false,dt,0.0,1.0,1.0,state.positions.handle(),state.positions.handle(),None);}
    let mut toi_advanced: f64 = 0.0;
    let mut full_remaining_sweep_accepted = false;
    let mut step: u32 = 1;
    let mut final_step = false;
    let mut final_clock_initialized = false;
    let mut last_toi: f32 = 1.0;
    // This witness exists only for this advance and is set only after the
    // final successful root certificate fitted the current eval_x.
    let mut certified_final_fit_step: Option<u32>;
    let final_fit_mode = std::env::var_os("PPF_REUSE_FINAL_FIT");
    let final_fit_reuse = final_fit_mode.as_deref() == Some(std::ffi::OsStr::new("1"));
    let final_fit_receipt = matches!(final_fit_mode.as_deref(), Some(v) if v == "0" || v == "1");

    // RECORDED AND NOT PRINTED. The head of the step is not one of the
    // solver's own printed scopes, so a printed row here would read, in a
    // transcript diff, as this driver's own noise. The stream file still
    // carries it.
    log::mark(
        "advance",
        "pre_newton",
        log::Marked::quiet(_pre_newton.elapsed().as_secs_f64() * 1000.0),
    );
    loop {
        certified_final_fit_step = None;
        let _nt_head = std::time::Instant::now();
        // STEP B0. The bounded escape. A bound, not a heuristic: no progress
        // window and no tuned epsilon. See the module docs for why the other
        // escape does not cover this.
        let active_roots=root_attempt.as_ref().is_some_and(|a|!a.body_indices.is_empty());
        if root_final_budget_exhausted(step,prm.max_newton_steps,final_step,active_roots) {
            // BOTH LINES BELONG TO THE SOLVER'S OWN TRANSCRIPT, so both are
            // bare. The two exponents are rebuilt because C's `%.2e` writes
            // `5.96e-09` where Rust's `{:.2e}` writes `5.96e-9`.
            log::message!(
                "### newton stalled: no acceptable step after {step} iterations \
                 (last toi: {}, toi_advanced: {})",
                log::c_exponential(f64::from(last_toi), 2),
                log::c_exponential(toi_advanced, 2)
            );
            log::message!(
                "### an over-constrained configuration cannot be advanced: a prescribed pin \
                 driven into geometry that cannot yield has no way to resolve. Re-author the \
                 pin's path, or make it a soft pull pin."
            );
            analytic_gap_trace.flush("Newton iteration bound (original rejection retained)",(*param).ccd_eps,prm.line_search_max_t);
            // TERMINAL ONLY: cap is already true. No read/sync before B0 refusal.
            let terminal_result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut terminal=AnalyticGapTrace::new_terminal(prm.time,vertices,(*param).ccd_eps);
                if terminal.enabled() {
                    terminal.capture_props(device,state.prop_vertex.handle());
                    let list=super::scene::slice(&data.pdrd_vert_list);
                    let rest=super::scene::slice(&data.pdrd_rest_centered);
                    let props=super::scene::slice(&data.prop.pdrd_body);
                    terminal.root_layout=Some(serde_json::json!({
                        "pdrd_vertex_list":list,
                        "rest_centered_gpu_source_bits":rest.iter().map(|p|std::array::from_fn::<_,3,_>(|a|p[a].to_bits())).collect::<Vec<_>>(),
                        "body_spans":props.iter().enumerate().map(|(body,p)|serde_json::json!({"body":body,"start":p.vertex_start,"count":p.vertex_count})).collect::<Vec<_>>(),
                        "controlled_bodies":root_attempt.as_ref().map(|a|a.body_indices.as_slice()),
                        "reduced_body_base":state.rigid.body_base,"reduced_dimension":state.rigid.dim}));
                    let query_toi=state.analytic.as_mut().and_then(|a|a.diagnostic_toi(device).ok());
                    terminal.capture(device,"terminal_B0_pre_reject",step,final_step,dt,toi_advanced,1.0,last_toi,
                        state.positions.handle(),state.eval_x.handle(),query_toi,state.floor.host(),root_attempt.as_ref());
                    let root_view=state.roots.view(&state.rigid);
                    terminal.capture_root_buffers(device,"terminal-current-root-workspaces",pdrd_bodies as usize,
                        state.rigid_state.handle(),state.rigid_rotation_step.handle(),state.rigid_running_rotation.handle(),state.rigid_centroid.handle(),root_view);
                    terminal.capture_root_direction(device,state.dx.handle(),state.pcg_rigid.xr.handle(),vertices,state.pcg_rigid.xr.len());
                    let mut extras=serde_json::Map::new();
                    if let Some(count)=vertices.checked_mul(3) {
                        for (name,handle,len) in [
                            ("accepted_previous_positions",state.positions_prev.handle(),state.positions_prev.len()),
                            ("rebuilt_current_target_workspace",state.target.handle(),state.target.len()),
                            ("current_rigid_target_workspace",state.rigid_target.handle(),state.rigid_target.len())] {
                            let bits=if len==count {sand_offset_read::<f32>(device,handle,count).map(|v|v.into_iter().map(f32::to_bits).collect::<Vec<_>>())} else {None};
                            extras.insert(name.into(),serde_json::json!({"expected_f32_count":count,"allocated_f32_count":len,"gpu_f32_bits":bits}));
                        }
                    }
                    if let Some(phase)=terminal.phases.back_mut() {
                        if let Some(object)=phase.as_object_mut() {
                            object.insert("terminal_only".into(),serde_json::json!({
                                "history":false,"bound_iteration":step,"last_completed_iteration":step.saturating_sub(1),
                                "last_toi_bits":last_toi.to_bits(),"accepted_previous_dt_bits":prm.prev_dt.to_bits(),
                                "accepted_clock_bits":prm.time.to_bits(),"extra_actual_gpu_buffers":extras,
                                "roles":"x0=accepted positions; x1=current eval iterate; target has been rebuilt by B33 and is NOT last CCD origin. Rigid target, fitted state, running_R, dtheta, dx/xr and TOI are current terminal workspaces; last writer is not inferred. No re-fit, re-projection or historical replay."}));
                        }
                    }
                    let assembly_ledger=analytic_assembly_ledger.read(device,state,step);
                    let witness_result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||
                        frozen_newton_witness.at_cap(device,&state.rigid,step)));
                    let witness=match witness_result {Ok(value)=>value,Err(_)=>serde_json::json!({
                        "status":"OPTIONAL_WITNESS_PANICKED","original_rejection_retained":true})};
                    if let Some(phase)=terminal.phases.back_mut().and_then(serde_json::Value::as_object_mut) {
                        phase.insert("frozen_operator_witness".into(),witness);
                    }
                    if let Some(phase)=terminal.phases.back_mut().and_then(serde_json::Value::as_object_mut) {
                        phase.insert("assembly_stored_ledger".into(),assembly_ledger);
                    }
                    terminal.flush("Terminal-only B0 Newton bound; no stage history; original rejection retained",(*param).ccd_eps,prm.line_search_max_t);
                }
            }));
            if terminal_result.is_err() {let _=std::io::Write::write_fmt(&mut std::io::stderr().lock(),format_args!("ANALYTIC_STALL_TERMINAL_CAPTURE optional observation panicked; original rejection retained\n"));}
            let _=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||analytic_assembly_ledger.cleanup(device)));
    let _=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||frozen_newton_witness.cleanup(device)));
            result.newton_progress = false;
            *newton_candidate=Some((dt,step,last_toi));
            if let Some(attempt)=root_attempt.as_ref(){state.roots.transaction.as_mut().unwrap().reject(attempt).map_err(|e| Fatal::invariant(e.to_string()))?;}
                    return Ok(result);
        }

        // STEP B1. Open the iteration: every accumulator is cleared by the pass
        // that opens the assembly, never by a kernel inside it.
        // THE OPENING CLEAR, on the device. Every element model scatters into
        // this array and the seam refuses a stale mirror, so the zero it opens
        // at is written by a dispatch rather than by the host.
        let force_seed = VecFillArgs {
            array: state.force.handle(),
            value: 0.0,
            count: (3 * vertices) as u32,
            seam_arena_count: 0,
        };

        // THE SEARCH DIRECTION opens at zero on the device, as the right-hand
        // side above does.
        let dx_seed = VecFillArgs {
            array: state.dx.handle(),
            value: 0.0,
            count: (3 * vertices) as u32,
            seam_arena_count: 0,
        };

        // THE BLOCK DIAGONAL opens at zero on the device, as the two arrays
        // above do; momentum and the Dirichlet rows accumulate into it.
        let diagonal_seed = VecFillArgs {
            array: state.diagonal.handle(),
            value: 0.0,
            count: (9 * vertices) as u32,
            seam_arena_count: 0,
        };

        let fixed_bytes = state.fixed_values.len() * std::mem::size_of::<f32>();
        let fixed_handle = state.fixed_values.handle();
        // Contact assembly is the first writer of this per-iteration counter.
        // Clear it with the other Newton seeds, before any contact can run.
        let contact_count_bytes = state.statistics_contact_count.len() * size_of::<u32>();
        let contact_count_handle = if contact_count_bytes > 0 {
            Some(state.statistics_contact_count.handle())
        } else {
            None
        };
        device.run("step.open.clear", |encoder| {
            encoder.elements(&force_seed, (3 * vertices) as u32)?;
            encoder.elements(&dx_seed, (3 * vertices) as u32)?;
            encoder.elements(&diagonal_seed, (9 * vertices) as u32)?;
            ppf_cts_compute::encode_zero(encoder, fixed_handle, fixed_bytes)?;
            if let Some(counter) = contact_count_handle {
                ppf_cts_compute::encode_zero(encoder, counter, contact_count_bytes)?;
            }
            Ok(())
        })?;

        // THE STEP ANNOUNCES ITSELF, and the two wordings are a HOST CONTRACT.
        // `------ newton step N ------` opens an ordinary iteration and
        // `------ error reduction step ------` the trailing one. A reader
        // following a run, and a gate counting iterations, have nothing to read
        // without these two lines: a structurally marked step that prints
        // nothing shows a scene finishing with no evidence that any Newton
        // iteration happened.
        if final_step {
            log::message!("------ error reduction step ------");
        } else {
            log::message!("------ newton step {step} ------");
        }

        if sand_offset_trace.enabled {sand_offset_trace.capture(device,"pre-B3",step,final_step,dt,toi_advanced,1.0,1.0,state.positions.handle(),state.eval_x.handle(),None);}
        if analytic_gap_trace.enabled() {analytic_gap_trace.capture(device,"pre-B3",step,final_step,dt,toi_advanced,1.0,1.0,state.positions.handle(),state.eval_x.handle(),None,state.floor.host(),root_attempt.as_ref());}
        if analytic_gap_trace.enabled() {
            let root_view=state.roots.view(&state.rigid);
            analytic_gap_trace.capture_root_buffers(device,"pre-B3-root-workspace",pdrd_bodies as usize,
                state.rigid_state.handle(),state.rigid_rotation_step.handle(),state.rigid_running_rotation.handle(),state.rigid_centroid.handle(),root_view);
            analytic_gap_trace.capture_root_direction(device,state.dx.handle(),state.pcg_rigid.xr.handle(),vertices,state.pcg_rigid.xr.len());
        }

        // STEP B3. The trailing error-reduction iteration: shrink the clock to
        // what was actually integrated and bring the kinematic pins back in
        // step with it BEFORE the targets are rebuilt off them.
        if final_step {
            if initialize_final_clock(final_step,final_clock_initialized) {
            #[cfg(all(test,feature="cpu"))]
            if PRIVATE_TEST_B3_PCG.compare_exchange(1,2,std::sync::atomic::Ordering::SeqCst,std::sync::atomic::Ordering::SeqCst).is_ok() {
                // Exercise the real B3 shortening and solve on a controlled CCD fraction.
                toi_advanced *= 0.5;
            }
            // Host-only candidate. The host will resample authored input and
            // retry through CCD after full precommit rollback.
            if kinematics_shortened.is_some() && toi_advanced < 1.0 {
                let shortened = (f64::from(dt) * toi_advanced) as f32;
                if shortened.is_finite() && shortened > 0.0 && shortened < dt {
                    *kinematics_shortened = Some(shortened);
                    result.newton_progress = false;
                    return Ok(result);
                }
            }
            if toi_advanced < 1.0 && !state.fix.is_empty() {
                let back = 1.0 - toi_advanced as f32;
                let pins = state.fix.len() as u32;
                let rewind_args = RewindFixArgs {
                    fix: state.fix.handle(),
                    back,
                    count: pins,
                    seam_arena_count: 0,
                };
                device.launch("step.rewind_fix", &rewind_args, pins)?;
            }
            // IN DOUBLE, THEN NARROWED, WHICH IS NOT THE OBVIOUS SPELLING.
            // `dt` is fp32 and `toi_advanced` is double, so the product is
            // formed in double and rounded ONCE on the way back. Multiplying in
            // f32 rounds twice and gives a different final step size, which is
            // the clock the whole trailing iteration integrates against.
            dt = (f64::from(dt) * toi_advanced) as f32;
            let certified_requested_dt=dt;
            if let Some(old)=root_attempt.take(){
                let next=state.roots.shorten(device,&old,dt)?;
                // Use the exact returned endpoint's native representation in
                // the target, joint damping, momentum and accepted prev_dt.
                dt=next.integration_dt_s.ok_or_else(||Fatal::invariant("ROOT_MOTION_TIME_MISMATCH: shortened native attempt has no integration duration"))?;
                root_attempt=Some(next);
            }
            let old_fem_end=state.fem_roots.attempt.as_ref().map(|a|a.end);
            dt=if let Some(shared)=root_attempt.as_ref(){state.fem_roots.shorten_shared(shared.start_s,shared.end_s,dt,certified_requested_dt)?}else{state.fem_roots.shorten(dt)?};
            state.fem_roots.gate_shared_clock(root_attempt.as_ref())?;
            if state.fem_roots.attempt.as_ref().map(|a|a.end)!=old_fem_end {state.fem_roots.retarget(device,&mut state.fix)?;}
            final_clock_initialized=true;
            // CCD can shorten the final physical duration inside this Newton
            // attempt. Refresh damping dt without recapturing accepted history.
            super::restoring::begin_step(device,state,prm.time,dt)?;
            }
            if let Some(attempt)=root_attempt.as_ref(){state.joints.set_step_end_time(attempt.end_s,dt)?;}
            else if let Some(attempt)=state.fem_roots.attempt.as_ref(){state.joints.set_step_end_time(attempt.end,dt)?;}
            compute_target(
                device,
                &prm,
                curr,
                prev,
                vertices,
                state.target.handle(),
                state.fix.handle(),
                dt,
                state.fix_index.handle(),
                state.field.acceleration(),
            )?;
            state.roots.target(device,state.rigid_staged.vert_list.handle(),state.rigid_staged.rest_centered.handle(),state.target.handle(),data.pdrd_vert_list.size,vertices as u32)?;
        }

        if sand_offset_trace.enabled {sand_offset_trace.capture(device,"post-B3-pre-assembly",step,final_step,dt,toi_advanced,1.0,1.0,state.positions.handle(),state.eval_x.handle(),None);}
        if analytic_gap_trace.enabled() {analytic_gap_trace.capture(device,"post-B3-pre-assembly",step,final_step,dt,toi_advanced,1.0,1.0,state.positions.handle(),state.eval_x.handle(),None,state.floor.host(),root_attempt.as_ref());}
        if analytic_gap_trace.enabled() {
            let root_view=state.roots.view(&state.rigid);
            analytic_gap_trace.capture_root_buffers(device,"post-B3-pre-assembly-root-workspace",pdrd_bodies as usize,
                state.rigid_state.handle(),state.rigid_rotation_step.handle(),state.rigid_running_rotation.handle(),state.rigid_centroid.handle(),root_view);
            analytic_gap_trace.capture_root_direction(device,state.dx.handle(),state.pcg_rigid.xr.handle(),vertices,state.pcg_rigid.xr.len());
        }

        // The pattern's handles, taken once: both matrices below are built
        // over the SAME pattern and each keeps its own copy, which is what
        // makes a push look its block up in the pattern that matrix was built
        // over rather than in whichever one a caller happened to pass.
        let pattern_refs = state.fixed_pattern_refs();
        // The two matrices, wrapped fresh over the scene's pattern. The value
        // arrays are the state's, so nothing here allocates.
        // THE PATTERN IS WALKED ONCE PER RUN, NOT ONCE PER ITERATION. The
        // verdict is cached on the pattern's identity, which is what makes the
        // skip safe: a different pattern re-validates in full.
        let mut checked = state.fixed_pattern_checked;
        let mut fixed = FixedCsr::adopt_validated(
            device,
            pattern_refs,
            data,
            std::mem::take(&mut state.fixed_values),
            &mut checked,
        )?;
        state.fixed_pattern_checked = checked;

        // THE ASSEMBLY SCOPE OPENS HERE, NOT AT THE MOMENTUM LAYER. It opens
        // immediately above this seed and closes after the `num_contact` mark,
        // so it encloses the seed, every elastic layer, the snapshot, both
        // strain limiters, the whole contact assembly and the three marks. It
        // must not open at the momentum layer and close after the stitch: that
        // measures a strict subset and reports it under the `matrix_assembly`
        // name, and the printed order is the tell, since `> matrix_assembly`
        // belongs AFTER `* num_contact`.
        let _assembly = std::time::Instant::now();
        // STEP B4. Seed the search direction on every prescribed row.
        let seed_args = DxSeedArgs {
            eval_x: state.eval_x.handle(),
            target: state.target.handle(),
            prop: state.prop_vertex.handle(),
            dx: state.dx.handle(),
            count: vertices as u32,
            seam_arena_count: 0,
        };
        device.launch("step.dx_seed", &seed_args, vertices as u32)?;

        // RECORDED AND NOT PRINTED: the head of a Newton iteration is not one
        // of the solver's own printed scopes. Its span overlaps the assembly's,
        // which is fine, the two being independent clocks.
        log::mark(
            "advance",
            "nt_head",
            log::Marked::quiet(_nt_head.elapsed().as_secs_f64() * 1000.0),
        );
        // STEP B6. The momentum layer. INSIDE the assembly scope opened at
        // STEP B4 and OUTSIDE the `asm_elastic` sub-timer below, so the
        // `matrix assembly` total counts it and `asm_elastic` does not.
        let _phase = super::phase::start("assemble.momentum");
        assemble::momentum(device, data, param, state, dt)?;
        drop(_phase);

        // THE ELASTIC SUB-TIMER spans the five elastic layers alone: the
        // momentum layer above and the stitch below are outside it, so the
        // channel measures elasticity and nothing else.
        let _asm_elastic = std::time::Instant::now();
        // STEP B7a. The rod bending layer, FIRST of the five elastic
        // dispatches, so it is assembled before every other elastic term. The
        // order is load-bearing because the assembled sums are fp32.
        assemble::rod_bend(device, data, state, &mut fixed, dt)?;

        // STEP B7b. The rod stretch layer, SECOND of the five.
        assemble::rod_stretch(device, data, state, &mut fixed, dt)?;

        // STEP B7c. The shell membrane layer. BEFORE the tets: the five
        // elastic dispatches run as rod bend, rod, shell face, tet and hinge,
        // and the assembled sums are fp32, so the order decides the last bits.
        let _phase = super::phase::start("assemble.shell_membrane");
        assemble::shell_membrane(device, data, prm.eiganalysis_eps, state, &mut fixed, dt)?;
        drop(_phase);

        // STEP B7d. The tet elastic layer.
        state.factor_tet.begin_tet(device,&mut fixed)?;
        assemble::tet_elastic(device, data, prm.eiganalysis_eps, state, &mut fixed, dt)?;
        state.factor_tet.end_tet(device,&mut fixed)?;

        // STEP B7e. The shell hinge bending layer, LAST of the five elastic
        // dispatches, so it is assembled after the membrane and after the
        // tets.
        let _phase = super::phase::start("assemble.shell_bending");
        assemble::shell_bending(device, data, state, &mut fixed, dt)?;
        drop(_phase);
        // Name: Assembly: Elastic
        // Format: list[(time, ms)]
        // Description:
        // Diagnostic sub-timer of "matrix assembly": wall-clock ms spent
        // assembling the elastic (membrane / bending / solid) Hessian and
        // force into the fixed matrix.
        log::mark("advance", "asm_elastic", _asm_elastic.elapsed());

        // STEP B10. The cross-stitch layer. IT RUNS AHEAD OF B9 IN THIS FILE,
        // because the stitch blocks must land in the live matrix BEFORE that
        // matrix is snapshotted, and the snapshot is what the contact stiffness
        // contracts. Placing it after the snapshot would leave every contact on
        // a stitched scene measuring itself against a matrix the stitch blocks
        // are missing from.
        let _phase = super::phase::start("assemble.stitch");
        assemble::stitch(device, data, prm.stitch_length_factor, state, &mut fixed)?;
        drop(_phase);

        // STEP B9. Snapshot the elastic matrix. The contact stiffness reads
        // this reference rather than the matrix it is about to add to, so the
        // snapshot boundary is where it is for a reason: moving it would change
        // what a later phase reads.
        {
            // THE SNAPSHOT IS A DEVICE-TO-DEVICE COPY, not a host round trip:
            // the seam's copy verb is what keeps the matrix on the device
            // instead of reading it back and writing it out again.
            let _asm_copy = std::time::Instant::now();
            state.tmp_fixed_values.copy_from(device, fixed.value_readback())?;
            let non_tet=fixed.values_mut();
            let full=state.tmp_fixed_values.handle();
            state.factor_tet.full_reference(device,non_tet,full,state.tmp_fixed_values.len() as u32)?;

            // Name: Assembly: Fixed Copy
            // Format: list[(time, ms)]
            // Description:
            // Diagnostic sub-timer of "matrix assembly": wall-clock ms to snapshot
            // the elastic fixed matrix into tmp_fixed (the contact-stiffness
            // reference read by the contact assembly).
            log::mark("advance", "asm_copy", _asm_copy.elapsed());
        }

        // STEP B12 and B13. The two strain limiters, shell then rod, reading
        // the snapshot above and writing the live
        // matrix. Both are dispatched on an element count alone, so a scene
        // reaches them the moment it carries a shell face or a rod whose
        // material sets `strain-limit`.
        //
        // THE SNAPSHOT IS WRAPPED RATHER THAN BORROWED, so the two matrices are
        // separate objects the borrow checker can hold at once, and the values
        // go back to the state afterwards. Nothing allocates: `adopt_from_dataset`
        // reuses the array it is handed.
        {
            let mut checked = state.fixed_pattern_checked;
            let mut reference = FixedCsr::adopt_validated(
                device,
                pattern_refs,
                data,
                std::mem::take(&mut state.tmp_fixed_values),
                &mut checked,
            )?;
            state.fixed_pattern_checked = checked;
            // THE STRAIN-LIMIT SUB-TIMER spans both limiters and stops BEFORE
            // the contact assembly opens its own scope. The `shell` and `rod`
            // results are inspected at the bottom of this block rather than
            // here, so what the channel measures is the two dispatches
            // themselves.
            let _asm_strainlimit = std::time::Instant::now();
            let shell = assemble::shell_strain(
                device,
                data,
                prm.eiganalysis_eps,
                prm.barrier,
                state,
                &mut reference,
                &mut fixed,
            );
            let rod = if shell.is_ok() {
                assemble::rod_strain(device, data, prm.barrier, state, &mut reference, &mut fixed)
            } else {
                Ok(())
            };
            // Name: Assembly: Strain Limit
            // Format: list[(time, ms)]
            // Description:
            // Diagnostic sub-timer of "matrix assembly": wall-clock ms for the
            // strain-limiting Hessian / force contributions.
            log::mark("advance", "asm_strainlimit", _asm_strainlimit.elapsed());
            // STEP B15. Contact assembly, INSIDE this block because it reads the
            // same elastic snapshot: `barrier::compute_stiffness` measures the
            // contact's stiffness against the elastic Hessian the contact is
            // about to be added to, and reading the live matrix instead would
            // let one contact's stiffness depend on another's already being
            // assembled.
            // THE STATISTICS COUNTER IS CLEARED ONCE PER NEWTON ITERATION: it
            // counts the contacts THIS iteration's
            // assembly deposits, so a counter carried across iterations would
            // report the running total of every iteration in the step. The
            // clear runs whether or not the assembly is taken, so a step that
            // skips contact reports zero rather than the previous iteration's
            // count.
            // THE PER-OBJECT STATISTICS CHANNEL, built once for the three
            // assemblies below because all three record into ONE counter, and
            // rebuilt each iteration because taking the counter's handle stales
            // its mirror, which is what makes `fetch` download before reading.
            let statistics = crate::driver::contact::StatisticsRefs {
                contact_count_size: state.statistics_contact_count.len() as u32,
                contact_count: state.statistics_contact_count.handle(),
                object_index_size: state.statistics_object_index.len() as u32,
                object_index: state.statistics_object_index.handle(),
                static_object_index_size: state.statistics_static_object_index.len() as u32,
                static_object_index: state.statistics_static_object_index.handle(),
            };
            // Slip prediction uses m/h² and grain contact uses h*omega; both
            // must use the same integration duration h as momentum and grain
            // integration, including playback and trial shortening. This is
            // not the scene-clock increment h/playback.
            let contact_h = dt;
            // THE COLLAPSED-SEPARATION SLOTS ARE CLEARED ONCE, HERE, BEFORE
            // THE FIRST OF THE TWO ASSEMBLY PASSES. Every pass writes its
            // query's slot first-writer-wins, so a clear BETWEEN them drops the
            // earlier pass's report before `collect_assembly_overlap` below
            // reads it. Clearing at the top of `Contact::assemble` would be
            // correct only if that pass ran first, and it does not, so the
            // collision-mesh kinds 7, 8 and 9 would be written and then
            // erased. The line search clears them again
            // before the sweep, which reads them under its own decoder.
            if let Some(contact) = state.contact.as_mut() {
                contact.clear_overlap_slots(device)?;
                if shell.is_ok() && rod.is_ok() && !prm.disable_contact {
                    contact.refresh_contact_queries(
                        device, data, mesh_refs, state.eval_x.handle(), windows,
                    )?;
                }
            }
            // THE RESIDUAL SNAPSHOT THIS PASS'S FRICTION ANCHORS ON. Taken
            // here rather than read out of `force` directly because the
            // contacts below deposit into `force` through atomics, so a pair
            // reading that vector would read a row another pair had already
            // moved. TWO SNAPSHOTS, NOT ONE, because this driver deposits both
            // halves straight into `force` rather than accumulating the
            // self-contact half into a vector of its own and folding it in at
            // the end. The second copy therefore carries the analytic and
            // collision-mesh forces, which is what the self-contact pass must
            // anchor on.
            {
                let bytes = 3 * state.sizes.vertices * size_of::<f32>();
                let (residual, force) = (state.residual.handle(), state.force.handle());
                device.copy(residual, 0, force, 0, bytes)?;
            }
            // STEP B15a. The constraint layer: geometry outside the solved
            // namespace. The analytic sphere and floor barriers run whether or
            // not `disable-contact` is set; the collision-mesh half is inside
            // that gate.
            //
            // FIRST, before the self-contact assembly, and reading the SAME
            // elastic snapshot. The self-contact block below says why the order
            // is load-bearing.
            let constraint_result = if shell.is_ok() && rod.is_ok() {
                // Single optional D2D slot, only the iteration immediately before the original cap.
                let record_analytic_ledger=analytic_assembly_ledger.selected(step,prm.max_newton_steps)
                    && vertices==16 && state.rigid.dim==12 && pdrd_bodies==2 && locked_groups==0
                    && !state.hard.active() && grains==0 && state.fix.is_empty() && state.sphere.is_empty()
                    && state.floor.len()==1 && !state.floor.host()[0].kinematic
                    && root_attempt.as_ref().is_some_and(|a|a.body_indices.as_slice()==[0]);
                if record_analytic_ledger {
                    let observed=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||
                        analytic_assembly_ledger.before(device,state,step,final_step,dt,root_attempt.as_ref())));
                    if observed.is_err() {analytic_assembly_ledger.pre_complete=false;analytic_assembly_ledger.error=Some("optional before-copy panicked".into());}
                }
                let SolverState {
                    analytic: analytic_state,
                    contact: contact_state,
                    force,
                    diagonal,
                    eval_x,
                    fix,
                    sphere,
                    floor,
                    push,
                    // THE THREE GRAIN SCHUR BLOCKS, per vertex and owned here.
                    // They are sized with the scene whenever it carries grains,
                    // so the analytic pass writes into an array that exists
                    // whether or not this scene has a collider.
                    grain_angular: grain_angular_buffer,
                    grain_coupling: grain_coupling_buffer,
                    grain_rotational: grain_rotational_buffer,
                    residual: residual_buffer,
                    ..
                } = &mut *state;
                let residual = residual_buffer.handle();
                let grain_angular = grain_angular_buffer.handle();
                let grain_coupling = grain_coupling_buffer.handle();
                let grain_rotational = grain_rotational_buffer.handle();
                let iterate = eval_x.handle();
                let Some(analytic) = analytic_state.as_mut() else {
                    return Err(Fatal::invariant(
                        "solver driver: the analytic collider assembly was reached with \
                         no collider layer allocated, so a scene's sphere or \
                         floor would hold nothing at all",
                    ));
                };
                let mut outcome = analytic.assemble(
                    device,
                    data,
                    mesh_refs,
                    param,
                    contact_h,
                    curr,
                    iterate,
                    fix.handle(),
                    sphere,
                    floor,
                    &mut reference,
                    &mut fixed,
                    push,
                    force.handle(),
                    diagonal.handle(),
                    grain_angular,
                    grain_coupling,
                    grain_rotational,
                    residual,
                    statistics,
                );
                if outcome.is_ok() && record_analytic_ledger {
                    let observed=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||
                        analytic_assembly_ledger.after(device,force.handle(),diagonal.handle())));
                    if observed.is_err() {analytic_assembly_ledger.post_complete=false;analytic_assembly_ledger.error=Some("optional after-copy panicked".into());}
                }
                if outcome.is_ok() && !prm.disable_contact {
                    if let Some(contact) = contact_state.as_mut() {
                        outcome = contact.assemble_collision_mesh_prepared(
                            device,
                            data,
                            mesh_refs,
                            param,
                            contact_h,
                            curr,
                            iterate,
                            &mut reference,
                            &mut fixed,
                            force.handle(),
                            push,
                            residual,
                            statistics,
                        );
                    }
                }
                outcome
            } else {
                Ok(())
            };
            // THE RESIDUAL SNAPSHOT THIS PASS'S FRICTION ANCHORS ON. Taken
            // here rather than read out of `force` directly because the
            // contacts below deposit into `force` through atomics, so a pair
            // reading that vector would read a row another pair had already
            // moved. TWO SNAPSHOTS, NOT ONE, because this driver deposits both
            // halves straight into `force` rather than accumulating the
            // self-contact half into a vector of its own and folding it in at
            // the end. The second copy therefore carries the analytic and
            // collision-mesh forces, which is what the self-contact pass must
            // anchor on.
            {
                let bytes = 3 * state.sizes.vertices * size_of::<f32>();
                let (residual, force) = (state.residual.handle(), state.force.handle());
                device.copy(residual, 0, force, 0, bytes)?;
            }
            // THE SELF-CONTACT ASSEMBLY, timed as `asm_contact`.
            //
            // AFTER the analytic and collision-mesh contacts, and the order is
            // load-bearing rather than
            // incidental: every friction term anchors itself on the
            // residual assembled so far, and the contacts above are the ones
            // carrying the normal load in most scenes, so a mesh contact built
            // after them reads a tangential drive with that load already
            // removed. The reverse order would hand a vertex resting on a floor
            // its full weight as the sideways drive of a contact against a
            // neighbor. Both are fp32 running sums either way, so the order is
            // not free to change back.
            let _asm_contact = std::time::Instant::now();
            let contact_result = if shell.is_ok() && rod.is_ok() && !prm.disable_contact {
                let SolverState {
                    contact: contact_state,
                    force,
                    eval_x,
                    push,
                    grain_omega: omega,
                    grain_torque: torque,
                    grain_stiffness: stiffness,
                    grain_normal: normal,
                    residual: residual_buffer,
                    ..
                } = &mut *state;
                let residual = residual_buffer.handle();
                let iterate = eval_x.handle();
                let grain_omega = omega.handle();
                let (grain_torque, grain_stiffness, grain_normal) =
                    (torque.handle(), stiffness.handle(), normal.handle());
                match contact_state.as_mut() {
                    Some(contact) => contact.assemble_prepared(
                        device,
                        data,
                        mesh_refs,
                        param,
                        contact_h,
                        curr,
                        iterate,
                        &mut reference,
                        &mut fixed,
                        force.handle(),
                        push,
                        grain_omega,
                        grain_torque,
                        grain_stiffness,
                        grain_normal,
                        residual,
                        statistics,
                    ),
                    None => Err(Fatal::invariant(
                        "solver driver: the contact assembly was reached with no contact \
                         subsystem allocated",
                    )),
                }
            } else {
                Ok(())
            };
            // Given back before any failure is reported, so a step that stops
            // here leaves the state holding its buffer rather than an empty one
            // the next step would size again.
            state.tmp_fixed_values = reference.into_values();
            shell?;
            rod?;
            // Name: Assembly: Contact
            // Format: list[(time, ms)]
            // Description:
            // Diagnostic sub-timer of "matrix assembly": wall-clock ms for the
            // self-contact / collision-mesh Hessian + force assembly (the CSR fill
            // path, including the dynamic-matrix rebuild).
            log::mark("advance", "asm_contact", _asm_contact.elapsed());
            contact_result?;
            if let Err(error)=constraint_result.as_ref() {
                if analytic_gap_trace.enabled() { analytic_gap_trace.capture(device,"barrier-failure-read",step,final_step,dt,toi_advanced,1.0,1.0,state.positions.handle(),state.eval_x.handle(),None,state.floor.host(),root_attempt.as_ref()); }
                analytic_gap_trace.flush(&error.detail,(*param).ccd_eps,prm.line_search_max_t);
            }
            constraint_result?;
            // A PAIR THE ASSEMBLY FOUND ALREADY COLLAPSED TO ITS OFFSET ends
            // the advance here, and the report is a HOST CONTRACT: the kind
            // string, the pair, the world-space lengths, and for a
            // collision-mesh kind the note saying which space the second index
            // is in. `contact_separated = false` is what turns it into a
            // structured OverlappingStart on the host.
            if let Some(contact) = state.contact.as_mut() {
                if let Some(overlap) = contact.collect_assembly_overlap(device)? {
                    sand_offset_trace.report(device,data,state,&overlap,(*param).ccd_eps,prm.line_search_max_t);
                    log::message!("### {}", overlap.describe());
                    if overlap.collision_mesh {
                        log::message!(
                            "### the second index is in the static collision-mesh \
                             vertex space; the first is a dynamic vertex."
                        );
                    }
                    log::message!(
                        "### no intersection allowance covers this pair, so contact has to \
                         keep it apart. An allowance that names it (self, inter-object, or \
                         a pin's) takes the pair out of contact altogether."
                    );
                    log::message!(
                        "### give the initial geometry a small clearance so nothing starts \
                         in contact, or check whether a stitch or pin is pulling elements \
                         together faster than contact can resolve."
                    );
                    result.contact_separated = false;
                    if let Some(attempt)=root_attempt.as_ref(){state.roots.transaction.as_mut().unwrap().reject(attempt).map_err(|e| Fatal::invariant(e.to_string()))?;}
                    return Ok(result);
                }
            }
        }
        // How many contact pairs the assembly deposited. Telemetry rather than
        // a term of the solve, and the one
        // stream that separates a scene whose geometry never touches from a
        // backend whose broad phase found nothing.
        {
            let (self_contact, static_contact) = if prm.disable_contact {
                (0, 0)
            } else {
                state.contact.as_ref().map_or((0, 0), |contact| {
                    (contact.assembled, contact.assembled_collision_mesh)
                })
            };
            let constraint = state
                .analytic
                .as_ref()
                .map_or(0, |analytic| analytic.assembled);
            // Name: Total Contact Count
            // Format: list[(time, count)]
            // Description:
            // Total number of active contact and constraint pairs assembled
            // into the system matrix for this Newton iteration, summed across
            // self-contact, collision-mesh contact, and analytic constraints
            // (sphere, floor). A useful proxy for how crowded the collision
            // scene is at this iteration.
            log::mark("advance", "num_contact", (self_contact + static_contact + constraint) as f64);
            log::mark("advance", "num_contact_dynamic", self_contact as f64);
            log::mark("advance", "num_contact_static_mesh", static_contact as f64);
            log::mark("advance", "num_contact_analytic", constraint as f64);
        }
        // Name: Matrix Assembly Time
        // Format: list[(time, ms)]
        // Description:
        // Wall-clock time in milliseconds spent assembling the global
        // system matrix and right-hand side for the Newton linear solve,
        // including inertia, elastic, stitch, strain-limiting, and contact
        // contributions. One entry per Newton iteration.
        //
        // CLOSED AFTER `num_contact` AND NOT BEFORE IT, which is what puts
        // `> matrix_assembly` below `* num_contact` in the stream.
        log::mark("advance", "matrix_assembly", _assembly.elapsed());

        // STEP B15b. THE SAND GRAIN'S SPIN, CONDENSED OUT OF THE NEWTON SYSTEM.
        // After the whole system is assembled and BEFORE the Dirichlet removal
        // and the solve. The grain's
        // angular degree of freedom is eliminated into the translation block
        // it couples to, so the solve below stays a translation-only system.
        if grains > 0 {
            // THE STATE'S OWN BLOCKS, present whenever the scene has grains.
            // Asking the collider and substituting `Handle::NONE` when there is
            // none left this pass with an unresolvable buffer in exactly the
            // scenes that carry grains and no floor or sphere.
            let (angular, coupling, rotational) = (
                state.grain_angular.handle(),
                state.grain_coupling.handle(),
                state.grain_rotational.handle(),
            );
            let condense = SandGrainCondenseRowArgs {
                inverse_center_inertia: state.grain_inv_inertia_center.handle(),
                angular,
                coupling,
                rotational_gradient: rotational,
                previous_omega: state.grain_omega_prev.handle(),
                diagonal: state.diagonal.handle(),
                force: state.force.handle(),
                dt,
                // The fraction of a grain's spin angular momentum fed back into
                // its translation, read from the environment and defaulting to
                // 0.5, which is near-textbook rolling while staying bounded:
                // fully included it is a positive-feedback runaway.
                spin_couple: sand_spin_couple(),
                count: vertices as u32,
                seam_arena_count: 0,
            };
            device.launch("sand.condense", &condense, vertices as u32)?;
        }

        let _dirichlet = std::time::Instant::now();
        // STEP B16. Dirichlet DOF removal, both passes, WITH the lifting term,
        // over BOTH matrices. The contact matrix carries a moving collider's
        // coupling to the geometry it is advancing into, so a reduction that
        // skipped it would drop exactly the signal the lifting term exists to
        // deliver, and the step would deadlock rather than compute a wrong
        // number.
        //
        // The flat contact arrays are TAKEN and given back, for the reason
        // `tmp_fixed_values` is: the lift needs the driver's force, iterate and
        // target at the same time as the matrix, and they live in one struct.
        if removed > 0 {
            state.factor_tet.lift(device,state.mesh_tet.handle(),state.dof_mask.handle(),
                state.eval_x.handle(),state.target.handle(),state.force.handle())?;
            let mut flat = state
                .contact
                .as_mut()
                .map(|contact| std::mem::take(&mut contact.flat));
            let dynamic = if let Some(flat) = flat.as_mut() {
                dirichlet::lift(device, state, flat.offset.handle(), flat.index.handle(), flat.value.handle())
            } else {
                Ok(())
            };
            let (offset, column) =
                (state.fixed_offset.handle(), state.fixed_index.handle());
            let fixed_lift = if dynamic.is_ok() {
                dirichlet::lift(device, state, offset, column, fixed.values_mut())
            } else {
                Ok(())
            };
            if let (Some(flat), Some(contact)) = (flat, state.contact.as_mut()) {
                contact.flat = flat;
            }
            dynamic?;
            fixed_lift?;
            dirichlet::prescribe(device, state)?;
            // THE CHANNEL CARRIES NO DOCSTRING ABOVE IT, so it is deliberately
            // absent from `session.get.log.names()`, and `asm_dirichlet` is the
            // name it is read under.
            //
            // INSIDE THE `removed > 0` BRANCH: a scene with no prescribed row
            // does no Dirichlet work and so prints no line, and marking it
            // unconditionally here would put a row in the stream for work that
            // did not happen.
            log::mark("advance", "asm_dirichlet", _dirichlet.elapsed());
        }
        state.factor_tet.finish(device,&mut fixed,state.dof_mask.handle(),
            state.fixed_offset.handle(),state.fixed_index.handle())?;
        let tet_factor_view=state.factor_tet.view(state.mesh_tet.handle(),state.dof_mask.handle());
        // THE LINEAR SOLVE'S TIMER OPENS HERE: after the Dirichlet scope
        // closes and before anything the solve needs is prepared. That places
        // the block-Jacobi build inside it, the preconditioner being part of
        // the solve, and it covers all THREE of this driver's solve branches.
        // A timer around the unreduced branch alone would leave every PDRD and
        // every locked scene with no `> linsolve` line at all.
        let _linsolve = std::time::Instant::now();
        // STEP B17. The linear solve.
        // The pattern handles are taken BEFORE the destructure below, which
        // holds the state's fields disjointly and so cannot lend the whole of
        // it to a method.
        let pattern_refs = state.fixed_pattern_refs();
        // Only material-frame scenes prepare this additional frozen operator.
        // Do the actual PDRD fit before nodal diagonal inversion: mixed FEM/
        // rigid angular coefficients must use one current pose in every action.
        if state.joints.soft.active() {
            let vert_list=state.rigid_staged.vert_list.handle();
            let body_prop=state.rigid_staged.body_prop.handle();
            let rest=state.rigid_staged.rest_centered.handle();
            let mut reduction=super::pdrd::Reduction {map:&state.rigid,staged:&mut state.rigid_staged};
            reduction.fit(device,vert_list,state.prop_vertex.handle(),body_prop,rest,
                state.eval_x.handle(),state.rigid_fit_scratch.handle(),state.rigid_state.handle(),
                data.pdrd_vert_list.size,&state.pcg_rigid.zero)?;
            state.joints.soft.freeze(device,state.eval_x.handle(),state.rigid_state.handle(),
                state.joints.reference_handle(),&state.rigid,dt,None).map_err(|error|{
                    let _=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||eprintln!("FEM_PREPARE_CONTEXT source=B17_iterate parameter_time={:.17e} newton_step={} adaptive_retry_token={} dt_bits={:08x} accepted_commit_clock=unavailable",prm.time,step,state.adaptive_retry_token,dt.to_bits())));error
                })?;
        }
        let soft_view=state.joints.soft.view(&state.rigid,dt,state.joints.curves.view());
        // The disjoint field borrows the solve needs, taken once: the operator
        // reads the block diagonal while the preconditioner, the right-hand
        // side, the direction and the PCG workspace are written.
        let SolverState {
            diagonal,
            precond_diagonal,
            precond_inverse: inverse,
            force,
            eval_x,
            dx,
            pcg: work,
            contact: contact_state,
            ..
        } = state;
        let operator = Operator {
            tet_factor:tet_factor_view,
            dynamic: contact_state.as_mut().map(|contact| DynamicView {
                offset: contact.flat.offset.handle(),
                index: contact.flat.index.handle(),
                value: contact.flat.value.handle(),
                // THE TRANSPOSE ARRAYS ARE SPANNED TO THE STORED TOTAL, which
                // is what a host truncate would do: the pass sizes them for the
                // widest case and only the kernel knows the real width.
                transpose_offset: contact.transpose.offset.handle(),
                transpose_index: contact.transpose.index.span(0, contact.transpose.total),
                transpose_value: contact.transpose.value.span(0, contact.transpose.total),
            }),
            fixed: fixed.view(),
            fixed_pattern: Some(fixed.pattern()),
            diagonal: diagonal.span(0, 9 * vertices),
        };
        operator.precond_diagonal(device, precond_diagonal.span(0, 9 * vertices))?;
        if let Some(soft)=soft_view {
            soft.add(device,5,soft.dummy,precond_diagonal.span(0,9*vertices))?;
        }
        // THE INVERSION IS A DISPATCH, not a host walk. The loop over the rows
        // is the kernel's, and so is the verdict, which comes back through the
        // diagnostic lane rather than as a returned row. The payload the entry
        // raises is the largest eigenvalue, the two extremes of the spectrum
        // and the row, which is everything a per-row message would otherwise
        // read out of the block itself.
        let invert_args = BlockJacobiInvertRowArgs {
            diagonal: precond_diagonal.span(0, 9 * vertices),
            inverse: inverse.span(0, 9 * vertices),
            count: vertices as u32,
            seam_arena_count: 0,
        };
        let inverted = device.run("step.block_jacobi_invert", |encoder| {
            // Safety: both arrays are borrowed for the whole call and hold one
            // 3x3 block per vertex.
            unsafe { encoder.elements(&invert_args, vertices as u32) }
        });
        if let Err(fault) = inverted {
            return Err(match fault {
                Fault::Device { diag, .. } => Fatal::invariant(format!(
                    "solver driver: {} block-Jacobi diagonal block(s) are not positive \
                     definite. Every block is inverted through a floored symmetric \
                     eigendecomposition, so this means the assembled block is NaN, \
                     infinite, or has a non-positive largest eigenvalue, which is an \
                     assembly defect upstream rather than a tolerance. The first, by row, \
                     reports [lambda_max, lambda_min, lambda_max_again, row] as {}",
                    diag.failures,
                    diag.first
                        .as_ref()
                        .map_or(String::new(), |first| format!("{first}"))
                )),
                other => Fatal::from(other),
            });
        }
        // THE LOCKED SOLVE IS A DIFFERENT SYSTEM, not the same one with extra
        // steps, so it is a branch of its own:
        // the dispatcher prepares the affine feasible correction against the
        // positions the frames were built at, runs `Q M Q z = Q (b - M q)`, and
        // verifies the solved correction carries no forbidden angular
        // increment.
        let mut report = if pdrd_bodies > 0 || soft_view.is_some() {
            // THE REDUCED SOLVE, taken whenever
            // the scene carries a body. A PDRD body's vertices own no per-vertex
            // degrees of freedom, so the whole system, cloth rows included,
            // routes through the reduction as soon as one exists.
            //
            // THE FIT AND THE SCATTER RUN FIRST AND EVERY ITERATION. The body's
            // rotation moves under the Newton loop, and both the prolongation
            // and the force restriction read each vertex's rotated rest vector
            // as the rigid Jacobian's moment arm; fitting once per step would
            // leave the reduction working in a frame the body has left.
            let SolverState {
                rigid,
                rigid_staged,
                rigid_state,
                rigid_blocks,
                rigid_factor,
                rigid_fit_scratch,
                rigid_rotation_step,
                pcg_rigid,
                eval_x,
                target,
                prop_vertex,
                pull,
                // THE LOCK STATE, for a scene that carries both. The reduced
                // solve projects the FULL-SPACE product inside its own operator
                // rather than being wrapped by the lock, which is the order
                // `cg_rigid_translation_locked` uses and the only order that is
                // the same operator.
                lock,
                pcg_locked,
                translation_lock,
                translation_lock_index,
                translation_lock_initial,
                dof_mask,
                rigid_running_rotation,
                rigid_rotation_seeded,
                joints,
                hard,
                roots,
                ..
            } = state;
            let body_vertices = data.pdrd_vert_list.size;
            let vert_list = rigid_staged.vert_list.handle();
            let body_prop = rigid_staged.body_prop.handle();
            let rest_centered = rigid_staged.rest_centered.handle();
            let mut reduction = super::pdrd::Reduction {
                map: rigid,
                staged: rigid_staged,
            };
            reduction.fit(
                device,
                vert_list,
                prop_vertex.handle(),
                body_prop,
                rest_centered,
                eval_x.handle(),
                rigid_fit_scratch.handle(),
                rigid_state.handle(),
                body_vertices,
                &pcg_rigid.zero,
            )?;
            // SEED THE RUNNING ROTATION FROM THE POSE THIS RUN WAS HANDED,
            // once, behind the `pdrd_rprev_seeded` latch. The fit above has
            // just written each body's absolute
            // best-fit rotation into `rigid_state`, so the seed is a copy.
            //
            // THE IDENTITY IS ONLY RIGHT ON A FRESH SCENE. An unrotated body
            // fits to the identity and the two agree, which is why this is
            // invisible on every run that starts from the authored pose. A
            // `--load` resume starts from a pose whose bodies have already
            // turned, and seeding the identity there aims the anchored rigidify
            // at the authored orientation rather than the saved one.
            //
            // AFTER THE FIRST STEP IT MUST NOT RUN AGAIN: the running rotation
            // is then the composition of the increments the line search
            // accepted, and re-fitting it to a contact-sheared pose is the
            // drift the anchored rigidify exists to avoid.
            if !*rigid_rotation_seeded {
                reduction.copy_state_rotation(
                    device,
                    rigid_state.handle(),
                    rigid_running_rotation.handle(),
                )?;
                *rigid_rotation_seeded = true;
            }
            reduction.scatter_rotated_rest(
                device,
                vert_list,
                prop_vertex.handle(),
                rigid_state.handle(),
                rest_centered,
                body_vertices,
            )?;
            let empty_dyn = (
                reduction.staged.empty_dyn_offset.handle(),
                reduction.staged.empty_dyn_index.handle(),
                reduction.staged.empty_dyn_value.handle(),
            );
            let precond_inputs = super::pdrd::PrecondInputs {
                vert_list,
                vertex_prop: prop_vertex.handle(),
                state: rigid_state.handle(),
                rest_centered,
                // AN EMPTY MATRIX WHEN THERE IS NO CONTACT, never
                // `Handle::NONE`. The sandwich row resolves all three buffers
                // before its body runs and then indexes `dyn_offset[v]` and
                // `dyn_offset[v + 1]` unconditionally, so a scene that
                // configures no contact needs a real `rows + 1` run of zeros
                // here rather than an absent buffer. `Handle::NONE` carries
                // `u32::MAX` as its arena and trapped the entry's own assert.
                dyn_index: operator
                    .dynamic
                    .as_ref()
                    .map_or_else(|| empty_dyn.1, |d| d.index),
                dyn_offset: operator
                    .dynamic
                    .as_ref()
                    .map_or_else(|| empty_dyn.0, |d| d.offset),
                dyn_value: operator
                    .dynamic
                    .as_ref()
                    .map_or_else(|| empty_dyn.2, |d| d.value),
                fixed_index: pattern_refs.index,
                fixed_offset: pattern_refs.offset,
                fixed_value: fixed.value_handle(),
                rows: pattern_refs.rows,
                body_vertices,
            };
            let mut joint_view=joints.view(rigid_state.handle(), rigid, dt);
            let root_context=if hard.active(){roots.hard_context()?}else{None};
            let root_view=roots.view(rigid);
            if let Some(context)=root_context{hard.freeze_known_root(device,rigid_state.handle(),(joint_view.twist_context,joint_view.count),rigid,context)?;}
            else{hard.freeze(device,rigid_state.handle(),(joint_view.twist_context,joint_view.count),rigid)?;}
            // THE DIRECTIONAL JOINT GEOMETRY GATE, and this is its only
            // per-solve site. It runs HERE, after the fit above has rewritten
            // `rigid_state` for the current iterate and before the
            // preconditioner consumes the joint's `J^T H J` contribution below.
            // `joints.reference` still holds the SUBSTEP-START fit taken in
            // `begin_step`, so the damping increment is measured between the two
            // poses the assembly will actually use. A requested row whose live
            // chart has become unusable -- of either pose -- refuses the step
            // here rather than being accepted with that term silently missing;
            // the check is once per frozen pose, never inside a PCG application.
            super::joints::gate_frozen_pose(
                device,
                &mut *joints,
                rigid,
                rigid_state.handle(),
                dt,
            )?;
            let auto_frozen = pcg::runtime_auto_policy::candidate(pcg::runtime_auto_policy::Domain {
                cuda: cfg!(cuda_backend), fp32: reduction.staged.pcg_precision == crate::args::PcgPrecision::Fp32,
                newton: true, bodies: reduction.map.n_bodies > 0,
                joints: joint_view.count != 0,
                compatible: !hard.active() && locked_groups == 0 && !reduction.map.needs_projection()
                    && (*param).precond != crate::data::PrecondMode::Schwarz,
                explicit: pcg::runtime_auto_policy::explicit_controls(),
            });
            let frozen_joint_tangent = frozen_joint_tangent || auto_frozen;
            if let Some(soft) = joint_view.soft.as_mut().filter(|_| frozen_joint_tangent) {
                soft.frozen_restoring = Some(joints.soft.prepare_restoring_cache(device, *soft)?);
                joints.soft.check_restoring_cache_fault(device)?;
            }
            if frozen_joint_tangent && joint_view.count > 0 {
                // The angular V2 tangent depends only on this fitted Newton
                // pose and remains fixed for every PCG search-direction
                // product. Rebuild it once here; the epoch prevents a stale
                // matrix from leaking into the next Newton iteration.
                joint_view = joints.prepare_angular_tangents(device, joint_view)?;
                // Directional control (when no restoring curve owns it) and
                // angular limit/drive extensions have their own pose-frozen
                // Jacobian/Hessian data. Prepare them at the same Newton
                // linearization point so PCG applications only perform the
                // necessary cached HVP and retain the epoch fallback.
                joint_view = joints.prepare_control_extension_tangents(device, joint_view)?;
                joint_view.frozen_ball = Some(joints.prepare_ball_cache(device, joint_view)?);
                joints.check_ball_cache_fault(device)?;
            }
            if frozen_joint_tangent && joint_view.curves.is_some() && joint_view.control_count > 0 {
                joint_view.frozen_angular = Some(joints.prepare_angular_cache(device, joint_view)?);
                joints.check_angular_cache_fault(device)?;
            }
            reduction.staged.reduced_schwarz.select(device,reduction.map,
                (*param).precond==crate::data::PrecondMode::Schwarz && locked_groups==0,(*param).schwarz_levels)?;
            if (*param).precond==crate::data::PrecondMode::Schwarz && locked_groups>0 {
                eprintln!("REDUCED_SCHWARZ_CONSTRAINT_FALLBACK {{\"reason\":\"nonlocal_aggregate_projector\",\"preconditioner\":\"original_block_jacobi_with_original_outer_Q\",\"groups\":{}}}",locked_groups);
            }
            reduction.build_precond(
                device,
                &precond_inputs,
                rigid_blocks,
                rigid_factor,
                dt,
                Some(joint_view),
            )?;
            let requests_arithmetic = reduction.staged.pcg_precision
                == crate::args::PcgPrecision::MixedFp16Arithmetic;
            let arithmetic_eligible = requests_arithmetic
                && reduction.map.n_bodies > 0 && reduction.map.n_cloth > 0
                && !reduction.map.needs_projection() && root_view.is_none()
                && locked_groups == 0 && !hard.active()
                && !reduction.staged.reduced_schwarz.active()
                && prm.cg_max_iter > 1;
            if requests_arithmetic && reduction.staged.last_arithmetic_eligibility != Some(arithmetic_eligible) {
                reduction.staged.last_arithmetic_eligibility = Some(arithmetic_eligible);
                ::log::info!(
                    "PCG_PRECISION_EFFECTIVE: {} (rigids={}, nodes={}, body_projection={}, roots={}, lock_groups={}, hard={}, schwarz={}, max_iterations={})",
                    if arithmetic_eligible { "fp16_nodal_arithmetic" } else { "fp32" },
                    reduction.map.n_bodies, reduction.map.n_cloth, reduction.map.needs_projection(),
                    root_view.is_some(), locked_groups, hard.active(),
                    reduction.staged.reduced_schwarz.active(), prm.cg_max_iter,
                );
            }
            reduction.prepare_mixed_precond(device, inverse.span(0, 9 * vertices), arithmetic_eligible)?;
            reduction.staged.reduced_schwarz.rebuild(device,reduction.map,&operator,
                reduction.staged.empty_dyn_offset.handle(),joint_view,
                reduction.staged.vbody.handle(),reduction.staged.prot.handle(),
                eval_x.handle(),dof_mask.handle(),super::constraint_space::Inputs{
                    joint_mode:reduction.staged.jmode.handle(),joint_axis:reduction.staged.jaxis.handle(),
                    translation_lock:reduction.staged.tlock.handle(),translation_axis:reduction.staged.tlock_axis.handle(),translation_mode:reduction.staged.tlock_mode.handle(),
                    rotation_lock:reduction.staged.rlock.handle(),rotation_axis:reduction.staged.rlock_axis.handle(),rotation_mode:reduction.staged.rlock_mode.handle()},
                &roots.constraint_enabled_bodies())?;
            // The projector, built exactly as the locked branch builds it, and
            // `None` when the scene carries no lock so the reduced apply keeps
            // its unlocked shape.
            let mut lock_projector = if locked_groups > 0 {
                let lock_rows = super::lock::RowInputs {
                    lock_index: translation_lock_index.handle(),
                    locks: translation_lock.handle(),
                    prop: prop_vertex.handle(),
                    positions: eval_x.handle(),
                    dof_mask: dof_mask.handle(),
                    initial: translation_lock_initial.handle(),
                };
                let mut projector = super::lock::Projector {
                    scratch: lock,
                    groups: locked_groups,
                    vertices,
                    rows: lock_rows,
                };
                projector.prepare(
                    device,
                    locked_records,
                    dx.handle(),
                    pcg_locked.q.handle(),
                )?;
                // THE HINGE-AND-LOCK FEASIBILITY CHECK, which the device side
                // already assumes has run:
                // `pdrd_translation_lock_particular_row` returns early for a
                // hinged body on the stated grounds that the host has checked
                // it. Without this the early return silently discards a
                // correction the scene asked for rather than refusing an
                // impossible one. `prepare` has just written the drift the
                // verdict reads.
                super::lock::check_hinge_lock_feasible(
                    device,
                    &mut projector.scratch.drift,
                    locked_records,
                    &reduction.map.jmode,
                    &reduction.map.tlock,
                )?;
                Some(projector)
            } else {
                None
            };
            if root_context.is_some(){hard.capture_affine_force(device,&mut reduction,force.handle(),joint_view)?;}
            // Borrow the already assembled block handle; optional capture only
            // reads it after the same frozen solve reports failure.
            pcg_rigid.failure_precond_blocks = Some(rigid_blocks.handle());
            let solved = pcg::solve_rigid(
                device,
                &operator,
                &mut reduction,
                rigid_factor.handle(),
                inverse.span(0, 9 * vertices),
                force.handle(),
                dx.handle(),
                work,
                pcg_rigid,
                // THE ROTATION EXPORT, and it is not optional on this path.
                // The rigidify commit rebuilds each body from the running
                // rotation, so a rotation the solve took but never exported is
                // one the commit then OVERWRITES: the body would translate
                // correctly and never turn, which is worse than not rigidifying
                // at all because the motion is silently wrong rather than
                // absent.
                Some(rigid_rotation_step.handle()),
                lock_projector.as_mut(),
                joint_view,
                root_view,
                translation_lock.handle(),
                if hard.active(){prm.cg_tol.min(1e-8)}else{prm.cg_tol},
                prm.cg_max_iter,
                arithmetic_eligible,
                !hard.active() && !(prm.max_newton_steps==2048 && analytic_assembly_ledger.selected(step,prm.max_newton_steps) && analytic_assembly_ledger.post_complete),
                dof_mask.handle(),
                if hard.active() { None } else { Some(pcg::InertialGuessContext {
                    eval_x: eval_x.handle(),
                    target: target.handle(),
                    prop: prop_vertex.handle(),
                    pull: pull.handle(),
                    pull_count: pull.len(),
                }) },
            )?;
            // Optional selected-iteration D2D freeze; no read/probe before B0 cap.
            if prm.max_newton_steps==2048 && analytic_assembly_ledger.selected(step,prm.max_newton_steps) && analytic_assembly_ledger.post_complete {
                let observed=std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let tag=serde_json::json!({"step":step,"final":final_step,"integration_dt_bits":dt.to_bits(),
                        "label":"B17_after_frozen_solve","assembly_tag":analytic_assembly_ledger.tag,
                        "solve_iterations":solved.iterations,"solve_outcome":format!("{:?}",solved.outcome),
                        "solve_relative_bits":solved.relative_residual.to_bits()});
                    frozen_newton_witness.freeze(device,&operator,&mut reduction,joint_view,root_view,
                        force.handle(),eval_x.handle(),pcg_rigid,tag);
                }));
                if observed.is_err(){frozen_newton_witness.optional_panic();}
            }

            // Same frozen op/reduction/view/factor as u0. No smooth gradient is
            // reassembled during any pure-RHS inverse. Only the sealed one-source
            // V7 union takes the principal affine branch; a released source
            // takes the original rootless path with newly frozen geometry.
            if let Some(context)=root_context{
                hard.correct_known_root(device,&operator,&mut reduction,rigid_factor.handle(),
                    inverse.span(0,9*vertices),dx,rigid_rotation_step,work,pcg_rigid,
                    joint_view,root_view.ok_or_else(||Fatal::invariant("ROOT_HARD_SOURCE: active root view absent"))?,context,
                    &solved,prm.cg_tol.min(1e-8),prm.cg_max_iter,locked_groups)?;
            }else{
                hard.correct(device,&operator,&mut reduction,rigid_factor.handle(),
                    inverse.span(0,9*vertices),dx,rigid_rotation_step,work,pcg_rigid,
                    joint_view,&solved,prm.cg_tol.min(1e-8),prm.cg_max_iter,0,locked_groups)?;
            }
            if joint_view.soft.is_some_and(|view| view.frozen_restoring.is_some()) {
                joints.soft.check_restoring_cache_fault(device)?;
            }
            if joint_view.frozen_ball.is_some() {
                joints.check_ball_cache_fault(device)?;
            }
            if joint_view.frozen_angular.is_some() {
                joints.check_angular_cache_fault(device)?;
            }
            // THE TANGENT CHECK ON THIS ARM TOO, easy to read as belonging
            // only to the deformable
            // path. A PDRD body's own six DOFs are constrained exactly by the
            // reduced projector, but a locked GROUP in a scene that also
            // carries a rigid body still has deformable and SAND members, and
            // the aggregate projector is all that constrains their rows.
            //
            // GATED ON A CONVERGED SOLVE, as it is on the arm below: an
            // unconverged correction has not been projected to completion, so a
            // violation read off it would raise a fatal where an ordinary
            // failed step is the right answer.
            if let Some(projector) = lock_projector.as_mut() {
                if matches!(
                    solved.outcome,
                    pcg::Outcome::Converged | pcg::Outcome::AcceptedAnimation | pcg::Outcome::CurvatureTruncated
                ) {
                    eval_x.download(device)?;
                    let mut step_host = vec![0.0f32; 3 * vertices];
                    dx.read(device, 0, &mut step_host)?;
                    let mass_host: Vec<f32> = prop_vertex.host()[..vertices]
                        .iter()
                        .map(|prop| prop.mass)
                        .collect();
                    projector.check_tangent(
                        device,
                        locked_records,
                        dx.handle(),
                        eval_x.host(),
                        &step_host,
                        translation_lock_index.host(),
                        &mass_host,
                        "constrained Newton correction",
                    )?;
                }
            }
            reduction.staged.reduced_schwarz.receipt(device,"frozen-newton-including-hard-inverse")?;
            solved
        } else if locked_groups > 0 {
            let SolverState {
                lock,
                pcg_locked,
                translation_lock,
                translation_lock_index,
                translation_lock_initial,
                prop_vertex,
                dof_mask,
                eval_x,
                schwarz,
                ..
            } = state;
            let rows = super::lock::RowInputs {
                lock_index: translation_lock_index.handle(),
                locks: translation_lock.handle(),
                prop: prop_vertex.handle(),
                positions: eval_x.handle(),
                dof_mask: dof_mask.handle(),
                initial: translation_lock_initial.handle(),
            };
            let mut projector = super::lock::Projector {
                scratch: lock,
                groups: locked_groups,
                vertices,
                rows,
            };
            // `prepare` seeds from the caller's initial guess, which is `dx`:
            // the pinned rows already carry their exact prescribed correction.
            projector.prepare(
                device,
                locked_records,
                dx.handle(),
                pcg_locked.q.handle(),
            )?;
            // THE PRECONDITIONER THE SCENE ASKED FOR, ON THIS PATH TOO. The
            // hierarchy is installed before the locked branch is taken, so a
            // locked scene that set `precond = schwarz` gets Schwarz rather
            // than block-Jacobi under a Schwarz label.
            let sweep = if (*param).precond == crate::data::PrecondMode::Schwarz {
                install_schwarz(device, schwarz, &operator, vertices, param)?;
                Some(&mut *schwarz)
            } else {
                None
            };
            super::dump_linsys::maybe_dump(device, &operator, force.handle(), eval_x.handle(), vertices)?;
            let solved = pcg::solve_locked(
                device,
                &operator,
                &mut projector,
                inverse.span(0, 9 * vertices),
                force.handle(),
                dx.handle(),
                work,
                pcg_locked,
                sweep,
                prm.cg_tol,
                prm.cg_max_iter,
            )?;
            // ON SUCCESS ONLY: a failed solve has no correction worth
            // verifying, and the check is read-only either way. It aborts on a
            // violation rather than snapping, because a snap after the CCD line
            // search would reintroduce penetration.
            //
            // SUCCESS IS CONVERGENCE OR STEIHAUG TRUNCATION, AND AN
            // ITERATION-CAP EXHAUSTION IS NEITHER. The locked solve reports
            // `Converged` when the residual falls, `CurvatureTruncated` when an
            // unresolvable curvature stops it, and a failure at the cap or on a
            // non-finite residual. A gate spelled as "not a breakdown" would let
            // the cap through instead, and that costs twice: the whole pose and
            // the whole search direction would cross the seam to verify a
            // direction the step abandons at `### cg failed` a few lines below,
            // and the verdict would be taken on an UNCONVERGED correction, so a
            // violation it reported would raise a Fatal where an ordinary failed
            // step is the right answer.
            if matches!(
                solved.outcome,
                pcg::Outcome::Converged | pcg::Outcome::AcceptedAnimation | pcg::Outcome::CurvatureTruncated
            ) {
                eval_x.download(device)?;
                // `dx` IS A PLAIN DEVICE BUFFER, so the check reads it back
                // explicitly. The host re-verification below is the only thing
                // that wants it, and it runs once per solve rather than per
                // iteration.
                let mut step_host = vec![0.0f32; 3 * vertices];
                dx.read(device, 0, &mut step_host)?;
                let mass_host: Vec<f32> =
                    prop_vertex.host()[..vertices].iter().map(|p| p.mass).collect();
                // THE TWO MIRRORS ARE LENT, NOT COPIED. `host()` already answers
                // out of a host array the buffer owns for the whole run, so a
                // `to_vec()` on either argument would be a second copy of
                // `vertices` position triples and `vertices` indices per Newton
                // iteration, held only for the duration of one call.
                projector.check_tangent(
                    device,
                    locked_records,
                    dx.handle(),
                    eval_x.host(),
                    &step_host,
                    translation_lock_index.host(),
                    &mass_host,
                    "constrained Newton correction",
                )?;
            }
            solved
        } else {
            // THE PRECONDITIONER THE SCENE ASKED FOR. Schwarz is REBUILT when
            // the row count changes and reused otherwise: the factorization is
            // quadratic in the dense-block cap and the partition is a
            // sequential host sweep, so paying either per Newton step would
            // cost more than the preconditioner saves.
            let sweep = if (*param).precond == crate::data::PrecondMode::Schwarz {
                install_schwarz(device, &mut state.schwarz, &operator, vertices, param)?;
                Some(&mut state.schwarz)
            } else {
                None
            };
            super::dump_linsys::maybe_dump(device, &operator, force.handle(), eval_x.handle(), vertices)?;
            pcg::solve(
                device,
                &operator,
                inverse.span(0, 9 * vertices),
                force.handle(),
                dx.handle(),
                work,
                sweep,
                prm.cg_tol,
                prm.cg_max_iter,
            )?
        };
        #[cfg(all(test,feature="cpu"))]
        if final_step && PRIVATE_TEST_B3_PCG.compare_exchange(2,0,std::sync::atomic::Ordering::SeqCst,std::sync::atomic::Ordering::SeqCst).is_ok() {
            PRIVATE_TEST_B3_DT.store(dt.to_bits(),std::sync::atomic::Ordering::SeqCst);
            report.outcome=pcg::Outcome::MaxIterations;
            report.iterations=prm.cg_max_iter;
            report.relative_residual=prm.cg_tol.max(1.0);
        }
        drop(operator);
        // Host tags only. In the qualified no-hard/no-lock path these solve buffers persist to B0.
        if analytic_assembly_ledger.post_complete {
            let _=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||
                analytic_assembly_ledger.solve_finished(step,final_step,dt,&report)));
        }
        // Give the value array back so the next iteration reuses it.
        state.factor_tet.check(device)?;
        state.fixed_values = fixed.into_values();

        // Name: Linear Solve Time
        // Format: list[(time, ms)]
        // Map: pcg_linsolve
        // Description:
        // Wall-clock time in milliseconds spent in the preconditioned
        // conjugate gradient (PCG) linear solve for the Newton step
        // direction. One entry per Newton iteration. Typically the
        // dominant per-iteration cost.
        //
        // RECORDED BEFORE THE VERDICT IS READ, because the scope closes before
        // the outcome is tested: a solve that ran to the iteration
        // cap still spent the time it spent, and dropping the row there would
        // leave the stream shortest exactly where it is most worth reading. A
        // `?` inside a branch above escapes without this row, and that path is
        // a `Fatal` that ends the run rather than a failed step.
        log::mark("advance", "linsolve", _linsolve.elapsed());

        // STEP B17a. The solve's verdict.
        // Name: Linear Solve Iteration Count
        // Format: list[(time, iterations)]
        // Map: pcg_iter
        // Description:
        // Number of preconditioned conjugate gradient (PCG) iterations
        // consumed during the linear solve for this Newton iteration.
        // High values indicate an ill-conditioned system or a tight
        // tolerance and often correlate with long linear-solve times.
        log::mark("advance", "iter", report.iterations as f64);
        // Name: Linear Solve Relative Residual
        // Format: list[(time, ratio)]
        // Map: pcg_resid
        // Description:
        // Final relative residual reached by the PCG linear solve for this
        // Newton iteration. When this stays well below the configured
        // tolerance, the solve converged cleanly, values close to the
        // tolerance indicate the iteration cap was hit. In a scene with rigid
        // (PDRD) bodies the reduced solve measures each degree-of-freedom group
        // against its own initial residual, and this reports the worst group,
        // so a body and the cloth cannot mask each other.
        log::mark("advance", "reresid", report.relative_residual as f64);
        // Name: Schwarz Block-Jacobi Fallback
        // Format: list[(time, count)]
        // Description:
        // 1 if the solver fell back from the Schwarz preconditioner to the
        // SPD-safe block-Jacobi base for this Newton iteration's PCG solve, else
        // 0. Always 0 under the block-jacobi preconditioner; a nonzero entry is
        // worth reviewing. Recorded every iteration but only printed when
        // nonzero so the common 0 case does not clutter the log.
        //
        // A STRUCTURAL ZERO HERE, NOT AN UNMEASURED ONE, which is why it is
        // `quiet` rather than a number whose print is decided per iteration.
        // Two things could raise it and this driver has neither: there is no
        // memory guard that degrades Schwarz to block-Jacobi, and `schwarz.rs`
        // builds its aggregate term so the `rz <= 0` breakdown cannot arise.
        // The channel is recorded so a reader asking for it by name gets an
        // answer, and the day a degrade path is added this becomes a `Number`
        // and starts printing.
        log::mark("advance", "schwarz_fallback", log::Marked::quiet(0.0));
        match report.outcome {
            // BOTH ARE SUCCESSES, and the truncation reports itself at the
            // site where it happens (`pcg.rs`). Nothing is added here.
            pcg::Outcome::Converged | pcg::Outcome::AcceptedAnimation | pcg::Outcome::CurvatureTruncated => {}
            pcg::Outcome::MaxIterations | pcg::Outcome::NoProgress => {
                let no_progress = matches!(report.outcome, pcg::Outcome::NoProgress);
                // A FAILURE: the cap is reported as `### cg failed` with
                // `pcg_success = false` and an abandoned step.
                //
                // THE ITERATE IS STILL RETURNED, AND THAT IS NOT PERMISSION TO
                // CONTINUE. A direction that never met `cg-tol` is not a Newton
                // direction, and taking it produces a run that exits 0 with
                // every frame written and no statement anywhere that the linear
                // solves never converged.
                //
                // `CurvatureTruncated` above stays a SUCCESS on the opposite
                // grounds: Steihaug truncation on an unresolvable curvature
                // stops at the best direction fp32 can see, rather than at an
                // unconverged one.
                //
                // THE LINE BELONGS TO THE SOLVER'S OWN TRANSCRIPT, so it is
                // bare. The iteration count and residual are this driver's
                // addition and the exponent is rebuilt to C's `%.3e` for the
                // same reason every other mark is.
                log::message!(
                    "### cg failed ({}, {} actual iterations, relative residual {})",
                    if no_progress { "NoProgress" } else { "iteration cap" },
                    report.iterations,
                    log::c_exponential(f64::from(report.relative_residual), 3)
                );
                if adaptive_retry {
                    result.retry_iterations = report.iterations;
                    result.retry_residual = report.relative_residual;
                    let classify = if no_progress { super::adaptive_retry::classify_no_progress } else { super::adaptive_retry::classify_finite_cap };
                    let retryable_cap = classify(
                        super::adaptive_retry::PcgCap {
                            iterations: report.iterations,
                            iteration_limit: prm.cg_max_iter,
                            relative_residual: report.relative_residual,
                            tolerance: prm.cg_tol,
                        },
                        [result.ccd_success, result.intersection_free,
                         result.newton_progress, result.pin_feasible,
                         result.contact_separated],
                    );
                    if retryable_cap {
                        // Hard-constraint trial state has a separate transaction
                        // protocol; until it is integrated, fail closed by name.
                        if state.hard.active() {
                            result.retry_reason = crate::data::RETRY_REASON_UNSUPPORTED_STATE;
                            log::message!("CG_RETRY_UNSUPPORTED_STATE: capability=hard_joint_trial_restore");
                        } else {
                            result.retry_reason = crate::data::RETRY_REASON_PCG_MAX_ITERATIONS;
                        }
                    }
                }
                result.pcg_success = false;
                if let Some(attempt)=root_attempt.as_ref(){state.roots.transaction.as_mut().unwrap().reject(attempt).map_err(|e| Fatal::invariant(e.to_string()))?;}
                    return Ok(result);
            }
            // A BREAKDOWN IS FATAL, NOT A FAILED STEP, and the distinction is
            // the whole of the `pAp<=0` triage. The Newton
            // matrix and the block-Jacobi preconditioner are SPD BY
            // CONSTRUCTION, every per-element Hessian block that can be
            // indefinite being PSD-projected at the source, so a sign that
            // survives the round-off bound is a real regression upstream: a
            // lost projection, a sign or assembly error, a dropped CSR block.
            // Reporting it as an ordinary unconverged step collapses it onto
            // the same `CrashKind::Cg` an iteration-cap exhaustion produces,
            // and the natural response to THAT is to raise `cg-max-iter` or
            // loosen `cg-tol`, which buries the defect instead of surfacing it.
            // Every one of these sites aborts the run.
            pcg::Outcome::IndefiniteMatrix => {
                return Err(Fatal::invariant(format!(
                    "PCG breakdown: p^T A p is negative beyond the round-off \
                     bound of its own sum at iteration {}, relative residual \
                     {:.3e}. The assembled Newton Hessian is not SPD. Beyond \
                     that bound the sign is real, so this is a per-element \
                     Hessian missing its PSD projection, a sign or assembly \
                     error, or a dropped off-diagonal block. It is NOT a \
                     tolerance to loosen: a genuine defect puts the Rayleigh \
                     quotient at order one, six orders above the bound.",
                    report.iterations, report.relative_residual
                )));
            }
            pcg::Outcome::NonSpdPreconditioner => {
                return Err(Fatal::invariant(format!(
                    "PCG breakdown: r^T M^-1 r is non-positive at iteration \
                     {}, relative residual {:.3e}. The preconditioner is not \
                     SPD. Every block-Jacobi diagonal block is inverted \
                     through a floored symmetric eigendecomposition, so each \
                     per-vertex term is positive by construction and their sum \
                     cannot be negative in exact arithmetic: a non-positive \
                     value here means a block is NaN or infinite, or the \
                     residual itself is not finite.",
                    report.iterations, report.relative_residual
                )));
            }
        }

        // STEP B18a. THE SPIN RECOVERED from the solved translation increment,
        // by Schur back-substitution, immediately after the solve that produced
        // the increment. It uses the RAW solve direction: the line search's toi
        // rescale is not applied to the recovered spin, which is a small-toi
        // approximation stated here rather than corrected.
        if grains > 0 {
            // THE STATE'S OWN BLOCKS, present whenever the scene has grains.
            // Asking the collider and substituting `Handle::NONE` when there is
            // none left this pass with an unresolvable buffer in exactly the
            // scenes that carry grains and no floor or sphere.
            let (angular, coupling, rotational) = (
                state.grain_angular.handle(),
                state.grain_coupling.handle(),
                state.grain_rotational.handle(),
            );
            let recover = SandGrainRecoverRowArgs {
                inverse_center_inertia: state.grain_inv_inertia_center.handle(),
                angular,
                coupling,
                rotational_gradient: rotational,
                previous_omega: state.grain_omega_prev.handle(),
                increment: dx.handle(),
                omega: state.grain_omega.handle(),
                dt,
                count: vertices as u32,
                seam_arena_count: 0,
            };
            device.launch("sand.recover", &recover, vertices as u32)?;
        }

        let _nt_mid = std::time::Instant::now();
        // STEP B19. The search direction's largest per-vertex magnitude, and
        // the rescale that keeps any one vertex inside `max_dx`.
        let magnitude_args = DxMagnitudeArgs {
            direction: state.dx.handle(),
            magnitude: state.scalar.handle(),
            count: vertices as u32,
            seam_arena_count: 0,
        };
        device.launch("step.dx_magnitude", &magnitude_args, vertices as u32)?;
        // AND THIS ONE FOLDS ON THE DEVICE TOO; see the start-of-step pair.
        let max_dx = {
            let a = state.scalar.handle();
            // Safety: the buffer outlives the call and names `vertices` floats.
            unsafe { state.fold.max(device, "step.max_dx", a, vertices as u32, 0.0) }?
        };
        // Name: Max Search Direction Magnitude
        // Format: list[(time, meters)]
        // Map: max_search_dir
        // Description:
        // Maximum per-vertex magnitude (L2 norm) of the Newton search
        // direction returned by the linear solve for this Newton
        // iteration, in meters. Compared against the max_dx parameter to
        // decide whether the search direction must be rescaled before
        // the line search.
        log::mark("advance", "max_dx", max_dx as f64);
        let toi_recale = (prm.max_dx / max_dx).min(1.0);
        // Name: Search Direction Rescale Factor
        // Format: list[(time, ratio)]
        // Description:
        // Scalar in (0, 1] applied to the Newton search direction before
        // the line search, so that no per-vertex displacement exceeds the
        // configured max_dx. A value of 1.0 means the direction was
        // already within budget, smaller values clamp an over-eager step.
        log::mark("advance", "toi_recale", toi_recale as f64);

        // STEP B20. `target` is reused to hold the iterate's pre-step
        // positions, then the rescaled step is applied to the iterate.
        state.target.copy_from(device, &state.eval_x)?;
        let step_args = PositionStepArgs {
            eval_x: state.eval_x.handle(),
            direction: state.dx.handle(),
            scale: toi_recale,
            count: vertices as u32,
            seam_arena_count: 0,
        };
        device.launch("step.position_step", &step_args, vertices as u32)?;

        // STEP B21 and B22. The `fix-xz` position drag, the half that moves the
        // accepted step. Its momentum half is assembled above, and the two ship
        // together: the momentum term is what the linear system is built from
        // and this clamp is what the step does, so a backend carrying one alone
        // would solve a system that does not describe the step it then takes.
        //
        // Anchor to the incoming accepted current pose for this simulation
        // step, not the n-1 velocity-history pose or mutable Newton iterate.
        // Saturation holds x_(n+1) = x_n instead of oscillating over two steps.
        if prm.fix_xz != 0.0 {
            let drag_args = FixXzDragArgs {
                eval_x: state.eval_x.handle(),
                previous: curr,
                dof_removed: state.dof_mask.handle(),
                threshold: prm.fix_xz,
                count: vertices as u32,
                seam_arena_count: 0,
            };
            device.launch("step.fix_xz_drag", &drag_args, vertices as u32)?;
        }

        // RECORDED AND NOT PRINTED: the rescale and the position step are not
        // one of the solver's own printed scopes.
        log::mark(
            "advance",
            "nt_mid",
            log::Marked::quiet(_nt_mid.elapsed().as_secs_f64() * 1000.0),
        );
        // STEP B28. The AABB refresh the line search queries against. The
        // leaves are re-bounded over the WHOLE candidate step, from the
        // pre-step positions `target` now holds to the proposed iterate, so a
        // pair that ends the step apart but crossed inside it is still a
        // candidate.
        //
        // The refresh comes BEFORE the sweep and after the step was proposed:
        // bounding the old pose alone would miss every crossing the step
        // introduces.
        if !prm.disable_contact {
            let _aabb = std::time::Instant::now();
            let SolverState {
                contact: contact_state,
                target,
                eval_x,
                ..
            } = &mut *state;
            let (start, finish) = (target.handle(), eval_x.handle());
            if let Some(contact) = contact_state.as_mut() {
                contact.refresh_leaves(device, data, mesh_refs, start, finish, prm.line_search_max_t, windows)?;
            }
            // THIS SCOPE CLOSES BEFORE THE LINE SEARCH OPENS, which is what
            // puts `> aabb_update` above `> line_search` in the stream. It
            // carries no docstring, so the channel is absent from `names()` and
            // only the printed line exists.
            log::mark("advance", "aabb_update", _aabb.elapsed());
        }

        if sand_offset_trace.enabled {sand_offset_trace.capture(device,"pre-CCD",step,final_step,dt,toi_advanced,toi_recale,prm.line_search_max_t,state.target.handle(),state.eval_x.handle(),None);}
        if analytic_gap_trace.enabled() {analytic_gap_trace.capture(device,"pre-CCD",step,final_step,dt,toi_advanced,toi_recale,prm.line_search_max_t,state.target.handle(),state.eval_x.handle(),None,state.floor.host(),root_attempt.as_ref());}

        // STEP B29. The line search. The ACCD filter is one of the two things
        // that make this solver penetration-free: it returns the largest
        // fraction of the proposed step that crosses nothing, and no step past
        // it is ever taken. With `disable-contact` set there are no pairs to
        // sweep and the filter returns its whole span, which the ratio below
        // turns into 1.
        // THE LINE SEARCH, timed as `line_search`.
        let _line_search = std::time::Instant::now();
        let (mut toi, b29_certificate) = {
            let filter = if prm.disable_contact {
                None
            } else {
                let SolverState {
                    contact: contact_state,
                    target,
                    eval_x,
                    ..
                } = &mut *state;
                let (start, finish) = (target.handle(), eval_x.handle());
                // NO POSE CROSSES TO THE HOST HERE. The sweep is six device
                // dispatches over the two handles above, each walking a BVH
                // with the ACCD advance as a per-hit functor. What comes back
                // is two per-primitive float arrays reduced to one number
                // each.
                match contact_state.as_mut() {
                    Some(contact) => Some(contact.line_search(
                        device,
                        data,
                        mesh_refs,
                        &*param,
                        start,
                        finish,
                        windows,
                    )?),
                    None => {
                        return Err(Fatal::invariant(
                            "solver driver: the line search was reached with no contact \
                             subsystem allocated, so nothing would bound the step",
                        ))
                    }
                }
            };
            // A PAIR THAT BEGAN THE STEP INSIDE ITS OFFSET RETURNS EXACTLY
            // ZERO, and reading that as "no progress, try a shorter step" spins
            // forever on a state the conservative advance cannot resolve. It is
            // asked about BEFORE the time is used, and the step ends naming the
            // pair.
            // THE ANALYTIC SWEEP, which runs OUTSIDE the `disable-contact`
            // gate: a prescribed vertex driven through a floor tunnels it
            // whether or not mesh contact is live, and no barrier can stop
            // it.
            let sweep = {
                let SolverState {
                    analytic: analytic_state,
                    target,
                    eval_x,
                    fix,
                    sphere,
                    floor,
                    ..
                } = &mut *state;
                let (start, finish) = (target.handle(), eval_x.handle());
                let Some(analytic) = analytic_state.as_mut() else {
                    return Err(Fatal::invariant(
                        "solver driver: the analytic collider sweep was reached with no \
                         collider layer allocated, so nothing would bound a \
                         vertex against a sphere or a floor",
                    ));
                };
                let sweep = analytic.line_search(device, data, mesh_refs, param, start, finish, fix.handle(), sphere, floor)?;
                if analytic_gap_trace.enabled() { let query_toi=analytic.diagnostic_toi(device).ok(); analytic_gap_trace.capture(device,"B29-analytic-sweep-input",step,final_step,dt,toi_advanced,toi_recale,prm.line_search_max_t,start,finish,query_toi,floor.host(),root_attempt.as_ref()); }
                sweep
            };
            // A FIX-PINNED VERTEX DRIVEN THROUGH A COLLIDER IS REPORTED, NOT
            // CLAMPED. It has no freedom to yield, so a clamped time of impact
            // would stall the whole solve without preventing the crossing. The
            // step fails naming the vertex.
            if let Some(bad) = sweep.infeasible_pin {
                // THE WORDING IS A HOST CONTRACT, character for character. A
                // Rust string literal keeps the indentation of its continuation
                // lines, so the naive spelling carries a run of spaces
                // mid-sentence and matches nothing grepping for the message.
                // Concatenated adjacent literals are what avoid that.
                log::message!(
                    "### infeasible pin: prescribed vertex {bad} is driven \
                     through an analytic collider (floor/sphere) it cannot \
                     yield to"
                );
                log::message!(
                    "### re-author the pin's path so it stays outside the \
                     collider, or make it a soft pull pin so it can yield."
                );
                // THE ROW IS RECORDED AFTER THE REPORT: both messages, then
                // the scope's own row. The row has to be written at all on an
                // early return, because an unbalanced scope is an error in the
                // log, but it belongs after the two lines that explain the
                // failure and not before them.
                log::mark("advance", "line_search", _line_search.elapsed());
                result.pin_feasible = false;
                if let Some(attempt)=root_attempt.as_ref(){state.roots.transaction.as_mut().unwrap().reject(attempt).map_err(|e| Fatal::invariant(e.to_string()))?;}
                    return Ok(result);
            }
            if let Some(overlap) = filter.as_ref().and_then(super::ccd::Filter::overlapping_start)
            {
                log::message!("### contact starts overlapping: {}", overlap.describe());
                log::message!(
                    "### give the initial geometry a small clearance so nothing starts in \
                     contact, or check whether a pin is pulling elements together faster than \
                     contact can resolve."
                );
                // Recorded after the report, for the reason above.
                log::mark("advance", "line_search", _line_search.elapsed());
                result.contact_separated = false;
                if let Some(attempt)=root_attempt.as_ref(){state.roots.transaction.as_mut().unwrap().reject(attempt).map_err(|e| Fatal::invariant(e.to_string()))?;}
                    return Ok(result);
            }
            // THE CONTACT FILTER'S OWN FRACTION, recorded before the analytic
            // sweep and the strain limiters narrow it. It is a different
            // question from the accepted `toi` below: this one says whether
            // MESH CONTACT bit, which is the only reading that distinguishes a
            // step contact bounded from one a limiter did.
            //
            // RECORDED AND NOT PRINTED. The value keeps its stream file, which
            // `examples/metal_pdrd_*_fixture.py` read, and stays out of the
            // printed transcript.
            log::mark(
                "advance",
                "contact_toi",
                log::Marked::quiet(f64::from(
                    filter
                        .as_ref()
                        .map_or(prm.line_search_max_t, super::ccd::Filter::time_of_impact)
                        / prm.line_search_max_t,
                )),
            );
            let smallest = filter
                .as_ref()
                .map_or(prm.line_search_max_t, super::ccd::Filter::time_of_impact)
                .min(sweep.time_of_impact);
            let certificate = filter.as_ref().map(|mesh| b32_shadow::SweepCertificate {
                raw_toi: mesh.time_of_impact(), ceiling: prm.line_search_max_t,
                overlapping_start: mesh.overlapping_start().is_some(),
            });
            ((smallest / prm.line_search_max_t).min(1.0), certificate)
        };
        // THE STRAIN LIMITER'S OWN HALF, folded in after the contact CCD.
        // This is the half that makes the limit
        // a limit: the barrier above raises the cost of stretching and this
        // REFUSES the fraction of the step that would cross it, so an authored
        // limit bounds the accepted pose rather than merely discouraging it.
        //
        // `SL_toi` OPENS AT 1.0 AND IS THE MINIMUM OVER BOTH HALVES, which is
        // what the failure message below reports; `toi` takes each half
        // separately. Both reductions fold the per-element
        // array the walk wrote, so the answer does not depend on how the range
        // was cut.
        let mut sl_toi = 1.0f32;
        if state.sizes.shell_faces > 0 {
            assemble::shell_strain_toi(device, data, state, prm.line_search_max_t)?;
            let faces = state.sizes.shell_faces;
            let toi_array = state.face_strain.toi.handle();
            // Safety: the buffer outlives the call and names `faces` floats.
            let smallest = unsafe {
                state.fold.min(
                    device,
                    "step.shell_strain_toi",
                    toi_array,
                    faces as u32,
                    prm.line_search_max_t,
                )
            }?;
            let shell_toi = smallest / prm.line_search_max_t;
            sl_toi = sl_toi.min(shell_toi);
            toi = toi.min(shell_toi);
            // Name: Strain-Limit Time of Impact
            // Format: list[(time, ratio)]
            // Description:
            // Fraction in (0, 1] of the rescaled search direction that can
            // be taken without violating the configured shell or rod
            // strain limits, as returned by the strain-limiting line
            // search. A value of 1.0 means strain limits never bound the
            // step, smaller values mean the strain limiter clamped it.
            log::mark("advance", "SL_toi", shell_toi as f64);
        }
        if state.sizes.rods > 0 {
            assemble::rod_strain_toi(device, data, state, prm.line_search_max_t)?;
            let rods = state.sizes.rods;
            let toi_array = state.rod_strain.toi.handle();
            // Safety: the buffer outlives the call and names `rods` floats.
            let smallest = unsafe {
                state.fold.min(
                    device,
                    "step.rod_strain_toi",
                    toi_array,
                    rods as u32,
                    prm.line_search_max_t,
                )
            }?;
            let rod_toi = smallest / prm.line_search_max_t;
            sl_toi = sl_toi.min(rod_toi);
            toi = toi.min(rod_toi);
            // Name: SL-rod-toi
            // Format: list[(time, ratio)]
            // Map: SL-rod-toi
            // Description:
            // Fraction in (0, 1] of the rescaled search direction the ROD
            // strain limiter allows, reported separately from the shell
            // limiter's because a rod-only scene never marks that one. The
            // docstring above is what makes the channel askable by name: a
            // value that only reaches stdout ties every caller to the exact
            // phrasing of a printed line.
            log::mark("advance", "SL_rod_toi", rod_toi as f64);
        }
        // Name: Line Search Time
        // Format: list[(time, ms)]
        // Description:
        // Wall-clock time in milliseconds spent in the per-iteration
        // line search, which runs continuous collision detection (CCD)
        // plus strain-limit CCD to find the largest feasible substep
        // along the rescaled search direction. One entry per Newton
        // iteration.
        //
        // CLOSED AFTER BOTH STRAIN LIMITERS, NOT BEFORE THEM. The scope opens
        // above the contact sweep and closes after `SL_toi` and `SL_rod_toi`
        // are marked, so the limiters are inside the measured span and their
        // two lines print ABOVE `> line_search`. Recording it where the contact
        // sweep ends would report a shorter time under the `line_search` name
        // and put the lines in the wrong order.
        log::mark("advance", "line_search", _line_search.elapsed());
        if analytic_gap_trace.enabled() { analytic_gap_trace.capture(device,"B31-original-before-certification",step,final_step,dt,toi_advanced,toi_recale,toi,state.target.handle(),state.eval_x.handle(),None,state.floor.host(),root_attempt.as_ref()); }
        if analytic_gap_trace.enabled() {
            let root_view=state.roots.view(&state.rigid);
            analytic_gap_trace.capture_root_buffers(device,"B31-original-before-certification-root-workspace",pdrd_bodies as usize,
                state.rigid_state.handle(),state.rigid_rotation_step.handle(),state.rigid_running_rotation.handle(),state.rigid_centroid.handle(),root_view);
            analytic_gap_trace.capture_root_direction(device,state.dx.handle(),state.pcg_rigid.xr.handle(),vertices,state.pcg_rigid.xr.len());
        }
        // Determine the actual safe rounded candidate BEFORE advancing B30's clock.
        let controlled=root_attempt.as_ref().map_or(&[][..],|a|a.body_indices.as_slice());
        let toi = if toi>f32::EPSILON {
            let origin=state.target.handle();let proposed=state.eval_x.handle();
            let certify_mesh=!prm.disable_contact && (state.joints.active() || !controlled.is_empty());
            let SolverState {analytic,contact,floor,..}=&mut *state;
            // The B29 certificate lives only in this Newton iteration. Between
            // B29 and here shell/rod strain TOI and optional traces do not write
            // target, eval_x, windows or the controlled set. Thus origin and
            // proposal contents, mesh/topology, dt and CCD parameters remain
            // those used by the sweep; handle equality alone is not the proof.
            // The acceptor separately compares every finite rounded candidate
            // coordinate bitwise with its captured proposal on the first full trial.
            let mode=std::env::var("PPF_B31_CERTIFICATE_MODE").unwrap_or_default();
            let shadow_enabled=mode=="shadow";
            let reuse_enabled=mode=="reuse";
            let receipt=std::env::var_os("PPF_B31_REUSE_RECEIPT").is_some_and(|v|v=="1");
            let compare=certify_mesh && (shadow_enabled || reuse_enabled)
                && b29_certificate.is_some_and(b32_shadow::SweepCertificate::covers_endpoint)
                && toi.to_bits()==1.0_f32.to_bits();
            let mut check_mesh=|device:&mut D,candidate:ppf_cts_compute::Handle,trial:u32,fraction:f32,same:Option<bool>,compare_ms:f64|->FatalResult<bool>{
                let eligible=compare && trial==0 && fraction.to_bits()==1.0_f32.to_bits() && same==Some(true);
                if reuse_enabled && eligible {
                    if receipt {eprintln!("B31_REUSE_RECEIPT trial={} fraction={:.9e} eligible=true same={:?} used=true skipped_mesh_sweep_refit=true compare_ms={:.3}",trial,fraction,same,compare_ms);}
                    return Ok(true);
                }
                let started=std::time::Instant::now();
                let original=contact.as_mut().ok_or_else(||Fatal::invariant("MESH_ROUNDED_ACCEPT_UNRESOLVED: no contact layer"))?
                    .certify_rounded_candidate(device,data,mesh_refs,&*param,origin,candidate,windows);
                if shadow_enabled {
                    let original_ms=started.elapsed().as_secs_f64()*1000.0;
                    let original_pass=original.as_ref().ok().copied();
                    eprintln!("B31_CERTIFICATE_SHADOW trial={} fraction={:.9e} raw_toi={:?} ceiling={:?} same={:?} eligible={} read_error={} original_pass={:?} disagreement={} compare_ms={:.3} original_ms={:.3} skipped_mesh_sweep_refit=false",
                        trial,fraction,b29_certificate.map(|c|c.raw_toi),b29_certificate.map(|c|c.ceiling),same,eligible,
                        compare && trial==0 && fraction.to_bits()==1.0_f32.to_bits() && same.is_none(),original_pass,
                        eligible && original_pass!=Some(true),compare_ms,original_ms);
                }
                if receipt {eprintln!("B31_REUSE_RECEIPT trial={} fraction={:.9e} eligible={} same={:?} used=false skipped_mesh_sweep_refit=false compare_ms={:.3}",trial,fraction,eligible,same,compare_ms);}
                original
            };
            let mesh_check:Option<&mut dyn FnMut(&mut D,ppf_cts_compute::Handle,u32,f32,Option<bool>,f64)->FatalResult<bool>>=
                if certify_mesh {Some(&mut check_mesh)}else{None};
            let acceptance=analytic.as_mut().ok_or_else(||Fatal::invariant("ANALYTIC_ACCEPT_UNRESOLVED: no analytic layer"))?
                .certified_plane_accept(device,mesh_refs,origin,proposed,vertices,toi,floor,controlled,compare,mesh_check);
            match acceptance {
                Ok(alpha)=>alpha,
                Err(error)=>{
                    // ACCEPT FAILURE OBSERVER: BEGIN B31 before original reject
                    capture_acceptance_failure(device,data,state,root_attempt.as_ref(),"B31_acceptance_error_pre_reject",
                        step,final_step,dt,toi_advanced,toi_recale,toi,origin,proposed,pdrd_bodies as usize,vertices,prm.time,
                        (*param).ccd_eps,prm.line_search_max_t,&error.detail);
                    // ACCEPT FAILURE OBSERVER: END B31 before original reject
                    analytic_gap_trace.flush(&error.detail,(*param).ccd_eps,prm.line_search_max_t);
                    if let Some(a)=root_attempt.as_ref(){state.roots.transaction.as_mut().unwrap().reject(a).map_err(|e|Fatal::invariant(e.to_string()))?;}return Err(error);
                }
            }
        } else { toi }; // Preserve the original CCD refusal before any accept.
        if analytic_gap_trace.enabled() { analytic_gap_trace.capture(device,"B31-certified-actual-before-clock",step,final_step,dt,toi_advanced,toi_recale,toi,state.target.handle(),state.eval_x.handle(),None,state.floor.host(),root_attempt.as_ref()); }
        // Name: Line Search Time of Impact
        // Format: list[(time, ratio)]
        // Description:
        // Fraction in (0, 1] of the rescaled Newton search direction that
        // can be taken without causing a collision or violating strain
        // limits, as the minimum of the contact CCD result and the
        // strain-limit TOI. A value of 1.0 means the full Newton step was
        // accepted, smaller values mean the line search cut it short.
        log::mark("advance", "toi", toi as f64);
        last_toi = toi;
        if toi <= f32::EPSILON {
            // BOTH LINES BELONG TO THE SOLVER'S OWN TRANSCRIPT, both at `%.2e`.
            log::message!("### ccd failed (toi: {})", log::c_exponential(f64::from(toi), 2));
            if sl_toi < 1.0 {
                log::message!(
                    "strain limiting toi: {}",
                    log::c_exponential(f64::from(sl_toi), 2)
                );
            }
            result.ccd_success = false;
            if let Some(attempt)=root_attempt.as_ref(){state.roots.transaction.as_mut().unwrap().reject(attempt).map_err(|e| Fatal::invariant(e.to_string()))?;}
                    return Ok(result);
        }

        let _nt_tail = std::time::Instant::now();
        // STEP B30. How far through its span the step has advanced. Kept in
        // double, because it accumulates across every Newton iteration.
        if !final_step {
            toi_advanced += (1.0 - toi_advanced).max(0.0) * f64::from(toi_recale * toi);
            full_remaining_sweep_accepted |= toi_recale == 1.0 && toi == 1.0;
        }
        // PRINTED AT EVERY NEWTON ITERATION, so it is a bare transcript line
        // rather than a debug one. It
        // reads like a mark and is not: the recorded channel is written once
        // per step at STEP C1, and this is the running total the reader watches
        // climb toward `target_toi`.
        log::message!("* toi_advanced: {}", log::c_exponential(toi_advanced, 2));

        let sand_query_toi=if sand_offset_trace.enabled {state.contact.as_mut().and_then(|c|c.sand_offset_diagnostic_toi(device))}else{None};
        if sand_offset_trace.enabled {sand_offset_trace.capture(device,"CCD-bound-before-accept",step,final_step,dt,toi_advanced,toi_recale,toi,state.target.handle(),state.eval_x.handle(),sand_query_toi);}
        if analytic_gap_trace.enabled() { analytic_gap_trace.capture(device,"B31-post-certification",step,final_step,dt,toi_advanced,toi_recale,toi,state.target.handle(),state.eval_x.handle(),None,state.floor.host(),root_attempt.as_ref()); }

        // STEP B31. Accept the line search's fraction, from the pre-step
        // positions `target` is holding.
        let accept_args = PositionAcceptArgs {
            origin: state.target.handle(),
            proposed: state.eval_x.handle(),
            fraction: toi,
            count: vertices as u32,
            seam_arena_count: 0,
        };
        // certified_plane_accept has already committed precisely its certified scratch.
        // Keep the record/fraction for trace and the original B31 phase ordering.
        let _ = accept_args;

        // This branch remains unreachable until the trusted mode overlay is
        // explicitly enabled after complete root-path and checkpoint support.
        // Do not infer turns from endpoint orientations alone.
        let twist_b31=if let Some(selected)=state.joints.twist_selection().map(|x|x.to_vec()) {
            let mut fitted=vec![crate::data::PdrdRigidState::default();state.rigid.n_bodies];
            state.rigid_state.read(device,0,&mut fitted)?;
            let origin_fit=state.rigid_state.handle();
            let origin=super::joints::capture_twist_pose(device,data,state,&selected,dt,super::joints::TwistPoseSource::Target,Some(origin_fit)).map_err(|error|{
                let _=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||{
                    eprintln!("TWIST_SAMPLE_STEP_CONTEXT source=B31_target parameter_time={:.17e} newton_step={} adaptive_retry_token={} dt_bits={:08x}",prm.time,step,state.adaptive_retry_token,dt.to_bits());
                    analytic_gap_trace.flush(&error.detail,(*param).ccd_eps,prm.line_search_max_t);
                }));error
            })?;
            let origin_sequence=state.joints.twist_sample_sequence();
            let end=match super::joints::capture_twist_pose(device,data,state,&selected,dt,super::joints::TwistPoseSource::Iterate,None) {
                Ok(end)=>end,
                Err(mut error)=>{
                    let _=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||{
                        eprintln!("TWIST_SAMPLE_STEP_CONTEXT source=B31_iterate parameter_time={:.17e} newton_step={} adaptive_retry_token={} dt_bits={:08x}",prm.time,step,state.adaptive_retry_token,dt.to_bits());
                        analytic_gap_trace.flush(&error.detail,(*param).ccd_eps,prm.line_search_max_t);
                    }));
                    if allow_swing_retry {
                        if let Err(qualification)=state.joints.mark_swing_sample_trial(device,&state.rigid,&selected,origin_sequence,prm.time,dt) {
                            error.detail.push_str(&format!("; swing trial qualification unavailable: {qualification:?}"));
                        }
                    }
                    return Err(error);
                }
            };
            let view=state.joints.view(state.rigid_state.handle(),&state.rigid,dt);
            if !state.joints.advance_segment_pieces(device,view,&selected,&origin,&end,None)? {
                state.joints.mark_uncertified_path_retry(device,dt,"B31")?;
                return Err(Fatal::invariant("CONTINUOUS_TWIST_PATH_UNCERTIFIED: B31"));
            }
            Some((selected,end,fitted))
        } else {None};

        if sand_offset_trace.enabled {sand_offset_trace.capture(device,"post-B31",step,final_step,dt,toi_advanced,toi_recale,toi,state.target.handle(),state.eval_x.handle(),None);}
        if analytic_gap_trace.enabled() {analytic_gap_trace.capture(device,"post-B31",step,final_step,dt,toi_advanced,toi_recale,toi,state.target.handle(),state.eval_x.handle(),None,state.floor.host(),root_attempt.as_ref());}
        if analytic_gap_trace.enabled() {
            let root_view=state.roots.view(&state.rigid);
            analytic_gap_trace.capture_root_buffers(device,"post-B31-root-workspace",pdrd_bodies as usize,
                state.rigid_state.handle(),state.rigid_rotation_step.handle(),state.rigid_running_rotation.handle(),state.rigid_centroid.handle(),root_view);
            analytic_gap_trace.capture_root_direction(device,state.dx.handle(),state.pcg_rigid.xr.handle(),vertices,state.pcg_rigid.xr.len());
        }

        // STEP B32. THE PDRD RIGIDIFY COMMIT, and it is the reason a body
        // stays a body across frames rather than only within a step.
        //
        // The reduced solve moves each body rigidly, but the iterate it lands
        // on has also been through contact, the strain limiters and the line
        // search, all of which act PER VERTEX and none of which knows the body
        // is rigid. So the accepted pose is very slightly sheared, and refitting
        // to it next frame would fold that shear into the fit. Across a long run
        // it accumulates as a non-rigid shrink.
        //
        // The fix is to rebuild each body from a rotation that is ANCHORED
        // rather than refitted: `running_rotation` carries the rotation the
        // solve has actually applied, composed across every iteration and every
        // frame, so the rigid image owes nothing to the sheared iterate but its
        // CENTROID.
        //
        // AND THE COMMIT IS CCD-FILTERED LIKE ANY OTHER STEP. Snapping to the
        // rigid image would move vertices, and a move that crosses geometry
        // penetrates whatever it crossed no matter how principled the pose it
        // moves toward. So the same sweep that bounds the Newton step bounds
        // this one, and the same two failures end the step naming what they hit.
        if pdrd_bodies > 0 {
            {
                let SolverState {
                    rigid,
                    rigid_staged,
                    rigid_running_rotation,
                    rigid_state,
                    roots,
                    rigid_rotation_step,
                    rigid_centroid,
                    eval_x,
                    rigid_target,
                    prop_vertex,
                    pcg_rigid,
                    ..
                } = &mut *state;
                let body_vertices = data.pdrd_vert_list.size;
                let vert_list = rigid_staged.vert_list.handle();
                let body_prop = rigid_staged.body_prop.handle();
                let rest_centered = rigid_staged.rest_centered.handle();
                let mut reduction = super::pdrd::Reduction {
                    map: rigid,
                    staged: rigid_staged,
                };
                if analytic_gap_trace.enabled() {
                    let root_view=roots.view(&*reduction.map);
                    analytic_gap_trace.capture_root_buffers(device,"B32-before-compose-root-workspace",pdrd_bodies as usize,
                        rigid_state.handle(),rigid_rotation_step.handle(),rigid_running_rotation.handle(),rigid_centroid.handle(),root_view);
                }
                reduction.compose_running_rotation(
                    device,
                    rigid_running_rotation.handle(),
                    rigid_rotation_step.handle(),
                    -(toi_recale * toi),
                )?;
                if analytic_gap_trace.enabled() {
                    let root_view=roots.view(&*reduction.map);
                    analytic_gap_trace.capture_root_buffers(device,"B32-pre-controlled-retract-root-workspace",pdrd_bodies as usize,
                        rigid_state.handle(),rigid_rotation_step.handle(),rigid_running_rotation.handle(),rigid_centroid.handle(),root_view);
                }
                roots.controlled_retract(device,rigid_state.handle(),rigid_rotation_step.handle(),rigid_running_rotation.handle(),toi_recale*toi)?;
                if analytic_gap_trace.enabled() {
                    let root_view=roots.view(&*reduction.map);
                    analytic_gap_trace.capture_root_buffers(device,"B32-post-controlled-retract-root-workspace",pdrd_bodies as usize,
                        rigid_state.handle(),rigid_rotation_step.handle(),rigid_running_rotation.handle(),rigid_centroid.handle(),root_view);
                }
                // The target starts as the iterate, so a NON-PDRD vertex is
                // carried through the commit untouched and its lerp below is a
                // no-op. Only the bodies' own vertices are overwritten.
                rigid_target.copy_from(device, eval_x)?;
                reduction.rigidify(
                    device,
                    vert_list,
                    prop_vertex.handle(),
                    body_prop,
                    rest_centered,
                    eval_x.handle(),
                    rigid_running_rotation.handle(),
                    rigid_centroid.handle(),
                    rigid_target.handle(),
                    body_vertices,
                    &pcg_rigid.zero,
                )?;
            }

            // The sweep, from the iterate to the rigid image.
            let (toi_rig, b32_certificate) = {
                // THE `rigidify_ccd` SCOPE OPENS HERE and closes after the
                // sweep and its readback and BEFORE the infeasibility report,
                // so the span covers the AABB refresh and both halves of the
                // sweep. The mesh and analytic halves are separate calls here,
                // so the mark below closes after the second.
                let _rigidify_ccd = std::time::Instant::now();
                if !prm.disable_contact {
                    let SolverState {
                        contact: contact_state,
                        eval_x,
                        rigid_target,
                        ..
                    } = &mut *state;
                    let (start, finish) = (eval_x.handle(), rigid_target.handle());
                    if let Some(contact) = contact_state.as_mut() {
                        contact.refresh_leaves(
                            device, data, mesh_refs, start, finish,
                            prm.line_search_max_t, windows,
                        )?;
                    }
                }
                let filter = if prm.disable_contact {
                    None
                } else {
                    let SolverState {
                        contact: contact_state,
                        eval_x,
                        rigid_target,
                        ..
                    } = &mut *state;
                    let (start, finish) = (eval_x.handle(), rigid_target.handle());
                    // THE SAME DEVICE NARROW PHASE AS B29, over the two handles
                    // above: six dispatches, each walking a BVH with the ACCD
                    // advance as a per-hit functor, and no pose on the host.
                    match contact_state.as_mut() {
                        Some(contact) => Some(contact.line_search(
                            device, data, mesh_refs, &*param, start, finish,
                            windows,
                        )?),
                        None => {
                            return Err(Fatal::invariant(
                                "solver driver: the rigidify commit was reached with no contact \
                                 subsystem allocated, so nothing would bound it",
                            ))
                        }
                    }
                };
                // THE ANALYTIC SWEEP IS INSIDE THE GATE HERE, AND THAT IS NOT
                // WHAT B29 DOES. The two sites gate at different levels and the
                // difference is easy to read backwards.
                //
                // At B29 only the MESH sweep is behind `disable-contact`: the
                // analytic and pin sweep runs regardless, because a prescribed
                // vertex driven through a floor tunnels it whether or not mesh
                // contact is live.
                //
                // At the RIGIDIFY COMMIT the whole sweep is behind the gate,
                // the AABB refresh included, so with contact disabled neither
                // the mesh nor the analytic half runs. Hoisting the analytic
                // half out of the gate here, on the grounds that B29 does so,
                // reads the inner gate as if it were the outer one.
                let sweep = if prm.disable_contact {
                    super::collider::Sweep::unobstructed(prm.line_search_max_t)
                } else {
                    let SolverState {
                        analytic: analytic_state,
                        eval_x,
                        rigid_target,
                        fix,
                        sphere,
                        floor,
                        ..
                    } = &mut *state;
                    let (start, finish) = (eval_x.handle(), rigid_target.handle());
                    let Some(analytic) = analytic_state.as_mut() else {
                        return Err(Fatal::invariant(
                            "solver driver: the rigidify commit's analytic collider sweep was \
                             reached with no collider layer allocated",
                        ));
                    };
                    let sweep = analytic.line_search(
                        device, data, mesh_refs, param, start, finish,
                        fix.handle(), sphere, floor,
                    )?;
                    if analytic_gap_trace.enabled() { let query_toi=analytic.diagnostic_toi(device).ok(); analytic_gap_trace.capture(device,"B32-analytic-sweep-input",step,final_step,dt,toi_advanced,toi_recale,prm.line_search_max_t,start,finish,query_toi,floor.host(),root_attempt.as_ref()); }
                    sweep
                };
                // GATED, because the scope covers work that does not happen
                // with contact off: neither half of the sweep runs, so a row
                // here would time nothing.
                if !prm.disable_contact {
                    log::mark("advance", "rigidify_ccd", _rigidify_ccd.elapsed());
                }
                if let Some(bad) = sweep.infeasible_pin {
                    log::message!(
                        "### infeasible pin: prescribed vertex {bad} is driven through an \
                         analytic collider during the rigidify commit"
                    );
                    result.pin_feasible = false;
                    if let Some(attempt)=root_attempt.as_ref(){state.roots.transaction.as_mut().unwrap().reject(attempt).map_err(|e| Fatal::invariant(e.to_string()))?;}
                    return Ok(result);
                }
                if let Some(overlap) =
                    filter.as_ref().and_then(super::ccd::Filter::overlapping_start)
                {
                    log::message!(
                        "### contact starts overlapping during the rigidify commit: {}",
                        overlap.describe()
                    );
                    result.contact_separated = false;
                    if let Some(attempt)=root_attempt.as_ref(){state.roots.transaction.as_mut().unwrap().reject(attempt).map_err(|e| Fatal::invariant(e.to_string()))?;}
                    return Ok(result);
                }
                let smallest = filter
                    .as_ref()
                    .map_or(prm.line_search_max_t, super::ccd::Filter::time_of_impact)
                    .min(sweep.time_of_impact);
                let certificate = filter.as_ref().map(|mesh| b32_shadow::SweepCertificate {
                    raw_toi: mesh.time_of_impact(),
                    ceiling: prm.line_search_max_t,
                    overlapping_start: mesh.overlapping_start().is_some(),
                });
                ((smallest / prm.line_search_max_t).min(1.0), certificate)
            };


            // The lerp toward the rigid image. `position_accept` writes into
            // its `proposed` argument, so the accepted pose lands in the target
            // and is copied back: the arithmetic is the same interpolation
            // every other accept uses, so the commit advances along its segment
            // by the same rule as every other accepted step.
            let accept = PositionAcceptArgs {
                origin: state.eval_x.handle(),
                proposed: state.rigid_target.handle(),
                fraction: toi_rig,
                count: vertices as u32,
                seam_arena_count: 0,
            };
            if analytic_gap_trace.enabled() { analytic_gap_trace.capture(device,"B32-original-before-certification",step,final_step,dt,toi_advanced,toi_recale,toi_rig,accept.origin,accept.proposed,None,state.floor.host(),root_attempt.as_ref()); }
            let controlled=root_attempt.as_ref().map_or(&[][..],|a|a.body_indices.as_slice());
            let toi_rig=if prm.disable_contact {
                // Keep the original whole B32 contact-off gate.
                device.launch("step.rigidify_accept",&accept,vertices as u32)?;
                toi_rig
            } else {
                let certify_mesh=state.joints.active() || !controlled.is_empty();
                let SolverState {analytic,contact,floor,..}=&mut *state;
                let mode = std::env::var("PPF_B32_CERTIFICATE_MODE").unwrap_or_default();
                let shadow_enabled = mode == "shadow";
                let reuse_enabled = mode == "reuse";
                let reuse_receipt = std::env::var_os("PPF_B32_REUSE_RECEIPT").is_some_and(|v| v == "1");
                let compare = certify_mesh && (shadow_enabled || reuse_enabled)
                    && b32_certificate.is_some_and(b32_shadow::SweepCertificate::covers_endpoint)
                    && toi_rig.to_bits() == 1.0_f32.to_bits();
                let mut check_mesh=|device:&mut D,candidate:ppf_cts_compute::Handle,trial:u32,fraction:f32,same:Option<bool>,compare_ms:f64|->FatalResult<bool>{
                    // Device comparison is restricted to the first complete trial.
                    let eligible = compare && trial == 0 && fraction.to_bits() == 1.0_f32.to_bits()
                        && same == Some(true);
                    if reuse_enabled && eligible {
                        if reuse_receipt { eprintln!("B32_REUSE_RECEIPT trial={} fraction={:.9e} used=true fallback=false same={:?}", trial, fraction, same); }
                        return Ok(true);
                    }
                    if reuse_receipt { eprintln!("B32_REUSE_RECEIPT trial={} fraction={:.9e} used=false fallback=true reuse_enabled={} compare={} same={:?}", trial, fraction, reuse_enabled, compare, same); }
                    let started = std::time::Instant::now();
                    let original = contact.as_mut().ok_or_else(||Fatal::invariant("MESH_ROUNDED_ACCEPT_UNRESOLVED: no contact layer"))?
                        .certify_rounded_candidate(device,data,mesh_refs,&*param,accept.origin,candidate,windows);
                    if shadow_enabled {
                        let original_ms = started.elapsed().as_secs_f64() * 1000.0;
                        let original_pass = original.as_ref().ok().copied();
                        eprintln!("B32_CERTIFICATE_SHADOW trial={} fraction={:.9e} raw_toi={:?} ceiling={:?} same={:?} eligible={} read_error={} original_pass={:?} disagreement={} compare_ms={:.3} original_ms={:.3}",
                            trial, fraction, b32_certificate.map(|c|c.raw_toi),
                            b32_certificate.map(|c|c.ceiling), same, eligible, compare && trial == 0 && fraction.to_bits() == 1.0_f32.to_bits() && same.is_none(), original_pass,
                            eligible && original_pass != Some(true), compare_ms, original_ms);
                    }
                    original
                };
                if reuse_receipt && !certify_mesh { eprintln!("B32_REUSE_RECEIPT used=false fallback=false reason=no_mesh_certification"); }
                let mesh_check:Option<&mut dyn FnMut(&mut D,ppf_cts_compute::Handle,u32,f32,Option<bool>,f64)->FatalResult<bool>>=
                    if certify_mesh {Some(&mut check_mesh)}else{None};
                let acceptance=analytic.as_mut().ok_or_else(||Fatal::invariant("ANALYTIC_ACCEPT_UNRESOLVED: no analytic layer"))?
                    .certified_plane_accept(device,mesh_refs,accept.origin,accept.proposed,vertices,toi_rig,floor,controlled,compare,mesh_check);
                match acceptance {
                    Ok(alpha)=>alpha,
                    Err(error)=>{
                        // ACCEPT FAILURE OBSERVER: BEGIN B32 before original reject
                        capture_acceptance_failure(device,data,state,root_attempt.as_ref(),"B32_acceptance_error_pre_reject",
                            step,final_step,dt,toi_advanced,toi_recale,toi_rig,accept.origin,accept.proposed,pdrd_bodies as usize,vertices,prm.time,
                            (*param).ccd_eps,prm.line_search_max_t,&error.detail);
                        // ACCEPT FAILURE OBSERVER: END B32 before original reject
                        analytic_gap_trace.flush(&error.detail,(*param).ccd_eps,prm.line_search_max_t);
                        if let Some(a)=root_attempt.as_ref(){state.roots.transaction.as_mut().unwrap().reject(a).map_err(|e|Fatal::invariant(e.to_string()))?;}return Err(error);
                    }
                }
            };
            log::mark("advance", "rigidify_toi", f64::from(toi_rig));
            if analytic_gap_trace.enabled() { analytic_gap_trace.capture(device,"B32-certified-actual-before-copy",step,final_step,dt,toi_advanced,toi_recale,toi_rig,accept.origin,accept.proposed,None,state.floor.host(),root_attempt.as_ref()); }
            let SolverState { eval_x, rigid_target, .. } = &mut *state;
            eval_x.copy_from(device, rigid_target)?;
        }

        if sand_offset_trace.enabled {sand_offset_trace.capture(device,"post-B32",step,final_step,dt,toi_advanced,toi_recale,toi,state.target.handle(),state.eval_x.handle(),None);}
        if analytic_gap_trace.enabled() {analytic_gap_trace.capture(device,"post-B32",step,final_step,dt,toi_advanced,toi_recale,toi,state.target.handle(),state.eval_x.handle(),None,state.floor.host(),root_attempt.as_ref());}
        if analytic_gap_trace.enabled() {
            let root_view=state.roots.view(&state.rigid);
            analytic_gap_trace.capture_root_buffers(device,"post-B32-root-workspace",pdrd_bodies as usize,
                state.rigid_state.handle(),state.rigid_rotation_step.handle(),state.rigid_running_rotation.handle(),state.rigid_centroid.handle(),root_view);
            analytic_gap_trace.capture_root_direction(device,state.dx.handle(),state.pcg_rigid.xr.handle(),vertices,state.pcg_rigid.xr.len());
        }

        // The accepted pose changed after the frozen solve. Validate the
        // extension domain/arithmetic before this iterate can exit or commit.
        #[cfg(test)]
        if private_test_fem_injection(){state.joints.soft.test_inject_captured_candidate(device,state.eval_x.handle())?;}
        #[cfg(all(test,feature="cpu"))]
        if private_test_fem_row2_injection(){state.joints.soft.test_inject_row2_captured_scratch(device,state.eval_x.handle())?;}
        if let Some((selected,origin,fitted))=twist_b31 {
            super::joints::fit_candidate(device,data,state)?;
            let final_fit=state.rigid_state.handle();
            let end=super::joints::capture_twist_pose(device,data,state,&selected,dt,super::joints::TwistPoseSource::Iterate,Some(final_fit)).map_err(|error|{
                let _=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||{
                    eprintln!("TWIST_SAMPLE_STEP_CONTEXT source=B32_iterate parameter_time={:.17e} newton_step={} adaptive_retry_token={} dt_bits={:08x}",prm.time,step,state.adaptive_retry_token,dt.to_bits());
                    analytic_gap_trace.flush(&error.detail,(*param).ccd_eps,prm.line_search_max_t);
                }));error
            })?;
            let mut dtheta=vec![0.0_f32;3*state.rigid.n_bodies];
            state.rigid_rotation_step.read(device,0,&mut dtheta)?;
            let fraction=f64::from((toi_recale*toi).abs());
            let mut commanded:Vec<f64>=dtheta.chunks_exact(3).map(|v|{
                let norm=v.iter().map(|x|f64::from(*x).powi(2)).sum::<f64>().sqrt();
                fraction*norm+1e-5
            }).collect();
            let body_selected=state.joints.twist_body_selection(&selected,state.rigid.n_bodies)?;
            let Some(root_extra)=state.roots.twist_root_extra(&body_selected,&fitted,&dtheta,toi_recale*toi) else {
                state.joints.mark_uncertified_path_retry(device,dt,"B32")?;
                return Err(Fatal::invariant("CONTINUOUS_TWIST_PATH_UNCERTIFIED: B32 root override"));
            };
            for (bound,extra) in commanded.iter_mut().zip(root_extra) {*bound+=extra;}
            let view=state.joints.view(final_fit,&state.rigid,dt);
            if !state.joints.advance_segment_pieces(device,view,&selected,&origin,&end,Some(&commanded))? {
                state.joints.mark_uncertified_path_retry(device,dt,"B32")?;
                return Err(Fatal::invariant("CONTINUOUS_TWIST_PATH_UNCERTIFIED: B32"));
            }
            super::joints::check_candidate(device,state,dt,super::joints::B32Complete).map_err(|error|{
                let _=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||eprintln!("FEM_PREPARE_CONTEXT source=B32_candidate parameter_time={:.17e} newton_step={} adaptive_retry_token={} dt_bits={:08x} accepted_commit_clock=unavailable",prm.time,step,state.adaptive_retry_token,dt.to_bits())));error
            })?;
        } else {super::joints::gate_candidate(device,data,state,dt,super::joints::B32Complete).map_err(|error|{
            let _=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||eprintln!("FEM_PREPARE_CONTEXT source=B32_candidate parameter_time={:.17e} newton_step={} adaptive_retry_token={} dt_bits={:08x} accepted_commit_clock=unavailable",prm.time,step,state.adaptive_retry_token,dt.to_bits())));error
        })?;}

        // RECORDED AND NOT PRINTED: the accept and the rigidify commit are not
        // one of the solver's own printed scopes.
        log::mark(
            "advance",
            "nt_tail",
            log::Marked::quiet(_nt_tail.elapsed().as_secs_f64() * 1000.0),
        );
        // STEP B33. Loop control. A continuing iteration REBUILDS the target,
        // which B20 clobbered; without it the next iteration would integrate
        // against the previous iterate's positions.
        if !result.success() {break;}
        if final_step {
            if let Some(attempt)=root_attempt.as_ref().filter(|a|!a.body_indices.is_empty()) {
                // Probe native member evidence; never call a committing function.
                // API/nonfinite/improper failures remain hard refusals.
                match super::roots::certify_candidate(device,data,state,attempt)? {
                    super::roots::CandidateCertification::Ready{..}=>{certified_final_fit_step=Some(step);},
                    super::roots::CandidateCertification::NeedsRefinement(error)=>{
                        if prm.max_newton_steps==0 {
                            // No new unbounded final loop when the old budget is disabled.
                            return Err(error);
                        }
                        step+=1;
                        continue;
                    },
                }
            }
            break;
        }
        if toi_advanced >= f64::from(prm.target_toi) && step >= prm.min_newton_steps
            && root_boundary_ready(preserve_root_boundary,full_remaining_sweep_accepted,step,prm.max_newton_steps) {
            final_step = true;
        } else {
            step += 1;
            compute_target(
                device,
                &prm,
                curr,
                prev,
                vertices,
                state.target.handle(),
                state.fix.handle(),
                dt,
                state.fix_index.handle(),
                state.field.acceleration(),
            )?;
            state.roots.target(device,state.rigid_staged.vert_list.handle(),state.rigid_staged.rest_centered.handle(),state.target.handle(),data.pdrd_vert_list.size,vertices as u32)?;
        }
    }

    // Optional storage is also retired when the selected pass completes normally.
    let _=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||analytic_assembly_ledger.cleanup(device)));
    let _=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||frozen_newton_witness.cleanup(device)));
    if result.success() {
        let mut reverse_pin_face_edges = Vec::new();
        // STEP C1 and C2. The final AABB refresh and `check_intersection`: the
        // SECOND of the two things that make this solver penetration-free. The
        // ACCD filter refuses a step that would cross; this reports a crossing
        // that got through anyway, at the pose the step is about to commit.
        //
        // Reporting is GATED ON "DYNAMIC" and stays gated: an intersection
        // between two fully prescribed elements cannot be resolved by either
        // side yielding, so reporting it would only abort a run over geometry
        // the solver was never going to fix. The three intersection allowances
        // reach `intersection.rs`'s one predicate, the same one that takes
        // their pairs out of contact and the line search.
        if !prm.disable_contact {
            let SolverState {
                contact: contact_state,
                eval_x,
                positions: last_accepted,
                prop_vertex,
                fix,
                pull,
                ..
            } = &mut *state;
            let pose = eval_x.handle();
            let Some(contact) = contact_state.as_mut() else {
                return Err(Fatal::invariant(
                    "solver driver: the intersection gate was reached with no contact subsystem \
                     allocated, so a penetrating pose would be committed unreported",
                ));
            };
            let _phase = super::phase::start("contact.refresh_leaves");
            let _aabb = std::time::Instant::now();
            contact.refresh_leaves(device, data, mesh_refs, pose, pose, 1.0, windows)?;
            drop(_phase);
            // THE SECOND SITE OF THE SAME CHANNEL. This scope closes before
            // the intersection gate opens, so `> aabb_update` prints above
            // `> check_intersection`. One channel, two sites per step.
            log::mark("advance", "aabb_update", _aabb.elapsed());
            // The scan walks the committed pose on the host, so it downloads
            // here. `pose` was taken above and staled the mirror, which is why
            // this is not optional.
            //
            // AND THE SCAN IS WHERE THE DIVERGENCE IS. The gate should run
            // three traversals per edge in ONE dispatch and move a 4-byte
            // counter, at most `capacity` records and two flag arrays, which is
            // the shape `contact/intersect_geometry.entry.cpp` already
            // declares. `intersection.rs` instead walks a materialized pair
            // list serially on the host, and that is a recorded divergence: a
            // pass whose work is proportional to the mesh belongs on the
            // device. The CCD half of the same row closed with
            // `contact/ccd_sweep.kernel.cpp`, which is the shape this one takes
            // next; nothing blocks it. This download is the transport that host
            // walk needs and comes out with it.
            // THE FINAL PENETRATION GATE, timed as `check_intersection`. The
            // download above is inside the scope because it is part of what the
            // gate costs today.
            let _gate = std::time::Instant::now();
            eval_x.download(device)?;
            let mut report =
                contact.check_intersection(device, data, mesh_refs, pose, last_accepted.handle(), eval_x.host(), prop_vertex.host(), windows, !prm.disable_pin_dof_removal)?;
            if !prm.disable_pin_dof_removal {
                reverse_pin_face_edges = super::reverse_pin_contacts::filter_trial(
                    device, data, last_accepted, eval_x.host(), prop_vertex.host(), fix,
                    pull.host(), &contact.reverse_pin_face_edges, &mut report,
                )?;
                if !reverse_pin_face_edges.is_empty() {
                    log::message!(
                        "REVERSE_PIN_FE_SCOPED: start={:.17e} scoped_pairs={} rejected_pairs={}",
                        prm.time, reverse_pin_face_edges.len(), report.found
                    );
                }
            }
            if report.scoped_soft_static_fe > 0 {
                log::message!(
                    "NARROW_FE_SCOPED: start={:.17e} scoped_pairs={} rejected_pairs={}",
                    prm.time, report.scoped_soft_static_fe, report.found
                );
            }
            if report.is_clean() {
                // THE CLEAN PATH IS `debug!`, SO THE TRANSCRIPT ONLY SPEAKS
                // WHEN THE GATE FINDS SOMETHING. A line per committed pose puts
                // one sentence per step into a stream that otherwise carries
                // marks and timings, and on a real scene that is most of what a
                // reader sees go by.
                //
                // IT IS AT `debug!` RATHER THAN ABSENT, because the argument for
                // having it is still good: this is the final penetration gate,
                // a gate nothing can observe is not a gate, and a run that died
                // in the first line search would otherwise report zero
                // intersections while proving nothing.
                // A PDRD cloth contact scene asserts the gate RAN and
                // sets `RUST_LOG=debug` to see it.
                ::log::debug!("check_intersection: the committed pose carries no \
                              intersecting pair");
            } else {
                // THE LINE BELONGS TO THE SOLVER'S OWN TRANSCRIPT, so it is
                // bare. The count is this driver's addition.
                log::message!(
                    "### intersection detected: {} intersecting pair(s) at the candidate pose",
                    report.found
                );
                result.intersection_free = false;
            }
            // THE ROW IS RECORDED AFTER THE REPORT: `### intersection
            // detected` prints inside the scope and the timing line follows the
            // finding rather than preceding it. No docstring here, so the
            // channel is absent from `names()` and only the printed line
            // exists; the INITIALIZE-time scan is the one that carries a
            // docstring, under `initial_check_intersection`.
            log::mark("advance", "check_intersection", _gate.elapsed());
            super::publish_intersection_records(&report);
        }

        // C2 may turn a previously successful trial into a rejected one.  Do
        // not certify or commit that pose, regardless of root/FEM presence.
        if !result.success() {
            let rejected = if let Some(attempt) = root_attempt.as_ref() {
                match state.roots.transaction.as_mut() {
                    Some(transaction) => transaction.reject(attempt).map_err(|e| e.to_string()),
                    None => Err("missing root transaction".to_string()),
                }
            } else { Ok(()) };
            state.fem_roots.abort();
            if let Err(error) = rejected {
                return Err(Fatal::invariant(format!(
                    "final intersection rejected; root transaction rejection failed: {error}"
                )));
            }
            return Ok(result);
        }

        // Name: Advanced Fractional Step Size
        // Format: list[(time, ratio)]
        // Description:
        // Fraction in (0, 1] of the target step size that the Newton loop
        // actually advanced, accumulated across all its iterations. The
        // final Final Step Size equals this fraction times the target dt.
        // A value of 1.0 means the full target step completed, smaller
        // values mean contacts or strain limits forced a partial step.
        log::mark("advance", "toi_advanced", toi_advanced);
        // Name: Newton Iteration Count
        // Format: list[(time, iterations)]
        // Description:
        // Number of Newton iterations consumed in this simulation step
        // (before the trailing error-reduction iteration). Values above
        // the configured min_newton_steps indicate the solver needed
        // extra iterations to reach the target advanced step size.
        log::mark("advance", "newton_steps", step as f64);
        // Name: Final Step Size
        // Format: list[(time, seconds)]
        // Description:
        // Step size in seconds that was actually integrated this
        // simulation step. In easy cases this matches the target dt, but
        // it is reduced by the advanced TOI fraction when contacts or
        // strain limits shorten the step, and can also be reduced when
        // enable_retry is on and the PCG solve fails.
        log::mark("advance", "final_dt", dt as f64);

        // Certify the actual nonlinear, post-line-search/post-rigidify pose
        // BEFORE any clock or accepted-position write. Failure is a typed fatal;
        // no snapping or unreviewed hard retry is inserted after CCD.
        if state.hard.active() {
            let twist_context=state.joints.twist_context();
        let SolverState {hard,rigid,rigid_staged,prop_vertex,eval_x,..}=&mut *state;
            let mut reduction=super::pdrd::Reduction {map:rigid,staged:rigid_staged};
            hard.gate_candidate(device,&mut reduction,prop_vertex.handle(),eval_x.handle(),
                data.pdrd_vert_list.size,false,twist_context)?;
        }

        // Prepare the exact F candidate before any root, clock or position commit.
        if state.joints.material_spin_active() || state.joints.soft.material_spin_active() {
            // Only the ready certificate in this final iteration creates the
            // witness. C1/C2 reads eval_x and writes private contact buffers.
            // Hard constraints and debug observers conservatively use original.
            let certified_same_iteration = certified_final_fit_step.take() == Some(step);
            let can_reuse = final_fit_reuse && certified_same_iteration
                && result.success() && !state.hard.active()
                && !analytic_gap_trace.enabled() && !sand_offset_trace.enabled;
            let started = final_fit_receipt.then(std::time::Instant::now);
            if !can_reuse {
                super::joints::fit_candidate(device,data,state)?;
            }
            if let Some(started) = started {
                eprintln!("PPF_REUSE_FINAL_FIT_RECEIPT route={} saved_fit_count={} final_step={} root_certified={} hard={} contact_success={} diagnostic={} elapsed_ms={:.6}",
                    if can_reuse {"reuse"} else {"original"}, u8::from(can_reuse), step,
                    certified_same_iteration, state.hard.active(), result.success(),
                    analytic_gap_trace.enabled() || sand_offset_trace.enabled,
                    started.elapsed().as_secs_f64()*1000.0);
            }
        }
        if state.joints.material_spin_active() {
            let view=state.joints.view(state.rigid_state.handle(),&state.rigid,dt);
            if let Some(code)=state.joints.sample_material_spin_candidate(device,view)? {
                return Err(Fatal::invariant(format!("MATERIAL_SPIN_CANDIDATE_FAULT: code={code}")));
            }
        }
        if state.joints.soft.material_spin_active() {
            if let Some(code)=state.joints.soft.sample_material_spin_candidate(device,state.eval_x.handle(),state.rigid_state.handle(),&state.rigid,dt)? {
                if code==8 {state.joints.mark_material_spin_retry(dt);}
                return Err(Fatal::invariant(format!("FEM_MATERIAL_SPIN_CANDIDATE_FAULT: code={code}")));
            }
        }

        // Compute and validate the same scene-clock duration before writing
        // the accepted clock or previous integration step.
        let (accepted_end, accepted_scene_dt) = if kinematics_shortened.is_some() {
            (root_plan.end_s, (root_plan.end_s - root_plan.start_s) as f32)
        } else if let Some(attempt) = root_attempt.as_ref() {
            (attempt.end_s, (attempt.end_s - attempt.start_s) as f32)
        } else if let Some(attempt) = state.fem_roots.attempt.as_ref() {
            (attempt.end, (attempt.end - attempt.start) as f32)
        } else {
            // Preserve the existing scene-clock increment and avoid cancellation
            // from subtracting two large absolute times.
            let elapsed = dt / prm.playback;
            ((*param).time + f64::from(elapsed), elapsed)
        };
        if state.plastic.any() && (!accepted_scene_dt.is_finite() || accepted_scene_dt <= 0.0
            || !accepted_end.is_finite() || accepted_end <= (*param).time) {
            if let Some(transaction) = state.roots.transaction.as_mut() { transaction.abort_trial(); }
            state.fem_roots.abort();
            return Err(Fatal::invariant("plasticity: invalid accepted scene duration"));
        }
        if let Some(attempt)=root_attempt.as_ref(){
            if !super::roots::curve_gate(device,data,state,attempt,mesh_refs,windows,param)? {
                state.roots.transaction.as_mut().unwrap().reject(attempt).map_err(|e|Fatal::invariant(e.to_string()))?;
                state.roots.request_curve_retry(dt);result.ccd_success=false;return Ok(result);
            }
            if analytic_gap_trace.enabled() {
                analytic_gap_trace.capture(device,"root-pre-commit-gate",step,final_step,dt,toi_advanced,1.0,last_toi,state.target.handle(),state.eval_x.handle(),None,state.floor.host(),Some(attempt));
                let root_view=state.roots.view(&state.rigid);
                analytic_gap_trace.capture_root_buffers(device,"root-pre-commit-workspace",pdrd_bodies as usize,
                    state.rigid_state.handle(),state.rigid_rotation_step.handle(),state.rigid_running_rotation.handle(),state.rigid_centroid.handle(),root_view);
            }
            if let Err(error)=super::roots::commit_candidate(device,data,state,attempt) {
                if analytic_gap_trace.enabled() {
                    analytic_gap_trace.capture(device,"root-commit-failure-read",step,final_step,dt,toi_advanced,1.0,last_toi,state.target.handle(),state.eval_x.handle(),None,state.floor.host(),Some(attempt));
                    let root_view=state.roots.view(&state.rigid);
                    analytic_gap_trace.capture_root_buffers(device,"root-failed-commit-refit-workspace",pdrd_bodies as usize,
                        state.rigid_state.handle(),state.rigid_rotation_step.handle(),state.rigid_running_rotation.handle(),state.rigid_centroid.handle(),root_view);
                    analytic_gap_trace.flush(&error.detail,(*param).ccd_eps,prm.line_search_max_t);
                }
                return Err(error);
            }
        }
        let _commit = std::time::Instant::now();
        // STEP C3. Commit the clock. `prev_dt` is what the NEXT step's velocity
        // reconstruction divides by, and `backend.rs` reads it back into the
        // checkpoint, so both fields are written through to the host's record.
        (*param).prev_dt = dt;
        (*param).time = accepted_end;
        (*param).time_f32 = (*param).time as f32;
        if state.joints.curves.active() {
            state.restoring_commit_count=state.restoring_commit_count.checked_add(1).ok_or_else(||Fatal::invariant("restoring real commit count overflow"))?;
            state.restoring_commit_time=Some((*param).time);
        }

        #[cfg(test)]
        if private_test_commit_stage(1) {return Err(Fatal::invariant("PRIVATE_TEST_C4"));}

        // STEP C4. The translation-lock invariant check, on the pose about to
        // be committed.
        //
        // READ-ONLY BY DESIGN. It accumulates each group's mass-weighted
        // perpendicular center-of-mass drift and the largest per-axis
        // displacement in that group, then reaches a host verdict against a
        // round-off bound. It never snaps a position: a correction applied
        // after the CCD line search would move a vertex the line search had
        // already certified, which is how a penetration gets reintroduced. A
        // group that has drifted off its commanded line is reported instead.
        if locked_groups > 0 {
            let rows = super::lock::RowInputs {
                lock_index: state.translation_lock_index.handle(),
                locks: state.translation_lock.handle(),
                prop: state.prop_vertex.handle(),
                positions: state.eval_x.handle(),
                dof_mask: state.dof_mask.handle(),
                initial: state.translation_lock_initial.handle(),
            };
            let iterate = state.eval_x.handle();
            super::lock::check_invariant(
                device,
                rows,
                iterate,
                locked_records,
                &mut state.lock_drift,
                &mut state.lock_max_displacement,
                vertices,
                "completed Newton step",
            )?;
        }

        // STEP C5. Commit the positions: the current pose becomes the previous
        // one and the iterate becomes current.
        // ON THE DEVICE, in that order. `fetch()` is what puts these in the
        // arrays `backend.rs` reads.
        let twist_commit=if state.joints.twist_selection().is_some(){Some(state.joints.prepare_twist_commit((*param).time)?)}else{None};
        let material_commit=if state.joints.material_spin_active(){Some(state.joints.prepare_material_spin_commit((*param).time)?)}else{None};
        let fem_material_commit=if state.joints.soft.material_spin_active(){Some(state.joints.soft.prepare_material_spin_commit((*param).time)?)}else{None};
        state.positions_prev.copy_from(device, &state.positions)?;
        state.positions.copy_from(device, &state.eval_x)?;
        if let Some(contact) = &mut state.contact {
            contact.reverse_pin_face_edges = reverse_pin_face_edges;
        }
        if let Some(accepted)=twist_commit {state.joints.finish_twist_commit(accepted);}
        if let Some(accepted)=material_commit {state.joints.finish_material_spin_commit(accepted);}
        if let Some(accepted)=fem_material_commit {state.joints.soft.finish_material_spin_commit(accepted);}
        *c5_committed=true;
        #[cfg(test)]
        if private_test_commit_stage(2) {return Err(Fatal::invariant("PRIVATE_TEST_POST_C5"));}

        // STEP C6. THE SAND SPIN INTEGRATE, on the pose just committed, which
        // is where the committed pose makes it meaningful. It consumes the CONVERGED friction
        // torque the contact fold summed over the final Newton iteration's
        // simultaneous contacts, and the angular stiffness beside it is what
        // makes omega approach the rolling rate rather than overshoot it.
        //
        // A GRAIN-GRAIN PAIR IS A STAGGERED, POST-SOLVE UPDATE, unlike the
        // analytic contacts, whose spin is condensed into the solve itself.
        //
        // THE TWO TRAILING ARGUMENTS DO NOT AGREE, THOUGH BOTH LOOK LIKE
        // TUNING. `c_roll` is a literal zero below and is the optional linear
        // resistance in the implicit denominator; `roll_resist` is the
        // anti-pump under-roll cap and defaults to 0.05. Passing zero for both
        // leaves friction able to do net positive work on every grain-grain
        // contact in every SAND scene.
        if grains > 0 && !sand_no_roll() {
            // THE BLOCKS ARE THE STATE'S, so this reads a real per-vertex array
            // whether or not the scene has an analytic collider. Asking the
            // collider and substituting `Handle::NONE` when there is none does
            // not work: a generated entry resolves the handle BEFORE the body
            // runs, and the body reads `angular[i]` unconditionally, so a grain
            // scene with no floor and no sphere would have nothing valid to
            // read.
            let angular = state.grain_angular.handle();
            // THE STATE'S OWN ACCUMULATORS, NOT THE CONTACT LAYER'S. Taking
            // them from that layer would couple this to something
            // `disable-contact` does not build, while the analytic colliders
            // are assembled outside it, so the grain would have spin to
            // integrate and nowhere to accumulate it.
            let torque = state.grain_torque.handle();
            let stiffness = state.grain_stiffness.handle();
            let normal = state.grain_normal.handle();
            let integrate = SandGrainIntegrateRowArgs {
                inverse_rolling_inertia: state.grain_inv_inertia.handle(),
                angular,
                prop: state.prop_vertex.handle(),
                params: state.param_vertex.handle(),
                curr: state.positions.handle(),
                prev: state.positions_prev.handle(),
                torque,
                normal_sum: normal,
                angular_stiffness: stiffness,
                omega: state.grain_omega.handle(),
                dt,
                c_roll: 0.0,
                roll_resist: sand_roll_resist(),
                count: vertices as u32,
                seam_arena_count: 0,
            };
            device.launch("sand.integrate", &integrate, vertices as u32)?;
            // THE NEXT STEP'S CONDENSE READS THIS STEP'S SPIN, so the snapshot
            // is taken here rather than at the top of the next step: `omega` is
            // the one piece of SAND state that carries across steps.
            state.grain_omega_prev.copy_from(device, &state.grain_omega)?;
        }

        // STEP C7 and C8. The four plastic creeps, on the pose just committed
        // and in a fixed order: shell faces, tets, hinges, then interior rod
        // vertices.
        // Creep uses the accepted scene-clock duration, including any
        // shortening, without altering the integration duration or prev_dt.
        super::plasticity::creep(device, data, state, accepted_scene_dt)?;
        state.joints.curves.check(device,"restoring accepted commit")?;
        super::restoring_receipt::observe(device,&mut state.joints.curves,(*param).time)?;

        super::fem_roots::accept(device,state,&super::scene::SceneView::new(data),(*param).time,(*param).prev_dt)?;
        result.time = (*param).time;
        // RECORDED AND NOT PRINTED: the commit is not one of the solver's own
        // printed scopes.
        log::mark(
            "advance",
            "commit",
            log::Marked::quiet(_commit.elapsed().as_secs_f64() * 1000.0),
        );
    }
    if !result.success(){if let Some(attempt)=root_attempt.as_ref(){state.roots.transaction.as_mut().unwrap().reject(attempt).map_err(|e| Fatal::invariant(e.to_string()))?;}}
    Ok(result)
}

/// Every scene class the driver does not implement, checked before it steps.
///
/// A SECOND CALL TO THE SAME GATE, DELIBERATELY, and it is one function rather
/// than two lists on purpose. `super::refusal::scene_refusals` runs once at
/// `initialize()` against the scene as built; this runs it again every step
/// against the scene as it now is, which is where a per-step upload can
/// introduce something the first call never saw: `make_constraint` is rebuilt
/// from the schedule every step, so a collider that switches on at t = 2 is
/// invisible to a gate that only ran at t = 0.
///
/// TWO LISTS WOULD BE THE DEFECT THIS GUARDS AGAINST. If the gate and the
/// driver could disagree, one of the two orders is a scene refused that the
/// driver could step (loud and merely annoying) and the other is a scene
/// stepped that the gate meant to refuse, which writes plausible frames with
/// nothing in the output to say so. Sharing the function makes the second
/// impossible rather than unlikely.
///
/// # Safety
/// `param` must address a live `ParamSet`.
unsafe fn assert_unsupported_absent(data: &DataSet, param: *const ParamSet) -> FatalResult<()> {
    let refusals = super::refusal::scene_refusals(data, &*param);
    if refusals.is_empty() {
        return Ok(());
    }
    let named: Vec<String> = refusals.iter().map(super::refusal::Refusal::describe).collect();
    Err(Fatal::invariant(format!(
        "solver driver: advance() reached a scene the capability gate refuses: {}. Either \
         initialize() accepted it, which would mean the driver steps a scene it cannot solve, \
         or the scene gained the feature partway through the run. The step stops either way",
        named.join("; ")
    )))
}
/// The implicit target every vertex is solved toward, for one step size.
///
/// A free function rather than a closure over the step's locals, because it now
/// takes the device and a closure holding a mutable borrow of it could not be
/// called from inside the Newton loop where the device is also used.
///
/// # Safety
/// `current` and `previous` must each address `3 * vertices` position
/// components that outlive the dispatch.
#[allow(clippy::too_many_arguments)]
unsafe fn compute_target<D: Device>(
    device: &mut D,
    prm: &StepParams,
    current: Handle,
    previous: Handle,
    vertices: usize,
    target: Handle,
    fix: Handle,
    dt: f32,
    fix_index: Handle,
    external: Handle,
) -> Result<(), Fault> {
    let args = ComputeTargetSeedArgs {
        current,
        previous,
        fix_index,
        fix,
        dt,
        previous_dt: prm.prev_dt,
        // Gravity rides the record as three scalars rather than as a sixth
        // buffer. It is one scene-wide vector every thread reads identically,
        // and it lives on this stack frame beside `dt`, so there is no
        // allocation for a handle to name and never will be.
        gravity_x: prm.gravity[0],
        gravity_y: prm.gravity[1],
        gravity_z: prm.gravity[2],
        // THE FIELD'S ACCELERATION RIDES BESIDE GRAVITY, per vertex. It is
        // this step's evaluation at the step's starting position, so every
        // rebuild of the target within the step reads the same values.
        external,
        inactive_momentum: i32::from(prm.inactive_momentum),
        target,
        count: vertices as u32,
        seam_arena_count: 0,
    };
    device.launch("step.compute_target", &args, vertices as u32)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn authored_root_boundary_leaves_the_last_iteration_for_shorter_recertification() {
        assert!(!super::root_boundary_ready(true, false, 2, 4));
        assert!(super::root_boundary_ready(true, false, 3, 4));
        assert!(!super::root_final_budget_exhausted(3, 4, true, true));
        assert!(super::root_boundary_ready(true, true, 2, 4));
        assert!(super::root_boundary_ready(false, false, 2, 4));
    }

    #[test]
    fn the_step_delay_defaults_to_zero_and_parses_milliseconds() {
        use std::time::Duration;
        // ZERO IS THE DEFAULT AND THE ABSENT CASE, which is what keeps the hook
        // off for every run that did not ask for it. The deleted stub defaulted
        // to 1000 ms and made seven scenarios observable by accident; this one
        // makes nobody slower until asked.
        assert_eq!(super::parse_step_delay(None), Duration::ZERO);
        assert_eq!(super::parse_step_delay(Some("0")), Duration::ZERO);
        assert_eq!(
            super::parse_step_delay(Some("200")),
            Duration::from_millis(200)
        );
        // Whitespace survives a shell that quotes awkwardly.
        assert_eq!(
            super::parse_step_delay(Some(" 600 ")),
            Duration::from_millis(600)
        );
        // Unparsable is zero, not a panic: see `parse_step_delay`.
        assert_eq!(super::parse_step_delay(Some("fast")), Duration::ZERO);
        assert_eq!(super::parse_step_delay(Some("-5")), Duration::ZERO);
        assert_eq!(super::parse_step_delay(Some("")), Duration::ZERO);
    }

    use super::*;
    use crate::driver::launch::host_device;
    use crate::driver::state::SolverState;
    use crate::driver::test_scene::TestScene;

    /// THE PIN INDEX GATHER.
    ///
    /// This module exists because nothing else covers the dispatch that fills
    /// `state.fix_index`: zeroing that kernel's only write leaves every one of
    /// the 454 tests outside this module passing. A per-element write that no
    /// assertion reads is the same gap the rod bending slot key slipped
    /// through.
    /// THE DOF MASK AND ITS FOLD, both on the device.
    ///
    /// The mask alone would be half a fix: a build-on-device that downloaded
    /// the whole mask to count it would keep the reduction on the host while
    /// moving the walk.
    #[test]
    fn the_dof_mask_and_its_sum_are_both_computed_on_the_device() {
        let mut scene = TestScene::new(5);
        {
            let props = scene.data.prop.vertex.as_mut_slice();
            // THREE SHAPES THE MASK MUST TELL APART: unpinned, pinned and free,
            // and pinned INSIDE a PDRD body. The last owns no per-vertex degree
            // of freedom, so it is not eliminated and its anchor keeps the
            // barrier; a mask that ignored `pdrd_body_index` would count it.
            props[0].fix_index = 0;
            props[1].fix_index = 2;
            props[2].fix_index = 5;
            props[3].fix_index = 9;
            props[3].pdrd_body_index = 1;
            props[4].fix_index = 0;
        }
        let mut state = SolverState::default();
        let mut device = host_device();
        // Safety: the scene lives in its box for the whole test.
        unsafe { state.allocate(&mut device, &scene.data) }
            .expect("the fixture scene allocates");
        crate::driver::state::reseed_props(&mut device, &mut state, &scene.data);

        let count = state.sizes.vertices as u32;
        let args = crate::driver::kernels::VertexDofRemovalMaskArgs {
            prop: state.prop_vertex.handle(),
            disable: 0,
            mask: state.dof_mask.handle(),
            count,
            seam_arena_count: 0,
        };
        // Safety: both arrays are borrowed for the whole call.
        unsafe { device.launch("test.dof_mask", &args, count) }
            .expect("the mask dispatches");

        // ONE LEVEL IS ENOUGH AT THIS SIZE, which is the fold's own tail case.
        let fold = crate::driver::kernels::VecBlockSumU32Args {
            source: state.dof_mask.handle(),
            length: count,
            width: crate::driver::state::DOF_FOLD_WIDTH as u32,
            total: state.dof_total.handle(),
            count: 1,
            seam_arena_count: 0,
        };
        // Safety: as above.
        unsafe { device.launch("test.dof_fold", &fold, 1) }.expect("the fold dispatches");
        state.dof_total.download(&mut device).expect("the total reads back");
        assert_eq!(
            state.dof_total.host()[0],
            2,
            "two vertices are pinned and free of a PDRD body; the third pin is \
             inside one and owns no degree of freedom to remove"
        );
    }

    #[test]
    fn the_pin_index_gather_copies_every_vertex_record() {
        let mut scene = TestScene::new(4);
        {
            let props = scene.data.prop.vertex.as_mut_slice();
            // A MIX RATHER THAN ALL-PINNED: a gather that wrote a constant, or
            // that wrote the right value at the wrong slot, has to be told
            // apart from one that copies each record.
            props[0].fix_index = 0;
            props[1].fix_index = 3;
            props[2].fix_index = 0;
            props[3].fix_index = 7;
        }
        let mut state = SolverState::default();
        let mut device = host_device();
        // Safety: the scene lives in its box for the whole test.
        unsafe { state.allocate(&mut device, &scene.data) }
            .expect("the fixture scene allocates");
        crate::driver::state::reseed_props(&mut device, &mut state, &scene.data);

        let count = state.sizes.vertices as u32;
        let args = crate::driver::kernels::VertexFixIndexFromRecordsArgs {
            prop: state.prop_vertex.handle(),
            fix_index: state.fix_index.handle(),
            count,
            seam_arena_count: 0,
        };
        // Safety: both arrays are borrowed for the whole call.
        unsafe { device.launch("test.fix_index", &args, count) }
            .expect("the gather dispatches");
        state
            .fix_index
            .download(&mut device)
            .expect("the gather reads back");
        assert_eq!(
            &state.fix_index.host()[..4],
            &[0u32, 3, 0, 7][..],
            "the gather copies each vertex's own pin index"
        );
    }
}

// Opt-in, single last-bound-iteration GPU stored-deposit ledger. No kernel/ABI.
// D2D snapshots can affect scheduling. No normal-path D2H or operator probe.
struct AnalyticAssemblyLedger {
    enabled: bool,
    storage: ppf_cts_compute::Buffer<f32>,
    tag: Option<serde_json::Value>,
    pre_complete: bool,
    post_complete: bool,
    solve_tag: Option<serde_json::Value>,
    error: Option<String>,
}
impl AnalyticAssemblyLedger {
    fn new() -> Self {
        Self { enabled: std::env::var_os("PPF_ANALYTIC_STALL_TERMINAL_CAPTURE_DIR").is_some_and(|v| !v.is_empty()),
            storage: Default::default(), tag: None, pre_complete: false,
            post_complete: false, solve_tag: None, error: None }
    }
    fn selected(&self, step:u32, bound:u32) -> bool {
        self.enabled && bound>0 && step==bound-1
    }
    fn before(&mut self, device:&mut impl Device, state:&mut SolverState, step:u32,
        final_step:bool, dt:f32, attempt:Option<&crate::joint_root_motion_state::Attempt>) {
        self.pre_complete=false; self.post_complete=false; self.solve_tag=None; self.error=None;
        let eval=state.eval_x.handle(); let force=state.force.handle(); let diagonal=state.diagonal.handle();
        let prop_handle=state.prop_vertex.handle(); let param_handle=state.param_vertex.handle();
        self.tag=Some(serde_json::json!({"step":step,"final":final_step,"integration_dt_bits":dt.to_bits(),
            "vertex":13,"label":"B15a_before_analytic","vertices":state.sizes.vertices,
            "property_source":"actual immutable prop_vertex/param_vertex read at B0; same handles within attempt",
            "source_handles":{"eval_x":format!("{:?}",eval),"force":format!("{:?}",force),
                "C_diagonal":format!("{:?}",diagonal),"prop_vertex":format!("{:?}",prop_handle),
                "param_vertex":format!("{:?}",param_handle)},
            "attempt":attempt.map(|a|serde_json::json!({"sequence":a.sequence(),"start_bits":a.start_s.to_bits(),
                "end_bits":a.end_s.to_bits(),"integration_dt_bits":a.integration_dt_s.map(f32::to_bits),"bodies":a.body_indices})),
            "runtime_floor_uploaded_host_bits":state.floor.host().iter().map(|f|serde_json::json!({
                "ground_bits":std::array::from_fn::<_,3,_>(|k|f.ground[k].to_bits()),"up_bits":std::array::from_fn::<_,3,_>(|k|f.up[k].to_bits()),"ghat_bits":f.ghat.to_bits(),
                "friction_bits":f.friction.to_bits(),"thickness_bits":f.thickness.to_bits(),"kinematic":f.kinematic})).collect::<Vec<_>>() }));
        // The qualified scene has 16 vertices. Explicit lengths protect every slice copy.
        if state.eval_x.len()!=48 || state.force.len()!=48 || state.diagonal.len()!=144 {
            self.error=Some("source shape unavailable".into()); return;
        }
        let result=(|| -> Result<(),Fault> {
            self.storage.size(device,27,ppf_cts_compute::AllocLabel("analytic.terminal.assembly"))?;
            device.copy(self.storage.handle(),0,eval,39*4,3*4)?;
            device.copy(self.storage.handle(),3*4,force,39*4,3*4)?;
            device.copy(self.storage.handle(),6*4,diagonal,117*4,9*4)
        })();
        match result {Ok(())=>self.pre_complete=true,Err(e)=>self.error=Some(e.to_string())}
    }
    fn after(&mut self, device:&mut impl Device, force:Handle, diagonal:Handle) {
        if !self.pre_complete {return;}
        let result=(|| -> Result<(),Fault> {
            device.copy(self.storage.handle(),15*4,force,39*4,3*4)?;
            device.copy(self.storage.handle(),18*4,diagonal,117*4,9*4)
        })();
        match result {Ok(())=>self.post_complete=true,Err(e)=>self.error=Some(e.to_string())}
    }
    fn solve_finished(&mut self, step:u32, final_step:bool, dt:f32, report:&pcg::Report) {
        if !self.post_complete || self.tag.as_ref().and_then(|x|x["step"].as_u64())!=Some(u64::from(step)) {return;}
        self.solve_tag=Some(serde_json::json!({"step":step,"final":final_step,"integration_dt_bits":dt.to_bits(),
            "label":"B17_solve_exit","iterations":report.iterations,"outcome":format!("{:?}",report.outcome),
            "relative_residual_bits":report.relative_residual.to_bits(),
            "retained_writer_contract":"no hard/aggregate lock; B31/B32/B33 do not write force/C/precond or reduced f/xr/r; no new solve or probe"}));
    }
    fn read_at<T:ppf_cts_compute::Pod+Default>(device:&mut impl Device, handle:Handle, first:usize, count:usize) -> Option<Vec<T>> {
        let width=std::mem::size_of::<T>(); let offset=first.checked_mul(width)?; let bytes=count.checked_mul(width)?;
        if bytes>isize::MAX as usize {return None;}
        let mut values=Vec::new(); values.try_reserve_exact(count).ok()?; values.resize_with(count,T::default);
        let dst=unsafe {std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(),bytes)};
        device.read(handle,offset,dst).ok()?; Some(values)
    }
    fn read(&self, device:&mut impl Device, state:&mut SolverState, bound_step:u32) -> serde_json::Value {
        let complete=self.pre_complete && self.post_complete && self.solve_tag.as_ref().and_then(|x|x["step"].as_u64())==Some(u64::from(bound_step.saturating_sub(1)));
        let stored=if complete && self.storage.len()==27 {sand_offset_read::<f32>(device,self.storage.handle(),27)}else{None};
        let bits=|v:Option<Vec<f32>>|v.map(|v|v.into_iter().map(f32::to_bits).collect::<Vec<_>>());
        let slice=|start:usize,count:usize|stored.as_ref().map(|v|v[start..start+count].iter().map(|v|v.to_bits()).collect::<Vec<_>>());
        let retained=complete && state.force.len()==48 && state.diagonal.len()==144 && state.precond_diagonal.len()==144
            && state.pcg_rigid.f.len()==12 && state.pcg_rigid.xr.len()==12 && state.pcg_rigid.r.len()==12 && state.dof_mask.len()==16;
        let force_handle=state.force.handle(); let C_handle=state.diagonal.handle(); let precond_handle=state.precond_diagonal.handle();
        let rhs_handle=state.pcg_rigid.f.handle(); let xr_handle=state.pcg_rigid.xr.handle(); let r_handle=state.pcg_rigid.r.handle();
        let dof_handle=state.dof_mask.handle(); let prop_handle=state.prop_vertex.handle(); let param_handle=state.param_vertex.handle();
        let final_force=if retained {bits(Self::read_at::<f32>(device,force_handle,39,3))}else{None};
        let final_C=if retained {bits(Self::read_at::<f32>(device,C_handle,117,9))}else{None};
        let precond=if retained {bits(Self::read_at::<f32>(device,precond_handle,117,9))}else{None};
        let rhs=if retained {bits(sand_offset_read::<f32>(device,rhs_handle,12))}else{None};
        let xr=if retained {bits(sand_offset_read::<f32>(device,xr_handle,12))}else{None};
        let recursive_r=if retained {bits(sand_offset_read::<f32>(device,r_handle,12))}else{None};
        let mask=if retained {sand_offset_read::<u32>(device,dof_handle,16)}else{None};
        let prop=if retained && state.prop_vertex.len()==16 {Self::read_at::<crate::data::VertexProp>(device,prop_handle,13,1).and_then(|v|v.first().copied())}else{None};
        let param=prop.and_then(|p|if (p.param_index as usize)<state.param_vertex.len() {
            Self::read_at::<crate::data::VertexParam>(device,param_handle,p.param_index as usize,1).and_then(|v|v.first().copied())
        } else {None});
        serde_json::json!({"schema":"analytic-assembly-stored-ledger-v1","history":false,"selected_step_only":true,
            "assembly_tag":self.tag,"solve_tag":self.solve_tag,"copies_complete":complete,"copy_error":self.error,
            "assembly_xyz13_bits":slice(0,3),"pre_force13_bits":slice(3,3),"pre_C13_column_major_bits":slice(6,9),
            "post_force13_bits":slice(15,3),"post_C13_column_major_bits":slice(18,9),
            "final_force13_bits":final_force,"final_C13_column_major_bits":final_C,"precond_diagonal13_column_major_bits":precond,
            "reduced_rhs_bits":rhs,"reduced_xr_bits":xr,"recursive_residual_bits":recursive_r,
            "raw_DOF_mask_u32":mask,"raw_vertex_prop13":prop,
            "raw_vertex_param13_bits":param.map(|p|[p.ghat.to_bits(),p.offset.to_bits(),p.friction.to_bits()]),
            "group_initial_bits":if retained {Some(state.pcg_rigid.group_initial.iter().map(|x|x.to_bits()).collect::<Vec<_>>())}else{None},
            "group_current_bits":if retained {Some(state.pcg_rigid.group_current.iter().map(|x|x.to_bits()).collect::<Vec<_>>())}else{None},
            "note":"GPU stored deposits, not local kernel f/H/gap. No true residual, frozen operator, spectrum or history. Copies may change diagnostic scheduling."})
    }
    fn cleanup(&mut self, device:&mut impl Device) {
        if self.storage.len()>0 {let _=self.storage.free(device);}
    }
}

// PPF_SAND_OFFSET_CAPTURE=1 enables diagnostic readbacks, default[.35,1.0]s.
// Buffer snapshots cover ALL vertices and the last16 phase records. The actual
// assembly overlap chooses the offender AFTER the failure; no guessed pair IDs.
// I/O/readback errors are diagnostic omissions, never physics decisions.
struct SandOffsetPhase {
    label:&'static str,step:u32,final_step:bool,dt:f32,advanced:f64,rescale:f32,fraction:f32,
    x0:Vec<f32>,x1:Vec<f32>,query_toi:Option<Vec<f32>>,
}
struct SandOffsetTrace {enabled:bool,time:f64,vertices:usize,accepted:Option<Vec<f32>>,phases:std::collections::VecDeque<SandOffsetPhase>}
fn sand_offset_read<T:ppf_cts_compute::Pod+Default>(device:&mut impl Device,handle:Handle,count:usize)->Option<Vec<T>> {
    let mut values=Vec::new();values.try_reserve_exact(count).ok()?;values.resize_with(count,T::default);
    let bytes=unsafe {std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(),values.len()*std::mem::size_of::<T>())};
    device.read(handle,0,bytes).ok()?;Some(values)
}
impl SandOffsetTrace {
    fn new(time:f64,vertices:usize)->Self {
        let begin=std::env::var("PPF_SAND_OFFSET_CAPTURE_BEGIN").ok().and_then(|s|s.parse::<f64>().ok()).unwrap_or(0.35);
        let end=std::env::var("PPF_SAND_OFFSET_CAPTURE_END").ok().and_then(|s|s.parse::<f64>().ok()).unwrap_or(1.0);
        Self{enabled:std::env::var_os("PPF_SAND_OFFSET_CAPTURE").is_some_and(|x|x=="1") && time>=begin && time<=end,time,vertices,accepted:None,phases:std::collections::VecDeque::new()}
    }
    #[allow(clippy::too_many_arguments)]
    fn capture(&mut self,device:&mut impl Device,label:&'static str,step:u32,final_step:bool,dt:f32,advanced:f64,rescale:f32,fraction:f32,x0:Handle,x1:Handle,query_toi:Option<Vec<f32>>) {
        if !self.enabled{return;}
        let Some(a)=sand_offset_read::<f32>(device,x0,3*self.vertices)else{return;};
        if label=="accepted-current" {self.accepted=Some(a);return;}
        let Some(b)=sand_offset_read::<f32>(device,x1,3*self.vertices)else{return;};
        if self.phases.len()==16 {self.phases.pop_front();}
        self.phases.push_back(SandOffsetPhase{label,step,final_step,dt,advanced,rescale,fraction,x0:a,x1:b,query_toi});
    }
    fn report(&self,device:&mut impl Device,data:&DataSet,state:&mut SolverState,overlap:&super::ccd::AssemblyOverlap,ccd_eps:f32,max_t:f32) {
        if !self.enabled{return;}
        use std::io::Write;
        let (a,b)=(overlap.elements.0 as usize,overlap.elements.1 as usize);
        if overlap.collision_mesh || a>=self.vertices || b>=self.vertices{return;}
        let Some(props)=sand_offset_read::<crate::data::VertexProp>(device,state.prop_vertex.handle(),self.vertices)else{return;};
        let Some(params)=sand_offset_read::<crate::data::VertexParam>(device,state.param_vertex.handle(),state.param_vertex.len())else{return;};
        let pa=props[a];let pb=props[b];let Some(va)=params.get(pa.param_index as usize)else{return;};let Some(vb)=params.get(pb.param_index as usize)else{return;};
        // Kind 6 is shared by ALL self-contact primitive families. Its two IDs
        // are only the first/last vertices, not enough to identify a PP pair.
        // Retained full snapshots permit lossless reconstruction of every topology
        // candidate consistent with those IDs, without another CUDA entry.
        let mut candidates:Vec<(&str,usize,Vec<usize>)>=vec![("PP",0,vec![a,b])];
        for (i,f) in data.mesh.mesh.face.as_slice().iter().enumerate() {
            if f[2] as usize==b {candidates.push(("PT",i,vec![a,f[0] as usize,f[1] as usize,b]));}
        }
        for (i,e) in data.mesh.mesh.edge.as_slice().iter().enumerate() {
            if e[1] as usize==b {candidates.push(("PE",i,vec![a,e[0] as usize,b]));}
        }
        for (i,e0) in data.mesh.mesh.edge.as_slice().iter().enumerate() {
            if e0[0] as usize!=a {continue;}
            for (j,e1) in data.mesh.mesh.edge.as_slice().iter().enumerate() {
                if e1[1] as usize==b {candidates.push(("EE",i,vec![a,e0[1] as usize,e1[0] as usize,b]));
                    let _=writeln!(std::io::stderr().lock(),"SAND_OFFSET_CAPTURE {}",serde_json::json!({"event":"ee-candidate-edges","edge_a":i,"edge_b":j,"vertices":[a,e0[1] as usize,e1[0] as usize,b]}));}
            }
        }
        candidates.retain(|(_,_,ids)|ids.iter().all(|&v|v<self.vertices));
        let face_props=sand_offset_read::<crate::data::FaceProp>(device,state.prop_face.handle(),state.prop_face.len());
        let face_params=sand_offset_read::<crate::data::FaceParam>(device,state.param_face.handle(),state.param_face.len());
        let edge_props=sand_offset_read::<crate::data::EdgeProp>(device,state.prop_edge.handle(),state.prop_edge.len());
        let edge_params=sand_offset_read::<crate::data::EdgeParam>(device,state.param_edge.handle(),state.param_edge.len());
        let candidate_triple=|x:&[f32],v:usize|[x[3*v],x[3*v+1],x[3*v+2]];
        for (family,primitive,ids) in &candidates {
            let vp:Vec<_>=ids.iter().map(|&v|props[v]).collect();
            let vparams:Vec<_>=ids.iter().map(|&v|params.get(props[v].param_index as usize).copied()).collect();
            let fp=if *family=="PT" {face_props.as_ref().and_then(|x|x.get(*primitive)).copied()}else{None};
            let ep=if *family=="PE" {edge_props.as_ref().and_then(|x|x.get(*primitive)).copied()}else{None};
            let fparam=fp.and_then(|p|face_params.as_ref().and_then(|x|x.get(p.param_index as usize)).copied());
            let eparam=ep.and_then(|p|edge_params.as_ref().and_then(|x|x.get(p.param_index as usize)).copied());
            let _=writeln!(std::io::stderr().lock(),"SAND_OFFSET_CAPTURE {}",serde_json::json!({"event":"primitive-candidate","family":family,"primitive":primitive,"vertices":ids,"vertex_props":vp,"vertex_params":vparams,"face_prop":fp,"face_param":fparam,"edge_prop":ep,"edge_param":eparam,"status":"topology-compatible; not yet proven actual offender"}));
            if let Some(x)=&self.accepted {
                let pts:Vec<_>=ids.iter().map(|&v|candidate_triple(x,v)).collect();let bits:Vec<_>=pts.iter().map(|p|p.map(f32::to_bits)).collect();
                let _=writeln!(std::io::stderr().lock(),"SAND_OFFSET_CAPTURE {}",serde_json::json!({"event":"primitive-phase","family":family,"primitive":primitive,"vertices":ids,"label":"accepted-current","time":self.time,"x_bits":bits}));
            }
            for p in &self.phases {
                let x0:Vec<_>=ids.iter().map(|&v|candidate_triple(&p.x0,v).map(f32::to_bits)).collect();
                let x1:Vec<_>=ids.iter().map(|&v|candidate_triple(&p.x1,v).map(f32::to_bits)).collect();
                let q:Vec<_>=ids.iter().map(|&v|p.query_toi.as_ref().and_then(|x|x.get(v)).copied()).collect();
                let _=writeln!(std::io::stderr().lock(),"SAND_OFFSET_CAPTURE {}",serde_json::json!({"event":"primitive-phase","family":family,"primitive":primitive,"vertices":ids,"label":p.label,"time":self.time,"step":p.step,"final":p.final_step,"dt":p.dt,"advanced":p.advanced,"rescale":p.rescale,"fraction_or_max_t":p.fraction,"x0_bits":x0,"x1_bits":x1,"query_toi":q}));
            }
        }
        let metadata=serde_json::json!({"event":"sand-offset-offender","time":self.time,"elements":[a,b],"kind":overlap.kind,"world_d2":overlap.d2,"world_offset":overlap.offset,"surface_count":data.surface_vert_count,"ccd_eps":ccd_eps,"line_search_max_t":max_t,"pa":pa,"pb":pb,"vparam_a":va,"vparam_b":vb,"material_sum_f32":va.offset+vb.offset,"ghat_average_f32":0.5*(va.ghat+vb.ghat),"ring_records":self.phases.len()});
        let _=writeln!(std::io::stderr().lock(),"SAND_OFFSET_CAPTURE {}",metadata);
        let triple=|x:&[f32],v:usize|[x[3*v],x[3*v+1],x[3*v+2]];
        let bits=|x:[f32;3]|x.map(f32::to_bits);
        if let Some(x)=&self.accepted {let aa=triple(x,a);let bb=triple(x,b);let _=writeln!(std::io::stderr().lock(),"SAND_OFFSET_CAPTURE {}",serde_json::json!({"event":"accepted-current","time":self.time,"a":aa,"b":bb,"a_bits":bits(aa),"b_bits":bits(bb)}));}
        for p in &self.phases {
            let a0=triple(&p.x0,a);let a1=triple(&p.x1,a);let b0=triple(&p.x0,b);let b1=triple(&p.x1,b);
            let d2=|u:[f32;3],v:[f32;3]|{let d=std::array::from_fn::<_,3,_>(|i|u[i]-v[i]);d[0]*d[0]+d[1]*d[1]+d[2]*d[2]};
            let row=serde_json::json!({"event":"phase","label":p.label,"time":self.time,"step":p.step,"final":p.final_step,"dt":p.dt,"advanced":p.advanced,"rescale":p.rescale,"fraction_or_max_t":p.fraction,"a0":a0,"a1":a1,"b0":b0,"b1":b1,"a0_bits":bits(a0),"a1_bits":bits(a1),"b0_bits":bits(b0),"b1_bits":bits(b1),"world_d2_x0_f32":d2(a0,b0),"world_d2_x1_f32":d2(a1,b1),"query_toi_a":p.query_toi.as_ref().and_then(|x|x.get(a)).copied(),"query_toi_b":p.query_toi.as_ref().and_then(|x|x.get(b)).copied()});
            let _=writeln!(std::io::stderr().lock(),"SAND_OFFSET_CAPTURE {}",row);
        }
    }
}


// ACCEPT FAILURE OBSERVER: BEGIN failure-only step context
/// Best-effort observation after acceptance already returned its original Fatal.
/// # Safety
/// Uses advance's live same-device state/data/position handle contracts. Handles
/// cover 3*vertices f32; body POD/RootTarget layouts use the existing checked trace.
/// No refit, projection, transaction update or clock update is performed here.
#[allow(clippy::too_many_arguments)]
unsafe fn capture_acceptance_failure(device:&mut impl Device,data:&DataSet,state:&mut SolverState,
    attempt:Option<&crate::joint_root_motion_state::Attempt>,label:&'static str,step:u32,final_step:bool,
    dt:f32,advanced:f64,rescale:f32,fraction:f32,origin:Handle,proposal:Handle,bodies:usize,vertices:usize,
    time:f64,ccd_eps:f32,max_t:f32,reason:&str) {
    // Env lookup only before failure; no normal-path observer allocation/read/copy.
    if !std::env::var_os("PPF_ANALYTIC_STALL_TERMINAL_CAPTURE_DIR")
        .is_some_and(|p|std::path::Path::new(&p).is_absolute()) {return;}
    let observed=std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut terminal=AnalyticGapTrace::new_terminal(time,vertices,ccd_eps);
        if !terminal.enabled(){return;}
        let ledger=state.analytic.as_mut().and_then(|a|a.take_accept_failure_ledger());
        terminal.capture_props(device,state.prop_vertex.handle());
        let list=super::scene::slice(&data.pdrd_vert_list);
        let rest=super::scene::slice(&data.pdrd_rest_centered);
        let spans=super::scene::slice(&data.prop.pdrd_body);
        terminal.root_layout=Some(serde_json::json!({"pdrd_vertex_list":list,
            "rest_centered_gpu_source_bits":rest.iter().map(|p|std::array::from_fn::<_,3,_>(|a|p[a].to_bits())).collect::<Vec<_>>(),
            "body_spans":spans.iter().enumerate().map(|(body,p)|serde_json::json!({"body":body,"start":p.vertex_start,"count":p.vertex_count})).collect::<Vec<_>>(),
            "controlled_bodies":attempt.map(|a|a.body_indices.as_slice()),
            "reduced_body_base":state.rigid.body_base,"reduced_dimension":state.rigid.dim}));
        let query_toi=state.analytic.as_mut().and_then(|a|a.diagnostic_toi(device).ok());
        terminal.capture(device,label,step,final_step,dt,advanced,rescale,fraction,origin,proposal,
            query_toi,state.floor.host(),attempt);
        let root_view=state.roots.view(&state.rigid);
        terminal.capture_root_buffers(device,"acceptance-error-current-root-workspaces",bodies,
            state.rigid_state.handle(),state.rigid_rotation_step.handle(),state.rigid_running_rotation.handle(),state.rigid_centroid.handle(),root_view);
        terminal.capture_root_direction(device,state.dx.handle(),state.pcg_rigid.xr.handle(),vertices,state.pcg_rigid.xr.len());
        if let Some(phase)=terminal.phases.back_mut() {
            phase.as_object_mut().unwrap().insert("acceptance_failure_ledger".into(),serde_json::json!({
                "history":false,"root_body_uuid_native_index_map":state.roots.transaction.as_ref().and_then(|tx|tx.effective_roots().ok())
                    .map(|rows|rows.into_iter().map(|r|serde_json::json!({"body_uuid":r.body_uuid,"runtime_body_index":r.runtime_body_index})).collect::<Vec<_>>()),
                "all_body_uuid_map":null,"all_body_uuid_map_source":"bind sealed scene-context/receipt in readonly analyzer; SolverState retains no full UUID map",
                "transaction_generation":state.roots.transaction.as_ref().map(|tx|tx.hard_epoch().0),
                "attempt_sequence":attempt.map(|a|a.sequence()),"observer":ledger,
                "B32_alpha_advances_clock":false,
                "note":"Failure-only current raw root workspaces, not B17 frozen solve state. Phase x0/x1 are actual immutable acceptance origin/original caller proposal. Per-vertex TOI is current analytic workspace, not a certified mesh raw TOI."}));
        }
        terminal.flush(reason,ccd_eps,max_t);
    }));
    if observed.is_err() {
        // An optional stderr failure must not replace the original Fatal/reject.
        let _=std::io::Write::write_fmt(&mut std::io::stderr().lock(),format_args!("ACCEPTANCE_FAILURE_OBSERVER_PANIC: original Fatal retained\n"));
    }
}
// ACCEPT FAILURE OBSERVER: END failure-only step context

// Opt-in, read-only diagnostic. No kernel/ABI/numerical changes.
struct AnalyticGapTrace {
    path: Option<std::path::PathBuf>,
    time: f64,
    vertices: usize,
    ccd_eps: f32,
    vertex_props: Option<serde_json::Value>,
    root_layout: Option<serde_json::Value>,
    sequence: u64,
    phases: std::collections::VecDeque<serde_json::Value>,
    flushed: bool,
}
impl AnalyticGapTrace {
    fn new(time: f64, vertices: usize, ccd_eps: f32) -> Self {
        let path = std::env::var_os("PPF_ANALYTIC_GAP_CAPTURE_DIR")
            .map(std::path::PathBuf::from).filter(|p| p.is_absolute());
        Self { path, time, vertices, ccd_eps, vertex_props: None, root_layout: None, sequence: 0,
            phases: std::collections::VecDeque::new(), flushed: false }
    }
    // Constructed only after the original B0 bound has already fired.
    fn new_terminal(time:f64,vertices:usize,ccd_eps:f32)->Self {
        let path=std::env::var_os("PPF_ANALYTIC_STALL_TERMINAL_CAPTURE_DIR")
            .map(std::path::PathBuf::from).filter(|p|p.is_absolute());
        Self {path,time,vertices,ccd_eps,vertex_props:None,root_layout:None,sequence:0,
            phases:std::collections::VecDeque::new(),flushed:false}
    }
    fn enabled(&self) -> bool { self.path.is_some() }
    fn capture_props(&mut self, device: &mut impl Device, props: Handle) {
        if !self.enabled() { return; }
        self.vertex_props = sand_offset_read::<crate::data::VertexProp>(device,props,self.vertices)
            .and_then(|p|serde_json::to_value(p).ok());
    }
    #[allow(clippy::too_many_arguments)]
    fn capture(&mut self, device: &mut impl Device, label: &'static str,
        step: u32, final_step: bool, dt: f32, advanced: f64, rescale: f32,
        fraction: f32, x0: Handle, x1: Handle, query_toi: Option<Vec<f32>>,
        floors: &[crate::data::Floor],
        attempt: Option<&crate::joint_root_motion_state::Attempt>) {
        if !self.enabled() { return; }
        let a = sand_offset_read::<f32>(device, x0, 3 * self.vertices);
        let b = sand_offset_read::<f32>(device, x1, 3 * self.vertices);
        let sweep_reference = if label.ends_with("analytic-sweep-input") {
            a.as_ref().zip(b.as_ref()).map(|(a,b)| {
                a.chunks_exact(3).zip(b.chunks_exact(3)).enumerate().map(|(v,(p0,pn))| {
                    let p1: [f32;3] = std::array::from_fn(|k| fraction*(pn[k]-p0[k])+p0[k]);
                    let rows: Vec<_> = floors.iter().enumerate().map(|(j,f)| {
                        let dot = |p:&[f32]| {
                            let products: [f32;3] = std::array::from_fn(|k| (p[k]-f.ground[k])*f.up[k]);
                            (products[0]+products[1])+products[2]
                        };
                        let h0=dot(p0); let h1=dot(&p1);
                        let park=if h0>0.0 {
                            let floor_clear=(2.0*self.ccd_eps).max(1.0e-2*f.ghat);
                            let allowance=if self.ccd_eps>0.0 {0.5*self.ccd_eps} else {0.5*h0};
                            floor_clear.min(h0-allowance.min(0.5*h0))
                        } else {0.0};
                        // Reference only; authoritative device TOI is recorded separately.
                        serde_json::json!({"plane":j,"host_h0_separate_bits":h0.to_bits(),
                            "host_h1_separate_bits":h1.to_bits(),
                            "host_park_bits":park.to_bits(),
                            "host_crossing_fraction_bits":((h0-park)/(h0-h1)).to_bits(),
                            "host_horizon_position_bits":p1.map(f32::to_bits)})
                    }).collect();
                    serde_json::json!({"vertex":v,"planes":rows})
                }).collect::<Vec<_>>()
            })
        } else { None };
        let points = |values: Option<Vec<f32>>| -> serde_json::Value {
            let Some(values) = values else { return serde_json::json!({"read_error":true}); };
            let rows: Vec<_> = values.chunks_exact(3).enumerate().map(|(v,p)| {
                let gaps: Vec<_> = floors.iter().enumerate().map(|(j,f)| {
                    let d: [f32;3] = std::array::from_fn(|k| p[k]-f.ground[k]);
                    let products: [f32;3] = std::array::from_fn(|k| d[k]*f.up[k]);
                    let separate = (products[0]+products[1])+products[2];
                    let fused = d[0].mul_add(f.up[0],d[1].mul_add(f.up[1],products[2]));
                    let precise: f64 = (0..3).map(|k|
                        (f64::from(p[k])-f64::from(f.ground[k]))*f64::from(f.up[k])).sum();
                    serde_json::json!({"plane":j,"host_f64_gap":precise,
                        "host_f32_separate_bits":separate.to_bits(),
                        "host_f32_nested_fma_bits":fused.to_bits()})
                }).collect();
                serde_json::json!({"vertex":v,"gpu_position_bits":p.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),
                    "reference_gaps_not_gpu_evaluation":gaps})
            }).collect();
            serde_json::json!(rows)
        };
        let planes: Vec<_> = floors.iter().enumerate().map(|(j,f)|serde_json::json!({
            "index":j,"ground_bits":std::array::from_fn::<_,3,_>(|k|f.ground[k].to_bits()),
            "up_bits":std::array::from_fn::<_,3,_>(|k|f.up[k].to_bits()),
            "ghat_bits":f.ghat.to_bits(),"friction_bits":f.friction.to_bits(),
            "thickness_bits":f.thickness.to_bits(),"kinematic":f.kinematic})).collect();
        // Opt-in history only: preserve the accepted start and the approach to
        // an acceptance refusal, rather than just the last Newton iteration.
        if self.phases.len()==8192 { self.phases.pop_front(); }
        self.sequence+=1;
        self.phases.push_back(serde_json::json!({"diagnostic_id":self.sequence,"label":label,"step":step,"final":final_step,
            "accepted_physical_time":self.time,"integration_dt_bits":dt.to_bits(),
            "toi_advanced":advanced,"toi_rescale_bits":rescale.to_bits(),
            "fraction_or_max_t_bits":fraction.to_bits(),
            "per_vertex_gpu_toi_bits":query_toi.map(|q|q.iter().map(|t|t.to_bits()).collect::<Vec<_>>()),
            "pending_root_clock":attempt.map(|a|serde_json::json!({"start":a.start_s,"end":a.end_s,
                "integration_dt_bits":a.integration_dt_s.map(f32::to_bits),
                "start_bits":a.start_s.to_bits(),"end_bits":a.end_s.to_bits(),
                "frozen_authored_pose_f64_bits":a.body_indices.iter().zip(&a.target).map(|(body,pose)|serde_json::json!({
                    "body":body,"center_bits":pose.center_world_m.map(f64::to_bits),
                    "rotation_wxyz_bits":pose.rotation_world_wxyz.map(f64::to_bits)})).collect::<Vec<_>>()})),
            "planes":planes,"sweep_reference_not_gpu_evaluation":sweep_reference,
            "x0":points(a),"x1":points(b)}));
    }
    // Observe existing GPU workspaces; do not launch, refit or alter a pose.
    // All reads use existing typed POD readback at synchronized Device boundaries.
    #[allow(clippy::too_many_arguments)]
    fn capture_root_buffers(&mut self,device:&mut impl Device,label:&'static str,bodies:usize,
        fitted:Handle,dtheta:Handle,running:Handle,centroid:Handle,view:Option<super::roots::View>) {
        if !self.enabled() {return;}
        let Some(triples)=bodies.checked_mul(3) else {return;};
        let Some(nines)=bodies.checked_mul(9) else {return;};
        if !bodies.checked_mul(std::mem::size_of::<crate::data::PdrdRigidState>()).is_some_and(|n|n<=isize::MAX as usize) {return;}
        let view=view.filter(|v|v.count as usize<=bodies);
        let states=sand_offset_read::<crate::data::PdrdRigidState>(device,fitted,bodies).map(|rows|
            rows.into_iter().enumerate().map(|(body,s)|serde_json::json!({"body":body,
                "anchor_bits":s.anchor.map(f32::to_bits),"x_bits":s.x.map(f32::to_bits),
                "rotation_column_major_bits":s.rotation.map(f32::to_bits),"count":s.count,
                "host_reference_world_center":std::array::from_fn::<_,3,_>(|a|f64::from(s.anchor[a])+f64::from(s.x[a]))})).collect::<Vec<_>>());
        let bits=|values:Option<Vec<f32>>|values.map(|v|v.into_iter().map(f32::to_bits).collect::<Vec<_>>());
        let dtheta=bits(sand_offset_read::<f32>(device,dtheta,triples));
        let running=bits(sand_offset_read::<f32>(device,running,nines));
        let centroid=bits(sand_offset_read::<f32>(device,centroid,triples));
        let targets=view.and_then(|v|sand_offset_read::<crate::joint_root_motion_gpu::RootTargetGpu>(device,v.targets,v.count as usize)).map(|rows|
            rows.into_iter().map(|r|serde_json::json!({"body":r.body,"enabled":r.enabled,"body_in_range":(r.body as usize)<bodies,"list_start":r.list_start,"list_count":r.list_count,
                "center_bits":r.center.map(f32::to_bits),"rotation_column_major_bits":r.rotation.map(f32::to_bits)})).collect::<Vec<_>>());
        if let Some(phase)=self.phases.back_mut() {
            let object=phase.as_object_mut().unwrap();
            let records=object.entry("actual_gpu_root_workspaces").or_insert_with(||serde_json::json!([]));
            records.as_array_mut().unwrap().push(serde_json::json!({"label":label,"bodies":bodies,
                "fitted_state":states,"dtheta_bits":dtheta,"running_rotation_bits":running,
                "rigid_centroid_bits":centroid,"actual_retract_targets":targets,
                "note":"raw GPU POD/buffer bits; computed world center is host reference only; missing read is null"}));
        }
    }
    fn capture_root_direction(&mut self,device:&mut impl Device,dx:Handle,xr:Handle,vertices:usize,dim:usize) {
        if !self.enabled() {return;}
        let Some(count)=vertices.checked_mul(3) else {return;};
        if !count.max(dim).checked_mul(std::mem::size_of::<f32>()).is_some_and(|n|n<=isize::MAX as usize) {return;}
        let mut bits=|h,n|sand_offset_read::<f32>(device,h,n).map(|v|v.into_iter().map(f32::to_bits).collect::<Vec<_>>());
        let direction=bits(dx,count);let reduced=bits(xr,dim);
        if let Some(phase)=self.phases.back_mut() {
            phase.as_object_mut().unwrap().insert("actual_gpu_solved_direction".to_owned(),serde_json::json!({
                "vertex_dx_bits":direction,"reduced_xr_bits":reduced,
                "note":"existing post-solve workspaces at named stage, not an extra projection/particular evaluation"}));
        }
    }
    fn flush(&mut self, reason: &str, ccd_eps: f32, max_t: f32) {
        if self.flushed { return; }
        let Some(dir)=self.path.as_ref() else { return; };
        self.flushed=true;
        let result=(|| -> std::io::Result<()> {
            std::fs::create_dir_all(dir)?;
            let file=std::fs::OpenOptions::new().write(true).create_new(true)
                .open(dir.join("analytic-gap-failure.json"))?;
            serde_json::to_writer_pretty(file,&serde_json::json!({
                "schema":"analytic-gap-readonly-v1","reason":reason,"time":self.time,
                "vertices":self.vertices,"ccd_eps_bits":ccd_eps.to_bits(),
                "line_search_max_t_bits":max_t.to_bits(),"vertex_props":self.vertex_props,"root_layout":self.root_layout,"phases":self.phases,
                "note":"GPU buffer bits/TOIs are authoritative; host reference gaps do not replace GPU arithmetic or its original diagnostic. No numerical correction."}))?;
            Ok(())
        })();
        eprintln!("ANALYTIC_GAP_CAPTURE {}: {:?}",dir.display(),result);
    }
}







