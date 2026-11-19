//! Small policy surface shared by the opt-in adaptive step path and host tests.
//! It classifies only typed finite PCG iteration-cap reports; it does not
//! inspect logs, commit time, or mutate solver state.

use crate::data::StepResult;

pub const MAX_REJECTIONS: u32 = 16;
/// Identifies the typed, bounded pre-C5 FEM material retry host contract.
pub(crate) const FEM_RETRY_HOST_CAPABILITY: &str = "typed_b32_v1";
pub const REASON_NONE: u32 = crate::data::RETRY_REASON_NONE;
pub const REASON_PCG_CAP: u32 = crate::data::RETRY_REASON_PCG_MAX_ITERATIONS;
pub const REASON_UNSUPPORTED: u32 = crate::data::RETRY_REASON_UNSUPPORTED_STATE;

pub struct RetryAttempt {
    pub result: StepResult,
    pub continuation: u64,
}

pub struct RetryEvent {
    pub attempt_index: u32,
    pub requested_h: f32,
    pub iterations: u32,
    pub residual: f32,
    pub native_rejections: u32,
    pub next_h: f32,
    pub remaining_budget: u32,
}

pub struct RetryRun {
    pub save_requested: bool,
    pub result: StepResult,
    pub requested_h: f32,
    pub retry_count: u32,
    pub events: Vec<RetryEvent>,
    /// Present only for the host coordinator: changed by CG caps, never FEM.
    pub cg_ceiling_candidate: Option<f32>,
    pub final_remaining_budget: Option<u32>,
}

/// One concrete backend/native-attempt adapter. Production and CPU fake
/// implementations both use the coordinator below, so budget, token, h
/// halving, cancellation, and exactly-once cleanup are shared executable code.
pub trait RetryAttemptOps {
    fn prepare_and_advance(
        &mut self,
        requested_h: f32,
        attempt_index: u32,
        continuation: u64,
        remaining_budget: u32,
    ) -> RetryAttempt;
    fn cancel(&mut self, continuation: u64);
    fn finish(&mut self);
}

pub struct ClosureRetryAttemptOps<A, C, F> {
    attempt: A,
    cancel: C,
    finish: F,
}

impl<A, C, F> ClosureRetryAttemptOps<A, C, F> {
    pub fn new(attempt: A, cancel: C, finish: F) -> Self {
        Self { attempt, cancel, finish }
    }
}

impl<A, C, F> RetryAttemptOps for ClosureRetryAttemptOps<A, C, F>
where
    A: FnMut(f32, u32, u64, u32) -> RetryAttempt,
    C: FnMut(u64),
    F: FnMut(),
{
    fn prepare_and_advance(&mut self, h:f32, index:u32, token:u64, budget:u32) -> RetryAttempt {
        (self.attempt)(h,index,token,budget)
    }
    fn cancel(&mut self, token:u64) { (self.cancel)(token); }
    fn finish(&mut self) { (self.finish)(); }
}

/// Run complete candidates from one accepted start. `prepare_and_advance`
/// owns the real input/seed preparation and native call; this coordinator owns
/// only retry policy. `finish` is invoked once on every ordinary return path.
pub fn run_adaptive_attempts<O: RetryAttemptOps>(
    ops: &mut O,
    nominal_h: f32,
    start_time: f64,
    rejection_budget: u32,
) -> RetryRun {
    let mut requested_h = nominal_h;
    let mut remaining = rejection_budget;
    let mut continuation = 0u64;
    let mut retry_count = 0u32;
    let mut events = Vec::new();
    let result = loop {
        let attempt = ops.prepare_and_advance(
            requested_h,
            retry_count,
            continuation,
            remaining,
        );
        let retryable = !attempt.result.success()
            && attempt.result.retry_reason == REASON_PCG_CAP
            && attempt.continuation != 0;
        if retryable {
            let next_budget = remaining_after(remaining, attempt.result.retry_rejections, true);
            let next_h = next_half_h(requested_h, start_time);
            if let (Some(next_h), Some(next_budget)) = (next_h, next_budget) {
                events.push(RetryEvent {
                    attempt_index: retry_count,
                    requested_h,
                    iterations: attempt.result.retry_iterations,
                    residual: attempt.result.retry_residual,
                    native_rejections: attempt.result.retry_rejections,
                    next_h,
                    remaining_budget: next_budget,
                });
                requested_h = next_h;
                remaining = next_budget;
                continuation = attempt.continuation;
                retry_count += 1;
                continue;
            }
        }
        if attempt.continuation != 0 {
            ops.cancel(attempt.continuation);
        }
        break attempt.result;
    };
    ops.finish();
    RetryRun { save_requested:false, result, requested_h, retry_count, events, cg_ceiling_candidate:None, final_remaining_budget:None }
}

/// Material-aware host coordinator. The callback must resample every authored
/// input and restore its pre-override seed for each requested h.
pub fn run_host_attempts<A,C,F>(mut attempt:A,mut cancel:C,mut finish:F,initial_h:f32,start:f64,budget:u32,initial_ceiling:f32)
    ->Result<RetryRun,super::scene::Fatal>
where A:FnMut(f32,u32,u64,u32)->Result<super::HostAdvanceOutcome,super::scene::Fatal>,
      C:FnMut(u64)->Result<(),super::scene::Fatal>,F:FnMut()->Result<(),super::scene::Fatal> {
    use super::HostAdvanceOutcome as O;
    let mut h=initial_h;let mut remaining=budget;let mut token=0u64;let mut count=0u32;let mut events=Vec::new();let mut ceiling=initial_ceiling;
    let mut fem_first:Option<String>=None;let mut fem_last:Option<String>=None;
    let mut swing_first:Option<String>=None;let mut swing_last:Option<String>=None;
    let mut chart_first:Option<String>=None;let mut chart_last:Option<String>=None;
    let outcome=(||->Result<RetryRun,super::scene::Fatal>{loop {
        let next=match attempt(h,count,token,remaining){Ok(v)=>v,Err(mut original)=>{
            if token!=0 {if let Err(e)=cancel(token){original.detail.push_str(&format!("; host continuation cancellation failed: {}",e.detail));}}
            return Err(original);
        }};
        token=0;
        match next {
            O::KinematicsShortened { actual_h, continuation } => {
                let next_budget = remaining_after(remaining, 0, true);
                if continuation == 0 || next_budget.is_none() || !actual_h.is_finite()
                    || actual_h <= 0.0 || actual_h >= h || start + f64::from(actual_h) <= start {
                    if continuation != 0 { cancel(continuation)?; }
                    return Err(super::scene::Fatal::invariant("KINEMATICS_RETRY_UNRESOLVED: no strictly shorter valid step or retry budget exhausted"));
                }
                // Keep a small margin below the last CCD limit so floating-point
                // rounding does not repeatedly reject an unchanged boundary.
                const KINEMATICS_RETRY_MARGIN: f32 = 0.99;
                h = actual_h * KINEMATICS_RETRY_MARGIN;
                if h <= 0.0 || start + f64::from(h) <= start {
                    cancel(continuation)?;
                    return Err(super::scene::Fatal::invariant("KINEMATICS_RETRY_UNRESOLVED: margin underflow"));
                }
                remaining = next_budget.unwrap();
                token = continuation;
                count += 1;
            }            O::SaveRequested{continuation}=>{
                if continuation==0 {return Err(super::scene::Fatal::invariant("HOST_SAVE: missing continuation"));}
                cancel(continuation)?;
                let mut result=StepResult::default();result.newton_progress=false;
                return Ok(RetryRun{save_requested:true,result,requested_h:h,retry_count:count,events,cg_ceiling_candidate:Some(ceiling),final_remaining_budget:Some(remaining)});
            }
            O::Accepted(result)=>{
                remaining=remaining_after(remaining,result.retry_rejections,false).ok_or_else(||super::scene::Fatal::invariant("HOST_RETRY: native rejection budget overflow"))?;
                return Ok(RetryRun{save_requested:false,result,requested_h:h,retry_count:count,events,cg_ceiling_candidate:Some(ceiling),final_remaining_budget:Some(remaining)});
            }
            O::Refused(result)=>{
                remaining=remaining_after(remaining,result.retry_rejections,false).ok_or_else(||super::scene::Fatal::invariant("HOST_RETRY: refused native rejection budget overflow"))?;
                return Ok(RetryRun{save_requested:false,result,requested_h:h,retry_count:count,events,cg_ceiling_candidate:Some(ceiling),final_remaining_budget:Some(remaining)});
            }
            O::PcgCap{result,continuation,failed_dt}=>{
                let next_budget=remaining_after(remaining,result.retry_rejections,true);
                let next_h=next_half_h(failed_dt,start);
                if continuation==0 || next_budget.is_none() || next_h.is_none(){
                    if continuation!=0 {if let Err(error)=cancel(continuation){return Err(super::scene::Fatal::invariant(format!("PCG cap iterations={} residual={:?} requested_h={:?}; cancellation failed: {}",result.retry_iterations,result.retry_residual,h,error.detail)));}}
                    return Ok(RetryRun{save_requested:false,result,requested_h:h,retry_count:count,events,cg_ceiling_candidate:Some(ceiling),final_remaining_budget:Some(remaining)});
                }
                let new_h=next_h.unwrap();let new_budget=next_budget.unwrap();
                events.push(RetryEvent{attempt_index:count,requested_h:h,iterations:result.retry_iterations,residual:result.retry_residual,native_rejections:result.retry_rejections,next_h:new_h,remaining_budget:new_budget});
                ceiling=ceiling.min(new_h);h=new_h;remaining=new_budget;token=continuation;count+=1;
            }
            O::NewtonCapRejected{refusal,continuation}=>{
                let next_budget=remaining_after(remaining,refusal.native_rejections,true);
                let next_h=next_half_h(refusal.failed_dt,start);
                if continuation==0 || next_budget.is_none() || next_h.is_none(){
                    let mut original=refusal.original;
                    if continuation!=0 {if let Err(e)=cancel(continuation){original.detail.push_str(&format!("; Newton continuation cancellation failed: {}",e.detail));}}
                    original.detail.push_str("; NEWTON_CAP_UNRESOLVED: budget exhausted or dt cannot advance");return Err(original);
                }
                ::log::info!("NEWTON_CAP_RETRY generation={} start_bits={:016x} step={} toi={:?} failed_h={:?} retry_h={:?}",refusal.generation,refusal.start_bits,refusal.step,refusal.last_toi,refusal.failed_dt,next_h.unwrap());
                h=next_h.unwrap();remaining=next_budget.unwrap();token=continuation;count+=1;
            }
            O::FemMaterialRejected{refusal,continuation}=>{
                let evidence=format!("fault={} row={} side={} generation={} start_bits={:016x} failed_h_bits={:08x} vertices={:?} statuses={:?} reason={:?} original={}",
                    refusal.token.fault,refusal.token.row,refusal.token.side,refusal.token.generation,refusal.token.start_bits,refusal.token.dt_bits,refusal.token.vertices,refusal.token.statuses,refusal.token.reason,refusal.original.detail);
                if fem_first.is_none(){fem_first=Some(evidence.clone());}fem_last=Some(evidence);
                let next_budget=remaining_after(remaining,refusal.native_rejections,true);
                let failed=f32::from_bits(refusal.token.dt_bits);let next_h=next_half_h(failed,start);
                if continuation==0 || next_budget.is_none() || next_h.is_none(){
                    let mut original=refusal.original;
                    if continuation!=0 {if let Err(e)=cancel(continuation){original.detail.push_str(&format!("; FEM continuation cancellation failed: {}",e.detail));}}
                    original.detail.push_str("; FEM_MATERIAL_DOMAIN_UNRESOLVED: rejection budget exhausted or dt cannot advance");return Err(original);
                }
                let new_h=next_h.unwrap();remaining=next_budget.unwrap();
                ::log::info!("FEM_MATERIAL_RETRY row={} side={} failed_h={:?} retry_h={:?} remaining={}",refusal.token.row,refusal.token.side,failed,new_h,remaining);
                h=new_h;token=continuation;count+=1;
            }
            O::MaterialSpinRejected{refusal,continuation}=>{
                let next_budget=remaining_after(remaining,refusal.native_rejections,true);
                let next_h=next_half_h(refusal.failed_dt,start);
                if continuation==0 || next_budget.is_none() || next_h.is_none(){
                    let mut original=refusal.original;
                    if continuation!=0 {if let Err(e)=cancel(continuation){original.detail.push_str(&format!("; material continuation cancellation failed: {}",e.detail));}}
                    original.detail.push_str("; MATERIAL_LOCAL_STEP_UNRESOLVED: rejection budget exhausted or dt cannot advance");return Err(original);
                }
                ::log::info!("MATERIAL_LOCAL_STEP_RETRY generation={} start_bits={:016x} failed_h={:?} retry_h={:?}",refusal.generation,refusal.start_bits,refusal.failed_dt,next_h.unwrap());
                h=next_h.unwrap();remaining=next_budget.unwrap();token=continuation;count+=1;
            }
            O::ChartPathRejected{refusal,continuation}=>{
                let evidence=format!("reason={:?} generation={} start_bits={:016x} failed_h_bits={:08x} original={}",refusal.token.reason,refusal.token.trial,refusal.token.start_bits,refusal.token.dt_bits,refusal.original.detail);
                if chart_first.is_none(){chart_first=Some(evidence.clone());}chart_last=Some(evidence);
                let next_budget=remaining_after(remaining,refusal.native_rejections,true);
                let failed=f32::from_bits(refusal.token.dt_bits);let next_h=next_half_h(failed,start);
                if continuation==0 || next_budget.is_none() || next_h.is_none(){
                    let mut original=refusal.original;
                    if continuation!=0 {if let Err(e)=cancel(continuation){original.detail.push_str(&format!("; chart continuation cancellation failed: {}",e.detail));}}
                    original.detail.push_str("; CONTINUOUS_CHART_PATH_UNRESOLVED: rejection budget exhausted or dt cannot advance");return Err(original);
                }
                let new_h=next_h.unwrap();remaining=next_budget.unwrap();
                ::log::info!("CONTINUOUS_CHART_PATH_RETRY reason={:?} failed_h={:?} retry_h={:?} remaining={}",refusal.token.reason,failed,new_h,remaining);
                h=new_h;token=continuation;count+=1;
            }
            O::SwingSampleRejected{refusal,continuation}=>{
                let sample=refusal.token.sample;
                let evidence=format!("fault={} joint={} bodies=({},{}) sample_sequence={} origin_sequence={} generation={} start_bits={:016x} failed_h_bits={:08x} source=B31_Iterate original={}",
                    sample.code,sample.joint_index,sample.body_a,sample.body_b,sample.call_sequence,refusal.token.origin_sequence,
                    refusal.token.generation,refusal.token.start_bits,refusal.token.dt_bits,refusal.original.detail);
                if swing_first.is_none(){swing_first=Some(evidence.clone());}swing_last=Some(evidence);
                let next_budget=remaining_after(remaining,refusal.native_rejections,true);
                let failed=f32::from_bits(refusal.token.dt_bits);let next_h=next_half_h(failed,start);
                if continuation==0 || next_budget.is_none() || next_h.is_none(){
                    let mut original=refusal.original;
                    if continuation!=0 {if let Err(e)=cancel(continuation){original.detail.push_str(&format!("; swing continuation cancellation failed: {}",e.detail));}}
                    original.detail.push_str("; CONTINUOUS_SWING_DOMAIN_UNRESOLVED: rejection budget exhausted or dt cannot advance");return Err(original);
                }
                let new_h=next_h.unwrap();remaining=next_budget.unwrap();
                ::log::info!("CONTINUOUS_SWING_SAMPLE_RETRY joint={} sequence={} failed_h={:?} retry_h={:?} remaining={}",sample.joint_index,sample.call_sequence,failed,new_h,remaining);
                h=new_h;token=continuation;count+=1;
            }
        }
    }})();
    let cleanup=finish();
    let result=match (outcome,cleanup) {(Err(mut original),Err(e))=>{original.detail.push_str(&format!("; host retry cleanup failed: {}",e.detail));Err(original)},(Err(e),_)=>Err(e),(Ok(_),Err(e))=>Err(e),(Ok(r),Ok(()))=>Ok(r)};
    result.map_err(|mut error|{
        if let Some(first)=fem_first {error.detail.push_str(&format!("; preceding FEM first=[{first}] last=[{}]",fem_last.unwrap_or_default()));}
        if let Some(first)=swing_first {error.detail.push_str(&format!("; preceding swing sample first=[{first}] last=[{}]",swing_last.unwrap_or_default()));}
        if let Some(first)=chart_first {error.detail.push_str(&format!("; preceding chart path first=[{first}] last=[{}]",chart_last.unwrap_or_default()));}
        error
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PcgCap {
    pub iterations: u32,
    pub iteration_limit: u32,
    pub relative_residual: f32,
    pub tolerance: f32,
}

/// The five non-PCG StepResult gates must all remain true. A cap is retryable
/// only when the enum already says MaxIterations and all numeric evidence is
/// internally consistent and finite.
pub fn classify_finite_cap(cap: PcgCap, other_gates: [bool; 5]) -> bool {
    other_gates.into_iter().all(|gate| gate)
        && cap.iteration_limit > 0
        && cap.iterations == cap.iteration_limit
        && cap.relative_residual.is_finite()
        && cap.relative_residual > cap.tolerance
        && cap.tolerance.is_finite()
        && cap.tolerance >= 0.0
}

pub fn classify_no_progress(cap: PcgCap, other_gates: [bool; 5]) -> bool {
    other_gates.into_iter().all(|gate| gate)
        && cap.iterations > 0 && cap.iterations < cap.iteration_limit
        && cap.relative_residual.is_finite() && cap.tolerance.is_finite()
        && cap.tolerance >= 0.0 && cap.relative_residual >= cap.tolerance
}

/// Halve a positive finite candidate only when both the f32 native dt and its
/// f64 host endpoint can advance. Callers retain the immutable nominal dt.
pub fn next_half_h(current_h: f32, start_time: f64) -> Option<f32> {
    if !current_h.is_finite() || current_h <= 0.0 || !start_time.is_finite() {
        return None;
    }
    let next = current_h * 0.5;
    let end = start_time + f64::from(next);
    if !next.is_finite() || next < f32::MIN_POSITIVE || end <= start_time || !end.is_finite() {
        None
    } else {
        Some(next)
    }
}

/// Consume native root/chart rejections plus the current failed CG candidate
/// from one logical-step budget. `None` means the caller must not continue.
pub fn remaining_after(budget: u32, native_rejections: u32, cg_candidate_rejected: bool) -> Option<u32> {
    let spent = native_rejections.checked_add(u32::from(cg_candidate_rejected))?;
    if spent <= budget {
        Some(budget - spent)
    } else {
        None
    }
}

pub fn continuation_matches(open_token: u64, requested_token: u64, start_time: f64, current_time: f64) -> bool {
    requested_token != 0 && open_token == requested_token
        && start_time.is_finite() && current_time.is_finite()
        && start_time == current_time
}

/// The native entry only opens/reuses a token when another CG rejection can
/// still be charged after native root/chart candidates already spent budget.
pub fn may_continue(retry_reason: u32, native_rejections: u32, budget: u32) -> bool {
    retry_reason == REASON_PCG_CAP && native_rejections < budget
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host_success()->StepResult {StepResult{time:1.000625,ccd_success:true,pcg_success:true,intersection_free:true,newton_progress:true,pin_feasible:true,contact_separated:true,..Default::default()}}
    fn host_fem(h:f32,continuation:u64)->super::super::HostAdvanceOutcome {
        super::super::HostAdvanceOutcome::FemMaterialRejected{continuation,refusal:super::super::step::FemHostRefusal{
            token:super::super::soft_joints::MaterialFrameCandidateRejection{generation:1,start_bits:1f64.to_bits(),dt_bits:h.to_bits(),row:1,side:1,vertices:[0,1,2,3],fault:259,statuses:[0,259,0,0],reason:super::super::soft_joints::MaterialFrameRejectReason::StrictF32Negative},
            original:super::super::scene::Fatal::invariant("material frame prepare: row 1, native fault 259"),native_rejections:0}}
    }
    #[test] fn host_chart_swing_and_cg_share_one_budget() {
        use super::super::{HostAdvanceOutcome as O,scene::Fatal,step::{ChartHostRefusal,SwingHostRefusal},joints::{test_chart_token,test_swing_token}};
        let mut samples=Vec::new();let mut finished=0;
        let run=run_host_attempts(|h,i,_,_|{samples.push(h);Ok(match i {
            0=>O::ChartPathRejected{refusal:ChartHostRefusal{token:test_chart_token(h),original:Fatal::invariant("chart source"),native_rejections:0},continuation:1},
            1=>O::SwingSampleRejected{refusal:SwingHostRefusal{token:test_swing_token(h),original:Fatal::invariant("swing source"),native_rejections:0},continuation:2},
            2=>{let mut result=host_success();result.newton_progress=false;result.retry_reason=REASON_PCG_CAP;O::PcgCap{result,continuation:3,failed_dt:h}},
            _=>O::Accepted(host_success())
        })},|_|Ok(()),||{finished+=1;Ok(())},0.008,1.,16,0.008).unwrap();
        assert_eq!(samples,vec![0.008,0.004,0.002,0.001]);assert_eq!(run.final_remaining_budget,Some(13));assert_eq!(finished,1);
        assert_eq!(run.cg_ceiling_candidate,Some(0.001));
    }
    #[test] fn host_chart_and_swing_keep_causes_through_budget_and_cleanup_failure() {
        use super::super::{HostAdvanceOutcome as O,scene::Fatal,step::{ChartHostRefusal,SwingHostRefusal},joints::{test_chart_token,test_swing_token}};
        let error=match run_host_attempts(|h,i,_,_|Ok(if i==0 {
            O::ChartPathRejected{refusal:ChartHostRefusal{token:test_chart_token(h),original:Fatal::invariant("chart original"),native_rejections:0},continuation:1}
        }else {
            O::SwingSampleRejected{refusal:SwingHostRefusal{token:test_swing_token(h),original:Fatal::invariant("swing original"),native_rejections:0},continuation:2}
        }),|_|Err(Fatal::invariant("cancel failure")),||Err(Fatal::invariant("finish failure")),0.008,1.,1,0.008){Err(e)=>e,Ok(_)=>panic!("budget exhaustion must fail")};
        assert!(error.detail.contains("chart original"));assert!(error.detail.contains("swing original"));
        assert!(error.detail.contains("CONTINUOUS_SWING_DOMAIN_UNRESOLVED"));assert!(error.detail.contains("cancel failure"));assert!(error.detail.contains("finish failure"));
    }
    #[test] fn host_fem_and_cg_share_budget_without_fem_sticky() {
        use super::super::HostAdvanceOutcome as O;
        let mut samples=Vec::new();let mut finished=0;
        let run=run_host_attempts(|h,i,_,_|{samples.push(h);Ok(match i {
            0=>host_fem(h,1),
            1=>{let mut r=host_success();r.newton_progress=false;r.retry_reason=REASON_PCG_CAP;O::PcgCap{result:r,continuation:2,failed_dt:h}},
            2=>host_fem(h,3),_=>O::Accepted(host_success())})},|_|Ok(()),||{finished+=1;Ok(())},0.005,1.,16,0.005).unwrap();
        assert_eq!(samples,vec![0.005,0.0025,0.00125,0.000625]);assert_eq!(run.cg_ceiling_candidate,Some(0.00125));
        assert_eq!(run.final_remaining_budget,Some(13));assert_eq!(finished,1);
        let run=run_host_attempts(|h,i,_,_|Ok(if i==0{host_fem(h,7)}else{O::Accepted(host_success())}),|_|Ok(()),||Ok(()),0.005,1.,16,0.005).unwrap();
        assert_eq!(run.cg_ceiling_candidate,Some(0.005));assert_eq!(run.final_remaining_budget,Some(15));
        let run=run_host_attempts(|h,i,_,_|Ok(match i {0=>{let mut r=host_success();r.newton_progress=false;r.retry_reason=REASON_PCG_CAP;O::PcgCap{result:r,continuation:8,failed_dt:h}},1=>host_fem(h,9),_=>O::Accepted(host_success())}),|_|Ok(()),||Ok(()),0.005,1.,16,0.005).unwrap();
        assert_eq!(run.cg_ceiling_candidate,Some(0.0025));assert_eq!(run.requested_h,0.00125);assert_eq!(run.final_remaining_budget,Some(14));
        let run=run_host_attempts(|h,i,_,_|Ok(if i==0{let mut r=host_success();r.newton_progress=false;r.retry_reason=REASON_PCG_CAP;O::PcgCap{result:r,continuation:10,failed_dt:h}}else{O::Accepted(host_success())}),|_|Ok(()),||Ok(()),0.005,1.,16,0.005).unwrap();
        assert_eq!(run.cg_ceiling_candidate,Some(0.0025));assert_eq!(run.final_remaining_budget,Some(15));
    }
    #[test] fn root_clipped_pcg_and_newton_share_budget_using_actual_dt() {
        use super::super::{HostAdvanceOutcome as O,scene::Fatal,step::NewtonCapHostRefusal};
        let mut requested=Vec::new();
        let run=run_host_attempts(|h,i,_,_|{requested.push(h);Ok(match i {
            0=>{let mut r=host_success();r.pcg_success=false;r.retry_reason=REASON_PCG_CAP;O::PcgCap{result:r,continuation:1,failed_dt:0.003}},
            1=>O::NewtonCapRejected{refusal:NewtonCapHostRefusal{failed_dt:0.001, generation:1,start_bits:1f64.to_bits(),native_rejections:0,step:2048,last_toi:0.00001,original:Fatal::invariant("newton cap")},continuation:2},
            _=>O::Accepted(host_success())
        })},|_|Ok(()),||Ok(()),0.01,1.,16,0.01).unwrap();
        assert!(requested[1]<=0.0015 && requested[2]<=0.0005);
        assert_eq!(run.final_remaining_budget,Some(14));
        assert!(run.cg_ceiling_candidate.unwrap()<=0.0015);
    }
    #[test] fn save_request_cancels_pending_once_before_next_trial() {
        use super::super::{HostAdvanceOutcome as O,scene::Fatal,step::NewtonCapHostRefusal};
        let mut calls=0;let mut cancelled=0;let mut finished=0;
        let run=run_host_attempts(|h,_,token,_|{calls+=1;Ok(if token==0 {
            O::NewtonCapRejected{refusal:NewtonCapHostRefusal{failed_dt:h,generation:1,start_bits:1f64.to_bits(),native_rejections:0,step:2048,last_toi:0.00001,original:Fatal::invariant("newton cap")},continuation:7}
        } else {O::SaveRequested{continuation:token}})},|token|{assert_eq!(token,7);cancelled+=1;Ok(())},||{finished+=1;Ok(())},0.01,1.,16,0.01).unwrap();
        assert!(run.save_requested);assert_eq!((calls,cancelled,finished),(2,1,1));
    }
    #[test] fn host_fem_budget_keeps_original_cause_and_finishes_once() {
        let mut finished=0;let mut cancelled=0;
        let error=match run_host_attempts(|h,_,_,_|Ok(host_fem(h,0)),|_|{cancelled+=1;Ok(())},||{finished+=1;Ok(())},0.005,1.,0,0.005){Err(e)=>e,Ok(_)=>panic!("budget must refuse")};
        assert!(error.detail.contains("native fault 259")&&error.detail.contains("FEM_MATERIAL_DOMAIN_UNRESOLVED"));assert_eq!((finished,cancelled),(1,0));
    }
    #[test] fn host_fem_context_survives_callback_cancel_and_cleanup_failures(){
        use super::super::{HostAdvanceOutcome as O,scene::Fatal};
        let err=match run_host_attempts(|h,i,_,_|if i==0{Ok(host_fem(h,3))}else{Err(Fatal::invariant("fresh callback failed"))},|_|Err(Fatal::invariant("cancel failed")),||Err(Fatal::invariant("finish failed")),0.005,1.,16,0.005){Err(e)=>e,Ok(_)=>panic!("must refuse")};
        assert!(err.detail.starts_with("fresh callback failed")&&err.detail.contains("cancel failed")&&err.detail.contains("finish failed")&&err.detail.contains("native fault 259"));
        let err=match run_host_attempts(|h,i,_,_|Ok(if i==0{host_fem(h,4)}else{let mut r=host_success();r.newton_progress=false;r.retry_reason=REASON_PCG_CAP;r.retry_iterations=17;r.retry_residual=0.25;O::PcgCap{result:r,continuation:5,failed_dt:h}}),|_|Err(Fatal::invariant("cancel failed")),||Ok(()),0.005,1.,1,0.005){Err(e)=>e,Ok(_)=>panic!("must refuse")};
        assert!(err.detail.contains("PCG cap iterations=17")&&err.detail.contains("cancel failed")&&err.detail.contains("native fault 259"));
    }

    fn valid() -> PcgCap {
        PcgCap { iterations: 120, iteration_limit: 120, relative_residual: 0.02, tolerance: 0.005 }
    }

    #[test]
    fn only_consistent_finite_cap_is_retryable() {
        assert!(classify_finite_cap(valid(), [true; 5]));
        let mut c = valid(); c.iterations -= 1;
        assert!(!classify_finite_cap(c, [true; 5]));
        let mut c = valid(); c.relative_residual = f32::NAN;
        assert!(!classify_finite_cap(c, [true; 5]));
        let mut c = valid(); c.relative_residual = c.tolerance;
        assert!(!classify_finite_cap(c, [true; 5]));
        for i in 0..5 {
            let mut gates = [true; 5]; gates[i] = false;
            assert!(!classify_finite_cap(valid(), gates));
        }
    }

    #[test]
    fn candidate_halving_rejects_underflow_and_lost_clock_progress() {
        assert_eq!(next_half_h(0.01, 1.0), Some(0.005));
        assert_eq!(next_half_h(f32::MIN_POSITIVE, 0.0), None);
        assert_eq!(next_half_h(1.0e-10, 1.0e30), None);
        assert_eq!(next_half_h(f32::NAN, 0.0), None);
    }

    #[test]
    fn root_chart_and_cg_share_one_rejection_budget() {
        assert_eq!(remaining_after(16, 3, true), Some(12));
        assert_eq!(remaining_after(16, 16, false), Some(0));
        assert_eq!(remaining_after(16, 16, true), None);
        assert_eq!(remaining_after(0, 0, false), Some(0));
        assert_eq!(remaining_after(0, 0, true), None);
        assert_eq!(remaining_after(u32::MAX, u32::MAX, true), None);
    }

    #[test]
    fn retry_continuation_is_bound_to_one_uncommitted_start() {
        assert!(continuation_matches(7, 7, 1.25, 1.25));
        assert!(!continuation_matches(7, 8, 1.25, 1.25));
        assert!(!continuation_matches(7, 7, 1.25, 1.5));
        assert!(!continuation_matches(0, 0, 1.25, 1.25));
        assert!(!continuation_matches(7, 7, f64::NAN, f64::NAN));
        assert!(may_continue(REASON_PCG_CAP, 15, 16));
        assert!(!may_continue(REASON_PCG_CAP, 16, 16));
        assert!(!may_continue(REASON_UNSUPPORTED, 0, 16));
    }
}
