//! Host-only, bounded diagnostic of the existing resident scalar recurrence.
use std::sync::{Mutex, OnceLock, atomic::{AtomicU64, Ordering}};

const MIN_STEPS: usize = 32;
const MAX_STEPS: usize = 256;
const MAX_SEGMENTS: usize = 2;

#[derive(serde::Serialize)]
struct Step { iteration:u32, alpha:f32, beta:f32, pap:f32 }

#[derive(serde::Serialize)]
struct Segment {
    solve_id:u64, restart_generation:u32, physical_start_s:f64,
    sample_time_s:f32, dt_s:f32, dim:usize, body_base:usize, bodies:usize,
    first_iteration:u32, last_iteration:u32, end_reason:&'static str,
    steps:Vec<Step>,
}

#[derive(Default)]
struct State { segments:Vec<Segment>, emitted:bool }
impl State {
    fn admit(&mut self,segment:Segment)->bool {
        if segment.steps.len()<MIN_STEPS || self.emitted || self.segments.len()>=MAX_SEGMENTS {return false;}
        self.segments.push(segment);self.segments.len()==MAX_SEGMENTS
    }
}
static STATE: OnceLock<Mutex<State>> = OnceLock::new();
static ENABLED: OnceLock<bool> = OnceLock::new();
static NEXT_SOLVE: AtomicU64 = AtomicU64::new(0);

fn emit(state:&mut State) {
    if state.emitted { return; }
    state.emitted=true;
    let output=serde_json::json!({"schema":"ppf.pcg-lanczos-trace.v1",
        "diagnostic_only":true,"sampling":"first_contiguous_256_valid_status0_steps",
        "min_steps":MIN_STEPS,"max_segments":MAX_SEGMENTS,
        "segments":state.segments});
    use std::io::Write;
    let _=writeln!(std::io::stderr().lock(),"PCG_LANCZOS_TRACE {}",output);
}

extern "C" fn flush_at_exit() {
    let _=std::panic::catch_unwind(|| {
        if let Some(state)=STATE.get() {
            let mut state=state.lock().unwrap_or_else(|e|e.into_inner());
            emit(&mut state);
        }
    });
}
extern "C" { fn atexit(callback: extern "C" fn()) -> std::ffi::c_int; }

/// Count all reduced solve calls while tracing is enabled, not just admitted ones.
pub(super) fn solve_id() -> Option<u64> {
    let enabled=*ENABLED.get_or_init(|| {
        if !std::env::var_os("PPF_PCG_LANCZOS_TRACE").is_some_and(|v|v=="1") { return false; }
        STATE.get_or_init(||Mutex::new(State::default()));
        // C runtime exit invokes this on normal process completion, including
        // runs with fewer than two qualifying segments. A forced kill cannot.
        unsafe { atexit(flush_at_exit)==0 }
    });
    enabled.then(|| NEXT_SOLVE.fetch_add(1,Ordering::Relaxed)+1)
}

fn submit(segment:Segment) {
    if let Some(state)=STATE.get() {
        let mut state=state.lock().unwrap_or_else(|e|e.into_inner());
        if state.admit(segment) {emit(&mut state);}
    }
}

pub(super) struct Trace { segment:Option<Segment> }
impl Trace {
    pub(super) fn new(solve_id:u64,physical_start_s:f64,sample_time_s:f32,dt_s:f32,
        dim:usize,body_base:usize,bodies:usize)->Option<Self> {
        let mut steps=Vec::new();steps.try_reserve_exact(MAX_STEPS).ok()?;
        Some(Self{segment:Some(Segment{solve_id,restart_generation:0,physical_start_s,
            sample_time_s,dt_s,dim,body_base,bodies,first_iteration:1,last_iteration:0,
            end_reason:"abnormal_exit",steps})})
    }
    pub(super) fn record(&mut self,iteration:u32,alpha:f32,beta:f32,pap:f32) {
        let Some(s)=self.segment.as_mut() else{return};
        if iteration!=s.last_iteration+1 || !alpha.is_finite() || alpha<=0.0 ||
            !beta.is_finite() || beta<0.0 || !pap.is_finite() || pap<=0.0 {
            self.finish("invalid_or_noncontiguous");return;
        }
        s.steps.push(Step{iteration,alpha,beta,pap});s.last_iteration=iteration;
        if s.steps.len()==MAX_STEPS { self.finish("prefix_limit"); }
    }
    pub(super) fn finish(&mut self,reason:&'static str) {
        if let Some(mut s)=self.segment.take() {s.end_reason=reason;submit(s);}
    }
}
impl Drop for Trace { fn drop(&mut self) {self.finish("abnormal_or_route_exit");} }

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn bounded_prefix_and_invalid_end() {
        let mut t=Trace::new(1,0.,0.,0.01,6,3,1).unwrap();
        for i in 1..=31 {t.record(i,1.,0.5,2.);}
        t.record(32,f32::NAN,0.5,2.);
        assert!(t.segment.is_none());
        let mut t=Trace::new(2,0.,0.,0.01,6,3,1).unwrap();
        for i in 1..=MAX_STEPS as u32 {t.record(i,1.,0.5,2.);}
        assert!(t.segment.is_none());
        t.record(257,1.,0.5,2.);assert!(t.segment.is_none());
    }
    #[test] fn segment_does_not_cross_restart() {
        let mut t=Trace::new(3,0.,0.,0.01,6,3,1).unwrap();
        for i in 1..=32 {t.record(i,1.,0.5,2.);}
        assert_eq!(t.segment.as_ref().unwrap().steps.len(),32);
        t.finish("true_residual_restart");assert!(t.segment.is_none());
        t.record(33,1.,0.5,2.);assert!(t.segment.is_none());
    }
    #[test] fn short_segments_do_not_consume_quota() {
        fn segment(id:u64,n:u32)->Segment {
            let mut t=Trace::new(id,0.,0.,0.01,6,3,1).unwrap();
            for i in 1..=n {t.record(i,1.,0.5,2.);}
            t.segment.take().unwrap()
        }
        let mut s=State::default();assert!(!s.admit(segment(1,31)));assert_eq!(s.segments.len(),0);
        assert!(!s.admit(segment(2,32)));assert_eq!(s.segments.len(),1);
        assert!(s.admit(segment(3,33)));assert!(!s.admit(segment(4,34)));assert_eq!(s.segments.len(),2);
    }
    #[test] fn exit_flush_child() {
        if std::env::var_os("PPF_TRACE_TEST_CHILD").is_none(){return;}
        let id=solve_id().unwrap();let mut t=Trace::new(id,0.,0.,0.01,6,3,1).unwrap();
        for i in 1..=32 {t.record(i,1.,0.5,2.);}
        t.finish("converged");
    }
    #[test] fn normal_exit_flushes_one_record_with_one_segment() {
        let output=std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact","driver::pcg::lanczos_trace::tests::exit_flush_child","--nocapture"])
            .env("PPF_TRACE_TEST_CHILD","1").env("PPF_PCG_LANCZOS_TRACE","1")
            .output().unwrap();
        assert!(output.status.success());
        let stderr=String::from_utf8(output.stderr).unwrap();
        let records:Vec<_>=stderr.lines().filter_map(|line|line.strip_prefix("PCG_LANCZOS_TRACE ")).collect();
        assert_eq!(records.len(),1);
        let json:serde_json::Value=serde_json::from_str(records[0]).unwrap();
        assert_eq!(json["schema"],"ppf.pcg-lanczos-trace.v1");
        assert_eq!(json["segments"].as_array().unwrap().len(),1);
    }
}
