//! Actual gradient-assembly device observations, collected only after commit.
//! No extra curve evaluation or configured binding count is called consumption.
use ppf_cts_compute::{Device, Fault};
use crate::restoring_curve_gpu::CurveHistory;
use super::restoring::Curves;
use serde_json::{json, Value};

#[derive(Clone, Default,Debug,PartialEq,serde::Serialize,serde::Deserialize)]
pub(crate) struct BindingStats {
    replacements: u64,
    nonzero_q: u64,
    nonzero_effort: u64,
    q_abs_max: f64,
    effort_component_abs_max: f64,
    energy_max: f64,
    first_q: f32,
    first_reference_q: f32,
    history_first: Option<CurveHistory>,
    history_last: Option<CurveHistory>,
}
#[derive(Clone, Default,Debug,PartialEq,serde::Serialize,serde::Deserialize)]
pub(crate) struct RuntimeStats {
    accepted_steps: u64,
    first_start_time: Option<f64>,
    last_accepted_time: Option<f64>,
    dt_min: f32,
    dt_max: f32,
    samples: Vec<BindingStats>,
}
fn refuse(detail: &str) -> Fault {
    Fault::Platform { call: "restoring actual observations", detail: format!("RESTORING_RUNTIME_EVIDENCE_INVALID: {detail}") }
}

impl RuntimeStats {
    pub(crate) fn checkpoint_commits(&self)->(u64,Option<f64>){(self.accepted_steps,self.last_accepted_time)}
    pub(crate) fn validate_checkpoint(&self,time:f64,bindings:usize)->Result<(),String>{
        if self.accepted_steps==0 {
            if self.first_start_time.is_some()||self.last_accepted_time.is_some()||!self.samples.is_empty(){return Err("CHECKPOINT_RESTORING_STATS_INVALID: nonempty zero-count history".into());}
            return Ok(());
        }
        if self.samples.len()!=bindings || self.last_accepted_time!=Some(time) ||
            !self.first_start_time.is_some_and(|t|t.is_finite()&&t>=0.&&t<time) ||
            !self.dt_min.is_finite()||!self.dt_max.is_finite()||self.dt_min<=0.||self.dt_max<self.dt_min {
            return Err("CHECKPOINT_RESTORING_STATS_INVALID: clock/count/duration".into());
        }
        for s in &self.samples {
            if s.replacements!=self.accepted_steps||s.nonzero_q>s.replacements||s.nonzero_effort>s.replacements||
                ![s.q_abs_max,s.effort_component_abs_max,s.energy_max].iter().all(|v|v.is_finite()&&*v>=0.)||
                !s.first_q.is_finite()||!s.first_reference_q.is_finite(){return Err("CHECKPOINT_RESTORING_STATS_INVALID: binding accumulator".into());}
            for h in [s.history_first,s.history_last].into_iter().flatten(){if h.initialized!=1||h.epoch==0||h.reserved!=0||!h.q0.is_finite()||!h.d0.iter().all(|v|v.is_finite()){return Err("CHECKPOINT_RESTORING_STATS_INVALID: captured history".into());}}
        }
        Ok(())
    }
}

/// This is called AFTER real physical commit, before any subsequent trial clears
/// the device bank. Rows are q, actual effort, E, physical dt, replaced=1, q_ref.
/// Receipt describes the last real assembly sample of that committed substep,
/// not a separately evaluated final-pose oracle.
pub fn observe<D: Device>(device: &mut D, curves: &mut Curves, accepted_time: f64) -> Result<(), Fault> {
    if !curves.active() { return Ok(()); }
    curves.check(device, "before accepted-step observation")?;
    let start = f64::from_bits(curves.accepted_time.ok_or_else(|| refuse("history was never captured"))?);
    let dt = curves.dt;
    if !accepted_time.is_finite() || !dt.is_finite() || dt <= 0.0 || accepted_time <= start
        || (accepted_time-start-f64::from(dt)).abs() > (1e-8f64).max(f64::from(dt)*1e-5)
        || curves.runtime.last_accepted_time.is_some_and(|old| accepted_time <= old) {
        return Err(refuse("accepted physical time/dt mismatch or duplicate commit"));
    }
    let mut rows = vec![0.0f32; curves.observations.len()];
    curves.observations.read(device, 0, &mut rows)?;
    let mut history = vec![CurveHistory::default(); curves.history.len()];
    curves.history.read(device, 0, &mut history)?;
    if rows.len() != curves.binding_metadata.len()*6 { return Err(refuse("observation row count")); }
    // All device rows are checked before any accumulator mutation.
    let mut candidate = curves.runtime.clone();
    if candidate.samples.is_empty() { candidate.samples.resize_with(curves.binding_metadata.len(), BindingStats::default); }
    if candidate.samples.len() != curves.binding_metadata.len() { return Err(refuse("binding inventory changed")); }
    for (index, binding) in curves.binding_metadata.iter().enumerate() {
        let row = &rows[index*6..index*6+6];
        if !row.iter().all(|v| v.is_finite()) || row[4] != 1.0 || row[3].to_bits() != dt.to_bits() || row[2] < 0.0 {
            return Err(refuse("missing replacement, nonfinite assembly value, energy or actual dt mismatch"));
        }
        let sample = &mut candidate.samples[index];
        let captured = if binding.history_index == u32::MAX { None } else {
            let h = *history.get(binding.history_index as usize).ok_or_else(|| refuse("history index"))?;
            if h.initialized != 1 || h.epoch != curves.epoch || h.epoch == 0 || !h.q0.is_finite() || !h.d0.iter().all(|v| v.is_finite()) {
                return Err(refuse("uninitialized/nonfinite/wrong-epoch device history"));
            }
            if sample.history_last.is_some_and(|previous| h.epoch <= previous.epoch) { return Err(refuse("history epoch did not advance after accepted transition")); }
            Some(h)
        };
        if sample.replacements == 0 {
            sample.first_q=row[0]; sample.first_reference_q=row[5]; sample.history_first=captured;
        }
        sample.replacements=sample.replacements.checked_add(1).ok_or_else(|| refuse("replacement counter overflow"))?;
        sample.nonzero_q += u64::from(row[0] != 0.0);
        sample.nonzero_effort += u64::from(row[1] != 0.0);
        sample.q_abs_max=sample.q_abs_max.max(f64::from(row[0]).abs());
        sample.effort_component_abs_max=sample.effort_component_abs_max.max(f64::from(row[1]).abs());
        sample.energy_max=sample.energy_max.max(f64::from(row[2]));
        sample.history_last=captured;
    }
    if candidate.accepted_steps == 0 { candidate.first_start_time=Some(start);candidate.dt_min=dt;candidate.dt_max=dt; }
    else { candidate.dt_min=candidate.dt_min.min(dt);candidate.dt_max=candidate.dt_max.max(dt); }
    candidate.accepted_steps=candidate.accepted_steps.checked_add(1).ok_or_else(|| refuse("accepted step counter overflow"))?;
    candidate.last_accepted_time=Some(accepted_time);
    curves.runtime=candidate;
    Ok(())
}
fn history_value(history: Option<CurveHistory>) -> Value {
    history.map_or(Value::Null, |h| json!({"epoch":h.epoch,"initialized":h.initialized,"d0":h.d0,"q0":h.q0}))
}
fn channel(kind:u32, channel:u32)-> &'static str {
    if kind==5 {"cross_link_distance"} else if kind==4 {"stitch"} else {match channel {0=>"twist_x",1=>"swing_y",2=>"swing_z",3=>"anchor",_=>"UNKNOWN"}}
}
pub fn receipt(curves:&Curves)->Value {
    if !curves.active() { return Value::Null; }
    let runtime=&curves.runtime;
    let bindings:Vec<Value>=curves.binding_metadata.iter().enumerate().map(|(i,b)| {
        let s=runtime.samples.get(i).cloned().unwrap_or_default();
        json!({"binding_index":i,"kind":b.kind,"target_index":b.target_index,"channel":channel(b.kind,b.channel),"receipt_index":b.receipt_index,
            "actual_replacement_samples":s.replacements,"nonzero_q_samples":s.nonzero_q,"nonzero_effort_samples":s.nonzero_effort,
            "q_abs_max":s.q_abs_max,"effort_component_abs_max":s.effort_component_abs_max,"energy_max":s.energy_max,"first_q":s.first_q,"first_reference_q":s.first_reference_q,
            "history_first":history_value(s.history_first),"history_last":history_value(s.history_last)})
    }).collect();
    json!({"kind":"ppf.restoring_curves.runtime_observations","version":1,"source":"ACTUAL_GRADIENT_ASSEMBLY_DEVICE_BUFFER",
        "sample_location":"LAST_REAL_ASSEMBLY_BEFORE_ACCEPTED_COMMIT","accepted_steps":runtime.accepted_steps,
        "first_start_time":runtime.first_start_time,"last_accepted_time":runtime.last_accepted_time,"dt_min":runtime.dt_min,"dt_max":runtime.dt_max,"bindings":bindings})
}
/// A FINISHED native writer must use this gate. Unobserved configuration or a
/// partial diagnostic snapshot cannot be published as completed consumption.
pub fn final_receipt(curves: &Curves) -> Result<Value, Fault> {
    if !curves.active() { return Ok(Value::Null); }
    if curves.runtime.accepted_steps == 0 || curves.runtime.last_accepted_time.is_none()
        || curves.runtime.samples.len() != curves.binding_metadata.len()
        || curves.runtime.samples.iter().any(|s| s.replacements != curves.runtime.accepted_steps) {
        return Err(refuse("active curves have no complete actual accepted-step replacement evidence"));
    }
    Ok(receipt(curves))
}
