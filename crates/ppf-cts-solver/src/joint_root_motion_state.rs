//! Host clock/transaction for prescribed roots. No pose correction after fetch.
//! A retry starts at the last committed clock; key cursors are never advanced
//! by a rejected attempt. Full rotational trajectories are planned here, but
//! native affine-CCD selection refuses them until arc inflation is integrated.
use crate::joint_root_motion_wire::*;

pub const CODE_TIME: &str = "ROOT_MOTION_TIME_MISMATCH";
pub const CODE_COMMIT: &str = "ROOT_MOTION_COMMIT_MISMATCH";
fn fail(code: &'static str, detail: &str) -> RootMotionError {
    RootMotionError { code, body_uuid: None, key_index: None,
        field: "root.transaction".into(), detail: detail.into() }
}

#[derive(Clone, Debug)]
pub struct SweepSegment {
    pub start_s: f64, pub end_s: f64,
    pub start: Vec<RootPose>, pub end: Vec<RootPose>,
    /// Per-root max deviation of its rigid arc from this segment's chord.
    /// Pair margins ADD both participants' radii. A margin never proves CCD.
    pub arc_margin_m: Vec<f64>,
}

pub fn plan_sweep(roots: &[ValidatedRootMotion], radius_m: &[f64],
                  start_s: f64, end_s: f64) -> RootResult<Vec<SweepSegment>> {
    if roots.len() != radius_m.len() || !start_s.is_finite() || start_s<0.0
        || !end_s.is_finite() || end_s<=start_s
        || radius_m.iter().any(|r| !r.is_finite() || *r<0.0) {
        return Err(fail(CODE_TIME,"invalid sweep domain"));
    }
    let mut cuts=vec![start_s,end_s];
    for root in roots {
        for key in &root.keys {
            if key.time_s>start_s && key.time_s<end_s { cuts.push(key.time_s); }
        }
    }
    cuts.sort_by(f64::total_cmp); cuts.dedup();
    let mut refined=vec![start_s];
    for pair in cuts.windows(2) {
        let mut angle=0.0_f64;
        for root in roots {
            angle=angle.max(pose_angle(root.sample_native(pair[0])?,root.sample_native(pair[1])?));
        }
        let n=(angle/(std::f64::consts::PI/18.0)).ceil().max(1.0) as usize;
        if refined.len()+n>65 { return Err(fail(CODE_NATIVE_CAPABILITY,"sweep exceeds 64 subintervals")); }
        for i in 1..=n {
            let t=if i==n {pair[1]} else {pair[0]+(pair[1]-pair[0])*(i as f64/n as f64)};
            if t<=*refined.last().unwrap() { return Err(fail(CODE_TIME,"sweep interval is not representable")); }
            refined.push(t);
        }
    }
    refined.windows(2).map(|p| {
        let start=roots.iter().map(|r|r.sample_native(p[0])).collect::<RootResult<Vec<_>>>()?;
        let end=roots.iter().map(|r|r.sample_native(p[1])).collect::<RootResult<Vec<_>>>()?;
        let arc_margin_m=roots.iter().zip(start.iter().zip(&end)).zip(radius_m).map(|((root,(a,b)),r)| {
            // 2*sin(theta/4)^2 is stable near zero, unlike 1-cos(theta/2).
            let ref_offset=root.reference_transform.map_or(0.0,|d|d.center_world_m.iter().map(|v|v*v).sum::<f64>().sqrt());
            2.0*(r+ref_offset)*(pose_angle(*a,*b)*0.25).sin().powi(2)
        }).collect();
        Ok(SweepSegment{start_s:p[0],end_s:p[1],start,end,arc_margin_m})
    }).collect()
}

fn pose_angle(a: RootPose,b: RootPose)->f64 {
    let dot=a.rotation_world_wxyz.iter().zip(b.rotation_world_wxyz)
        .map(|(a,b)|a*b).sum::<f64>().abs().clamp(0.0,1.0);
    2.0*dot.acos()
}

/// Static orientation / piecewise linear center are affine vertex paths. A
/// changing rotation requires the curved-path contact seam, and is refused even
/// with disable-contact: the first native milestone's capability is bounded.
pub fn require_affine_native(roots: &[ValidatedRootMotion]) -> RootResult<()> {
    for root in roots {
        let q=root.keys[0].pose.rotation_world_wxyz;
        if root.keys.iter().any(|k|k.pose.rotation_world_wxyz != q) {
            return Err(fail(CODE_NATIVE_CAPABILITY,"changing root rotation requires curved CCD integration"));
        }
    }
    Ok(())
}

/// Read-only native candidate evidence; only commit advances accepted history.
pub enum CandidateValidation {Ready,NeedsRefinement(RootMotionError)}

#[derive(Clone, Debug)]
pub struct Attempt {
    generation: u64, sequence: u64,
    pub start_s: f64, pub end_s: f64,
    pub body_indices: Vec<u32>, pub target: Vec<RootPose>,
    /// Rounded GPU representation of the exact elapsed physical duration.
    pub integration_dt_s: Option<f32>,
}
impl Attempt { pub fn sequence(&self)->u64 {self.sequence} }

/// The event clock stays f64. The integrator represents its duration in f32,
/// just as it represents masses and positions; representation roundoff is not
/// another physical step and must never replace accepted velocity history.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NativeStep {
    pub start_s: f64, pub end_s: f64, pub integration_dt_s: f32,
    requested_dt_s: f32,
    pub lands_on_boundary: bool,
}
// Shared physical-clock/f32-representation contract. This is representation
// error only, not permission to enlarge a certified physical sweep.
fn native_duration_matches(elapsed:f64,dt:f32)->bool {
    elapsed.is_finite() && elapsed>0.0 && dt.is_finite() && dt>0.0
        && (f64::from(dt)-elapsed).abs()<=f64::from(f32::EPSILON)*elapsed
}
pub fn native_step_before_key(start_s:f64,requested_dt_s:f32,roots:&[ValidatedRootMotion])->RootResult<NativeStep> {
    let mut next_key=None::<f64>;
    for root in roots {
        if let Some(key)=root.keys.iter().find(|k|k.time_s>start_s) {
            next_key=Some(next_key.map_or(key.time_s,|t|t.min(key.time_s)));
        }
    }
    native_step_before_boundary(start_s,requested_dt_s,next_key,!roots.is_empty())
}

/// Shared exact boundary clock; FEM controls supply real key/event times,
/// without inventing a PDRD runtime index or root record.
pub fn native_step_before_boundary(start_s:f64,requested_dt_s:f32,next_key:Option<f64>,event_clock:bool)->RootResult<NativeStep> {

    if !start_s.is_finite() || start_s<0.0 || !requested_dt_s.is_finite() || requested_dt_s<=0.0 {
        return Err(fail(CODE_TIME,"invalid native event clock"));
    }
    let nominal=f64::from(requested_dt_s);
    let mut end_s=start_s+nominal;
    if let Some(key)=next_key {
        let gap=key-start_s;
        let count=(gap/nominal).round();
        // Plan BEFORE integration: distribute only the binary32 representation
        // residual over equal physical intervals to the next exact author key.
        // Membership in the SAME rounding cell, not an epsilon/time floor, is
        // the contract. In particular a real tiny key gap is never coalesced.
        // Integers beyond 2^53 cannot provide an exact interval count in f64.
        if count>=1.0 && count<=9_007_199_254_740_992.0
            && (gap/count) as f32==requested_dt_s {
            let planned=if count==1.0 {key} else {start_s+gap/count};
            // Absolute-clock cancellation must still satisfy the driver seam.
            if planned>start_s && planned<=key
                && (planned-start_s) as f32==requested_dt_s {end_s=planned;}
        }
        end_s=end_s.min(key);
    }
    let elapsed=end_s-start_s;
    let integration_dt_s=if !event_clock{requested_dt_s}else{elapsed as f32};
    if !end_s.is_finite() || end_s<=start_s || !integration_dt_s.is_finite() || integration_dt_s<=0.0 {
        return Err(fail(CODE_TIME,"event duration cannot be represented by native integration"));
    }
    if event_clock && next_key!=Some(end_s) && integration_dt_s!=requested_dt_s {
        return Err(fail(CODE_TIME,"absolute event clock cannot represent unchanged native integration dt"));
    }
    // No minimum-step clamp, key retiming or clock snap. The authored endpoint
    // is exact; the normal f32 rounding bound must hold for its GPU duration.
    if !native_duration_matches(elapsed,integration_dt_s) {
        return Err(fail(CODE_TIME,"native event duration exceeds normal f32 representation error"));
    }
    Ok(NativeStep{start_s,end_s,integration_dt_s,requested_dt_s,lands_on_boundary:next_key==Some(end_s)})
}

/// Plan ordinary kinematics in scene time while integrating in physical time.
/// Root and FEM transactions use the separate physical-clock contract above.
pub fn native_step_before_scene_boundary(start_s:f64, requested_scene_h:f32, next_event:Option<f64>, playback:f32)->RootResult<NativeStep> {
    if !playback.is_finite() || playback<=0.0 {
        return Err(fail(CODE_TIME,"invalid kinematics playback"));
    }
    let mut plan=native_step_before_boundary(start_s,requested_scene_h,next_event,true)?;
    if playback==1.0 {return Ok(plan);}
    let physical=(plan.end_s-plan.start_s)*f64::from(playback);
    let integration=physical as f32;
    if !native_duration_matches(physical,integration) {
        return Err(fail(CODE_TIME,"kinematics physical duration cannot be represented"));
    }
    plan.integration_dt_s=integration;
    Ok(plan)
}
#[cfg(test)]
mod event_clock_tests {
    use super::*;
    #[test]
    fn scene_boundary_scales_physical_duration_without_merging_events() {
        for playback in [0.5_f32,1.0,2.0] {
            let first=native_step_before_scene_boundary(0.0,0.1,Some(0.037),playback).unwrap();
            assert_eq!(first.end_s,0.037);
            assert_eq!(first.integration_dt_s,(0.037_f64*f64::from(playback)) as f32);
            let second=native_step_before_scene_boundary(first.end_s,0.1,Some(0.037001),playback).unwrap();
            assert_eq!(second.end_s,0.037001);
            assert!(second.integration_dt_s>0.0);
            let whole=native_step_before_scene_boundary(0.0,0.01,None,playback).unwrap();
            assert_eq!(whole.integration_dt_s, (f64::from(0.01_f32)*f64::from(playback)) as f32);
        }
        assert!(native_step_before_scene_boundary(0.0,0.1,None,0.0).is_err());
        assert!(native_step_before_scene_boundary(0.0,0.1,None,f32::NAN).is_err());
    }
    fn moving(body:u32)->ValidatedRootMotion {
        let pose=|x|RootPose{center_world_m:[x,1.0,0.0],rotation_world_wxyz:[1.0,0.0,0.0,0.0]};
        ValidatedRootMotion{scene_uuid:"scene".into(),body_uuid:format!("body{body}"),runtime_body_index:body,
            mode:RootMotionMode::Kinematic,control_events:vec![],reference_transform:None,keys:vec![NormalizedKey{time_s:0.0,pose:pose(0.0)},
                NormalizedKey{time_s:0.01,pose:pose(0.005)},NormalizedKey{time_s:0.025,pose:pose(0.008)}]}
    }
    fn reference_track(mode:RootMotionMode,controls:Vec<RootControlEvent>)->ValidatedRootMotion {
        let s=std::f64::consts::FRAC_1_SQRT_2;
        let pose=|t,c,q|NormalizedKey{time_s:t,pose:RootPose{center_world_m:c,rotation_world_wxyz:q}};
        ValidatedRootMotion{scene_uuid:"scene".into(),body_uuid:"reference-body".into(),runtime_body_index:0,mode,
            keys:vec![pose(0.0,[0.0;3],[1.0,0.0,0.0,0.0]),pose(1.0,[1.0,0.0,0.0],[s,0.0,0.0,s]),
                pose(2.0,[2.0,0.0,0.0],[0.0,0.0,0.0,1.0])],control_events:controls,
            reference_transform:Some(RootPose{center_world_m:[1.0,0.0,0.0],rotation_world_wxyz:[1.0,0.0,0.0,0.0]})}
    }
    #[test] fn future_control_event_is_not_a_no_event_program(){
        let plain=RootState::new(1,vec![moving(0)]).unwrap();assert!(!plain.has_control_events());
        let mut future=moving(0);future.control_events.push(RootControlEvent{time_s:1.0,mode:RootControlMode::Static,pose_binding:Some(PoseBinding::CurrentHold)});
        let state=RootState::new(2,vec![future]).unwrap();assert!(state.due_controls().is_empty());assert!(state.has_control_events());
    }
    #[test]
    fn decimal_key_uses_four_normal_steps_without_roundoff_microsteps() {
        let mut state=RootState::new(1,vec![moving(0),moving(1)]).unwrap();
        for i in 0..4 {
            let plan=state.native_step(0.0025).unwrap();
            assert!(plan.integration_dt_s>0.0024);
            assert_eq!(plan.lands_on_boundary,i==3);
            let a=state.begin_native(plan).unwrap();
            state.commit(&a,a.end_s,&a.target,0.0,1e-6).unwrap();
        }
        assert_eq!(state.committed_time_s,0.01);
        assert!(state.native_step(0.0025).unwrap().integration_dt_s>0.0024);
        assert_eq!(state.commit_sequence,4);
    }
    #[test]
    fn two_clock_ticks_before_root_key_plan_keeps_the_exact_boundary() {
        let key = 11.0_f64 / 30.0;
        let start = f64::from_bits(key.to_bits() - 2);
        let plan = native_step_before_boundary(start, 0.005, Some(key), true).unwrap();
        assert!(plan.lands_on_boundary);
        assert_eq!(plan.end_s, key);
        assert_eq!(f64::from(plan.integration_dt_s), key - start);
    }
    #[test]
    fn native_rebuild_preserves_exact_event_and_shortening_refreezes_target() {
        let mut state=RootState::new(1,vec![moving(0)]).unwrap();
        let plan=state.native_step(0.02).unwrap();
        let a=state.begin_native(plan).unwrap();
        assert_eq!(a.end_s,0.01);
        let same=state.shorten_affine(&a,f64::from(plan.integration_dt_s)).unwrap();
        assert_eq!(same.end_s,0.01);assert_eq!(same.sequence(),a.sequence());
        let short=state.shorten_affine(&a,f64::from(0.005f32)).unwrap();
        assert!(short.end_s<a.end_s);assert!(short.sequence()>a.sequence());
        assert!(state.commit(&a,a.end_s,&a.target,0.0,1e-6).is_err());
        state.commit(&short,short.end_s,&short.target,0.0,1e-6).unwrap();
    }
    #[test]
    fn root_key_and_native_shorten_produce_earlier_retry_endpoint() {
        let mut root=moving(0);root.keys[1].time_s=0.003;
        let mut state=RootState::new(1,vec![root]).unwrap();
        let plan=state.native_step(0.01).unwrap();
        assert!((plan.integration_dt_s-0.003).abs()<1e-8);
        let first=state.begin_native(plan).unwrap();
        let short=state.shorten_native(&first,f64::from(0.002f32)).unwrap();
        let failed=short.integration_dt_s.unwrap();
        state.reject(&short).unwrap();
        let next=crate::driver::adaptive_retry::next_half_h(failed,0.).unwrap();
        let retry=state.begin_native(state.native_step(next).unwrap()).unwrap();
        assert!(retry.end_s<short.end_s && retry.end_s<=0.001);
        assert_ne!(retry.target[0].center_world_m,short.target[0].center_world_m);
    }
    #[test]
    fn event_plan_cannot_change_committed_clock_or_integration_duration() {
        let mut state=RootState::new(1,vec![moving(0)]).unwrap();
        let mut bad=state.native_step(0.0025).unwrap();bad.integration_dt_s=0.003;
        assert!(state.begin_native(bad).is_err());
        let good=state.native_step(0.0025).unwrap();let mut a=state.begin_native(good).unwrap();
        a.integration_dt_s=Some(0.003);
        assert!(state.commit(&a,a.end_s,&a.target,0.0,1e-6).is_err());
    }
    #[test]
    fn ordinary_without_root_retains_exact_requested_gpu_dt() {
        let dt=0.0025f32;
        let plan=native_step_before_key(1234.5,dt,&[]).unwrap();
        assert_eq!(plan.integration_dt_s.to_bits(),dt.to_bits());
        assert_eq!(plan.end_s,1234.5+f64::from(dt));
    }
    #[test]
    fn root_state_samples_d_then_left_control_epoch_and_release_clears_epoch() {
        let event=RootControlEvent{time_s:1.0,mode:RootControlMode::Kinematic,pose_binding:Some(PoseBinding::RigidDeltaV1)};
        let mut root=reference_track(RootMotionMode::Dynamic,vec![event]);
        root.keys.truncate(3);
        let raw=root.clone();let mut state=RootState::new(3,vec![root]).unwrap();
        let attempt=state.begin(0.0,1.0).unwrap();
        assert!(attempt.body_indices.is_empty());
        state.commit(&attempt,1.0,&[],0.0,0.0).unwrap();
        let actual=RootPose{center_world_m:[5.0,-2.0,3.0],rotation_world_wxyz:[1.0,0.0,0.0,0.0]};
        state.apply_controls(&[(0,actual)],0.0,0.0).unwrap();
        assert_eq!(state.root_pose(0,1.0).unwrap(),actual);
        let epoch=compose_pose(actual,inverse_pose(raw.sample_native(1.0).unwrap()).unwrap()).unwrap();
        assert_eq!(state.root_pose(0,1.5).unwrap(),compose_pose(epoch,raw.sample_native(1.5).unwrap()).unwrap());
        state.release(0).unwrap();
        assert!(state.active_indices().is_empty());
        assert_eq!(state.root_pose(0,1.5).unwrap(),raw.sample_native(1.5).unwrap());
    }
    #[test]
    fn static_reference_is_held_in_native_frame_and_sweep_radius_includes_d_offset() {
        let mut static_root=reference_track(RootMotionMode::Static,vec![]);static_root.keys.truncate(1);
        let state=RootState::new(4,vec![static_root.clone()]).unwrap();
        let initial=static_root.sample_native(0.0).unwrap();
        assert_eq!(state.root_pose(0,0.5).unwrap(),initial);
        assert_eq!(state.effective_roots().unwrap()[0].sample_native(0.5).unwrap(),initial);
        let rotating=reference_track(RootMotionMode::Kinematic,vec![]);
        let sweep=plan_sweep(&[rotating],&[0.5],0.0,0.5).unwrap();
        let theta=std::f64::consts::FRAC_PI_4;
        let expected=2.0*1.5*((theta/5.0)*0.25).sin().powi(2);
        assert!((sweep[0].arc_margin_m[0]-expected).abs()<1e-12);
    }
    #[test]
    fn absolute_takeover_matches_pd_and_current_hold_keeps_readback() {
        let absolute=RootControlEvent{time_s:1.0,mode:RootControlMode::Static,pose_binding:Some(PoseBinding::Absolute)};
        let mut state=RootState::new(5,vec![reference_track(RootMotionMode::Kinematic,vec![absolute])]).unwrap();
        let attempt=state.begin(0.0,1.0).unwrap();let target=attempt.target[0];
        state.commit(&attempt,1.0,&[target],0.0,0.0).unwrap();
        state.apply_controls(&[(0,target)],0.0,0.0).unwrap();
        assert_eq!(state.root_pose(0,1.5).unwrap(),target);

        let hold=RootControlEvent{time_s:1.0,mode:RootControlMode::Static,pose_binding:Some(PoseBinding::CurrentHold)};
        let mut state=RootState::new(6,vec![reference_track(RootMotionMode::Dynamic,vec![hold])]).unwrap();
        let attempt=state.begin(0.0,1.0).unwrap();
        state.commit(&attempt,1.0,&[],0.0,0.0).unwrap();
        let actual=RootPose{center_world_m:[8.0,3.0,-2.0],rotation_world_wxyz:[1.0,0.0,0.0,0.0]};
        state.apply_controls(&[(0,actual)],0.0,0.0).unwrap();
        assert_eq!(state.root_pose(0,1.5).unwrap(),actual);
    }
}

/// Convert a key-bounded physical duration to the actual fp32 solver dt, toward
/// zero rather than rounding past a key. The attempt samples at T0+f64(dt), the
/// SAME clock the driver commits. A vanished duration is a typed refusal.
pub fn native_dt_before_key(start_s:f64, requested_dt:f32, roots:&[ValidatedRootMotion]) -> RootResult<f32> {
    if !start_s.is_finite() || start_s<0.0 || !requested_dt.is_finite() || requested_dt<=0.0 {
        return Err(fail(CODE_TIME,"invalid native step clock"));
    }
    let mut allowed=f64::from(requested_dt);
    for root in roots {
        if let Some(key)=root.keys.iter().find(|k|k.time_s>start_s) {allowed=allowed.min(key.time_s-start_s);}
    }
    let mut dt=allowed as f32;
    if f64::from(dt)>allowed {dt=f32::from_bits(dt.to_bits()-1);}
    if !dt.is_finite() || dt<=0.0 || start_s+f64::from(dt)<=start_s {
        return Err(fail(CODE_TIME,"key-bounded native dt cannot advance the physical clock"));
    }
    Ok(dt)
}

// A fixed rigid epoch is applied to targets and every certified curve sample.
fn qmul(a:[f64;4],b:[f64;4])->[f64;4] {
    let [w,x,y,z]=a;let [v,i,j,k]=b;
    [w*v-x*i-y*j-z*k,w*i+x*v+y*k-z*j,w*j-x*k+y*v+z*i,w*k+x*j-y*i+z*v]
}
fn compose_pose(a:RootPose,b:RootPose)->RootResult<RootPose> {
    let r=rotate_world(a.rotation_world_wxyz,b.center_world_m);
    let mut q=qmul(a.rotation_world_wxyz,b.rotation_world_wxyz);
    let norm=q.iter().map(|v|v*v).sum::<f64>().sqrt();
    let center=std::array::from_fn(|i|a.center_world_m[i]+r[i]);
    if !norm.is_finite() || norm<=0.0 || center.iter().any(|v|!v.is_finite()) {return Err(fail(CODE_INITIAL_POSE,"rigid rebase overflow"));}
    for v in &mut q {*v/=norm;}
    Ok(RootPose{center_world_m:center,rotation_world_wxyz:q})
}
fn inverse_pose(a:RootPose)->RootResult<RootPose> {
    let [w,x,y,z]=a.rotation_world_wxyz;let q=[w,-x,-y,-z];
    let center=rotate_world(q,a.center_world_m.map(|v|-v));
    if center.iter().any(|v|!v.is_finite()) {return Err(fail(CODE_INITIAL_POSE,"inverse rebase overflow"));}
    Ok(RootPose{center_world_m:center,rotation_world_wxyz:q})
}

#[derive(Debug)]
#[derive(Clone)]
pub struct RootState {
    epochs: Vec<Option<RootPose>>, modes: Vec<RootControlMode>, held: Vec<RootPose>, control_cursor: Vec<usize>,
    roots: Vec<ValidatedRootMotion>, active: Vec<bool>,
    generation: u64, next_sequence: u64, pending: Option<Attempt>,
    pub committed_time_s: f64, pub previous_time_s: f64,
    pub commit_sequence: u64,
}
impl RootState {
    /// Restore accepted physics after a failed in-process trial while keeping
    /// attempt identifiers monotone within this same launch.
    pub(crate) fn restore_trial_snapshot(&mut self,mut before:Self)->Result<(),String>{
        if self.generation!=before.generation{return Err("root trial generation changed".into());}
        before.next_sequence=before.next_sequence.max(self.next_sequence);
        *self=before;Ok(())
    }
    /// Immutable transaction identity for root+hard frozen inverse lifetime.
    pub fn hard_epoch(&self)->(u64,u64){(self.generation,self.next_sequence)}

    pub fn new(generation: u64, roots: Vec<ValidatedRootMotion>) -> RootResult<Self> {
        if generation==0 || roots.is_empty() || roots.len()>MAX_ROOTS_V2
            || roots.iter().any(|r|r.keys.is_empty())
            || roots.windows(2).any(|p|p[0].runtime_body_index>=p[1].runtime_body_index) {
            return Err(fail(CODE_ROOT_BATCH,"invalid generation or sorted root batch"));
        }
        let held=roots.iter().map(|r|r.sample_native(0.0)).collect::<RootResult<Vec<_>>>()?;
        Ok(Self {epochs:vec![None;roots.len()],modes:roots.iter().map(|r|match r.mode {RootMotionMode::Static=>RootControlMode::Static,RootMotionMode::Kinematic=>RootControlMode::Kinematic,RootMotionMode::Dynamic=>RootControlMode::Dynamic}).collect(),
            held,control_cursor:vec![0;roots.len()],active:roots.iter().map(|r|r.mode!=RootMotionMode::Dynamic).collect(),roots,generation,next_sequence:0,pending:None,
            committed_time_s:0.0,previous_time_s:0.0,commit_sequence:0})
    }
    pub fn abort_trial(&mut self) {self.pending=None;}
    pub fn active_indices(&self)->Vec<u32> {
        self.roots.iter().zip(&self.active).filter(|(_,a)|**a).map(|(r,_)|r.runtime_body_index).collect()
    }
    pub(crate) fn single_kinematic_retry_root(&self)->bool {
        self.roots.len()==1 && self.active[0] && self.modes[0]==RootControlMode::Kinematic
            && !self.rotating() && !self.has_control_events()
    }
    pub fn pose_at_committed(&self,body:u32)->RootResult<RootPose> {
        self.root_pose(body,self.committed_time_s)
    }
    pub fn native_dt(&self,requested_dt:f32)->RootResult<f32> {
        // Released roots no longer constrain physical time to their keys.
        let roots=self.roots.iter().zip(&self.active).filter(|(_,a)|**a).map(|(r,_)|r.clone()).collect::<Vec<_>>();
        native_dt_before_key(self.committed_time_s,requested_dt,&roots)
    }
    pub fn native_step(&self,requested_dt:f32)->RootResult<NativeStep> {
        // Planning-only copies add exact control cuts, including released bodies.
        // They never replace authored keys or the sampler in the transaction.
        let mut cuts=Vec::new();
        for (i,root) in self.roots.iter().enumerate() {
            let mut cut=root.clone();
            if self.modes[i]!=RootControlMode::Kinematic {cut.keys=vec![NormalizedKey{time_s:0.0,pose:self.held[i]}];cut.reference_transform=None;}
            for event in &root.control_events[self.control_cursor[i]..] {
                if event.time_s>self.committed_time_s && !cut.keys.iter().any(|k|k.time_s==event.time_s) {
                    cut.keys.push(NormalizedKey{time_s:event.time_s,pose:self.held[i]});
                }
            }
            cut.keys.sort_by(|a,b|a.time_s.total_cmp(&b.time_s)); cuts.push(cut);
        }
        let mut step=native_step_before_key(self.committed_time_s,requested_dt,&cuts)?;
        let mut angle=0.0_f64;
        for (i,root) in self.roots.iter().enumerate() {
            if self.modes[i]==RootControlMode::Kinematic {angle=angle.max(pose_angle(self.root_pose(root.runtime_body_index,step.start_s)?,self.root_pose(root.runtime_body_index,step.end_s)?));}
        }
        let cap=std::f64::consts::PI/18.0;
        if angle>cap {
            let duration=(step.end_s-step.start_s)*(cap/angle)*(1.0-4.0*f64::from(f32::EPSILON));
            step.end_s=step.start_s+duration;step.integration_dt_s=(step.end_s-step.start_s) as f32;
            step.lands_on_boundary=false;
            if step.end_s<=step.start_s || step.integration_dt_s<=0.0 {return Err(fail(CODE_TIME,"angular substep cannot advance the physical clock"));}
        }
        Ok(step)
    }
    pub fn rotating(&self)->bool {
        self.roots.iter().any(|r|r.keys.iter().any(|k|k.pose.rotation_world_wxyz!=r.keys[0].pose.rotation_world_wxyz))
    }
    /// Retry scope must reject even future, not-yet-due control events.
    pub fn has_control_events(&self)->bool {self.roots.iter().any(|r|!r.control_events.is_empty())}
    pub fn root_pose(&self,body:u32,time:f64)->RootResult<RootPose> {
        let i=self.roots.iter().position(|r|r.runtime_body_index==body).ok_or_else(||fail(CODE_BODY_BINDING,"unknown root"))?;
        if self.modes[i]==RootControlMode::Static {return Ok(self.held[i]);}
        let authored=self.roots[i].sample_native(time)?;
        self.epochs[i].map_or(Ok(authored),|epoch|compose_pose(epoch,authored))
    }
    /// Author-root frame with the existing left-world control epoch applied,
    /// before D. Used only to construct exact per-vertex curve envelopes.
    pub fn author_pose(&self,body:u32,time:f64)->RootResult<RootPose> {
        let i=self.roots.iter().position(|r|r.runtime_body_index==body).ok_or_else(||fail(CODE_BODY_BINDING,"unknown root"))?;
        let author=self.roots[i].sample(time)?;
        self.epochs[i].map_or(Ok(author),|epoch|compose_pose(epoch,author))
    }
    pub fn reference_transform(&self,body:u32)->RootResult<RootPose> {
        let i=self.roots.iter().position(|r|r.runtime_body_index==body).ok_or_else(||fail(CODE_BODY_BINDING,"unknown root"))?;
        Ok(self.roots[i].reference_transform.unwrap_or(RootPose{center_world_m:[0.0;3],rotation_world_wxyz:[1.0,0.0,0.0,0.0]}))
    }
    pub fn curve_frame(&self,body:u32,time:f64)->RootResult<(RootPose,RootPose)> {
        let i=self.roots.iter().position(|r|r.runtime_body_index==body).ok_or_else(||fail(CODE_BODY_BINDING,"unknown root"))?;
        if self.modes[i]==RootControlMode::Static {
            return Ok((self.held[i],RootPose{center_world_m:[0.0;3],rotation_world_wxyz:[1.0,0.0,0.0,0.0]}));
        }
        Ok((self.author_pose(body,time)?,self.reference_transform(body)?))
    }
    pub fn effective_roots(&self)->RootResult<Vec<ValidatedRootMotion>> {
        self.roots.iter().enumerate().filter(|(i,_)|self.active[*i]).map(|(i,r)| {
            let mut effective=r.clone();
            if self.modes[i]==RootControlMode::Static {effective.keys=vec![NormalizedKey{time_s:0.0,pose:self.held[i]}];effective.reference_transform=None;}
            else if let Some(epoch)=self.epochs[i] {
                for key in &mut effective.keys {key.pose=compose_pose(epoch,key.pose)?;}
            }
            Ok(effective)
        }).collect()
    }
    pub fn plan_sweep(&self,radius_m:&[f64],start_s:f64,end_s:f64)->RootResult<Vec<SweepSegment>> {
        plan_sweep(&self.effective_roots()?,radius_m,start_s,end_s)
    }
    pub fn due_control_policies(&self)->Vec<(u32,RootControlMode,Option<PoseBinding>)> {
        self.roots.iter().enumerate().filter_map(|(i,r)|r.control_events.get(self.control_cursor[i])
            .filter(|e|e.time_s==self.committed_time_s).map(|e|(r.runtime_body_index,e.mode,e.pose_binding))).collect()
    }
    pub fn due_controls(&self)->Vec<(u32,RootControlMode)> {
        self.roots.iter().enumerate().filter_map(|(i,r)|r.control_events.get(self.control_cursor[i])
            .filter(|e|e.time_s==self.committed_time_s).map(|e|(r.runtime_body_index,e.mode))).collect()
    }
    /// Atomic batch at an ACCEPTED physical boundary. No pose/history writes.
    pub fn apply_controls(&mut self,actual:&[(u32,RootPose)],tolerance_m:f64,tolerance_rad:f64)->RootResult<()> {
        if self.pending.is_some() || !tolerance_m.is_finite() || tolerance_m<0.0 || !tolerance_rad.is_finite() || tolerance_rad<0.0 {
            return Err(fail(CODE_TIME,"control boundary requires finite tolerance and no pending sweep"));
        }
        let due=self.due_controls();
        if due.len()!=actual.len() || due.iter().zip(actual).any(|((body,_),(b,_))|body!=b) {
            return Err(fail(CODE_BODY_BINDING,"control readback order/count differs from due batch"));
        }
        let mut changes=Vec::new();
        for ((body,mode),(_,pose)) in due.iter().zip(actual) {
            let i=self.roots.iter().position(|r|r.runtime_body_index==*body).unwrap();
            let policy=self.roots[i].control_events[self.control_cursor[i]].pose_binding.unwrap_or(PoseBinding::Absolute);
            let target=match policy {
                PoseBinding::CurrentHold|PoseBinding::RigidDeltaV1=>*pose,
                PoseBinding::Absolute=>self.roots[i].sample_native(self.committed_time_s)?,
            };
            let norm=pose.rotation_world_wxyz.iter().map(|x|x*x).sum::<f64>();
            if pose.center_world_m.iter().chain(pose.rotation_world_wxyz.iter()).any(|x|!x.is_finite()) || (norm-1.0).abs()>1e-6 {
                return Err(fail(CODE_INITIAL_POSE,"mode boundary native readback is invalid"));
            }
            if *mode!=RootControlMode::Dynamic && (pose.center_world_m.iter().zip(target.center_world_m).any(|(a,b)|(a-b).abs()>tolerance_m) || pose_angle(*pose,target)>tolerance_rad) {
                return Err(fail(CODE_INITIAL_POSE,"takeover requires authored current pose; teleport is forbidden"));
            }
            let epoch=if policy==PoseBinding::RigidDeltaV1 {
                Some(compose_pose(*pose,inverse_pose(self.roots[i].sample_native(self.committed_time_s)?)?)?)
            } else {None};
            changes.push((i,*mode,target,epoch));
        }
        for (i,mode,target,epoch) in changes {self.epochs[i]=epoch;self.modes[i]=mode;self.active[i]=mode!=RootControlMode::Dynamic;self.held[i]=target;self.control_cursor[i]+=1;}
        Ok(())
    }
    pub fn begin_native(&mut self,step:NativeStep)->RootResult<Attempt> {
        if step!=self.native_step(step.requested_dt_s)? {
            return Err(fail(CODE_TIME,"native event plan differs from committed clock or active roots"));
        }
        let mut attempt=self.begin(step.start_s,step.end_s-step.start_s)?;
        if attempt.end_s!=step.end_s {return Err(fail(CODE_TIME,"native event endpoint changed"));}
        attempt.integration_dt_s=Some(step.integration_dt_s);
        self.pending=Some(attempt.clone());
        Ok(attempt)
    }
    /// An additional authenticated controller may bound this root's planned
    /// interval from inside. The original strict entry above is unchanged.
    pub fn begin_native_bounded(&mut self,step:NativeStep)->RootResult<Attempt> {
        let planned=self.native_step(step.requested_dt_s)?;
        if self.pending.is_some()||step.start_s!=self.committed_time_s||step.start_s!=planned.start_s
            ||!step.end_s.is_finite()||step.end_s<=step.start_s||step.end_s>planned.end_s
            ||!step.integration_dt_s.is_finite()||step.integration_dt_s<=0.||step.integration_dt_s>planned.integration_dt_s
            ||(step.end_s-step.start_s) as f32!=step.integration_dt_s
            ||!native_duration_matches(step.end_s-step.start_s,step.integration_dt_s) {
            return Err(fail(CODE_TIME,"shared native endpoint is outside original root plan"));
        }
        let result=self.begin(step.start_s,step.end_s-step.start_s);
        match result {
            Ok(mut attempt) if attempt.end_s==step.end_s=>{attempt.integration_dt_s=Some(step.integration_dt_s);self.pending=Some(attempt.clone());Ok(attempt)},
            _=>{self.pending=None;Err(fail(CODE_TIME,"shared native begin changed bounded endpoint"))}
        }
    }
    /// The result is in physical seconds. For this first milestone playback
    /// must be exactly one; install gate refuses any other playback value.
    pub fn begin(&mut self, host_time_s: f64, requested_dt_s: f64) -> RootResult<Attempt> {
        if self.pending.is_some() || host_time_s != self.committed_time_s
            || !requested_dt_s.is_finite() || requested_dt_s<=0.0 {
            return Err(fail(CODE_TIME,"attempt must start at the committed clock with no pending attempt"));
        }
        let mut end=host_time_s+requested_dt_s;
        if !end.is_finite() || end<=host_time_s {return Err(fail(CODE_TIME,"unrepresentable attempt duration"));}
        let mut body_indices=Vec::new(); let mut target=Vec::new();
        // Cut at the earliest key over every active root, so one common clock
        // remains within one affine segment for the entire prescribed batch.
        for (i,root) in self.roots.iter().enumerate() {
            if self.modes[i]==RootControlMode::Kinematic {
                if let Some(key)=root.keys.iter().find(|k|k.time_s>host_time_s) {end=end.min(key.time_s);}
            }
            if let Some(event)=root.control_events.get(self.control_cursor[i]) {
                if event.time_s>host_time_s {end=end.min(event.time_s);}
                else {return Err(fail(CODE_TIME,"due control boundary must be read back before beginning a sweep"));}
            }
        }
        for (root,active) in self.roots.iter().zip(&self.active) {
            if *active {body_indices.push(root.runtime_body_index);target.push(self.root_pose(root.runtime_body_index,end)?);}
        }
        self.next_sequence=self.next_sequence.checked_add(1).ok_or_else(||fail(CODE_TIME,"attempt sequence exhausted"))?;
        let attempt=Attempt {generation:self.generation,sequence:self.next_sequence,
            start_s:host_time_s,end_s:end,body_indices,target,integration_dt_s:None};
        self.pending=Some(attempt.clone()); Ok(attempt)
    }
    pub fn reject(&mut self, attempt: &Attempt) -> RootResult<()> {
        self.verify_pending(attempt)?; self.pending=None; Ok(())
    }
    /// Native B3 error-reduction retarget. This is safe ONLY for the first
    /// affine slice: every earlier root sweep is on the same straight prefix.
    /// The native caller still runs a solve/CCD/scan to the new endpoint. Full
    /// rotational trajectories cannot borrow this retarget path.
    pub fn shorten_affine(&mut self, attempt:&Attempt, actual_dt_s:f64)->RootResult<Attempt> {
        require_affine_native(&self.roots)?;
        self.shorten_native(attempt,actual_dt_s)
    }
    /// A curve cut freezes a NEW sampled endpoint; caller must solve and certify
    /// the complete new curve again. An affine-prefix certificate is not reused.
    pub fn shorten_native(&mut self,attempt:&Attempt,actual_dt_s:f64)->RootResult<Attempt> {
        self.verify_pending(attempt)?;
        let mut physical_dt=if let Some(dt)=attempt.integration_dt_s {
            // B3 can rebuild at unchanged dt. Keep the exact event endpoint.
            if actual_dt_s==f64::from(dt) {return Ok(attempt.clone());}
            if actual_dt_s>=f64::from(dt) {return Err(fail(CODE_TIME,"native retarget must shorten integration dt"));}
            let shortened=actual_dt_s as f32;
            if f64::from(shortened)!=actual_dt_s {
                return Err(fail(CODE_TIME,"native retarget dt is not the actual f32 duration"));
            }
            // Preserve the SAME affine sweep fraction. Re-running the nominal
            // event planner here could enlarge the already checked CCD prefix.
            (attempt.end_s-attempt.start_s)*(actual_dt_s/f64::from(dt))
        } else {actual_dt_s};
        let mut end=self.committed_time_s+physical_dt;
        let mapped_end=end;
        // Keep the checked physical prefix. If its nearest f32 duration would
        // enlarge the actual GPU step, take a STRICTER prefix instead; never
        // move its endpoint outward to fit a float representation.
        if attempt.integration_dt_s.is_some() && end.is_finite()
            && end>self.committed_time_s && actual_dt_s>0.0
            && (end-self.committed_time_s) as f32>actual_dt_s as f32 {
            end=end.min(self.committed_time_s+actual_dt_s);
            // Absolute-clock addition can round outward. One inward f64 ULP
            // is sufficient when that boundary is representable; the existing
            // checks below refuse coarse/unrepresentable clocks otherwise.
            if (end-self.committed_time_s) as f32>actual_dt_s as f32 {
                end=f64::from_bits(end.to_bits()-1);
            }
            physical_dt=end-self.committed_time_s;
        }
        // Every consumer uses the representation of this actual endpoint.
        let integration_dt_s=attempt.integration_dt_s.map(|_| (end-self.committed_time_s) as f32);
        if !actual_dt_s.is_finite() || actual_dt_s<=0.0
            || !end.is_finite() || end<=self.committed_time_s || end>attempt.end_s
            || end>mapped_end
            || integration_dt_s.is_some_and(|dt| f64::from(dt)>actual_dt_s || !native_duration_matches(end-self.committed_time_s,dt)) {
            return Err(fail(CODE_TIME,"shortened endpoint must be positive and inside pending sweep"));
        }
        self.pending=None;
        match self.begin(self.committed_time_s,physical_dt) {
            Ok(mut next)=> {
                if next.end_s!=end || next.end_s>mapped_end {
                    self.pending=Some(attempt.clone());
                    return Err(fail(CODE_TIME,"refrozen endpoint differs from bounded physical prefix"));
                }
                next.integration_dt_s=integration_dt_s;
                self.pending=Some(next.clone()); Ok(next)
            }
            Err(error)=> {self.pending=Some(attempt.clone()); Err(error)}
        }
    }
    fn verify_pending(&self, attempt: &Attempt) -> RootResult<()> {
        let Some(p)=&self.pending else {return Err(fail(CODE_COMMIT,"no pending attempt"));};
        if attempt.generation!=self.generation || attempt.sequence!=p.sequence
            || attempt.start_s!=p.start_s || attempt.end_s!=p.end_s
            || attempt.integration_dt_s!=p.integration_dt_s
            || attempt.body_indices!=p.body_indices || attempt.target!=p.target {
            return Err(fail(CODE_COMMIT,"attempt differs from frozen pending state"));
        } Ok(())
    }
    /// `actual` is the GPU fitted root pose, in frozen runtime-body order.
    /// The final commit caller still runs native solve/CCD/intersection gates
    /// at this exact frozen endpoint; this probe advances no accepted state.
    /// Verify the immutable pending transaction and native pose without writes.
    /// Only finite pose residuals may request another same-endpoint solve.
    /// Invalid clocks/counts/tolerances/quaternions remain hard refusals.
    pub fn validate_candidate(&self, attempt: &Attempt, accepted_time_s: f64,
                  actual: &[RootPose], position_tolerance_m: f64, rotation_tolerance_rad: f64) -> RootResult<CandidateValidation> {
        self.verify_pending(attempt)?;
        if accepted_time_s!=attempt.end_s || actual.len()!=attempt.target.len()
            || attempt.body_indices.len()!=attempt.target.len()
            || !position_tolerance_m.is_finite() || position_tolerance_m<0.0
            || !rotation_tolerance_rad.is_finite() || rotation_tolerance_rad<0.0 {
            return Err(fail(CODE_COMMIT,"accepted clock/count/tolerance differs from frozen sweep"));
        }
        let mut needs_refinement=false;
        for (a,b) in actual.iter().zip(&attempt.target) {
            let norm=a.rotation_world_wxyz.iter().map(|v|v*v).sum::<f64>();
            if a.center_world_m.iter().chain(a.rotation_world_wxyz.iter()).any(|v|!v.is_finite())
                || !norm.is_finite() || (norm-1.0).abs()>1e-6 {
                return Err(fail(CODE_COMMIT,"native committed root does not equal swept target"));
            }
            let angle=pose_angle(*a,*b);
            if !angle.is_finite() || a.center_world_m.iter().zip(b.center_world_m).any(|(x,y)|!(x-y).is_finite()) {
                return Err(fail(CODE_COMMIT,"native committed root does not equal swept target"));
            }
            if a.center_world_m.iter().zip(b.center_world_m).any(|(x,y)|(x-y).abs()>position_tolerance_m)
                || angle>rotation_tolerance_rad {needs_refinement=true;}
        }
        if needs_refinement {return Ok(CandidateValidation::NeedsRefinement(fail(CODE_COMMIT,"native committed root does not equal swept target")));}
        Ok(CandidateValidation::Ready)
    }
    pub fn commit(&mut self, attempt: &Attempt, accepted_time_s: f64,
                  actual: &[RootPose], position_tolerance_m: f64, rotation_tolerance_rad: f64) -> RootResult<()> {
        match self.validate_candidate(attempt,accepted_time_s,actual,position_tolerance_m,rotation_tolerance_rad)? {
            CandidateValidation::Ready=>{},
            CandidateValidation::NeedsRefinement(error)=>return Err(error),
        }
        self.previous_time_s=self.committed_time_s; self.committed_time_s=accepted_time_s;
        self.commit_sequence=self.commit_sequence.checked_add(1).ok_or_else(||fail(CODE_TIME,"commit sequence exhausted"))?;
        self.pending=None; Ok(())
    }
    /// Between accepted steps only. Changes ONLY the mask and DOES NOT return
    /// synthetic pose history for the caller to install. Native
    /// velocity stays (committed - previous)/prev_dt, including the actual last
    /// shortened step. Endpoint hold never implicitly releases a root.
    pub fn release(&mut self, body: u32) -> RootResult<()> {
        if self.pending.is_some() {return Err(fail(CODE_TIME,"release during attempt forbidden"));}
        let index=self.roots.iter().position(|r|r.runtime_body_index==body)
            .ok_or_else(||fail(CODE_BODY_BINDING,"unknown release body"))?;
        if !self.active[index] {return Err(fail(CODE_UNSUPPORTED_DOMAIN,"body already released"));}
        self.epochs[index]=None; self.active[index]=false; self.modes[index]=RootControlMode::Dynamic; Ok(())
    }
    /// Re-taking a released body requires readback matching the trajectory's
    /// current pose. A teleport needs its own collision sweep and is refused.
    pub fn takeover(&mut self, body: u32, actual: RootPose, tolerance_m: f64,
                    tolerance_rad: f64) -> RootResult<()> {
        if self.pending.is_some() {return Err(fail(CODE_TIME,"takeover during attempt forbidden"));}
        let index=self.roots.iter().position(|r|r.runtime_body_index==body)
            .ok_or_else(||fail(CODE_BODY_BINDING,"unknown takeover body"))?;
        let target=self.roots[index].sample_native(self.committed_time_s)?;
        if self.active[index] || !tolerance_m.is_finite() || tolerance_m<0.0
            || !tolerance_rad.is_finite() || tolerance_rad<0.0
            || actual.center_world_m.iter().chain(actual.rotation_world_wxyz.iter()).any(|v|!v.is_finite())
            || (actual.rotation_world_wxyz.iter().map(|v|v*v).sum::<f64>()-1.0).abs()>1e-6
            || actual.center_world_m.iter().zip(target.center_world_m).any(|(a,b)|(a-b).abs()>tolerance_m)
            || pose_angle(actual,target)>tolerance_rad {
            return Err(fail(CODE_INITIAL_POSE,"takeover requires current pose match, no teleport"));
        }
        self.epochs[index]=None; self.active[index]=true; self.modes[index]=RootControlMode::Kinematic; Ok(())
    }
}


/// Accepted controller history only. Author programs remain owned by the new runtime.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RootSnapshot {
    pub source_generation: u64,
    pub body_identities: Vec<(String, String, u32)>,
    pub epochs: Vec<Option<RootPose>>, pub modes: Vec<RootControlMode>,
    pub held: Vec<RootPose>, pub control_cursor: Vec<usize>, pub active: Vec<bool>,
    pub committed_time_s: f64, pub previous_time_s: f64, pub commit_sequence: u64,
}
fn checkpoint_pose_valid(p: RootPose)->bool {
    p.center_world_m.iter().chain(p.rotation_world_wxyz.iter()).all(|x|x.is_finite())
        && (p.rotation_world_wxyz.iter().map(|x|x*x).sum::<f64>()-1.).abs()<=1e-6
}
impl RootState {
    pub fn export_checkpoint(&self)->RootResult<RootSnapshot> {
        if self.pending.is_some() {return Err(fail(CODE_COMMIT,"checkpoint requires no pending root attempt"));}
        let s=RootSnapshot {source_generation:self.generation,
            body_identities:self.roots.iter().map(|r|(r.scene_uuid.clone(),r.body_uuid.clone(),r.runtime_body_index)).collect(),
            epochs:self.epochs.clone(),modes:self.modes.clone(),held:self.held.clone(),control_cursor:self.control_cursor.clone(),active:self.active.clone(),
            committed_time_s:self.committed_time_s,previous_time_s:self.previous_time_s,commit_sequence:self.commit_sequence};
        self.validate_checkpoint_history(&s,self.committed_time_s)?; Ok(s)
    }
    fn validate_checkpoint_history(&self,s:&RootSnapshot,accepted_time:f64)->RootResult<()> {
        let invalid=||fail(CODE_COMMIT,"invalid accepted root checkpoint history");
        let n=self.roots.len();
        if s.source_generation==0 || !accepted_time.is_finite() || accepted_time<0. || s.committed_time_s!=accepted_time
            || !s.previous_time_s.is_finite() || s.previous_time_s<0. || s.previous_time_s>s.committed_time_s
            || (s.commit_sequence==0 && (s.committed_time_s!=0. || s.previous_time_s!=0.))
            || (s.commit_sequence>0 && s.previous_time_s>=s.committed_time_s)
            || s.body_identities.len()!=n || s.epochs.len()!=n || s.modes.len()!=n || s.held.len()!=n || s.control_cursor.len()!=n || s.active.len()!=n {return Err(invalid());}
        for (i,r) in self.roots.iter().enumerate() {
            if s.body_identities[i]!=(r.scene_uuid.clone(),r.body_uuid.clone(),r.runtime_body_index)
                || s.control_cursor[i]!=r.control_events.iter().filter(|e|e.time_s<=accepted_time).count()
                || s.active[i]!=(s.modes[i]!=RootControlMode::Dynamic)
                || !checkpoint_pose_valid(s.held[i]) || s.epochs[i].is_some_and(|p|!checkpoint_pose_valid(p))
                || (s.epochs[i].is_some() && s.modes[i]!=RootControlMode::Kinematic) {return Err(invalid());}
            // A rigid-delta epoch can only come from an applied rigid-delta event.
            if s.epochs[i].is_some() && (s.control_cursor[i]==0 || r.control_events[s.control_cursor[i]-1].pose_binding!=Some(PoseBinding::RigidDeltaV1)) {return Err(invalid());}
            if let Some(epoch)=s.epochs[i] {
                let event=&r.control_events[s.control_cursor[i]-1];
                let expected=compose_pose(s.held[i],inverse_pose(r.sample_native(event.time_s)?)?)?;
                if epoch!=expected {return Err(invalid());}
            }
            let authored_mode=if s.control_cursor[i]>0 {r.control_events[s.control_cursor[i]-1].mode} else {match r.mode {RootMotionMode::Static=>RootControlMode::Static,RootMotionMode::Kinematic=>RootControlMode::Kinematic,RootMotionMode::Dynamic=>RootControlMode::Dynamic}};
            // Manual release/takeover can produce Dynamic/Kinematic, never Static.
            if s.modes[i]==RootControlMode::Static && authored_mode!=RootControlMode::Static {return Err(invalid());}
        }
        Ok(())
    }
    pub fn validate_checkpoint(&self,s:&RootSnapshot,accepted_time:f64)->RootResult<()> {
        if self.pending.is_some() || self.generation==s.source_generation {return Err(fail(CODE_COMMIT,"checkpoint import requires no pending attempt and a fresh generation"));}
        self.validate_checkpoint_history(s,accepted_time)
    }
    pub fn import_checkpoint(&mut self,s:RootSnapshot,accepted_time:f64)->RootResult<()> {
        self.validate_checkpoint(&s,accepted_time)?;
        self.epochs=s.epochs;self.modes=s.modes;self.held=s.held;self.control_cursor=s.control_cursor;self.active=s.active;
        self.committed_time_s=s.committed_time_s;self.previous_time_s=s.previous_time_s;self.commit_sequence=s.commit_sequence;
        // The destination generation and monotonically increasing attempt sequence stay local.
        Ok(())
    }
}

#[cfg(test)]
mod checkpoint_tests {
    use super::*;
    fn root()->ValidatedRootMotion {
        let pose=RootPose{center_world_m:[0.;3],rotation_world_wxyz:[1.,0.,0.,0.]};
        ValidatedRootMotion{scene_uuid:"scene".into(),body_uuid:"body".into(),runtime_body_index:0,mode:RootMotionMode::Dynamic,
            keys:vec![NormalizedKey{time_s:0.,pose},NormalizedKey{time_s:1.,pose:RootPose{center_world_m:[1.,0.,0.],..pose}}],
            reference_transform:None,control_events:vec![RootControlEvent{time_s:0.1,mode:RootControlMode::Kinematic,pose_binding:Some(PoseBinding::RigidDeltaV1)}]}
    }
    #[test]fn failed_trial_restore_preserves_accepted_root_and_monotone_attempts(){
        let mut state=RootState::new(1,vec![root()]).unwrap();
        let before=state.clone();let old=state.begin(0.,0.05).unwrap();
        state.restore_trial_snapshot(before).unwrap();
        assert_eq!(state.committed_time_s,0.0);assert_eq!(state.commit_sequence,0);
        let next=state.begin(0.,0.05).unwrap();assert!(next.sequence>old.sequence);
        assert!(state.commit(&old,old.end_s,&old.target,0.,0.).is_err());
        let wrong=RootState::new(2,vec![root()]).unwrap();assert!(state.restore_trial_snapshot(wrong).is_err());
    }
    #[test]fn checkpoint_roundtrip_rebases_tokens_and_preserves_epoch_history(){
        let mut source=RootState::new(1,vec![root()]).unwrap();let a=source.begin(0.,0.1).unwrap();
        assert!(source.export_checkpoint().is_err());source.commit(&a,0.1,&[],0.,0.).unwrap();
        assert!(source.export_checkpoint().is_err(),"event must be applied first");
        let pose=RootPose{center_world_m:[4.,2.,1.],rotation_world_wxyz:[1.,0.,0.,0.]};source.apply_controls(&[(0,pose)],0.,0.).unwrap();
        let snapshot=source.export_checkpoint().unwrap();let bytes=serde_json::to_vec(&snapshot).unwrap();
        let mut target=RootState::new(2,vec![root()]).unwrap();target.import_checkpoint(serde_json::from_slice(&bytes).unwrap(),0.1).unwrap();
        assert_eq!(source.root_pose(0,0.4).unwrap(),target.root_pose(0,0.4).unwrap());assert_eq!(target.generation,2);assert_eq!(target.previous_time_s,0.);assert_eq!(target.commit_sequence,1);
        assert!(target.commit(&a,0.1,&[],0.,0.).is_err());let old=source.begin(0.1,0.1).unwrap();let next=target.begin(0.1,0.1).unwrap();
        assert!(target.commit(&old,old.end_s,&old.target,0.,0.).is_err());target.commit(&next,next.end_s,&next.target,0.,0.).unwrap();
    }
    #[test]fn checkpoint_invalid_fields_and_pending_leave_root_unchanged(){
        let source=RootState::new(1,vec![root()]).unwrap();let good=source.export_checkpoint().unwrap();let mut target=RootState::new(2,vec![root()]).unwrap();
        let before=format!("{:?}",target);
        let mut cases=vec![];let mut b=good.clone();b.held.clear();cases.push(b);let mut b=good.clone();b.held[0].rotation_world_wxyz=[0.;4];cases.push(b);
        let mut b=good.clone();b.held[0].center_world_m[0]=f64::NAN;cases.push(b);let mut b=good.clone();b.control_cursor[0]=1;cases.push(b);
        let mut b=good.clone();b.active[0]=true;cases.push(b);let mut b=good.clone();b.previous_time_s=1.;cases.push(b);let mut b=good.clone();b.source_generation=2;cases.push(b);
        for b in cases {assert!(target.import_checkpoint(b,0.).is_err());assert_eq!(format!("{:?}",target),before);}
        let a=target.begin(0.,0.05).unwrap();assert!(target.import_checkpoint(good,0.).is_err());assert!(target.reject(&a).is_ok());
    }
}
