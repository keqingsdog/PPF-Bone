//! CPU-only runtime horizon policy. This does not edit sealed SimArgs or author keys.
use crate::joint_root_motion_wire::{ValidatedRootMotion,RootMotionMode,RootControlMode};
use crate::fem_control_events::{Body,Mode};

pub const CODE_REQUEST:&str="RESUME_REQUEST_INVALID";
pub const CODE_COVERAGE:&str="RESUME_CONTROL_COVERAGE_INSUFFICIENT";
pub const CODE_GENERIC:&str="RESUME_GENERIC_ANIMATION_COVERAGE_UNPROVEN";
#[derive(Clone,Debug,PartialEq)]
pub struct Request {
    pub load:i32,
    pub sealed_frames:i32,
    /// Final frame index, never an additional-frame count.
    pub effective_frames:i32,
    pub requested_time:Option<f64>,
    pub overridden:bool,
}
fn invalid(detail:&str)->String {format!("{CODE_REQUEST}: {detail}")}
/// `load` has already resolved --load -1. A horizon override is resume-only.
/// Without an override load may equal the sealed final frame (already finished),
/// but loading beyond that sealed horizon requires an explicit override. accepted_time is the actual
/// checkpoint physical clock, not load/fps (native steps can cross a frame).
pub fn validate(load:i32,target:Option<i32>,sealed:i32,fps:f64,accepted_time:Option<f64>)->Result<Request,String> {
    if load<0 {return Err(invalid("load must be a resolved nonnegative checkpoint index"));}
    if !fps.is_finite() || fps<=0. {return Err(invalid("fps must be finite and strictly positive"));}
    if accepted_time.is_some_and(|t|!t.is_finite() || t<0.) {return Err(invalid("accepted physical time must be finite and nonnegative"));}
    let (effective_frames,requested_time)=match target {
        None=>{if load>sealed {return Err(invalid("load exceeds the sealed horizon; an explicit target override is required"));}(sealed,None)},
        Some(target)=>{
            if load==0 || target<=load {return Err(invalid("target requires a resolved nonzero load and must exceed it"));}
            let time=f64::from(target)/fps;
            if !time.is_finite() || time<=0. || accepted_time.is_some_and(|accepted|time<=accepted) {return Err(invalid("target physical time must be finite and strictly after the accepted checkpoint"));}
            (target,Some(time))
        }
    };
    Ok(Request{load,sealed_frames:sealed,effective_frames,requested_time,overridden:target.is_some()})
}

/// These validated PDRD/FEM types are produced only by the strict `hold`
/// extrapolation parsers. Their existing sampler defines the entire sealed
/// tail. This executes that program; it does not claim newly sampled author
/// motion. Frontend author-tail checks remain required for live Blender input.
pub fn validate_control_coverage(roots:&[ValidatedRootMotion],bodies:&[Body],requested_time:f64)->Result<(),String> {
    if !requested_time.is_finite() || requested_time<0. {return Err(format!("{CODE_COVERAGE}: target physical time is invalid"));}
    for r in roots {
        kinematic_required_end(r.mode==RootMotionMode::Kinematic,r.control_events.iter().map(|e|(e.time_s,e.mode==RootControlMode::Kinematic)),requested_time)?;
        check_keys(&r.body_uuid,r.keys.iter().map(|k|k.time_s),None)?;
    }
    for b in bodies {
        kinematic_required_end(b.initial_mode==Mode::Kinematic,b.events.iter().map(|e|(e.time_s,e.mode==Mode::Kinematic)),requested_time)?;
        check_keys(&b.body_uuid,b.keys.iter().map(|k|k.time_s),None)?;
    }
    Ok(())
}
/// The ordinary strict extension parser authenticates the schema first. A
/// multi-key sampled angular drive also needs authored coverage; linear_hold
/// alone cannot prove what an author intended beyond the sealed sampling window.
pub fn validate_angular_drive_coverage(value:Option<&serde_json::Value>,target:f64)->Result<(),String>{
    let Some(value)=value else{return Ok(());};
    let bad=||format!("{CODE_COVERAGE}: invalid sealed angular drive coverage");
    for axis in ["twist_x","swing_y","swing_z"] {
        let Some(modes)=value.get(axis) else{continue;};
        for channel in modes.as_object().ok_or_else(bad)?.values(){
            if channel.get("enabled").and_then(|v|v.as_bool())!=Some(true){continue;}
            if channel.get("interpolation").and_then(|v|v.as_str())!=Some("linear_hold"){return Err(bad());}
            let keys=channel.get("target_keys").and_then(|v|v.as_array()).ok_or_else(bad)?;
            let times=keys.iter().map(|v|v.get("time_s").and_then(|v|v.as_f64()).ok_or_else(bad)).collect::<Result<Vec<_>,_>>()?;
            check_keys(axis,times.iter().copied(),if times.len()>1{Some(target)}else{None})?;
        }
    }
    Ok(())
}
// An interval ending at a STATIC/FREE handoff needs its exact endpoint sample,
// while events after the requested horizon cannot extend this requirement.
fn kinematic_required_end(initial:bool,events:impl Iterator<Item=(f64,bool)>,target:f64)->Result<Option<f64>,String> {
    let mut active=initial;let mut required=None;let mut previous=0.;
    for (time,next_active) in events {
        if !time.is_finite() || time<=previous {return Err(format!("{CODE_COVERAGE}: invalid sealed control event clock"));}
        previous=time;
        if time>target {continue;}
        if active {required=Some(time);}
        active=next_active;
    }
    if active {required=Some(target);}
    Ok(required)
}
fn check_keys(body:&str,keys:impl Iterator<Item=f64>,required:Option<f64>)->Result<(),String> {
    let keys=keys.collect::<Vec<_>>();
    if keys.is_empty() || keys[0]!=0. || keys.iter().any(|t|!t.is_finite() || *t<0.) || keys.windows(2).any(|w|w[0]>=w[1]) {return Err(format!("{CODE_COVERAGE}: body {body} has invalid sealed key times"));}
    if let Some(end)=required {if *keys.last().unwrap()<end {return Err(format!("{CODE_COVERAGE}: body {body} KINEMATIC interval endpoint {end} exceeds sealed author endpoint {}",keys.last().unwrap()));}}
    Ok(())
}
/// Callers must inspect non-root animation separately before applying an
/// override. Passing true is a typed refusal until that coverage is proven.
pub fn validate_generic_animation_coverage(has_unverified_generic_animation:bool)->Result<(),String> {
    if has_unverified_generic_animation {Err(format!("{CODE_GENERIC}: non-root scene animation needs an upper-layer sealed coverage check"))}else{Ok(())}
}

#[cfg(test)]
mod tests {
    #[test]fn sampled_drive_coverage_and_constant_hold(){
        let multi=serde_json::json!({"coordinate_model":"parent_local_swing_twist_x_v1","swing_y":{"position":{"enabled":true,"interpolation":"linear_hold","target_keys":[{"time_s":0.,"target_rad":0.},{"time_s":1.,"target_rad":0.2}]}}});
        assert!(super::validate_angular_drive_coverage(Some(&multi),1.).is_ok());assert!(super::validate_angular_drive_coverage(Some(&multi),2.).is_err());
        let mut constant=multi.clone();constant["swing_y"]["position"]["target_keys"].as_array_mut().unwrap().truncate(1);assert!(super::validate_angular_drive_coverage(Some(&constant),2.).is_ok());
    }
    use super::*;
    use crate::joint_root_motion_wire::{RootPose,NormalizedKey,RootControlEvent,PoseBinding};
    use crate::fem_control_events::{Event,Binding};
    fn pose()->RootPose {RootPose{center_world_m:[0.;3],rotation_world_wxyz:[1.,0.,0.,0.]}}
    fn root(mode:RootMotionMode)->ValidatedRootMotion {ValidatedRootMotion{scene_uuid:"scene".into(),body_uuid:"root".into(),runtime_body_index:0,mode,
        keys:vec![NormalizedKey{time_s:0.,pose:pose()},NormalizedKey{time_s:1.,pose:pose()}],control_events:vec![],reference_transform:None}}
    fn body(mode:Mode)->Body {Body{body_uuid:"fem".into(),vertices:vec![0,1,2,3],initial_mode:mode,
        keys:vec![NormalizedKey{time_s:0.,pose:pose()},NormalizedKey{time_s:1.,pose:pose()}],reference:pose(),events:vec![]}}
    #[test]fn override_resolved_nonzero_load_final_frame_and_actual_clock(){
        let r=validate(30,Some(60),30,30.,Some(1.01)).unwrap();assert_eq!(r.effective_frames,60);assert_eq!(r.sealed_frames,30);assert_eq!(r.requested_time,Some(2.));assert!(r.overridden);
        for (load,target) in [(0,60),(-1,60),(30,30),(30,29),(30,-1)] {assert!(validate(load,Some(target),30,30.,None).is_err());}
        assert!(validate(30,Some(31),30,30.,Some(1.04)).is_err());
        assert!(validate(30,Some(60),30,30.,Some(2.)).is_err());
        assert!(validate(30,Some(60),30,30.,Some(f64::NAN)).is_err());
        for fps in [0.,-1.,f64::NAN,f64::INFINITY,f64::MIN_POSITIVE] {assert!(validate(30,Some(60),30,fps,None).is_err());}
    }
    #[test]fn no_override_preserves_fresh_and_ordinary_terminal_semantics(){
        for load in [0,30] {let r=validate(load,None,30,30.,Some(2.)).unwrap();assert_eq!(r.effective_frames,30);assert_eq!(r.requested_time,None);assert!(!r.overridden);}
        assert!(validate(40,None,30,30.,None).unwrap_err().contains("explicit target override"));
        assert!(validate(0,None,0,0.,None).is_err());assert!(validate(0,None,0,-1.,None).is_err());
        assert!(validate(30,Some(40),30,30.,Some(1.)).is_ok());
    }
    #[test]fn both_actual_sealed_samplers_hold_last_pose_through_199_at_60fps(){
        let last=179./60.;let target=199./60.;let mut r=root(RootMotionMode::Kinematic);let mut b=body(Mode::Kinematic);
        let end=RootPose{center_world_m:[1.,2.,3.],rotation_world_wxyz:[1.,0.,0.,0.]};
        r.keys[1]=NormalizedKey{time_s:last,pose:end};b.keys[1]=NormalizedKey{time_s:last,pose:end};
        assert_eq!(r.sample_native(last).unwrap(),r.sample_native(target).unwrap());assert_eq!(b.sample(last).unwrap(),b.sample(target).unwrap());
        assert_eq!(r.sample_native(target).unwrap(),end);assert_eq!(b.sample(target).unwrap(),end);
        validate_control_coverage(&[r],&[b],target).unwrap();
    }
    #[test]fn static_dynamic_hold_and_kinematic_event_boundary(){
        for mode in [RootMotionMode::Static,RootMotionMode::Dynamic] {assert!(validate_control_coverage(&[root(mode)],&[],2.).is_ok());}
        for mode in [Mode::Static,Mode::Free] {assert!(validate_control_coverage(&[],&[body(mode)],2.).is_ok());}
        let mut r=root(RootMotionMode::Dynamic);r.control_events.push(RootControlEvent{time_s:2.,mode:RootControlMode::Kinematic,pose_binding:Some(PoseBinding::RigidDeltaV1)});
        assert!(validate_control_coverage(&[r.clone()],&[],1.5).is_ok());assert!(validate_control_coverage(&[r],&[],2.).is_ok());
        let mut b=body(Mode::Static);b.events.push(Event{time_s:1.5,mode:Mode::Kinematic,binding:Some(Binding::CurrentFollow)});assert!(validate_control_coverage(&[],&[b],2.).is_ok());
    }
    #[test]fn sealed_future_events_execute_after_the_last_pose_key(){
        let mut r=root(RootMotionMode::Kinematic);r.control_events=vec![RootControlEvent{time_s:1.5,mode:RootControlMode::Static,pose_binding:Some(PoseBinding::CurrentHold)}];
        let mut b=body(Mode::Kinematic);b.events=vec![Event{time_s:1.5,mode:Mode::Free,binding:None}];
        validate_control_coverage(&[r.clone()],&[b.clone()],2.).unwrap();
        let mut root=crate::joint_root_motion_state::RootState::new(31,vec![r]).unwrap();
        for start in [0.,1.] {let a=root.begin(start,1.).unwrap();let end=a.end_s;let actual=a.target.clone();root.commit(&a,end,&actual,0.,0.).unwrap();}
        root.apply_controls(&[(0,pose())],0.,0.).unwrap();let saved=root.export_checkpoint().unwrap();assert_eq!(saved.control_cursor,[1]);assert_eq!(saved.modes,[RootControlMode::Static]);
        let xyz=[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]];
        let mut fem=crate::fem_control_events::Runtime::new(vec![b],&xyz,"tail-event-cpu".into()).unwrap();
        let a=fem.begin(1.5,1.5).unwrap();fem.accept(&a,&xyz,|_|Ok(pose())).unwrap();assert_eq!(fem.states[0].cursor,1);assert_eq!(fem.states[0].mode,Mode::Free);assert_eq!(fem.bodies[0].events.len(),1);
    }
    #[test]fn requested_horizon_does_not_invent_next_step_or_retime_key(){
        let mut r=root(RootMotionMode::Kinematic);r.keys[1].time_s=0.035;
        let p=crate::joint_root_motion_state::native_step_before_key(0.03,0.01,&[r.clone()]).unwrap();assert_eq!(p.end_s,0.035);
        validate_control_coverage(&[r],&[],0.035).unwrap();assert!(p.integration_dt_s<0.01);
        assert!(validate_generic_animation_coverage(true).unwrap_err().starts_with(CODE_GENERIC));assert!(validate_generic_animation_coverage(false).is_ok());
    }
}

#[cfg(test)]
mod interval_tests {
    use super::*;
    use crate::joint_root_motion_wire::{NormalizedKey,RootPose,RootControlEvent,PoseBinding};
    use crate::fem_control_events::{Event,Binding};
    fn pose()->RootPose {RootPose{center_world_m:[0.;3],rotation_world_wxyz:[1.,0.,0.,0.]}}
    fn keys(end:f64)->Vec<NormalizedKey>{vec![NormalizedKey{time_s:0.,pose:pose()},NormalizedKey{time_s:end,pose:pose()}]}
    #[test]fn ended_kinematic_interval_only_requires_its_handoff_endpoint(){
        let mut r=ValidatedRootMotion{scene_uuid:"scene".into(),body_uuid:"root".into(),runtime_body_index:0,mode:RootMotionMode::Kinematic,keys:keys(1.),
            control_events:vec![RootControlEvent{time_s:1.,mode:RootControlMode::Static,pose_binding:Some(PoseBinding::CurrentHold)}],reference_transform:None};
        validate_control_coverage(&[r.clone()],&[],2.).unwrap();r.keys=keys(0.9);assert!(validate_control_coverage(&[r],&[],2.).is_ok());
        let mut b=Body{body_uuid:"fem".into(),vertices:vec![0,1,2,3],initial_mode:Mode::Static,keys:keys(1.8),reference:pose(),events:vec![
            Event{time_s:1.5,mode:Mode::Kinematic,binding:Some(Binding::CurrentFollow)},Event{time_s:1.8,mode:Mode::Static,binding:Some(Binding::CurrentHold)},
            Event{time_s:3.,mode:Mode::Kinematic,binding:Some(Binding::CurrentFollow)}]};
        validate_control_coverage(&[],&[b.clone()],2.).unwrap();b.keys=keys(1.7);assert!(validate_control_coverage(&[],&[b],2.).is_ok());
    }
    #[test]fn interval_mode_at_exact_target_still_needs_the_endpoint_sample(){
        assert_eq!(kinematic_required_end(false,[(1.,true),(2.,false)].into_iter(),1.).unwrap(),Some(1.));
        assert_eq!(kinematic_required_end(true,[(1.,false),(3.,true)].into_iter(),2.).unwrap(),Some(1.));
        assert_eq!(kinematic_required_end(false,[(3.,true)].into_iter(),2.).unwrap(),None);
        assert_eq!(kinematic_required_end(true,[(1.,false),(1.5,true),(1.8,false)].into_iter(),2.).unwrap(),Some(1.8));
    }
}
