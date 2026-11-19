//! Whole-node FEM control program and accepted-boundary transaction state.
//! No solver position/history writes and no PDRD body-index namespace.
use crate::joint_root_motion_wire::{RootPose,NormalizedKey,sample_normalized_keys,compose_root_pose,inverse_root_pose,rotate_world};
use serde::{Deserialize,Serialize};
use std::sync::Arc;
pub const FILE:&str="bin/fem_body_controls_v2.json";
pub const FEATURE:&str="fem_root_control_events_v1";
type Result<T>=std::result::Result<T,String>;
#[derive(Clone,Copy,Debug,PartialEq,Eq,Deserialize,Serialize)]
pub enum Mode {#[serde(rename="FREE")] Free,#[serde(rename="STATIC")] Static,#[serde(rename="KINEMATIC")] Kinematic}
#[derive(Clone,Copy,Debug,PartialEq,Eq,Deserialize,Serialize)]
#[serde(rename_all="snake_case")]
pub enum Binding {CurrentHold,CurrentFollow,Absolute}
#[derive(Clone,Debug,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Key {pub time_s:f64,pub center_world_m:[f64;3],pub rotation_world_wxyz:[f64;4]}
#[derive(Clone,Debug,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reference {pub translation_local_m:[f64;3],pub rotation_local_wxyz:[f64;4]}
#[derive(Clone,Debug,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventWire {pub time_s:f64,pub mode:Mode, pub pose_binding:serde_json::Value}
#[derive(Clone,Debug,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BodyWire {pub body_uuid:String,pub source:String,pub vertex_indices:Vec<u32>,pub mode:Mode,pub keys:Vec<Key>,pub reference_transform:Reference,pub control_events:Vec<EventWire>}
#[derive(Clone,Debug,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramWire {pub version:u32,pub kind:String,pub scene_uuid:String,pub interpolation:String,pub extrapolation:String,pub bodies:Vec<BodyWire>}
#[derive(Clone,Debug)]
pub struct Event {pub time_s:f64,pub mode:Mode,pub binding:Option<Binding>}
#[derive(Clone,Debug)]
pub struct Body {pub body_uuid:String,pub vertices:Vec<u32>,pub initial_mode:Mode,pub keys:Vec<NormalizedKey>,pub reference:RootPose,pub events:Vec<Event>}
fn finite(x:f64)->bool {x.is_finite()&&(x as f32).is_finite()&&(x==0.||(x as f32)!=0.)}
fn pose(t:[f64;3],mut q:[f64;4])->Result<RootPose>{
    let norm=q.iter().map(|v|v*v).sum::<f64>().sqrt();
    if !t.iter().chain(q.iter()).all(|v|finite(*v))||(norm-1.).abs()>1e-6 {return Err("finite representable pose with normalized quaternion required".into());}
    for v in &mut q {*v/=norm;}
    Ok(RootPose{center_world_m:t,rotation_world_wxyz:q})
}
fn canonical_uuid(s:&str)->bool{s!="00000000-0000-0000-0000-000000000000"&&s.len()==36&&s.bytes().enumerate().all(|(i,c)|if [8,13,18,23].contains(&i){c==b'-'}else{c.is_ascii_digit()||(b'a'..=b'f').contains(&c)})}
impl ProgramWire {
    /// Identity/topology and user-pin membership are additionally authenticated
    /// by the sealed scene binder; this validates only the closed program.
    pub fn validate(self,scene_uuid:&str)->Result<Vec<Body>> {
        if self.version!=2||self.kind!="ppf.fem_body_controls"||self.scene_uuid!=scene_uuid||self.interpolation!="linear_center_shortest_slerp_v1"||self.extrapolation!="hold"||self.bodies.is_empty() {return Err("FEM event program identity mismatch".into());}
        if self.bodies.len()>128||self.bodies.iter().map(|b|b.keys.len()).sum::<usize>()>65536{return Err("root/key capacity exceeded".into());}
        let mut result=Vec::new();let mut previous=String::new();
        for b in self.bodies {
            if !canonical_uuid(&b.body_uuid)||b.body_uuid<=previous||!matches!(b.source.as_str(),"POSE"|"OBJECT")||b.vertex_indices.is_empty()||b.vertex_indices.windows(2).any(|w|w[0]>=w[1]) {return Err("sorted unique FEM root identity and full vertex order required".into());}
            previous=b.body_uuid.clone();
            if b.keys.is_empty()||b.keys.len()>4096||b.keys[0].time_s!=0. {return Err("zero-start bounded nonempty author keys required".into());}
            let mut keys:Vec<NormalizedKey>=Vec::new();let mut last=-1.;
            for k in b.keys {
                if !k.time_s.is_finite()||k.time_s<0.||k.time_s<=last{return Err("strict finite author key times required".into());}last=k.time_s;
                let mut p=pose(k.center_world_m,k.rotation_world_wxyz)?;
                if let Some(previous)=keys.last(){let mut dot: f64=previous.pose.rotation_world_wxyz.iter().zip(p.rotation_world_wxyz).map(|(a,b)|a*b).sum();if dot<0.{for v in &mut p.rotation_world_wxyz{*v = -*v;}dot = -dot;}if dot<=1e-8{return Err("ambiguous adjacent half-turn".into());}}
                keys.push(NormalizedKey{time_s:k.time_s,pose:p});
            }
            let reference=pose(b.reference_transform.translation_local_m,b.reference_transform.rotation_local_wxyz)?;
            if b.control_events.len()>4096 {return Err("event capacity exceeded".into());}
            let mut mode=b.mode;let mut last=0.;let mut events=Vec::new();
            for e in b.control_events {
                if !e.time_s.is_finite()||e.time_s<=last||e.mode==mode{return Err("strict positive event times and actual mode changes required".into());}
                let binding=match (&e.mode,&e.pose_binding) {
                    (Mode::Free,serde_json::Value::Null)=>None,
                    (Mode::Static|Mode::Kinematic,serde_json::Value::String(s)) if s=="absolute"=>Some(Binding::Absolute),
                    (Mode::Static,serde_json::Value::String(s)) if s=="current_hold"=>Some(Binding::CurrentHold),
                    (Mode::Kinematic,serde_json::Value::String(s)) if s=="current_follow"=>Some(Binding::CurrentFollow),
                    _=>return Err("unsupported FEM event pose binding".into())
                };
                last=e.time_s;mode=e.mode;events.push(Event{time_s:e.time_s,mode,binding});
            }
            if (b.mode==Mode::Kinematic||events.iter().any(|e|e.mode==Mode::Kinematic))&&keys.len()<2 {return Err("KINEMATIC requires at least two author keys".into());}
            if b.mode==Mode::Free&&events.is_empty(){return Err("inert FREE root must not be registered".into());}
            result.push(Body{body_uuid:b.body_uuid,vertices:b.vertex_indices,initial_mode:b.mode,keys,reference,events});
        }
        Ok(result)
    }
}
impl Body {
    pub fn sample(&self,t:f64)->Result<RootPose>{compose_root_pose(sample_normalized_keys(&self.body_uuid,&self.keys,t).map_err(|e|e.to_string())?,self.reference).map_err(|e|e.to_string())}
}
fn transform(p:RootPose,x:[f64;3])->Result<[f64;3]>{let r=rotate_world(p.rotation_world_wxyz,x);let y=std::array::from_fn(|i|r[i]+p.center_world_m[i]);if !y.iter().all(|v|finite(*v)){return Err("FEM target transform overflow".into());}Ok(y)}
#[derive(Clone,Debug,PartialEq,Deserialize,Serialize)]
#[serde(deny_unknown_fields)]
pub struct Capture {pub absolute_hold_targets:Option<Vec<[f64;3]>>,pub positions:Vec<[f64;3]>,pub locals:Vec<[f64;3]>,pub material_frame:Option<RootPose>,pub capture_time:f64,pub epoch_transform:RootPose}
#[derive(Clone,Debug)]
pub struct State {pub mode:Mode,pub binding:Option<Binding>,pub cursor:usize,pub epoch:u64,pub capture:Option<Arc<Capture>>}
#[derive(Clone,Debug)]
pub struct Segment {pub start:f64,pub end:f64,pub states:Vec<State>}
#[derive(Clone,Debug,PartialEq,Deserialize,Serialize)]
#[serde(deny_unknown_fields)]
pub struct ControlSegment {
    pub body_uuid:String,pub segment_mode:Mode,pub segment_pose_binding:Option<Binding>,pub segment_epoch:u64,pub segment_event_cursor:u64,
    pub accepted_end_mode:Mode,pub accepted_end_pose_binding:Option<Binding>,pub accepted_end_epoch:u64,pub accepted_end_event_cursor:u64,
}
pub struct OutputEvidence {pub program_sha256:String,pub rows:Vec<ControlSegment>,pub pins:Vec<(u32,([f64;3],[f64;3],[f64;3]))>}
#[derive(Clone,Debug)]
pub struct Runtime {pub immutable_rest_locals:Vec<Vec<[f64;3]>>,pub bodies:Vec<Body>,pub states:Vec<State>,pub accepted_time:f64,pub last_segment:Option<Segment>,pub launch_id:String,next_sequence:u64,pending:Option<Attempt>}
#[derive(Clone,Debug,PartialEq)]
pub struct Attempt {launch_id:String,sequence:u64,pub start:f64,pub end:f64,pub dt:f32}
impl Runtime {
    /// Restore accepted state without reusing an attempt identifier.
    pub(crate) fn restore_trial_snapshot(&mut self,mut before:Self)->Result<()>{
        if self.launch_id!=before.launch_id{return Err("FEM trial launch identity changed".into());}
        before.next_sequence=before.next_sequence.max(self.next_sequence);
        *self=before;Ok(())
    }
    /// Adopt the canonical endpoint already bounded by the PDRD controller.
    /// Validate the ORIGINAL certified duration, never apply a second inward
    /// endpoint calculation to its rounded integration dt.
    pub fn shorten_shared(&mut self,a:&Attempt,start:f64,end:f64,dt:f32,certified_requested_dt:f32)->Result<Attempt>{
        if self.pending.as_ref()!=Some(a)||start!=a.start||!end.is_finite()||!dt.is_finite()||!certified_requested_dt.is_finite()||dt<=0.||certified_requested_dt<=0.||dt>certified_requested_dt||certified_requested_dt>a.dt{return Err("invalid shared FEM shortening identity/duration".into());}
        if certified_requested_dt==a.dt {if start==a.start&&end==a.end&&dt==a.dt{return Ok(a.clone());}return Err("unchanged shared duration changed endpoint".into());}
        let bound=a.start+(a.end-a.start)*(f64::from(certified_requested_dt)/f64::from(a.dt));
        if end<=start||end>=a.end||end>bound||(end-start) as f32!=dt{return Err("shared FEM endpoint is outside certified original prefix".into());}
        if self.next_sequence==u64::MAX{return Err("FEM attempt sequence overflow".into());}
        self.pending=None;let result=self.begin(end,dt);if result.is_err(){self.pending=Some(a.clone());}result
    }
    pub fn abort_pending(&mut self){self.pending=None;}
    pub fn shorten(&mut self,a:&Attempt,dt:f32)->Result<Attempt>{
        if self.pending.as_ref()!=Some(a)||!dt.is_finite()||dt<=0.||dt>a.dt{return Err("invalid FEM shortening token/duration".into());}
        if dt==a.dt{return Ok(a.clone());}
        let mut end=a.start+(a.end-a.start)*(f64::from(dt)/f64::from(a.dt));
        let bounded=end;
        if (end-a.start) as f32>dt {end=end.min(a.start+f64::from(dt));if (end-a.start) as f32>dt{end=f64::from_bits(end.to_bits()-1);}}
        if end<=a.start||end>a.end{return Err("unrepresentable FEM shortened endpoint".into());}
        if end>bounded{return Err("FEM shortening enlarged certified sweep".into());}
        if self.next_sequence==u64::MAX{return Err("FEM attempt sequence overflow".into());}
        self.pending=None;let result=self.begin(end,(end-a.start) as f32);if result.is_err(){self.pending=Some(a.clone());}result
    }
    pub fn plan(&self,requested:f32,other_boundary:Option<f64>)->Result<crate::joint_root_motion_state::NativeStep>{
        let next=self.bodies.iter().flat_map(|b|b.keys.iter().map(|k|k.time_s).chain(b.events.iter().map(|e|e.time_s))).chain(other_boundary).filter(|t|*t>self.accepted_time).min_by(f64::total_cmp);
        crate::joint_root_motion_state::native_step_before_boundary(self.accepted_time,requested,next,true).map_err(|e|e.to_string())
    }
    /// Initial captures are prepared from immutable full-node rest data by the
    /// authenticated binder. No position or velocity is changed here.
    pub fn new(bodies:Vec<Body>,positions:&[[f64;3]],launch_id:String)->Result<Self>{
        if launch_id.is_empty(){return Err("actual launch identity absent".into());}
        let mut states=Vec::new();let mut immutable_rest_locals=Vec::new();
        for b in &bodies {
            let inv0=inverse_root_pose(b.sample(0.)?).map_err(|e|e.to_string())?;
            immutable_rest_locals.push(b.vertices.iter().map(|v|positions.get(*v as usize).copied().ok_or_else(||"initial FEM vertex out of bounds".into()).and_then(|x|transform(inv0,x))).collect::<Result<Vec<_>>>()?);
            let capture=if b.initial_mode==Mode::Free{None}else{
                // Initial absolute control uses the sealed author proxy B(0).
                // Material Pe defines only a later current_follow epoch.
                let frame=b.sample(0.)?;
                let inv=inverse_root_pose(frame).map_err(|e|e.to_string())?;
                let mut actual=Vec::new();let mut locals=Vec::new();
                for &v in &b.vertices {let x=*positions.get(v as usize).ok_or("initial FEM vertex out of bounds")?;if !x.iter().all(|v|finite(*v)){return Err("initial FEM node nonfinite/unrepresentable".into());}actual.push(x);if b.initial_mode==Mode::Kinematic{locals.push(transform(inv,x)?);}}
                Some(Arc::new(Capture{absolute_hold_targets:None,positions:actual,locals,material_frame:if b.initial_mode==Mode::Kinematic{Some(frame)}else{None},capture_time:0.,epoch_transform:RootPose{center_world_m:[0.;3],rotation_world_wxyz:[1.,0.,0.,0.]}}))
            };
            let state=State{mode:b.initial_mode,binding:if b.initial_mode==Mode::Free{None}else{Some(Binding::Absolute)},cursor:0,epoch:0,capture};
            if let Some(target)=Self::target(b,&state,0.)? {for (&v,y) in b.vertices.iter().zip(target){let x=positions[v as usize];let error=(0..3).map(|i|(y[i]-x[i]).powi(2)).sum::<f64>().sqrt();let norm=x.iter().map(|v|v*v).sum::<f64>().sqrt();if error>2e-6*(1.+norm)||!y.iter().all(|v|finite(*v)){return Err("initial absolute target differs at actual FEM node".into());}}}
            states.push(state);
        }
        Ok(Self{immutable_rest_locals,bodies,states,accepted_time:0.,last_segment:None,launch_id,next_sequence:0,pending:None})
    }
    pub fn target(body:&Body,state:&State,t:f64)->Result<Option<Vec<[f64;3]>>>{
        if state.mode==Mode::Free{return Ok(None);}
        let c=state.capture.as_ref().ok_or("controlled FEM capture absent")?;
        if state.mode==Mode::Static{return Ok(Some(c.absolute_hold_targets.as_ref().unwrap_or(&c.positions).clone()));}
        let p=compose_root_pose(c.epoch_transform,body.sample(t)?).map_err(|e|e.to_string())?;
        c.locals.iter().map(|x|transform(p,*x)).collect::<Result<Vec<_>>>().map(Some)
    }
    /// Planning happens before construction of FixPair targets. Accepted time
    /// remains exact f64 even though integration duration is rounded binary32.
    pub fn begin(&mut self,end:f64,dt:f32)->Result<Attempt>{
        if self.pending.is_some()||!end.is_finite()||end<=self.accepted_time||!dt.is_finite()||dt<=0.||(f64::from(dt)-(end-self.accepted_time)).abs()>f64::from(f32::EPSILON)*(end-self.accepted_time){return Err("invalid FEM trial clock/lifetime".into());}
        for b in &self.bodies {if b.events.iter().any(|e|self.accepted_time<e.time_s&&e.time_s<end){return Err("FEM accepted bracket crosses event".into());}}
        let sequence=self.next_sequence;self.next_sequence=self.next_sequence.checked_add(1).ok_or("FEM attempt sequence overflow")?;
        let a=Attempt{launch_id:self.launch_id.clone(),sequence,start:self.accepted_time,end,dt};self.pending=Some(a.clone());Ok(a)
    }
    pub fn abort(&mut self,a:&Attempt)->Result<()>{if self.pending.as_ref()!=Some(a){return Err("stale FEM attempt token".into());}self.pending=None;Ok(())}
    /// Caller must invoke after C5 commits accepted coordinates, including the
    /// final requested boundary. Prepare every capture before atomic commit.
    /// The returned old segment owns capture snapshots for exact output.
    pub fn accept<F>(&mut self,a:&Attempt,positions:&[[f64;3]],mut material_frame:F)->Result<Segment> where F:FnMut(&str)->Result<RootPose>{
        if self.pending.as_ref()!=Some(a){return Err("stale FEM accept token".into());}
        let segment=Segment{start:a.start,end:a.end,states:self.states.clone()};let mut next=self.states.clone();
        for ((b,s),rest_local) in self.bodies.iter().zip(&mut next).zip(&self.immutable_rest_locals){
            if let Some(e)=b.events.get(s.cursor){
                if e.time_s<a.end{return Err("missed FEM event boundary".into());}
                if e.time_s==a.end {
                    let capture=if e.mode==Mode::Free{None}else{
                        let actual=b.vertices.iter().map(|v|positions.get(*v as usize).copied().filter(|x|x.iter().all(|v|finite(*v))).ok_or_else(||"invalid accepted full-node capture".to_string())).collect::<Result<Vec<_>>>()?;
                        let mut absolute_hold_targets=None;
                        let (locals,epoch_transform,frame)=if e.binding==Some(Binding::Absolute) {
                            let author=b.sample(a.end)?;
                            let targets=rest_local.iter().map(|x|transform(author,*x)).collect::<Result<Vec<_>>>()?;
                            if targets.len()!=actual.len(){return Err("FEM_CONTROL_ABSOLUTE_MISMATCH: full-node extent differs".into());}
                            for ((&vertex,x),y) in b.vertices.iter().zip(&actual).zip(&targets){let error=(0..3).map(|i|(x[i]-y[i]).powi(2)).sum::<f64>().sqrt();let norm=x.iter().map(|v|v*v).sum::<f64>().sqrt();if !error.is_finite()||error>2e-6*(1.+norm){return Err(format!("FEM_CONTROL_ABSOLUTE_MISMATCH: body {} node {} error {}",b.body_uuid,vertex,error));}}
                            if e.mode==Mode::Static{absolute_hold_targets=Some(targets);}
                            (if e.mode==Mode::Kinematic{rest_local.clone()}else{Vec::new()},RootPose{center_world_m:[0.;3],rotation_world_wxyz:[1.,0.,0.,0.]},Some(author))
                        }else if e.mode==Mode::Kinematic {
                            let pe=material_frame(&b.body_uuid)?;let pe=pose(pe.center_world_m,pe.rotation_world_wxyz)?;let inv=inverse_root_pose(pe).map_err(|e|e.to_string())?;
                            let locals=actual.iter().map(|x|transform(inv,*x)).collect::<Result<Vec<_>>>()?;
                            let epoch=compose_root_pose(pe,inverse_root_pose(b.sample(a.end)?).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;(locals,epoch,Some(pe))
                        }else{(Vec::new(),RootPose{center_world_m:[0.;3],rotation_world_wxyz:[1.,0.,0.,0.]},None)};
                        Some(Arc::new(Capture{absolute_hold_targets,positions:actual,locals,material_frame:frame,capture_time:a.end,epoch_transform}))
                    };
                    s.mode=e.mode;s.binding=e.binding;s.cursor+=1;s.epoch=s.epoch.checked_add(1).ok_or("FEM epoch overflow")?;s.capture=capture;
                }
            }
        }
        self.states=next;self.accepted_time=a.end;self.last_segment=Some(segment.clone());self.pending=None;Ok(segment)
    }
    pub fn controlled_vertices(&self)->Vec<u32>{let mut v:Vec<_>=self.bodies.iter().zip(&self.states).filter(|(_,s)|s.mode!=Mode::Free).flat_map(|(b,_)|b.vertices.iter().copied()).collect();v.sort_unstable();v}
    pub fn output(&self,time:f64,start:f64,end:f64,program_sha256:String)->Result<OutputEvidence>{
        if !time.is_finite()||time<start||time>end||self.accepted_time!=end{return Err("FEM output clock differs from accepted state".into());}
        let segment=if start==0.&&end==0.&&self.last_segment.is_none(){Segment{start,end,states:self.states.clone()}}else{self.last_segment.clone().ok_or("FEM accepted segment absent")?};
        if segment.start!=start||segment.end!=end{return Err("FEM output bracket differs from immutable segment".into());}
        let mut rows=Vec::new();let mut pins=Vec::new();
        for ((body,old),right) in self.bodies.iter().zip(&segment.states).zip(&self.states){
            let cursor=body.events.iter().filter(|e|e.time_s<=start).count();let end_cursor=body.events.iter().filter(|e|e.time_s<=end).count();
            if old.cursor!=cursor||right.cursor!=end_cursor||body.events.iter().any(|e|start<e.time_s&&e.time_s<end){return Err("FEM output cursor/bracket mismatch".into());}
            rows.push(ControlSegment{body_uuid:body.body_uuid.clone(),segment_mode:old.mode,segment_pose_binding:old.binding,segment_epoch:old.epoch,segment_event_cursor:old.cursor as u64,accepted_end_mode:right.mode,accepted_end_pose_binding:right.binding,accepted_end_epoch:right.epoch,accepted_end_event_cursor:right.cursor as u64});
            if let Some(at_f)=Self::target(body,old,time)?{let at_p=Self::target(body,old,start)?.ok_or("FEM previous target absent")?;let at_c=Self::target(body,old,end)?.ok_or("FEM end target absent")?;for (((&v,f),p),c) in body.vertices.iter().zip(at_f).zip(at_p).zip(at_c){pins.push((v,(f,p,c)));}}
        }
        Ok(OutputEvidence{program_sha256,rows,pins})
    }
}

#[cfg(test)]mod tests {
    use super::*;
    fn identity()->RootPose{RootPose{center_world_m:[0.;3],rotation_world_wxyz:[1.,0.,0.,0.]}}
    fn body()->Body {Body{body_uuid:"00000000-0000-0000-0000-000000000001".into(),vertices:vec![0,1,2,3],initial_mode:Mode::Free,keys:vec![NormalizedKey{time_s:0.,pose:identity()},NormalizedKey{time_s:1.,pose:RootPose{center_world_m:[1.,0.,0.],..identity()}}],reference:identity(),events:vec![Event{time_s:0.1,mode:Mode::Static,binding:Some(Binding::CurrentHold)},Event{time_s:0.2,mode:Mode::Kinematic,binding:Some(Binding::CurrentFollow)},Event{time_s:0.3,mode:Mode::Free,binding:None}]}}
    fn shape()->Vec<[f64;3]>{vec![[0.,0.,0.],[1.2,0.1,0.],[0.2,1.1,0.3],[0.1,0.2,0.9]]}
    #[test]fn failed_trial_restore_preserves_fem_state_and_monotone_attempts(){
        let x=shape();let mut state=Runtime::new(vec![body()],&x,"launch".into()).unwrap();
        let before=state.clone();let old=state.begin(0.1,0.1).unwrap();
        state.restore_trial_snapshot(before).unwrap();assert_eq!(state.accepted_time,0.0);
        let next=state.begin(0.1,0.1).unwrap();assert!(next.sequence>old.sequence);
        assert!(state.accept(&old,&x,|_|Ok(identity())).is_err());
        let wrong=Runtime::new(vec![body()],&x,"other".into()).unwrap();assert!(state.restore_trial_snapshot(wrong).is_err());
    }
    #[test]fn initial_absolute_uses_author_proxy_not_material_frame(){
        let x=shape();let mut b=body();b.initial_mode=Mode::Kinematic;
        b.reference=RootPose{center_world_m:[2.,3.,4.],rotation_world_wxyz:[0.5f64.sqrt(),0.,0.,0.5f64.sqrt()]};
        let r=Runtime::new(vec![b.clone()],&x,"initial".into()).unwrap();let s=&r.states[0];let c=s.capture.as_ref().unwrap();
        assert_eq!(c.material_frame,Some(b.sample(0.).unwrap()));
        for (actual,target) in x.iter().zip(Runtime::target(&b,s,0.).unwrap().unwrap()){for i in 0..3{assert!((actual[i]-target[i]).abs()<1e-14);}}
        let y=Runtime::target(&b,s,0.05).unwrap().unwrap();for (a,y) in x.iter().zip(y){assert!((y[0]-a[0]-0.05).abs()<1e-14);assert!((y[1]-a[1]).abs()<1e-14);assert!((y[2]-a[2]).abs()<1e-14);}
        b.initial_mode=Mode::Static;let r=Runtime::new(vec![b.clone()],&x,"static".into()).unwrap();let c=r.states[0].capture.as_ref().unwrap();assert!(c.locals.is_empty());assert!(c.material_frame.is_none());assert_eq!(Runtime::target(&b,&r.states[0],0.5).unwrap(),Some(x.clone()));
        let mut invalid=x;invalid[0][0]=f64::INFINITY;assert!(Runtime::new(vec![b],&invalid,"invalid".into()).is_err());
    }
    #[test]fn full_deformed_shape_hold_follow_release_and_old_segment(){
        let x=shape();let mut r=Runtime::new(vec![body()],&x,"test-launch".into()).unwrap();
        let a=r.begin(0.1,0.1).unwrap();let seg=r.accept(&a,&x,|_|Ok(identity())).unwrap();
        assert_eq!(seg.states[0].mode,Mode::Free);assert_eq!(r.states[0].mode,Mode::Static);assert_eq!(r.states[0].cursor,1);
        assert_eq!(Runtime::target(&r.bodies[0],&r.states[0],0.15).unwrap().unwrap(),x);
        let a=r.begin(0.2,0.1).unwrap();let seg=r.accept(&a,&x,|_|Ok(identity())).unwrap();assert_eq!(seg.states[0].mode,Mode::Static);
        let same=Runtime::target(&r.bodies[0],&r.states[0],0.2).unwrap().unwrap();assert_eq!(same,x);
        let moved=Runtime::target(&r.bodies[0],&r.states[0],0.25).unwrap().unwrap();for (y,x) in moved.iter().zip(&x){assert!((y[0]-x[0]-0.05).abs()<1e-14);assert_eq!(y[1],x[1]);assert_eq!(y[2],x[2]);}
        let a=r.begin(0.3,0.1).unwrap();let seg=r.accept(&a,&moved,|_|panic!("release must not measure/fit nodes")).unwrap();assert_eq!(seg.states[0].mode,Mode::Kinematic);assert_eq!(r.states[0].epoch,3);assert!(Runtime::target(&r.bodies[0],&r.states[0],0.4).unwrap().is_none());assert!(r.controlled_vertices().is_empty());assert_eq!(x,shape());
    }
    #[test]fn abort_retry_stale_tokens_and_failed_capture_do_not_commit(){
        let x=shape();let mut r=Runtime::new(vec![body()],&x,"test-launch".into()).unwrap();assert!(r.begin(0.15,0.15).is_err());let a=r.begin(0.1,0.1).unwrap();r.abort(&a).unwrap();let b=r.begin(0.1,0.1).unwrap();assert_ne!(a,b);assert!(r.accept(&a,&x,|_|Ok(identity())).is_err());
        let mut bad=x.clone();bad[3][2]=f64::NAN;assert!(r.accept(&b,&bad,|_|Ok(identity())).is_err());assert_eq!(r.accepted_time,0.);assert_eq!(r.states[0].cursor,0);assert_eq!(r.states[0].epoch,0);r.accept(&b,&x,|_|Ok(identity())).unwrap();assert_eq!(r.accepted_time,0.1);
    }
    #[test]fn simultaneous_boundary_capture_is_atomic_across_bodies(){let x=shape();let mut one=body();one.events[0]=Event{time_s:0.1,mode:Mode::Kinematic,binding:Some(Binding::CurrentFollow)};let mut two=one.clone();two.body_uuid="00000000-0000-0000-0000-000000000002".into();let mut r=Runtime::new(vec![one,two],&x,"test-launch".into()).unwrap();let a=r.begin(0.1,0.1).unwrap();let mut count=0;assert!(r.accept(&a,&x,|_|{count+=1;if count==2{Err("inverted material frame".into())}else{Ok(identity())}}).is_err());assert!(r.states.iter().all(|s|s.cursor==0&&s.epoch==0&&s.mode==Mode::Free));assert_eq!(r.accepted_time,0.);}
    #[test]fn explicit_null_is_required_and_event_schema_is_closed(){
        assert!(serde_json::from_str::<EventWire>(r#"{"time_s":1.0,"mode":"FREE"}"#).is_err());
        assert!(serde_json::from_str::<EventWire>(r#"{"time_s":1.0,"mode":"FREE","pose_binding":null,"extra":0}"#).is_err());
        assert!(serde_json::from_str::<EventWire>(r#"{"time_s":1.0,"mode":"FREE","pose_binding":null,"mode":"STATIC"}"#).is_err());
        assert!(serde_json::from_str::<EventWire>(r#"{"time_s":1.0,"mode":"FREE","pose_binding":null}"#).is_ok());
    }
    fn absolute_body(mode:Mode)->Body{let mut b=body();b.events=vec![Event{time_s:0.1,mode,binding:Some(Binding::Absolute)}];b}
    #[test]fn absolute_static_holds_author_target_separate_from_actual_capture(){
        let x=shape();let b=absolute_body(Mode::Static);let mut r=Runtime::new(vec![b],&x,"abs-static".into()).unwrap();
        let mut actual=x.clone();for p in &mut actual{p[0]+=0.1;}actual[3][1]+=1e-7;
        let unchanged=actual.clone();let a=r.begin(0.1,0.1).unwrap();let seg=r.accept(&a,&actual,|_|panic!("absolute does not use Pe")).unwrap();
        assert_eq!(actual,unchanged);assert_eq!(seg.states[0].mode,Mode::Free);
        let c=r.states[0].capture.as_ref().unwrap();assert_eq!(c.positions,actual);assert!(c.locals.is_empty());assert_eq!(c.material_frame,Some(r.bodies[0].sample(0.1).unwrap()));
        let hold=Runtime::target(&r.bodies[0],&r.states[0],0.8).unwrap().unwrap();assert_eq!(hold,c.absolute_hold_targets.clone().unwrap());assert_ne!(hold,c.positions);
        for (p,y) in x.iter().zip(hold){assert!((y[0]-p[0]-0.1).abs()<1e-14);assert!((y[1]-p[1]).abs()<1e-14);}
    }
    #[test]fn absolute_kinematic_uses_immutable_locals_after_current_capture(){
        let x=shape();let mut b=body();b.events=vec![Event{time_s:0.1,mode:Mode::Static,binding:Some(Binding::CurrentHold)},Event{time_s:0.2,mode:Mode::Kinematic,binding:Some(Binding::Absolute)}];
        let mut r=Runtime::new(vec![b],&x,"abs-after-current".into()).unwrap();let rest=r.immutable_rest_locals.clone();let mut deformed=x.clone();deformed[3][2]+=0.2;
        let a=r.begin(0.1,0.1).unwrap();r.accept(&a,&deformed,|_|panic!()).unwrap();let old=r.states[0].capture.clone().unwrap();
        let a=r.begin(0.2,0.1).unwrap();let before=format!("{:?}",r.states);let err=r.accept(&a,&deformed,|_|panic!()).unwrap_err();assert!(err.starts_with("FEM_CONTROL_ABSOLUTE_MISMATCH:"));assert_eq!(format!("{:?}",r.states),before);assert_eq!(r.accepted_time,0.1);assert_eq!(r.immutable_rest_locals,rest);
        let mut matching=x;for p in &mut matching{p[0]+=0.2;}let segment=r.accept(&a,&matching,|_|panic!()).unwrap();assert!(Arc::ptr_eq(segment.states[0].capture.as_ref().unwrap(),&old));
        let capture=r.states[0].capture.as_ref().unwrap();assert_eq!(capture.locals,rest[0]);assert!(capture.absolute_hold_targets.is_none());assert_eq!(capture.epoch_transform,identity());
        let future=Runtime::target(&r.bodies[0],&r.states[0],0.3).unwrap().unwrap();for (x,y) in matching.iter().zip(future){assert!((y[0]-x[0]-0.1).abs()<1e-14);assert_eq!(y[1],x[1]);assert_eq!(y[2],x[2]);}
    }
    #[test]fn absolute_batch_interior_mismatch_and_nonfinite_are_atomic(){
        let mut x=shape();x.push([0.2,0.2,0.2]);let mut one=absolute_body(Mode::Static);one.vertices.push(4);let mut two=one.clone();two.body_uuid="00000000-0000-0000-0000-000000000002".into();
        let mut r=Runtime::new(vec![one,two],&x,"abs-batch".into()).unwrap();let a=r.begin(0.1,0.1).unwrap();let mut matching=x;for p in &mut matching{p[0]+=0.1;}
        // First body's target matches; only second body's immutable interior
        // reference differs. No partial first-body capture may be committed.
        r.immutable_rest_locals[1][4][2]+=0.02;let before=format!("{:?}",r.states);assert!(r.accept(&a,&matching,|_|panic!()).unwrap_err().contains("node 4"));assert_eq!(format!("{:?}",r.states),before);assert_eq!(r.accepted_time,0.);assert!(r.last_segment.is_none());assert_eq!(r.pending.as_ref(),Some(&a));
        r.immutable_rest_locals[1][4][2]-=0.02;let mut bad=matching.clone();bad[4][1]=f64::NAN;assert!(r.accept(&a,&bad,|_|panic!()).is_err());assert_eq!(format!("{:?}",r.states),before);
        r.accept(&a,&matching,|_|panic!()).unwrap();assert!(r.states.iter().all(|s|s.cursor==1&&s.epoch==1&&s.binding==Some(Binding::Absolute)));
    }
    #[test]fn exact_nonframe_clock_segment_cursors_and_unchanged_shortening(){
        let x=shape();let mut b=body();b.events[0].time_s=0.035;
        let mut r=Runtime::new(vec![b],&x,"clock-launch".into()).unwrap();
        while r.accepted_time<0.035 {let plan=r.plan(0.01,None).unwrap();let a=r.begin(plan.end_s,plan.integration_dt_s).unwrap();let same=r.shorten(&a,a.dt).unwrap();assert_eq!(same,a);r.accept(&a,&x,|_|Ok(identity())).unwrap();}
        assert_eq!(r.accepted_time,0.035);assert_ne!(r.accepted_time,f64::from(0.035f32));
        let last=r.last_segment.as_ref().unwrap();let out=r.output(0.035,last.start,last.end,"hash".into()).unwrap();assert_eq!(out.rows[0].segment_event_cursor,0);assert_eq!(out.rows[0].accepted_end_event_cursor,1);assert_eq!(out.rows[0].segment_mode,Mode::Free);assert_eq!(out.rows[0].accepted_end_mode,Mode::Static);
        let plan=r.plan(0.01,None).unwrap();let a=r.begin(plan.end_s,plan.integration_dt_s).unwrap();r.accept(&a,&x,|_|Ok(identity())).unwrap();let out=r.output(a.start,a.start,a.end,"hash".into()).unwrap();assert_eq!(out.rows[0].segment_event_cursor,1);assert_eq!(out.pins.len(),4);assert!(r.output(a.end,0.,a.end,"hash".into()).is_err());
    }
    #[test]fn launch_namespace_rejects_token_from_other_run(){let x=shape();let mut a=Runtime::new(vec![body()],&x,"one".into()).unwrap();let mut b=Runtime::new(vec![body()],&x,"two".into()).unwrap();let token=a.begin(0.1,0.1).unwrap();b.begin(0.1,0.1).unwrap();assert!(b.accept(&token,&x,|_|Ok(identity())).is_err());assert_eq!(b.accepted_time,0.);}
    #[test]fn current_follow_retains_deformation_under_rotation(){
        let x=shape();let mut b=body();b.events[0]=Event{time_s:0.1,mode:Mode::Kinematic,binding:Some(Binding::CurrentFollow)};b.events.truncate(1);
        let angle=std::f64::consts::FRAC_PI_4;b.keys[1].pose.rotation_world_wxyz=[angle.cos(),0.,0.,angle.sin()];
        let frame=RootPose{center_world_m:[0.3,0.2,-0.1],rotation_world_wxyz:[(angle/2.).cos(),0.,0.,(angle/2.).sin()]};
        let mut r=Runtime::new(vec![b],&x,"rotating".into()).unwrap();let a=r.begin(0.1,0.1).unwrap();r.accept(&a,&x,|_|Ok(frame)).unwrap();
        let same=Runtime::target(&r.bodies[0],&r.states[0],0.1).unwrap().unwrap();let moved=Runtime::target(&r.bodies[0],&r.states[0],0.6).unwrap().unwrap();
        for (a,b) in same.iter().zip(&x){for k in 0..3{assert!((a[k]-b[k]).abs()<2e-14);}}
        for i in 0..4{for j in 0..4{let d=|v:&Vec<[f64;3]>|(0..3).map(|k|(v[i][k]-v[j][k]).powi(2)).sum::<f64>();assert!((d(&moved)-d(&x)).abs()<2e-14);}}
        assert!((moved[1][1]-x[1][1]).abs()>0.1);
    }
    #[test]fn shortened_trial_preserves_certified_physical_prefix(){let x=shape();let mut b=body();b.events[0].time_s=0.035;let mut r=Runtime::new(vec![b],&x,"short".into()).unwrap();let p=r.plan(0.1,None).unwrap();assert_eq!(p.end_s,0.035);let a=r.begin(p.end_s,p.integration_dt_s).unwrap();let dt=(a.dt as f64*0.4) as f32;let next=r.shorten(&a,dt).unwrap();assert!(next.end<=a.start+(a.end-a.start)*f64::from(dt)/f64::from(a.dt));assert!(next.end<a.end);assert_eq!(r.states[0].cursor,0);r.accept(&next,&x,|_|Ok(identity())).unwrap();assert_eq!(r.states[0].mode,Mode::Free);}
    fn pdrd(cut:f64)->crate::joint_root_motion_state::RootState {
        use crate::joint_root_motion_wire::{ValidatedRootMotion,RootMotionMode};
        crate::joint_root_motion_state::RootState::new(1,vec![ValidatedRootMotion{scene_uuid:"scene".into(),body_uuid:"pdrd".into(),runtime_body_index:0,mode:RootMotionMode::Kinematic,keys:vec![NormalizedKey{time_s:0.,pose:identity()},NormalizedKey{time_s:cut,pose:RootPose{center_world_m:[0.01,0.,0.],..identity()}}],control_events:vec![],reference_transform:None}]).unwrap()
    }
    #[test]fn bounded_pdrd_begin_accepts_fem_earlier_event_and_keeps_strict_api(){
        let x=shape();let mut b=body();b.events[0].time_s=0.035;let mut fem=Runtime::new(vec![b],&x,"shared-begin".into()).unwrap();let mut root=pdrd(1.);
        let original=root.native_step(0.1).unwrap();let shared=fem.plan(0.1,Some(original.end_s)).unwrap();assert_eq!(shared.end_s,0.035);assert!(root.begin_native(shared).is_err());
        let mut bad=shared;bad.end_s=original.end_s+1.;assert!(root.begin_native_bounded(bad).is_err());
        bad=shared;bad.start_s=0.001;assert!(root.begin_native_bounded(bad).is_err());
        bad=shared;bad.integration_dt_s=f32::from_bits(shared.integration_dt_s.to_bits()+1);assert!(root.begin_native_bounded(bad).is_err());
        // All refusals left no pending transaction; the exact valid shared
        // entry can now start, and cannot be entered a second time.
        let ra=root.begin_native_bounded(shared).unwrap();let fa=fem.begin(shared.end_s,shared.integration_dt_s).unwrap();assert_eq!(ra.start_s,fa.start);assert_eq!(ra.end_s,fa.end);assert_eq!(ra.integration_dt_s,Some(fa.dt));assert!(root.begin_native_bounded(shared).is_err());root.abort_trial();assert!(root.begin_native(original).is_ok());
    }
    #[test]fn shared_shortening_uses_one_inward_rounded_endpoint_and_refuses_stale(){
        let x=shape();let native=0.01f32;let previous=f32::from_bits(native.to_bits()-1);let cut=(f64::from(native)+f64::from(previous))*0.5+1e-14;assert_eq!(cut as f32,native);
        let mut b=body();b.events[0].time_s=cut;let mut fem=Runtime::new(vec![b],&x,"shared-shrink".into()).unwrap();let mut root=pdrd(cut);let plan=fem.plan(0.02,Some(root.native_step(0.02).unwrap().end_s)).unwrap();let ra=root.begin_native_bounded(plan).unwrap();let fa=fem.begin(plan.end_s,plan.integration_dt_s).unwrap();let certified=0.006f32;let rn=root.shorten_native(&ra,f64::from(certified)).unwrap();let dt=rn.integration_dt_s.unwrap();assert!(dt<certified,"fixture exercises inward duration rounding");
        assert!(fem.shorten_shared(&fa,rn.start_s+1e-4,rn.end_s,dt,certified).is_err());assert!(fem.shorten_shared(&fa,rn.start_s,fa.end,dt,certified).is_err());
        let next=fem.shorten_shared(&fa,rn.start_s,rn.end_s,dt,certified).unwrap();assert_eq!(next.start,rn.start_s);assert_eq!(next.end,rn.end_s);assert_eq!(next.dt,dt);assert!(fem.shorten_shared(&fa,rn.start_s,rn.end_s,dt,certified).is_err());fem.abort(&next).unwrap();assert!(fem.begin(plan.end_s,plan.integration_dt_s).is_ok());assert_eq!(fem.accepted_time,0.);assert_eq!(fem.states[0].cursor,0);
    }
}


#[derive(Clone,Debug,PartialEq,Deserialize,Serialize)]
#[serde(deny_unknown_fields)]
pub struct StateSnapshot {pub mode:Mode,pub binding:Option<Binding>,pub cursor:usize,pub epoch:u64,pub capture:Option<Capture>}
impl StateSnapshot {
    fn from_state(s:&State)->Self {Self{mode:s.mode,binding:s.binding,cursor:s.cursor,epoch:s.epoch,capture:s.capture.as_deref().cloned()}}
    fn into_state(self)->State {State{mode:self.mode,binding:self.binding,cursor:self.cursor,epoch:self.epoch,capture:self.capture.map(Arc::new)}}
}
#[derive(Clone,Debug,PartialEq,Deserialize,Serialize)]
#[serde(deny_unknown_fields)]
pub struct SegmentSnapshot {pub start:f64,pub end:f64,pub states:Vec<StateSnapshot>}
#[derive(Clone,Debug,PartialEq,Deserialize,Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeSnapshot {
    pub source_launch_id:String,pub body_identities:Vec<(String,Vec<u32>)>,
    pub immutable_rest_locals:Vec<Vec<[f64;3]>>,pub states:Vec<StateSnapshot>,
    pub accepted_time:f64,pub last_segment:Option<SegmentSnapshot>,
}
fn checkpoint_points(v:&[[f64;3]],n:usize)->bool {v.len()==n && v.iter().flatten().all(|x|finite(*x))}
fn checkpoint_pose(p:RootPose)->bool {pose(p.center_world_m,p.rotation_world_wxyz).is_ok()}
fn checkpoint_near(a:&[[f64;3]],b:&[[f64;3]])->bool {
    a.len()==b.len() && a.iter().zip(b).all(|(a,b)|a.iter().zip(b).all(|(a,b)|(a-b).abs()<=2e-6*(1.+a.abs().max(b.abs()))))
}
impl Runtime {
    pub fn export_checkpoint(&self)->Result<RuntimeSnapshot> {
        if self.pending.is_some(){return Err("FEM checkpoint requires no pending attempt".into());}
        let s=RuntimeSnapshot{source_launch_id:self.launch_id.clone(),body_identities:self.bodies.iter().map(|b|(b.body_uuid.clone(),b.vertices.clone())).collect(),
            immutable_rest_locals:self.immutable_rest_locals.clone(),states:self.states.iter().map(StateSnapshot::from_state).collect(),accepted_time:self.accepted_time,
            last_segment:self.last_segment.as_ref().map(|s|SegmentSnapshot{start:s.start,end:s.end,states:s.states.iter().map(StateSnapshot::from_state).collect()})};
        self.validate_checkpoint_history(&s,self.accepted_time)?;Ok(s)
    }
    fn validate_checkpoint_states(&self,states:&[StateSnapshot],time:f64,rest:&[Vec<[f64;3]>])->Result<()> {
        if states.len()!=self.bodies.len(){return Err("FEM checkpoint state count mismatch".into());}
        let identity=RootPose{center_world_m:[0.;3],rotation_world_wxyz:[1.,0.,0.,0.]};
        for ((b,s),rest) in self.bodies.iter().zip(states).zip(rest) {
            let cursor=b.events.iter().filter(|e|e.time_s<=time).count();
            let (mode,binding,capture_time)=if cursor==0 {(b.initial_mode,if b.initial_mode==Mode::Free{None}else{Some(Binding::Absolute)},0.)}
                else {let e=&b.events[cursor-1];(e.mode,e.binding,e.time_s)};
            if s.cursor!=cursor || s.epoch!=cursor as u64 || s.mode!=mode || s.binding!=binding {return Err("FEM checkpoint event clock/mode/epoch mismatch".into());}
            if mode==Mode::Free {if s.capture.is_some(){return Err("FREE FEM checkpoint has a capture".into());}continue;}
            let c=s.capture.as_ref().ok_or("controlled FEM checkpoint capture absent")?;let n=b.vertices.len();
            if c.capture_time!=capture_time || !checkpoint_points(&c.positions,n) || !checkpoint_points(&c.locals,if mode==Mode::Kinematic{n}else{0})
                || !checkpoint_pose(c.epoch_transform) || c.material_frame.is_some_and(|p|!checkpoint_pose(p))
                || c.absolute_hold_targets.as_ref().is_some_and(|v|!checkpoint_points(v,n)) {return Err("FEM checkpoint capture shape/pose/time mismatch".into());}
            let want_hold=mode==Mode::Static && binding==Some(Binding::Absolute) && cursor>0;
            let want_frame=mode==Mode::Kinematic || (binding==Some(Binding::Absolute) && cursor>0);
            if c.absolute_hold_targets.is_some()!=want_hold || c.material_frame.is_some()!=want_frame {return Err("FEM checkpoint capture binding contract mismatch".into());}
            if binding==Some(Binding::Absolute) {
                if c.epoch_transform!=identity || (mode==Mode::Kinematic && c.locals!=*rest) {return Err("FEM absolute checkpoint changed rest/epoch".into());}
                let author=b.sample(capture_time)?;
                if want_frame && c.material_frame!=Some(author){return Err("FEM absolute checkpoint material frame mismatch".into());}
                let targets=rest.iter().map(|x|transform(author,*x)).collect::<Result<Vec<_>>>()?;
                if !checkpoint_near(&targets,&c.positions) || c.absolute_hold_targets.as_ref().is_some_and(|v|*v!=targets) {return Err("FEM absolute checkpoint target mismatch".into());}
            }else if binding==Some(Binding::CurrentHold) {
                if mode!=Mode::Static || c.epoch_transform!=identity {return Err("FEM hold checkpoint contract mismatch".into());}
            }else if binding==Some(Binding::CurrentFollow) {
                if mode!=Mode::Kinematic {return Err("FEM follow checkpoint mode mismatch".into());}
                let frame=c.material_frame.ok_or("FEM follow checkpoint frame absent")?;
                let inverse=inverse_root_pose(frame).map_err(|e|e.to_string())?;
                let locals=c.positions.iter().map(|x|transform(inverse,*x)).collect::<Result<Vec<_>>>()?;
                let epoch=compose_root_pose(frame,inverse_root_pose(b.sample(capture_time)?).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
                if !checkpoint_near(&locals,&c.locals) || c.epoch_transform!=epoch {return Err("FEM follow checkpoint local/epoch mismatch".into());}
            }else{return Err("FEM controlled checkpoint binding absent".into());}
            // Also exercise the production target transform for overflow.
            Self::target(b,&s.clone().into_state(),time)?;
        }
        Ok(())
    }
    fn validate_checkpoint_history(&self,s:&RuntimeSnapshot,accepted_time:f64)->Result<()> {
        if s.source_launch_id.is_empty() || !accepted_time.is_finite() || accepted_time<0. || s.accepted_time!=accepted_time
            || s.body_identities!=self.bodies.iter().map(|b|(b.body_uuid.clone(),b.vertices.clone())).collect::<Vec<_>>()
            || s.immutable_rest_locals.len()!=self.bodies.len() || self.immutable_rest_locals.len()!=self.bodies.len() {return Err("FEM checkpoint identity/time mismatch".into());}
        for ((b,rest),expected) in self.bodies.iter().zip(&s.immutable_rest_locals).zip(&self.immutable_rest_locals) {
            if !checkpoint_points(rest,b.vertices.len()) || rest!=expected {return Err("FEM checkpoint immutable rest mismatch".into());}
        }
        self.validate_checkpoint_states(&s.states,accepted_time,&s.immutable_rest_locals)?;
        match &s.last_segment {
            None if accepted_time==0.=>{},
            Some(segment) if segment.start.is_finite() && segment.start>=0. && segment.start<segment.end && segment.end==accepted_time=>{
                self.validate_checkpoint_states(&segment.states,segment.start,&s.immutable_rest_locals)?;
                for (i,b) in self.bodies.iter().enumerate(){
                    if b.events.iter().any(|e|segment.start<e.time_s&&e.time_s<segment.end) {return Err("FEM checkpoint segment crosses event".into());}
                    if !b.events.iter().any(|e|e.time_s==segment.end) && segment.states[i]!=s.states[i] {return Err("FEM checkpoint states changed without event".into());}
                }
            },
            _=>return Err("FEM checkpoint accepted segment mismatch".into()),
        }
        Ok(())
    }
    pub fn validate_checkpoint(&self,s:&RuntimeSnapshot,accepted_time:f64)->Result<()> {
        if self.pending.is_some() || self.launch_id==s.source_launch_id {return Err("FEM checkpoint requires no pending attempt and a fresh launch identity".into());}
        self.validate_checkpoint_history(s,accepted_time)
    }
    pub fn import_checkpoint(&mut self,s:RuntimeSnapshot,accepted_time:f64)->Result<()> {
        self.validate_checkpoint(&s,accepted_time)?;
        self.states=s.states.into_iter().map(StateSnapshot::into_state).collect();self.accepted_time=s.accepted_time;
        self.last_segment=s.last_segment.map(|s|Segment{start:s.start,end:s.end,states:s.states.into_iter().map(StateSnapshot::into_state).collect()});
        // Rest/program and destination launch identity/attempt sequence stay local.
        Ok(())
    }
}

#[cfg(test)]
mod checkpoint_tests {
    use super::*;
    fn identity()->RootPose {RootPose{center_world_m:[0.;3],rotation_world_wxyz:[1.,0.,0.,0.]}}
    fn shape()->Vec<[f64;3]> {vec![[0.;3],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]}
    fn body()->Body {Body{body_uuid:"body".into(),vertices:vec![0,1,2,3],initial_mode:Mode::Free,
        keys:vec![NormalizedKey{time_s:0.,pose:identity()},NormalizedKey{time_s:1.,pose:RootPose{center_world_m:[1.,0.,0.],..identity()}}],reference:identity(),
        events:vec![Event{time_s:0.1,mode:Mode::Static,binding:Some(Binding::CurrentHold)},Event{time_s:0.2,mode:Mode::Kinematic,binding:Some(Binding::CurrentFollow)},Event{time_s:0.3,mode:Mode::Free,binding:None}]}}
    #[test]fn checkpoint_roundtrip_all_modes_capture_segment_and_stale_tokens(){
        let x=shape();let mut source=Runtime::new(vec![body()],&x,"old".into()).unwrap();
        for end in [0.,0.1,0.2,0.3] {
            if end>0. {let a=source.begin(end,(end-source.accepted_time) as f32).unwrap();assert!(source.export_checkpoint().is_err());source.accept(&a,&x,|_|Ok(identity())).unwrap();}
            let snapshot=source.export_checkpoint().unwrap();let bytes=serde_json::to_vec(&snapshot).unwrap();
            let mut target=Runtime::new(vec![body()],&x,"new".into()).unwrap();target.import_checkpoint(serde_json::from_slice(&bytes).unwrap(),end).unwrap();
            let got=target.export_checkpoint().unwrap();assert_eq!(got.states,snapshot.states);assert_eq!(got.last_segment,snapshot.last_segment);assert_eq!(target.launch_id,"new");
            let end2=end+0.01;let old=source.begin(end2,0.01).unwrap();let next=target.begin(end2,0.01).unwrap();
            assert!(target.accept(&old,&x,|_|Ok(identity())).is_err());target.accept(&next,&x,|_|Ok(identity())).unwrap();source.abort(&old).unwrap();
        }
    }
    #[test]fn checkpoint_rejects_invalid_capture_clock_rest_and_is_atomic(){
        let x=shape();let mut source=Runtime::new(vec![body()],&x,"old".into()).unwrap();let a=source.begin(0.1,0.1).unwrap();source.accept(&a,&x,|_|Ok(identity())).unwrap();
        let good=source.export_checkpoint().unwrap();let mut target=Runtime::new(vec![body()],&x,"new".into()).unwrap();let before=format!("{:?}",target);
        let mut cases=vec![];let mut b=good.clone();b.states.clear();cases.push(b);let mut b=good.clone();b.states[0].cursor=0;cases.push(b);
        let mut b=good.clone();b.states[0].capture.as_mut().unwrap().positions.pop();cases.push(b);let mut b=good.clone();b.states[0].capture.as_mut().unwrap().positions[0][0]=f64::NAN;cases.push(b);
        let mut b=good.clone();b.states[0].capture.as_mut().unwrap().epoch_transform.rotation_world_wxyz=[0.;4];cases.push(b);let mut b=good.clone();b.states[0].capture.as_mut().unwrap().capture_time=0.;cases.push(b);
        let mut b=good.clone();b.immutable_rest_locals[0][0][0]=1.;cases.push(b);let mut b=good.clone();b.last_segment.as_mut().unwrap().start=0.2;cases.push(b);
        let mut b=good.clone();b.source_launch_id="new".into();cases.push(b);
        let mut b=good.clone();b.states[0].binding=Some(Binding::Absolute);cases.push(b);
        let mut b=good.clone();b.states[0].epoch=9;cases.push(b);
        let mut b=good.clone();b.states[0].capture.as_mut().unwrap().locals=vec![[0.;3];4];cases.push(b);
        let mut b=good.clone();b.states[0].capture.as_mut().unwrap().material_frame=Some(identity());cases.push(b);
        for b in cases {assert!(target.import_checkpoint(b,0.1).is_err());assert_eq!(format!("{:?}",target),before);}
        let a=target.begin(0.05,0.05).unwrap();assert!(target.import_checkpoint(good,0.1).is_err());target.abort(&a).unwrap();
    }
    #[test]fn checkpoint_absolute_modes_validate_full_capture_contract(){
        let x=shape();
        for mode in [Mode::Static,Mode::Kinematic] {
            let mut b=body();b.initial_mode=mode;b.events.clear();let r=Runtime::new(vec![b.clone()],&x,"one".into()).unwrap();r.export_checkpoint().unwrap();
            b.initial_mode=Mode::Free;b.events=vec![Event{time_s:0.1,mode,binding:Some(Binding::Absolute)}];
            let mut r=Runtime::new(vec![b],&x,"one".into()).unwrap();let a=r.begin(0.1,0.1).unwrap();let mut moved=x.clone();for v in &mut moved {v[0]+=0.1;}r.accept(&a,&moved,|_|panic!()).unwrap();r.export_checkpoint().unwrap();
        }
    }
}

#[cfg(test)]
mod checkpoint_rotated_follow_tests {
    use super::*;
    #[test]fn serde_cbor_preserves_nontrivial_follow_material_and_old_segment(){
        let q=|a:f64|[(a/2.).cos(),0.,0.,(a/2.).sin()];
        let body=Body{body_uuid:"body".into(),vertices:vec![0,1,2,3],initial_mode:Mode::Free,
            keys:vec![NormalizedKey{time_s:0.,pose:RootPose{center_world_m:[0.1,0.2,0.3],rotation_world_wxyz:q(0.1)}},NormalizedKey{time_s:1.,pose:RootPose{center_world_m:[1.2,-0.1,0.4],rotation_world_wxyz:q(1.1)}}],
            reference:RootPose{center_world_m:[0.02,0.03,-0.04],rotation_world_wxyz:q(0.2)},events:vec![Event{time_s:0.1,mode:Mode::Kinematic,binding:Some(Binding::CurrentFollow)}]};
        let x=vec![[0.1,0.2,0.3],[1.2,0.1,0.2],[0.3,1.4,0.1],[0.1,0.3,1.6]];
        let mut r=Runtime::new(vec![body.clone()],&x,"old".into()).unwrap();let a=r.begin(0.1,0.1).unwrap();
        let frame=RootPose{center_world_m:[0.22,-0.4,0.8],rotation_world_wxyz:q(0.37)};r.accept(&a,&x,|_|Ok(frame)).unwrap();
        let good=r.export_checkpoint().unwrap();let mut bytes=vec![];ciborium::ser::into_writer(&good,&mut bytes).unwrap();let decoded=ciborium::de::from_reader(bytes.as_slice()).unwrap();
        let mut target=Runtime::new(vec![body],&x,"new".into()).unwrap();target.import_checkpoint(decoded,0.1).unwrap();
        assert_eq!(Runtime::target(&r.bodies[0],&r.states[0],0.7).unwrap(),Runtime::target(&target.bodies[0],&target.states[0],0.7).unwrap());
        let mut bad=good;bad.states[0].capture.as_mut().unwrap().locals[0][0]+=0.01;assert!(target.import_checkpoint(bad,0.1).is_err());
    }
}
