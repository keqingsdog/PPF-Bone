//! Copy to crates/ppf-cts-solver/src/driver/roots.rs after registering generated
//! root args/IDs. This module stages endpoints and dispatches exact DOF algebra;
//! the step loop owns CCD, rollback and accepted position/history commits.
use ppf_cts_compute::{AllocLabel,Buffer,Device,Encoder,EncoderExt,Fault,Handle};
#[path="root_geometry.rs"] pub(super) mod root_geometry;
use crate::joint_root_motion_gpu::RootTargetGpu;
use crate::joint_root_motion_state::{Attempt,RootState,NativeStep,native_step_before_key};
use crate::joint_root_motion_wire::{ValidatedRootMotion,RootPose};
use super::kernels::{RootProjectArgs,RootParticularArgs,RootTargetVerticesArgs};

// SAFETY: repr(C), scalar-only 64-byte record matched to generated CUDA/Metal.
unsafe impl ppf_cts_compute::Pod for RootTargetGpu {}
fn fault(e:impl std::fmt::Display)->Fault {Fault::Platform {call:"root motion",detail:e.to_string()}}

#[derive(Clone,Copy,Debug,Default,Eq,PartialEq)]
pub struct ConstraintDomain {pub owner_generation:u64,pub revision:u64}
impl ConstraintDomain { pub fn is_cacheable(&self)->bool {self.owner_generation!=0 && self.revision!=0} }
static NEXT_DOMAIN_OWNER:std::sync::atomic::AtomicU64=std::sync::atomic::AtomicU64::new(1);
fn new_domain_owner()->u64 {
    NEXT_DOMAIN_OWNER.fetch_update(std::sync::atomic::Ordering::Relaxed,std::sync::atomic::Ordering::Relaxed,|n|n.checked_add(1)).unwrap_or(0)
}
struct DomainTracker {stamp:ConstraintDomain, enabled:Vec<u32>, published:bool}
impl Default for DomainTracker {fn default()->Self {Self{stamp:ConstraintDomain{owner_generation:new_domain_owner(),revision:1},enabled:Vec::new(),published:false}}}
impl DomainTracker {
 fn reinstall(&mut self) {self.stamp=ConstraintDomain{owner_generation:new_domain_owner(),revision:1};self.enabled.clear();self.published=false;}
 fn observe(&mut self, records:&[RootTargetGpu])->ConstraintDomain {
    if !self.published || !self.stamp.is_cacheable() {return ConstraintDomain::default();}
    let mut enabled:Vec<_>=records.iter().filter(|r|r.enabled==1).map(|r|r.body).collect();
    enabled.sort_unstable();enabled.dedup();
    if enabled!=self.enabled {self.enabled=enabled;self.stamp.revision=self.stamp.revision.checked_add(1).unwrap_or(0);}
    self.stamp
 }
}
#[derive(Clone,Copy)]
pub struct View {pub targets:Handle,pub count:u32,pub bodies:u32,pub base:u32,pub dim:u32,pub constraint_domain:ConstraintDomain}
#[derive(Default)]
pub struct Roots {
    pub transaction:Option<RootState>,
    targets:Buffer<RootTargetGpu>,
    records:Vec<RootTargetGpu>,
    domain_tracker:DomainTracker,
    // Sealed pdrd body vertex_start/count by runtime body index.
    spans:Vec<(u32,u32)>,
    retry_dt:Option<f32>,
    // Last uncommitted native duration, including event/final-clock shortening.
    trial_dt:Option<f32>,
    pub curve_retry:bool,
    curve:CurveStaging,
}
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct HardContext {pub body:u32,pub generation:u64,pub sequence:u64}
impl Roots {
    pub(crate) fn invalidate_trial_targets<D:Device>(&mut self,device:&mut D)->Result<(),Fault> {
        self.domain_tracker.published=false;
        self.records.clear();
        self.retry_step(None);
        self.targets.free(device)
    }
    /// Guard the direct target-matrix branch of controlled_retract. The other
    /// branch is bounded by its explicit scaled rotation vector.
    pub(crate) fn twist_root_extra(&self,selected:&[bool],fitted:&[crate::data::PdrdRigidState],dtheta:&[f32],fraction:f32)->Option<Vec<f64>> {
        use nalgebra::{Matrix3,UnitQuaternion,Vector3};
        if selected.len()!=fitted.len() || dtheta.len()!=3*fitted.len() || !fraction.is_finite() {return None;}
        let mut extra=vec![0.0;selected.len()];
        for root in &self.records {
            let b=root.body as usize;
            if root.enabled!=1 {continue;}
            if b>=selected.len() {return None;}
            if !selected[b] {continue;}
            if fraction==1.0 {
                let theta=Vector3::new(-(dtheta[3*b] as f64),-(dtheta[3*b+1] as f64),-(dtheta[3*b+2] as f64));
                let r=Matrix3::from_column_slice(&fitted[b].rotation.map(f64::from));
                let target=Matrix3::from_column_slice(&root.rotation.map(f64::from));
                if theta.iter().chain(r.iter()).chain(target.iter()).any(|x|!x.is_finite()) {return None;}
                let identity=Matrix3::identity();
                if (r.transpose()*r-identity).norm()>1e-4 || (target.transpose()*target-identity).norm()>1e-4 || r.determinant()<=0.0 || target.determinant()<=0.0 {return None;}
                let expected=UnitQuaternion::from_scaled_axis(theta).to_rotation_matrix().matrix()*r;
                if (target-expected).norm()>1e-4 {return None;}
                extra[b]=1e-3;
            }
        }
        Some(extra)
    }
    pub fn constraint_enabled_bodies(&self)->Vec<u32>{self.records.iter().filter(|r|r.enabled==1).map(|r|r.body).collect()}

    pub fn install<D:Device>(&mut self,device:&mut D,generation:u64,
                            roots:Vec<ValidatedRootMotion>,spans:Vec<(u32,u32)>)->Result<(),Fault> {
        self.domain_tracker.reinstall();
        if roots.is_empty() {self.transaction=None;self.records.clear();self.domain_tracker.published=true;return Ok(());}
        if roots.iter().any(|r|r.runtime_body_index as usize>=spans.len()) {return Err(fault("ROOT_MOTION_BODY_BINDING: span missing"));}
        let tx=RootState::new(generation,roots).map_err(fault)?;
        self.targets.size(device,tx.active_indices().len(),AllocLabel("roots.targets"))?;
        self.spans=spans;self.transaction=Some(tx);Ok(())
    }
    /// Real enabled native target identity, frozen for this Newton call.
    pub fn hard_context(&self)->Result<Option<HardContext>,Fault>{
        let active:Vec<_>=self.records.iter().filter(|r|r.enabled==1).collect();
        if active.is_empty(){return Ok(None);}
        if active.len()!=1{return Err(fault("ROOT_HARD_SOURCE: exactly one active root required"));}
        let tx=self.transaction.as_ref().ok_or_else(||fault("ROOT_HARD_SOURCE: missing transaction"))?;
        let (generation,sequence)=tx.hard_epoch();
        Ok(Some(HardContext{body:active[0].body,generation,sequence}))
    }
    pub fn active(&self)->bool {self.transaction.as_ref().is_some_and(|s|!s.active_indices().is_empty())}
    pub fn plan_step(&self,time_s:f64,requested:f32)->Result<NativeStep,Fault> {
        let requested=self.retry_dt.unwrap_or(requested);
        match &self.transaction {Some(s)=>s.native_step(requested).map_err(fault),None=>native_step_before_key(time_s,requested,&[]).map_err(fault)}
    }
    /// Call after planning the physical endpoint and its rounded fp32 dt.
    /// Freeze one target for the whole attempt. On a shortened sweep refreeze;
    /// overwrite accepted positions to force this old target to fit a new dt.
    pub fn begin<D:Device>(&mut self,device:&mut D,step:NativeStep)->Result<Option<Attempt>,Fault> {
        self.begin_impl(device,step,false)
    }
    pub fn begin_shared<D:Device>(&mut self,device:&mut D,step:NativeStep)->Result<Option<Attempt>,Fault> {
        self.begin_impl(device,step,true)
    }
    fn begin_impl<D:Device>(&mut self,device:&mut D,step:NativeStep,shared:bool)->Result<Option<Attempt>,Fault> {
        let Some(tx)=&mut self.transaction else {return Ok(None);};
        let attempt=if shared{tx.begin_native_bounded(step)}else{tx.begin_native(step)}.map_err(fault)?;
        let mut records=Vec::with_capacity(attempt.body_indices.len());
        for (&body,&pose) in attempt.body_indices.iter().zip(&attempt.target) {
            let (start,count)=self.spans[body as usize];
            records.push(RootTargetGpu::sampled(body,pose,start,count).map_err(fault)?);
        }
        self.domain_tracker.published=false;
        self.targets.size(device,records.len(),AllocLabel("roots.targets"))?;
        if !records.is_empty() {self.targets.write(device,0,&records)?;}
        self.records=records;
        self.domain_tracker.published=true;
        self.trial_dt=attempt.integration_dt_s;
        Ok(Some(attempt))
    }
    /// Call from B3 immediately after dt is narrowed to the actual TOI clock,
    /// before compute_target. Native solve/CCD/intersection remain mandatory.
    pub fn shorten<D:Device>(&mut self,device:&mut D,attempt:&Attempt,dt:f32)->Result<Attempt,Fault> {
        let tx=self.transaction.as_mut().ok_or_else(||fault("ROOT_MOTION_COMMIT_MISMATCH: no root transaction"))?;
        let next=tx.shorten_native(attempt,f64::from(dt)).map_err(fault)?;
        let mut records=Vec::with_capacity(next.body_indices.len());
        for (&body,&pose) in next.body_indices.iter().zip(&next.target) {
            let (start,count)=self.spans[body as usize];
            records.push(RootTargetGpu::sampled(body,pose,start,count).map_err(fault)?);
        }
        self.domain_tracker.published=false;
        self.targets.size(device,records.len(),AllocLabel("roots.targets"))?;
        if !records.is_empty(){self.targets.write(device,0,&records)?;}
        self.records=records;
        self.domain_tracker.published=true;
        self.trial_dt=next.integration_dt_s;
        Ok(next)
    }
    /// Native caller supplies its accepted readback history and changes this
    /// mask between accepted steps. No curr/prev or prev_dt write occurs here.
    pub fn release<D:Device>(&mut self,device:&mut D,body:u32)->Result<(),Fault> {
        self.transaction.as_mut().ok_or_else(||fault("ROOT_MOTION_BODY_BINDING: no installed roots"))?
            .release(body).map_err(fault)?;
        self.domain_tracker.published=false;
        for r in &mut self.records {if r.body==body {r.enabled=0;}}
        if !self.records.is_empty(){self.targets.write(device,0,&self.records)?;}
        self.domain_tracker.published=true;
        Ok(())
    }
    pub fn takeover<D:Device>(&mut self,device:&mut D,body:u32,actual:RootPose,
                              tolerance_m:f64,tolerance_rad:f64)->Result<(),Fault> {
        let tx=self.transaction.as_mut().ok_or_else(||fault("ROOT_MOTION_BODY_BINDING: no installed roots"))?;
        tx.takeover(body,actual,tolerance_m,tolerance_rad).map_err(fault)?;
        let pose=tx.pose_at_committed(body).map_err(fault)?;
        let (start,count)=self.spans[body as usize];
        let record=RootTargetGpu::sampled(body,pose,start,count).map_err(fault)?;
        self.domain_tracker.published=false;
        if let Some(row)=self.records.iter_mut().find(|r|r.body==body){*row=record;}
        else {self.records.push(record);self.records.sort_by_key(|r|r.body);}
        self.targets.size(device,self.records.len(),AllocLabel("roots.targets"))?;
        self.targets.write(device,0,&self.records)?;self.domain_tracker.published=true;Ok(())
    }
    pub fn view(&mut self,map:&super::rigid_map::RigidMap)->Option<View> {
        let constraint_domain=self.domain_tracker.observe(&self.records);
        if self.targets.len()==0 || !self.active(){return None;}
        Some(View{targets:self.targets.handle(),count:self.targets.len() as u32,
            bodies:map.n_bodies as u32,base:map.body_base as u32,dim:map.dim as u32,constraint_domain})
    }
    /// Override only implicit target rows INSIDE the native solve, immediately
    /// after every compute_target rebuild. No curr/prev/accepted buffer writes.
    pub unsafe fn target<D:Device>(&mut self,device:&mut D,list:Handle,rest:Handle,
                                   target:Handle,body_vertices:u32,vertices:u32)->Result<(),Fault> {
        if !self.active() || self.targets.len()==0{return Ok(());}
        let args=RootTargetVerticesArgs{targets:self.targets.handle(),list,rest,target,
            roots:self.targets.len() as u32,vertices,count:body_vertices,seam_arena_count:0};
        device.launch("roots.target_vertices",&args,body_vertices)?;Ok(())
    }
}
impl View {
    /// Encode the same homogeneous root projection inside an ordered region.
    pub unsafe fn encode_project(&self,encoder:&mut dyn Encoder,vector:Handle)->Result<(),Fault> {
        let args=RootProjectArgs{targets:self.targets,vector,base:self.base,dim:self.dim,
            count:self.count,seam_arena_count:0};
        encoder.elements(&args,self.count)
    }
    /// Q v; used on every reduced homogeneous vector and both sides Q A Q.
    pub unsafe fn project<D:Device>(&self,device:&mut D,vector:Handle)->Result<(),Fault> {
        let args=RootProjectArgs{targets:self.targets,vector,base:self.base,dim:self.dim,
            count:self.count,seam_arena_count:0};
        device.launch("roots.project",&args,self.count)?;Ok(())
    }
    /// Set root rows of the AFFINE direction to (anchor+x)-target. The native
    /// position_step subtracts this direction, matching the dx_seed convention.
    /// All other
    /// reduced rows stay untouched. Apply A*p before projecting its output.
    pub unsafe fn particular<D:Device>(&self,device:&mut D,state:Handle,vector:Handle)->Result<(),Fault> {
        let args=RootParticularArgs{targets:self.targets,state,vector,bodies:self.bodies,
            base:self.base,dim:self.dim,count:self.count,seam_arena_count:0};
        device.launch("roots.particular",&args,self.count)?;Ok(())
    }
}

// Production helpers. Callers retain ownership of native Newton/CCD history.

pub unsafe fn install_native<D:Device>(device:&mut D,data:&crate::data::DataSet,
    state:&mut super::state::SolverState,owned:&crate::joint::OwnedJointScene,sealed_initial:Option<&[f32]>)->super::scene::FatalResult<()> {
    if owned.roots.is_empty(){return Ok(());}
    use crate::joint_root_motion_wire::{BodyIdentity,NativeFrameEvidence};
    if state.rigid.any_joint||state.rigid.any_translation_lock||state.rigid.any_rotation_lock||data.translation_lock.size>0 {
        return Err(super::scene::Fatal::invariant("ROOT_MOTION_NATIVE_CAPABILITY: aggregate/native lock composition is unavailable"));
    }
    let props=super::scene::slice(&data.prop.pdrd_body);
    let list=super::scene::slice(&data.pdrd_vert_list);
    let rest=super::scene::slice(&data.pdrd_rest_centered);
    if sealed_initial.is_none(){state.positions.download(device)?;}
    // Runtime rebinding validates the same immutable initial shape on resume.
    // Accepted deformation is installed separately, never treated as rest.
    let positions=sealed_initial.unwrap_or_else(||state.positions.host());
    let ids:Vec<_>=owned.bodies.iter().map(|(uuid,&runtime_body_index)|BodyIdentity{uuid,runtime_body_index}).collect();
    let mut spans=Vec::new();let mut rr=Vec::new();let mut xx=Vec::new();let mut tolerances=Vec::new();
    for r in &owned.roots {
        let p=props.get(r.runtime_body_index as usize).ok_or_else(||fault("ROOT_MOTION_BODY_BINDING: native row absent"))?;
        let start=p.vertex_start as usize;let count=p.vertex_count as usize;
        let members=list.get(start..start+count).ok_or_else(||fault("ROOT_MOTION_BODY_BINDING: native span absent"))?;
        let mut y=Vec::new();let mut x=Vec::new();let mut extent=0.0_f64;let mut world=1.0_f64;
        for (i,&v) in members.iter().enumerate() {
            let rest=rest.get(start+i).ok_or_else(||fault("ROOT_MOTION_BODY_BINDING: native rest absent"))?;
            let components=positions.get(3*v as usize..3*v as usize+3).ok_or_else(||fault("ROOT_MOTION_BODY_BINDING: native vertex absent"))?;
            let a:[f64;3]=std::array::from_fn(|a|f64::from(rest[a]));
            let b:[f64;3]=std::array::from_fn(|a|f64::from(components[a]));
            extent=extent.max(a.iter().map(|v|v*v).sum::<f64>().sqrt());world=world.max(b.iter().map(|v|v.abs()).fold(0.0,f64::max));
            y.push(a);x.push(b);
        }
        spans.push(members);rr.push(y);xx.push(x);
        tolerances.push((16.0*f64::from(f32::EPSILON)*extent).max(4.0*f64::from(f32::EPSILON)*world));
    }
    let evidence:Vec<_>=owned.roots.iter().enumerate().map(|(i,r)|NativeFrameEvidence{
        context_scene_uuid:&owned.scene_uuid,context_bodies:&ids,joint_bodies:&ids,
        native_body:BodyIdentity{uuid:&r.body_uuid,runtime_body_index:r.runtime_body_index},
        native_body_is_free:props[r.runtime_body_index as usize].joint_mode==0,
        // Union of the legacy rigid known-parent check and the retained
        // strict-prepare FEM/mixed PDRD-parent proof. The legacy scan alone is
        // blind to material edges, whose joints live in the soft graph, so a
        // mixed/FEM-parent root was refused here on the second (runtime
        // actual-geometry) rebind. Nothing else in this helper changes: the FREE
        // identity, body identity, pin, fit-tolerance and root-curve/clock checks
        // in `bind_native_frames_with_body_controls` and below are untouched.
        has_enabled_known_parent_joint:owned.joints.iter().any(|j|j.wire.enabled&&j.parent_runtime_body_index==r.runtime_body_index)
            || owned.fem_pdrd_parent_proof().contains(r.runtime_body_index),
        vertex_indices:spans[i],rest_centered_m:&rr[i],initial_world_m:&xx[i],fit_tolerance_m:tolerances[i],
    }).collect();
    let vertex_props=super::scene::slice(&data.prop.vertex);
    let pins:Vec<_>=vertex_props.iter().enumerate().filter(|(_,p)|p.fix_index>0||p.pull_index>0).map(|(v,_)|v as u32).collect();
    crate::joint_root_motion_binding::bind_native_frames_with_body_controls(&owned.roots,&evidence,&pins,owned.body_controls_v1).map_err(fault)?;
    let generation=crate::status_writer::launch_id().and_then(|s|u64::from_str_radix(&s,16).ok())
        .filter(|n|*n!=0).ok_or_else(||fault("ROOT_MOTION_COMMIT_MISMATCH: actual launch generation absent"))?;
    state.roots.install(device,generation,owned.roots.clone(),props.iter().map(|p|(p.vertex_start,p.vertex_count)).collect())?;
    Ok(())
}

pub unsafe fn gate_runtime(data:&crate::data::DataSet,state:&super::state::SolverState,playback:f32)->super::scene::FatalResult<()> {
    if state.roots.transaction.is_some() && playback!=1.0 {
        return Err(super::scene::Fatal::invariant("ROOT_MOTION_NATIVE_CAPABILITY: installed root controller requires unit physical clock"));
    }
    if !state.roots.active(){return Ok(());}
    if playback!=1.0||data.translation_lock.size>0||state.rigid.needs_projection(){
        return Err(super::scene::Fatal::invariant("ROOT_MOTION_NATIVE_CAPABILITY: runtime playback/lock changed"));
    }
    let active=state.roots.transaction.as_ref().unwrap().active_indices();
    for p in super::scene::slice(&data.prop.vertex) {
        if p.pdrd_body_index>0&&active.contains(&(p.pdrd_body_index-1))&&(p.fix_index>0||p.pull_index>0){
            return Err(super::scene::Fatal::invariant("ROOT_MOTION_OLD_PIN_OVERLAP: runtime pin touches prescribed root"));
        }
    }
    Ok(())
}

/// Root member evidence only; transaction/time/accepted history remain untouched.
pub enum CandidateCertification {
    Ready {actual:Vec<RootPose>,max_tol:f64},
    NeedsRefinement(super::scene::Fatal),
}

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
enum RootComponentCertification {Invalid,NeedsRefinement,Ready}
fn classify_root_component(want:f64,got:f64,tol:f64)->RootComponentCertification {
    if !want.is_finite()||!got.is_finite()||!(want-got).is_finite()||!tol.is_finite()||tol<0.0 {RootComponentCertification::Invalid}
    else if (want-got).abs()>tol {RootComponentCertification::NeedsRefinement}
    else {RootComponentCertification::Ready}
}

/// Read-only candidate evidence. The final call follows native intersection
/// checks and precedes any accepted clock/position/history writes. Refit the actual final iterate,
/// compare every member against immutable rest plus the sampled authored pose,
/// returning evidence only. Never commit, correct or snap positions here.
/// # Safety
/// Scene records, rest/member spans and fitted/position buffers must stay live
/// on this device for the same native scene. The frozen attempt must remain
/// pending; every body index and typed buffer range satisfies the original
/// Reduction::fit/readback contracts. Only fit scratch/host mirrors may change.
pub unsafe fn certify_candidate<D:Device>(device:&mut D,data:&crate::data::DataSet,
    state:&mut super::state::SolverState,attempt:&Attempt)->super::scene::FatalResult<CandidateCertification> {
    if attempt.body_indices.len()!=attempt.target.len() {
        return Err(fault("ROOT_MOTION_COMMIT_MISMATCH: accepted clock/count/tolerance differs from frozen sweep").into());
    }
    let super::state::SolverState{rigid,rigid_staged,rigid_state,rigid_fit_scratch,pcg_rigid,eval_x,prop_vertex,roots,..}=state;
    if !attempt.body_indices.is_empty(){
        let list=rigid_staged.vert_list.handle();let props=rigid_staged.body_prop.handle();let rest=rigid_staged.rest_centered.handle();
        let mut reduction=super::pdrd::Reduction{map:rigid,staged:rigid_staged};
        reduction.fit(device,list,prop_vertex.handle(),props,rest,eval_x.handle(),rigid_fit_scratch.handle(),rigid_state.handle(),data.pdrd_vert_list.size,&pcg_rigid.zero)?;
    }
    eval_x.download(device)?;
    let list=super::scene::slice(&data.pdrd_vert_list);let rest=super::scene::slice(&data.pdrd_rest_centered);
    let vertex_props=super::scene::slice(&data.prop.vertex);
    let geometry_rest=root_geometry::rest_vectors(rest);
    let mut actual=Vec::new();let mut max_tol=0.0_f64;let mut needs_refinement=None;
    for (&body,&target) in attempt.body_indices.iter().zip(&attempt.target) {
        let fitted=rigid_state.read_one(device,body as usize)?;
        let (start,count)=roots.spans[body as usize];
        if fitted.count!=count {return Err(fault("ROOT_MOTION_COMMIT_MISMATCH: fitted body count differs").into());}
        let geometry=root_geometry::fit_body(eval_x.host(),&geometry_rest,list,body,start,count,1.0,
            |v|vertex_props.get(v as usize).map(|p|p.pdrd_body_index)).map_err(|e|fault(format!("ROOT_MOTION_COMMIT_MISMATCH: {e}")))?;
        let center=geometry.center;
        let matrix=nalgebra::Matrix3::<f64>::from_column_slice(&fitted.rotation.map(f64::from));
        if !matrix.iter().all(|v|v.is_finite())||(matrix.transpose()*matrix-nalgebra::Matrix3::identity()).norm()>1e-4||matrix.determinant()<=0.0 {
            return Err(fault("ROOT_MOTION_COMMIT_MISMATCH: fitted rotation not proper").into());
        }
        let pose=RootPose{center_world_m:center,rotation_world_wxyz:geometry.quaternion};
        let mut radius=0.0_f64;
        for i in start as usize..(start+count) as usize {radius=radius.max(rest[i].iter().map(|v|f64::from(*v).powi(2)).sum::<f64>().sqrt());}
        let world=target.center_world_m.iter().chain(center.iter()).map(|v|v.abs()).fold(1.0,f64::max);
        let tol=(32.0*f64::from(f32::EPSILON)*radius).max(8.0*f64::from(f32::EPSILON)*(world+radius));
        max_tol=max_tol.max(tol);
        if geometry.max_component_residual>tol && needs_refinement.is_none() {
            needs_refinement=Some(super::scene::Fatal::from(fault("ROOT_MOTION_COMMIT_MISMATCH: actual root geometry is not representable within native tolerance")));
        }
        for i in start as usize..(start+count) as usize {
            let rotated=crate::joint_root_motion_wire::rotate_world(target.rotation_world_wxyz,std::array::from_fn(|a|f64::from(rest[i][a])));
            let v=list[i] as usize;
            for a in 0..3 {let want=target.center_world_m[a]+rotated[a];let got=f64::from(eval_x.host()[3*v+a]);
                match classify_root_component(want,got,tol) {
                    RootComponentCertification::Invalid=>return Err(fault(format!("ROOT_MOTION_COMMIT_MISMATCH: body {body} vertex {v} axis {a}, expected {want}, actual {got}, tolerance {tol}")).into()),
                    RootComponentCertification::NeedsRefinement=>{
                        if needs_refinement.is_none(){needs_refinement=Some(super::scene::Fatal::from(fault(format!("ROOT_MOTION_COMMIT_MISMATCH: body {body} vertex {v} axis {a}, expected {want}, actual {got}, tolerance {tol}"))));}
                    },
                    RootComponentCertification::Ready=>{},
                }
            }
        }
        actual.push(pose);
    }
    // Probe the same frozen transaction domain/pose checks without committing.
    // A later invalid aggregate pose must not be swallowed by a member miss.
    match roots.transaction.as_ref().ok_or_else(||fault("ROOT_MOTION_COMMIT_MISMATCH: transaction absent"))?
        .validate_candidate(attempt,attempt.end_s,&actual,max_tol,1e-4).map_err(fault)? {
        crate::joint_root_motion_state::CandidateValidation::Ready=>{},
        crate::joint_root_motion_state::CandidateValidation::NeedsRefinement(error)=>{
            if needs_refinement.is_none(){needs_refinement=Some(fault(error).into());}
        },
    }
    if let Some(error)=needs_refinement {return Ok(CandidateCertification::NeedsRefinement(error));}
    Ok(CandidateCertification::Ready{actual,max_tol})
}

/// Final certification and transaction commit remain after all C2 gates.
/// A loop probe never calls this function: it must not commit a transaction.
/// # Safety
/// Same live scene/state/attempt and typed buffer domains as certify_candidate.
pub unsafe fn commit_candidate<D:Device>(device:&mut D,data:&crate::data::DataSet,
    state:&mut super::state::SolverState,attempt:&Attempt)->super::scene::FatalResult<()> {
    let (actual,max_tol)=match certify_candidate(device,data,state,attempt)? {
        CandidateCertification::Ready{actual,max_tol}=>(actual,max_tol),
        CandidateCertification::NeedsRefinement(error)=>return Err(error),
    };
    state.roots.transaction.as_mut().ok_or_else(||fault("ROOT_MOTION_COMMIT_MISMATCH: transaction absent"))?
        .commit(attempt,attempt.end_s,&actual,max_tol,1e-4).map_err(fault)?;
    Ok(())
}

include!("roots_curve.fragment.rs");

#[cfg(test)]
mod constraint_domain_tests {
 use super::*;
 fn record(body:u32, enabled:u32)->RootTargetGpu {let mut r=RootTargetGpu::default();r.body=body;r.enabled=enabled;r}
 #[test]
 fn tracks_published_root_set_and_owner() {
  let mut t=DomainTracker::default();
  assert!(!t.observe(&[record(1,1)]).is_cacheable());
  t.published=true;
  let a=t.observe(&[record(1,1),record(2,0)]);assert!(a.is_cacheable());
  assert_eq!(a,t.observe(&[record(2,0),record(1,1)]));
  let b=t.observe(&[record(1,1),record(2,1)]);assert_ne!(a,b);
  let c=t.observe(&[record(2,1)]);assert_ne!(b,c);
  t.published=false;assert!(!t.observe(&[record(2,1)]).is_cacheable());
  t.published=true;assert_eq!(c,t.observe(&[record(2,1)]));
  t.reinstall();t.published=true;let d=t.observe(&[record(2,1)]);assert_ne!(c,d);
  let mut other=DomainTracker::default();other.published=true;assert_ne!(d,other.observe(&[record(2,1)]));
 }
 #[test]
 fn revision_overflow_is_not_cacheable() {
  let mut t=DomainTracker::default();t.published=true;t.stamp.revision=u64::MAX;
  assert!(!t.observe(&[record(7,1)]).is_cacheable());
  assert!(!t.observe(&[]).is_cacheable());
 }
}
