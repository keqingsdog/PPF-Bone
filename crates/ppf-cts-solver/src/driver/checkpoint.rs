//! Accepted runtime state only; no device handle or Newton attempt is serialized.
use super::{scene::{Fatal,FatalResult},state::SolverState};
use ppf_cts_compute::Device;
#[derive(Clone,Debug,PartialEq,serde::Serialize,serde::Deserialize)]
pub(crate) struct Snapshot {
    pub running_rotation:Vec<f32>,pub rotation_seeded:bool,
    pub root:Option<crate::joint_root_motion_state::RootSnapshot>,
    pub fem:Option<crate::fem_control_events::RuntimeSnapshot>,pub fem_steps:u64,
    pub restoring:Option<super::restoring::Snapshot>,
    pub restoring_commit_count:u64,pub restoring_commit_time:Option<f64>,
}
impl Snapshot {
    pub(crate) fn semantic_eq(&self,other:&Self)->bool{
        let mut a=self.clone();let mut b=other.clone();
        if let Some(r)=&mut a.root{r.source_generation=1;}if let Some(r)=&mut b.root{r.source_generation=1;}
        if let Some(r)=&mut a.fem{r.source_launch_id.clear();}if let Some(r)=&mut b.fem{r.source_launch_id.clear();}
        // Exact scalar bytes after removing only process-local attempt identity.
        // In particular preserve every f32 rotation/history bit without hashing.
        bincode::serialize(&a).ok().zip(bincode::serialize(&b).ok()).is_some_and(|(a,b)|a==b)
    }
}
pub(crate) fn validate(s:&Snapshot,time:f64,bodies:usize,
    root:Option<&crate::joint_root_motion_state::RootState>,
    fem:Option<&crate::fem_control_events::Runtime>,
    bindings:&[crate::restoring_curve_gpu::CurveBinding])->Result<(),String>{
    if !time.is_finite()||time<0.{return Err("CHECKPOINT_CLOCK_INVALID: accepted time".into());}
    crate::checkpoint_values::validate_running_rotation(&s.running_rotation,bodies)?;
    match(&s.root,root){(Some(a),Some(b))=>b.validate_checkpoint(a,time).map_err(|e|e.to_string())?,(None,None)=>{},_=>return Err("CHECKPOINT_ROOT_INVALID: static/runtime presence".into())}
    match(&s.fem,fem){(Some(a),Some(b))=>b.validate_checkpoint(a,time)?,(None,None) if s.fem_steps==0=>{},_=>return Err("CHECKPOINT_FEM_INVALID: static/runtime presence".into())}
    match(&s.restoring,bindings.is_empty()){
        (Some(a),false)=>{a.validate(time,bindings)?;if (s.restoring_commit_time!=Some(time)&&s.restoring_commit_count!=0)||a.runtime.checkpoint_commits()!=(s.restoring_commit_count,s.restoring_commit_time){return Err("CHECKPOINT_RESTORING_INVALID: driver commit time/count".into());}},
        (None,true) if s.restoring_commit_count==0&&s.restoring_commit_time.is_none()=>{},
        _=>return Err("CHECKPOINT_RESTORING_INVALID: static/runtime presence".into())
    }
    Ok(())
}
pub(super) fn export<D:Device>(device:&mut D,state:&mut SolverState,time:f64)->FatalResult<Snapshot>{
    if state.adaptive_retry_token!=0||state.fem_roots.attempt.is_some(){return Err(Fatal::invariant("CHECKPOINT_PENDING_ATTEMPT: only accepted boundary may be saved"));}
    let root=state.roots.transaction.as_ref().map(|r|r.export_checkpoint().map_err(|e|Fatal::invariant(e.to_string()))).transpose()?;
    let fem=state.fem_roots.runtime.as_ref().map(|r|r.export_checkpoint().map_err(Fatal::invariant)).transpose()?;
    let mut rotation=vec![0.;state.rigid_running_rotation.len()];if !rotation.is_empty(){state.rigid_running_rotation.read(device,0,&mut rotation)?;}
    let s=Snapshot{running_rotation:rotation,rotation_seeded:state.rigid_rotation_seeded,root,fem,fem_steps:state.fem_roots.accepted_steps,
        restoring:state.joints.curves.export_checkpoint(device,time)?,restoring_commit_count:state.restoring_commit_count,restoring_commit_time:state.restoring_commit_time};
    crate::checkpoint_values::validate_running_rotation(&s.running_rotation,state.rigid.n_bodies).map_err(Fatal::invariant)?;
    if s.restoring.as_ref().is_some_and(|r|r.runtime.checkpoint_commits()!=(s.restoring_commit_count,s.restoring_commit_time)){
        return Err(Fatal::invariant("CHECKPOINT_ACCEPTED_HISTORY_INVALID: rotation or commit count"));
    }
    // Snapshot export validates itself against the current static program; DTO
    // import validators intentionally reject same launch identity, so physical
    // validation there is performed against the next launch's static runtimes.
    Ok(s)
}
pub(super) fn import<D:Device>(device:&mut D,state:&mut SolverState,s:&Snapshot,time:f64)->FatalResult<()>{
    validate(s,time,state.rigid.n_bodies,state.roots.transaction.as_ref(),state.fem_roots.runtime.as_ref(),&state.joints.curves.binding_metadata).map_err(Fatal::invariant)?;
    if state.adaptive_retry_token!=0||state.fem_roots.attempt.is_some(){return Err(Fatal::invariant("CHECKPOINT_PENDING_ATTEMPT: fresh import required"));}
    let mut root_candidate=state.roots.transaction.clone();
    let mut fem_candidate=state.fem_roots.runtime.clone();
    if let (Some(runtime),Some(saved))=(&mut root_candidate,&s.root){runtime.import_checkpoint(saved.clone(),time).map_err(|e|Fatal::invariant(e.to_string()))?;}
    if let (Some(runtime),Some(saved))=(&mut fem_candidate,&s.fem){runtime.import_checkpoint(saved.clone(),time).map_err(Fatal::invariant)?;}
    let membership=if let Some(rt)=&fem_candidate {
        let all:std::collections::HashSet<_>=rt.bodies.iter().flat_map(|b|b.vertices.iter().copied()).collect();
        let after=rt.controlled_vertices();let mut fix=vec![crate::data::FixPair::default();state.fix.len()];state.fix.read(device,0,&mut fix)?;
        fix.retain(|p|!all.contains(&p.index));fix.extend(super::fem_roots::pins(rt,time,time)?);
        Some((after,fix,state.pull.host().to_vec()))
    }else{None};
    // Every CPU contract/capture/target is prepared before the first upload or
    // accepted host mutation. A device write failure is fatal, never continued.
    if !s.running_rotation.is_empty(){state.rigid_running_rotation.write(device,0,&s.running_rotation)?;}
    if let Some(curves)=&s.restoring{state.joints.curves.import_checkpoint(device,curves,time)?;}
    state.roots.transaction=root_candidate;state.fem_roots.runtime=fem_candidate;
    state.rigid_rotation_seeded=s.rotation_seeded;state.fem_roots.accepted_steps=s.fem_steps;
    state.restoring_commit_count=s.restoring_commit_count;state.restoring_commit_time=s.restoring_commit_time;
    if let Some((after,fix,pull))=membership {
        let view=super::require_scene("checkpoint FEM membership");
        unsafe {
            super::constraint::rebuild_pins(&view,&fix,&pull)?;
            state.stash_pins(device,&fix,&pull)?;
            state.restage_pin_indices(device,view.vertex_props_mut())?;
            state.restage_element_fixed(device,view.face_props_mut(),view.edge_props_mut(),view.tet_props_mut(),view.hinge_props_mut())?;
        }
        state.joints.soft.update_prescribed_vertices(device,&state.rigid,&after)?;
    }
    if let Some(contact) = &mut state.contact {
        contact.reverse_pin_face_edges.clear();
    }
    Ok(())
}
