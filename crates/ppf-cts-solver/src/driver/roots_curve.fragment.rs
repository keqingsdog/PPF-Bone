// Append this fragment to driver/roots.rs. Kept in production as an include
// so the typed native path is compiled by the full-source check.
#[derive(Default)]
struct CurveStaging {
    vertex_prop:Buffer<crate::data::VertexProp>,face_prop:Buffer<crate::data::FaceProp>,edge_prop:Buffer<crate::data::EdgeProp>,
    vertex_param:Buffer<crate::data::VertexParam>,face_param:Buffer<crate::data::FaceParam>,edge_param:Buffer<crate::data::EdgeParam>,
    margin:Buffer<f32>,refused:ppf_cts_compute::ReadbackBuffer<u32>,
}
impl Roots {
    pub fn rotating(&self)->bool {self.transaction.as_ref().is_some_and(RootState::rotating)}
    pub fn has_control_events(&self)->bool {self.transaction.as_ref().is_some_and(RootState::has_control_events)}
    pub fn retry_request(&self)->Option<f32> {self.retry_dt}
    pub fn trial_dt(&self)->Option<f32> {self.trial_dt}
    pub fn request_curve_retry(&mut self,dt:f32) {self.retry_dt=Some(dt*0.5);self.curve_retry=true;}
    pub fn retry_step(&mut self,dt:Option<f32>) {self.retry_dt=dt;self.curve_retry=false;self.trial_dt=None;}
}
fn fitted_pose(state:crate::data::PdrdRigidState)->Result<RootPose,Fault> {
    let matrix=nalgebra::Matrix3::<f64>::from_column_slice(&state.rotation.map(f64::from));
    if !matrix.iter().all(|x|x.is_finite()) || (matrix.transpose()*matrix-nalgebra::Matrix3::identity()).norm()>1e-4 || matrix.determinant()<=0.0 {
        return Err(fault("ROOT_MOTION_INITIAL_POSE: invalid physical boundary fit"));
    }
    let q=nalgebra::UnitQuaternion::from_matrix(&matrix);
    let q=q.quaternion();
    Ok(RootPose{center_world_m:std::array::from_fn(|i|f64::from(state.anchor[i])+f64::from(state.x[i])),rotation_world_wxyz:[q.w,q.i,q.j,q.k]})
}
/// Evaluate authored controls on an accepted boundary, never after a failed
/// attempt. Physical curr/prev/prev_dt and output remain untouched.
#[cfg(test)]
pub(super) static PRIVATE_TEST_BOUNDARY_COUNT:std::sync::atomic::AtomicU32=std::sync::atomic::AtomicU32::new(0);
pub unsafe fn control_boundary<D:Device>(device:&mut D,data:&crate::data::DataSet,state:&mut super::state::SolverState,param:*const crate::data::ParamSet)->super::scene::FatalResult<()> {
    #[cfg(test)]
    PRIVATE_TEST_BOUNDARY_COUNT.fetch_add(1,std::sync::atomic::Ordering::SeqCst);
    let policies=state.roots.transaction.as_ref().map(RootState::due_control_policies).unwrap_or_default();
    let due=state.roots.transaction.as_ref().map(RootState::due_controls).unwrap_or_default();
    if due.is_empty(){return Ok(());}
    // Rare physical-boundary readback records real finite-difference velocity
    // and the untouched history bits for native release/takeover receipts.
    state.positions.download(device)?;state.positions_prev.download(device)?;
    let list=super::scene::slice(&data.pdrd_vert_list);
    let mut history=Vec::new();
    for &(body,mode) in &due {
        let (first,count)=state.roots.spans[body as usize];
        let mut velocity=[0.0_f64;3];let mut hash=0xcbf29ce484222325_u64;
        for at in first as usize..(first+count) as usize {
            let v=list[at] as usize;
            for a in 0..3 {
                let curr=state.positions.host()[3*v+a];let prev=state.positions_prev.host()[3*v+a];
                if (*param).prev_dt>0.0 {velocity[a]+=(f64::from(curr)-f64::from(prev))/f64::from((*param).prev_dt)/f64::from(count);}
                for word in [curr.to_bits(),prev.to_bits()] {hash=(hash^u64::from(word)).wrapping_mul(0x100000001b3);}
            }
        }
        history.push((body,mode,velocity,hash));
    }
    // Snapshot the downloaded accepted bytes before physical fit dispatches
    // invalidate the device buffer host mirror. Never read a target as actual.
    let accepted_positions=state.positions.host().to_vec();
    let super::state::SolverState{rigid,rigid_staged,rigid_state,rigid_fit_scratch,pcg_rigid,positions,prop_vertex,roots,..}=state;
    let mut reduction=super::pdrd::Reduction{map:rigid,staged:rigid_staged};
    reduction.fit(device,reduction.staged.vert_list.handle(),prop_vertex.handle(),reduction.staged.body_prop.handle(),reduction.staged.rest_centered.handle(),positions.handle(),rigid_fit_scratch.handle(),rigid_state.handle(),data.pdrd_vert_list.size,&pcg_rigid.zero)?;
    let mut actual=Vec::new();
    let rest=super::scene::slice(&data.pdrd_rest_centered);
    let vertex_props=super::scene::slice(&data.prop.vertex);
    let geometry_rest=root_geometry::rest_vectors(rest);
    let mut measured=Vec::new();
    for &(body,_) in &due {
        // Keep the existing finite/proper physical-fit refusal as well.
        fitted_pose(rigid_state.read_one(device,body as usize)?)?;
        let (start,count)=roots.spans[body as usize];
        let geometry=root_geometry::fit_body(&accepted_positions,&geometry_rest,list,body,start,count,1.0,
            |v|vertex_props.get(v as usize).map(|p|p.pdrd_body_index)).map_err(|e|fault(format!("ROOT_MOTION_INITIAL_POSE: {e}")))?;
        actual.push((body,RootPose{center_world_m:geometry.center,rotation_world_wxyz:geometry.quaternion}));
        measured.push(geometry);
    }
    // The same representational contract as the accepted readback gate, with
    // actual body/world scale rather than a user-supplied teleport tolerance.
    let rest=super::scene::slice(&data.pdrd_rest_centered);
    let mut scale=1.0_f64;
    for &(body,pose) in &actual {
        scale=scale.max(pose.center_world_m.iter().map(|x|x.abs()).fold(0.0,f64::max));
        let (start,count)=roots.spans[body as usize];
        for y in &rest[start as usize..(start+count) as usize] {scale=scale.max(y.iter().map(|x|f64::from(*x).powi(2)).sum::<f64>().sqrt());}
    }
    let representation_tolerance=32.0*f64::from(f32::EPSILON)*scale;
    if measured.iter().any(|g|g.max_component_residual>representation_tolerance) {
        return Err(fault("ROOT_MOTION_INITIAL_POSE: actual boundary geometry is not representable within native tolerance").into());
    }
    // Evidence writes are best-effort stderr, independent of logger filtering;
    // I/O errors are ignored and cannot affect the physical transaction.
    use std::io::Write;
    for &(body,mode,velocity,hash) in &history {
        let _=writeln!(std::io::stderr().lock(),"root-control-attempt-history: time={:.17} body={} mode={:?} prev_dt={:.9e} actual_center_velocity={:?} curr_prev_bits_fnv64={:016x}",(*param).time,body,mode,(*param).prev_dt,velocity,hash);
    }
    for &(body,pose) in &actual {
        let authored=roots.transaction.as_ref().unwrap().root_pose(body,(*param).time).map_err(fault)?;
        let _=writeln!(std::io::stderr().lock(),"root-control-pose: time={:.17} body={} actual={:?} authored={:?}",(*param).time,body,pose,authored);
    }
    for &(body,mode,policy) in &policies {
        let _=writeln!(std::io::stderr().lock(),"root-control-binding: time={:.17} body={} mode={:?} pose_binding={:?}",(*param).time,body,mode,policy);
    }
    roots.transaction.as_mut().unwrap().apply_controls(&actual,32.0*f64::from(f32::EPSILON)*scale,1e-4).map_err(fault)?;
    for record in &mut roots.records {record.enabled=u32::from(roots.transaction.as_ref().unwrap().active_indices().contains(&record.body));}
    if !roots.records.is_empty(){roots.targets.write(device,0,&roots.records)?;}
    for (body,mode,velocity,hash) in history {
        let _=writeln!(std::io::stderr().lock(),"root-control-history: time={:.17} body={} mode={:?} prev_dt={:.9e} actual_center_velocity={:?} curr_prev_bits_fnv64={:016x}; control changed projection mask only",(*param).time,body,mode,(*param).prev_dt,velocity,hash);
    }
    let _=writeln!(std::io::stderr().lock(),"root-control: physical boundary {} accepted modes {:?}; native position/velocity history retained",roots.transaction.as_ref().unwrap().committed_time_s,due);
    Ok(())
}

/// Supplemental whole-physical-step curve CCD. The original Newton and final
/// intersection paths already ran. This uses ONLY local shadow prop/material
/// tables, the ORIGINAL BVH/ACCD neutral kernels and their original pair filter.
/// False asks the outer advance wrapper to restore uncommitted trial state and
/// solve a smaller freshly sampled physical step. Nothing is committed here.
pub unsafe fn curve_gate<D:Device>(device:&mut D,data:&crate::data::DataSet,state:&mut super::state::SolverState,
    attempt:&Attempt,mesh:super::contact::MeshRefs,windows:super::contact::Windows,param:*const crate::data::ParamSet)->super::scene::FatalResult<bool> {
    use crate::joint_root_motion_curve::{point_envelope,inflated_offset,upward_f32};
    let tx=state.roots.transaction.as_ref().unwrap();
    let mut moving=false;
    for (&body,&end) in attempt.body_indices.iter().zip(&attempt.target) {
        if tx.root_pose(body,attempt.start_s).map_err(fault)?.rotation_world_wxyz!=end.rotation_world_wxyz {moving=true;}
    }
    if !moving {return Ok(true);}
    if state.sphere.host().iter().any(|s|s.bowl && !s.kinematic) {
        return Err(super::scene::Fatal::invariant("ROOT_MOTION_NATIVE_CAPABILITY: rotating root clipped bowl curve CCD is not installed"));
    }
    state.positions.download(device)?;state.eval_x.download(device)?;
    let start=state.positions.host();let end=state.eval_x.host();
    let vertices=state.sizes.vertices;
    let list=super::scene::slice(&data.pdrd_vert_list);let rest=super::scene::slice(&data.pdrd_rest_centered);
    let mut margin=vec![0.0_f64;vertices];let mut owner=vec![None;vertices];
    for (&body,_) in attempt.body_indices.iter().zip(&attempt.target) {
        let (p0,d)=tx.curve_frame(body,attempt.start_s).map_err(fault)?;
        let (p1,_)=tx.curve_frame(body,attempt.end_s).map_err(fault)?;
        let (first,count)=state.roots.spans[body as usize];
        for at in first as usize..(first+count) as usize {
            let v=list[at] as usize;owner[v]=Some(body);
            let y:[f64;3]=std::array::from_fn(|a|f64::from(rest[at][a]));
            let x0=std::array::from_fn(|a|f64::from(start[3*v+a]));let x1=std::array::from_fn(|a|f64::from(end[3*v+a]));
            let qy=crate::joint_root_motion_wire::rotate_world(d.rotation_world_wxyz,y);
            let local=std::array::from_fn(|a|d.center_world_m[a]+qy[a]);
            margin[v]=point_envelope(p0,p1,local,x0,x1).map_err(fault)?;
        }
    }
    let mut vp=state.prop_vertex.host().to_vec();let mut fp=state.prop_face.host().to_vec();let mut ep=state.prop_edge.host().to_vec();
    let original_v=state.param_vertex.host();let original_f=state.param_face.host();let original_e=state.param_edge.host();
    let max_offset=original_v.iter().map(|p|f64::from(p.offset)).chain(original_f.iter().map(|p|f64::from(p.offset))).chain(original_e.iter().map(|p|f64::from(p.offset)))
        .chain(super::scene::slice(&data.constraint.mesh.param_arrays.vertex).iter().map(|p|f64::from(p.offset)))
        .chain(super::scene::slice(&data.constraint.mesh.param_arrays.face).iter().map(|p|f64::from(p.offset)))
        .chain(super::scene::slice(&data.constraint.mesh.param_arrays.edge).iter().map(|p|f64::from(p.offset))).fold(0.0,f64::max);
    let mut vparams=Vec::new();let mut fparams=Vec::new();let mut eparams=Vec::new();
    for (v,p) in vp.iter_mut().enumerate() {
        let mut material=*original_v.get(p.param_index as usize).ok_or_else(||fault("ROOT_MOTION_BODY_BINDING: vertex material index"))?;
        material.offset=inflated_offset(material.offset,margin[v],max_offset).map_err(fault)?;p.param_index=v as u32;vparams.push(material);
    }
    let faces=state.mesh_face.host();let edges=state.mesh_edge.host();
    for (i,p) in fp.iter_mut().enumerate() {
        let ids=&faces[3*i..3*i+3];let first=owner[ids[0] as usize];
        if ids.iter().any(|&v|owner[v as usize]!=first) && ids.iter().any(|&v|owner[v as usize].is_some()) {
            return Err(super::scene::Fatal::invariant("ROOT_MOTION_BODY_BINDING: root triangle membership must be closed"));
        }
        let envelope=ids.iter().map(|&v|margin[v as usize]).fold(0.0,f64::max);
        let mut material=*original_f.get(p.param_index as usize).ok_or_else(||fault("ROOT_MOTION_BODY_BINDING: face material index"))?;
        material.offset=inflated_offset(material.offset,envelope,max_offset).map_err(fault)?;p.param_index=i as u32;fparams.push(material);
    }
    for (i,p) in ep.iter_mut().enumerate() {
        let ids=&edges[2*i..2*i+2];
        if owner[ids[0] as usize]!=owner[ids[1] as usize] && ids.iter().any(|&v|owner[v as usize].is_some()) {
            return Err(super::scene::Fatal::invariant("ROOT_MOTION_BODY_BINDING: root edge membership must be closed"));
        }
        let envelope=ids.iter().map(|&v|margin[v as usize]).fold(0.0,f64::max);
        let mut material=*original_e.get(p.param_index as usize).ok_or_else(||fault("ROOT_MOTION_BODY_BINDING: edge material index"))?;
        material.offset=inflated_offset(material.offset,envelope,max_offset).map_err(fault)?;p.param_index=i as u32;eparams.push(material);
    }
    fn upload<D:Device,T:ppf_cts_compute::Pod>(device:&mut D,buffer:&mut Buffer<T>,rows:&[T],label:&'static str)->Result<(),Fault> {
        buffer.size(device,rows.len(),AllocLabel(label))?;if !rows.is_empty(){buffer.write(device,0,rows)?;}Ok(())
    }
    let curve=&mut state.roots.curve;
    upload(device,&mut curve.vertex_prop,&vp,"roots.curve.vertex.prop")?;upload(device,&mut curve.face_prop,&fp,"roots.curve.face.prop")?;upload(device,&mut curve.edge_prop,&ep,"roots.curve.edge.prop")?;
    upload(device,&mut curve.vertex_param,&vparams,"roots.curve.vertex.param")?;upload(device,&mut curve.face_param,&fparams,"roots.curve.face.param")?;upload(device,&mut curve.edge_param,&eparams,"roots.curve.edge.param")?;
    let margins=margin.into_iter().map(upward_f32).collect::<crate::joint_root_motion_wire::RootResult<Vec<_>>>().map_err(fault)?;
    upload(device,&mut curve.margin,&margins,"roots.curve.margin")?;
    let mut shadow=mesh;shadow.vertex_prop=curve.vertex_prop.handle();shadow.face_prop=curve.face_prop.handle();shadow.edge_prop=curve.edge_prop.handle();
    shadow.vertex_param=curve.vertex_param.handle();shadow.face_param=curve.face_param.handle();shadow.edge_param=curve.edge_param.handle();
    shadow.covered_point_edges = false;
    let mut curve_param=std::ptr::read(param);curve_param.line_search_max_t=1.0;
    if !(*param).disable_contact {
        let contact=state.contact.as_mut().ok_or_else(||super::scene::Fatal::invariant("ROOT_MOTION_NATIVE_CAPABILITY: curve contact layer absent"))?;
        contact.refresh_leaves(device,data,shadow,state.positions.handle(),state.eval_x.handle(),1.0,windows)?;
        let filter=contact.line_search(device,data,shadow,&curve_param,state.positions.handle(),state.eval_x.handle(),windows)?;
        if filter.overlapping_start().is_some() || filter.time_of_impact()<1.0 {return Ok(false);}
    }
    let count=data.pdrd_vert_list.size;
    curve.refused.size(device,count as usize,AllocLabel("roots.curve.analytic.refused"))?;
    let args=super::kernels::RootCurveAnalyticArgs {targets:state.roots.targets.handle(),list:state.rigid_staged.vert_list.handle(),x0:state.positions.handle(),x1:state.eval_x.handle(),margin:curve.margin.handle(),
        spheres:state.sphere.handle(),sphere_count:state.sphere.len() as u32,floors:state.floor.handle(),floor_count:state.floor.len() as u32,
        refused:curve.refused.handle(),roots:state.roots.targets.len() as u32,vertices:vertices as u32,count,seam_arena_count:0};
    device.launch("roots.curve.analytic",&args,count)?;
    curve.refused.download(device)?;
    Ok(curve.refused.host().iter().all(|&r|r==0))
}

impl Roots {
    pub unsafe fn controlled_retract<D:Device>(&self,device:&mut D,state:Handle,dtheta:Handle,running:Handle,fraction:f32)->Result<(),Fault> {
        if self.records.is_empty(){return Ok(());}
        let args=super::kernels::RootControlledRetractArgs{targets:self.targets.handle(),state,dtheta,running,bodies:self.spans.len() as u32,fraction,count:self.records.len() as u32,seam_arena_count:0};
        device.launch("roots.controlled_retract",&args,self.records.len() as u32)?;Ok(())
    }
}
