//! Solver adapter for authenticated full-node FEM controls.
use ppf_cts_compute::Device;
use crate::{data::{FixPair,Vec3f},fem_control_events::{Runtime,Attempt},joint_root_motion_state::NativeStep};
use super::{state::SolverState,scene::{Fatal,FatalResult,SceneView}};
#[derive(Default)]
pub struct FemRoots {pub runtime:Option<Runtime>,pub bindings:Vec<crate::soft_pose_output::MaterialBinding>,pub program_sha256:String,pub attempt:Option<Attempt>,pub accepted_steps:u64,pub prepared_end:Option<f64>}
fn fail(e:impl ToString)->Fatal{Fatal::invariant(format!("FEM_CONTROL_EVENTS: {}",e.to_string()))}
fn raw<D:Device>(device:&mut D,handle:ppf_cts_compute::Handle,bytes:usize)->FatalResult<Vec<u8>>{let mut data=vec![0;bytes];if bytes>0{device.read(handle,0,&mut data)?;}Ok(data)}
fn hash(bytes:&[u8])->String{use sha2::{Digest,Sha256};format!("{:x}",Sha256::digest(bytes))}
fn field_u32(bytes:&[u8],offset:usize)->u32{u32::from_le_bytes(bytes[offset..offset+4].try_into().unwrap())}
fn field_f32(bytes:&[u8],offset:usize)->f32{f32::from_le_bytes(bytes[offset..offset+4].try_into().unwrap())}
fn fixed_free_hash<T>(bytes:&[u8],fixed_offset:usize)->String{let mut masked=bytes.to_vec();for row in masked.chunks_exact_mut(std::mem::size_of::<T>()){row[fixed_offset]=0;}hash(&masked)}
pub fn pins(rt:&Runtime,start:f64,end:f64)->FatalResult<Vec<FixPair>>{
    let mut pins=Vec::new();
    for (b,s) in rt.bodies.iter().zip(&rt.states){
        let Some(target)=Runtime::target(b,s,end).map_err(fail)? else{continue;};
        let initial=Runtime::target(b,s,start).map_err(fail)?.ok_or_else(||fail("missing start target"))?;
        for ((&index,x),a) in b.vertices.iter().zip(target).zip(initial){pins.push(FixPair{position:Vec3f::from(x.map(|v|v as f32)),step_delta:Vec3f::from(std::array::from_fn::<_,3,_>(|i|(x[i]-a[i]) as f32)),ghat:0.,index,kinematic:s.mode==crate::fem_control_events::Mode::Kinematic,allow_intersection:false});}
    }
    Ok(pins)
}
impl FemRoots {
    pub fn abort(&mut self){if let Some(rt)=&mut self.runtime{rt.abort_pending();}self.attempt=None;}
    pub fn plan(&self,time:f64,requested:f32,other:NativeStep)->FatalResult<NativeStep>{
        let Some(rt)=&self.runtime else{return Ok(other);};
        if rt.accepted_time!=time{return Err(fail("host/accepted clock mismatch"));}
        rt.plan(requested,Some(other.end_s)).map_err(fail)
    }
    pub fn begin(&mut self,plan:NativeStep)->FatalResult<()>{
        if let Some(rt)=&mut self.runtime {rt.abort_pending();self.attempt=Some(rt.begin(plan.end_s,plan.integration_dt_s).map_err(fail)?);}Ok(())
    }
    pub fn shorten(&mut self,dt:f32)->FatalResult<f32>{
        if let Some(rt)=&mut self.runtime {let a=self.attempt.as_ref().ok_or_else(||fail("FEM trial absent"))?;let next=rt.shorten(a,dt).map_err(fail)?;let dt=next.dt;self.attempt=Some(next);return Ok(dt);}Ok(dt)
    }
    pub fn shorten_shared(&mut self,start:f64,end:f64,dt:f32,certified_dt:f32)->FatalResult<f32>{
        if let Some(rt)=&mut self.runtime {let a=self.attempt.as_ref().ok_or_else(||fail("shared FEM trial absent"))?;let next=rt.shorten_shared(a,start,end,dt,certified_dt).map_err(fail)?;self.attempt=Some(next);}Ok(dt)
    }
    pub fn gate_shared_clock(&self,other:Option<&crate::joint_root_motion_state::Attempt>)->FatalResult<()>{
        if let (Some(a),Some(b))=(&self.attempt,other){if a.start!=b.start_s||a.end!=b.end_s||Some(a.dt)!=b.integration_dt_s{return Err(fail("shared controllers disagree before target/assembly"));}}
        Ok(())
    }
    /// Replace only event-root rows after generic rewind_fix has shortened the
    /// user rows. Resample exact path; do not retain its rotational chord.
    pub fn retarget<D:Device>(&self,device:&mut D,fix:&mut ppf_cts_compute::Buffer<FixPair>)->FatalResult<()> {
        let (Some(rt),Some(a))=(&self.runtime,&self.attempt) else{return Ok(());};
        if rt.states.iter().all(|s|s.mode==crate::fem_control_events::Mode::Free){return Ok(());}
        let mut existing=vec![FixPair::default();fix.len()];fix.read(device,0,&mut existing)?;
        let controlled:std::collections::HashSet<_>=rt.bodies.iter().flat_map(|b|b.vertices.iter().copied()).collect();
        let targets:std::collections::HashMap<_,_>=pins(rt,a.start,a.end)?.into_iter().map(|p|(p.index,p)).collect();
        let mut found=std::collections::HashSet::new();
        for pin in &mut existing {if controlled.contains(&pin.index){*pin=*targets.get(&pin.index).ok_or_else(||fail("trial pin membership differs from accepted mode"))?;found.insert(pin.index);}}
        if targets.len()!=found.len(){return Err(fail("FEM target missing from rebuilt constraint"));}
        if !existing.is_empty(){fix.write(device,0,&existing)?;}Ok(())
    }
}
/// Called only after C5 AND the original accepted plasticity/material updates.
/// No accepted position or velocity-history buffer is written.
pub unsafe fn accept<D:Device>(device:&mut D,state:&mut SolverState,view:&SceneView,accepted_time:f64,prev_dt:f32)->FatalResult<()> {
    let mut controls=std::mem::take(&mut state.fem_roots);
    let result=(||{
        let Some(rt)=controls.runtime.as_mut() else{return Ok(());};
        let a=controls.attempt.as_ref().ok_or_else(||fail("accepted FEM trial absent"))?;
        if a.end!=accepted_time{return Err(fail("native accepted clock differs from frozen FEM endpoint"));}
        let due=rt.bodies.iter().zip(&rt.states).any(|(b,s)|b.events.get(s.cursor).is_some_and(|e|e.time_s==a.end));
        let step_trace_dir=std::env::var_os("PPF_FEM_CONTROL_ACCEPTED_TRACE_DIR").map(std::path::PathBuf::from);
        let event_trace_dir=if due{std::env::var_os("PPF_FEM_CONTROL_EVENT_TRACE_DIR").map(std::path::PathBuf::from)}else{None};
        let tracing=step_trace_dir.is_some()||event_trace_dir.is_some();
        let xyz=if due||tracing{state.positions.download(device)?;state.positions.host().to_vec()}else{Vec::new()};
        let prev_before=if tracing{state.positions_prev.download(device)?;Some(state.positions_prev.host().to_vec())}else{None};
        use crate::data::{TetProp,HingeProp,TetParam};use std::mem::{size_of,offset_of};
        let material_before=if tracing{Some((raw(device,state.prop_tet.handle(),state.prop_tet.len()*size_of::<TetProp>())?,raw(device,state.prop_hinge.handle(),state.prop_hinge.len()*size_of::<HingeProp>())?,raw(device,state.inv_rest3x3.handle(),state.inv_rest3x3.len()*4)?,raw(device,state.param_tet.handle(),state.param_tet.len()*size_of::<TetParam>())?))}else{None};
        let positions:Vec<[f64;3]>=xyz.chunks_exact(3).map(|x|[x[0] as f64,x[1] as f64,x[2] as f64]).collect();
        let bytes:Vec<u8>=xyz.iter().flat_map(|v|v.to_le_bytes()).collect();
        let before=rt.controlled_vertices();
        let segment=rt.accept(a,&positions,|uuid|{
            let binding=controls.bindings.iter().find(|b|b.body_uuid==uuid).ok_or("material binding absent")?;
            Ok(crate::soft_pose_output::root_pose(&crate::soft_pose_output::measure(binding,&bytes)?))
        }).map_err(fail)?;
        let after=rt.controlled_vertices();
        if before!=after {
            let all:std::collections::HashSet<_>=rt.bodies.iter().flat_map(|b|b.vertices.iter().copied()).collect();
            let mut fix=vec![FixPair::default();state.fix.len()];state.fix.read(device,0,&mut fix)?;fix.retain(|p|!all.contains(&p.index));fix.extend(pins(rt,rt.accepted_time,rt.accepted_time)?);
            let pull=state.pull.host().to_vec();
            super::constraint::rebuild_pins(view,&fix,&pull)?;
            state.stash_pins(device,&fix,&pull)?;
            state.restage_pin_indices(device,view.vertex_props_mut())?;
            state.restage_element_fixed(device,view.face_props_mut(),view.edge_props_mut(),view.tet_props_mut(),view.hinge_props_mut())?;
            state.joints.soft.update_prescribed_vertices(device,&state.rigid,&after)?;
        }
        controls.accepted_steps=controls.accepted_steps.checked_add(1).ok_or_else(||fail("FEM accepted step count overflow"))?;
        if tracing {
            state.positions.download(device)?;state.positions_prev.download(device)?;
            let curr_after=state.positions.host();let prev_after=state.positions_prev.host();let prev_before=prev_before.as_ref().unwrap();
            if curr_after!=xyz.as_slice()||prev_after!=prev_before.as_slice(){return Err(fail("boundary event changed accepted position/history"));}
            let mapping=state.joints.soft.prescribed_mapping_readback(device)?;
            let (tet_before,hinge_before,rest_before,param_before)=material_before.as_ref().unwrap();
            let tet_after=raw(device,state.prop_tet.handle(),tet_before.len())?;let hinge_after=raw(device,state.prop_hinge.handle(),hinge_before.len())?;let rest_after=raw(device,state.inv_rest3x3.handle(),rest_before.len())?;let param_after=raw(device,state.param_tet.handle(),param_before.len())?;
            let before_hashes=serde_json::json!({"tet_nonfixed_fields_sha256":fixed_free_hash::<TetProp>(tet_before,offset_of!(TetProp,fixed)),"hinge_nonfixed_fields_sha256":fixed_free_hash::<HingeProp>(hinge_before,offset_of!(HingeProp,fixed)),"inv_rest3x3_sha256":hash(rest_before),"tet_params_sha256":hash(param_before)});
            let after_hashes=serde_json::json!({"tet_nonfixed_fields_sha256":fixed_free_hash::<TetProp>(&tet_after,offset_of!(TetProp,fixed)),"hinge_nonfixed_fields_sha256":fixed_free_hash::<HingeProp>(&hinge_after,offset_of!(HingeProp,fixed)),"inv_rest3x3_sha256":hash(&rest_after),"tet_params_sha256":hash(&param_after)});
            if before_hashes!=after_hashes{return Err(fail("event changed live rest/plastic/material fields"));}
            let rows:Vec<_>=rt.bodies.iter().zip(&segment.states).zip(&rt.states).map(|((b,old),new)|{
                let nodes=|flat:&[f32]|b.vertices.iter().map(|v|{let i=3*(*v as usize);[flat[i],flat[i+1],flat[i+2]]}).collect::<Vec<_>>();
                let pose=|p:crate::joint_root_motion_wire::RootPose|serde_json::json!({"center_world_m":p.center_world_m,"rotation_world_wxyz":p.rotation_world_wxyz});
                let capture=|s:&crate::fem_control_events::State|s.capture.as_ref().map(|c|serde_json::json!({"capture_time_seconds":c.capture_time,"positions_world_m":c.positions,"absolute_hold_targets":c.absolute_hold_targets,"local_positions":c.locals,"material_frame":c.material_frame.map(pose),"epoch_transform":pose(c.epoch_transform)}));
                let tet_rows=|props:&[u8],params:&[u8]|view.tets().iter().enumerate().filter(|(_,t)|t.iter().any(|v|b.vertices.binary_search(v).is_ok())).map(|(index,t)|{let row=&props[index*size_of::<TetProp>()..(index+1)*size_of::<TetProp>()];let pi=field_u32(row,offset_of!(TetProp,param_index)) as usize;let param=&params[pi*size_of::<TetParam>()..(pi+1)*size_of::<TetParam>()];serde_json::json!({"tet_index":index,"vertex_indices":[t[0],t[1],t[2],t[3]],"fixed":row[offset_of!(TetProp,fixed)]!=0,"rest_excluded":row[offset_of!(TetProp,rest_excluded)]!=0,"param_index":pi,"mu":field_f32(param,offset_of!(TetParam,mu)),"lambda":field_f32(param,offset_of!(TetParam,lambda)),"plasticity":field_f32(param,offset_of!(TetParam,plasticity)),"deform_damping":field_f32(param,offset_of!(TetParam,deform_damping))})}).collect::<Vec<_>>();
                let hinge_rows=|props:&[u8]|view.hinges().iter().enumerate().filter(|(_,t)|t.iter().any(|v|b.vertices.binary_search(v).is_ok())).map(|(index,t)|{let row=&props[index*size_of::<HingeProp>()..(index+1)*size_of::<HingeProp>()];serde_json::json!({"hinge_index":index,"vertex_indices":[t[0],t[1],t[2],t[3]],"fixed":row[offset_of!(HingeProp,fixed)]!=0,"rest_angle":field_f32(row,offset_of!(HingeProp,rest_angle))})}).collect::<Vec<_>>();
                serde_json::json!({"body_uuid":b.body_uuid,"vertex_indices":b.vertices,"immutable_rest_local_positions":rt.immutable_rest_locals[rt.bodies.iter().position(|candidate|candidate.body_uuid==b.body_uuid).expect("trace body belongs to runtime")],"event_applied":old.cursor!=new.cursor,"old_segment":{"mode":old.mode,"pose_binding":old.binding,"epoch":old.epoch,"event_cursor":old.cursor,"capture":capture(old)},"new_state":{"mode":new.mode,"pose_binding":new.binding,"epoch":new.epoch,"event_cursor":new.cursor,"capture":capture(new)},"before_event":{"curr":nodes(&xyz),"prev":nodes(prev_before),"prev_dt_seconds":prev_dt},"after_event":{"curr":nodes(curr_after),"prev":nodes(prev_after),"prev_dt_seconds":prev_dt},"private_vertex_to_free_after":b.vertices.iter().map(|v|mapping.get(*v as usize).copied()).collect::<Vec<_>>(),"tet_material_before":tet_rows(tet_before,param_before),"tet_material_after":tet_rows(&tet_after,&param_after),"hinge_before":hinge_rows(hinge_before),"hinge_after":hinge_rows(&hinge_after)})
            }).collect();
            for (directory,is_step) in event_trace_dir.into_iter().map(|p|(p,false)).chain(step_trace_dir.into_iter().map(|p|(p,true))) {
            let trace=serde_json::json!({"kind":if is_step{"ppf.fem_control.accepted_step_trace"}else{"ppf.fem_control.accepted_event_trace"},"version":1,"launch_id":rt.launch_id,"control_program_sha256":controls.program_sha256,"accepted_time_seconds":accepted_time,"accepted_time_f32":accepted_time as f32,"accepted_steps":controls.accepted_steps,"accepted_bracket_start_seconds":segment.start,"rows":rows,"history_unchanged":true,"controlled_vertices_after":after,"private_none_sentinel":u32::MAX,"material_state_before":before_hashes,"material_state_after":after_hashes,"material_kernel_executed_for_trace":false});
            std::fs::create_dir_all(&directory).map_err(fail)?;
            let path=directory.join(format!("fem_{}_{}_step_{}.json",if is_step{"step"}else{"event"},rt.launch_id,controls.accepted_steps));let tmp=path.with_extension("json.tmp");
            if path.exists(){return Err(fail("accepted trace identity already exists"));}
            use std::io::Write;let mut file=std::fs::OpenOptions::new().write(true).create_new(true).open(&tmp).map_err(fail)?;file.write_all(&serde_json::to_vec(&trace).map_err(fail)?).map_err(fail)?;file.sync_all().map_err(fail)?;std::fs::rename(tmp,path).map_err(fail)?;
            }
        }
        controls.attempt=None;Ok(())
    })();
    state.fem_roots=controls;result
}
