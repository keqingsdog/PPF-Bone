//! Separate restoring V1 uploads. No legacy record acquires a field.
use ppf_cts_compute::{AllocLabel,Buffer,Device,Fault,Handle,ReadbackBuffer};
use crate::restoring_curve_gpu::{CurveDescriptor,CurveNode,CurveBinding,CurveHistory};
unsafe impl ppf_cts_compute::Pod for CurveDescriptor {}
unsafe impl ppf_cts_compute::Pod for CurveNode {}
unsafe impl ppf_cts_compute::Pod for CurveBinding {}
unsafe impl ppf_cts_compute::Pod for CurveHistory {}
#[derive(Clone,Copy)]
pub struct View {
 pub descriptors:Handle,pub nodes:Handle,pub bindings:Handle,pub history:Handle,pub fault:Handle,pub ball_fault:Handle,pub angular_fault:Handle,pub fem_fault:Handle,pub stitch_fault:Handle,pub observations:Handle,pub dummy:Handle,pub dt:f32,
 pub descriptor_count:u32,pub node_count:u32,pub binding_count:u32,pub history_count:u32,
}
#[derive(Default)]
pub struct Curves {
 pub(crate) observations:Buffer<f32>,pub(crate) binding_metadata:Vec<CurveBinding>,pub(crate) runtime:super::restoring_receipt::RuntimeStats,
 empty_transport:bool,
 descriptors:Buffer<CurveDescriptor>,nodes:Buffer<CurveNode>,bindings:Buffer<CurveBinding>,
 pub(crate) history:Buffer<CurveHistory>,empty_history:Buffer<CurveHistory>,fault:ReadbackBuffer<u32>,ball_fault:ReadbackBuffer<u32>,angular_fault:ReadbackBuffer<u32>,fem_fault:ReadbackBuffer<u32>,stitch_fault:ReadbackBuffer<u32>,pub(crate) dt:f32,dummy:Buffer<f32>,pub(crate) accepted_time:Option<u64>,pub(crate) epoch:u64,
}
#[derive(Clone,Debug,PartialEq,serde::Serialize,serde::Deserialize)]
pub(crate) struct Snapshot {
 pub history:Vec<CurveHistory>,pub dt:f32,pub accepted_time:Option<u64>,pub epoch:u64,
 pub runtime:super::restoring_receipt::RuntimeStats,
}
impl Snapshot {
 pub fn validate(&self,time:f64,bindings:&[CurveBinding])->Result<(),String>{
  let used:Vec<u32>=bindings.iter().map(|b|b.history_index).collect();
  crate::checkpoint_values::validate_curve_history(&self.history,self.dt,self.accepted_time,self.epoch,&used,time)?;
  self.runtime.validate_checkpoint(time,bindings.len())
 }
}
impl Curves {
 /// In-process accepted-state transaction, not a portable checkpoint. Do not
 /// inspect or clear fault banks: failed-candidate diagnostics remain intact.
 pub(crate) fn snapshot_trial<D:Device>(&self,device:&mut D)->Result<Option<Snapshot>,Fault>{
  if !self.active(){return Ok(None);}
  let mut history=vec![CurveHistory::default();self.history.len()];self.history.read(device,0,&mut history)?;
  Ok(Some(Snapshot{history,dt:self.dt,accepted_time:self.accepted_time,epoch:self.epoch,runtime:self.runtime.clone()}))
 }
 pub(crate) fn restore_trial<D:Device>(&mut self,device:&mut D,s:&Option<Snapshot>)->Result<(),Fault>{
  match s {Some(s) if self.active()=>{self.history.write(device,0,&s.history)?;self.dt=s.dt;self.accepted_time=s.accepted_time;self.epoch=s.epoch;self.runtime=s.runtime.clone();Ok(())},None if !self.active()=>Ok(()),_=>Err(Fault::Platform{call:"restoring trial rollback",detail:"binding state changed".into()})}
 }
 pub(crate) fn export_checkpoint<D:Device>(&mut self,device:&mut D,time:f64)->Result<Option<Snapshot>,Fault>{
  if !self.active(){return Ok(None);}
  self.check(device,"checkpoint accepted boundary")?;
  let mut history=vec![CurveHistory::default();self.history.len()];self.history.read(device,0,&mut history)?;
  let s=Snapshot{history,dt:self.dt,accepted_time:self.accepted_time,epoch:self.epoch,runtime:self.runtime.clone()};
  s.validate(time,&self.binding_metadata).map_err(|detail|Fault::Platform{call:"checkpoint restoring",detail})?;Ok(Some(s))
 }
 pub(crate) fn import_checkpoint<D:Device>(&mut self,device:&mut D,s:&Snapshot,time:f64)->Result<(),Fault>{
  s.validate(time,&self.binding_metadata).map_err(|detail|Fault::Platform{call:"checkpoint restoring",detail})?;
  self.history.write(device,0,&s.history)?;self.dt=s.dt;self.accepted_time=s.accepted_time;self.epoch=s.epoch;self.runtime=s.runtime.clone();Ok(())
 }
 pub fn active(&self)->bool {self.bindings.len()!=0}
 pub(crate) fn enable_empty_transport<D:Device>(&mut self,device:&mut D,lanes:usize)->Result<(),Fault>{
  if self.active(){return Ok(());}
  if self.empty_transport && self.empty_history.len()!=0 && [&self.fault,&self.ball_fault,&self.angular_fault,&self.fem_fault,&self.stitch_fault].iter().all(|bank|bank.len()>=lanes.max(1)){return Ok(());}
  self.dummy.size(device,64,AllocLabel("restoring.empty_transport"))?;
  // Generated entries resolve every pointer even when its count is zero.
  // CurveHistory contains a u64 and needs 8-byte alignment, which the f32
  // transport dummy cannot promise. Keep its zero-count placeholder typed.
  self.empty_history.size(device,1,AllocLabel("restoring.empty_history"))?;
  self.empty_history.write(device,0,&[CurveHistory::default()])?;
  for bank in [&mut self.fault,&mut self.ball_fault,&mut self.angular_fault,&mut self.fem_fault,&mut self.stitch_fault]{bank.size(device,lanes.max(1),AllocLabel("restoring.empty_fault"))?;bank.seed(device,&vec![0;lanes.max(1)])?;}
  self.empty_transport=true;Ok(())
 }
 pub fn install<D:Device>(&mut self,device:&mut D,tables:&crate::restoring_curve::PackedTables,bindings:&[CurveBinding],fault_lanes:usize)->Result<(),Fault>{
  if bindings.is_empty(){return Ok(());}let history_count=bindings.iter().filter(|b|b.history_index!=u32::MAX).map(|b|b.history_index as usize+1).max().unwrap_or(0);
  self.descriptors.size(device,tables.descriptors.len(),AllocLabel("restoring.descriptors"))?;self.descriptors.write(device,0,&tables.descriptors)?;
  self.nodes.size(device,tables.nodes.len(),AllocLabel("restoring.nodes"))?;self.nodes.write(device,0,&tables.nodes)?;
  self.bindings.size(device,bindings.len(),AllocLabel("restoring.bindings"))?;self.bindings.write(device,0,bindings)?;
  // Angular-only descriptors still pass valid unused history storage to entries.
  self.history.size(device,history_count.max(1),AllocLabel("restoring.accepted_history"))?;self.history.write(device,0,&vec![CurveHistory::default();history_count.max(1)])?;
  self.fault.size(device,fault_lanes.max(bindings.len()).max(1),AllocLabel("restoring.fault"))?;
  for bank in [&mut self.ball_fault,&mut self.angular_fault,&mut self.fem_fault,&mut self.stitch_fault]{bank.size(device,fault_lanes.max(bindings.len()).max(1),AllocLabel("restoring.family_fault"))?;}
  self.dummy.size(device,64,AllocLabel("restoring.unused_transport"))?;
  self.observations.size(device,bindings.len().checked_mul(6).ok_or_else(||Fault::Platform{call:"restoring observations",detail:"observation count overflow".into()})?,AllocLabel("restoring.actual_assembly_observations"))?;
  self.observations.write(device,0,&vec![0.;self.observations.len()])?;
  self.binding_metadata=bindings.to_vec();self.runtime=super::restoring_receipt::RuntimeStats::default();
  self.accepted_time=None;self.epoch=0;self.dt=0.;self.clear(device)?;Ok(())
 }
 pub fn reset_observations<D:Device>(&mut self,device:&mut D)->Result<(),Fault>{if self.active(){self.check(device,"before observation reset")?;self.observations.write(device,0,&vec![0.;self.observations.len()])?;}Ok(())}
 pub fn view(&mut self)->Option<View>{if !self.active(){if !self.empty_transport{return None;}let d=self.dummy.handle();return Some(View{descriptors:d,nodes:d,bindings:d,history:self.empty_history.handle(),fault:self.fault.handle(),ball_fault:self.ball_fault.handle(),angular_fault:self.angular_fault.handle(),fem_fault:self.fem_fault.handle(),stitch_fault:self.stitch_fault.handle(),observations:d,dt:self.dt,dummy:d,descriptor_count:0,node_count:0,binding_count:0,history_count:0});}Some(View{descriptors:self.descriptors.handle(),nodes:self.nodes.handle(),bindings:self.bindings.handle(),history:self.history.handle(),fault:self.fault.handle(),ball_fault:self.ball_fault.handle(),angular_fault:self.angular_fault.handle(),fem_fault:self.fem_fault.handle(),stitch_fault:self.stitch_fault.handle(),observations:self.observations.handle(),dt:self.dt,dummy:self.dummy.handle(),descriptor_count:self.descriptors.len() as u32,node_count:self.nodes.len() as u32,binding_count:self.bindings.len() as u32,history_count:self.history.len() as u32})}
 /// Candidate dt changes on every retry, but accepted-history identity does not.
 pub fn prepare_step(&mut self,time:f64,dt:f32)->Result<Option<u64>,Fault>{
  if !self.active(){return Ok(None);}
  if !time.is_finite()||time<0.||!dt.is_finite()||dt<=0.{return Err(Fault::Platform{call:"restoring history",detail:"RESTORING_CONTEXT_INVALID: invalid physical start time or dt".into()});}
  let time=if time==0.{0.}else{time};
  if self.accepted_time.is_some_and(|old|time<f64::from_bits(old)){return Err(Fault::Platform{call:"restoring history",detail:"RESTORING_HISTORY_INVALID: accepted physical time moved backwards".into()});}
  let next=if self.accepted_time==Some(time.to_bits()){None}else{Some(self.epoch.checked_add(1).ok_or_else(||Fault::Platform{call:"restoring history",detail:"RESTORING_HISTORY_INVALID: epoch overflow".into()})?)};
  self.dt=dt;
  Ok(next)
 }
 /// Commit only after the history capture dispatch has passed its typed gate.
 pub fn commit_capture(&mut self,time:f64,epoch:u64){self.accepted_time=Some((if time==0.{0.}else{time}).to_bits());self.epoch=epoch;}
 pub fn clear<D:Device>(&mut self,device:&mut D)->Result<(),Fault>{if self.active()||self.empty_transport{self.check(device,"before clearing restoring faults")?;for bank in [&mut self.fault,&mut self.ball_fault,&mut self.angular_fault,&mut self.fem_fault,&mut self.stitch_fault]{bank.seed(device,&vec![0;bank.len()])?;}}Ok(())}
 /// Capture every fault before another dispatch can overwrite a bank.
 pub fn snapshot_faults<D:Device>(&mut self,device:&mut D)->Result<Vec<(&'static str,usize,u32)>,Fault>{
  if !self.active()&&!self.empty_transport{return Ok(Vec::new());}
  for bank in [&mut self.fault,&mut self.ball_fault,&mut self.angular_fault,&mut self.fem_fault,&mut self.stitch_fault]{bank.download(device)?;}
  let mut faults=Vec::new();
  for (family,bank) in [("history",&self.fault),("rigid-anchor",&self.ball_fault),("angular",&self.angular_fault),("fem",&self.fem_fault),("stitch",&self.stitch_fault)] {
   faults.extend(bank.host().iter().copied().enumerate().filter(|(_,code)|*code!=0).map(|(lane,code)|(family,lane,code)));
  }
  Ok(faults)
 }
 pub(crate) fn consume_material_faults<D:Device>(&mut self,device:&mut D)->Result<(),Fault>{
  let faults=self.snapshot_faults(device)?;
  if !faults.iter().all(|(family,_,code)|matches!(*family,"angular"|"fem")&&*code==8){return Err(Fault::Platform{call:"material spin retry",detail:format!("unexpected fault bank {faults:?}")});}
  for bank in [&mut self.fault,&mut self.ball_fault,&mut self.angular_fault,&mut self.fem_fault,&mut self.stitch_fault]{bank.seed(device,&vec![0;bank.len()])?;}Ok(())
 }
 /// Consume only the exact candidate fault snapshot after physical rollback.
 pub fn consume_candidate_faults<D:Device>(&mut self,device:&mut D,expected:&[(&'static str,usize,u32)])->Result<(),Fault>{
  if !expected.iter().all(|(family,_,code)|*family=="angular" && matches!(*code,5|6|134|135))
   || self.snapshot_faults(device)? != expected {
   return Err(Fault::Platform{call:"restoring retry",detail:"fault snapshot mismatch or non-retryable fault".into()});
  }
  for bank in [&mut self.fault,&mut self.ball_fault,&mut self.angular_fault,&mut self.fem_fault,&mut self.stitch_fault]{bank.seed(device,&vec![0;bank.len()])?;}
  Ok(())
 } pub fn fault_from_snapshot(stage:&str,faults:&[(&'static str,usize,u32)],retry_eligible:bool)->Option<Fault>{
  let &(family,lane,code)=faults.first()?;
  let category=match code{160=>"RESTORING_CONTEXT_INVALID",161=>"RESTORING_BINDING_INVALID",162=>"RESTORING_SCALAR_INVALID",163=>"RESTORING_HISTORY_INVALID",164=>"RESTORING_ARITHMETIC_FAULT",_=>"RESTORING_GEOMETRY_INVALID"};
  Some(Fault::Platform{call:"restoring curve",detail:format!("{category}: {stage}: family {family} lane {lane} fault {code}; retry_eligible={retry_eligible}; all_curves_faults={faults:?}")})
 }
 pub fn check<D:Device>(&mut self,device:&mut D,stage:&str)->Result<(),Fault>{
  let faults=self.snapshot_faults(device)?;
  if let Some(fault)=Self::fault_from_snapshot(stage,&faults,false){
   if faults.iter().any(|(_,_,code)|*code==162) {
    // Diagnostic-only readback. Preserve the original typed fatal even if
    // the device or stderr fails while gathering observations.
    let _=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||{
     let mut rows=vec![0f32;self.observations.len()];
     match self.observations.read(device,0,&mut rows) {
      Ok(())=>{
       let mut found=false;
       for (index,binding) in self.binding_metadata.iter().enumerate() {
        let row=&rows[6*index..6*index+6];
        let matching=faults.iter().any(|(family,lane,code)|{
         *code==162 && *lane==binding.target_index as usize &&
          ((*family=="fem" && matches!(binding.kind,1|3)) ||
           (*family=="angular" && binding.kind==0) ||
           (*family=="rigid-anchor" && matches!(binding.kind,2|5)) ||
           (*family=="stitch" && binding.kind==4))
        });
        if matching && row[0]==-162. {
         found=true;
         eprintln!("RESTORING_SCALAR_OBSERVATION stage={stage} binding={index} kind={} target={} channel={} table={} dt={:.9e} dt_bits={:#010x} coordinate={:.9e} coordinate_bits={:#010x} args={:?} arg_bits={:?} subtype_reason={:.9e} subtype_reason_bits={:#010x}",
          binding.kind,binding.target_index,binding.channel,binding.table_index,self.dt,self.dt.to_bits(),row[1],row[1].to_bits(),&row[2..5],row[2..5].iter().map(|v|v.to_bits()).collect::<Vec<_>>(),row[5],row[5].to_bits());
        }
       }
       if !found {eprintln!("RESTORING_SCALAR_OBSERVATION stage={stage} no_matching_diagnostic");}
      },
      Err(error)=>eprintln!("RESTORING_SCALAR_OBSERVATION stage={stage} read_failed={error:?}"),
     }
    }));
   }
   return Err(fault);
  }
  Ok(())
 }}/// Capture only from accepted positions/reference fits. Retry at identical time
/// retains both the same bytes and the same epoch; apply entries never write it.
pub unsafe fn begin_step<D:Device>(device:&mut D,state:&mut super::state::SolverState,time:f64,dt:f32)->super::scene::FatalResult<()> {
 if !state.joints.curves.active(){return Ok(());}
 // A previous failure must never be erased by beginning another trial.
 state.joints.curves.check(device,"before physical substep")?;
 let epoch=state.joints.curves.prepare_step(time,dt)?;
 state.joints.curves.clear(device)?;
 // Every trial requires fresh actual assembly writes; old replaced flags cannot count.
 state.joints.curves.reset_observations(device)?;
 let reference=state.joints.reference_handle();let view=state.joints.view(reference,&state.rigid,dt);let c=view.curves.expect("active curves");let soft=view.soft;
 if let Some(epoch)=epoch {
 let args=super::kernels::RestoringHistoryCaptureArgs{bindings:c.bindings,binding_count:c.binding_count,count:c.binding_count,history:c.history,history_count:c.history_count,ball:if view.count>0{view.ball}else{c.dummy},ball_count:view.count,fem_cache:soft.map_or(c.dummy,|v|v.cache),fem_count:soft.map_or(0,|v|v.count),reference:if view.bodies>0{reference}else{c.dummy},bodies:view.bodies,accepted_xyz:state.positions.handle(),vertex_count:state.sizes.vertices as u32,stitch_indices:if state.sizes.stitches>0{state.stitch.index.handle()}else{c.dummy},stitch_weights:if state.sizes.stitches>0{state.stitch.weight.handle()}else{c.dummy},stitch_count:state.sizes.stitches as u32,epoch_low:epoch as u32,epoch_high:(epoch>>32) as u32,fault:c.fault,seam_arena_count:0};
 device.launch("restoring.history.capture",&args,c.binding_count)?;state.joints.curves.check(device,"accepted step history")?;
 state.joints.curves.commit_capture(time,epoch);
 }
 // Retry retains q0/d0/epoch but checks geometry using its actual candidate dt.
 view.add(device,3,c.dummy,c.dummy)?;
 if state.sizes.stitches>0 {
  let args=super::kernels::RestoringStitchApplyArgs{xyz:state.positions.handle(),vertex_count:state.sizes.vertices as u32,stitch_indices:state.stitch.index.handle(),stitch_weights:state.stitch.weight.handle(),stitch_count:state.sizes.stitches as u32,dt:c.dt,action:3,gradient:c.dummy,hessian:c.dummy,energy:c.dummy,fault:c.stitch_fault,observations:c.observations,descriptors:c.descriptors,nodes:c.nodes,bindings:c.bindings,history:c.history,descriptor_count:c.descriptor_count,node_count:c.node_count,binding_count:c.binding_count,history_count:c.history_count,count:state.sizes.stitches as u32,seam_arena_count:0};
  device.launch("restoring.stitch.gate",&args,state.sizes.stitches as u32)?;
 }
 state.joints.curves.check(device,"accepted step restoring geometry")?;Ok(())
}

#[cfg(all(test,feature="cpu"))]
mod empty_transport_tests {
 use super::*;

 #[test]
 fn history_placeholder_is_aligned_and_not_active_history() {
  let mut device=super::super::launch::host_device();
  // Perturb preceding arena allocation; correctness must not depend on a
  // float dummy coincidentally landing on an eight-byte boundary.
  let mut preceding=Buffer::<u32>::default();
  preceding.size(&mut device,1,AllocLabel("test.empty_transport.preceding")).unwrap();
  let mut curves=Curves::default();curves.enable_empty_transport(&mut device,3).unwrap();
  assert!(!curves.active());assert_eq!(curves.history.len(),0);
  let view=curves.view().unwrap();assert_eq!(view.dummy.off%8,4);
  assert_eq!(view.history.off%8,0);assert_eq!(view.history_count,0);
  assert_eq!((view.descriptor_count,view.node_count,view.binding_count),(0,0,0));
  assert!(curves.snapshot_trial(&mut device).unwrap().is_none());
  assert!(curves.export_checkpoint(&mut device,0.).unwrap().is_none());
  curves.restore_trial(&mut device,&None).unwrap();
  curves.enable_empty_transport(&mut device,3).unwrap();
  assert_eq!(curves.view().unwrap().history,view.history);
 }
}



impl View {
    pub(super) fn with_preview_buffers(
        mut self,
        map: &mut impl FnMut(Handle, bool) -> Result<Handle, Fault>,
    ) -> Result<Self, Fault> {
        self.fault = map(self.fault, true)?;
        self.ball_fault = map(self.ball_fault, true)?;
        self.angular_fault = map(self.angular_fault, true)?;
        self.fem_fault = map(self.fem_fault, true)?;
        self.stitch_fault = map(self.stitch_fault, true)?;
        // Some FEM families read earlier observations during the product.
        // Keep their bits while isolating the buffer from all trial writes.
        self.observations = map(self.observations, false)?;
        Ok(self)
    }
}
