//! Device storage for the immutable packed forest. Explicitly released by caller.
use ppf_cts_compute::{AllocLabel,Buffer,Device,Encoder,EncoderExt,Fault,Handle,KernelArgs};
use super::kernels::BodyFactorApplyArgs;
use super::body_factor_pack::PackedFactor;

pub struct BodyFactorDevice { buffers:Vec<Buffer<u32>>, lengths:[u32;10], pub node_count:u32, pub component_count:u32, pub body_count:u32, pub root_count:u32, instance:u64, live:bool }
fn fail(detail:&str)->Fault {Fault::Platform{call:"body factor device",detail:detail.into()}}
pub(super) fn upload_failure(call:&'static str, primary:Fault, cleanup:Result<(),Fault>)->Fault {
 match cleanup {
  Ok(()) => primary,
  Err(cleanup) => Fault::Platform{call,detail:format!("upload: {primary}; cleanup: {cleanup}")},
 }
}
#[cfg(test)]
mod upload_failure_tests {
 use super::*;
 #[test]
 fn preserves_primary_when_cleanup_succeeds() {
  let primary=Fault::Alloc{label:AllocLabel("factor.test"),bytes:16,detail:"upload failed".into()};
  assert!(matches!(upload_failure("factor cleanup",primary,Ok(())),Fault::Alloc{label:AllocLabel("factor.test"),bytes:16,..}));
 }
 #[test]
 fn reports_both_failures_when_cleanup_fails() {
  let primary=Fault::Alloc{label:AllocLabel("factor.test"),bytes:16,detail:"upload failed".into()};
  let cleanup=Fault::Platform{call:"free test",detail:"release failed".into()};
  match upload_failure("body factor upload cleanup",primary,Err(cleanup)) {
   Fault::Platform{call,detail} => {assert_eq!(call,"body factor upload cleanup");assert!(detail.contains("upload failed"));assert!(detail.contains("release failed"));},
   other => panic!("unexpected fault: {other:?}"),
  }
 }
}
impl BodyFactorDevice {
 /// Allocation is transactional: every acquired handle is freed on failure.
 pub fn upload(device:&mut impl Device,p:&PackedFactor)->Result<Self,Fault>{
  let instance=device.instance_id().ok_or_else(||fail("device identity unavailable"))?;
  let caps=device.group_limits().ok_or_else(||fail("group limits unavailable"))?;
  if device.info().backend!="cuda" || caps.max_threads<256 || caps.max_scratch_bytes<16416 || device.missing().contains(&BodyFactorApplyArgs::KERNEL){return Err(fail("unsupported device"));}
  if p.version!=1 || p.node_count==0 || p.component_count==0 || p.component_count>p.node_count || p.body_count<p.node_count{return Err(fail("unsupported factor extent"));}
  let arrays:[&[u32];10]=[&p.body_index,&p.component_offsets,&p.parent,&p.depth,&p.max_depth,&p.child_offsets,&p.children,&p.root_body_indices,&p.l_words,&p.w_words];
  let mut lengths=[0u32;10];for (i,a) in arrays.iter().enumerate(){lengths[i]=u32::try_from(a.len()).map_err(|_|fail("extent overflow"))?;}
  let n=p.node_count as usize;let c=p.component_count as usize;let factor_words=n.checked_mul(72).ok_or_else(||fail("extent overflow"))?;
  if arrays[0].len()!=n || arrays[1].len()!=c+1 || arrays[2].len()!=n || arrays[3].len()!=n || arrays[4].len()!=c || arrays[5].len()!=n+1 || arrays[6].len()!=n-c || arrays[7].len()!=p.body_count as usize-n || arrays[8].len()!=factor_words || arrays[9].len()!=factor_words{return Err(fail("packed extent"));}
  if p.component_offsets.first()!=Some(&0)||p.component_offsets.last()!=Some(&p.node_count)||p.child_offsets.first()!=Some(&0)||p.child_offsets.last()!=Some(&(p.children.len() as u32)) {return Err(fail("packed offsets"));}
  let mut seen=vec![false;p.body_count as usize];for &b in p.body_index.iter().chain(&p.root_body_indices){let b=b as usize;if b>=seen.len()||seen[b]{return Err(fail("body mapping"));}seen[b]=true;}
  let mut child_seen=vec![false;n];
  for ci in 0..c {let start=p.component_offsets[ci] as usize;let end=p.component_offsets[ci+1] as usize;if start>=end||end>n||end-start>170{return Err(fail("component offsets"));}let mut max=0;
   for i in start..end {let pa=p.parent[i];if i==start {if pa!=u32::MAX||p.depth[i]!=0{return Err(fail("component root"));}} else {let parent=pa as usize;if parent<start||parent>=i||p.depth[i]!=p.depth[parent]+1{return Err(fail("parent depth"));}}max=max.max(p.depth[i]);
    let a=p.child_offsets[i] as usize;let b=p.child_offsets[i+1] as usize;if a>b||b>p.children.len(){return Err(fail("child offsets"));}let mut prev=u32::MAX;for &ch in &p.children[a..b]{let x=ch as usize;if ch>=prev||x>=end||x<=i||p.parent[x]!=i as u32||child_seen[x]{return Err(fail("children"));}child_seen[x]=true;prev=ch;}
    for k in 0..36{for words in [&p.l_words,&p.w_words]{let j=72*i+2*k;let v=f64::from_bits(words[j] as u64|((words[j+1] as u64)<<32));if !v.is_finite()||(std::ptr::eq(words,&p.l_words)&&k/6==k%6&&v<=0.){return Err(fail("factor value"));}}}
   }if p.max_depth[ci]!=max{return Err(fail("component depth"));}
  }
  for i in 0..n {if (p.parent[i]!=u32::MAX)!=child_seen[i]{return Err(fail("child coverage"));}}
  let mut owner=Self{buffers:(0..12).map(|_|Buffer::none()).collect(),lengths,node_count:p.node_count,component_count:p.component_count,body_count:p.body_count,root_count:lengths[7],instance,live:true};
  let result=(|| {for (i,data) in arrays.iter().enumerate(){owner.buffers[i].size(device,data.len().max(1),AllocLabel("body_factor.data"))?;if !data.is_empty(){owner.buffers[i].write(device,0,data)?;}}for i in 10..12{owner.buffers[i].size(device,1,AllocLabel("body_factor.control"))?;}Ok(())})();
  if let Err(e)=result{return Err(upload_failure("body factor upload cleanup",e,owner.free(device)))}Ok(owner)
 }
 /// Build the immutable record after the caller has validated live spans,
 /// device identity and non-overlap. The owner, inputs and any region must
 /// remain live without grow/free until the region has been released.
 fn args(&self,rhs:Handle,z:Handle,body_base:u32,stop:Option<Handle>)->Result<BodyFactorApplyArgs,Fault>{
  if !self.live{return Err(fail("owner freed"));}
  // Dummy u32 zero has the same representation as f32 +0.0 for disabled stop.
  let stop_handle=stop.unwrap_or_else(||self.buffers[11].handle());
  let l=self.lengths;let h=|i:usize|self.buffers[i].handle();
  Ok(BodyFactorApplyArgs{body_index:h(0),component_offsets:h(1),parent:h(2),depth:h(3),max_depth:h(4),child_offsets:h(5),children:h(6),root_body_indices:h(7),l_words:h(8),w_words:h(9),rhs,z,fault:h(10),stop:stop_handle,node_count:self.node_count,component_count:self.component_count,body_count:self.body_count,root_count:self.root_count,body_base,stop_enabled:u32::from(stop.is_some()),body_index_len:l[0],component_offsets_len:l[1],parent_len:l[2],depth_len:l[3],max_depth_len:l[4],child_offsets_len:l[5],children_len:l[6],root_body_indices_len:l[7],l_words_len:l[8],w_words_len:l[9],rhs_len:rhs.size,z_len:z.size,fault_len:1,stop_len:stop_handle.size,groups:self.component_count,seam_arena_count:0})
 }
 /// Clear once when recording a solve, not before each replay.
 pub fn encode_clear_fault(&self,encoder:&mut dyn Encoder)->Result<(),Fault>{if !self.live || encoder.instance_id()!=Some(self.instance){return Err(fail("owner device mismatch"));}encoder.validate_current_span(self.fault_handle(),4,4,4)?;ppf_cts_compute::encode_zero(encoder,self.fault_handle(),4)}
 /// Encodes a single group per component. Caller retains all buffers and the
 /// region without freeing or growing allocations until release.
 pub unsafe fn encode_apply(&self,encoder:&mut dyn Encoder,rhs:Handle,z:Handle,body_base:u32,stop:Option<Handle>)->Result<(),Fault>{
  unsafe { self.encode_apply_with_fault_target(encoder,rhs,z,body_base,stop,None) }
 }
 /// The external word is borrowed for this region; the caller clears and checks it.
 pub unsafe fn encode_apply_with_fault_target(&self,encoder:&mut dyn Encoder,rhs:Handle,z:Handle,body_base:u32,stop:Option<Handle>,fault:Option<Handle>)->Result<(),Fault>{
  if !self.live || encoder.instance_id()!=Some(self.instance){return Err(fail("owner device mismatch"));}
  let needed=(body_base as usize).checked_add((self.body_count as usize).checked_mul(6).ok_or_else(||fail("body extent"))?).ok_or_else(||fail("body extent"))?;
  if needed>rhs.size as usize||needed>z.size as usize{return Err(fail("mixed extent"));}
  let mut spans:Vec<(Handle,usize)>=Vec::new();
  for (i,b) in self.buffers.iter().enumerate(){let len=if i<10 {self.lengths[i] as usize} else {1};let bytes=len.max(1).checked_mul(4).ok_or_else(||fail("extent"))?;encoder.validate_current_span(b.handle(),4,bytes,4)?;spans.push((b.handle(),bytes));}
  for h in [rhs,z] {let bytes=(h.size as usize).checked_mul(4).ok_or_else(||fail("extent"))?;encoder.validate_current_span(h,4,bytes,4)?;spans.push((h,bytes));}
  if let Some(h)=stop {let bytes=(h.size as usize).checked_mul(4).ok_or_else(||fail("extent"))?;if bytes<4{return Err(fail("stop extent"));}encoder.validate_current_span(h,4,bytes,4)?;spans.push((h,bytes));}
  if let Some(h)=fault {if h.size!=1{return Err(fail("fault extent"));}encoder.validate_current_span(h,4,4,4)?;spans.push((h,4));}
  for i in 0..spans.len(){for j in i+1..spans.len(){let (a,an)=spans[i];let (b,bn)=spans[j];if a.arena==b.arena && (a.off as u64)<b.off as u64+bn as u64 && (b.off as u64)<a.off as u64+an as u64{return Err(fail("overlapping spans"));}}}
  let mut args=self.args(rhs,z,body_base,stop)?;if let Some(h)=fault {args.fault=h;}unsafe{encoder.groups(&args,self.component_count,256)}
 }
 /// Immediate phase-one apply; clear the sticky fault once per solve.
 pub unsafe fn apply(&self,device:&mut impl Device,rhs:Handle,z:Handle,body_base:u32)->Result<(),Fault>{
  if device.instance_id()!=Some(self.instance){return Err(fail("owner device mismatch"));}
  device.run("body_factor.apply",|encoder|{self.encode_clear_fault(encoder)?;unsafe{self.encode_apply(encoder,rhs,z,body_base,None)}})?;
  self.check_fault(device)
 }
 pub fn check_fault(&self,device:&mut impl Device)->Result<(),Fault>{if !self.live || device.instance_id()!=Some(self.instance){return Err(fail("owner device mismatch"));}device.validate_current_span(self.fault_handle(),4,4,4)?;let mut v=[0u32;1];self.buffers[10].read(device,0,&mut v)?;if v[0]!=0{return Err(fail("device factor fault"));}Ok(())}
 pub fn fault_handle(&self)->Handle {self.buffers[10].handle()}
 pub fn free(&mut self,device:&mut impl Device)->Result<(),Fault>{if device.instance_id()!=Some(self.instance){return Err(fail("owner device mismatch"));}self.live=false;let mut result=Ok(());for b in &mut self.buffers{let r=b.free(device);if result.is_ok(){result=r;}}result}
}

/// Opt-in diagnostic: independent temporary buffers only; never commits a solve.
pub fn packet_self_check(device:&mut impl Device,owner:&BodyFactorDevice)->Result<(),Fault>{
 use ppf_cts_compute::ReadbackBuffer;
 let mut rhs=Buffer::<f32>::none();let mut z=Buffer::<f32>::none();let mut packet=ReadbackBuffer::<f32>::default();
 let result=(|| -> Result<(),Fault>{
  owner.check_fault(device)?;
  let n=(owner.body_count as usize).checked_mul(6).ok_or_else(||fail("self check extent"))?;
  rhs.size(device,n,AllocLabel("body_factor.check.rhs"))?;z.size(device,n,AllocLabel("body_factor.check.z"))?;
  packet.size(device,13,AllocLabel("body_factor.check.packet"))?;
  for case in 0..3 {
   let mut seed=[0.0f32;13];if case==2 {seed[9]=1.0;}packet.seed(device,&seed)?;
   rhs.write(device,0,&vec![if case==0 {0.0} else {f32::NAN};n])?;
   z.write(device,0,&vec![7.0;n])?;
   let fault=packet.span(12,1);let stop=packet.span(9,1);
   device.run("body_factor.packet_self_check",|enc|unsafe{owner.encode_apply_with_fault_target(enc,rhs.handle(),z.handle(),0,Some(stop),Some(fault))})?;
   packet.download(device)?;let bits=packet.host()[12].to_bits();
   if (case==1 && bits==0)||(case!=1 && bits!=0){return Err(fail("self check fault route"));}
   owner.check_fault(device)?;
   if case==2 {let mut actual=vec![0.0f32;n];z.read(device,0,&mut actual)?;if packet.host()[9]!=1.0 || actual.iter().any(|&v|v!=7.0){return Err(fail("self check stop"));}}
  }
  let overlap=packet.span(9,1);
  let rejected=device.run("body_factor.packet_overlap_check",|enc|unsafe{owner.encode_apply_with_fault_target(enc,rhs.handle(),z.handle(),0,Some(overlap),Some(overlap))});
  match rejected {Err(Fault::Platform{call:"body factor device",detail}) if detail=="overlapping spans"=>{},Err(e)=>return Err(e),Ok(_)=>return Err(fail("self check overlap accepted"))}
  Ok(())
 })();
 let f1=rhs.free(device);let f2=z.free(device);let f3=packet.free(device);
 result?;f1?;f2?;f3?;Ok(())
}

/// Isolated diagnostic; all replayed work targets temporary buffers.
pub fn batch8_packet_self_check(device:&mut impl Device,owner:&BodyFactorDevice)->Result<(),Fault>{
 use ppf_cts_compute::ReadbackBuffer;
 use super::kernels::{VecFillArgs,JointPcgBatch8FactorFaultArgs,JointPcgBatch8StopArgs,FemPcgReducedDirectionArgs};
 let mut rhs=Buffer::<f32>::none(); let mut z=Buffer::<f32>::none(); let mut p=Buffer::<f32>::none();
 let mut packet=ReadbackBuffer::<f32>::default();
 let result=(|| -> Result<(),Fault>{
  let n=(owner.body_count as usize).checked_mul(6).ok_or_else(||fail("batch8 check extent"))?;
  let count=u32::try_from(n).map_err(|_|fail("batch8 check extent"))?;
  rhs.size(device,n,AllocLabel("factor_batch8.check.rhs"))?;
  z.size(device,n,AllocLabel("factor_batch8.check.z"))?;
  p.size(device,n,AllocLabel("factor_batch8.check.p"))?;
  packet.size(device,13,AllocLabel("factor_batch8.check.packet"))?;
  for target in [1u32,4,8] {
   rhs.write(device,0,&vec![0.0;n])?; z.write(device,0,&vec![7.0;n])?; p.write(device,0,&vec![7.0;n])?;
   let mut seed=[0.0f32;13]; seed[1]=1.0;seed[2]=1.0;seed[3]=1.0;seed[4]=1.0; packet.seed(device,&seed)?;
   let status=packet.span(9,1);let first_iteration=packet.span(10,1);let current_iteration=packet.span(11,1);let fault=packet.span(12,1);let beta=packet.span(0,1);
   let region=device.record("factor_batch8.self_check",|enc|unsafe{
    for sub in 1..=8 {
     if sub==target {enc.elements(&VecFillArgs{array:rhs.handle(),value:f32::NAN,count,seam_arena_count:0},count)?;}
     owner.encode_apply_with_fault_target(enc,rhs.handle(),z.handle(),0,Some(status),Some(fault))?;
     enc.elements(&JointPcgBatch8FactorFaultArgs{fault,status,first_iteration,current_iteration,extent:1,count:1,seam_arena_count:0},1)?;
     enc.elements(&JointPcgBatch8StopArgs{norm:packet.span(1,1),initial:packet.span(2,1),initial_all:1.0,groups:1,cloth_present:1,rz_next:packet.span(3,1),rz_previous:packet.span(4,1),beta,relative:packet.span(5,1),status,tolerance:0.01,first_iteration,current_iteration,max_iterations:100,extent:1,count:1,seam_arena_count:0},1)?;
     enc.elements(&FemPcgReducedDirectionArgs{z:z.handle(),p:p.handle(),beta,status,extent:count,count,seam_arena_count:0},count)?;
    } Ok(())
   })?;
   let run=(|| -> Result<(),Fault>{
    device.replay(&region,1)?;packet.download(device)?;
    if packet.host()[12].to_bits()==0 || packet.host()[9]!=3.0 || packet.host()[10].to_bits()!=target || packet.host()[11].to_bits()!=target {return Err(fail("batch8 check fault latch"));}
    let mut before=vec![0.0;n];p.read(device,0,&mut before)?;let mut before_z=vec![0.0;n];z.read(device,0,&mut before_z)?;
    device.replay(&region,1)?;packet.download(device)?;
    let mut after=vec![0.0;n];p.read(device,0,&mut after)?;let mut after_z=vec![0.0;n];z.read(device,0,&mut after_z)?;
    if packet.host()[9]!=3.0 || packet.host()[10].to_bits()!=target || packet.host()[11].to_bits()!=target || before!=after || before_z!=after_z {return Err(fail("batch8 check sticky replay"));}
    Ok(())
   })(); device.release(region);run?;
  }
  // Stop is already latched; a later NaN must not touch fault, z or direction.
  rhs.write(device,0,&vec![f32::NAN;n])?; z.write(device,0,&vec![7.0;n])?; p.write(device,0,&vec![7.0;n])?;
  let mut seed=[0.0f32;13];seed[2]=1.0;seed[3]=1.0;seed[4]=1.0;seed[11]=f32::from_bits(1);packet.seed(device,&seed)?;
  let status=packet.span(9,1);let first_iteration=packet.span(10,1);let current_iteration=packet.span(11,1);let fault=packet.span(12,1);let beta=packet.span(0,1);
  let region=device.record("factor_batch8.stopped_self_check",|enc|unsafe{
     enc.elements(&JointPcgBatch8StopArgs{norm:packet.span(1,1),initial:packet.span(2,1),initial_all:1.0,groups:1,cloth_present:1,rz_next:packet.span(3,1),rz_previous:packet.span(4,1),beta,relative:packet.span(5,1),status,tolerance:0.01,first_iteration,current_iteration,max_iterations:100,extent:1,count:1,seam_arena_count:0},1)?;
   for _ in 0..8 {
    owner.encode_apply_with_fault_target(enc,rhs.handle(),z.handle(),0,Some(status),Some(fault))?;
    enc.elements(&JointPcgBatch8FactorFaultArgs{fault,status,first_iteration,current_iteration,extent:1,count:1,seam_arena_count:0},1)?;
    enc.elements(&FemPcgReducedDirectionArgs{z:z.handle(),p:p.handle(),beta,status,extent:count,count,seam_arena_count:0},count)?;
   } Ok(())
  })?;
  let run=(|| -> Result<(),Fault>{device.replay(&region,2)?;packet.download(device)?;
   let mut actual_z=vec![0.0;n];let mut actual_p=vec![0.0;n];z.read(device,0,&mut actual_z)?;p.read(device,0,&mut actual_p)?;
   if packet.host()[12].to_bits()!=0 || packet.host()[9]!=1.0 || packet.host()[10].to_bits()!=2 || packet.host()[11].to_bits()!=2 || actual_z.iter().chain(&actual_p).any(|&v|v!=7.0) {return Err(fail("batch8 check prior stop"));} Ok(())})();
  device.release(region);run?;
  owner.check_fault(device)?; Ok(())
 })();
 let f1=rhs.free(device);let f2=z.free(device);let f3=p.free(device);let f4=packet.free(device);
 result?;f1?;f2?;f3?;f4?;Ok(())
}
