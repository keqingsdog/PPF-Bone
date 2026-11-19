//! Same-Newton original FEM/body/root orthogonal constraint space.
use ppf_cts_compute::{AllocLabel,Buffer,ReadbackBuffer,Device,Handle,Fault};
use super::{kernels::*,reduced_model::Revision};
fn refusal(s:impl Into<String>)->Fault{Fault::Platform{call:"constraint consistent Schwarz",detail:s.into()}}
#[derive(Clone,Copy)]
pub struct Inputs{pub joint_mode:Handle,pub joint_axis:Handle,pub translation_lock:Handle,pub translation_axis:Handle,pub translation_mode:Handle,pub rotation_lock:Handle,pub rotation_axis:Handle,pub rotation_mode:Handle}
#[derive(Clone,Copy)]
pub struct View{pub q:Handle,pub cloth:u32,pub bodies:u32,pub revision:Revision}
#[derive(Default)]
pub struct Frozen{q:Buffer<f32>,roots:Buffer<u32>,fault:ReadbackBuffer<u32>,dummy_u32:Buffer<u32>,dummy_f32:Buffer<f32>,ready:Option<Revision>}
impl Frozen{
 pub unsafe fn rebuild<D:Device>(&mut self,d:&mut D,mut input:Inputs,row_to_vertex:Handle,dof_mask:Handle,cloth:u32,bodies:u32,roots:&[u32],revision:Revision)->Result<View,Fault>{
 self.ready=None;let units=cloth.checked_add(bodies).ok_or_else(||refusal("unit count overflow"))?;
 let cells=units.checked_mul(36).ok_or_else(||refusal("Q capacity overflow"))?;
 for h in [input.joint_mode,input.joint_axis,input.translation_lock,input.translation_axis,input.translation_mode,input.rotation_lock,input.rotation_axis,input.rotation_mode]{if h.size<bodies{return Err(refusal("original body projector input extent"));}}
 for h in [input.joint_axis,input.translation_axis,input.rotation_axis]{if u64::from(h.size)<3*u64::from(bodies){return Err(refusal("original body axis scalar extent"));}}
 if bodies==0{self.dummy_u32.size(d,1,AllocLabel("rqc.inactive_u32"))?;self.dummy_u32.write(d,0,&[0])?;
 self.dummy_f32.size(d,3,AllocLabel("rqc.inactive_f32"))?;self.dummy_f32.write(d,0,&[0.,0.,0.])?;
 let u=self.dummy_u32.handle();let f=self.dummy_f32.handle();input=Inputs{joint_mode:u,joint_axis:f,translation_lock:u,translation_axis:f,translation_mode:u,rotation_lock:u,rotation_axis:f,rotation_mode:u};}
 self.q.size(d,(cells as usize).max(36),AllocLabel("rqc.q"))?;
 self.roots.size(d,(bodies as usize).max(1),AllocLabel("rqc.roots"))?;
 let mut mask=vec![0u32;(bodies as usize).max(1)];for &b in roots{if b>=bodies{return Err(refusal("enabled root body outside map"));}mask[b as usize]=1;}
 self.roots.write(d,0,&mask)?;self.fault.size(d,(units as usize).max(1),AllocLabel("rqc.fault"))?;
 if units>0{let a=RqcFreezeArgs{row_to_vertex,dof_mask,joint_mode:input.joint_mode,joint_axis:input.joint_axis,translation_lock:input.translation_lock,translation_axis:input.translation_axis,translation_mode:input.translation_mode,rotation_lock:input.rotation_lock,rotation_axis:input.rotation_axis,rotation_mode:input.rotation_mode,root_mask:self.roots.handle(),q:self.q.handle(),fault:self.fault.handle(),cloth,bodies,extent:units,count:units,seam_arena_count:0};d.launch("rqc.freeze",&a,units)?;self.fault.download(d)?;if self.fault.host()[..units as usize].iter().any(|&x|x!=0){return Err(refusal("actual original constraint basis refused"));}}
 self.ready=Some(revision);Ok(View{q:self.q.handle(),cloth,bodies,revision})
 }
}

/// Exact Newton selector: no diagnostics or readbacks during ordinary timing.
pub fn diagnostic_selected(revision:Revision)->Result<bool,Fault>{
 match std::env::var("PPF_SCHWARZ_CONSTRAINT_CHECK_NEWTON"){
 Err(std::env::VarError::NotPresent)=>Ok(false),
 Ok(v)=>{let n=v.parse::<u64>().map_err(|_|refusal("constraint diagnostic Newton must be u64"))?;Ok(n==revision.operator)},
 Err(_)=>Err(refusal("constraint diagnostic selector is not UTF8"))}
}
pub unsafe fn read_f32<D:Device>(d:&mut D,h:Handle,n:usize)->Result<Vec<f32>,Fault>{let mut v=vec![0f32;n];let bytes=n.checked_mul(4).ok_or_else(||refusal("diagnostic bytes overflow"))?;if n>h.size as usize{return Err(refusal("diagnostic f32 extent"));}d.read(h,0,std::slice::from_raw_parts_mut(v.as_mut_ptr().cast(),bytes))?;Ok(v)}
pub unsafe fn read_u32<D:Device>(d:&mut D,h:Handle,n:usize)->Result<Vec<u32>,Fault>{let mut v=vec![0u32;n];if n>h.size as usize{return Err(refusal("diagnostic u32 extent"));}d.read(h,0,std::slice::from_raw_parts_mut(v.as_mut_ptr().cast(),n*4))?;Ok(v)}
pub fn q_entry(q:&[f32],cloth:u32,ra:u32,ia:usize,rb:u32,ib:usize)->f64{
 let unit=|r:u32|if r<cloth{r}else{cloth+(r-cloth)/2};
 let axis=|r:u32,i:usize|if r<cloth{i}else{3*((r-cloth)%2)as usize+i};
 if unit(ra)!=unit(rb){0.}else{f64::from(q[36*unit(ra)as usize+6*axis(ra,ia)+axis(rb,ib)])}
}
pub unsafe fn check_q<D:Device>(d:&mut D,v:View,roots:&[u32])->Result<(),Fault>{
 let units=(v.cloth+v.bodies)as usize;if units>100000{return Err(refusal("constraint diagnostic unit bound"));}
 let q=read_f32(d,v.q,36*units)?;let mut idem=0f64;let mut sym=0f64;
 for u in 0..units{let n=if u<v.cloth as usize{3}else{6};for i in 0..n{for j in 0..n{
 let a=f64::from(q[36*u+6*i+j]);let z=(0..n).map(|k|f64::from(q[36*u+6*i+k])*f64::from(q[36*u+6*k+j])).sum::<f64>();
 idem=idem.max((z-a).abs());sym=sym.max((a-f64::from(q[36*u+6*j+i])).abs());}}}
 let roots_zero=roots.iter().all(|&b|q[36*(v.cloth+b)as usize..36*(v.cloth+b+1)as usize].iter().all(|&x|x==0.));
 let bound=512.*f64::from(f32::EPSILON);if !idem.is_finite()||idem>bound||sym>bound||!roots_zero{return Err(refusal(format!("frozen Q invariant failed idempotence={idem} symmetry={sym} roots_zero={roots_zero}")));}
 eprintln!("SCHWARZ_CONSTRAINT_Q_CHECK {}",serde_json::json!({"revision":v.revision.operator,"idempotence_max":idem,"symmetry_max":sym,"enabled_root_rows_exact_zero":roots_zero,"bound":bound}));Ok(())
}
