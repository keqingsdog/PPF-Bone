use ppf_cts_compute::{AllocLabel,Buffer,Device};
use super::kernels::PdrdComposeRunningRotationRowArgs;

fn put<T: ppf_cts_compute::Pod>(d:&mut impl Device, v:&[T])->Buffer<T>{
    let mut b=Buffer::default();b.size(d,v.len(),AllocLabel("running_rotation.test")).unwrap();b.write(d,0,v).unwrap();b
}
fn orth(m:&[f32])->f64 {
    let mut s=0f64; for i in 0..3 { for j in 0..3 {let mut dot=0.;for r in 0..3 {dot+=(m[3*i+r] as f64)*(m[3*j+r] as f64);}let e=dot-if i==j{1.}else{0.};s+=e*e;}} s.sqrt()
}
fn mul(a:&[f64;9],b:&[f64;9])->[f64;9]{let mut x=[0.;9];for i in 0..3{for j in 0..3{for k in 0..3{x[3*j+i]+=a[3*k+i]*b[3*j+k];}}}x}
fn exp(w:[f32;3])->[f64;9]{let x=w[0] as f64;let y=w[1] as f64;let z=w[2] as f64;let t=(x*x+y*y+z*z).sqrt();if t==0.{return [1.,0.,0.,0.,1.,0.,0.,0.,1.];}let (s,c)=t.sin_cos();let (a,b)=(s/t,(1.-c)/(t*t));[1.-b*(y*y+z*z),a*z+b*x*y,-a*y+b*x*z,-a*z+b*x*y,1.-b*(x*x+z*z),a*x+b*y*z,a*y+b*x*z,-a*x+b*y*z,1.-b*(x*x+y*y)]}
#[test]
fn actual_host_compose_100k_and_rejections(){unsafe{
 let mut d=super::launch::host_device();let identity=[1f32,0.,0.,0.,1.,0.,0.,0.,1.];
 let mut running=put(&mut d,&identity);let mut step=put(&mut d,&[0f32;3]);let mut fault=put(&mut d,&[0u32]);let mut observed=put(&mut d,&[0f32;41]);
 let mut reference=[1f64,0.,0.,0.,1.,0.,0.,0.,1.];let mut seed=0x13579bdu32;let mut maxorth=0f64;
 for k in 0..100_000 {let w=if k<40_000 {[0.001f32,0.,0.]} else if k<60_000 {[-0.001f32,0.,0.]} else {let mut a=[0f32;3];for v in &mut a {seed=seed.wrapping_mul(1664525).wrapping_add(1013904223);*v=((seed>>8) as f32/16777216.-0.5)*0.002;}a};
  step.write(&mut d,0,&w).unwrap();let args=PdrdComposeRunningRotationRowArgs{running:running.handle(),dtheta:step.handle(),fault:fault.handle(),observed:observed.handle(),scale:1.,count:1,seam_arena_count:0};d.launch("pdrd.compose_running_rotation",&args,1).unwrap();
  let mut f=[0u32];fault.read(&mut d,0,&mut f).unwrap();assert_eq!(f[0],0,"step {k}");let mut r=[0f32;9];running.read(&mut d,0,&mut r).unwrap();maxorth=maxorth.max(orth(&r));assert!(maxorth<=5e-6,"step {k} orth {maxorth}");reference=mul(&exp(w),&reference);
 }
 let mut final_r=[0f32;9];running.read(&mut d,0,&mut final_r).unwrap();let mut e2=0.;for i in 0..9{e2+=(final_r[i] as f64-reference[i]).powi(2);}let angle=(e2.sqrt()/(2.*2f64.sqrt())).min(1.).asin()*2.;println!("100k max_orth={maxorth:.9e} final_angle_error={angle:.9e}");assert!(angle<=1e-3,"reference angle {angle}, orth {maxorth}");
 for mut bad in [[f32::MAX;9],[f32::NAN;9],[-1.,0.,0.,0.,1.,0.,0.,0.,1.],[1.0002,0.,0.,0.,1.,0.,0.,0.,1.]] {running.write(&mut d,0,&bad).unwrap();step.write(&mut d,0,&[0.,0.,0.]).unwrap();let bits=bad.map(f32::to_bits);let args=PdrdComposeRunningRotationRowArgs{running:running.handle(),dtheta:step.handle(),fault:fault.handle(),observed:observed.handle(),scale:1.,count:1,seam_arena_count:0};d.launch("pdrd.compose_running_rotation",&args,1).unwrap();let mut f=[0u32];fault.read(&mut d,0,&mut f).unwrap();assert_eq!(f[0],1);running.read(&mut d,0,&mut bad).unwrap();assert_eq!(bits,bad.map(f32::to_bits));}
}}

#[test]
fn actual_host_compose_state23_and_bad_increment(){unsafe{
 let mut d=super::launch::host_device();
 let state23=[0.9310082f32,-0.30135453,-0.20592062,-0.27407545,-0.20462617,-0.93968576,0.24104464,0.93129563,-0.2731038];
 let mut running=put(&mut d,&state23);let mut step=put(&mut d,&[0f32;3]);let mut fault=put(&mut d,&[0u32]);let observed=put(&mut d,&[0f32;41]);
 let dispatch=|d:&mut ppf_cts_compute::HostDevice,running:&Buffer<f32>,step:&Buffer<f32>,fault:&Buffer<u32>,scale:f32|{
  let args=PdrdComposeRunningRotationRowArgs{running:running.handle(),dtheta:step.handle(),fault:fault.handle(),observed:observed.handle(),scale,count:1,seam_arena_count:0};d.launch("pdrd.compose_running_rotation",&args,1).unwrap();
 };
 dispatch(&mut d,&running,&step,&fault,1.);let mut f=[0u32];fault.read(&mut d,0,&mut f).unwrap();assert_eq!(f[0],0);let mut valid=[0f32;9];running.read(&mut d,0,&mut valid).unwrap();assert!(orth(&valid)<=5e-6);
 for (w,scale) in [([f32::NAN,0.,0.],1.),([f32::INFINITY,0.,0.],1.),([f32::MAX,0.,0.],2.),([0.,0.,0.],f32::NAN)] {
  running.write(&mut d,0,&state23).unwrap();step.write(&mut d,0,&w).unwrap();dispatch(&mut d,&running,&step,&fault,scale);
  fault.read(&mut d,0,&mut f).unwrap();assert_eq!(f[0],2);let mut after=[0f32;9];running.read(&mut d,0,&mut after).unwrap();assert_eq!(after.map(f32::to_bits),state23.map(f32::to_bits));
 }
}}


#[test]
fn actual_host_reduction_reports_fatal_bits(){unsafe{
 let mut d=super::launch::host_device();
 let map=super::rigid_map::RigidMap{n_bodies:1,..Default::default()};let mut staged=super::rigid_map::Staged::default();
 let original=[f32::from_bits(0x7fc01234),0.,0.,0.,1.,0.,0.,0.,1.];
 let mut running=put(&mut d,&original);let step=put(&mut d,&[0f32;3]);
 let mut reduction=super::pdrd::Reduction{map:&map,staged:&mut staged};
 let err=reduction.compose_running_rotation(&mut d,running.handle(),step.handle(),1.).unwrap_err();
 let message=format!("{err:?}");assert!(message.contains("body=0")&&message.contains("stage=1")&&message.contains("7fc01234")&&message.contains("unavailable"),"{message}");
 let mut after=[0f32;9];running.read(&mut d,0,&mut after).unwrap();assert_eq!(after.map(f32::to_bits),original.map(f32::to_bits));
}}

#[test]
fn actual_host_controlled_root_full_partial_free(){unsafe{
 use super::kernels::RootControlledRetractArgs;
 use crate::{data::PdrdRigidState,joint_root_motion_gpu::RootTargetGpu};
 let mut d=super::launch::host_device();let identity=[1f32,0.,0.,0.,1.,0.,0.,0.,1.];
 let target=[0f32,1.,0.,-1.,0.,0.,0.,0.,1.];let pose=PdrdRigidState{rotation:identity,..Default::default()};
 let roots=put(&mut d,&[RootTargetGpu{body:0,enabled:1,rotation:target,..Default::default()}]);
 let state=put(&mut d,&[pose,pose]);let w=[0.25f32,0.,0.,0.,0.,0.];let step=put(&mut d,&w);let mut both=identity.to_vec();both.extend(identity);
 let mut running=put(&mut d,&both);let mut fault=put(&mut d,&[0u32;2]);let observation=put(&mut d,&[0f32;82]);
 for fraction in [0f32,0.5,1.] {running.write(&mut d,0,&both).unwrap();
  let compose=PdrdComposeRunningRotationRowArgs{running:running.handle(),dtheta:step.handle(),fault:fault.handle(),observed:observation.handle(),scale:-fraction,count:2,seam_arena_count:0};d.launch("pdrd.compose_running_rotation",&compose,2).unwrap();
  let args=RootControlledRetractArgs{targets:roots.handle(),state:state.handle(),dtheta:step.handle(),running:running.handle(),bodies:2,fraction,count:1,seam_arena_count:0};d.launch("roots.controlled_retract",&args,1).unwrap();
  let mut got=[0f32;18];running.read(&mut d,0,&mut got).unwrap();if fraction==1. {assert_eq!(got[..9].iter().map(|v|v.to_bits()).collect::<Vec<_>>(),target.map(f32::to_bits));} else {let expected=exp([-fraction*w[0],0.,0.]);for i in 0..9 {assert!((got[i] as f64-expected[i]).abs()<2e-6,"fraction={fraction} i={i}");}}
  for i in 0..9 {assert!((got[9+i]-identity[i]).abs()<2e-6,"free body {i}");}
 }
}}


