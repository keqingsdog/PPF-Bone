//! Actual CPU Device dispatch using captured 03 f32 endpoints; no scene mutation.
use ppf_cts_compute::{AllocLabel,Buffer,Device,Pod};
use super::kernels::JointFemPrepareArgs;
use crate::soft_joint_gpu::{FemJointGpu,FemFrameGpu,FEM_TET,FEM_WORLD};

fn put<T:Pod>(d:&mut impl Device,v:&[T])->Buffer<T>{
    let mut b=Buffer::default();b.size(d,v.len(),AllocLabel("private.fem_polar_dispatch")).unwrap();b.write(d,0,v).unwrap();b
}

#[test]
fn actual_cpu_fem_prepare_accepts_unique_inversion_rejects_rank_zero(){unsafe{
    const D:[u32;9]=[1063426473,3250987223,3194281851,1104096495,3250774850,1103759681,1103557674,3214533382,3251428429];
    const POS:[u32;12]=[1038262790,1068293810,3193753708,1035346165,1068797009,3192552172,1033690301,1068419915,3194568000,1034892874,1068507054,3193864950];
    const NEG:[u32;12]=[1038332988,1068285496,3193817454,1035423330,1068789632,3192621234,1033771489,1068411806,3194637710,1034957302,1068499072,3193920620];
    const REF:[u32;12]=[1038185868,1068302809,3193682896,1035257740,1068804855,3192471305,1033599515,1068428565,3194487672,1034824517,1068515716,3193804597];
    let mut device=super::launch::host_device();
    let mut record:FemJointGpu=std::mem::zeroed();record.enabled=1;record.a.kind=FEM_WORLD;
    record.b.kind=FEM_TET;record.b.vertices=[0,1,2,3];record.b.dm_inv=D.map(f32::from_bits);
    record.b.anchor_bary=[1.,0.,0.,0.];record.b.center_bary=[1.,0.,0.,0.];
    record.b.rest_rotation=[1.,0.,0.,0.,1.,0.,0.,0.,1.];
    let records=put(&mut device,&[record]);let mut current=put(&mut device,&POS.map(f32::from_bits));
    let reference=put(&mut device,&REF.map(f32::from_bits));let dummy=put(&mut device,&[0f32;36]);
    let frames: [FemFrameGpu;4]=std::mem::zeroed();let mut cache=put(&mut device,&frames);let mut fault=put(&mut device,&[0u32]);
    let args=JointFemPrepareArgs{records:records.handle(),record_count:1,current_xyz:current.handle(),reference_xyz:reference.handle(),vertex_count:4,rigid_current:dummy.handle(),rigid_reference:dummy.handle(),bodies:0,cache:cache.handle(),fault:fault.handle(),count:1,seam_arena_count:0};
    device.launch("joints.fem.prepare",&args,1).unwrap();let mut code=[999u32];fault.read(&mut device,0,&mut code).unwrap();
    let mut got:[FemFrameGpu;4]=std::mem::zeroed();cache.read(&mut device,0,&mut got).unwrap();
    assert_eq!(code,[0]);assert_eq!(got.map(|f|f.status),[0,0,0,0]);assert!(got[1].rotation.iter().all(|v|v.is_finite()));
    assert!(got[1].rotation_jacobian.iter().chain(got[3].rotation_jacobian.iter()).all(|v|v.is_finite()));
    current.write(&mut device,0,&NEG.map(f32::from_bits)).unwrap();device.launch("joints.fem.prepare",&args,1).unwrap();
    fault.read(&mut device,0,&mut code).unwrap();cache.read(&mut device,0,&mut got).unwrap();
    assert_eq!(code,[0]);assert_eq!(got.map(|f|f.status),[0,0,0,0]);
    assert!(got[1].rotation.iter().chain(got[1].rotation_jacobian.iter()).all(|v|v.is_finite()));
    // Coincident nodes make a rank-zero endpoint and do not define an SO(3)
    // material rotation, so the native guard must still reject it.
    let mut collapsed=[0f32;12];let first=POS.map(f32::from_bits);
    for i in 0..4 {collapsed[3*i..3*i+3].copy_from_slice(&first[..3]);}
    current.write(&mut device,0,&collapsed).unwrap();device.launch("joints.fem.prepare",&args,1).unwrap();
    fault.read(&mut device,0,&mut code).unwrap();cache.read(&mut device,0,&mut got).unwrap();
    assert_eq!(code,[259]);assert_eq!(got.map(|f|f.status),[0,259,0,0]);
}}
