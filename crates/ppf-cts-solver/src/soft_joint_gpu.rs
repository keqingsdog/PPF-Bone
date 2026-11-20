//! Exact transport PODs for neutral material-frame FEM joints.
use crate::joint_angular_gpu::AngularJointGpuV2;
use crate::joint_angular_control_gpu::AngularControlGpu;
pub const FEM_WORLD:u32=0;
pub const FEM_PDRD:u32=1;
pub const FEM_TET:u32=2;
#[repr(C)]
#[derive(Clone,Copy,Debug,PartialEq)]
pub struct FemEndpointGpu {
    pub kind:u32,pub id:u32,pub vertices:[u32;4],
    pub dm_inv:[f32;9],pub anchor_bary:[f32;4],pub center_bary:[f32;4],
    pub rest_rotation:[f32;9],pub local_anchor:[f32;3],
}
#[repr(C)]
#[derive(Clone,Copy,Debug,Default,PartialEq)]
pub struct PmxJointGpu {
    pub enabled: u32,
    pub modes: [u32;6],
    pub lower: [f32;6],
    pub upper: [f32;6],
    pub springs: [f32;6],
    pub limit_weight: [f32;2],
    pub frame_a: [f32;4],
    pub frame_b: [f32;4],
}
#[repr(C)]
#[derive(Clone,Copy,Debug,PartialEq)]
pub struct FemJointGpu {
    pub a:FemEndpointGpu,pub b:FemEndpointGpu,pub enabled:u32,pub ball_stiffness:f32,
    pub v2:AngularJointGpuV2,pub control:AngularControlGpu,
    pub pmx: PmxJointGpu,
}
#[repr(C)]
#[derive(Clone,Copy,Debug,PartialEq)]
pub struct FemFrameGpu {
    pub rotation:[f32;9],pub anchor:[f32;3],pub center:[f32;3],
    pub rotation_jacobian:[f32;36],pub status:u32,
}
const _:()=assert!(std::mem::size_of::<FemEndpointGpu>()==140);
const _:()=assert!(std::mem::offset_of!(FemEndpointGpu,dm_inv)==24);
const _:()=assert!(std::mem::offset_of!(FemEndpointGpu,anchor_bary)==60);
const _:()=assert!(std::mem::offset_of!(FemEndpointGpu,rest_rotation)==92);
const _:()=assert!(std::mem::size_of::<PmxJointGpu>()==140);
const _:()=assert!(std::mem::size_of::<FemJointGpu>()==588);
const _:()=assert!(std::mem::offset_of!(FemJointGpu,pmx)==448);
const _:()=assert!(std::mem::offset_of!(FemJointGpu,v2)==288);
const _:()=assert!(std::mem::offset_of!(FemJointGpu,control)==360);
const _:()=assert!(std::mem::size_of::<FemFrameGpu>()==208);
const _:()=assert!(std::mem::offset_of!(FemFrameGpu,rotation_jacobian)==60);
const _:()=assert!(std::mem::offset_of!(FemFrameGpu,status)==204);
