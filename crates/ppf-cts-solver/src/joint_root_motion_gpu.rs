//! Neutral transport only. The live crate adds the Pod implementation beside
//! its existing joint GPU records after copying this module into src/.
use crate::joint_root_motion_wire::{rotate_world,RootPose};
#[repr(C)]
#[derive(Clone,Copy,Debug,Default,PartialEq)]
pub struct RootTargetGpu {
    pub body:u32,pub enabled:u32,pub center:[f32;3],pub rotation:[f32;9],
    pub list_start:u32,pub list_count:u32,
}
const _:()=assert!(std::mem::size_of::<RootTargetGpu>()==64);
impl RootTargetGpu {
    pub fn sampled(body:u32,pose:RootPose,list_start:u32,list_count:u32)->Result<Self,String> {
        if list_count==0 || list_start.checked_add(list_count).is_none()
            || pose.center_world_m.iter().chain(pose.rotation_world_wxyz.iter()).any(|x|!x.is_finite())
            || (pose.rotation_world_wxyz.iter().map(|x|x*x).sum::<f64>()-1.0).abs()>1e-6 {
            return Err("ROOT_MOTION_BODY_BINDING: invalid sampled target".into());
        }
        let center=pose.center_world_m.map(|x|x as f32);
        let mut rotation=[0.0f32;9];
        for c in 0..3 {
            let axis=std::array::from_fn(|a|if a==c {1.0}else{0.0});
            let col=rotate_world(pose.rotation_world_wxyz,axis);
            for r in 0..3 {rotation[r+3*c]=col[r] as f32;}
        }
        if center.iter().chain(rotation.iter()).any(|x|!x.is_finite()) {
            return Err("ROOT_MOTION_INVALID_KEYS: target exceeds float32".into());
        }
        Ok(Self{body,enabled:1,center,rotation,list_start,list_count})
    }
}
