//! Conservative per-point curve envelope; never a pose correction.
use crate::joint_root_motion_wire::{RootPose,RootResult,RootMotionError,rotate_world,CODE_NATIVE_CAPABILITY};
fn fail(detail:&str)->RootMotionError {RootMotionError{code:CODE_NATIVE_CAPABILITY,body_uuid:None,key_index:None,field:"curved_ccd".into(),detail:detail.into()}}
pub fn upward_f32(value:f64)->RootResult<f32> {
    if !value.is_finite() || value<0.0 {return Err(fail("nonfinite/negative envelope"));}
    let mut out=value as f32;
    if f64::from(out)<value {out=f32::from_bits(out.to_bits()+1);}
    if !out.is_finite(){return Err(fail("envelope exceeds binary32"));} Ok(out)
}
pub fn angle(a:RootPose,b:RootPose)->f64 {
    let dot=a.rotation_world_wxyz.iter().zip(b.rotation_world_wxyz).map(|(a,b)|a*b).sum::<f64>().abs().clamp(0.0,1.0);
    2.0*dot.acos()
}
pub fn point_envelope(a:RootPose,b:RootPose,rest:[f64;3],start:[f64;3],end:[f64;3])->RootResult<f64> {
    if rest.iter().chain(start.iter()).chain(end.iter()).any(|x|!x.is_finite()) {return Err(fail("invalid native curve vertex"));}
    let y0=rotate_world(a.rotation_world_wxyz,rest);let y1=rotate_world(b.rotation_world_wxyz,rest);
    let radius=rest.iter().map(|x|x*x).sum::<f64>().sqrt();
    let theta=angle(a,b);
    if !theta.is_finite() || theta>std::f64::consts::PI/18.0*(1.0+f64::from(f32::EPSILON)) {return Err(fail("curve slice exceeds ten degrees"));}
    let error0=(0..3).map(|i|(a.center_world_m[i]+y0[i]-start[i]).powi(2)).sum::<f64>().sqrt();
    let error1=(0..3).map(|i|(b.center_world_m[i]+y1[i]-end[i]).powi(2)).sum::<f64>().sqrt();
    // Chord of a rigid circular arc, plus convex interpolation of measured
    // endpoint errors. Roundoff term covers quaternion evaluation, including
    // the sampler's nlerp branch below a 2.9e-6 rad total key angle.
    Ok(2.0*radius*(theta*0.25).sin().powi(2)+error0.max(error1)
        +64.0*f64::EPSILON*(radius+a.center_world_m.iter().chain(b.center_world_m.iter()).map(|x|x.abs()).fold(0.0,f64::max)))
}
pub fn inflated_offset(base:f32,envelope:f64,max_scene_offset:f64)->RootResult<f32> {
    if !base.is_finite() || base<0.0 || !max_scene_offset.is_finite() || max_scene_offset<0.0 {return Err(fail("invalid contact offset"));}
    if envelope==0.0 {return Ok(base);}
    // Only THIS primitive is padded. The scene offset scale bounds rounding
    // of the shared kernel's sum with any partner (one f32 addition). The
    // original barrier/contact tables and nonroot offsets remain bit-exact.
    upward_f32(f64::from(base)+envelope+4.0*f64::from(f32::EPSILON)*(f64::from(base)+max_scene_offset+envelope))
}

#[cfg(test)]
mod reference_curve_tests {
    use super::*;
    use crate::joint_root_motion_wire::compose_root_pose;
    #[test]
    fn per_vertex_curve_margin_uses_author_path_and_d_times_rest() {
        let s=std::f64::consts::FRAC_1_SQRT_2;
        let p0=RootPose{center_world_m:[0.0;3],rotation_world_wxyz:[1.0,0.0,0.0,0.0]};
        let theta=std::f64::consts::PI/20.0;
        let p1=RootPose{center_world_m:[0.0;3],rotation_world_wxyz:[(theta*0.5).cos(),0.0,0.0,(theta*0.5).sin()]};
        let d=RootPose{center_world_m:[1.0,0.0,0.0],rotation_world_wxyz:[s,s,0.0,0.0]};
        let rest=[0.0,1.0,0.0];
        let qrest=rotate_world(d.rotation_world_wxyz,rest);
        let local=std::array::from_fn(|i|d.center_world_m[i]+qrest[i]);
        let native0=compose_root_pose(p0,d).unwrap();let native1=compose_root_pose(p1,d).unwrap();
        let offset0=rotate_world(native0.rotation_world_wxyz,rest);let offset1=rotate_world(native1.rotation_world_wxyz,rest);
        let x0=std::array::from_fn(|i|native0.center_world_m[i]+offset0[i]);
        let x1=std::array::from_fn(|i|native1.center_world_m[i]+offset1[i]);
        let bound=point_envelope(p0,p1,local,x0,x1).unwrap();
        let expected=2.0*2.0_f64.sqrt()*(theta*0.25).sin().powi(2);
        assert!((bound-expected).abs()<1e-12);
    }
}
