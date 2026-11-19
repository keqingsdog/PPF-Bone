//! Self-contained hard geometry transport. NOT the old CUDA ABI.
use crate::{joint_d6_hard_wire::ValidatedD6, joint_angular_hard_wire::{ValidatedHardHingeV1, ValidatedTwistV1}};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HardJointGpu {
    pub joint_index: u32, pub body_a: u32, pub body_b: u32, pub enabled: u32,
    pub anchor_a: [f32;3], pub anchor_b: [f32;3],
    pub frame_a_wxyz: [f32;4], pub frame_b_wxyz: [f32;4], pub rest_relative_wxyz: [f32;4],
    pub rest_translation_m: [f32;3],
    pub modes: [u32;6], pub lower: [f32;6], pub upper: [f32;6],
}
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HardGeometryGpu {
    pub joint_index: u32, pub body_a: u32, pub body_b: u32, pub fault: u32,
    pub coordinate: [f32;6], pub jacobian: [f32;72],
}
impl Default for HardGeometryGpu {
    fn default()->Self {Self {joint_index:0,body_a:0,body_b:0,fault:0,coordinate:[0.;6],jacobian:[0.;72]}}
}
// Large arrays' Default is provided manually on older compiler toolchains.
const _: () = {
    assert!(std::mem::size_of::<HardJointGpu>()==172);
    assert!(std::mem::offset_of!(HardJointGpu,anchor_a)==16);
    assert!(std::mem::offset_of!(HardJointGpu,frame_a_wxyz)==40);
    assert!(std::mem::offset_of!(HardJointGpu,rest_translation_m)==88);
    assert!(std::mem::offset_of!(HardJointGpu,modes)==100);
    assert!(std::mem::offset_of!(HardJointGpu,lower)==124);
    assert!(std::mem::offset_of!(HardJointGpu,upper)==148);
    assert!(std::mem::size_of::<HardGeometryGpu>()==328);
    assert!(std::mem::offset_of!(HardGeometryGpu,coordinate)==16);
    assert!(std::mem::offset_of!(HardGeometryGpu,jacobian)==40);
};

pub fn hinge_axes(h: ValidatedHardHingeV1) -> ValidatedD6 {
    let mut d=ValidatedD6 { modes:[0,0,0,0,2,2], lower:[0.;6], upper:[0.;6], rest:[0.;3] };
    if let ValidatedTwistV1::Bounded { lower_rad,upper_rad }=h.twist_x {
        d.modes[3]=1; d.lower[3]=lower_rad; d.upper[3]=upper_rad;
    }
    d
}

pub fn pack(joint_index:u32, body_a:u32, body_b:u32, enabled:bool,
    anchor_a:[f64;3], anchor_b:[f64;3], frame_a:[f64;4], frame_b:[f64;4], rest:[f64;4],
    axes:ValidatedD6) -> Result<HardJointGpu,&'static str> {
    if body_a==u32::MAX || body_b==u32::MAX || body_a==body_b { return Err("G2_HARD_UNSUPPORTED_DOMAIN"); }
    // Geometry components, including the fitted frame's tiny roundoff lanes,
    // have an ABSOLUTE transport contract. Underflow is not a lost physical
    // stiffness/axis request: this function does not narrow those coefficients.
    // Their wire validators retain their separate nonzero-underflow refusal.
    fn geometry<const N:usize>(x:[f64;N])->Result<([f32;N],f64),&'static str> {
        fn up(x:f64)->f64 {if x==0.0 {f64::from_bits(1)}else{f64::from_bits(x.to_bits()+1)}}
        let mut y=[0.;N]; for i in 0..N { y[i]=x[i] as f32;
            if !x[i].is_finite()||!y[i].is_finite() { return Err("G2_HARD_INVALID_TRANSPORT"); }
        }
        let mut error=0.;
        for i in 0..N {
            let cast=y[i] as f64;
            // Include subtraction/addition rounding and possible FP64 FTZ.
            let subtraction=up(up(2.0_f64.powi(-53)*up(x[i].abs()+cast.abs()))+f64::MIN_POSITIVE);
            let float_daz=if cast!=0.0&&cast.abs()<(f32::MIN_POSITIVE as f64){cast.abs()}else{0.0};
            error=up(error+up(up((x[i]-cast).abs()+subtraction)+float_daz));
        }
        if !error.is_finite()||error>1e-7 {return Err("G2_HARD_GEOMETRY_TRANSPORT_BUDGET");}
        Ok((y,error))
    }
    fn quat(x:[f64;4])->Result<([f32;4],f64),&'static str> {
        let y=geometry(x)?;
        if (x.iter().map(|v|v*v).sum::<f64>().sqrt()-1.).abs()>1e-4 { return Err("G2_HARD_INVALID_FRAME"); }
        Ok(y)
    }
    let (anchor_a,ea)=geometry(anchor_a)?;let (anchor_b,eb)=geometry(anchor_b)?;
    let (frame_a,qa)=quat(frame_a)?;let (frame_b,qb)=quat(frame_b)?;let (rest,qr)=quat(rest)?;
    eprintln!("HARD_GEOMETRY_TRANSPORT joint_index={joint_index} anchor_a_l1_error_bound_m={ea:.12e} anchor_b_l1_error_bound_m={eb:.12e} frame_a_l1_error_bound={qa:.12e} frame_b_l1_error_bound={qb:.12e} rest_quaternion_l1_error_bound={qr:.12e} per_field_l1_budget=1.000000000000e-7 scope=finite_f64_to_f32_geometry_only_no_axis_or_force_underflow_exception");
    Ok(HardJointGpu { joint_index,body_a,body_b,enabled:u32::from(enabled),
        anchor_a,anchor_b,frame_a_wxyz:frame_a,
        frame_b_wxyz:frame_b,rest_relative_wxyz:rest,
        rest_translation_m:axes.rest,modes:axes.modes,lower:axes.lower,upper:axes.upper })
}
