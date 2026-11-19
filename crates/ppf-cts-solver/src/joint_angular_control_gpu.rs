//! Directional (per-axis) SO(3) angular control transport.
//!
//! Additive and parallel to `AngularJointGpuV2`: the isotropic 72-byte record is
//! untouched. One 88-byte row is emitted PER JOINT, in the validated joint
//! order, which is what keeps `scene.joints[i]`, this record `i` and the
//! per-joint fault lane slot `i` the same joint. A joint without
//! `angular_control_v1`, and a disabled joint, emit `enabled == 0` and zero
//! coefficients, so the kernel's `active()` gate skips them and the V2
//! isotropic path behaves exactly as before.
//!
//! Units are physical (Nm/rad, Nms/rad); there is no dt^2 scaling.
use crate::joint::{JointWire, OwnedJointScene};

pub const CONTROL_WORLD_ENDPOINT: u32 = u32::MAX;
/// The prior canonical coordinate model. Only this chart is accepted; anything
/// else is refused at ingest rather than silently evaluated in a different
/// coordinate system.
pub const COORDINATE_MODEL: &str = "parent_local_swing_twist_x_v1";
pub const COORDINATE_MODEL_CODE: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AngularControlGpu {
    pub body_a: u32,
    pub body_b: u32,
    /// 1 exactly when this row must be evaluated: the joint is enabled and it
    /// carries a directional record. A record with all-zero coefficients is
    /// still requested — it needs its chart bound — and simply contributes
    /// exactly zero force.
    pub enabled: u32,
    pub coordinate_model: u32,
    pub frame_a_wxyz: [f32; 4],
    pub frame_b_wxyz: [f32; 4],
    pub rest_relative_wxyz: [f32; 4],
    pub stiffness_nm_per_rad: [f32; 3],
    pub damping_nms_per_rad: [f32; 3],
}
const _: () = assert!(std::mem::size_of::<AngularControlGpu>() == 88);
const _: () = assert!(std::mem::offset_of!(AngularControlGpu, frame_a_wxyz) == 16);
const _: () = assert!(std::mem::offset_of!(AngularControlGpu, frame_b_wxyz) == 32);
const _: () = assert!(std::mem::offset_of!(AngularControlGpu, rest_relative_wxyz) == 48);
const _: () = assert!(std::mem::offset_of!(AngularControlGpu, stiffness_nm_per_rad) == 64);
const _: () = assert!(std::mem::offset_of!(AngularControlGpu, damping_nms_per_rad) == 76);

fn f32_checked(x: f64, field: &str) -> Result<f32, String> {
    if !x.is_finite() { return Err(format!("{field}: nonfinite f64")); }
    let y = x as f32;
    if !y.is_finite() { return Err(format!("{field}: outside finite f32 range")); }
    if x != 0.0 && y == 0.0 { return Err(format!("{field}: underflows the f32 domain")); }
    Ok(y)
}
fn quat(q: &[f64; 4], field: &str) -> Result<[f32; 4], String> {
    Ok([f32_checked(q[0], field)?, f32_checked(q[1], field)?,
        f32_checked(q[2], field)?, f32_checked(q[3], field)?])
}
fn vec3(v: [f64; 3], field: &str) -> Result<[f32; 3], String> {
    Ok([f32_checked(v[0], field)?, f32_checked(v[1], field)?, f32_checked(v[2], field)?])
}

/// True when the wire requests the directional record's physical terms.
///
/// This is the *request*, deliberately not "are the coefficients nonzero": a
/// row that carries only soft limits/drives (a later group) still needs its
/// chart and Jacobian validated, and a per-axis coefficient of zero simply
/// contributes exactly zero force. A joint without `angular_control_v1` emits
/// `enabled == 0`, which keeps the V2 isotropic path authoritative and makes the
/// row a true no-op with no chart requirement.
pub fn requested(wire: &JointWire) -> bool {
    wire.enabled && wire.angular_control_v1.is_some()
}

/// One row per validated joint, in joint order, when ANY joint requests a
/// directional record.
///
/// A scene where no joint carries `angular_control_v1` gets an EMPTY vector, not
/// a run of disabled rows: `Joints::install` then allocates no control buffer,
/// no fault lane, and every dispatch of the family is skipped by its explicit
/// count, so a V2 scene never launches an unallocated new family and keeps the
/// exact behaviour it had before directional control existed.
///
/// Once any record is present the mapping is the FULL joint order -- one row per
/// joint, `enabled == 0` for the joints without a record -- so row `i` is always
/// joint `i`, which is what keeps the per-joint fault lane slot meaningful.
pub fn from_validated(scene: &OwnedJointScene) -> Result<Vec<AngularControlGpu>, String> {
    if !scene.joints.iter().any(|j| j.wire.angular_control_v1.is_some()) { return Ok(Vec::new()); }
    scene.joints.iter().map(|j| {
        let wire = &j.wire;
        let (k, c) = match wire.angular_control_v1.as_ref() {
            Some(control) => (vec3(control.stiffness(), "angular_control_v1.stiffness_Nm_per_rad")?,
                              vec3(control.damping(), "angular_control_v1.damping_Nms_per_rad")?),
            None => ([0.0; 3], [0.0; 3]),
        };
        Ok(AngularControlGpu {
            body_a: j.parent_runtime_body_index,
            body_b: j.child_runtime_body_index,
            enabled: u32::from(requested(wire)),
            coordinate_model: COORDINATE_MODEL_CODE,
            frame_a_wxyz: quat(&wire.parent_local_frame.quaternion_wxyz, "frame_a_wxyz")?,
            frame_b_wxyz: quat(&wire.child_local_frame.quaternion_wxyz, "frame_b_wxyz")?,
            rest_relative_wxyz: quat(&wire.rest_relative_quaternion_wxyz, "rest_relative_wxyz")?,
            stiffness_nm_per_rad: k,
            damping_nms_per_rad: c,
        })
    }).collect()
}

// repr(C) fixed-width transport, no pointers or padding.
unsafe impl ppf_cts_compute::Pod for AngularControlGpu {}
