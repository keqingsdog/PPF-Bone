//! Version-2 angular spring/damper records. The final reserved float carries an
//! optional isotropic angular deviation limit without changing the 72-byte ABI.
use crate::joint::OwnedJointScene;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AngularJointGpuV2 {
    pub body_a: u32,
    pub body_b: u32,
    pub enabled: u32,
    pub max_angle_rad: f32,
    pub frame_a_wxyz: [f32; 4],
    pub frame_b_wxyz: [f32; 4],
    pub rest_relative_wxyz: [f32; 4],
    pub stiffness_nm_per_rad: f32,
    pub damping_nms_per_rad: f32,
}
const _: () = assert!(std::mem::size_of::<AngularJointGpuV2>() == 72);
const _: () = assert!(std::mem::offset_of!(AngularJointGpuV2, frame_a_wxyz) == 16);
const _: () = assert!(std::mem::offset_of!(AngularJointGpuV2, rest_relative_wxyz) == 48);
const _: () = assert!(std::mem::offset_of!(AngularJointGpuV2, stiffness_nm_per_rad) == 64);

fn f32_checked(x: f64, field: &str) -> Result<f32, String> {
    if !x.is_finite() { return Err(format!("{field}: nonfinite f64")); }
    let y = x as f32;
    if !y.is_finite() { return Err(format!("{field}: outside finite f32 range")); }
    Ok(y)
}
fn quat(q: &[f64; 4], field: &str) -> Result<[f32; 4], String> {
    Ok([f32_checked(q[0], field)?, f32_checked(q[1], field)?,
        f32_checked(q[2], field)?, f32_checked(q[3], field)?])
}
pub fn from_validated(scene: &OwnedJointScene) -> Result<Vec<AngularJointGpuV2>, String> {
    scene.joints.iter().map(|joint| {
        let wire = &joint.wire;
        Ok(AngularJointGpuV2 {
            body_a: joint.parent_runtime_body_index,
            body_b: joint.child_runtime_body_index,
            enabled: u32::from(wire.enabled),
            max_angle_rad: f32_checked(wire.max_angle_rad, "max_angle_rad")?,
            frame_a_wxyz: quat(&wire.parent_local_frame.quaternion_wxyz, "frame_a_wxyz")?,
            frame_b_wxyz: quat(&wire.child_local_frame.quaternion_wxyz, "frame_b_wxyz")?,
            rest_relative_wxyz: quat(&wire.rest_relative_quaternion_wxyz, "rest_relative_wxyz")?,
            stiffness_nm_per_rad: f32_checked(wire.angle_stiffness_nm_per_rad, "angle_stiffness_Nm_per_rad")?,
            damping_nms_per_rad: f32_checked(wire.angle_damping_nms_per_rad, "angle_damping_Nms_per_rad")?,
        })
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joint;
    use std::path::Path;

    fn fixture() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../local/share/ppf-cts/git-ppf-unified/r1-binding-20260923-a/r1-binding-20260923-a")
    }
    fn scene() -> joint::OwnedJointScene {
        let fixture = fixture();
        let bytes = std::fs::read(fixture.join("bin/pdrd_body.bin")).unwrap();
        let rows = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
        joint::ingest_validated(&fixture, &rows).unwrap().unwrap()
    }
    #[test]
    fn converts_legacy_payload_to_inert_v2_angular_record() {
        let records = from_validated(&scene()).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].frame_a_wxyz, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(records[0].frame_b_wxyz, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(records[0].rest_relative_wxyz, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(records[0].stiffness_nm_per_rad, 0.0);
        assert_eq!(records[0].damping_nms_per_rad, 0.0);
        assert_eq!(records[0].max_angle_rad, 0.0);
    }
    #[test]
    fn rejects_coefficients_that_overflow_native_float() {
        let mut scene = scene();
        scene.joints[0].wire.angle_damping_nms_per_rad = f64::MAX;
        assert!(from_validated(&scene).unwrap_err().contains("angle_damping_Nms_per_rad: outside finite f32 range"));
    }
}

// repr(C) fixed-width transport, with no pointers or padding.
unsafe impl ppf_cts_compute::Pod for AngularJointGpuV2 {}
