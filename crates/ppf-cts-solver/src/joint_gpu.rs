//! ABI-checked ball-anchor transport consumed by the neutral PPF driver.
use crate::joint::OwnedJointScene;

pub const WORLD_ENDPOINT: u32 = u32::MAX;
pub const BALL_ANCHOR_FLAG_ACTIVE: u32 = 1;
pub const BALL_ANCHOR_FLAG_NO_COLLIDE: u32 = 2;
pub const BALL_ANCHOR_FLAG_MASK: u32 = BALL_ANCHOR_FLAG_ACTIVE | BALL_ANCHOR_FLAG_NO_COLLIDE;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BallAnchorGpu {
    pub body_a: u32, pub body_b: u32, pub record_index: u32,
    /// ABI-stable flag word: ACTIVE and, only for an active joint,
    /// NO_COLLIDE when `collide_connected` is false.
    pub enabled: u32,
    pub anchor_a: [f32; 3], pub anchor_b: [f32; 3], pub stiffness: f32,
}
const _: () = assert!(std::mem::size_of::<BallAnchorGpu>() == 44);
const _: () = assert!(std::mem::offset_of!(BallAnchorGpu, anchor_a) == 16);
const _: () = assert!(std::mem::offset_of!(BallAnchorGpu, anchor_b) == 28);
const _: () = assert!(std::mem::offset_of!(BallAnchorGpu, stiffness) == 40);
fn f32_checked(x: f64, field: &str) -> Result<f32, String> {
    if !x.is_finite() { return Err(format!("{field}: nonfinite f64")); }
    let y = x as f32;
    if !y.is_finite() { return Err(format!("{field}: outside finite f32 range")); }
    Ok(y)
}
pub fn from_validated(scene: &OwnedJointScene) -> Result<Vec<BallAnchorGpu>, String> {
    scene.joints.iter().enumerate().map(|(record_index, j)| {
        let a = &j.wire.parent_local_frame.translation_m; let b = &j.wire.child_local_frame.translation_m;
        Ok(BallAnchorGpu { body_a: j.parent_runtime_body_index, body_b: j.child_runtime_body_index,
            record_index: u32::try_from(record_index).map_err(|_| "record_index overflow")?,
            enabled: (if j.wire.enabled { BALL_ANCHOR_FLAG_ACTIVE } else { 0 }) |
                (if j.wire.enabled && !j.wire.collide_connected
                    && j.parent_runtime_body_index != WORLD_ENDPOINT
                    && j.child_runtime_body_index != WORLD_ENDPOINT {
                    BALL_ANCHOR_FLAG_NO_COLLIDE
                } else { 0 }),
            anchor_a: [f32_checked(a[0], "anchor_a[0]")?, f32_checked(a[1], "anchor_a[1]")?, f32_checked(a[2], "anchor_a[2]")?],
            anchor_b: [f32_checked(b[0], "anchor_b[0]")?, f32_checked(b[1], "anchor_b[1]")?, f32_checked(b[2], "anchor_b[2]")?],
            stiffness: f32_checked(j.wire.stiffness_N_per_m, "stiffness")? })
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

    fn validated_fixture() -> joint::OwnedJointScene {
        let fixture = fixture();
        let bytes = std::fs::read(fixture.join("bin/pdrd_body.bin")).unwrap();
        let rows = bytes
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        joint::ingest_validated(&fixture, &rows).unwrap().unwrap()
    }

    #[test]
    fn converts_real_validated_source_fixture() {
        let scene = validated_fixture();
        let records = from_validated(&scene).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].body_a, 0);
        assert_eq!(records[0].body_b, 1);
        assert_eq!(records[0].record_index, 0);
        assert_eq!(records[0].enabled, 0);
        assert_eq!(records[0].anchor_a, [0.1, 0.0, 0.0]);
        assert_eq!(records[0].anchor_b, [0.0, 0.2, 0.0]);
        assert_eq!(records[0].stiffness, 12.5);
    }

    #[test]
    fn encodes_connected_collision_flag_without_changing_abi() {
        let mut scene = validated_fixture();
        scene.joints[0].wire.enabled = true;
        scene.joints[0].wire.collide_connected = false;
        let records = from_validated(&scene).unwrap();
        assert_eq!(std::mem::size_of::<BallAnchorGpu>(), 44);
        assert_eq!(records[0].enabled,
                   BALL_ANCHOR_FLAG_ACTIVE | BALL_ANCHOR_FLAG_NO_COLLIDE);

        scene.joints[0].wire.enabled = false;
        let disabled = from_validated(&scene).unwrap();
        assert_eq!(disabled[0].enabled, 0);
    }

    #[test]
    fn rejects_f64_values_outside_finite_f32_range() {
        let mut scene = validated_fixture();
        scene.joints[0].wire.stiffness_N_per_m = f64::MAX;
        assert!(from_validated(&scene)
            .unwrap_err()
            .contains("stiffness: outside finite f32 range"));
    }
}

// repr(C) fixed-width transport, with no pointers or padding.
unsafe impl ppf_cts_compute::Pod for BallAnchorGpu {}
