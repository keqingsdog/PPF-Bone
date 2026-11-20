//! Independent PMX joint graph. Row ordering has no skeletal parent semantics.
use std::collections::HashSet;

use serde::Deserialize;

use crate::{
    joint::{self, JointError, LocalFrame, ValidatedJointInput},
    joint_d6_hard_wire::{Axis, AxisMap},
    soft_joint_gpu::{FemEndpointGpu, FemJointGpu, PmxJointGpu, FEM_PDRD},
};

pub const FEATURE: &str = "pmx_joints_v1";
pub const FILE: &str = "bin/pmx_joints_v1.json";
const KIND: &str = "ppf.pmx_joints";
pub const LINEAR_LIMIT_WEIGHT: f64 = 200_000.0;
pub const ANGULAR_LIMIT_WEIGHT: f64 = 2_000.0;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Payload {
    version: u32,
    kind: String,
    scene_uuid: String,
    joints: Vec<Joint>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Joint {
    joint_uuid: String,
    body_a_uuid: String,
    body_b_uuid: String,
    frame_a: LocalFrame,
    frame_b: LocalFrame,
    translation: AxisMap,
    rotation: AxisMap,
    #[serde(rename = "linear_spring_N_per_m")]
    linear_spring: [f64; 3],
    #[serde(rename = "angular_spring_Nm_per_rad")]
    angular_spring: [f64; 3],
}

fn bad(field: &str, detail: &str) -> JointError {
    joint::err("pmx-joints", "PMX_JOINT_INVALID", field, None, None, detail)
}

fn narrow(value: f64) -> Result<f32, JointError> {
    crate::soft_joint::float(value)
        .map_err(|_| bad("number", "finite representable f32 required"))
}

fn canonical(id: &str) -> bool {
    joint::canonical_uuid(id).as_deref() == Some(id)
}

pub fn prepare(input: &ValidatedJointInput, scale: f64) -> Result<Vec<FemJointGpu>, JointError> {
    if input.contains(FILE) != input.requires_feature(FEATURE)? {
        return Err(bad("required_features", "PMX sidecar and feature must appear together"));
    }
    if !input.contains(FILE) {
        return Ok(Vec::new());
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err(bad("world_scaling", "positive finite scale required"));
    }
    let bytes = input.bytes(FILE)?;
    if bytes.len() > 16 * 1024 * 1024 {
        return Err(bad("payload", "16 MiB limit exceeded"));
    }
    let strict: joint::StrictValue = serde_json::from_slice(bytes)
        .map_err(|e| bad("payload", &e.to_string()))?;
    let payload: Payload = serde_json::from_value(strict.0)
        .map_err(|e| bad("payload", &e.to_string()))?;
    if payload.version != 1 || payload.kind != KIND || payload.scene_uuid != input.records().scene_uuid
        || payload.joints.is_empty() || payload.joints.len() > 65536 {
        return Err(bad("identity", "expected a nonempty bounded PMX graph for the sealed scene"));
    }
    let bind = |id: &str, frame: &LocalFrame| -> Result<(FemEndpointGpu, [f32; 4]), JointError> {
        let runtime = input.records().bodies.get(id)
            .filter(|_| canonical(id))
            .ok_or_else(|| bad("body_uuid", "endpoint must resolve to a sealed PDRD body"))?;
        let norm = frame.quaternion_wxyz.iter().map(|q| q * q).sum::<f64>().sqrt();
        if !norm.is_finite() || (norm - 1.0).abs() > 1e-8 {
            return Err(bad("quaternion_wxyz", "finite unit frame quaternion required"));
        }
        let mut endpoint = FemEndpointGpu {
            kind: FEM_PDRD, id: *runtime, vertices: [0; 4], dm_inv: [0.; 9],
            anchor_bary: [0.; 4], center_bary: [0.; 4],
            rest_rotation: [1., 0., 0., 0., 1., 0., 0., 0., 1.],
            local_anchor: [0.; 3],
        };
        for (out, value) in endpoint.local_anchor.iter_mut().zip(frame.translation_m) {
            *out = narrow(value * scale)?;
        }
        Ok((endpoint, frame.quaternion_wxyz.map(|q| q as f32)))
    };
    let mut ids: HashSet<_> = input.records().joints.iter().map(|j| j.wire.joint_uuid.as_str()).collect();
    let mut records = Vec::with_capacity(payload.joints.len());
    for row in &payload.joints {
        if !canonical(&row.joint_uuid) || !ids.insert(&row.joint_uuid) || row.body_a_uuid == row.body_b_uuid {
            return Err(bad("joint_uuid", "unique joint identity and distinct endpoints required"));
        }
        let (a, frame_a) = bind(&row.body_a_uuid, &row.frame_a)?;
        let (b, frame_b) = bind(&row.body_b_uuid, &row.frame_b)?;
        let mut pmx = PmxJointGpu {
            enabled: 1, frame_a, frame_b,
            limit_weight: [narrow(LINEAR_LIMIT_WEIGHT)?, narrow(ANGULAR_LIMIT_WEIGHT)?],
            ..Default::default()
        };
        let springs = row.linear_spring.into_iter().chain(row.angular_spring);
        for (i, (axis, spring)) in row.translation.ordered().into_iter()
            .chain(row.rotation.ordered()).zip(springs).enumerate() {
            let bound_scale = if i < 3 { scale } else { 1.0 };
            if spring < 0.0 {
                return Err(bad("spring", "nonnegative coefficient required"));
            }
            // World scaling changes lengths. Force/torque parameters remain
            // the authored SI values, as for other scene joint records.
            pmx.springs[i] = narrow(spring)?;
            let (mode, lower, upper) = match *axis {
                Axis::Free => (0, 0.0, 0.0),
                Axis::Locked { value } => (2, value, value),
                Axis::Limited { lower, upper } => (1, lower, upper),
            };
            pmx.modes[i] = mode;
            pmx.lower[i] = narrow(lower * bound_scale)?;
            pmx.upper[i] = narrow(upper * bound_scale)?;
            if mode == 1 && pmx.lower[i] >= pmx.upper[i] {
                return Err(bad("translation/rotation", "limited interval must remain ordered in f32"));
            }
            if i >= 3 && (lower.abs() > f64::from(std::f32::consts::PI) || upper.abs() > f64::from(std::f32::consts::PI)) {
                return Err(bad("rotation", "Euler bounds outside [-pi, pi]"));
            }
        }
        let mut record = crate::cross_links::inert_row(a, b);
        record.pmx = pmx;
        records.push(record);
    }
    Ok(records)
}
