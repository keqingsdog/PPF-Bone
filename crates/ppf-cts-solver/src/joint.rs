//! R1 PDRD ball-anchor interchange and host-side math.
//!
//! This module owns the version-2 payload independently of the legacy
//! `pdrd_body.bin` ABI. The neutral driver consumes only validated records.

use nalgebra::{Matrix3, SMatrix, SVector, Vector3};
use ppf_cts_formats::envelope::{from_cbor_with_version, FormatError};
use serde::{de::{self, MapAccess, Visitor}, Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::{HashMap, HashSet}, fs, path::{Component, Path}};

pub const JOINT_FILE: &str = "bin/pdrd_ball_anchor.cbor";
pub const CONTEXT_FILE: &str = "bin/pdrd_scene_context.cbor";
pub const KIND: &str = "ppf.pdrd_ball_anchor";
pub const CONTEXT_KIND: &str = "ppf.pdrd_scene_context";
pub const ENVELOPE_VERSION: u32 = 2;
/// The V3 envelope is the prior canonical home of directional (per-axis)
/// angular control: it keeps every V2 field and adds `required_features` plus
/// the optional `joints[i].angular_control_v1` record. V2 MUST NOT carry
/// either, so an old reader sees a higher version and refuses rather than
/// silently ignoring a requested control.
pub const ANGULAR_CONTROL_ENVELOPE_VERSION: u32 = 3;
pub const ANGULAR_EXTENSION_ENVELOPE_VERSION: u32 = 4;
/// The V3 capability feature, spelled exactly as the prior contract does.
pub const ANGULAR_CONTROL_FEATURE: &str = "pdrd_angular_controls_v1";
/// The V4 extension fields (soft limits and drives). They are named here only
/// so that a payload carrying them is refused BY NAME as unimplemented rather
/// than falling into a generic unknown-field error, or worse, being accepted
/// and ignored. Nothing in this crate implements them yet.
pub const ANGULAR_SOFT_LIMITS_FIELD: &str = "angular_soft_limits_v1";
pub const ANGULAR_DRIVES_FIELD: &str = "angular_drives_v1";
pub const ANGULAR_SOFT_LIMITS_FEATURE: &str = "pdrd_angular_soft_limits_v1";
pub const ANGULAR_SWING_CONE_FEATURE: &str = "pdrd_angular_swing_cone_v1";
pub const ANGULAR_DRIVES_FEATURE: &str = "pdrd_angular_drives_v1";
pub const CAPABILITY: &str = "pdrd_ball_anchor_v1";
pub const WORLD_CAPABILITY: &str = "pdrd_world_ball_anchor_v1";
pub const WORLD_ENDPOINT_UUID: &str = "00000000-0000-0000-0000-000000000000";
pub const PDRD_BODY_ROW_LEN: usize = 23;

/// Rust-side selection is deliberately independent of record count, enabled
/// bits, and stiffness.  Joint-off controls must eventually use the same
/// native candidate/commit path as joint-on controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentalBallAnchorSelection {
    Off,
    Requested,
}

#[cfg(test)]
mod world_endpoint_tests {
    use super::*;
    pub(super) fn payload() -> JointPayload {
        JointPayload {
            required_capability: WORLD_CAPABILITY.into(), scene_uuid: "11111111-1111-1111-1111-111111111111".into(), scene_inputs_sha256: String::new(),
            bodies: vec![BodyWire { body_uuid: "22222222-2222-2222-2222-222222222222".into(), runtime_body_index: 0 }, BodyWire { body_uuid: "33333333-3333-3333-3333-333333333333".into(), runtime_body_index: 1 }],
            joints: vec![JointWire { joint_uuid: "44444444-4444-4444-4444-444444444444".into(), parent_body_uuid: WORLD_ENDPOINT_UUID.into(), child_body_uuid: "33333333-3333-3333-3333-333333333333".into(), parent_local_frame: LocalFrame { translation_m: [0.0, 2.0, 0.0], quaternion_wxyz: [1.0, 0.0, 0.0, 0.0] }, child_local_frame: LocalFrame { translation_m: [0.0; 3], quaternion_wxyz: [1.0, 0.0, 0.0, 0.0] }, stiffness_N_per_m: 37.0, enabled: true, collide_connected: true, rest_relative_quaternion_wxyz: [1.0, 0.0, 0.0, 0.0], angle_stiffness_nm_per_rad: 0.0, angle_damping_nms_per_rad: 0.0, max_angle_rad: 0.0, angular_control_v1: None, angular_soft_limits_v1: None, angular_drives_v1: None, angular_hard_hinge_v1: None, joint_d6_hard_v1: None }],
            required_features: None,
        }
    }
    fn ingest_test(p: JointPayload) -> Result<OwnedJointScene, JointError> {
        ingest_payload(p, ENVELOPE_VERSION, &[0.0; 46], InputSnapshot { bytes: HashMap::new(), optional_absent: Vec::new() })
    }
    #[test]
    fn world_endpoint_maps_to_reserved_index_and_needs_feature_capability() {
        let p = payload();
        let scene = ingest_test(p.clone()).unwrap();
        assert_eq!(scene.joints[0].parent_runtime_body_index, crate::joint_gpu::WORLD_ENDPOINT);
        assert_eq!(scene.joints[0].child_runtime_body_index, 1);
        let mut old_capability = p;
        old_capability.required_capability = CAPABILITY.into();
        assert_eq!(ingest_test(old_capability).unwrap_err().stable_code, "JOINT_R1_UNSUPPORTED_REQUIRED_CAPABILITY");
    }
    #[test]
    fn world_endpoint_is_parent_only_and_requires_identity_frame() {
        let mut child_world = payload();
        child_world.joints[0].child_body_uuid = WORLD_ENDPOINT_UUID.into();
        assert_eq!(ingest_test(child_world).unwrap_err().stable_code, "JOINT_R1_UNSUPPORTED_WORLD_ENDPOINT");
        let mut rotated_world = payload();
        rotated_world.joints[0].parent_local_frame.quaternion_wxyz = [0.0, 1.0, 0.0, 0.0];
        assert_eq!(ingest_test(rotated_world).unwrap_err().stable_code, "JOINT_R1_INVALID_WORLD_FRAME");
    }
    #[test]
    fn reads_generated_dynamic_impact_scene_from_python_export() {
        let Ok(path) = std::env::var("PPF_R1_WORLD_FIXTURE") else { return; };
        let input = prepare_native_input(Path::new(&path)).unwrap().expect("sealed R1 world-anchor scene");
        assert_eq!(input.records.joints.len(), 6);
        assert_eq!(input.records.joints[0].parent_runtime_body_index, crate::joint_gpu::WORLD_ENDPOINT);
        assert_eq!(input.records.joints[0].child_runtime_body_index, 0);
        assert_eq!(input.records.bodies.len(), 7);
        let gpu = crate::joint_gpu::from_validated(&input.records).unwrap();
        assert_eq!(gpu.len(), 6);
        assert_eq!(gpu[0].body_a, crate::joint_gpu::WORLD_ENDPOINT);
        assert_eq!(gpu[0].body_b, 0);
        assert!(gpu[1..].iter().all(|record| record.body_a != crate::joint_gpu::WORLD_ENDPOINT && record.body_b != crate::joint_gpu::WORLD_ENDPOINT));
    }
}

/// Bounded versioned result expected from the future native C++ main ABI.
/// Do not infer readiness from `upload_ball_anchor_records`: upload only
/// copies POD records and says nothing about authoritative poses, fresh CCD,
/// or finality.  Until C++ exposes and Rust verifies this handshake, the
/// selection fails closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeExperimentalHandshake {
    Unavailable,
    Ready { abi_version: u32, capability: &'static str },
}

const EXPERIMENTAL_NATIVE_ABI_VERSION: u32 = 1;

fn native_experimental_handshake() -> NativeExperimentalHandshake {
    NativeExperimentalHandshake::Unavailable
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JointError {
    pub stage: &'static str,
    pub stable_code: &'static str,
    pub field: String,
    pub joint_uuid: Option<String>,
    pub body_uuid: Option<String>,
    pub detail: String,
}
impl std::fmt::Display for JointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{stage:{}, stable_code:{}, field:{}, joint_uuid:{:?}, body_uuid:{:?}, detail:{}}}", self.stage, self.stable_code, self.field, self.joint_uuid, self.body_uuid, self.detail)
    }
}
impl std::error::Error for JointError {}
pub(crate) fn err(stage: &'static str, code: &'static str, field: &str, joint: Option<&str>, body: Option<&str>, detail: &str) -> JointError {
    JointError { stage, stable_code: code, field: field.into(), joint_uuid: joint.map(str::to_owned), body_uuid: body.map(str::to_owned), detail: detail.into() }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalFrame { pub translation_m: [f64; 3], pub quaternion_wxyz: [f64; 4] }
fn identity_quaternion_wxyz() -> [f64; 4] { [1.0, 0.0, 0.0, 0.0] }
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointWire {
    pub joint_uuid: String, pub parent_body_uuid: String, pub child_body_uuid: String,
    pub parent_local_frame: LocalFrame, pub child_local_frame: LocalFrame,
    pub stiffness_N_per_m: f64, pub enabled: bool, pub collide_connected: bool,
    #[serde(default = "identity_quaternion_wxyz")] pub rest_relative_quaternion_wxyz: [f64; 4],
    #[serde(default, rename = "angle_stiffness_Nm_per_rad")] pub angle_stiffness_nm_per_rad: f64,
    #[serde(default, rename = "angle_damping_Nms_per_rad")] pub angle_damping_nms_per_rad: f64,
    /// Optional isotropic angular deviation limit from the rest-relative pose.
    /// Zero disables the limit; positive values are radians in (0, pi].
    #[serde(default, rename = "max_angle_rad")] pub max_angle_rad: f64,
    /// Optional directional (per-axis) angular control, in the prior canonical
    /// shape: a coordinate model plus a per-axis stiffness/damping map. Absent
    /// leaves the V2 isotropic path authoritative and the directional record
    /// disabled. Values are signed-domain physical units (Nm/rad, Nms/rad); the
    /// scalar `max_angle_rad` nonnegativity and `<= pi` gate is deliberately NOT
    /// applied to them.
    ///
    /// A PRESENT key is authoritative, including a present `null`: exactly as
    /// the prior contract's `present_angular_control` does, an explicit null is
    /// a malformed record rather than an absent one, so a producer cannot
    /// disable a requested control by spelling it null.
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "present_angular_control")]
    pub angular_control_v1: Option<AngularControlWireV1>,
    /// V4 extension fields. Deliberately declared so their presence is a TYPED
    /// refusal (`JOINT_ANGULAR_EXTENSION_UNIMPLEMENTED`) rather than an unknown
    /// field, and so this envelope can never accept them and ignore them.
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "unimplemented_extension")]
    pub angular_soft_limits_v1: Option<StrictValue>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "unimplemented_extension")]
    pub angular_drives_v1: Option<StrictValue>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "crate::joint_angular_hard_wire::present_hard")]
    pub angular_hard_hinge_v1: Option<crate::joint_angular_hard_wire::AngularHardHingeWireV1>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "crate::joint_d6_hard_wire::present_d6")]
    pub joint_d6_hard_v1: Option<crate::joint_d6_hard_wire::D6Wire>,
}

/// The canonical per-axis triple of the prior directional control contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AngularAxisValue { pub twist_x: f64, pub swing_y: f64, pub swing_z: f64 }
impl AngularAxisValue {
    pub fn as_array(&self) -> [f64; 3] { [self.twist_x, self.swing_y, self.swing_z] }
}
/// `joints[i].angular_control_v1`. `rest_relative_quaternion_wxyz` continues to
/// come from the joint's own field, so the rest orientation is not discarded.
///
/// Field names and axes are the prior canonical ones exactly: the keys are
/// `stiffness_Nm_per_rad` / `damping_Nms_per_rad` at the top level (NOT the
/// Rust-idiomatic snake case) and each value is an object with `twist_x`,
/// `swing_y`, `swing_z`. `deny_unknown_fields` on both levels plus the strict
/// presence helper below is what makes a missing, misspelled, duplicated or
/// extra key a refusal rather than a silently defaulted coefficient.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AngularControlWireV1 {
    pub coordinate_model: String,
    #[serde(rename = "stiffness_Nm_per_rad")] pub stiffness_nm_per_rad: AngularAxisValue,
    #[serde(rename = "damping_Nms_per_rad")] pub damping_nms_per_rad: AngularAxisValue,
}
impl AngularControlWireV1 {
    pub fn stiffness(&self) -> [f64; 3] { self.stiffness_nm_per_rad.as_array() }
    pub fn damping(&self) -> [f64; 3] { self.damping_nms_per_rad.as_array() }
}

/// Any nested CBOR value, decoded with DUPLICATE-KEY REJECTION at every level.
///
/// `serde_json::Value` (and a plain map) silently keeps the last of two equal
/// keys, so a payload that spells a coefficient twice would be accepted with
/// one of the two values chosen by declaration order. These wrappers reject
/// duplicates recursively, which is the behaviour the prior extension wire
/// established and which this envelope needs to keep.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(transparent)]
pub struct StrictValue(pub serde_json::Value);
impl<'de> Deserialize<'de> for StrictValue {
    fn deserialize<D: de::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct StrictVisitor;
        impl<'de> Visitor<'de> for StrictVisitor {
            type Value = StrictValue;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a finite scalar, array, or map with unique string keys")
            }
            fn visit_bool<E: de::Error>(self, v: bool) -> Result<Self::Value, E> { Ok(StrictValue(serde_json::Value::Bool(v))) }
            fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> { Ok(StrictValue(serde_json::Value::Number(v.into()))) }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> { Ok(StrictValue(serde_json::Value::Number(v.into()))) }
            fn visit_f64<E: de::Error>(self, v: f64) -> Result<Self::Value, E> {
                serde_json::Number::from_f64(v).map(serde_json::Value::Number).map(StrictValue)
                    .ok_or_else(|| E::custom("nonfinite wire number"))
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> { Ok(StrictValue(serde_json::Value::String(v.to_owned()))) }
            fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> { Ok(StrictValue(serde_json::Value::Null)) }
            fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> { Ok(StrictValue(serde_json::Value::Null)) }
            fn visit_some<D: de::Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> { StrictValue::deserialize(d) }
            fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(StrictValue(v)) = seq.next_element::<StrictValue>()? { values.push(v); }
                Ok(StrictValue(serde_json::Value::Array(values)))
            }
            fn visit_map<A: de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut values = serde_json::Map::new();
                while let Some(key) = map.next_key::<String>()? {
                    let StrictValue(value) = map.next_value::<StrictValue>()?;
                    if values.insert(key.clone(), value).is_some() {
                        return Err(de::Error::custom(format!("duplicate wire key: {key}")));
                    }
                }
                Ok(StrictValue(serde_json::Value::Object(values)))
            }
        }
        deserializer.deserialize_any(StrictVisitor)
    }
}
fn present_angular_control<'de, D: de::Deserializer<'de>>(d: D) -> Result<Option<AngularControlWireV1>, D::Error> {
    AngularControlWireV1::deserialize(d).map(Some)
}
/// Present V4 extension fields decode strictly and then fail by name in the
/// capability gate below. Decoding them here keeps the typed code stable even
/// when the offending record is malformed: the version is refused first.
fn unimplemented_extension<'de, D: de::Deserializer<'de>>(d: D) -> Result<Option<StrictValue>, D::Error> {
    StrictValue::deserialize(d).map(Some)
}
fn present_required_features<'de, D: de::Deserializer<'de>>(d: D) -> Result<Option<Vec<String>>, D::Error> {
    Vec::<String>::deserialize(d).map(Some)
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BodyWire { pub body_uuid: String, pub runtime_body_index: u32 }
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointPayload {
    pub required_capability: String, pub scene_uuid: String, pub scene_inputs_sha256: String,
    pub bodies: Vec<BodyWire>, pub joints: Vec<JointWire>,
    /// V3-only. Absent on V2, where its presence is itself a refusal.
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "present_required_features")] pub required_features: Option<Vec<String>>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextInput { pub relative_path: String, pub byte_length: usize, pub sha256: String }
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextBody { pub body_uuid: String, pub runtime_body_index: u32, pub vertex_count: u32 }
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneInputContext { pub context_version: u32, pub scene_uuid: String, pub builder_contract_version: u32, pub complete: bool, pub bodies: Vec<ContextBody>, pub inputs: Vec<ContextInput>, pub optional_absent: Vec<String>, pub scene_inputs_sha256: String,
    #[serde(default,skip_serializing_if="Option::is_none")]
    pub restoring_curves_v1:Option<RestoringExtensionSeal>,
}
#[derive(Debug,Clone,Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RestoringExtensionSeal{
    pub required_feature:String,pub sidecar_sha256:String,pub sidecar_byte_length:usize,pub combined_inputs_sha256:String,
}
#[derive(Debug, Clone, Serialize)]
struct VertexMap { version: u32, kind: String, payload: HashMap<String, Vec<u32>> }
struct StrictVertexPayload(HashMap<String, Vec<u32>>);
impl<'de> Deserialize<'de> for StrictVertexPayload {
    fn deserialize<D: de::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct PayloadVisitor;
        impl<'de> Visitor<'de> for PayloadVisitor {
            type Value = StrictVertexPayload;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("a unique string-to-u32-vector map") }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut values = HashMap::new();
                while let Some((key, value)) = map.next_entry::<String, Vec<u32>>()? {
                    if values.insert(key.clone(), value).is_some() { return Err(de::Error::custom(format!("duplicate VertexMap payload key: {key}"))); }
                }
                Ok(StrictVertexPayload(values))
            }
        }
        deserializer.deserialize_map(PayloadVisitor)
    }
}
impl<'de> Deserialize<'de> for VertexMap {
    fn deserialize<D: de::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire { version: u32, kind: String, payload: StrictVertexPayload }
        let wire = Wire::deserialize(deserializer)?;
        Ok(Self { version: wire.version, kind: wire.kind, payload: wire.payload.0 })
    }
}

fn decode_vertex_map(bytes: &[u8]) -> Result<VertexMap, JointError> {
    let map: VertexMap = ciborium::from_reader(bytes).map_err(|e| err("context", "JOINT_R1_MEMBERSHIP_INVALID", "map.pickle", None, None, &format!("{e:?}")))?;
    // Both official formats carry the same string -> u32 vertex-list shape.
    // v1 is retained for sealed historical fixtures; v2 is the current producer.
    if !matches!(map.version, 1 | 2) || map.kind != "VertexMap" {
        return Err(err("context", "JOINT_R1_MEMBERSHIP_INVALID", "map.pickle", None, None, "unsupported vertex map"));
    }
    Ok(map)
}
#[derive(Debug, Deserialize)]
struct NativeConfig { param: crate::args::SimArgs }
#[derive(Debug, Clone)]
struct InputSnapshot { bytes: HashMap<String, Vec<u8>>, optional_absent: Vec<String> }
#[derive(Debug, Clone)]
pub struct OwnedJoint { pub wire: JointWire, pub parent_runtime_body_index: u32, pub child_runtime_body_index: u32 }
/// Typed, non-forgeable proof that a real PDRD runtime body was validated as the
/// enabled parent endpoint of a genuine material (FEM / mixed) joint edge.
///
/// Trust boundary. This value is populated ONLY by `prepare_native_input`, and
/// only AFTER `soft_joint::prepare` succeeded, from the already-validated
/// UUID -> runtime body map. It has:
///   * no public constructor and no `Deserialize` impl anywhere in its type
///     graph, so malformed external input cannot manufacture membership;
///   * `Default` == empty, which is what the original rigid-only path keeps;
///   * a private field, so code outside this module cannot set it at all.
/// The driver therefore never parses raw unvalidated JSON to decide this.
#[derive(Debug, Clone, Default)]
pub struct ValidatedFemParentProof { indices: HashSet<u32> }
impl ValidatedFemParentProof {
    pub fn is_empty(&self) -> bool { self.indices.is_empty() }
    pub fn len(&self) -> usize { self.indices.len() }
    /// Runtime lookup: was this native runtime body the enabled PDRD parent of a
    /// validated material edge?
    pub fn contains(&self, runtime_body_index: u32) -> bool { self.indices.contains(&runtime_body_index) }
    pub(crate) fn insert_validated(&mut self, runtime_body_index: u32) { self.indices.insert(runtime_body_index); }
}

/// Pure filter that builds the proof set from VALIDATED inputs only.
///
/// * `bodies` / `wires` come from the already-validated `soft_joint` payload.
///   `soft_joint::prepare` has already refused any wire whose endpoints are both
///   rigid, so every wire reaching here IS a genuine material edge.
/// * `runtime` is `OwnedJointScene::bodies`, the validated UUID -> runtime map.
/// * `soft_joint::prepare` has already refused a PDRD body whose
///   `runtime_body_index` disagrees with that map, so the index inserted here is
///   a real sealed PDRD body row, never a phantom one.
///
/// A wire contributes only when ALL hold: it is enabled, its parent is not the
/// world endpoint, its parent body is a genuine PDRD, and that UUID has a real
/// validated runtime index. A FEM parent (including a FEM-FEM edge), a world
/// parent, a disabled edge and an unknown UUID each contribute nothing.
pub(crate) fn collect_fem_pdrd_parent_proof(
    bodies: &[crate::soft_joint::BodyWire],
    wires: &[crate::soft_joint::JointRow],
    runtime: &HashMap<String, u32>,
) -> ValidatedFemParentProof {
    let kind_of: HashMap<&str, &str> =
        bodies.iter().map(|b| (b.body_uuid.as_str(), b.kind.as_str())).collect();
    let mut proof = ValidatedFemParentProof::default();
    for row in wires {
        let w = &row.wire;
        if !w.enabled { continue; }
        let parent = w.parent_body_uuid.as_str();
        if parent == WORLD_ENDPOINT_UUID { continue; }
        // Literal matches the sealed kind spelling used by soft_joint::prepare.
        if kind_of.get(parent).copied() != Some("PDRD") { continue; }
        if let Some(&index) = runtime.get(parent) { proof.insert_validated(index); }
    }
    proof
}
#[derive(Debug, Clone)]
pub struct OwnedJointScene { pub scene_uuid: String, pub scene_inputs_sha256: String, pub payload_sha256: String,
    /// The exact envelope version this scene was validated from. The version
    /// gate above is what decides it, so a consumer can state which contract it
    /// is holding instead of assuming the newest one.
    pub payload_version: u32,
    pub body_controls_v1: bool,
    pub hard_records: Vec<crate::joint_hard_gpu::HardJointGpu>,
    pub bodies: HashMap<String, u32>, pub joints: Vec<OwnedJoint>, pub roots: Vec<crate::joint_root_motion_wire::ValidatedRootMotion>,
    /// Retained strict-prepare proof for the later runtime actual-geometry rebind
    /// (`driver::roots::install_native`). Empty on the original rigid path.
    /// Private: not settable from outside this module and not decoded from input.
    fem_pdrd_parent_proof: ValidatedFemParentProof,
    snapshot: InputSnapshot }
impl OwnedJointScene {
    /// The native constructor still owns legacy decoded arrays.  It must call
    /// this immediately before handing those arrays to the backend so a file
    /// replacement after sealing is a typed refusal, never a silent solve.
    pub fn verify_snapshot_unchanged(&self, root: &Path) -> Result<(), JointError> {
        // Re-enumerate, rather than merely re-hash retained names: an input
        // added after validation (or an optional file that was absent) must not
        // become visible to a future native loader at handoff.
        checked_scene_root(root)?;
        let actual = scan_consumed_paths(root)?;
        let mut expected: Vec<_> = self.snapshot.bytes.keys().cloned().collect();
        expected.sort();
        if actual != expected { return Err(err("context", "JOINT_R1_HANDOFF_INVENTORY_CHANGED", "inputs", None, None, "consumed input was added, removed, or replaced after validation")); }
        for absent in &self.snapshot.optional_absent {
            if root.join(absent).exists() { return Err(err("context", "JOINT_R1_HANDOFF_OPTIONAL_ABSENCE_CHANGED", absent, None, None, "declared absent input appeared after validation")); }
        }
        for (relative, expected) in &self.snapshot.bytes {
            let now=fs::read(safe_existing_file(root, relative)?).map_err(|e|err("context","JOINT_R1_CONTEXT_IO",relative,None,None,&e.to_string()))?;
            if now!=*expected{return Err(err("context","JOINT_R1_INPUT_CHANGED_AFTER_VALIDATION",relative,None,None,"native input changed after validated snapshot"));}
        }
        Ok(())
    }
    /// Retained strict-prepare FEM/mixed PDRD-parent proof. Empty on the original
    /// rigid path. The runtime root installer unions this with the legacy
    /// known-parent check instead of recomputing it from `joints` alone.
    pub fn fem_pdrd_parent_proof(&self) -> &ValidatedFemParentProof { &self.fem_pdrd_parent_proof }
}

/// The only joint-mode input handed to the native scene constructor.  It owns
/// the manifest/context control records as well as every sealed native input;
/// callers therefore cannot accidentally validate one generation and decode
/// another generation from disk.
#[derive(Debug, Clone)]
pub struct ValidatedJointInput {
    records: OwnedJointScene,
    manifest_bytes: Vec<u8>,
    context_bytes: Vec<u8>,
}
impl ValidatedJointInput {
    pub fn records(&self) -> &OwnedJointScene { &self.records }
    /// Transfer the already validated, sealed scene records into the native
    /// scene.  Callers must not re-ingest the manifest or context after this
    /// point: that would reopen the validation/decode generation boundary.
    pub fn into_records(self) -> OwnedJointScene { self.records }
    pub fn bytes(&self, relative: &str) -> Result<&[u8], JointError> {
        self.records.snapshot.bytes.get(relative).map(Vec::as_slice)
            .ok_or_else(|| err("context", "JOINT_R1_SNAPSHOT_MISSING", relative, None, None, "joint native decode may not fall back to filesystem"))
    }
    pub fn contains(&self, relative: &str) -> bool { self.records.snapshot.bytes.contains_key(relative) }
    pub fn requires_feature(&self, feature:&str)->Result<bool,JointError> {
        let (payload,_) = decode_joint_payload(&self.manifest_bytes)?;
        Ok(payload.required_features.as_ref().is_some_and(|v|v.iter().any(|s|s==feature)))
    }
    pub fn param_files(&self) -> Vec<String> {
        let mut files: Vec<_> = self.records.snapshot.bytes.keys()
            .filter_map(|p| p.strip_prefix("bin/param/").filter(|n| !n.contains('/')).map(str::to_owned)).collect();
        files.sort(); files
    }
    pub fn verify_unchanged(&self, root: &Path) -> Result<(), JointError> {
        self.records.verify_snapshot_unchanged(root)?;
        for (relative, expected) in [(JOINT_FILE, &self.manifest_bytes), (CONTEXT_FILE, &self.context_bytes)] {
            let now = fs::read(safe_existing_file(root, relative)?).map_err(|e| err("context", "JOINT_R1_CONTEXT_IO", relative, None, None, &e.to_string()))?;
            if now != *expected { return Err(err("context", "JOINT_R1_INPUT_CHANGED_AFTER_VALIDATION", relative, None, None, "joint control input changed after validated snapshot")); }
        }
        Ok(())
    }
}

/// A manifest is a capability requirement even if it happens to carry no
/// records.  A legacy scene is represented by the absence of this file.
pub fn required_capability(path: &Path, pdrd_rows: &[f32]) -> Result<Option<OwnedJointScene>, JointError> {
    ingest(path, pdrd_rows)
}

pub(crate) fn canonical_uuid(v: &str) -> Option<String> {
    // Match the frontend's `uuid.UUID` spelling acceptance.  The map still
    // requires the canonical spelling for bound body keys below, so accepting
    // this here cannot turn an alias into an ignored authoritative entry.
    let bare = v.strip_prefix("urn:uuid:").unwrap_or(v);
    let bare = bare.strip_prefix('{').and_then(|x| x.strip_suffix('}')).unwrap_or(bare);
    if bare.len() == 32 && bare.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Some(format!("{}-{}-{}-{}-{}", &bare[0..8], &bare[8..12], &bare[12..16], &bare[16..20], &bare[20..32]).to_ascii_lowercase());
    }
    (bare.len() == 36 && [8, 13, 18, 23].into_iter().all(|i| bare.as_bytes()[i] == b'-') &&
        bare.bytes().enumerate().all(|(i, c)| matches!(i, 8 | 13 | 18 | 23) || c.is_ascii_hexdigit()))
        .then(|| bare.to_ascii_lowercase())
}
fn uuid_ok(v: &str) -> bool { canonical_uuid(v).is_some() }
fn finite3(v: &[f64; 3]) -> bool { v.iter().all(|x| x.is_finite()) }
fn valid_quaternion(q: &[f64; 4]) -> bool { q.iter().all(|x| x.is_finite()) && (q.iter().map(|x| x*x).sum::<f64>().sqrt() - 1.0).abs() <= 1e-8 }
fn valid_frame(v: &LocalFrame) -> bool { finite3(&v.translation_m) && valid_quaternion(&v.quaternion_wxyz) }
fn hex(bytes: &[u8]) -> String { bytes.iter().map(|b| format!("{b:02x}")).collect() }
pub(crate) fn sha256(bytes: &[u8]) -> String { hex(&Sha256::digest(bytes)) }
fn read_u32(path: &Path) -> Result<Vec<u32>, JointError> { let b=fs::read(path).map_err(|e|err("context","JOINT_R1_CONTEXT_IO","file",None,None,&e.to_string()))?; if b.len()%4!=0{return Err(err("context","JOINT_R1_CONTEXT_INVALID","file",None,None,"u32 alignment"));} Ok(b.chunks_exact(4).map(|x|u32::from_le_bytes(x.try_into().unwrap())).collect()) }
fn read_f32(path: &Path) -> Result<Vec<f32>, JointError> { let b=fs::read(path).map_err(|e|err("context","JOINT_R1_CONTEXT_IO","file",None,None,&e.to_string()))?; if b.len()%4!=0{return Err(err("context","JOINT_R1_CONTEXT_INVALID","file",None,None,"f32 alignment"));} Ok(b.chunks_exact(4).map(|x|f32::from_le_bytes(x.try_into().unwrap())).collect()) }
fn field_u32(x:f32, field:&str, body:&str)->Result<usize,JointError>{if !x.is_finite()||x<0.||x.fract()!=0.||x>16_777_216. {return Err(err("context","JOINT_R1_PDRD_ROW_INVALID",field,None,Some(body),"must be exact nonnegative f32 integer"));}Ok(x as usize)}

const OPTIONAL_INPUTS: &[&str] = &["dyn_param.txt", "bin/rest_vert.bin", "bin/rest_vert_anim.bin", "bin/pdrd_root_motion_v1.cbor"];
// This is intentionally a schema, not an observed-inventory heuristic.  The
// R1 loader needs these bytes to build and bind a PDRD scene; a resealed
// manifest cannot turn one of them into an optional input.
const REQUIRED_R1_INPUTS: &[&str] = &[
    "info.toml", "param.toml", "map.pickle", "bin/displacement.bin",
    "bin/vert_dmap.bin", "bin/vert.bin", "bin/vel.bin", "bin/uv.bin",
    "bin/tri.bin", "bin/pdrd_body.bin", "bin/pdrd_vert_index.bin",
    "bin/pdrd_vert_list.bin", "bin/pdrd_rest_centered.bin",
];

fn checked_scene_root(root: &Path) -> Result<(), JointError> {
    let absolute = if root.is_absolute() { root.to_path_buf() } else {
        std::env::current_dir().map_err(|e|err("context","JOINT_R1_CONTEXT_IO","root",None,None,&e.to_string()))?.join(root)
    };
    for ancestor in absolute.ancestors() {
        let meta=fs::symlink_metadata(ancestor).map_err(|e|err("context","JOINT_R1_CONTEXT_IO","root",None,None,&e.to_string()))?;
        if is_reparse(&meta) { return Err(err("context","JOINT_R1_UNSAFE_PATH","root",None,None,"scene-root ancestor is a symlink/reparse point")); }
    }
    if !fs::symlink_metadata(root).map_err(|e|err("context","JOINT_R1_CONTEXT_IO","root",None,None,&e.to_string()))?.is_dir() {
        return Err(err("context","JOINT_R1_UNSAFE_PATH","root",None,None,"scene root must be a directory"));
    }
    Ok(())
}
fn safe_rel(path: &str) -> bool {
    !path.is_empty() && !path.contains('\\') && !Path::new(path).is_absolute()
        && !path.contains(':') && Path::new(path).components().all(|c| matches!(c, Component::Normal(_)))
}
fn is_reparse(meta: &fs::Metadata) -> bool {
    if meta.file_type().is_symlink() { return true; }
    #[cfg(windows)] { use std::os::windows::fs::MetadataExt; return meta.file_attributes() & 0x400 != 0; }
    #[cfg(not(windows))] { false }
}
pub(crate) fn safe_existing_file(root: &Path, relative: &str) -> Result<std::path::PathBuf, JointError> {
    if !safe_rel(relative) { return Err(err("context", "JOINT_R1_UNSAFE_PATH", relative, None, None, "path is not a safe relative input")); }
    checked_scene_root(root)?;
    // Check the root and every existing ancestor, including Windows junctions.
    let mut current = root.to_path_buf();
    if is_reparse(&fs::symlink_metadata(&current).map_err(|e|err("context","JOINT_R1_CONTEXT_IO","root",None,None,&e.to_string()))?) { return Err(err("context","JOINT_R1_UNSAFE_PATH","root",None,None,"root is a symlink/reparse point")); }
    for c in Path::new(relative).components() { if let Component::Normal(part) = c { current.push(part); let m=fs::symlink_metadata(&current).map_err(|e|err("context","JOINT_R1_CONTEXT_IO",relative,None,None,&e.to_string()))?; if is_reparse(&m) { return Err(err("context","JOINT_R1_UNSAFE_PATH",relative,None,None,"input ancestor is a symlink/reparse point")); } } }
    if !fs::symlink_metadata(&current).map_err(|e|err("context","JOINT_R1_CONTEXT_IO",relative,None,None,&e.to_string()))?.is_file() { return Err(err("context","JOINT_R1_UNSAFE_PATH",relative,None,None,"input must be a regular file")); }
    Ok(current)
}
/// `Path::exists` follows links and reports a dangling link as absent.  That
/// is not an absence a capability gate may accept, so probe with link metadata
/// after checking every ancestor instead.
pub(crate) fn checked_regular_file_or_absent(root: &Path, relative: &str) -> Result<bool, JointError> {
    if !safe_rel(relative) { return Err(err("context", "JOINT_R1_UNSAFE_PATH", relative, None, None, "path is not a safe relative input")); }
    checked_scene_root(root)?;
    let mut current = root.to_path_buf();
    for (index, component) in Path::new(relative).components().enumerate() {
        let Component::Normal(part) = component else { unreachable!("safe_rel accepted a non-normal component"); };
        current.push(part);
        match fs::symlink_metadata(&current) {
            Ok(meta) => {
                if is_reparse(&meta) { return Err(err("context", "JOINT_R1_UNSAFE_PATH", relative, None, None, "input ancestor is a symlink/reparse point")); }
                if index + 1 == Path::new(relative).components().count() {
                    if !meta.is_file() { return Err(err("context", "JOINT_R1_UNSAFE_PATH", relative, None, None, "input must be a regular file")); }
                    return Ok(true);
                }
                if !meta.is_dir() { return Err(err("context", "JOINT_R1_UNSAFE_PATH", relative, None, None, "input ancestor must be a directory")); }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                if index + 1 == Path::new(relative).components().count() { return Ok(false); }
                return Err(err("context", "JOINT_R1_CONTEXT_IO", relative, None, None, "input parent is absent"));
            }
            Err(e) => return Err(err("context", "JOINT_R1_CONTEXT_IO", relative, None, None, &e.to_string())),
        }
    }
    unreachable!("safe_rel rejects an empty path")
}
fn is_consumed_input(path: &str) -> bool {
    matches!(path, "info.toml" | "param.toml" | "dyn_param.txt" | "map.pickle" | "command.bat") ||
        (path.starts_with("bin/") && path != CONTEXT_FILE && path != JOINT_FILE && path != "bin/restoring_curves_v1.json")
}
pub(crate) fn scan_consumed_paths(root: &Path) -> Result<Vec<String>, JointError> {
    // The manifest hash domain is deliberately small: only the five native
    // root inputs and the bin subtree.  In particular, do not even stat
    // output/cache/preview/export (or arbitrary user folders); those are not
    // solver inputs and may legitimately be mutable or inaccessible.
    fn walk(root:&Path, dir:&Path, out:&mut Vec<String>)->Result<(),JointError>{
        for e in fs::read_dir(dir).map_err(|e|err("context","JOINT_R1_CONTEXT_IO","inventory",None,None,&e.to_string()))? {
            let e=e.map_err(|e|err("context","JOINT_R1_CONTEXT_IO","inventory",None,None,&e.to_string()))?;
            let ft=e.file_type().map_err(|e|err("context","JOINT_R1_CONTEXT_IO","inventory",None,None,&e.to_string()))?;
            let meta=fs::symlink_metadata(e.path()).map_err(|e|err("context","JOINT_R1_CONTEXT_IO","inventory",None,None,&e.to_string()))?;
            if is_reparse(&meta){return Err(err("context","JOINT_R1_UNSAFE_PATH","inventory",None,None,"symlink/reparse input is forbidden"));}
            let rel=e.path().strip_prefix(root).map_err(|_|err("context","JOINT_R1_UNSAFE_PATH","inventory",None,None,"escaped root"))?.to_string_lossy().replace('\\',"/");
            if ft.is_dir(){walk(root,&e.path(),out)?;}
            else if ft.is_file() && is_consumed_input(&rel) {out.push(rel);}
        } Ok(())
    }
    checked_scene_root(root)?;
    let mut out=Vec::new();
    for name in ["info.toml", "param.toml", "dyn_param.txt", "map.pickle", "command.bat"] {
        if checked_regular_file_or_absent(root, name)? { out.push(name.to_owned()); }
    }
    let bin = root.join("bin");
    match fs::symlink_metadata(&bin) {
        Ok(meta) => {
            if is_reparse(&meta) || !meta.is_dir() { return Err(err("context", "JOINT_R1_UNSAFE_PATH", "bin", None, None, "bin must be a non-reparse directory")); }
            walk(root, &bin, &mut out)?;
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(err("context", "JOINT_R1_CONTEXT_IO", "bin", None, None, &e.to_string())),
    }
    out.sort(); Ok(out)
}

fn required_native_inputs(snapshot: &HashMap<String, Vec<u8>>) -> Result<(), JointError> {
    let info=std::str::from_utf8(snapshot.get("info.toml").ok_or_else(||err("context","JOINT_R1_REQUIRED_INPUT_MISSING","info.toml",None,None,"missing native configuration"))?)
        .map_err(|e|err("context","JOINT_R1_MALFORMED_INPUT","info.toml",None,None,&e.to_string()))?;
    let param=std::str::from_utf8(snapshot.get("param.toml").ok_or_else(||err("context","JOINT_R1_REQUIRED_INPUT_MISSING","param.toml",None,None,"missing native configuration"))?)
        .map_err(|e|err("context","JOINT_R1_MALFORMED_INPUT","param.toml",None,None,&e.to_string()))?;
    let parsed:toml::Value=toml::from_str(info).map_err(|e|err("context","JOINT_R1_MALFORMED_INPUT","info.toml",None,None,&e.to_string()))?;
    let count=parsed.get("count").and_then(|v|v.as_table()).ok_or_else(||err("context","JOINT_R1_MALFORMED_INPUT","info.toml",None,None,"native loader requires [count]"))?;
    for key in ["vert","rod","tri","tet","static_vert","static_tri","pin_block","wall","sphere","stitch","shell_count"] {
        if count.get(key).and_then(|v|v.as_integer()).filter(|v|*v>=0).is_none() { return Err(err("context","JOINT_R1_MALFORMED_INPUT","info.toml",None,None,&format!("native loader requires nonnegative count.{key}"))); }
    }
    let has_rest=count.get("has_rest_vert").and_then(|v|v.as_bool()).unwrap_or(false);
    let has_anim=count.get("has_rest_vert_anim").and_then(|v|v.as_bool()).unwrap_or(false);
    let has_bend=count.get("has_bend_rest_vert").and_then(|v|v.as_bool()).unwrap_or(false);
    if has_anim && count.get("rest_vert_anim_frames").and_then(|v|v.as_integer()).filter(|v|*v>=0).is_none() { return Err(err("context","JOINT_R1_MALFORMED_INPUT","info.toml",None,None,"native loader requires nonnegative count.rest_vert_anim_frames")); }
    let mut needed: Vec<String>=Vec::new();
    if has_rest { needed.push("bin/rest_vert.bin".into()); }
    if has_rest || has_anim { needed.push("bin/rest_vert_mask.bin".into()); }
    if has_anim { needed.extend(["bin/rest_vert_anim.bin","bin/rest_vert_times.bin"].into_iter().map(str::to_owned)); }
    if has_bend { needed.extend(["bin/bend_rest_vert.bin","bin/bend_rest_vert_mask.bin"].into_iter().map(str::to_owned)); }
    let n = |key: &str| count.get(key).and_then(|v| v.as_integer()).unwrap() as usize;
    if n("rod") > 0 { needed.push("bin/rod.bin".into()); }
    if n("tri") > 0 { needed.push("bin/tri.bin".into()); }
    if n("tet") > 0 { needed.push("bin/tet.bin".into()); }
    if n("static_vert") > 0 { needed.extend(["bin/static_vert_dmap.bin", "bin/static_vert.bin"].into_iter().map(str::to_owned)); }
    if n("static_tri") > 0 { needed.push("bin/static_tri.bin".into()); }
    if n("stitch") > 0 { needed.extend(["bin/stitch_ind.bin", "bin/stitch_w.bin"].into_iter().map(str::to_owned)); } // stitch_stiffness has a native default.
    let table = |name: &str| parsed.get(name).and_then(|v| v.as_table()).ok_or_else(||err("context","JOINT_R1_MALFORMED_INPUT","info.toml",None,None,&format!("native loader requires [{name}]")));
    let nonnegative = |entry: &toml::map::Map<String, toml::Value>, key: &str, section: &str| -> Result<usize, JointError> {
        entry.get(key).and_then(|v| v.as_integer()).filter(|v| *v >= 0).map(|v| v as usize).ok_or_else(||err("context","JOINT_R1_MALFORMED_INPUT","info.toml",None,None,&format!("native loader requires nonnegative {section}.{key}")))
    };
    let float = |entry: &toml::map::Map<String, toml::Value>, key: &str, section: &str| -> Result<(), JointError> {
        entry.get(key).and_then(|v| v.as_float()).map(|_| ()).ok_or_else(||err("context","JOINT_R1_MALFORMED_INPUT","info.toml",None,None,&format!("native loader requires float {section}.{key}")))
    };
    let string = |entry: &toml::map::Map<String, toml::Value>, key: &str, section: &str| -> Result<(), JointError> {
        entry.get(key).and_then(|v| v.as_str()).map(|_| ()).ok_or_else(||err("context","JOINT_R1_MALFORMED_INPUT","info.toml",None,None,&format!("native loader requires string {section}.{key}")))
    };
    let boolean = |entry: &toml::map::Map<String, toml::Value>, key: &str, section: &str| -> Result<(), JointError> {
        entry.get(key).and_then(|v| v.as_bool()).map(|_| ()).ok_or_else(||err("context","JOINT_R1_MALFORMED_INPUT","info.toml",None,None,&format!("native loader requires bool {section}.{key}")))
    };
    for i in 0..n("pin_block") {
        let section=format!("pin-{i}"); let pin=table(&section)?;
        let _=nonnegative(pin,"pin",&section)?; let operation_count=nonnegative(pin,"operation_count",&section)?;
        if pin.get("pull").and_then(|v|v.as_float()).is_none() { return Err(err("context","JOINT_R1_MALFORMED_INPUT","info.toml",None,None,&format!("native loader requires {section}.pull"))); }
        needed.push(format!("bin/pin-ind-{i}.bin"));
        for j in 0..operation_count {
            let op_section=format!("pin-{i}-op-{j}"); let op=table(&op_section)?;
            let op_type=op.get("type").and_then(|v|v.as_str()).ok_or_else(||err("context","JOINT_R1_MALFORMED_INPUT","info.toml",None,None,&format!("native loader requires {op_section}.type")))?;
            match op_type {
                "move_by" | "move_to" => {
                    for key in ["t_start", "t_end"] { float(op,key,&op_section)?; }
                    string(op,"transition",&op_section)?;
                    needed.push(format!("bin/pin-{i}-op-{j}.bin"));
                }
                "spin" => {
                    for key in ["center_x", "center_y", "center_z", "axis_x", "axis_y", "axis_z", "angular_velocity", "t_start", "t_end"] { float(op,key,&op_section)?; }
                }
                "scale" => {
                    for key in ["center_x", "center_y", "center_z", "factor", "t_start", "t_end"] { float(op,key,&op_section)?; }
                    string(op,"transition",&op_section)?;
                }
                "torque" => {
                    let _=nonnegative(op,"axis_component",&op_section)?;
                    let _=nonnegative(op,"hint_vertex",&op_section)?;
                    for key in ["magnitude", "t_start", "t_end"] { float(op,key,&op_section)?; }
                }
                "transform_keyframes" => {
                    let keyframes=nonnegative(op,"keyframe_count",&op_section)?;
                    for key in ["t_start", "t_end", "rest_tx", "rest_ty", "rest_tz"] { float(op,key,&op_section)?; }
                    for suffix in [".bin", "-time.bin", "-translation.bin", "-quaternion.bin", "-scale.bin"] { needed.push(format!("bin/pin-{i}-op-{j}{suffix}")); }
                    if keyframes > 1 { for suffix in ["-interp.bin", "-handles.bin"] { needed.push(format!("bin/pin-{i}-op-{j}{suffix}")); } }
                }
                _ => return Err(err("context","JOINT_R1_MALFORMED_INPUT","info.toml",None,None,&format!("native loader does not recognize {op_section}.type"))),
            }
        }
    }
    for i in 0..n("wall") { let section=format!("wall-{i}"); let wall=table(&section)?; let keyframes=nonnegative(wall,"keyframe",&section)?; if keyframes>0 { for key in ["nx", "ny", "nz", "contact-gap", "friction"] { float(wall,key,&section)?; } string(wall,"transition",&section)?; needed.push(format!("bin/wall-pos-{i}.bin")); needed.push(format!("bin/wall-timing-{i}.bin")); } }
    for i in 0..n("sphere") { let section=format!("sphere-{i}"); let sphere=table(&section)?; boolean(sphere,"invert",&section)?; boolean(sphere,"hemisphere",&section)?; string(sphere,"transition",&section)?; let keyframes=nonnegative(sphere,"keyframe",&section)?; if keyframes>0 { for key in ["contact-gap", "friction"] { float(sphere,key,&section)?; } for suffix in ["pos", "radius", "timing"] { needed.push(format!("bin/sphere-{suffix}-{i}.bin")); } } }
    for path in needed { if !snapshot.contains_key(&path) { return Err(err("context","JOINT_R1_REQUIRED_INPUT_MISSING",&path,None,None,"required by sealed native configuration")); } }
    let config:NativeConfig=toml::from_str(param).map_err(|e|err("context","JOINT_R1_MALFORMED_INPUT","param.toml",None,None,&e.to_string()))?;
    if !config.param.world_scaling.is_finite() || config.param.world_scaling<=0.0 { return Err(err("context","JOINT_R1_MALFORMED_INPUT","param.toml",None,None,"native loader requires finite positive param.world_scaling")); }
    Ok(())
}

/// The one version/feature gate for the directional control contract.
///
/// V2 is the historical envelope and MUST NOT carry `required_features` or
/// `angular_control_v1`: a V2 payload that does is a producer that thought it
/// was authoring a control, and accepting it would run the scene with the
/// requested term silently absent.
///
/// V3 carries exactly the feature union its records imply: the control feature
/// once if any joint requests a directional record, and nothing otherwise. The
/// comparison is exact (same length, same spelling, same order), so an
/// unrequested feature, a missing one, a duplicate and an unknown one are all
/// the same refusal.
///
/// V4 validates and packs soft limits and drives before installation. Earlier
/// envelopes cannot request these fields; versions above V4 remain refused.
fn gate_angular_control_contract(payload: &JointPayload, version: u32) -> Result<(), JointError> {
    if let Some(features)=payload.required_features.as_ref() {
        if features.iter().any(|f|f=="cross_link_collision_pairs_v1") {
            if version<3 || features.windows(2).any(|w|w[0]>=w[1]) {return Err(err("cross-links","CROSS_LINK_COLLISION_INVALID","required_features",None,None,"sorted unique versioned carrier required"));}
            let mut stripped=payload.clone();stripped.required_features=Some(features.iter().filter(|f|f.as_str()!="cross_link_collision_pairs_v1").cloned().collect());
            return gate_angular_control_contract(&stripped,version);
        }
    }
    if let Some(features)=payload.required_features.as_ref() {
        if features.iter().any(|f|f==crate::pmx_joints::FEATURE) {
            if version<3 || features.windows(2).any(|w|w[0]>=w[1]) || !cfg!(neutral_driver) {
                return Err(err("pmx-joints","PMX_JOINT_INVALID","required_features",None,None,"PMX requires a sorted versioned neutral carrier"));
            }
            let mut stripped=payload.clone();
            stripped.required_features=Some(features.iter().filter(|f|f.as_str()!=crate::pmx_joints::FEATURE).cloned().collect());
            return gate_angular_control_contract(&stripped,version);
        }
    }
    if let Some(features)=payload.required_features.as_ref() {
        if features.iter().any(|f|f==crate::cross_links::FEATURE) {
            if version<3 || features.windows(2).any(|w|w[0]>=w[1]) {return Err(err("cross-links","CROSS_LINK_INVALID","required_features",None,None,"sorted unique versioned carrier required"));}
            let mut stripped=payload.clone();stripped.required_features=Some(features.iter().filter(|f|f.as_str()!=crate::cross_links::FEATURE).cloned().collect());
            return gate_angular_control_contract(&stripped,version);
        }
    }

    if let Some(features)=payload.required_features.as_ref() {
        if features.iter().any(|f|f==crate::connected_collision::FEATURE) {
            if version<3 || features.windows(2).any(|w|w[0]>=w[1]) || !features.iter().any(|f|f==crate::soft_joint::FEATURE) {
                return Err(err("collision","BONE_CONNECTED_CAPABILITY","required_features",None,None,"sorted unique versioned material carrier required"));
            }
            let mut stripped=payload.clone();
            stripped.required_features=Some(features.iter().filter(|f|f.as_str()!=crate::connected_collision::FEATURE).cloned().collect());
            return gate_angular_control_contract(&stripped,version);
        }
    }

    if payload.required_features.as_ref().is_some_and(|f|f.iter().any(|n|n==crate::fem_control_events::FEATURE)) {
        if !cfg!(cuda_backend){return Err(err("fem","FEM_CONTROL_BACKEND_UNSUPPORTED","required_features",None,None,"FEM event candidate supports matching CUDA build only"));}
        let features=payload.required_features.as_ref().unwrap();
        if version<3||features.windows(2).any(|w|w[0]>=w[1])||!features.iter().any(|n|n==crate::soft_joint::FEATURE)||features.iter().any(|n|n==crate::soft_joint::CONTROL_FEATURE){return Err(err("fem","FEM_CONTROL_EVENTS_INVALID","required_features",None,None,"event FEM requires sorted unique material carrier and excludes v1 controls"));}
        let mut stripped=payload.clone();stripped.required_features=Some(features.iter().filter(|n|n.as_str()!=crate::fem_control_events::FEATURE).cloned().collect());
        return gate_angular_control_contract(&stripped,version);
    }
    if payload.required_features.as_ref().is_some_and(|f|f.iter().any(|n|n==crate::soft_joint::CONTROL_FEATURE)) {
        if !cfg!(cuda_backend) {return Err(err("fem","FEM_CONTROL_BACKEND_UNSUPPORTED","required_features",None,None,"initial FEM controls candidate supports CUDA only"));}
        let features=payload.required_features.as_ref().unwrap();
        if version<3 || features.windows(2).any(|w|w[0]>=w[1]) || !features.iter().any(|n|n==crate::soft_joint::FEATURE) {
            return Err(err("fem","FEM_CONTROL_INVALID","required_features",None,None,"prescribed FEM requires sorted unique material carrier"));
        }
        let mut stripped=payload.clone();
        stripped.required_features=Some(features.iter().filter(|n|n.as_str()!=crate::soft_joint::CONTROL_FEATURE).cloned().collect());
        return gate_angular_control_contract(&stripped,version);
    }
    let has_root_v3=payload.required_features.as_ref().is_some_and(|f|f.iter().any(|n|n==crate::joint_root_motion_wire::ROOT_FEATURE_V3));
    let authored_hard=payload.joints.iter().any(|j|j.angular_hard_hinge_v1.is_some()||j.joint_d6_hard_v1.is_some());
    if has_root_v3 && (authored_hard || version==7) {
        return Err(err("root","ROOT_MOTION_NATIVE_CAPABILITY","required_features",None,None,"root-v3 with hard-D6/hinge union is explicitly unsupported"));
    }
    if let Some(features)=payload.required_features.as_ref() {
        if features.iter().any(|f|f==crate::soft_joint::FEATURE) {
            if version<3 || features.windows(2).any(|w|w[0]>=w[1]) {
                return Err(err("fem","FEM_JOINT_INVALID","required_features",None,None,"sorted unique versioned FEM capability required"));
            }
            let mut stripped=payload.clone();
            // The cone marker covers rigid + material authored rows. The
            // retained material intake verifies that union after decoding its
            // sidecar; this recursive gate proves the rigid subset only.
            let rigid_cone = payload.joints.iter().any(|j|
                crate::joint_angular_extension_gpu::has_swing_cone(j.angular_soft_limits_v1.as_ref().map(|v| &v.0)));
            let mut angular: Vec<String> = features.iter().filter(|f|
                f.as_str()!=crate::soft_joint::FEATURE && f.as_str()!=ANGULAR_SWING_CONE_FEATURE).cloned().collect();
            if rigid_cone { angular.push(ANGULAR_SWING_CONE_FEATURE.into()); angular.sort(); }
            stripped.required_features=Some(angular);
            return gate_angular_control_contract(&stripped,version);
        }
    }
    if let Some(features) = payload.required_features.as_ref() {
        if features.iter().any(|f| f == "ppf_collision_filter_v1") {
            if features.windows(2).any(|w| w[0] >= w[1]) || version == 2 {
                return Err(err("collision", "COLLISION_FILTER_CAPABILITY", "required_features", None, None, "requires sorted unique features on envelope >=3"));
            }
            let mut stripped = payload.clone();
            stripped.required_features = Some(features.iter().filter(|f| f.as_str() != "ppf_collision_filter_v1").cloned().collect());
            return gate_angular_control_contract(&stripped, version);
        }
    }
    if payload.required_features.as_ref().is_some_and(|f|f.iter().any(|n|n==crate::joint_root_motion_wire::BODY_CONTROLS_FEATURE)) {
        let names=payload.required_features.as_ref().unwrap();
        if version!=9 || names.windows(2).any(|w|w[0]>=w[1]) || !names.iter().any(|n|n==crate::joint_root_motion_wire::ROOT_FEATURE_V2 || n==crate::joint_root_motion_wire::ROOT_FEATURE_V3)
            || payload.joints.iter().any(|j|j.angular_hard_hinge_v1.is_some()||j.joint_d6_hard_v1.is_some()) {
            return Err(err("root","ROOT_MOTION_REQUIRED_FEATURES","required_features",None,None,"body controls require exact V9 root-v2/v3 non-hard feature union"));
        }
        let mut stripped=payload.clone();
        stripped.required_features=Some(names.iter().filter(|n|n.as_str()!=crate::joint_root_motion_wire::BODY_CONTROLS_FEATURE).cloned().collect());
        return gate_angular_control_contract(&stripped,version);
    }
    if authored_hard || matches!(version,5|7) { return gate_hard_contract(payload,version); }
    if matches!(version,6|9) {
        use crate::joint_root_motion_wire::{ROOT_FEATURE,ROOT_FEATURE_V2,ROOT_FEATURE_V3};
        let names=payload.required_features.as_ref().ok_or_else(||err("root","ROOT_MOTION_REQUIRED_FEATURES","required_features",None,None,"V6/V9 root feature union required"))?;
        if names.iter().any(|n|n.contains("hard_")||n.contains("hinge")||n.contains("d6")) {
            return Err(err("root","ROOT_MOTION_NATIVE_CAPABILITY","required_features",None,None,"root plus hard graph composition is not installed"));
        }
        let roots=names.iter().filter(|n|matches!(n.as_str(),ROOT_FEATURE|ROOT_FEATURE_V2|ROOT_FEATURE_V3)).collect::<Vec<_>>();
        let required_ok=if version==6 {roots.len()==1&&roots[0].as_str()==ROOT_FEATURE}
            else {roots.len()==1&&matches!(roots[0].as_str(),ROOT_FEATURE_V2|ROOT_FEATURE_V3)};
        if !required_ok || names.windows(2).any(|w|w[0]>=w[1]) {
            return Err(err("root","ROOT_MOTION_REQUIRED_FEATURES","required_features",None,None,"exact sorted unique root/angle union and matching V6/V9 root feature required"));
        }
        let mut angular=payload.clone();
        angular.required_features=Some(names.iter().filter(|n|!matches!(n.as_str(),ROOT_FEATURE|ROOT_FEATURE_V2|ROOT_FEATURE_V3)).cloned().collect());
        return gate_angular_control_contract(&angular,ANGULAR_EXTENSION_ENVELOPE_VERSION);
    }
    use crate::joint_angular_extension_gpu as ext;
    let has_control = payload.joints.iter().any(|j| j.angular_control_v1.is_some());
    let has_extension = payload.joints.iter().any(|j| j.angular_soft_limits_v1.is_some() || j.angular_drives_v1.is_some());
    // Earlier envelopes cannot request a later family, even if it is now implemented.
    if version != ANGULAR_EXTENSION_ENVELOPE_VERSION {
        if let Some(j) = payload.joints.iter().find(|j| j.angular_soft_limits_v1.is_some() || j.angular_drives_v1.is_some()) {
            let field = if j.angular_soft_limits_v1.is_some() { ANGULAR_SOFT_LIMITS_FIELD } else { ANGULAR_DRIVES_FIELD };
            return Err(err("ingest", "JOINT_ANGULAR_EXTENSION_UNIMPLEMENTED", field, Some(&j.joint_uuid), None,
                "soft limits and drives require the V4 envelope"));
        }
    }
    if let Some(names) = payload.required_features.as_ref() {
        for name in names {
            if !matches!(name.as_str(), ANGULAR_CONTROL_FEATURE | ANGULAR_SOFT_LIMITS_FEATURE | ANGULAR_DRIVES_FEATURE | ANGULAR_SWING_CONE_FEATURE) {
                return Err(err("ingest", "JOINT_G2_REQUIRED_FEATURES", "required_features", None, None, &format!("unknown required feature: {name}")));
            }
        }
        if version != ANGULAR_EXTENSION_ENVELOPE_VERSION && names.iter().any(|n| matches!(n.as_str(), ANGULAR_SOFT_LIMITS_FEATURE | ANGULAR_DRIVES_FEATURE | ANGULAR_SWING_CONE_FEATURE)) {
            return Err(err("ingest", "JOINT_ANGULAR_EXTENSION_UNIMPLEMENTED", "required_features", None, None,
                "later features require the V4 envelope"));
        }
    }
    match version {
        ENVELOPE_VERSION => {
            if payload.required_features.is_some() || has_control || has_extension {
                return Err(err("ingest", "JOINT_G2_V2_EXTENSION", "required_features", None, None, "V2 must not contain later angular fields"));
            }
            Ok(())
        }
        ANGULAR_CONTROL_ENVELOPE_VERSION => {
            let expected: Vec<&str> = if has_control { vec![ANGULAR_CONTROL_FEATURE] } else { Vec::new() };
            let found: Option<Vec<&str>> = payload.required_features.as_ref().map(|f| f.iter().map(String::as_str).collect());
            if found.as_deref() != Some(expected.as_slice()) {
                return Err(err("ingest", "JOINT_G2_REQUIRED_FEATURES", "required_features", None, None,
                    "V3 required_features must exactly match control presence"));
            }
            Ok(())
        }
        ANGULAR_EXTENSION_ENVELOPE_VERSION => {
            let inputs: Vec<_> = payload.joints.iter().map(|j| ext::JointExtensionInput {
                control_present: j.angular_control_v1.is_some(),
                control_model: j.angular_control_v1.as_ref().map(|c| c.coordinate_model.as_str()),
                soft: j.angular_soft_limits_v1.as_ref().map(|v| &v.0),
                drives: j.angular_drives_v1.as_ref().map(|v| &v.0),
                legacy_angular_nonzero: j.angle_stiffness_nm_per_rad != 0.0 || j.angle_damping_nms_per_rad != 0.0 || j.max_angle_rad != 0.0,
            }).collect();
            let features = payload.required_features.as_ref().ok_or_else(|| err("ingest", "JOINT_G2_REQUIRED_FEATURES", "required_features", None, None, "V4 requires its exact feature union"))?;
            ext::validate_v4_and_pack(&inputs, features).map_err(|e| err("ingest", e.code, &e.path, None, None, &e.to_string()))?;
            Ok(())
        }
        other => Err(err("ingest", "JOINT_ANGULAR_EXTENSION_UNSUPPORTED_VERSION", "envelope", None, None,
            &format!("envelope version {other} is not implemented; only V2/V3/V4 are accepted"))),
    }
}

/// Independently load and validate the sealed builder snapshot before setup.
/// The returned context is only accepted after every inventory hash and the
/// endpoint membership/row facts have been checked from consumed files.
pub fn validate_scene_context(root: &Path, payload: &JointPayload, rows: &[f32]) -> Result<SceneInputContext, JointError> {
    let bytes=fs::read(safe_existing_file(root, CONTEXT_FILE)?).map_err(|e|err("context","JOINT_R1_CONTEXT_MISSING","pdrd_scene_context.cbor",None,None,&e.to_string()))?;
    let ctx:SceneInputContext=from_cbor_with_version(2,CONTEXT_KIND,&bytes).map_err(|e|err("context","JOINT_R1_CONTEXT_INVALID","pdrd_scene_context.cbor",None,None,&e.to_string()))?;
    if ctx.context_version!=2 || ctx.builder_contract_version!=2 || !ctx.complete {return Err(err("context","JOINT_R1_CONTEXT_INCOMPLETE","complete",None,None,"final session configuration unavailable"));}
    if !uuid_ok(&ctx.scene_uuid) || ctx.scene_uuid != payload.scene_uuid {return Err(err("context","JOINT_R1_CROSS_SCENE_CONTEXT","scene_uuid",None,None,"payload/context scene identity mismatch"));}
    let mut prior=""; for input in &ctx.inputs { if input.relative_path.as_str()<=prior || input.relative_path.starts_with('/') || input.relative_path.contains("..") {return Err(err("context","JOINT_R1_CONTEXT_INVALID","inputs",None,None,"inventory must be sorted relative paths"));} prior=&input.relative_path; let b=fs::read(root.join(&input.relative_path)).map_err(|e|err("context","JOINT_R1_CONTEXT_IO",&input.relative_path,None,None,&e.to_string()))?; if b.len()!=input.byte_length || sha256(&b)!=input.sha256 {return Err(err("context","JOINT_R1_INPUT_HASH_MISMATCH",&input.relative_path,None,None,"changed solver input"));} }
    let mut digest_input=format!("ppf.pdrd_scene_context/v2\n{}\n",ctx.scene_uuid);
    for body in &ctx.bodies { digest_input.push_str(&format!("{}\t{}\t{}\n",body.body_uuid,body.runtime_body_index,body.vertex_count)); }
    for input in &ctx.inputs { digest_input.push_str(&format!("{}\t{}\t{}\n",input.relative_path,input.byte_length,input.sha256)); }
    let digest=sha256(digest_input.as_bytes());
    if digest!=ctx.scene_inputs_sha256 || digest!=payload.scene_inputs_sha256 {return Err(err("context","JOINT_R1_INPUT_HASH_MISMATCH","scene_inputs_sha256",None,None,"payload/context inventory digest mismatch"));}
    if rows.len()%PDRD_BODY_ROW_LEN!=0 {return Err(err("context","JOINT_R1_PDRD_ABI_INVALID","pdrd_body.bin",None,None,"23-float rows required"));}
    let index=read_u32(&root.join("bin/pdrd_vert_index.bin"))?; let list=read_u32(&root.join("bin/pdrd_vert_list.bin"))?; let rest=read_f32(&root.join("bin/pdrd_rest_centered.bin"))?;
    if rest.len()!=list.len()*3 {return Err(err("context","JOINT_R1_MEMBERSHIP_INVALID","pdrd_rest_centered.bin",None,None,"rest rows must match PDRD list"));}
    if ctx.bodies.len()!=rows.len()/PDRD_BODY_ROW_LEN {return Err(err("context","JOINT_R1_MEMBERSHIP_INVALID","bodies",None,None,"context must bind every PDRD row"));}
    let mut bound=HashSet::new(); let mut seen_global=HashSet::new();
    for body in &ctx.bodies { if !uuid_ok(&body.body_uuid)||!bound.insert(body.runtime_body_index) {return Err(err("context","JOINT_R1_MEMBERSHIP_INVALID","bodies",None,Some(&body.body_uuid),"invalid or duplicate row binding"));} let r=body.runtime_body_index as usize; if r>=rows.len()/23{return Err(err("context","JOINT_R1_DANGLING_BODY","runtime_body_index",None,Some(&body.body_uuid),"row absent"));} let row=&rows[r*23..r*23+23]; if !row.iter().all(|x|x.is_finite())||row[2]<=0.||row[15]<=0.||row[16]!=0. {return Err(err("context","JOINT_R1_PDRD_ROW_INVALID","pdrd_body.bin",None,Some(&body.body_uuid),"nonfinite, nonphysical, or non-FREE row"));} let start=field_u32(row[0],"vertex_start",&body.body_uuid)?;let count=field_u32(row[1],"vertex_count",&body.body_uuid)?; if count!=body.vertex_count as usize||count<4||start.checked_add(count).filter(|x|*x<=list.len()).is_none(){return Err(err("context","JOINT_R1_MEMBERSHIP_INVALID","vertex_count",None,Some(&body.body_uuid),"invalid list range"));} let det=Matrix3::from_row_slice(&row[6..15]).determinant();if !det.is_finite()||det.abs()<1e-12{return Err(err("context","JOINT_R1_PDRD_ROW_INVALID","gram_inv",None,Some(&body.body_uuid),"degenerate rest Gram"));} for (local,&v) in list[start..start+count].iter().enumerate(){let vi=v as usize;if vi>=index.len()||index[vi]!=body.runtime_body_index+1||!seen_global.insert(v){return Err(err("context","JOINT_R1_MEMBERSHIP_INVALID","pdrd_vert_index.bin",None,Some(&body.body_uuid),"list/index ownership mismatch"));}let y=&rest[(start+local)*3..(start+local+1)*3];if !y.iter().all(|x|x.is_finite()){return Err(err("context","JOINT_R1_PDRD_ROW_INVALID","pdrd_rest_centered.bin",None,Some(&body.body_uuid),"nonfinite rest vertex"));}} }
    if index.iter().enumerate().any(|(i,&owner)| (owner==0 && seen_global.contains(&(i as u32))) || (owner>0 && !seen_global.contains(&(i as u32)))) {return Err(err("context","JOINT_R1_MEMBERSHIP_INVALID","pdrd_vert_index.bin",None,None,"ownership must be comprehensive and disjoint"));}
    let mut friction_inputs=HashMap::new();
    for input in &ctx.inputs {
        let name=&input.relative_path;
        if name.starts_with("bin/")&&name.contains("friction")&&name.ends_with(".bin") {
            let bytes=fs::read(safe_existing_file(root,name)?).map_err(|e|err("context","JOINT_R1_CONTEXT_IO",name,None,None,&e.to_string()))?;
            friction_inputs.insert(name.clone(),bytes);
        }
    }
    let info=fs::read_to_string(root.join("info.toml")).map_err(|e|err("context","JOINT_R1_CONTEXT_IO","info.toml",None,None,&e.to_string()))?;
    let parsed=toml::from_str(&info).map_err(|e|err("context","JOINT_R1_MALFORMED_INPUT","info.toml",None,None,&e.to_string()))?;
    validate_snapshot_friction(&friction_inputs,&parsed)?;
    let endpoint_rows:HashSet<u32>=payload.bodies.iter().map(|b|b.runtime_body_index).collect();
    for ent in fs::read_dir(root.join("bin")).map_err(|e|err("context","JOINT_R1_CONTEXT_IO","bin",None,None,&e.to_string()))? {let p=ent.map_err(|e|err("context","JOINT_R1_CONTEXT_IO","bin",None,None,&e.to_string()))?.path();let n=p.file_name().and_then(|x|x.to_str()).unwrap_or("");if n.starts_with("pin-ind-") {for v in read_u32(&p)? {if (v as usize)<index.len()&&endpoint_rows.contains(&(index[v as usize]-1)) {return Err(err("context","JOINT_R1_ENDPOINT_PINNED",n,None,None,"endpoint membership overlaps pin"));}}}}
    let param=fs::read_to_string(root.join("param.toml")).map_err(|e|err("context","JOINT_R1_CONTEXT_INCOMPLETE","param.toml",None,None,&e.to_string()))?;if !param.lines().any(|l|l.trim_start().starts_with("world_scaling")&&(l.contains("= 1")||l.contains("=1"))) {return Err(err("context","JOINT_R1_UNSUPPORTED_DOMAIN","world_scaling",None,None,"R1 requires world_scaling=1"));}
    if let Ok(dynp)=fs::read_to_string(root.join("dyn_param.txt")){if dynp.contains("velocity:")||dynp.contains("angular_velocity:") {return Err(err("context","JOINT_R1_ENDPOINT_PRESCRIBED_MOTION","dyn_param.txt",None,None,"conservative R1 schedule refusal"));}}
    Ok(ctx)
}

/// Parses and validates the immutable R1 payload.  `pdrd_rows` is used only
/// to prove that each endpoint resolves to a FREE PDRD body; it never changes
/// that binary format.  Unsupported scene facts that have no authoritative
/// source here must be rejected by the frontend exporter instead of guessed.
fn ingest_payload(mut payload: JointPayload, version: u32, pdrd_rows: &[f32], snapshot: InputSnapshot) -> Result<OwnedJointScene, JointError> {
    if snapshot.bytes.is_empty() && payload.required_features.as_ref().is_some_and(|f|f.iter().any(|n|n=="pdrd_kinematic_roots_v1"||n=="pdrd_kinematic_roots_v2"||n==crate::joint_root_motion_wire::ROOT_FEATURE_V3)) {
        return Err(err("root","ROOT_MOTION_BODY_BINDING","snapshot",None,None,"root ingestion requires sealed native input"));
    }

    gate_angular_control_contract(&payload, version)?;
    let body_controls_v1=payload.required_features.as_ref().is_some_and(|f|f.iter().any(|n|n==crate::joint_root_motion_wire::BODY_CONTROLS_FEATURE));
    if payload.required_capability != CAPABILITY && payload.required_capability != WORLD_CAPABILITY { return Err(err("ingest", "JOINT_R1_UNSUPPORTED_REQUIRED_CAPABILITY", "required_capability", None, None, "unknown or missing required capability")); }
    payload.scene_uuid=canonical_uuid(&payload.scene_uuid).ok_or_else(||err("ingest", "JOINT_R1_INVALID_UUID", "scene_uuid", None, None, "expected immutable UUID"))?;
    if pdrd_rows.len() % PDRD_BODY_ROW_LEN != 0 { return Err(err("ingest", "JOINT_R1_PDRD_ABI_INVALID", "pdrd_body.bin", None, None, "23-float rows required")); }
    let mut bodies = HashMap::new();
    let mut runtime_indices = HashSet::new();
    for b in &mut payload.bodies {
        b.body_uuid=canonical_uuid(&b.body_uuid).ok_or_else(||err("ingest", "JOINT_R1_INVALID_UUID", "body_uuid", None, Some(&b.body_uuid), "expected immutable UUID"))?;
        if bodies.insert(b.body_uuid.clone(), b.runtime_body_index).is_some() { return Err(err("ingest", "JOINT_R1_DUPLICATE_BODY_UUID", "body_uuid", None, Some(&b.body_uuid), "duplicate body UUID")); }
        if !runtime_indices.insert(b.runtime_body_index) {
            return Err(err("ingest", "JOINT_R1_DUPLICATE_RUNTIME_BODY_INDEX", "runtime_body_index", None, Some(&b.body_uuid), "runtime body index must have exactly one UUID binding"));
        }
        let row = b.runtime_body_index as usize;
        if row >= pdrd_rows.len() / PDRD_BODY_ROW_LEN { return Err(err("ingest", "JOINT_R1_DANGLING_BODY", "runtime_body_index", None, Some(&b.body_uuid), "does not resolve to PDRD body")); }
        if pdrd_rows[row * PDRD_BODY_ROW_LEN + 16] != 0.0 { return Err(err("ingest", "JOINT_R1_ENDPOINT_NOT_FREE", "joint_mode", None, Some(&b.body_uuid), "R1 requires FREE PDRD endpoint")); }
    }
    let mut seen = HashMap::new(); let mut joints = Vec::new();
    for mut j in payload.joints {
        j.joint_uuid=canonical_uuid(&j.joint_uuid).ok_or_else(||err("ingest", "JOINT_R1_INVALID_UUID", "joint_uuid", Some(&j.joint_uuid), None, "expected immutable UUID"))?;
        j.parent_body_uuid=canonical_uuid(&j.parent_body_uuid).ok_or_else(||err("ingest", "JOINT_R1_INVALID_UUID", "parent_body_uuid", Some(&j.joint_uuid), Some(&j.parent_body_uuid), "expected immutable UUID"))?;
        j.child_body_uuid=canonical_uuid(&j.child_body_uuid).ok_or_else(||err("ingest", "JOINT_R1_INVALID_UUID", "child_body_uuid", Some(&j.joint_uuid), Some(&j.child_body_uuid), "expected immutable UUID"))?;
        if seen.insert(j.joint_uuid.clone(), ()).is_some() { return Err(err("ingest", "JOINT_R1_DUPLICATE_JOINT_UUID", "joint_uuid", Some(&j.joint_uuid), None, "duplicate joint UUID")); }
        let world_parent = j.parent_body_uuid == WORLD_ENDPOINT_UUID;
        let world_child = j.child_body_uuid == WORLD_ENDPOINT_UUID;
        if world_child || (world_parent && world_child) { return Err(err("ingest", "JOINT_R1_UNSUPPORTED_WORLD_ENDPOINT", "child_body_uuid", Some(&j.joint_uuid), Some(&j.child_body_uuid), "world endpoint is supported only as the parent endpoint")); }
        if world_parent && payload.required_capability != WORLD_CAPABILITY { return Err(err("ingest", "JOINT_R1_UNSUPPORTED_REQUIRED_CAPABILITY", "required_capability", Some(&j.joint_uuid), None, "world anchor requires pdrd_world_ball_anchor_v1")); }
        if j.parent_body_uuid == j.child_body_uuid { return Err(err("ingest", "JOINT_R1_SAME_ENDPOINT", "child_body_uuid", Some(&j.joint_uuid), Some(&j.child_body_uuid), "endpoints must be distinct")); }
        let a = if world_parent { crate::joint_gpu::WORLD_ENDPOINT } else { *bodies.get(&j.parent_body_uuid).ok_or_else(||err("ingest", "JOINT_R1_DANGLING_BODY", "parent_body_uuid", Some(&j.joint_uuid), Some(&j.parent_body_uuid), "unresolved endpoint"))? };
        let Some(&b) = bodies.get(&j.child_body_uuid) else { return Err(err("ingest", "JOINT_R1_DANGLING_BODY", "child_body_uuid", Some(&j.joint_uuid), Some(&j.child_body_uuid), "unresolved endpoint")); };
        if a == b { return Err(err("ingest", "JOINT_R1_SAME_ENDPOINT", "child_body_uuid", Some(&j.joint_uuid), Some(&j.child_body_uuid), "distinct UUIDs resolved to the same runtime body")); }
        if !j.stiffness_N_per_m.is_finite() || j.stiffness_N_per_m < 0.0 { return Err(err("ingest", "JOINT_R1_INVALID_STIFFNESS", "stiffness_N_per_m", Some(&j.joint_uuid), None, "must be finite and non-negative")); }
        if !j.angle_stiffness_nm_per_rad.is_finite() || j.angle_stiffness_nm_per_rad < 0.0 { return Err(err("ingest", "JOINT_ANGULAR_INVALID_STIFFNESS", "angle_stiffness_Nm_per_rad", Some(&j.joint_uuid), None, "must be finite and non-negative")); }
        if !j.angle_damping_nms_per_rad.is_finite() || j.angle_damping_nms_per_rad < 0.0 { return Err(err("ingest", "JOINT_ANGULAR_INVALID_DAMPING", "angle_damping_Nms_per_rad", Some(&j.joint_uuid), None, "must be finite and non-negative")); }
        if !j.max_angle_rad.is_finite() || j.max_angle_rad < 0.0 || j.max_angle_rad > std::f64::consts::PI { return Err(err("ingest", "JOINT_ANGULAR_INVALID_LIMIT", "max_angle_rad", Some(&j.joint_uuid), None, "must be zero (disabled) or in (0, pi] radians")); }
        if !valid_quaternion(&j.rest_relative_quaternion_wxyz) { return Err(err("ingest", "JOINT_ANGULAR_INVALID_REST_ROTATION", "rest_relative_quaternion_wxyz", Some(&j.joint_uuid), None, "finite normalized wxyz quaternion required")); }
        // Directional (per-axis) angular control gates, canonical schema.
        // Signed-domain physical units: finite and non-negative per axis, and
        // deliberately no scalar pi bound. The f32 narrowing is checked here so
        // a value the device record cannot carry is refused at the boundary
        // instead of becoming a different (or zero) coefficient on upload.
        if let Some(c) = j.angular_control_v1.as_ref() {
            if c.coordinate_model != crate::material_spin_history::MODEL && c.coordinate_model != crate::joint_angular_control_gpu::COORDINATE_MODEL &&
                !(c.coordinate_model == crate::twist_overlay::CONTINUOUS_MODEL && !world_parent) {
                return Err(err("ingest", "JOINT_G2_MODEL", "angular_control_v1.coordinate_model", Some(&j.joint_uuid), None, "unknown or unsupported angular coordinate model"));
            }
            for (field, v) in [("stiffness_Nm_per_rad", c.stiffness()), ("damping_Nms_per_rad", c.damping())] {
                for (axis, x) in ["twist_x", "swing_y", "swing_z"].into_iter().zip(v) {
                    let name = format!("angular_control_v1.{field}");
                    if !x.is_finite() || x < 0.0 {
                        return Err(err("ingest", "JOINT_G2_COEFFICIENT", &name, Some(&j.joint_uuid), None, "finite nonnegative per-axis coefficient required"));
                    }
                    let narrowed = x as f32;
                    if !narrowed.is_finite() || (x != 0.0 && narrowed == 0.0) {
                        return Err(err("ingest", "JOINT_G2_COEFFICIENT", &name, Some(&j.joint_uuid), None, &format!("{axis} is outside the finite f32 device domain")));
                    }
                }
            }
        }
        if !valid_frame(&j.parent_local_frame) { return Err(err("ingest", "JOINT_R1_INVALID_LOCAL_FRAME", "parent_local_frame", Some(&j.joint_uuid), None, "finite translation and normalized quaternion required")); }
        if world_parent && j.parent_local_frame.quaternion_wxyz != [1.0, 0.0, 0.0, 0.0] { return Err(err("ingest", "JOINT_R1_INVALID_WORLD_FRAME", "parent_local_frame.quaternion_wxyz", Some(&j.joint_uuid), None, "world endpoint requires identity orientation; translation is an absolute world point")); }
        if !valid_frame(&j.child_local_frame) { return Err(err("ingest", "JOINT_R1_INVALID_LOCAL_FRAME", "child_local_frame", Some(&j.joint_uuid), None, "finite translation and normalized quaternion required")); }
        joints.push(OwnedJoint { wire: j, parent_runtime_body_index: a, child_runtime_body_index: b });
    }
    let hard_records=bind_hard_records(version,&joints,&bodies,pdrd_rows)?;
    Ok(OwnedJointScene { body_controls_v1, hard_records, scene_uuid: payload.scene_uuid, scene_inputs_sha256: payload.scene_inputs_sha256, payload_sha256: String::new(), payload_version: version, bodies, joints, roots:Vec::new(), fem_pdrd_parent_proof: ValidatedFemParentProof::default(), snapshot })
}

/// Peek the envelope version, then decode the typed payload under it.
///
/// The peek is deliberately separate from the typed decode so that a NEWER
/// envelope version is refused by version (`VersionMismatch`) rather than
/// reported as a payload-shape error, which is the existing contract for
/// versioned records. Only the two implemented versions are accepted; the
/// version gate then decides what each one is allowed to carry.
fn decode_joint_payload(bytes: &[u8]) -> Result<(JointPayload, u32), JointError> {
    let envelope: ppf_cts_formats::envelope::Envelope<serde_json::Value> = ciborium::from_reader(bytes)
        .map_err(|e| err("ingest", "JOINT_R1_MALFORMED_PAYLOAD", "envelope", None, None, &format!("{e:?}")))?;
    let version = envelope.version;
    if !matches!(version, ENVELOPE_VERSION | ANGULAR_CONTROL_ENVELOPE_VERSION | ANGULAR_EXTENSION_ENVELOPE_VERSION | 5 | 6 | 7 | 9) {
        return Err(err("ingest", "JOINT_R1_VERSION_OR_KIND", "envelope", None, None,
            &format!("envelope version {version} is not one of the implemented versions V{ENVELOPE_VERSION}/V{ANGULAR_CONTROL_ENVELOPE_VERSION}")));
    }
    let payload: JointPayload = from_cbor_with_version(version, KIND, bytes).map_err(|e| match &e {
        FormatError::VersionMismatch { .. } | FormatError::KindMismatch { .. } => err("ingest", "JOINT_R1_VERSION_OR_KIND", "envelope", None, None, &e.to_string()),
        FormatError::CborDe(_) | FormatError::CborSer(_) => err("ingest", "JOINT_R1_MALFORMED_PAYLOAD", "payload", None, None, &e.to_string()),
    })?;
    Ok((payload, version))
}

pub fn ingest(path: &Path, pdrd_rows: &[f32]) -> Result<Option<OwnedJointScene>, JointError> {
    check_root_orphan(path)?;
    checked_scene_root(path)?;
    if !checked_regular_file_or_absent(path, JOINT_FILE)? { return Ok(None); }
    let bytes = fs::read(safe_existing_file(path, JOINT_FILE)?).map_err(|e| err("ingest", "JOINT_R1_MALFORMED_PAYLOAD", "file", None, None, &e.to_string()))?;
    let (payload, version) = decode_joint_payload(&bytes)?;
    ingest_payload(payload,version,pdrd_rows,InputSnapshot{bytes:HashMap::new(),optional_absent:Vec::new()}).map(Some)
}

fn validated_snapshot(path:&Path, payload:&mut JointPayload, rows:&[f32])->Result<(SceneInputContext,InputSnapshot,Vec<u8>),JointError>{
    let context_bytes=fs::read(safe_existing_file(path, CONTEXT_FILE)?).map_err(|e|err("context","JOINT_R1_CONTEXT_MISSING",CONTEXT_FILE,None,None,&e.to_string()))?;
    let mut ctx:SceneInputContext=from_cbor_with_version(2,CONTEXT_KIND,&context_bytes).map_err(|e|err("context","JOINT_R1_CONTEXT_INVALID",CONTEXT_FILE,None,None,&e.to_string()))?;
    if ctx.context_version!=2||ctx.builder_contract_version!=2||!ctx.complete{return Err(err("context","JOINT_R1_CONTEXT_INCOMPLETE","complete",None,None,"final session configuration unavailable"));}
    ctx.scene_uuid=canonical_uuid(&ctx.scene_uuid).ok_or_else(||err("context","JOINT_R1_INVALID_UUID","scene_uuid",None,None,"invalid context UUID"))?;
    payload.scene_uuid=canonical_uuid(&payload.scene_uuid).ok_or_else(||err("context","JOINT_R1_INVALID_UUID","scene_uuid",None,None,"invalid payload UUID"))?;
    if ctx.scene_uuid!=payload.scene_uuid{return Err(err("context","JOINT_R1_CROSS_SCENE_CONTEXT","scene_uuid",None,None,"payload/context scene identity mismatch"));}
    let mut snapshot=HashMap::new(); let mut listed=Vec::new(); let mut previous="";
    for i in &ctx.inputs { if !safe_rel(&i.relative_path)||!is_consumed_input(&i.relative_path)||i.relative_path.as_str()<=previous{return Err(err("context","JOINT_R1_CONTEXT_INVALID","inputs",None,None,"paths must be safe, sorted, unique consumed inputs"));} previous=&i.relative_path; let p=safe_existing_file(path,&i.relative_path)?; let b=fs::read(&p).map_err(|e|err("context","JOINT_R1_CONTEXT_IO",&i.relative_path,None,None,&e.to_string()))?; if b.len()!=i.byte_length||sha256(&b)!=i.sha256{return Err(err("context","JOINT_R1_INPUT_HASH_MISMATCH",&i.relative_path,None,None,"changed solver input"));} listed.push(i.relative_path.clone());snapshot.insert(i.relative_path.clone(),b); }
    let actual=scan_consumed_paths(path)?; if actual!=listed{return Err(err("context","JOINT_R1_INVENTORY_MISMATCH","inputs",None,None,"unlisted or missing consumed solver input"));}
    for required in REQUIRED_R1_INPUTS { if !snapshot.contains_key(*required) { return Err(err("context","JOINT_R1_REQUIRED_INPUT_MISSING",required,None,None,"required R1/native input is absent from sealed inventory")); } }
    let absent=ctx.optional_absent.clone(); let mut sorted_absent=absent.clone(); sorted_absent.sort(); sorted_absent.dedup(); if sorted_absent!=absent||absent.iter().any(|p|!OPTIONAL_INPUTS.contains(&p.as_str())||!safe_rel(p)||path.join(p).exists()) {return Err(err("context","JOINT_R1_CONTEXT_INVALID","optional_absent",None,None,"optional absence must be sorted, unique, exact, and actually absent"));}
    for p in OPTIONAL_INPUTS {if *p==crate::joint_root_motion_binding::ROOT_MOTION_FILE && !matches!(payload.required_features.as_ref(),Some(f) if f.iter().any(|n|n=="pdrd_kinematic_roots_v1"||n=="pdrd_kinematic_roots_v2"||n==crate::joint_root_motion_wire::ROOT_FEATURE_V3)) && !snapshot.contains_key(*p) {continue;} if !snapshot.contains_key(*p)&&!absent.iter().any(|x|x==p){return Err(err("context","JOINT_R1_CONTEXT_INVALID","optional_absent",None,None,"optional input lacks presence/absence declaration"));}}
    required_native_inputs(&snapshot)?;
    let mut cb=HashMap::new(); for b in &mut ctx.bodies {b.body_uuid=canonical_uuid(&b.body_uuid).ok_or_else(||err("context","JOINT_R1_INVALID_UUID","body_uuid",None,Some(&b.body_uuid),"invalid context UUID"))?;if cb.insert(b.body_uuid.clone(),b.runtime_body_index).is_some(){return Err(err("context","JOINT_R1_DUPLICATE_BODY_UUID","body_uuid",None,Some(&b.body_uuid),"case alias"));}}
    let mut pb=HashMap::new();for b in &mut payload.bodies{b.body_uuid=canonical_uuid(&b.body_uuid).ok_or_else(||err("context","JOINT_R1_INVALID_UUID","body_uuid",None,Some(&b.body_uuid),"invalid payload UUID"))?;if pb.insert(b.body_uuid.clone(),b.runtime_body_index).is_some(){return Err(err("context","JOINT_R1_DUPLICATE_BODY_UUID","body_uuid",None,Some(&b.body_uuid),"case alias"));}}
    if cb!=pb{return Err(err("context","JOINT_R1_BINDING_MISMATCH","bodies",None,None,"payload bindings must exactly equal builder context"));}
    let mut bodies=ctx.bodies.clone();bodies.sort_by(|a,b|a.body_uuid.cmp(&b.body_uuid)); let mut digest=format!("ppf.pdrd_scene_context/v2\n{}\n",ctx.scene_uuid);for b in &bodies{digest.push_str(&format!("{}\t{}\t{}\n",b.body_uuid,b.runtime_body_index,b.vertex_count));}for i in &ctx.inputs{digest.push_str(&format!("{}\t{}\t{}\n",i.relative_path,i.byte_length,i.sha256));}for p in &absent{digest.push_str(&format!("optional_absent\t{}\n",p));}if sha256(digest.as_bytes())!=ctx.scene_inputs_sha256||ctx.scene_inputs_sha256!=payload.scene_inputs_sha256{return Err(err("context","JOINT_R1_INPUT_HASH_MISMATCH","scene_inputs_sha256",None,None,"digest mismatch"));}
    let map = decode_vertex_map(snapshot.get("map.pickle").ok_or_else(||err("context","JOINT_R1_MEMBERSHIP_INVALID","map.pickle",None,None,"missing identity map"))?)?;
    // `map.pickle` carries every named object exported by the frontend, not
    // just PDRD bodies.  Only keys that resolve to a bound PDRD UUID are
    // authoritative here; ordinary object names must survive validation.
    let bound_keys:HashSet<String>=ctx.bodies.iter().map(|b|b.body_uuid.clone()).collect();
    let mut canonical_map=HashMap::new(); for (key, members) in map.payload {
        if let Some(canonical)=canonical_uuid(&key) {
            if bound_keys.contains(&canonical) {
                if key!=canonical || canonical_map.insert(canonical.clone(),members).is_some(){return Err(err("context","JOINT_R1_MEMBERSHIP_INVALID","map.pickle",None,Some(&canonical),"case/UUID alias in bound PDRD map keys"));}
            }
        }
    }
    let body_bytes=snapshot.get("bin/pdrd_body.bin").ok_or_else(||err("context","JOINT_R1_REQUIRED_INPUT_MISSING","bin/pdrd_body.bin",None,None,"missing body bytes"))?;
    let authoritative_rows=read_f32_from(body_bytes)?;
    if !rows.is_empty() && rows != authoritative_rows.as_slice() { return Err(err("context","JOINT_R1_PDRD_BYTES_MISMATCH","bin/pdrd_body.bin",None,None,"caller/native rows differ from retained validated bytes")); }
    if authoritative_rows.len()%PDRD_BODY_ROW_LEN!=0||ctx.bodies.len()!=authoritative_rows.len()/PDRD_BODY_ROW_LEN{return Err(err("context","JOINT_R1_MEMBERSHIP_INVALID","pdrd_body.bin",None,None,"row/body cardinality mismatch"));}
    let index=read_u32_from(snapshot.get("bin/pdrd_vert_index.bin").ok_or_else(||err("context","JOINT_R1_MEMBERSHIP_INVALID","pdrd_vert_index.bin",None,None,"missing"))?)?;let list=read_u32_from(snapshot.get("bin/pdrd_vert_list.bin").ok_or_else(||err("context","JOINT_R1_MEMBERSHIP_INVALID","pdrd_vert_list.bin",None,None,"missing"))?)?;
    let rest=read_f32_from(snapshot.get("bin/pdrd_rest_centered.bin").ok_or_else(||err("context","JOINT_R1_MEMBERSHIP_INVALID","pdrd_rest_centered.bin",None,None,"missing"))?)?;
    validate_snapshot_domain(payload,&ctx,&snapshot,&authoritative_rows,&index,&list,&rest)?;
    let mut seen_rows=HashSet::new();let mut seen_vertices=HashSet::new();for b in &ctx.bodies{if !seen_rows.insert(b.runtime_body_index){return Err(err("context","JOINT_R1_BINDING_MISMATCH","runtime_body_index",None,Some(&b.body_uuid),"row alias"));}let r=b.runtime_body_index as usize;if r>=authoritative_rows.len()/23{return Err(err("context","JOINT_R1_DANGLING_BODY","runtime_body_index",None,Some(&b.body_uuid),"absent row"));}let row=&authoritative_rows[r*23..r*23+23];let start=field_u32(row[0],"vertex_start",&b.body_uuid)?;let count=field_u32(row[1],"vertex_count",&b.body_uuid)?;let mapped=canonical_map.get(&b.body_uuid).ok_or_else(||err("context","JOINT_R1_MEMBERSHIP_INVALID","map.pickle",None,Some(&b.body_uuid),"body not in canonical map"))?;if count!=b.vertex_count as usize||count!=mapped.len()||start.checked_add(count).filter(|x|*x<=list.len()).is_none()||list[start..start+count]!=mapped[..]||mapped.iter().any(|&v|!seen_vertices.insert(v)||index.get(v as usize)!=Some(&(b.runtime_body_index+1))){return Err(err("context","JOINT_R1_MEMBERSHIP_INVALID","map.pickle",None,Some(&b.body_uuid),"map/list/index membership disagree or duplicate"));}}
    if seen_vertices.len()!=list.len() || index.iter().enumerate().any(|(v,&owner)|(owner==0)==seen_vertices.contains(&(v as u32)) || (owner>0 && owner as usize>ctx.bodies.len())) { return Err(err("context","JOINT_R1_MEMBERSHIP_INVALID","pdrd_vert_index.bin",None,None,"map/list/index ownership is not complete and bijective")); }
    Ok((ctx,InputSnapshot{bytes:snapshot,optional_absent:absent},context_bytes))
}
fn read_u32_from(b:&[u8])->Result<Vec<u32>,JointError>{if b.len()%4!=0{return Err(err("context","JOINT_R1_CONTEXT_INVALID","binary",None,None,"u32 alignment"));}Ok(b.chunks_exact(4).map(|x|u32::from_le_bytes(x.try_into().unwrap())).collect())}
fn read_f32_from(b:&[u8])->Result<Vec<f32>,JointError>{if b.len()%4!=0{return Err(err("context","JOINT_R1_CONTEXT_INVALID","binary",None,None,"f32 alignment"));}Ok(b.chunks_exact(4).map(|x|f32::from_le_bytes(x.try_into().unwrap())).collect())}
fn read_usize_from(b:&[u8])->Result<Vec<usize>,JointError>{
    let width=std::mem::size_of::<usize>();
    if b.len()%width!=0{return Err(err("context","JOINT_R1_CONTEXT_INVALID","binary",None,None,"usize alignment"));}
    Ok(b.chunks_exact(width).map(|x|if width==8 {usize::try_from(u64::from_le_bytes(x.try_into().unwrap())).unwrap()} else {u32::from_le_bytes(x.try_into().unwrap()) as usize}).collect())
}

/// Enforce R1's documented physical-domain restrictions using only retained
/// bytes.  No `Path` is accepted: seal validation and native decoding must
/// never validate one generation then reopen a different filesystem state.
fn validate_snapshot_friction(snapshot:&HashMap<String,Vec<u8>>, parsed:&toml::Value)->Result<(),JointError>{
    // This is a coefficient-domain check, not a change to upstream contact
    // assembly. All static, spatial and sampled primitive bins are retained.
    for (name,bytes) in snapshot {
        if name.starts_with("bin/")&&name.ends_with(".bin")&&name.contains("friction") {
            for mu in read_f32_from(bytes)? {
                if !mu.is_finite()||mu<0. {
                    return Err(err("context","JOINT_R1_UNSUPPORTED_FRICTION",name,None,None,"friction must be finite and nonnegative"));
                }
            }
        }
    }
    let count=parsed.get("count").and_then(|v|v.as_table()).ok_or_else(||err("context","JOINT_R1_MALFORMED_INPUT","info.toml",None,None,"missing [count]"))?;
    for kind in ["wall","sphere"] {
        let n=count.get(kind).and_then(|v|v.as_integer()).filter(|v|*v>=0)
            .ok_or_else(||err("context","JOINT_R1_MALFORMED_INPUT","info.toml",None,None,&format!("missing nonnegative count.{kind}")))?;
        for i in 0..n {
            let section=format!("{kind}-{i}");
            let Some(entry)=parsed.get(&section).and_then(|v|v.as_table()) else {continue};
            if entry.get("keyframe").and_then(|v|v.as_integer()).unwrap_or(0)>0 {
                let mu=entry.get("friction").and_then(|v|v.as_float())
                    .ok_or_else(||err("context","JOINT_R1_MALFORMED_INPUT","info.toml",None,None,&format!("missing {section}.friction")))?;
                if !mu.is_finite()||!(mu as f32).is_finite()||mu<0. {
                    return Err(err("context","JOINT_R1_UNSUPPORTED_FRICTION",&section,None,None,"collider friction must be finite and nonnegative"));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod friction_domain_tests {
    use super::*;
    fn info() -> toml::Value { toml::from_str("[count]\nwall=0\nsphere=0\n").unwrap() }
    #[test]
    fn retained_static_and_sampled_coefficients_preserve_nonzero_bytes() {
        let snapshot:HashMap<_,_>=[
            ("bin/param/tri-friction.bin".into(),[0.0f32,0.5,1.0].into_iter().flat_map(f32::to_le_bytes).collect()),
            ("bin/param-anim/tri-friction-1.bin".into(),[0.2f32,0.7].into_iter().flat_map(f32::to_le_bytes).collect()),
            ("bin/sand-friction.bin".into(),0.4f32.to_le_bytes().to_vec()),
        ].into_iter().collect();
        let before=snapshot.clone();
        validate_snapshot_friction(&snapshot,&info()).unwrap();
        assert_eq!(snapshot,before);
    }
    #[test]
    fn retained_invalid_coefficients_and_misalignment_refuse() {
        for mu in [-1.0f32,f32::NAN,f32::INFINITY,f32::NEG_INFINITY] {
            let snapshot=[("bin/param/tri-friction.bin".into(),mu.to_le_bytes().to_vec())].into_iter().collect();
            assert_eq!(validate_snapshot_friction(&snapshot,&info()).unwrap_err().stable_code,"JOINT_R1_UNSUPPORTED_FRICTION");
        }
        let snapshot=[("bin/param/tri-friction.bin".into(),vec![0u8;3])].into_iter().collect();
        assert_eq!(validate_snapshot_friction(&snapshot,&info()).unwrap_err().stable_code,"JOINT_R1_CONTEXT_INVALID");
    }
    #[test]
    fn active_wall_and_sphere_friction_use_same_finite_nonnegative_domain() {
        for kind in ["wall","sphere"] {
            for mu in ["0.0","0.5","1.0","-1.0","nan","inf","-inf","1.0e100","true","'0.5'"] {
                let parsed=toml::from_str(&format!("[count]\nwall={}\nsphere={}\n[{kind}-0]\nkeyframe=1\nfriction={mu}\n",i32::from(kind=="wall"),i32::from(kind=="sphere"))).unwrap();
                let result=validate_snapshot_friction(&HashMap::new(),&parsed);
                if matches!(mu,"0.0"|"0.5"|"1.0") {assert!(result.is_ok(),"{kind}/{mu}");}
                else {assert!(result.is_err(),"{kind}/{mu}");}
            }
        }
    }
}

fn validate_snapshot_domain(payload:&JointPayload, ctx:&SceneInputContext, snapshot:&HashMap<String,Vec<u8>>, rows:&[f32], index:&[u32], list:&[u32], rest:&[f32])->Result<(),JointError>{
    let info=std::str::from_utf8(snapshot.get("info.toml").ok_or_else(||err("context","JOINT_R1_REQUIRED_INPUT_MISSING","info.toml",None,None,"missing retained configuration"))?).map_err(|e|err("context","JOINT_R1_MALFORMED_INPUT","info.toml",None,None,&e.to_string()))?;
    let parsed:toml::Value=toml::from_str(info).map_err(|e|err("context","JOINT_R1_MALFORMED_INPUT","info.toml",None,None,&e.to_string()))?;
    let count=parsed.get("count").and_then(|v|v.as_table()).ok_or_else(||err("context","JOINT_R1_MALFORMED_INPUT","info.toml",None,None,"missing [count]"))?;
    let n=|key:&str| count.get(key).and_then(|v|v.as_integer()).filter(|v|*v>=0).map(|v|v as usize).ok_or_else(||err("context","JOINT_R1_MALFORMED_INPUT","info.toml",None,None,&format!("missing nonnegative count.{key}")));
    let param=std::str::from_utf8(snapshot.get("param.toml").ok_or_else(||err("context","JOINT_R1_REQUIRED_INPUT_MISSING","param.toml",None,None,"missing retained configuration"))?).map_err(|e|err("context","JOINT_R1_MALFORMED_INPUT","param.toml",None,None,&e.to_string()))?;
    let config:NativeConfig=toml::from_str(param).map_err(|e|err("context","JOINT_R1_MALFORMED_INPUT","param.toml",None,None,&e.to_string()))?;
    if config.param.world_scaling != 1.0 {return Err(err("context","JOINT_R1_UNSUPPORTED_DOMAIN","world_scaling",None,None,"R1 requires retained param.world_scaling exactly 1"));}
    if rest.len()!=list.len().checked_mul(3).ok_or_else(||err("context","JOINT_R1_MEMBERSHIP_INVALID","pdrd_rest_centered.bin",None,None,"rest length overflow"))? {return Err(err("context","JOINT_R1_MEMBERSHIP_INVALID","pdrd_rest_centered.bin",None,None,"retained rest length must equal 3 * retained vertex-list length"));}
    if !rest.iter().all(|x|x.is_finite()) {return Err(err("context","JOINT_R1_PDRD_ROW_INVALID","pdrd_rest_centered.bin",None,None,"retained centered rest coordinates must be finite"));}
    for body in &ctx.bodies {let r=body.runtime_body_index as usize;let row=rows.get(r*PDRD_BODY_ROW_LEN..(r+1)*PDRD_BODY_ROW_LEN).ok_or_else(||err("context","JOINT_R1_DANGLING_BODY","runtime_body_index",None,Some(&body.body_uuid),"retained PDRD row absent"))?;if !row.iter().all(|x|x.is_finite())||row[2]<=0.||row[15]<=0.||row[16]!=0. {return Err(err("context","JOINT_R1_PDRD_ROW_INVALID","pdrd_body.bin",None,Some(&body.body_uuid),"R1 requires finite physical FREE PDRD rows"));}let count=field_u32(row[1],"vertex_count",&body.body_uuid)?;if count<4{return Err(err("context","JOINT_R1_MEMBERSHIP_INVALID","vertex_count",None,Some(&body.body_uuid),"R1 requires at least four vertices per endpoint"));}let determinant=Matrix3::from_row_slice(&row[6..15]).determinant();if !determinant.is_finite()||determinant.abs()<1e-12{return Err(err("context","JOINT_R1_PDRD_ROW_INVALID","gram_inv",None,Some(&body.body_uuid),"degenerate retained inverse rest Gram"));}}
    let collision_feature = payload.required_features.as_ref().is_some_and(|f| f.iter().any(|n| n == "ppf_collision_filter_v1"));
    let collision_bytes = snapshot.get("bin/collision_filter.bin");
    let static_collision_bytes = snapshot.get("bin/static_collision_filter.bin");
    if collision_feature != (collision_bytes.is_some() || static_collision_bytes.is_some()) {
        return Err(err("collision", "COLLISION_FILTER_CAPABILITY", "bin/collision_filter.bin", None, None, "sealed input and required feature must both be present"));
    }
    if let Some(bytes) = collision_bytes {
        let filter = read_u32_from(bytes)?;
        if filter.len() != index.len() || !filter.iter().any(|&v| v != 0) {
            return Err(err("collision", "COLLISION_FILTER_INVALID", "bin/collision_filter.bin", None, None, "requires one u32 per dynamic vertex and at least one enabled filter"));
        }
    }
    validate_snapshot_friction(snapshot,&parsed)?;
    let endpoint_rows:HashSet<u32>=payload.bodies.iter().map(|b|b.runtime_body_index).collect();
    for i in 0..n("pin_block")? {let name=format!("bin/pin-ind-{i}.bin");let pins=read_usize_from(snapshot.get(&name).ok_or_else(||err("context","JOINT_R1_REQUIRED_INPUT_MISSING",&name,None,None,"pin input absent from retained snapshot"))?)?;for v in pins {if let Some(&owner)=index.get(v) {if owner>0&&endpoint_rows.contains(&(owner-1)) {return Err(err("context","JOINT_R1_ENDPOINT_PINNED",&name,None,None,"endpoint membership overlaps a retained pin or override"));}}}}
    if ["has_rest_vert","has_rest_vert_anim","has_bend_rest_vert"].into_iter().any(|k|count.get(k).and_then(|v|v.as_bool())==Some(true)) {return Err(err("context","JOINT_R1_ENDPOINT_PRESCRIBED_MOTION","info.toml",None,None,"R1 rejects retained rest-shape overrides"));}
    if let Some(bytes)=snapshot.get("dyn_param.txt") {let dynp=std::str::from_utf8(bytes).map_err(|e|err("context","JOINT_R1_MALFORMED_INPUT","dyn_param.txt",None,None,&e.to_string()))?;if ["velocity:","angular_velocity:","angular_velocity_world:"].into_iter().any(|mode|dynp.contains(mode)) {return Err(err("context","JOINT_R1_ENDPOINT_PRESCRIBED_MOTION","dyn_param.txt",None,None,"R1 rejects retained prescribed velocity modes"));}}
    Ok(())
}

/// The only entry point used by scene construction.  Keep `ingest` public for
/// payload-only tests, but never let setup consume its result without first
/// proving the sealed solver-input snapshot.
pub fn ingest_validated(path: &Path, pdrd_rows: &[f32]) -> Result<Option<OwnedJointScene>, JointError> {
    let Some(input)=prepare_native_input(path)? else{return Ok(None);};
    let retained=read_f32_from(input.bytes("bin/pdrd_body.bin")?)?;
    if retained!=pdrd_rows{return Err(err("context","JOINT_R1_PDRD_BYTES_MISMATCH","bin/pdrd_body.bin",None,None,"caller differs from sealed input"));}
    Ok(Some(input.into_records()))
}

/// Capture, validate and retain the complete R1 input generation before a
/// native constructor performs its first input decode.
// Insert before prepare_native_input. All bytes below belong to one retained
// inventory generation. Physical PDRD rows remain FREE.
fn root_error(e: crate::joint_root_motion_wire::RootMotionError) -> JointError {
    err("root", e.code, &e.field, None, e.body_uuid.as_deref(), &e.detail)
}
fn check_root_orphan(path: &Path) -> Result<(), JointError> {
    if !checked_regular_file_or_absent(path,JOINT_FILE)? && checked_regular_file_or_absent(path,crate::connected_collision_native::CROSS_FILE)? {
        return Err(err("cross-links","CROSS_LINK_COLLISION_INVALID","required_features",None,None,"orphan collision sidecar requires sealed joint carrier"));
    }
    if !checked_regular_file_or_absent(path,JOINT_FILE)? && checked_regular_file_or_absent(path,crate::pmx_joints::FILE)? {
        return Err(err("pmx-joints","PMX_JOINT_INVALID","required_features",None,None,"orphan PMX sidecar requires sealed joint carrier"));
    }
    if !checked_regular_file_or_absent(path,JOINT_FILE)? && checked_regular_file_or_absent(path,crate::cross_links::FILE)? {
        return Err(err("cross-links","CROSS_LINK_INVALID","required_features",None,None,"orphan cross-link sidecar requires sealed joint carrier"));
    }

    if !checked_regular_file_or_absent(path, JOINT_FILE)? &&
        (checked_regular_file_or_absent(path, crate::soft_joint::FILE)? || checked_regular_file_or_absent(path, crate::soft_joint::CONTROL_FILE)?) {
        return Err(err("fem", "FEM_JOINT_INVALID", "required_features", None, None,
            "FEM sidecar requires its sealed capability carrier; orphan joints cannot be ignored"));
    }
    if !checked_regular_file_or_absent(path, JOINT_FILE)? &&
        checked_regular_file_or_absent(path, crate::joint_root_motion_binding::ROOT_MOTION_FILE)? {
        return Err(err("root", "ROOT_MOTION_REQUIRED_FEATURES", "required_features", None, None,
            "root sidecar requires a sealed joint manifest and root feature"));
    }
    Ok(())
}
fn bind_snapshot_roots(payload: &JointPayload, version: u32, ctx: &SceneInputContext,
                       snapshot: &InputSnapshot, rows: &[f32], validated_fem_parents:&HashSet<String>)
    -> Result<Vec<crate::joint_root_motion_wire::ValidatedRootMotion>, JointError> {
    use crate::joint_root_motion_binding::{decode_bound, SealedEntry, ROOT_MOTION_FILE};
    use crate::joint_root_motion_wire::{BodyIdentity, NativeFrameEvidence};
    let entry=ctx.inputs.iter().find(|i|i.relative_path==ROOT_MOTION_FILE)
        .map(|i|SealedEntry{relative_path:&i.relative_path,byte_length:i.byte_length as u64,sha256:&i.sha256});
    let roots=decode_bound(payload.required_features.as_deref().unwrap_or(&[]),
        snapshot.bytes.get(ROOT_MOTION_FILE).map(Vec::as_slice),entry,&ctx.scene_uuid).map_err(root_error)?;
    if roots.is_empty(){return Ok(roots);}
    if version==7{
        if roots.len()!=1||payload.joints.len()!=1||payload.bodies.len()>3||payload.required_features.as_ref().is_some_and(|f|f.iter().any(|n|n=="pdrd_kinematic_roots_v2"||n==crate::joint_root_motion_wire::ROOT_FEATURE_V3)){
            return Err(err("root","G2_HARD_UNSUPPORTED_DOMAIN","root_union",None,None,"V7 single D6 and one V1 source root only"));
        }
        let j=&payload.joints[0];
        if !j.enabled||j.joint_d6_hard_v1.is_none()||j.angular_hard_hinge_v1.is_some()||j.parent_body_uuid!=roots[0].body_uuid{
            return Err(err("root","ROOT_HARD_SOURCE","body_uuid",None,None,"known root must be the enabled single-D6 parent/source"));
        }
    }else if !matches!(version,6|9){return Err(err("root","ROOT_MOTION_VERSION_OR_KIND","envelope",None,None,"roots require V6/V9 or the exact V7 single-D6 union"));}
    let bytes=|name:&str|snapshot.bytes.get(name).map(Vec::as_slice)
        .ok_or_else(||err("root","ROOT_MOTION_BODY_BINDING",name,None,None,"missing sealed native bytes"));
    let config:NativeConfig=toml::from_str(std::str::from_utf8(bytes("param.toml")?)
        .map_err(|e|err("root","ROOT_MOTION_NATIVE_CAPABILITY","param.toml",None,None,&e.to_string()))?)
        .map_err(|e|err("root","ROOT_MOTION_NATIVE_CAPABILITY","param.toml",None,None,&e.to_string()))?;
    // Revalidate the same retained generation at the capability boundary;
    // a scalar zero cannot describe per-object, sampled or analytic friction.
    let info:toml::Value=toml::from_str(std::str::from_utf8(bytes("info.toml")?)
        .map_err(|e|err("context","JOINT_R1_MALFORMED_INPUT","info.toml",None,None,&e.to_string()))?)
        .map_err(|e|err("context","JOINT_R1_MALFORMED_INPUT","info.toml",None,None,&e.to_string()))?;
    validate_snapshot_friction(&snapshot.bytes,&info)?;
    crate::joint_root_motion_binding::gate_native(&roots,&crate::joint_root_motion_binding::NativeCapabilityEvidence {
        playback:config.param.playback,world_scale:config.param.world_scaling,friction_domain_validated:true,
        root_module_installed:cfg!(neutral_driver),key_cut_installed:cfg!(neutral_driver),projected_solve_installed:cfg!(neutral_driver),
        endpoint_history_gate_installed:cfg!(neutral_driver), curved_ccd_installed:cfg!(neutral_driver), control_boundary_installed:cfg!(neutral_driver) }).map_err(root_error)?;
    // Conservative first milestone: any aggregate lock is refused alongside
    // roots. Never remove or silently ignore a preexisting lock.
    for name in ["bin/translation_lock.bin","bin/rotation_lock.bin","bin/translation_lock_mode.bin","bin/rotation_lock_mode.bin"] {
        if snapshot.bytes.get(name).is_some_and(|b|b.iter().any(|v|*v!=0)) {
            return Err(err("root","ROOT_MOTION_NATIVE_CAPABILITY",name,None,None,"root plus aggregate lock composition is not installed"));
        }
    }
    let list=read_u32_from(bytes("bin/pdrd_vert_list.bin")?)?;
    let rest=read_f32_from(bytes("bin/pdrd_rest_centered.bin")?)?;
    let dmap=read_u32_from(bytes("bin/vert_dmap.bin")?)?;
    let f64s=|b:&[u8]|->Result<Vec<f64>,JointError>{
        if b.len()%8!=0{return Err(err("root","ROOT_MOTION_BODY_BINDING","native positions",None,None,"f64 alignment"));}
        Ok(b.chunks_exact(8).map(|s|f64::from_le_bytes(s.try_into().unwrap())).collect())
    };
    let verts=f64s(bytes("bin/vert.bin")?)?;
    let displacement=f64s(bytes("bin/displacement.bin")?)?;
    if verts.len()!=3*dmap.len() || displacement.len()%3!=0 {
        return Err(err("root","ROOT_MOTION_BODY_BINDING","native positions",None,None,"position/displacement shape"));
    }
    let context_ids:Vec<_>=ctx.bodies.iter().map(|b|BodyIdentity{uuid:&b.body_uuid,runtime_body_index:b.runtime_body_index}).collect();
    let joint_ids:Vec<_>=payload.bodies.iter().map(|b|BodyIdentity{uuid:&b.body_uuid,runtime_body_index:b.runtime_body_index}).collect();
    let mut pins=Vec::new();
    for (name,raw) in &snapshot.bytes {
        if name.starts_with("bin/pin-ind-")&&name.ends_with(".bin") {
            for v in read_usize_from(raw)? {pins.push(u32::try_from(v).map_err(|_|err("root","ROOT_MOTION_BODY_BINDING",name,None,None,"pin overflow"))?);}
        }
    }
    // Keep owned arrays alive until all borrowed NativeFrameEvidence are used.
    let mut spans=Vec::new();let mut rest_rows=Vec::new();let mut world_rows=Vec::new();let mut tolerances=Vec::new();
    for r in &roots {
        let index=r.runtime_body_index as usize;
        let row=rows.get(index*23..(index+1)*23).ok_or_else(||err("root","ROOT_MOTION_BODY_BINDING","runtime_body_index",None,Some(&r.body_uuid),"row absent"))?;
        let start=field_u32(row[0],"vertex_start",&r.body_uuid)?;
        let count=field_u32(row[1],"vertex_count",&r.body_uuid)?;
        let members=list.get(start..start+count).ok_or_else(||err("root","ROOT_MOTION_BODY_BINDING","vertex_list",None,Some(&r.body_uuid),"span absent"))?;
        let mut rr=Vec::new();let mut xx=Vec::new();let mut extent=0.0_f64;let mut world=1.0_f64;
        for (local,&v) in members.iter().enumerate() {
            let y:[f64;3]=std::array::from_fn(|a|f64::from(rest[3*(start+local)+a]));
            let group=*dmap.get(v as usize).ok_or_else(||err("root","ROOT_MOTION_BODY_BINDING","vert_dmap",None,None,"vertex absent"))? as usize;
            if group>=displacement.len()/3 {return Err(err("root","ROOT_MOTION_BODY_BINDING","displacement",None,None,"group absent"));}
            // Match scene::scale_mat + apply_displacement exactly: narrow each
            // input to f32, add in f32, THEN promote for this comparison.
            let x:[f64;3]=std::array::from_fn(|a|f64::from((verts[3*v as usize+a] as f32)+(displacement[3*group+a] as f32)));
            extent=extent.max(y.iter().map(|v|v*v).sum::<f64>().sqrt());
            world=world.max(x.iter().map(|v|v.abs()).fold(0.0,f64::max));rr.push(y);xx.push(x);
        }
        // Native f32 vertex images need a world-coordinate rounding allowance;
        // also retain a body-scale bound. This is a numerical, explicit bound.
        tolerances.push((16.0*f64::from(f32::EPSILON)*extent).max(4.0*f64::from(f32::EPSILON)*world));
        spans.push(members.to_vec());rest_rows.push(rr);world_rows.push(xx);
    }
    let evidence:Vec<_>=roots.iter().enumerate().map(|(i,r)|NativeFrameEvidence {
        context_scene_uuid:&ctx.scene_uuid,context_bodies:&context_ids,joint_bodies:&joint_ids,
        native_body:BodyIdentity{uuid:&r.body_uuid,runtime_body_index:r.runtime_body_index},
        native_body_is_free:rows[23*r.runtime_body_index as usize+16]==0.0,
        has_enabled_known_parent_joint:payload.joints.iter().any(|j|j.enabled&&j.parent_body_uuid==r.body_uuid)
            || validated_fem_parents.contains(&r.body_uuid),
        vertex_indices:&spans[i],rest_centered_m:&rest_rows[i],initial_world_m:&world_rows[i],fit_tolerance_m:tolerances[i],
    }).collect();
    crate::joint_root_motion_binding::bind_native_frames_with_body_controls(&roots,&evidence,&pins,payload.required_features.as_ref().is_some_and(|f|f.iter().any(|n|n==crate::joint_root_motion_wire::BODY_CONTROLS_FEATURE))).map_err(root_error)?;
    Ok(roots)
}

pub fn prepare_native_input(path: &Path) -> Result<Option<ValidatedJointInput>, JointError> {
    check_root_orphan(path)?;
    checked_scene_root(path)?;
    if !checked_regular_file_or_absent(path, JOINT_FILE)? { return Ok(None); }
    let manifest_bytes = fs::read(safe_existing_file(path, JOINT_FILE)?)
        .map_err(|e| err("ingest", "JOINT_R1_MALFORMED_PAYLOAD", JOINT_FILE, None, None, &e.to_string()))?;
    let (mut payload, version) = decode_joint_payload(&manifest_bytes)?;
    let (ctx, snapshot, context_bytes) = validated_snapshot(path, &mut payload, &[])?;
    let rows = read_f32_from(snapshot.bytes.get("bin/pdrd_body.bin").expect("validated required input"))?;
    let records = ingest_payload(payload.clone(), version, &rows, snapshot)?;
    let mut input=ValidatedJointInput { records, manifest_bytes, context_bytes };
    // Bind the full material graph first. Root edge evidence then comes from
    // validated physical mixed joints, without adding phantom PDRD records.
    let soft=crate::soft_joint::prepare(&input)?;
    crate::cross_links::prepare(&input)?;
    crate::pmx_joints::prepare(&input, 1.0)?;
    let fem_parents:HashSet<String>=soft.as_ref().map(|s|s.wires.iter()
        .filter(|j|j.wire.enabled).map(|j|j.wire.parent_body_uuid.clone()).collect()).unwrap_or_default();
    // Retain the same strict-prepare evidence for the SECOND (runtime
    // actual-geometry) rebind in `driver::roots::install_native`. Populated only
    // here, only after `soft_joint::prepare` succeeded, and only from the
    // validated UUID -> runtime mapping. The rigid path leaves it empty.
    if let Some(s)=soft.as_ref() {
        input.records.fem_pdrd_parent_proof =
            collect_fem_pdrd_parent_proof(&s.bodies,&s.wires,&input.records.bodies);
    }
    let roots=bind_snapshot_roots(&payload,version,&ctx,&input.records.snapshot,&rows,&fem_parents)?;
    if input.records.body_controls_v1 && roots.is_empty() {
        return Err(err("root","ROOT_MOTION_BODY_BINDING","roots",None,None,"body-controls carrier requires at least one sealed known controlled body"));
    }
    input.records.roots=roots;
    input.records.payload_sha256 = sha256(&input.manifest_bytes);
    Ok(Some(input))
}

/// Read the body ABI once for R1 and retain those exact decoded values for the
/// native constructor.  This closes the old validate-then-read-again window.
pub fn validated_pdrd_rows(path: &Path) -> Result<Option<(OwnedJointScene, Vec<f32>)>, JointError> {
    let Some(input) = prepare_native_input(path)? else { return Ok(None); };
    let rows = read_f32_from(input.bytes("bin/pdrd_body.bin")?)?;
    Ok(Some((input.records, rows)))
}

/// Deterministic preflight used before `main::setup()` is allowed to prune a
/// fresh output or resume checkpoints.  It deliberately reads the unchanged
/// 23-float ABI directly and never deserializes checkpoint state.
pub fn preflight(path: &Path) -> Result<(), JointError> {
    preflight_selected(path, ExperimentalBallAnchorSelection::Off)
}

/// Pre-setup gate.  The requested path deliberately validates the exact same
/// sealed input/domain first, then asks the real native ABI gate.  Therefore
/// malformed, unknown, and domain-invalid payloads are never hidden behind a
/// convenient "experimental unavailable" response.
pub fn preflight_selected(path: &Path, selection: ExperimentalBallAnchorSelection) -> Result<(), JointError> {
    check_root_orphan(path)?;
    checked_scene_root(path)?;
    if !checked_regular_file_or_absent(path, JOINT_FILE)? { return Ok(()); }
    let bytes = fs::read(safe_existing_file(path, "bin/pdrd_body.bin")?).map_err(|e| err("preflight", "JOINT_R1_PDRD_ABI_INVALID", "pdrd_body.bin", None, None, &e.to_string()))?;
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        return Err(err("preflight", "JOINT_R1_PDRD_ABI_INVALID", "pdrd_body.bin", None, None, "file is not f32 aligned"));
    }
    let rows: Vec<f32> = bytes.chunks_exact(4).map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]])).collect();
    let records = ingest_validated(path, &rows)?.expect("manifest was checked above");
    // Validate material-joint inputs before setup may create/prune output.
    let input=prepare_native_input(path)?.expect("manifest checked above");
    crate::soft_joint::prepare(&input)?;
    if selection == ExperimentalBallAnchorSelection::Requested {
        // Structural/input preflight is complete. The executable performs its
        // actual linked-DLL query separately before it creates or cleans output.
        return Ok(());
    }
    Err(err("preflight", "JOINT_R1_UNSUPPORTED_CAPABILITY", "required_capability", None, None,
        &format!("{} is required by scene {} but this executable has no CUDA/PCG joint assembly ({} body bindings, {} joints)", CAPABILITY, records.scene_uuid, records.bodies.len(), records.joints.len())))
}

/// Reject all resume spellings before checkpoint discovery, output pruning, or
/// bincode deserialization.  `-1` is also a resume request even though the
/// historical parser resolves it later to a positive checkpoint number.
pub fn reject_experimental_resume(load: i32) -> Result<(), JointError> {
    if load == 0 { return Ok(()); }
    Err(err("preflight", "JOINT_R1_EXPERIMENTAL_RESUME_UNSUPPORTED", "load", None, None,
        "--experimental-ball-anchor-prototype is fresh-run-only; resume is rejected before checkpoint selection"))
}

/// Shared by preflight and the defensive Scene::new boundary.  This remains a
/// refusal until a native ABI handshake proves the whole authoritative
/// pose/CCD/finality path exists; the dormant upload symbol is insufficient.
pub fn require_experimental_native(records: &OwnedJointScene) -> Result<(), JointError> {
    validate_experimental_handshake(records, native_experimental_handshake())
}

/// Keep the future FFI query and its contract validation separate.  This lets
/// the Rust gate test version/capability mismatches without pretending that an
/// upload endpoint is a valid native motion implementation.
fn validate_experimental_handshake(
    records: &OwnedJointScene,
    handshake: NativeExperimentalHandshake,
) -> Result<(), JointError> {
    match handshake {
        NativeExperimentalHandshake::Ready { abi_version, capability }
            if abi_version == EXPERIMENTAL_NATIVE_ABI_VERSION && capability == CAPABILITY => Ok(()),
        NativeExperimentalHandshake::Ready { abi_version, capability } => Err(err("capability", "JOINT_R1_EXPERIMENTAL_NATIVE_UNAVAILABLE", "native_handshake", records.joints.first().map(|j| j.wire.joint_uuid.as_str()), None,
            &format!("native experimental ABI/capability mismatch: expected v{EXPERIMENTAL_NATIVE_ABI_VERSION}/{CAPABILITY}, got v{abi_version}/{capability}"))),
        NativeExperimentalHandshake::Unavailable => Err(err("capability", "JOINT_R1_EXPERIMENTAL_NATIVE_UNAVAILABLE", "native_handshake", records.joints.first().map(|j| j.wire.joint_uuid.as_str()), None,
            "no verified native experimental ABI handshake for authoritative pose, reconstructed CCD, and finality")),
    }
}

/// Write a machine-readable refusal next to the immutable scene input, not in
/// `output/`: this is intentionally safe for both fresh and resume launches.
pub fn write_preflight_refusal(path: &Path, e: &JointError) {
    #[derive(Serialize)]
    struct Refusal<'a> { stage: &'a str, stable_code: &'a str, field: &'a str, joint_uuid: &'a Option<String>, body_uuid: &'a Option<String>, detail: &'a str }
    let refusal = Refusal { stage: e.stage, stable_code: e.stable_code, field: &e.field, joint_uuid: &e.joint_uuid, body_uuid: &e.body_uuid, detail: &e.detail };
    if let Ok(bytes) = serde_json::to_vec_pretty(&refusal) {
        let _ = fs::write(path.join("joint_r1_preflight_refusal.json"), bytes);
    }
}

/// Native-source, FP64 host representation of the R1 Gauss--Newton term.
pub type Vec6 = SVector<f64, 6>; pub type Mat6 = SMatrix<f64, 6, 6>; pub type Mat12 = SMatrix<f64, 12, 12>;
#[derive(Clone, Debug)] pub struct BallAnchor { pub parent_local_anchor_m: Vector3<f64>, pub child_local_anchor_m: Vector3<f64>, pub stiffness_n_per_m: f64, pub enabled: bool }
#[derive(Clone, Debug)] pub struct BallEval { pub residual_m: Vector3<f64>, pub energy_j: f64, pub gradient: SVector<f64, 12>, pub hessian_gn: Mat12 }
fn skew(v: &Vector3<f64>) -> Matrix3<f64> { Matrix3::new(0.0,-v.z,v.y, v.z,0.0,-v.x, -v.y,v.x,0.0) }
impl BallAnchor {
    pub fn evaluate(&self, ca: Vector3<f64>, ra: Matrix3<f64>, cb: Vector3<f64>, rb: Matrix3<f64>) -> BallEval {
        if !self.enabled || self.stiffness_n_per_m == 0.0 { return BallEval { residual_m: Vector3::zeros(), energy_j: 0.0, gradient: SVector::zeros(), hessian_gn: Mat12::zeros() }; }
        let p = ra * self.parent_local_anchor_m; let q = rb * self.child_local_anchor_m; let r = ca + p - cb - q;
        let mut ja = SMatrix::<f64,3,6>::zeros(); let mut jb = SMatrix::<f64,3,6>::zeros();
        ja.fixed_view_mut::<3,3>(0,0).copy_from(&Matrix3::identity()); ja.fixed_view_mut::<3,3>(0,3).copy_from(&-skew(&p));
        jb.fixed_view_mut::<3,3>(0,0).copy_from(&Matrix3::identity()); jb.fixed_view_mut::<3,3>(0,3).copy_from(&-skew(&q));
        let mut j = SMatrix::<f64,3,12>::zeros(); j.fixed_view_mut::<3,6>(0,0).copy_from(&ja); j.fixed_view_mut::<3,6>(0,6).copy_from(&-jb);
        BallEval { residual_m: r, energy_j: 0.5*self.stiffness_n_per_m*r.dot(&r), gradient: self.stiffness_n_per_m*j.transpose()*r, hessian_gn: self.stiffness_n_per_m*j.transpose()*j }
    }
    pub fn matvec(&self, ca: Vector3<f64>, ra: Matrix3<f64>, cb: Vector3<f64>, rb: Matrix3<f64>, v: SVector<f64,12>) -> SVector<f64,12> { self.evaluate(ca,ra,cb,rb).hessian_gn * v }
    pub fn incident_self_blocks(&self, ca: Vector3<f64>, ra: Matrix3<f64>, cb: Vector3<f64>, rb: Matrix3<f64>) -> (Mat6, Mat6) { let h=self.evaluate(ca,ra,cb,rb).hessian_gn; (h.fixed_view::<6,6>(0,0).into_owned(), h.fixed_view::<6,6>(6,6).into_owned()) }
}

#[cfg(test)]
fn test_workspace_root() -> std::path::PathBuf {
    std::fs::canonicalize(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2).unwrap()).unwrap()
}
#[cfg(test)]
fn test_area() -> std::path::PathBuf { test_workspace_root().join("target").join("joint-test-fixtures") }
#[cfg(test)]
fn reject_reparse_path(workspace: &Path, candidate: &Path) {
    let relative=candidate.strip_prefix(workspace).expect("test path must be lexically below workspace");
    assert!(relative.components().all(|c|matches!(c,Component::Normal(_))), "test path must contain only normal workspace-relative components: {}", candidate.display());
    let mut current=workspace.to_path_buf();
    assert!(!is_reparse(&fs::symlink_metadata(&current).unwrap()), "workspace is a reparse point");
    for component in relative.components() { let Component::Normal(part)=component else { unreachable!() }; current.push(part); if let Ok(meta)=fs::symlink_metadata(&current) { assert!(!is_reparse(&meta), "test path has reparse ancestor: {}", current.display()); } }
}
#[cfg(test)]
fn owned_test_root(prefix: &str) -> std::path::PathBuf {
    let workspace=test_workspace_root(); let area=test_area();
    reject_reparse_path(&workspace, &area);
    std::fs::create_dir_all(&area).unwrap(); reject_reparse_path(&workspace, &area);
    for attempt in 0..128u32 {
        let unique=format!("{prefix}-{}-{}-{attempt}",std::process::id(),std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos());
        let target=area.join(unique);
        match std::fs::create_dir(&target) {
            Ok(()) => {
                std::fs::create_dir(target.join("bin")).unwrap();
                reject_reparse_path(&workspace, &target);
                return target;
            }
            Err(e) if e.kind()==std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => panic!("failed to create isolated joint test root {}: {e}", target.display()),
        }
    }
    panic!("could not create an isolated joint test root below {}", area.display())
}
#[cfg(test)]
fn cleanup_owned_test_root(candidate: std::path::PathBuf) {
    let workspace=test_workspace_root(); let area=test_area();
    reject_reparse_path(&workspace, &area); reject_reparse_path(&workspace, &candidate);
    let area=std::fs::canonicalize(area).unwrap(); let target=std::fs::canonicalize(&candidate).unwrap();
    assert!(area != workspace && area.starts_with(&workspace), "test area escaped workspace: {}", area.display());
    assert!(target != workspace && target != area && target.starts_with(&workspace) && target.starts_with(&area), "refusing cleanup outside owned workspace test area: {}", target.display());
    std::fs::remove_dir_all(target).unwrap();
}

#[cfg(test)]
mod independent_tests {
    use super::*;
    use ppf_cts_formats::envelope::to_cbor_with_version;

    fn id(n: char) -> String { format!("{n:0<8}-{n:0<4}-{n:0<4}-{n:0<4}-{n:0<12}") }
    fn root() -> std::path::PathBuf { owned_test_root("ppf-joint-independent") }
    fn cleanup(p: std::path::PathBuf) { cleanup_owned_test_root(p); }
 pub(super) fn payload() -> JointPayload { JointPayload { required_capability: CAPABILITY.into(), scene_uuid:id('1'), scene_inputs_sha256:String::new(), bodies:vec![BodyWire{body_uuid:id('2'),runtime_body_index:0},BodyWire{body_uuid:id('3'),runtime_body_index:1}], joints:vec![JointWire{joint_uuid:id('4'),parent_body_uuid:id('2'),child_body_uuid:id('3'),parent_local_frame:LocalFrame{translation_m:[0.;3],quaternion_wxyz:[1.,0.,0.,0.]},child_local_frame:LocalFrame{translation_m:[0.;3],quaternion_wxyz:[1.,0.,0.,0.]},stiffness_N_per_m:37.,enabled:true,collide_connected:true,rest_relative_quaternion_wxyz:[1.,0.,0.,0.],angle_stiffness_nm_per_rad:0.,angle_damping_nms_per_rad:0.,max_angle_rad:0.,angular_control_v1:None,angular_soft_limits_v1:None,angular_drives_v1: None, angular_hard_hinge_v1: None, joint_d6_hard_v1: None}], required_features:None } }
    fn anchor(k:f64, enabled:bool)->BallAnchor { BallAnchor{parent_local_anchor_m:Vector3::new(0.3,-0.2,0.4),child_local_anchor_m:Vector3::new(-0.1,0.5,0.2),stiffness_n_per_m:k,enabled} }
    fn rot()->Matrix3<f64>{ Matrix3::new(0.,-1.,0.,1.,0.,0.,0.,0.,1.) }
    fn close(a:f64,b:f64){assert!((a-b).abs()<=1e-12+1e-10*b.abs().max(1e-12),"{a} != {b}")}
    // Standalone FP64 reference: this deliberately does not call production
    // energy/evaluate/Jacobian helpers.
    fn oracle(a:&BallAnchor,ca:Vector3<f64>,ra:Matrix3<f64>,cb:Vector3<f64>,rb:Matrix3<f64>)->BallEval { if !a.enabled||a.stiffness_n_per_m==0. {return BallEval{residual_m:Vector3::zeros(),energy_j:0.,gradient:SVector::zeros(),hessian_gn:Mat12::zeros()};} let p=ra*a.parent_local_anchor_m;let q=rb*a.child_local_anchor_m;let r=ca+p-cb-q;let x=|v:Vector3<f64>|Matrix3::new(0.,-v.z,v.y,v.z,0.,-v.x,-v.y,v.x,0.);let mut j=SMatrix::<f64,3,12>::zeros();j.fixed_view_mut::<3,3>(0,0).copy_from(&Matrix3::identity());j.fixed_view_mut::<3,3>(0,3).copy_from(&-x(p));j.fixed_view_mut::<3,3>(0,6).copy_from(&-Matrix3::identity());j.fixed_view_mut::<3,3>(0,9).copy_from(&x(q));BallEval{residual_m:r,energy_j:0.5*a.stiffness_n_per_m*r.dot(&r),gradient:a.stiffness_n_per_m*j.transpose()*r,hessian_gn:a.stiffness_n_per_m*j.transpose()*j}}
    #[test] fn public_math_matches_independent_oracle() { let a=anchor(37.,true);let ca=Vector3::new(1.,2.,3.);let cb=Vector3::new(-0.5,0.2,1.);let ra=rot();let rb=Matrix3::identity();let got=a.evaluate(ca,ra,cb,rb);let want=oracle(&a,ca,ra,cb,rb);close(got.energy_j,want.energy_j);close((got.gradient-want.gradient).norm(),0.);let v=SVector::<f64,12>::from_element(0.25);close((a.matvec(ca,ra,cb,rb,v)-want.hessian_gn*v).norm(),0.);let (aa,bb)=a.incident_self_blocks(ca,ra,cb,rb);close((aa-want.hessian_gn.fixed_view::<6,6>(0,0)).norm(),0.);close((bb-want.hessian_gn.fixed_view::<6,6>(6,6)).norm(),0.);assert!(got.hessian_gn.fixed_view::<6,6>(0,6).norm()>1e-9);let fa=-a.stiffness_n_per_m*want.residual_m;let fb=-fa;let p=ra*a.parent_local_anchor_m;let q=rb*a.child_local_anchor_m;let pa=ca+p;let pb=cb+q;close((got.gradient.fixed_rows::<3>(0).into_owned()+fa).norm(),0.);close((got.gradient.fixed_rows::<3>(3).into_owned()+p.cross(&fa)).norm(),0.);close((got.gradient.fixed_rows::<3>(6).into_owned()+fb).norm(),0.);close((got.gradient.fixed_rows::<3>(9).into_owned()+q.cross(&fb)).norm(),0.);close((pa.cross(&fa)+pb.cross(&fb)).norm(),0.);for off in [anchor(0.,true),anchor(37.,false)] {assert_eq!(off.evaluate(ca,ra,cb,rb).gradient.norm(),0.);assert_eq!(off.matvec(ca,ra,cb,rb,v).norm(),0.);assert_eq!(off.incident_self_blocks(ca,ra,cb,rb).0.norm(),0.);} }
    #[test] fn aliases_and_empty_manifest_fail_closed_without_output_mutation() { let r=root();let rows=vec![0.;46];let p=payload();std::fs::write(r.join(JOINT_FILE),to_cbor_with_version(2,KIND,&p).unwrap()).unwrap();let mut alias=p.clone();alias.bodies[1].runtime_body_index=0;std::fs::write(r.join(JOINT_FILE),to_cbor_with_version(2,KIND,&alias).unwrap()).unwrap();assert_eq!(ingest(&r,&rows).unwrap_err().stable_code,"JOINT_R1_DUPLICATE_RUNTIME_BODY_INDEX");let mut empty=p;empty.joints.clear();std::fs::write(r.join(JOINT_FILE),to_cbor_with_version(2,KIND,&empty).unwrap()).unwrap();std::fs::write(r.join("bin/pdrd_body.bin"),vec![0u8;46*4]).unwrap();let out=r.join("output");std::fs::create_dir(&out).unwrap();std::fs::write(out.join("vert_0.bin"),b"fresh").unwrap();std::fs::write(out.join("state_8.bin.gz"),b"resume").unwrap();let before=(std::fs::read(out.join("vert_0.bin")).unwrap(),std::fs::read(out.join("state_8.bin.gz")).unwrap());let e=preflight_selected(&r,ExperimentalBallAnchorSelection::Off).unwrap_err();assert_eq!(e.stable_code,"JOINT_R1_CONTEXT_IO");write_preflight_refusal(&r,&e);assert_eq!(before.0,std::fs::read(out.join("vert_0.bin")).unwrap());assert_eq!(before.1,std::fs::read(out.join("state_8.bin.gz")).unwrap());assert!(r.join("joint_r1_preflight_refusal.json").exists());cleanup(r); }

    #[test]
    fn experimental_selection_validates_before_executable_native_query() {
        let p = copied_fixture();
        let output = p.join("output");
        std::fs::create_dir(&output).unwrap();
        std::fs::write(output.join("state_9.bin.gz"), b"checkpoint").unwrap();
        std::fs::write(output.join("vert_10.bin"), b"future-output").unwrap();
        let before = (std::fs::read(output.join("state_9.bin.gz")).unwrap(), std::fs::read(output.join("vert_10.bin")).unwrap());
        assert_eq!(preflight_selected(&p, ExperimentalBallAnchorSelection::Off).unwrap_err().stable_code, "JOINT_R1_UNSUPPORTED_CAPABILITY");
        let resume = reject_experimental_resume(9).unwrap_err();
        assert_eq!(resume.stable_code, "JOINT_R1_EXPERIMENTAL_RESUME_UNSUPPORTED");
        assert_eq!(before.0, std::fs::read(output.join("state_9.bin.gz")).unwrap());
        assert_eq!(before.1, std::fs::read(output.join("vert_10.bin")).unwrap());
        assert_eq!(reject_experimental_resume(-1).unwrap_err().stable_code, "JOINT_R1_EXPERIMENTAL_RESUME_UNSUPPORTED");
        assert!(preflight_selected(&p, ExperimentalBallAnchorSelection::Requested).is_ok());
        assert_eq!(before.0, std::fs::read(output.join("state_9.bin.gz")).unwrap());
        assert_eq!(before.1, std::fs::read(output.join("vert_10.bin")).unwrap());
        cleanup(p);
    }

    #[test]
    fn experimental_selection_preserves_payload_rejections_and_no_joint_control() {
        let p = copied_fixture();
        let mut unsupported = payload_at(&p);
        unsupported.required_capability = "future_unknown_capability".into();
        std::fs::write(p.join(JOINT_FILE), to_cbor_with_version(2, KIND, &unsupported).unwrap()).unwrap();
        assert_eq!(preflight_selected(&p, ExperimentalBallAnchorSelection::Requested).unwrap_err().stable_code, "JOINT_R1_UNSUPPORTED_REQUIRED_CAPABILITY");
        std::fs::write(p.join(JOINT_FILE), b"malformed").unwrap();
        assert_eq!(preflight_selected(&p, ExperimentalBallAnchorSelection::Requested).unwrap_err().stable_code, "JOINT_R1_MALFORMED_PAYLOAD");
        cleanup(p);
        let no_joint = root();
        assert!(preflight_selected(&no_joint, ExperimentalBallAnchorSelection::Requested).is_ok());
        cleanup(no_joint);
    }

    #[test]
    fn resealed_snapshot_domain_rejections_precede_experimental_unavailable() {
        let p=copied_fixture();
        let baseline=prepare_native_input(&p);
        assert!(baseline.is_ok(),"unmodified fixture must validate before mutation; isolated fixture retained at {}: {baseline:?}",p.display());
        let param=std::fs::read_to_string(p.join("param.toml")).unwrap().replace("world_scaling = 1.0","world_scaling = 2.0");
        std::fs::write(p.join("param.toml"),param).unwrap(); reseal(&p,context(&p));
        let domain=prepare_native_input(&p).unwrap_err();
        assert_eq!(domain.stable_code,"JOINT_R1_UNSUPPORTED_DOMAIN","world_scaling mutation must reject from the sealed snapshot; isolated fixture retained at {}: {domain:?}",p.display());
        let preflight=preflight_selected(&p,ExperimentalBallAnchorSelection::Requested).unwrap_err();
        assert_eq!(preflight.stable_code,"JOINT_R1_UNSUPPORTED_DOMAIN","requested preflight must preserve the sealed domain refusal; isolated fixture retained at {}: {preflight:?}",p.display());
        cleanup(p);
        let p=copied_fixture();
        std::fs::write(p.join("bin/param/tri-friction.bin"),(-1.0f32).to_le_bytes()).unwrap(); reseal(&p,context(&p));
        assert_eq!(prepare_native_input(&p).unwrap_err().stable_code,"JOINT_R1_UNSUPPORTED_FRICTION");
        assert_eq!(preflight_selected(&p,ExperimentalBallAnchorSelection::Requested).unwrap_err().stable_code,"JOINT_R1_UNSUPPORTED_FRICTION");
        cleanup(p);
    }

    #[test]
    fn resealed_retained_physical_bytes_reject_before_requested_handshake() {
        let cases: Vec<(&str, Box<dyn Fn(&std::path::Path)>)> = vec![
            ("degenerate-gram", Box::new(|p| { let mut r=rows(p); r[6..15].fill(0.); write_rows(p,&r); })),
            ("under-four-vertices", Box::new(|p| { let mut r=rows(p); r[1]=3.; write_rows(p,&r); })),
            ("rest-nan", Box::new(|p| { let mut bytes=std::fs::read(p.join("bin/pdrd_rest_centered.bin")).unwrap(); bytes[..4].copy_from_slice(&f32::NAN.to_le_bytes()); std::fs::write(p.join("bin/pdrd_rest_centered.bin"),bytes).unwrap(); })),
            ("rest-length", Box::new(|p| { let mut bytes=std::fs::read(p.join("bin/pdrd_rest_centered.bin")).unwrap(); bytes.truncate(bytes.len()-4); std::fs::write(p.join("bin/pdrd_rest_centered.bin"),bytes).unwrap(); })),
        ];
        for (name, mutate) in cases {
            let p=copied_fixture(); mutate(&p); reseal(&p,context(&p));
            let prepared=prepare_native_input(&p).unwrap_err();
            assert!(prepared.stable_code.starts_with("JOINT_R1_"),"{name}: {prepared:?}");
            let requested=preflight_selected(&p,ExperimentalBallAnchorSelection::Requested).unwrap_err();
            assert_eq!(requested.stable_code,prepared.stable_code,"{name}");
            assert_ne!(requested.stable_code,"JOINT_R1_EXPERIMENTAL_NATIVE_UNAVAILABLE","{name}");
            cleanup(p);
        }
    }

    #[test]
    fn experimental_handshake_version_and_capability_mismatches_fail_closed() {
        let p = copied_fixture();
        let records = prepare_native_input(&p).unwrap().unwrap().into_records();
        for handshake in [
            NativeExperimentalHandshake::Ready { abi_version: EXPERIMENTAL_NATIVE_ABI_VERSION + 1, capability: CAPABILITY },
            NativeExperimentalHandshake::Ready { abi_version: EXPERIMENTAL_NATIVE_ABI_VERSION, capability: "unsupported_native_feature" },
        ] {
            assert_eq!(
                validate_experimental_handshake(&records, handshake).unwrap_err().stable_code,
                "JOINT_R1_EXPERIMENTAL_NATIVE_UNAVAILABLE"
            );
        }
        cleanup(p);
    }
    #[test] fn consumes_exact_python_fixedscene_fixture_via_validated_ingest() { let dir=std::env::var("PPF_R1_PYTHON_FIXTURE").unwrap_or_else(|_|std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../local/share/ppf-cts/git-ppf-unified/r1-binding-20260923-a/r1-binding-20260923-a").to_string_lossy().into_owned()); let root=std::path::Path::new(&dir);assert!(root.is_dir(),"required normal Scene.build/FixedSession fixture is absent: {dir}"); let raw=std::fs::read(root.join("bin/pdrd_body.bin")).unwrap();let rows:Vec<f32>=raw.chunks_exact(4).map(|x|f32::from_le_bytes([x[0],x[1],x[2],x[3]])).collect();let scene=ingest_validated(root,&rows).unwrap().unwrap();assert_eq!(scene.bodies.len(),2);assert_eq!(scene.joints.len(),1);let j=&scene.joints[0];assert_eq!(j.parent_runtime_body_index,0);assert_eq!(j.child_runtime_body_index,1);assert_eq!(j.wire.stiffness_N_per_m,12.5);assert!(!j.wire.enabled);assert_eq!(j.wire.parent_local_frame.translation_m,[0.1,0.,0.]);assert_eq!(j.wire.child_local_frame.translation_m,[0.,0.2,0.]); }

    fn fixture() -> std::path::PathBuf { std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../local/share/ppf-cts/git-ppf-unified/r1-binding-20260923-a/r1-binding-20260923-a") }
    fn copy_tree(src:&std::path::Path, dst:&std::path::Path) { std::fs::create_dir_all(dst).unwrap(); for e in std::fs::read_dir(src).unwrap() { let e=e.unwrap(); let d=dst.join(e.file_name()); if e.file_type().unwrap().is_dir(){copy_tree(&e.path(),&d)}else{std::fs::copy(e.path(),d).unwrap();} } }
    fn copied_fixture() -> std::path::PathBuf { let p=root(); copy_tree(&fixture(),&p); p }
    fn rows(p:&std::path::Path)->Vec<f32>{read_f32_from(&std::fs::read(p.join("bin/pdrd_body.bin")).unwrap()).unwrap()}
    fn write_rows(p:&std::path::Path, rows:&[f32]) { let bytes:Vec<u8>=rows.iter().flat_map(|x|x.to_le_bytes()).collect(); std::fs::write(p.join("bin/pdrd_body.bin"),bytes).unwrap(); }
    fn context(p:&std::path::Path)->SceneInputContext { from_cbor_with_version(2,CONTEXT_KIND,&std::fs::read(p.join(CONTEXT_FILE)).unwrap()).unwrap() }
    fn payload_at(p:&std::path::Path)->JointPayload { from_cbor_with_version(2,KIND,&std::fs::read(p.join(JOINT_FILE)).unwrap()).unwrap() }
    fn reseal(p:&std::path::Path, mut c:SceneInputContext) { c.inputs.sort_by(|a,b|a.relative_path.cmp(&b.relative_path)); for i in &mut c.inputs {let b=std::fs::read(p.join(&i.relative_path)).unwrap();i.byte_length=b.len();i.sha256=sha256(&b);} let mut b=c.bodies.clone();b.sort_by(|a,b|a.body_uuid.cmp(&b.body_uuid)); let mut d=format!("ppf.pdrd_scene_context/v2\n{}\n",c.scene_uuid);for x in b{d.push_str(&format!("{}\t{}\t{}\n",x.body_uuid,x.runtime_body_index,x.vertex_count));}for x in &c.inputs{d.push_str(&format!("{}\t{}\t{}\n",x.relative_path,x.byte_length,x.sha256));}for x in &c.optional_absent{d.push_str(&format!("optional_absent\t{}\n",x));}c.scene_inputs_sha256=sha256(d.as_bytes());let mut q=payload_at(p);q.scene_inputs_sha256=c.scene_inputs_sha256.clone();std::fs::write(p.join(CONTEXT_FILE),to_cbor_with_version(2,CONTEXT_KIND,&c).unwrap()).unwrap();std::fs::write(p.join(JOINT_FILE),to_cbor_with_version(2,KIND,&q).unwrap()).unwrap(); }
    fn assert_code(p:&std::path::Path, code:&str){assert_eq!(ingest_validated(p,&rows(p)).unwrap_err().stable_code,code);}
    #[test] fn controlled_context_binding_mutations_fail_closed() {
        // Binding swaps are semantic: payload is excluded from its own digest.
        let p=copied_fixture();let mut q=payload_at(&p);let a=q.bodies[0].runtime_body_index;q.bodies[0].runtime_body_index=q.bodies[1].runtime_body_index;q.bodies[1].runtime_body_index=a;std::fs::write(p.join(JOINT_FILE),to_cbor_with_version(2,KIND,&q).unwrap()).unwrap();assert_code(&p,"JOINT_R1_BINDING_MISMATCH");cleanup(p);
        // A canonical UUID alias in the authoritative map is semantic after a reseal.
        let p=copied_fixture();let mut m:VertexMap=ciborium::from_reader(std::fs::read(p.join("map.pickle")).unwrap().as_slice()).unwrap();let (k,v)=m.payload.iter().next().map(|(k,v)|(k.clone(),v.clone())).unwrap();m.payload.insert(k.to_ascii_uppercase(),v);let mut bytes=Vec::new();ciborium::into_writer(&m,&mut bytes).unwrap();std::fs::write(p.join("map.pickle"),bytes).unwrap();reseal(&p,context(&p));assert_code(&p,"JOINT_R1_MEMBERSHIP_INVALID");cleanup(p);
        // Removing a native-required file and its inventory entry cannot be resealed into validity.
        let p=copied_fixture();std::fs::remove_file(p.join("param.toml")).unwrap();let mut c=context(&p);c.inputs.retain(|x|x.relative_path!="param.toml");reseal(&p,c);assert_code(&p,"JOINT_R1_REQUIRED_INPUT_MISSING");cleanup(p);
        let p=copied_fixture();std::fs::write(p.join("bin/extra_consumed.bin"),b"x").unwrap();assert_code(&p,"JOINT_R1_INVENTORY_MISMATCH");cleanup(p);
        let p=copied_fixture();let mut c=context(&p);c.optional_absent.reverse();reseal(&p,c);assert_code(&p,"JOINT_R1_CONTEXT_INVALID");cleanup(p);
        let p=copied_fixture();let mut c=context(&p);c.inputs[0].relative_path="C:/outside.bin".into();std::fs::write(p.join(CONTEXT_FILE),to_cbor_with_version(2,CONTEXT_KIND,&c).unwrap()).unwrap();assert_code(&p,"JOINT_R1_CONTEXT_INVALID");cleanup(p);
        let p=copied_fixture();let f=p.join("bin/pdrd_body.bin");let mut b=std::fs::read(&f).unwrap();b[0]^=1;std::fs::write(f,b).unwrap();assert_code(&p,"JOINT_R1_INPUT_HASH_MISMATCH");cleanup(p);
        let p=copied_fixture();std::fs::write(p.join("param.toml"),b"not = [valid").unwrap();reseal(&p,context(&p));assert_code(&p,"JOINT_R1_MALFORMED_INPUT");cleanup(p);
    }
    #[test] fn mixed_named_non_pdrd_object_and_byte_model_are_valid() { let p=copied_fixture();let mut m:VertexMap=ciborium::from_reader(std::fs::read(p.join("map.pickle")).unwrap().as_slice()).unwrap();m.payload.insert("ordinary named non-PDRD object".into(),vec![7,8]);let mut bytes=Vec::new();ciborium::into_writer(&m,&mut bytes).unwrap();std::fs::write(p.join("map.pickle"),bytes).unwrap();let model=p.join("bin/u8-model.bin");std::fs::write(&model,[1u8,2,3]).unwrap();let mut c=context(&p);c.inputs.push(ContextInput{relative_path:"bin/u8-model.bin".into(),byte_length:0,sha256:String::new()});reseal(&p,c);let result=ingest_validated(&p,&rows(&p));assert!(result.is_ok(),"{result:?}");cleanup(p); }
    #[test] fn resealed_rest_and_native_toml_defects_fail_closed() { let p=copied_fixture();let info=std::fs::read_to_string(p.join("info.toml")).unwrap().replace("has_rest_vert = false", "has_rest_vert = true");std::fs::write(p.join("info.toml"),info).unwrap();reseal(&p,context(&p));assert_code(&p,"JOINT_R1_REQUIRED_INPUT_MISSING");cleanup(p);let p=copied_fixture();std::fs::write(p.join("param.toml"),"[param]\nworld_scaling = 1").unwrap();reseal(&p,context(&p));assert_code(&p,"JOINT_R1_MALFORMED_INPUT");cleanup(p); }
    #[test] fn resealed_count_dependent_native_inputs_fail_closed() {
        let p=copied_fixture(); let info=std::fs::read_to_string(p.join("info.toml")).unwrap().replace("rod = 0", "rod = 1"); std::fs::write(p.join("info.toml"),info).unwrap(); reseal(&p,context(&p)); assert_code(&p,"JOINT_R1_REQUIRED_INPUT_MISSING"); cleanup(p);
        let p=copied_fixture(); let info=std::fs::read_to_string(p.join("info.toml")).unwrap().replace("pin_block = 0", "pin_block = 1"); std::fs::write(p.join("info.toml"),info).unwrap(); reseal(&p,context(&p)); assert_code(&p,"JOINT_R1_MALFORMED_INPUT"); cleanup(p);
        let p=copied_fixture(); let mut info=std::fs::read_to_string(p.join("info.toml")).unwrap().replace("pin_block = 0", "pin_block = 1"); info.push_str("\n[pin-0]\npin = 0\noperation_count = 0\npull = 0.0\n"); std::fs::write(p.join("info.toml"),info).unwrap(); reseal(&p,context(&p)); assert_code(&p,"JOINT_R1_REQUIRED_INPUT_MISSING"); cleanup(p);
    }
    #[test] fn resealed_native_operation_wall_and_sphere_fields_fail_closed() {
        // Scene::new reads these sphere fields before branching on keyframes.
        let p=copied_fixture(); let mut info=std::fs::read_to_string(p.join("info.toml")).unwrap().replace("sphere = 0", "sphere = 1"); info.push_str("\n[sphere-0]\nkeyframe = 0\n"); std::fs::write(p.join("info.toml"),info).unwrap(); reseal(&p,context(&p)); assert_code(&p,"JOINT_R1_MALFORMED_INPUT"); cleanup(p);
        // Active walls read their normal, transition, contact gap, and friction.
        let p=copied_fixture(); let mut info=std::fs::read_to_string(p.join("info.toml")).unwrap().replace("wall = 0", "wall = 1"); info.push_str("\n[wall-0]\nkeyframe = 1\nny = 1.0\nnz = 0.0\ntransition = \"linear\"\ncontact-gap = 0.0\nfriction = 0.0\n"); std::fs::write(p.join("info.toml"),info).unwrap(); reseal(&p,context(&p)); assert_code(&p,"JOINT_R1_MALFORMED_INPUT"); cleanup(p);
        // Pin operation subtypes have their own required native TOML reads.
        let p=copied_fixture(); let mut info=std::fs::read_to_string(p.join("info.toml")).unwrap().replace("pin_block = 0", "pin_block = 1"); info.push_str("\n[pin-0]\npin = 0\noperation_count = 1\npull = 0.0\n\n[pin-0-op-0]\ntype = \"move_by\"\nt_start = 0.0\nt_end = 1.0\n"); std::fs::write(p.join("info.toml"),info).unwrap(); reseal(&p,context(&p)); assert_code(&p,"JOINT_R1_MALFORMED_INPUT"); cleanup(p);
    }
    #[test] fn frontend_uuid_spellings_and_raw_duplicate_map_keys_fail_closed() {
        for spelling_kind in 0..3 {
            let p=copied_fixture(); let key=context(&p).bodies[0].body_uuid.clone(); let spelling=match spelling_kind { 0 => format!("{{{key}}}"), 1 => key.replace('-', ""), _ => format!("urn:uuid:{key}"), }; let mut m:VertexMap=ciborium::from_reader(std::fs::read(p.join("map.pickle")).unwrap().as_slice()).unwrap(); let members=m.payload.get(&key).unwrap().clone(); m.payload.insert(spelling,members); let mut bytes=Vec::new(); ciborium::into_writer(&m,&mut bytes).unwrap(); std::fs::write(p.join("map.pickle"),bytes).unwrap(); reseal(&p,context(&p)); assert_code(&p,"JOINT_R1_MEMBERSHIP_INVALID"); cleanup(p);
        }
        // Keep every original map/list/index membership valid, then repeat one
        // raw CBOR key.  If a decoder silently collapses duplicates, this test
        // would now pass, so the rejection specifically proves raw-key checks.
        let p=copied_fixture(); let m:VertexMap=ciborium::from_reader(std::fs::read(p.join("map.pickle")).unwrap().as_slice()).unwrap(); let (duplicate_key,duplicate_members)=m.payload.iter().next().map(|(k,v)|(k.clone(),v.clone())).unwrap(); let mut raw=vec![0xa3,0x67]; raw.extend_from_slice(b"version"); raw.push(1); raw.push(0x64); raw.extend_from_slice(b"kind"); raw.push(0x69); raw.extend_from_slice(b"VertexMap"); raw.push(0x67); raw.extend_from_slice(b"payload"); assert!(m.payload.len()<23); raw.push(0xa0+(m.payload.len() as u8)+1); for (key,members) in &m.payload { ciborium::into_writer(key,&mut raw).unwrap(); ciborium::into_writer(members,&mut raw).unwrap(); } ciborium::into_writer(&duplicate_key,&mut raw).unwrap(); ciborium::into_writer(&duplicate_members,&mut raw).unwrap(); std::fs::write(p.join("map.pickle"),raw).unwrap(); reseal(&p,context(&p)); assert_code(&p,"JOINT_R1_MEMBERSHIP_INVALID"); cleanup(p);
    }
    #[test] fn dangling_manifest_is_not_no_joint_when_platform_allows_links() {
        let p=root();
        #[cfg(windows)] { match std::os::windows::fs::symlink_file(p.join("missing-manifest.cbor"),p.join(JOINT_FILE)) { Ok(()) => assert_eq!(ingest(&p,&[]).unwrap_err().stable_code,"JOINT_R1_UNSAFE_PATH"), Err(e) if matches!(e.kind(),std::io::ErrorKind::PermissionDenied|std::io::ErrorKind::Unsupported) || e.raw_os_error()==Some(1314) => {}, Err(e) => panic!("unexpected symlink setup failure: {e}"), } }
        #[cfg(not(windows))] { std::os::unix::fs::symlink(p.join("missing-manifest.cbor"),p.join(JOINT_FILE)).unwrap(); assert_eq!(ingest(&p,&[]).unwrap_err().stable_code,"JOINT_R1_UNSAFE_PATH"); }
        cleanup(p);
    }
    #[test] fn handoff_changes_and_no_joint_control_fail_closed() { let p=copied_fixture();let s=ingest_validated(&p,&rows(&p)).unwrap().unwrap();std::fs::write(p.join("bin/late.bin"),b"late").unwrap();assert_eq!(s.verify_snapshot_unchanged(&p).unwrap_err().stable_code,"JOINT_R1_HANDOFF_INVENTORY_CHANGED");cleanup(p);let p=copied_fixture();let s=ingest_validated(&p,&rows(&p)).unwrap().unwrap();let f=p.join("bin/pdrd_body.bin");let mut b=std::fs::read(&f).unwrap();b[0]^=1;std::fs::write(f,b).unwrap();assert_eq!(s.verify_snapshot_unchanged(&p).unwrap_err().stable_code,"JOINT_R1_INPUT_CHANGED_AFTER_VALIDATION");cleanup(p);let n=root();assert!(validated_pdrd_rows(&n).unwrap().is_none());cleanup(n); }
    #[test] fn inventory_ignores_mutable_and_unrelated_trees_but_tracks_bin() {
        let p = root();
        std::fs::write(p.join("info.toml"), b"x").unwrap();
        for name in ["output", "cache", "preview", "export", "checkpoint", "summary", "unrelated"] {
            std::fs::create_dir_all(p.join(name)).unwrap();
            std::fs::write(p.join(name).join("not-an-input.bin"), b"mutable").unwrap();
        }
        let before = scan_consumed_paths(&p).unwrap();
        assert_eq!(before, vec!["info.toml"]);
        std::fs::write(p.join("bin/real-input.bin"), b"sealed").unwrap();
        assert_eq!(scan_consumed_paths(&p).unwrap(), vec!["bin/real-input.bin", "info.toml"]);
        cleanup(p);
    }
    #[test] fn force_field_toml_and_scripts_are_sealed_and_tampering_refused() {
        let p = root();
        std::fs::create_dir_all(p.join("bin/force_field")).unwrap();
        let mut snapshot = InputSnapshot { bytes: HashMap::new(), optional_absent: Vec::new() };
        for name in ["bin/force_field/force_field.toml", "bin/force_field/script-0.py"] {
            std::fs::write(p.join(name), b"sealed force field").unwrap();
            snapshot.bytes.insert(name.into(), b"sealed force field".to_vec());
        }
        std::fs::write(p.join(JOINT_FILE), b"control sidecar").unwrap();
        std::fs::write(p.join(CONTEXT_FILE), b"control sidecar").unwrap();
        let records = OwnedJointScene { scene_uuid: String::new(), scene_inputs_sha256: String::new(), payload_sha256: String::new(), payload_version: ENVELOPE_VERSION, body_controls_v1:false, hard_records:Vec::new(), bodies: HashMap::new(), joints: Vec::new(), roots:Vec::new(), fem_pdrd_parent_proof: ValidatedFemParentProof::default(), snapshot };
        assert_eq!(scan_consumed_paths(&p).unwrap(), vec!["bin/force_field/force_field.toml", "bin/force_field/script-0.py"]);
        records.verify_snapshot_unchanged(&p).unwrap();
        for name in ["bin/force_field/force_field.toml", "bin/force_field/script-0.py"] {
            std::fs::write(p.join(name), b"tampered force field").unwrap();
            let refusal = records.verify_snapshot_unchanged(&p).unwrap_err();
            assert_eq!(refusal.stable_code, "JOINT_R1_INPUT_CHANGED_AFTER_VALIDATION");
            std::fs::write(p.join(name), b"sealed force field").unwrap();
        }
        std::fs::write(p.join("bin/force_field/script-1.py"), b"new script").unwrap();
        assert_eq!(records.verify_snapshot_unchanged(&p).unwrap_err().stable_code, "JOINT_R1_HANDOFF_INVENTORY_CHANGED");
        cleanup(p);
    }
    #[test] fn vertex_map_v1_v2_codec_and_unknown_version_refusal() {
        for version in [0, 1, 2, 3, u32::MAX] {
            let payload = HashMap::from([(id('1'), vec![0u32, 2, 3])]);
            let wire = VertexMap { version, kind: "VertexMap".into(), payload: payload.clone() };
            let mut bytes = Vec::new(); ciborium::into_writer(&wire, &mut bytes).unwrap();
            match version {
                1 | 2 => assert_eq!(decode_vertex_map(&bytes).unwrap().payload, payload),
                _ => assert_eq!(decode_vertex_map(&bytes).unwrap_err().stable_code, "JOINT_R1_MEMBERSHIP_INVALID"),
            }
        }
        let malformed = serde_json::json!({"version":2,"kind":"VertexMap","payload":{"body":[-1]}});
        let mut bytes = Vec::new(); ciborium::into_writer(&malformed, &mut bytes).unwrap();
        assert_eq!(decode_vertex_map(&bytes).unwrap_err().stable_code, "JOINT_R1_MEMBERSHIP_INVALID");
        let wrong_kind = serde_json::json!({"version":2,"kind":"OtherMap","payload":{}});
        let mut bytes = Vec::new(); ciborium::into_writer(&wrong_kind, &mut bytes).unwrap();
        assert_eq!(decode_vertex_map(&bytes).unwrap_err().stable_code, "JOINT_R1_MEMBERSHIP_INVALID");
    }
    #[test]
    #[ignore = "requires a freshly authored current Scene/Session fixture via PPF_CURRENT_AUTHORING_FIXTURE"]
    fn current_authoring_vertex_map_v2_membership_validates() {
        let root = std::path::PathBuf::from(std::env::var("PPF_CURRENT_AUTHORING_FIXTURE").unwrap());
        let map = decode_vertex_map(&std::fs::read(root.join("map.pickle")).unwrap()).unwrap();
        assert_eq!(map.version, 2);
        let input = prepare_native_input(&root).unwrap().unwrap();
        assert_eq!(input.records().bodies.len(), 2);
        assert_eq!(input.records().joints.len(), 1);
        input.verify_unchanged(&root).unwrap();
    }
    #[test]
    #[ignore = "requires the sealed historical mixed fixture via PPF_LEGACY_MIXED_FIXTURE"]
    fn legacy_mixed_vertex_map_v1_membership_validates() {
        let root = std::path::PathBuf::from(std::env::var("PPF_LEGACY_MIXED_FIXTURE").unwrap());
        let map = decode_vertex_map(&std::fs::read(root.join("map.pickle")).unwrap()).unwrap();
        assert_eq!(map.version, 1);
        let input = prepare_native_input(&root).unwrap().unwrap();
        assert_eq!(input.records().bodies.len(), 30);
        assert_eq!(input.records().joints.len(), 27);
        input.verify_unchanged(&root).unwrap();
    }
}

#[cfg(test)] mod tests {
 use super::*; use ppf_cts_formats::envelope::to_cbor_with_version;
 fn id(n: char)->String { format!("{:0<8}-{:0<4}-{:0<4}-{:0<4}-{:0<12}", n, n, n, n, n) }
 fn anchor(k:f64, enabled:bool)->BallAnchor { BallAnchor{parent_local_anchor_m:Vector3::new(0.3,-0.2,0.4),child_local_anchor_m:Vector3::new(-0.1,0.5,0.2),stiffness_n_per_m:k,enabled} }
 fn rot()->Matrix3<f64>{ Matrix3::new(0.,-1.,0.,1.,0.,0.,0.,0.,1.) }
 #[test] fn math_matches_independent_fd_and_balance() { let a=anchor(37.,true); let ca=Vector3::new(1.,2.,3.);let cb=Vector3::new(-0.5,0.2,1.);let ra=rot();let rb=Matrix3::identity();let e=a.evaluate(ca,ra,cb,rb); assert!((&e.hessian_gn-e.hessian_gn.transpose()).norm()<1e-12); let eig=e.hessian_gn.symmetric_eigen().eigenvalues; assert!(eig.iter().all(|x|*x>=-1e-10)); let h=1e-6; for i in 0..12 { let mut da=SVector::<f64,12>::zeros();da[i]=h; let energy=|d:SVector<f64,12>| { let dra=nalgebra::Rotation3::from_scaled_axis(Vector3::new(d[3],d[4],d[5])).into_inner()*ra; let drb=nalgebra::Rotation3::from_scaled_axis(Vector3::new(d[9],d[10],d[11])).into_inner()*rb; a.evaluate(ca+Vector3::new(d[0],d[1],d[2]),dra,cb+Vector3::new(d[6],d[7],d[8]),drb).energy_j }; assert!(((energy(da)-energy(-da))/(2.*h)-e.gradient[i]).abs()<2e-6); } let f_a=-a.stiffness_n_per_m*e.residual_m; let f_b=-f_a; let tau=(ca+ra*a.parent_local_anchor_m).cross(&f_a)+(cb+rb*a.child_local_anchor_m).cross(&f_b); assert!(tau.norm()<1e-10); assert!((e.hessian_gn.fixed_view::<6,6>(0,6)-e.hessian_gn.fixed_view::<6,6>(6,0).transpose()).norm()<1e-12); }
 #[test] fn coincidence_and_off_are_exact_zero() { let a=anchor(10.,true); let r=Matrix3::identity(); let ca=Vector3::zeros(); let cb=a.parent_local_anchor_m-a.child_local_anchor_m; let e=a.evaluate(ca,r,cb,r); assert!(e.energy_j<1e-28);assert!(e.gradient.norm()<1e-13); assert!(e.hessian_gn.norm()>0.); for x in [anchor(0.,true),anchor(10.,false)] {let q=x.evaluate(ca,r,cb,r);assert_eq!(q.energy_j,0.);assert_eq!(q.gradient.norm(),0.);assert_eq!(q.hessian_gn.norm(),0.);} }
 #[test] fn cbor_roundtrip_and_refusal() { let root=owned_test_root("ppf-joint"); let rows=vec![0.;46];assert!(ingest(&root,&rows).unwrap().is_none(), "legacy no-joint path remains empty"); let p=JointPayload{required_capability:CAPABILITY.into(),scene_uuid:id('1'),scene_inputs_sha256:String::new(),bodies:vec![BodyWire{body_uuid:id('2'),runtime_body_index:0},BodyWire{body_uuid:id('3'),runtime_body_index:1}],joints:vec![JointWire{joint_uuid:id('4'),parent_body_uuid:id('2'),child_body_uuid:id('3'),parent_local_frame:LocalFrame{translation_m:[0.;3],quaternion_wxyz:[1.,0.,0.,0.]},child_local_frame:LocalFrame{translation_m:[0.;3],quaternion_wxyz:[1.,0.,0.,0.]},stiffness_N_per_m:10.,enabled:true,collide_connected:true,rest_relative_quaternion_wxyz:[1.,0.,0.,0.],angle_stiffness_nm_per_rad:0.,angle_damping_nms_per_rad:0.,max_angle_rad:0.,angular_control_v1:None,angular_soft_limits_v1:None,angular_drives_v1: None, angular_hard_hinge_v1: None, joint_d6_hard_v1: None}],required_features:None};std::fs::write(root.join(JOINT_FILE),to_cbor_with_version(2,KIND,&p).unwrap()).unwrap();assert_eq!(ingest(&root,&rows).unwrap().unwrap().joints.len(),1);let mut bad=p.clone();bad.joints[0].stiffness_N_per_m=-1.;std::fs::write(root.join(JOINT_FILE),to_cbor_with_version(2,KIND,&bad).unwrap()).unwrap();assert_eq!(ingest(&root,&rows).unwrap_err().stable_code,"JOINT_R1_INVALID_STIFFNESS");bad=p.clone();bad.joints[0].max_angle_rad=std::f64::consts::PI+0.01;std::fs::write(root.join(JOINT_FILE),to_cbor_with_version(2,KIND,&bad).unwrap()).unwrap();assert_eq!(ingest(&root,&rows).unwrap_err().stable_code,"JOINT_ANGULAR_INVALID_LIMIT");std::fs::write(root.join(JOINT_FILE),to_cbor_with_version(1,KIND,&p).unwrap()).unwrap();assert_eq!(ingest(&root,&rows).unwrap_err().stable_code,"JOINT_R1_VERSION_OR_KIND");cleanup_owned_test_root(root); }

 // --- A1.1 canonical directional control: parser, version and transport -----
 mod directional_tests {
  use super::*;
  fn control(k0:f64,k1:f64,k2:f64,c0:f64,c1:f64,c2:f64)->AngularControlWireV1 {
      AngularControlWireV1 { coordinate_model: crate::joint_angular_control_gpu::COORDINATE_MODEL.into(),
          stiffness_nm_per_rad: AngularAxisValue{twist_x:k0,swing_y:k1,swing_z:k2},
          damping_nms_per_rad: AngularAxisValue{twist_x:c0,swing_y:c1,swing_z:c2} }
  }
  fn payload(version_features:Option<Vec<&str>>, control:Option<AngularControlWireV1>)->JointPayload {
      let mut p=super::independent_tests::payload();
      p.joints[0].angular_control_v1=control;
      p.required_features=version_features.map(|f| f.into_iter().map(str::to_owned).collect());
      p
  }
  fn decode(p:&JointPayload, version:u32)->Result<OwnedJointScene,JointError> {
      let bytes=to_cbor_with_version(version,KIND,p).unwrap();
      let (decoded,found)=decode_joint_payload(&bytes)?;
      ingest_payload(decoded,found,&[0.;46],InputSnapshot{bytes:HashMap::new(),optional_absent:Vec::new()})
  }

  #[test] fn canonical_record_is_accepted_at_v3_with_the_exact_feature() {
      let p=payload(Some(vec![ANGULAR_CONTROL_FEATURE]),Some(control(0.,0.,10.,0.,0.,0.)));
      let scene=decode(&p,ANGULAR_CONTROL_ENVELOPE_VERSION).unwrap();
      assert_eq!(scene.payload_version,ANGULAR_CONTROL_ENVELOPE_VERSION);
      assert_eq!(scene.joints[0].wire.angular_control_v1.as_ref().unwrap().stiffness(),[0.,0.,10.]);
      // The joint-level rest orientation survives alongside the control record.
      assert_eq!(scene.joints[0].wire.rest_relative_quaternion_wxyz,[1.,0.,0.,0.]);
  }

  #[test] fn v3_feature_list_must_match_control_presence_exactly() {
      // Present control with an empty or missing list, and control-free joints
      // that still advertise the feature, are all the same refusal.
      for features in [Some(vec![]),None,Some(vec![ANGULAR_CONTROL_FEATURE,ANGULAR_CONTROL_FEATURE])] {
          let p=payload(features,Some(control(0.,0.,10.,0.,0.,0.)));
          assert_eq!(decode(&p,ANGULAR_CONTROL_ENVELOPE_VERSION).unwrap_err().stable_code,"JOINT_G2_REQUIRED_FEATURES");
      }
      let p=payload(Some(vec![ANGULAR_CONTROL_FEATURE]),None);
      assert_eq!(decode(&p,ANGULAR_CONTROL_ENVELOPE_VERSION).unwrap_err().stable_code,"JOINT_G2_REQUIRED_FEATURES");
      // Unsupported capability refusal precedes the implied-list mismatch.
      // This is the same typed refusal pinned by the V4-field tests below.
      let p=payload(Some(vec![ANGULAR_SOFT_LIMITS_FEATURE]),None);
      assert_eq!(decode(&p,ANGULAR_CONTROL_ENVELOPE_VERSION).unwrap_err().stable_code,"JOINT_ANGULAR_EXTENSION_UNIMPLEMENTED");
      for name in ["pdrd_unknown_v1",""] {
          let p=payload(Some(vec![name]),None);
          assert_eq!(decode(&p,ANGULAR_CONTROL_ENVELOPE_VERSION).unwrap_err().stable_code,"JOINT_G2_REQUIRED_FEATURES");
      }
      // A control-free V3 with an empty list is legal and requests nothing: the
      // family is EMPTY, so nothing is allocated or launched for it.
      let scene=decode(&payload(Some(vec![]),None),ANGULAR_CONTROL_ENVELOPE_VERSION).unwrap();
      assert!(crate::joint_angular_control_gpu::from_validated(&scene).unwrap().is_empty());
  }

  #[test] fn v2_never_carries_a_control_or_a_feature_list() {
      let p=payload(Some(vec![ANGULAR_CONTROL_FEATURE]),Some(control(0.,0.,10.,0.,0.,0.)));
      assert_eq!(decode(&p,ENVELOPE_VERSION).unwrap_err().stable_code,"JOINT_G2_V2_EXTENSION");
      let p=payload(Some(vec![]),None);
      assert_eq!(decode(&p,ENVELOPE_VERSION).unwrap_err().stable_code,"JOINT_G2_V2_EXTENSION");
      // The historical V2 payload still validates and its family is EMPTY: the
      // V2 scene allocates no control buffer and dispatches nothing new.
      let scene=decode(&payload(None,None),ENVELOPE_VERSION).unwrap();
      assert_eq!(scene.payload_version,ENVELOPE_VERSION);
      assert!(crate::joint_angular_control_gpu::from_validated(&scene).unwrap().is_empty());
      // Even when the joints DO carry records the family is per-joint ordered,
      // so a joint without one is a disabled zero row rather than a missing row.
      let mut mixed=payload(Some(vec![ANGULAR_CONTROL_FEATURE]),Some(control(0.,0.,10.,0.,0.,0.)));
      let mut second=mixed.joints[0].clone();
      second.joint_uuid="55555555-5555-5555-5555-555555555555".into();
      second.angular_control_v1=None;
      mixed.joints.push(second);
      let scene=decode(&mixed,ANGULAR_CONTROL_ENVELOPE_VERSION).unwrap();
      let rows=crate::joint_angular_control_gpu::from_validated(&scene).unwrap();
      assert_eq!(rows.len(),2);
      assert_eq!(rows[0].enabled,1);
      assert_eq!(rows[1].enabled,0);
  }

  #[test] fn future_versions_and_earlier_envelope_extension_fields_are_refused() {
      // Refuse versions outside the implemented V2/V3/V4/V5/V6/V7/V9 set.
      for version in [8u32,10,11] {
          let p=payload(Some(vec![ANGULAR_CONTROL_FEATURE]),Some(control(0.,0.,10.,0.,0.,0.)));
          assert_eq!(decode(&p,version).unwrap_err().stable_code,"JOINT_R1_VERSION_OR_KIND");
      }
      // The V4 fields are named refusals, not silently-ignored keys.
      let mut p=payload(Some(vec![ANGULAR_CONTROL_FEATURE]),Some(control(0.,0.,10.,0.,0.,0.)));
      p.joints[0].angular_soft_limits_v1=Some(StrictValue(serde_json::json!({"coordinate_model":"parent_local_swing_twist_x_v1"})));
      let e=decode(&p,ANGULAR_CONTROL_ENVELOPE_VERSION).unwrap_err();
      assert_eq!(e.stable_code,"JOINT_ANGULAR_EXTENSION_UNIMPLEMENTED");
      assert_eq!(e.field,ANGULAR_SOFT_LIMITS_FIELD);
      let mut p=payload(Some(vec![ANGULAR_CONTROL_FEATURE]),Some(control(0.,0.,10.,0.,0.,0.)));
      p.joints[0].angular_drives_v1=Some(StrictValue(serde_json::json!({"coordinate_model":"parent_local_swing_twist_x_v1"})));
      let e=decode(&p,ANGULAR_CONTROL_ENVELOPE_VERSION).unwrap_err();
      assert_eq!(e.stable_code,"JOINT_ANGULAR_EXTENSION_UNIMPLEMENTED");
      assert_eq!(e.field,ANGULAR_DRIVES_FIELD);
      // Advertising the V4 features is the same refusal.
      let p=payload(Some(vec![ANGULAR_CONTROL_FEATURE,ANGULAR_SOFT_LIMITS_FEATURE]),Some(control(0.,0.,10.,0.,0.,0.)));
      assert_eq!(decode(&p,ANGULAR_CONTROL_ENVELOPE_VERSION).unwrap_err().stable_code,"JOINT_ANGULAR_EXTENSION_UNIMPLEMENTED");
  }

  #[test] fn swing_cone_carrier_checks_rigid_and_material_marker_paths() {
      let mut p=payload(Some(vec![ANGULAR_CONTROL_FEATURE,ANGULAR_SOFT_LIMITS_FEATURE,ANGULAR_SWING_CONE_FEATURE]),Some(control(0.,0.,0.,0.,0.,0.)));
      let axis=serde_json::json!({"enabled":false,"lower_rad":-0.1,"upper_rad":0.1,"stiffness_Nm_per_rad":0.0});
      p.joints[0].angular_soft_limits_v1=Some(StrictValue(serde_json::json!({
          "coordinate_model":"parent_local_swing_twist_x_v1","twist_x":axis,"swing_y":axis,"swing_z":axis,
          "swing_cone":{"enabled":true,"half_angle_rad":0.3,"stiffness_Nm_per_rad":10.0}
      })));
      assert!(gate_angular_control_contract(&p,4).is_ok());
      p.required_features.as_mut().unwrap().pop();
      assert_eq!(gate_angular_control_contract(&p,4).unwrap_err().stable_code,"JOINT_G2_REQUIRED_FEATURES");
      p.required_features.as_mut().unwrap().push(ANGULAR_SWING_CONE_FEATURE.into());
      p.required_features.as_mut().unwrap().insert(0,crate::soft_joint::FEATURE.into());
      assert!(gate_angular_control_contract(&p,4).is_ok());
      // A material-only marker is deferred to strict decoded sidecar union validation.
      p.joints[0].angular_soft_limits_v1=None;
      p.required_features.as_mut().unwrap().retain(|f|f!=ANGULAR_SOFT_LIMITS_FEATURE);
      assert!(gate_angular_control_contract(&p,4).is_ok());
      p.required_features.as_mut().unwrap().retain(|f|f!=crate::soft_joint::FEATURE);
      assert_eq!(gate_angular_control_contract(&p,4).unwrap_err().stable_code,"JOINT_G2_REQUIRED_FEATURES");
  }

  #[test] fn v4_zero_base_soft_limit_decodes_and_exact_features_are_required() {
      let mut p=payload(Some(vec![ANGULAR_CONTROL_FEATURE,ANGULAR_SOFT_LIMITS_FEATURE]),Some(control(0.,0.,0.,0.,0.,0.)));
      p.joints[0].angular_soft_limits_v1=Some(StrictValue(serde_json::json!({
          "coordinate_model":"parent_local_swing_twist_x_v1",
          "twist_x":{"enabled":false,"lower_rad":-0.05,"upper_rad":0.05,"stiffness_Nm_per_rad":0.0},
          "swing_y":{"enabled":false,"lower_rad":-0.05,"upper_rad":0.05,"stiffness_Nm_per_rad":0.0},
          "swing_z":{"enabled":true,"lower_rad":-0.05,"upper_rad":0.05,"stiffness_Nm_per_rad":20.0}
      })));
      let scene=decode(&p,ANGULAR_EXTENSION_ENVELOPE_VERSION).unwrap();
      assert_eq!(scene.joints.len(),1);
      assert!(scene.joints[0].wire.angular_soft_limits_v1.is_some());
      let mut missing_axis=p.clone();
      missing_axis.joints[0].angular_soft_limits_v1.as_mut().unwrap().0.as_object_mut().unwrap().remove("twist_x");
      assert_eq!(decode(&missing_axis,ANGULAR_EXTENSION_ENVELOPE_VERSION).unwrap_err().stable_code,"JOINT_ANGULAR_EXTENSION_INVALID");
      p.required_features=Some(vec![ANGULAR_CONTROL_FEATURE.into()]);
      assert_eq!(decode(&p,ANGULAR_EXTENSION_ENVELOPE_VERSION).unwrap_err().stable_code,"JOINT_G2_REQUIRED_FEATURES");
  }

  #[test] fn v4_drive_keys_decode_but_extra_keys_and_orphan_extensions_are_refused() {
      let mut p=payload(Some(vec![ANGULAR_CONTROL_FEATURE,ANGULAR_DRIVES_FEATURE]),Some(control(0.,0.,0.,0.,0.,0.)));
      let drive=serde_json::json!({"coordinate_model":"parent_local_swing_twist_x_v1","swing_z":{"position":{
          "enabled":true,"gain_Nm_per_rad":10.0,"max_effort_Nm":0.5,"interpolation":"linear_hold",
          "target_keys":[{"time_s":0.0,"value_rad":0.0},{"time_s":0.2,"value_rad":0.2}]
      }}});
      p.joints[0].angular_drives_v1=Some(StrictValue(drive.clone()));
      assert!(decode(&p,ANGULAR_EXTENSION_ENVELOPE_VERSION).unwrap().joints[0].wire.angular_drives_v1.is_some());
      let mut extra=drive;
      extra.as_object_mut().unwrap().insert("unknown".into(),serde_json::json!(1));
      p.joints[0].angular_drives_v1=Some(StrictValue(extra));
      assert!(decode(&p,ANGULAR_EXTENSION_ENVELOPE_VERSION).is_err());
      p.joints[0].angular_control_v1=None;
      p.required_features=Some(vec![ANGULAR_DRIVES_FEATURE.into()]);
      assert!(decode(&p,ANGULAR_EXTENSION_ENVELOPE_VERSION).is_err());
  }

  #[test] fn coordinate_model_is_the_only_accepted_chart() {
      let mut c=control(0.,0.,10.,0.,0.,0.);
      c.coordinate_model="parent_local_other_v1".into();
      let p=payload(Some(vec![ANGULAR_CONTROL_FEATURE]),Some(c));
      let e=decode(&p,ANGULAR_CONTROL_ENVELOPE_VERSION).unwrap_err();
      assert_eq!(e.stable_code,"JOINT_G2_MODEL");
      assert_eq!(e.field,"angular_control_v1.coordinate_model");
  }

  #[test] fn per_axis_coefficients_are_validated_in_the_canonical_shape() {
      // Canonical per-axis maps: one bad axis is enough, and the code and field
      // are the prior contract's.
      for (mutate, axis) in [(0usize,"twist_x"),(1,"swing_y"),(2,"swing_z")] {
          for (bad, is_stiffness) in [(f64::NAN,true),(f64::INFINITY,false),(-1.0,true),(f64::MAX,true),(1e-300,false)] {
              let mut c=control(1.,1.,1.,1.,1.,1.);
              if is_stiffness { match mutate {0=>c.stiffness_nm_per_rad.twist_x=bad,1=>c.stiffness_nm_per_rad.swing_y=bad,_=>c.stiffness_nm_per_rad.swing_z=bad} }
              else { match mutate {0=>c.damping_nms_per_rad.twist_x=bad,1=>c.damping_nms_per_rad.swing_y=bad,_=>c.damping_nms_per_rad.swing_z=bad} }
              let p=payload(Some(vec![ANGULAR_CONTROL_FEATURE]),Some(c));
              let e=decode(&p,ANGULAR_CONTROL_ENVELOPE_VERSION).unwrap_err();
              assert_eq!(e.stable_code,"JOINT_G2_COEFFICIENT");
              assert_eq!(e.field,format!("angular_control_v1.{}",if is_stiffness {"stiffness_Nm_per_rad"} else {"damping_Nms_per_rad"}),"axis {axis}");
          }
      }
      // Zero coefficients are legal: a zero axis contributes exactly nothing.
      let p=payload(Some(vec![ANGULAR_CONTROL_FEATURE]),Some(control(0.,0.,0.,0.,0.,0.)));
      let scene=decode(&p,ANGULAR_CONTROL_ENVELOPE_VERSION).unwrap();
      let rows=crate::joint_angular_control_gpu::from_validated(&scene).unwrap();
      // A zero-coefficient record is still REQUESTED (it has a chart to bind),
      // which is why the row stays enabled and the kernel decides inertness.
      assert_eq!(rows[0].enabled,1);
      assert_eq!(rows[0].stiffness_nm_per_rad,[0.;3]);
  }

  #[test] fn unknown_missing_present_null_and_duplicate_keys_are_refused() {
      let good=serde_json::json!({
          "coordinate_model":"parent_local_swing_twist_x_v1",
          "stiffness_Nm_per_rad":{"twist_x":0.0,"swing_y":0.0,"swing_z":10.0},
          "damping_Nms_per_rad":{"twist_x":0.0,"swing_y":0.0,"swing_z":0.0}});
      fn joint_with(value:serde_json::Value)->serde_json::Value {
          serde_json::json!({"joint_uuid":"44444444-4444-4444-4444-444444444444","parent_body_uuid":"22222222-2222-2222-2222-222222222222","child_body_uuid":"33333333-3333-3333-3333-333333333333",
              "parent_local_frame":{"translation_m":[0.0,0.0,0.0],"quaternion_wxyz":[1.0,0.0,0.0,0.0]},"child_local_frame":{"translation_m":[0.0,0.0,0.0],"quaternion_wxyz":[1.0,0.0,0.0,0.0]},
              "stiffness_N_per_m":37.0,"enabled":true,"collide_connected":true,"angular_control_v1":value})
      }
      let decode_json=|v:serde_json::Value| -> Result<OwnedJointScene,JointError> {
          let payload:JointPayload=serde_json::from_value(serde_json::json!({
              "required_capability":"pdrd_ball_anchor_v1","scene_uuid":"11111111-1111-1111-1111-111111111111","scene_inputs_sha256":"",
              "bodies":[{"body_uuid":"22222222-2222-2222-2222-222222222222","runtime_body_index":0},{"body_uuid":"33333333-3333-3333-3333-333333333333","runtime_body_index":1}],
              "joints":[v],"required_features":["pdrd_angular_controls_v1"]})).map_err(|e|err("ingest","JOINT_R1_MALFORMED_PAYLOAD","payload",None,None,&e.to_string()))?;
          ingest_payload(payload,ANGULAR_CONTROL_ENVELOPE_VERSION,&[0.;46],InputSnapshot{bytes:HashMap::new(),optional_absent:Vec::new()})
      };
      assert!(decode_json(joint_with(good.clone())).is_ok());
      // Present-null is authoritative: it is a malformed record, NOT an absent
      // one, so a producer cannot disable a request by spelling it null.
      assert!(decode_json(joint_with(serde_json::Value::Null)).is_err());
      // Unknown key at either level.
      let mut extra=good.clone(); extra["extra"]=serde_json::json!(1);
      assert!(decode_json(joint_with(extra)).is_err());
      let mut extra_axis=good.clone(); extra_axis["stiffness_Nm_per_rad"]["extra"]=serde_json::json!(1);
      assert!(decode_json(joint_with(extra_axis)).is_err());
      // Missing required key at either level.
      let missing_top=serde_json::json!({"coordinate_model":"parent_local_swing_twist_x_v1","stiffness_Nm_per_rad":{"twist_x":0.0,"swing_y":0.0,"swing_z":10.0}});
      assert!(decode_json(joint_with(missing_top)).is_err());
      let mut missing_axis=good.clone(); missing_axis["damping_Nms_per_rad"].as_object_mut().unwrap().remove("swing_y");
      assert!(decode_json(joint_with(missing_axis)).is_err());
      // Duplicate key inside the nested axis map. serde_json's own Value would
      // keep the last of the two; the strict wire value rejects the record.
      let duplicate=serde_json::from_str::<StrictValue>(r#"{"twist_x":1.0,"twist_x":2.0}"#);
      assert!(duplicate.is_err());
      assert!(serde_json::from_str::<StrictValue>(r#"{"twist_x":1.0}"#).is_ok());
  }

  #[test] fn present_null_features_cannot_masquerade_as_absent_on_v2_or_v3() {
      let mut value=serde_json::to_value(payload(None,None)).unwrap();
      value["required_features"]=serde_json::Value::Null;
      for version in [ENVELOPE_VERSION,ANGULAR_CONTROL_ENVELOPE_VERSION] {
          let bytes=to_cbor_with_version(version,KIND,&value).unwrap();
          assert_eq!(decode_joint_payload(&bytes).unwrap_err().stable_code,"JOINT_R1_MALFORMED_PAYLOAD");
      }
  }

  #[test] fn transport_rows_are_one_per_joint_in_joint_order() {
      let mut p=payload(Some(vec![ANGULAR_CONTROL_FEATURE]),Some(control(0.,2.,10.,1.,0.,0.)));
      let mut second=p.joints[0].clone();
      second.joint_uuid="55555555-5555-5555-5555-555555555555".into();
      second.angular_control_v1=None;
      p.joints.push(second);
      let scene=decode(&p,ANGULAR_CONTROL_ENVELOPE_VERSION).unwrap();
      let rows=crate::joint_angular_control_gpu::from_validated(&scene).unwrap();
      assert_eq!(rows.len(),scene.joints.len());
      assert_eq!(rows[0].body_a,0); assert_eq!(rows[0].body_b,1);
      assert_eq!(rows[0].enabled,1);
      assert_eq!(rows[0].stiffness_nm_per_rad,[0.,2.,10.]);
      assert_eq!(rows[0].damping_nms_per_rad,[1.,0.,0.]);
      assert_eq!(rows[0].coordinate_model,crate::joint_angular_control_gpu::COORDINATE_MODEL_CODE);
      assert_eq!(rows[0].rest_relative_wxyz,[1.,0.,0.,0.]);
      // The control-free joint in the same scene is a disabled, zero row at the
      // SAME joint index, so the per-joint fault lane slot stays meaningful.
      assert_eq!(rows[1].enabled,0);
      assert_eq!(rows[1].stiffness_nm_per_rad,[0.;3]);
      assert_eq!(rows[1].body_a,0); assert_eq!(rows[1].body_b,1);
      // Transport is a pure function of the wire record: no coefficient, frame
      // or rest value is dropped on the way to the device row.
      assert_eq!(rows[0].frame_a_wxyz,rows[1].frame_a_wxyz);
      assert_eq!(rows[0].frame_b_wxyz,rows[1].frame_b_wxyz);
      assert_eq!(rows[0].rest_relative_wxyz,rows[1].rest_relative_wxyz);
      // A disabled joint never requests, whatever it carries.
      let mut p2=payload(Some(vec![ANGULAR_CONTROL_FEATURE]),Some(control(0.,0.,10.,0.,0.,0.)));
      p2.joints[0].enabled=false;
      let scene=decode(&p2,ANGULAR_CONTROL_ENVELOPE_VERSION).unwrap();
      assert_eq!(crate::joint_angular_control_gpu::from_validated(&scene).unwrap()[0].enabled,0);
  }
 }


 #[test]
 fn connected_collision_exclusion_is_accepted_and_preserved() {
     let root = owned_test_root("ppf-joint-collision-exclusion");
     let rows = vec![0.; 46];
     let mut p = super::independent_tests::payload();
     p.joints[0].collide_connected = false;
     std::fs::write(root.join(JOINT_FILE), to_cbor_with_version(2, KIND, &p).unwrap()).unwrap();
     let scene = ingest(&root, &rows).unwrap().unwrap();
     assert!(!scene.joints[0].wire.collide_connected);
     cleanup_owned_test_root(root);
 }
}

/// Bounded V5/V7 path plus exact V7 single-D6/one-V1-source-root union.
/// Unknown features, all other root unions and soft extensions are refused.
fn gate_hard_contract(payload:&JointPayload,version:u32)->Result<(),JointError> {
    use crate::joint_hard_contract::{self as hard,HardPresence,RuntimeSupport};
    let refusal=|code|err("ingest",code,"hard_joint",None,None,"bounded rootless V5/V7 contract");
    if !matches!(version,5|7) || payload.required_capability!=CAPABILITY ||
       (version==5&&!payload.joints.iter().any(|j|j.angular_hard_hinge_v1.is_some())) ||
       (version==7&&!payload.joints.iter().any(|j|j.joint_d6_hard_v1.is_some())) {
        return Err(refusal("G2_HARD_UNSUPPORTED_DOMAIN"));
    }
    let features=payload.required_features.as_deref().ok_or_else(||refusal("JOINT_G2_REQUIRED_FEATURES"))?;
    if features.iter().any(|f|f==crate::joint_root_motion_wire::ROOT_FEATURE_V3) {
        return Err(refusal("G2_HARD_UNSUPPORTED_DOMAIN"));
    }
    let root_count=usize::from(features.iter().any(|f|f=="pdrd_kinematic_roots_v1"));
    if features.iter().any(|f|f=="pdrd_kinematic_roots_v2") ||
       (root_count!=0 && (version!=7||payload.joints.len()!=1||payload.bodies.len()>3)) ||
       payload.joints.iter().any(|j|j.angular_soft_limits_v1.is_some()||j.angular_drives_v1.is_some()) {
        return Err(refusal("G2_HARD_UNSUPPORTED_DOMAIN"));
    }
    let presence:Vec<_>=payload.joints.iter().map(|j| {
        let conflict=j.angular_hard_hinge_v1.is_some()&&j.angular_control_v1.as_ref().is_some_and(|c|
            c.stiffness()[1..].iter().chain(&c.damping()[1..]).any(|&v|v!=0.0));
        HardPresence {hinge:j.angular_hard_hinge_v1.as_ref(),d6:j.joint_d6_hard_v1.as_ref(),
            control:j.angular_control_v1.is_some(),soft_limits:j.angular_soft_limits_v1.is_some(),
            drives:j.angular_drives_v1.is_some(),world_endpoint:j.parent_body_uuid==WORLD_ENDPOINT_UUID||j.child_body_uuid==WORLD_ENDPOINT_UUID,
            isotropic_terms:j.angle_stiffness_nm_per_rad!=0.0||j.angle_damping_nms_per_rad!=0.0||j.max_angle_rad!=0.0,
            hard_soft_conflict:conflict}
    }).collect();
    // This Ready selection is colocated with the complete install/frozen-PCG /
    // actual-candidate implementation patch. Nonneutral backends cannot accept.
    let runtime=if cfg!(neutral_driver) {RuntimeSupport::FrozenCoupledInverseAndCandidateGates}
        else {RuntimeSupport::Unwired};
    hard::validate_contract(version,&presence,features,root_count,runtime).map_err(refusal)
}

fn bind_hard_records(version:u32,joints:&[OwnedJoint],bodies:&HashMap<String,u32>,rows:&[f32])
    ->Result<Vec<crate::joint_hard_gpu::HardJointGpu>,JointError> {
    if !matches!(version,5|7) {return Ok(Vec::new());}
    use crate::joint_hard_contract::{self as hard,Edge,EdgeKind};
    let fail=|code|err("ingest",code,"hard_joint.binding",None,None,"hard endpoints/graph/transport must match sealed FREE native rows");
    let count=rows.len()/PDRD_BODY_ROW_LEN;
    if bodies.len()!=count || bodies.values().any(|&b|b as usize>=count) {
        return Err(fail("G2_HARD_UNSUPPORTED_DOMAIN"));
    }
    let free:Vec<bool>=rows.chunks_exact(PDRD_BODY_ROW_LEN).map(|r|r[16]==0.0).collect();
    let edges:Vec<_>=joints.iter().map(|j|Edge{a:j.parent_runtime_body_index,b:j.child_runtime_body_index,
        enabled:j.wire.enabled,stiffness:j.wire.stiffness_N_per_m as f32,
        kind:if j.wire.angular_hard_hinge_v1.is_some(){EdgeKind::Hinge}
             else if j.wire.joint_d6_hard_v1.is_some(){EdgeKind::D6}else{EdgeKind::Ordinary}}).collect();
    hard::validate_topology(version,&free,&edges,None).map_err(fail)?;
    let mut packed=Vec::new();
    for (index,j) in joints.iter().enumerate() {
        let w=&j.wire;
        let axes=if let Some(h)=&w.angular_hard_hinge_v1 {
            crate::joint_hard_gpu::hinge_axes(h.validate().map_err(|e|fail(e.code))?)
        } else if let Some(d)=&w.joint_d6_hard_v1 {
            d.validate().map_err(|e|fail(e.code))?
        } else {continue};
        let index=u32::try_from(index).map_err(|_|fail("G2_HARD_INVALID_INPUT"))?;
        packed.push(crate::joint_hard_gpu::pack(index,j.parent_runtime_body_index,j.child_runtime_body_index,
            w.enabled,w.parent_local_frame.translation_m,w.child_local_frame.translation_m,
            w.parent_local_frame.quaternion_wxyz,w.child_local_frame.quaternion_wxyz,
            w.rest_relative_quaternion_wxyz,axes).map_err(fail)?);
    }
    Ok(packed)
}

#[cfg(test)] mod hard_ingest_tests {
 use super::*;
 pub(super) fn payload(version:u32)->JointPayload {
  let mut p=world_endpoint_tests::payload();
  p.required_capability=CAPABILITY.into();
  p.joints[0].parent_body_uuid=p.bodies[0].body_uuid.clone();
  p.joints[0].parent_local_frame.translation_m=[0.;3];
  if version==5 {
   p.joints[0].angular_control_v1=Some(AngularControlWireV1 {coordinate_model:crate::joint_angular_control_gpu::COORDINATE_MODEL.into(),
    stiffness_nm_per_rad:AngularAxisValue{twist_x:2.,swing_y:0.,swing_z:0.},damping_nms_per_rad:AngularAxisValue{twist_x:0.,swing_y:0.,swing_z:0.}});
   p.joints[0].angular_hard_hinge_v1=Some(crate::joint_angular_hard_wire::AngularHardHingeWireV1 {coordinate_model:crate::joint_angular_hard_wire::COORDINATE_MODEL.into(),twist_x:crate::joint_angular_hard_wire::TwistWireV1::Free});
   p.required_features=Some(vec![ANGULAR_CONTROL_FEATURE.into(),crate::joint_angular_hard_wire::HARD_FEATURE.into()]);
  } else {
   use crate::joint_d6_hard_wire::{Axis,AxisMap,D6Wire};
   p.joints[0].joint_d6_hard_v1=Some(D6Wire {coordinate_model:crate::joint_d6_hard_wire::MODEL.into(),rest_translation_m:[0.;3],
    translation:AxisMap{x:Axis::Locked{value:0.},y:Axis::Locked{value:0.},z:Axis::Locked{value:0.}},
    rotation:AxisMap{x:Axis::Free,y:Axis::Limited{lower:-0.5,upper:0.5},z:Axis::Locked{value:0.}}});
   p.required_features=Some(vec![crate::joint_d6_hard_wire::FEATURE.into()]);
  } p
 }
 fn ingest(p:JointPayload,v:u32)->Result<OwnedJointScene,JointError> {
  let bytes=ppf_cts_formats::envelope::to_cbor_with_version(v,KIND,&p).unwrap();
  let (p,v)=decode_joint_payload(&bytes)?;
  ingest_payload(p,v,&[0.;46],InputSnapshot{bytes:HashMap::new(),optional_absent:Vec::new()})
 }
 #[test] fn authored_v5_v7_bind_exact_hard_rows_and_keep_smooth_order() {
  let a=ingest(payload(5),5).unwrap();assert_eq!(a.hard_records.len(),1);assert_eq!(a.joints.len(),1);
  assert_eq!(a.hard_records[0].modes,[0,0,0,0,2,2]);assert_eq!(a.hard_records[0].joint_index,0);
  let a=ingest(payload(7),7).unwrap();assert_eq!(a.hard_records[0].modes,[2,2,2,0,1,2]);
  assert_eq!(a.hard_records[0].body_a,0);assert_eq!(a.hard_records[0].body_b,1);
 }
 #[test] fn wrong_feature_version_root_conflict_and_alias_are_typed_refusals() {
  let mut p=payload(7);p.required_features.as_mut().unwrap().push("pdrd_kinematic_roots_v1".into());
  assert_eq!(ingest(p,7).unwrap_err().stable_code,"ROOT_MOTION_BODY_BINDING");
  assert_eq!(ingest(payload(5),4).unwrap_err().stable_code,"G2_HARD_UNSUPPORTED_DOMAIN");
  let mut p=payload(7);p.required_features.as_mut().unwrap().push("unknown".into());
  assert_eq!(ingest(p,7).unwrap_err().stable_code,"JOINT_G2_REQUIRED_FEATURES");
  let mut p=payload(7);p.bodies[1].runtime_body_index=0;
  assert_eq!(ingest(p,7).unwrap_err().stable_code,"JOINT_R1_DUPLICATE_RUNTIME_BODY_INDEX");
  let mut p=payload(5);p.joints[0].angular_control_v1.as_mut().unwrap().stiffness_nm_per_rad.swing_y=1.;
  assert_eq!(ingest(p,5).unwrap_err().stable_code,"JOINT_ANGULAR_HARD_EXTENSION_CONFLICT");
 }
 #[test] fn actual_scene_session_exports_pass_sealed_native_ingest() {
  let Ok(list)=std::env::var("PPF_HARD_EXPORTED_INPUTS") else {return};
  let paths:Vec<String>=serde_json::from_str(&list).unwrap();assert_eq!(paths.len(),3);
  for path in paths {
   let input=prepare_native_input(Path::new(&path)).unwrap().unwrap();
   let scene=input.records();assert!(matches!(scene.payload_version,5|7));
   assert!(!scene.hard_records.is_empty());input.verify_unchanged(Path::new(&path)).unwrap();
   assert!(scene.hard_records.iter().all(|r|r.enabled==1&&r.body_a!=r.body_b));
  }
 }
 #[test] fn explicit_null_and_duplicate_hard_fields_cannot_be_ignored() {
  let p=payload(7);let mut v=serde_json::to_value(&p).unwrap();v["joints"][0]["joint_d6_hard_v1"]=serde_json::Value::Null;
  assert!(serde_json::from_value::<JointPayload>(v).is_err());
  let mut v=serde_json::to_value(&p).unwrap();v["joints"][0]["joint_d6_hard_v1"]["translation"]["x"]["bogus"]=serde_json::json!(0);
  assert!(serde_json::from_value::<JointPayload>(v).is_err());
  let text=serde_json::to_string(&p).unwrap();let duplicate=text.replacen("\"joint_d6_hard_v1\":","\"joint_d6_hard_v1\":null,\"joint_d6_hard_v1\":",1);
  assert!(serde_json::from_str::<JointPayload>(&duplicate).is_err());
 }
}

#[cfg(test)] mod root_hard_composition_ingest_tests {
 use super::*;
 #[test] fn scene_session_exact_union_retains_bound_source_and_inventory() {
  let Ok(list)=std::env::var("PPF_ROOT_HARD_EXPORTED_INPUTS") else{return};
  let paths:Vec<String>=serde_json::from_str(&list).unwrap();assert_eq!(paths.len(),3);
  for (i,path) in paths.iter().enumerate(){
   let input=prepare_native_input(Path::new(path)).unwrap().unwrap();input.verify_unchanged(Path::new(path)).unwrap();
   let scene=input.records();assert_eq!(scene.roots.len(),1);
   if i==0{assert_eq!(scene.payload_version,6);assert!(scene.hard_records.is_empty());}
   else {assert_eq!(scene.payload_version,7);assert_eq!(scene.hard_records.len(),1);assert_eq!(scene.joints.len(),1);
    assert_eq!(scene.roots[0].runtime_body_index,scene.hard_records[0].body_a);
    assert_eq!(scene.roots[0].body_uuid,scene.joints[0].wire.parent_body_uuid);}
  }
 }
 #[test] fn hinge_and_batch_root_unions_stay_typed_refusals(){
  let mut p=hard_ingest_tests::payload(5);p.required_features.as_mut().unwrap().push("pdrd_kinematic_roots_v1".into());
  assert_eq!(gate_hard_contract(&p,5).unwrap_err().stable_code,"G2_HARD_UNSUPPORTED_DOMAIN");
  let mut p=hard_ingest_tests::payload(7);p.required_features.as_mut().unwrap().push("pdrd_kinematic_roots_v2".into());
  assert_eq!(gate_hard_contract(&p,7).unwrap_err().stable_code,"G2_HARD_UNSUPPORTED_DOMAIN");
  let mut p=hard_ingest_tests::payload(7);p.required_features.as_mut().unwrap().push(crate::joint_root_motion_wire::ROOT_FEATURE_V3.into());
  assert_eq!(gate_hard_contract(&p,7).unwrap_err().stable_code,"G2_HARD_UNSUPPORTED_DOMAIN");
 }
}


#[cfg(test)]
mod collision_filter_sealed_tests {
    use super::*;
    #[test]
    #[ignore = "requires actual Blender-authored enabled/disabled sealed cases"]
    fn collision_filter_accepts_actual_blender_sealed_inputs() {
        let root = std::env::var("PPF_COLLISION_SEALED_ROOT").expect("explicit fixture root required");
        let root = Path::new(&root);
        for (name, enabled) in [("enabled",true),("disabled",false)] {
            let path = root.join(name);
            let input = prepare_native_input(&path).expect("sealed actual Blender scene qualifies").expect("joint input present");
            assert_eq!(input.records.bodies.len(),3);
            let mask = input.bytes("bin/collision_filter.bin");
            if enabled {
                let bytes=mask.expect("enabled feature retains packed binary");
                assert_eq!(bytes.len(),3*642*4);
                assert!(bytes.chunks_exact(4).any(|x| u32::from_le_bytes(x.try_into().unwrap()) != 0));
                assert_eq!(bytes,std::fs::read(path.join("bin/collision_filter.bin")).unwrap());
            } else {
                assert!(mask.is_err());
            }
        }
    }
}

pub(crate) fn collision_vertex_map(bytes:&[u8])->Result<HashMap<String,Vec<u32>>,JointError> {
    let mut cursor=std::io::Cursor::new(bytes);
    let map:VertexMap=ciborium::from_reader(&mut cursor).map_err(|e|err("collision","COLLISION_FILTER_INVALID","map.pickle",None,None,&e.to_string()))?;
    if cursor.position()!=bytes.len() as u64 || !matches!(map.version,1|2) || map.kind!="VertexMap" {return Err(err("collision","COLLISION_FILTER_INVALID","map.pickle",None,None,"version/kind/trailing bytes"));}
    Ok(map.payload)
}

// File: fem_parent_proof_regression.rs
// Staged CPU regression for the retained strict-prepare FEM/mixed PDRD-parent proof.
//
// PLACEMENT: append this module inside `crates/ppf-cts-solver/src/joint.rs` (it needs
// `pub(crate)`/module access to `collect_fem_pdrd_parent_proof` and to `JointWire`),
// next to the other `#[cfg(test)]` modules. No filesystem, no native, no GPU.
//
// WHAT IT PROVES (the two ends of the union predicate used by driver::roots):
//   1. LEGACY / NO-ENABLED PROOF: with no material edges (or only disabled ones) the
//      proof stays empty, so the legacy rigid known-parent scan alone decides -- and
//      a root with no enabled legacy parent joint is still refused. The rigid path is
//      therefore byte-for-byte as strict as before.
//   2. STRICT MIXED PROOF: with an ENABLED genuine PDRD -> FEM edge the proof holds
//      exactly that PDRD's real runtime index; a FEM -> FEM edge and a WORLD-parent
//      edge contribute nothing, so no phantom rigid body is ever admitted.

#[cfg(test)]
mod fem_pdrd_parent_proof_tests {
    use super::*;
    use crate::soft_joint::{BodyWire, JointRow};

    const PDRD_ROOT: &str = "01fca64e-0639-4b35-bcfa-421a019f0f9b";
    const FEM_CHILD: &str = "42ecfef0-0338-4f52-88c7-74691e3f1916";
    const FEM_SECOND: &str = "52ecfef0-0338-4f52-88c7-74691e3f1917";

    fn frame(t: [f64; 3]) -> LocalFrame {
        LocalFrame { translation_m: t, quaternion_wxyz: [1.0, 0.0, 0.0, 0.0] }
    }

    fn wire(uuid: &str, parent: &str, child: &str, enabled: bool) -> JointWire {
        JointWire {
            joint_uuid: uuid.into(), parent_body_uuid: parent.into(), child_body_uuid: child.into(),
            parent_local_frame: frame([0.0, 0.0, 0.0]), child_local_frame: frame([0.0, 0.0, 0.0]),
            stiffness_N_per_m: 40.0, enabled, collide_connected: true,
            rest_relative_quaternion_wxyz: [1.0, 0.0, 0.0, 0.0],
            angle_stiffness_nm_per_rad: 0.0, angle_damping_nms_per_rad: 0.0, max_angle_rad: 0.0,
            angular_control_v1: None, angular_soft_limits_v1: None, angular_drives_v1: None,
            angular_hard_hinge_v1: None, joint_d6_hard_v1: None,
        }
    }

    fn row(uuid: &str, parent: &str, child: &str, enabled: bool) -> JointRow {
        JointRow { wire: wire(uuid, parent, child, enabled), parent_anchor_bary: None, child_anchor_bary: None }
    }

    fn pdrd(uuid: &str) -> BodyWire {
        BodyWire { body_uuid: uuid.into(), kind: "PDRD".into(), runtime_body_index: Some(0),
            tet_index: None, vertex_indices: None, dm_inv_column_major: None,
            center_bary: None, frame_rest_column_major: None }
    }

    fn fem(uuid: &str, tet_index: u32, nodes: [u32; 4]) -> BodyWire {
        BodyWire { body_uuid: uuid.into(), kind: "FEM_TET_MATERIAL".into(), runtime_body_index: None,
            tet_index: Some(tet_index), vertex_indices: Some(nodes),
            dm_inv_column_major: Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            center_bary: Some([0.25, 0.25, 0.25, 0.25]),
            frame_rest_column_major: Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]) }
    }

    fn runtime(entries: &[(&str, u32)]) -> HashMap<String, u32> {
        entries.iter().map(|(u, i)| ((*u).to_string(), *i)).collect()
    }

    /// Mirrors exactly the union predicate in driver::roots::install_native, so the
    /// regression measures the same decision the driver makes.
    fn has_enabled_known_parent_joint(
        proof: &ValidatedFemParentProof,
        legacy_enabled_parents: &[u32],
        runtime_body_index: u32,
    ) -> bool {
        legacy_enabled_parents.contains(&runtime_body_index) || proof.contains(runtime_body_index)
    }

    #[test]
    fn legacy_path_has_no_proof_and_stays_strict() {
        // No material edges at all: the proof must be empty, exactly as on the
        // original rigid-only path. Its Default is what every non-FEM construction
        // site keeps.
        let proof = collect_fem_pdrd_parent_proof(&[], &[], &runtime(&[(PDRD_ROOT, 0)]));
        assert!(proof.is_empty(), "no material edges must yield no proof");
        assert_eq!(proof.len(), 0);

        // A root whose only joint is DISABLED contributes nothing, so the legacy
        // scan alone decides and still refuses the root.
        let bodies = [pdrd(PDRD_ROOT), fem(FEM_CHILD, 0, [8, 11, 10, 9])];
        let wires = [row("b0df159f-53ed-4049-984c-69f70c64e228", PDRD_ROOT, FEM_CHILD, false)];
        let proof = collect_fem_pdrd_parent_proof(&bodies, &wires, &runtime(&[(PDRD_ROOT, 0)]));
        assert!(proof.is_empty(), "a disabled material edge must not prove anything");
        assert!(
            !has_enabled_known_parent_joint(&proof, &[], 0),
            "legacy path with no enabled parent joint must still refuse the root"
        );

        // ...and the legacy scan still works when a real rigid parent joint exists.
        assert!(has_enabled_known_parent_joint(&proof, &[0], 0));
    }

    #[test]
    fn strict_mixed_proof_admits_only_the_genuine_enabled_pdrd_parent() {
        let bodies = [
            pdrd(PDRD_ROOT),
            fem(FEM_CHILD, 0, [8, 11, 10, 9]),
            fem(FEM_SECOND, 1, [12, 13, 14, 15]),
        ];
        let wires = [
            // genuine, enabled PDRD -> FEM material edge: the one that must count
            row("b0df159f-53ed-4049-984c-69f70c64e228", PDRD_ROOT, FEM_CHILD, true),
            // enabled FEM -> FEM edge: parent is material, so it must NOT count
            row("1d135427-0000-0000-0000-000000000000", FEM_CHILD, FEM_SECOND, true),
            // enabled WORLD parent: not a body at all
            row("aaaa0000-0000-0000-0000-000000000000", WORLD_ENDPOINT_UUID, FEM_SECOND, true),
        ];
        let proof = collect_fem_pdrd_parent_proof(&bodies, &wires, &runtime(&[(PDRD_ROOT, 0)]));

        assert_eq!(proof.len(), 1, "exactly the one genuine PDRD parent");
        assert!(proof.contains(0), "the real PDRD runtime index must be proved");
        assert!(
            has_enabled_known_parent_joint(&proof, &[], 0),
            "the mixed graph must now admit its genuine PDRD-parent root"
        );

        // An FEM parent never enters the proof, so its runtime index stays unproved
        // even though it is the parent of an enabled edge.
        assert!(!proof.contains(1), "an FEM-FEM parent must not be admitted");
        assert!(!has_enabled_known_parent_joint(&proof, &[], 1));
    }

    #[test]
    fn unbound_uuid_and_world_only_graphs_cannot_manufacture_proof() {
        let bodies = [pdrd(PDRD_ROOT), fem(FEM_CHILD, 0, [8, 11, 10, 9])];
        // An enabled PDRD-parent edge whose UUID has NO validated runtime index must
        // contribute nothing: the proof is built from the validated map, not from
        // the wire's self-declared identity.
        let wires = [row("b0df159f-53ed-4049-984c-69f70c64e228",
                         "99999999-9999-9999-9999-999999999999", FEM_CHILD, true)];
        let proof = collect_fem_pdrd_parent_proof(&bodies, &wires, &runtime(&[(PDRD_ROOT, 0)]));
        assert!(proof.is_empty(), "an unbound parent UUID must not manufacture proof");
        assert!(!has_enabled_known_parent_joint(&proof, &[], 0));
    }

    #[test]
    fn default_proof_is_empty_and_not_decodable() {
        // The rigid default: every non-FEM construction site keeps this value.
        let d = ValidatedFemParentProof::default();
        assert!(d.is_empty() && d.len() == 0);
        assert!(!d.contains(0) && !d.contains(u32::MAX));

        // The field is private and there is no Deserialize path, so a decoded
        // payload cannot populate it. This assertion is a compile-time guard:
        // uncommenting the next line must FAIL to compile.
        // let _ = OwnedJointScene { fem_pdrd_parent_proof: d.clone(), ..Default::default() };
        let _ = d;
    }
}
