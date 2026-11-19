//! D03 root-motion sidecar wire and pure CPU sampler.
//!
//! This module deliberately has no native ABI or host-crate dependencies. The
//! eventual V6 ingest can deserialize the typed envelope directly from CBOR;
//! converting through a generic map would erase duplicate-key evidence.

use serde::{de, Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;
use std::io::Cursor;

pub const ROOT_FEATURE: &str = "pdrd_kinematic_roots_v1";
pub const ROOT_KIND: &str = "ppf.pdrd_root_motion";
pub const INTERPOLATION: &str = "linear_center_shortest_slerp_v1";
pub const EXTRAPOLATION: &str = "hold";
pub const CODE_VERSION_OR_KIND: &str = "ROOT_MOTION_VERSION_OR_KIND";
pub const CODE_REQUIRED_FEATURES: &str = "ROOT_MOTION_REQUIRED_FEATURES";
pub const CODE_SIDECAR_MISSING: &str = "ROOT_MOTION_SIDECAR_MISSING";
pub const CODE_BODY_BINDING: &str = "ROOT_MOTION_BODY_BINDING";
pub const CODE_INVALID_KEYS: &str = "ROOT_MOTION_INVALID_KEYS";
pub const CODE_INITIAL_POSE: &str = "ROOT_MOTION_INITIAL_POSE";
pub const CODE_OLD_PIN_OVERLAP: &str = "ROOT_MOTION_OLD_PIN_OVERLAP";
pub const CODE_UNSUPPORTED_DOMAIN: &str = "ROOT_MOTION_UNSUPPORTED_DOMAIN";
pub const CODE_INPUT_HASH_MISMATCH: &str = "ROOT_MOTION_INPUT_HASH_MISMATCH";
pub const CODE_NATIVE_CAPABILITY: &str = "ROOT_MOTION_NATIVE_CAPABILITY";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RootMotionError {
    pub code: &'static str,
    pub body_uuid: Option<String>,
    pub key_index: Option<usize>,
    pub field: String,
    pub detail: String,
}
impl fmt::Display for RootMotionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at {}", self.code, self.field)?;
        if let Some(body) = &self.body_uuid { write!(f, " body={body}")?; }
        if let Some(key) = self.key_index { write!(f, " key={key}")?; }
        write!(f, ": {}", self.detail)
    }
}
impl std::error::Error for RootMotionError {}
pub type RootResult<T> = Result<T, RootMotionError>;
fn error(code: &'static str, body: Option<&str>, key: Option<usize>, field: impl Into<String>, detail: impl Into<String>) -> RootMotionError {
    RootMotionError { code, body_uuid: body.map(str::to_owned), key_index: key,
        field: field.into(), detail: detail.into() }
}

/// CBOR floating number only: ciborium widens f16/f32 encodings to f64, while
/// integer and boolean substitutions remain invalid for these real fields.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(transparent)]
pub struct WireF64(pub f64);
impl<'de> Deserialize<'de> for WireF64 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ExactF64;
        impl<'de> de::Visitor<'de> for ExactF64 {
            type Value = WireF64;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str("a CBOR f64") }
            fn visit_f64<E: de::Error>(self, v: f64) -> Result<Self::Value, E> { Ok(WireF64(v)) }
        }
        deserializer.deserialize_any(ExactF64)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RootMotionEnvelopeWireV1 {
    pub version: u32,
    pub kind: String,
    pub payload: RootMotionPayloadWireV1,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RootMotionPayloadWireV1 {
    pub scene_uuid: String,
    pub interpolation: String,
    pub extrapolation: String,
    pub bodies: Vec<RootMotionBodyWireV1>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RootMotionBodyWireV1 {
    pub body_uuid: String,
    pub runtime_body_index: u32,
    pub mode: RootMotionMode,
    pub keys: Vec<RootMotionKeyWireV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub control_events: Vec<RootControlEventWire>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RootControlMode { Static, Kinematic, Dynamic }
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RootControlEventWire {
    pub time_s: WireF64, pub mode: RootControlMode,
    #[serde(default,skip_serializing_if="Option::is_none",deserialize_with="decode_pose_binding")]
    pub pose_binding: Option<PoseBinding>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all="snake_case")]
pub enum PoseBinding { Absolute, CurrentHold, RigidDeltaV1 }
fn decode_pose_binding<'de,D:Deserializer<'de>>(d:D)->Result<Option<PoseBinding>,D::Error> {
    PoseBinding::deserialize(d).map(Some)
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootControlEvent { pub time_s: f64, pub mode: RootControlMode, pub pose_binding: Option<PoseBinding> }
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RootMotionMode { Static, Kinematic, Dynamic }
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RootMotionKeyWireV1 {
    pub time_s: WireF64,
    pub center_world_m: [WireF64; 3],
    pub rotation_world_wxyz: [WireF64; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RootPose {
    pub center_world_m: [f64; 3],
    pub rotation_world_wxyz: [f64; 4],
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalizedKey { pub time_s: f64, pub pose: RootPose }
#[derive(Clone, Debug, PartialEq)]
pub struct ValidatedRootMotion {
    pub scene_uuid: String,
    pub body_uuid: String,
    pub runtime_body_index: u32,
    pub mode: RootMotionMode,
    pub keys: Vec<NormalizedKey>,
    pub control_events: Vec<RootControlEvent>,
    /// Immutable right-local transform from the author frame to the native
    /// carrier frame. None preserves the exact v1/v2 path.
    pub reference_transform: Option<RootPose>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RootMotionEnvelopeWireV3 {
    pub version: u32,
    pub kind: String,
    pub payload: RootMotionPayloadWireV3,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RootMotionPayloadWireV3 {
    pub scene_uuid: String,
    pub interpolation: String,
    pub extrapolation: String,
    pub bodies: Vec<RootMotionBodyWireV3>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RootMotionBodyWireV3 {
    pub body_uuid: String,
    pub runtime_body_index: u32,
    pub mode: RootMotionMode,
    pub keys: Vec<RootMotionKeyWireV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub control_events: Vec<RootControlEventWire>,
    pub reference_transform: RootReferenceTransformWireV3,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RootReferenceTransformWireV3 {
    pub translation_local_m: [WireF64; 3],
    pub rotation_local_wxyz: [WireF64; 4],
}

fn validate_controls(events:&[RootControlEventWire],initial:RootMotionMode,uuid:&str)->RootResult<Vec<RootControlEvent>> {
    if events.len()>4096 {return Err(error(CODE_CAPACITY_OVERFLOW,Some(uuid),None,"control_events","at most 4096 events per body"));}
    let mut previous=0.0; let mut mode=match initial {RootMotionMode::Static=>RootControlMode::Static,RootMotionMode::Kinematic=>RootControlMode::Kinematic,RootMotionMode::Dynamic=>RootControlMode::Dynamic};
    let mut result=Vec::new();
    for (i,event) in events.iter().enumerate() {
        if !event.time_s.0.is_finite() || event.time_s.0<=previous || event.mode==mode {
            return Err(error(CODE_INVALID_KEYS,Some(uuid),Some(i),"control_events","strictly increasing positive physical times and actual mode changes required"));
        }
        if event.mode==RootControlMode::Dynamic && event.pose_binding.is_some()
            || event.pose_binding==Some(PoseBinding::CurrentHold) && event.mode!=RootControlMode::Static
            || event.pose_binding==Some(PoseBinding::RigidDeltaV1) && event.mode!=RootControlMode::Kinematic {
            return Err(error(CODE_INVALID_KEYS,Some(uuid),Some(i),"control_events.pose_binding","policy does not match target control mode"));
        }
        previous=event.time_s.0; mode=event.mode;
        result.push(RootControlEvent {time_s:previous,mode,pose_binding:event.pose_binding});
    }
    Ok(result)
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeSegment { pub start_s: f64, pub end_s: f64 }

pub fn decode_sidecar(bytes: &[u8]) -> RootResult<ValidatedRootMotion> {
    let mut reader = Cursor::new(bytes);
    let raw: RootMotionEnvelopeWireV1 = ciborium::de::from_reader(&mut reader)
        .map_err(|e| error(CODE_INVALID_KEYS, None, None, "sidecar.cbor", e.to_string()))?;
    if reader.position() != bytes.len() as u64 {
        return Err(error(CODE_INVALID_KEYS, None, None, "sidecar.cbor", "trailing CBOR bytes"));
    }
    raw.validate()
}

/// Compare the closed context-v2 inventory entry against the exact sidecar
/// bytes before V6 initialization. The caller still verifies inventory name.
pub fn verify_sidecar_snapshot(bytes: &[u8], expected_byte_length: u64, expected_sha256: &str) -> RootResult<()> {
    let digest = Sha256::digest(bytes);
    let actual = format!("{digest:x}");
    if bytes.len() as u64 != expected_byte_length || actual != expected_sha256 {
        return Err(error(CODE_INPUT_HASH_MISMATCH, None, None, "bin/pdrd_root_motion_v1.cbor", "byte length or SHA-256 differs from sealed inventory"));
    }
    Ok(())
}

/// Validate the typed wire form. This is also usable after a host CBOR ingest
/// that embeds the same typed fields without generic-map reinterpretation.
impl RootMotionEnvelopeWireV1 {
    pub fn validate(self) -> RootResult<ValidatedRootMotion> {
        if self.version != 1 || self.kind != ROOT_KIND {
            return Err(error(CODE_VERSION_OR_KIND, None, None, "sidecar.version/kind", "expected version 1 and ppf.pdrd_root_motion"));
        }
        let payload = self.payload;
        if !canonical_nonworld_uuid(&payload.scene_uuid) {
            return Err(error(CODE_BODY_BINDING, None, None, "payload.scene_uuid", "nonzero canonical lowercase UUID required"));
        }
        if payload.interpolation != INTERPOLATION || payload.extrapolation != EXTRAPOLATION {
            return Err(error(CODE_VERSION_OR_KIND, None, None, "payload.interpolation/extrapolation", "unsupported sampler contract"));
        }
        if payload.bodies.len() != 1 {
            return Err(error(CODE_UNSUPPORTED_DOMAIN, None, None, "payload.bodies", "first slice requires exactly one root body"));
        }
        let body = payload.bodies.into_iter().next().unwrap();
        let uuid = body.body_uuid;
        if !canonical_nonworld_uuid(&uuid) {
            return Err(error(CODE_BODY_BINDING, Some(&uuid), None, "body.body_uuid", "nonzero canonical lowercase UUID required"));
        }
        if body.mode==RootMotionMode::Dynamic || body.control_events.iter().any(|e|e.pose_binding.is_some()) {
            return Err(error(CODE_VERSION_OR_KIND,Some(&uuid),None,"body.mode/control_events.pose_binding","body control extensions require root v2 and body-controls feature"));
        }
        let required_count = if body.mode == RootMotionMode::Static { 1 } else { 2 };
        if body.keys.len() > MAX_KEYS_PER_ROOT_V2 {
            return Err(error(CODE_CAPACITY_OVERFLOW, Some(&uuid), None, "body.keys", "per-root key count exceeds 4096"));
        }
        if body.keys.len() < required_count || (body.mode == RootMotionMode::Static && body.control_events.is_empty() && body.keys.len() != 1) {
            return Err(error(CODE_INVALID_KEYS, Some(&uuid), None, "body.keys", "static needs one key; kinematic needs at least two"));
        }
        let mut keys: Vec<NormalizedKey> = Vec::with_capacity(body.keys.len());
        for (i, raw) in body.keys.into_iter().enumerate() {
            let t = raw.time_s.0;
            if !t.is_finite() || t < 0.0 || (i == 0 && t != 0.0) ||
                keys.last().is_some_and(|prev| t <= prev.time_s) {
                return Err(error(CODE_INVALID_KEYS, Some(&uuid), Some(i), "time_s", "finite nonnegative, first zero, strictly increasing time required"));
            }
            let center = raw.center_world_m.map(|v| v.0);
            for (axis, value) in center.iter().enumerate() {
                if !value.is_finite() {
                    return Err(error(CODE_INVALID_KEYS, Some(&uuid), Some(i), format!("center_world_m[{axis}]"), "finite f64 required"));
                }
            }
            let mut q = raw.rotation_world_wxyz.map(|v| v.0);
            for (axis, value) in q.iter().enumerate() {
                if !value.is_finite() {
                    return Err(error(CODE_INVALID_KEYS, Some(&uuid), Some(i), format!("rotation_world_wxyz[{axis}]"), "finite f64 required"));
                }
            }
            let norm = q.iter().map(|v| v * v).sum::<f64>().sqrt();
            if !norm.is_finite() || (norm - 1.0).abs() > 1e-6 {
                return Err(error(CODE_INVALID_KEYS, Some(&uuid), Some(i), "rotation_world_wxyz", "quaternion norm outside 1e-6 of one"));
            }
            for v in &mut q { *v /= norm; }
            if let Some(prev) = keys.last() {
                let mut dot = dot4(prev.pose.rotation_world_wxyz, q);
                if dot < 0.0 { for v in &mut q { *v = -*v; } dot = -dot; }
                if dot <= 1e-8 {
                    return Err(error(CODE_INVALID_KEYS, Some(&uuid), Some(i), "rotation_world_wxyz", "ambiguous adjacent half-turn"));
                }
            }
            keys.push(NormalizedKey { time_s: t,
                pose: RootPose { center_world_m: center, rotation_world_wxyz: q } });
        }
        let control_events=validate_controls(&body.control_events,body.mode,&uuid)?;
        Ok(ValidatedRootMotion { scene_uuid: payload.scene_uuid, body_uuid: uuid,
            runtime_body_index: body.runtime_body_index, mode: body.mode, keys, control_events, reference_transform: None })
    }
}

fn canonical_nonworld_uuid(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 36 && b.iter().enumerate().all(|(i, c)| {
        if [8, 13, 18, 23].contains(&i) { *c == b'-' }
        else { c.is_ascii_digit() || (b'a'..=b'f').contains(c) }
    }) && s != "00000000-0000-0000-0000-000000000000"
}

// ---------------------------------------------------------------------------
// Root-motion sidecar **version 2** (CONTRACT.md §4.1, decision U1)
//
// One physical path `bin/pdrd_root_motion_v1.cbor` serves both schema versions; the
// CBOR envelope `version` distinguishes them.  A present sidecar whose envelope
// version does not match the required root feature refuses **before any solve** - an
// unexpected version is never ignored and never treated as absent.  Root-v1
// semantics above are untouched.
// ---------------------------------------------------------------------------
pub const BODY_CONTROLS_FEATURE: &str = "pdrd_body_controls_v1";
pub const ROOT_FEATURE_V2: &str = "pdrd_kinematic_roots_v2";
pub const ROOT_FEATURE_V3: &str = "pdrd_kinematic_roots_v3";
pub const SIDECAR_VERSION_V2: u32 = 2;
pub const SIDECAR_VERSION_V3: u32 = 3;
pub const MAX_ROOTS_V2: usize = 128;
pub const MAX_KEYS_PER_ROOT_V2: usize = 4096;
pub const MAX_KEY_RECORDS_TOTAL_V2: usize = 65536;
/// V9 stable codes for the root batch (ABI_LAYOUT.json `error_codes.v9_additions`).
pub const CODE_ROOT_BATCH: &str = "G2_D6_ROOT_BATCH_INVALID";
pub const CODE_CAPACITY_OVERFLOW: &str = "G2_D6_CAPACITY_OVERFLOW";

/// One validated root of a version-2 sidecar, keyed by its runtime body index.
#[derive(Clone, Debug, PartialEq)]
pub struct ValidatedRootMotionV2 {
    pub scene_uuid: String,
    pub bodies: Vec<ValidatedRootMotion>,
    pub total_keys: usize,
}

impl ValidatedRootMotionV2 {
    pub fn rooted_body_count(&self) -> usize {
        self.bodies.len()
    }

    /// The one atomic physical clock: every root is sampled at the same `T1` (§4.3).
    pub fn sample_all(&self, time_s: f64) -> RootResult<Vec<RootPose>> {
        self.bodies.iter().map(|body| body.sample(time_s)).collect()
    }

    /// Root body indices ascending, which is the frozen read order.
    pub fn runtime_body_indices(&self) -> Vec<u32> {
        self.bodies.iter().map(|body| body.runtime_body_index).collect()
    }
}

/// Peek the envelope version without validating the payload (U1 dispatch).
pub fn peek_sidecar_version(bytes: &[u8]) -> RootResult<(u32, String)> {
    #[derive(Deserialize)]
    struct Probe { version: u32, kind: String }
    let mut reader = Cursor::new(bytes);
    let probe: Probe = ciborium::de::from_reader(&mut reader)
        .map_err(|e| error(CODE_VERSION_OR_KIND, None, None, "sidecar.cbor", e.to_string()))?;
    Ok((probe.version, probe.kind))
}

/// Decode a version-2 sidecar: 1..=128 bodies, sorted ascending and unique,
/// per-root keys strictly increasing from exactly `time_s = 0`, valid unit
/// quaternions, and a total key bound independent of the root count.
pub fn decode_sidecar_v2(bytes: &[u8]) -> RootResult<ValidatedRootMotionV2> {
    let mut reader = Cursor::new(bytes);
    let raw: RootMotionEnvelopeWireV1 = ciborium::de::from_reader(&mut reader)
        .map_err(|e| error(CODE_INVALID_KEYS, None, None, "sidecar.cbor", e.to_string()))?;
    if reader.position() != bytes.len() as u64 {
        return Err(error(CODE_INVALID_KEYS, None, None, "sidecar.cbor", "trailing CBOR bytes"));
    }
    if raw.version != SIDECAR_VERSION_V2 || raw.kind != ROOT_KIND {
        return Err(error(
            CODE_VERSION_OR_KIND,
            None,
            None,
            "sidecar.version/kind",
            "expected version 2 and ppf.pdrd_root_motion for the root-v2 surface",
        ));
    }
    let payload = raw.payload;
    if !canonical_nonworld_uuid(&payload.scene_uuid) {
        return Err(error(CODE_BODY_BINDING, None, None, "payload.scene_uuid", "nonzero canonical lowercase UUID required"));
    }
    if payload.interpolation != INTERPOLATION || payload.extrapolation != EXTRAPOLATION {
        return Err(error(CODE_VERSION_OR_KIND, None, None, "payload.interpolation/extrapolation", "unsupported sampler contract"));
    }
    if payload.bodies.is_empty() || payload.bodies.len() > MAX_ROOTS_V2 {
        return Err(error(CODE_ROOT_BATCH, None, None, "payload.bodies", "1..=128 root bodies required"));
    }
    // Refuse an out-of-order batch: the frozen order is part of the contract, so a
    // mis-ordered batch is never silently reordered.
    let indices = payload.bodies.iter().map(|b| b.runtime_body_index).collect::<Vec<_>>();
    for pair in indices.windows(2) {
        if pair[0] >= pair[1] {
            return Err(error(
                CODE_ROOT_BATCH,
                None,
                None,
                "payload.bodies.runtime_body_index",
                "bodies must already be sorted ascending by runtime_body_index and unique",
            ));
        }
    }
    let mut bodies = Vec::with_capacity(payload.bodies.len());
    let mut total_keys = 0usize;
    let mut uuids = std::collections::HashSet::new();
    for body in payload.bodies {
        // The per-body rules are exactly v1's, including the mode-dependent minimum
        // key count and the first-key-zero / strictly-increasing time rule.
        let uuid = body.body_uuid;
        if !canonical_nonworld_uuid(&uuid) {
            return Err(error(CODE_BODY_BINDING, Some(&uuid), None, "body.body_uuid", "nonzero canonical lowercase UUID required"));
        }
        if !uuids.insert(uuid.clone()) {
            return Err(error(CODE_ROOT_BATCH, Some(&uuid), None, "body.body_uuid", "duplicate root UUID"));
        }
        let required_count = if body.mode==RootMotionMode::Kinematic
            || body.mode==RootMotionMode::Dynamic && body.control_events.iter().any(|e|e.mode==RootControlMode::Kinematic)
            || body.control_events.iter().any(|e|e.pose_binding==Some(PoseBinding::RigidDeltaV1)) {2} else {1};
        if body.keys.len() < required_count
            || (body.mode == RootMotionMode::Static && body.control_events.is_empty() && body.keys.len() != 1)
        {
            return Err(error(CODE_INVALID_KEYS, Some(&uuid), None, "body.keys", "static needs one key; kinematic needs at least two"));
        }
        if body.keys.len() > MAX_KEYS_PER_ROOT_V2 {
            return Err(error(CODE_CAPACITY_OVERFLOW, Some(&uuid), None, "body.keys", "per-root key count exceeds 4096"));
        }
        total_keys += body.keys.len();
        if total_keys > MAX_KEY_RECORDS_TOTAL_V2 {
            return Err(error(CODE_CAPACITY_OVERFLOW, Some(&uuid), None, "body.keys", "total key records exceed 65536"));
        }
        let mut keys: Vec<NormalizedKey> = Vec::with_capacity(body.keys.len());
        for (i, raw) in body.keys.into_iter().enumerate() {
            let t = raw.time_s.0;
            if !t.is_finite() || t < 0.0 || (i == 0 && t != 0.0)
                || keys.last().is_some_and(|prev| t <= prev.time_s)
            {
                return Err(error(CODE_INVALID_KEYS, Some(&uuid), Some(i), "time_s", "finite nonnegative, first zero, strictly increasing time required"));
            }
            let center = raw.center_world_m.map(|v| v.0);
            for (axis, value) in center.iter().enumerate() {
                if !value.is_finite() {
                    return Err(error(CODE_INVALID_KEYS, Some(&uuid), Some(i), format!("center_world_m[{axis}]"), "finite f64 required"));
                }
            }
            let mut q = raw.rotation_world_wxyz.map(|v| v.0);
            for (axis, value) in q.iter().enumerate() {
                if !value.is_finite() {
                    return Err(error(CODE_INVALID_KEYS, Some(&uuid), Some(i), format!("rotation_world_wxyz[{axis}]"), "finite f64 required"));
                }
            }
            let norm = q.iter().map(|v| v * v).sum::<f64>().sqrt();
            if !norm.is_finite() || (norm - 1.0).abs() > 1e-6 {
                return Err(error(CODE_INVALID_KEYS, Some(&uuid), Some(i), "rotation_world_wxyz", "quaternion norm outside 1e-6 of one"));
            }
            for v in &mut q { *v /= norm; }
            if let Some(prev) = keys.last() {
                let mut dot = dot4(prev.pose.rotation_world_wxyz, q);
                if dot < 0.0 { for v in &mut q { *v = -*v; } dot = -dot; }
                if dot <= 1e-8 {
                    return Err(error(CODE_INVALID_KEYS, Some(&uuid), Some(i), "rotation_world_wxyz", "ambiguous adjacent half-turn"));
                }
            }
            keys.push(NormalizedKey { time_s: t,
                pose: RootPose { center_world_m: center, rotation_world_wxyz: q } });
        }
        let control_events=validate_controls(&body.control_events,body.mode,&uuid)?;
        bodies.push(ValidatedRootMotion {
            scene_uuid: payload.scene_uuid.clone(),
            body_uuid: uuid,
            runtime_body_index: body.runtime_body_index,
            mode: body.mode,
            keys, control_events, reference_transform: None,
        });
    }
    Ok(ValidatedRootMotionV2 { scene_uuid: payload.scene_uuid, bodies, total_keys })
}

/// Decode the strict v3 envelope separately from v1/v2, then reuse v2's
/// unchanged root/key/control validation after capturing each immutable D.
pub fn decode_sidecar_v3(bytes: &[u8]) -> RootResult<ValidatedRootMotionV2> {
    let mut reader=Cursor::new(bytes);
    let raw:RootMotionEnvelopeWireV3=ciborium::de::from_reader(&mut reader)
        .map_err(|e|error(CODE_INVALID_KEYS,None,None,"sidecar.cbor",e.to_string()))?;
    if reader.position()!=bytes.len() as u64 {
        return Err(error(CODE_INVALID_KEYS,None,None,"sidecar.cbor","trailing CBOR bytes"));
    }
    if raw.version!=SIDECAR_VERSION_V3 || raw.kind!=ROOT_KIND {
        return Err(error(CODE_VERSION_OR_KIND,None,None,"sidecar.version/kind","expected version 3 and ppf.pdrd_root_motion"));
    }
    let mut refs=Vec::with_capacity(raw.payload.bodies.len());
    let mut bodies=Vec::with_capacity(raw.payload.bodies.len());
    for (i,b) in raw.payload.bodies.into_iter().enumerate() {
        let t=b.reference_transform.translation_local_m.map(|v|v.0);
        if t.iter().any(|v|!v.is_finite()) {
            return Err(error(CODE_INVALID_KEYS,Some(&b.body_uuid),None,format!("bodies[{i}].reference_transform.translation_local_m"),"finite f64 values required"));
        }
        let mut q=b.reference_transform.rotation_local_wxyz.map(|v|v.0);
        if q.iter().any(|v|!v.is_finite()) {
            return Err(error(CODE_INVALID_KEYS,Some(&b.body_uuid),None,format!("bodies[{i}].reference_transform.rotation_local_wxyz"),"finite f64 values required"));
        }
        let norm=q.iter().map(|v|v*v).sum::<f64>().sqrt();
        if !norm.is_finite() || norm<=0.0 || (norm-1.0).abs()>1e-6 {
            return Err(error(CODE_INVALID_KEYS,Some(&b.body_uuid),None,format!("bodies[{i}].reference_transform.rotation_local_wxyz"),"quaternion norm outside 1e-6 of one"));
        }
        for v in &mut q {*v/=norm;}
        refs.push(RootPose{center_world_m:t,rotation_world_wxyz:q});
        bodies.push(RootMotionBodyWireV1{body_uuid:b.body_uuid,runtime_body_index:b.runtime_body_index,
            mode:b.mode,keys:b.keys,control_events:b.control_events});
    }
    let v2=RootMotionEnvelopeWireV1{version:SIDECAR_VERSION_V2,kind:raw.kind,payload:RootMotionPayloadWireV1{
        scene_uuid:raw.payload.scene_uuid,interpolation:raw.payload.interpolation,
        extrapolation:raw.payload.extrapolation,bodies}};
    let mut encoded=Vec::new();
    ciborium::ser::into_writer(&v2,&mut encoded).map_err(|e|error(CODE_INVALID_KEYS,None,None,"sidecar.cbor",e.to_string()))?;
    let mut validated=decode_sidecar_v2(&encoded)?;
    for (body,d) in validated.bodies.iter_mut().zip(refs) {body.reference_transform=Some(d);}
    Ok(validated)
}

/// One path, version-distinguished (U1): refuse before any solve on a mismatch.
///
/// `required_feature` is the root feature the V9 envelope advertises, so `None`
/// (a wholly rootless forest) with a present sidecar is also a refusal.
pub fn validate_sidecar_version_obligation_v9(
    required_feature: Option<&str>,
    sidecar_present: bool,
    envelope_version: Option<u32>,
) -> RootResult<()> {
    match (required_feature, sidecar_present, envelope_version) {
        (None, false, _) => Ok(()),
        (None, true, _) => Err(error(
            CODE_VERSION_OR_KIND,
            None,
            None,
            "bin/pdrd_root_motion_v1.cbor",
            "a present sidecar with no advertised root feature is refused, never ignored",
        )),
        (Some(_), false, _) => Err(error(
            CODE_SIDECAR_MISSING,
            None,
            None,
            "bin/pdrd_root_motion_v1.cbor",
            "an advertised root feature requires the sealed sidecar",
        )),
        (Some(ROOT_FEATURE), true, Some(1)) => Ok(()),
        (Some(ROOT_FEATURE_V2), true, Some(2)) => Ok(()),
        (Some(ROOT_FEATURE_V3), true, Some(3)) => Ok(()),
        (Some(_), true, Some(other)) => Err(error(
            CODE_VERSION_OR_KIND,
            None,
            None,
            "bin/pdrd_root_motion_v1.cbor",
            format!("a present sidecar with envelope version {other} does not match the required root feature"),
        )),
        (Some(_), true, None) => Err(error(
            CODE_VERSION_OR_KIND,
            None,
            None,
            "bin/pdrd_root_motion_v1.cbor",
            "an undecodable sidecar version is a refusal, never treated as absent",
        )),
    }
}

/// Multi-root scenes require the root-v2 symbols; a v1 batch is never split (§4.0).
pub fn validate_root_surface_symbols(
    prescribed_roots: usize,
    root_v2_symbols_loaded: bool,
) -> RootResult<()> {
    if prescribed_roots > 1 && !root_v2_symbols_loaded {
        return Err(error(
            CODE_ROOT_BATCH,
            None,
            None,
            "pdrd_root_motion_v2_query_v1",
            "multi-root requires the root-v2 surface",
        ));
    }
    Ok(())
}

/// Upload rules of the root-v2 batch (CONTRACT.md §4.2).
pub fn validate_upload_v2(
    body_count: u32,
    body_stride: u32,
    key_stride: u32,
    key_count_total: u32,
    keys_per_root: &[(u32, u32)],
) -> RootResult<()> {
    if body_stride != 64
        || key_stride != 64
    {
        return Err(error(CODE_INVALID_KEYS, None, None, "body_stride/key_stride", "body_stride and key_stride must both be exactly 64"));
    }
    if body_count == 0 || body_count as usize > MAX_ROOTS_V2 {
        return Err(error(CODE_ROOT_BATCH, None, None, "body_count", "1..=128 root bodies required"));
    }
    if keys_per_root.len() != body_count as usize {
        return Err(error(CODE_ROOT_BATCH, None, None, "keys_per_root", "one key span per body required"));
    }
    if key_count_total as usize > MAX_KEY_RECORDS_TOTAL_V2 {
        return Err(error(CODE_CAPACITY_OVERFLOW, None, None, "key_count_total", "total key records exceed 65536"));
    }
    let mut expected_start = 0u32;
    for (start, count) in keys_per_root.iter() {
        if *start != expected_start {
            return Err(error(CODE_INVALID_KEYS, None, None, "key_start/key_count", "key spans must be contiguous, non-overlapping and start at zero"));
        }
        if *count == 0 || *count as usize > MAX_KEYS_PER_ROOT_V2 {
            return Err(error(CODE_CAPACITY_OVERFLOW, None, None, "key_count", "per-root key_count must be 1..=4096"));
        }
        expected_start = expected_start.saturating_add(*count);
        if expected_start > key_count_total {
            return Err(error(CODE_CAPACITY_OVERFLOW, None, None, "key_count_total", "per-root spans must stay inside key_count_total"));
        }
    }
    if expected_start != key_count_total {
        return Err(error(CODE_CAPACITY_OVERFLOW, None, None, "key_count_total", "per-root spans must cover exactly key_count_total"));
    }
    Ok(())
}

fn dot4(a: [f64; 4], b: [f64; 4]) -> f64 { (0..4).map(|i| a[i] * b[i]).sum() }

fn checked_time(t: f64, body: &str, field: &str) -> RootResult<()> {
    if !t.is_finite() || t < 0.0 {
        Err(error(CODE_INVALID_KEYS, Some(body), None, field, "finite nonnegative physical seconds required"))
    } else { Ok(()) }
}
/// Shared author curve evaluator; has no PDRD or FEM body-index identity.
pub fn sample_normalized_keys(body:&str, keys:&[NormalizedKey], time_s:f64)->RootResult<RootPose> {
    if keys.is_empty() {return Err(error(CODE_INVALID_KEYS,Some(body),None,"keys","nonempty validated keys required"));}

        checked_time(time_s, body, "physical_T_s")?;
        match keys.binary_search_by(|key| key.time_s.total_cmp(&time_s)) {
            Ok(i) => Ok(keys[i].pose),
            Err(0) => Ok(keys[0].pose),
            Err(i) if i == keys.len() => Ok(keys[keys.len() - 1].pose),
            Err(i) => {
                let a = keys[i - 1]; let b = keys[i];
                let u = (time_s - a.time_s) / (b.time_s - a.time_s);
                if !u.is_finite() || !(0.0..=1.0).contains(&u) {
                    return Err(error(CODE_INVALID_KEYS, Some(body), Some(i), "time_s", "segment interpolation arithmetic invalid"));
                }
                let center = std::array::from_fn(|j| (1.0-u) * a.pose.center_world_m[j] + u * b.pose.center_world_m[j]);
                if center.iter().any(|v| !v.is_finite()) {
                    return Err(error(CODE_INVALID_KEYS, Some(body), Some(i), "center_world_m", "interpolated center overflow"));
                }
                Ok(RootPose { center_world_m: center,
                    rotation_world_wxyz: shortest_slerp(a.pose.rotation_world_wxyz, b.pose.rotation_world_wxyz, u) })
            }
        }
    }
impl ValidatedRootMotion {
    /// Stateless, deterministic world-native-frame sample at physical T.
    pub fn sample(&self, time_s: f64) -> RootResult<RootPose> {
        sample_normalized_keys(&self.body_uuid,&self.keys,time_s)
    }

    /// Sample the author path first and apply the immutable right-local D.
    /// Control epochs are deliberately composed later by RootState.
    pub fn sample_native(&self,time_s:f64)->RootResult<RootPose> {
        let author=self.sample(time_s)?;
        match self.reference_transform {
            Some(d) if d.center_world_m==[0.0;3]
                && (d.rotation_world_wxyz==[1.0,0.0,0.0,0.0]||d.rotation_world_wxyz==[-1.0,0.0,0.0,0.0])=>Ok(author),
            Some(d)=>compose_root_pose(author,d),None=>Ok(author)
        }
    }

    /// Exact sweep intervals, cut at every strictly interior key boundary.
    pub fn partition_step(&self, start_s: f64, dt_s: f64) -> RootResult<Vec<TimeSegment>> {
        checked_time(start_s, &self.body_uuid, "physical_T_s")?;
        let end_s = start_s + dt_s;
        if !dt_s.is_finite() || dt_s <= 0.0 || !end_s.is_finite() || end_s <= start_s {
            return Err(error(CODE_INVALID_KEYS, Some(&self.body_uuid), None, "physical_dt_s", "positive representable forward physical step required"));
        }
        let mut out = Vec::new(); let mut start = start_s;
        for key in &self.keys {
            if key.time_s > start_s && key.time_s < end_s {
                out.push(TimeSegment { start_s: start, end_s: key.time_s });
                start = key.time_s;
            }
        }
        out.push(TimeSegment { start_s: start, end_s });
        Ok(out)
    }
}

#[cfg(test)]
mod reference_v3_tests {
    use super::*;
    fn key(t:f64,c:[f64;3],q:[f64;4])->RootMotionKeyWireV1 {
        RootMotionKeyWireV1{time_s:WireF64(t),center_world_m:c.map(WireF64),rotation_world_wxyz:q.map(WireF64)}
    }
    fn encoded(keys:Vec<RootMotionKeyWireV1>,d:[f64;3],dq:[f64;4])->Vec<u8> {
        let body=RootMotionBodyWireV3{body_uuid:"22222222-2222-2222-2222-222222222222".into(),runtime_body_index:0,
            mode:RootMotionMode::Kinematic,keys,control_events:vec![],reference_transform:RootReferenceTransformWireV3{
                translation_local_m:d.map(WireF64),rotation_local_wxyz:dq.map(WireF64)}};
        let wire=RootMotionEnvelopeWireV3{version:3,kind:ROOT_KIND.into(),payload:RootMotionPayloadWireV3{
            scene_uuid:"11111111-1111-1111-1111-111111111111".into(),interpolation:INTERPOLATION.into(),extrapolation:EXTRAPOLATION.into(),bodies:vec![body]}};
        let mut out=Vec::new();ciborium::ser::into_writer(&wire,&mut out).unwrap();out
    }
    fn find_value_mut<'a>(v:&'a mut ciborium::value::Value,name:&str)->Option<&'a mut ciborium::value::Value> {
        use ciborium::value::Value;
        match v {
            Value::Map(entries)=>{
                if let Some(i)=entries.iter().position(|(k,_)|k==&Value::Text(name.into())) {return Some(&mut entries[i].1);}
                for (_,item) in entries {if let Some(found)=find_value_mut(item,name){return Some(found);}}
                None
            },
            Value::Array(items)=>{for item in items {if let Some(found)=find_value_mut(item,name){return Some(found);}}None},
            _=>None,
        }
    }
    fn generic_value(bytes:&[u8])->ciborium::value::Value {ciborium::de::from_reader(bytes).unwrap()}
    fn encode_value(v:&ciborium::value::Value)->Vec<u8>{let mut b=Vec::new();ciborium::ser::into_writer(v,&mut b).unwrap();b}
    fn close(a:f64,b:f64){assert!((a-b).abs()<1e-12,"{a} != {b}");}
    #[test]
    fn eccentric_ninety_degree_midpoint_applies_d_after_author_slerp() {
        let s=std::f64::consts::FRAC_1_SQRT_2;
        let bytes=encoded(vec![key(0.0,[0.0;3],[1.0,0.0,0.0,0.0]),key(1.0,[0.0;3],[s,0.0,0.0,s])],[1.0,0.0,0.0],[1.0,0.0,0.0,0.0]);
        let roots=decode_sidecar_v3(&bytes).unwrap();
        let got=roots.bodies[0].sample_native(0.5).unwrap();
        close(got.center_world_m[0],s);close(got.center_world_m[1],s);
        close(got.rotation_world_wxyz[0],std::f64::consts::FRAC_PI_8.cos());
        close(got.rotation_world_wxyz[3],std::f64::consts::FRAC_PI_8.sin());
        assert!(decode_sidecar_v2(&bytes).is_err(),"v2 typed schema must reject v3 fields/version");
    }
    #[test]
    fn translated_author_track_and_rotated_reference_compose_in_right_local_order() {
        let s=std::f64::consts::FRAC_1_SQRT_2;
        let qx=[s,s,0.0,0.0];let qz=[s,0.0,0.0,s];
        let bytes=encoded(vec![key(0.0,[1.0,2.0,0.0],qz),key(1.0,[3.0,2.0,0.0],qz)],[0.0,2.0,0.0],qx);
        let r=decode_sidecar_v3(&bytes).unwrap().bodies.remove(0);
        let p=r.sample_native(0.5).unwrap();
        close(p.center_world_m[0],0.0);close(p.center_world_m[1],2.0);close(p.center_world_m[2],0.0);
        let expected=compose_root_pose(RootPose{center_world_m:[2.0,2.0,0.0],rotation_world_wxyz:qz},RootPose{center_world_m:[0.0,2.0,0.0],rotation_world_wxyz:qx}).unwrap();
        assert_eq!(p,expected);
    }
    #[test]
    fn varying_axis_and_translation_keep_the_exact_composed_midpoint() {
        let s=std::f64::consts::FRAC_1_SQRT_2;let c=std::f64::consts::FRAC_PI_8.cos();let h=std::f64::consts::FRAC_PI_8.sin();
        let root=decode_sidecar_v3(&encoded(vec![key(0.0,[0.0;3],[1.0,0.0,0.0,0.0]),key(1.0,[2.0,-1.0,0.0],[s,s,0.0,0.0])],
            [0.0,1.0,0.0],[s,0.0,0.0,s])).unwrap().bodies.remove(0);
        let p=root.sample_native(0.5).unwrap();
        close(p.center_world_m[0],1.0);close(p.center_world_m[1],-0.5+s);close(p.center_world_m[2],s);
        close(p.rotation_world_wxyz[0],c*s);close(p.rotation_world_wxyz[1],h*s);
        close(p.rotation_world_wxyz[2],-h*s);close(p.rotation_world_wxyz[3],c*s);
    }
    #[test]
    fn v3_rejects_exact_adjacent_half_turn_and_normalizes_quaternion_sign() {
        let half=encoded(vec![key(0.0,[0.0;3],[1.0,0.0,0.0,0.0]),key(1.0,[0.0;3],[0.0,0.0,0.0,1.0])],[0.0;3],[1.0,0.0,0.0,0.0]);
        assert!(decode_sidecar_v3(&half).is_err());
        let s=std::f64::consts::FRAC_1_SQRT_2;
        let signed=encoded(vec![key(0.0,[0.0;3],[1.0,0.0,0.0,0.0]),key(1.0,[0.0;3],[-s,0.0,0.0,-s])],[0.0;3],[1.0,0.0,0.0,0.0]);
        let p=decode_sidecar_v3(&signed).unwrap().bodies[0].sample(1.0).unwrap();
        assert!(p.rotation_world_wxyz[0]>0.0&&p.rotation_world_wxyz[3]>0.0);
    }
    #[test]
    fn initial_native_fit_uses_author_initial_composed_with_reference() {
        let s=std::f64::consts::FRAC_1_SQRT_2;
        let bytes=encoded(vec![key(0.0,[1.0,2.0,0.0],[s,0.0,0.0,s]),key(1.0,[2.0,2.0,0.0],[s,0.0,0.0,s])],
            [1.0,0.0,0.0],[1.0,0.0,0.0,0.0]);
        let root=decode_sidecar_v3(&bytes).unwrap().bodies.remove(0);
        let rest=[[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]];
        let native=reconstruct_native_vertices(root.sample_native(0.0).unwrap(),&rest);
        let ids=[1,2,3,4];let bodies=[BodyIdentity{uuid:"22222222-2222-2222-2222-222222222222",runtime_body_index:0}];
        let evidence=NativeFrameEvidence{context_scene_uuid:"11111111-1111-1111-1111-111111111111",context_bodies:&bodies,joint_bodies:&bodies,
            native_body:BodyIdentity{uuid:"22222222-2222-2222-2222-222222222222",runtime_body_index:0},native_body_is_free:true,has_enabled_known_parent_joint:true,vertex_indices:&ids,
            rest_centered_m:&rest,initial_world_m:&native,fit_tolerance_m:1e-12};
        assert_eq!(root.validate_initial_native_frame(&evidence).unwrap(),0.0);
        let author=reconstruct_native_vertices(root.keys[0].pose,&rest);
        let author_evidence=NativeFrameEvidence{initial_world_m:&author,..evidence};
        assert!(root.validate_initial_native_frame(&author_evidence).is_err(),"author-only initial fit must refuse");
        let baked=decode_sidecar_v3(&encoded(vec![key(0.0,[1.0,2.0,0.0],[s,0.0,0.0,s]),key(1.0,[2.0,2.0,0.0],[s,0.0,0.0,s]),
            ],[1.0,0.0,0.0],[1.0,0.0,0.0,0.0])).unwrap().bodies.remove(0);
        let baked_native=reconstruct_native_vertices(baked.keys[0].pose,&rest);
        let doubled_evidence=NativeFrameEvidence{initial_world_m:&baked_native,..evidence};
        assert!(baked.validate_initial_native_frame(&doubled_evidence).is_err(),"already-offset keys plus D must fail initial agreement");
    }
    #[test]
    fn sidecar_feature_and_version_matrix_is_closed() {
        for (feature,present,version,ok) in [(None,false,None,true),(None,true,Some(3),false),
            (Some(ROOT_FEATURE_V2),true,Some(3),false),(Some(ROOT_FEATURE_V3),true,Some(2),false),
            (Some(ROOT_FEATURE_V3),true,Some(3),true),(Some(ROOT_FEATURE_V3),false,None,false)] {
            assert_eq!(validate_sidecar_version_obligation_v9(feature,present,version).is_ok(),ok);
        }
    }
    #[test]
    fn identity_reference_preserves_author_sample_exactly() {
        let s=std::f64::consts::FRAC_1_SQRT_2;
        let bytes=encoded(vec![key(0.0,[0.2,0.3,0.4],[1.0,0.0,0.0,0.0]),key(1.0,[1.2,0.3,0.4],[s,0.0,0.0,s])],[0.0;3],[1.0,0.0,0.0,0.0]);
        let r=decode_sidecar_v3(&bytes).unwrap().bodies.remove(0);
        for t in [0.0,0.1,0.5,1.0] {assert_eq!(r.sample_native(t).unwrap(),r.sample(t).unwrap());}
    }
    #[test]
    fn reference_transform_is_required_closed_finite_and_f64_typed() {
        use ciborium::value::Value;
        let s=std::f64::consts::FRAC_1_SQRT_2;
        let base=encoded(vec![key(0.0,[0.0;3],[1.0,0.0,0.0,0.0]),key(1.0,[1.0,0.0,0.0],[s,0.0,0.0,s])],[0.0;3],[1.0,0.0,0.0,0.0]);
        let mut missing=generic_value(&base);
        if let Value::Map(envelope)=&mut missing {
            if let Some((_,Value::Map(payload)))=envelope.iter_mut().find(|(k,_)|k==&Value::Text("payload".into())) {
                if let Some((_,Value::Array(bodies)))=payload.iter_mut().find(|(k,_)|k==&Value::Text("bodies".into())) {
                    if let Value::Map(body)=&mut bodies[0] {body.retain(|(k,_)|k!=&Value::Text("reference_transform".into()));}
                }
            }
        }
        assert!(decode_sidecar_v3(&encode_value(&missing)).is_err());
        let mut wrong_type=generic_value(&base);
        if let Some(Value::Array(values))=find_value_mut(&mut wrong_type,"translation_local_m") {values[0]=Value::Integer(1.into());}
        assert!(decode_sidecar_v3(&encode_value(&wrong_type)).is_err());
        let mut unknown=generic_value(&base);
        if let Some(Value::Map(values))=find_value_mut(&mut unknown,"reference_transform") {
            values.push((Value::Text("future".into()),Value::Bool(true)));
        }
        assert!(decode_sidecar_v3(&encode_value(&unknown)).is_err());
        assert!(decode_sidecar_v3(&encoded(vec![key(0.0,[0.0;3],[1.0,0.0,0.0,0.0]),key(1.0,[1.0,0.0,0.0],[s,0.0,0.0,s])],[f64::NAN,0.0,0.0],[1.0,0.0,0.0,0.0])).is_err());
        assert!(decode_sidecar_v3(&encoded(vec![key(0.0,[0.0;3],[1.0,0.0,0.0,0.0]),key(1.0,[1.0,0.0,0.0],[s,0.0,0.0,s])],[0.0;3],[0.0;4])).is_err());
    }
}

pub fn compose_root_pose(a:RootPose,b:RootPose)->RootResult<RootPose> {
    let [w,x,y,z]=a.rotation_world_wxyz;let [v,i,j,k]=b.rotation_world_wxyz;
    let mut q=[w*v-x*i-y*j-z*k,w*i+x*v+y*k-z*j,w*j-x*k+y*v+z*i,w*k+x*j-y*i+z*v];
    let r=rotate_world(a.rotation_world_wxyz,b.center_world_m);
    let center=std::array::from_fn(|n|a.center_world_m[n]+r[n]);
    let norm=q.iter().map(|v|v*v).sum::<f64>().sqrt();
    if !norm.is_finite()||norm<=0.0||center.iter().any(|v|!v.is_finite()) {
        return Err(error(CODE_INITIAL_POSE,None,None,"reference_transform","pose composition overflow/invalid rotation"));
    }
    for v in &mut q {*v/=norm;}
    Ok(RootPose{center_world_m:center,rotation_world_wxyz:q})
}

pub fn inverse_root_pose(a:RootPose)->RootResult<RootPose> {
    let [w,x,y,z]=a.rotation_world_wxyz;let q=[w,-x,-y,-z];
    let center=rotate_world(q,a.center_world_m.map(|v|-v));
    if center.iter().any(|v|!v.is_finite()) {return Err(error(CODE_INITIAL_POSE,None,None,"reference_transform","inverse pose overflow"));}
    Ok(RootPose{center_world_m:center,rotation_world_wxyz:q})
}

fn shortest_slerp(a: [f64; 4], b: [f64; 4], u: f64) -> [f64; 4] {
    let dot = dot4(a, b).clamp(-1.0, 1.0);
    let result = if dot > 1.0 - 1e-12 {
        std::array::from_fn(|i| a[i] + u * (b[i] - a[i]))
    } else {
        let theta = dot.acos(); let denom = theta.sin();
        let wa = ((1.0 - u) * theta).sin() / denom;
        let wb = (u * theta).sin() / denom;
        std::array::from_fn(|i| wa * a[i] + wb * b[i])
    };
    let norm = result.iter().map(|v| v * v).sum::<f64>().sqrt();
    result.map(|v| v / norm)
}

/// V6's exact lexically sorted feature list, calculated from authored joint
/// key presence, including disabled records. The host supplies those booleans.
#[derive(Clone, Copy, Debug, Default)]
pub struct JointFeaturePresence {
    pub control: bool,
    pub soft_limits: bool,
    pub drives: bool,
    pub hard_hinge: bool,
}
pub fn expected_features_v6(joints: &[JointFeaturePresence]) -> RootResult<Vec<&'static str>> {
    if joints.iter().any(|j| j.hard_hinge) {
        return Err(error(CODE_UNSUPPORTED_DOMAIN, None, None, "joint.angular_hard_hinge_v1", "V6 first slice excludes hard hinges"));
    }
    let mut out = vec![ROOT_FEATURE];
    if joints.iter().any(|j| j.control) { out.push("pdrd_angular_controls_v1"); }
    if joints.iter().any(|j| j.soft_limits) { out.push("pdrd_angular_soft_limits_v1"); }
    if joints.iter().any(|j| j.drives) { out.push("pdrd_angular_drives_v1"); }
    out.sort_unstable();
    Ok(out)
}
pub fn validate_features_v6(actual: &[String], joints: &[JointFeaturePresence], has_enabled_known_parent_joint: bool) -> RootResult<()> {
    let expected = expected_features_v6(joints)?;
    if !has_enabled_known_parent_joint {
        return Err(error(CODE_UNSUPPORTED_DOMAIN, None, None, "joints.parent_body", "enabled joint from known nonworld parent required"));
    }
    if actual.iter().map(String::as_str).collect::<Vec<_>>() != expected {
        return Err(error(CODE_REQUIRED_FEATURES, None, None, "required_features", "expected exact unique lexically sorted V6 features"));
    }
    Ok(())
}

/// Presence check before any native initialization. A V6 sidecar without its
/// feature or any sidecar on an older/no-joint scene is never ignored.
pub fn validate_sidecar_obligation(joint_version: Option<u32>, root_feature_present: bool, sidecar_present: bool) -> RootResult<()> {
    if matches!(joint_version, Some(7|8|9)) {
        return match (root_feature_present, sidecar_present) {
            (true, true) | (false, false) => Ok(()),
            (true, false) => Err(error(CODE_SIDECAR_MISSING, None, None, "bin/pdrd_root_motion_v1.cbor", "root feature requires sealed sidecar")),
            (false, true) => Err(error(CODE_REQUIRED_FEATURES, None, None, "required_features", "sidecar requires root feature")),
        };
    }
    if joint_version != Some(6) {
        if sidecar_present || root_feature_present {
            return Err(error(CODE_VERSION_OR_KIND, None, None, "bin/pdrd_root_motion_v1.cbor", "root motion sidecar requires V6 joint envelope"));
        }
        return Ok(());
    }
    if !root_feature_present {
        return Err(error(CODE_REQUIRED_FEATURES, None, None, "required_features", "V6 first slice requires root feature"));
    }
    if !sidecar_present {
        return Err(error(CODE_SIDECAR_MISSING, None, None, "bin/pdrd_root_motion_v1.cbor", "required sidecar missing"));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub struct BodyIdentity<'a> { pub uuid: &'a str, pub runtime_body_index: u32 }
/// Inputs are supplied from the already sealed native-frame adapter. Rest
/// coordinates are immutable and include any initial nonuniform scale.
#[derive(Clone, Copy)]
pub struct NativeFrameEvidence<'a> {
    pub context_scene_uuid: &'a str,
    pub context_bodies: &'a [BodyIdentity<'a>],
    pub joint_bodies: &'a [BodyIdentity<'a>],
    pub native_body: BodyIdentity<'a>,
    pub native_body_is_free: bool,
    pub has_enabled_known_parent_joint: bool,
    pub vertex_indices: &'a [u32],
    pub rest_centered_m: &'a [[f64; 3]],
    pub initial_world_m: &'a [[f64; 3]],
    pub fit_tolerance_m: f64,
}
pub fn rotate_world(q: [f64; 4], y: [f64; 3]) -> [f64; 3] {
    let [w, x, yy, z] = q;
    let cross = [yy*y[2]-z*y[1], z*y[0]-x*y[2], x*y[1]-yy*y[0]];
    let cross2 = [yy*cross[2]-z*cross[1], z*cross[0]-x*cross[2], x*cross[1]-yy*cross[0]];
    std::array::from_fn(|i| y[i] + 2.0*(w*cross[i]+cross2[i]))
}
pub fn reconstruct_native_vertices(pose: RootPose, rest_centered_m: &[[f64; 3]]) -> Vec<[f64; 3]> {
    rest_centered_m.iter().map(|&y| {
        let r = rotate_world(pose.rotation_world_wxyz, y);
        std::array::from_fn(|i| pose.center_world_m[i] + r[i])
    }).collect()
}
impl ValidatedRootMotion {
    pub fn validate_initial_native_frame(&self, e: &NativeFrameEvidence<'_>) -> RootResult<f64> {
        self.validate_initial_native_frame_with_body_controls(e,false)
    }
    pub fn validate_initial_native_frame_with_body_controls(&self,e:&NativeFrameEvidence<'_>,body_controls:bool)->RootResult<f64> {
        let body = Some(self.body_uuid.as_str());
        if e.context_scene_uuid != self.scene_uuid {
            return Err(error(CODE_BODY_BINDING, body, None, "scene_uuid", "sidecar/context mismatch"));
        }
        let matches = |list: &[BodyIdentity<'_>]| list.iter().filter(|b| b.uuid == self.body_uuid && b.runtime_body_index == self.runtime_body_index).count() == 1
            && list.iter().filter(|b| b.uuid == self.body_uuid || b.runtime_body_index == self.runtime_body_index).count() == 1;
        if !matches(e.context_bodies) || !matches(e.joint_bodies) ||
            e.native_body.uuid != self.body_uuid || e.native_body.runtime_body_index != self.runtime_body_index {
            return Err(error(CODE_BODY_BINDING, body, None, "body_uuid/runtime_body_index", "context, joint and native row must identify same unique body"));
        }
        if !e.native_body_is_free || (!body_controls && !e.has_enabled_known_parent_joint) {
            return Err(error(CODE_UNSUPPORTED_DOMAIN, body, None, "native_body/joints", "FREE root and enabled known-parent joint required"));
        }
        let n = e.vertex_indices.len();
        if n < 4 || e.rest_centered_m.len() != n || e.initial_world_m.len() != n {
            return Err(error(CODE_BODY_BINDING, body, None, "vertex_membership", "at least four corresponding native vertices required"));
        }
        let mut indices = e.vertex_indices.to_vec(); indices.sort_unstable();
        if indices.windows(2).any(|w| w[0] == w[1]) {
            return Err(error(CODE_BODY_BINDING, body, None, "vertex_membership", "duplicate vertex index"));
        }
        if !e.fit_tolerance_m.is_finite() || e.fit_tolerance_m < 0.0 {
            return Err(error(CODE_INITIAL_POSE, body, Some(0), "fit_tolerance_m", "finite nonnegative adapter tolerance required"));
        }
        let initial=self.sample_native(0.0)?;
        let expected = reconstruct_native_vertices(initial, e.rest_centered_m);
        let mut max_residual = 0.0_f64;
        for (k, (want, actual)) in expected.iter().zip(e.initial_world_m).enumerate() {
            for axis in 0..3 {
                let v = e.rest_centered_m[k][axis];
                let residual = (want[axis] - actual[axis]).abs();
                if !v.is_finite() || !actual[axis].is_finite() || !want[axis].is_finite() || !residual.is_finite() || residual > e.fit_tolerance_m {
                    return Err(error(CODE_INITIAL_POSE, body, Some(0), format!("vertex[{}].component[{axis}]", e.vertex_indices[k]), "native initial vertex does not fit sealed rest/pose"));
                }
                max_residual = max_residual.max(residual);
            }
        }
        Ok(max_residual)
    }
}
