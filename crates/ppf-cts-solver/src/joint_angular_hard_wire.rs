//! Strict V5 `angular_hard_hinge_v1` authoring value.
//!
//! Intended as the eventual `JointWire` field type. Deserialization sees the
//! CBOR map directly; no intermediate JSON map may erase duplicate keys.

use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use serde::ser::SerializeMap;

pub const WIRE_VERSION: u32 = 5;
pub const HARD_KEY: &str = "angular_hard_hinge_v1";
pub const HARD_FEATURE: &str = "pdrd_angular_hard_hinge_v1";
pub const CONTROL_FEATURE: &str = "pdrd_angular_controls_v1";
pub const DRIVE_FEATURE: &str = "pdrd_angular_drives_v1";
pub const SOFT_FEATURE: &str = "pdrd_angular_soft_limits_v1";
pub const COORDINATE_MODEL: &str = "parent_local_swing_twist_x_v1";
// Match the native f32 chart threshold, including its rounding.
pub const CHART_LIMIT_RAD: f64 = (std::f32::consts::PI - 1e-3_f32) as f64;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AngularHardHingeWireV1 {
    pub coordinate_model: String,
    pub twist_x: TwistWireV1,
}

/// Preserve field presence: Serde's ordinary Option decoder treats an
/// explicitly authored CBOR null as absent and would bypass V5 validation.
pub fn present_hard<'de, D: Deserializer<'de>>(d: D) -> Result<Option<AngularHardHingeWireV1>, D::Error> {
    AngularHardHingeWireV1::deserialize(d).map(Some)
}

#[derive(Clone, Debug, PartialEq)]
pub enum TwistWireV1 {
    Free,
    Bounded { lower_rad: f64, upper_rad: f64 },
}

// `Option<f64>` normally interprets a present null as absent. Here presence
// itself is part of the exact free/bounded shape, so null must fail decoding.
fn present_f64<'de, D: Deserializer<'de>>(d: D) -> Result<Option<f64>, D::Error> {
    f64::deserialize(d).map(Some)
}

impl<'de> Deserialize<'de> for TwistWireV1 {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct RawTwist {
            mode: String,
            #[serde(default, deserialize_with = "present_f64")]
            lower_rad: Option<f64>,
            #[serde(default, deserialize_with = "present_f64")]
            upper_rad: Option<f64>,
        }
        let raw = RawTwist::deserialize(d)?;
        match (raw.mode.as_str(), raw.lower_rad, raw.upper_rad) {
            ("free", None, None) => Ok(Self::Free),
            ("bounded", Some(lower_rad), Some(upper_rad)) =>
                Ok(Self::Bounded { lower_rad, upper_rad }),
            _ => Err(de::Error::custom("invalid hard twist mode or bound-key shape")),
        }
    }
}

impl Serialize for TwistWireV1 {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut map = s.serialize_map(Some(if matches!(self, Self::Free) { 1 } else { 3 }))?;
        match self {
            Self::Free => map.serialize_entry("mode", "free")?,
            Self::Bounded { lower_rad, upper_rad } => {
                map.serialize_entry("mode", "bounded")?;
                map.serialize_entry("lower_rad", lower_rad)?;
                map.serialize_entry("upper_rad", upper_rad)?;
            }
        }
        map.end()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HardWireError {
    pub code: &'static str,
    pub path: &'static str,
}
impl std::fmt::Display for HardWireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at {}", self.code, self.path)
    }
}
impl std::error::Error for HardWireError {}
type WireResult<T> = std::result::Result<T, HardWireError>;
fn invalid(code: &'static str, path: &'static str) -> HardWireError { HardWireError { code, path } }

pub const CODE_INVALID: &str = "JOINT_ANGULAR_EXTENSION_INVALID";
pub const CODE_INTERVAL: &str = "JOINT_ANGULAR_EXTENSION_INVALID_INTERVAL";
pub const CODE_OUT_OF_DOMAIN: &str = "JOINT_ANGULAR_EXTENSION_OUT_OF_DOMAIN";
pub const CODE_REQUIRES_CONTROL: &str = "JOINT_ANGULAR_EXTENSION_REQUIRES_CONTROL";
pub const CODE_HARD_CONFLICT: &str = "JOINT_ANGULAR_HARD_EXTENSION_CONFLICT";
pub const CODE_REQUIRED_FEATURES: &str = "JOINT_G2_REQUIRED_FEATURES";

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ValidatedTwistV1 {
    Free,
    Bounded { lower_rad: f32, upper_rad: f32 },
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ValidatedHardHingeV1 { pub twist_x: ValidatedTwistV1 }

fn finite_f32(value: f64, path: &'static str) -> WireResult<f32> {
    let narrowed = value as f32;
    if !value.is_finite() || !narrowed.is_finite() ||
        (value != 0.0 && value.abs() < f32::from_bits(1) as f64) ||
        (value != 0.0 && narrowed == 0.0) {
        return Err(invalid(CODE_INVALID, path));
    }
    Ok(narrowed)
}

impl AngularHardHingeWireV1 {
    /// Validate numerics before packing, including disabled author records.
    /// Equality after f32 conversion is refused; it is not a strict bound.
    pub fn validate(&self) -> WireResult<ValidatedHardHingeV1> {
        if self.coordinate_model != COORDINATE_MODEL {
            return Err(invalid(CODE_INVALID, "angular_hard_hinge_v1.coordinate_model"));
        }
        let twist_x = match self.twist_x {
            TwistWireV1::Free => ValidatedTwistV1::Free,
            TwistWireV1::Bounded { lower_rad, upper_rad } => {
                const LOWER: &str = "angular_hard_hinge_v1.twist_x.lower_rad";
                const UPPER: &str = "angular_hard_hinge_v1.twist_x.upper_rad";
                let lower = finite_f32(lower_rad, LOWER)?;
                let upper = finite_f32(upper_rad, UPPER)?;
                if lower_rad.abs() >= CHART_LIMIT_RAD || (lower as f64).abs() >= CHART_LIMIT_RAD {
                    return Err(invalid(CODE_OUT_OF_DOMAIN, LOWER));
                }
                if upper_rad.abs() >= CHART_LIMIT_RAD || (upper as f64).abs() >= CHART_LIMIT_RAD {
                    return Err(invalid(CODE_OUT_OF_DOMAIN, UPPER));
                }
                if lower_rad >= upper_rad || lower >= upper {
                    return Err(invalid(CODE_INTERVAL, LOWER));
                }
                ValidatedTwistV1::Bounded { lower_rad: lower, upper_rad: upper }
            }
        };
        Ok(ValidatedHardHingeV1 { twist_x })
    }
}

/// Exact V5 list from all authored joint keys. A disabled hard object still
/// requires the hard feature. The caller must compare this against the actual
/// required_features vector before native initialization.
#[derive(Clone, Copy, Debug, Default)]
pub struct JointFeaturePresence {
    pub angular_control: bool,
    pub soft_limits: bool,
    pub drives: bool,
    pub hard_hinge: bool,
}
pub fn required_features_v5(joints: &[JointFeaturePresence]) -> Vec<&'static str> {
    let mut out = Vec::new();
    if joints.iter().any(|j| j.angular_control) { out.push(CONTROL_FEATURE); }
    if joints.iter().any(|j| j.drives) { out.push(DRIVE_FEATURE); }
    if joints.iter().any(|j| j.hard_hinge) { out.push(HARD_FEATURE); }
    if joints.iter().any(|j| j.soft_limits) { out.push(SOFT_FEATURE); }
    out // Already in lexical order; duplicates cannot occur.
}
pub fn validate_required_features_v5(joints: &[JointFeaturePresence], actual: &[String]) -> WireResult<()> {
    if actual.iter().map(String::as_str).collect::<Vec<_>>() != required_features_v5(joints) {
        Err(invalid(CODE_REQUIRED_FEATURES, "required_features"))
    } else { Ok(()) }
}
