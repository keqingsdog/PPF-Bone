//! Strict V7 six-axis hard joint authoring value.
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use serde::ser::SerializeMap;

pub const FEATURE: &str = "pdrd_joint_d6_hard_v1";
pub const MODEL: &str = "parent_local_d6_swing_twist_x_v1";
pub const CHART_LIMIT_RAD: f64 = (std::f32::consts::PI - 1e-3_f32) as f64;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct D6Wire {
    pub coordinate_model: String,
    pub rest_translation_m: [f64; 3],
    pub translation: AxisMap,
    pub rotation: AxisMap,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AxisMap { pub x: Axis, pub y: Axis, pub z: Axis }
impl AxisMap { pub fn ordered(&self) -> [&Axis; 3] { [&self.x, &self.y, &self.z] } }

#[derive(Clone, Debug, PartialEq)]
pub enum Axis { Free, Limited { lower: f64, upper: f64 }, Locked { value: f64 } }
fn present_f64<'de, D: Deserializer<'de>>(d: D) -> Result<Option<f64>, D::Error> {
    f64::deserialize(d).map(Some)
}
impl<'de> Deserialize<'de> for Axis {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Raw {
            mode: String,
            #[serde(default, deserialize_with = "present_f64")] lower: Option<f64>,
            #[serde(default, deserialize_with = "present_f64")] upper: Option<f64>,
            #[serde(default, deserialize_with = "present_f64")] value: Option<f64>,
        }
        let r = Raw::deserialize(d)?;
        match (r.mode.as_str(), r.lower, r.upper, r.value) {
            ("free", None, None, None) => Ok(Self::Free),
            ("limited", Some(lower), Some(upper), None) => Ok(Self::Limited { lower, upper }),
            ("locked", None, None, Some(value)) => Ok(Self::Locked { value }),
            _ => Err(de::Error::custom("invalid D6 axis mode or exact bound-key shape")),
        }
    }
}
impl Serialize for Axis {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut m = s.serialize_map(Some(match self { Self::Free => 1, Self::Limited { .. } => 3, Self::Locked { .. } => 2 }))?;
        match self {
            Self::Free => m.serialize_entry("mode", "free")?,
            Self::Limited { lower, upper } => { m.serialize_entry("mode", "limited")?; m.serialize_entry("lower", lower)?; m.serialize_entry("upper", upper)?; },
            Self::Locked { value } => { m.serialize_entry("mode", "locked")?; m.serialize_entry("value", value)?; },
        }
        m.end()
    }
}
pub fn present_d6<'de, D: Deserializer<'de>>(d: D) -> Result<Option<D6Wire>, D::Error> {
    D6Wire::deserialize(d).map(Some)
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct D6Error { pub code: &'static str, pub field: &'static str, pub detail: &'static str }
impl std::fmt::Display for D6Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}: {}: {}", self.code, self.field, self.detail) }
}
impl std::error::Error for D6Error {}
fn bad(code: &'static str, field: &'static str, detail: &'static str) -> D6Error { D6Error { code, field, detail } }
fn narrow(x: f64, field: &'static str) -> Result<f32, D6Error> {
    if !x.is_finite() { return Err(bad("G2_D6_INVALID_AXIS", field, "nonfinite value")); }
    let y = x as f32;
    if !y.is_finite() || (x != 0.0 && y == 0.0) {
        return Err(bad("G2_D6_INVALID_AXIS", field, "f32 overflow or underflow"));
    }
    Ok(y)
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ValidatedD6 { pub modes: [u32; 6], pub lower: [f32; 6], pub upper: [f32; 6], pub rest: [f32; 3] }
impl D6Wire {
    pub fn validate(&self) -> Result<ValidatedD6, D6Error> {
        if self.coordinate_model != MODEL { return Err(bad("G2_D6_UNSUPPORTED_DOMAIN", "coordinate_model", "unknown D6 coordinate model")); }
        let mut out = ValidatedD6 { modes: [0; 6], lower: [0.0; 6], upper: [0.0; 6], rest: [0.0; 3] };
        for i in 0..3 { out.rest[i] = narrow(self.rest_translation_m[i], "rest_translation_m")?; }
        for (i, axis) in self.translation.ordered().into_iter().chain(self.rotation.ordered()).enumerate() {
            let angular = i >= 3;
            match axis {
                Axis::Free => {},
                Axis::Limited { lower, upper } => {
                    let l = narrow(*lower, "lower")?; let u = narrow(*upper, "upper")?;
                    if l >= u || (angular && !((l as f64) > -CHART_LIMIT_RAD && (u as f64) < CHART_LIMIT_RAD && *lower > -CHART_LIMIT_RAD && *upper < CHART_LIMIT_RAD)) {
                        return Err(bad("G2_D6_INVALID_AXIS", "lower/upper", "invalid narrowed interval or angular chart domain"));
                    }
                    out.modes[i] = 1; out.lower[i] = l; out.upper[i] = u;
                },
                Axis::Locked { value } => {
                    let v = narrow(*value, "value")?;
                    if angular && !(*value > -CHART_LIMIT_RAD && *value < CHART_LIMIT_RAD && (v as f64) > -CHART_LIMIT_RAD && (v as f64) < CHART_LIMIT_RAD) {
                        return Err(bad("G2_D6_INVALID_AXIS", "value", "angular lock outside chart"));
                    }
                    out.modes[i] = 2; out.lower[i] = v; out.upper[i] = v;
                },
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn exact_axis_and_narrowing() {
        let ok = D6Wire { coordinate_model: MODEL.into(), rest_translation_m: [0.0; 3],
            translation: AxisMap { x: Axis::Limited { lower: -0.02, upper: 0.03 }, y: Axis::Free, z: Axis::Locked { value: 0.0 } },
            rotation: AxisMap { x: Axis::Free, y: Axis::Free, z: Axis::Free } };
        assert_eq!(ok.validate().unwrap().modes, [1,0,2,0,0,0]);
        let mut bad = ok.clone(); bad.translation.x = Axis::Limited { lower: 1.0, upper: 1.0 + f64::EPSILON };
        assert_eq!(bad.validate().unwrap_err().code, "G2_D6_INVALID_AXIS");
        let mut bad = ok.clone(); bad.rest_translation_m[0] = f64::MIN_POSITIVE;
        assert!(bad.validate().is_err());
        for text in [r#"{"mode":"free","value":null}"#,r#"{"mode":"free","upper":1}"#,r#"{"mode":"locked","value":0,"lower":0}"#] {
            assert!(serde_json::from_str::<Axis>(text).is_err());
        }
    }
}
