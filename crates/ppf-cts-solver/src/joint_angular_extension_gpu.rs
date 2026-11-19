//! A1.2 angular extension transport: canonical V4 validation and record packing.
//!
//! STAGED FILE. Intended destination:
//! `crates/ppf-cts-solver/src/joint_angular_extension_gpu.rs`
//! plus `mod joint_angular_extension_gpu;` in `src/main.rs`. Nothing here is
//! installed by the staging step; see `angular-a12-handoff.md`.
//!
//! This is the OLD pure parser (`OLD/crates/ppf-cts-solver/src/
//! joint_angular_extension_gpu.rs`) carried onto the CURRENT official tree. The
//! record ABI, the canonical field spellings, the refusal codes and the budgets
//! are the old contract's exactly, because they are the authoring/wire contract
//! the design fixes and not a property of the engine it is being ported to. What
//! changed is the SHAPE OF THE ENTRY POINT and the types it names:
//!
//! * It takes IMMUTABLE REFERENCES the caller already owns — the full joint
//!   order as `&[JointExtensionInput]`, each field as a `&serde_json::Value` —
//!   and returns owned packed records. It does NOT require simultaneous changes
//!   to the live `joint.rs`: the caller builds the input slice from
//!   `OwnedJointScene` (the exact snippet is in the handoff) and the only thing
//!   the live envelope owes is that its two V4 fields decode to a raw `Value`.
//! * It never names a native symbol, an upload, or a session. The device-side
//!   half is `zz_joint_ext/angular_extension.kernel.cpp`, staged beside this
//!   file.
//!
//! TWO V4 KEYS, TWO OPTIONAL RECORD FAMILIES, ONE ROW PER JOINT.
//! `angular_soft_limits_v1` is one 56-byte row per joint, in validated joint
//! order, so row `i` is joint `i`. `angular_drives_v1` is a variable-length
//! family of 32-byte drive rows plus 16-byte key rows; the drive rows come out
//! sorted by (joint, axis, mode) and their key spans partition the key array in
//! that same order, which is what lets the device side reach a joint's channels
//! without a search.
//!
//! UNITS ARE PHYSICAL: radians, seconds, N*m, N*m/rad, N*m*s/rad. There is no
//! `dt^2` anywhere and no frame index anywhere.
//!
//! NOTHING IS SILENTLY ACCEPTED. Every object is matched against its exact key
//! set (required AND optional), so a misspelled, missing, extra or explicitly
//! null field is a refusal rather than a default. A magnitude that would flush
//! to zero in the 16-byte key's `f32` payload, a non-finite number, a key whose
//! first time is not exactly zero, a non-increasing key time, and a bound or
//! position target at or outside the chart's principal domain are all typed
//! refusals.

use serde_json::{Map, Value};

/// The envelope version these records belong to. V4 is the first version that
/// may carry them; V2 and V3 are refused by the live version gate, not here.
pub const WIRE_VERSION: u32 = 4;
pub const FEATURE_CONTROL: &str = "pdrd_angular_controls_v1";
pub const FEATURE_SOFT: &str = "pdrd_angular_soft_limits_v1";
pub const FEATURE_SWING_CONE: &str = "pdrd_angular_swing_cone_v1";
pub const FEATURE_DRIVES: &str = "pdrd_angular_drives_v1";
/// The one accepted chart, spelled exactly as the native chart constant is.
pub const COORDINATE_MODEL: &str = "parent_local_swing_twist_x_v1";
pub const CONTINUOUS_MODEL: &str = "parent_local_swing_twist_x_continuous_v2";
pub const CONTINUOUS_DOMAIN_RAD: f64 = 512.0;
/// Match the native f32 chart threshold, INCLUDING its rounding: the native
/// domain is `(f32::PI - 1e-3_f32) as f64`, so computing it in f64 would admit
/// a bound the device chart refuses.
pub const DOMAIN_RAD: f64 = (std::f32::consts::PI - 1e-3_f32) as f64;
pub const MAX_KEYS_PER_DRIVE: usize = 1024;
pub const MAX_TOTAL_BYTES: usize = 16 * 1024 * 1024;
/// Capability bits the loaded backend must advertise: control, soft, drives and
/// the telemetry family, one bit each in the canonical order.
pub const REQUIRED_CAPABILITY_BITS: u64 = 0b1111;
const AXES: [&str; 3] = ["twist_x", "swing_y", "swing_z"];
const MODES: [&str; 3] = ["position", "velocity", "torque"];

/// 56 bytes, alignment 4. Field order and offsets mirror
/// `AngularSoftLimitGpuV1` in the neutral kernel; `layout` below asserts them.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AngularSoftLimitGpuV1 {
    pub joint_index: u32,
    pub enabled_axis_mask: u32,
    pub lower_rad: [f32; 3],
    pub upper_rad: [f32; 3],
    pub stiffness_nm_per_rad: [f32; 3],
    pub swing_cone_enabled: u32,
    pub swing_cone_half_angle_rad: f32,
    pub swing_cone_stiffness_nm_per_rad: f32,
}

/// 32 bytes, alignment 4. `mode` is 1/2/3 for position/velocity/torque.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AngularDriveGpuV1 {
    pub joint_index: u32,
    pub axis: u32,
    pub mode: u32,
    pub enabled: u32,
    pub gain: f32,
    pub max_effort_nm: f32,
    pub key_offset: u32,
    pub key_count: u32,
}

/// 16 bytes, alignment 8. `time_s` stays an `f64` because the canonical record
/// is 16 bytes with an 8-byte alignment and the time is the one field whose
/// precision matters across a long schedule.
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AngularDriveKeyGpuV1 {
    pub time_s: f64,
    pub target_value: f32,
    pub reserved: u32,
}

/// Neutral device transport, distinct from the canonical f64-time wire key.
/// Kernel seam records use f32 scalars, including physical schedule seconds.
/// Convert with `device_keys()` and reject unrepresentable/colliding times.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AngularDriveDeviceKeyV1 {
    pub time_s: f32,
    pub target_value: f32,
    pub reserved0: f32,
    pub reserved1: f32,
}

/// Historical canonical backend capability layout; retained for ABI validation.
/// This module does not load a DLL or provide this symbol. A caller may invoke
/// the validator only on a capability record obtained from an actual backend.
/// Neutral key/telemetry layouts are separate and are not advertised by this POD.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PpfAngularExtensionCapabilitiesV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub feature_bits: u64,
    pub control_stride: u32,
    pub limit_stride: u32,
    pub drive_stride: u32,
    pub key_stride: u32,
    pub telemetry_stride: u32,
    pub max_keys_per_drive: u32,
    pub max_total_bytes: u64,
}

/// Per-joint telemetry row, one slot per joint, written at the accepted sample
/// time. Source order is limit, position, velocity, torque; channel target order
/// is position, velocity, torque; saturation bits are position=1, velocity=2,
/// torque=4.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AngularExtensionTelemetryV1 {
    pub sample_time_s: f64,
    pub coordinate_rad: [f32; 3],
    pub start_coordinate_rad: [f32; 3],
    pub sampled_target: [[f32; 3]; 3],
    pub source_energy_j: [[f32; 4]; 3],
    pub source_gradient_nm: [[f32; 4]; 3],
    pub gn_weight: [f32; 3],
    pub lower_excess_rad: [f32; 3],
    pub upper_excess_rad: [f32; 3],
    pub reaction_world_nm: [f32; 6],
    pub saturation_mask: [u32; 3],
}

/// Compile-time ABI assertions. A silent layout drift here is a device kernel
/// reading a different field than the host wrote, which is why these are
/// `const` assertions rather than tests.
macro_rules! layout {
    ($t:ty, $size:expr, $align:expr, {$($field:ident: $offset:expr),+ $(,)?}) => {
        const _: () = assert!(std::mem::size_of::<$t>() == $size);
        const _: () = assert!(std::mem::align_of::<$t>() == $align);
        $(const _: () = assert!(std::mem::offset_of!($t, $field) == $offset);)+
    };
}
layout!(AngularSoftLimitGpuV1, 56, 4, {joint_index: 0, enabled_axis_mask: 4,
    lower_rad: 8, upper_rad: 20, stiffness_nm_per_rad: 32,
    swing_cone_enabled: 44, swing_cone_half_angle_rad: 48, swing_cone_stiffness_nm_per_rad: 52});
layout!(AngularDriveGpuV1, 32, 4, {joint_index: 0, axis: 4, mode: 8, enabled: 12,
    gain: 16, max_effort_nm: 20, key_offset: 24, key_count: 28});
layout!(AngularDriveKeyGpuV1, 16, 8, {time_s: 0, target_value: 8, reserved: 12});
layout!(AngularDriveDeviceKeyV1, 16, 4, {time_s: 0, target_value: 4, reserved0: 8, reserved1: 12});
layout!(PpfAngularExtensionCapabilitiesV1, 48, 8, {abi_version: 0, struct_size: 4,
    feature_bits: 8, control_stride: 16, limit_stride: 20, drive_stride: 24,
    key_stride: 28, telemetry_stride: 32, max_keys_per_drive: 36, max_total_bytes: 40});
layout!(AngularExtensionTelemetryV1, 240, 8, {sample_time_s: 0, coordinate_rad: 8,
    start_coordinate_rad: 20, sampled_target: 32, source_energy_j: 68,
    source_gradient_nm: 116, gn_weight: 164, lower_excess_rad: 176,
    upper_excess_rad: 188, reaction_world_nm: 200, saturation_mask: 224});

/// The 104-byte neutral rigid state's rotational block offset, asserted where
/// the reduced indices are built rather than assumed. The extension family
/// touches ROTATIONAL slots only: `base + 6*body + 3 .. base + 6*body + 6`.
pub const REDUCED_ROTATIONAL_OFFSET: u32 = 3;
pub const REDUCED_BODY_STRIDE: u32 = 6;
/// The directional control row these records ride on. Unchanged in A1.2.
pub const CONTROL_RECORD_BYTES: usize = 88;
pub const SOFT_RECORD_BYTES: usize = 56;
/// Included verbatim by the standard identity generator in the loaded-backend
/// layout identity. Keep offsets explicit so old 44-byte backends are refused.
pub const ANGULAR_EXTENSION_LAYOUT_SPEC: &str = "AngularSoftLimitGpuV1:size=56;align=4;joint_index:u32@0;enabled_axis_mask:u32@4;lower_rad:f32[3]@8;upper_rad:f32[3]@20;stiffness_nm_per_rad:f32[3]@32;swing_cone_enabled:u32@44;swing_cone_half_angle_rad:f32@48;swing_cone_stiffness_nm_per_rad:f32@52;swing-cone-soft-radial-semantic-v1";

pub const DRIVE_RECORD_BYTES: usize = 32;
pub const KEY_RECORD_BYTES: usize = 16;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtensionError {
    /// Stable machine-readable refusal code, spelled as the contract does.
    pub code: &'static str,
    /// Dotted path to the refused field, starting at `joints[i].`.
    pub path: String,
}
impl std::fmt::Display for ExtensionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at {}", self.code, self.path)
    }
}
impl std::error::Error for ExtensionError {}
type Result<T> = std::result::Result<T, ExtensionError>;
fn bad(code: &'static str, path: impl Into<String>) -> ExtensionError {
    ExtensionError { code, path: path.into() }
}
/// Codes. These are the canonical ones the old contract established and the
/// frontend already raises, so the same refusal is greppable through the whole
/// stack. The `required_features` mismatch uses the live gate's own code.
const INVALID: &str = "JOINT_ANGULAR_EXTENSION_INVALID";
const DOMAIN: &str = "JOINT_ANGULAR_EXTENSION_OUT_OF_DOMAIN";
const INTERVAL: &str = "JOINT_ANGULAR_EXTENSION_INVALID_INTERVAL";
const ORDER: &str = "JOINT_ANGULAR_EXTENSION_KEY_ORDER";
const BUDGET: &str = "JOINT_ANGULAR_EXTENSION_KEY_LIMIT";
const INTERPOLATION: &str = "JOINT_ANGULAR_EXTENSION_UNSUPPORTED_INTERPOLATION";
const CONFLICT: &str = "JOINT_G2_LEGACY_CONFLICT";
const REQUIRES_CONTROL: &str = "JOINT_ANGULAR_EXTENSION_REQUIRES_CONTROL";
const FEATURES: &str = "JOINT_G2_REQUIRED_FEATURES";

/// One joint's V4 extension surface, as the caller already holds it.
///
/// `soft` and `drives` are raw V4 object values, preserving strict scalar types
/// until validation runs. One entry is required for EVERY ordered joint,
/// including a joint with no extension at all, because the soft family emits
/// one row per joint and the index of that row IS the joint's identity.
#[derive(Clone, Copy, Debug)]
pub struct JointExtensionInput<'a> {
    /// True when this joint's `angular_control_v1` record is PRESENT. The
    /// extension rides on that row's chart and Jacobian, so an extension without
    /// it is refused rather than evaluated in some other coordinate system.
    pub control_present: bool,
    pub control_model: Option<&'a str>,
    pub soft: Option<&'a Value>,
    pub drives: Option<&'a Value>,
    /// True when this joint's legacy isotropic angular coefficients
    /// (`angle_stiffness_Nm_per_rad`, `angle_damping_Nms_per_rad`,
    /// `max_angle_rad`) are not all exactly zero.
    pub legacy_angular_nonzero: bool,
}

impl<'a> JointExtensionInput<'a> {
    /// The absent row: no extension, no directional control, no legacy angular
    /// term. A scene built entirely from these packs to `None`.
    pub const EMPTY: Self = Self {
        control_present: false,
        control_model: None,
        soft: None,
        drives: None,
        legacy_angular_nonzero: false,
    };
}

/// The packed, upload-ready records. One soft row per joint; drive rows sorted
/// by (joint, axis, mode); key rows in that same order with contiguous spans.
#[derive(Clone, Debug)]
pub struct PackedAngularExtensionV1 {
    pub limits: Vec<AngularSoftLimitGpuV1>,
    pub drives: Vec<AngularDriveGpuV1>,
    pub keys: Vec<AngularDriveKeyGpuV1>,
}
impl PackedAngularExtensionV1 {
    pub fn total_bytes(&self) -> usize {
        self.limits.len() * SOFT_RECORD_BYTES
            + self.drives.len() * DRIVE_RECORD_BYTES
            + self.keys.len() * KEY_RECORD_BYTES
    }
    /// The (limit, drive, key) counts a launcher takes, refused rather than
    /// truncated when any of them leaves `u32`.
    pub fn checked_upload_counts(&self) -> Result<(u32, u32, u32)> {
        Ok((
            u32::try_from(self.limits.len()).map_err(|_| bad(BUDGET, "limits"))?,
            u32::try_from(self.drives.len()).map_err(|_| bad(BUDGET, "drives"))?,
            u32::try_from(self.keys.len()).map_err(|_| bad(BUDGET, "keys"))?,
        ))
    }
    /// Explicit canonical wire -> neutral device conversion. The native driver
    /// must upload this array, never `keys`, which has a different time layout.
    pub fn device_keys(&self) -> Result<Vec<AngularDriveDeviceKeyV1>> {
        let mut device = Vec::with_capacity(self.keys.len());
        for (i, k) in self.keys.iter().enumerate() {
            let time = k.time_s as f32;
            if !time.is_finite() || time < 0.0 || (k.time_s != 0.0 && (time == 0.0 || time.is_subnormal())) {
                return Err(bad("JOINT_ANGULAR_EXTENSION_DEVICE_TIME", format!("keys[{i}].time_s")));
            }
            device.push(AngularDriveDeviceKeyV1 { time_s: time, target_value: k.target_value, reserved0: 0.0, reserved1: 0.0 });
        }
        for d in &self.drives {
            self.key_span(d)?;
            let start=d.key_offset as usize;
            let end=start+d.key_count as usize;
            if device[start..end].windows(2).any(|w| w[1].time_s <= w[0].time_s) {
                return Err(bad("JOINT_ANGULAR_EXTENSION_DEVICE_TIME", format!("joints[{}].angular_drives_v1", d.joint_index)));
            }
        }
        Ok(device)
    }
    /// The exact key span one drive row owns. Refuses a span that leaves the
    /// packed array instead of slicing a shorter run.
    pub fn key_span(&self, drive: &AngularDriveGpuV1) -> Result<&[AngularDriveKeyGpuV1]> {
        let start = usize::try_from(drive.key_offset).map_err(|_| bad(BUDGET, "keys"))?;
        let count = usize::try_from(drive.key_count).map_err(|_| bad(BUDGET, "keys"))?;
        let end = start.checked_add(count).ok_or_else(|| bad(BUDGET, "keys"))?;
        self.keys.get(start..end).ok_or_else(|| bad(BUDGET, "keys"))
    }
}

/// The exact, ORDERED feature list V4 must advertise for this joint sequence.
///
/// The order is lexical and fixed: controls, drives, soft limits. A scene that
/// advertises a superset, a subset, a duplicate, a reordering or an unknown
/// feature is the same refusal, which is what makes the list a statement about
/// the records rather than a hint.
pub fn required_features(joints: &[JointExtensionInput<'_>]) -> Vec<&'static str> {
    let mut out = Vec::new();
    if joints.iter().any(|j| j.control_present) {
        out.push(FEATURE_CONTROL);
    }
    if joints.iter().any(|j| j.drives.is_some()) {
        out.push(FEATURE_DRIVES);
    }
    if joints.iter().any(|j| j.soft.is_some()) {
        out.push(FEATURE_SOFT);
    }
    if joints.iter().any(|j| has_swing_cone(j.soft)) {
        out.push(FEATURE_SWING_CONE);
    }
    out
}

/// Presence, including a disabled or malformed authored cone, requires the new
/// feature. Validation below refuses malformed objects rather than coercing.
pub fn has_swing_cone(soft: Option<&Value>) -> bool {
    soft.and_then(Value::as_object).is_some_and(|m| m.contains_key("swing_cone"))
}

// ---------------------------------------------------------------------------
// Strict value access. Every helper refuses rather than coerces.
// ---------------------------------------------------------------------------

/// An object whose key set is EXACTLY `required + optional`.
fn object<'a>(
    v: &'a Value,
    path: &str,
    required: &[&str],
    optional: &[&str],
) -> Result<&'a Map<String, Value>> {
    let m = v.as_object().ok_or_else(|| bad(INVALID, path))?;
    if required.iter().any(|k| !m.contains_key(*k))
        || m.keys().any(|k| !required.contains(&k.as_str()) && !optional.contains(&k.as_str()))
    {
        return Err(bad(INVALID, path));
    }
    Ok(m)
}
/// A required field. A present `null` decodes to `Value::Null`, which the type
/// helpers then refuse, so an explicit null is never read as "absent".
fn field<'a>(m: &'a Map<String, Value>, key: &str, path: &str) -> Result<&'a Value> {
    m.get(key).ok_or_else(|| bad(INVALID, format!("{path}.{key}")))
}
fn name(v: &Value, expected: &str, path: &str) -> Result<()> {
    if v.as_str() == Some(expected) {
        Ok(())
    } else {
        Err(bad(INVALID, path))
    }
}
/// Strict `bool`: the JSON/CBOR boolean, not a number that happens to be 0 or 1.
fn boolean(v: &Value, path: &str) -> Result<bool> {
    v.as_bool().ok_or_else(|| bad(INVALID, path))
}
/// A finite real, optionally non-negative. `as_f64` is exact for every CBOR
/// integer and float the envelope can carry, so a value that is not a real
/// number at all is refused here.
fn real(v: &Value, path: &str, nonnegative: bool) -> Result<f64> {
    let n = v.as_f64().ok_or_else(|| bad(INVALID, path))?;
    if !n.is_finite() || (nonnegative && n < 0.0) {
        return Err(bad(INVALID, path));
    }
    Ok(n)
}
/// A real that survives narrowing to `f32`: finite, and not a magnitude that
/// would flush to zero. A value that changes meaning under the narrowing is a
/// refusal, because the device only ever sees the narrowed coefficient.
fn narrow(v: &Value, path: &str, nonnegative: bool) -> Result<f32> {
    let n = real(v, path, nonnegative)?;
    let x = n as f32;
    if !x.is_finite()
        || (n != 0.0 && x == 0.0)
        || (n != 0.0 && n.abs() < f32::from_bits(1) as f64)
    {
        return Err(bad(INVALID, path));
    }
    Ok(x)
}
/// Strictly inside the chart's principal domain: `abs(x) < pi - 1e-3`, the
/// same threshold and the same rounding the native chart refuses at.
fn in_domain(x: f32, path: &str) -> Result<f32> {
    if (x as f64).abs() >= DOMAIN_RAD {
        Err(bad(DOMAIN, path))
    } else {
        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// Soft limits.
// ---------------------------------------------------------------------------

/// `joints[i].angular_soft_limits_v1`.
///
/// All three axis keys are REQUIRED even when disabled: the record is one
/// full row per joint and a missing axis would silently mean "no bound", which
/// is a different physical request. A disabled axis is still fully validated and
/// simply does not set its mask bit. Endpoints are SIGNED and preserved.
fn parse_soft(v: &Value, joint: u32, model:&str) -> Result<AngularSoftLimitGpuV1> {
    let path = format!("joints[{joint}].angular_soft_limits_v1");
    let m = object(
        v,
        &path,
        &["coordinate_model", "twist_x", "swing_y", "swing_z"],
        &["swing_cone"],
    )?;
    name(field(m, "coordinate_model", &path)?, model, &path)?;
    let mut row = AngularSoftLimitGpuV1 {
        joint_index: joint,
        enabled_axis_mask: 0,
        lower_rad: [0.0; 3],
        upper_rad: [0.0; 3],
        stiffness_nm_per_rad: [0.0; 3],
        swing_cone_enabled: 0,
        swing_cone_half_angle_rad: 0.0,
        swing_cone_stiffness_nm_per_rad: 0.0,
    };
    for (axis, key) in AXES.iter().enumerate() {
        let p = format!("{path}.{key}");
        let a = object(
            field(m, key, &path)?,
            &p,
            &["enabled", "lower_rad", "upper_rad", "stiffness_Nm_per_rad"],
            &[],
        )?;
        if boolean(field(a, "enabled", &p)?, &p)? {
            row.enabled_axis_mask |= 1 << axis;
        }
        let bound=|x:f32| if model==crate::material_spin_history::MODEL || (axis==0 && model==CONTINUOUS_MODEL) {if (x as f64).abs()<CONTINUOUS_DOMAIN_RAD {Ok(x)} else {Err(bad(DOMAIN,&p))}} else {in_domain(x,&p)};
        let lower = bound(narrow(field(a, "lower_rad", &p)?, &p, false)?)?;
        let upper = bound(narrow(field(a, "upper_rad", &p)?, &p, false)?)?;
        if lower > upper {
            return Err(bad(INTERVAL, &p));
        }
        row.lower_rad[axis] = lower;
        row.upper_rad[axis] = upper;
        row.stiffness_nm_per_rad[axis] = narrow(field(a, "stiffness_Nm_per_rad", &p)?, &p, true)?;
    }
    if let Some(value) = m.get("swing_cone") {
        let p = format!("{path}.swing_cone");
        let cone = object(value, &p, &["enabled", "half_angle_rad", "stiffness_Nm_per_rad"], &[])?;
        row.swing_cone_enabled = u32::from(boolean(field(cone, "enabled", &p)?, &format!("{p}.enabled"))?);
        let angle_path = format!("{p}.half_angle_rad");
        let angle = in_domain(narrow(field(cone, "half_angle_rad", &p)?, &angle_path, true)?, &angle_path)?;
        if angle <= 0.0 || angle.is_subnormal() { return Err(bad(DOMAIN, angle_path)); }
        row.swing_cone_half_angle_rad = angle;
        let k_path = format!("{p}.stiffness_Nm_per_rad");
        let k = narrow(field(cone, "stiffness_Nm_per_rad", &p)?, &k_path, true)?;
        if k.is_subnormal() { return Err(bad(INVALID, k_path)); }
        row.swing_cone_stiffness_nm_per_rad = k;
        if row.swing_cone_enabled != 0 && row.enabled_axis_mask & 0b110 != 0 {
            return Err(bad("JOINT_ANGULAR_SWING_CONE_CONFLICT", p));
        }
    }
    Ok(row)
}

// ---------------------------------------------------------------------------
// Drives.
// ---------------------------------------------------------------------------

fn gain_key(mode: usize) -> &'static str {
    ["gain_Nm_per_rad", "gain_Nms_per_rad", "gain"][mode]
}
fn value_key(mode: usize) -> &'static str {
    ["value_rad", "value_rad_per_s", "value_Nm"][mode]
}

/// `linear_hold` at a physical time in seconds: the first key holds before it,
/// the last holds after it, linear in between. The packed array has at least one
/// key and strictly increasing times by construction.
fn sample(keys: &[AngularDriveKeyGpuV1], t: f64) -> f32 {
    if t <= keys[0].time_s {
        return keys[0].target_value;
    }
    let last = keys.len() - 1;
    if t >= keys[last].time_s {
        return keys[last].target_value;
    }
    let hi = keys.partition_point(|k| k.time_s < t);
    let a = keys[hi - 1];
    let b = keys[hi];
    let w = (t - a.time_s) / (b.time_s - a.time_s);
    ((a.target_value as f64) * (1.0 - w) + (b.target_value as f64) * w) as f32
}

/// The combined swing position target must stay inside the chart's swing disk.
///
/// Checked only when BOTH swing position channels are enabled: with one channel
/// disabled the other's target alone is already inside the domain per key. The
/// union of the two channels' key times bounds every segment, because the disk
/// is convex and a linear-hold segment interior cannot leave a convex set whose
/// endpoints are inside it, so sampling at key times is sufficient.
fn check_swing_disk(out: &PackedAngularExtensionV1, joint: u32) -> Result<()> {
    let position = |axis: u32| {
        out.drives
            .iter()
            .find(|d| d.joint_index == joint && d.axis == axis && d.mode == 1 && d.enabled == 1)
    };
    let (Some(y), Some(z)) = (position(1), position(2)) else {
        return Ok(());
    };
    let yk = out.key_span(y)?;
    let zk = out.key_span(z)?;
    for t in yk.iter().chain(zk.iter()).map(|k| k.time_s) {
        let yy = sample(yk, t) as f64;
        let zz = sample(zk, t) as f64;
        if yy * yy + zz * zz >= DOMAIN_RAD * DOMAIN_RAD {
            return Err(bad(
                DOMAIN,
                format!("joints[{joint}].angular_drives_v1.swing_y_swing_z"),
            ));
        }
    }
    Ok(())
}

/// `joints[i].angular_drives_v1`.
///
/// `coordinate_model` is required and the three axis keys are optional, but an
/// axis that IS present must contain only the three mode keys and a mode that is
/// present must be a complete channel. The traversal order (axis, then mode)
/// with exact key spans is what makes the emitted drive array sorted by
/// (joint, axis, mode) with contiguous key spans, which the native side relies
/// on for its binary search and its span check.
fn parse_drives(v: &Value, joint: u32, model:&str, out: &mut PackedAngularExtensionV1) -> Result<()> {
    let path = format!("joints[{joint}].angular_drives_v1");
    let m = object(v, &path, &["coordinate_model"], &AXES)?;
    name(field(m, "coordinate_model", &path)?, model, &path)?;
    for (axis, key) in AXES.iter().enumerate() {
        let Some(axis_value) = m.get(*key) else {
            continue;
        };
        let ap = format!("{path}.{key}");
        let am = object(axis_value, &ap, &[], &MODES)?;
        for (mode, mode_key) in MODES.iter().enumerate() {
            let Some(channel) = am.get(*mode_key) else {
                continue;
            };
            let cp = format!("{ap}.{mode_key}");
            let gm = gain_key(mode);
            let vk = value_key(mode);
            let cm = object(
                channel,
                &cp,
                &["enabled", gm, "max_effort_Nm", "interpolation", "target_keys"],
                &[],
            )?;
            let enabled = boolean(field(cm, "enabled", &cp)?, &cp)?;
            let gain = narrow(field(cm, gm, &cp)?, &cp, true)?;
            let max_effort_nm = narrow(field(cm, "max_effort_Nm", &cp)?, &cp, true)?;
            if field(cm, "interpolation", &cp)?.as_str() != Some("linear_hold") {
                return Err(bad(INTERPOLATION, &cp));
            }
            let key_values = field(cm, "target_keys", &cp)?
                .as_array()
                .ok_or_else(|| bad(INVALID, &cp))?;
            if key_values.is_empty() {
                return Err(bad(INVALID, &cp));
            }
            if key_values.len() > MAX_KEYS_PER_DRIVE {
                return Err(bad(BUDGET, &cp));
            }
            let key_offset = u32::try_from(out.keys.len()).map_err(|_| bad(BUDGET, &cp))?;
            let key_count = u32::try_from(key_values.len()).map_err(|_| bad(BUDGET, &cp))?;
            let mut prev = -1.0_f64;
            for (i, kv) in key_values.iter().enumerate() {
                let kp = format!("{cp}.target_keys[{i}]");
                let km = object(kv, &kp, &["time_s", vk], &[])?;
                let time_s = real(field(km, "time_s", &kp)?, &kp, true)?;
                // The schedule is anchored at exactly zero and strictly
                // increasing: a duplicate or out-of-order key would make the
                // `linear_hold` segment a division by zero or a backwards
                // interpolation, neither of which is a physical target.
                if (i == 0 && time_s != 0.0) || (i > 0 && time_s <= prev) {
                    return Err(bad(ORDER, &kp));
                }
                let target_value = narrow(field(km, vk, &kp)?, &kp, false)?;
                // Only a position target is a CHART coordinate and therefore
                // bounded by the principal domain. A velocity target in rad/s
                // and an effort target in N*m are unbounded by the chart.
                if mode == 0 {
                    if model==crate::material_spin_history::MODEL || (axis==0 && model==CONTINUOUS_MODEL) {
                        if (target_value as f64).abs()>=CONTINUOUS_DOMAIN_RAD {return Err(bad(DOMAIN,&kp));}
                    } else {in_domain(target_value, &kp)?;}
                }
                out.keys.push(AngularDriveKeyGpuV1 {
                    time_s,
                    target_value,
                    reserved: 0,
                });
                prev = time_s;
            }
            out.drives.push(AngularDriveGpuV1 {
                joint_index: joint,
                axis: axis as u32,
                mode: mode as u32 + 1,
                enabled: u32::from(enabled),
                gain,
                max_effort_nm,
                key_offset,
                key_count,
            });
            if out.total_bytes() > MAX_TOTAL_BYTES {
                return Err(bad(BUDGET, &cp));
            }
        }
    }
    if model==crate::material_spin_history::MODEL {Ok(())} else {check_swing_disk(out, joint)}
}

// ---------------------------------------------------------------------------
// The entry point.
// ---------------------------------------------------------------------------

/// Validate a V4 envelope's extension records and pack them by full joint order.
///
/// `joints` is the immutable, full, validated joint order, one entry per joint.
/// `features` is the envelope's `required_features` exactly as it arrived.
///
/// `Ok(None)` means V4 carries NO extension key at all and the V3 transport is
/// unchanged. `Ok(Some(packed))` means at least one joint requested an extension
/// and the packed records are complete for EVERY joint.
///
/// Refusals, in the order they are taken:
///
/// * `JOINT_G2_REQUIRED_FEATURES` — the advertised feature list is not exactly
///   the list these records imply.
/// * `JOINT_G2_LEGACY_CONFLICT` — a joint carries both a directional record and
///   nonzero legacy isotropic angular coefficients. Two angular models on one
///   joint is two physics, and picking either silently is wrong.
/// * `JOINT_ANGULAR_EXTENSION_REQUIRES_CONTROL` — an extension without the
///   directional record whose chart and Jacobian define its coordinates.
/// * then the per-field codes above, with the offending path.
pub fn validate_v4_and_pack(
    joints: &[JointExtensionInput<'_>],
    features: &[String],
) -> Result<Option<PackedAngularExtensionV1>> {
    let expected = required_features(joints);
    if features.iter().map(String::as_str).collect::<Vec<_>>() != expected {
        return Err(bad(FEATURES, "required_features"));
    }
    let has_extension = joints.iter().any(|j| j.soft.is_some() || j.drives.is_some());
    for (i, j) in joints.iter().enumerate() {
        if j.control_present && j.legacy_angular_nonzero {
            return Err(bad(CONFLICT, format!("joints[{i}].angular_control_v1")));
        }
        if (j.soft.is_some() || j.drives.is_some()) && !j.control_present {
            return Err(bad(REQUIRES_CONTROL, format!("joints[{i}]")));
        }
        if (j.soft.is_some() || j.drives.is_some()) && !matches!(j.control_model,Some(COORDINATE_MODEL|CONTINUOUS_MODEL|crate::material_spin_history::MODEL)) {
            return Err(bad(INVALID,format!("joints[{i}].angular_control_v1.coordinate_model")));
        }
    }
    if !has_extension {
        return Ok(None);
    }
    // One soft row per joint at 56 bytes is the floor of this family, so a joint
    // count that cannot fit the byte budget is refused before anything is built.
    if joints.len() > MAX_TOTAL_BYTES / SOFT_RECORD_BYTES {
        return Err(bad(BUDGET, "joints"));
    }
    let count = u32::try_from(joints.len()).map_err(|_| bad(BUDGET, "joints"))?;
    let mut out = PackedAngularExtensionV1 {
        limits: Vec::with_capacity(joints.len()),
        drives: Vec::new(),
        keys: Vec::new(),
    };
    for (i, j) in joints.iter().enumerate() {
        let index = i as u32;
        let extension_model=j.control_model.unwrap_or(COORDINATE_MODEL);
        let row = match j.soft {
            Some(v) => parse_soft(v, index, extension_model)?,
            // An absent soft key is a present, all-zero, all-disabled row: the
            // family is one row per joint so that row `i` is joint `i`.
            None => AngularSoftLimitGpuV1 {
                joint_index: index,
                enabled_axis_mask: 0,
                lower_rad: [0.0; 3],
                upper_rad: [0.0; 3],
                stiffness_nm_per_rad: [0.0; 3],
                swing_cone_enabled: 0,
                swing_cone_half_angle_rad: 0.0,
                swing_cone_stiffness_nm_per_rad: 0.0,
            },
        };
        out.limits.push(row);
        if out.total_bytes() > MAX_TOTAL_BYTES {
            return Err(bad(BUDGET, "limits"));
        }
        if let Some(v) = j.drives {
            parse_drives(v, index, extension_model, &mut out)?;
        }
    }
    if out.limits.len() != count as usize || out.total_bytes() > MAX_TOTAL_BYTES {
        return Err(bad(BUDGET, "extension"));
    }
    out.checked_upload_counts()?;
    out.device_keys()?;
    Ok(Some(out))
}

/// Must run against the ALREADY LOADED backend, before V4 initialize/advance.
///
/// The strides are the record ABIs this module packs, and the control stride is
/// the unchanged 88-byte directional row: a backend that advertises a different
/// stride would read a different field than the host wrote, so a mismatch is a
/// refusal rather than something to work around.
pub fn validate_capabilities(c: &PpfAngularExtensionCapabilitiesV1) -> Result<()> {
    if c.abi_version != 1
        || c.struct_size != 48
        || c.feature_bits & REQUIRED_CAPABILITY_BITS != REQUIRED_CAPABILITY_BITS
        || c.control_stride != CONTROL_RECORD_BYTES as u32
        || c.limit_stride != SOFT_RECORD_BYTES as u32
        || c.drive_stride != DRIVE_RECORD_BYTES as u32
        || c.key_stride != KEY_RECORD_BYTES as u32
        || c.telemetry_stride != 240
        || c.max_keys_per_drive != MAX_KEYS_PER_DRIVE as u32
        || c.max_total_bytes != MAX_TOTAL_BYTES as u64
    {
        return Err(bad("JOINT_ANGULAR_EXTENSION_BAD_ABI", "capabilities"));
    }
    Ok(())
}

#[cfg(test)]
mod unit {
    use super::*;
    use serde_json::json;

    fn soft_value() -> Value {
        json!({
            "coordinate_model": "parent_local_swing_twist_x_v1",
            "twist_x": {"enabled": false, "lower_rad": -0.05, "upper_rad": 0.05,
                        "stiffness_Nm_per_rad": 20.0},
            "swing_y": {"enabled": false, "lower_rad": -0.05, "upper_rad": 0.05,
                        "stiffness_Nm_per_rad": 20.0},
            "swing_z": {"enabled": true, "lower_rad": -0.05, "upper_rad": 0.05,
                        "stiffness_Nm_per_rad": 20.0}
        })
    }
    fn drive_value() -> Value {
        json!({
            "coordinate_model": "parent_local_swing_twist_x_v1",
            "swing_z": {"position": {"enabled": true, "gain_Nm_per_rad": 10.0,
                "max_effort_Nm": 0.5, "interpolation": "linear_hold",
                "target_keys": [{"time_s": 0.0, "value_rad": 0.0},
                                {"time_s": 0.1, "value_rad": 0.2},
                                {"time_s": 0.2, "value_rad": 0.2}]}}
        })
    }
    fn one<'a>(soft: Option<&'a Value>, drives: Option<&'a Value>)
        -> Vec<JointExtensionInput<'a>> {
        vec![JointExtensionInput { control_present: true, control_model:Some(COORDINATE_MODEL), soft, drives,
            legacy_angular_nonzero: false }]
    }
    fn features_of(joints: &[JointExtensionInput<'_>]) -> Vec<String> {
        required_features(joints).into_iter().map(str::to_owned).collect()
    }

    fn cone_value(enabled: bool) -> Value {
        let mut value=soft_value();
        value["swing_z"]["enabled"]=json!(false);
        value["swing_cone"]=json!({"enabled":enabled,"half_angle_rad":0.3,"stiffness_Nm_per_rad":12.0});
        value
    }

    #[test]
    fn swing_cone_packs_tail_and_requires_presence_feature() {
        for enabled in [true,false] {
            let mut soft=cone_value(enabled);
            soft["twist_x"]["enabled"]=json!(true);
            let joints=one(Some(&soft),None);
            let features=features_of(&joints);
            assert_eq!(features,[FEATURE_CONTROL,FEATURE_SOFT,FEATURE_SWING_CONE]);
            let packed=validate_v4_and_pack(&joints,&features).unwrap().unwrap();
            let row=packed.limits[0];
            assert_eq!((row.swing_cone_enabled,row.swing_cone_half_angle_rad,row.swing_cone_stiffness_nm_per_rad),(u32::from(enabled),0.3,12.0));
            assert_eq!(row.enabled_axis_mask,1);
            assert_eq!(packed.total_bytes(),56);
            assert_eq!(validate_v4_and_pack(&joints,&features[..2]).unwrap_err().code,FEATURES);
        }
        let soft=soft_value();
        let joints=one(Some(&soft),None);
        let mut features=features_of(&joints);features.push(FEATURE_SWING_CONE.into());
        assert_eq!(validate_v4_and_pack(&joints,&features).unwrap_err().code,FEATURES);
    }

    #[test]
    fn swing_cone_is_strict_even_disabled_and_refuses_interval_conflicts() {
        for enabled in [true,false] {
            for field in ["enabled","half_angle_rad","stiffness_Nm_per_rad"] {
                let mut soft=cone_value(enabled);soft["swing_cone"].as_object_mut().unwrap().remove(field);
                assert!(parse_soft(&soft,0,COORDINATE_MODEL).is_err(),"missing {field}");
                let mut soft=cone_value(enabled);soft["swing_cone"][field]=Value::Null;
                assert!(parse_soft(&soft,0,COORDINATE_MODEL).is_err(),"null {field}");
            }
            for angle in [0.0,-0.1,DOMAIN_RAD,4.0,1e-40,1e-320] {
                let mut soft=cone_value(enabled);soft["swing_cone"]["half_angle_rad"]=json!(angle);
                assert!(parse_soft(&soft,0,COORDINATE_MODEL).is_err(),"angle {angle}");
            }
            for k in [-1.0,1e40,1e-40,1e-320] {
                let mut soft=cone_value(enabled);soft["swing_cone"]["stiffness_Nm_per_rad"]=json!(k);
                assert!(parse_soft(&soft,0,COORDINATE_MODEL).is_err(),"stiffness {k}");
            }
            let mut soft=cone_value(enabled);soft["swing_cone"]["enabled"]=json!(1);
            assert!(parse_soft(&soft,0,COORDINATE_MODEL).is_err());
            let mut soft=cone_value(enabled);soft["swing_cone"]["stiffness_nm_per_rad"]=json!(1.0);
            assert!(parse_soft(&soft,0,COORDINATE_MODEL).is_err());
            let mut soft=cone_value(enabled);soft["swing_cone"]=Value::Null;
            assert!(parse_soft(&soft,0,COORDINATE_MODEL).is_err());
        }
        for axis in ["swing_y","swing_z"] {
            let mut soft=cone_value(true);soft[axis]["enabled"]=json!(true);
            assert_eq!(parse_soft(&soft,0,COORDINATE_MODEL).unwrap_err().code,"JOINT_ANGULAR_SWING_CONE_CONFLICT");
            soft["swing_cone"]["enabled"]=json!(false);
            assert!(parse_soft(&soft,0,COORDINATE_MODEL).is_ok());
        }
        let mut soft=cone_value(true);soft["swing_cone"]["stiffness_Nm_per_rad"]=json!(0.0);
        assert!(parse_soft(&soft,0,COORDINATE_MODEL).is_ok());
    }

    #[test]
    fn canonical_soft_row_packs_one_row_per_joint() {
        let soft = soft_value();
        let joints = one(Some(&soft), None);
        let packed = validate_v4_and_pack(&joints, &features_of(&joints)).unwrap().unwrap();
        assert_eq!(packed.limits.len(), 1);
        assert_eq!(packed.limits[0].joint_index, 0);
        assert_eq!(packed.limits[0].enabled_axis_mask, 0b100);
        assert_eq!(packed.limits[0].lower_rad, [-0.05, -0.05, -0.05]);
        assert_eq!(packed.limits[0].stiffness_nm_per_rad, [20.0, 20.0, 20.0]);
        assert_eq!((packed.limits[0].swing_cone_enabled, packed.limits[0].swing_cone_half_angle_rad,
            packed.limits[0].swing_cone_stiffness_nm_per_rad), (0, 0.0, 0.0));
        assert!(packed.drives.is_empty() && packed.keys.is_empty());
        assert_eq!(packed.total_bytes(), 56);
    }

    #[test]
    fn canonical_drive_channel_packs_sorted_with_contiguous_spans() {
        let drives = drive_value();
        let joints = vec![
            JointExtensionInput { control_present: true, control_model:Some(COORDINATE_MODEL), soft: None, drives: None,
                legacy_angular_nonzero: false },
            JointExtensionInput { control_present: true, control_model:Some(COORDINATE_MODEL), soft: None, drives: Some(&drives),
                legacy_angular_nonzero: false },
        ];
        let packed = validate_v4_and_pack(&joints, &features_of(&joints)).unwrap().unwrap();
        assert_eq!(packed.limits.len(), 2, "one soft row per joint, even with no soft key");
        assert_eq!(packed.drives.len(), 1);
        assert_eq!(packed.drives[0].joint_index, 1);
        assert_eq!((packed.drives[0].axis, packed.drives[0].mode, packed.drives[0].enabled),
                   (2, 1, 1));
        assert_eq!(packed.drives[0].gain, 10.0);
        assert_eq!(packed.drives[0].max_effort_nm, 0.5);
        assert_eq!((packed.drives[0].key_offset, packed.drives[0].key_count), (0, 3));
        let span = packed.key_span(&packed.drives[0]).unwrap();
        assert_eq!(span.iter().map(|k| k.time_s).collect::<Vec<_>>(), [0.0, 0.1, 0.2]);
        assert_eq!(span.iter().map(|k| k.target_value).collect::<Vec<_>>(), [0.0, 0.2, 0.2]);
        assert!(span.iter().all(|k| k.reserved == 0));
    }

    #[test]
    fn no_extension_key_keeps_the_v3_transport() {
        let joints = vec![JointExtensionInput::EMPTY];
        assert!(validate_v4_and_pack(&joints, &[]).unwrap().is_none());
    }

    #[test]
    fn feature_list_must_be_exactly_ordered() {
        let soft = soft_value();
        let drives = drive_value();
        let joints = one(Some(&soft), Some(&drives));
        // The canonical order is controls, drives, soft limits.
        assert_eq!(features_of(&joints),
            ["pdrd_angular_controls_v1", "pdrd_angular_drives_v1",
             "pdrd_angular_soft_limits_v1"]);
        for wrong in [
            vec!["pdrd_angular_controls_v1".to_string()],
            vec!["pdrd_angular_controls_v1".to_string(),
                 "pdrd_angular_soft_limits_v1".to_string(),
                 "pdrd_angular_drives_v1".to_string()],
            vec!["pdrd_angular_controls_v1".to_string(),
                 "pdrd_angular_drives_v1".to_string(),
                 "pdrd_angular_soft_limits_v1".to_string(),
                 "pdrd_angular_soft_limits_v1".to_string()],
            vec!["pdrd_angular_controls_v1".to_string(),
                 "pdrd_angular_drives_v1".to_string(),
                 "pdrd_angular_soft_limits_v1".to_string(),
                 "pdrd_angular_hard_hinge_v5".to_string()],
            vec![],
        ] {
            let e = validate_v4_and_pack(&joints, &wrong).unwrap_err();
            assert_eq!(e.code, FEATURES, "{wrong:?}");
        }
    }

    #[test]
    fn legacy_angular_conflict_and_missing_control_are_refused() {
        let soft = soft_value();
        let conflicting = vec![JointExtensionInput { control_present: true, control_model:Some(COORDINATE_MODEL), soft: Some(&soft),
            drives: None, legacy_angular_nonzero: true }];
        let e = validate_v4_and_pack(&conflicting, &features_of(&conflicting)).unwrap_err();
        assert_eq!(e.code, CONFLICT);
        let orphan = vec![JointExtensionInput { control_present: false, control_model:None, soft: Some(&soft),
            drives: None, legacy_angular_nonzero: false }];
        let e = validate_v4_and_pack(&orphan, &features_of(&orphan)).unwrap_err();
        assert_eq!(e.code, REQUIRES_CONTROL);
    }

    #[test]
    fn unknown_missing_null_and_extra_fields_are_all_refused() {
        let mut missing = soft_value();
        missing["swing_z"].as_object_mut().unwrap().remove("stiffness_Nm_per_rad");
        let mut extra = soft_value();
        extra["swing_z"]["gain"] = json!(1.0);
        let mut nulled = soft_value();
        nulled["swing_z"]["enabled"] = Value::Null;
        let mut renamed = soft_value();
        renamed["coordinate_model"] = json!("parent_local_swing_twist_x_v2");
        let mut integer_flag = soft_value();
        integer_flag["swing_z"]["enabled"] = json!(1);
        let mut absent_axis = soft_value();
        absent_axis.as_object_mut().unwrap().remove("twist_x");
        for (why, value) in [("missing", missing), ("extra", extra), ("null", nulled),
            ("renamed", renamed), ("integer flag", integer_flag), ("absent axis", absent_axis)] {
            let joints = one(Some(&value), None);
            let e = validate_v4_and_pack(&joints, &features_of(&joints)).unwrap_err();
            assert_eq!(e.code, INVALID, "{why}");
            assert!(e.path.starts_with("joints[0].angular_soft_limits_v1"), "{why}: {e}");
        }
    }

    #[test]
    fn bounds_are_signed_ordered_and_inside_the_chart_domain() {
        let mut inverted = soft_value();
        inverted["swing_z"]["lower_rad"] = json!(0.2);
        inverted["swing_z"]["upper_rad"] = json!(-0.2);
        let joints = one(Some(&inverted), None);
        let e = validate_v4_and_pack(&joints, &features_of(&joints)).unwrap_err();
        assert_eq!(e.code, INTERVAL);
        let domain = DOMAIN_RAD.to_string();
        for bad_bound in [domain.as_str(), "3.14159265", "1e30"] {
            let mut value = soft_value();
            value["twist_x"]["upper_rad"] = serde_json::from_str(bad_bound).unwrap();
            let joints = one(Some(&value), None);
            let e = validate_v4_and_pack(&joints, &features_of(&joints)).unwrap_err();
            assert_eq!(e.code, DOMAIN, "{bad_bound}");
        }
    }

    #[test]
    fn continuous_twist_domains_require_matching_model_and_keep_swing_bounded() {
        let mut soft=soft_value();
        soft["coordinate_model"]=json!(CONTINUOUS_MODEL);
        soft["twist_x"]["lower_rad"]=json!(-4.0*std::f64::consts::PI);
        soft["twist_x"]["upper_rad"]=json!(4.0*std::f64::consts::PI);
        let mut drives=drive_value();
        drives["coordinate_model"]=json!(CONTINUOUS_MODEL);
        drives["twist_x"]=json!({"position":{"enabled":true,"gain_Nm_per_rad":1.0,"max_effort_Nm":1.0,"interpolation":"linear_hold","target_keys":[{"time_s":0.0,"value_rad":12.56637}]}});
        let mut joints=one(Some(&soft),Some(&drives));
        joints[0].control_model=Some(CONTINUOUS_MODEL);
        assert!(validate_v4_and_pack(&joints,&features_of(&joints)).is_ok());
        joints[0].control_model=Some(COORDINATE_MODEL);
        assert!(validate_v4_and_pack(&joints,&features_of(&joints)).is_err());
        drop(joints);
        soft["swing_y"]["upper_rad"]=json!(4.0);
        let mut joints=one(Some(&soft),None);joints[0].control_model=Some(CONTINUOUS_MODEL);
        assert_eq!(validate_v4_and_pack(&joints,&features_of(&joints)).unwrap_err().code,DOMAIN);
        drop(joints);
        soft["swing_y"]["upper_rad"]=json!(0.1);
        soft["twist_x"]["upper_rad"]=json!(512.0);
        let mut joints=one(Some(&soft),None);joints[0].control_model=Some(CONTINUOUS_MODEL);
        assert_eq!(validate_v4_and_pack(&joints,&features_of(&joints)).unwrap_err().code,DOMAIN);
    }

    #[test]
    fn key_order_origin_budget_and_narrowing_are_refused() {
        let key = |t: f64, v: f64| json!({"time_s": t, "value_rad": v});
        let channel = |keys: Value| {
            json!({"coordinate_model": "parent_local_swing_twist_x_v1",
                "swing_z": {"position": {"enabled": true, "gain_Nm_per_rad": 10.0,
                    "max_effort_Nm": 0.5, "interpolation": "linear_hold",
                    "target_keys": keys}}})
        };
        let cases: [(&str, Value, &str); 6] = [
            ("first key not zero", channel(json!([key(0.1, 0.0)])), ORDER),
            ("duplicate time", channel(json!([key(0.0, 0.0), key(0.0, 0.1)])), ORDER),
            ("decreasing time", channel(json!([key(0.0, 0.0), key(0.2, 0.1), key(0.1, 0.2)])), ORDER),
            ("negative time", channel(json!([key(-0.1, 0.0)])), INVALID),
            ("empty keys", channel(json!([])), INVALID),
            ("target outside domain", channel(json!([key(0.0, 3.2)])), DOMAIN),
        ];
        for (why, value, code) in cases {
            let joints = one(None, Some(&value));
            let e = validate_v4_and_pack(&joints, &features_of(&joints)).unwrap_err();
            assert_eq!(e.code, code, "{why}");
        }
        // The key budget is per channel and its refusal is its own code.
        let oversized: Vec<Value> = (0..=MAX_KEYS_PER_DRIVE)
            .map(|i| key(i as f64 * 1e-3, 0.0))
            .collect();
        let value = channel(Value::Array(oversized));
        let joints = one(None, Some(&value));
        let e = validate_v4_and_pack(&joints, &features_of(&joints)).unwrap_err();
        assert_eq!(e.code, BUDGET);
        // A magnitude that flushes to zero in the 16-byte key is a refusal, not a
        // silently inert channel.
        let tiny = channel(json!([key(0.0, 1e-320)]));
        let joints = one(None, Some(&tiny));
        let e = validate_v4_and_pack(&joints, &features_of(&joints)).unwrap_err();
        assert_eq!(e.code, INVALID);
    }

    #[test]
    fn both_swing_position_targets_stay_inside_the_swing_disk() {
        let value = json!({"coordinate_model": "parent_local_swing_twist_x_v1",
            "swing_y": {"position": {"enabled": true, "gain_Nm_per_rad": 10.0,
                "max_effort_Nm": 0.5, "interpolation": "linear_hold",
                "target_keys": [{"time_s": 0.0, "value_rad": 0.0},
                                {"time_s": 0.1, "value_rad": 2.3}]}},
            "swing_z": {"position": {"enabled": true, "gain_Nm_per_rad": 10.0,
                "max_effort_Nm": 0.5, "interpolation": "linear_hold",
                "target_keys": [{"time_s": 0.0, "value_rad": 0.0},
                                {"time_s": 0.1, "value_rad": 2.3}]}}});
        let joints = one(None, Some(&value));
        let e = validate_v4_and_pack(&joints, &features_of(&joints)).unwrap_err();
        assert_eq!(e.code, DOMAIN);
        assert!(e.path.ends_with("swing_y_swing_z"), "{e}");
        // One channel alone at the same radius is inside the domain per key, so
        // the disk check must not fire for it.
        let single = json!({"coordinate_model": "parent_local_swing_twist_x_v1",
            "swing_y": {"position": {"enabled": true, "gain_Nm_per_rad": 10.0,
                "max_effort_Nm": 0.5, "interpolation": "linear_hold",
                "target_keys": [{"time_s": 0.0, "value_rad": 2.3}]}}});
        let joints = one(None, Some(&single));
        assert!(validate_v4_and_pack(&joints, &features_of(&joints)).is_ok());
    }

    #[test]
    fn every_drive_mode_spells_its_own_gain_and_value_key() {
        let value = json!({"coordinate_model": "parent_local_swing_twist_x_v1",
            "twist_x": {
                "position": {"enabled": true, "gain_Nm_per_rad": 1.0, "max_effort_Nm": 1.0,
                    "interpolation": "linear_hold",
                    "target_keys": [{"time_s": 0.0, "value_rad": 0.1}]},
                "velocity": {"enabled": true, "gain_Nms_per_rad": 2.0, "max_effort_Nm": 1.0,
                    "interpolation": "linear_hold",
                    "target_keys": [{"time_s": 0.0, "value_rad_per_s": 0.1}]},
                "torque": {"enabled": true, "gain": 3.0, "max_effort_Nm": 1.0,
                    "interpolation": "linear_hold",
                    "target_keys": [{"time_s": 0.0, "value_Nm": 0.1}]}}});
        let joints = one(None, Some(&value));
        let packed = validate_v4_and_pack(&joints, &features_of(&joints)).unwrap().unwrap();
        assert_eq!(packed.drives.iter().map(|d| (d.axis, d.mode, d.gain, d.key_count))
            .collect::<Vec<_>>(), [(0, 1, 1.0, 1), (0, 2, 2.0, 1), (0, 3, 3.0, 1)]);
        // A mode spelled with the wrong mode's gain key is a refusal: the
        // channels are physically different and a shared key would merge them.
        let wrong = json!({"coordinate_model": "parent_local_swing_twist_x_v1",
            "twist_x": {"velocity": {"enabled": true, "gain_Nm_per_rad": 2.0,
                "max_effort_Nm": 1.0, "interpolation": "linear_hold",
                "target_keys": [{"time_s": 0.0, "value_rad_per_s": 0.1}]}}});
        let joints = one(None, Some(&wrong));
        let e = validate_v4_and_pack(&joints, &features_of(&joints)).unwrap_err();
        assert_eq!(e.code, INVALID);
        // An unsupported interpolation is named as its own refusal.
        let held = json!({"coordinate_model": "parent_local_swing_twist_x_v1",
            "twist_x": {"position": {"enabled": true, "gain_Nm_per_rad": 1.0,
                "max_effort_Nm": 1.0, "interpolation": "cubic",
                "target_keys": [{"time_s": 0.0, "value_rad": 0.1}]}}});
        let joints = one(None, Some(&held));
        let e = validate_v4_and_pack(&joints, &features_of(&joints)).unwrap_err();
        assert_eq!(e.code, INTERPOLATION);
    }

    #[test]
    fn swing_cone_identity_spec_covers_the_actual_packed_layout() {
        let expected=format!("AngularSoftLimitGpuV1:size={};align={};joint_index:u32@{};enabled_axis_mask:u32@{};lower_rad:f32[3]@{};upper_rad:f32[3]@{};stiffness_nm_per_rad:f32[3]@{};swing_cone_enabled:u32@{};swing_cone_half_angle_rad:f32@{};swing_cone_stiffness_nm_per_rad:f32@{};swing-cone-soft-radial-semantic-v1",
            std::mem::size_of::<AngularSoftLimitGpuV1>(),std::mem::align_of::<AngularSoftLimitGpuV1>(),
            std::mem::offset_of!(AngularSoftLimitGpuV1,joint_index),std::mem::offset_of!(AngularSoftLimitGpuV1,enabled_axis_mask),
            std::mem::offset_of!(AngularSoftLimitGpuV1,lower_rad),std::mem::offset_of!(AngularSoftLimitGpuV1,upper_rad),
            std::mem::offset_of!(AngularSoftLimitGpuV1,stiffness_nm_per_rad),std::mem::offset_of!(AngularSoftLimitGpuV1,swing_cone_enabled),
            std::mem::offset_of!(AngularSoftLimitGpuV1,swing_cone_half_angle_rad),std::mem::offset_of!(AngularSoftLimitGpuV1,swing_cone_stiffness_nm_per_rad));
        assert_eq!(ANGULAR_EXTENSION_LAYOUT_SPEC,expected);
    }

    #[test]
    fn capabilities_must_match_the_packed_abi() {
        let good = PpfAngularExtensionCapabilitiesV1 {
            abi_version: 1, struct_size: 48, feature_bits: REQUIRED_CAPABILITY_BITS,
            control_stride: 88, limit_stride: 56, drive_stride: 32, key_stride: 16,
            telemetry_stride: 240, max_keys_per_drive: 1024,
            max_total_bytes: MAX_TOTAL_BYTES as u64,
        };
        assert!(validate_capabilities(&good).is_ok());
        let mut missing_bit = good;
        missing_bit.feature_bits = 0b0111;
        assert!(validate_capabilities(&missing_bit).is_err());
        let mut old_row = good;
        old_row.limit_stride = 44;
        assert!(validate_capabilities(&old_row).is_err());
        let mut moved_row = good;
        moved_row.control_stride = 92;
        assert!(validate_capabilities(&moved_row).is_err());
    }

    #[test]
    fn sample_holds_the_first_and_last_key() {
        let keys = [
            AngularDriveKeyGpuV1 { time_s: 0.0, target_value: 0.0, reserved: 0 },
            AngularDriveKeyGpuV1 { time_s: 0.1, target_value: 0.2, reserved: 0 },
            AngularDriveKeyGpuV1 { time_s: 0.2, target_value: 0.2, reserved: 0 },
        ];
        assert_eq!(sample(&keys, -1.0), 0.0);
        assert_eq!(sample(&keys, 0.05), 0.1);
        assert_eq!(sample(&keys, 0.1), 0.2);
        assert_eq!(sample(&keys, 5.0), 0.2);
        // One key holds everywhere, including at its own time.
        assert_eq!(sample(&keys[..1], 0.0), 0.0);
        assert_eq!(sample(&keys[..1], 3.0), 0.0);
    }
}
