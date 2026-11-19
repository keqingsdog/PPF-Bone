// A1.2 per-axis angular SOFT LIMITS and DRIVES, in the neutral seam.
//
// STAGED FILE. Intended destination:
// `crates/ppf-cts-solver/src/kernels/zz_joint_ext/angular_extension.kernel.cpp`
// Nothing here is installed by the staging step; see
// `coordination/port-20260930/angular-a12-handoff.md` for dispatch integration.
// This file includes the existing directional body; that body needs no edit.
//
// WHAT IT REUSES AND WHAT IT DOES NOT REINVENT. The per-axis chart
// `parent_local_swing_twist_x_v1`, its exact 3x6 Jacobian with respect to the
// world-left increments of both endpoints, the rigid-state view, the reduced
// offset resolution and the typed fault lane are the A1.1 directional control's.
// They are reached by including that neutral source; this file declares NO
// second chart, NO second rotation model and NO second control kernel. That
// include is the whole integration surface on the C++ side, and it is why the
// coordinate a soft bound clamps is by construction the coordinate a directional
// spring acts on rather than a numerically similar one.
//
// `#pragma once` prevents redefinition when a generated translation unit
// reaches the directional body through multiple dependency paths. Production
// dispatch registration remains separate and is performed by the main thread.
//
// PHYSICAL UNITS, NO dt^2, NO FRAME INDEX. Coordinates are radians, bounds are
// radians, stiffness is N*m/rad, damping is N*m*s/rad, torque is N*m. The
// damping and velocity channels divide by the ACTUAL substep duration in seconds
// and the drive targets are sampled at the ACTUAL substep time, both of which
// arrive as physical seconds from the driver. The reference coordinate used by
// velocity control is the coordinate at the SUBSTEP START, which is the same
// reference the directional damping uses, so the two channel families cannot
// disagree about where the substep began.
//
// Curvatures are coordinate-space second derivatives: interval limit k, the
// coupled YZ cone Hessian, Huber gain on the quadratic branch and zero on the
// saturated branch; prescribed torque zero. The assembled J^T W J is a
// positive semidefinite Gauss-Newton operator.
// It omits SO(3) chart second derivatives and is not the full rotation Hessian.
// Torque uses the frozen-substep generalized work -effort*(c-c0), so its trial
// energy derivative matches the prescribed generalized effort.
//
// THE SEAM THIS DESIGN USES FOR ITS GATE. A requested chart that is unusable at
// a frozen pose must refuse the step rather than be silently dropped, and the
// driver already has that seam for A1.1: the per-joint fault lane is checked
// after the current pose is fitted and before the preconditioner that consumes
// the joint blocks, the reduced right-hand side or any PCG application. This
// kernel's `check` action runs on that same lane with the same slot convention
// (one row per joint, row `i` is joint `i`) and, unlike A1.1's control check,
// stays ACTIVE for a row whose base stiffness and damping are all zero, because
// an enabled soft limit or drive is a requested term that rides on that row's
// chart. It writes a slot only when this family actually has a requested term
// there, so it lays no value over another family's slot and needs no extra
// download: run it once per frozen pose beside the control check, before the
// preconditioner, and read the lane once.
#pragma once
#include "angular_control.kernel.cpp"

// One row per joint, in validated joint order: row `i` is joint `i`.
// 56 bytes: mask 0x1/0x2/0x4 selects twist_x/swing_y/swing_z.
struct AngularSoftLimitGpu {
unsigned joint_index;
unsigned enabled_axis_mask;
float lower_rad[3];
float upper_rad[3];
float stiffness_Nm_per_rad[3];
unsigned swing_cone_enabled;
float swing_cone_half_angle_rad;
float swing_cone_stiffness_nm_per_rad;
};
// POD is uploaded verbatim. Refuse layout drift in every compiled backend.
static_assert(sizeof(AngularSoftLimitGpu)==56&&alignof(AngularSoftLimitGpu)==4,"AngularSoftLimitGpu layout");


// One row per (joint, axis, mode) channel, sorted by (joint, axis, mode) with
// key spans that partition the key array. 32 bytes. `mode` is
// DRIVE_POSITION/DRIVE_VELOCITY/DRIVE_TORQUE below.
struct AngularDriveGpu {
unsigned joint_index;
unsigned axis;
unsigned mode;
unsigned enabled;
float gain;
float max_effort_Nm;
unsigned key_offset;
unsigned key_count;
};

// Neutral device key: 16 bytes, alignment 4, target at byte 4.
// This is NOT the canonical wire key (f64 time, alignment 8, target at byte 8).
// Rust must use device_keys() to convert and validate schedules before upload.
struct AngularDriveKeyGpu {
float time_s;
float target_value;
float reserved0;
float reserved1;
};

namespace joint_angular_extension {

[[seam::host_device_fn]] inline bool finite(float x) { return !fmath::isnan(x) && !fmath::isinf(x); }

enum : int { NCOORD = 3, NINCR = 6 };
enum : unsigned {
    DRIVE_POSITION = 1u,
    DRIVE_VELOCITY = 2u,
    DRIVE_TORQUE = 3u,
    // The canonical 56/32/16 record sizes, restated where a reader of this file
    // can see them. A drift here is a device reading a different field than the
    // host wrote, which is why the driver asserts them too.
    SOFT_RECORD_BYTES = 56u,
    DRIVE_RECORD_BYTES = 32u,
    KEY_RECORD_BYTES = 16u,
    TANGENT_STRIDE = 22u, TANGENT_NONE = 0xffffffffu,
};

// ---------------------------------------------------------------------------
// Fault signalling.
//
// The A1.1 lane carries NONE and the geometry codes; because this family shares
// that lane, its own codes are disjoint from every one of them. A slot is
// written ONLY when this family has a requested term at that joint, so a joint
// this family does not touch keeps whatever the control check put there and the
// driver's single readback names the right cause.
// ---------------------------------------------------------------------------
enum : unsigned {
    EXTENSION_FAULT_NONE = 0u,
    // 1..4 are the A1.1 directional codes and are never written here.
    EXTENSION_FAULT_CHART_CURRENT = 128u,
    EXTENSION_FAULT_CHART_REFERENCE = 129u,
    EXTENSION_FAULT_BODY_INDEX = 130u,
    EXTENSION_FAULT_REDUCED_RANGE = 131u,
    // The sampled instant is not a physical instant: a non-positive or
    // non-finite substep duration, or a sample time that would need a frame
    // index to be meaningful. Refused rather than clamped, because a clamped
    // duration silently changes the units of a velocity channel.
    EXTENSION_FAULT_STEP_DOMAIN = 132u,
    EXTENSION_FAULT_RECORD_ARITHMETIC = 133u,
    EXTENSION_FAULT_CURRENT_TWIST_DOMAIN = 134u,
    EXTENSION_FAULT_CURRENT_SWING_DOMAIN = 135u,
    EXTENSION_FAULT_TWIST_HISTORY = 136u,
};

// One axis, one channel: the whole per-axis arithmetic of this family.
//
// Source energy, coordinate gradient and coordinate curvature; all float32.
struct AxisContribution {
    float energy_limit, energy_position, energy_velocity, energy_torque;
    float gradient_limit, gradient_position, gradient_velocity, gradient_torque;
    float weight_limit, weight_position, weight_velocity;
    float lower_excess, upper_excess;
    unsigned position_saturated, velocity_saturated, torque_capped;
};
[[seam::host_device_fn]] inline void clear(AxisContribution& a) {
    a.energy_limit = 0.f; a.energy_position = 0.f;
    a.energy_velocity = 0.f; a.energy_torque = 0.f;
    a.gradient_limit = 0.f; a.gradient_position = 0.f;
    a.gradient_velocity = 0.f; a.gradient_torque = 0.f;
    a.weight_limit = 0.f; a.weight_position = 0.f; a.weight_velocity = 0.f;
    a.lower_excess = 0.f; a.upper_excess = 0.f;
    a.position_saturated = 0u; a.velocity_saturated = 0u; a.torque_capped = 0u;
}
// The energy assembled into the solve, all four sources, one axis.
[[seam::host_device_fn]] inline float total_energy(const AxisContribution& a) {
    return a.energy_limit + a.energy_position + a.energy_velocity + a.energy_torque;
}
// dE/dc for one axis: what the chart's Jacobian multiplies.
[[seam::host_device_fn]] inline float total_gradient(const AxisContribution& a) {
    return a.gradient_limit + a.gradient_position + a.gradient_velocity + a.gradient_torque;
}
// The Gauss-Newton curvature for one axis. A torque contributes none.
[[seam::host_device_fn]] inline float total_weight(const AxisContribution& a) {
    return a.weight_limit + a.weight_position + a.weight_velocity;
}
[[seam::host_device_fn]] inline bool acceptable(const AxisContribution& a) {
    return finite(total_energy(a)) && finite(total_gradient(a)) &&
        finite(total_weight(a)) && total_weight(a) >= 0.f &&
        finite(a.lower_excess) && finite(a.upper_excess);
}

// ---------------------------------------------------------------------------
// linear_hold sampling over physical seconds.
//
// Past the last key the last value HOLDS, before the first the first holds, and
// between two keys the value is linear. The packer guarantees at least one key
// and strictly increasing first-key-zero times, so the segment lookup cannot
// divide by zero; the guards below are about a corrupt record rather than a
// legitimate one, and they refuse rather than interpolate nonsense.
// ---------------------------------------------------------------------------
[[seam::host_device_fn]] inline bool sample_key(const AngularDriveGpu& d, const AngularDriveKeyGpu* keys, float t, float& target) {
    target = 0.f;
    if (!keys || d.key_count == 0u) return false;
    const AngularDriveKeyGpu* k = keys + d.key_offset;
    const AngularDriveKeyGpu& first = k[0];
    if (!finite(first.time_s) || !finite(first.target_value)) return false;
    if (t <= first.time_s) { target = first.target_value; return finite(target); }
    const unsigned last = d.key_count - 1u;
    const AngularDriveKeyGpu& final_key = k[last];
    if (!finite(final_key.time_s) || !finite(final_key.target_value)) return false;
    if (t >= final_key.time_s) { target = final_key.target_value; return finite(target); }
    unsigned lo = 0u, hi = last;
    while (hi - lo > 1u) {
        const unsigned mid = lo + (hi - lo) / 2u;
        if (t < k[mid].time_s) hi = mid; else lo = mid;
    }
    const AngularDriveKeyGpu& a = k[lo];
    const AngularDriveKeyGpu& b = k[hi];
    const float span = b.time_s - a.time_s;
    if (!(span > 0.f) || !finite(span)) return false;
    const float w = (t - a.time_s) / span;
    if (!finite(w)) return false;
    target = (1.f - w) * a.target_value + w * b.target_value;
    return finite(target);
}

// ---------------------------------------------------------------------------
// One source term at a time. A disabled/zero-effort channel is exactly inert.
// Zero base K/C does not disable an enabled limit/drive on the same chart row.
// ---------------------------------------------------------------------------

// Signed soft bound: a one-sided quadratic penalty outside [lower, upper] with
// the exact coordinate gradient and coordinate curvature, and NOTHING inside. A soft penalty
// is not a clamp: the coordinate may pass the bound, and the reported excess is
// how far it did.
[[seam::host_device_fn]] inline bool add_limit(const AngularSoftLimitGpu& l, int axis, float c, AxisContribution& out) {
    if ((l.enabled_axis_mask & (1u << axis)) == 0u) return true;
    const float k = l.stiffness_Nm_per_rad[axis];
    if (!finite(k) || k < 0.f) return false;
    if (k == 0.f) return true;
    const float lower = l.lower_rad[axis], upper = l.upper_rad[axis];
    if (!finite(lower) || !finite(upper) || !finite(c)) return false;
    float delta;
    if (c < lower) { delta = c - lower; out.lower_excess = -delta; }
    else if (c > upper) { delta = c - upper; out.upper_excess = delta; }
    else return true;
    out.energy_limit = .5f * k * delta * delta;
    out.gradient_limit = k * delta;
    out.weight_limit = k;
    return finite(out.energy_limit) && finite(out.gradient_limit);
}

// Circular swing penalty in the existing parent-local swing chart. The exact
// coordinate Hessian is PSD; J^T H J omits chart second derivatives (GN).
// Per-axis energy is split evenly only for accounting; gradient and diagonal
// entries retain their actual coordinate meanings. yz is the symmetric H_yz.
[[seam::host_device_fn]] inline bool add_swing_cone(const AngularSoftLimitGpu& l,
        const float c[NCOORD], AxisContribution out[NCOORD], float& yz) {
    yz=0.f;
    if(l.swing_cone_enabled==0u)return true;
    constexpr float DOMAIN_RAD=3.14159265358979323846f-1e-3f;
    const float alpha=l.swing_cone_half_angle_rad,k=l.swing_cone_stiffness_nm_per_rad;
    if(l.swing_cone_enabled!=1u||(l.enabled_axis_mask&6u)!=0u||
       !finite(alpha)||!(alpha>0.f)||!(alpha<DOMAIN_RAD)||!finite(k)||k<0.f||
       !finite(c[1])||!finite(c[2]))return false;
    // Scale before squaring: valid normal chart values may have subnormal
    // squares, even when their force and curvature are fully representable.
    const float ay=c[1]<0.f?-c[1]:c[1],az=c[2]<0.f?-c[2]:c[2];
    const float scale=ay>az?ay:az;
    if(scale==0.f)return true;
    const float uy=c[1]/scale,uz=c[2]/scale;
    const float length=fmath::sqrt(uy*uy+uz*uz);
    const float r=scale*length;
    if(!finite(r))return false;
    if(r<=alpha||k==0.f)return true;
    const float excess=r-alpha,ny=uy/length,nz=uz/length;
    const float tangent=excess/r,radial=alpha/r;
    const float effort=k*excess;
    // Halve the larger factor before the final product, avoiding an overflow
    // of twice the energy when the energy itself remains finite.
    const float energy=finite(effort)?
        (excess>effort?effort*(excess*.5f):(effort*.5f)*excess):((.5f*k)*excess)*excess;
    // Preserve total accounting under FTZ when half a normal energy is
    // subnormal. This split has no effect on coordinate force or curvature.
    out[1].energy_limit=.5f*energy;out[2].energy_limit=energy-out[1].energy_limit;
    out[1].gradient_limit=finite(effort)?effort*ny:(k*ny)*excess;
    out[2].gradient_limit=finite(effort)?effort*nz:(k*nz)*excess;
    out[1].weight_limit=k*(tangent+radial*ny*ny);
    out[2].weight_limit=k*(tangent+radial*nz*nz);
    yz=k*radial*ny*nz;
    return finite(energy)&&finite(yz)&&finite(out[1].gradient_limit)&&
        finite(out[2].gradient_limit)&&finite(out[1].weight_limit)&&finite(out[2].weight_limit);
}

// Capped Huber effort. Coordinate curvature is gain on the quadratic branch
// (including the boundary) and zero after saturation. The operator uses this
// curvature with the exact chart Jacobian as a Gauss-Newton approximation.
[[seam::host_device_fn]] inline bool huber(float r, float a, float m, float& energy, float& gradient,
        float& weight, unsigned& saturated) {
    energy = 0.f; gradient = 0.f; weight = 0.f; saturated = 0u;
    if (!finite(r) || !finite(a) || !finite(m) || a < 0.f || m < 0.f) return false;
    if (a == 0.f || m == 0.f) return true;
    const float ar = a * r;
    if (fabsf(ar) <= m) {
        energy = .5f * a * r * r;
        gradient = ar;
        weight = a;
    } else {
        const float magnitude = fabsf(r);
        energy = m * magnitude - m * m / (2.f * a);
        gradient = (ar < 0.f ? -m : m);
        weight = 0.f;
        saturated = 1u;
    }
    return finite(energy) && finite(gradient) && finite(weight);
}

// Frozen-substep generalized work for a capped prescribed effort. Its
// coordinate gradient is -effort and its coordinate curvature is zero.
[[seam::host_device_fn]] inline bool add_torque(const AngularDriveGpu& d, float target, float c, float c0, AxisContribution& out) {
    if (!finite(target) || !finite(d.gain) || !finite(d.max_effort_Nm) ||
            d.gain < 0.f || d.max_effort_Nm < 0.f) return false;
    float effort = 0.f;
    out.torque_capped = 0u;
    if (target != 0.f && d.gain > 0.f && d.max_effort_Nm > 0.f) {
        // Compare against the saturation threshold BEFORE multiplying. Valid
        // finite gain*target may exceed fp32 while its capped effort is finite.
        const float magnitude = fabsf(target);
        const float threshold = d.max_effort_Nm / d.gain;
        if (magnitude >= threshold) {
            effort = target < 0.f ? -d.max_effort_Nm : d.max_effort_Nm;
            out.torque_capped = magnitude > threshold ? 1u : 0u;
        } else {
            effort = d.gain * target;
            // Division/multiply rounding at the threshold must never let the
            // computed effort escape its physical cap by an ulp.
            if (effort > d.max_effort_Nm) { effort=d.max_effort_Nm; out.torque_capped=1u; }
            else if (effort < -d.max_effort_Nm) { effort=-d.max_effort_Nm; out.torque_capped=1u; }
        }
    }
    out.energy_torque = -effort * (c - c0);
    out.gradient_torque = -effort;
    return finite(out.energy_torque) && finite(out.gradient_torque);
}

// `sample_time_s` is the ACTUAL substep end in seconds and `dt_s` the ACTUAL
// substep duration, so the velocity channel's `(c - c0) / dt` is a real angular
// rate and the reference `c0` is the substep-start coordinate.
[[seam::host_device_fn]] inline bool add_drive(const AngularDriveGpu& d, const AngularDriveKeyGpu* keys,
        float sample_time_s, float c, float c0, float dt_s, AxisContribution& out) {
    if (d.enabled != 1u) return true;
    // An enabled channel with neither a gain nor a cap applies no effort at all:
    // exactly inert, and never a chart requirement.
    if (d.gain == 0.f || d.max_effort_Nm == 0.f) return true;
    float target;
    if (!sample_key(d, keys, sample_time_s, target)) return false;
    if (d.mode == DRIVE_POSITION) {
        return huber(c - target, d.gain, d.max_effort_Nm, out.energy_position,
            out.gradient_position, out.weight_position,
            out.position_saturated);
    }
    if (d.mode == DRIVE_VELOCITY) {
        return huber((c - c0) - dt_s * target, d.gain / dt_s, d.max_effort_Nm,
            out.energy_velocity, out.gradient_velocity, out.weight_velocity, out.velocity_saturated);
    }
    if (d.mode == DRIVE_TORQUE) return add_torque(d, target, c, c0, out);
    // An unknown mode is a refused record, never a skipped term.
    return false;
}

// ---------------------------------------------------------------------------
// The row predicate.
//
// `joint_angular_control::row_requested` is the directional row's own request
// (the joint is enabled and the chart model is the canonical one) and is used
// here for exactly that. It is deliberately NOT `contributes`: a row carrying
// only soft limits or drives has all-zero per-axis spring coefficients, and
// `contributes` is a statement about those coefficients. The extension's own
// `active` below is what decides whether THIS family has a term on the row, and
// it reads the limit mask and the drive channels rather than the base.
// ---------------------------------------------------------------------------
[[seam::host_device_fn]] inline bool row_requested(const AngularControlGpu& base) {
    return joint_angular_control::row_requested(base);
}

// True when this family must evaluate the row: at least one enabled limit with a
// positive stiffness exactly on an axis whose mask bit is set, or one enabled
// drive channel that can produce effort. `active` is what forces chart
// validation, so it is written per axis and per channel rather than summed into
// a single "does this joint have any extension" flag.
[[seam::host_device_fn]] inline bool drive_span(const unsigned* offsets,unsigned drive_count,unsigned joint_index,
        unsigned& begin,unsigned& end) {
    begin=0u;end=drive_count;
    if(!offsets)return true;
    begin=offsets[joint_index];end=offsets[joint_index+1u];
    return begin<=end&&end<=drive_count;
}
[[seam::host_device_fn]] inline bool active(const AngularControlGpu& base, const AngularSoftLimitGpu& limit,
        const AngularDriveGpu* drives, unsigned drive_count, unsigned joint_index,const unsigned* drive_offsets=nullptr) {
    if (!row_requested(base)) return false;
    // Invalid enabled records must reach evaluate and fail its arithmetic gate.
    if(limit.swing_cone_enabled!=0u)return true;
    for (int axis = 0; axis < NCOORD; ++axis) {
        if ((limit.enabled_axis_mask & (1u << axis)) != 0u &&
            limit.stiffness_Nm_per_rad[axis] > 0.f) return true;
    }
    unsigned begin=0u,end=0u;
    if(!drive_span(drive_offsets,drive_count,joint_index,begin,end))return true;
    for (unsigned i = begin; i < end; ++i) {
        const AngularDriveGpu& d = drives[i];
        if (d.joint_index != joint_index) continue;
        if (d.enabled == 1u && (d.gain > 0.f && d.max_effort_Nm > 0.f)) return true;
    }
    return false;
}

[[seam::host_device_fn]] inline bool resolve_row(const AngularControlGpu& base, const PdrdRigidState* s,
        const PdrdRigidState* s0, unsigned bodies, float c[NCOORD], float J[NCOORD * NINCR],
        float c0[NCOORD], float J0[NCOORD * NINCR], unsigned& code, const AngularTwistEvalContext* twist=nullptr, [[seam::thread]] const MaterialSpinContext* material=nullptr) {
    // Share the chart resolution and branch with directional control.
    const unsigned common = material && material->model!=0u
        ? joint_angular_control::resolve_material(base,s,bodies,*material,c,J,c0)
        : joint_angular_control::resolve(base,s,s0,bodies,c,J,c0,J0,twist);
    switch (common) {
    case joint_angular_control::CONTROL_FAULT_NONE: code=EXTENSION_FAULT_NONE; return true;
    case joint_angular_control::CONTROL_FAULT_CHART_REFERENCE: code=EXTENSION_FAULT_CHART_REFERENCE; break;
    case joint_angular_control::CONTROL_FAULT_BODY_INDEX: code=EXTENSION_FAULT_BODY_INDEX; break;
    case joint_angular_control::CONTROL_FAULT_CURRENT_TWIST_DOMAIN: code=EXTENSION_FAULT_CURRENT_TWIST_DOMAIN; break;
    case joint_angular_control::CONTROL_FAULT_CURRENT_SWING_DOMAIN: code=EXTENSION_FAULT_CURRENT_SWING_DOMAIN; break;
    case joint_angular_control::CONTROL_FAULT_TWIST_HISTORY: code=EXTENSION_FAULT_TWIST_HISTORY; break;
    case joint_angular_control::CONTROL_FAULT_MATERIAL_LOCAL_STEP: code=joint_angular_control::CONTROL_FAULT_MATERIAL_LOCAL_STEP; break;
    case joint_angular_control::CONTROL_FAULT_MATERIAL_INVALID: code=joint_angular_control::CONTROL_FAULT_MATERIAL_INVALID; break;
    default: code=EXTENSION_FAULT_CHART_CURRENT; break;
    }
    return false;
}
// The reduced slot pair, resolved through A1.1's own offset rule so the two
// families cannot disagree about which reduced row a body owns.
[[seam::host_device_fn]] inline bool resolve_slots(const AngularControlGpu& base, unsigned base_offset,
        unsigned dim, unsigned& a, unsigned& b, unsigned& code) {
    if (base.body_a != joint_angular_control::CONTROL_WORLD_ENDPOINT &&
        !joint_angular_control::reduced_offset(base_offset, base.body_a, dim, &a)) {
        code = EXTENSION_FAULT_REDUCED_RANGE;
        return false;
    }
    if (base.body_b != joint_angular_control::CONTROL_WORLD_ENDPOINT &&
        !joint_angular_control::reduced_offset(base_offset, base.body_b, dim, &b)) {
        code = EXTENSION_FAULT_REDUCED_RANGE;
        return false;
    }
    return true;
}

// Evaluate one joint's whole extension surface at the frozen pose.
//
// The base row's own stiffness and damping are NOT added here: A1.1 already
// assembles them, and adding them a second time is the one way this family could
// silently double a joint's force. What this evaluates is limits and drives.
//
// `contributions` is the per-axis coordinate gradient, which is what the three
// operator actions need. `per_axis_out` is optional and carries the whole
// per-axis breakdown for the energy action and for a receipt; passing a null
// pointer costs nothing and keeps the gradient path from computing an energy it
// will not use.
[[seam::host_device_fn]] inline bool evaluate(const AngularControlGpu& base, const AngularSoftLimitGpu& limit,
        const AngularDriveGpu* drives, unsigned drive_count, const AngularDriveKeyGpu* keys,
        unsigned joint_index, float sample_time_s, float dt_s, const PdrdRigidState* s,
        const PdrdRigidState* s0, unsigned bodies, float contributions[NCOORD],
        float c[NCOORD], float J[NCOORD * NINCR], AxisContribution* per_axis_out, unsigned& code, float* swing_yz_out=nullptr, const AngularTwistEvalContext* twist=nullptr, [[seam::thread]] const MaterialSpinContext* material=nullptr,const unsigned* drive_offsets=nullptr) {
    if(swing_yz_out)*swing_yz_out=0.f;
    for (int axis = 0; axis < NCOORD; ++axis) contributions[axis] = 0.f;
    if (per_axis_out) {
        for (int axis = 0; axis < NCOORD; ++axis) clear(per_axis_out[axis]);
    }
    if (!row_requested(base)) return true;
    unsigned drive_begin=0u,drive_end=0u;
    if(!drive_span(drive_offsets,drive_count,joint_index,drive_begin,drive_end)){code=EXTENSION_FAULT_RECORD_ARITHMETIC;return false;}
    if(!active(base,limit,drives,drive_count,joint_index,drive_offsets))return true;
    // The step domain is checked BEFORE the chart: a velocity channel's units
    // depend on it, and a clamped duration would be a silently different unit.
    if (!(dt_s > 0.f) || !finite(dt_s) || (!finite(sample_time_s) || sample_time_s <= 0.f)) {
        code = EXTENSION_FAULT_STEP_DOMAIN;
        return false;
    }
    float c0[NCOORD], J0[NCOORD * NINCR];
    if (!resolve_row(base, s, s0, bodies, c, J, c0, J0, code,twist,material)) return false;
    code = EXTENSION_FAULT_RECORD_ARITHMETIC;
    AxisContribution per_axis[NCOORD];
    for (int axis = 0; axis < NCOORD; ++axis) clear(per_axis[axis]);
    for (int axis = 0; axis < NCOORD; ++axis) {
        if (!add_limit(limit, axis, c[axis], per_axis[axis])) return false;
    }
    float swing_yz=0.f;
    if(!add_swing_cone(limit,c,per_axis,swing_yz))return false;
    if(swing_yz_out)*swing_yz_out=swing_yz;
    for (unsigned i = drive_begin; i < drive_end; ++i) {
        const AngularDriveGpu& d = drives[i];
        if (d.joint_index != joint_index) continue;
        if (d.axis >= (unsigned)NCOORD || d.mode < DRIVE_POSITION || d.mode > DRIVE_TORQUE) return false;
        if (!add_drive(d, keys, sample_time_s, c[d.axis], c0[d.axis], dt_s, per_axis[d.axis])) return false;
    }
    for (int axis = 0; axis < NCOORD; ++axis) {
        if (!acceptable(per_axis[axis])) return false;
        contributions[axis] = total_gradient(per_axis[axis]);
        if (per_axis_out) per_axis_out[axis] = per_axis[axis];
    }
    // The pre-solve gate validates aggregates and the frozen reduced
    // operator, not only individual source records. This is the SAME
    // arithmetic order used by energy, gradient, Hv and self-block assembly.
    float energy_sum = 0.f;
    float h[NCOORD];
    for (int r=0;r<NCOORD;++r) {
        energy_sum += total_energy(per_axis[r]);
        h[r] = total_weight(per_axis[r]);
    }
    if (!finite(energy_sum)) return false;
    for (int k=0;k<NINCR;++k) {
        float gradient = 0.f;
        for (int r=0;r<NCOORD;++r) gradient += J[NINCR*r+k]*contributions[r];
        if (!finite(gradient)) return false;
        for (int column=0;column<NINCR;++column) {
            float entry = 0.f;
            for (int r=0;r<NCOORD;++r) entry += J[NINCR*r+k]*h[r]*J[NINCR*r+column];
            if(swing_yz!=0.f)entry+=swing_yz*(J[NINCR+k]*J[2*NINCR+column]+J[2*NINCR+k]*J[NINCR+column]);
            if (!finite(entry)) return false;
        }
    }
    code = EXTENSION_FAULT_NONE;
    return true;
}

// The diagonal weights and optional symmetric YZ cross weight of the SAME
// frozen pose, for the operator actions. Kept
// separate from `evaluate` so the gradient half never pays for a second chart
// and so a reader can see that the gradient and the operator share one chart.
[[seam::host_device_fn]] inline bool weights(const AngularControlGpu& base, const AngularSoftLimitGpu& limit,
        const AngularDriveGpu* drives, unsigned drive_count, const AngularDriveKeyGpu* keys,
        unsigned joint_index, float sample_time_s, float dt_s, const PdrdRigidState* s,
        const PdrdRigidState* s0, unsigned bodies, float h[NCOORD], float J[NCOORD*NINCR], unsigned& code, float* swing_yz_out=nullptr, const AngularTwistEvalContext* twist=nullptr, [[seam::thread]] const MaterialSpinContext* material=nullptr,const unsigned* drive_offsets=nullptr) {
    float g[NCOORD], c[NCOORD];
    AxisContribution per_axis[NCOORD];
    if (!evaluate(base,limit,drives,drive_count,keys,joint_index,sample_time_s,dt_s,s,s0,bodies,g,c,J,per_axis,code,swing_yz_out,twist,material,drive_offsets)) return false;
    for (int axis=0;axis<NCOORD;++axis) h[axis]=total_weight(per_axis[axis]);
    return true;
}

// The coordinate gradient assembled into the reduced right-hand side.
//
// PPF solves `A u = +dE/dq` and applies the candidate with `exp(-u_rot)`, so what
// is assembled is the POSITIVE coordinate gradient times the chart Jacobian,
// exactly as A1.1's `gradient_kernel` states for the directional spring. Row
// major `J[r*6+k]` is d coordinate r / d increment k, the six increments being
// body A's three world-left components then body B's.
[[seam::host_device_fn]] inline void assemble_gradient(const float contributions[NCOORD], const float J[NCOORD * NINCR],
        float out[NINCR]) {
    for (int k = 0; k < NINCR; ++k) {
        float v = 0.f;
        for (int r = 0; r < NCOORD; ++r) v += J[NINCR * r + k] * contributions[r];
        out[k] = v;
    }
}

// The Gauss-Newton Hessian-vector product. Row major `J[r*6+k]`, so `z = h . (J x)`
// and `y = J^T z`, which is the same operator A1.1's `matvec_kernel` forms for
// the directional spring with the extension's weights substituted.
[[seam::host_device_fn]] inline void assemble_matvec(const float h[NCOORD], const float J[NCOORD * NINCR],
        const float x[NINCR], float y[NINCR], float swing_yz=0.f) {
    float z[NCOORD],dots[NCOORD];
    for (int r = 0; r < NCOORD; ++r) {
        float dot = 0.f;
        for (int k = 0; k < NINCR; ++k) dot += J[NINCR * r + k] * x[k];
        dots[r]=dot;
        z[r] = h[r] * dot;
    }
    if(swing_yz!=0.f){z[1]+=swing_yz*dots[2];z[2]+=swing_yz*dots[1];}
    for (int k = 0; k < NINCR; ++k) {
        float v = 0.f;
        for (int r = 0; r < NCOORD; ++r) v += J[NINCR * r + k] * z[r];
        y[k] = v;
    }
}

// The 3x3 rotational self block, `J^T H J` (H has diagonal h and symmetric YZ cross weight) restricted to one endpoint's
// rotational increments. The translational rows are deliberately absent: both
// endpoint positions are prescribed at this level, exactly as in A1.1, so the
// block goes into the body's own 6x6 slot at rotational rows and columns 3..5.
[[seam::host_device_fn]] inline void assemble_self_block(const float h[NCOORD], const float J[NCOORD * NINCR],
        int endpoint, float block[9], float swing_yz=0.f) {
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            float v = 0.f;
            for (int r = 0; r < NCOORD; ++r) v += J[NINCR * r + 3 * endpoint + row] * h[r] * J[NINCR * r + 3 * endpoint + column];
            if(swing_yz!=0.f)v+=swing_yz*(J[NINCR+3*endpoint+row]*J[2*NINCR+3*endpoint+column]+J[2*NINCR+3*endpoint+row]*J[NINCR+3*endpoint+column]);
            block[3 * row + column] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// The four actions, one thread per joint. Each is a true no-op for a joint this
// family does not touch: no chart read, no fault, no write.
// ---------------------------------------------------------------------------
[[seam::device_fn]] inline void energy_kernel(const AngularControlGpu* base, const AngularTwistEvalContext* contexts, const MaterialSpinContext* material_contexts, const AngularSoftLimitGpu* limits,
        unsigned count, const AngularDriveGpu* drives, unsigned drive_count, const unsigned* drive_offsets, const AngularDriveKeyGpu* keys,
        const PdrdRigidState* s, const PdrdRigidState* s0, unsigned bodies, float sample_time_s, float dt_s,
        compute::atomic_float_t* out, unsigned* fault, unsigned i) {
    if (i >= count) return;
    const AngularControlGpu row = base[i];
    const AngularSoftLimitGpu limit = limits[i];
    const MaterialSpinContext material = material_contexts[i];
    if (!row_requested(row) || !active(row, limit, drives, drive_count, i,drive_offsets)) return;
    float contributions[NCOORD], c[NCOORD], J[NCOORD * NINCR];
    AxisContribution per_axis[NCOORD];
    unsigned code = EXTENSION_FAULT_NONE;
    if (!evaluate(row, limit, drives, drive_count, keys, i, sample_time_s, dt_s, s, s0, bodies,
            contributions, c, J, per_axis, code,nullptr,contexts+i,&material,drive_offsets)) {
        if (fault) fault[i] = code;
        return;
    }
    float e = 0.f;
    for (int axis = 0; axis < NCOORD; ++axis) e += total_energy(per_axis[axis]);
    if (!finite(e)) { if (fault) fault[i] = EXTENSION_FAULT_RECORD_ARITHMETIC; return; }
    compute::atomic_add(out, e);
}

[[seam::device_fn]] inline void gradient_kernel(const AngularControlGpu* base, const AngularTwistEvalContext* contexts, const MaterialSpinContext* material_contexts, const AngularSoftLimitGpu* limits,
        unsigned count, const AngularDriveGpu* drives, unsigned drive_count, const unsigned* drive_offsets, const AngularDriveKeyGpu* keys,
        const PdrdRigidState* s, const PdrdRigidState* s0, unsigned bodies, unsigned base_offset, unsigned dim,
        float sample_time_s, float dt_s, compute::atomic_float_t* out, unsigned* fault, unsigned i) {
    if (i >= count) return;
    const AngularControlGpu row = base[i];
    const AngularSoftLimitGpu limit = limits[i];
    const MaterialSpinContext material = material_contexts[i];
    if (!row_requested(row) || !active(row, limit, drives, drive_count, i,drive_offsets)) return;
    float contributions[NCOORD], c[NCOORD], J[NCOORD * NINCR];
    unsigned code = EXTENSION_FAULT_NONE;
    if (!evaluate(row, limit, drives, drive_count, keys, i, sample_time_s, dt_s, s, s0, bodies,
            contributions, c, J, nullptr, code,nullptr,contexts+i,&material,drive_offsets)) {
        if (fault) fault[i] = code;
        return;
    }
    unsigned a = 0u, b = 0u;
    if (!resolve_slots(row, base_offset, dim, a, b, code)) {
        if (fault) fault[i] = code;
        return;
    }
    float y[NINCR];
    assemble_gradient(contributions, J, y);
    if (row.body_a != joint_angular_control::CONTROL_WORLD_ENDPOINT) {
        for (int k = 0; k < 3; ++k) compute::atomic_add(out + a + 3 + k, y[k]);
    }
    if (row.body_b != joint_angular_control::CONTROL_WORLD_ENDPOINT) {
        for (int k = 0; k < 3; ++k) compute::atomic_add(out + b + 3 + k, y[3 + k]);
    }
}

[[seam::device_fn]] inline void matvec_kernel(const AngularControlGpu* base, const AngularTwistEvalContext* contexts, const MaterialSpinContext* material_contexts, const AngularSoftLimitGpu* limits,
        unsigned count, const AngularDriveGpu* drives, unsigned drive_count, const unsigned* drive_offsets, const AngularDriveKeyGpu* keys,
        const PdrdRigidState* s, const PdrdRigidState* s0, unsigned bodies, unsigned base_offset, unsigned dim,
        float sample_time_s, float dt_s, const float* in, compute::atomic_float_t* out, unsigned* fault,
        unsigned i) {
    if (i >= count) return;
    const AngularControlGpu row = base[i];
    const AngularSoftLimitGpu limit = limits[i];
    const MaterialSpinContext material = material_contexts[i];
    if (!row_requested(row) || !active(row, limit, drives, drive_count, i,drive_offsets)) return;
    float h[NCOORD], c[NCOORD], J[NCOORD * NINCR],swing_yz=0.f;
    unsigned code = EXTENSION_FAULT_NONE;
    // `c` is written by `weights` and not read again here: the operator this
    // action assembles is `J^T H J` (H has diagonal h and symmetric YZ cross weight), and the tilde keeps that explicit
    // rather than leaving an unused local to be inferred as an oversight.
    (void)c;
    if (!weights(row, limit, drives, drive_count, keys, i, sample_time_s, dt_s, s, s0, bodies, h, J, code,&swing_yz,contexts+i,&material,drive_offsets)) {
        if (fault) fault[i] = code;
        return;
    }
    // A row whose every weight is exactly zero is exactly inert: it is not a
    // fault and it writes nothing, which is what keeps a fully saturated or
    // entirely inactive row from being reported as a failure.
    float total = 0.f;
    for (int axis = 0; axis < NCOORD; ++axis) total += h[axis];
    if (total == 0.f) return;
    unsigned a = 0u, b = 0u;
    if (!resolve_slots(row, base_offset, dim, a, b, code)) {
        if (fault) fault[i] = code;
        return;
    }
    float x[NINCR], y[NINCR];
    for (int k = 0; k < NINCR; ++k) x[k] = 0.f;
    if (row.body_a != joint_angular_control::CONTROL_WORLD_ENDPOINT) {
        for (int k = 0; k < 3; ++k) x[k] = in[a + 3 + k];
    }
    if (row.body_b != joint_angular_control::CONTROL_WORLD_ENDPOINT) {
        for (int k = 0; k < 3; ++k) x[3 + k] = in[b + 3 + k];
    }
    assemble_matvec(h, J, x, y,swing_yz);
    if (row.body_a != joint_angular_control::CONTROL_WORLD_ENDPOINT) {
        for (int k = 0; k < 3; ++k) compute::atomic_add(out + a + 3 + k, y[k]);
    }
    if (row.body_b != joint_angular_control::CONTROL_WORLD_ENDPOINT) {
        for (int k = 0; k < 3; ++k) compute::atomic_add(out + b + 3 + k, y[3 + k]);
    }
}

// Freeze the exact extension Gauss-Newton tangent once for one Newton pose.
// Layout is J[3x6], diagonal weights[3], swing YZ cross weight, endpoint offsets.
[[seam::device_fn]] inline void tangent_build_kernel(const AngularControlGpu* base,const AngularTwistEvalContext* contexts,
        const MaterialSpinContext* material_contexts,const AngularSoftLimitGpu* limits,unsigned count,
        const AngularDriveGpu* drives,unsigned drive_count,const unsigned* drive_offsets,const AngularDriveKeyGpu* keys,
        const PdrdRigidState* s,const PdrdRigidState* s0,unsigned bodies,unsigned base_offset,unsigned dim,
        float sample_time_s,float dt_s,float* factors,unsigned* offsets,unsigned* published,unsigned epoch,
        unsigned* fault,unsigned i) {
    if(i>=count)return;
    published[i]=0u;
    unsigned a=TANGENT_NONE,b=TANGENT_NONE,code=EXTENSION_FAULT_NONE;
    float* row=factors+TANGENT_STRIDE*i;
    for(unsigned k=0;k<TANGENT_STRIDE;++k)row[k]=0.f;
    if(epoch==0u){if(fault)fault[i]=EXTENSION_FAULT_RECORD_ARITHMETIC;return;}
    const AngularControlGpu joint=base[i];
    const AngularSoftLimitGpu limit=limits[i];
    const MaterialSpinContext material=material_contexts[i];
    if(!row_requested(joint)||!active(joint,limit,drives,drive_count,i,drive_offsets)) {
        offsets[2u*i]=a;offsets[2u*i+1u]=b;
        if(fault)fault[i]=EXTENSION_FAULT_NONE;
        published[i]=epoch;return;
    }
    float h[NCOORD],c[NCOORD],J[NCOORD*NINCR],swing_yz=0.f;
    if(!weights(joint,limit,drives,drive_count,keys,i,sample_time_s,dt_s,s,s0,bodies,h,J,code,&swing_yz,contexts+i,&material,drive_offsets)) {
        if(fault)fault[i]=code;
        return;
    }
    if(joint.body_a!=joint_angular_control::CONTROL_WORLD_ENDPOINT && !resolve_slots(joint,base_offset,dim,a,b,code)) {
        if(fault)fault[i]=code;
        return;
    }
    if(joint.body_b!=joint_angular_control::CONTROL_WORLD_ENDPOINT && !resolve_slots(joint,base_offset,dim,a,b,code)) {
        if(fault)fault[i]=code;
        return;
    }
    for(unsigned k=0;k<NCOORD*NINCR;++k)row[k]=J[k];
    for(unsigned k=0;k<NCOORD;++k)row[NCOORD*NINCR+k]=h[k];
    row[NCOORD*NINCR+NCOORD]=swing_yz;
    offsets[2u*i]=a;offsets[2u*i+1u]=b;
    if(fault)fault[i]=EXTENSION_FAULT_NONE;
    published[i]=epoch;
}

[[seam::device_fn]] inline void tangent_apply_kernel(const AngularControlGpu* base,const AngularTwistEvalContext* contexts,
        const MaterialSpinContext* material_contexts,const AngularSoftLimitGpu* limits,unsigned count,
        const AngularDriveGpu* drives,unsigned drive_count,const unsigned* drive_offsets,const AngularDriveKeyGpu* keys,
        const PdrdRigidState* s,const PdrdRigidState* s0,unsigned bodies,unsigned base_offset,unsigned dim,
        float sample_time_s,float dt_s,const float* in,compute::atomic_float_t* out,unsigned* fault,
        const float* factors,const unsigned* tangent_offsets,const unsigned* published,unsigned epoch,unsigned i) {
    if(i>=count)return;
    if(epoch==0u||published[i]!=epoch) {
        matvec_kernel(base,contexts,material_contexts,limits,count,drives,drive_count,drive_offsets,keys,
            s,s0,bodies,base_offset,dim,sample_time_s,dt_s,in,out,fault,i);
        return;
    }
    const unsigned a=tangent_offsets[2u*i],b=tangent_offsets[2u*i+1u];
    if(a==TANGENT_NONE&&b==TANGENT_NONE)return;
    const float* row=factors+TANGENT_STRIDE*i;
    float x[NINCR]={},y[NINCR];
    if(a!=TANGENT_NONE)for(unsigned k=0;k<3u;++k)x[k]=in[a+3u+k];
    if(b!=TANGENT_NONE)for(unsigned k=0;k<3u;++k)x[3u+k]=in[b+3u+k];
    // Keep the same small arithmetic order as the uncached action.
    float h[NCOORD]={row[18],row[19],row[20]},J[NCOORD*NINCR];
    for(unsigned k=0;k<NCOORD*NINCR;++k)J[k]=row[k];
    assemble_matvec(h,J,x,y,row[21]);
    if(a!=TANGENT_NONE)for(unsigned k=0;k<3u;++k)compute::atomic_add(out+a+3u+k,y[k]);
    if(b!=TANGENT_NONE)for(unsigned k=0;k<3u;++k)compute::atomic_add(out+b+3u+k,y[3u+k]);
}

[[seam::device_fn]] inline void blocks_kernel(const AngularControlGpu* base, const AngularTwistEvalContext* contexts, const MaterialSpinContext* material_contexts, const AngularSoftLimitGpu* limits,
        unsigned count, const AngularDriveGpu* drives, unsigned drive_count, const unsigned* drive_offsets, const AngularDriveKeyGpu* keys,
        const PdrdRigidState* s, const PdrdRigidState* s0, unsigned bodies, float sample_time_s, float dt_s,
        compute::atomic_float_t* blocks, unsigned* fault, unsigned i) {
    if (i >= count) return;
    const AngularControlGpu row = base[i];
    const AngularSoftLimitGpu limit = limits[i];
    const MaterialSpinContext material = material_contexts[i];
    if (!row_requested(row) || !active(row, limit, drives, drive_count, i,drive_offsets)) return;
    float h[NCOORD], c[NCOORD], J[NCOORD * NINCR],swing_yz=0.f;
    unsigned code = EXTENSION_FAULT_NONE;
    (void)c;   // The self block is `J^T H J` (H has diagonal h and symmetric YZ cross weight); see the matvec action above.
    if (!weights(row, limit, drives, drive_count, keys, i, sample_time_s, dt_s, s, s0, bodies, h, J, code,&swing_yz,contexts+i,&material,drive_offsets)) {
        if (fault) fault[i] = code;
        return;
    }
    float total = 0.f;
    for (int axis = 0; axis < NCOORD; ++axis) total += h[axis];
    if (total == 0.f) return;
    unsigned a = 0u, b = 0u;
    if (row.body_a != joint_angular_control::CONTROL_WORLD_ENDPOINT &&
        !joint_angular_control::block_offset(row.body_a, &a)) code = EXTENSION_FAULT_REDUCED_RANGE;
    else if (row.body_b != joint_angular_control::CONTROL_WORLD_ENDPOINT &&
        !joint_angular_control::block_offset(row.body_b, &b)) code = EXTENSION_FAULT_REDUCED_RANGE;
    if (code != EXTENSION_FAULT_NONE) { if (fault) fault[i] = code; return; }
    float block[9];
    for (int endpoint = 0; endpoint < 2; ++endpoint) {
        const unsigned body = endpoint ? row.body_b : row.body_a;
        if (body == joint_angular_control::CONTROL_WORLD_ENDPOINT) continue;
        const unsigned slot = endpoint ? b : a;
        assemble_self_block(h, J, endpoint, block,swing_yz);
        for (int row_i = 0; row_i < 3; ++row_i) {
            for (int column = 0; column < 3; ++column) {
                compute::atomic_add(blocks + slot + 6 * (3 + row_i) + 3 + column, block[3 * row_i + column]);
            }
        }
    }
}

// The per-pose gate, on the shared fault lane.
//
// A slot is written ONLY when this family has a requested term at that joint, so
// a joint it does not touch keeps the value another family's check left there.
// The joint index is the row index, which is the whole reason the soft family is
// one row per joint.
[[seam::device_fn]] inline void check_kernel(const AngularControlGpu* base, const AngularTwistEvalContext* contexts, const MaterialSpinContext* material_contexts, const AngularSoftLimitGpu* limits,
        unsigned count, const AngularDriveGpu* drives, unsigned drive_count, const unsigned* drive_offsets, const AngularDriveKeyGpu* keys,
        const PdrdRigidState* s, const PdrdRigidState* s0, unsigned bodies, unsigned base_offset,
        unsigned dim, float sample_time_s, float dt_s, unsigned* fault, unsigned i) {
    if (i >= count) return;
    const AngularControlGpu row = base[i];
    const AngularSoftLimitGpu limit = limits[i];
    const MaterialSpinContext material = material_contexts[i];
    if (!row_requested(row) || !active(row, limit, drives, drive_count, i,drive_offsets)) return;
    float g[NCOORD], c[NCOORD], J[NCOORD*NINCR];
    AxisContribution per_axis[NCOORD];
    unsigned code = EXTENSION_FAULT_NONE;
    if (!evaluate(row,limit,drives,drive_count,keys,i,sample_time_s,dt_s,s,s0,bodies,g,c,J,per_axis,code,nullptr,contexts+i,&material,drive_offsets)) {
        if (fault) fault[i] = code;
        return;
    }
    unsigned a=0u,b=0u;
    if (!resolve_slots(row,base_offset,dim,a,b,code)) { if (fault) fault[i] = code; return; }
    if ((row.body_a != joint_angular_control::CONTROL_WORLD_ENDPOINT && !joint_angular_control::block_offset(row.body_a,&a)) ||
        (row.body_b != joint_angular_control::CONTROL_WORLD_ENDPOINT && !joint_angular_control::block_offset(row.body_b,&b))) {
        if (fault) fault[i] = EXTENSION_FAULT_REDUCED_RANGE;
    }
}

} // namespace joint_angular_extension

// ---------------------------------------------------------------------------
// The entry point.
//
// `sample_time_s` is the actual substep end and `dt_s` the actual substep
// duration, both in physical seconds and both already validated finite by the
// driver. There is no frame index anywhere in this signature and none is
// reconstructed inside: a target sampled at a frame boundary is a different
// physical target under a variable step, which is exactly the defect this
// signature exists to make impossible.
//
// Actions: 0 gradient, 1 Hessian-vector, 2 6x6 self blocks, 3 complete frozen-pose gate, 4 scalar energy.
// ---------------------------------------------------------------------------
[[seam::entry(i)]]
[[seam::device_fn]] inline void joint_angular_extension_apply(const AngularControlGpu *records, const AngularTwistEvalContext *twist_context, const MaterialSpinContext *material_spin_context, const AngularSoftLimitGpu *limits,
        const AngularDriveGpu *drives, unsigned drive_count, const unsigned *drive_offsets, const AngularDriveKeyGpu *keys,
        const PdrdRigidState *state, const PdrdRigidState *reference, unsigned bodies, unsigned base,
        unsigned dim, float sample_time_s, float dt_s, unsigned action, const float *input,
        float *tangent_cache,unsigned *tangent_offsets,unsigned *tangent_published,unsigned epoch,
        compute::atomic_float_t *output, unsigned *fault, unsigned i) {
    if(action==0u) joint_angular_extension::gradient_kernel(records,twist_context,material_spin_context,limits,0xffffffffu,drives,drive_count,drive_offsets,keys,state,reference,bodies,base,dim,sample_time_s,dt_s,output,fault,i);
    else if(action==1u) joint_angular_extension::matvec_kernel(records,twist_context,material_spin_context,limits,0xffffffffu,drives,drive_count,drive_offsets,keys,state,reference,bodies,base,dim,sample_time_s,dt_s,input,output,fault,i);
    else if(action==2u) joint_angular_extension::blocks_kernel(records,twist_context,material_spin_context,limits,0xffffffffu,drives,drive_count,drive_offsets,keys,state,reference,bodies,sample_time_s,dt_s,output,fault,i);
    else if(action==4u) joint_angular_extension::energy_kernel(records,twist_context,material_spin_context,limits,0xffffffffu,drives,drive_count,drive_offsets,keys,state,reference,bodies,sample_time_s,dt_s,output,fault,i);
    else if(action==6u) joint_angular_extension::tangent_build_kernel(records,twist_context,material_spin_context,limits,0xffffffffu,drives,drive_count,drive_offsets,keys,state,reference,bodies,base,dim,sample_time_s,dt_s,tangent_cache,tangent_offsets,tangent_published,epoch,fault,i);
    else if(action==7u) joint_angular_extension::tangent_apply_kernel(records,twist_context,material_spin_context,limits,0xffffffffu,drives,drive_count,drive_offsets,keys,state,reference,bodies,base,dim,sample_time_s,dt_s,input,output,fault,tangent_cache,tangent_offsets,tangent_published,epoch,i);
    else joint_angular_extension::check_kernel(records,twist_context,material_spin_context,limits,0xffffffffu,drives,drive_count,drive_offsets,keys,state,reference,bodies,base,dim,sample_time_s,dt_s,fault,i);
}



