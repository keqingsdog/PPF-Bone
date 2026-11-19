// Directional (per-axis) SO(3) angular control, ported from the bounded
// angular_control_v1 math (OLD/crates/ppf-cts-solver/src/cpp/joint/
// angular_control_v1_math.hpp, sha256 0b48e16268e8e0f4...).
//
// The chart decomposes the relative rotation into a twist coordinate and a
// two-component swing vector, and carries its exact 3x6 derivative with
// respect to world-left incremental rotations of body A and body B. Stiffness
// and damping are per axis, in physical units (Nm/rad, Nms/rad): there is no
// dt^2 scaling anywhere in this file. Damping is referenced to the true
// substep-start rigid state, which the driver samples once per step.
#pragma once
#include "../energy/model/pdrd_rigid.kernel.cpp"
#include "restoring_context.hpp"
#include "restoring_observation.hpp"
#include "material_spin_v3.hpp"

struct AngularControlGpu {
unsigned body_a;
unsigned body_b;
unsigned enabled;
unsigned coordinate_model;
float frame_a_wxyz[4];
float frame_b_wxyz[4];
float rest_relative_wxyz[4];
float stiffness_Nm_per_rad[3];
float damping_Nms_per_rad[3];
};

// Dynamic evaluation state; authored 88-byte row remains ABI-stable.
struct AngularTwistEvalContext {
float current_anchor;
float reference_anchor;
unsigned continuous;
unsigned initialized;
};

// Accepted history, indexed by the full angular-control row index.
struct MaterialSpinContext {
float accepted_q_wxyz[4];
float accepted_a[3];
unsigned model;
};

namespace joint_angular_control {
[[seam::host_device_fn]] inline bool finite(float x) { return !fmath::isnan(x) && !fmath::isinf(x); }
enum : unsigned { CONTROL_WORLD_ENDPOINT = 0xffffffffu, COORDINATE_MODEL_SWING_TWIST_X = 1u };
enum : int { NCOORD = 3, NINCR = 6 };
enum : unsigned { CHART_DOMAIN_NONE=0u, CHART_DOMAIN_TWIST=1u, CHART_DOMAIN_SWING=2u };
enum : unsigned { CONTROL_TANGENT_STRIDE = 21u, CONTROL_TANGENT_NONE = 0xffffffffu };

// ---------------------------------------------------------------------------
// Forward-mode dual numbers: value plus the six world-left increments
// [A xyz, B xyz]. Branch-consistent derivatives of the evaluated chart.
// ---------------------------------------------------------------------------
struct Dual {
    float v; float d[NINCR];
    [[seam::host_device_fn]] Dual(): v(0.f) { for(int i=0;i<NINCR;++i)d[i]=0.f; }
    [[seam::host_device_fn]] explicit Dual(float x): v(x) { for(int i=0;i<NINCR;++i)d[i]=0.f; }
};
[[seam::host_device_fn]] inline Dual dadd(Dual a, Dual b) { Dual o(a.v+b.v); for(int i=0;i<NINCR;++i)o.d[i]=a.d[i]+b.d[i]; return o; }
[[seam::host_device_fn]] inline Dual dsub(Dual a, Dual b) { Dual o(a.v-b.v); for(int i=0;i<NINCR;++i)o.d[i]=a.d[i]-b.d[i]; return o; }
[[seam::host_device_fn]] inline Dual dneg(Dual a) { Dual o(-a.v); for(int i=0;i<NINCR;++i)o.d[i]=-a.d[i]; return o; }
[[seam::host_device_fn]] inline Dual dmul(Dual a, Dual b) { Dual o(a.v*b.v); for(int i=0;i<NINCR;++i)o.d[i]=a.d[i]*b.v+a.v*b.d[i]; return o; }
[[seam::host_device_fn]] inline Dual ddiv(Dual a, Dual b) { Dual o(a.v/b.v); const float inv=1.f/(b.v*b.v); for(int i=0;i<NINCR;++i)o.d[i]=(a.d[i]*b.v-a.v*b.d[i])*inv; return o; }
[[seam::host_device_fn]] inline Dual droot(Dual a) { Dual o(fmath::sqrt(a.v)); const float k=.5f/o.v; for(int i=0;i<NINCR;++i)o.d[i]=k*a.d[i]; return o; }
[[seam::host_device_fn]] inline Dual dangle(Dual y, Dual x) { Dual o(fmath::atan2(y.v,x.v)); const float den=x.v*x.v+y.v*y.v; for(int i=0;i<NINCR;++i)o.d[i]=(x.v*y.d[i]-y.v*x.d[i])/den; return o; }
[[seam::host_device_fn]] inline bool dfinite(Dual x) { if(!finite(x.v))return false; for(int i=0;i<NINCR;++i)if(!finite(x.d[i]))return false; return true; }

struct Quat { Dual w,x,y,z; };
[[seam::host_device_fn]] inline Quat qmul(Quat a, Quat b) {
    Quat o;
    o.w=dsub(dsub(dsub(dmul(a.w,b.w),dmul(a.x,b.x)),dmul(a.y,b.y)),dmul(a.z,b.z));
    o.x=dadd(dadd(dsub(dmul(a.w,b.x),dmul(a.z,b.y)),dmul(a.x,b.w)),dmul(a.y,b.z));
    o.y=dadd(dadd(dadd(dmul(a.w,b.y),dmul(a.z,b.x)),dmul(a.y,b.w)),dneg(dmul(a.x,b.z)));
    o.z=dadd(dsub(dadd(dmul(a.w,b.z),dmul(a.x,b.y)),dmul(a.y,b.x)),dmul(a.z,b.w));
    return o;
}
[[seam::host_device_fn]] inline Quat qinv(Quat q) { Quat o; o.w=q.w; o.x=dneg(q.x); o.y=dneg(q.y); o.z=dneg(q.z); return o; }
[[seam::host_device_fn]] inline Quat qnorm(Quat q) {
    Dual n=droot(dadd(dadd(dmul(q.w,q.w),dmul(q.x,q.x)),dadd(dmul(q.y,q.y),dmul(q.z,q.z))));
    Quat o; o.w=ddiv(q.w,n); o.x=ddiv(q.x,n); o.y=ddiv(q.y,n); o.z=ddiv(q.z,n); return o;
}
[[seam::host_device_fn]] inline Quat qraw(const float q[4]) { Quat o; o.w=Dual(q[0]); o.x=Dual(q[1]); o.y=Dual(q[2]); o.z=Dual(q[3]); return o; }

// R is an explicit row-major 3x3 here (see rigid_rowmajor below). The
// antisymmetric differences therefore have the opposite index order to the
// original column-major helper; otherwise the value is the inverse rotation
// while seed_world_left differentiates the forward rotation.
[[seam::host_device_fn]] inline bool quat_from_matrix(const float R[9], Quat& q) {
    for(int i=0;i<9;++i) if(!finite(R[i])) return false;
    float a[4]; const float tr=R[0]+R[4]+R[8];
    if(tr>0.f){ float s=2.f*fmath::sqrt(tr+1.f); a[0]=.25f*s; a[1]=(R[7]-R[5])/s; a[2]=(R[2]-R[6])/s; a[3]=(R[3]-R[1])/s; }
    else if(R[0]>=R[4]&&R[0]>=R[8]){ float s=2.f*fmath::sqrt(fmath::max(1.f+R[0]-R[4]-R[8],0.f)); a[0]=(R[7]-R[5])/s; a[1]=.25f*s; a[2]=(R[3]+R[1])/s; a[3]=(R[6]+R[2])/s; }
    else if(R[4]>=R[8]){ float s=2.f*fmath::sqrt(fmath::max(1.f+R[4]-R[0]-R[8],0.f)); a[0]=(R[2]-R[6])/s; a[1]=(R[3]+R[1])/s; a[2]=.25f*s; a[3]=(R[7]+R[5])/s; }
    else { float s=2.f*fmath::sqrt(fmath::max(1.f+R[8]-R[0]-R[4],0.f)); a[0]=(R[3]-R[1])/s; a[1]=(R[6]+R[2])/s; a[2]=(R[7]+R[5])/s; a[3]=.25f*s; }
    float n=fmath::sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]+a[3]*a[3]);
    if(!(n>0.f)||!finite(n)) return false;
    q=qraw(a); q=qnorm(q);
    return dfinite(q.w)&&dfinite(q.x)&&dfinite(q.y)&&dfinite(q.z);
}
// Seed the analytic derivative of a world-left incremental rotation:
// dq = 0.5 [0,e_k] q at the given slot.
[[seam::host_device_fn]] inline void seed_world_left(Quat& q, int offset) {
    const float w=q.w.v,x=q.x.v,y=q.y.v,z=q.z.v;
    q.w.d[offset]=-x*.5f; q.x.d[offset]=w*.5f;  q.y.d[offset]=-z*.5f; q.z.d[offset]=y*.5f;
    q.w.d[offset+1]=-y*.5f; q.x.d[offset+1]=z*.5f; q.y.d[offset+1]=w*.5f; q.z.d[offset+1]=-x*.5f;
    q.w.d[offset+2]=-z*.5f; q.x.d[offset+2]=-y*.5f; q.y.d[offset+2]=x*.5f; q.z.d[offset+2]=w*.5f;
}
[[seam::host_device_fn]] inline bool canonical_negative(Quat q) {
    return q.w.v<0.f||(q.w.v==0.f&&(q.x.v<0.f||(q.x.v==0.f&&(q.y.v<0.f||(q.y.v==0.f&&q.z.v<0.f)))));
}
// RigidState::rotation is column-major; the chart wants row-major.
[[seam::host_device_fn]] inline void rigid_rowmajor(const PdrdRigidState* s, unsigned body, float o[9]) {
    for(int r=0;r<3;++r) for(int c=0;c<3;++c) o[3*r+c]=s[body].R[3*c+r];
}

// Relative pose Q=(Ra Fa)^-1 (Rb Fb) rest^-1, without evaluating the
// global swing/twist chart. T=(Rb Fb rest^-1) maps right increments of Q
// to world-left body increments. Both endpoints may independently be world.
[[seam::host_device_fn]] inline bool material_relative_pose(const AngularControlGpu& j,
        const PdrdRigidState* state, unsigned bodies, float q[4], float T[9]) {
    if ((j.body_a!=CONTROL_WORLD_ENDPOINT && j.body_a>=bodies) ||
        (j.body_b!=CONTROL_WORLD_ENDPOINT && j.body_b>=bodies)) return false;
    Quat a,b; float R[9]; const float identity[4]={1.f,0.f,0.f,0.f};
    if(j.body_a==CONTROL_WORLD_ENDPOINT) a=qraw(identity);
    else { rigid_rowmajor(state,j.body_a,R); if(!quat_from_matrix(R,a))return false; }
    if(j.body_b==CONTROL_WORLD_ENDPOINT) b=qraw(identity);
    else { rigid_rowmajor(state,j.body_b,R); if(!quat_from_matrix(R,b))return false; }
    for(int k=0;k<4;++k) if(!finite(j.frame_a_wxyz[k]) || !finite(j.frame_b_wxyz[k]) || !finite(j.rest_relative_wxyz[k])) return false;
    Quat left=qmul(a,qraw(j.frame_a_wxyz));
    Quat right=qmul(qmul(b,qraw(j.frame_b_wxyz)),qinv(qraw(j.rest_relative_wxyz)));
    const float ln=left.w.v*left.w.v+left.x.v*left.x.v+left.y.v*left.y.v+left.z.v*left.z.v;
    const float rn=right.w.v*right.w.v+right.x.v*right.x.v+right.y.v*right.y.v+right.z.v*right.z.v;
    if(!(ln>0.f)||!(rn>0.f)||!finite(ln)||!finite(rn))return false;
    left=qnorm(left); right=qnorm(right); Quat rel=qnorm(qmul(qinv(left),right));
    q[0]=rel.w.v; q[1]=rel.x.v; q[2]=rel.y.v; q[3]=rel.z.v;
    const float w=right.w.v,x=right.x.v,y=right.y.v,z=right.z.v;
    T[0]=1.f-2.f*(y*y+z*z); T[1]=2.f*(x*y-z*w); T[2]=2.f*(x*z+y*w);
    T[3]=2.f*(x*y+z*w); T[4]=1.f-2.f*(x*x+z*z); T[5]=2.f*(y*z-x*w);
    T[6]=2.f*(x*z-y*w); T[7]=2.f*(y*z+x*w); T[8]=1.f-2.f*(x*x+y*y);
    for(int k=0;k<4;++k)if(!finite(q[k]))return false;
    for(int k=0;k<9;++k)if(!finite(T[k]))return false;
    return true;
}

// A world-left increment of either body induces a right increment of Q via
// T^T. The two sides have opposite signs; world endpoints have zero columns.
[[seam::host_device_fn]] inline bool material_body_jacobian(const AngularControlGpu& row,
        const float T[9], const float jac_body[9], float J[18]) {
    for(int r=0;r<3;++r) for(int k=0;k<3;++k) {
        float v=0.f;
        for(int c=0;c<3;++c) v+=jac_body[3*r+c]*T[3*k+c];
        if(!finite(v))return false;
        J[6*r+k]=row.body_a==CONTROL_WORLD_ENDPOINT?0.f:-v;
        J[6*r+3+k]=row.body_b==CONTROL_WORLD_ENDPOINT?0.f:v;
    }
    return true;
}

// Chart: relative rotation -> (twist, swing_y, swing_z) and its 3x6 Jacobian.
[[seam::host_device_fn]] inline bool chart(const AngularControlGpu& j, const float Ra[9], const float Rb[9], float c[NCOORD], float J[NCOORD*NINCR], unsigned* domain_reason=nullptr, bool continuous_twist=false, float* radius=nullptr) {
    if(domain_reason) *domain_reason=CHART_DOMAIN_NONE;
    Quat a,b;
    if(j.body_a==CONTROL_WORLD_ENDPOINT){ a=Quat(); a.w=Dual(1.f); a.x=Dual(); a.y=Dual(); a.z=Dual(); }
    else { if(!quat_from_matrix(Ra,a)) return false; seed_world_left(a,0); }
    if(j.body_b==CONTROL_WORLD_ENDPOINT){ b=Quat(); b.w=Dual(1.f); b.x=Dual(); b.y=Dual(); b.z=Dual(); }
    else { if(!quat_from_matrix(Rb,b)) return false; seed_world_left(b,3); }
    a=qmul(a,qraw(j.frame_a_wxyz)); b=qmul(b,qraw(j.frame_b_wxyz));
    Quat q=qnorm(qmul(qmul(qinv(a),b),qinv(qraw(j.rest_relative_wxyz))));
    if(canonical_negative(q)){ q.w=dneg(q.w); q.x=dneg(q.x); q.y=dneg(q.y); q.z=dneg(q.z); }
    Dual tnorm=droot(dadd(dmul(q.w,q.w),dmul(q.x,q.x)));
    if(!dfinite(tnorm)||tnorm.v<1e-4f) return false;
    if(radius) *radius=tnorm.v;
    Quat qt; qt.w=ddiv(q.w,tnorm); qt.x=ddiv(q.x,tnorm); qt.y=Dual(); qt.z=Dual();
    Quat qs=qnorm(qmul(q,qinv(qt)));
    Dual tw=dmul(Dual(2.f),dangle(qt.x,qt.w));
    Dual swing_s2=dadd(dmul(qs.y,qs.y),dmul(qs.z,qs.z));
    const float swing_len=fmath::sqrt(swing_s2.v);
    Dual scale;
    if(swing_len<1e-4f){
        // Taylor expansion of 2 atan2(s,w)/s including its derivative.
        Dual w=qs.w;
        const Dual w3=dmul(dmul(w,w),w);
        scale=dsub(ddiv(Dual(2.f),w),dmul(Dual(2.f/3.f),ddiv(swing_s2,w3)));
    } else { Dual swing_norm=droot(swing_s2); scale=ddiv(dmul(Dual(2.f),dangle(swing_norm,qs.w)),swing_norm); }
    Dual coords[NCOORD]; coords[0]=tw; coords[1]=dmul(scale,qs.y); coords[2]=dmul(scale,qs.z);
    const float swing=fmath::sqrt(coords[1].v*coords[1].v+coords[2].v*coords[2].v);
    if(!finite(swing)) return false;
    // Retained chart domain from the prior implementation: the twist and swing
    // magnitudes must stay inside the open chart, not merely be finite.
    constexpr float PI_F = 3.14159265358979323846f;
    if((!continuous_twist && fmath::abs(tw.v)>=PI_F-1e-3f)||swing>=PI_F-1e-3f) {
        // Keep the original boolean/cutoffs. Only a fully finite coordinate
        // AND derivative may identify a geometric candidate-domain violation.
        bool all_finite=true;
        for(int r=0;r<NCOORD;++r) if(!dfinite(coords[r])) all_finite=false;
        if(domain_reason && all_finite) *domain_reason=
            (!continuous_twist && fmath::abs(tw.v)>=PI_F-1e-3f)?CHART_DOMAIN_TWIST:CHART_DOMAIN_SWING;
        return false;
    }
    for(int r=0;r<NCOORD;++r){
        if(!dfinite(coords[r])) return false;
        c[r]=coords[r].v;
        for(int k=0;k<NINCR;++k) J[NINCR*r+k]=coords[r].d[k];
    }
    return true;
}

// Lift a freshly evaluated principal coordinate near a certified branch anchor.
// This does not certify the path: the caller must do that before selecting an
// anchor, and must reject a phase ambiguity rather than guessing its turn.
[[seam::host_device_fn]] inline bool lift_twist(float principal, float anchor, float& lifted) {
    constexpr float PI_F = 3.14159265358979323846f;
    constexpr float TAU_F = 6.28318530717958647692f;
    if (!finite(principal) || !finite(anchor) || fmath::abs(anchor) > 512.f ||
        fmath::abs(principal) > PI_F + 1e-5f) return false;
    const float turns = (anchor-principal)/TAU_F;
    const int nearest = int(turns + (turns >= 0.f ? .5f : -.5f));
    lifted = principal + float(nearest)*TAU_F;
    return finite(lifted) && fmath::abs(lifted-anchor) < PI_F-1e-5f;
}
// ---------------------------------------------------------------------------
// Fault signalling.
//
// An ACTIVE requested row whose live geometry is unusable must never silently
// disappear from the assembly: that would drop a requested term while the step
// is still accepted, which is silently wrong physics. Every action writes a
// typed code into the per-joint fault lane (each thread owns its own slot, so a
// plain store is enough and no atomics or extra synchronization are needed),
// and the driver reads the lane once per accepted step and rejects the step
// instead of accepting it. Disabled rows write 0 and stay a true no-op.
// ---------------------------------------------------------------------------
enum : unsigned {
    CONTROL_FAULT_NONE = 0u,
    CONTROL_FAULT_CHART_CURRENT = 1u,
    CONTROL_FAULT_CHART_REFERENCE = 2u,
    CONTROL_FAULT_BODY_INDEX = 3u,
    CONTROL_FAULT_REDUCED_RANGE = 4u,
    CONTROL_FAULT_CURRENT_TWIST_DOMAIN = 5u,
    CONTROL_FAULT_CURRENT_SWING_DOMAIN = 6u,
    CONTROL_FAULT_TWIST_HISTORY = 7u,
    CONTROL_FAULT_MATERIAL_LOCAL_STEP = 8u,
    CONTROL_FAULT_MATERIAL_INVALID = 9u,
};

[[seam::host_device_fn]] inline bool row_requested(const AngularControlGpu& j) {
    // "Requested" is the wire request, not whether the coefficients happen to be
    // nonzero: a row may legitimately carry only soft limits/drives in a later
    // group while its per-axis spring coefficients are all zero, and such a row
    // still needs the chart and Jacobian. Coefficients of zero simply contribute
    // exactly zero.
    return j.enabled==1u && j.coordinate_model==COORDINATE_MODEL_SWING_TWIST_X;
}
[[seam::host_device_fn]] inline bool contributes(const AngularControlGpu& j) {
    for(int k=0;k<NCOORD;++k) if(j.stiffness_Nm_per_rad[k]!=0.f||j.damping_Nms_per_rad[k]!=0.f) return true;
    return false;
}
// Per-axis spring + step-start-referenced damping. Physical units; no dt^2.
[[seam::host_device_fn]] inline void weights(const AngularControlGpu& j, const float c[NCOORD], const float c0[NCOORD], float dt, float g[NCOORD], float h[NCOORD], float* energy) {
    float e=0.f;
    for(int k=0;k<NCOORD;++k){
        const float d=c[k]-c0[k];
        const float kd=j.stiffness_Nm_per_rad[k];
        const float cd=j.damping_Nms_per_rad[k]/dt;
        g[k]=kd*c[k]+cd*d;
        h[k]=kd+cd;
        e+=.5f*(kd*c[k]*c[k]+cd*d*d);
    }
    if(energy) *energy=e;
}
[[seam::host_device_fn]] inline bool curve_weights(const AngularControlGpu& row,const restoring_curve::CurveContext& curves,unsigned kind,unsigned target,const float c[3],const float c0[3],float dt,float g[3],float h[3],float* energy,unsigned& code,float* observations=nullptr,float* diagnostics=nullptr) {
 float total=0.f;
 for(unsigned axis=0;axis<3;++axis){float e=0.f;const auto* binding=restoring_curve::lookup(curves,kind,target,axis,code);if(code)return false;
  if(binding){if(!restoring_curve::angular(curves,kind,target,axis,c[axis],c0[axis],dt,e,g[axis],h[axis],code,diagnostics))return false;if(observations)restoring_observe(curves,kind,target,axis,c[axis],g[axis],e,dt,c0[axis],observations);}
  else{float d=c[axis]-c0[axis],cd=row.damping_Nms_per_rad[axis]/dt;g[axis]=row.stiffness_Nm_per_rad[axis]*c[axis]+cd*d;h[axis]=row.stiffness_Nm_per_rad[axis]+cd;e=.5f*(row.stiffness_Nm_per_rad[axis]*c[axis]*c[axis]+cd*d*d);}
  total+=e;
 }
 if(!finite(total)){code=restoring_curve::CurveFaultArithmetic;return false;}if(energy)*energy=total;return true;
}
[[seam::host_device_fn]] inline bool eval(const AngularControlGpu& j, const PdrdRigidState* s, float c[NCOORD], float J[NCOORD*NINCR], unsigned* domain_reason=nullptr, bool continuous_twist=false, float* radius=nullptr) {
    float ra[9],rb[9];
    const float I[9]={1.f,0.f,0.f,0.f,1.f,0.f,0.f,0.f,1.f};
    if(j.body_a==CONTROL_WORLD_ENDPOINT){ for(int i=0;i<9;++i)ra[i]=I[i]; } else rigid_rowmajor(s,j.body_a,ra);
    if(j.body_b==CONTROL_WORLD_ENDPOINT){ for(int i=0;i<9;++i)rb[i]=I[i]; } else rigid_rowmajor(s,j.body_b,rb);
    return chart(j,ra,rb,c,J,domain_reason,continuous_twist,radius);
}
[[seam::host_device_fn]] inline bool reduced_offset(unsigned base, unsigned body, unsigned dim, unsigned* out) {
    if(base>dim) return false;
    if(body>=(dim-base)/6u) return false;
    const unsigned first=base+6u*body;
    if(first>dim||first+5u>=dim) return false;
    *out=first; return true;
}
[[seam::host_device_fn]] inline bool block_offset(unsigned body, unsigned* out) {
    if(body>0xffffffffu/36u-1u) return false;
    const unsigned first=36u*body;
    if(first>0xffffffffu||first+35u>0xffffffffu) return false;
    *out=first; return true;
}

// Resolve both charts and the live body indices, returning a typed fault code
// instead of a bare bool so the caller can report which joint and why.
[[seam::host_device_fn]] inline unsigned resolve(const AngularControlGpu& j, const PdrdRigidState* s, const PdrdRigidState* s0,
        unsigned bodies, float c[NCOORD], float J[NCOORD*NINCR], float c0[NCOORD], float J0[NCOORD*NINCR], const AngularTwistEvalContext* twist=nullptr) {
    if((j.body_a!=CONTROL_WORLD_ENDPOINT&&j.body_a>=bodies)||(j.body_b!=CONTROL_WORLD_ENDPOINT&&j.body_b>=bodies))
        return CONTROL_FAULT_BODY_INDEX;
    const bool continuous=twist && twist->continuous==1u;
    if(twist && (twist->continuous>1u || (continuous && twist->initialized!=1u))) return CONTROL_FAULT_TWIST_HISTORY;
    unsigned domain_reason=CHART_DOMAIN_NONE;
    if(!eval(j,s,c,J,&domain_reason,continuous)) {
        if(domain_reason!=CHART_DOMAIN_NONE) {
            // A current-domain candidate never masks an invalid frozen start.
            if(!eval(j,s0,c0,J0,nullptr,continuous)) return CONTROL_FAULT_CHART_REFERENCE;
            return domain_reason==CHART_DOMAIN_TWIST?CONTROL_FAULT_CURRENT_TWIST_DOMAIN:CONTROL_FAULT_CURRENT_SWING_DOMAIN;
        }
        return CONTROL_FAULT_CHART_CURRENT;
    }
    if(!eval(j,s0,c0,J0,nullptr,continuous)) return CONTROL_FAULT_CHART_REFERENCE;
    if(continuous && (!lift_twist(c[0],twist->current_anchor,c[0]) || !lift_twist(c0[0],twist->reference_anchor,c0[0]))) return CONTROL_FAULT_TWIST_HISTORY;
    return CONTROL_FAULT_NONE;
}

// Accepted material state is immutable through Newton and line search.
// This resolves the independently defined local-geodesic v3 material chart;
// it deliberately does not invoke the global v2 swing/twist chart.
[[seam::host_device_fn]] inline unsigned resolve_material(const AngularControlGpu& row,
        const PdrdRigidState* state, unsigned bodies, [[seam::thread]] const MaterialSpinContext& accepted,
        float c[3], float J[18], float c0[3]) {
    if(accepted.model!=3u)return CONTROL_FAULT_MATERIAL_INVALID;
    if((row.body_a!=CONTROL_WORLD_ENDPOINT&&row.body_a>=bodies)||
       (row.body_b!=CONTROL_WORLD_ENDPOINT&&row.body_b>=bodies))return CONTROL_FAULT_BODY_INDEX;
    float q[4],T[9];
    if(!material_relative_pose(row,state,bodies,q,T))return CONTROL_FAULT_MATERIAL_INVALID;
    material_spin_v3::Trial trial{};
    unsigned status=material_spin_v3::evaluate(accepted.accepted_q_wxyz,accepted.accepted_a,q,trial);
    if(status==material_spin_v3::LOCAL_STEP_AMBIGUOUS)return CONTROL_FAULT_MATERIAL_LOCAL_STEP;
    if(status!=material_spin_v3::OK || !material_body_jacobian(row,T,trial.jac_body,J))return CONTROL_FAULT_MATERIAL_INVALID;
    for(int k=0;k<3;++k){c[k]=trial.strain[k];c0[k]=accepted.accepted_a[k];}
    return CONTROL_FAULT_NONE;
}
[[seam::device_fn]] inline void gradient_kernel(const AngularControlGpu* js,const AngularTwistEvalContext* contexts,unsigned count,const PdrdRigidState* s,const PdrdRigidState* s0,unsigned bodies,unsigned base,unsigned dim,float dt,compute::atomic_float_t* out,unsigned* fault,unsigned i) {
    if(i>=count) return;
    const AngularControlGpu j=js[i];
    if(!row_requested(j)) return;          // disabled / non-requested: true no-op
    if(!contributes(j)) return;            // all-zero coefficients contribute exactly zero
    float c[NCOORD],c0[NCOORD],J[NCOORD*NINCR],J0[NCOORD*NINCR],g[NCOORD],h[NCOORD];
    unsigned code=resolve(j,s,s0,bodies,c,J,c0,J0,contexts+i);
    if(code==CONTROL_FAULT_NONE){
        unsigned a=0,b=0;
        if(j.body_a!=CONTROL_WORLD_ENDPOINT&&!reduced_offset(base,j.body_a,dim,&a)) code=CONTROL_FAULT_REDUCED_RANGE;
        else if(j.body_b!=CONTROL_WORLD_ENDPOINT&&!reduced_offset(base,j.body_b,dim,&b)) code=CONTROL_FAULT_REDUCED_RANGE;
        if(code==CONTROL_FAULT_NONE){
            weights(j,c,c0,dt,g,h,nullptr);
            // R_trial = exp(-increment) R_work, so the reduced direction is the
            // positive energy gradient, matching the positional joint convention.
            for(int k=0;k<NINCR;++k){
                float v=0.f;
                for(int r=0;r<NCOORD;++r) v+=J[NINCR*r+k]*g[r];
                if(k<3){ if(j.body_a!=CONTROL_WORLD_ENDPOINT) compute::atomic_add(out+a+3+k,v); }
                else { if(j.body_b!=CONTROL_WORLD_ENDPOINT) compute::atomic_add(out+b+k,v); }
            }
            return;
        }
    }
    // Requested but unusable: record, never silently vanish.
    if(fault) fault[i]=code;
}
[[seam::device_fn]] inline void matvec_kernel(const AngularControlGpu* js,const AngularTwistEvalContext* contexts,unsigned count,const PdrdRigidState* s,const PdrdRigidState* s0,unsigned bodies,unsigned base,unsigned dim,float dt,const float* in,compute::atomic_float_t* out,unsigned* fault,unsigned i) {
    if(i>=count) return;
    const AngularControlGpu j=js[i];
    if(!row_requested(j)) return;
    if(!contributes(j)) return;
    float c[NCOORD],c0[NCOORD],J[NCOORD*NINCR],J0[NCOORD*NINCR],g[NCOORD],h[NCOORD];
    unsigned code=resolve(j,s,s0,bodies,c,J,c0,J0,contexts+i);
    if(code==CONTROL_FAULT_NONE){
        unsigned a=0,b=0;
        if(j.body_a!=CONTROL_WORLD_ENDPOINT&&!reduced_offset(base,j.body_a,dim,&a)) code=CONTROL_FAULT_REDUCED_RANGE;
        else if(j.body_b!=CONTROL_WORLD_ENDPOINT&&!reduced_offset(base,j.body_b,dim,&b)) code=CONTROL_FAULT_REDUCED_RANGE;
        if(code==CONTROL_FAULT_NONE){
            weights(j,c,c0,dt,g,h,nullptr);
            float x[NINCR]={};
            if(j.body_a!=CONTROL_WORLD_ENDPOINT) for(int k=0;k<3;++k) x[k]=in[a+3+k];
            if(j.body_b!=CONTROL_WORLD_ENDPOINT) for(int k=0;k<3;++k) x[3+k]=in[b+3+k];
            float z[NCOORD]={};
            for(int r=0;r<NCOORD;++r){ for(int k=0;k<NINCR;++k) z[r]+=J[NINCR*r+k]*x[k]; z[r]*=h[r]; }
            for(int k=0;k<NINCR;++k){
                float v=0.f;
                for(int r=0;r<NCOORD;++r) v+=J[NINCR*r+k]*z[r];
                if(k<3){ if(j.body_a!=CONTROL_WORLD_ENDPOINT) compute::atomic_add(out+a+3+k,v); }
                else { if(j.body_b!=CONTROL_WORLD_ENDPOINT) compute::atomic_add(out+b+k,v); }
            }
            return;
        }
    }
    if(fault) fault[i]=code;
}
// Freeze the exact per-row Gauss-Newton tangent once for one Newton pose.
// Storage: J[3x6], h[3], two reduced body offsets and a publication epoch.
[[seam::device_fn]] inline void tangent_build_kernel(const AngularControlGpu* js,const AngularTwistEvalContext* contexts,
        unsigned count,const PdrdRigidState* s,const PdrdRigidState* s0,unsigned bodies,unsigned base,unsigned dim,float dt,
        float* factors,unsigned* offsets,unsigned* published,unsigned epoch,unsigned* fault,unsigned i) {
    if(i>=count)return;
    published[i]=0u;
    const AngularControlGpu j=js[i];
    unsigned a=CONTROL_TANGENT_NONE,b=CONTROL_TANGENT_NONE,code=CONTROL_FAULT_NONE;
    float* row=factors+CONTROL_TANGENT_STRIDE*i;
    for(unsigned k=0;k<CONTROL_TANGENT_STRIDE;++k)row[k]=0.f;
    if(epoch==0u){if(fault)fault[i]=CONTROL_FAULT_CHART_CURRENT;return;}
    if(!row_requested(j)||!contributes(j)){
        offsets[2u*i]=a;offsets[2u*i+1u]=b;
        if(fault)fault[i]=CONTROL_FAULT_NONE;
        published[i]=epoch;return;
    }
    float c[NCOORD],c0[NCOORD],J[NCOORD*NINCR],J0[NCOORD*NINCR],g[NCOORD],h[NCOORD];
    code=resolve(j,s,s0,bodies,c,J,c0,J0,contexts+i);
    if(code==CONTROL_FAULT_NONE){
        if(j.body_a!=CONTROL_WORLD_ENDPOINT&&!reduced_offset(base,j.body_a,dim,&a))code=CONTROL_FAULT_REDUCED_RANGE;
        else if(j.body_b!=CONTROL_WORLD_ENDPOINT&&!reduced_offset(base,j.body_b,dim,&b))code=CONTROL_FAULT_REDUCED_RANGE;
    }
    if(code==CONTROL_FAULT_NONE)weights(j,c,c0,dt,g,h,nullptr);
    if(code!=CONTROL_FAULT_NONE){if(fault)fault[i]=code;offsets[2u*i]=CONTROL_TANGENT_NONE;offsets[2u*i+1u]=CONTROL_TANGENT_NONE;return;}
    for(unsigned k=0;k<NCOORD*NINCR;++k)row[k]=J[k];
    for(unsigned k=0;k<NCOORD;++k)row[NCOORD*NINCR+k]=h[k];
    offsets[2u*i]=a;offsets[2u*i+1u]=b;
    if(fault)fault[i]=CONTROL_FAULT_NONE;
    published[i]=epoch;
}
[[seam::device_fn]] inline void tangent_apply_kernel(const AngularControlGpu* js,const AngularTwistEvalContext* contexts,unsigned count,
        const PdrdRigidState* s,const PdrdRigidState* s0,unsigned bodies,unsigned base,unsigned dim,float dt,const float* in,
        compute::atomic_float_t* out,unsigned* fault,const float* factors,const unsigned* offsets,const unsigned* published,
        unsigned epoch,unsigned i) {
    if(i>=count)return;
    if(epoch==0u||published[i]!=epoch){matvec_kernel(js,contexts,count,s,s0,bodies,base,dim,dt,in,out,fault,i);return;}
    const unsigned a=offsets[2u*i],b=offsets[2u*i+1u];
    if(a==CONTROL_TANGENT_NONE&&b==CONTROL_TANGENT_NONE)return;
    const float* row=factors+CONTROL_TANGENT_STRIDE*i;
    float x[NINCR]={},z[NCOORD]={},y[NINCR]={};
    if(a!=CONTROL_TANGENT_NONE)for(unsigned k=0;k<3;++k)x[k]=in[a+3u+k];
    if(b!=CONTROL_TANGENT_NONE)for(unsigned k=0;k<3;++k)x[3u+k]=in[b+3u+k];
    for(unsigned r=0;r<NCOORD;++r){for(unsigned k=0;k<NINCR;++k)z[r]+=row[NINCR*r+k]*x[k];z[r]*=row[NCOORD*NINCR+r];}
    for(unsigned k=0;k<NINCR;++k)for(unsigned r=0;r<NCOORD;++r)y[k]+=row[NINCR*r+k]*z[r];
    if(a!=CONTROL_TANGENT_NONE)for(unsigned k=0;k<3;++k)compute::atomic_add(out+a+3u+k,y[k]);
    if(b!=CONTROL_TANGENT_NONE)for(unsigned k=0;k<3;++k)compute::atomic_add(out+b+3u+k,y[3u+k]);
}
[[seam::device_fn]] inline void blocks_kernel(const AngularControlGpu* js,const AngularTwistEvalContext* contexts,unsigned count,const PdrdRigidState* s,const PdrdRigidState* s0,unsigned bodies,float dt,compute::atomic_float_t* blocks,unsigned* fault,unsigned i) {
    if(i>=count) return;
    const AngularControlGpu j=js[i];
    if(!row_requested(j)) return;
    if(!contributes(j)) return;
    float c[NCOORD],c0[NCOORD],J[NCOORD*NINCR],J0[NCOORD*NINCR],g[NCOORD],h[NCOORD];
    unsigned code=resolve(j,s,s0,bodies,c,J,c0,J0,contexts+i);
    if(code==CONTROL_FAULT_NONE){
        unsigned a=0,b=0;
        if(j.body_a!=CONTROL_WORLD_ENDPOINT&&!block_offset(j.body_a,&a)) code=CONTROL_FAULT_REDUCED_RANGE;
        else if(j.body_b!=CONTROL_WORLD_ENDPOINT&&!block_offset(j.body_b,&b)) code=CONTROL_FAULT_REDUCED_RANGE;
        if(code==CONTROL_FAULT_NONE){
            weights(j,c,c0,dt,g,h,nullptr);
            // J^T diag(h) J restricted to the rotational 3x3 block of each body.
            for(int ra=0;ra<3;++ra) for(int ca=0;ca<3;++ca){
                float v=0.f;
                for(int r=0;r<NCOORD;++r) v+=J[NINCR*r+ra]*h[r]*J[NINCR*r+ca];
                if(j.body_a!=CONTROL_WORLD_ENDPOINT) compute::atomic_add(blocks+a+6*(3+ra)+(3+ca),v);
                if(j.body_b!=CONTROL_WORLD_ENDPOINT) compute::atomic_add(blocks+b+6*(3+ra)+(3+ca),v);
            }
            return;
        }
    }
    if(fault) fault[i]=code;
}
// Full-lane overwrite of the per-joint fault codes at one frozen pose. Zero for
// a row that is not requested, so a disabled row can never be reported as a
// geometry failure, and zero for a requested row whose coefficients are all
// zero, because such a row contributes exactly nothing and must therefore be
// exactly inert: reporting its chart as unusable would refuse a step the physics
// never asked about. Runs once per pose, before assembly/PCG: no per-matvec sync.
[[seam::device_fn]] inline void check_kernel(const AngularControlGpu* js,const AngularTwistEvalContext* contexts,unsigned count,const PdrdRigidState* s,const PdrdRigidState* s0,unsigned bodies,unsigned* fault,unsigned i) {
    if(i>=count) return;
    const AngularControlGpu j=js[i];
    if(!row_requested(j)||!contributes(j)){ if(fault) fault[i]=CONTROL_FAULT_NONE; return; }
    float c[NCOORD],c0[NCOORD],J[NCOORD*NINCR],J0[NCOORD*NINCR];
    const unsigned code=resolve(j,s,s0,bodies,c,J,c0,J0,contexts+i);
    if(fault) fault[i]=code;
}

} // namespace joint_angular_control

[[seam::entry(i)]]
[[seam::device_fn]] inline void joint_angular_control_apply(const AngularControlGpu *records, const AngularTwistEvalContext *twist_context, const MaterialSpinContext *material_spin_context, const PdrdRigidState *state, const PdrdRigidState *reference, unsigned bodies, unsigned base, unsigned dim, float dt, unsigned action, const float *input, compute::atomic_float_t *output, unsigned *fault, float *tangent_cache, unsigned *tangent_offsets, unsigned *tangent_published, unsigned epoch, unsigned i) {
 if(action==5u) {
   if(input[i]==0.f){fault[i]=0u;return;}
   float q[4],T[9];
   if(!joint_angular_control::material_relative_pose(records[i],state,bodies,q,T)) {
     fault[i]=joint_angular_control::CONTROL_FAULT_CHART_CURRENT; return;
   }
   for(unsigned k=0;k<4;++k)compute::atomic_add(output+4u*i+k,q[k]);
   fault[i]=0u;
 }
 else if(action==0u) joint_angular_control::gradient_kernel(records,twist_context,0xffffffffu,state,reference,bodies,base,dim,dt,output,fault,i);
 else if(action==1u) joint_angular_control::matvec_kernel(records,twist_context,0xffffffffu,state,reference,bodies,base,dim,dt,input,output,fault,i);
 else if(action==2u) joint_angular_control::blocks_kernel(records,twist_context,0xffffffffu,state,reference,bodies,dt,output,fault,i);
 else if(action==3u) {
   if(material_spin_context[i].model==3u)fault[i]=0u;
   else joint_angular_control::check_kernel(records,twist_context,0xffffffffu,state,reference,bodies,fault,i);
 }
 else if(action==4u) {
   const AngularControlGpu j=records[i];
   if(input[i]==0.f) { fault[i]=0u; return; }
   if((j.body_a!=joint_angular_control::CONTROL_WORLD_ENDPOINT && j.body_a>=bodies) ||
      (j.body_b!=joint_angular_control::CONTROL_WORLD_ENDPOINT && j.body_b>=bodies)) {
     fault[i]=joint_angular_control::CONTROL_FAULT_BODY_INDEX; return;
   }
   float c[3],J[18],radius=0.f;
   unsigned domain_reason=joint_angular_control::CHART_DOMAIN_NONE;
   if(!joint_angular_control::eval(j,state,c,J,&domain_reason,true,&radius)) {
     fault[i]=domain_reason==joint_angular_control::CHART_DOMAIN_SWING
       ?joint_angular_control::CONTROL_FAULT_CURRENT_SWING_DOMAIN
       :joint_angular_control::CONTROL_FAULT_CHART_CURRENT;
     return;
   }
   joint_angular_control::Quat qa,qb;
   float ra[9],rb[9];
   if(j.body_a==joint_angular_control::CONTROL_WORLD_ENDPOINT) { qa.w=joint_angular_control::Dual(1.f); }
   else { joint_angular_control::rigid_rowmajor(state,j.body_a,ra); if(!joint_angular_control::quat_from_matrix(ra,qa)){fault[i]=joint_angular_control::CONTROL_FAULT_CHART_CURRENT;return;} }
   if(j.body_b==joint_angular_control::CONTROL_WORLD_ENDPOINT) { qb.w=joint_angular_control::Dual(1.f); }
   else { joint_angular_control::rigid_rowmajor(state,j.body_b,rb); if(!joint_angular_control::quat_from_matrix(rb,qb)){fault[i]=joint_angular_control::CONTROL_FAULT_CHART_CURRENT;return;} }
   const float out[10]={c[0],radius,qa.w.v,qa.x.v,qa.y.v,qa.z.v,qb.w.v,qb.x.v,qb.y.v,qb.z.v};
   for(unsigned k=0;k<10u;++k) compute::atomic_add(output+10u*i+k,out[k]);
   fault[i]=0u;
 }
  else if(action==6u) joint_angular_control::tangent_build_kernel(records,twist_context,0xffffffffu,state,reference,bodies,base,dim,dt,tangent_cache,tangent_offsets,tangent_published,epoch,fault,i);
  else if(action==7u) joint_angular_control::tangent_apply_kernel(records,twist_context,0xffffffffu,state,reference,bodies,base,dim,dt,input,output,fault,tangent_cache,tangent_offsets,tangent_published,epoch,i);
}






