// Hard-only factor export. Reuse the ACTUAL float chart/form_h/rot routines;
// never recompute material/contact coefficients or build H from basis probes.
#pragma once
#include "../zz_joint/ball_anchor.kernel.cpp"
#include "../zz_joint/angular_joint_v2.kernel.cpp"
#include "../zz_joint_ext/angular_control.kernel.cpp"
struct HardSmoothFactorsGpu {
    unsigned body_a;
    unsigned body_b;
    unsigned ball_active;
    unsigned angular_active;
    unsigned control_active;
    unsigned fault;
    float p_a[3];
    float p_b[3];
    float ball_k;
    float angular_h[9];
    float control_j[18];
    float control_h[3];
};
[[seam::entry(i)]]
[[seam::device_fn]] inline void hard_frozen_smooth_factors(
    const BallAnchorGpu* ball, const AngularJointGpuV2* angular,
    const AngularControlGpu* control, const AngularTwistEvalContext* twist_context, const PdrdRigidState* state,
    const PdrdRigidState* reference, unsigned bodies, float dt,
    unsigned control_count, HardSmoothFactorsGpu* output, unsigned i) {
    HardSmoothFactorsGpu f{};
    const BallAnchorGpu b=ball[i]; const AngularJointGpuV2 a=angular[i];
    f.body_a=b.body_a; f.body_b=b.body_b;
    if(a.body_a!=b.body_a || a.body_b!=b.body_b || b.body_a>=bodies || b.body_b>=bodies || !(dt>0.f)) f.fault=1u;
    if(f.fault==0u && joint_ball::valid(b,bodies)) {
        f.ball_active=1u; f.ball_k=b.stiffness;
        const PdrdRigidState sa=state[b.body_a], sb=state[b.body_b];
        joint_ball::rot(sa.R,b.anchor_a,f.p_a); joint_ball::rot(sb.R,b.anchor_b,f.p_b);
    }
    if(f.fault==0u && joint_angular::valid(a,bodies)) {
        f.angular_active=1u;
        joint_angular::form_h(a,state,reference,joint_angular::ANGULAR_WORLD_ENDPOINT_V2,dt,f.angular_h);
    }
    if(f.fault==0u && i<control_count) {
        const AngularControlGpu c=control[i];
        if(c.body_a!=b.body_a || c.body_b!=b.body_b) f.fault=2u;
        if(f.fault==0u && joint_angular_control::row_requested(c) && joint_angular_control::contributes(c)) {
            float q[3],q0[3],j0[18],g[3];
            f.fault=joint_angular_control::resolve(c,state,reference,bodies,q,f.control_j,q0,j0,twist_context+i);
            if(f.fault==0u) {
                f.control_active=1u;
                joint_angular_control::weights(c,q,q0,dt,g,f.control_h,nullptr);
            }
        }
    }
    output[i]=f;
}

