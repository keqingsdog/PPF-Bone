#pragma once
#include "../seam/seam.hpp"
#include "../float_math.hpp"
#ifndef SM_INLINE_DEVICE_HOST
#define SM_INLINE_DEVICE_HOST inline
#endif

// Version-3 incremental material-spin coordinates.  All quantities are local
// to the joint's existing relative material frame; see angular_control::chart
// for the order inv(Ra*Fa)*(Rb*Fb)*inv(rest).
//
// Accepted (Qn,a_n) is immutable during Newton/line search. For a short
// Qn->Qt step, delta=Log(Qn^-1 Qt) and the candidate strain is the exact
// integral of dot(tau)=omega_x, dot(b)=Rot2(tau)*omega_yz along constant body
// spin delta.  This is a path-memory constitutive model, not an atlas-only
// rewrite of the v2 swing/twist energy. This chosen geodesic defines the
// discrete trial; Newton/line-search traversal is not physical history. The
// caller must resolve time sampling and commit Q/a only on an accepted step.
namespace material_spin_v3 {

enum Status : unsigned {
    OK=0u,
    NONFINITE=1u,
    INVALID_QUATERNION=2u,
    LOCAL_STEP_AMBIGUOUS=3u,
    UNCERTIFIED_PATH=4u,
    LOST_PRECISION=5u
};

struct Trial {
    float accepted_candidate_q[4]; // wxyz, normalized and aligned with Qn
    float strain[3];               // tau, by, bz
    float increment[3];            // delta in accepted Qn body coordinates
    float jac_delta[9];            // row-major: d(strain)/d(delta)
    float jac_body[9];             // row-major: d(strain)/d(Qt right increment)
};

SM_INLINE_DEVICE_HOST bool finite(float x) {
    return !fmath::isnan(x) && !fmath::isinf(x);
}

SM_INLINE_DEVICE_HOST unsigned normalize_q(SM_THREAD const float q[4],SM_THREAD float out[4]) {
    float n2=0.f;
    for(int k=0;k<4;++k){if(!finite(q[k]))return NONFINITE;n2+=q[k]*q[k];}
    // Inputs are unit rotations, rather than arbitrary quaternion magnitudes.
    if(!finite(n2)||n2<0.998f||n2>1.002f)return INVALID_QUATERNION;
    const float inv=1.f/fmath::sqrt(n2);
    for(int k=0;k<4;++k){out[k]=q[k]*inv;if(!finite(out[k]))return NONFINITE;}
    return OK;
}

SM_INLINE_DEVICE_HOST void sinc_cosc(float x,SM_THREAD float& s,SM_THREAD float& c,SM_THREAD float& ds,SM_THREAD float& dc) {
    const float x2=x*x;
    if(fmath::abs(x)<0.25f){
        const float x4=x2*x2,x6=x4*x2;
        s=1.f-x2/6.f+x4/120.f-x6/5040.f;
        c=x*(.5f-x2/24.f+x4/720.f-x6/40320.f);
        ds=x*(-1.f/3.f+x2/30.f-x4/840.f+x6/45360.f);
        dc=.5f-x2/8.f+x4/144.f-x6/5760.f;
    }else{
        const float sx=fmath::sin_bounded(x),cx=fmath::cos_bounded(x);
        s=sx/x;c=(1.f-cx)/x;
        ds=(x*cx-sx)/x2;dc=(x*sx-(1.f-cx))/x2;
    }
}

SM_INLINE_DEVICE_HOST unsigned evaluate(SM_THREAD const float accepted_q[4],
    SM_THREAD const float accepted_a[3],SM_THREAD const float trial_q[4],SM_THREAD Trial& output) {
    for(int k=0;k<3;++k)if(!finite(accepted_a[k]))return NONFINITE;
    float n[4],t[4];
    unsigned code=normalize_q(accepted_q,n);if(code!=OK)return code;
    code=normalize_q(trial_q,t);if(code!=OK)return code;
    float dot=0.f;for(int k=0;k<4;++k)dot+=n[k]*t[k];
    if(!finite(dot))return NONFINITE;
    if(dot<0.f){dot=-dot;for(int k=0;k<4;++k)t[k]=-t[k];}
    // qrel=conj(Qn)*Qt, taking the representative aligned with Qn.
    const float qv[3]={
        n[0]*t[1]-n[1]*t[0]-n[2]*t[3]+n[3]*t[2],
        n[0]*t[2]+n[1]*t[3]-n[2]*t[0]-n[3]*t[1],
        n[0]*t[3]-n[1]*t[2]+n[2]*t[1]-n[3]*t[0]
    };
    const float s2=qv[0]*qv[0]+qv[1]*qv[1]+qv[2]*qv[2];
    if(!finite(s2))return NONFINITE;
    const float v=fmath::sqrt(s2);
    float scale;
    if(v<1e-4f)scale=2.f+s2/3.f+3.f*s2*s2/20.f;
    else scale=(2.f*fmath::atan2(v,dot))/v;
    Trial tmp{};
    for(int k=0;k<4;++k)tmp.accepted_candidate_q[k]=t[k];
    for(int k=0;k<3;++k)tmp.increment[k]=scale*qv[k];
    const float x=tmp.increment[0],y=tmp.increment[1],z=tmp.increment[2];
    const float norm2=x*x+y*y+z*z;
    if(!finite(norm2))return NONFINITE;
    const float local_angle=fmath::sqrt(norm2);
    // This is an integration-step domain, not a global bending-angle bound.
    // A caller must retry with a shorter time step rather than commit an
    // out-of-domain local step. It does not by itself prove no temporal alias.
    if(!(dot>0.70710678118654752440f))return LOCAL_STEP_AMBIGUOUS;
    float s,c,ds,dc;sinc_cosc(x,s,c,ds,dc);
    constexpr float HALF_PI=1.57079632679489661923f;
    const float ca=fmath::sin_periodic(accepted_a[0]+HALF_PI);
    const float sa=fmath::sin_periodic(accepted_a[0]);
    const float u=s*y-c*z,vv=c*y+s*z;
    const float du=ds*y-dc*z,dv=dc*y+ds*z;
    tmp.strain[0]=accepted_a[0]+x;
    tmp.strain[1]=accepted_a[1]+ca*u-sa*vv;
    tmp.strain[2]=accepted_a[2]+sa*u+ca*vv;
    tmp.jac_delta[0]=1.f;tmp.jac_delta[1]=0.f;tmp.jac_delta[2]=0.f;
    tmp.jac_delta[3]=ca*du-sa*dv;tmp.jac_delta[4]=ca*s-sa*c;tmp.jac_delta[5]=-ca*c-sa*s;
    tmp.jac_delta[6]=sa*du+ca*dv;tmp.jac_delta[7]=sa*s+ca*c;tmp.jac_delta[8]=-sa*c+ca*s;
    // J_r^-1(delta)=I+hat(delta)/2+c*hat(delta)^2, for a right increment.
    float coeff;
    if(local_angle<1e-3f)coeff=1.f/12.f+norm2/720.f+norm2*norm2/30240.f;
    else {
        const float half=.5f*local_angle;
        const float sh=fmath::sin_bounded(half),ch=fmath::cos_bounded(half);
        if(!finite(sh)||!finite(ch)||sh==0.f)return NONFINITE;
        coeff=(1.f-half*ch/sh)/norm2;
    }
    const float skew[9]={0.f,-z,y,z,0.f,-x,-y,x,0.f};
    float ri[9];
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){
        float s2ij=0.f;for(int k=0;k<3;++k)s2ij+=skew[3*i+k]*skew[3*k+j];
        ri[3*i+j]=(i==j?1.f:0.f)+.5f*skew[3*i+j]+coeff*s2ij;
    }
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){
        float value=0.f;for(int k=0;k<3;++k)value+=tmp.jac_delta[3*i+k]*ri[3*k+j];
        tmp.jac_body[3*i+j]=value;
    }
    for(int k=0;k<3;++k)if(!finite(tmp.strain[k])||!finite(tmp.increment[k]))return NONFINITE;
    for(int k=0;k<9;++k)if(!finite(tmp.jac_delta[k])||!finite(tmp.jac_body[k]))return NONFINITE;
    output=tmp;
    return OK;
}
} // namespace material_spin_v3
