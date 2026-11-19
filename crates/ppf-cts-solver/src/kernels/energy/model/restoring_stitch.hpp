#pragma once
#include "../../zz_joint_ext/restoring_context.hpp"

// Frozen smooth signed stitch coordinate. rest_length_m is ALREADY the Build
// sqrt(l0*l0 + epsilon*epsilon), not the unsmoothed contact-gap rest l0.
// No history mutation, legacy cap or scalar evaluator lives in this helper.
namespace restoring_stitch {
// Hessian entries are tensor storage, not authored coefficients or scalar
// forces. A tiny off-axis product can round to zero even when both principal
// stiffnesses are representable. Preserve the strict generic conversion, but
// admit IEEE binary32 round-to-nearest-even zero at this storage boundary only.
// Require a normal scale for THIS tensor so rounding cannot erase an entirely
// subnormal stiffness. Each changed entry is at most half a minimum subnormal;
// for a symmetric 3x3 matrix the norm perturbation is <= 1.5 epsilon * scale.
// Raw and PSD tensors supply separate scales; no claim of exact PSD survives
// component rounding. Scalar, gradient and eigenvalue checks remain strict.
SM_INLINE_DEVICE_HOST bool narrow_hessian_component(
    restoring_curve::device_f32::Scaled value,float tensor_scale,float& out) {
    namespace f=restoring_curve::device_f32;
    if(f::narrow(value,out))return true;
    if(!f::finite(value.m)||!f::finite(tensor_scale)||
       tensor_scale<1.1754943508222875e-38f)return false;
    // Scaled arithmetic normalizes a nonzero significand to [1,2) in magnitude.
    if(value.e < -150 || (value.e == -150 && f::abs(value.m)<=1.f)) {
        out=value.m<0.f?-0.f:0.f;return true;
    }
    return false;
}

struct Evaluation {
    float energy;
    float gradient[3];
    float raw_hessian[9];
    float projected_hessian[9];
    float coordinate;
    float radial_eigenvalue;
    float transverse_eigenvalue;
};

// Explicit host-callable device arithmetic for independent high-precision tests.
// Rationalization is mathematically identical to sqrt(d.d+eps^2)-rest, but
// preserves a small displacement before a large damping coefficient is applied.
SM_INLINE_DEVICE_HOST bool coordinate_f32(
    const float d[3],float epsilon_m,float rest_length_m,float& q,unsigned& fault) {
    namespace f=restoring_curve::device_f32;
    fault=0;
    if(!d||!f::finite(epsilon_m)||epsilon_m<=0.f||!f::finite(rest_length_m)||rest_length_m<epsilon_m){
        fault=restoring_curve::CurveFaultArithmetic;return false;
    }
    f::CoordinateSum numerator{f::scaled(0.f),f::scaled(0.f)};
    f::Scaled radius2=f::square(f::scaled(epsilon_m));
    float terms[4]={epsilon_m,0.f,0.f,0.f};
    for(unsigned i=0;i<3;++i){
        if(!f::finite(d[i])){fault=restoring_curve::CurveFaultArithmetic;return false;}
        terms[i+1]=f::abs(d[i]);
        f::Scaled value=f::scaled(d[i]);
        radius2=f::add(radius2,f::square(value));
    }
    // Cancel the largest square against rest first. Adding a tiny eps^2
    // before two equal huge squares would lose it even with one residual.
    for(unsigned i=0;i<4;++i)for(unsigned j=i+1;j<4;++j)
        if(terms[j]>terms[i]){float t=terms[i];terms[i]=terms[j];terms[j]=t;}
    f::coordinate_product(numerator,f::scaled(terms[0]),f::scaled(terms[0]));
    f::coordinate_product(numerator,f::neg(f::scaled(rest_length_m)),f::scaled(rest_length_m));
    for(unsigned i=1;i<4;++i)f::coordinate_product(numerator,f::scaled(terms[i]),f::scaled(terms[i]));
    f::Scaled value=f::div(f::add(numerator.hi,numerator.lo),
                         f::add(f::root(radius2),f::scaled(rest_length_m)));
    if(!f::narrow(value,q)){fault=restoring_curve::CurveFaultArithmetic;return false;}
    return true;
}

#if !defined(SM_MSL_CONCAT)
SM_INLINE_DEVICE_HOST bool coordinate_legacy(
    const float d[3], float epsilon_m, float rest_length_m,
    float& q, unsigned& fault) {
    fault=0;
    if (!d || !restoring_curve::finite(epsilon_m) || epsilon_m<=0.f ||
        !restoring_curve::finite(rest_length_m) || rest_length_m<epsilon_m) {
        fault=restoring_curve::CurveFaultArithmetic;return false;
    }
    double square=double(epsilon_m)*epsilon_m;
    for (unsigned i=0;i<3;++i) {
        if (!restoring_curve::finite(d[i])) {fault=restoring_curve::CurveFaultArithmetic;return false;}
        square+=double(d[i])*d[i];
    }
    const double value=restoring_curve::positive_sqrt(square)-double(rest_length_m);
    if (!restoring_curve::narrow(value,q)) {fault=restoring_curve::CurveFaultArithmetic;return false;}
    return true;
}
#endif
SM_INLINE_DEVICE_HOST bool coordinate(const float d[3],float epsilon_m,float rest_length_m,float& q,unsigned& fault) { return coordinate_f32(d,epsilon_m,rest_length_m,q,fault); }

// Shared scalar-distance law. The caller owns the family lookup, so internal
// cross-links reuse exactly the stitch arithmetic without spoofing Stitch rows.
SM_INLINE_DEVICE_HOST bool evaluate_binding_f32(
    const restoring_curve::CurveContext& ctx,const CurveBinding* b,
    const float d[3],float dt,Evaluation& output,unsigned& fault) {
    namespace f=restoring_curve::device_f32;
    if(!b||fault)return false;
    if(!f::finite(dt)||dt<=0.f||!f::finite(b->damping)||b->damping<0.f){fault=restoring_curve::CurveFaultArithmetic;return false;}
    const CurveHistory* history=restoring_curve::history_for(ctx,*b,fault);
    if(!history||fault)return false;
    if(!f::finite(history->q0)){fault=restoring_curve::CurveFaultHistory;return false;}
    if(!ctx.descriptors||b->table_index>=ctx.descriptor_count||ctx.descriptors[b->table_index].domain>1){fault=restoring_curve::CurveFaultBinding;return false;}
    Evaluation candidate{};
    if(!coordinate_f32(d,b->epsilon_m,b->rest_length_m,candidate.coordinate,fault))return false;
    auto scalar=restoring_curve::evaluate_f32(ctx,*b,candidate.coordinate);
    if(scalar.fault!=restoring_curve::Fault::None){fault=restoring_curve::CurveFaultScalar;return false;}
    const unsigned mask=b->reserved[0];
    if ((mask && (b->kind!=restoring_curve::CrossLinkDistance || !(mask&1u) || (mask&~7u))) || b->reserved[1] || b->reserved[2]) {fault=restoring_curve::CurveFaultBinding;return false;}
    const bool tension=!mask || (mask&2u), compression=!mask || (mask&4u);
    const float q=candidate.coordinate, q0=history->q0;
    const bool active=(q>=0.f ? tension : compression);
    const float projected=tension&&compression ? q : (tension ? (q>0.f?q:0.f) : (compression ? (q<0.f?q:0.f) : 0.f));
    auto cd=f::div(f::scaled(b->damping),f::scaled(dt));
    auto dq=f::sub(f::scaled(projected),f::scaled(q0));
    auto damp_square=f::square(dq);
    // Remove the constant energy at the inactive rest side without a
    // cancellation-prone subtraction of two nearly equal squares.
    const bool opposite=(tension&&!compression&&q0<0.f)||(compression&&!tension&&q0>0.f);
    if(opposite) damp_square=f::mul(f::scaled(projected),f::sub(f::scaled(projected),f::mul(f::scaled(2.f),f::scaled(q0))));
    if(!tension&&!compression) damp_square=f::scaled(0.f);
    auto e=f::add(f::scaled(scalar.energy),f::mul(f::scaled(.5f),f::mul(cd,damp_square)));
    auto g=f::add(f::scaled(scalar.gradient),active?f::mul(cd,dq):f::scaled(0.f));
    auto h=f::add(f::scaled(scalar.curvature),active?cd:f::scaled(0.f));
    if(!f::narrow(e,candidate.energy,true)){fault=restoring_curve::CurveFaultArithmetic;return false;}
    auto dn=f::norm(d,3);
    auto r=f::root(f::add(f::square(dn),f::square(f::scaled(b->epsilon_m))));
    auto transverse=f::div(g,r);
    auto radial=f::add(f::mul(h,f::square(f::div(dn,r))),
        f::mul(transverse,f::square(f::div(f::scaled(b->epsilon_m),r))));
    if(!f::narrow(radial,candidate.radial_eigenvalue)||!f::narrow(transverse,candidate.transverse_eigenvalue)){
        fault=restoring_curve::CurveFaultArithmetic;return false;
    }
    auto radial_psd=radial.m>0.f?radial:f::scaled(0.f);
    auto transverse_psd=transverse.m>0.f?transverse:f::scaled(0.f);
    const float raw_scale=f::abs(candidate.radial_eigenvalue)>f::abs(candidate.transverse_eigenvalue)
        ?f::abs(candidate.radial_eigenvalue):f::abs(candidate.transverse_eigenvalue);
    const float positive_radial=candidate.radial_eigenvalue>0.f?candidate.radial_eigenvalue:0.f;
    const float positive_transverse=candidate.transverse_eigenvalue>0.f?candidate.transverse_eigenvalue:0.f;
    const float projected_scale=positive_radial>positive_transverse?positive_radial:positive_transverse;
    f::Scaled n[3];
    for(unsigned i=0;i<3;++i)n[i]=dn.m!=0.f?f::div(f::scaled(d[i]),dn):f::scaled(0.f);
    for(unsigned i=0;i<3;++i){
        if(!f::narrow(f::mul(g,f::div(f::scaled(d[i]),r)),candidate.gradient[i])){
            fault=restoring_curve::CurveFaultArithmetic;return false;
        }
        for(unsigned j=0;j<3;++j){
            auto outer=f::mul(n[i],n[j]),other=f::sub(f::scaled(i==j?1.f:0.f),outer);
            auto exact=f::add(f::mul(radial,outer),f::mul(transverse,other));
            auto projected=f::add(f::mul(radial_psd,outer),f::mul(transverse_psd,other));
            if(!narrow_hessian_component(exact,raw_scale,candidate.raw_hessian[3*i+j])||!narrow_hessian_component(projected,projected_scale,candidate.projected_hessian[3*i+j])){
                fault=restoring_curve::CurveFaultArithmetic;return false;
            }
        }
    }
    output=candidate;return true;
}

SM_INLINE_DEVICE_HOST bool evaluate_detailed_f32(
    const restoring_curve::CurveContext& ctx,unsigned row_index,
    const float d[3],float dt,Evaluation& output,unsigned& fault) {
    const CurveBinding* binding=restoring_curve::lookup(ctx,restoring_curve::Stitch,row_index,restoring_curve::AnchorChannel,fault);
    return evaluate_binding_f32(ctx,binding,d,dt,output,fault);
}

#if !defined(SM_MSL_CONCAT)
SM_INLINE_DEVICE_HOST bool evaluate_detailed_legacy(
    const restoring_curve::CurveContext& ctx, unsigned row_index,
    const float d[3], float dt, Evaluation& output, unsigned& fault) {
    const CurveBinding* binding=restoring_curve::lookup(
        ctx,restoring_curve::Stitch,row_index,restoring_curve::AnchorChannel,fault);
    if (!binding || fault) return false; // Inactive: untouched legacy caller path.
    if (!restoring_curve::finite(dt) || dt<=0.f || !restoring_curve::finite(binding->damping) || binding->damping<0.f) {
        fault=restoring_curve::CurveFaultArithmetic;return false;
    }
    const CurveHistory* history=restoring_curve::history_for(ctx,*binding,fault);
    if (!history || fault) return false;
    if (!restoring_curve::finite(history->q0)) {fault=restoring_curve::CurveFaultHistory;return false;}
    if (!ctx.descriptors || binding->table_index>=ctx.descriptor_count ||
        ctx.descriptors[binding->table_index].domain>1) {
        fault=restoring_curve::CurveFaultBinding;return false; // SIGNED/ODD only.
    }
    Evaluation candidate{};
    if (!coordinate_legacy(d,binding->epsilon_m,binding->rest_length_m,candidate.coordinate,fault)) return false;
    const auto scalar=restoring_curve::evaluate_legacy(ctx,*binding,candidate.coordinate);
    if (scalar.fault!=restoring_curve::Fault::None) {fault=restoring_curve::CurveFaultScalar;return false;}
    const double damping=double(binding->damping)/dt;
    const double dq=double(candidate.coordinate)-history->q0;
    const double energy=double(scalar.energy)+.5*damping*dq*dq;
    const double g=double(scalar.gradient)+damping*dq;
    const double h=double(scalar.curvature)+damping;
    if (!restoring_curve::narrow_energy(energy,candidate.energy)) {fault=restoring_curve::CurveFaultArithmetic;return false;}
    const double eps=binding->epsilon_m;
    double norm2=0.;for(unsigned i=0;i<3;++i)norm2+=double(d[i])*d[i];
    const double r2=norm2+eps*eps;
    const double r=restoring_curve::positive_sqrt(r2);
    const double transverse=g/r;
    const double radial=h*norm2/r2+g*eps*eps/(r2*r);
    if (!restoring_curve::narrow(radial,candidate.radial_eigenvalue) ||
        !restoring_curve::narrow(transverse,candidate.transverse_eigenvalue)) {
        fault=restoring_curve::CurveFaultArithmetic;return false;
    }
    // At d=0 all directions coincide, grad=0 and H=max(0,g/eps)*I.
    // Else n=d/|d|, with independently projected longitudinal/transverse modes.
    const double radial_psd=radial>0.?radial:0.;
    const double transverse_psd=transverse>0.?transverse:0.;
    for(unsigned i=0;i<3;++i) {
        if (!restoring_curve::narrow(g*double(d[i])/r,candidate.gradient[i])) {
            fault=restoring_curve::CurveFaultArithmetic;return false;
        }
        for(unsigned j=0;j<3;++j) {
            const double outer=norm2>0.?double(d[i])*d[j]/norm2:0.;
            const double diagonal=i==j?1.:0.;
            const double exact=transverse*diagonal+(radial-transverse)*outer;
            const double projected=transverse_psd*diagonal+(radial_psd-transverse_psd)*outer;
            if (!restoring_curve::narrow(exact,candidate.raw_hessian[3*i+j]) ||
                !restoring_curve::narrow(projected,candidate.projected_hessian[3*i+j])) {
                fault=restoring_curve::CurveFaultArithmetic;return false;
            }
        }
    }
    output=candidate;return true;
}
#endif
SM_INLINE_DEVICE_HOST bool evaluate_detailed(const restoring_curve::CurveContext& ctx,unsigned row_index,const float d[3],float dt,Evaluation& output,unsigned& fault) { return evaluate_detailed_f32(ctx,row_index,d,dt,output,fault); }

SM_INLINE_DEVICE_HOST bool evaluate(
    const restoring_curve::CurveContext& ctx, unsigned row_index,
    const float d[3], float dt, float& energy, float gradient[3],
    float hessian[9], unsigned& fault) {
    Evaluation output{};
    if (!evaluate_detailed(ctx,row_index,d,dt,output,fault)) return false;
    energy=output.energy;
    for(unsigned i=0;i<3;++i)gradient[i]=output.gradient[i];
    for(unsigned i=0;i<9;++i)hessian[i]=output.projected_hessian[i];
    return true;
}
} // namespace restoring_stitch
