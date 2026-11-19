#pragma once
#include "../zzzzzzzzz_restoring/restoring_apply.kernel.cpp"

// A frozen cache is valid for exactly one Newton linear solve. The host owns
// unique nonzero epochs, initializes the fault bank before this dispatch,
// reaps it before publishing a View, and invalidates the View on every refit.
[[seam::entry(i)]]
[[seam::device_fn]] inline void restoring_fem_frozen_build(
    const FemJointGpu* records, const MaterialSpinContext* material_spin_context,
    unsigned record_count, const FemFrameGpu* frames,
    const unsigned* vertex_to_free, unsigned vertex_count, unsigned free_count,
    unsigned rigid_base, unsigned dim, float dt,
    const CurveDescriptor* descriptors, const CurveNode* nodes,
    const CurveBinding* bindings, const CurveHistory* history,
    unsigned descriptor_count, unsigned node_count,
    unsigned binding_count, unsigned history_count, float* observations,
    unsigned epoch, float* cached_hessian, unsigned* cached_index,
    unsigned* published_epoch, unsigned* fault, unsigned i) {
    if (i >= record_count) return;
    published_epoch[i] = 0u;
    if (epoch == 0u || record_count > 0xffffffffu / 576u) {
        fault[i] = joint_fem::BAD_STEP;
        return;
    }
    const FemJointGpu row = records[i];
    unsigned idx[24];
    for (unsigned k = 0; k < 24; ++k) idx[k] = joint_fem::NONE;
    if (row.enabled) {
        if (!joint_fem::mapping(row.a, vertex_to_free, vertex_count, free_count,
                                rigid_base, dim, idx) ||
            !joint_fem::mapping(row.b, vertex_to_free, vertex_count, free_count,
                                rigid_base, dim, idx + 12)) {
            fault[i] = joint_fem::BAD_MAPPING;
            return;
        }
        const restoring_curve::CurveContext ctx{
            descriptors, nodes, bindings, history, descriptor_count,
            node_count, binding_count, history_count
        };
        const MaterialSpinContext material = material_spin_context[i];
        float gradient[24], hessian[576], energy;
        unsigned code;
        if (!joint_fem::evaluate(row, frames + 4u * i, dt, gradient, hessian,
                                 energy, code, &ctx, i, nullptr, observations,
                                 &material)) {
            fault[i] = code;
            return;
        }
        for (unsigned k = 0; k < 576; ++k)
            cached_hessian[576u * i + k] = hessian[k];
    }
    for (unsigned k = 0; k < 24; ++k)
        cached_index[24u * i + k] = idx[k];
    fault[i] = 0u;
    published_epoch[i] = epoch;
}

// The per-row k/l loop and atomic scatter match joint_fem::scatter(action=1).
// Repeated indices are retained, including aliased endpoints.
[[seam::entry(i)]]
[[seam::device_fn]] inline void restoring_fem_frozen_apply(
    unsigned record_count, const float* cached_hessian,
    const unsigned* cached_index, const unsigned* published_epoch,
    unsigned epoch, const float* input, compute::atomic_float_t* output,
    unsigned* fault, unsigned i) {
    if (i >= record_count) return;
    if (epoch == 0u || published_epoch[i] != epoch ||
        record_count > 0xffffffffu / 576u) {
        fault[i] = joint_fem::BAD_STEP;
        return;
    }
    const unsigned* idx = cached_index + 24u * i;
    const float* hessian = cached_hessian + 576u * i;
    for (unsigned k = 0; k < 24; ++k) {
        if (idx[k] == joint_fem::NONE) continue;
        float value = 0.0f;
        for (unsigned l = 0; l < 24; ++l) {
            if (idx[l] != joint_fem::NONE)
                value += hessian[24u * k + l] * input[idx[l]];
        }
        compute::atomic_add(output + idx[k], value);
    }
}

// Frozen directional angular tangent: J is 3x6 and the curvature is diagonal.
// The per-row ordering exactly matches the reference action=1.
[[seam::entry(i)]]
[[seam::device_fn]] inline void restoring_angular_frozen_build(
    const AngularControlGpu* records, const AngularTwistEvalContext* twist,
    const MaterialSpinContext* material_context, unsigned count_rows,
    const PdrdRigidState* state, const PdrdRigidState* reference,
    unsigned bodies, unsigned base, unsigned dim, float dt,
    const CurveDescriptor* descriptors, const CurveNode* nodes,
    const CurveBinding* bindings, const CurveHistory* history,
    unsigned descriptor_count, unsigned node_count, unsigned binding_count,
    unsigned history_count, unsigned epoch, float* factors,
    unsigned* offsets, unsigned* published, unsigned* fault, unsigned i) {
    if (i >= count_rows) return;
    published[i]=0u;
    const AngularControlGpu row=records[i];
    unsigned a=joint_fem::NONE,b=joint_fem::NONE;
    if(!joint_angular_control::row_requested(row)) {
        offsets[2u*i]=a; offsets[2u*i+1u]=b;
        published[i]=epoch; return;
    }
    const restoring_curve::CurveContext ctx{descriptors,nodes,bindings,history,descriptor_count,node_count,binding_count,history_count};
    const MaterialSpinContext material=material_context[i];
    float c[3],c0[3],j[18],j0[18],g[3],h[3],e;
    unsigned code=material.model==0u
      ? joint_angular_control::resolve(row,state,reference,bodies,c,j,c0,j0,twist+i)
      : joint_angular_control::resolve_material(row,state,bodies,material,c,j,c0);
    if(!code && !joint_angular_control::curve_weights(row,ctx,restoring_curve::RigidAngular,i,c,c0,dt,g,h,&e,code,nullptr)) {fault[i]=code;return;}
    if(!code && row.body_a!=joint_angular_control::CONTROL_WORLD_ENDPOINT && !joint_angular_control::reduced_offset(base,row.body_a,dim,&a)) code=joint_angular_control::CONTROL_FAULT_REDUCED_RANGE;
    if(!code && row.body_b!=joint_angular_control::CONTROL_WORLD_ENDPOINT && !joint_angular_control::reduced_offset(base,row.body_b,dim,&b)) code=joint_angular_control::CONTROL_FAULT_REDUCED_RANGE;
    if(code){fault[i]=code;return;}
    for(unsigned k=0;k<18;k++) factors[21u*i+k]=j[k];
    for(unsigned k=0;k<3;k++) factors[21u*i+18u+k]=h[k];
    offsets[2u*i]=a; offsets[2u*i+1u]=b; published[i]=epoch;
}
[[seam::entry(i)]]
[[seam::device_fn]] inline void restoring_angular_frozen_apply(
    unsigned count_rows, const float* factors, const unsigned* offsets,
    const unsigned* published,unsigned epoch,const float* input,
    compute::atomic_float_t* output,unsigned* fault,unsigned i) {
    if(i>=count_rows)return;
    if(!epoch || published[i]!=epoch){fault[i]=joint_fem::BAD_STEP;return;}
    const unsigned a=offsets[2u*i],b=offsets[2u*i+1u];
    if(a==joint_fem::NONE && b==joint_fem::NONE)return;
    const float* j=factors+21u*i;const float* h=j+18;
    float x[6]={},axis[3]={};
    for(unsigned k=0;k<3;k++){if(a!=joint_fem::NONE)x[k]=input[a+3u+k];if(b!=joint_fem::NONE)x[k+3u]=input[b+3u+k];}
    for(unsigned r=0;r<3;r++){for(unsigned k=0;k<6;k++)axis[r]+=j[6u*r+k]*x[k];axis[r]*=h[r];}
    for(unsigned k=0;k<6;k++){float v=0;for(unsigned r=0;r<3;r++)v+=j[6u*r+k]*axis[r];
      if(k<3 && a!=joint_fem::NONE)compute::atomic_add(output+a+3u+k,v);
      if(k>=3 && b!=joint_fem::NONE)compute::atomic_add(output+b+k,v);}
}

// Ball anchor: p/q and H3 describe the exact current tangent. The mode bit
// preserves legacy stiffness multiplication ordering when no curve is bound.
[[seam::entry(i)]]
[[seam::device_fn]] inline void restoring_ball_frozen_build(
    const BallAnchorGpu* records,unsigned count_rows,const PdrdRigidState* state,
    unsigned bodies,unsigned base,unsigned dim,float dt,
    const CurveDescriptor* descriptors,const CurveNode* nodes,
    const CurveBinding* bindings,const CurveHistory* history,
    unsigned descriptor_count,unsigned node_count,unsigned binding_count,
    unsigned history_count,unsigned epoch,float* factors,unsigned* offsets,
    unsigned* published,unsigned* fault,unsigned i) {
    if(i>=count_rows)return;
    published[i]=0u;
    const BallAnchorGpu row=records[i];
    const restoring_curve::CurveContext ctx{descriptors,nodes,bindings,history,descriptor_count,node_count,binding_count,history_count};
    unsigned code=0u;
    const auto* binding=restoring_curve::lookup(ctx,restoring_curve::RigidAnchor,i,3,code);
    if(code){fault[i]=code;return;}
    unsigned a=joint_fem::NONE,b=joint_fem::NONE;
    float p[3]={},q[3]={},d[3],e,g[3],h[9]={};
    const bool wa=row.body_a==joint_ball::WORLD_ENDPOINT,wb=row.body_b==joint_ball::WORLD_ENDPOINT;
    if(!binding) {
        if(!joint_ball::valid(row,bodies) || (!wa&&!joint_ball::reduced_offset(base,row.body_a,dim,&a)) || (!wb&&!joint_ball::reduced_offset(base,row.body_b,dim,&b))) {
            offsets[3u*i]=joint_fem::NONE;offsets[3u*i+1u]=joint_fem::NONE;offsets[3u*i+2u]=0u;published[i]=epoch;return;
        }
        PdrdRigidState sa{},sb{};
        if(!wa)sa=state[row.body_a];
        if(!wb)sb=state[row.body_b];
        if(wa)for(unsigned k=0;k<3;k++)p[k]=row.anchor_a[k]; else joint_ball::rot(sa.R,row.anchor_a,p);
        if(wb)for(unsigned k=0;k<3;k++)q[k]=row.anchor_b[k]; else joint_ball::rot(sb.R,row.anchor_b,q);
        h[0]=row.stiffness;
    } else {
        if(!restoring_ball_geometry(row,state,bodies,p,q,d) || !restoring_curve::anchor(ctx,restoring_curve::RigidAnchor,i,d,dt,e,g,h,code) ||
           (!wa&&!joint_ball::reduced_offset(base,row.body_a,dim,&a)) || (!wb&&!joint_ball::reduced_offset(base,row.body_b,dim,&b))) {
            fault[i]=code?code:restoring_curve::CurveFaultBinding;return;
        }
    }
    for(unsigned k=0;k<3;k++){factors[15u*i+k]=p[k];factors[15u*i+3u+k]=q[k];}
    for(unsigned k=0;k<9;k++)factors[15u*i+6u+k]=h[k];
    offsets[3u*i]=a;offsets[3u*i+1u]=b;offsets[3u*i+2u]=binding?1u:0u;published[i]=epoch;
}
[[seam::entry(i)]]
[[seam::device_fn]] inline void restoring_ball_frozen_apply(
    unsigned count_rows,const float* factors,const unsigned* offsets,
    const unsigned* published,unsigned epoch,const float* input,
    compute::atomic_float_t* output,unsigned* fault,unsigned i) {
    if(i>=count_rows)return;
    if(!epoch||published[i]!=epoch){fault[i]=joint_fem::BAD_STEP;return;}
    const unsigned a=offsets[3u*i],b=offsets[3u*i+1u],mode=offsets[3u*i+2u];
    if(a==joint_fem::NONE&&b==joint_fem::NONE)return;
    const float* f=factors+15u*i;
    float p[3],q[3],h[9],ua[6]={},ub[6]={},delta[3],w[3],z[6];
    for(unsigned k=0;k<3;k++){p[k]=f[k];q[k]=f[3u+k];}
    for(unsigned k=0;k<9;k++)h[k]=f[6u+k];
    if(a!=joint_fem::NONE){for(unsigned k=0;k<6;k++)ua[k]=input[a+k];}
    if(b!=joint_fem::NONE){for(unsigned k=0;k<6;k++)ub[k]=input[b+k];}
    joint_ball::L_pair(p,ua,q,ub,delta);
    if(mode){for(unsigned k=0;k<3;k++){w[k]=0;for(unsigned l=0;l<3;l++)w[k]+=h[3u*k+l]*delta[l];}}
    else for(unsigned k=0;k<3;k++)w[k]=delta[k];
    if(a!=joint_fem::NONE){joint_ball::lt(p,w,z);for(unsigned k=0;k<6;k++)compute::atomic_add(output+a+k,mode?z[k]:h[0]*z[k]);}
    if(b!=joint_fem::NONE){joint_ball::lt(q,w,z);for(unsigned k=0;k<6;k++)compute::atomic_add(output+b+k,mode?-z[k]:-h[0]*z[k]);}
}
