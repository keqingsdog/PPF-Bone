#pragma once
// Stage7: one original Hessian output row per element lane. The Stage5 thin
// entries stay registered for fallback and direct numerical comparison.
#include "../zzzzzzzz_fem_joint/fem_joint.kernel.cpp"

namespace joint_fem_rows {
[[seam::device_fn]] inline bool repeated_destination(const unsigned idx[24]) {
    // Every lane evaluates the identical predicate; no group barrier/broadcast.
    for (unsigned a=0; a<24u; ++a)
        if (idx[a]!=joint_fem::NONE)
            for (unsigned b=a+1u; b<24u; ++b)
                if (idx[a]==idx[b]) return true;
    return false;
}

[[seam::device_fn]] inline void apply_row(const FemFrameGpu* cache,
    unsigned record_count, unsigned i, bool extension, unsigned k,
    const unsigned idx[24], const float* input,
    compute::atomic_float_t* output) {
    if (idx[k]==joint_fem::NONE) return;
    const unsigned char* h=joint_fem::hessian_row(cache,record_count,i,extension);
    float value=0.f;
    // Keep the old row's l=0..23 += chain and its one atomic destination.
    for (int l=0; l<24; ++l)
        if (idx[l]!=joint_fem::NONE)
            value+=joint_fem::hessian_read_float(h+4*(24*k+l))*input[idx[l]];
    compute::atomic_add(output+idx[k],value);
}
}

[[seam::entry(count,work_i)]] void joint_fem_cached_rows(
    const FemJointGpu* records,unsigned record_count,const FemFrameGpu* cache,
    const unsigned* vertex_to_free,unsigned vertex_count,unsigned free_count,
    unsigned rigid_base,unsigned dim,float dt,unsigned token,
    const float* input,compute::atomic_float_t* output,unsigned* fault,
    const float* stopped,unsigned count,unsigned work_i);
[[seam::device_fn]] inline void joint_fem_cached_rows(
    const FemJointGpu* records,unsigned record_count,const FemFrameGpu* cache,
    const unsigned* vertex_to_free,unsigned vertex_count,unsigned free_count,
    unsigned rigid_base,unsigned dim,float dt,unsigned token,
    const float* input,compute::atomic_float_t* output,unsigned* fault,
    const float* stopped,unsigned work_i) {
    if (*stopped!=0.0f) return;
    const unsigned i=work_i/24u, k=work_i%24u;
    if (i>=record_count) return;
    if (token==0u) {if(k==0u) fault[i]=joint_fem::BAD_STEP;return;}
    const FemJointGpu row=records[i];if(!row.enabled)return;
    unsigned idx[24];
    if(!joint_fem::mapping(row.a,vertex_to_free,vertex_count,free_count,rigid_base,dim,idx)||
       !joint_fem::mapping(row.b,vertex_to_free,vertex_count,free_count,rigid_base,dim,idx+12)) {
        if(k==0u) fault[i]=joint_fem::BAD_MAPPING;return;
    }
    if(!joint_fem::hessian_matches(cache,record_count,i,token,dt)) {
        if(k==0u) fault[i]=joint_fem::BAD_STEP;return;
    }
    if(joint_fem_rows::repeated_destination(idx)) {
        if(k==0u) joint_fem::hessian_matvec(cache,record_count,i,false,idx,input,output);
        return;
    }
    joint_fem_rows::apply_row(cache,record_count,i,false,k,idx,input,output);
}

[[seam::entry(count,work_i)]] void joint_fem_extension_cached_rows(
    const FemJointGpu* records,unsigned record_count,const FemFrameGpu* cache,
    const unsigned* vertex_to_free,unsigned vertex_count,unsigned free_count,
    unsigned rigid_base,unsigned dim,const AngularSoftLimitGpu* limits,
    const AngularDriveGpu* drives,unsigned drive_count,float dt,unsigned token,
    const float* input,compute::atomic_float_t* output,unsigned* fault,
    const float* stopped,unsigned count,unsigned work_i);
[[seam::device_fn]] inline void joint_fem_extension_cached_rows(
    const FemJointGpu* records,unsigned record_count,const FemFrameGpu* cache,
    const unsigned* vertex_to_free,unsigned vertex_count,unsigned free_count,
    unsigned rigid_base,unsigned dim,const AngularSoftLimitGpu* limits,
    const AngularDriveGpu* drives,unsigned drive_count,float dt,unsigned token,
    const float* input,compute::atomic_float_t* output,unsigned* fault,
    const float* stopped,unsigned work_i) {
    if (*stopped!=0.0f) return;
    const unsigned i=work_i/24u, k=work_i%24u;
    if (i>=record_count) return;
    if (token==0u) {if(k==0u) fault[i]=joint_fem::BAD_STEP;return;}
    const FemJointGpu row=records[i];if(!row.enabled)return;
    unsigned idx[24];
    if(!joint_fem::mapping(row.a,vertex_to_free,vertex_count,free_count,rigid_base,dim,idx)||
       !joint_fem::mapping(row.b,vertex_to_free,vertex_count,free_count,rigid_base,dim,idx+12)) {
        if(k==0u) fault[i]=joint_fem::BAD_MAPPING;return;
    }
    if(!joint_fem::hessian_matches(cache,record_count,i,token,dt)) {
        if(k==0u) fault[i]=joint_fem::BAD_STEP;return;
    }
    if(!joint_angular_extension::row_requested(row.control)||
       !joint_angular_extension::active(row.control,limits[i],drives,drive_count,i))return;
    if(joint_fem_rows::repeated_destination(idx)) {
        if(k==0u) joint_fem::hessian_matvec(cache,record_count,i,true,idx,input,output);
        return;
    }
    joint_fem_rows::apply_row(cache,record_count,i,true,k,idx,input,output);
}
