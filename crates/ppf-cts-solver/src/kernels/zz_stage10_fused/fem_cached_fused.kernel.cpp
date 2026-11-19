#pragma once
// Stage10: one indexed base+extension launch. Preserve the two frozen row
// evaluations and atomic contributions; only their dispatch is fused.
#include "../zzzzzzzzzzzzzzzzzz_fem_cached_indexed/fem_cached_indexed.kernel.cpp"

[[seam::entry(count,work_i)]] void joint_fem_cached_indexed_fused_rows(
    const FemJointGpu* records,unsigned record_count,const FemFrameGpu* cache,
    const unsigned* vertex_to_free,unsigned vertex_count,unsigned free_count,
    unsigned rigid_base,unsigned dim,const unsigned* index_cache,unsigned index_ready,
    const AngularSoftLimitGpu* limits,const AngularDriveGpu* drives,unsigned drive_count,
    float dt,unsigned token,const float* input,compute::atomic_float_t* output,
    unsigned* fault,const float* stopped,unsigned count,unsigned work_i);
[[seam::device_fn]] inline void joint_fem_cached_indexed_fused_rows(
    const FemJointGpu* records,unsigned record_count,const FemFrameGpu* cache,
    const unsigned* vertex_to_free,unsigned vertex_count,unsigned free_count,
    unsigned rigid_base,unsigned dim,const unsigned* index_cache,unsigned index_ready,
    const AngularSoftLimitGpu* limits,const AngularDriveGpu* drives,unsigned drive_count,
    float dt,unsigned token,const float* input,compute::atomic_float_t* output,
    unsigned* fault,const float* stopped,unsigned work_i) {
    if(*stopped!=0.0f)return;
    const unsigned i=work_i/24u,k=work_i%24u;
    if(i>=record_count)return;
    if(token==0u){if(k==0u)fault[i]=joint_fem::BAD_STEP;return;}
    const FemJointGpu row=records[i];if(!row.enabled)return;
    unsigned idx[24];bool duplicate=false;
    if(!joint_fem_indexed::get_mapping(row,vertex_to_free,vertex_count,free_count,
        rigid_base,dim,index_cache,i,token,index_ready,idx,duplicate)) {
        if(k==0u)fault[i]=joint_fem::BAD_MAPPING;return;
    }
    if(!joint_fem::hessian_matches(cache,record_count,i,token,dt)) {
        if(k==0u)fault[i]=joint_fem::BAD_STEP;return;
    }
    if(duplicate) {
        if(k==0u)joint_fem::hessian_matvec(cache,record_count,i,false,idx,input,output);
    } else {
        joint_fem_rows::apply_row(cache,record_count,i,false,k,idx,input,output);
    }
    if(!joint_angular_extension::row_requested(row.control)||
       !joint_angular_extension::active(row.control,limits[i],drives,drive_count,i))return;
    if(duplicate) {
        if(k==0u)joint_fem::hessian_matvec(cache,record_count,i,true,idx,input,output);
    } else {
        joint_fem_rows::apply_row(cache,record_count,i,true,k,idx,input,output);
    }
}
