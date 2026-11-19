#pragma once
#include "../zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz_restoring_cache/restoring_fem_frozen.kernel.cpp"

// One output row per lane; aliasing remains an atomic scatter. A stopped batch
// performs no reads or writes, including fault writes.
[[seam::entry(work_i)]]
[[seam::device_fn]] inline void restoring_fem_frozen_row(
    unsigned record_count,const float* cached_hessian,const unsigned* cached_index,
    const unsigned* published_epoch,unsigned epoch,const float* input,
    compute::atomic_float_t* output,unsigned* fault,const float* stopped,
    unsigned use_stopped,unsigned work_i,unsigned extent) {
    if(use_stopped!=0u && *stopped!=0.0f)return;
    if(work_i>=extent)return;
    const unsigned i=work_i/24u,k=work_i%24u;
    if(i>=record_count)return;
    if(record_count>0xffffffffu/24u || extent!=record_count*24u){
        if(k==0u)fault[i]=joint_fem::BAD_STEP;return;
    }
    if(epoch==0u || published_epoch[i]!=epoch || record_count>0xffffffffu/576u){
        if(k==0u)fault[i]=joint_fem::BAD_STEP;return;
    }
    const unsigned* ids=cached_index+24u*i;
    if(ids[k]==joint_fem::NONE)return;
    const float* h=cached_hessian+576u*i+24u*k;float value=0.f;
    for(unsigned l=0;l<24u;++l)if(ids[l]!=joint_fem::NONE)value+=h[l]*input[ids[l]];
    compute::atomic_add(output+ids[k],value);
}

[[seam::entry(work_i)]]
[[seam::device_fn]] inline void joint_fem_extension_frozen_row(
    const FemJointGpu* records,unsigned record_count,const FemFrameGpu* cache,
    const unsigned* vertex_to_free,unsigned vertex_count,unsigned free_count,
    unsigned rigid_base,unsigned dim,const AngularSoftLimitGpu* limits,
    const AngularDriveGpu* drives,unsigned drive_count,float dt,unsigned token,
    const float* input,compute::atomic_float_t* output,unsigned* fault,
    const float* stopped,unsigned use_stopped,unsigned work_i,unsigned extent) {
    if(use_stopped!=0u && *stopped!=0.0f)return;
    if(work_i>=extent)return;
    const unsigned i=work_i/24u,k=work_i%24u;
    if(i>=record_count)return;
    if(record_count>0xffffffffu/24u || extent!=record_count*24u){
        if(k==0u)fault[i]=joint_fem::BAD_STEP;return;
    }
    if(token==0u){if(k==0u)fault[i]=joint_fem::BAD_STEP;return;}
    const FemJointGpu row=records[i];if(!row.enabled)return;
    unsigned idx[24];
    if(!joint_fem::mapping(row.a,vertex_to_free,vertex_count,free_count,rigid_base,dim,idx) ||
       !joint_fem::mapping(row.b,vertex_to_free,vertex_count,free_count,rigid_base,dim,idx+12)) {
        if(k==0u)fault[i]=joint_fem::BAD_MAPPING;return;
    }
    if(!joint_fem::hessian_matches(cache,record_count,i,token,dt)){
        if(k==0u)fault[i]=joint_fem::BAD_STEP;return;
    }
    if(!joint_angular_extension::row_requested(row.control)||
       !joint_angular_extension::active(row.control,limits[i],drives,drive_count,i))return;
    if(idx[k]==joint_fem::NONE)return;
    const unsigned char* h=joint_fem::hessian_row(cache,record_count,i,true);
    float value=0.f;
    for(unsigned l=0;l<24u;++l)if(idx[l]!=joint_fem::NONE)
        value+=joint_fem::hessian_read_float(h+4u*(24u*k+l))*input[idx[l]];
    compute::atomic_add(output+idx[k],value);
}