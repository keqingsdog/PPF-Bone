#pragma once
// Stage5: only immutable cached action7; original all-action entries unchanged.
#include "../zzzzzzzz_fem_joint/fem_joint.kernel.cpp"

[[seam::entry(count,i)]] void joint_fem_cached_thin(const FemJointGpu* records,unsigned record_count,const FemFrameGpu* cache,const unsigned* vertex_to_free,unsigned vertex_count,unsigned free_count,unsigned rigid_base,unsigned dim,float dt,unsigned token,const float* input,compute::atomic_float_t* output,unsigned* fault,const float* stopped,unsigned count,unsigned i);
[[seam::device_fn]] inline void joint_fem_cached_thin(const FemJointGpu* records,unsigned record_count,const FemFrameGpu* cache,const unsigned* vertex_to_free,unsigned vertex_count,unsigned free_count,unsigned rigid_base,unsigned dim,float dt,unsigned token,const float* input,compute::atomic_float_t* output,unsigned* fault,const float* stopped,unsigned i) {
    if(*stopped!=0.0f)return;
    if(i>=record_count)return;
    if(token==0u){fault[i]=joint_fem::BAD_STEP;return;}
    const FemJointGpu row=records[i];if(!row.enabled)return;
    unsigned idx[24];if(!joint_fem::mapping(row.a,vertex_to_free,vertex_count,free_count,rigid_base,dim,idx)||!joint_fem::mapping(row.b,vertex_to_free,vertex_count,free_count,rigid_base,dim,idx+12)){fault[i]=joint_fem::BAD_MAPPING;return;}
    if(!joint_fem::hessian_matches(cache,record_count,i,token,dt)){fault[i]=joint_fem::BAD_STEP;return;}
    joint_fem::hessian_matvec(cache,record_count,i,false,idx,input,output);
}

[[seam::entry(count,i)]] void joint_fem_extension_cached_thin(const FemJointGpu* records,unsigned record_count,const FemFrameGpu* cache,const unsigned* vertex_to_free,unsigned vertex_count,unsigned free_count,unsigned rigid_base,unsigned dim,const AngularSoftLimitGpu* limits,const AngularDriveGpu* drives,unsigned drive_count,float dt,unsigned token,const float* input,compute::atomic_float_t* output,unsigned* fault,const float* stopped,unsigned count,unsigned i);
[[seam::device_fn]] inline void joint_fem_extension_cached_thin(const FemJointGpu* records,unsigned record_count,const FemFrameGpu* cache,const unsigned* vertex_to_free,unsigned vertex_count,unsigned free_count,unsigned rigid_base,unsigned dim,const AngularSoftLimitGpu* limits,const AngularDriveGpu* drives,unsigned drive_count,float dt,unsigned token,const float* input,compute::atomic_float_t* output,unsigned* fault,const float* stopped,unsigned i) {
    if(*stopped!=0.0f)return;
    if(i>=record_count)return;
    if(token==0u){fault[i]=joint_fem::BAD_STEP;return;}
    const FemJointGpu row=records[i];if(!row.enabled)return;
    unsigned idx[24];if(!joint_fem::mapping(row.a,vertex_to_free,vertex_count,free_count,rigid_base,dim,idx)||!joint_fem::mapping(row.b,vertex_to_free,vertex_count,free_count,rigid_base,dim,idx+12)){fault[i]=joint_fem::BAD_MAPPING;return;}
    if(!joint_fem::hessian_matches(cache,record_count,i,token,dt)){fault[i]=joint_fem::BAD_STEP;return;}
    if(!joint_angular_extension::row_requested(row.control)||!joint_angular_extension::active(row.control,limits[i],drives,drive_count,i))return;
    joint_fem::hessian_matvec(cache,record_count,i,true,idx,input,output);
}
