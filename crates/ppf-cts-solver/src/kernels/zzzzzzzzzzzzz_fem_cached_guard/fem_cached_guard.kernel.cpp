#pragma once
// Stage3: stopped batches must not touch original action7 fault/output banks.
#include "../zzzzzzzz_fem_joint/fem_joint.kernel.cpp"

[[seam::entry(count,i)]] void joint_fem_cached_guarded(const FemJointGpu* records,const MaterialSpinContext* material_spin_context,unsigned record_count,const FemFrameGpu* cache,const unsigned* vertex_to_free,unsigned vertex_count,unsigned free_count,unsigned rigid_base,unsigned dim,float dt,unsigned action,const float* input,compute::atomic_float_t* output,compute::atomic_float_t* nodal_blocks,compute::atomic_float_t* rigid_blocks,unsigned* fault,const float* stopped,unsigned count,unsigned i);
[[seam::device_fn]] inline void joint_fem_cached_guarded(const FemJointGpu* records,const MaterialSpinContext* material_spin_context,unsigned record_count,const FemFrameGpu* cache,const unsigned* vertex_to_free,unsigned vertex_count,unsigned free_count,unsigned rigid_base,unsigned dim,float dt,unsigned action,const float* input,compute::atomic_float_t* output,compute::atomic_float_t* nodal_blocks,compute::atomic_float_t* rigid_blocks,unsigned* fault,const float* stopped,unsigned i) {
    if(*stopped!=0.0f)return;
    joint_fem_apply(records,material_spin_context,record_count,cache,vertex_to_free,vertex_count,free_count,rigid_base,dim,dt,action,input,output,nodal_blocks,rigid_blocks,fault,i);
}

[[seam::entry(count,i)]] void joint_fem_extension_cached_guarded(const FemJointGpu* records,const MaterialSpinContext* material_spin_context,unsigned record_count,const FemFrameGpu* cache,const unsigned* vertex_to_free,unsigned vertex_count,unsigned free_count,unsigned rigid_base,unsigned dim,const AngularSoftLimitGpu* limits,const AngularDriveGpu* drives,unsigned drive_count,const AngularDriveKeyGpu* keys,float sample_time_s,float dt,unsigned action,const float* input,compute::atomic_float_t* output,compute::atomic_float_t* nodal_blocks,compute::atomic_float_t* rigid_blocks,unsigned* fault,const float* stopped,unsigned count,unsigned i);
[[seam::device_fn]] inline void joint_fem_extension_cached_guarded(const FemJointGpu* records,const MaterialSpinContext* material_spin_context,unsigned record_count,const FemFrameGpu* cache,const unsigned* vertex_to_free,unsigned vertex_count,unsigned free_count,unsigned rigid_base,unsigned dim,const AngularSoftLimitGpu* limits,const AngularDriveGpu* drives,unsigned drive_count,const AngularDriveKeyGpu* keys,float sample_time_s,float dt,unsigned action,const float* input,compute::atomic_float_t* output,compute::atomic_float_t* nodal_blocks,compute::atomic_float_t* rigid_blocks,unsigned* fault,const float* stopped,unsigned i) {
    if(*stopped!=0.0f)return;
    joint_fem_extension_apply(records,material_spin_context,record_count,cache,vertex_to_free,vertex_count,free_count,rigid_base,dim,limits,drives,drive_count,keys,sample_time_s,dt,action,input,output,nodal_blocks,rigid_blocks,fault,i);
}
