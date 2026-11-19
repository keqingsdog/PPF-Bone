#pragma once
// Stage8: freeze one private mapping per record. Keep the Stage7 row math.
#include "../zzzzzzzzzzzzzzzzz_fem_cached_rows/fem_cached_rows.kernel.cpp"

namespace joint_fem_indexed {
enum : unsigned { STRIDE=27u, VALID=24u, DUPLICATE=25u, EPOCH=26u };
[[seam::device_fn]] inline bool get_mapping(
    const FemJointGpu& row,const unsigned* vertex_to_free,
    unsigned vertex_count,unsigned free_count,unsigned rigid_base,unsigned dim,
    const unsigned* index_cache,unsigned record,unsigned token,
    unsigned index_ready,unsigned idx[24],bool& duplicate) {
    const unsigned base=STRIDE*record;
    if(index_ready!=0u && index_cache[base+EPOCH]==token) {
        if(index_cache[base+VALID]==0u)return false;
        for(unsigned k=0;k<24u;++k)idx[k]=index_cache[base+k];
        duplicate=index_cache[base+DUPLICATE]!=0u;
        return true;
    }
    // Unready/stale metadata takes the unchanged Stage7 runtime mapping path.
    if(!joint_fem::mapping(row.a,vertex_to_free,vertex_count,free_count,rigid_base,dim,idx)||
       !joint_fem::mapping(row.b,vertex_to_free,vertex_count,free_count,rigid_base,dim,idx+12))return false;
    duplicate=joint_fem_rows::repeated_destination(idx);
    return true;
}
}

[[seam::entry(count,i)]] void joint_fem_cached_index_precompute(
    const FemJointGpu* records,unsigned record_count,const unsigned* vertex_to_free,
    unsigned vertex_count,unsigned free_count,unsigned rigid_base,unsigned dim,
    unsigned token,unsigned* index_cache,unsigned count,unsigned i);
[[seam::device_fn]] inline void joint_fem_cached_index_precompute(
    const FemJointGpu* records,unsigned record_count,const unsigned* vertex_to_free,
    unsigned vertex_count,unsigned free_count,unsigned rigid_base,unsigned dim,
    unsigned token,unsigned* index_cache,unsigned i) {
    if(i>=record_count)return;
    const FemJointGpu row=records[i];
    unsigned idx[24];bool valid=false,duplicate=false;
    if(row.enabled) {
        valid=joint_fem::mapping(row.a,vertex_to_free,vertex_count,free_count,rigid_base,dim,idx)&&
              joint_fem::mapping(row.b,vertex_to_free,vertex_count,free_count,rigid_base,dim,idx+12);
        if(valid)duplicate=joint_fem_rows::repeated_destination(idx);
    }
    const unsigned base=joint_fem_indexed::STRIDE*i;
    for(unsigned k=0;k<24u;++k)index_cache[base+k]=valid?idx[k]:joint_fem::NONE;
    index_cache[base+joint_fem_indexed::VALID]=valid?1u:0u;
    index_cache[base+joint_fem_indexed::DUPLICATE]=duplicate?1u:0u;
    // Published last; host waits for the one-shot launch before map_ready=true.
    index_cache[base+joint_fem_indexed::EPOCH]=token;
}

[[seam::entry(count,work_i)]] void joint_fem_cached_indexed_rows(
    const FemJointGpu* records,unsigned record_count,const FemFrameGpu* cache,
    const unsigned* vertex_to_free,unsigned vertex_count,unsigned free_count,
    unsigned rigid_base,unsigned dim,const unsigned* index_cache,unsigned index_ready,
    float dt,unsigned token,const float* input,compute::atomic_float_t* output,
    unsigned* fault,const float* stopped,unsigned count,unsigned work_i);
[[seam::device_fn]] inline void joint_fem_cached_indexed_rows(
    const FemJointGpu* records,unsigned record_count,const FemFrameGpu* cache,
    const unsigned* vertex_to_free,unsigned vertex_count,unsigned free_count,
    unsigned rigid_base,unsigned dim,const unsigned* index_cache,unsigned index_ready,
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
    if(duplicate){
        if(k==0u)joint_fem::hessian_matvec(cache,record_count,i,false,idx,input,output);
        return;
    }
    joint_fem_rows::apply_row(cache,record_count,i,false,k,idx,input,output);
}

[[seam::entry(count,work_i)]] void joint_fem_extension_cached_indexed_rows(
    const FemJointGpu* records,unsigned record_count,const FemFrameGpu* cache,
    const unsigned* vertex_to_free,unsigned vertex_count,unsigned free_count,
    unsigned rigid_base,unsigned dim,const unsigned* index_cache,unsigned index_ready,
    const AngularSoftLimitGpu* limits,const AngularDriveGpu* drives,unsigned drive_count,
    float dt,unsigned token,const float* input,compute::atomic_float_t* output,
    unsigned* fault,const float* stopped,unsigned count,unsigned work_i);
[[seam::device_fn]] inline void joint_fem_extension_cached_indexed_rows(
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
    if(!joint_angular_extension::row_requested(row.control)||
       !joint_angular_extension::active(row.control,limits[i],drives,drive_count,i))return;
    if(duplicate){
        if(k==0u)joint_fem::hessian_matvec(cache,record_count,i,true,idx,input,output);
        return;
    }
    joint_fem_rows::apply_row(cache,record_count,i,true,k,idx,input,output);
}
