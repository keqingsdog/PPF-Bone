#pragma once
// Stage10 export candidate: original matvec source/scheduling unchanged.
#include "../zzzzzzzzzzzzzzzzzz_fem_cached_indexed/fem_cached_indexed.kernel.cpp"
namespace reduced_schwarz {
// Repeated support vertices, shared endpoint vertices and same-body endpoints
// are coalesced BEFORE CSR emission. WORLD/padded coordinates contribute none.
[[seam::host_device_fn]] inline unsigned coalesce(
    const unsigned idx[24], const float h[576], unsigned unique[24], float out[576]) {
    unsigned n=0, slot[24];
    for(unsigned k=0;k<24;++k){unique[k]=joint_fem::NONE;slot[k]=joint_fem::NONE;}
    for(unsigned k=0;k<576;++k)out[k]=0.f;
    for(unsigned k=0;k<24;++k)if(idx[k]!=joint_fem::NONE){
        unsigned s=0;while(s<n&&unique[s]!=idx[k])++s;
        if(s==n)unique[n++]=idx[k];slot[k]=s;
    }
    for(unsigned k=0;k<24;++k)if(slot[k]!=joint_fem::NONE)
        for(unsigned l=0;l<24;++l)if(slot[l]!=joint_fem::NONE)
            out[24*slot[k]+slot[l]]+=h[24*k+l];
    return n;
}

// Finite local entries can overflow when alias occurrences are added. Publish
// neither ids nor nonfinite values unless the COALESCED block passes bounds
// and arithmetic checks. Failed exports remain the empty zero block.
[[seam::host_device_fn]] inline bool checked_coalesce(
    const unsigned idx[24],const float h[576],unsigned dim,
    unsigned unique[24],float out[576],unsigned& count) {
    count=0;
    for(unsigned k=0;k<24;++k)unique[k]=joint_fem::NONE;
    for(unsigned k=0;k<576;++k)out[k]=0.f;
    for(unsigned k=0;k<24;++k)if(idx[k]!=joint_fem::NONE&&idx[k]>=dim)return false;
    for(unsigned k=0;k<576;++k)if(!joint_fem::finite(h[k]))return false;
    unsigned ids[24];float values[576];
    const unsigned n=coalesce(idx,h,ids,values);
    for(unsigned k=0;k<576;++k)if(!joint_fem::finite(values[k]))return false;
    for(unsigned k=0;k<24;++k)unique[k]=ids[k];
    for(unsigned k=0;k<576;++k)out[k]=values[k];
    count=n;return true;
}

// Kept separate from the existing matvec entry so no global operator code or
// existing kernel table entry changes. Every arithmetic operation is the
// existing extension-apply operation, at the identical frozen frames/time/dt.
[[seam::host_device_fn]] inline bool add_extension(
    const FemJointGpu& row,const FemFrameGpu* f,const AngularSoftLimitGpu& limit,
    const AngularDriveGpu* drives,unsigned drive_count,const AngularDriveKeyGpu* keys,
    unsigned record,float sample_time_s,float dt,float h[576],unsigned& fault,[[seam::thread]] const MaterialSpinContext* material) {
    AngularControlGpu control=row.control;
    control.body_a=row.a.kind==joint_fem::WORLD?joint_fem::NONE:0u;
    control.body_b=row.b.kind==joint_fem::WORLD?joint_fem::NONE:1u;
    PdrdRigidState cur[2],ref[2];joint_fem::pose_view(f[0],cur[0]);joint_fem::pose_view(f[1],cur[1]);
    joint_fem::pose_view(f[2],ref[0]);joint_fem::pose_view(f[3],ref[1]);
    float cg[3],c[3],j[18],ag[6]={},ah[36]={},g[24]={},swing_yz=0.f;
    joint_angular_extension::AxisContribution axis[3];
    if(!joint_angular_extension::evaluate(control,limit,drives,drive_count,keys,record,
        sample_time_s,dt,cur,ref,2,cg,c,j,axis,fault,&swing_yz,nullptr,material))return false;
    if(!joint_angular_extension::row_requested(control)||
       !joint_angular_extension::active(control,limit,drives,drive_count,record))return true;
    for(int k=0;k<6;++k){
        for(int r=0;r<3;++r)ag[k]+=j[6*r+k]*cg[r];
        for(int l=0;l<6;++l)for(int r=0;r<3;++r)
            ah[6*k+l]+=j[6*r+k]*joint_angular_extension::total_weight(axis[r])*j[6*r+l];
    }
    if(swing_yz!=0.f)for(int k=0;k<6;++k)for(int l=0;l<6;++l)ah[6*k+l]+=swing_yz*(j[6+k]*j[12+l]+j[12+k]*j[6+l]);
    joint_fem::add_angle(ag,ah,f[0],f[1],g,h);
    for(int k=0;k<24;++k)if(!joint_fem::finite(g[k])){fault=joint_fem::BAD_ARITHMETIC;return false;}
    for(int k=0;k<576;++k)if(!joint_fem::finite(h[k])){fault=joint_fem::BAD_ARITHMETIC;return false;}
    return true;
}
} // namespace reduced_schwarz

// source_mode: 0=recompute complete current model, 1=Stage10 current cached
// model. A stale requested cache refuses; it never reads an old Hessian tail.
[[seam::entry(i)]]
[[seam::device_fn]] inline void reduced_schwarz_joint_export(
    const FemJointGpu* records,const MaterialSpinContext* material_spin_context,unsigned record_count,const FemFrameGpu* cache,
    const unsigned* vertex_to_free,unsigned vertex_count,unsigned free_count,
    unsigned rigid_base,unsigned dim,const unsigned* index_cache,unsigned index_ready,
    unsigned source_mode,unsigned token,float hessian_dt,
    const AngularSoftLimitGpu* limits,const AngularDriveGpu* drives,unsigned drive_count,
    const AngularDriveKeyGpu* keys,unsigned extension_count,float sample_time_s,float dt,
    unsigned curves_enabled,const CurveDescriptor* descriptors,const CurveNode* nodes,
    const CurveBinding* bindings,const CurveHistory* history,
    unsigned descriptor_count,unsigned node_count,unsigned binding_count,unsigned history_count,
    unsigned* scalar_ids,float* local_hessian,unsigned* scalar_count,unsigned* fault,unsigned i) {
    if(i>=record_count)return;
    unsigned* ids=scalar_ids+24ull*i;float* values=local_hessian+576ull*i;
    for(unsigned k=0;k<24;++k)ids[k]=joint_fem::NONE;
    for(unsigned k=0;k<576;++k)values[k]=0.f;
    scalar_count[i]=0;fault[i]=0;
    if(record_count>0xffffffffu/576u||free_count>dim/3u||rigid_base!=3u*free_count||
       rigid_base>dim||(dim-rigid_base)%6u!=0u||source_mode>1u||curves_enabled>1u){
        fault[i]=joint_fem::BAD_MAPPING;return;
    }
    if(extension_count!=0u&&extension_count!=record_count){fault[i]=joint_fem::BAD_MAPPING;return;}
    const FemJointGpu row=records[i];if(!row.enabled)return;
    if(!(dt>0.f)||!joint_fem::finite(dt)){fault[i]=joint_fem::BAD_STEP;return;}
    unsigned idx[24];bool duplicate=false;
    // A recompute fallback uses original mapping, not old cached metadata from
    // a different frozen step. The cached route retains Stage10's epoch check
    // and its generic fallback for stale/unready mapping metadata.
    if(!joint_fem_indexed::get_mapping(row,vertex_to_free,vertex_count,free_count,
        rigid_base,dim,index_cache,i,token,source_mode==1u?index_ready:0u,idx,duplicate)){
        fault[i]=joint_fem::BAD_MAPPING;return;
    }
    float g[24],h[576],energy;unsigned code=0;
    if(source_mode==1u){
        // The current Stage10 tail is authoritative only on the original
        // no-body/no-curves cache domain. Never reinterpret a curves cache as
        // the base linear law. Kernel checks also defend stale host metadata.
        if(curves_enabled!=0u||dim!=rigid_base||dt!=hessian_dt||
           !joint_fem::hessian_matches(cache,record_count,i,token,dt)){
            fault[i]=joint_fem::BAD_STEP;return;
        }
        const unsigned char* base=joint_fem::hessian_row(cache,record_count,i,false);
        const unsigned char* ext=joint_fem::hessian_row(cache,record_count,i,true);
        for(unsigned k=0;k<576;++k){
            h[k]=joint_fem::hessian_read_float(base+4*k);
            if(extension_count!=0u)h[k]+=joint_fem::hessian_read_float(ext+4*k);
        }
    }else{
        const restoring_curve::CurveContext context{descriptors,nodes,bindings,history,
            descriptor_count,node_count,binding_count,history_count};
        const MaterialSpinContext material=material_spin_context[i];
        if(!joint_fem::evaluate(row,cache+4ull*i,dt,g,h,energy,code,
            curves_enabled?&context:nullptr,i,nullptr,nullptr,&material)){fault[i]=code;return;}
        // nullptr observations exactly matches global action1. No history or
        // restoring observations are mutated during this export.
        if(extension_count!=0u&&!reduced_schwarz::add_extension(row,cache+4ull*i,
            limits[i],drives,drive_count,keys,i,sample_time_s,dt,h,code,&material)){
            fault[i]=code;return;
        }
    }
    unsigned n=0;
    if(!reduced_schwarz::checked_coalesce(idx,h,dim,ids,values,n)){
        fault[i]=joint_fem::BAD_ARITHMETIC;return;
    }
    scalar_count[i]=n;
}
