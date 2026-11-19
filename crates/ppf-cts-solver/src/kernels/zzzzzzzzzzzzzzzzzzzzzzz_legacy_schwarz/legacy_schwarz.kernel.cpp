#pragma once
// Frozen legacy action1 coefficients only. Original global kernels are included,
// never edited; evaluation is read-only (nullptr observations, const history).
#include "../zzzzzzzzz_restoring/restoring_apply.kernel.cpp"
#include "../zz_joint/angular_joint_v2.kernel.cpp"
#include "../zz_joint_ext/angular_extension.kernel.cpp"

namespace legacy_schwarz {
enum : unsigned { NONE=0xffffffffu, BAD_MAPPING=1001u, BAD_STEP=1002u, BAD_ARITHMETIC=1003u };
[[seam::device_fn]] inline bool map(unsigned a,unsigned b,unsigned bodies,unsigned base,unsigned dim,unsigned ids[24]) {
    for(unsigned k=0;k<24;++k)ids[k]=NONE;
    if(base>dim || (dim-base)%6u || (dim-base)/6u!=bodies)return false;
    for(unsigned side=0;side<2;++side){
        const unsigned body=side?b:a;if(body==NONE)continue;
        if(body>=bodies)return false;
        const unsigned first=base+6u*body;
        for(unsigned k=0;k<6;++k)ids[6u*side+k]=first+k;
    }
    return true;
}
// The same alias coalescing contract as material export, with post-sum finite
// checks. WORLD/unused coordinates never enter either CSR or its domain graph.
[[seam::device_fn]] inline bool publish(const unsigned ids[24],const float h[576],unsigned dim,
    unsigned* output_ids,float* output_h,unsigned& count) {
    unsigned slots[24];count=0;
    for(unsigned k=0;k<24;++k){slots[k]=NONE;if(ids[k]==NONE)continue;if(ids[k]>=dim)return false;
        unsigned s=0;while(s<count&&output_ids[s]!=ids[k])++s;
        if(s==count)output_ids[count++]=ids[k];slots[k]=s;
    }
    for(unsigned i=0;i<24;++i)for(unsigned j=0;j<24;++j){
        const float v=h[24*i+j];if(!joint_ball::finite(v))return false;
        if(slots[i]!=NONE&&slots[j]!=NONE)output_h[24*slots[i]+slots[j]]+=v;
    }
    for(unsigned k=0;k<576;++k)if(!joint_ball::finite(output_h[k]))return false;
    return true;
}
// basis-to-matvec uses the exact original L/H/Lt order, including signs and
// translation/rotation slots. It does not substitute a different joint model.
[[seam::device_fn]] inline void ball_h(const float p[3],const float q[3],const float w[9],
    bool curved,float stiffness,float h[576]) {
    for(unsigned col=0;col<12;++col){
        float ua[6]={},ub[6]={},va[3],vb[3],z[3]={},ya[6],yb[6];
        if(col<6)ua[col]=1.f;else ub[col-6]=1.f;
        joint_ball::L(p,ua,va);joint_ball::L(q,ub,vb);
        if(curved){for(unsigned r=0;r<3;++r)for(unsigned k=0;k<3;++k)z[r]+=w[3*r+k]*(va[k]-vb[k]);}
        else for(unsigned r=0;r<3;++r)z[r]=va[r]-vb[r];
        joint_ball::lt(p,z,ya);joint_ball::lt(q,z,yb);
        for(unsigned r=0;r<6;++r){h[24*r+col]=curved?ya[r]:stiffness*ya[r];
            h[24*(r+6)+col]=curved?-yb[r]:-stiffness*yb[r];}
    }
}
[[seam::device_fn]] inline void control_h(const float j[18],const float w[3],float h[576]) {
    for(unsigned col=0;col<6;++col){
        float x[6]={},z[3]={};x[col]=1.f;
        for(unsigned r=0;r<3;++r){for(unsigned k=0;k<6;++k)z[r]+=j[6*r+k]*x[k];z[r]*=w[r];}
        const unsigned c=col<3?col+3:col+6;
        for(unsigned k=0;k<6;++k){float v=0.f;for(unsigned r=0;r<3;++r)v+=j[6*r+k]*z[r];
            const unsigned row=k<3?k+3:k+6;h[24*row+c]=v;}
    }
}
}

// Family-index preserving records: [ball count][isotropic count][directional
// control_count][extension extension_count]. Multiple records for an edge are
// additive CSR terms, as material records already are; none overwrite another.
[[seam::entry(i)]]
[[seam::device_fn]] inline void legacy_schwarz_export(
    const BallAnchorGpu* ball,const AngularJointGpuV2* angular,unsigned base_count,
    const AngularControlGpu* control,const AngularTwistEvalContext* twist_context,const MaterialSpinContext* material_spin_context,unsigned control_count,
    const AngularSoftLimitGpu* limits,unsigned extension_count,
    const AngularDriveGpu* drives,unsigned drive_count,const AngularDriveKeyGpu* keys,
    const PdrdRigidState* state,const PdrdRigidState* reference,
    unsigned bodies,unsigned base,unsigned dim,float dt,float sample_time_s,
    unsigned curves_enabled,const CurveDescriptor* descriptors,const CurveNode* nodes,
    const CurveBinding* bindings,const CurveHistory* history,
    unsigned descriptor_count,unsigned node_count,unsigned binding_count,unsigned history_count,
    unsigned* scalar_ids,float* local_hessian,unsigned* scalar_count,unsigned* fault,
    unsigned extent,unsigned i) {
    if(i>=extent)return;
    unsigned* output_ids=scalar_ids+24ull*i;float* output_h=local_hessian+576ull*i;
    for(unsigned k=0;k<24;++k)output_ids[k]=legacy_schwarz::NONE;
    for(unsigned k=0;k<576;++k)output_h[k]=0.f;
    scalar_count[i]=0;fault[i]=0;
    if(base_count>0xffffffffu/2u || control_count>0xffffffffu-2u*base_count ||
       extension_count>0xffffffffu-2u*base_count-control_count ||
       extent!=2u*base_count+control_count+extension_count || extent>0xffffffffu/576u || curves_enabled>1u){fault[i]=legacy_schwarz::BAD_MAPPING;return;}
    if(!(dt>0.f)||!joint_ball::finite(dt)||!joint_ball::finite(sample_time_s)){fault[i]=legacy_schwarz::BAD_STEP;return;}
    const restoring_curve::CurveContext ctx{descriptors,nodes,bindings,history,descriptor_count,node_count,binding_count,history_count};
    unsigned ids[24],code=0,a=legacy_schwarz::NONE,b=legacy_schwarz::NONE;
    float h[576]={};
    if(i<base_count){
        const BallAnchorGpu row=ball[i];a=row.body_a;b=row.body_b;
        const auto* binding=curves_enabled?restoring_curve::lookup(ctx,restoring_curve::RigidAnchor,i,3,code):nullptr;
        if(code){fault[i]=code;return;}
        float p[3],q[3],d[3],w[9]={};
        if(binding){
            float e,g[3];if(!restoring_ball_geometry(row,state,bodies,p,q,d)){fault[i]=legacy_schwarz::BAD_MAPPING;return;}
            if(!restoring_curve::anchor(ctx,restoring_curve::RigidAnchor,i,d,dt,e,g,w,code)){fault[i]=code;return;}
        }else{
            if(!joint_ball::valid(row,bodies))return;
            if(a==legacy_schwarz::NONE)for(unsigned k=0;k<3;++k)p[k]=row.anchor_a[k];else joint_ball::rot(state[a].R,row.anchor_a,p);
            if(b==legacy_schwarz::NONE)for(unsigned k=0;k<3;++k)q[k]=row.anchor_b[k];else joint_ball::rot(state[b].R,row.anchor_b,q);
            w[0]=w[4]=w[8]=row.stiffness;
        }
        legacy_schwarz::ball_h(p,q,w,binding!=nullptr,row.stiffness,h);
    }else if(i<2u*base_count){
        const AngularJointGpuV2 row=angular[i-base_count];a=row.body_a;b=row.body_b;
        if(!joint_angular::valid(row,bodies))return;
        float w[9];joint_angular::form_h(row,state,reference,joint_angular::ANGULAR_WORLD_ENDPOINT_V2,dt,w);
        // Original action1 is H*(omega_b-omega_a), scattered with -/+ signs.
        for(unsigned col=0;col<6;++col){float x[3]={},y[3];x[col%3]=col<3?-1.f:1.f;
            for(unsigned r=0;r<3;++r)y[r]=w[3*r]*x[0]+w[3*r+1]*x[1]+w[3*r+2]*x[2];
            const unsigned c=col<3?col+3:col+6;
            for(unsigned r=0;r<3;++r){h[24*(r+3)+c]=-y[r];h[24*(r+9)+c]=y[r];}}
    }else if(i<2u*base_count+control_count){
        const unsigned record=i-2u*base_count;const AngularControlGpu row=control[record];a=row.body_a;b=row.body_b;
        if(!joint_angular_control::row_requested(row))return;
        if(!curves_enabled&&!joint_angular_control::contributes(row))return;
        float c[3],c0[3],j[18],j0[18],g[3],w[3];
        const MaterialSpinContext material=material_spin_context[record];
        code=material.model==0u
            ? joint_angular_control::resolve(row,state,reference,bodies,c,j,c0,j0,twist_context+record)
            : joint_angular_control::resolve_material(row,state,bodies,material,c,j,c0);
        if(code){fault[i]=code;return;}
        if(curves_enabled){
            if(!joint_angular_control::curve_weights(row,ctx,restoring_curve::RigidAngular,record,c,c0,dt,g,w,nullptr,code,nullptr)){fault[i]=code;return;}
        }else joint_angular_control::weights(row,c,c0,dt,g,w,nullptr);
        legacy_schwarz::control_h(j,w,h);
    }else{
        const unsigned record=i-2u*base_count-control_count;const AngularControlGpu row=control[record];a=row.body_a;b=row.body_b;
        const AngularSoftLimitGpu limit=limits[record];
        const MaterialSpinContext material=material_spin_context[record];
        if(!joint_angular_extension::row_requested(row)||!joint_angular_extension::active(row,limit,drives,drive_count,record))return;
        float w[3],j[18],swing_yz=0.f;
        if(!joint_angular_extension::weights(row,limit,drives,drive_count,keys,record,sample_time_s,dt,state,reference,bodies,w,j,code,&swing_yz,twist_context+record,&material)){fault[i]=code;return;}
        float total=0.f;for(unsigned k=0;k<3;++k)total+=w[k];if(total==0.f)return;
        // Use extension's own matvec including its accumulation order.
        for(unsigned col=0;col<6;++col){float x[6]={},y[6];x[col]=1.f;joint_angular_extension::assemble_matvec(w,j,x,y,swing_yz);
            const unsigned c=col<3?col+3:col+6;
            for(unsigned k=0;k<6;++k){const unsigned r=k<3?k+3:k+6;h[24*r+c]=y[k];}}
    }
    if(!legacy_schwarz::map(a,b,bodies,base,dim,ids)){fault[i]=legacy_schwarz::BAD_MAPPING;return;}
    unsigned n=0;
    if(!legacy_schwarz::publish(ids,h,dim,output_ids,output_h,n)){
        for(unsigned k=0;k<24;++k)output_ids[k]=legacy_schwarz::NONE;
        for(unsigned k=0;k<576;++k)output_h[k]=0.f;
        fault[i]=legacy_schwarz::BAD_ARITHMETIC;return;
    }
    scalar_count[i]=n;
}

