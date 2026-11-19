// Host-only adapter. C's kernelgen host translation unit supplies compute,
// fmath and all rendered includes. Never compile neutral sources as a second
// implementation: these calls are the same helpers used by solver assembly.
#include "zzzzzzzz_fem_joint/fem_joint.kernel.cpp"
#include "zz_joint/ball_anchor.kernel.cpp"
#include "energy/model/pdrd_lock_projector.kernel.cpp"
#include <vector>
#include <cstring>

extern "C" int restoring_mass_rest_states(const void* vertex_properties,
    const void* body_properties,const void* positions,unsigned vertex_count,
    const unsigned* vertex_list,const void* rest_centered,unsigned member_count,
    unsigned body_count,void* output_states) {
    if((vertex_count&&(!vertex_properties||!positions))||
       (body_count&&(!body_properties||!output_states))||
       (member_count&&(!vertex_list||!rest_centered)))return 1;
    const auto* vp=static_cast<const VertexProp*>(vertex_properties);
    const auto* bp=static_cast<const PdrdBodyProp*>(body_properties);
    const auto* x=static_cast<const Vec3f*>(positions);
    const auto* rest=static_cast<const Vec3f*>(rest_centered);
    for(unsigned b=0;b<body_count;++b){
        if(!bp[b].vertex_count||bp[b].vertex_start>member_count||
           bp[b].vertex_count>member_count-bp[b].vertex_start)return 2;
        for(unsigned k=0;k<bp[b].vertex_count;++k){
            const unsigned v=vertex_list[bp[b].vertex_start+k];
            if(v>=vertex_count||vp[v].pdrd_body_index!=b+1)return 3;
        }
    }
    for(unsigned j=0;j<member_count;++j)if(vertex_list[j]>=vertex_count||
        !vp[vertex_list[j]].pdrd_body_index||vp[vertex_list[j]].pdrd_body_index>body_count)return 3;
    if(body_count>0xffffffffu/12u)return 4;
    std::vector<compute::atomic_float_t> accum(12u*body_count);
    std::vector<float> image(12u*body_count);
    // Host float slots deliberately expose only atomic_add. The seam permits
    // arena byte initialization/readback; do not invent float store/load APIs.
    std::memset(accum.data(),0,accum.size()*sizeof(compute::atomic_float_t));
    for(unsigned j=0;j<member_count;++j)
        pdrd_fit_centroid_row(vertex_list,vp,bp,x,accum.data(),j);
    std::memcpy(image.data(),accum.data(),image.size()*sizeof(float));
    for(unsigned j=0;j<member_count;++j)
        pdrd_fit_covariance_row(vertex_list,vp,bp,x,rest,image.data(),accum.data(),j);
    std::memcpy(image.data(),accum.data(),image.size()*sizeof(float));
    auto* out=static_cast<PdrdRigidState*>(output_states);
    for(unsigned b=0;b<body_count;++b){
        pdrd_fit_finish_row(vertex_list,bp,x,image.data(),out,b);
        for(unsigned k=0;k<9;++k)if(!joint_fem::finite(out[b].R[k])||!joint_fem::finite(out[b].Iref[k]))return 5;
        for(unsigned k=0;k<3;++k)if(!joint_fem::finite(out[b].anchor[k])||!joint_fem::finite(out[b].x[k]))return 5;
    }
    return 0;
}

extern "C" int restoring_mass_body_basis(unsigned joint_mode,const float* joint_axis,
    unsigned has_translation,const float* translation_axis,unsigned translation_mode,
    unsigned has_rotation,const float* rotation_axis,unsigned rotation_mode,
    const float* fixed_relative_xyz,unsigned fixed_count,float* basis36,unsigned* count) {
    if(!joint_axis||!translation_axis||!rotation_axis||!basis36||!count||
       (fixed_count&&!fixed_relative_xyz))return 1;
    unsigned status=PDRD_LOCK_OK;
    for(unsigned k=0;k<36;++k)basis36[k]=0.f;
    unsigned n=pdrd_build_body_basis(joint_mode,Vec3f(joint_axis[0],joint_axis[1],joint_axis[2]),
        has_translation!=0u,Vec3f(translation_axis[0],translation_axis[1],translation_axis[2]),translation_mode,
        has_rotation!=0u,Vec3f(rotation_axis[0],rotation_axis[1],rotation_axis[2]),rotation_mode,basis36,&status);
    for(unsigned v=0;v<fixed_count;++v)for(unsigned axis=0;axis<3;++axis) {
        float unit[3]={},row[6];unit[axis]=1.f;
        joint_ball::lt(fixed_relative_xyz+3*v,unit,row);
        n=pdrd_append_constraint(basis36,n,row,&status);
    }
    if(status!=PDRD_LOCK_OK)return static_cast<int>(status);
    *count=n;return 0;
}

extern "C" int restoring_mass_joint_jacobian(
    unsigned family, unsigned channel, const void* record,
    const float* xyz, unsigned vertex_count, const void* states,
    unsigned body_count, float* jacobian72) {
    if (!record || !jacobian72 || (vertex_count && !xyz) ||
        (body_count && !states) || family > 3u || channel > 3u)
        return 1;
    const auto* rigid = static_cast<const PdrdRigidState*>(states);
    float out[72] = {};
    if (family == 0u) {
        if (channel >= 3u) return 2;
        const auto& row = *static_cast<const AngularControlGpu*>(record);
        if ((row.body_a != 0xffffffffu && row.body_a >= body_count) ||
            (row.body_b != 0xffffffffu && row.body_b >= body_count)) return 3;
        float ra[9], rb[9], coordinates[3], j[18];
        joint_fem::identity(ra); joint_fem::identity(rb);
        if (row.body_a != 0xffffffffu)
            joint_angular_control::rigid_rowmajor(rigid,row.body_a,ra);
        if (row.body_b != 0xffffffffu)
            joint_angular_control::rigid_rowmajor(rigid,row.body_b,rb);
        if (!joint_angular_control::chart(row,ra,rb,coordinates,j)) return 4;
        for (unsigned k=0;k<3;++k) {
            out[3+k]=j[6*channel+k];
            out[15+k]=j[6*channel+3+k];
        }
    } else if (family == 2u) {
        if (channel != 3u) return 2;
        const auto& row = *static_cast<const BallAnchorGpu*>(record);
        if ((row.body_a != 0xffffffffu && row.body_a >= body_count) ||
            (row.body_b != 0xffffffffu && row.body_b >= body_count)) return 3;
        float p[3]={},q[3]={};
        if (row.body_a!=0xffffffffu) joint_ball::rot(rigid[row.body_a].R,row.anchor_a,p);
        if (row.body_b!=0xffffffffu) joint_ball::rot(rigid[row.body_b].R,row.anchor_b,q);
        for (unsigned col=0;col<6;++col) {
            float basis[6]={},a[3],b[3]; basis[col]=1.f;
            joint_ball::L(p,basis,a);joint_ball::L(q,basis,b);
            for (unsigned r=0;r<3;++r) {
                if(row.body_a!=0xffffffffu) out[24*r+col]=a[r];
                if(row.body_b!=0xffffffffu) out[24*r+12+col]=-b[r];
            }
        }
    } else {
        const auto& row=*static_cast<const FemJointGpu*>(record);
        FemFrameGpu a{},b{};
        if(joint_fem::frame(row.a,xyz,vertex_count,rigid,body_count,a)!=joint_fem::OK ||
           joint_fem::frame(row.b,xyz,vertex_count,rigid,body_count,b)!=joint_fem::OK) return 5;
        if(family==3u) {
            if(channel!=3u) return 2;
            float aj[36],bj[36];
            joint_fem::anchor_jacobian(row.a,a,aj);
            joint_fem::anchor_jacobian(row.b,b,bj);
            for(unsigned r=0;r<3;++r)for(unsigned k=0;k<12;++k) {
                out[24*r+k]=aj[12*r+k];out[24*r+12+k]=-bj[12*r+k];
            }
        } else {
            if(channel>=3u) return 2;
            AngularControlGpu control=row.control;
            control.body_a=row.a.kind==joint_fem::WORLD?0xffffffffu:0u;
            control.body_b=row.b.kind==joint_fem::WORLD?0xffffffffu:1u;
            float ra[9],rb[9],coordinates[3],j[18];
            for(unsigned r=0;r<3;++r)for(unsigned c=0;c<3;++c) {
                ra[3*r+c]=a.rotation[3*c+r];rb[3*r+c]=b.rotation[3*c+r];
            }
            if(!joint_angular_control::chart(control,ra,rb,coordinates,j)) return 4;
            for(unsigned k=0;k<12;++k)for(unsigned r=0;r<3;++r) {
                out[k]+=j[6*channel+r]*a.rotation_jacobian[12*r+k];
                out[12+k]+=j[6*channel+3+r]*b.rotation_jacobian[12*r+k];
            }
        }
    }
    for(unsigned k=0;k<72;++k)if(!joint_fem::finite(out[k]))return 6;
    for(unsigned k=0;k<72;++k)jacobian72[k]=out[k];
    return 0;
}

// Independent coordinate-image entry for finite differences in a test-only
// harness. Calls values of the actual frame/chart; never derives from returned J.
extern "C" int restoring_mass_joint_coordinates(unsigned family,unsigned channel,
    const void* record,const float* xyz,unsigned vertex_count,const void* states,
    unsigned body_count,float* coordinates3) {
    if(!record||!coordinates3||family>3u||channel>3u||
       (vertex_count&&!xyz)||(body_count&&!states))return 1;
    const auto* rigid=static_cast<const PdrdRigidState*>(states);
    float result[3]={};
    if(family==0u){
        if(channel>=3u)return 2;
        const auto& row=*static_cast<const AngularControlGpu*>(record);
        if((row.body_a!=0xffffffffu&&row.body_a>=body_count)||
           (row.body_b!=0xffffffffu&&row.body_b>=body_count))return 3;
        float ra[9],rb[9],c[3],j[18];joint_fem::identity(ra);joint_fem::identity(rb);
        if(row.body_a!=0xffffffffu)joint_angular_control::rigid_rowmajor(rigid,row.body_a,ra);
        if(row.body_b!=0xffffffffu)joint_angular_control::rigid_rowmajor(rigid,row.body_b,rb);
        if(!joint_angular_control::chart(row,ra,rb,c,j))return 4;result[0]=c[channel];
    }else if(family==2u){
        if(channel!=3u)return 2;
        const auto& row=*static_cast<const BallAnchorGpu*>(record);
        if((row.body_a!=0xffffffffu&&row.body_a>=body_count)||
           (row.body_b!=0xffffffffu&&row.body_b>=body_count))return 3;
        float p[3]={},q[3]={};if(row.body_a!=0xffffffffu)joint_ball::rot(rigid[row.body_a].R,row.anchor_a,p);
        if(row.body_b!=0xffffffffu)joint_ball::rot(rigid[row.body_b].R,row.anchor_b,q);
        for(unsigned k=0;k<3;++k){const float a=row.body_a==0xffffffffu?row.anchor_a[k]:rigid[row.body_a].anchor[k]+rigid[row.body_a].x[k]+p[k];const float b=row.body_b==0xffffffffu?row.anchor_b[k]:rigid[row.body_b].anchor[k]+rigid[row.body_b].x[k]+q[k];result[k]=a-b;}
    }else{
        const auto& row=*static_cast<const FemJointGpu*>(record);FemFrameGpu a{},b{};
        if(joint_fem::frame(row.a,xyz,vertex_count,rigid,body_count,a)!=joint_fem::OK||
           joint_fem::frame(row.b,xyz,vertex_count,rigid,body_count,b)!=joint_fem::OK)return 5;
        if(family==3u){if(channel!=3u)return 2;for(unsigned k=0;k<3;++k)result[k]=a.anchor[k]-b.anchor[k];}
        else{if(channel>=3u)return 2;AngularControlGpu control=row.control;control.body_a=row.a.kind==joint_fem::WORLD?0xffffffffu:0u;control.body_b=row.b.kind==joint_fem::WORLD?0xffffffffu:1u;
            float ra[9],rb[9],c[3],j[18];for(unsigned r=0;r<3;++r)for(unsigned k=0;k<3;++k){ra[3*r+k]=a.rotation[3*k+r];rb[3*r+k]=b.rotation[3*k+r];}if(!joint_angular_control::chart(control,ra,rb,c,j))return 4;result[0]=c[channel];}
    }
    for(unsigned k=0;k<3;++k)if(!joint_fem::finite(result[k]))return 6;
    for(unsigned k=0;k<3;++k)coordinates3[k]=result[k];return 0;
}
