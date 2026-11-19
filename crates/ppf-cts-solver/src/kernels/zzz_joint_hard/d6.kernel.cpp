// Hard row geometry, ported from old joint_d6_geometry_v1.hpp. No penalty.
#pragma once
#include "../zz_joint_ext/angular_control.kernel.cpp"

struct HardJointGpu {
unsigned joint_index;
unsigned body_a;
unsigned body_b;
unsigned enabled;
float anchor_a[3];
float anchor_b[3];
float frame_a_wxyz[4];
float frame_b_wxyz[4];
float rest_relative_wxyz[4];
float rest_translation_m[3];
unsigned modes[6];
float lower[6];
float upper[6];
};
struct HardGeometryGpu {
unsigned joint_index;
unsigned body_a;
unsigned body_b;
unsigned fault;
float coordinate[6];
float jacobian[72];
};
namespace joint_hard {
enum : unsigned { OK=0u, BODY=1u, FRAME=2u, CHART=3u, AXIS=4u, RANGE=5u, HISTORY=6u };
[[seam::host_device_fn]] inline bool finite(float x) {return joint_angular_control::finite(x);}
[[seam::host_device_fn]] inline float dot(const float a[3],const float b[3]) {return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
[[seam::host_device_fn]] inline void cross(const float a[3],const float b[3],float o[3]) {o[0]=a[1]*b[2]-a[2]*b[1];o[1]=a[2]*b[0]-a[0]*b[2];o[2]=a[0]*b[1]-a[1]*b[0];}
[[seam::host_device_fn]] inline bool finite3(const float a[3]) {return finite(a[0])&&finite(a[1])&&finite(a[2]);}
[[seam::host_device_fn]] inline bool unit(const float q[4]) {float n=0.f;for(int k=0;k<4;++k){if(!finite(q[k]))return false;n+=q[k]*q[k];}return fmath::abs(fmath::sqrt(n)-1.f)<=1e-4f;}
[[seam::host_device_fn]] inline bool rotation(const float R[9]) {
    float c[3][3];for(int j=0;j<3;++j)for(int k=0;k<3;++k){c[j][k]=R[3*j+k];if(!finite(c[j][k]))return false;}
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)if(fmath::abs(dot(c[i],c[j])-(i==j?1.f:0.f))>1e-3f)return false;
    float v[3];cross(c[0],c[1],v);return fmath::abs(dot(v,c[2])-1.f)<=1e-3f;
}
[[seam::host_device_fn]] inline void rot(const float R[9],const float x[3],float o[3]) {for(int i=0;i<3;++i)o[i]=R[i]*x[0]+R[3+i]*x[1]+R[6+i]*x[2];}
[[seam::host_device_fn]] inline void transpose(const float R[9],const float x[3],float o[3]) {for(int i=0;i<3;++i)o[i]=R[3*i]*x[0]+R[3*i+1]*x[1]+R[3*i+2]*x[2];}
[[seam::host_device_fn]] inline bool angular_needed(const HardJointGpu& j) {return j.modes[3]!=0u||j.modes[4]!=0u||j.modes[5]!=0u;}
[[seam::host_device_fn]] inline unsigned evaluate(const HardJointGpu& j,const PdrdRigidState* s,unsigned bodies,HardGeometryGpu& result,const AngularTwistEvalContext& twist) {
    if(j.enabled==0u)return OK;
    if(j.enabled!=1u||j.body_a>=bodies||j.body_b>=bodies||j.body_a==j.body_b)return BODY;
    const PdrdRigidState a=s[j.body_a],b=s[j.body_b];
    if(!finite3(j.anchor_a)||!finite3(j.anchor_b)||!finite3(j.rest_translation_m)||
       !finite3(a.anchor)||!finite3(b.anchor)||!finite3(a.x)||!finite3(b.x)||!rotation(a.R)||!rotation(b.R)||
       !unit(j.frame_a_wxyz)||!unit(j.frame_b_wxyz)||!unit(j.rest_relative_wxyz))return FRAME;
    for(int i=0;i<6;++i)if(j.modes[i]>2u||!finite(j.lower[i])||!finite(j.upper[i])||
        (j.modes[i]==1u&&j.lower[i]>=j.upper[i])||
        (j.modes[i]==2u&&j.lower[i]!=j.upper[i]))return AXIS;
    float La[9],Qa[9];
    const float w=j.frame_a_wxyz[0],x=j.frame_a_wxyz[1],y=j.frame_a_wxyz[2],z=j.frame_a_wxyz[3];
    const float n=w*w+x*x+y*y+z*z;
    // Column-major unit quaternion rotation.
    La[0]=1.f-2.f*(y*y+z*z)/n;La[1]=2.f*(x*y+w*z)/n;La[2]=2.f*(x*z-w*y)/n;
    La[3]=2.f*(x*y-w*z)/n;La[4]=1.f-2.f*(x*x+z*z)/n;La[5]=2.f*(y*z+w*x)/n;
    La[6]=2.f*(x*z+w*y)/n;La[7]=2.f*(y*z-w*x)/n;La[8]=1.f-2.f*(x*x+y*y)/n;
    for(int c=0;c<3;++c)for(int r=0;r<3;++r){Qa[3*c+r]=0.f;for(int k=0;k<3;++k)Qa[3*c+r]+=a.R[3*k+r]*La[3*c+k];}
    float sa[3],tb[3],d[3],local[3];rot(a.R,j.anchor_a,sa);rot(b.R,j.anchor_b,tb);
    // NEW upstream representation: actual center = immutable anchor + x.
    // Subtract anchors first, preserving small deltas at large world offsets.
    for(int k=0;k<3;++k)d[k]=(b.anchor[k]-a.anchor[k])+(b.x[k]-a.x[k])+(tb[k]-sa[k]);
    transpose(Qa,d,local);for(int i=0;i<3;++i)result.coordinate[i]=local[i]-j.rest_translation_m[i];
    float sum[3];for(int i=0;i<3;++i)sum[i]=d[i]+sa[i];
    for(int k=0;k<3;++k){float e[3]={},pa[3],pb[3],qa[3],qb[3];e[k]=1.f;
        cross(sum,e,pa);cross(tb,e,pb);transpose(Qa,pa,qa);transpose(Qa,pb,qb);
        for(int r=0;r<3;++r){result.jacobian[12*r+k]=-Qa[3*r+k];result.jacobian[12*r+3+k]=qa[r];
            result.jacobian[12*r+6+k]=Qa[3*r+k];result.jacobian[12*r+9+k]=-qb[r];}}
    if(angular_needed(j)){
        AngularControlGpu angular{};angular.body_a=j.body_a;angular.body_b=j.body_b;angular.enabled=1u;angular.coordinate_model=1u;
        for(int k=0;k<4;++k){angular.frame_a_wxyz[k]=j.frame_a_wxyz[k];angular.frame_b_wxyz[k]=j.frame_b_wxyz[k];angular.rest_relative_wxyz[k]=j.rest_relative_wxyz[k];}
        if(twist.continuous>1u || (twist.continuous==1u && twist.initialized!=1u))return HISTORY;
        const bool continuous=twist.continuous==1u;
        float c[3],J[18];if(!joint_angular_control::eval(angular,s,c,J,nullptr,continuous))return CHART;
        if(continuous && !joint_angular_control::lift_twist(c[0],twist.current_anchor,c[0]))return HISTORY;
        for(int r=0;r<3;++r){result.coordinate[3+r]=c[r];for(int k=0;k<3;++k){result.jacobian[12*(3+r)+3+k]=J[6*r+k];result.jacobian[12*(3+r)+9+k]=J[6*r+3+k];}}
    }
    for(int i=0;i<6;++i)if(!finite(result.coordinate[i]))return FRAME;
    for(int i=0;i<72;++i)if(!finite(result.jacobian[i]))return FRAME;
    return OK;
}
[[seam::host_device_fn]] inline bool offsets(const HardGeometryGpu& g,unsigned base,unsigned dim,unsigned& a,unsigned& b) {
    return joint_angular_control::reduced_offset(base,g.body_a,dim,&a)&&joint_angular_control::reduced_offset(base,g.body_b,dim,&b);
}
} // joint_hard

// One row per AUTHORED hard owner, preserving joint_index independently.
[[seam::entry(i)]]
[[seam::device_fn]] inline void joint_hard_geometry(const HardJointGpu* records,const AngularTwistEvalContext* twist_context,unsigned twist_count,const PdrdRigidState* state,unsigned bodies,HardGeometryGpu* output,unsigned i) {
    HardGeometryGpu result{};result.joint_index=records[i].joint_index;result.body_a=records[i].body_a;result.body_b=records[i].body_b;
    result.fault=records[i].joint_index>=twist_count?joint_hard::HISTORY:joint_hard::evaluate(records[i],state,bodies,result,twist_context[records[i].joint_index]);
    output[i]=result; // full overwrite: a reused allocation never carries stale verdicts
}

// B*x, physical additive derivative. Never multiplies by stiffness or dt.
[[seam::entry(i)]]
[[seam::device_fn]] inline void joint_hard_rows_apply(const HardJointGpu* records,const HardGeometryGpu* geometry,unsigned base,unsigned dim,const float* input,float* output,unsigned* fault,unsigned i) {
    const HardJointGpu j=records[i];const HardGeometryGpu g=geometry[i];
    for(int r=0;r<6;++r)output[6*i+r]=0.f;
    fault[i]=g.fault;if(j.enabled==0u)return;
    unsigned a=0,b=0;if(g.fault)return;if(!joint_hard::offsets(g,base,dim,a,b)){fault[i]=joint_hard::RANGE;return;}
    for(int r=0;r<6;++r)if(j.modes[r]!=0u)for(int k=0;k<6;++k)output[6*i+r]+=g.jacobian[12*r+k]*input[a+k]+g.jacobian[12*r+6+k]*input[b+k];
}

// B^T*lambda, additive reaction covector on BOTH bodies.
[[seam::entry(i)]]
[[seam::device_fn]] inline void joint_hard_rows_transpose(const HardJointGpu* records,const HardGeometryGpu* geometry,unsigned base,unsigned dim,const float* input,compute::atomic_float_t* output,unsigned* fault,unsigned i) {
    const HardJointGpu j=records[i];const HardGeometryGpu g=geometry[i];fault[i]=g.fault;if(j.enabled==0u)return;
    unsigned a=0,b=0;if(g.fault)return;if(!joint_hard::offsets(g,base,dim,a,b)){fault[i]=joint_hard::RANGE;return;}
    for(int k=0;k<6;++k){float va=0.f,vb=0.f;for(int r=0;r<6;++r)if(j.modes[r]!=0u){va+=g.jacobian[12*r+k]*input[6*i+r];vb+=g.jacobian[12*r+6+k]*input[6*i+r];}
        compute::atomic_add(output+a+k,va);compute::atomic_add(output+b+k,vb);}
}

