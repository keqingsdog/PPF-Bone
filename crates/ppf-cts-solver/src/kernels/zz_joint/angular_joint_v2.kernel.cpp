// Isotropic SO(3) V2 spring, damper and finite cone penalty.
#pragma once
#include "../energy/model/pdrd_rigid.kernel.cpp"
struct AngularJointGpuV2 {
unsigned body_a;
unsigned body_b;
unsigned enabled;
float max_angle_rad;
float frame_a_wxyz[4];
float frame_b_wxyz[4];
float rest_relative_wxyz[4];
float stiffness_Nm_per_rad;
float damping_Nms_per_rad;
};
namespace joint_angular {
[[seam::host_device_fn]] inline bool finite(float x) { return !fmath::isnan(x) && !fmath::isinf(x); }
enum : unsigned { ANGULAR_WORLD_ENDPOINT_V2=0xffffffffu };

using M3 = float[9];
[[seam::host_device_fn]] inline void ident( [[seam::thread]] float a[9]) { for(int i=0;i<9;++i)a[i]=0.f; a[0]=a[4]=a[8]=1.f; }
[[seam::host_device_fn]] inline void transpose( [[seam::thread]] const float a[9], [[seam::thread]] float o[9]) { for(int r=0;r<3;++r)for(int c=0;c<3;++c)o[3*r+c]=a[3*c+r]; }
[[seam::host_device_fn]] inline void mul( [[seam::thread]] const float a[9], [[seam::thread]] const float b[9], [[seam::thread]] float o[9]) { float t[9];for(int r=0;r<3;++r)for(int c=0;c<3;++c)t[3*r+c]=a[3*r]*b[c]+a[3*r+1]*b[3+c]+a[3*r+2]*b[6+c];for(int i=0;i<9;++i)o[i]=t[i]; }
[[seam::host_device_fn]] inline void quat_matrix( [[seam::thread]] const float q[4], [[seam::thread]] float o[9]) {
    const float w=q[0],x=q[1],y=q[2],z=q[3];
    o[0]=1.f-2.f*(y*y+z*z); o[1]=2.f*(x*y-z*w); o[2]=2.f*(x*z+y*w);
    o[3]=2.f*(x*y+z*w); o[4]=1.f-2.f*(x*x+z*z); o[5]=2.f*(y*z-x*w);
    o[6]=2.f*(x*z-y*w); o[7]=2.f*(y*z+x*w); o[8]=1.f-2.f*(x*x+y*y);
}
[[seam::host_device_fn]] inline void body_rotation(const PdrdRigidState* s,unsigned body, [[seam::thread]] float o[9]) {
    for(int r=0;r<3;++r)for(int c=0;c<3;++c)o[3*r+c]=s[body].R[3*c+r];
}
[[seam::host_device_fn]] inline void frame_rotation(const PdrdRigidState* s,unsigned body,unsigned world, [[seam::thread]] const float q[4], [[seam::thread]] float o[9]) {
    float qmat[9]; quat_matrix(q,qmat);
    if(body==world) { for(int i=0;i<9;++i)o[i]=qmat[i]; return; }
    float r[9]; body_rotation(s,body,r); mul(r,qmat,o);
}
[[seam::host_device_fn]] inline void so3_log( [[seam::thread]] const float r[9], [[seam::thread]] float v[3]) {
    float q[4], tr=r[0]+r[4]+r[8];
    if(tr>0.f) { float s=fmath::sqrt(fmath::max(tr+1.f,0.f))*2.f; q[0]=.25f*s; q[1]=(r[7]-r[5])/s; q[2]=(r[2]-r[6])/s; q[3]=(r[3]-r[1])/s; }
    else if(r[0]>=r[4]&&r[0]>=r[8]) { float s=fmath::sqrt(fmath::max(1.f+r[0]-r[4]-r[8],0.f))*2.f; q[0]=(r[7]-r[5])/s; q[1]=.25f*s; q[2]=(r[1]+r[3])/s; q[3]=(r[2]+r[6])/s; }
    else if(r[4]>=r[8]) { float s=fmath::sqrt(fmath::max(1.f+r[4]-r[0]-r[8],0.f))*2.f; q[0]=(r[2]-r[6])/s; q[1]=(r[1]+r[3])/s; q[2]=.25f*s; q[3]=(r[5]+r[7])/s; }
    else { float s=fmath::sqrt(fmath::max(1.f+r[8]-r[0]-r[4],0.f))*2.f; q[0]=(r[3]-r[1])/s; q[1]=(r[2]+r[6])/s; q[2]=(r[5]+r[7])/s; q[3]=.25f*s; }
    float n=fmath::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    if(!(n>0.f)||!finite(n)){v[0]=v[1]=v[2]=fmath::div(0.f,0.f);return;}
    for(int i=0;i<4;++i)q[i]/=n;
    if(q[0]<0.f)for(int i=0;i<4;++i)q[i]=-q[i];
    float sn=fmath::sqrt(q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    if(sn<1e-7f){v[0]=2.f*q[1];v[1]=2.f*q[2];v[2]=2.f*q[3];return;}
    float theta=2.f*fmath::atan2(sn,fmath::max(q[0],0.f)),scale=theta/sn;
    v[0]=scale*q[1];v[1]=scale*q[2];v[2]=scale*q[3];
}
[[seam::host_device_fn]] inline void left_log_jacobian_inverse( [[seam::thread]] const float p[3], [[seam::thread]] float o[9]) {
    const float t2=p[0]*p[0]+p[1]*p[1]+p[2]*p[2],t=fmath::sqrt(t2);
    float a;
    if(t<1e-3f) a=1.f/12.f+t2/720.f+t2*t2/30240.f;
    else { const float half_angle=.5f*t; const float cot=fmath::cos_bounded(half_angle)/fmath::sin_bounded(half_angle); a=(1.f-half_angle*cot)/t2; }
    const float x=p[0],y=p[1],z=p[2];
    const float k[9]={0.f,-z,y,z,0.f,-x,-y,x,0.f};
    float k2[9];mul(k,k,k2); ident(o);
    for(int i=0;i<9;++i)o[i]=o[i]-.5f*k[i]+a*k2[i];
}
[[seam::host_device_fn]] inline bool frame_jacobian([[seam::thread]] const AngularJointGpuV2& j,const PdrdRigidState* cur,
    const PdrdRigidState* ref,unsigned world,bool damping, [[seam::thread]] float phi[3], [[seam::thread]] float m[9]) {
    float a[9],b[9],at[9],e[9],d[9],dt[9],f[9],jl[9],tmp[9];
    frame_rotation(cur,j.body_a,world,j.frame_a_wxyz,a);
    frame_rotation(cur,j.body_b,world,j.frame_b_wxyz,b);
    transpose(a,at);mul(at,b,e);
    if(damping) {
        float ar[9],br[9],art[9];
        frame_rotation(ref,j.body_a,world,j.frame_a_wxyz,ar);
        frame_rotation(ref,j.body_b,world,j.frame_b_wxyz,br);
        transpose(ar,art);mul(art,br,d);
    } else quat_matrix(j.rest_relative_wxyz,d);
    transpose(d,dt);mul(dt,e,f);so3_log(f,phi);
    if(!finite(phi[0])||!finite(phi[1])||!finite(phi[2]))return false;
    left_log_jacobian_inverse(phi,jl);mul(jl,dt,tmp);mul(tmp,at,m);
    for(int i=0;i<9;++i)if(!finite(m[i]))return false;
    return true;
}
[[seam::host_device_fn]] inline bool valid([[seam::thread]] const AngularJointGpuV2& j,unsigned n) {
    const bool wa=j.body_a==ANGULAR_WORLD_ENDPOINT_V2,wb=j.body_b==ANGULAR_WORLD_ENDPOINT_V2;
    if(!j.enabled||(wa&&wb)||(!wa&&j.body_a>=n)||(!wb&&j.body_b>=n)||(!wa&&!wb&&j.body_a==j.body_b))return false;
    if(!finite(j.max_angle_rad)||j.max_angle_rad<0.f||j.max_angle_rad>3.14159265358979323846f||!finite(j.stiffness_Nm_per_rad)||j.stiffness_Nm_per_rad<0.f||!finite(j.damping_Nms_per_rad)||j.damping_Nms_per_rad<0.f)return false;
    for(int i=0;i<4;++i)if(!finite(j.frame_a_wxyz[i])||!finite(j.frame_b_wxyz[i])||!finite(j.rest_relative_wxyz[i]))return false;
    return j.stiffness_Nm_per_rad>0.f||j.damping_Nms_per_rad>0.f||j.max_angle_rad>0.f;
}
[[seam::host_device_fn]] inline bool offset(unsigned base,unsigned body,unsigned dim,[[seam::thread]] unsigned* out) {
    if(base>dim || body>=(dim-base)/6u)return false; const unsigned k=base+6u*body;
    if(k>(unsigned)dim||k+5u>=(unsigned)dim)return false;*out=k;return true;
}
[[seam::host_device_fn]] inline void transpose_mul_vec( [[seam::thread]] const float m[9], [[seam::thread]] const float x[3], [[seam::thread]] float y[3]) { for(int c=0;c<3;++c)y[c]=m[c]*x[0]+m[3+c]*x[1]+m[6+c]*x[2]; }
[[seam::host_device_fn]] inline void angular_potential([[seam::thread]] const AngularJointGpuV2& j, [[seam::thread]] const float phi[3], [[seam::thread]] float q[3], [[seam::thread]] float h[9]) {
    const float theta=fmath::sqrt(phi[0]*phi[0]+phi[1]*phi[1]+phi[2]*phi[2]);
    const float k=j.stiffness_Nm_per_rad;
    for(int r=0;r<3;++r)for(int c=0;c<3;++c)h[3*r+c]=(r==c?k:0.f);
    for(int r=0;r<3;++r)q[r]=k*phi[r];
    if(j.max_angle_rad>0.f&&theta>j.max_angle_rad&&theta>1e-7f) {
        // A steep, positive-semidefinite penalty keeps the relative SO(3)
        // rotation inside a near-hard cone while retaining a well-defined CG
        // Hessian. The regular spring remains active inside the limit.
        const float kl=fmath::max(2000.f,10.f*k),excess=theta-j.max_angle_rad;
        const float ratio=excess/theta;
        for(int r=0;r<3;++r) {
            const float u=phi[r]/theta;
            q[r]+=kl*excess*u;
            for(int c=0;c<3;++c) {
                const float v=phi[c]/theta;
                h[3*r+c]+=kl*(ratio*(r==c?1.f:0.f)+(1.f-ratio)*u*v);
            }
        }
    }
}
[[seam::host_device_fn]] inline void form_h([[seam::thread]] const AngularJointGpuV2& j,const PdrdRigidState* cur,const PdrdRigidState* ref,unsigned world,float dt, [[seam::thread]] float h[9]) {
    float ps[3],pd[3],ms[9],md[9],hp[9],q[3];for(int i=0;i<9;++i)h[i]=0.f;
    if((j.stiffness_Nm_per_rad>0.f||j.max_angle_rad>0.f)&&frame_jacobian(j,cur,ref,world,false,ps,ms)) {
        angular_potential(j,ps,q,hp);
        for(int r=0;r<3;++r)for(int c=0;c<3;++c)for(int a=0;a<3;++a)for(int b=0;b<3;++b)h[3*r+c]+=ms[3*a+r]*hp[3*a+b]*ms[3*b+c];
    }
    if(j.damping_Nms_per_rad>0.f&&frame_jacobian(j,cur,ref,world,true,pd,md)) {
        const float cd=j.damping_Nms_per_rad/dt;
        for(int r=0;r<3;++r)for(int c=0;c<3;++c)for(int k=0;k<3;++k)h[3*r+c]+=cd*md[3*k+r]*md[3*k+c];
    }
}
[[seam::device_fn]] inline void gradient_kernel(const AngularJointGpuV2* js,unsigned count,const PdrdRigidState* cur,const PdrdRigidState* ref,unsigned bodies,unsigned base,unsigned dim,float dt,compute::atomic_float_t* out, unsigned i) {
    if(i>=count)return;const auto j=js[i];if(!valid(j,bodies))return;
    const bool wa=j.body_a==ANGULAR_WORLD_ENDPOINT_V2,wb=j.body_b==ANGULAR_WORLD_ENDPOINT_V2;unsigned oa=0,ob=0;
    if((!wa&&!offset(base,j.body_a,dim,&oa))||(!wb&&!offset(base,j.body_b,dim,&ob)))return;
    float g[3]={},phi[3],m[9],v[3],hp[9],q[3];
    if((j.stiffness_Nm_per_rad>0.f||j.max_angle_rad>0.f)&&frame_jacobian(j,cur,ref,ANGULAR_WORLD_ENDPOINT_V2,false,phi,m)) {angular_potential(j,phi,q,hp);transpose_mul_vec(m,q,v);for(int k=0;k<3;++k)g[k]-=v[k];}
    if(j.damping_Nms_per_rad>0.f&&frame_jacobian(j,cur,ref,ANGULAR_WORLD_ENDPOINT_V2,true,phi,m)) {transpose_mul_vec(m,phi,v);for(int k=0;k<3;++k)g[k]-=(j.damping_Nms_per_rad/dt)*v[k];}
    for(int k=0;k<3;++k){if(!wa)compute::atomic_add(out+oa+3+k,g[k]);if(!wb)compute::atomic_add(out+ob+3+k,-g[k]);}
}
[[seam::device_fn]] inline void matvec_kernel(const AngularJointGpuV2* js,unsigned count,const PdrdRigidState* cur,const PdrdRigidState* ref,unsigned bodies,unsigned base,unsigned dim,float dt,const float* in,compute::atomic_float_t* out,const float* tangent_cache,const unsigned* tangent_epoch,unsigned epoch, unsigned i) {
    if(i>=count)return;const auto j=js[i];if(!valid(j,bodies))return;
    const bool wa=j.body_a==ANGULAR_WORLD_ENDPOINT_V2,wb=j.body_b==ANGULAR_WORLD_ENDPOINT_V2;unsigned oa=0,ob=0;
    if((!wa&&!offset(base,j.body_a,dim,&oa))||(!wb&&!offset(base,j.body_b,dim,&ob)))return;
    float h[9],x[3]={},y[3];
    if(epoch!=0u&&tangent_epoch[i]==epoch)for(int k=0;k<9;++k)h[k]=tangent_cache[9*i+k];
    else form_h(j,cur,ref,ANGULAR_WORLD_ENDPOINT_V2,dt,h);
    for(int k=0;k<3;++k)x[k]=(wb?0.f:in[ob+3+k])-(wa?0.f:in[oa+3+k]);
    for(int r=0;r<3;++r)y[r]=h[3*r]*x[0]+h[3*r+1]*x[1]+h[3*r+2]*x[2];
    for(int k=0;k<3;++k){if(!wa)compute::atomic_add(out+oa+3+k,-y[k]);if(!wb)compute::atomic_add(out+ob+3+k,y[k]);}
}
[[seam::device_fn]] inline void build_tangent_kernel(const AngularJointGpuV2* js,unsigned count,const PdrdRigidState* cur,const PdrdRigidState* ref,unsigned bodies,float dt,float* tangent_cache,unsigned* tangent_epoch,unsigned epoch,unsigned i) {
    if(i>=count)return;
    const auto j=js[i];float h[9];
    if(valid(j,bodies))form_h(j,cur,ref,ANGULAR_WORLD_ENDPOINT_V2,dt,h);
    else for(int k=0;k<9;++k)h[k]=0.f;
    for(int k=0;k<9;++k)tangent_cache[9*i+k]=h[k];
    // Publish only after all nine values for this row have been written.
    tangent_epoch[i]=epoch;
}
[[seam::device_fn]] inline void self_blocks_kernel(const AngularJointGpuV2* js,unsigned count,const PdrdRigidState* cur,const PdrdRigidState* ref,unsigned bodies,float dt,compute::atomic_float_t* blocks, unsigned i) {
    if(i>=count)return;const auto j=js[i];if(!valid(j,bodies))return;
    const bool wa=j.body_a==ANGULAR_WORLD_ENDPOINT_V2,wb=j.body_b==ANGULAR_WORLD_ENDPOINT_V2;float h[9];form_h(j,cur,ref,ANGULAR_WORLD_ENDPOINT_V2,dt,h);
    if(!wa)for(int r=0;r<3;++r)for(int c=0;c<3;++c)compute::atomic_add(blocks+36*j.body_a+6*(3+r)+(3+c),h[3*r+c]);
    if(!wb)for(int r=0;r<3;++r)for(int c=0;c<3;++c)compute::atomic_add(blocks+36*j.body_b+6*(3+r)+(3+c),h[3*r+c]);
}

}
[[seam::entry(i)]]
[[seam::device_fn]] inline void joint_angular_apply(const AngularJointGpuV2 *records, const PdrdRigidState *state, const PdrdRigidState *reference, unsigned bodies, unsigned base, unsigned dim, float dt, unsigned action, const float *input, compute::atomic_float_t *output, float *tangent_cache, unsigned *tangent_epoch, unsigned epoch, unsigned i) {
 if(action==0u) joint_angular::gradient_kernel(records,0xffffffffu,state,reference,bodies,base,dim,dt,output,i);
 else if(action==1u) joint_angular::matvec_kernel(records,0xffffffffu,state,reference,bodies,base,dim,dt,input,output,tangent_cache,tangent_epoch,epoch,i);
 else if(action==2u) joint_angular::self_blocks_kernel(records,0xffffffffu,state,reference,bodies,dt,output,i);
 else if(action==3u && epoch!=0u) joint_angular::build_tangent_kernel(records,0xffffffffu,state,reference,bodies,dt,tangent_cache,tangent_epoch,epoch,i);
}
