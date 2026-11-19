// V2 joint arithmetic ported from the bounded basic snapshot.
#pragma once
#include "../energy/model/pdrd_rigid.kernel.cpp"
struct BallAnchorGpu {
unsigned body_a;
unsigned body_b;
unsigned record_index;
unsigned enabled;
float anchor_a[3];
float anchor_b[3];
float stiffness;
};
namespace joint_ball {
[[seam::host_device_fn]] inline bool finite(float x) { return !fmath::isnan(x) && !fmath::isinf(x); }
enum : unsigned { WORLD_ENDPOINT=0xffffffffu, BALL_ANCHOR_FLAG_ACTIVE=1u };

[[seam::host_device_fn]] inline bool finite3(const float* a) { return finite(a[0]) && finite(a[1]) && finite(a[2]); }
[[seam::host_device_fn]] inline bool valid([[seam::thread]] const BallAnchorGpu& j, unsigned n) {
    const bool world_a = j.body_a == WORLD_ENDPOINT;
    const bool world_b = j.body_b == WORLD_ENDPOINT;
    const bool bodies_ok = !(world_a && world_b) &&
        (world_a || j.body_a < n) && (world_b || j.body_b < n) &&
        (world_a || world_b || j.body_a != j.body_b);
    return (j.enabled & BALL_ANCHOR_FLAG_ACTIVE) != 0 &&
        j.stiffness != 0.0f && finite(j.stiffness) &&
        j.stiffness > 0.0f && bodies_ok && finite3(j.anchor_a) && finite3(j.anchor_b);
}
[[seam::host_device_fn]] inline void rot( [[seam::thread]] const float R[9], [[seam::thread]] const float a[3], [[seam::thread]] float p[3]) {
    // RigidState::R is column-major.
    p[0]=R[0]*a[0]+R[3]*a[1]+R[6]*a[2]; p[1]=R[1]*a[0]+R[4]*a[1]+R[7]*a[2]; p[2]=R[2]*a[0]+R[5]*a[1]+R[8]*a[2];
}
[[seam::host_device_fn]] inline void lt( [[seam::thread]] const float p[3], [[seam::thread]] const float v[3], [[seam::thread]] float out[6]) {
    out[0]=v[0]; out[1]=v[1]; out[2]=v[2];
    // (-skew(p))^T v = p x v.
    out[3]=p[1]*v[2]-p[2]*v[1]; out[4]=p[2]*v[0]-p[0]*v[2]; out[5]=p[0]*v[1]-p[1]*v[0];
}
[[seam::host_device_fn]] inline void L( [[seam::thread]] const float p[3], [[seam::thread]] const float u[6], [[seam::thread]] float out[3]) {
    out[0]=u[0]-p[1]*u[5]+p[2]*u[4]; out[1]=u[1]+p[0]*u[5]-p[2]*u[3]; out[2]=u[2]-p[0]*u[4]+p[1]*u[3];
}
[[seam::host_device_fn]] inline void L_pair([[seam::thread]] const float p[3], [[seam::thread]] const float ua[6], [[seam::thread]] const float q[3], [[seam::thread]] const float ub[6], [[seam::thread]] float out[3]) {
    out[0]=(ua[0]-ub[0])+(-p[1]*ua[5]+p[2]*ua[4])-(-q[1]*ub[5]+q[2]*ub[4]);
    out[1]=(ua[1]-ub[1])+(p[0]*ua[5]-p[2]*ua[3])-(q[0]*ub[5]-q[2]*ub[3]);
    out[2]=(ua[2]-ub[2])+(-p[0]*ua[4]+p[1]*ua[3])-(-q[0]*ub[4]+q[1]*ub[3]);
}
[[seam::host_device_fn]] inline bool reduced_offset(unsigned base, unsigned body, unsigned n, [[seam::thread]] unsigned* offset) {
    if(base>n || body>=(n-base)/6u) return false;
    const unsigned first = base + 6u * body;
    if (first > static_cast<unsigned>(n) || first + 5u >= static_cast<unsigned>(n)) return false;
    *offset = first;
    return true;
}
[[seam::host_device_fn]] inline bool block_offset(unsigned body, [[seam::thread]] unsigned* offset) {
    if(body>0xffffffffu/36u-1u) return false;
    const unsigned first = 36u * body;
    if (first > 0xffffffffu || first + 35u > 0xffffffffu) return false;
    *offset = first;
    return true;
}
[[seam::device_fn]] inline void gradient_kernel(const BallAnchorGpu* js,unsigned count,const PdrdRigidState* s,unsigned bodies,unsigned base,unsigned size,compute::atomic_float_t* out, unsigned i) {
    if(i>=count) return;
    const BallAnchorGpu j=js[i]; const bool wa=j.body_a==WORLD_ENDPOINT, wb=j.body_b==WORLD_ENDPOINT;
    unsigned oa=0,ob=0;
    if(!valid(j,bodies)||(!wa&&!reduced_offset(base,j.body_a,size,&oa))||(!wb&&!reduced_offset(base,j.body_b,size,&ob))) return;
    PdrdRigidState sa{},sb{};if(!wa)sa=s[j.body_a];if(!wb)sb=s[j.body_b];
    float p[3],q[3],r[3],a[6],b[6];
    if(wa) { for(int x=0;x<3;++x)p[x]=j.anchor_a[x]; } else rot(sa.R,j.anchor_a,p);
    if(wb) { for(int x=0;x<3;++x)q[x]=j.anchor_b[x]; } else rot(sb.R,j.anchor_b,q);
    // Cancel shared translation before mixing it with rotated local offsets.
    for(int x=0;x<3;++x) {
        r[x]=((wa?j.anchor_a[x]:s[j.body_a].anchor[x])-(wb?j.anchor_b[x]:s[j.body_b].anchor[x]))
            +((wa?0.f:p[x])-(wb?0.f:q[x]))
            +((wa?0.f:s[j.body_a].x[x])-(wb?0.f:s[j.body_b].x[x]));
    }
    if(!wa) { lt(p,r,a); for(int x=0;x<6;++x) compute::atomic_add(out+oa+x,j.stiffness*a[x]); }
    if(!wb) { lt(q,r,b); for(int x=0;x<6;++x) compute::atomic_add(out+ob+x,-j.stiffness*b[x]); }
}
[[seam::device_fn]] inline void energy_kernel(const BallAnchorGpu* js,unsigned count,const PdrdRigidState* s,unsigned bodies,compute::atomic_float_t* out, unsigned i) {
    if(i>=count) return;
    const BallAnchorGpu j=js[i]; const bool wa=j.body_a==WORLD_ENDPOINT, wb=j.body_b==WORLD_ENDPOINT; if(!valid(j,bodies)) return;
    PdrdRigidState sa{},sb{};if(!wa)sa=s[j.body_a];if(!wb)sb=s[j.body_b];
    float p[3],q[3],r[3];
    if(wa) { for(int x=0;x<3;++x)p[x]=j.anchor_a[x]; } else rot(sa.R,j.anchor_a,p);
    if(wb) { for(int x=0;x<3;++x)q[x]=j.anchor_b[x]; } else rot(sb.R,j.anchor_b,q);
    // Cancel shared translation before mixing it with rotated local offsets.
    for(int x=0;x<3;++x) {
        r[x]=((wa?j.anchor_a[x]:s[j.body_a].anchor[x])-(wb?j.anchor_b[x]:s[j.body_b].anchor[x]))
            +((wa?0.f:p[x])-(wb?0.f:q[x]))
            +((wa?0.f:s[j.body_a].x[x])-(wb?0.f:s[j.body_b].x[x]));
    }
    compute::atomic_add(out, 0.5f*j.stiffness*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]));
}
[[seam::device_fn]] inline void matvec_kernel(const BallAnchorGpu* js,unsigned count,const PdrdRigidState* s,unsigned bodies,unsigned base,unsigned size,const float* in,compute::atomic_float_t* out, unsigned i) {
    if(i>=count) return;
    const BallAnchorGpu j=js[i]; const bool wa=j.body_a==WORLD_ENDPOINT, wb=j.body_b==WORLD_ENDPOINT;
    unsigned oa=0,ob=0;
    if(!valid(j,bodies)||(!wa&&!reduced_offset(base,j.body_a,size,&oa))||(!wb&&!reduced_offset(base,j.body_b,size,&ob))) return;
    PdrdRigidState sa{},sb{};if(!wa)sa=s[j.body_a];if(!wb)sb=s[j.body_b];
    float p[3],q[3],ua[6]={},ub[6]={},w[3],a[6],b[6];
    if(wa) { for(int x=0;x<3;++x)p[x]=j.anchor_a[x]; } else rot(sa.R,j.anchor_a,p);
    if(wb) { for(int x=0;x<3;++x)q[x]=j.anchor_b[x]; } else rot(sb.R,j.anchor_b,q);
    if(!wa) { for(int x=0;x<6;++x)ua[x]=in[oa+x]; }
    if(!wb) { for(int x=0;x<6;++x)ub[x]=in[ob+x]; }
    L_pair(p,ua,q,ub,w);
    if(!wa) { lt(p,w,a); for(int x=0;x<6;++x)compute::atomic_add(out+oa+x,j.stiffness*a[x]); }
    if(!wb) { lt(q,w,b); for(int x=0;x<6;++x)compute::atomic_add(out+ob+x,-j.stiffness*b[x]); }
}
[[seam::device_fn]] inline void blocks_kernel(const BallAnchorGpu* js,unsigned count,const PdrdRigidState* s,unsigned bodies,compute::atomic_float_t* blocks, unsigned i) {
    if(i>=count) return;
    const BallAnchorGpu j=js[i];const bool wa=j.body_a==WORLD_ENDPOINT,wb=j.body_b==WORLD_ENDPOINT;unsigned oa=0,ob=0;
    if(!valid(j,bodies)||(!wa&&!block_offset(j.body_a,&oa))||(!wb&&!block_offset(j.body_b,&ob)))return;
    PdrdRigidState sa{},sb{};if(!wa)sa=s[j.body_a];if(!wb)sb=s[j.body_b];
    float p[3],q[3],e[6],v[3],z[6];if(wa){for(int x=0;x<3;++x)p[x]=j.anchor_a[x];}else rot(sa.R,j.anchor_a,p);if(wb){for(int x=0;x<3;++x)q[x]=j.anchor_b[x];}else rot(sb.R,j.anchor_b,q);
    for(int col=0;col<6;++col){for(int x=0;x<6;++x)e[x]=0.f;e[col]=1.f;if(!wa){L(p,e,v);lt(p,v,z);for(int row=0;row<6;++row)compute::atomic_add(blocks+oa+6*row+col,j.stiffness*z[row]);}if(!wb){L(q,e,v);lt(q,v,z);for(int row=0;row<6;++row)compute::atomic_add(blocks+ob+6*row+col,j.stiffness*z[row]);}}
}

}
[[seam::entry(i)]]
[[seam::device_fn]] inline void joint_ball_apply(const BallAnchorGpu *records, const PdrdRigidState *state, unsigned bodies, unsigned base, unsigned dim, unsigned action, const float *input, compute::atomic_float_t *output, unsigned i) {
 if(action==0u) joint_ball::gradient_kernel(records,0xffffffffu,state,bodies,base,dim,output,i);
 else if(action==1u) joint_ball::matvec_kernel(records,0xffffffffu,state,bodies,base,dim,input,output,i);
 else joint_ball::blocks_kernel(records,0xffffffffu,state,bodies,output,i);
}
