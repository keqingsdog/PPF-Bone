// Prescribed rigid roots in the current neutral backend. No penalty energy.
#pragma once
#include "../energy/model/pdrd_rigid.kernel.cpp"
#include "../zz_joint/angular_joint_v2.kernel.cpp"

// 64 bytes, column-major R, one immutable sampled endpoint per active root.
struct RootTargetGpu {
unsigned body;
unsigned enabled;
float center[3];
float R[9];
unsigned list_start;
unsigned list_count;
};

namespace root_motion {
[[seam::host_device_fn]] inline bool finite(float x) {return !fmath::isnan(x)&&!fmath::isinf(x);}
[[seam::host_device_fn]] inline bool offset(unsigned base,unsigned body,unsigned dim,[[seam::thread]] unsigned* k) {
    if(base>dim||body>=(dim-base)/6u)return false;*k=base+6u*body;return true;
}
// Q removes ALL six root increments. Called on reduced RHS, both sides of
// every Hessian application, every preconditioned residual/search direction and
// final homogeneous correction. Unique ascending body records avoid races.
[[seam::host_device_fn]] inline void project([[seam::thread]] const RootTargetGpu& root,
    unsigned base,unsigned dim,float* vector) {
    unsigned k=0;if(!root.enabled||!offset(base,root.body,dim,&k))return;
    for(unsigned a=0;a<6;++a)vector[k+a]=0.f;
}
// Exact subtractive SO(3) tangent to the sampled prescribed orientation.
// This is an affine prescribed DOF, not a projected force or a stiff spring.
[[seam::host_device_fn]] inline void particular([[seam::thread]] const RootTargetGpu& root,
    const PdrdRigidState* state,unsigned bodies,unsigned base,unsigned dim,float* vector) {
    unsigned k=0;if(!root.enabled||root.body>=bodies||!offset(base,root.body,dim,&k))return;
    // New state is anchored: world centroid = anchor + x. Keep the subtraction
    // ordered at body scale; do not interpret x as the old unanchored centroid.
    // position_step SUBTRACTS direction: p = current - target, as dx_seed.
    for(unsigned a=0;a<3;++a)vector[k+a]=state[root.body].x[a]-(root.center[a]-state[root.body].anchor[a]);
    // Row-major M = R_target R_current^T; compose_running_rotation applies
    // exp(-direction_rot). Keep the newest anchored 104-byte state layout.
    float m[9]={0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    for(unsigned r=0;r<3;++r)for(unsigned c=0;c<3;++c)for(unsigned j=0;j<3;++j)
        m[3*r+c]+=root.R[r+3*j]*state[root.body].R[c+3*j];
    float phi[3];joint_angular::so3_log(m,phi);
    for(unsigned a=0;a<3;++a)vector[k+3+a]=-phi[a];
}
} // namespace root_motion

[[seam::entry(i)]]
[[seam::device_fn]] inline void root_project(const RootTargetGpu* targets,float* vector,unsigned base,unsigned dim,unsigned i) {
    root_motion::project(targets[i],base,dim,vector);
}
[[seam::entry(i)]]
[[seam::device_fn]] inline void root_particular(const RootTargetGpu* targets,const PdrdRigidState* state,float* vector,unsigned bodies,unsigned base,unsigned dim,unsigned i) {
    root_motion::particular(targets[i],state,bodies,base,dim,vector);
}

// Write the solver's prescribed target (not accepted output positions). The
// Newton/CCD driver seeds its particular dx from this and keeps the accepted
// pose/history untouched until the candidate passes the real CCD/scan gates.
[[seam::entry(i)]]
[[seam::device_fn]] inline void root_target_vertices(const RootTargetGpu* targets,const unsigned* list,const Vec3f* rest,Vec3f* target,unsigned roots,unsigned vertices,unsigned i) {
    for(unsigned r=0;r<roots;++r) {
        const RootTargetGpu root=targets[r];
        if(!root.enabled||i<root.list_start||i-root.list_start>=root.list_count)continue;
        const unsigned v=list[i];if(v>=vertices)return;
        for(unsigned a=0;a<3;++a) {
            float x=root.center[a];
            for(unsigned c=0;c<3;++c)x+=root.R[a+3*c]*rest[i][c];
            target[v][a]=x;
        }
        return;
    }
}
