// Optional prescribed-curve certificate. Default analytic/contact paths stay.
#pragma once
#include "../zzzz_root/root_motion.kernel.cpp"

namespace root_curve {
[[seam::host_device_fn]] inline float segment_radius_min(Vec3f a,Vec3f b,Vec3f c) {
    Vec3f d=b-a;const float dd=d.dot(d);
    const float t=dd>0.f?fmath::max(0.f,fmath::min(1.f,(c-a).dot(d)/dd)):0.f;
    return (a+t*d-c).norm();
}
[[seam::host_device_fn]] inline bool analytic_clear(Vec3f a,Vec3f b,float margin,
    const Sphere* spheres,unsigned ns,const Floor* floors,unsigned nf) {
    if(!root_motion::finite(margin)||margin<0.f)return false;
    for(unsigned s=0;s<ns;++s) {
        const Sphere sphere=spheres[s];if(sphere.kinematic)continue;
        // Bowls need a curved clipped-sphere/cylinder envelope, never the
        // endpoint-only legacy sphere test. Host capability gate refuses them.
        if(sphere.bowl)return false;
        const Vec3f center=sphere.center;
        const float rounding=16.f*1.1920928955078125e-7f*
            (a.norm()+b.norm()+center.norm()+fmath::abs(sphere.radius)+margin);
        const float pad=margin+rounding;
        if(sphere.reverse) {
            if(!(fmath::max((a-center).norm(),(b-center).norm())+pad<sphere.radius))return false;
        } else if(!(segment_radius_min(a,b,center)>sphere.radius+pad))return false;
    }
    for(unsigned f=0;f<nf;++f) {
        const Floor floor=floors[f];if(floor.kinematic)continue;
        const Vec3f up=floor.up,ground=floor.ground;
        const float norm=up.norm();if(!(norm>0.f))return false;
        const float rounding=16.f*1.1920928955078125e-7f*
            (a.norm()+b.norm()+ground.norm()+margin)*norm;
        if(!(fmath::min(up.dot(a-ground),up.dot(b-ground))>margin*norm+rounding))return false;
    }
    return true;
}
}
[[seam::entry(i)]]
[[seam::device_fn]] inline void root_curve_analytic(const RootTargetGpu* targets,
    const unsigned* list,const Vec3f* x0,const Vec3f* x1,const float* margin,
    const Sphere* spheres,unsigned sphere_count,const Floor* floors,unsigned floor_count,
    unsigned* refused,unsigned roots,unsigned vertices,unsigned i) {
    refused[i]=0u;
    for(unsigned r=0;r<roots;++r) {
        const RootTargetGpu root=targets[r];
        if(!root.enabled||i<root.list_start||i-root.list_start>=root.list_count)continue;
        const unsigned v=list[i];
        if(v>=vertices){refused[i]=1u;return;}
        const Vec3f a=x0[v],b=x1[v];
        refused[i]=root_curve::analytic_clear(a,b,margin[v],spheres,sphere_count,floors,floor_count)?0u:1u;
        return;
    }
}

// Native controlled-DOF retraction BEFORE rigidify's unchanged sweep/gates.
// Particular was computed in the current fitted R frame. Using accumulated
// free-running R here would integrate the refit's tiny mismatch each step.
[[seam::entry(i)]]
[[seam::device_fn]] inline void root_controlled_retract(const RootTargetGpu* targets,
    const PdrdRigidState* state,const float* dtheta,float* running,
    unsigned bodies,float fraction,unsigned i) {
    const RootTargetGpu root=targets[i];
    if(!root.enabled||root.body>=bodies)return;
    const unsigned b=root.body;
    // At a FULL accepted Newton correction this is the exact frozen prescribed
    // orientation, the constrained counterpart to exp(-dtheta) R for free DOFs.
    // It is NOT a commit/output/gate correction; rigidify CCD still runs next.
    if(fraction==1.f) {
        for(unsigned e=0;e<9u;++e)running[9u*b+e]=root.R[e];
        return;
    }
    const Vec3f increment(-fraction*dtheta[3u*b],-fraction*dtheta[3u*b+1u],-fraction*dtheta[3u*b+2u]);
    const Mat3x3f delta=pdrd_exp_so3(increment);
    Mat3x3f fitted;
    for(unsigned e=0;e<9u;++e)fitted.data()[e]=state[b].R[e];
    const Mat3x3f retracted=delta*fitted;
    for(unsigned e=0;e<9u;++e)running[9u*b+e]=retracted.data()[e];
}
