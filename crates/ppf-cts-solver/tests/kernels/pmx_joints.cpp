// Test the same six-coordinate evaluation used by native Newton/contact steps.
#include "data.hpp"
#include "zzzzzzzz_fem_joint/fem_joint.kernel.cpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

static int failures;
static void check(bool ok, const char* label) {
    if (!ok) { std::printf("FAIL %s\n", label); ++failures; }
}

static void rotate(PdrdRigidState& body, int axis, float angle) {
    const float c=std::cos(angle), s=std::sin(angle);
    const int u=(axis+1)%3, v=(axis+2)%3;
    for (int column=0; column<3; ++column) {
        float* a=body.R+3*column;
        const float x=a[u],y=a[v];
        a[u]=c*x-s*y;
        a[v]=s*x+c*y;
    }
}

struct Test {
    FemJointGpu joint{};
    PdrdRigidState bodies[2]{};

    Test() {
        joint.enabled=joint.pmx.enabled=1;
        joint.a.kind=joint.b.kind=joint_fem::RIGID;
        joint.b.id=1;
        joint.pmx.frame_a[0]=joint.pmx.frame_b[0]=1.f;
        joint.pmx.limit_weight[0]=100.f;
        joint.pmx.limit_weight[1]=80.f;
        for (auto& body:bodies) joint_fem::identity(body.R);
    }

    float evaluate(float* gradient, float* hessian) const {
        FemFrameGpu frames[4]{};
        frames[0].status=joint_fem::frame(joint.a,nullptr,0,bodies,2,frames[0]);
        frames[1].status=joint_fem::frame(joint.b,nullptr,0,bodies,2,frames[1]);
        frames[2]=frames[0];
        frames[3]=frames[1];
        float energy=0.f;
        unsigned fault=0;
        check(joint_fem::evaluate(joint,frames,0.1f,gradient,hessian,energy,fault) && fault==0,"evaluation");
        return energy;
    }
};

static void check_modes() {
    Test t;
    float g[24],h[576];
    t.joint.pmx.modes[0]=2;
    t.joint.pmx.modes[1]=1;
    t.joint.pmx.lower[1]=-.03f;
    t.joint.pmx.upper[1]=.08f;
    t.bodies[1].x[2]=.9f;
    t.bodies[1].x[1]=.075f;
    check(t.evaluate(g,h)==0.f && g[13]==0.f && g[14]==0.f,"inside and free axes generate no force");
    t.bodies[1].x[0]=-.02f;
    t.bodies[1].x[1]=.1f;
    t.evaluate(g,h);
    check(std::abs(g[12]+2.f)<1e-5f && std::abs(g[13]-2.f)<1e-5f && g[14]==0.f,
        "bilateral lock and upper stop generate restoring forces without springs");
    t.bodies[1].x[1]=-.05f;
    t.evaluate(g,h);
    check(std::abs(g[13]+2.f)<1e-5f,"lower stop generates the opposite force");

    Test angular;
    angular.joint.pmx.modes[3]=1;
    angular.joint.pmx.lower[3]=-.35f;
    angular.joint.pmx.upper[3]=-.1f;
    rotate(angular.bodies[1],0,.2f);
    check(angular.evaluate(g,h)==0.f,"positive physical Rx is negative Bullet XYZ coordinate");
    rotate(angular.bodies[1],0,-.4f);
    check(angular.evaluate(g,h)>0.f && g[15]<0.f,"asymmetric rotation upper stop reacts with correct sign");

    Test rotated_frame;
    const float half_root=.7071067811865475f;
    rotated_frame.joint.pmx.frame_a[0]=rotated_frame.joint.pmx.frame_b[0]=half_root;
    rotated_frame.joint.pmx.frame_a[3]=rotated_frame.joint.pmx.frame_b[3]=half_root;
    rotated_frame.joint.pmx.modes[3]=1;
    rotated_frame.joint.pmx.lower[3]=-.35f;
    rotated_frame.joint.pmx.upper[3]=-.1f;
    rotate(rotated_frame.bodies[1],1,.2f);
    check(rotated_frame.evaluate(g,h)==0.f,"rotated local frame preserves angular limit sign");

    Test spring;
    spring.joint.pmx.springs[1]=20.f;
    spring.bodies[1].x[1]=.2f;
    spring.bodies[1].x[2]=.9f;
    spring.evaluate(g,h);
    check(std::abs(g[13]-4.f)<1e-5f && g[14]==0.f,"spring only acts on its authored axis");
}

static void check_gradient() {
    Test t;
    t.joint.a.local_anchor[1]=.11f;
    t.joint.b.local_anchor[2]=.08f;
    t.bodies[0].x[0]=.04f;
    t.bodies[1].x[0]=.22f;
    rotate(t.bodies[0],2,.45f);
    rotate(t.bodies[1],1,-.2f);
    rotate(t.bodies[1],0,.4f);
    for (int i=0;i<6;++i) {
        t.joint.pmx.modes[i]=2;
        t.joint.pmx.springs[i]=3.f+i;
    }
    float g[24],h[576];
    t.evaluate(g,h);
    const float step=5e-4f;
    for (int side=0;side<2;++side) for (int k=0;k<6;++k) {
        Test plus=t,minus=t;
        if (k<3) {
            plus.bodies[side].x[k]+=step;
            minus.bodies[side].x[k]-=step;
        } else {
            rotate(plus.bodies[side],k-3,step);
            rotate(minus.bodies[side],k-3,-step);
        }
        float scratch_g[24],scratch_h[576];
        const float numerical=(plus.evaluate(scratch_g,scratch_h)-minus.evaluate(scratch_g,scratch_h))/(2*step);
        const float analytic=g[12*side+k];
        check(std::abs(numerical-analytic)<3e-3f*(1.f+std::abs(analytic)),"world-space derivative");
    }
}

int main() {
    check_modes();
    check_gradient();
    std::printf("PMX native joint evaluation %s (%d failures)\n",failures?"FAIL":"PASS",failures);
    return failures?1:0;
}
