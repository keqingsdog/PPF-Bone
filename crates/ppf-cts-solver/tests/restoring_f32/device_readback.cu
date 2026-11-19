// Compile with the real generated CUDA prologue/body include roots. This calls
// the same neutral Schwarz bodies as their generated production entry points.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "data.hpp"
#include "linalg/eigsolve.hpp"
#include "diagnostics/diagnostics.hpp"
#include "zzzzzzzzzzzzzzzzzzzzzzz_legacy_schwarz/legacy_schwarz.kernel.cu"
#include "zzzzzzzzzzzzzzzzzzzzz_reduced_schwarz/reduced_schwarz.kernel.cu"

__global__ void channel_probe(unsigned kind,const float* a,float* out) {
    bool radial=kind==1;
    float k=a[8];
    CurveNode nodes[3]={{-1.f,-k,.5f*k,k},{0.f,0.f,0.f,k},{1.f,k,.5f*k,k}};
    CurveDescriptor desc{};desc.node_count=radial?2:3;desc.domain=radial?2:0;
    CurveBinding b{};b.kind=radial?restoring_curve::RigidAnchor:restoring_curve::Stitch;b.channel=3;b.damping=a[1];
    CurveHistory history{};history.initialized=1;
    restoring_curve::CurveContext ctx{&desc,nodes+(radial?1:0),&b,&history,1,radial?2u:3u,1,1};
    unsigned fault=0;float dt=a[0];
    if(kind==0){
        b.kind=restoring_curve::RigidAngular;b.channel=0;float e=0,g=0,h=0;
        bool ok=restoring_curve::angular(ctx,b.kind,0,0,a[3],a[2],dt,e,g,h,fault);
        out[0]=ok;out[1]=float(fault);out[2]=e;out[3]=g;out[4]=h;return;
    }
    float d[3]={a[5],a[6],a[7]};
    if(radial){
        for(unsigned i=0;i<3;++i)history.d0[i]=a[2+i];
        float e=0,g[3]{},h[9]{};bool ok=restoring_curve::anchor(ctx,b.kind,0,d,dt,e,g,h,fault);
        out[0]=ok;out[1]=float(fault);out[2]=e;
        for(unsigned i=0;i<3;++i)out[3+i]=g[i];for(unsigned i=0;i<9;++i)out[6+i]=h[i];return;
    }
    b.epsilon_m=a[2];b.rest_length_m=a[3];history.q0=a[4];
    restoring_stitch::Evaluation e{};bool ok=restoring_stitch::evaluate_detailed(ctx,0,d,dt,e,fault);
    out[0]=ok;out[1]=float(fault);out[2]=e.energy;out[3]=e.coordinate;
    out[4]=e.radial_eigenvalue;out[5]=e.transverse_eigenvalue;
    for(unsigned i=0;i<3;++i)out[6+i]=e.gradient[i];
    for(unsigned i=0;i<9;++i){out[9+i]=e.raw_hessian[i];out[18+i]=e.projected_hessian[i];}
}

__global__ void schwarz_probe(float* out) {
    CurveNode nodes[2]={{0.f,0.f,0.f,7.f},{1.f,7.f,3.5f,7.f}};
    CurveDescriptor desc{};desc.node_count=2;desc.domain=2;
    CurveBinding binding{};binding.channel=3;binding.damping=.2f;
    CurveHistory history{};history.initialized=1;
    PdrdRigidState state{};state.R[0]=state.R[4]=state.R[8]=1.f;state.x[0]=.3f;
    BallAnchorGpu ball{};ball.body_b=0xffffffffu;ball.enabled=1;ball.stiffness=2.f;
    unsigned ids[48],count[2],fault[2];float matrix[1152];
    for(unsigned active=0;active<2;++active){
        binding.kind=restoring_curve::RigidAnchor;
        legacy_schwarz_export(&ball,nullptr,1,nullptr,0,nullptr,0,nullptr,0,nullptr,
            &state,&state,1,0,6,.1f,0.f,active,&desc,nodes,&binding,&history,1,2,1,1,
            ids,matrix,count,fault,2,0);
        unsigned offset=8*active;out[offset]=float(fault[0]);out[offset+1]=float(count[0]);
        for(unsigned i=0;i<6;++i)out[offset+2+i]=matrix[24*i+i];
    }
    FemJointGpu row{};row.enabled=1;row.ball_stiffness=2.f;row.a.kind=joint_fem::RIGID;row.b.kind=joint_fem::WORLD;
    FemFrameGpu frames[4]{};
    for(unsigned i=0;i<4;++i){frames[i].rotation[0]=frames[i].rotation[4]=frames[i].rotation[8]=1.f;}
    frames[0].anchor[0]=frames[0].center[0]=.3f;
    for(unsigned active=0;active<2;++active){
        binding.kind=restoring_curve::FemAnchor;
        reduced_schwarz_joint_export(&row,1,frames,nullptr,0,0,0,6,nullptr,0,0,1,.1f,
            nullptr,nullptr,0,nullptr,0,0.f,.1f,active,&desc,nodes,&binding,&history,1,2,1,1,
            ids,matrix,count,fault,0);
        unsigned offset=16+8*active;out[offset]=float(fault[0]);out[offset+1]=float(count[0]);
        for(unsigned i=0;i<6;++i)out[offset+2+i]=matrix[24*i+i];
    }
}
int main(int argc,char** argv) {
    if(argc<2)return 2;
    cudaDeviceProp prop{};if(cudaGetDeviceProperties(&prop,0)!=cudaSuccess)return 70;
    fprintf(stderr,"GPU %s sm_%d%d\n",prop.name,prop.major,prop.minor);
    float *a=nullptr,*out=nullptr;
    if(cudaMallocManaged(&a,9*sizeof(float))!=cudaSuccess||cudaMallocManaged(&out,32*sizeof(float))!=cudaSuccess)return 71;
    for(unsigned i=0;i<9;++i)a[i]=0.f;for(unsigned i=0;i<32;++i)out[i]=0.f;a[8]=1.f;
    unsigned kind=std::strcmp(argv[1],"angular")==0?0:std::strcmp(argv[1],"anchor")==0?1:2;
    unsigned length=kind==0?5:kind==1?15:27;
    bool schwarz=std::strcmp(argv[1],"schwarz")==0;
    if(schwarz){schwarz_probe<<<1,1>>>(out);length=32;}
    else {for(int i=2;i<argc&&i<11;++i)a[i-2]=strtof(argv[i],nullptr);channel_probe<<<1,1>>>(kind,a,out);}
    cudaError_t code=cudaGetLastError();if(code==cudaSuccess)code=cudaDeviceSynchronize();
    if(code!=cudaSuccess){fprintf(stderr,"CUDA_ERROR %s\n",cudaGetErrorString(code));return 72;}
    for(unsigned i=0;i<length;++i)printf(i?" %.9g":"%.9g",out[i]);puts("");
    if(schwarz){
        for(unsigned group=0;group<4;++group){
            unsigned offset=8*group;if(out[offset]!=0.f||out[offset+1]!=6.f)return 73;
            float expected=group%2?9.f:2.f;
            for(unsigned i=0;i<6;++i)if(std::fabs(out[offset+2+i]-(i<3?expected:0.f))>2.e-5f)return 74;
        }
    }
    cudaFree(a);cudaFree(out);return 0;
}
