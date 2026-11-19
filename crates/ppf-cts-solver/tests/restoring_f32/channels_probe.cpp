#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "energy/model/restoring_stitch.hpp"
int main(int argc,char** argv){
 if(argc<2)return 2;bool radial=std::strcmp(argv[1],"anchor")==0;
 CurveNode nodes[3]={{-1.f,-1.f,.5f,1.f},{0.f,0.f,0.f,1.f},{1.f,1.f,.5f,1.f}};
 float stiffness=argc>10?strtof(argv[10],nullptr):1.f;
 for(auto& n:nodes){n.effort*=stiffness;n.area*=stiffness;n.right_slope*=stiffness;}
 CurveDescriptor desc{};desc.node_count=radial?2:3;desc.domain=radial?2:0;
 CurveBinding b{};b.kind=radial?restoring_curve::RigidAnchor:restoring_curve::Stitch;b.channel=3;
 CurveHistory history{};history.initialized=1;
 restoring_curve::CurveContext ctx{&desc,nodes+(radial?1:0),&b,&history,1,radial?2u:3u,1,1};
 float dt=strtof(argv[2],nullptr);b.damping=strtof(argv[3],nullptr);unsigned code=0;
 if(std::strcmp(argv[1],"angular")==0){
  b.kind=restoring_curve::RigidAngular;b.channel=0;float e=0,g=0,h=0;
  bool ok=restoring_curve::angular_f32(ctx,b.kind,0,0,strtof(argv[5],nullptr),strtof(argv[4],nullptr),dt,e,g,h,code);
  printf("%d %u %.9g %.9g %.9g\n",ok,code,e,g,h);return 0;
 }
 if(radial){
  for(unsigned i=0;i<3;++i)history.d0[i]=strtof(argv[4+i],nullptr);
  float d[3];for(unsigned i=0;i<3;++i)d[i]=strtof(argv[7+i],nullptr);
  float e=0,g[3]{},h[9]{};bool ok=restoring_curve::anchor_f32(ctx,b.kind,0,d,dt,e,g,h,code);
  printf("%d %u %.9g",ok,code,e);for(float x:g)printf(" %.9g",x);for(float x:h)printf(" %.9g",x);puts("");return 0;
 }
 b.epsilon_m=strtof(argv[4],nullptr);b.rest_length_m=strtof(argv[5],nullptr);history.q0=strtof(argv[6],nullptr);
 float d[3];for(unsigned i=0;i<3;++i)d[i]=strtof(argv[7+i],nullptr);
 restoring_stitch::Evaluation out{};bool ok=restoring_stitch::evaluate_detailed_f32(ctx,0,d,dt,out,code);
 printf("%d %u %.9g %.9g %.9g %.9g",ok,code,out.energy,out.coordinate,out.radial_eigenvalue,out.transverse_eigenvalue);
 for(float x:out.gradient)printf(" %.9g",x);for(float x:out.raw_hessian)printf(" %.9g",x);for(float x:out.projected_hessian)printf(" %.9g",x);puts("");return 0;
}
