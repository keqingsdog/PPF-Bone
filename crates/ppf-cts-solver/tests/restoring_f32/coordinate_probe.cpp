#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <limits>
#include <initializer_list>
#include "energy/model/restoring_stitch.hpp"
int main(int argc,char** argv) {
 if(argc==1){
  namespace f=restoring_curve::device_f32;
  float out=0.f;
  float norms[2]={1.e20f,1.e-30f};
  for(float x:norms){float d[3]={x,0.f,0.f};assert(f::narrow(f::norm(d,3),out));assert(out==x);}
  assert(f::narrow(f::mul(f::scaled(1.e20f),f::div(f::scaled(1.e20f),f::scaled(1.e20f))),out)&&out==1.e20f);
  auto cd=f::div(f::scaled(1.e-30f),f::scaled(1.e30f));
  auto delta=f::sub(f::scaled(1.f),f::scaled(-1.e30f));
  assert(f::narrow(f::mul(f::scaled(.5f),f::mul(cd,f::square(delta))),out,true));
  assert(out>.499999f&&out<.500001f);
  float tiny=std::numeric_limits<float>::denorm_min();
  restoring_curve::HostNode nodes[2]={{0.f,0.f,0.f},{1.f,2.f*tiny,tiny}};
  auto value=restoring_curve::eval_f32({nodes,2,restoring_curve::Domain::Radial,restoring_curve::Tail::Linear},.25f);
  assert(value.fault==restoring_curve::Fault::Arithmetic);
  assert(!f::narrow(f::div(f::scaled(1.f),f::scaled(0.f)),out));
  assert(!f::narrow(f::root(f::scaled(-1.f)),out));
  auto maximum=f::scaled(std::numeric_limits<float>::max());
  assert(f::narrow(maximum,out)&&out==std::numeric_limits<float>::max());
  assert(!f::narrow(f::mul(maximum,f::scaled(1.00000011920928955078125f)),out));
  assert(f::narrow(f::mul(maximum,f::scaled(.999999940395355224609375f)),out));
  auto half=f::scaled(1.f,-150);
  assert(!f::narrow(half,out));assert(f::narrow(half,out,true)&&out==0.f);
  assert(f::narrow_angular_effort(half,out)&&out==0.f);
  assert(f::narrow_angular_effort(f::neg(half),out)&&out==0.f);
  assert(f::narrow_angular_effort(f::scaled(1.00000011920928955078125f,-150),out)&&out==tiny);
  assert(f::narrow_angular_effort(f::scaled(-1.00000011920928955078125f,-150),out)&&out==-tiny);
  assert(!f::narrow_angular_effort(f::mul(maximum,f::scaled(2.f)),out));
  assert(!f::narrow_angular_effort(f::scaled(std::numeric_limits<float>::infinity()),out));
  assert(!f::narrow_angular_effort(f::scaled(std::numeric_limits<float>::quiet_NaN()),out));
  assert(f::narrow(f::scaled(1.00000011920928955078125f,-150),out)&&out==tiny);
  assert(!f::narrow(f::scaled(.999999940395355224609375f,-150),out));
  restoring_curve::HostNode oddnodes[3]={{0.f,0.f,0.f},{1.f,1.f,.5f},{2.f,3.f,2.5f}};
  restoring_curve::HostView odd{oddnodes,3,restoring_curve::Domain::Odd,restoring_curve::Tail::Linear};
  auto knot=restoring_curve::eval_f32(odd,1.f);assert(knot.curvature==2.f&&knot.gradient==1.f&&knot.energy==.5f);
  auto left=restoring_curve::eval_f32(odd,.999999940395355224609375f);assert(left.curvature==1.f);
  auto reflected=restoring_curve::eval_f32(odd,-1.f);assert(reflected.curvature==2.f&&reflected.gradient==-1.f&&reflected.energy==.5f);
  auto tail=restoring_curve::eval_f32(odd,3.f);assert(tail.energy==6.5f&&tail.gradient==5.f&&tail.curvature==2.f);
  odd.tail=restoring_curve::Tail::HoldEffort;tail=restoring_curve::eval_f32(odd,3.f);
  assert(tail.energy==5.5f&&tail.gradient==3.f&&tail.curvature==0.f);
  odd.domain=restoring_curve::Domain::Radial;auto zero=restoring_curve::eval_f32(odd,0.f);
  assert(zero.energy==0.f&&zero.gradient==0.f&&zero.curvature==1.f);
  assert(restoring_curve::eval_f32(odd,-1.f).fault==restoring_curve::Fault::InvalidCoordinate);
  // Reproduce the actual initial angular coordinate on both sides of zero.
  CurveNode actual_nodes[2]={{0.f,0.f,0.f,0.f},{1.f,.0004735227848868817f,.00023676139244344085f,.0004735227848868817f}};
  CurveDescriptor descriptor{};descriptor.node_count=2;descriptor.domain=1;
  CurveBinding binding{};binding.kind=restoring_curve::FemAngular;binding.channel=2;binding.damping=.00010550889419391751f;
  restoring_curve::CurveContext ctx{&descriptor,actual_nodes,&binding,nullptr,1,2,1,0};
  float coordinate=27.f*tiny,energy=0.f,gradient=0.f,curvature=0.f;unsigned fault=0;
  for(float x : {coordinate,-coordinate}) {
   assert(restoring_curve::angular_f32(ctx,binding.kind,0,2,x,0.f,.01f,energy,gradient,curvature,fault));
   assert(energy==0.f&&gradient==0.f&&curvature>0.f&&fault==0);
  }
  // Neither .375*tiny term is representable by itself, while their sum is.
  actual_nodes[1].effort=.375f;actual_nodes[1].area=.1875f;binding.damping=.375f;
  assert(restoring_curve::angular_f32(ctx,binding.kind,0,2,tiny,0.f,1.f,energy,gradient,curvature,fault));
  assert(gradient==tiny&&curvature==.75f);
  assert(restoring_curve::angular_f32(ctx,binding.kind,0,2,-tiny,0.f,1.f,energy,gradient,curvature,fault));
  assert(gradient==-tiny&&curvature==.75f);
  puts("PASS scaled range/norm/damping/nonzero-force-underflow/domain checks");return 0;
 }
 if(argc!=6)return 2;
 float d[3]={strtof(argv[1],nullptr),strtof(argv[2],nullptr),strtof(argv[3],nullptr)};
 float eps=strtof(argv[4],nullptr),rest=strtof(argv[5],nullptr),q=0.f,legacy=0.f;unsigned code=0,oldcode=0;
 bool ok=restoring_stitch::coordinate_f32(d,eps,rest,q,code);
 bool old=restoring_stitch::coordinate_legacy(d,eps,rest,legacy,oldcode);
 printf("%d %.9g %u %d %.9g %u\n",ok,q,code,old,legacy,oldcode);
 return 0;
}
