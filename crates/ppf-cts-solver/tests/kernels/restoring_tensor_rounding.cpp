#include "data.hpp"
#include "energy/model/restoring_stitch.hpp"
#include <cstdio>
#include <cmath>
#include <limits>
#ifdef PPF_TEST_BORROWED_RAW_HEADER
#include PPF_TEST_BORROWED_RAW_HEADER
#endif
namespace f=restoring_curve::device_f32;
int failures=0;void ck(bool ok,const char*n){printf("%s %s\n",ok?"PASS":"FAIL",n);failures+=!ok;}
struct BindingCase {
 CurveDescriptor descriptor{};CurveNode nodes[2]{};CurveBinding binding{};CurveHistory history{};
 BindingCase(float stiffness=11.f){descriptor.node_count=2;descriptor.domain=1;nodes[1]={1.f,stiffness,.5f*stiffness,stiffness};binding.kind=restoring_curve::CrossLinkDistance;binding.channel=3;binding.epsilon_m=.75f;binding.rest_length_m=1.f;history.initialized=1;}
 restoring_curve::CurveContext context(){return {&descriptor,nodes,&binding,&history,1,2,1,1};}
 bool eval(const float d[3],restoring_stitch::Evaluation& out,unsigned& fault){auto ctx=context();return restoring_stitch::evaluate_binding_f32(ctx,&binding,d,1.f,out,fault);}
};
void binding_tests(){
 {BindingCase c;float d[3]={1e-30f,1e-30f,1.f};restoring_stitch::Evaluation out{};unsigned fault=0;
  bool ok=c.eval(d,out,fault);
  ck(ok&&fault==0&&out.radial_eigenvalue>1.f&&out.transverse_eigenvalue>1.f&&out.gradient[2]>1.f,"real binding normal tensor tiny offdiag accepted");
  ck(ok&&out.raw_hessian[1]==0.f&&out.projected_hessian[1]==0.f,"real raw and PSD offdiag round to zero at storage");
 }
 {BindingCase c(std::numeric_limits<float>::denorm_min());c.binding.epsilon_m=std::ldexp(1.f,-80);c.binding.rest_length_m=std::ldexp(1.f,100);
  float axis[3]={1.f,0.f,0.f};restoring_stitch::Evaluation out{};unsigned fault=0;
  bool ok=c.eval(axis,out,fault);
  ck(ok&&fault==0&&out.transverse_eigenvalue<-std::numeric_limits<float>::min()&&out.radial_eigenvalue==std::numeric_limits<float>::denorm_min(),"real binding normal negative raw mode and subnormal PSD principal witness");
  float diagonal[3]={1.f,1.f,0.f};out.energy=123.f;fault=0;ok=c.eval(diagonal,out,fault);
  ck(!ok&&fault==restoring_curve::CurveFaultArithmetic&&out.energy==123.f,"real binding PSD subnormal tiny component rejected without borrowing raw scale");
#ifdef PPF_TEST_BORROWED_RAW_HEADER
  auto ctx=c.context();restoring_stitch_borrowed_raw::Evaluation borrowed{};fault=0;
  ck(restoring_stitch_borrowed_raw::evaluate_binding_f32(ctx,&c.binding,diagonal,1.f,borrowed,fault)&&fault==0,"negative-control borrowed raw scale passes proving PSD storage rejection seam");
#endif
 }
}
int main(){binding_tests();float v=0;auto cv=[](f::Scaled a,float scale,float &v){return restoring_stitch::narrow_hessian_component(a,scale,v);};
ck(cv({1.5f,-160},11.f,v)&&v==0&&!std::signbit(v),"positive far subnormal rounds positive zero");
ck(cv({-1.5f,-160},11.f,v)&&v==0&&std::signbit(v),"negative far subnormal rounds negative zero");
ck(cv({1.f,-150},11.f,v)&&v==0&&!std::signbit(v),"positive exact half minimum subnormal tie to even zero");
ck(cv({-1.f,-150},11.f,v)&&v==0&&std::signbit(v),"negative exact half minimum subnormal tie to even negative zero");
ck(cv({1.5f,-150},11.f,v)&&v==std::numeric_limits<float>::denorm_min(),"above half minimum subnormal retained");
ck(cv({1.f,-149},11.f,v)&&v==std::numeric_limits<float>::denorm_min(),"minimum subnormal retained");
ck(!cv({1.5f,-160},std::numeric_limits<float>::denorm_min(),v),"do not erase globally subnormal stiffness");
ck(!cv({1.f,128},11.f,v),"overflow still rejected");
ck(!cv({std::numeric_limits<float>::quiet_NaN(),0},11.f,v),"NaN still rejected");
ck(!cv({std::numeric_limits<float>::infinity(),0},11.f,v),"infinity still rejected");
ck(!f::narrow({1.5f,-160},v),"generic scalar narrowing stays strict");
ck(!f::narrow({1.f,128},v),"generic scalar overflow stays strict");
ck(cv({1.375f,3},11.f,v)&&v==11.f,"normal principal-scale tensor unchanged");
ck(cv({std::nextafter(1.f,2.f),-150},11.f,v)&&v==std::numeric_limits<float>::denorm_min(),"immediately above half minimum retains denormal");
ck(cv({-std::nextafter(1.f,2.f),-150},11.f,v)&&v==-std::numeric_limits<float>::denorm_min(),"negative immediately above half retains signed denormal");
ck(!cv({1.5f,-160},std::numeric_limits<float>::denorm_min(),v),"PSD subnormal scale may not borrow negative raw normal scale");
ck(cv({0.f,0},0.f,v)&&v==0.f,"fully zero PSD passes original narrowing");
printf("failures=%d\n",failures);return failures?1:0;}

