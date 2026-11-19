#pragma once
#include "restoring_curve.hpp"
// Mirrors B revision2 frozen restoring_curve_gpu.rs; these PODs alone upload.
struct CurveDescriptor {
 unsigned node_offset;
 unsigned node_count;
 unsigned domain;
 unsigned tail;
 unsigned reserved[4];
};
struct CurveNode {
 float x;
 float effort;
 float area;
 float right_slope;
};
struct CurveBinding {
 unsigned kind;
 unsigned target_index;
 unsigned channel;
 unsigned table_index;
 unsigned history_index;
 unsigned receipt_index;
 float damping;
 float epsilon_m;
 float rest_length_m;
 unsigned reserved[3];
};
struct CurveHistory {
 float d0[3];
 float q0;
 unsigned long long epoch;
 unsigned initialized;
 unsigned reserved;
};
namespace restoring_curve {
// Local execution view ONLY. Construct from neutral entry's individual arrays;
// never upload this struct or its process pointers as ABI/POD.
struct CurveContext {
 const CurveDescriptor* descriptors; const CurveNode* nodes;
 const CurveBinding* bindings; const CurveHistory* history;
 unsigned descriptor_count,node_count,binding_count,history_count;
};
// Neutral positive square root: double intermediates without host-only <cmath>.
#if !defined(SM_MSL_CONCAT)
SM_INLINE_DEVICE_HOST double positive_sqrt(double x) {
 if(x==0.)return 0.;if(!finite(x)||x<0.)return -1.;
 double scale=1.,n=x;while(n>=4.){n*=.25;scale*=2.;}while(n<1.){n*=4.;scale*=.5;}
 double r=2.;for(unsigned k=0;k<8;++k)r=.5*(r+n/r);return r*scale;
}
#endif
enum : unsigned { RigidAngular=0,FemAngular=1,RigidAnchor=2,FemAnchor=3,Stitch=4,CrossLinkDistance=5,AnchorChannel=3,
 CurveFaultContext=160,CurveFaultBinding=161,CurveFaultScalar=162,CurveFaultHistory=163,CurveFaultArithmetic=164 };
// Failure-only telemetry in the existing six-float observation row. Values
// are copied as f32 so even NaN/Inf payload bits survive device readback.
SM_INLINE_DEVICE_HOST void scalar_fault_diagnostic(const CurveContext& c,const CurveBinding& b,float coordinate,float a0,float a1,float a2,unsigned subtype,Fault reason,float* observations) {
 if(!observations)return;unsigned row=c.binding_count;
 for(unsigned i=0;i<c.binding_count;++i)if(c.bindings+i==&b){row=i;break;}
 if(row>=c.binding_count)return;float* out=observations+6*row;
 out[0]=-162.f;out[1]=coordinate;out[2]=a0;out[3]=a1;out[4]=a2;
 out[5]=float(subtype*16u+static_cast<unsigned>(reason));
}
SM_INLINE_DEVICE_HOST const CurveBinding* lookup(const CurveContext& c,unsigned kind,unsigned target,unsigned channel,unsigned& fault) {
 fault=0;const CurveBinding* found=nullptr;
 if(c.binding_count&&!c.bindings){fault=CurveFaultContext;return nullptr;}
 for(unsigned i=0;i<c.binding_count;++i){const CurveBinding* b=c.bindings+i;if(b->kind==kind&&b->target_index==target&&b->channel==channel){if(found){fault=CurveFaultBinding;return nullptr;}found=b;}}
 return found;
}
SM_INLINE_DEVICE_HOST bool scalar_view(const CurveContext& c,const CurveBinding& binding,HostNode local[64],HostView& view) {
 if(!c.descriptors||!c.nodes||binding.table_index>=c.descriptor_count)return false;
 const CurveDescriptor d=c.descriptors[binding.table_index];
 if(d.node_count<2||d.node_count>64||d.node_offset>c.node_count||d.node_count>c.node_count-d.node_offset||d.domain>2||d.tail>1)return false;
 for(unsigned i=0;i<d.node_count;++i){const CurveNode n=c.nodes[d.node_offset+i];local[i]={n.x,n.effort,n.area};}
 const Domain domain=d.domain==0?Domain::Signed:(d.domain==1?Domain::Odd:Domain::Radial);
 view={local,d.node_count,domain,d.tail==0?Tail::Linear:Tail::HoldEffort};return true;
}
SM_INLINE_DEVICE_HOST Result evaluate_f32(const CurveContext& c,const CurveBinding& binding,float x,SM_THREAD device_f32::Scaled* angular_gradient=nullptr) {
 HostNode local[64];HostView view{};if(!scalar_view(c,binding,local,view))return fault(Fault::InvalidTable);
 return eval_f32(view,x,angular_gradient);
}
#if !defined(SM_MSL_CONCAT)
SM_INLINE_DEVICE_HOST Result evaluate_legacy(const CurveContext& c,const CurveBinding& binding,float x) {
 HostNode local[64];HostView view{};if(!scalar_view(c,binding,local,view))return fault(Fault::InvalidTable);
 return eval_legacy(view,x);
}
#endif
SM_INLINE_DEVICE_HOST Result evaluate(const CurveContext& c,const CurveBinding& binding,float x) { return evaluate_f32(c,binding,x); }
SM_INLINE_DEVICE_HOST const CurveHistory* history_for(const CurveContext& c,const CurveBinding& b,unsigned& fault) {
 if(!c.history||b.history_index>=c.history_count||c.history[b.history_index].initialized!=1){fault=CurveFaultHistory;return nullptr;}
 return c.history+b.history_index;
}
SM_INLINE_DEVICE_HOST bool angular_f32(const CurveContext& c,unsigned kind,unsigned target,unsigned axis,float x,float x0,float dt,float& e,float& g,float& h,unsigned& code,float* diagnostics=nullptr) {
 namespace f=device_f32;
 const CurveBinding* b=lookup(c,kind,target,axis,code);if(code||!b)return false;
 f::Scaled elastic_gradient{};
 const auto r=evaluate_f32(c,*b,x,&elastic_gradient);if(r.fault!=Fault::None){scalar_fault_diagnostic(c,*b,x,x0,0.f,0.f,1u,r.fault,diagnostics);code=CurveFaultScalar;return false;}
 if(!f::finite(dt)||dt<=0.f||!f::finite(b->damping)||b->damping<0.f||!f::finite(x0)){code=CurveFaultArithmetic;return false;}
 auto cd=f::div(f::scaled(b->damping),f::scaled(dt)),d=f::sub(f::scaled(x),f::scaled(x0));
 if(!f::narrow(f::add(f::scaled(r.energy),f::mul(f::scaled(.5f),f::mul(cd,f::square(d)))),e,true)||
    !f::narrow_angular_effort(f::add(elastic_gradient,f::mul(cd,d)),g)||
    !f::narrow(f::add(f::scaled(r.curvature),cd),h)){code=CurveFaultArithmetic;return false;}
 return true;
}
#if !defined(SM_MSL_CONCAT)
SM_INLINE_DEVICE_HOST bool angular_legacy(const CurveContext& c,unsigned kind,unsigned target,unsigned axis,float x,float x0,float dt,float& e,float& g,float& h,unsigned& code,float* diagnostics=nullptr) {
 const CurveBinding* b=lookup(c,kind,target,axis,code);if(code||!b)return false;
 const auto r=evaluate_legacy(c,*b,x);if(r.fault!=Fault::None){scalar_fault_diagnostic(c,*b,x,x0,0.f,0.f,1u,r.fault,diagnostics);code=CurveFaultScalar;return false;}
 if(!finite(dt)||dt<=0||!finite(b->damping)||b->damping<0){code=CurveFaultArithmetic;return false;}
 const double cd=double(b->damping)/dt,d=double(x)-x0;
 if(!narrow_energy(double(r.energy)+.5*cd*d*d,e)||!narrow(double(r.gradient)+cd*d,g)||!narrow(double(r.curvature)+cd,h)){code=CurveFaultArithmetic;return false;}
 return true;
}
#endif
SM_INLINE_DEVICE_HOST bool angular(const CurveContext& c,unsigned kind,unsigned target,unsigned axis,float x,float x0,float dt,float& e,float& g,float& h,unsigned& code,float* diagnostics=nullptr) { return angular_f32(c,kind,target,axis,x,x0,dt,e,g,h,code,diagnostics); }
SM_INLINE_DEVICE_HOST bool anchor_f32(const CurveContext& c,unsigned kind,unsigned target,const float d[3],float dt,float& energy,float gradient[3],float hessian[9],unsigned& code,float* diagnostics=nullptr) {
 namespace f=device_f32;
 const CurveBinding* b=lookup(c,kind,target,AnchorChannel,code);if(code||!b)return false;
 const CurveHistory* history=history_for(c,*b,code);if(!history)return false;
 if(!f::finite(dt)||dt<=0.f||!f::finite(b->damping)||b->damping<0.f){code=CurveFaultArithmetic;return false;}
 for(unsigned k=0;k<3;++k)if(!f::finite(d[k])||!f::finite(history->d0[k])){code=CurveFaultArithmetic;return false;}
 auto r=f::norm(d,3);float radius;
 if(!f::narrow(r,radius)){code=CurveFaultArithmetic;return false;}
 const auto value=evaluate_f32(c,*b,radius);
 if(value.fault!=Fault::None){scalar_fault_diagnostic(c,*b,radius,d[0],d[1],d[2],2u,value.fault,diagnostics);code=CurveFaultScalar;return false;}
 if(c.descriptors[b->table_index].domain!=2){code=CurveFaultBinding;return false;}
 auto cd=f::div(f::scaled(b->damping),f::scaled(dt)),e=f::scaled(value.energy);
 f::Scaled n[3];
 for(unsigned k=0;k<3;++k){
  n[k]=r.m!=0.f?f::div(f::scaled(d[k]),r):f::scaled(0.f);
  auto delta=f::sub(f::scaled(d[k]),f::scaled(history->d0[k]));
  e=f::add(e,f::mul(f::scaled(.5f),f::mul(cd,f::square(delta))));
  if(!f::narrow(f::add(f::mul(f::scaled(value.gradient),n[k]),f::mul(cd,delta)),gradient[k])){code=CurveFaultArithmetic;return false;}
 }
 if(!f::narrow(e,energy,true)){code=CurveFaultArithmetic;return false;}
 auto transverse=r.m!=0.f?f::div(f::scaled(value.gradient),r):f::scaled(value.curvature);
 for(unsigned k=0;k<3;++k)for(unsigned l=0;l<3;++l){
  auto nn=f::mul(n[k],n[l]);
  auto h=f::add(f::mul(f::scaled(value.curvature),nn),
      f::mul(transverse,f::sub(f::scaled(k==l?1.f:0.f),nn)));
  if(k==l)h=f::add(h,cd);
  if(!f::narrow(h,hessian[3*k+l])){code=CurveFaultArithmetic;return false;}
 }
 return true;
}
#if !defined(SM_MSL_CONCAT)
SM_INLINE_DEVICE_HOST bool anchor_legacy(const CurveContext& c,unsigned kind,unsigned target,const float d[3],float dt,float& energy,float gradient[3],float hessian[9],unsigned& code,float* diagnostics=nullptr) {
 const CurveBinding* b=lookup(c,kind,target,AnchorChannel,code);if(code||!b)return false;
 const CurveHistory* history=history_for(c,*b,code);if(!history)return false;
 if(!finite(dt)||dt<=0||!finite(b->damping)||b->damping<0){code=CurveFaultArithmetic;return false;}
 double norm2=0.;for(unsigned k=0;k<3;++k){if(!finite(d[k])||!finite(history->d0[k])){code=CurveFaultArithmetic;return false;}norm2+=double(d[k])*d[k];}
 const double r=positive_sqrt(norm2);float radius;if(!narrow(r,radius)){code=CurveFaultArithmetic;return false;}
 const auto value=evaluate_legacy(c,*b,radius);if(value.fault!=Fault::None){scalar_fault_diagnostic(c,*b,radius,d[0],d[1],d[2],2u,value.fault,diagnostics);code=CurveFaultScalar;return false;}
 if(c.descriptors[b->table_index].domain!=2){code=CurveFaultBinding;return false;}
 const double cd=double(b->damping)/dt;double e=value.energy;
 for(unsigned k=0;k<3;++k){double delta=double(d[k])-history->d0[k];e+=.5*cd*delta*delta;double spring=r>0 ? double(value.gradient)*d[k]/r:0.;if(!narrow(spring+cd*delta,gradient[k])){code=CurveFaultArithmetic;return false;}}
 if(!narrow_energy(e,energy)){code=CurveFaultArithmetic;return false;}
 const double transverse=r>0 ? double(value.gradient)/r:value.curvature;
 for(unsigned k=0;k<3;++k)for(unsigned l=0;l<3;++l){double nn=r>0 ? double(d[k])*d[l]/norm2:0.;double h=transverse*(k==l?1.:0.)+(double(value.curvature)-transverse)*nn+(k==l?cd:0.);if(!narrow(h,hessian[3*k+l])){code=CurveFaultArithmetic;return false;}}
 return true;
}
#endif
SM_INLINE_DEVICE_HOST bool anchor(const CurveContext& c,unsigned kind,unsigned target,const float d[3],float dt,float& energy,float gradient[3],float hessian[9],unsigned& code,float* diagnostics=nullptr) { return anchor_f32(c,kind,target,d,dt,energy,gradient,hessian,code,diagnostics); }
} // namespace restoring_curve
static_assert(sizeof(CurveDescriptor)==32&&alignof(CurveDescriptor)==4,"CurveDescriptor layout");
static_assert(sizeof(CurveNode)==16&&alignof(CurveNode)==4,"CurveNode layout");
static_assert(sizeof(CurveBinding)==48&&alignof(CurveBinding)==4,"CurveBinding layout");
static_assert(sizeof(CurveHistory)==32&&alignof(CurveHistory)==8,"CurveHistory layout");
