#pragma once
#include "restoring_context.hpp"
// Separate diagnostic storage, written only by actual force assembly. Never
// read by a physical operator, and no process pointer is embedded in a POD.
SM_INLINE_DEVICE_HOST void restoring_observe(const restoring_curve::CurveContext& ctx,unsigned kind,unsigned target,unsigned channel,float q,float effort,float energy,float dt,float reference_q,float* observations) {
 if(!observations)return;
 for(unsigned i=0;i<ctx.binding_count;++i){const CurveBinding b=ctx.bindings[i];if(b.kind==kind&&b.target_index==target&&b.channel==channel){observations[6*i]=q;observations[6*i+1]=effort;observations[6*i+2]=energy;observations[6*i+3]=dt;observations[6*i+4]=1.f;observations[6*i+5]=reference_q;}}
}
SM_INLINE_DEVICE_HOST float restoring_observed_length(const float d[3]) {
 float value=0.f;
 // Telemetry does not participate in force assembly; preserve the old cast's
 // overflow result rather than turn a diagnostic overflow into a solver fault.
 for(unsigned i=0;i<3;++i)if(!restoring_curve::device_f32::finite(d[i]))return -1.f;
 auto r=restoring_curve::device_f32::norm(d,3);
 if(!restoring_curve::device_f32::narrow(r,value))return r.m*restoring_curve::device_f32::shift(1.f,r.e);
 return value;
}
SM_INLINE_DEVICE_HOST float restoring_observed_effort(const float g[3]) {
 float v=restoring_curve::device_f32::abs(g[0]);for(unsigned k=1;k<3;++k){float x=restoring_curve::device_f32::abs(g[k]);if(x>v)v=x;}return v;
}
