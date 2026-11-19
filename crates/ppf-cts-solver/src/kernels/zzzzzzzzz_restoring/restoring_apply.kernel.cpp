#pragma once
#include "../zz_joint/ball_anchor.kernel.cpp"
#include "../zzzzzzzz_fem_joint/fem_joint.kernel.cpp"
#include "../energy/model/restoring_stitch.hpp"
// One gathered displacement is used by accepted-history capture and all
// candidate operators. Form each barycentric endpoint relative to the same
// centroid so translating the whole scene does not alter damping history.
[[seam::host_device_fn]] inline bool restoring_stitch_displacement(const float* xyz,unsigned vertex_count,const unsigned* indices,const float* weights,unsigned row,float d[3]) {
 float x[18],cog[3]={},a[3]={},b[3]={};
 for(unsigned s=0;s<6;++s){unsigned v=indices[6*row+s];if(v>=vertex_count)return false;for(unsigned k=0;k<3;++k){x[3*s+k]=xyz[3*v+k];if(!restoring_curve::finite(x[3*s+k]))return false;cog[k]+=x[3*s+k]/6.f;}}
 for(unsigned s=0;s<6;++s){float w=weights[6*row+s];if(!restoring_curve::finite(w))return false;for(unsigned k=0;k<3;++k){float v=w*(x[3*s+k]-cog[k]);if(s<3)a[k]+=v;else b[k]+=v;}}
 for(unsigned k=0;k<3;++k){d[k]=a[k]-b[k];if(!restoring_curve::finite(d[k]))return false;}return true;
}
// Appended entries preserve every existing dispatch ID and legacy declaration.
[[seam::entry(i)]]
[[seam::device_fn]] inline void joint_restoring_angular_apply(const AngularControlGpu* records,const AngularTwistEvalContext* twist_context,const MaterialSpinContext* material_spin_context,unsigned record_count,const PdrdRigidState* state,const PdrdRigidState* reference,unsigned bodies,unsigned base,unsigned dim,float dt,unsigned action,const float* input,compute::atomic_float_t* output,unsigned* fault,const CurveDescriptor* descriptors,const CurveNode* nodes,const CurveBinding* bindings,const CurveHistory* history,unsigned descriptor_count,unsigned node_count,unsigned binding_count,unsigned history_count,float* observations,unsigned i) {
 if(i>=record_count)return;const AngularControlGpu row=records[i];if(!joint_angular_control::row_requested(row)){if(action==3u||action==5u)fault[i]=0;return;}
 if(action==5u){
  const MaterialSpinContext accepted=material_spin_context[i];float q[4],T[9];
  if(accepted.model==0u){fault[i]=0u;return;}
  if(accepted.model!=3u || !joint_angular_control::material_relative_pose(row,state,bodies,q,T)){fault[i]=joint_angular_control::CONTROL_FAULT_MATERIAL_INVALID;return;}
  material_spin_v3::Trial trial{};const unsigned status=material_spin_v3::evaluate(accepted.accepted_q_wxyz,accepted.accepted_a,q,trial);
  if(status!=material_spin_v3::OK){fault[i]=status==material_spin_v3::LOCAL_STEP_AMBIGUOUS?joint_angular_control::CONTROL_FAULT_MATERIAL_LOCAL_STEP:joint_angular_control::CONTROL_FAULT_MATERIAL_INVALID;return;}
  for(unsigned k=0;k<4;++k)compute::atomic_add(output+7u*i+k,trial.accepted_candidate_q[k]);
  for(unsigned k=0;k<3;++k)compute::atomic_add(output+7u*i+4u+k,trial.strain[k]);
  fault[i]=0u;return;
 }
 const restoring_curve::CurveContext ctx{descriptors,nodes,bindings,history,descriptor_count,node_count,binding_count,history_count};
 const MaterialSpinContext material=material_spin_context[i];
 float c[3],c0[3],j[18],j0[18],g[3],h[3],e;unsigned code=material.model==0u
  ? joint_angular_control::resolve(row,state,reference,bodies,c,j,c0,j0,twist_context+i)
  : joint_angular_control::resolve_material(row,state,bodies,material,c,j,c0);
 if(!code&&!joint_angular_control::curve_weights(row,ctx,restoring_curve::RigidAngular,i,c,c0,dt,g,h,&e,code,action==0u?observations:nullptr)){fault[i]=code;return;}
 unsigned a=0,b=0;if(!code&&row.body_a!=joint_angular_control::CONTROL_WORLD_ENDPOINT&&!joint_angular_control::reduced_offset(base,row.body_a,dim,&a))code=joint_angular_control::CONTROL_FAULT_REDUCED_RANGE;
 if(!code&&row.body_b!=joint_angular_control::CONTROL_WORLD_ENDPOINT&&!joint_angular_control::reduced_offset(base,row.body_b,dim,&b))code=joint_angular_control::CONTROL_FAULT_REDUCED_RANGE;
 if(code){fault[i]=code;return;}if(action==3u){fault[i]=0;return;}if(action==4u){compute::atomic_add(output,e);return;}
 if(action==2u){for(unsigned side=0;side<2;++side){unsigned body=side?row.body_b:row.body_a;if(body==joint_angular_control::CONTROL_WORLD_ENDPOINT)continue;for(unsigned r=0;r<3;++r)for(unsigned s=0;s<3;++s){float v=0;for(unsigned k=0;k<3;++k)v+=j[6*k+3*side+r]*h[k]*j[6*k+3*side+s];compute::atomic_add(output+36*body+6*(3+r)+3+s,v);}}return;}
 float axis[3]={};if(action==1u){float x[6]={};for(unsigned k=0;k<3;++k){if(row.body_a!=joint_angular_control::CONTROL_WORLD_ENDPOINT)x[k]=input[a+3+k];if(row.body_b!=joint_angular_control::CONTROL_WORLD_ENDPOINT)x[k+3]=input[b+3+k];}for(unsigned r=0;r<3;++r){for(unsigned k=0;k<6;++k)axis[r]+=j[6*r+k]*x[k];axis[r]*=h[r];}}
 else for(unsigned r=0;r<3;++r)axis[r]=g[r];
 for(unsigned k=0;k<6;++k){float v=0;for(unsigned r=0;r<3;++r)v+=j[6*r+k]*axis[r];if(k<3&&row.body_a!=joint_angular_control::CONTROL_WORLD_ENDPOINT)compute::atomic_add(output+a+3+k,v);if(k>=3&&row.body_b!=joint_angular_control::CONTROL_WORLD_ENDPOINT)compute::atomic_add(output+b+k,v);}
}
[[seam::entry(i)]]
[[seam::device_fn]] inline void joint_restoring_fem_apply(const FemJointGpu* records,const MaterialSpinContext* material_spin_context,unsigned record_count,const FemFrameGpu* cache,const unsigned* vertex_to_free,unsigned vertex_count,unsigned free_count,unsigned rigid_base,unsigned dim,float dt,unsigned action,const float* input,compute::atomic_float_t* output,compute::atomic_float_t* nodal_blocks,compute::atomic_float_t* rigid_blocks,unsigned* fault,const CurveDescriptor* descriptors,const CurveNode* nodes,const CurveBinding* bindings,const CurveHistory* history,unsigned descriptor_count,unsigned node_count,unsigned binding_count,unsigned history_count,float* observations,unsigned i) {
 if(i>=record_count)return;const FemJointGpu row=records[i];
 const MaterialSpinContext material=material_spin_context[i];
 if(action==8u||action==9u){joint_fem::material_sample(row,cache,material,action,output,fault,i);return;}
 if(!row.enabled){if(action==3u)fault[i]=0;return;}
 unsigned idx[24];if(!joint_fem::mapping(row.a,vertex_to_free,vertex_count,free_count,rigid_base,dim,idx)||!joint_fem::mapping(row.b,vertex_to_free,vertex_count,free_count,rigid_base,dim,idx+12)){fault[i]=joint_fem::BAD_MAPPING;return;}
 const restoring_curve::CurveContext ctx{descriptors,nodes,bindings,history,descriptor_count,node_count,binding_count,history_count};float g[24],h[576],e;unsigned code;if(!joint_fem::evaluate(row,cache+4*i,dt,g,h,e,code,&ctx,i,action==0u?observations:nullptr,observations,&material)){fault[i]=code;return;}if(action==3u){fault[i]=0;return;}joint_fem::scatter(action,row,idx,g,h,e,input,output,nodal_blocks,rigid_blocks);
}
[[seam::host_device_fn]] inline bool restoring_ball_geometry(const BallAnchorGpu& row,const PdrdRigidState* state,unsigned bodies,float p[3],float q[3],float d[3]) {
 const bool wa=row.body_a==joint_ball::WORLD_ENDPOINT,wb=row.body_b==joint_ball::WORLD_ENDPOINT;
 if(!row.enabled||(wa&&wb)||(!wa&&row.body_a>=bodies)||(!wb&&row.body_b>=bodies)||!joint_ball::finite3(row.anchor_a)||!joint_ball::finite3(row.anchor_b))return false;
 if(wa)for(unsigned k=0;k<3;++k)p[k]=row.anchor_a[k];else joint_ball::rot(state[row.body_a].R,row.anchor_a,p);
 if(wb)for(unsigned k=0;k<3;++k)q[k]=row.anchor_b[k];else joint_ball::rot(state[row.body_b].R,row.anchor_b,q);
 for(unsigned k=0;k<3;++k)d[k]=(wa?p[k]:state[row.body_a].anchor[k]+state[row.body_a].x[k]+p[k])-(wb?q[k]:state[row.body_b].anchor[k]+state[row.body_b].x[k]+q[k]);return true;
}
[[seam::entry(i)]]
[[seam::device_fn]] inline void joint_restoring_ball_apply(const BallAnchorGpu* records,unsigned record_count,const PdrdRigidState* state,unsigned bodies,unsigned base,unsigned dim,float dt,unsigned action,const float* input,compute::atomic_float_t* output,unsigned* fault,const CurveDescriptor* descriptors,const CurveNode* nodes,const CurveBinding* bindings,const CurveHistory* history,unsigned descriptor_count,unsigned node_count,unsigned binding_count,unsigned history_count,float* observations,unsigned i) {
 if(i>=record_count)return;const restoring_curve::CurveContext ctx{descriptors,nodes,bindings,history,descriptor_count,node_count,binding_count,history_count};unsigned code=0;const auto* binding=restoring_curve::lookup(ctx,restoring_curve::RigidAnchor,i,3,code);
 if(code){fault[i]=code;return;}if(!binding){if(action==0u)joint_ball::gradient_kernel(records,record_count,state,bodies,base,dim,output,i);else if(action==1u)joint_ball::matvec_kernel(records,record_count,state,bodies,base,dim,input,output,i);else if(action==2u)joint_ball::blocks_kernel(records,record_count,state,bodies,output,i);else if(action==4u)joint_ball::energy_kernel(records,record_count,state,bodies,output,i);else fault[i]=0;return;}
 const BallAnchorGpu row=records[i];float p[3],q[3],d[3],e,g[3],h[9];if(!restoring_ball_geometry(row,state,bodies,p,q,d)){fault[i]=restoring_curve::CurveFaultBinding;return;}
 if(!restoring_curve::anchor(ctx,restoring_curve::RigidAnchor,i,d,dt,e,g,h,code)){fault[i]=code;return;}
 unsigned a=0,b=0;const bool wa=row.body_a==joint_ball::WORLD_ENDPOINT,wb=row.body_b==joint_ball::WORLD_ENDPOINT;
 if((!wa&&!joint_ball::reduced_offset(base,row.body_a,dim,&a))||(!wb&&!joint_ball::reduced_offset(base,row.body_b,dim,&b))){fault[i]=restoring_curve::CurveFaultBinding;return;}
 if(action==3u){fault[i]=0;return;}if(action==4u){compute::atomic_add(output,e);return;}
 if(action==0u){const auto* prior=restoring_curve::history_for(ctx,*binding,code);restoring_observe(ctx,restoring_curve::RigidAnchor,i,3,restoring_observed_length(d),restoring_observed_effort(g),e,dt,prior?restoring_observed_length(prior->d0):0.f,observations);}
 if(action==2u){for(unsigned side=0;side<2;++side){unsigned body=side?row.body_b:row.body_a;if(body==joint_ball::WORLD_ENDPOINT)continue;for(unsigned col=0;col<6;++col){float u[6]={},v[3],w[3]={},z[6];u[col]=1;joint_ball::L(side?q:p,u,v);for(unsigned k=0;k<3;++k)for(unsigned l=0;l<3;++l)w[k]+=h[3*k+l]*v[l];joint_ball::lt(side?q:p,w,z);for(unsigned k=0;k<6;++k)compute::atomic_add(output+36*body+6*k+col,z[k]);}}return;}
 float w[3];for(unsigned k=0;k<3;++k)w[k]=g[k];if(action==1u){float ua[6]={},ub[6]={},delta[3];if(!wa){for(unsigned k=0;k<6;++k)ua[k]=input[a+k];}if(!wb){for(unsigned k=0;k<6;++k)ub[k]=input[b+k];}joint_ball::L_pair(p,ua,q,ub,delta);for(unsigned k=0;k<3;++k){w[k]=0;for(unsigned l=0;l<3;++l)w[k]+=h[3*k+l]*delta[l];}}
 float z[6];if(!wa){joint_ball::lt(p,w,z);for(unsigned k=0;k<6;++k)compute::atomic_add(output+a+k,z[k]);}if(!wb){joint_ball::lt(q,w,z);for(unsigned k=0;k<6;++k)compute::atomic_add(output+b+k,-z[k]);}
}
[[seam::entry(i)]]
[[seam::device_fn]] inline void restoring_history_capture(const CurveBinding* bindings,unsigned binding_count,CurveHistory* history,unsigned history_count,const BallAnchorGpu* ball,unsigned ball_count,const FemFrameGpu* fem_cache,unsigned fem_count,const PdrdRigidState* reference,unsigned bodies,const float* accepted_xyz,unsigned vertex_count,const unsigned* stitch_indices,const float* stitch_weights,unsigned stitch_count,unsigned epoch_low,unsigned epoch_high,unsigned* fault,unsigned i) {
 if(i>=binding_count)return;const CurveBinding b=bindings[i];fault[i]=0;if(b.kind<2)return;if(b.history_index>=history_count){fault[i]=restoring_curve::CurveFaultHistory;return;}CurveHistory next{};next.epoch=static_cast<unsigned long long>(epoch_low)+(static_cast<unsigned long long>(epoch_high)<<32);next.initialized=1;
 if(b.kind==2){if(b.target_index>=ball_count){fault[i]=restoring_curve::CurveFaultBinding;return;}float p[3],q[3];if(!restoring_ball_geometry(ball[b.target_index],reference,bodies,p,q,next.d0)){fault[i]=restoring_curve::CurveFaultBinding;return;}}
 else if(b.kind==3||b.kind==5){if(b.target_index>=fem_count){fault[i]=restoring_curve::CurveFaultBinding;return;}const FemFrameGpu a=fem_cache[4*b.target_index],c=fem_cache[4*b.target_index+1];if(a.status||c.status){fault[i]=restoring_curve::CurveFaultHistory;return;}for(unsigned k=0;k<3;++k)next.d0[k]=a.anchor[k]-c.anchor[k];if(b.kind==5){unsigned code=0;if(!restoring_stitch::coordinate(next.d0,b.epsilon_m,b.rest_length_m,next.q0,code)){fault[i]=code;return;}}}
 else if(b.kind==4){if(b.target_index>=stitch_count){fault[i]=restoring_curve::CurveFaultBinding;return;}float d[3];if(!restoring_stitch_displacement(accepted_xyz,vertex_count,stitch_indices,stitch_weights,b.target_index,d)){fault[i]=restoring_curve::CurveFaultBinding;return;}for(unsigned k=0;k<3;++k)next.d0[k]=d[k];unsigned code=0;if(!restoring_stitch::coordinate(d,b.epsilon_m,b.rest_length_m,next.q0,code)){fault[i]=code;return;}}
 else{fault[i]=restoring_curve::CurveFaultBinding;return;}history[b.history_index]=next;
}
// Overlay only active rows after the unchanged legacy gather. The resulting
// gradient/Hessian buffers feed the existing scatter, Hv and self blocks.
[[seam::entry(i)]]
[[seam::device_fn]] inline void restoring_stitch_apply(const float* xyz,unsigned vertex_count,const unsigned* stitch_indices,const float* stitch_weights,unsigned stitch_count,float dt,unsigned action,Mat3x6f* gradient,Mat18x18f* hessian,compute::atomic_float_t* energy,unsigned* fault,const CurveDescriptor* descriptors,const CurveNode* nodes,const CurveBinding* bindings,const CurveHistory* history,unsigned descriptor_count,unsigned node_count,unsigned binding_count,unsigned history_count,float* observations,unsigned i) {
 if(i>=stitch_count)return;const restoring_curve::CurveContext ctx{descriptors,nodes,bindings,history,descriptor_count,node_count,binding_count,history_count};unsigned code=0;const auto* binding=restoring_curve::lookup(ctx,restoring_curve::Stitch,i,3,code);if(code){fault[i]=code;return;}if(!binding){fault[i]=0;return;}
 float d[3],e,g[3],h[9];if(!restoring_stitch_displacement(xyz,vertex_count,stitch_indices,stitch_weights,i,d)){fault[i]=restoring_curve::CurveFaultBinding;return;}if(!restoring_stitch::evaluate(ctx,i,d,dt,e,g,h,code)){fault[i]=code?code:restoring_curve::CurveFaultBinding;return;}fault[i]=0;if(action==3u)return;if(action==4u){compute::atomic_add(energy,e);return;}
 if(action==0u){float q=0.f;unsigned code=0;restoring_stitch::coordinate(d,binding->epsilon_m,binding->rest_length_m,q,code);const auto* prior=restoring_curve::history_for(ctx,*binding,code);restoring_observe(ctx,restoring_curve::Stitch,i,3,q,restoring_observed_effort(g),e,dt,prior?prior->q0:0.f,observations);}
 Mat3x6f result_g;Mat18x18f result_h;for(unsigned a=0;a<6;++a){float wa=(a<3?1.f:-1.f)*stitch_weights[6*i+a];for(unsigned k=0;k<3;++k)result_g(k,a)=wa*g[k];for(unsigned b=0;b<6;++b){float wb=(b<3?1.f:-1.f)*stitch_weights[6*i+b];for(unsigned k=0;k<3;++k)for(unsigned l=0;l<3;++l)result_h(3*a+k,3*b+l)=wa*wb*h[3*k+l];}}gradient[i]=result_g;hessian[i]=result_h;
}

