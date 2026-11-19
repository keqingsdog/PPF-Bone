#include "../energy/model/pdrd_lock_projector.kernel.cpp"
// One 6x6 block per independent unit: cloth uses its leading 3x3.
[[seam::entry(u)]]
[[seam::device_fn]] inline void rqc_freeze(
 const unsigned *row_to_vertex,const unsigned *dof_mask,
 const unsigned *joint_mode,const float *joint_axis,
 const unsigned *translation_lock,const float *translation_axis,const unsigned *translation_mode,
 const unsigned *rotation_lock,const float *rotation_axis,const unsigned *rotation_mode,
 const unsigned *root_mask,float *q,unsigned *fault,unsigned cloth,unsigned bodies,unsigned u,unsigned extent) {
 if(u>=extent)return;
 fault[u]=0u;float *out=q+36u*u;for(unsigned k=0;k<36u;++k)out[k]=0.f;
 if(u<cloth){if(dof_mask[row_to_vertex[u]]==0u)for(unsigned i=0;i<3u;++i)out[6u*i+i]=1.f;return;}
 const unsigned b=u-cloth;if(b>=bodies){fault[u]=80u;return;}
 if(root_mask[b]!=0u)return;
 float basis[PDRD_LOCK_BASIS_FLOATS];unsigned status=PDRD_LOCK_OK;
 const Vec3f a(joint_axis[3u*b],joint_axis[3u*b+1u],joint_axis[3u*b+2u]),t(translation_axis[3u*b],translation_axis[3u*b+1u],translation_axis[3u*b+2u]),w(rotation_axis[3u*b],rotation_axis[3u*b+1u],rotation_axis[3u*b+2u]);
 const unsigned n=pdrd_build_body_basis(joint_mode[b],a,translation_lock[b]!=PDRD_CLOTH_MARKER,t,translation_mode[b],rotation_lock[b]!=PDRD_CLOTH_MARKER,w,rotation_mode[b],basis,&status);
 if(status!=PDRD_LOCK_OK){fault[u]=81u;return;}
 for(unsigned j=0;j<6u;++j){float v[6]={0,0,0,0,0,0};v[j]=1.f;pdrd_apply_body_basis(basis,n,v);for(unsigned i=0;i<6u;++i){if(!(v[i]==v[i]) || fabs(v[i])>3.402823466e38f){fault[u]=82u;return;}out[6u*i+j]=v[i];}}
}
[[seam::host_device_fn]] inline unsigned rqc_unit(unsigned row,unsigned cloth){return row<cloth?row:cloth+(row-cloth)/2u;}
[[seam::host_device_fn]] inline unsigned rqc_axis(unsigned row,unsigned axis,unsigned cloth){return row<cloth?axis:3u*((row-cloth)%2u)+axis;}
[[seam::host_device_fn]] inline float rqc_q(const float *q,unsigned ra,unsigned ia,unsigned rb,unsigned ib,unsigned cloth){const unsigned a=rqc_unit(ra,cloth),b=rqc_unit(rb,cloth);return a==b?q[36u*a+6u*rqc_axis(ra,ia,cloth)+rqc_axis(rb,ib,cloth)]:0.f;}
[[seam::entry(domain)]]
[[seam::device_fn]] inline void rqc_local(
 const unsigned *member_offset,const unsigned *members,const float *q,float *dense,unsigned *fault,unsigned cloth,unsigned domain,unsigned extent){
 if(domain>=extent)return;if(fault[domain]!=0u)return;
 const unsigned first=member_offset[domain],n=member_offset[domain+1u]-first,d=3u*n;
 if(n==0u||n>16u){fault[domain]=83u;return;}
 for(unsigned a=0;a<n;++a){unsigned row=members[first+a];if(row>=cloth){unsigned peer=cloth+((row-cloth)^1u);bool found=false;for(unsigned b=0;b<n;++b)found=found||members[first+b]==peer;if(!found){fault[domain]=84u;return;}}}
 float temp[2304];float *a=dense+2304u*domain;
 for(unsigned i=0;i<d;++i)for(unsigned j=0;j<d;++j){float v=0.f;for(unsigned k=0;k<d;++k)if(rqc_unit(members[first+i/3u],cloth)==rqc_unit(members[first+k/3u],cloth))v+=rqc_q(q,members[first+i/3u],i%3u,members[first+k/3u],k%3u,cloth)*a[k*d+j];temp[i*d+j]=v;}
 for(unsigned i=0;i<d;++i)for(unsigned j=0;j<d;++j){float v=0.f;for(unsigned k=0;k<d;++k)if(rqc_unit(members[first+k/3u],cloth)==rqc_unit(members[first+j/3u],cloth))v+=temp[i*d+k]*rqc_q(q,members[first+k/3u],k%3u,members[first+j/3u],j%3u,cloth);v+=(i==j?1.f:0.f)-rqc_q(q,members[first+i/3u],i%3u,members[first+j/3u],j%3u,cloth);if(!(v==v)||fabs(v)>3.402823466e38f){fault[domain]=85u;return;}a[i*d+j]=v;}
}
[[seam::entry(u)]]
[[seam::device_fn]] inline void rqc_modes(
 const unsigned *owner,const float *q,float *modes,unsigned *fault,unsigned cloth,unsigned bodies,unsigned u,unsigned extent){
 if(u>=extent)return;const unsigned row=u<cloth?u:cloth+2u*(u-cloth),d=u<cloth?3u:6u;
 if(u>=cloth && (u-cloth>=bodies||owner[row]!=owner[row+1u])){fault[owner[row]]=86u;return;}
 float out[36];for(unsigned i=0;i<d;++i)for(unsigned j=0;j<6u;++j){float v=0.f;for(unsigned k=0;k<d;++k)v+=q[36u*u+6u*i+k]*modes[18u*(row+k/3u)+6u*(k%3u)+j];out[6u*i+j]=v;if(!(v==v)||fabs(v)>3.402823466e38f){fault[owner[row]]=87u;return;}}
 for(unsigned i=0;i<d;++i)for(unsigned j=0;j<6u;++j)modes[18u*(row+i/3u)+6u*(i%3u)+j]=out[6u*i+j];
}
