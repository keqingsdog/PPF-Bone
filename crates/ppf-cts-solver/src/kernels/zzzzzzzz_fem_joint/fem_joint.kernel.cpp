// Material tet joint endpoint; no ghost mass, rigid projection, or new angle law.
#pragma once
#include "../zz_joint/angular_joint_v2.kernel.cpp"
#include "../zz_joint_ext/angular_extension.kernel.cpp"
#include "../energy/model/restoring_stitch.hpp"
#include "../float_math.hpp"
#include "../utility/svd3x3.kernel.cpp"

struct FemEndpointGpu {
    unsigned kind;
    unsigned id;
    unsigned vertices[4];
    float dm_inv[9];
    float anchor_bary[4];
    float center_bary[4];
    float rest_rotation[9];
    float local_anchor[3];
};
struct FemJointGpu {
    FemEndpointGpu a;
    FemEndpointGpu b;
    unsigned enabled;
    float ball_stiffness;
    AngularJointGpuV2 v2;
    AngularControlGpu control;
};
struct FemFrameGpu {
    float rotation[9];
    float anchor[3];
    float center[3];
    float rotation_jacobian[36];
    unsigned status;
};
namespace joint_fem {
enum : unsigned { WORLD=0u, RIGID=1u, TET=2u, NONE=0xffffffffu,
    OK=0u, BAD_ENDPOINT=256u, BAD_REST=257u, BAD_CURRENT=258u,
    BAD_POLAR=259u, BAD_MAPPING=260u, BAD_STEP=261u, BAD_ARITHMETIC=262u };
[[seam::host_device_fn]] inline bool finite(float v) { return !fmath::isnan(v)&&!fmath::isinf(v); }
[[seam::host_device_fn]] inline void identity(float a[9]) { for(int k=0;k<9;++k)a[k]=0.f;a[0]=a[4]=a[8]=1.f; }
// All 3x3 storage here is column major. All Jacobians are row major.
[[seam::host_device_fn]] inline void mm(const float a[9],const float b[9],float c[9]) {
    float t[9]={};for(int j=0;j<3;++j)for(int i=0;i<3;++i)for(int k=0;k<3;++k)t[3*j+i]+=a[3*k+i]*b[3*j+k];for(int k=0;k<9;++k)c[k]=t[k];
}
[[seam::host_device_fn]] inline void tm(const float a[9],const float b[9],float c[9]) {
    for(int j=0;j<3;++j)for(int i=0;i<3;++i){c[3*j+i]=0.f;for(int k=0;k<3;++k)c[3*j+i]+=a[3*i+k]*b[3*j+k];}
}
[[seam::host_device_fn]] inline float det(const float a[9]) {return a[0]*(a[4]*a[8]-a[7]*a[5])-a[3]*(a[1]*a[8]-a[7]*a[2])+a[6]*(a[1]*a[5]-a[4]*a[2]);}
[[seam::host_device_fn]] inline bool inv(const float a[9],float o[9]) {
    float d=det(a);if(!finite(d)||!(d>0.f))return false;
    o[0]=(a[4]*a[8]-a[7]*a[5])/d;o[3]=(a[6]*a[5]-a[3]*a[8])/d;o[6]=(a[3]*a[7]-a[6]*a[4])/d;
    o[1]=(a[7]*a[2]-a[1]*a[8])/d;o[4]=(a[0]*a[8]-a[6]*a[2])/d;o[7]=(a[6]*a[1]-a[0]*a[7])/d;
    o[2]=(a[1]*a[5]-a[4]*a[2])/d;o[5]=(a[3]*a[2]-a[0]*a[5])/d;o[8]=(a[0]*a[4]-a[3]*a[1])/d;
    for(int k=0;k<9;++k)if(!finite(o[k]))return false;return true;
}
[[seam::host_device_fn]] inline bool proper(const float r[9]) {
    float s[9];tm(r,r,s);float err=0.f;for(int j=0;j<3;++j)for(int i=0;i<3;++i){if(!finite(r[3*j+i]))return false;err=fmath::max(err,fmath::abs(s[3*j+i]-(i==j?1.f:0.f)));}return err<2e-5f&&fmath::abs(det(r)-1.f)<2e-5f;
}
// These bit helpers validate no-op proofs and the sensitive correction; a no-op requires both complete polar
// certificates passed. Nonzero subnormals, FTZ underflow and overflow are
// untrusted; signed zero remains a legitimate exact zero.
[[seam::host_device_fn]] inline bool noop_bad(float x) {
    const unsigned u=fmath::float_bits(x)&0x7fffffffu;
    return u>=0x7f800000u||((u&0x7f800000u)==0u&&(u&0x007fffffu)!=0u);
}
[[seam::host_device_fn]] inline bool noop_nonzero(float x){return (fmath::float_bits(x)&0x7fffffffu)!=0u;}
[[seam::host_device_fn]] inline bool noop_same(float a,float b){
    const unsigned x=fmath::float_bits(a),y=fmath::float_bits(b);
    return x==y||(!(x&0x7fffffffu)&&!(y&0x7fffffffu));
}
[[seam::host_device_fn]] inline bool noop_mul(float a,float b,float& out){
    if(noop_bad(a)||noop_bad(b))return false;volatile float p=a*b;out=p;
    return !noop_bad(out)&&!(noop_nonzero(a)&&noop_nonzero(b)&&!noop_nonzero(out));
}
[[seam::host_device_fn]] inline bool noop_add(float a,float b,float& out){
    if(noop_bad(a)||noop_bad(b))return false;volatile float p=a+b;out=p;return !noop_bad(out);
}
[[seam::host_device_fn]] inline bool noop_up(float x,float& y){
    const unsigned u=fmath::float_bits(x);if(noop_bad(x)||!noop_nonzero(x)||(u&0x80000000u))return false;
    y=fmath::float_from_bits(u+1u);return !noop_bad(y)&&noop_nonzero(y);
}
[[seam::host_device_fn]] inline bool noop_down(float x,float& y){
    const unsigned u=fmath::float_bits(x);if(noop_bad(x)||!noop_nonzero(x)||(u&0x80000000u))return false;
    y=fmath::float_from_bits(u-1u);return !noop_bad(y)&&noop_nonzero(y);
}
[[seam::host_device_fn]] inline bool noop_interval(const float oldr[9],const float candidate[9]){
    float upper=0.f,lowernorm=0.f;
    for(int k=0;k<9;++k){
        if(noop_bad(oldr[k])||noop_bad(candidate[k]))return false;
        if(!noop_same(oldr[k],candidate[k])){
            volatile float delta=candidate[k]-oldr[k];if(noop_bad(delta)||!noop_nonzero(delta))return false;
            const float ad=fmath::abs(delta);float up=0.f,sq=0.f,squp=0.f,next=0.f;
            if(!noop_up(ad,up)||!noop_mul(up,up,sq)||!noop_up(sq,squp))return false;
            if(!noop_nonzero(upper))upper=squp;else if(!noop_add(upper,squp,next)||!noop_up(next,upper))return false;
        }
        if(noop_nonzero(oldr[k])){
            float sq=0.f,sqdown=0.f,next=0.f;
            if(!noop_mul(fmath::abs(oldr[k]),fmath::abs(oldr[k]),sq)||!noop_down(sq,sqdown))return false;
            if(!noop_nonzero(lowernorm))lowernorm=sqdown;else if(!noop_add(lowernorm,sqdown,next)||!noop_down(next,lowernorm))return false;
        }
    }
    if(!noop_nonzero(lowernorm))return false;
    float threshold=0.f,lower=0.f;
    if(!noop_mul(1.4210854715202004e-14f,lowernorm,threshold)||!noop_down(threshold,lower))return false;
    return upper<=lower;
}
[[seam::host_device_fn]] inline bool noop_allowed(const float f[9],const float r[9],const float s[9],
    const float candidate[9],const float stretch[9],const float cki[9],const float w[3],float w2,float oldki[9]){
    if(noop_bad(w2)||w2<0.f||w2>1e-6f)return false;
    bool anyw=false;for(int k=0;k<3;++k){if(noop_bad(w[k]))return false;anyw=anyw||noop_nonzero(w[k]);}
    if(!anyw||!noop_nonzero(w2))return false;
    for(int k=0;k<9;++k)if(noop_bad(f[k])||noop_bad(r[k])||noop_bad(s[k])||noop_bad(candidate[k])||noop_bad(stretch[k])||noop_bad(cki[k]))return false;
    if(!proper(r))return false;
    float maxs=0.f;for(int k=0;k<9;++k)maxs=fmath::max(maxs,fmath::abs(s[k]));
    if(!(maxs>0.f)||noop_bad(maxs))return false;
    for(int j=0;j<3;++j)for(int i=0;i<3;++i)if(fmath::abs(s[3*j+i]-s[3*i+j])>2e-5f*maxs)return false;
    if(!(s[0]>0.f&&s[0]*s[4]-s[3]*s[1]>0.f&&det(s)>0.f))return false;
    float tr=s[0]+s[4]+s[8],kmat[9];for(int j=0;j<3;++j)for(int i=0;i<3;++i)kmat[3*j+i]=(i==j?tr:0.f)-s[3*j+i];
    if(!(kmat[0]>0.f&&kmat[0]*kmat[4]-kmat[3]*kmat[1]>0.f)||!inv(kmat,oldki))return false;
    for(int k=0;k<9;++k)if(noop_bad(oldki[k]))return false;
    if(!noop_interval(r,candidate))return false;
    const float unit=5.9604644775390625e-8f;float sixu=0.f,gamma=0.f,twogamma=0.f;
    if(!noop_mul(6.f,unit,sixu))return false;
    volatile float denominator=1.f-sixu;gamma=sixu/denominator;
    if(noop_bad(denominator)||noop_bad(gamma)||!noop_mul(2.f,gamma,twogamma))return false;
    const int pairs[3][2]={{0,1},{0,2},{1,2}};
    for(int p=0;p<3;++p){const int i=pairs[p][0],j=pairs[p][1];float sum=0.f;
        for(int k=0;k<3;++k){float x=0.f,y=0.f,next=0.f;
            if(!noop_mul(fmath::abs(r[3*i+k]),fmath::abs(f[3*j+k]),x)||!noop_mul(fmath::abs(r[3*j+k]),fmath::abs(f[3*i+k]),y)||!noop_add(sum,x,next))return false;sum=next;
            if(!noop_add(sum,y,next))return false;sum=next;
        }
        float bound=0.f;if(!noop_mul(twogamma,sum,bound))return false;
        volatile float delta=s[3*j+i]-s[3*i+j];if(noop_bad(delta)||(!noop_same(s[3*j+i],s[3*i+j])&&!noop_nonzero(delta)))return false;
        if(!(fmath::abs(delta)<=bound))return false;
    }
    return true;
}
enum RefineResult : unsigned { REFINE_REJECTED=0u, REFINE_APPLIED=1u, REFINE_NOOP=2u };
// One bounded stationarity correction for an otherwise proper frame whose
// measured R^T F narrowly fails the existing symmetry certificate. H is only
// an equation coefficient; the accepted stretch is always recomputed from F.
[[seam::host_device_fn]] inline RefineResult refine_stationarity(const float f[9],float r[9],float s[9],float ki[9]) {
    for(int k=0;k<9;++k)if(!finite(f[k]))return REFINE_REJECTED;
    const float fd=det(f);if(!finite(fd)||!(fd>0.f))return REFINE_REJECTED;
    float maxs=0.f;for(int k=0;k<9;++k){if(!finite(s[k]))return REFINE_REJECTED;maxs=fmath::max(maxs,fmath::abs(s[k]));}
    if(!(maxs>0.f)||!finite(maxs))return REFINE_REJECTED;
    float n[9],h[9],kmat[9],inverse[9];for(int k=0;k<9;++k){n[k]=s[k]/maxs;if(!finite(n[k]))return REFINE_REJECTED;}
    for(int j=0;j<3;++j)for(int i=0;i<3;++i){h[3*j+i]=.5f*(n[3*j+i]+n[3*i+j]);if(!finite(h[3*j+i]))return REFINE_REJECTED;}
    const float trace=h[0]+h[4]+h[8];if(!finite(trace))return REFINE_REJECTED;
    for(int j=0;j<3;++j)for(int i=0;i<3;++i){kmat[3*j+i]=(i==j?trace:0.f)-h[3*j+i];if(!finite(kmat[3*j+i]))return REFINE_REJECTED;}
    const float hd=det(h),kd=det(kmat);
    if(!finite(hd)||!finite(kd)||!(h[0]>0.f&&h[0]*h[4]-h[3]*h[1]>0.f&&hd>0.f)
       ||!(kmat[0]>0.f&&kmat[0]*kmat[4]-kmat[3]*kmat[1]>0.f&&kd>0.f)||!inv(kmat,inverse))return REFINE_REJECTED;
        float b[3]={},w[3]={};float skew=0.f;
    const int lhs[3]={5,6,1},rhs[3]={7,2,3};
    if(noop_bad(maxs))return REFINE_REJECTED;
    for(int i=0;i<3;++i){
        const float left=s[lhs[i]],right=s[rhs[i]];
        if(noop_bad(left)||noop_bad(right))return REFINE_REJECTED;
        volatile float delta=left-right;
        if(noop_bad(delta)||(!noop_same(left,right)&&!noop_nonzero(delta)))return REFINE_REJECTED;
        volatile float scaled_delta=delta/maxs;
        if(noop_bad(scaled_delta)||(noop_nonzero(delta)&&!noop_nonzero(scaled_delta)))return REFINE_REJECTED;
        b[i]=scaled_delta;
    }
    for(int i=0;i<3;++i){if(!finite(b[i]))return REFINE_REJECTED;skew=fmath::max(skew,fmath::abs(b[i]));for(int j=0;j<3;++j)w[i]+=inverse[3*j+i]*b[j];}
    float t2=0.f;for(int i=0;i<3;++i){if(!finite(w[i]))return REFINE_REJECTED;t2+=w[i]*w[i];}
    if(!finite(skew)||!(skew>0.f&&skew<=1e-3f)||!finite(t2)||!(t2<=1e-6f))return REFINE_REJECTED;
    float W[9]={0.f,w[2],-w[1],-w[2],0.f,w[0],w[1],-w[0],0.f},W2[9],q[9],candidate[9],stretch[9],candidate_ki[9];
    mm(W,W,W2);const float t4=t2*t2,a=1.f-t2/6.f+t4/120.f,bb=.5f-t2/24.f+t4/720.f;
    for(int k=0;k<9;++k)q[k]=a*W[k]+bb*W2[k];q[0]+=1.f;q[4]+=1.f;q[8]+=1.f;
    mm(r,q,candidate);if(!proper(candidate))return REFINE_REJECTED;tm(candidate,f,stretch);
    float newmax=0.f,newskew=0.f;for(int k=0;k<9;++k){if(!finite(stretch[k]))return REFINE_REJECTED;newmax=fmath::max(newmax,fmath::abs(stretch[k]));}
    if(!(newmax>0.f)||!finite(newmax))return REFINE_REJECTED;
    for(int j=0;j<3;++j)for(int i=0;i<3;++i){const float diff=fmath::abs(stretch[3*j+i]-stretch[3*i+j]);if(diff>2e-5f*newmax)return REFINE_REJECTED;newskew=fmath::max(newskew,diff);}
    if(!finite(newskew))return REFINE_REJECTED;const bool strict_improvement=newskew/newmax<skew;
    if(!(stretch[0]>0.f&&stretch[0]*stretch[4]-stretch[3]*stretch[1]>0.f&&det(stretch)>0.f))return REFINE_REJECTED;
    const float tr=stretch[0]+stretch[4]+stretch[8];float final_k[9];for(int j=0;j<3;++j)for(int i=0;i<3;++i)final_k[3*j+i]=(i==j?tr:0.f)-stretch[3*j+i];
    if(!(final_k[0]>0.f&&final_k[0]*final_k[4]-final_k[3]*final_k[1]>0.f)||!inv(final_k,candidate_ki))return REFINE_REJECTED;
    if(!strict_improvement){
        float oldki[9];if(!noop_allowed(f,r,s,candidate,stretch,candidate_ki,w,t2,oldki))return REFINE_REJECTED;
        for(int k=0;k<9;++k)ki[k]=oldki[k];return REFINE_NOOP;
    }
    for(int k=0;k<9;++k){r[k]=candidate[k];s[k]=stretch[k];ki[k]=candidate_ki[k];}return REFINE_APPLIED;
}
// Sensitive correction uses checked RN f32 expansions and a conservative fold-error bound; it does not claim correctly rounded dot products.
struct EcExpansion { float x[32]; unsigned n; };
[[seam::host_device_fn]] inline bool ec_add(float a,float b,float& out){if(noop_bad(a)||noop_bad(b))return false;volatile float v=a+b;if(noop_bad(v)||(!noop_same(a,-b)&&!noop_nonzero(v)))return false;out=v;return true;}
[[seam::host_device_fn]] inline bool ec_sub(float a,float b,float& out){if(noop_bad(a)||noop_bad(b))return false;volatile float v=a-b;if(noop_bad(v)||(!noop_same(a,b)&&!noop_nonzero(v)))return false;out=v;return true;}
[[seam::host_device_fn]] inline bool ec_mul(float a,float b,float& out){if(noop_bad(a)||noop_bad(b))return false;volatile float v=a*b;if(noop_bad(v)||(noop_nonzero(a)&&noop_nonzero(b)&&!noop_nonzero(v)))return false;out=v;return true;}
[[seam::host_device_fn]] inline bool ec_sum(float a,float b,float& s,float& e){float bb,t1,t2,t3;return ec_add(a,b,s)&&ec_sub(s,a,bb)&&ec_sub(s,bb,t1)&&ec_sub(a,t1,t2)&&ec_sub(b,bb,t3)&&ec_add(t2,t3,e);}
[[seam::host_device_fn]] inline bool ec_split(float a,float& h,float& l){float c,t;return ec_mul(4097.f,a,c)&&ec_sub(c,a,t)&&ec_sub(c,t,h)&&ec_sub(a,h,l);}
[[seam::host_device_fn]] inline bool ec_prod(float a,float b,float& p,float& e){float ah,al,bh,bl,t0,t1,t2,t3,t4;return ec_mul(a,b,p)&&ec_split(a,ah,al)&&ec_split(b,bh,bl)&&ec_mul(ah,bh,t0)&&ec_sub(t0,p,t1)&&ec_mul(ah,bl,t2)&&ec_add(t1,t2,t3)&&ec_mul(al,bh,t2)&&ec_add(t3,t2,t4)&&ec_mul(al,bl,t2)&&ec_add(t4,t2,e);}
[[seam::host_device_fn]] inline bool ec_grow(EcExpansion& e,float x){EcExpansion o{};float q=x;for(unsigned i=0;i<e.n;++i){float s,h;if(!ec_sum(q,e.x[i],s,h))return false;if(noop_nonzero(h)){if(o.n>=32)return false;o.x[o.n++]=h;}q=s;}if(noop_nonzero(q)||o.n==0){if(o.n>=32)return false;o.x[o.n++]=q;}e=o;return true;}
[[seam::host_device_fn]] inline bool ec_nextup(float x,float& out){unsigned b=fmath::float_bits(x);if(noop_bad(x)||!noop_nonzero(x)||(b>>31)||b>=0x7f7fffffu)return false;out=fmath::float_from_bits(b+1u);return !noop_bad(out);}
[[seam::host_device_fn]] inline bool ec_bound(const EcExpansion& e,unsigned cap,float& out){if(e.n==0||e.n>cap)return false;float total=0.f;for(unsigned i=0;i<e.n;++i){float x=fmath::float_from_bits(fmath::float_bits(e.x[i])&0x7fffffffu);if(noop_bad(x))return false;if(!noop_nonzero(x))continue;if(!noop_nonzero(total)){total=x;continue;}float v;if(!ec_add(total,x,v)||!ec_nextup(v,total))return false;}if(e.n==1||!noop_nonzero(total)){out=0.f;return true;}float c=float(e.n-1)*1.1920928955078125e-7f,v;if(!ec_mul(c,total,v)||!ec_nextup(v,out))return false;return true;}
[[seam::host_device_fn]] inline bool ec_dot(const float* a,const float* b,const float* c,const float* d,bool difference,bool minus_one,float& value){EcExpansion e{};for(int q=0;q<3;++q){float p,lo;if(!ec_prod(a[q],b[q],p,lo)||!ec_grow(e,lo)||!ec_grow(e,p))return false;if(difference){if(!ec_prod(c[q],d[q],p,lo)||!ec_grow(e,-lo)||!ec_grow(e,-p))return false;}}if(minus_one&&!ec_grow(e,-1.f))return false;value=0.f;for(unsigned q=0;q<e.n;++q){float v;if(!ec_add(value,e.x[q],v))return false;value=v;}float bound;return ec_bound(e,difference?12u:8u,bound);}
[[seam::host_device_fn]] inline bool ec_mm(const float a[9],const float b[9],float out[9],bool transpose_a){for(int j=0;j<3;++j)for(int i=0;i<3;++i){float sum=0.f;for(int k=0;k<3;++k){float p,n;int ix=transpose_a?3*i+k:3*k+i;if(!ec_mul(a[ix],b[3*j+k],p)||!ec_add(sum,p,n))return false;sum=n;}out[3*j+i]=sum;}return true;}
[[seam::host_device_fn]] inline bool ec_rn(){volatile float one=1.f,minus_one=-1.f,offset=8.94069671630859375e-8f;volatile float p=one+offset,n=minus_one-offset;return fmath::float_bits(p)==0x3f800001u && fmath::float_bits(n)==0xbf800001u;}
[[seam::host_device_fn]] inline bool ec_correct(const float f[9],float r[9]){
 if(!ec_rn())return false;
 if(!proper(r))return false;float e[9],s[9],es[9],ste[9],h[9],kmat[9],ki[9],b[3],w[3]={};
 for(int j=0;j<3;++j)for(int i=0;i<3;++i)if(!ec_dot(r+3*i,r+3*j,nullptr,nullptr,false,i==j,e[3*j+i]))return false;
 if(!ec_mm(r,f,s,true)||!ec_mm(e,s,es,false)||!ec_mm(s,e,ste,true))return false;
 float maxs=0.f;for(int q=0;q<9;++q){if(noop_bad(s[q]))return false;maxs=fmath::max(maxs,fmath::abs(s[q]));}if(!(maxs>0.f)||noop_bad(maxs))return false;
 const int left[3]={2,0,1},right[3]={1,2,0};for(int z=0;z<3;++z){int i=left[z],j=right[z];float raw,diff,half_term,value;if(!ec_dot(r+3*i,f+3*j,r+3*j,f+3*i,true,false,raw)||!ec_sub(es[3*j+i],ste[3*j+i],diff)||!ec_mul(.5f,diff,half_term)||!ec_sub(raw,half_term,value))return false;volatile float scaled=value/maxs;if(noop_bad(scaled)||(noop_nonzero(value)&&!noop_nonzero(scaled)))return false;b[z]=scaled;}
 float tmp[9];for(int q=0;q<9;++q){float half_term,v;if(!ec_mul(.5f,es[q],half_term)||!ec_sub(s[q],half_term,v))return false;volatile float scaled=v/maxs;if(noop_bad(scaled)||(noop_nonzero(v)&&!noop_nonzero(scaled)))return false;tmp[q]=scaled;}
 for(int j=0;j<3;++j)for(int i=0;i<3;++i){float sum;if(!ec_add(tmp[3*j+i],tmp[3*i+j],sum)||!ec_mul(.5f,sum,h[3*j+i]))return false;}
 float trace=h[0]+h[4]+h[8];if(noop_bad(trace))return false;for(int j=0;j<3;++j)for(int i=0;i<3;++i)kmat[3*j+i]=(i==j?trace:0.f)-h[3*j+i];
 if(!(h[0]>0.f&&h[0]*h[4]-h[3]*h[1]>0.f&&det(h)>0.f&&kmat[0]>0.f&&kmat[0]*kmat[4]-kmat[3]*kmat[1]>0.f&&det(kmat)>0.f&&inv(kmat,ki)))return false;
 float skew=0.f,t2=0.f;for(int i=0;i<3;++i){skew=fmath::max(skew,fmath::abs(b[i]));for(int j=0;j<3;++j)w[i]+=ki[3*j+i]*b[j];if(!finite(w[i]))return false;t2+=w[i]*w[i];}if(!finite(t2)||!(skew<=1e-3f&&t2<=1e-6f))return false;
 float W[9]={0.f,w[2],-w[1],-w[2],0.f,w[0],w[1],-w[0],0.f},W2[9],dq[9],edq[9],delta[9],rd[9];if(!ec_mm(W,W,W2,false))return false;float aa=1.f-t2/6.f+t2*t2/120.f,bb=.5f-t2/24.f+t2*t2/720.f;for(int q=0;q<9;++q)dq[q]=aa*W[q]+bb*W2[q];if(!ec_mm(e,dq,edq,false))return false;for(int q=0;q<9;++q)delta[q]=dq[q]-.5f*e[q]-.5f*edq[q];if(!ec_mm(r,delta,rd,false))return false;float candidate[9];for(int q=0;q<9;++q){candidate[q]=r[q]+rd[q];if(!finite(candidate[q]))return false;}for(int q=0;q<9;++q)r[q]=candidate[q];return true;
}

[[seam::host_device_fn]] inline bool positive_polar(const float f[9],float r[9],float s[9],float ki[9]) {
    float scale=0.f;for(int k=0;k<9;++k){if(!finite(f[k]))return false;scale=fmath::max(scale,fmath::abs(f[k]));}
    if(!(scale>0.f)||!(det(f)>0.f))return false;
    // Newton polar iteration evaluates the frame only. It NEVER changes x.
    for(int k=0;k<9;++k)r[k]=f[k]/scale;
    bool sensitive=false;
    for(int it=0;it<32;++it){float ri[9],next[9],change=0.f;if(!inv(r,ri))return false;
        float ma=0.f,mi=0.f,sa=0.f,si=0.f;
        if(it==0 || sensitive){
            for(int k=0;k<9;++k){ma=fmath::max(ma,fmath::abs(r[k]));mi=fmath::max(mi,fmath::abs(ri[k]));}
            if(!(ma>0.f&&mi>0.f)||!finite(ma)||!finite(mi))return false;
            for(int k=0;k<9;++k){const float a=r[k]/ma,b=ri[k]/mi;sa+=a*a;si+=b*b;}
            if(!(sa>0.f&&si>0.f)||!finite(sa)||!finite(si))return false;
            if(it==0){const float risk=(((ma*1.1920928955078125e-7f)*mi)*fmath::sqrt(sa))*fmath::sqrt(si);if(!finite(risk))return false;sensitive=risk>=2e-5f;}
        }
        float mu=1.f;
        if(sensitive){mu=(fmath::sqrt(mi)/fmath::sqrt(ma))*fmath::sqrt(fmath::sqrt(si/sa));if(!(mu>0.f)||!finite(mu))return false;}
        for(int j=0;j<3;++j)for(int i=0;i<3;++i){int k=3*j+i;next[k]=sensitive?.5f*(mu*r[k]+ri[3*i+j]/mu):.5f*(r[k]+ri[3*i+j]);if(!finite(next[k]))return false;change=fmath::max(change,fmath::abs(next[k]-r[k]));}
        for(int k=0;k<9;++k)r[k]=next[k];if(change<2e-7f)break;
    }
    if(sensitive&&!ec_correct(f,r))return false;
    if(!proper(r))return false;tm(r,f,s);
    // Symmetry + Sylvester SPD certificate; no clamping, abs(det), or S repair.
    float maxs=0.f;for(int k=0;k<9;++k)maxs=fmath::max(maxs,fmath::abs(s[k]));
    bool symmetric=true;float skew=0.f;
    for(int j=0;j<3;++j)for(int i=0;i<3;++i){const float diff=fmath::abs(s[3*j+i]-s[3*i+j]);skew=fmath::max(skew,diff);if(diff>2e-5f*maxs)symmetric=false;}
    if(sensitive||!symmetric){
        if(sensitive&&symmetric&&skew==0.f){
            // A flushed nonzero difference is not an exact zero skew.
            for(int k=0;k<9;++k)if(noop_bad(f[k])||noop_bad(r[k])||noop_bad(s[k]))return false;
            for(int j=0;j<3;++j)for(int i=0;i<3;++i)if(!noop_same(s[3*j+i],s[3*i+j]))return false;
        } else { return refine_stationarity(f,r,s,ki)!=REFINE_REJECTED; }
    }
    if(!(s[0]>0.f&&s[0]*s[4]-s[3]*s[1]>0.f&&det(s)>0.f))return false;
    float trace=s[0]+s[4]+s[8],kmat[9];for(int j=0;j<3;++j)for(int i=0;i<3;++i)kmat[3*j+i]=(i==j?trace:0.f)-s[3*j+i];
    if(!(kmat[0]>0.f&&kmat[0]*kmat[4]-kmat[3]*kmat[1]>0.f))return false;
    return inv(kmat,ki);
}
// Closest proper rotation, including rank-two and orientation-reversing F.
// The elastic SNHK model accepts inverted tets, so their material frame must
// use the same real geometry rather than reflecting or clamping it. The SVD
// orders sigma increasingly. At rank two its smallest left vector is zero;
// the cross product of the two resolved directions supplies the unique proper
// orientation. Rank one and non-unique nearest rotations are rejected.
[[seam::host_device_fn]] inline bool proper_svd_polar(const float f[9],float r[9],float s[9],float ki[9]) {
    float scale=0.f;for(int i=0;i<9;++i){if(!finite(f[i]))return false;scale=fmath::max(scale,fmath::abs(f[i]));}
    if(!(scale>0.f)||!finite(scale))return false;
    Mat3x3f a,u,vt;Vec3f sv;
    for(int j=0;j<3;++j)for(int i=0;i<3;++i)a(i,j)=f[3*j+i]/scale;
    svd3x3(a,u,sv,vt);
    for(int i=0;i<3;++i)if(!finite(sv[i]))return false;
    if(!(sv[2]>0.f&&sv[1]>2e-6f*sv[2]))return false;
    float vdet=vt.determinant();if(!finite(vdet)||fmath::abs(vdet)<.9f)return false;
    float x[3],y[3],z[3],xn=0.f;
    for(int i=0;i<3;++i){x[i]=u(i,1);xn+=x[i]*x[i];}if(!(xn>0.f)||!finite(xn))return false;
    xn=fmath::sqrt(xn);float dot=0.f;for(int i=0;i<3;++i){x[i]/=xn;dot+=x[i]*u(i,2);}
    float yn=0.f;for(int i=0;i<3;++i){y[i]=u(i,2)-dot*x[i];yn+=y[i]*y[i];}if(!(yn>0.f)||!finite(yn))return false;
    yn=fmath::sqrt(yn);for(int i=0;i<3;++i)y[i]/=yn;
    z[0]=x[1]*y[2]-x[2]*y[1];z[1]=x[2]*y[0]-x[0]*y[2];z[2]=x[0]*y[1]-x[1]*y[0];
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)r[3*j+i]=(vdet<0.f?-z[i]:z[i])*vt(0,j)+x[i]*vt(1,j)+y[i]*vt(2,j);
    if(!proper(r))return false;
    tm(r,f,s);float maxs=0.f;
    for(int i=0;i<9;++i){if(!finite(s[i]))return false;maxs=fmath::max(maxs,fmath::abs(s[i]));}
    if(!(maxs>0.f)||!finite(maxs))return false;
    float h[9],km[9];for(int i=0;i<3;++i)for(int j=0;j<3;++j){
        if(fmath::abs(s[3*j+i]-s[3*i+j])>2e-5f*maxs)return false;
        // Symmetrize only the equation coefficient, never the actual stretch.
        h[3*j+i]=.5f*(s[3*j+i]/maxs+s[3*i+j]/maxs);
    }
    const float tr=h[0]+h[4]+h[8];for(int i=0;i<3;++i)for(int j=0;j<3;++j)km[3*j+i]=(i==j?tr:0.f)-h[3*j+i];
    // K's positive definiteness certifies a locally unique closest SO(3)
    // frame. Scale-relative margins reject ill-conditioned/non-unique cases.
    if(!(km[0]>1e-6f&&km[0]*km[4]-km[3]*km[1]>1e-6f&&det(km)>1e-6f))return false;
    float kn[9];if(!inv(km,kn))return false;
    for(int i=0;i<9;++i){ki[i]=kn[i]/maxs;if(!finite(ki[i]))return false;}
    return true;
}
[[seam::host_device_fn]] inline bool polar(const float f[9],float r[9],float s[9],float ki[9]) {
    if(positive_polar(f,r,s,ki))return true;
    return proper_svd_polar(f,r,s,ki);
}
[[seam::host_device_fn]] inline unsigned frame(const FemEndpointGpu& e,const float* xyz,unsigned nv,const PdrdRigidState* rigid,unsigned nr,FemFrameGpu& o) {
    for(int k=0;k<36;++k)o.rotation_jacobian[k]=0.f;identity(o.rotation);
    if(e.kind==WORLD){for(int k=0;k<3;++k){if(!finite(e.local_anchor[k]))return BAD_ENDPOINT;o.anchor[k]=e.local_anchor[k];o.center[k]=0.f;}return OK;}
    if(e.kind==RIGID){if(e.id>=nr)return BAD_ENDPOINT;const PdrdRigidState s=rigid[e.id];if(!proper(s.R))return BAD_CURRENT;for(int k=0;k<9;++k)o.rotation[k]=s.R[k];for(int k=0;k<3;++k){o.center[k]=s.anchor[k]+s.x[k];o.anchor[k]=o.center[k];if(!finite(o.center[k])||!finite(e.local_anchor[k]))return BAD_CURRENT;for(int c=0;c<3;++c)o.anchor[k]+=s.R[3*c+k]*e.local_anchor[c];o.rotation_jacobian[12*k+3+k]=1.f;}return OK;}
    if(e.kind!=TET||!proper(e.rest_rotation))return BAD_REST;
    float suma=0.f,sumc=0.f;for(int v=0;v<4;++v){if(e.vertices[v]>=nv)return BAD_ENDPOINT;for(int w=0;w<v;++w)if(e.vertices[v]==e.vertices[w])return BAD_REST;if(!finite(e.anchor_bary[v])||!finite(e.center_bary[v]))return BAD_REST;suma+=e.anchor_bary[v];sumc+=e.center_bary[v];}
    // Reference winding is a convention; the material frame below certifies
    // a proper rotation and a unique derivative for the actual F.
    const float detDmInv=det(e.dm_inv);
    if(fmath::abs(suma-1.f)>2e-5f||fmath::abs(sumc-1.f)>2e-5f||!finite(detDmInv)||detDmInv==0.f)return BAD_REST;
    float dc[9],f[9],r[9],s[9],ki[9];for(int c=0;c<3;++c)for(int k=0;k<3;++k)dc[3*c+k]=xyz[3*e.vertices[c+1]+k]-xyz[3*e.vertices[0]+k];mm(dc,e.dm_inv,f);if(!polar(f,r,s,ki))return BAD_POLAR;mm(r,e.rest_rotation,o.rotation);
    for(int k=0;k<3;++k){o.anchor[k]=0.f;o.center[k]=0.f;for(int v=0;v<4;++v){float x=xyz[3*e.vertices[v]+k];if(!finite(x))return BAD_CURRENT;o.anchor[k]+=e.anchor_bary[v]*x;o.center[k]+=e.center_bary[v]*x;}}
    for(int v=0;v<4;++v)for(int axis=0;axis<3;++axis){float df[9]={},b[9],vee[3],omega[3]={};for(int c=0;c<3;++c){float grad=v==0?-(e.dm_inv[3*c]+e.dm_inv[3*c+1]+e.dm_inv[3*c+2]):e.dm_inv[3*c+v-1];df[3*c+axis]=grad;}tm(r,df,b);vee[0]=b[5]-b[7];vee[1]=b[6]-b[2];vee[2]=b[1]-b[3];for(int i=0;i<3;++i)for(int j=0;j<3;++j)omega[i]+=ki[3*j+i]*vee[j];for(int i=0;i<3;++i){float w=0.f;for(int j=0;j<3;++j)w+=r[3*j+i]*omega[j];o.rotation_jacobian[12*i+3*v+axis]=w;}}
    for(int k=0;k<36;++k)if(!finite(o.rotation_jacobian[k]))return BAD_ARITHMETIC;return OK;
}
[[seam::host_device_fn]] inline void pose_view(const FemFrameGpu& f,PdrdRigidState& s) {s=PdrdRigidState{};for(int k=0;k<9;++k)s.R[k]=f.rotation[k];for(int k=0;k<3;++k)s.x[k]=f.center[k];}
[[seam::host_device_fn]] inline void anchor_jacobian(const FemEndpointGpu& e,const FemFrameGpu& f,float j[36]) {
    for(int k=0;k<36;++k)j[k]=0.f;
    if(e.kind==TET){for(int v=0;v<4;++v)for(int k=0;k<3;++k)j[12*k+3*v+k]=e.anchor_bary[v];}
    else if(e.kind==RIGID){float p[3];for(int k=0;k<3;++k){p[k]=f.anchor[k]-f.center[k];j[12*k+k]=1.f;}j[5]=-p[1];j[4]=p[2];j[17]=p[0];j[15]=-p[2];j[27]=p[1];j[28]=-p[0];}
}
[[seam::host_device_fn]] inline bool mapping(const FemEndpointGpu& e,const unsigned* vf,unsigned nv,unsigned nf,unsigned base,unsigned dim,unsigned index[12]) {
    for(int k=0;k<12;++k)index[k]=NONE;
    if(e.kind==WORLD)return true;
    if(e.kind==RIGID){if(base>dim||e.id>=(dim-base)/6u)return false;for(int k=0;k<6;++k)index[k]=base+6*e.id+k;return true;}
    if(e.kind!=TET)return false;
    for(int v=0;v<4;++v){if(e.vertices[v]>=nv)return false;if(vf[e.vertices[v]]==NONE)continue;if(vf[e.vertices[v]]>=nf||base<3u||vf[e.vertices[v]]>(base-3u)/3u)return false;for(int k=0;k<3;++k)index[3*v+k]=3*vf[e.vertices[v]]+k;}return true;
}
[[seam::host_device_fn]] inline void add_angle(const float ag[6],const float ah[36],const FemFrameGpu& a,const FemFrameGpu& b,float g[24],float h[576]) {
    float j[144]={};for(int r=0;r<3;++r)for(int k=0;k<12;++k){j[24*r+k]=a.rotation_jacobian[12*r+k];j[24*(r+3)+12+k]=b.rotation_jacobian[12*r+k];}
    for(int k=0;k<24;++k){for(int r=0;r<6;++r)g[k]+=j[24*r+k]*ag[r];for(int l=0;l<24;++l)for(int r=0;r<6;++r)for(int s=0;s<6;++s)h[24*k+l]+=j[24*r+k]*ah[6*r+s]*j[24*s+l];}
}
// Cache lives strictly after the original 4*N FemFrame prefix. Byte copies
// preserve u32 token bits (>24 bits included) without float conversion or
// aliasing a float/FemFrame object as unsigned. Scalar payload remains f32.
[[seam::host_device_fn]] inline unsigned char* hessian_tail(const FemFrameGpu* cache,unsigned n) {
    return reinterpret_cast<unsigned char*>(const_cast<FemFrameGpu*>(cache))+4ull*n*sizeof(FemFrameGpu);
}
[[seam::host_device_fn]] inline void hessian_bytes_write(unsigned char* p,const unsigned char* v) {for(int k=0;k<4;++k)p[k]=v[k];}
[[seam::host_device_fn]] inline void hessian_write_float(unsigned char* p,float v) {hessian_bytes_write(p,reinterpret_cast<const unsigned char*>(&v));}
[[seam::host_device_fn]] inline float hessian_read_float(const unsigned char* p) {float v;unsigned char* b=reinterpret_cast<unsigned char*>(&v);for(int k=0;k<4;++k)b[k]=p[k];return v;}
[[seam::host_device_fn]] inline unsigned hessian_read_token(const unsigned char* p) {unsigned v;unsigned char* b=reinterpret_cast<unsigned char*>(&v);for(int k=0;k<4;++k)b[k]=p[k];return v;}
[[seam::host_device_fn]] inline unsigned char* hessian_row(const FemFrameGpu* cache,unsigned n,unsigned i,bool extension) {
    return hessian_tail(cache,n)+8ull*n+4ull*576*(i+(extension?n:0u));
}
[[seam::host_device_fn]] inline void hessian_store(const FemFrameGpu* cache,unsigned n,unsigned i,bool extension,unsigned token,float dt,const float* h) {
    unsigned char* meta=hessian_tail(cache,n)+8ull*i;
    hessian_bytes_write(meta,reinterpret_cast<const unsigned char*>(&token));hessian_write_float(meta+4,dt);
    unsigned char* out=hessian_row(cache,n,i,extension);for(int k=0;k<576;++k)hessian_write_float(out+4*k,h?h[k]:0.f);
}
[[seam::host_device_fn]] inline bool hessian_matches(const FemFrameGpu* cache,unsigned n,unsigned i,unsigned token,float dt) {
    const unsigned char* meta=hessian_tail(cache,n)+8ull*i;
    return token!=0u && token==hessian_read_token(meta) && dt==hessian_read_float(meta+4) && finite(dt)&&dt>0.f;
}
[[seam::device_fn]] inline void hessian_matvec(const FemFrameGpu* cache,unsigned n,unsigned i,bool extension,const unsigned idx[24],const float* input,compute::atomic_float_t* output) {
    const unsigned char* h=hessian_row(cache,n,i,extension);
    // Exact old k/l order and += chain. Base and extension remain distinct
    // entry launches and atomic additions; no pre-sum, reproject or symmetry.
    for(int k=0;k<24;++k)if(idx[k]!=NONE){float value=0.f;for(int l=0;l<24;++l)if(idx[l]!=NONE)value+=hessian_read_float(h+4*(24*k+l))*input[idx[l]];compute::atomic_add(output+idx[k],value);}
}


[[seam::host_device_fn]] inline bool evaluate(const FemJointGpu& row,const FemFrameGpu* cache,float dt,float g[24],float h[576],[[seam::thread]] float& energy,[[seam::thread]] unsigned& fault,const restoring_curve::CurveContext* curves=nullptr,unsigned target_index=0,float* observations=nullptr,float* diagnostics=nullptr,[[seam::thread]] const MaterialSpinContext* material=nullptr) {
    for(int k=0;k<24;++k)g[k]=0.f;for(int k=0;k<576;++k)h[k]=0.f;energy=0.f;fault=OK;if(!row.enabled)return true;
    for(int k=0;k<4;++k)if(cache[k].status!=OK){fault=cache[k].status;return false;}
    if(!(dt>0.f)||!finite(dt)){fault=BAD_STEP;return false;}
    if(!finite(row.ball_stiffness)||row.ball_stiffness<0.f){fault=BAD_ARITHMETIC;return false;}
    float aj[36],bj[36],j[72],res[3];anchor_jacobian(row.a,cache[0],aj);anchor_jacobian(row.b,cache[1],bj);
    for(int r=0;r<3;++r){res[r]=cache[0].anchor[r]-cache[1].anchor[r];energy+=.5f*row.ball_stiffness*res[r]*res[r];for(int k=0;k<12;++k){j[24*r+k]=aj[12*r+k];j[24*r+12+k]=-bj[12*r+k];}}
    float anchor_g[3],anchor_h[9],anchor_e=0.f;unsigned curve_code=0;bool active_anchor=false;
    if(curves)active_anchor=restoring_curve::anchor(*curves,restoring_curve::FemAnchor,target_index,res,dt,anchor_e,anchor_g,anchor_h,curve_code,diagnostics);
    const CurveBinding* distance_binding=nullptr;
    unsigned anchor_kind=restoring_curve::FemAnchor;
    float observed_coordinate=restoring_observed_length(res);
    if(curves&&!curve_code){
        distance_binding=restoring_curve::lookup(*curves,restoring_curve::CrossLinkDistance,target_index,3,curve_code);
        if(distance_binding&&!curve_code){
            if(active_anchor){fault=restoring_curve::CurveFaultBinding;return false;}
            restoring_stitch::Evaluation distance{};
            if(!restoring_stitch::evaluate_binding_f32(*curves,distance_binding,res,dt,distance,curve_code)){fault=curve_code?curve_code:restoring_curve::CurveFaultBinding;return false;}
            anchor_e=distance.energy;for(int k=0;k<3;++k)anchor_g[k]=distance.gradient[k];for(int k=0;k<9;++k)anchor_h[k]=distance.projected_hessian[k];
            observed_coordinate=distance.coordinate;anchor_kind=restoring_curve::CrossLinkDistance;active_anchor=true;
        }
    }
    if(curve_code){fault=curve_code;return false;}
    if(active_anchor){energy=anchor_e;unsigned code=0;const auto* binding=restoring_curve::lookup(*curves,anchor_kind,target_index,3,code);const auto* history=restoring_curve::history_for(*curves,*binding,code);restoring_observe(*curves,anchor_kind,target_index,3,observed_coordinate,restoring_observed_effort(anchor_g),anchor_e,dt,history?(anchor_kind==restoring_curve::CrossLinkDistance?history->q0:restoring_observed_length(history->d0)):0.f,observations);for(int k=0;k<24;++k){for(int r=0;r<3;++r)g[k]+=j[24*r+k]*anchor_g[r];for(int l=0;l<24;++l)for(int r=0;r<3;++r)for(int s=0;s<3;++s)h[24*k+l]+=j[24*r+k]*anchor_h[3*r+s]*j[24*s+l];}}
    else{for(int k=0;k<24;++k){for(int r=0;r<3;++r)g[k]+=row.ball_stiffness*j[24*r+k]*res[r];for(int l=0;l<24;++l)for(int r=0;r<3;++r)h[24*k+l]+=row.ball_stiffness*j[24*r+k]*j[24*r+l];}}
    PdrdRigidState cur[2],ref[2];pose_view(cache[0],cur[0]);pose_view(cache[1],cur[1]);pose_view(cache[2],ref[0]);pose_view(cache[3],ref[1]);
    AngularJointGpuV2 v2=row.v2;v2.body_a=row.a.kind==WORLD?NONE:0u;v2.body_b=row.b.kind==WORLD?NONE:1u;
    if(v2.enabled&&(v2.stiffness_Nm_per_rad!=0.f||v2.damping_Nms_per_rad!=0.f||v2.max_angle_rad!=0.f)){if(!joint_angular::valid(v2,2)){fault=BAD_ARITHMETIC;return false;}float ag[6]={},ah[36]={},phi[3],m[9],q[3],hp[9],grad[3]={},hh[9]={};
        if(v2.stiffness_Nm_per_rad>0.f||v2.max_angle_rad>0.f){if(!joint_angular::frame_jacobian(v2,cur,ref,NONE,false,phi,m)){fault=BAD_CURRENT;return false;}joint_angular::angular_potential(v2,phi,q,hp);for(int k=0;k<3;++k)for(int r=0;r<3;++r)grad[k]+=m[3*r+k]*q[r];for(int k=0;k<3;++k)for(int l=0;l<3;++l)for(int r=0;r<3;++r)for(int s=0;s<3;++s)hh[3*k+l]+=m[3*r+k]*hp[3*r+s]*m[3*s+l];float theta=fmath::sqrt(phi[0]*phi[0]+phi[1]*phi[1]+phi[2]*phi[2]);energy+=.5f*v2.stiffness_Nm_per_rad*theta*theta;if(v2.max_angle_rad>0.f&&theta>v2.max_angle_rad){float ex=theta-v2.max_angle_rad;energy+=.5f*fmath::max(2000.f,10.f*v2.stiffness_Nm_per_rad)*ex*ex;}}
        if(v2.damping_Nms_per_rad>0.f){if(!joint_angular::frame_jacobian(v2,cur,ref,NONE,true,phi,m)){fault=BAD_CURRENT;return false;}float cd=v2.damping_Nms_per_rad/dt;for(int k=0;k<3;++k){energy+=.5f*cd*phi[k]*phi[k];for(int r=0;r<3;++r)grad[k]+=cd*m[3*r+k]*phi[r];for(int l=0;l<3;++l)for(int r=0;r<3;++r)hh[3*k+l]+=cd*m[3*r+k]*m[3*r+l];}}
        for(int k=0;k<3;++k){ag[k]=-grad[k];ag[3+k]=grad[k];for(int l=0;l<3;++l){ah[6*k+l]=hh[3*k+l];ah[6*(k+3)+l+3]=hh[3*k+l];ah[6*k+l+3]=-hh[3*k+l];ah[6*(k+3)+l]=-hh[3*k+l];}}add_angle(ag,ah,cache[0],cache[1],g,h);
    }
    AngularControlGpu control=row.control;control.body_a=row.a.kind==WORLD?NONE:0u;control.body_b=row.b.kind==WORLD?NONE:1u;
    if(joint_angular_control::row_requested(control)){float c[3],c0[3],jcur[18],jref[18],cg[3],ch[3],ce,ag[6]={},ah[36]={};fault=material && material->model!=0u?joint_angular_control::resolve_material(control,cur,2,*material,c,jcur,c0):joint_angular_control::resolve(control,cur,ref,2,c,jcur,c0,jref);if(fault!=OK)return false;if(curves){unsigned code=0;if(!joint_angular_control::curve_weights(control,*curves,restoring_curve::FemAngular,target_index,c,c0,dt,cg,ch,&ce,code,observations,diagnostics)){fault=code;return false;}}else joint_angular_control::weights(control,c,c0,dt,cg,ch,&ce);energy+=ce;for(int k=0;k<6;++k){for(int r=0;r<3;++r)ag[k]+=jcur[6*r+k]*cg[r];for(int l=0;l<6;++l)for(int r=0;r<3;++r)ah[6*k+l]+=jcur[6*r+k]*ch[r]*jcur[6*r+l];}add_angle(ag,ah,cache[0],cache[1],g,h);}
    if(!finite(energy)){fault=BAD_ARITHMETIC;return false;}for(int k=0;k<24;++k)if(!finite(g[k])){fault=BAD_ARITHMETIC;return false;}for(int k=0;k<576;++k)if(!finite(h[k])){fault=BAD_ARITHMETIC;return false;}return true;
}
[[seam::device_fn]] inline void scatter(unsigned action,const FemJointGpu& row,const unsigned idx[24],const float g[24],const float h[576],float energy,const float* input,compute::atomic_float_t* output,compute::atomic_float_t* nodal,compute::atomic_float_t* rigid) {
    if(action==4u){compute::atomic_add(output,energy);return;}
    if(action==0u||action==1u){for(int k=0;k<24;++k)if(idx[k]!=NONE){float value=g[k];if(action==1u){value=0.f;for(int l=0;l<24;++l)if(idx[l]!=NONE)value+=h[24*k+l]*input[idx[l]];}compute::atomic_add(output+idx[k],value);}return;}
    // Aggregate aliased endpoint occurrences BEFORE extracting self blocks.
    // Nodal storage follows upstream 9*FULL_VERTEX, never compact free index.
    for(int side=0;side<2;++side){const FemEndpointGpu e=side==0?row.a:row.b;int off=12*side;
        if(action==2u&&e.kind==RIGID){if(side==1&&row.a.kind==RIGID&&row.a.id==e.id)continue;for(int r=0;r<6;++r)for(int c=0;c<6;++c){float value=0.f;for(int k=0;k<24;++k)if(idx[k]==idx[off+r])for(int l=0;l<24;++l)if(idx[l]==idx[off+c])value+=h[24*k+l];compute::atomic_add(rigid+36*e.id+6*r+c,value);}}
        else if(action==5u&&e.kind==TET){for(int v=0;v<4;++v){if(idx[off+3*v]==NONE)continue;bool prior=false;if(side==1&&row.a.kind==TET)for(int w=0;w<4;++w)if(e.vertices[v]==row.a.vertices[w])prior=true;if(prior)continue;for(int r=0;r<3;++r)for(int c=0;c<3;++c){float value=0.f;for(int k=0;k<24;++k)if(idx[k]==idx[off+3*v+r])for(int l=0;l<24;++l)if(idx[l]==idx[off+3*v+c])value+=h[24*k+l];compute::atomic_add(nodal+9*e.vertices[v]+3*r+c,value);}}}
    }
}
[[seam::device_fn]] inline void material_sample(const FemJointGpu& row,const FemFrameGpu* cache,
        [[seam::thread]] const MaterialSpinContext& material,unsigned action,compute::atomic_float_t* output,unsigned* fault,unsigned i) {
    if(!row.enabled||material.model==0u){fault[i]=0u;return;}
    if(material.model!=3u){fault[i]=joint_angular_control::CONTROL_FAULT_MATERIAL_INVALID;return;}
    const FemFrameGpu a=cache[4u*i],b=cache[4u*i+1u];if(a.status||b.status){fault[i]=a.status?a.status:b.status;return;}
    PdrdRigidState cur[2];pose_view(a,cur[0]);pose_view(b,cur[1]);
    AngularControlGpu control=row.control;control.body_a=row.a.kind==WORLD?NONE:0u;control.body_b=row.b.kind==WORLD?NONE:1u;
    float q[4],T[9];if(!joint_angular_control::material_relative_pose(control,cur,2,q,T)){fault[i]=joint_angular_control::CONTROL_FAULT_MATERIAL_INVALID;return;}
    if(action==9u){for(unsigned k=0;k<4;++k)compute::atomic_add(output+4u*i+k,q[k]);fault[i]=0u;return;}
    material_spin_v3::Trial trial{};const unsigned status=material_spin_v3::evaluate(material.accepted_q_wxyz,material.accepted_a,q,trial);
    if(status!=material_spin_v3::OK){fault[i]=status==material_spin_v3::LOCAL_STEP_AMBIGUOUS?joint_angular_control::CONTROL_FAULT_MATERIAL_LOCAL_STEP:joint_angular_control::CONTROL_FAULT_MATERIAL_INVALID;return;}
    for(unsigned k=0;k<4;++k)compute::atomic_add(output+7u*i+k,trial.accepted_candidate_q[k]);
    for(unsigned k=0;k<3;++k)compute::atomic_add(output+7u*i+4u+k,trial.strain[k]);fault[i]=0u;
}
} // namespace joint_fem
[[seam::entry(i)]]
[[seam::device_fn]] inline void joint_fem_extension_apply(const FemJointGpu* records,const MaterialSpinContext* material_spin_context,unsigned record_count,const FemFrameGpu* cache,const unsigned* vertex_to_free,unsigned vertex_count,unsigned free_count,unsigned rigid_base,unsigned dim,const AngularSoftLimitGpu* limits,const AngularDriveGpu* drives,unsigned drive_count,const AngularDriveKeyGpu* keys,float sample_time_s,float dt,unsigned action,const float* input,compute::atomic_float_t* output,compute::atomic_float_t* nodal_blocks,compute::atomic_float_t* rigid_blocks,unsigned* fault,unsigned i) {
    if(i>=record_count)return;
    const bool caching=(action&7u)>=6u;
    const unsigned token=caching?action>>3:0u;if(caching)action&=7u;
    if(caching&&token==0u){fault[i]=joint_fem::BAD_STEP;return;}
    const FemJointGpu row=records[i];if(!row.enabled){if(action==6u)joint_fem::hessian_store(cache,record_count,i,true,token,dt,nullptr);if(action==3u)fault[i]=0;return;}unsigned idx[24];if(!joint_fem::mapping(row.a,vertex_to_free,vertex_count,free_count,rigid_base,dim,idx)||!joint_fem::mapping(row.b,vertex_to_free,vertex_count,free_count,rigid_base,dim,idx+12)){fault[i]=joint_fem::BAD_MAPPING;return;}
    const MaterialSpinContext material=material_spin_context[i];
    if(action==7u){if(!joint_fem::hessian_matches(cache,record_count,i,token,dt)){fault[i]=joint_fem::BAD_STEP;return;}if(!joint_angular_extension::row_requested(row.control)||!joint_angular_extension::active(row.control,limits[i],drives,drive_count,i))return;joint_fem::hessian_matvec(cache,record_count,i,true,idx,input,output);return;}
    const FemFrameGpu* f=cache+4*i;for(int k=0;k<4;++k)if(f[k].status!=0){fault[i]=f[k].status;return;}
    PdrdRigidState cur[2],ref[2];joint_fem::pose_view(f[0],cur[0]);joint_fem::pose_view(f[1],cur[1]);joint_fem::pose_view(f[2],ref[0]);joint_fem::pose_view(f[3],ref[1]);
    AngularControlGpu control=row.control;control.body_a=row.a.kind==joint_fem::WORLD?joint_fem::NONE:0u;control.body_b=row.b.kind==joint_fem::WORLD?joint_fem::NONE:1u;
    float cg[3],c[3],j[18],ag[6]={},ah[36]={},g[24]={},h[576]={},energy=0.f,swing_yz=0.f;unsigned code=0;joint_angular_extension::AxisContribution axis[3];
    if(!joint_angular_extension::evaluate(control,limits[i],drives,drive_count,keys,i,sample_time_s,dt,cur,ref,2,cg,c,j,axis,code,&swing_yz,nullptr,&material)){fault[i]=code;return;}
    // evaluate is inert for disabled/non-requested extensions; its optional
    // outputs are not otherwise initialized, so retain the exact no-op path.
    if(!joint_angular_extension::row_requested(control)||!joint_angular_extension::active(control,limits[i],drives,drive_count,i)){if(action==6u)joint_fem::hessian_store(cache,record_count,i,true,token,dt,nullptr);if(action==3u)fault[i]=0;return;}
    for(int r=0;r<3;++r)energy+=joint_angular_extension::total_energy(axis[r]);
    for(int k=0;k<6;++k){for(int r=0;r<3;++r)ag[k]+=j[6*r+k]*cg[r];for(int l=0;l<6;++l)for(int r=0;r<3;++r)ah[6*k+l]+=j[6*r+k]*joint_angular_extension::total_weight(axis[r])*j[6*r+l];}
    if(swing_yz!=0.f)for(int k=0;k<6;++k)for(int l=0;l<6;++l)ah[6*k+l]+=swing_yz*(j[6+k]*j[12+l]+j[12+k]*j[6+l]);
    joint_fem::add_angle(ag,ah,f[0],f[1],g,h);for(int k=0;k<24;++k)if(!joint_fem::finite(g[k])){fault[i]=joint_fem::BAD_ARITHMETIC;return;}for(int k=0;k<576;++k)if(!joint_fem::finite(h[k])){fault[i]=joint_fem::BAD_ARITHMETIC;return;}if(action==6u){joint_fem::hessian_store(cache,record_count,i,true,token,dt,h);return;}if(action==3u){fault[i]=0;return;}joint_fem::scatter(action,row,idx,g,h,energy,input,output,nodal_blocks,rigid_blocks);
}
[[seam::entry(i)]]
[[seam::device_fn]] inline void joint_fem_prepare(const FemJointGpu* records,unsigned record_count,const float* current_xyz,const float* reference_xyz,unsigned vertex_count,const PdrdRigidState* rigid_current,const PdrdRigidState* rigid_reference,unsigned bodies,FemFrameGpu* cache,unsigned* fault,unsigned i) {
    if(i>=record_count)return;const FemJointGpu row=records[i];unsigned code=0;
    for(int k=0;k<4;++k){FemFrameGpu f{};if(row.enabled)f.status=joint_fem::frame(k%2==0?row.a:row.b,k<2?current_xyz:reference_xyz,vertex_count,k<2?rigid_current:rigid_reference,bodies,f);cache[4*i+k]=f;if(code==0&&f.status!=0)code=f.status;}fault[i]=code;
}
[[seam::entry(i)]]
[[seam::device_fn]] inline void joint_fem_apply(const FemJointGpu* records,const MaterialSpinContext* material_spin_context,unsigned record_count,const FemFrameGpu* cache,const unsigned* vertex_to_free,unsigned vertex_count,unsigned free_count,unsigned rigid_base,unsigned dim,float dt,unsigned action,const float* input,compute::atomic_float_t* output,compute::atomic_float_t* nodal_blocks,compute::atomic_float_t* rigid_blocks,unsigned* fault,unsigned i) {
    if(i>=record_count)return;
    const MaterialSpinContext material=material_spin_context[i];
    const FemJointGpu row=records[i];
    if(action==8u||action==9u){joint_fem::material_sample(row,cache,material,action,output,fault,i);return;}
    const bool caching=(action&7u)>=6u;
    const unsigned token=caching?action>>3:0u;if(caching)action&=7u;
    if(caching&&token==0u){fault[i]=joint_fem::BAD_STEP;return;}
    if(!row.enabled){if(action==6u)joint_fem::hessian_store(cache,record_count,i,false,token,dt,nullptr);if(action==3u)fault[i]=0;return;}unsigned idx[24];if(!joint_fem::mapping(row.a,vertex_to_free,vertex_count,free_count,rigid_base,dim,idx)||!joint_fem::mapping(row.b,vertex_to_free,vertex_count,free_count,rigid_base,dim,idx+12)){fault[i]=joint_fem::BAD_MAPPING;return;}
    if(action==7u){if(!joint_fem::hessian_matches(cache,record_count,i,token,dt)){fault[i]=joint_fem::BAD_STEP;return;}joint_fem::hessian_matvec(cache,record_count,i,false,idx,input,output);return;}
    float g[24],h[576],energy;unsigned code;if(!joint_fem::evaluate(row,cache+4*i,dt,g,h,energy,code,nullptr,0,nullptr,nullptr,&material)){fault[i]=code;return;}if(action==6u){joint_fem::hessian_store(cache,record_count,i,false,token,dt,h);return;}if(action==3u){fault[i]=0;return;}joint_fem::scatter(action,row,idx,g,h,energy,input,output,nodal_blocks,rigid_blocks);
}






