#pragma once
#include "../seam/seam.hpp"
// Standalone scalar host harness has no backend prologue. Production uses
// the existing seam macro (__device__ __host__ inline on CUDA).
#ifndef SM_INLINE_DEVICE_HOST
#define SM_INLINE_DEVICE_HOST inline
#endif

// M0 scalar law only. These host views are NOT uploaded POD/ABI records.
// B owns serialized descriptors and areas; an adapter follows its reviewed
// handoff. No existing kernel/driver includes this header in M1.




namespace restoring_curve {
enum class Domain { Signed, Odd, Radial };
enum class Tail { Linear, HoldEffort };
enum class Fault { None, InvalidTable, InvalidCoordinate, Arithmetic };
struct HostNode { float x, f, area; };
struct HostView { const HostNode* nodes; unsigned count; Domain domain; Tail tail; };
struct Result { float energy, gradient, curvature; Fault fault; };

// A single f32 significand with a separate exponent. This only extends the
// range of local restoring intermediates: C/dt may under/overflow f32 even
// when its final product is representable. It is not an ABI or storage type.
namespace device_f32 {
SM_INLINE_DEVICE_HOST bool finite(float x) { return x==x && x<=3.4028234663852886e38f && x>=-3.4028234663852886e38f; }
SM_INLINE_DEVICE_HOST float abs(float x) { return x<0.f?-x:x; }
struct Scaled { float m; int e; };
SM_INLINE_DEVICE_HOST Scaled scaled(float m,int e=0) {
    if(m==0.f)return {0.f,0};
    if(!finite(m))return {m,0};
    while(abs(m)>=2.f){m*=.5f;++e;}
    while(abs(m)<1.f){m*=2.f;--e;}
    return {m,e};
}
SM_INLINE_DEVICE_HOST Scaled neg(Scaled a) { return {-a.m,a.e}; }
SM_INLINE_DEVICE_HOST float shift(float m,int n) {
    while(n>0){m*=2.f;--n;}while(n<0){m*=.5f;++n;}return m;
}
SM_INLINE_DEVICE_HOST Scaled add(Scaled a,Scaled b) {
    if(a.m==0.f)return b;if(b.m==0.f)return a;
    if(a.e<b.e){Scaled t=a;a=b;b=t;}
    if(a.e-b.e>25)return a;
    return scaled(a.m+shift(b.m,b.e-a.e),a.e);
}
SM_INLINE_DEVICE_HOST Scaled sub(Scaled a,Scaled b) { return add(a,neg(b)); }
SM_INLINE_DEVICE_HOST Scaled mul(Scaled a,Scaled b) { return scaled(a.m*b.m,a.e+b.e); }
SM_INLINE_DEVICE_HOST Scaled div(Scaled a,Scaled b) { return scaled(a.m/b.m,a.e-b.e); }
SM_INLINE_DEVICE_HOST Scaled square(Scaled a) { return mul(a,a); }
SM_INLINE_DEVICE_HOST Scaled root(Scaled a) {
    if(a.m==0.f)return a;
    if(!finite(a.m)||a.m<0.f)return {-1.f,100000};
    if(a.e%2){a.m*=2.f;--a.e;}
    float r=2.f;for(unsigned i=0;i<6;++i)r=.5f*(r+a.m/r);
    return scaled(r,a.e/2);
}
SM_INLINE_DEVICE_HOST bool narrow(Scaled a,float& out,bool energy=false) {
    if(!finite(a.m)||(energy&&a.m<0.f)||a.e>127)return false;
    if(a.m==0.f){out=0.f;return true;}
    if(a.e < -150)out=0.f;
    else if(a.e < -126)out=shift(a.m,a.e+126)*1.1754943508222875e-38f;
    else out=shift(a.m,a.e);
    return finite(out)&&(energy||out!=0.f);
}
// Angular torque may be smaller than half the least positive f32. In that
// one case round to zero, but still reject an unexpectedly flushed value that
// should round to a representable subnormal.
SM_INLINE_DEVICE_HOST bool narrow_angular_effort(Scaled a,float& out) {
    if(narrow(a,out))return true;
    if(finite(a.m) && a.m!=0.f &&
       (a.e < -150 || (a.e == -150 && abs(a.m)==1.f))) {
        out=0.f;return true;
    }
    return false;
}
// Compensation is local to the difference of squared lengths. In particular,
// retain d*d when sqrt(eps*eps+d*d) would first round to eps.
struct CoordinateSum { Scaled hi,lo; };
SM_INLINE_DEVICE_HOST CoordinateSum two_sum(Scaled a,Scaled b) {
    if(a.m==0.f)return {b,scaled(0.f)};if(b.m==0.f)return {a,scaled(0.f)};
    if(a.e<b.e){Scaled t=a;a=b;b=t;}
    if(a.e-b.e>25)return {a,b};
    float x=a.m,y=shift(b.m,b.e-a.e),s=x+y,z=s-x;
    return {scaled(s,a.e),scaled((x-(s-z))+(y-z),a.e)};
}
SM_INLINE_DEVICE_HOST void coordinate_add(CoordinateSum& sum,Scaled value) {
    CoordinateSum p=two_sum(sum.hi,value);
    CoordinateSum q=two_sum(p.hi,add(sum.lo,p.lo));sum=q;
}
SM_INLINE_DEVICE_HOST void coordinate_product(CoordinateSum& sum,Scaled a,Scaled b) {
    float p=a.m*b.m;
    coordinate_add(sum,scaled(p,a.e+b.e));
    coordinate_add(sum,scaled(fmath::fma(a.m,b.m,-p),a.e+b.e));
}
SM_INLINE_DEVICE_HOST Scaled norm(const float* d,unsigned count) {
    Scaled sum=scaled(0.f);
    for(unsigned i=0;i<count;++i)sum=add(sum,square(scaled(d[i])));
    return root(sum);
}
} // namespace device_f32

SM_INLINE_DEVICE_HOST bool finite(float x) { return device_f32::finite(x); }
SM_INLINE_DEVICE_HOST Result fault(Fault value) { return {0.f, 0.f, 0.f, value}; }
#if !defined(SM_MSL_CONCAT)
// Explicit legacy host reference is excluded from MSL, which has no f64 type.
SM_INLINE_DEVICE_HOST bool finite(double x) { return x==x && x<=1.7976931348623157e308 && x>=-1.7976931348623157e308; }
SM_INLINE_DEVICE_HOST double absolute(double x) { return x<0. ? -x:x; }
SM_INLINE_DEVICE_HOST bool narrow(double value, float& out) {
    if (!finite(value) || absolute(value)>3.4028234663852886e38) return false;
    out=static_cast<float>(value);
    return finite(out) && (value==0. || out!=0.f);
}
// Energy is a nonnegative scalar measure. IEEE round-to-zero is valid when a
// finite sub-f32 energy is below the representable range; force/curvature and
// structural quantities keep using `narrow`, which still rejects underflow.
SM_INLINE_DEVICE_HOST bool narrow_energy(double value, float& out) {
    if (!finite(value) || value<0. || value>3.4028234663852886e38) return false;
    out=static_cast<float>(value);
    return finite(out) && out>=0.f;
}
#endif

// Shape/finite checks guard this standalone host API. Full strict author/area
// recomputation belongs to B's packer; finite node areas are already sealed.
SM_INLINE_DEVICE_HOST bool valid_structure(const HostView& table) {
    if (!table.nodes || table.count<2 || table.count>64) return false;
    if (table.domain!=Domain::Signed && table.domain!=Domain::Odd && table.domain!=Domain::Radial) return false;
    if (table.tail!=Tail::Linear && table.tail!=Tail::HoldEffort) return false;
    bool zero=false;
    for (unsigned i=0;i<table.count;++i) {
        const auto n=table.nodes[i];
        if (!finite(n.x)||!finite(n.f)||!finite(n.area)) return false;
        if (n.x==0.f) { if (zero||n.f!=0.f||n.area!=0.f) return false; zero=true; }
        if (i) {
            const auto prev=table.nodes[i-1];
            if (!(n.x>prev.x)||n.f<prev.f) return false;
        }
    }
    if (!zero) return false;
    if (table.domain==Domain::Signed) return table.nodes[0].x<0.f && table.nodes[table.count-1].x>0.f;
    return table.nodes[0].x==0.f && table.nodes[0].f==0.f && table.nodes[table.count-1].x>0.f;
}
#if !defined(SM_MSL_CONCAT)
SM_INLINE_DEVICE_HOST bool valid_view(const HostView& table) {
    if(!valid_structure(table))return false;
    for(unsigned i=1;i<table.count;++i){
        const auto n=table.nodes[i],p=table.nodes[i-1];float dx,df,slope;
        if(!narrow(double(n.x)-p.x,dx)||!narrow(double(n.f)-p.f,df)||
           !narrow(double(df)/dx,slope)||slope<0.f)return false;
    }
    return true;
}
#endif
struct Segment { float x;unsigned a,b,anchor;bool mirrored,exterior; };
SM_INLINE_DEVICE_HOST Segment select_segment(const HostView& table,float coordinate) {
    Segment s{};s.mirrored=table.domain==Domain::Odd&&coordinate<0.f;
    s.x=s.mirrored?-coordinate:coordinate;s.b=1;
    if(s.x<table.nodes[0].x){s.exterior=true;}
    else if(s.x>=table.nodes[table.count-1].x){s.exterior=true;s.a=table.count-2;s.b=table.count-1;s.anchor=s.b;}
    else {
        // Strict > chooses the right interval at an interior stored knot.
        for(unsigned i=1;i<table.count;++i)if(table.nodes[i].x>s.x){s.a=i-1;s.b=i;break;}
        // Use one fixed segment anchor toward zero, never switch rounded areas.
        s.anchor=device_f32::abs(table.nodes[s.a].x)<=device_f32::abs(table.nodes[s.b].x)?s.a:s.b;
    }
    return s;
}

#if !defined(SM_MSL_CONCAT)
SM_INLINE_DEVICE_HOST Result eval_legacy(const HostView& table, float coordinate) {
    if (!finite(coordinate)|| (table.domain==Domain::Radial && coordinate<0.f)) return fault(Fault::InvalidCoordinate);
    if (!valid_view(table)) return fault(Fault::InvalidTable);
    const auto segment=select_segment(table,coordinate);
    const bool mirrored=segment.mirrored;
    const double x=segment.x;
    const HostNode* nodes=table.nodes;
    const unsigned a=segment.a,b=segment.b,anchor=segment.anchor;
    const bool exterior=segment.exterior;
    const double h=(exterior && table.tail==Tail::HoldEffort) ? 0. :
        (double(nodes[b].f)-nodes[a].f)/(double(nodes[b].x)-nodes[a].x);
    const double t=x-double(nodes[anchor].x);
    const double g=double(nodes[anchor].f)+h*t;
    const double e=double(nodes[anchor].area)+double(nodes[anchor].f)*t+.5*h*t*t;
    Result out{};out.fault=Fault::None;
    if (!narrow_energy(e,out.energy)||!narrow(mirrored ? -g:g,out.gradient)||!narrow(h,out.curvature)||out.curvature<0.f)
        return fault(Fault::Arithmetic);
    return out;
}
#endif

// Host-callable single-precision path. Domain, knot tie-breaking and the
// fixed toward-zero segment anchor match the legacy scalar law above.
SM_INLINE_DEVICE_HOST bool valid_view_f32(const HostView& table) {
    namespace f=device_f32;
    if(!valid_structure(table))return false;
    for(unsigned i=1;i<table.count;++i){
        const auto n=table.nodes[i];
            const auto p=table.nodes[i-1];float dx,df,slope;
            if(!f::narrow(f::sub(f::scaled(n.x),f::scaled(p.x)),dx)||
               !f::narrow(f::sub(f::scaled(n.f),f::scaled(p.f)),df)||
               !f::narrow(f::div(f::scaled(df),f::scaled(dx)),slope)||slope<0.f)return false;
    }
    return true;
}
SM_INLINE_DEVICE_HOST Result eval_f32(const HostView& table,float coordinate,SM_THREAD device_f32::Scaled* angular_gradient=nullptr) {
    namespace f=device_f32;
    if(!f::finite(coordinate)||(table.domain==Domain::Radial&&coordinate<0.f))return fault(Fault::InvalidCoordinate);
    if(!valid_view_f32(table))return fault(Fault::InvalidTable);
    const auto segment=select_segment(table,coordinate);
    const bool mirrored=segment.mirrored,exterior=segment.exterior;
    const float x=segment.x;const unsigned a=segment.a,b=segment.b,anchor=segment.anchor;
    f::Scaled h=(exterior&&table.tail==Tail::HoldEffort)?f::scaled(0.f):
        f::div(f::sub(f::scaled(table.nodes[b].f),f::scaled(table.nodes[a].f)),
               f::sub(f::scaled(table.nodes[b].x),f::scaled(table.nodes[a].x)));
    f::Scaled t=f::sub(f::scaled(x),f::scaled(table.nodes[anchor].x));
    f::Scaled g=f::add(f::scaled(table.nodes[anchor].f),f::mul(h,t));
    f::Scaled e=f::add(f::scaled(table.nodes[anchor].area),
        f::add(f::mul(f::scaled(table.nodes[anchor].f),t),f::mul(f::scaled(.5f),f::mul(h,f::square(t)))));
    Result out{};out.fault=Fault::None;
    const f::Scaled signed_g=mirrored?f::neg(g):g;
    if(!f::narrow(e,out.energy,true)||!f::narrow(h,out.curvature)||out.curvature<0.f)return fault(Fault::Arithmetic);
    if(angular_gradient) {
        if(!f::narrow_angular_effort(signed_g,out.gradient))return fault(Fault::Arithmetic);
        // Preserve an unrepresentably small elastic torque for the damping
        // sum; representable torques keep their previous f32 rounding.
        *angular_gradient=out.gradient==0.f?signed_g:f::scaled(out.gradient);
    } else if(!f::narrow(signed_g,out.gradient))return fault(Fault::Arithmetic);
    return out;
}
SM_INLINE_DEVICE_HOST Result eval(const HostView& table,float coordinate) { return eval_f32(table,coordinate); }
} // namespace restoring_curve
