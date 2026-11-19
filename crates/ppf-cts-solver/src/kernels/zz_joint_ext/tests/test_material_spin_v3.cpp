#include "../material_spin_v3.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

using material_spin_v3::Trial;
using material_spin_v3::OK;
static constexpr double PI=3.14159265358979323846;
static int failures=0;
#define CHECK(x) do { if(!(x)){std::printf("FAIL line %d: %s\n",__LINE__,#x);++failures;} }while(0)

static void mul(const float a[4],const float b[4],float out[4]){
    out[0]=a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3];
    out[1]=a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2];
    out[2]=a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1];
    out[3]=a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0];
}
static void qexp(const double d[3],float q[4]){
    const double n=std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
    const double s=n==0.?0.:std::sin(n/2.)/n;
    q[0]=float(std::cos(n/2.));for(int k=0;k<3;++k)q[k+1]=float(s*d[k]);
}
static void qaxis(int axis,double degrees,float q[4]){
    double d[3]={0.,0.,0.};d[axis]=degrees*PI/180.;qexp(d,q);
}
static double energy(const float a[3]){
    const double k[3]={1.2,1.7,.9};double E=0.;for(int i=0;i<3;++i)E+=.5*k[i]*double(a[i])*double(a[i]);return E;
}
static void body_force(const Trial& t,double out[3]){
    const double k[3]={1.2,1.7,.9};
    for(int j=0;j<3;++j){out[j]=0.;for(int i=0;i<3;++i)out[j]+=double(t.jac_body[3*i+j])*k[i]*t.strain[i];}
}
static void run_sample(float qn[4],float an[3],const float qt[4]){
    Trial out{};unsigned status=material_spin_v3::evaluate(qn,an,qt,out);CHECK(status==OK);
    if(status==OK){std::memcpy(qn,out.accepted_candidate_q,sizeof(out.accepted_candidate_q));std::memcpy(an,out.strain,sizeof(out.strain));}
}
static void planar(){
    float qn[4]={1,0,0,0},a[3]={0,0,0},q[4];
    for(int deg=1;deg<=200;++deg){qaxis(1,double(deg),q);run_sample(qn,a,q);
        if(deg==179||deg==180||deg==181||deg==200){
            std::printf("PLANAR deg=%d strain=[%.9g,%.9g,%.9g]\n",deg,a[0],a[1],a[2]);
            CHECK(std::abs(double(a[1])-deg*PI/180.)<1e-4);CHECK(std::abs(a[0])<1e-5f);CHECK(std::abs(a[2])<1e-5f);
        }
    }
    for(int deg=199;deg>=0;--deg){qaxis(1,double(deg),q);run_sample(qn,a,q);}
    for(float x:a)CHECK(std::abs(x)<1e-4f);
}
static void twist_and_bend(){
    float qn[4]={1,0,0,0},a[3]={0,0,0},q[4];
    for(int deg=2;deg<=720;deg+=2){qaxis(0,double(deg),q);run_sample(qn,a,q);}
    CHECK(std::abs(double(a[0])-4.*PI)<2e-4);
    for(int deg=1;deg<=200;++deg){float bend[4],spin[4];qaxis(1,double(deg),bend);qaxis(0,720.,spin);mul(bend,spin,q);run_sample(qn,a,q);}
    std::printf("MIXED twist720 bend200 strain=[%.9g,%.9g,%.9g]\n",a[0],a[1],a[2]);
    CHECK(std::abs(double(a[0])-4.*PI)<2e-4);CHECK(std::abs(double(a[1])-200.*PI/180.)<2e-4);CHECK(std::abs(a[2])<3e-4f);
    for(int deg=199;deg>=0;--deg){float bend[4],spin[4];qaxis(1,double(deg),bend);qaxis(0,720.,spin);mul(bend,spin,q);run_sample(qn,a,q);}
    for(int deg=718;deg>=0;deg-=2){qaxis(0,double(deg),q);run_sample(qn,a,q);}
    for(float x:a)CHECK(std::abs(x)<3e-4f);
}
static void jacobians(){
    float qn[4],qt[4],step[4];double dn[3]={.4,-.2,.5},dt[3]={.22,-.31,.14};qexp(dn,qn);qexp(dt,step);mul(qn,step,qt);
    float a[3]={1.1f,.8f,-.35f};const float snapshot[3]={a[0],a[1],a[2]};Trial t{};
    CHECK(material_spin_v3::evaluate(qn,a,qt,t)==OK);
    const double eps=.001;double force[3];body_force(t,force);double maxJ=0.,maxE=0.;
    for(int j=0;j<3;++j){
        double d[3]={0,0,0};d[j]=eps;float ep[4],em[4],qp[4],qm[4];qexp(d,ep);d[j]=-eps;qexp(d,em);mul(qt,ep,qp);mul(qt,em,qm);
        Trial p{},m{};CHECK(material_spin_v3::evaluate(qn,a,qp,p)==OK);CHECK(material_spin_v3::evaluate(qn,a,qm,m)==OK);
        for(int i=0;i<3;++i){double fd=(double(p.strain[i])-double(m.strain[i]))/(2*eps);maxJ=std::max(maxJ,std::abs(fd-t.jac_body[3*i+j]));}
        double ge=(energy(p.strain)-energy(m.strain))/(2*eps);maxE=std::max(maxE,std::abs(ge-force[j]));
    }
    std::printf("JAC body_max_abs=%.9g energy_grad_max_abs=%.9g\n",maxJ,maxE);
    CHECK(maxJ<4e-4);CHECK(maxE<8e-4);CHECK(std::memcmp(a,snapshot,sizeof(a))==0);
    float neg[4];for(int k=0;k<4;++k)neg[k]=-qt[k];Trial flipped{};CHECK(material_spin_v3::evaluate(qn,a,neg,flipped)==OK);
    for(int k=0;k<3;++k)CHECK(std::abs(flipped.strain[k]-t.strain[k])<1e-6f);
    for(int k=0;k<9;++k)CHECK(std::abs(flipped.jac_body[k]-t.jac_body[k])<1e-6f);
}
static void cross_continuity(){
    float qn[4],qm[4],qp[4];qaxis(1,179.,qn);qaxis(1,179.999, qm);qaxis(1,180.001,qp);
    float a[3]={0.f,float(179.*PI/180.),0.f};Trial m{},p{};
    CHECK(material_spin_v3::evaluate(qn,a,qm,m)==OK);CHECK(material_spin_v3::evaluate(qn,a,qp,p)==OK);
    double fm[3],fp[3];body_force(m,fm);body_force(p,fp);
    std::printf("CROSS force_y_before=%.9g after=%.9g energy_delta=%.9g\n",fm[1],fp[1],energy(p.strain)-energy(m.strain));
    CHECK(std::abs(fm[1]-fp[1])<2e-4);CHECK(std::abs(energy(p.strain)-energy(m.strain))<.001);
}
static void invalid_cases(){
    float qn[4]={1,0,0,0},a[3]={0,0,0},qt[4];Trial out;std::memset(&out,0x5a,sizeof(out));Trial saved=out;
    qaxis(0,89.,qt);CHECK(material_spin_v3::evaluate(qn,a,qt,out)==OK);
    out=saved;qaxis(0,91.,qt);CHECK(material_spin_v3::evaluate(qn,a,qt,out)==material_spin_v3::LOCAL_STEP_AMBIGUOUS);CHECK(std::memcmp(&out,&saved,sizeof(out))==0);
    float zero[4]={0,0,0,0};CHECK(material_spin_v3::evaluate(qn,a,zero,out)==material_spin_v3::INVALID_QUATERNION);CHECK(std::memcmp(&out,&saved,sizeof(out))==0);
    a[1]=std::numeric_limits<float>::quiet_NaN();CHECK(material_spin_v3::evaluate(qn,a,qn,out)==material_spin_v3::NONFINITE);CHECK(std::memcmp(&out,&saved,sizeof(out))==0);
}
int main(){planar();twist_and_bend();jacobians();cross_continuity();invalid_cases();std::printf("TOTAL failures=%d\n",failures);return failures?1:0;}
