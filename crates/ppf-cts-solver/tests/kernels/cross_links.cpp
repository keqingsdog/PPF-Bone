// Direct neutral functions: no GPU launch, native solve or GUI claim.
#include "data.hpp"
#include "zzzzzzzzz_restoring/restoring_apply.kernel.cpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
static int failures=0;
static void check(bool ok,const char* name){if(!ok){std::printf("FAIL %s\n",name);++failures;}}
static bool close(float a,float b,float tol=3e-3f){return std::abs(a-b)<=tol*(1.f+std::max(std::abs(a),std::abs(b)));}
struct Case {
    float xyz[24]{};PdrdRigidState rigid[2]{};FemJointGpu row{};FemFrameGpu cache[4]{};
    CurveDescriptor desc{};CurveNode nodes[2]{};CurveBinding binding{};CurveHistory history{};
    unsigned vf[8]{0,1,2,3,4,5,6,7};unsigned fault[2]{};
    restoring_curve::CurveContext ctx(){return {&desc,nodes,&binding,&history,1,2,1,1};}
    void frames(){cache[0].status=joint_fem::frame(row.a,xyz,8,rigid,2,cache[0]);cache[1].status=joint_fem::frame(row.b,xyz,8,rigid,2,cache[1]);check(!cache[0].status&&!cache[1].status,"actual frame evaluation");}
    bool eval(float* g,float* h,float& e,float dt=.1f){frames();unsigned code=0;auto c=ctx();bool ok=joint_fem::evaluate(row,cache,dt,g,h,e,code,&c,0);check(ok&&code==0,"distance evaluate");return ok;}
    Case(bool aTet,bool bTet,bool initial_rotation=true){
        float base[12]={0,0,0,1,0,0,0,1,0,0,0,1};for(int k=0;k<12;++k){xyz[k]=base[k];xyz[12+k]=base[k]+(k%3==0?2.f:0.f);}
        for(int side=0;side<2;++side){auto& r=rigid[side];r.R[0]=r.R[4]=r.R[8]=1;r.anchor[0]=2.f*side+.25f;r.anchor[1]=r.anchor[2]=.25f;}
        auto endpoint=[&](unsigned side,bool tet){FemEndpointGpu e{};e.kind=tet?2:1;e.id=side;e.rest_rotation[0]=e.rest_rotation[4]=e.rest_rotation[8]=1;e.dm_inv[0]=e.dm_inv[4]=e.dm_inv[8]=1;
            e.local_anchor[1]=side?-.1f:.1f;for(int k=0;k<4;++k){e.vertices[k]=4*side+k;e.anchor_bary[k]=e.center_bary[k]=.25f;}
            e.anchor_bary[0]-=e.local_anchor[1];e.anchor_bary[2]+=e.local_anchor[1];return e;};
        row.a=endpoint(0,aTet);row.b=endpoint(1,bTet);row.enabled=1;if(initial_rotation)rotate();frames();cache[2]=cache[0];cache[3]=cache[1];
        float d[3];for(int k=0;k<3;++k)d[k]=cache[0].anchor[k]-cache[1].anchor[k];
        binding.kind=restoring_curve::CrossLinkDistance;binding.channel=3;binding.epsilon_m=.001f;binding.rest_length_m=std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]+binding.epsilon_m*binding.epsilon_m);binding.damping=2.f;
        desc.node_count=2;desc.domain=1;nodes[0]={0,0,0,100};nodes[1]={1,100,50,100};
        restoring_history_capture(&binding,1,&history,1,nullptr,0,cache,1,rigid,2,xyz,8,nullptr,nullptr,0,1,0,fault,0);
        check(fault[0]==0&&history.initialized==1&&history.epoch==1,"accepted cache history kind5");
    }
    void shiftB(float x){if(row.b.kind==2){for(int v=4;v<8;++v)xyz[3*v]+=x;}else rigid[1].x[0]+=x;}
    void rotate(){for(int v=0;v<8;++v){float x=xyz[3*v];xyz[3*v]=-xyz[3*v+1]+7;xyz[3*v+1]=x-3;}
        for(auto& r:rigid){float x=r.anchor[0]+r.x[0],y=r.anchor[1]+r.x[1];r.anchor[0]=-y+7;r.anchor[1]=x-3;for(float& v:r.x)v=0;float old[9];std::memcpy(old,r.R,sizeof old);for(int c=0;c<3;++c){r.R[3*c]=-old[3*c+1];r.R[3*c+1]=old[3*c];r.R[3*c+2]=old[3*c+2];}}}
};
static void derivatives(bool at,bool bt){
    Case c(at,bt);float g[24],h[576],e;auto original=c.history;c.eval(g,h,e);
    float maxg=0;for(float x:g)maxg=std::max(maxg,std::abs(x));check(maxg<5e-5f&&e<1e-8f,"initial zero force");
    c.rotate();c.eval(g,h,e);maxg=0;for(float x:g)maxg=std::max(maxg,std::abs(x));check(maxg<2e-4f,"common rotation+translation objective zero force");
    for(float stretch:{.2f,-.2f}){
        Case t(at,bt);t.shiftB(stretch);t.eval(g,h,e);check(e>0,"stretch/compression positive energy");
        float ja[36],jb[36],j[72],d[3];joint_fem::anchor_jacobian(t.row.a,t.cache[0],ja);joint_fem::anchor_jacobian(t.row.b,t.cache[1],jb);
        for(int r=0;r<3;++r){d[r]=t.cache[0].anchor[r]-t.cache[1].anchor[r];for(int k=0;k<12;++k){j[24*r+k]=ja[12*r+k];j[24*r+12+k]=-jb[12*r+k];}}
        auto ctx=t.ctx();restoring_stitch::Evaluation radial{};unsigned code=0;check(restoring_stitch::evaluate_binding_f32(ctx,&t.binding,d,.1f,radial,code),"shared distance math");
        for(int k=0;k<24;++k)for(int l=0;l<24;++l){float expected=0;for(int r=0;r<3;++r)for(int s=0;s<3;++s)expected+=j[24*r+k]*radial.projected_hessian[3*r+s]*j[24*s+l];check(close(h[24*k+l],expected,1e-5f),"independent Jt PSD(Hd) J");check(close(h[24*k+l],h[24*l+k],1e-5f),"symmetric mapped Hessian");}
        for(int trial=0;trial<8;++trial){float v[24],vHv=0;for(int k=0;k<24;++k)v[k]=std::sin(float(k+trial*7));for(int k=0;k<24;++k)for(int l=0;l<24;++l)vHv+=v[k]*h[24*k+l]*v[l];check(vHv>=-1e-3f,"PSD modes");}
        // Exact nodal derivatives; rigid rotational derivatives use a small
        // world-axis rotation, not a fictitious additive angular coordinate.
        for(int side=0;side<2;++side){bool tet=side?bt:at;int n=tet?12:6;
            for(int k=0;k<n;++k){Case plus=t,minus=t;float step=5e-4f;
                auto perturb=[&](Case& a,float dx){if(tet)a.xyz[12*side+k]+=dx;else if(k<3)a.rigid[side].x[k]+=dx;else {int axis=k-3;float q[9]={};float co=std::cos(dx),si=std::sin(dx);for(int col=0;col<3;++col){float x[3];for(int r=0;r<3;++r)x[r]=a.rigid[side].R[3*col+r];for(int r=0;r<3;++r){int u=(axis+1)%3,v=(axis+2)%3;q[3*col+r]=r==axis?x[r]:(r==u?co*x[u]-si*x[v]:si*x[u]+co*x[v]);}}std::memcpy(a.rigid[side].R,q,sizeof q);}};
                perturb(plus,step);perturb(minus,-step);float pg[24],ph[576],pe,mg[24],mh[576],me;plus.eval(pg,ph,pe);minus.eval(mg,mh,me);check(close((pe-me)/(2*step),g[12*side+k],4e-3f),"accurate energy gradient FD");
            }
        }
        if(!at&&!bt){float total[3]={},moment[3]={};for(int side=0;side<2;++side){const float* f=g+12*side;const auto& frame=t.cache[side];for(int k=0;k<3;++k){total[k]+=f[k];moment[k]+=f[3+k]+frame.center[(k+1)%3]*f[(k+2)%3]-frame.center[(k+2)%3]*f[(k+1)%3];}}for(int k=0;k<3;++k){check(std::abs(total[k])<1e-4f,"equal opposite endpoint force");check(std::abs(moment[k])<1e-4f,"total force moment balance");}}
        for(int k=0;k<3;++k){float dp[3],dm[3];std::memcpy(dp,d,sizeof d);std::memcpy(dm,d,sizeof d);dp[k]+=.001f;dm[k]-=.001f;restoring_stitch::Evaluation ep{},em{};restoring_stitch::evaluate_binding_f32(ctx,&t.binding,dp,.1f,ep,code);restoring_stitch::evaluate_binding_f32(ctx,&t.binding,dm,.1f,em,code);for(int r=0;r<3;++r)check(close((ep.gradient[r]-em.gradient[r])/.002f,radial.raw_hessian[3*r+k],3e-3f),"raw distance Hd gradient FD");}
        // Retry duration changes damping, never accepted history.
        float fast[24],fh[576],fe;t.eval(fast,fh,fe,.05f);check(fe>e,"shorter h increases radial damping energy");check(std::memcmp(&t.history,&original,sizeof original)==0,"candidate/retry reads accepted history only");
    }
}
static void scatter_alias_and_loop(){
    Case c(true,true);c.shiftB(.2f);float g[24],h[576],e;c.eval(g,h,e);
    // Share the same original support slot between two endpoints. Scatter must
    // aggregate repeated occurrences before extracting a nodal self block.
    c.row.b.vertices[0]=c.row.a.vertices[0];unsigned idx[24];joint_fem::mapping(c.row.a,c.vf,8,8,24,24,idx);joint_fem::mapping(c.row.b,c.vf,8,8,24,24,idx+12);
    float dense[576]={},input[24],out[24]={},nodal[72]={};for(int k=0;k<24;++k)input[k]=std::cos(float(k));
    // Independent gather oracle: no repeated-index += scatter in the oracle.
    for(int r=0;r<24;++r)for(int s=0;s<24;++s){float value=0;for(int k=0;k<24;++k)if(idx[k]==unsigned(r))for(int l=0;l<24;++l)if(idx[l]==unsigned(s))value+=h[24*k+l];dense[24*r+s]=value;}
    compute::atomic_float_t atomic_out[24]{},atomic_nodal[72]{};joint_fem::scatter(1,c.row,idx,g,h,e,input,atomic_out,nullptr,nullptr);joint_fem::scatter(5,c.row,idx,g,h,e,input,nullptr,atomic_nodal,nullptr);std::memcpy(out,atomic_out,sizeof out);std::memcpy(nodal,atomic_nodal,sizeof nodal);
    for(int k=0;k<24;++k){float expected=0;for(int l=0;l<24;++l)expected+=dense[24*k+l]*input[l];check(close(out[k],expected,1e-4f),"aliased mapped Hv");}
    for(int v=0;v<8;++v)for(int r=0;r<3;++r)for(int k=0;k<3;++k)check(close(nodal[9*v+3*r+k],dense[24*(3*v+r)+3*v+k],1e-4f),"aliased nodal self block");
    // Gather each destination independently, including all repeated occurrences.
    // A source-scatter oracle repeats the product's algorithm and can obscure
    // alias errors; this oracle has no indirect read-modify-write destination.
    float network[576]={};for(int edge=0;edge<3;++edge){unsigned map[24];for(int k=0;k<24;++k)map[k]=(idx[k]+3*edge)%24;for(int r=0;r<24;++r)for(int s=0;s<24;++s){float value=0;for(int k=0;k<24;++k)if(map[k]==unsigned(r))for(int l=0;l<24;++l)if(map[l]==unsigned(s))value+=h[24*k+l];network[24*r+s]+=value;}}
    float vHv=0;for(int k=0;k<24;++k)for(int l=0;l<24;++l)vHv+=input[k]*network[24*k+l]*input[l];check(vHv>=-1e-3f,"closed incident network PSD assembly");
}
int main(){derivatives(false,false);derivatives(true,true);derivatives(false,true);scatter_alias_and_loop();std::printf("cross_links neutral math: %s (%d failures)\n",failures?"FAIL":"PASS",failures);return failures?1:0;}
