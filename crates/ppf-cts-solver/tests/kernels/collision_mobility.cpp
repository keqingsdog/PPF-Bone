// Contact eligibility uses physical free nodes, not an element's inertia share.
#include "data.hpp"
#include "contact/collision_narrow.kernel.cpp"
#include "contact/ccd_sweep.kernel.cpp"
#include "contact/intersect_geometry.kernel.cpp"
#include <cstdio>
static int failures=0;
static void check(bool yes,const char* label){std::printf("%s %s\n",yes?"PASS":"FAIL",label);if(!yes)++failures;}
int main(){
    VertexProp node[3]{};Vec2u edge;edge[0]=0;edge[1]=1;Vec3u face;face[0]=0;face[1]=1;face[2]=2;
    FaceProp fp{};EdgeProp ep{};
    for(auto& n:node){n.mass=1.f;n.pdrd_body_index=1;}
    check(fp.mass==0.f&&ep.mass==0.f,"zero face/edge inertia shares retained");
    check(collision_face_has_free_mass(face,node,fp.fixed),"PDRD mass-zero face with free physical vertices eligible");
    check(collision_edge_has_free_mass(edge,node,ep.fixed),"non-ROD mass-zero edge with free physical vertices eligible");
    check(!collision_face_has_free_mass(face,node,true)&&!collision_edge_has_free_mass(edge,node,true),"explicit element-fixed veto retained");
    for(auto& n:node)n.fix_index=1;
    check(!collision_face_has_free_mass(face,node,false)&&!collision_edge_has_free_mass(edge,node,false),"fully prescribed positive-mass vertices ineligible");
    node[1].fix_index=0;
    check(collision_face_has_free_mass(face,node,false)&&collision_edge_has_free_mass(edge,node,false),"free positive-mass nonfirst vertex is sufficient");
    for(auto& n:node){n.fix_index=0;n.mass=0.f;}
    check(!collision_face_has_free_mass(face,node,false)&&!collision_edge_has_free_mass(edge,node,false),"massless incident vertices ineligible");
    node[2].mass=1.f;
    check(collision_face_has_free_mass(face,node,false)&&!collision_edge_has_free_mass(edge,node,false),"only actual incident nodes count");
    for(auto& n:node){n.mass=1.f;n.pdrd_body_index=0;}
    check(collision_face_has_free_mass(face,node,false)&&collision_edge_has_free_mass(edge,node,false),"ordinary positive-mass nodal elements remain eligible");

    // Real C2M CCD visitor: no dynamic vertex goes near the static point.
    // The entire dynamic triangle sweeps through it, so only the face path
    // can bound this motion. Its FaceProp.mass remains zero throughout.
    Vec3f start[3],end[3],collider[2];start[0]=Vec3f(-1.f,1.f,-1.f);start[1]=Vec3f(1.f,1.f,-1.f);start[2]=Vec3f(0.f,1.f,1.f);
    for(int v=0;v<3;++v){end[v]=start[v];end[v][1]=-1.f;node[v].pdrd_body_index=1;}
    collider[0]=Vec3f(0.f,0.f,0.f);collider[1]=Vec3f(0.f,0.f,1.f);VertexProp static_props[2]{};FaceParam face_param{};VertexParam vertex_param{};
    face_param.ghat=.001f;vertex_param.ghat=.001f;
    CcdCollisionPointFaceC2mVisitor c2m{};c2m.x0=start;c2m.x1=end;c2m.collider_vertex=collider;c2m.face=&face;c2m.vertex_prop=node;c2m.face_prop=&fp;c2m.collider_vertex_prop=static_props;c2m.face_param=&face_param;c2m.collider_vertex_param=&vertex_param;c2m.vertex_index=0;c2m.face_count=1;c2m.max_t=1.f;c2m.ccd_eps=1e-4f;c2m.toi=1.f;
    check(c2m(0)&&c2m.toi>0.f&&c2m.toi<1.f,"mass-zero dynamic face real C2M CCD limits crossing");
    fp.fixed=true;c2m.toi=1.f;check(!c2m(0)&&c2m.toi==1.f,"fixed face real C2M CCD remains excluded");fp.fixed=false;

    // Real EE CCD visitor over an independently crossing static edge.
    start[0]=Vec3f(-1.f,1.f,0.f);start[1]=Vec3f(1.f,1.f,0.f);end[0]=Vec3f(-1.f,-1.f,0.f);end[1]=Vec3f(1.f,-1.f,0.f);
    collider[0]=Vec3f(0.f,0.f,-1.f);collider[1]=Vec3f(0.f,0.f,1.f);EdgeProp static_edge_prop{};EdgeParam edge_param{};edge_param.ghat=.001f;
    CcdCollisionEdgeEdgeVisitor ee{};ee.x0=start;ee.x1=end;ee.collider_vertex=collider;ee.edge=&edge;ee.collider_edge=&edge;ee.vertex_prop=node;ee.collider_vertex_prop=static_props;ee.edge_prop=&ep;ee.collider_edge_prop=&static_edge_prop;ee.edge_param=&edge_param;ee.collider_edge_param=&edge_param;ee.edge_index=0;ee.collider_edge_count=1;ee.max_t=1.f;ee.ccd_eps=1e-4f;ee.toi=1.f;
    check(ee(0)&&ee.toi>0.f&&ee.toi<1.f,"mass-zero dynamic edge real EE CCD limits crossing");
    for(auto& n:node)n.fix_index=1;ee.toi=1.f;check(!ee(0)&&ee.toi==1.f,"fully prescribed edge real EE CCD remains excluded");
    check(fp.mass==0.f&&ep.mass==0.f,"tests never restore or fabricate element masses");
    std::printf("collision mobility: %s (%d failures)\n",failures?"FAIL":"PASS",failures);return failures?1:0;
}
