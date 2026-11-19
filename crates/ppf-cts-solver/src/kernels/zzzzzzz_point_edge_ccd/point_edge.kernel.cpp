// Standalone faceless-edge CCD coverage. No change to ACCD or old sweeps.
#pragma once
#include "../contact/ccd_sweep.kernel.cpp"

struct CcdStandalonePointEdgeVisitor {
    const Vec3f *x0;
    const Vec3f *x1;
    const Vec2u *edge;
    const VertexProp *vertex_prop;
    const unsigned *start_link_index;
    const unsigned *start_link_offset;
    unsigned has_start_link;
    const EdgeProp *edge_prop;
    const VertexParam *vertex_param;
    const EdgeParam *edge_param;
    const unsigned *edge_face_offset;
    unsigned has_edge_face;
    unsigned vertex_index;
    unsigned vertex_count;
    unsigned edge_count;
    float max_t;
    float ccd_eps;
    float toi;
    CcdOverlapRecord overlap;
    DiagHandle diag;

    [[seam::device_fn]] bool test(const AABB &box, const AABB &q) const {
        return aabb_overlap(box, q);
    }
    [[seam::device_fn]] bool operator()(unsigned index) {
        DIAG_ASSERT4(diag,index<edge_count,static_cast<float>(index),
                     static_cast<float>(edge_count),static_cast<float>(vertex_index),0.0f);
        if(index>=edge_count)return false;
        // Minimal missing domain: rod/faceless edges. Surface edge/face routes
        // remain the existing sweeps. A missing adjacency table means no faces.
        if(has_edge_face!=0u && edge_face_offset[index+1u]>edge_face_offset[index])return false;
        const Vec2u e=edge[index];
        DIAG_ASSERT4(diag,e[0]<vertex_count && e[1]<vertex_count,
                     static_cast<float>(e[0]),static_cast<float>(e[1]),
                     static_cast<float>(vertex_count),static_cast<float>(index));
        if(e[0]>=vertex_count || e[1]>=vertex_count)return false;
        // Only a point belonging to THIS edge is adjacent. Isolated points
        // have no incident edge/face and must never be removed by index order.
        if(e[0]==vertex_index || e[1]==vertex_index)return false;
        const VertexProp vp=vertex_prop[vertex_index];
        const VertexProp anchor=vertex_prop[e[0]];
        const EdgeProp ep=edge_prop[index];
        if(!contact_pair_admitted(pair_side_of_vertex(vp,vertex_index),
                                  pair_side_of_edge(anchor,ep,e),
                                  start_link_index,start_link_offset,has_start_link))return false;
        const VertexParam pv=vertex_param[vp.param_index];
        const EdgeParam pe=edge_param[ep.param_index];
        const float offset=pv.offset+pe.offset;
        const float ghat=0.5f*(pv.ghat+pe.ghat);
        const Vec3f p0=x0[vertex_index],p1=x1[vertex_index];
        const Vec3f e00=x0[e[0]],e01=x0[e[1]],e10=x1[e[0]],e11=x1[e[1]];
        accd::OverlapInfo info;info.d2=0.0f;info.offset=0.0f;info.flagged=0u;
        const float result=accd::point_edge_ccd(p0,p1,e00,e01,e10,e11,
                                               offset,ghat,max_t,ccd_eps,&info);
        ccd_record_overlap(overlap,CCD_SWEEP_POINT_EDGE,vertex_index,index,info);
        if(result<max_t){toi=fmath::min(toi,result);return true;}
        return false;
    }
};

[[seam::entry(element)]]
[[seam::device_fn]] inline void ccd_standalone_point_edge(
    const Vec3f *x0,const Vec3f *x1,const Vec2u *edge,unsigned edge_count,
    unsigned vertex_count,const VertexProp *vertex_prop,
    const unsigned *start_link_index,const unsigned *start_link_offset,
    unsigned has_start_link,const EdgeProp *edge_prop,
    const VertexParam *vertex_param,const EdgeParam *edge_param,
    const unsigned *edge_face_offset,unsigned has_edge_face,
    const unsigned *vertex_node,const unsigned *node,unsigned node_count,
    const AABB *aabb,unsigned root,const unsigned *active,unsigned has_active,
    float max_t,float ccd_eps,float *out_toi,CcdOverlapRecord *out_overlap,
    unsigned query_count,DiagHandle diag,unsigned element) {
    const unsigned i=ccd_morton_remap(vertex_node,element,query_count,diag);
    DIAG_ASSERT4(diag,i<vertex_count,static_cast<float>(i),
                 static_cast<float>(vertex_count),static_cast<float>(element),0.0f);
    if(i>=vertex_count)return;
    CcdStandalonePointEdgeVisitor op;
    op.x0=x0;op.x1=x1;op.edge=edge;op.vertex_prop=vertex_prop;
    op.start_link_index=start_link_index;op.start_link_offset=start_link_offset;
    op.has_start_link=has_start_link;op.edge_prop=edge_prop;
    op.vertex_param=vertex_param;op.edge_param=edge_param;
    op.edge_face_offset=edge_face_offset;op.has_edge_face=has_edge_face;
    op.vertex_index=i;op.vertex_count=vertex_count;op.edge_count=edge_count;
    op.max_t=max_t;op.ccd_eps=ccd_eps;op.toi=max_t;
    op.overlap=ccd_no_overlap();op.diag=diag;
    const VertexParam vp=vertex_param[vertex_prop[i].param_index];
    const AABB box=ccd_point_box(x0,x1,vp,max_t,active,has_active,i);
    aabb_query(node,node_count,aabb,root,op,box,diag);
    ccd_commit(out_toi,out_overlap,i,op.toi,op.overlap);
}
