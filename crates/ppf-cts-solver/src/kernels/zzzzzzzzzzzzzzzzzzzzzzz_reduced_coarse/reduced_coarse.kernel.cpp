// Exact-structure Galerkin hierarchy. All block matrices are column-major3x3.
// Physical modes/P are row-major3x6 per triplet; R is row-major6x6/node.
#pragma once
// Byte-for-byte input view of authoritative data.rs::PdrdRigidState and
// energy/model/pdrd_rigid.kernel.cpp. No simulation state is written here.
struct RmcRigidState {
    float anchor[3];
    float x[3];
    float R[9];
    float Iref[9];
    float mass_total;
    unsigned N;
};
struct RmcPosition {
    float data[3];
};
[[seam::device_fn]] inline bool rmc_finite(float x) {
    return x==x && x<=3.402823466e38f && x>=-3.402823466e38f;
}
[[seam::entry(row)]]
[[seam::device_fn]] inline void rmc_geometry(
    const unsigned *row_to_vertex,const RmcPosition *vertices,const RmcRigidState *state,
    const unsigned *dof_mask,float *position,unsigned *kind,unsigned *fault,
    unsigned row,unsigned cloth,unsigned extent) {
    if(row>=extent)return;
    if(row<cloth) {
        unsigned v=row_to_vertex[row];kind[row]=dof_mask[v]!=0u?3u:0u;
        for(unsigned i=0u;i<3u;++i)position[3u*row+i]=vertices[v].data[i];
    } else {
        unsigned b=(row-cloth)/2u;kind[row]=(row-cloth)%2u==0u?1u:2u;
        for(unsigned i=0u;i<3u;++i)position[3u*row+i]=state[b].anchor[i]+state[b].x[i];
    }
    for(unsigned i=0u;i<3u;++i)if(!rmc_finite(position[3u*row+i]))fault[row]=68u;
}
[[seam::entry(row)]]
[[seam::device_fn]] inline void rmc_child_geometry(
    const float *center,float *position,unsigned *kind,unsigned row,unsigned extent) {
    if(row>=extent)return;
    for(unsigned i=0u;i<3u;++i)position[3u*row+i]=center[3u*(row/2u)+i];
    kind[row]=row%2u==0u?1u:2u;
}
// position is LIVE world-space xyz per reduced triplet, not rest geometry.
// kind=0 FEM,1 body translation,2 body rotation. Body pairs share position.
[[seam::entry(node)]]
[[seam::device_fn]] inline void rmc_centers(
    const unsigned *member_offset,const unsigned *members,const float *position,
    const unsigned *kind,float *center,unsigned *fault,unsigned node,unsigned extent) {
    if(node>=extent)return;
    float c[3]={0.0f,0.0f,0.0f};unsigned n=0u;
    for(unsigned k=member_offset[node];k<member_offset[node+1u];++k) {
        unsigned r=members[k];if(kind[r]==2u)continue;
        ++n;for(unsigned i=0u;i<3u;++i)c[i]+=position[3u*r+i];
    }
    if(n==0u){fault[node]=60u;return;}
    for(unsigned i=0u;i<3u;++i) {
        float v=fmath::div(c[i],float(n));
        if(!rmc_finite(v)){fault[node]=60u;return;}
        center[3u*node+i]=v;
    }
}
[[seam::device_fn]] inline float rmc_physical(unsigned i,unsigned j,
    unsigned kind,float x,float y,float z) {
    if(kind==3u)return 0.0f;
    if(kind==2u)return j>=3u && i==j-3u?1.0f:0.0f;
    if(j<3u)return i==j?1.0f:0.0f;
    // delta=t-skew(position-center)*omega
    if(i==0u && j==4u)return z;
    if(i==0u && j==5u)return -y;
    if(i==1u && j==3u)return -z;
    if(i==1u && j==5u)return x;
    if(i==2u && j==3u)return y;
    if(i==2u && j==4u)return -x;
    return 0.0f;
}
[[seam::entry(row)]]
[[seam::device_fn]] inline void rmc_modes(
    const unsigned *owner,const unsigned *kind,const float *position,
    const float *center,float *modes,unsigned row,unsigned extent) {
    if(row>=extent)return;unsigned a=owner[row];
    float x=position[3u*row]-center[3u*a];
    float y=position[3u*row+1u]-center[3u*a+1u];
    float z=position[3u*row+2u]-center[3u*a+2u];
    for(unsigned i=0u;i<3u;++i)for(unsigned j=0u;j<6u;++j)
        modes[18u*row+6u*i+j]=rmc_physical(i,j,kind[row],x,y,z);
}
// At deeper levels child coordinates are Q coordinates. Never reinterpret
// normalized triplets as translation/rotation: M_parent=R_child*T_child,parent.
[[seam::entry(row)]]
[[seam::device_fn]] inline void rmc_parent_modes(
    const unsigned *owner,const float *child_center,const float *parent_center,
    const float *child_r,float *modes,unsigned row,unsigned extent) {
    if(row>=extent)return;unsigned child=row/2u,parent=owner[row];
    float x=child_center[3u*child]-parent_center[3u*parent];
    float y=child_center[3u*child+1u]-parent_center[3u*parent+1u];
    float z=child_center[3u*child+2u]-parent_center[3u*parent+2u];
    for(unsigned i=0u;i<3u;++i)for(unsigned j=0u;j<6u;++j) {
        float s=0.0f;
        for(unsigned k=0u;k<6u;++k)
            s+=child_r[36u*child+6u*(3u*(row%2u)+i)+k]
                *rmc_physical(k%3u,j,k<3u?1u:2u,x,y,z);
        modes[18u*row+6u*i+j]=s;
    }
}
// Weighted twice-reorthogonalized MGS. All duplicate diagonal CSR terms are
// summed. P=D^-1 Q and M is approximated by P R. Exact dependent columns and
// roundoff-indistinguishable candidate columns pad with zero P: this is only
// a coarse correction, and the independent fine local SPD sweep still covers
// every unconstrained coordinate. No original operator or tolerance is changed.
[[seam::entry(node)]]
[[seam::device_fn]] inline void rmc_basis(
    const unsigned *member_offset,const unsigned *members,const unsigned *offset,
    const unsigned *column,const float *value,const float *modes,
    float *p,float *transform,unsigned *rank,unsigned *fault,
    unsigned node,unsigned extent) {
    if(node>=extent)return;
    unsigned begin=member_offset[node],n=member_offset[node+1u]-begin,d=3u*n;
    if(n==0u || n>16u){fault[node]=61u;return;}
    float a[288],w[48],r[36];
    for(unsigned k=0u;k<36u;++k)r[k]=0.0f;
    for(unsigned t=0u;t<n;++t) {
        unsigned row=members[begin+t];
        for(unsigned i=0u;i<3u;++i) {
            float diagonal=0.0f;
            for(unsigned k=offset[row];k<offset[row+1u];++k)
                if(column[k]==row)diagonal+=value[9u*k+4u*i];
            if(!(diagonal>0.0f)||!rmc_finite(diagonal)){fault[node]=62u;return;}
            w[3u*t+i]=fmath::sqrt(diagonal);
            for(unsigned j=0u;j<6u;++j) {
                float source=modes[18u*row+6u*i+j];
                float v=w[3u*t+i]*source;
                if(v==0.0f && source!=0.0f){fault[node]=641u;return;}
                if(!rmc_finite(v)){fault[node]=63u;return;}
                a[6u*(3u*t+i)+j]=v;
            }
        }
    }
    unsigned qrank=0u;
    for(unsigned j=0u;j<6u;++j) {
        float original=0.0f;
        for(unsigned i=0u;i<d;++i)original+=a[6u*i+j]*a[6u*i+j];
        for(unsigned pass=0u;pass<2u;++pass)for(unsigned k=0u;k<qrank;++k) {
            float dot=0.0f;
            for(unsigned i=0u;i<d;++i)dot+=p[18u*members[begin+i/3u]+6u*(i%3u)+k]
                *w[i]*a[6u*i+j];
            r[6u*k+j]+=dot;
            for(unsigned i=0u;i<d;++i)a[6u*i+j]-=dot
                *p[18u*members[begin+i/3u]+6u*(i%3u)+k]*w[i];
        }
        float norm2=0.0f;
        for(unsigned i=0u;i<d;++i)norm2+=a[6u*i+j]*a[6u*i+j];
        if(!rmc_finite(norm2)||!rmc_finite(original)){fault[node]=63u;return;}
        if(norm2==0.0f) {
            // A squared norm may underflow even with nonzero coordinates.
            // That is unresolved numerical rank, not a dependent mode.
            for(unsigned i=0u;i<d;++i)if(a[6u*i+j]!=0.0f){fault[node]=642u;return;}
            continue;
        }
        // A direction below the rounding bound cannot be normalized reliably.
        // Omit it from this optional coarse space; the fine SPD correction
        // remains present and PCG checks the true unmodified operator residual.
        float gamma=float(12u*d)*1.192092896e-7f;
        if(norm2<=gamma*gamma*original)continue;
        float norm=fmath::sqrt(norm2);r[6u*qrank+j]=norm;
        for(unsigned i=0u;i<d;++i)p[18u*members[begin+i/3u]+6u*(i%3u)+qrank]
            =fmath::div(fmath::div(a[6u*i+j],norm),w[i]);
        ++qrank;
    }
    for(unsigned t=0u;t<n;++t)for(unsigned i=0u;i<3u;++i)
        for(unsigned j=qrank;j<6u;++j)p[18u*members[begin+t]+6u*i+j]=0.0f;
    for(unsigned k=0u;k<36u;++k)transform[36u*node+k]=r[k];
    rank[node]=qrank;
}
// One thread per coarse row. Canonical aggregate/scalar ordering produces the
// transpose from the SAME sum (not an independently rounded reverse sum).
// Reads every fine CSR additive duplicate exactly once per canonical entry.
[[seam::entry(row)]]
[[seam::device_fn]] inline void rmc_galerkin(
    const unsigned *member_offset,const unsigned *members,const unsigned *owner,
    const unsigned *fine_offset,const unsigned *fine_column,const float *fine_value,
    const float *p,const unsigned *rank,const unsigned *offset,
    const unsigned *column,float *value,unsigned *fault,unsigned row,unsigned extent) {
    if(row>=extent)return;
    for(unsigned block=offset[row];block<offset[row+1u];++block) {
        unsigned col=column[block];
        for(unsigned i=0u;i<3u;++i)for(unsigned j=0u;j<3u;++j) {
            unsigned ar=row/2u,ac=col/2u,si=3u*(row%2u)+i,sj=3u*(col%2u)+j;
            if(ar>ac || (ar==ac && si>sj)) {
                unsigned temp=ar;ar=ac;ac=temp;temp=si;si=sj;sj=temp;
            }
            float sum=0.0f;
            if(si>=rank[ar] || sj>=rank[ac]) {
                sum=ar==ac && si==sj?1.0f:0.0f;
            } else {
                // Both CSR orientations contribute to sym(P^T B P). They are
                // mathematically equal for symmetric B, but need not have the
                // same duplicate order/float rounding. Do not discard either.
                float oriented[2]={0.0f,0.0f};
                for(unsigned pass=0u;pass<2u;++pass) {
                    unsigned aa=pass==0u?ar:ac,bb=pass==0u?ac:ar;
                    unsigned ii=pass==0u?si:sj,jj=pass==0u?sj:si;
                    for(unsigned t=member_offset[aa];t<member_offset[aa+1u];++t) {
                        unsigned fr=members[t];
                        for(unsigned k=fine_offset[fr];k<fine_offset[fr+1u];++k) {
                            unsigned fc=fine_column[k];if(owner[fc]!=bb)continue;
                            for(unsigned u=0u;u<3u;++u)for(unsigned v=0u;v<3u;++v)
                                oriented[pass]+=p[18u*fr+6u*u+ii]*fine_value[9u*k+u+3u*v]
                                    *p[18u*fc+6u*v+jj];
                        }
                    }
                }
                sum=0.5f*oriented[0]+0.5f*oriented[1];
            }
            if(!rmc_finite(sum)){fault[row]=65u;return;}
            value[9u*block+i+3u*j]=sum;
        }
    }
}
[[seam::entry(node)]]
[[seam::device_fn]] inline void rmc_restrict(
    const unsigned *member_offset,const unsigned *members,const float *p,
    const float *fine,const unsigned *rank,float *coarse,unsigned *fault,
    unsigned node,unsigned extent) {
    if(node>=extent)return;
    for(unsigned j=0u;j<6u;++j) {
        float sum=0.0f;
        if(j<rank[node])for(unsigned t=member_offset[node];t<member_offset[node+1u];++t) {
            unsigned r=members[t];for(unsigned i=0u;i<3u;++i)
                sum+=p[18u*r+6u*i+j]*fine[3u*r+i];
        }
        if(!rmc_finite(sum)){fault[node]=66u;return;}
        coarse[6u*node+j]=sum;
    }
}
[[seam::entry(row)]]
[[seam::device_fn]] inline void rmc_prolong_add(
    const unsigned *owner,const float *p,const float *coarse,float *fine,
    unsigned *fault,unsigned row,unsigned extent) {
    if(row>=extent)return;unsigned node=owner[row];
    float out[3];
    for(unsigned i=0u;i<3u;++i) {
        float sum=0.0f;for(unsigned j=0u;j<6u;++j)
            sum+=p[18u*row+6u*i+j]*coarse[6u*node+j];
        out[i]=fine[3u*row+i]+sum;
        if(!rmc_finite(out[i])){fault[row]=67u;return;}
    }
    for(unsigned i=0u;i<3u;++i)fine[3u*row+i]=out[i];
}
