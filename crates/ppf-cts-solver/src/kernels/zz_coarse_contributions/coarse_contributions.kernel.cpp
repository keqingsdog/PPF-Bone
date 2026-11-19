#pragma once
[[seam::device_fn]] inline bool rmc_contrib_finite(float x) {
    return x==x && x<=3.402823466e+38f && x>=-3.402823466e+38f;
}
// Records are ordered exactly as old member/CSR loops. Slots are explicit so
// a future multi-support producer need not encode one owner per fine row.
[[seam::entry(row)]]
[[seam::device_fn]] inline void rmc_contribution_galerkin(
    const unsigned *pair_offsets,const unsigned *reverse,const unsigned *block_pair,
    const unsigned *records,const float *fine_value,const float *p,const unsigned *rank,
    const unsigned *offset,const unsigned *column,float *value,unsigned *fault,
    unsigned row,unsigned extent) {
    if(row>=extent)return;
    for(unsigned block=offset[row];block<offset[row+1u];++block) {
        unsigned col=column[block];
        for(unsigned i=0u;i<3u;++i)for(unsigned j=0u;j<3u;++j) {
            unsigned ar=row/2u,ac=col/2u,si=3u*(row%2u)+i,sj=3u*(col%2u)+j;
            unsigned forward=block_pair[block];
            if(ar>ac || (ar==ac && si>sj)) {
                unsigned temp=ar;ar=ac;ac=temp;temp=si;si=sj;sj=temp;
                forward=reverse[forward];
            }
            float sum=0.0f;
            if(si>=rank[ar] || sj>=rank[ac]) sum=ar==ac && si==sj?1.0f:0.0f;
            else {
                float oriented[2]={0.0f,0.0f};
                for(unsigned pass=0u;pass<2u;++pass) {
                    unsigned id=pass==0u?forward:reverse[forward];
                    unsigned ii=pass==0u?si:sj,jj=pass==0u?sj:si;
                    for(unsigned q=pair_offsets[id];q<pair_offsets[id+1u];++q) {
                        unsigned k=records[4u*q+1u],left=records[4u*q+2u],right=records[4u*q+3u];
                        for(unsigned u=0u;u<3u;++u)for(unsigned v=0u;v<3u;++v)
                            oriented[pass]+=p[18u*left+6u*u+ii]*fine_value[9u*k+u+3u*v]
                                *p[18u*right+6u*v+jj];
                    }
                }
                sum=0.5f*oriented[0]+0.5f*oriented[1];
            }
            if(!rmc_contrib_finite(sum)){fault[row]=65u;return;}
            value[9u*block+i+3u*j]=sum;
        }
    }
}

