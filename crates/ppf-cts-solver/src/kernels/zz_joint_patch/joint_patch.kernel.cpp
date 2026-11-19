#pragma once

// Every patch contains two independent units (a free 3-DOF node or a full
// 6-DOF rigid body). Consequently four triplets and twelve scalars suffice.
[[seam::host_device_fn]] inline unsigned jp_unit(unsigned row,unsigned cloth) {
    return row < cloth ? row : cloth+(row-cloth)/2u;
}
[[seam::host_device_fn]] inline unsigned jp_axis(unsigned row,unsigned axis,unsigned cloth) {
    return row < cloth ? axis : 3u*((row-cloth)%2u)+axis;
}
[[seam::host_device_fn]] inline float jp_q(const float* q,unsigned a,unsigned ai,unsigned b,unsigned bi,unsigned cloth) {
    unsigned u=jp_unit(a,cloth);
    return u==jp_unit(b,cloth) ? q[36u*u+6u*jp_axis(a,ai,cloth)+jp_axis(b,bi,cloth)] : 0.0f;
}
[[seam::host_device_fn]] inline bool jp_finite(float x) {
    return x==x && fmath::abs(x)<=3.402823466e38f;
}

// Assemble the exact principal block of the currently frozen reduced
// preconditioning matrix. Numerical failure disables the prototype before PCG.
// The physical operator is never changed. Store G=L^-1 D so G^T G is SPD.
[[seam::entry(patch)]]
[[seam::device_fn]] inline void joint_patch_factor(
    const unsigned* offset,const unsigned* column,const float* value,
    const unsigned* member_offset,const unsigned* members,const float* q,
    float* inverse,float* shift,unsigned* fault,unsigned cloth,
    unsigned patch,unsigned extent) {
    if(patch>=extent)return;
    unsigned first=member_offset[patch],n=member_offset[patch+1u]-first;
    if(n==0u || n>4u){fault[patch]=1u;return;}
    unsigned d=3u*n;float a[144],t[144],c[144],l[144],s[12];
    for(unsigned i=0;i<d*d;++i)a[i]=0.f;
    for(unsigned ri=0;ri<n;++ri) {
        unsigned row=members[first+ri];
        for(unsigned k=offset[row];k<offset[row+1u];++k)
            for(unsigned ci=0;ci<n;++ci)if(column[k]==members[first+ci])
                for(unsigned r=0;r<3u;++r)for(unsigned col=0;col<3u;++col)
                    a[(3u*ri+r)*d+3u*ci+col]+=value[9u*k+3u*col+r];
    }
    for(unsigned i=0;i<d;++i)for(unsigned j=0;j<d;++j){
        float v=0.f;for(unsigned k=0;k<d;++k)
            v+=jp_q(q,members[first+i/3u],i%3u,members[first+k/3u],k%3u,cloth)*a[k*d+j];
        t[i*d+j]=v;
    }
    for(unsigned i=0;i<d;++i)for(unsigned j=0;j<d;++j){
        float v=0.f;for(unsigned k=0;k<d;++k)
            v+=t[i*d+k]*jp_q(q,members[first+k/3u],k%3u,members[first+j/3u],j%3u,cloth);
        a[i*d+j]=v+(i==j?1.f:0.f)-jp_q(q,members[first+i/3u],i%3u,members[first+j/3u],j%3u,cloth);
    }
    for(unsigned i=0;i<d;++i){float v=fmath::abs(a[i*d+i]);
        s[i]=fmath::div(1.f,fmath::sqrt(v>1.e-30f?v:1.e-30f));
        if(!jp_finite(s[i])){fault[patch]=2u;return;}}
    for(unsigned i=0;i<d;++i)for(unsigned j=0;j<d;++j){
        c[i*d+j]=(0.5f*a[i*d+j]+0.5f*a[j*d+i])*s[i]*s[j];
        if(!jp_finite(c[i*d+j])){fault[patch]=2u;return;}}
    bool success=false;float jitter=0.f;
    for(unsigned attempt=0;attempt<5u;++attempt){
        for(unsigned i=0;i<d*d;++i)l[i]=0.f;bool good=true;
        for(unsigned i=0;i<d && good;++i)for(unsigned j=0;j<=i;++j){
            float x=c[i*d+j]+(i==j?jitter:0.f);
            for(unsigned k=0;k<j;++k)x-=l[i*d+k]*l[j*d+k];
            if(i==j){if(!(x>1.e-12f)||!jp_finite(x)){good=false;break;}l[i*d+j]=fmath::sqrt(x);}
            else l[i*d+j]=fmath::div(x,l[j*d+j]);
            if(!jp_finite(l[i*d+j])){good=false;break;}
        }
        if(good){success=true;break;}jitter=attempt==0u?1.e-7f:10.f*jitter;
    }
    shift[patch]=jitter;
    if(!success){fault[patch]=3u;return;}
    float* g=inverse+144u*patch;for(unsigned k=0;k<144u;++k)g[k]=0.f;
    for(unsigned col=0;col<d;++col)for(unsigned i=col;i<d;++i){
        float x=i==col?s[col]:0.f;for(unsigned k=col;k<i;++k)x-=l[i*d+k]*g[k*d+col];
        g[i*d+col]=fmath::div(x,l[i*d+i]);
        if(!jp_finite(g[i*d+col])){fault[patch]=4u;return;}
    }
    fault[patch]=0u;
}

// First stage writes disjoint per-patch scratch. Per-unit coverage weights are
// applied to both sides, preserving symmetry even for overlapping patches.
[[seam::entry(patch)]]
[[seam::device_fn]] inline void joint_patch_apply(
    const unsigned* member_offset,const unsigned* members,const float* weight,
    const float* inverse,const float* residual,float* scratch,unsigned* fault,
    unsigned patch,unsigned extent) {
    if(patch>=extent)return;
    unsigned first=member_offset[patch],n=member_offset[patch+1u]-first,d=3u*n;
    if(n==0u||n>4u){fault[patch]=5u;return;}
    const float* g=inverse+144u*patch;float y[12];
    for(unsigned i=0;i<12u;++i)scratch[12u*patch+i]=0.f;
    for(unsigned i=0;i<d;++i){float x=0.f;for(unsigned j=0;j<=i;++j){
        unsigned r=members[first+j/3u];x+=g[i*d+j]*weight[r]*residual[3u*r+j%3u];}
        if(!jp_finite(x)){fault[patch]=6u;return;}y[i]=x;}
    for(unsigned i=0;i<d;++i){float x=0.f;for(unsigned j=i;j<d;++j)x+=g[j*d+i]*y[j];
        unsigned r=members[first+i/3u];x*=weight[r];
        if(!jp_finite(x)){for(unsigned k=0;k<12u;++k)scratch[12u*patch+k]=0.f;fault[patch]=6u;return;}scratch[12u*patch+i]=x;}
}

// Reverse incidence gives one writer per output row and avoids floating point
// atomic races when many joints share a body.
[[seam::entry(row)]]
[[seam::device_fn]] inline void joint_patch_add(
    const unsigned* incident_offset,const unsigned* incident,const float* scratch,
    float* result,unsigned row,unsigned extent) {
    if(row>=extent)return;
    float v[3]={0.f,0.f,0.f};
    for(unsigned k=incident_offset[row];k<incident_offset[row+1u];++k){
        unsigned slot=incident[k],p=slot/4u,l=slot%4u;
        for(unsigned j=0;j<3u;++j)v[j]+=scratch[12u*p+3u*l+j];
    }
    for(unsigned j=0;j<3u;++j)result[3u*row+j]+=v[j];
}
