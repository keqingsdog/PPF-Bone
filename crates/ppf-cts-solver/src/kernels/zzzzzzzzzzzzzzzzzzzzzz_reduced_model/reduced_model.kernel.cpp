// Staged neutral implementation; append generated IDs only during integration.
// B = full J^T(A+B+C)J + complete joint H + explicit seed symmetry/shift delta.
// Body seed-counted direct diagonals excluded from CSR projection; subtract EXACT
// original analytic inertia before projecting full C. All 3x3 blocks column-major.
#pragma once
#include "../energy/model/pdrd_rigid.kernel.cpp"

[[seam::device_fn]] inline bool rsm_finite(float x) {
    return x == x && x <= 3.402823466e38f && x >= -3.402823466e38f;
}

// This is a MODEL operation, performed before CSR assembly and independent of
// the partition. Every later restriction/Galerkin level sees the same SPD seed.
// Input raw body excludes material-joint additions. Preserve symmetric SPD raw
// input exactly at zero shift. Symmetric projection and physical diagonal shift
// are explicitly materialized rather than hidden in a partition-local factor.
[[seam::entry(body)]]
[[seam::device_fn]] inline void rsm_body_seed(
    const float *raw_body,float *body_seed,float *body_shift,unsigned *fault,
    unsigned body,unsigned extent) {
    if(body>=extent) return;
    const float *raw=raw_body+36u*body;
    float a[36],c[36],l[36],scale[6];
    bool good=true;
    for(unsigned i=0u;i<6u;++i) for(unsigned j=0u;j<6u;++j) {
        a[6u*i+j]=0.5f*raw[6u*i+j]+0.5f*raw[6u*j+i];
        good=good && rsm_finite(raw[6u*i+j]) && rsm_finite(a[6u*i+j]);
    }
    for(unsigned i=0u;i<6u;++i) {
        const float diag=fmath::abs(a[6u*i+i]);
        scale[i]=fmath::div(1.0f,fmath::sqrt(diag>1.0e-30f?diag:1.0e-30f));
    }
    for(unsigned i=0u;i<6u;++i) for(unsigned j=0u;j<6u;++j) {
        c[6u*i+j]=a[6u*i+j]*scale[i]*scale[j];
        good=good && rsm_finite(c[6u*i+j]);
    }
    if(!good) {fault[body]=10u;return;}
    float shift=0.0f;bool success=false;
    for(unsigned attempt=0u;attempt<13u;++attempt) {
        good=true;
        for(unsigned k=0u;k<36u;++k) l[k]=0.0f;
        for(unsigned i=0u;i<6u && good;++i) for(unsigned j=0u;j<=i;++j) {
            float q=c[6u*i+j]+(i==j?shift:0.0f);
            for(unsigned k=0u;k<j;++k) q-=l[6u*i+k]*l[6u*j+k];
            if(i==j) {
                const float minimum=shift==0.0f?1.0e-12f:1.0e-6f;
                if(!(q>minimum) || !rsm_finite(q)) {good=false;break;}
                l[6u*i+j]=fmath::sqrt(q);
            } else l[6u*i+j]=fmath::div(q,l[6u*j+j]);
            good=good && rsm_finite(l[6u*i+j]);
            if(!good) break;
        }
        if(good) {success=true;break;}
        shift=attempt==0u?1.0e-7f:shift*10.0f;
    }
    body_shift[body]=shift;
    if(!success) {fault[body]=11u;return;}
    for(unsigned i=0u;i<6u;++i) for(unsigned j=0u;j<6u;++j) {
        // Equivalent to D^-1*(C+shift I)*D^-1, but avoiding
        // reconstruction roundoff of every original physical off-diagonal.
        const float physical_shift=i==j?fmath::div(fmath::div(shift,scale[i]),scale[i]):0.0f;
        const float value=a[6u*i+j]+physical_shift;
        body_seed[36u*body+6u*i+j]=value;
        good=good && rsm_finite(value);
    }
    if(!good) {fault[body]=12u;return;}
    // Validate the ACTUALLY materialized physical seed as well: forming a
    // shifted diagonal in physical units introduces one final float rounding.
    for(unsigned i=0u;i<6u;++i) for(unsigned j=0u;j<6u;++j)
        c[6u*i+j]=body_seed[36u*body+6u*i+j]*scale[i]*scale[j];
    for(unsigned k=0u;k<36u;++k) l[k]=0.0f;
    for(unsigned i=0u;i<6u && good;++i) for(unsigned j=0u;j<=i;++j) {
        float q=c[6u*i+j];
        for(unsigned k=0u;k<j;++k) q-=l[6u*i+k]*l[6u*j+k];
        if(i==j) {
            if(!(q>1.0e-12f) || !rsm_finite(q)) {good=false;break;}
            l[6u*i+j]=fmath::sqrt(q);
        } else l[6u*i+j]=fmath::div(q,l[6u*j+j]);
        good=good && rsm_finite(l[6u*i+j]);
        if(!good) break;
    }
    fault[body]=good?0u:13u;
}

[[seam::device_fn]] inline unsigned rsm_free(const unsigned *vertex_to_free,
                                           unsigned vert) {
    return vertex_to_free[vert];
}

// One free row owns its base entries. Body row owns two raw 6x6 subblocks.
// row_to_vertex is the actual inverse free map, NOT an assumption about vertex ordering.
[[seam::device_fn]] inline unsigned rsm_projection_parts(unsigned v,
    const unsigned* vertex_body,const unsigned* vertex_to_free,unsigned source_rows,
    unsigned cloth,unsigned bodies,unsigned rows[2],compute::atomic_uint_t* fault) {
    if(v>=source_rows){compute::atomic_add(fault,1u);return 0u;}
    const unsigned body=vertex_body[v];
    if(body==0xffffffffu){
        const unsigned free=vertex_to_free[v];
        if(free>=cloth){compute::atomic_add(fault,1u);return 0u;}
        rows[0]=free;return 1u;
    }
    if(body>=bodies){compute::atomic_add(fault,1u);return 0u;}
    rows[0]=cloth+2u*body;rows[1]=rows[0]+1u;return 2u;
}
[[seam::device_fn]] inline void rsm_projection_count(unsigned v,unsigned w,bool seed_diagonal,
    const unsigned* vertex_body,const unsigned* vertex_to_free,unsigned source_rows,
    unsigned cloth,unsigned bodies,const unsigned left[2],unsigned nl,
    compute::atomic_uint_t* row_count,compute::atomic_uint_t* fault) {
    // Raw seed contains only dynamic/fixed SAME FULL-VERTEX diagonal, not all
    // same-body edges. Reference/transpose spans always remain additive.
    if(seed_diagonal&&v==w&&vertex_body[v]!=0xffffffffu)return;
    unsigned right[2];const unsigned nr=rsm_projection_parts(w,vertex_body,vertex_to_free,source_rows,cloth,bodies,right,fault);
    for(unsigned p=0;p<nl;++p)compute::atomic_add(row_count+left[p],nr);
}
[[seam::device_fn]] inline Vec3f rsm_projection_basis(const unsigned* vertex_body,
    const float* rotated_rest,unsigned v,unsigned part,unsigned col) {
    Vec3f t=Vec3f::Zero(),w=Vec3f::Zero();
    if(part==0u)t[col]=1.f;else w[col]=1.f;
    if(vertex_body[v]==0xffffffffu)return t;
    const Vec3f p(rotated_rest[3u*v],rotated_rest[3u*v+1u],rotated_rest[3u*v+2u]);
    // Authoritative original reduction convention, not a second skew formula.
    return pdrd_prolong(t,w,p);
}
[[seam::device_fn]] inline unsigned rsm_projection_reserve(unsigned row,unsigned n,
    const unsigned* offset,const unsigned* base_count,compute::atomic_uint_t* cursor,
    compute::atomic_uint_t* fault) {
    const unsigned begin=offset[row],all_end=offset[row+1u];
    if(base_count[row]>all_end-begin){compute::atomic_add(fault,1u);return 0xffffffffu;}
    const unsigned end=begin+base_count[row],write=compute::atomic_add(cursor+row,n);
    if(write>end||n>end-write){compute::atomic_add(fault,1u);return 0xffffffffu;}
    return write;
}
[[seam::device_fn]] inline void rsm_projection_put(unsigned v,unsigned w,bool seed_diagonal,
    const float* a,bool transpose,const unsigned* vertex_body,const unsigned* vertex_to_free,
    const float* rotated_rest,unsigned source_rows,unsigned cloth,unsigned bodies,
    const unsigned left[2],unsigned nl,const unsigned* offset,const unsigned* base_count,
    compute::atomic_uint_t* cursor,unsigned* column,float* value,compute::atomic_uint_t* fault) {
    if(seed_diagonal&&v==w&&vertex_body[v]!=0xffffffffu)return;
    unsigned right[2];const unsigned nr=rsm_projection_parts(w,vertex_body,vertex_to_free,source_rows,cloth,bodies,right,fault);
    for(unsigned lp=0;lp<nl;++lp){
        const unsigned write=rsm_projection_reserve(left[lp],nr,offset,base_count,cursor,fault);
        if(write==0xffffffffu)continue;
        for(unsigned rp=0;rp<nr;++rp){
            column[write+rp]=right[rp];bool finite=true;
            for(unsigned c=0;c<3;++c){
                const Vec3f jc=rsm_projection_basis(vertex_body,rotated_rest,w,rp,c);
                Vec3f product=Vec3f::Zero();
                for(unsigned r=0;r<3;++r)for(unsigned k=0;k<3;++k)
                    product[r]+=a[transpose?3u*r+k:3u*k+r]*jc[k];
                for(unsigned r=0;r<3;++r){
                    const Vec3f jr=rsm_projection_basis(vertex_body,rotated_rest,v,lp,r);
                    float q=0.f;for(unsigned k=0;k<3;++k)q+=jr[k]*product[k];
                    value[9u*(write+rp)+3u*c+r]=q;finite=finite&&rsm_finite(q);
                }
            }
            if(!finite)compute::atomic_add(fault,1u);
        }
    }
}
// Per-full-vertex traversal. A source block emits 1/2/4 projected blocks.
[[seam::entry(row)]]
[[seam::device_fn]] inline void rsm_base_count(
    const unsigned *vertex_body, const unsigned *vertex_to_free,
    const unsigned *dynamic_index, const unsigned *dynamic_offset,
    const unsigned *reference_index, const unsigned *reference_offset,
    const unsigned *fixed_index, const unsigned *fixed_offset,
    const unsigned *transpose_pair, const unsigned *transpose_offset,
    compute::atomic_uint_t *row_count,compute::atomic_uint_t* projection_fault,
    unsigned n_cloth,unsigned bodies,unsigned row,unsigned extent) {
    const unsigned reduced=n_cloth+2u*bodies;
    if(row<reduced)compute::atomic_add(row_count+row,row<n_cloth?1u:2u);
    if(row>=extent)return;
    unsigned left[2];const unsigned nl=rsm_projection_parts(row,vertex_body,vertex_to_free,extent,n_cloth,bodies,left,projection_fault);
    for(unsigned k=dynamic_offset[row];k<dynamic_offset[row+1u];++k)
        rsm_projection_count(row,dynamic_index[k],true,vertex_body,vertex_to_free,extent,n_cloth,bodies,left,nl,row_count,projection_fault);
    for(unsigned k=reference_offset[row];k<reference_offset[row+1u];++k)
        rsm_projection_count(row,reference_index[k],false,vertex_body,vertex_to_free,extent,n_cloth,bodies,left,nl,row_count,projection_fault);
    for(unsigned k=fixed_offset[row];k<fixed_offset[row+1u];++k)
        rsm_projection_count(row,fixed_index[k],true,vertex_body,vertex_to_free,extent,n_cloth,bodies,left,nl,row_count,projection_fault);
    for(unsigned k=transpose_offset[row];k<transpose_offset[row+1u];++k)
        rsm_projection_count(row,transpose_pair[2u*k],false,vertex_body,vertex_to_free,extent,n_cloth,bodies,left,nl,row_count,projection_fault);
    // Full body C includes analytic contact, drag and pulls in addition to
    // point inertia. Private analytic inertia is subtracted from body seed.
    if(vertex_body[row]!=0xffffffffu)
        for(unsigned lp=0;lp<nl;++lp)compute::atomic_add(row_count+left[lp],nl);
}

// Local ID list is already alias-coalesced by frozen exporter. Build unique
// triplet rows; no local scalar can appear twice and WORLD has no scalar ID.
[[seam::device_fn]] inline unsigned rsm_triplets(
    const unsigned *ids, unsigned n, unsigned *triplet) {
    unsigned used = 0u;
    for (unsigned i=0u; i<n; ++i) {
        const unsigned r = ids[i]/3u;
        bool found = false;
        for (unsigned j=0u; j<used; ++j) found = found || triplet[j] == r;
        if (!found) triplet[used++] = r;
    }
    return used;
}

// Runs AFTER base_count, BEFORE exclusive scan. row_count[nrows] must be zero.
// Faults are checked before scan/fill: invalid exports cannot corrupt offsets.
[[seam::entry(record)]]
[[seam::device_fn]] inline void rsm_joint_count(
    const unsigned *scalar_ids, const unsigned *scalar_count,
    compute::atomic_uint_t *row_count, unsigned *fault, unsigned nrows,
    unsigned record, unsigned extent) {
    if (record>=extent) return;
    const unsigned n = scalar_count[record];
    const unsigned *ids = scalar_ids + 24u*record;
    unsigned invalid = n > 24u;
    for (unsigned i=0u; i<n && i<24u; ++i) {
        invalid |= ids[i] >= 3u*nrows;
        for (unsigned j=0u; j<i; ++j) invalid |= ids[i] == ids[j];
    }
    fault[record] = invalid;
    if (invalid) return;
    unsigned triplet[24];
    const unsigned used = rsm_triplets(ids, n, triplet);
    for (unsigned i=0u; i<used; ++i)
        compute::atomic_add(row_count+triplet[i], used);
}

[[seam::device_fn]] inline void rsm_copy(
    const float *source, float *target, bool transpose) {
    for (unsigned r=0u; r<3u; ++r)
        for (unsigned c=0u; c<3u; ++c)
            target[3u*c+r] = source[transpose ? 3u*r+c : 3u*c+r];
}

[[seam::device_fn]] inline void rsm_put(
    unsigned col, const unsigned *vertex_to_free, const float *source,
    unsigned *column, float *value, unsigned &write, bool transpose) {
    const unsigned free = rsm_free(vertex_to_free, col);
    if (free == 0xffffffffu) return;
    column[write] = free;
    rsm_copy(source, value+9u*write, transpose);
    ++write;
}

// Body seed is row-major36 AFTER rsm_body_seed, before material additions.
// Cloth diagonal is original operator.diagonal,
// NOT the material-augmented BJ precond diagonal. All four operator spans match
// schwarz_factor_gather exactly, including transpose-pair addressing.
[[seam::entry(row)]]
[[seam::device_fn]] inline void rsm_base_fill(
    const unsigned *row_to_vertex,const unsigned *vertex_body,const unsigned *vertex_to_free,
    const float *rotated_rest,const PdrdRigidState* body_state,
    const unsigned *dynamic_index, const float *dynamic_value,
    const unsigned *dynamic_offset, const unsigned *reference_index,
    const unsigned *reference_value, const unsigned *reference_offset,
    const float *global_value, const unsigned *fixed_index,
    const unsigned *fixed_offset, const float *fixed_value,
    const unsigned *transpose_pair, const unsigned *transpose_offset,
    const float *diagonal,const float *body_seed,const float* body_inertia,
    const unsigned *offset,const unsigned* base_count,compute::atomic_uint_t *cursor,
    unsigned *column,float *value,compute::atomic_uint_t* projection_fault,
    unsigned n_cloth,unsigned bodies,unsigned row,unsigned extent) {
    const unsigned reduced=n_cloth+2u*bodies;
    // Exactly one thread seeds each reduced row, competing only via checked
    // cursor reservations with the per-full-vertex projection writers below.
    if(row<reduced){
        const unsigned n=row<n_cloth?1u:2u;
        const unsigned write=rsm_projection_reserve(row,n,offset,base_count,cursor,projection_fault);
        if(write!=0xffffffffu){
            if(row<n_cloth){column[write]=row;rsm_copy(diagonal+9u*row_to_vertex[row],value+9u*write,false);}
            else{
                const unsigned b=(row-n_cloth)/2u,part=(row-n_cloth)%2u;
                for(unsigned cp=0;cp<2;++cp){column[write+cp]=n_cloth+2u*b+cp;bool finite=true;
                    for(unsigned r=0;r<3;++r)for(unsigned c=0;c<3;++c){
                        const unsigned k=36u*b+(3u*part+r)*6u+3u*cp+c;
                        // Empty bodies retain original identity seed; no full
                        // vertex C exists for them and original inertia emits I.
                        const float q=body_seed[k]-(body_state[b].N?body_inertia[k]:0.f);
                        value[9u*(write+cp)+3u*c+r]=q;finite=finite&&rsm_finite(q);
                    }
                    if(!finite)compute::atomic_add(projection_fault,1u);
                }
            }
        }
    }
    if(row>=extent)return;
    unsigned left[2];const unsigned nl=rsm_projection_parts(row,vertex_body,vertex_to_free,extent,n_cloth,bodies,left,projection_fault);
    for(unsigned k=dynamic_offset[row];k<dynamic_offset[row+1u];++k)
        rsm_projection_put(row,dynamic_index[k],true,dynamic_value+9u*k,false,vertex_body,vertex_to_free,rotated_rest,extent,n_cloth,bodies,left,nl,offset,base_count,cursor,column,value,projection_fault);
    for(unsigned k=reference_offset[row];k<reference_offset[row+1u];++k)
        rsm_projection_put(row,reference_index[k],false,global_value+9u*reference_value[k],true,vertex_body,vertex_to_free,rotated_rest,extent,n_cloth,bodies,left,nl,offset,base_count,cursor,column,value,projection_fault);
    for(unsigned k=fixed_offset[row];k<fixed_offset[row+1u];++k)
        rsm_projection_put(row,fixed_index[k],true,fixed_value+9u*k,false,vertex_body,vertex_to_free,rotated_rest,extent,n_cloth,bodies,left,nl,offset,base_count,cursor,column,value,projection_fault);
    for(unsigned k=transpose_offset[row];k<transpose_offset[row+1u];++k)
        rsm_projection_put(row,transpose_pair[2u*k],false,fixed_value+9u*transpose_pair[2u*k+1u],true,vertex_body,vertex_to_free,rotated_rest,extent,n_cloth,bodies,left,nl,offset,base_count,cursor,column,value,projection_fault);
    if(vertex_body[row]!=0xffffffffu)
        rsm_projection_put(row,row,false,diagonal+9u*row,false,vertex_body,vertex_to_free,rotated_rest,extent,n_cloth,bodies,left,nl,offset,base_count,cursor,column,value,projection_fault);
}

[[seam::entry(record)]]
[[seam::device_fn]] inline void rsm_joint_fill(
    const unsigned *scalar_ids, const unsigned *scalar_count,
    const float *local_hessian, compute::atomic_uint_t *cursor,
    unsigned *column, float *value, unsigned record, unsigned extent) {
    if (record>=extent) return;
    const unsigned n=scalar_count[record];
    const unsigned *ids=scalar_ids+24u*record;
    const float *h=local_hessian+576u*record;
    unsigned triplet[24];
    const unsigned used=rsm_triplets(ids,n,triplet);
    for (unsigned ri=0u; ri<used; ++ri) {
        const unsigned write=compute::atomic_add(cursor+triplet[ri],used);
        for (unsigned ci=0u; ci<used; ++ci) {
            column[write+ci]=triplet[ci];
            for (unsigned k=0u;k<9u;++k) value[9u*(write+ci)+k]=0.0f;
            for (unsigned i=0u;i<n;++i) if (ids[i]/3u==triplet[ri])
                for (unsigned j=0u;j<n;++j) if (ids[j]/3u==triplet[ci]) {
                    // Explicit symmetric projection addresses exporter roundoff;
                    // the global operator is untouched. Same pair formula in
                    // both triangles; no one-triangle atomic accumulation.
                    value[9u*(write+ci)+3u*(ids[j]%3u)+ids[i]%3u] +=
                        0.5f*h[24u*i+j]+0.5f*h[24u*j+i];
                }
        }
    }
}

// Check cursor closure and finite B values without downloading numerical B.
[[seam::entry(row)]]
[[seam::device_fn]] inline void rsm_check(
    const unsigned *offset,const unsigned *cursor,const unsigned *column,
    const float *value,unsigned *fault,unsigned row,unsigned extent) {
    if (row>=extent) return;
    unsigned bad=cursor[row]!=offset[row+1u];
    for (unsigned k=offset[row];k<offset[row+1u];++k) {
        bad |= column[k]>=extent;
        for (unsigned j=0u;j<9u;++j) bad |= !rsm_finite(value[9u*k+j]);
    }
    fault[row]=bad;
}

// One thread owns a complete domain. CSR duplicate columns ADD, preserving B.
// Dense is compact d*d row-major; storage stride is 48*48 per domain.
[[seam::entry(domain)]]
[[seam::device_fn]] inline void rsm_gather(
    const unsigned *offset,const unsigned *column,const float *value,
    const unsigned *member_offset,const unsigned *members,float *dense,
    unsigned *fault,unsigned domain,unsigned extent) {
    if (domain>=extent) return;
    const unsigned first=member_offset[domain], n=member_offset[domain+1u]-first;
    if (n==0u || n>16u) {fault[domain]=1u;return;}
    const unsigned d=3u*n;
    float *a=dense+2304u*domain;
    for (unsigned k=0u;k<d*d;++k) a[k]=0.0f;
    for (unsigned ri=0u;ri<n;++ri) {
        const unsigned row=members[first+ri];
        for (unsigned k=offset[row];k<offset[row+1u];++k)
            for (unsigned ci=0u;ci<n;++ci) if (column[k]==members[first+ci])
                for (unsigned r=0u;r<3u;++r)
                    for (unsigned c=0u;c<3u;++c)
                        a[(3u*ri+r)*d+3u*ci+c]+=value[9u*k+3u*c+r];
    }
    fault[domain]=0u;
}

// Mixed translation/rotation units require D*A*D, D_i=1/sqrt(abs(A_ii)).
// Retry diagonal shifts in NORMALIZED units until actual Cholesky succeeds.
// There is no per-pivot clamping: shift=0 factors the exact symmetric model;
// if roundoff/indefiniteness prevents it we record the actual applied shift.
// inverse stores G=L^-1 * D; apply G^T G restores physical units exactly.
[[seam::entry(domain)]]
[[seam::device_fn]] inline void rsm_factor(
    const unsigned *member_offset,const float *dense,float *normalized,
    float *lower,float *inverse,float *scale,float *shift,unsigned *fault,
    unsigned domain,unsigned extent) {
    if (domain>=extent) return;
    const unsigned n=member_offset[domain+1u]-member_offset[domain];
    if(n==0u || n>16u) {fault[domain]=2u;return;}
    const unsigned d=3u*n, base=2304u*domain;
    const float *a=dense+base;
    float *c=normalized+base,*l=lower+base,*g=inverse+base,*s=scale+48u*domain;
    bool good=true;
    for(unsigned i=0u;i<d;++i) {
        const float diag=fmath::abs(a[i*d+i]);
        // Avoid dividing by a zero scale while preserving small valid units.
        s[i]=fmath::div(1.0f,fmath::sqrt(diag>1.0e-30f?diag:1.0e-30f));
        good=good && rsm_finite(s[i]);
    }
    for(unsigned i=0u;i<d;++i) for(unsigned j=0u;j<d;++j) {
        c[i*d+j]=(0.5f*a[i*d+j]+0.5f*a[j*d+i])*s[i]*s[j];
        good=good && rsm_finite(c[i*d+j]);
    }
    if(!good) {fault[domain]=3u;return;}
    float regularization=0.0f;
    bool success=false;
    // This is only local float roundoff jitter, not body model repair. Larger
    // indefiniteness is a refused model/factor mismatch, never silently hidden.
    for(unsigned attempt=0u;attempt<5u;++attempt) {
        good=true;
        for(unsigned k=0u;k<d*d;++k) l[k]=0.0f;
        for(unsigned i=0u;i<d && good;++i) for(unsigned j=0u;j<=i;++j) {
            float q=c[i*d+j]+(i==j?regularization:0.0f);
            for(unsigned k=0u;k<j;++k) q-=l[i*d+k]*l[j*d+k];
            if(i==j) {
                if(!(q>1.0e-12f) || !rsm_finite(q)) {good=false;break;}
                l[i*d+j]=fmath::sqrt(q);
            } else l[i*d+j]=fmath::div(q,l[j*d+j]);
            good=good && rsm_finite(l[i*d+j]);
            if(!good) break;
        }
        if(good) {success=true;break;}
        regularization=attempt==0u?1.0e-7f:regularization*10.0f;
    }
    shift[domain]=regularization;
    if(!success) {fault[domain]=4u;return;}
    for(unsigned k=0u;k<d*d;++k) g[k]=0.0f;
    for(unsigned col=0u;col<d;++col) for(unsigned i=col;i<d;++i) {
        float q=i==col?s[col]:0.0f;
        for(unsigned k=col;k<i;++k) q-=l[i*d+k]*g[k*d+col];
        g[i*d+col]=fmath::div(q,l[i*d+i]);
        good=good && rsm_finite(g[i*d+col]);
    }
    fault[domain]=good?0u:5u;
}

// Native counters have an explicit sticky overflow witness in receipt[3].
[[seam::device_fn]] inline void rsm_receipt_add(compute::atomic_uint_t *receipt,unsigned field) {
    const unsigned old=compute::atomic_add(receipt+field,1u);
    if(old==0xffffffffu) compute::atomic_add(receipt+3u,1u);
}

// Disjoint complete domains, so each output scalar has exactly one writer.
// Caller applies original reduced projectors before AND after this operation.
[[seam::entry(domain)]]
[[seam::device_fn]] inline void rsm_apply(
    const unsigned *member_offset,const unsigned *members,const float *inverse,
    const float *residual,float *result,unsigned *apply_fault,compute::atomic_uint_t *receipt,
    unsigned domain,unsigned extent) {
    if (domain>=extent) return;
    if(domain==0u) rsm_receipt_add(receipt,0u);
    const unsigned first=member_offset[domain], n=member_offset[domain+1u]-first;
    if(n==0u || n>16u) {apply_fault[domain]=20u;rsm_receipt_add(receipt,2u);return;}
    const unsigned d=3u*n;
    const float *g=inverse+2304u*domain;
    float y[48],out[48];bool finite=true;
    for(unsigned i=0u;i<d;++i) {
        y[i]=0.0f;
        for(unsigned j=0u;j<=i;++j)
            y[i]+=g[i*d+j]*residual[3u*members[first+j/3u]+j%3u];
    }
    for(unsigned i=0u;i<d;++i) {
        float z=0.0f;
        for(unsigned j=i;j<d;++j) z+=g[j*d+i]*y[j];
        out[i]=z;finite=finite && rsm_finite(z) && rsm_finite(y[i]);
    }
    if(!finite) {apply_fault[domain]=21u;rsm_receipt_add(receipt,2u);return;}
    for(unsigned i=0u;i<d;++i) result[3u*members[first+i/3u]+i%3u]=out[i];
    rsm_receipt_add(receipt,1u);
}




