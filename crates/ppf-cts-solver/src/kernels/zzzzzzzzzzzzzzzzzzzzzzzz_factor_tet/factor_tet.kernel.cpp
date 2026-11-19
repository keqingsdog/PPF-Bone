// Frozen original spectral modes, relative-increment physical tet operator.
// Lambda/PSD branch expressions below copied verbatim from pinned original
// tet_spectral_hessian_fused; output stores p.col(1..3), never rounded p.col(0).
#pragma once
#include "../energy/tet_force.kernel.cpp"
[[seam::device_fn]] inline void ftt_capture_modes(
    const Vec3f &gradient_sigma,
    const Mat3x3f &hessian_sigma,
    const Mat3x3f &u, const Vec3f &sigma,
    const Mat3x3f &vt, float eps,
    const Mat3x3f &inverse_rest, float mass,
    float damping_scale,float *factor,float *weight,unsigned element) {
    Vec3f eigenvalues;
    Mat3x3f eigenvectors;
    linalg::eig::symm3x3(hessian_sigma, eigenvalues, eigenvectors);
    const float a = sigma[0];
    const float b = sigma[1];
    const float c = sigma[2];
    const float difference_ab = a - b;
    const float difference_ac = a - c;
    const float difference_bc = b - c;
    Vec9f lambda;
    lambda[0] = fmath::max(
        0.0f, fmath::div(gradient_sigma[0] + gradient_sigma[1], a + b));
    lambda[1] = fmath::max(
        0.0f, fmath::div(gradient_sigma[0] + gradient_sigma[2], a + c));
    lambda[2] = fmath::max(
        0.0f, fmath::div(gradient_sigma[1] + gradient_sigma[2], b + c));
    lambda[3] =
        fmath::max(0.0f,
                 fmath::abs(difference_ab) > eps
                     ? fmath::div(gradient_sigma[0] - gradient_sigma[1],
                                difference_ab)
                     : 0.5f * (hessian_sigma(0, 0) + hessian_sigma(1, 1)) -
                           0.5f *
                               (hessian_sigma(0, 1) + hessian_sigma(1, 0)));
    lambda[4] =
        fmath::max(0.0f,
                 fmath::abs(difference_ac) > eps
                     ? fmath::div(gradient_sigma[0] - gradient_sigma[2],
                                difference_ac)
                     : 0.5f * (hessian_sigma(0, 0) + hessian_sigma(2, 2)) -
                           0.5f *
                               (hessian_sigma(0, 2) + hessian_sigma(2, 0)));
    lambda[5] =
        fmath::max(0.0f,
                 fmath::abs(difference_bc) > eps
                     ? fmath::div(gradient_sigma[1] - gradient_sigma[2],
                                difference_bc)
                     : 0.5f * (hessian_sigma(1, 1) + hessian_sigma(2, 2)) -
                           0.5f *
                               (hessian_sigma(1, 2) + hessian_sigma(2, 1)));
    lambda[6] = fmath::max(0.0f, eigenvalues[0]);
    lambda[7] = fmath::max(0.0f, eigenvalues[1]);
    lambda[8] = fmath::max(0.0f, eigenvalues[2]);

    Vec3f shape_gradient =
        -inverse_rest.row(0) - inverse_rest.row(1) - inverse_rest.row(2);
    Mat3x4f shape;
    shape.col(0) = shape_gradient;
    shape.col(1) = inverse_rest.row(0);
    shape.col(2) = inverse_rest.row(1);
    shape.col(3) = inverse_rest.row(2);
    const Mat3x4f w = vt * shape;

    constexpr float inverse_sqrt_two = 0.7071067811865475f;
    // Twist (antisymmetric, sign -1) then flip (symmetric, sign +1) index
    // pairs, matching modes 0 through 5 of the non-fused form above.
    const unsigned mode_row[6] = {0u, 0u, 1u, 0u, 0u, 1u};
    const unsigned mode_column[6] = {1u, 2u, 2u, 1u, 2u, 2u};
    for (unsigned i = 0; i < 9; ++i) {
        if (lambda[i] == 0.0f) {
            continue;
        }
        Mat3x4f p;
        if (i < 6) {
            const float sign = (i < 3) ? -1.0f : 1.0f;
            const unsigned r = mode_row[i];
            const unsigned cc = mode_column[i];
            for (unsigned col = 0; col < 4; ++col) {
                p.col(col) = inverse_sqrt_two * (w(cc, col) * u.col(r) +
                                                 sign * w(r, col) * u.col(cc));
            }
        } else {
            const Vec3f v = eigenvectors.col(i - 6);
            for (unsigned col = 0; col < 4; ++col) {
                p.col(col) = v[0] * w(0, col) * u.col(0) +
                             v[1] * w(1, col) * u.col(1) +
                             v[2] * w(2, col) * u.col(2);
            }
        }
        weight[9u*element+i]=(mass*lambda[i])*damping_scale;
        for(unsigned a=1u;a<4u;++a)for(unsigned d=0u;d<3u;++d)
            factor[81u*element+9u*i+3u*(a-1u)+d]=p(d,a);
    }
}

[[seam::device_fn]] inline bool ftt_finite(float x) {
    return x==x && x<=3.402823466e38f && x>=-3.402823466e38f;
}
// Replaces only the tet embed pass. Constitutive/SVD/PSD code and full element
// Hessian for preconditioning use the pinned production functions unchanged.
[[seam::entry(element)]]
[[seam::device_fn]] inline void ftt_embed(
    const Vec3f *x,const Vec3f *current,const unsigned *tet,
    const Mat3x3f *inverse_rest,const unsigned *model,const float *mu,
    const float *lambda,const float *mass,const float *damping,
    float dt,float eigenanalysis_eps,compute::atomic_float_t *force,
    const unsigned *index,const unsigned *offset,compute::atomic_float_t *value,
    float *factor,float *weight,unsigned *fault,unsigned row_count,
    unsigned element,unsigned extent) {
    if(element>=extent)return;
    for(unsigned k=0u;k<81u;++k)factor[81u*element+k]=0.0f;
    for(unsigned k=0u;k<9u;++k)weight[9u*element+k]=0.0f;
    if(mu[element]<=0.0f)return;
    Vec4u ids;Vec3f pose[4],start[4];
    for(unsigned a=0u;a<4u;++a) {
        unsigned v=tet[4u*element+a];
        if(v>=row_count){fault[element]=1u;return;}
        ids[a]=v;pose[a]=x[v];start[a]=current[v];
    }
    const Mat3x3f rest=inverse_rest[element];
    Mat3x3f deformation=tet_deformation_gradient(pose[0],pose[1],pose[2],pose[3],rest);
    Mat3x3f u,vt;Vec3f sigma;svd3x3_rv(deformation,u,sigma,vt);
    DiffTable3 table;
    if(model[element]==ELASTIC_MODEL_ARAP) {
        table=ARAP::make_diff_table3(sigma,mu[element],lambda[element]);
    } else if(model[element]==ELASTIC_MODEL_STVK) {
        table=StVK::make_diff_table3(sigma,mu[element],lambda[element]);
    } else if(model[element]==ELASTIC_MODEL_SNHK) {
        table=SNHk::make_diff_table3(sigma,mu[element],lambda[element]);
    } else {fault[element]=2u;return;}
    Mat3x4f gradient=tet_convert_force(tet_spectral_force(table.deda,u,vt),rest,mass[element]);
    // E^T: one balance definition, not an independently rounded shape g0.
    for(unsigned d=0u;d<3u;++d)
        gradient(d,0)=-(gradient(d,1)+gradient(d,2)+gradient(d,3));
    Mat12x12f hessian=Mat12x12f::Zero();
    tet_spectral_hessian_fused(table.deda,table.d2ed2a,u,sigma,vt,
        eigenanalysis_eps,rest,mass[element],hessian);
    ftt_capture_modes(table.deda,table.d2ed2a,u,sigma,vt,
        eigenanalysis_eps,rest,mass[element],1.0f,factor,weight,element);
    const float beta=damping[element]>0.0f && dt>0.0f?fmath::div(damping[element],dt):0.0f;
    if(beta>0.0f) {
        float response[9];for(unsigned k=0u;k<9u;++k)response[k]=0.0f;
        for(unsigned m=0u;m<9u;++m) {
            float dot=0.0f;
            for(unsigned a=1u;a<4u;++a)for(unsigned d=0u;d<3u;++d) {
                const float relative=(pose[a][d]-start[a][d])-(pose[0][d]-start[0][d]);
                dot+=factor[81u*element+9u*m+3u*(a-1u)+d]*relative;
            }
            const float scaled=weight[9u*element+m]*dot;
            for(unsigned k=0u;k<9u;++k)response[k]+=factor[81u*element+9u*m+k]*scaled;
        }
        for(unsigned a=1u;a<4u;++a)for(unsigned d=0u;d<3u;++d)
            gradient(d,a)+=beta*response[3u*(a-1u)+d];
        for(unsigned d=0u;d<3u;++d)
            gradient(d,0)=-(gradient(d,1)+gradient(d,2)+gradient(d,3));
    }
    hessian*=1.0f+beta;
    for(unsigned m=0u;m<9u;++m) {
        weight[9u*element+m]*=1.0f+beta;
        if(!ftt_finite(weight[9u*element+m]) || weight[9u*element+m]<0.0f){fault[element]=3u;return;}
    }
    for(unsigned k=0u;k<81u;++k)if(!ftt_finite(factor[81u*element+k])){fault[element]=3u;return;}
    hinge_atomic_embed_force(ids,gradient,force);
    for(unsigned a=0u;a<4u;++a)for(unsigned b=0u;b<4u;++b) {
        Mat3x3f block=hessian.block<3,3>(3u*a,3u*b);
        bool stored=fixed_csr_atomic_push(index,offset,value,row_count,ids[a],ids[b],block);
        if(!stored && ids[a]<=ids[b]){fault[element]=4u;return;}
    }
}
// Per element scratch: 9 force components + physical energy + arithmetic bound.
// lift=0: input is unknown dx, masked vertices have0 increment.
// lift=1: input is eval_x, target gives prescribed eval_x-target on masked only.
[[seam::entry(element)]]
[[seam::device_fn]] inline void ftt_evaluate(
    const unsigned *tet,const float *factor,const float *weight,
    const float *input,const float *target,const unsigned *mask,float *response,
    unsigned *fault,unsigned lift,unsigned element,unsigned extent) {
    if(element>=extent)return;
    float delta[9],out[9];
    unsigned v0=tet[4u*element];
    for(unsigned a=1u;a<4u;++a)for(unsigned d=0u;d<3u;++d) {
        unsigned v=tet[4u*element+a];
        float a0=0.0f,aa=0.0f;
        if(lift==0u) {
            a0=mask[v0]==0u?input[3u*v0+d]:0.0f;
            aa=mask[v]==0u?input[3u*v+d]:0.0f;
        } else {
            a0=mask[v0]!=0u?input[3u*v0+d]-target[3u*v0+d]:0.0f;
            aa=mask[v]!=0u?input[3u*v+d]-target[3u*v+d]:0.0f;
        }
        delta[3u*(a-1u)+d]=aa-a0;
    }
    for(unsigned k=0u;k<9u;++k)out[k]=0.0f;
    float energy=0.0f,bound=0.0f;
    for(unsigned m=0u;m<9u;++m) {
        float dot=0.0f,absolute=0.0f;
        for(unsigned k=0u;k<9u;++k) {
            float term=factor[81u*element+9u*m+k]*delta[k];
            dot+=term;absolute+=fmath::abs(term);
        }
        float w=weight[9u*element+m],scaled=w*dot;
        energy+=dot*scaled;bound+=(w*absolute)*absolute;
        for(unsigned k=0u;k<9u;++k)out[k]+=factor[81u*element+9u*m+k]*scaled;
    }
    for(unsigned k=0u;k<9u;++k) {
        if(!ftt_finite(out[k])){fault[element]=5u;return;}
        response[11u*element+k]=out[k];
    }
    if(!ftt_finite(energy)||!ftt_finite(bound)){fault[element]=5u;return;}
    response[11u*element+9u]=energy;response[11u*element+10u]=bound;
}
// Deterministic row-owned gather, no element atomic accumulation in PCG.
// Adjacency stores4*element+local_vertex. Lift subtracts E^T H E prescribed dx
// only on free force rows; normal apply adds to non-tet matvec and curvatures.
[[seam::entry(row)]]
[[seam::device_fn]] inline void ftt_gather(
    const unsigned *adj_offset,const unsigned *adj,const unsigned *mask,const unsigned *tet,
    const float *response,float *result,float *curvature,float *absolute,
    unsigned *fault,unsigned lift,unsigned row,unsigned extent) {
    if(row>=extent || mask[row]!=0u)return;
    float sum[3]={0.0f,0.0f,0.0f},energy=0.0f,bound=0.0f;
    for(unsigned k=adj_offset[row];k<adj_offset[row+1u];++k) {
        unsigned element=adj[k]/4u,slot=adj[k]%4u;
        for(unsigned d=0u;d<3u;++d) {
            float f=slot==0u?-(response[11u*element+d]+response[11u*element+3u+d]+response[11u*element+6u+d])
                :response[11u*element+3u*(slot-1u)+d];
            sum[d]+=f;
        }
        // First unmasked slot owns each tet energy exactly once. slot0 can be
        // prescribed; ownership follows the actual frozen mask, not topology.
        bool first=true;
        for(unsigned a=0u;a<slot;++a)if(mask[tet[4u*element+a]]==0u)first=false;
        if(first){energy+=response[11u*element+9u];bound+=response[11u*element+10u];}
    }
    for(unsigned d=0u;d<3u;++d) {
        float v=result[3u*row+d]+(lift==0u?sum[d]:-sum[d]);
        if(!ftt_finite(v)){fault[row]=6u;return;}
        result[3u*row+d]=v;
    }
    if(lift==0u){curvature[row]+=energy;absolute[row]+=bound;}
}
// Reconstruction only, never cancellation subtraction. Full approxCSR remains
// original preconditioner/contact reference; non-tetCSR is actualbaseoperator.
[[seam::entry(index)]]
[[seam::device_fn]] inline void ftt_combine(const float *non_tet,const float *tet,
    float *full,unsigned index,unsigned extent) {
    if(index<extent)full[index]=non_tet[index]+tet[index];
}
[[seam::entry(row)]]
[[seam::device_fn]] inline void ftt_mask_values(const unsigned *offset,const unsigned *column,
    const unsigned *mask,float *value,unsigned row,unsigned extent) {
    if(row>=extent)return;
    for(unsigned k=offset[row];k<offset[row+1u];++k)
        if(mask[row]!=0u||mask[column[k]]!=0u)for(unsigned i=0u;i<9u;++i)value[9u*k+i]=0.0f;
}
