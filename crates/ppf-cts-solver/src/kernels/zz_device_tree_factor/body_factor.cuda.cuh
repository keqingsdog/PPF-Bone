#pragma once
#include <cuda_runtime.h>
#include <math.h>
namespace body_factor_cuda {
__device__ inline double word(const unsigned *p, unsigned node, unsigned i, unsigned j) {
    unsigned k = node * 72u + 2u * (i * 6u + j);
    return __longlong_as_double((static_cast<unsigned long long>(p[k+1]) << 32) | p[k]);
}
__device__ inline double load(const unsigned *p, unsigned i) { return __longlong_as_double((static_cast<unsigned long long>(p[2*i+1]) << 32) | p[2*i]); }
__device__ inline void store(unsigned *p, unsigned i, double v) { unsigned long long bits = __double_as_longlong(v); p[2*i] = static_cast<unsigned>(bits); p[2*i+1] = static_cast<unsigned>(bits >> 32); }
__device__ inline void apply(
    const unsigned *body_index,
    const unsigned *component_offsets,
    const unsigned *parent,
    const unsigned *depth,
    const unsigned *max_depth,
    const unsigned *child_offsets,
    const unsigned *children,
    const unsigned *root_body_indices,
    const unsigned *l_words,
    const unsigned *w_words,
    const float *rhs,
    float *z,
    unsigned *fault,
    const float *stop,
    unsigned node_count,
    unsigned component_count,
    unsigned body_count,
    unsigned root_count,
    unsigned body_base,
    unsigned stop_enabled,
    unsigned body_index_len,
    unsigned component_offsets_len,
    unsigned parent_len,
    unsigned depth_len,
    unsigned max_depth_len,
    unsigned child_offsets_len,
    unsigned children_len,
    unsigned root_body_indices_len,
    unsigned l_words_len,
    unsigned w_words_len,
    unsigned rhs_len,
    unsigned z_len,
    unsigned fault_len,
    unsigned stop_len,
    unsigned *scratch,
    unsigned lane,
    unsigned width,
    unsigned component) {
    unsigned &error=scratch[0], &begin=scratch[1], &end=scratch[2], &levels=scratch[3], &stopped=scratch[4], &init_error=scratch[5];
    unsigned *q=scratch+8, *x=q+2048;
    if (lane == 0u) {
        error = 0u; stopped = 0u; begin = end = levels = 0u;
        if (!fault_len || !node_count || !component_count || component_count > node_count || node_count > body_count || root_count != body_count-node_count || children_len != node_count-component_count || component >= component_count ||
            component_offsets_len <= component_count || body_index_len < node_count || parent_len < node_count ||
            depth_len < node_count || max_depth_len < component_count || child_offsets_len <= node_count ||
            l_words_len / 72u < node_count || w_words_len / 72u < node_count || root_body_indices_len < root_count ||
            body_base > rhs_len || body_count > (rhs_len-body_base)/6u || body_base > z_len ||
            body_count > (z_len-body_base)/6u || (stop_enabled && stop_len < 1u)) error = 2u;
        if (!error && (atomicCAS(fault, 0u, 0u) != 0u || (stop_enabled && stop[0] != 0.0f))) stopped = 1u;
        if (!error && !stopped) {
            begin=component_offsets[component]; end=component_offsets[component+1u]; levels=max_depth[component];
            if (begin>=end || end>node_count || end-begin>width || end-begin>170u || levels>=node_count) error=2u;
        }
        if (!error && !stopped) {
            for (unsigned u=begin; u<end && !error; ++u) {
                unsigned pa=parent[u], dep=depth[u], lo=child_offsets[u], hi=child_offsets[u+1u];
                if (body_index[u]>=body_count || dep>levels || lo>hi || hi>children_len ||
                    (pa==0xffffffffu ? dep!=0u : (pa<begin || pa>=u || depth[pa]+1u!=dep))) {error=2u; break;}
                for (unsigned k=lo; k<hi; ++k) {
                    unsigned ch=children[k];
                    if (ch<=u || ch>=end || parent[ch]!=u || depth[ch]!=dep+1u) {error=2u; break;}
                }
            }
            if (component==0u) for (unsigned k=0; k<root_count; ++k)
                if (root_body_indices[k]>=body_count) {error=2u; break;}
        }
        // This snapshot is immutable after the barrier. Arithmetic faults may
        // subsequently update error while other warps are still taking the exit.
        init_error = error;
    }
    __syncthreads();
    if (init_error) { if (lane==0u && fault_len) atomicCAS(fault,0u,init_error); return; }
    if (stopped) return;
    unsigned u=begin+lane;
    if (u<end) for (unsigned i=0;i<6u;++i) {
        double v=static_cast<double>(rhs[body_base+6u*body_index[u]+i]);
        store(q,6u*lane+i,v); if (!isfinite(v)) atomicCAS(&error,0u,3u);
    }
    __syncthreads();
    for (unsigned d=levels; d>0u; --d) {
        if (u<end && depth[u]==d-1u) for (unsigned k=child_offsets[u];k<child_offsets[u+1u];++k) {
            unsigned ch=children[k], ci=ch-begin;
            for (unsigned j=0;j<6u;++j) { double t=0.0;
                for (unsigned i=0;i<6u;++i) t=__dadd_rn(t,__dmul_rn(word(w_words,ch,i,j),load(q,6u*ci+i)));
                double v=__dsub_rn(load(q,6u*lane+j),t); store(q,6u*lane+j,v);
                if (!isfinite(v)) atomicCAS(&error,0u,3u);
            }
        }
        __syncthreads();
    }
    for (unsigned d=0u;d<=levels;++d) {
        if (u<end && depth[u]==d) {
            double y[6]; for(unsigned i=0;i<6u;++i) y[i]=load(q,6u*lane+i);
            for(unsigned i=0;i<6u;++i) {for(unsigned j=0;j<i;++j) y[i]=__dsub_rn(y[i],__dmul_rn(word(l_words,u,i,j),y[j])); y[i]=__ddiv_rn(y[i],word(l_words,u,i,i));}
            for(int i=5;i>=0;--i) {for(unsigned j=i+1;j<6u;++j) y[i]=__dsub_rn(y[i],__dmul_rn(word(l_words,u,j,i),y[j])); y[i]=__ddiv_rn(y[i],word(l_words,u,i,i));}
            if(parent[u]!=0xffffffffu) for(unsigned i=0;i<6u;++i) for(unsigned j=0;j<6u;++j)
                y[i]=__dsub_rn(y[i],__dmul_rn(word(w_words,u,i,j),load(x,6u*(parent[u]-begin)+j)));
            for(unsigned i=0;i<6u;++i) { store(x,6u*lane+i,y[i]); if(!isfinite(y[i]) || !isfinite(__double2float_rn(y[i]))) atomicCAS(&error,0u,3u); }
        }
        __syncthreads();
    }
    if(error) {if(lane==0u) atomicCAS(fault,0u,error); return;}
    if(u<end) for(unsigned i=0;i<6u;++i) z[body_base+6u*body_index[u]+i]=__double2float_rn(load(x,6u*lane+i));
    if(component==0u) for(unsigned k=lane;k<root_count;k+=width) for(unsigned i=0;i<6u;++i) z[body_base+6u*root_body_indices[k]+i]=0.0f;
}
}
