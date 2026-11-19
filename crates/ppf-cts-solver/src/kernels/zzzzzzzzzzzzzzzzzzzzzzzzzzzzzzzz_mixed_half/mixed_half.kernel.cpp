// Packed FP16 storage for nodal inverse Cholesky factors. Arithmetic remains FP32.
// A zero scale marks a rejected block; its apply uses the original FP32 inverse.
// The four u32 words use 16 B per node, in addition to the retained FP32 buffer.
// No Tensor Core or mixed-precision SpMV is implied by this storage format.
#include "../float_math.hpp"

[[seam::device_fn]] inline bool mh_finite(float x) {
    return (fmath::float_bits(x) & 0x7f800000u) != 0x7f800000u;
}
[[seam::device_fn]] inline float mh_abs(float x) {
    return fmath::float_from_bits(fmath::float_bits(x) & 0x7fffffffu);
}
[[seam::device_fn]] inline unsigned mh_to_half(float x) {
    unsigned b = fmath::float_bits(x), sign = (b >> 16) & 0x8000u;
    unsigned exp = (b >> 23) & 255u, m = b & 0x7fffffu;
    if (exp == 255u) return sign | 0x7c00u | (m ? 0x200u : 0u);
    int e = int(exp) - 112;
    if (e >= 31) return sign | 0x7c00u;
    if (e <= 0) {
        if (e < -10) return sign;
        m |= 0x800000u;
        unsigned shift = unsigned(14 - e), v = m >> shift;
        unsigned rem = m & ((1u << shift) - 1u), mid = 1u << (shift - 1u);
        if (rem > mid || (rem == mid && (v & 1u))) ++v;
        return sign | v;
    }
    unsigned v = m >> 13, rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (v & 1u))) ++v;
    if (v == 0x400u) { v = 0; ++e; }
    if (e >= 31) return sign | 0x7c00u;
    return sign | (unsigned(e) << 10) | v;
}
[[seam::device_fn]] inline float mh_from_half(unsigned h) {
    unsigned sign = (h & 0x8000u) << 16, e = (h >> 10) & 31u, m = h & 1023u;
    if (e == 0u) {
        if (m == 0u) return fmath::float_from_bits(sign);
        int exponent = -14;
        while ((m & 1024u) == 0u) { m <<= 1; --exponent; }
        return fmath::float_from_bits(sign | (unsigned(exponent + 127) << 23) | ((m & 1023u) << 13));
    }
    if (e == 31u) return fmath::float_from_bits(sign | 0x7f800000u | (m << 13));
    return fmath::float_from_bits(sign | ((e + 112u) << 23) | (m << 13));
}

// Column-major inverse. Quantization is rejected if ||L^-1 (Lq-L)||_F > 0.05.
// This bound limits distortion of Lq*Lq^T even for ill-conditioned blocks.
[[seam::device_fn]] inline void
mixed_half_pack_block(const float *inverse, unsigned *packed, unsigned i) {
    unsigned o = 4u*i, p = 9u*i;
    packed[o] = packed[o+1] = packed[o+2] = packed[o+3] = 0u;
    for (unsigned k=0;k<9;k++) if (!mh_finite(inverse[p+k])) return;
    if (inverse[p+1] != inverse[p+3] || inverse[p+2] != inverse[p+6] || inverse[p+5] != inverse[p+7]) return;
    float a = inverse[p], b = inverse[p+1], c = inverse[p+2];
    float d = inverse[p+4], e = inverse[p+5], f = inverse[p+8];
    if (!(a > 0.0f)) return;
    float l0=fmath::sqrt(a), l1=fmath::div(b,l0), l3=fmath::div(c,l0);
    float t=d-l1*l1; if (!(t>0.0f) || !mh_finite(t)) return;
    float l2=fmath::sqrt(t), l4=fmath::div(e-l3*l1,l2);
    t=f-l3*l3-l4*l4; if (!(t>0.0f) || !mh_finite(t)) return;
    float l5=fmath::sqrt(t);
    if (!mh_finite(l0)||!mh_finite(l1)||!mh_finite(l2)||!mh_finite(l3)||!mh_finite(l4)||!mh_finite(l5)) return;
    float maxv=mh_abs(l0);
    maxv=fmath::max(maxv,mh_abs(l1)); maxv=fmath::max(maxv,mh_abs(l2));
    maxv=fmath::max(maxv,mh_abs(l3)); maxv=fmath::max(maxv,mh_abs(l4)); maxv=fmath::max(maxv,mh_abs(l5));
    unsigned sb=fmath::float_bits(maxv)&0x7f800000u; if (sb==0u) sb=0x00800000u;
    float scale=fmath::float_from_bits(sb);
    float n0=fmath::div(l0,scale), n1=fmath::div(l1,scale), n2=fmath::div(l2,scale);
    float n3=fmath::div(l3,scale), n4=fmath::div(l4,scale), n5=fmath::div(l5,scale);
    unsigned h0=mh_to_half(n0),h1=mh_to_half(n1),h2=mh_to_half(n2);
    unsigned h3=mh_to_half(n3),h4=mh_to_half(n4),h5=mh_to_half(n5);
    float q0=mh_from_half(h0),q1=mh_from_half(h1),q2=mh_from_half(h2);
    float q3=mh_from_half(h3),q4=mh_from_half(h4),q5=mh_from_half(h5);
    if (!(q0>0.0f && q2>0.0f && q5>0.0f)) return;
    if (!mh_finite(q0*scale)||!mh_finite(q1*scale)||!mh_finite(q2*scale)||
        !mh_finite(q3*scale)||!mh_finite(q4*scale)||!mh_finite(q5*scale)) return;
    float d0=fmath::div(q0-n0,n0);
    float d1=fmath::div((q1-n1)-n1*d0,n2);
    float d2=fmath::div(q2-n2,n2);
    float d3=fmath::div((q3-n3)-n3*d0-n4*d1,n5);
    float d4=fmath::div((q4-n4)-n4*d2,n5);
    float d5=fmath::div(q5-n5,n5);
    float err=d0*d0+d1*d1+d2*d2+d3*d3+d4*d4+d5*d5;
    if (!(err<=0.0025f)) return;
    packed[o]=h0|(h1<<16); packed[o+1]=h2|(h3<<16); packed[o+2]=h4|(h5<<16); packed[o+3]=sb;
}
[[seam::device_fn]] inline void
mixed_half_apply_block(const float *inverse, const unsigned *packed,
                       const float *vector, float *result, unsigned i) {
    unsigned o=4u*i, p=9u*i, v=3u*i;
    float x=vector[v],y=vector[v+1],z=vector[v+2];
    if (packed[o+3]==0u) {
        result[v]=inverse[p]*x+inverse[p+3]*y+inverse[p+6]*z;
        result[v+1]=inverse[p+1]*x+inverse[p+4]*y+inverse[p+7]*z;
        result[v+2]=inverse[p+2]*x+inverse[p+5]*y+inverse[p+8]*z;
        return;
    }
    float s=fmath::float_from_bits(packed[o+3]);
    float a=mh_from_half(packed[o]&65535u)*s, b=mh_from_half(packed[o]>>16)*s;
    float c=mh_from_half(packed[o+1]&65535u)*s, d=mh_from_half(packed[o+1]>>16)*s;
    float e=mh_from_half(packed[o+2]&65535u)*s, f=mh_from_half(packed[o+2]>>16)*s;
    float t0=a*x+b*y+d*z,t1=c*y+e*z,t2=f*z;
    result[v]=a*t0; result[v+1]=b*t0+c*t1; result[v+2]=d*t0+e*t1+f*t2;
}

// Declarations describe the dispatch and let kernelgen generate the ABI.
[[seam::entry(count,i)]] void mixed_half_pack_block(
    const float *inverse, unsigned *packed, unsigned count, unsigned i);

[[seam::device_fn]] inline void mixed_half_pdrd_apply(
    const unsigned *vertex_body, const unsigned *cloth_offset,
    const float *inverse, const unsigned *packed, const float *residual,
    float *out, unsigned v) {
    if (vertex_body[v] != 0xffffffffu) return;
    const unsigned offset=cloth_offset[v];
    mixed_half_apply_block(inverse+9u*v, packed+4u*v,
                           residual+offset, out+offset, 0u);
}
[[seam::entry(count,v)]] void mixed_half_pdrd_apply(
    const unsigned *vertex_body, const unsigned *cloth_offset,
    const float *inverse, const unsigned *packed, const float *residual,
    float *out, unsigned count, unsigned v);

// CUDA half2 products, FP32 sums. Unsupported or unhealthy blocks use v1.
// Optional status bits: 1=half arithmetic, 2=fallback. Zero residual: no bit.
[[seam::device_fn]] inline bool mh_half_nonzero_preserved(float x) {
    return x==0.0f || mh_abs(x)>0.0000000298023223876953125f;
}
[[seam::device_fn]] inline void mixed_half_pdrd_arithmetic(
    const unsigned *vertex_body, const unsigned *cloth_offset,
    const float *inverse, const unsigned *packed, const float *residual,
    float *out, unsigned *status, unsigned record_status, unsigned v) {
    if (vertex_body[v] != 0xffffffffu) return;
    const unsigned off=cloth_offset[v], po=4u*v;
    float x=residual[off],y=residual[off+1],z=residual[off+2];
    float peak=fmath::max(mh_abs(x),fmath::max(mh_abs(y),mh_abs(z)));
    if (mh_finite(x)&&mh_finite(y)&&mh_finite(z)&&peak==0.0f) { out[off]=out[off+1]=out[off+2]=0.0f; return; }
    bool ok=fmath::packed_half2_supported() && packed[po+3]!=0u &&
            mh_finite(x) && mh_finite(y) && mh_finite(z) &&
            (fmath::float_bits(peak)&0x7f800000u)!=0u;
    float a=0.0f,b=0.0f,c=0.0f;
    if (ok) {
        float rs=fmath::float_from_bits(fmath::float_bits(peak)&0x7f800000u);
        float nx=fmath::div(x,rs),ny=fmath::div(y,rs),nz=fmath::div(z,rs);
        ok=mh_half_nonzero_preserved(nx)&&mh_half_nonzero_preserved(ny)&&mh_half_nonzero_preserved(nz);
        if (ok) {
        unsigned q0=packed[po],q1=packed[po+1],q2=packed[po+2];
        unsigned p0=fmath::packed_half2_mul(q0,fmath::packed_half2_pack(nx,ny));
        unsigned p1=fmath::packed_half2_mul((q1>>16)|(q1<<16),fmath::packed_half2_pack(nz,ny));
        unsigned p2=fmath::packed_half2_mul(q2,fmath::packed_half2_pack(nz,nz));
        float t0=(fmath::packed_half_unpack(p0)+fmath::packed_half_unpack(p0>>16))+fmath::packed_half_unpack(p1);
        float t1=fmath::packed_half_unpack(p1>>16)+fmath::packed_half_unpack(p2);
        float t2=fmath::packed_half_unpack(p2>>16);
        peak=fmath::max(mh_abs(t0),fmath::max(mh_abs(t1),mh_abs(t2)));
        ok=mh_finite(t0)&&mh_finite(t1)&&mh_finite(t2)&&peak>0.0f&&
           (fmath::float_bits(peak)&0x7f800000u)!=0u;
        if (ok) {
            float ts=fmath::float_from_bits(fmath::float_bits(peak)&0x7f800000u);
            float u=fmath::div(t0,ts),w=fmath::div(t1,ts),k=fmath::div(t2,ts);
            ok=mh_half_nonzero_preserved(u)&&mh_half_nonzero_preserved(w)&&mh_half_nonzero_preserved(k);
            if (ok) {
            p0=fmath::packed_half2_mul(q0,fmath::packed_half2_pack(u,u));
            p1=fmath::packed_half2_mul(q1,fmath::packed_half2_pack(w,u));
            p2=fmath::packed_half2_mul(q2,fmath::packed_half2_pack(w,k));
            float ca=fmath::packed_half_unpack(p0);
            float cb=fmath::packed_half_unpack(p0>>16)+fmath::packed_half_unpack(p1);
            float cc=(fmath::packed_half_unpack(p1>>16)+fmath::packed_half_unpack(p2))+fmath::packed_half_unpack(p2>>16);
            // All three scales are powers of two. Compose their exponents
            // before rounding, so cancelling scales cannot overflow midway.
            int re=int((fmath::float_bits(rs)>>23)&255u)-127;
            int te=int((fmath::float_bits(ts)>>23)&255u)-127;
            int se=int((packed[po+3]>>23)&255u)-127;
            int total=re+te+2*se;
            a=fmath::scale_pow2(ca,total);
            b=fmath::scale_pow2(cb,total);
            c=fmath::scale_pow2(cc,total);
            float out_peak=fmath::max(mh_abs(a),fmath::max(mh_abs(b),mh_abs(c)));
            ok=mh_finite(a)&&mh_finite(b)&&mh_finite(c)&&
               (ca==0.0f||a!=0.0f)&&(cb==0.0f||b!=0.0f)&&(cc==0.0f||c!=0.0f)&&
               out_peak>0.0f && (fmath::float_bits(out_peak)&0x7f800000u)!=0u;
            if (ok) {
                float dot=nx*fmath::div(a,out_peak)+ny*fmath::div(b,out_peak)+nz*fmath::div(c,out_peak);
                ok=mh_finite(dot)&&dot>0.0f;
            }
            }
        }
        }
    }
    if (ok) { out[off]=a; out[off+1]=b; out[off+2]=c; }
    else {
        unsigned p=9u*v;
        out[off]=inverse[p]*x+inverse[p+3]*y+inverse[p+6]*z;
        out[off+1]=inverse[p+1]*x+inverse[p+4]*y+inverse[p+7]*z;
        out[off+2]=inverse[p+2]*x+inverse[p+5]*y+inverse[p+8]*z;
    }
    if (record_status!=0u) status[v] |= ok ? 1u : 2u;
}
[[seam::entry(count,v)]] void mixed_half_pdrd_arithmetic(
    const unsigned *vertex_body, const unsigned *cloth_offset,
    const float *inverse, const unsigned *packed, const float *residual,
    float *out, unsigned *status, unsigned record_status, unsigned count, unsigned v);