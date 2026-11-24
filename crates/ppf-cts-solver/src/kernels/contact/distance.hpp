// Fork modification: changed from the upstream baseline.
// File: distance.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef DISTANCE_HPP
#define DISTANCE_HPP

#ifndef SM_MSL_CONCAT
#include "../common.hpp"
#include "../data.hpp"
#endif

#ifndef SM_THREAD
#define SM_THREAD
#define DISTANCE_UNDEF_THREAD
#endif
#ifndef SM_NUMERIC_MAX
#define SM_NUMERIC_MAX(T) std::numeric_limits<T>::max()
#define DISTANCE_UNDEF_NUMERIC_MAX
#endif

namespace proximity {

template <class T, class Y>
__device__ Vec3<Y> difference(SM_THREAD const Vec3<T> &a,
                              SM_THREAD const Vec3<T> &b) {
    Vec3<Y> result;
    for (unsigned dimension = 0; dimension < 3; ++dimension) {
        result[dimension] = static_cast<Y>(a[dimension] - b[dimension]);
    }
    return result;
}

template <class T, class Y>
__device__ Vec2<Y> point_edge_distance_coeff(
    SM_THREAD const Vec3<T> &p, SM_THREAD const Vec3<T> &e0,
    SM_THREAD const Vec3<T> &e1) {
    Vec3<Y> r = difference<T, Y>(e1, e0);
    Y d = r.dot(r);
    if (d > Y(0.0)) {
        Y t = r.dot(difference<T, Y>(p, e0)) / d;
        return Vec2<Y>(Y(1.0) - t, t);
    } else {
        return Vec2<Y>::Ones() / Y(2.0);
    }
}

// Barycentric coordinates of p projected perpendicularly onto the plane of the
// triangle. The result is unclamped, so a coordinate outside [0, 1] means the
// projection falls outside the triangle and names a point of its plane rather
// than of the triangle itself. Callers that need a point ON the triangle read
// the sign of the coordinates to decide, or use the closest-point routine
// below, which selects among the boundary as well.
//
// The coordinates are taken through the cross product rather than through the
// Gram system (r0.r0)(r1.r1) - (r0.r1)^2. The values are the same and the
// conditioning is not: that determinant subtracts two products which agree to
// within the SQUARED sine of the angle between the two edges, and both
// numerators cancel the same way, so on a triangle with one wide interior angle
// the quotient keeps only a few of single precision's digits. Each component of
// the cross product subtracts products agreeing only to within the FIRST power
// of that sine, a whole factor of the sine better. That matters because every
// caller tests the coordinates against 0 and 1 to decide whether the projection
// lies inside, and on a sliver the Gram form can put that verdict on the wrong
// side of the comparison.
template <class T, class Y>
__device__ Vec3<Y>
point_triangle_distance_coeff(SM_THREAD const Vec3<T> &p,
                              SM_THREAD const Vec3<T> &t0,
                              SM_THREAD const Vec3<T> &t1,
                              SM_THREAD const Vec3<T> &t2) {
    Vec3<Y> r0 = difference<T, Y>(t1, t0);
    Vec3<Y> r1 = difference<T, Y>(t2, t0);
    Vec3<Y> d = difference<T, Y>(p, t0);
    Vec3<Y> nrm = r0.cross(r1);
    Y nn = nrm.dot(nrm);
    if (nn > Y(0)) {
        Y inv_nn = Y(1) / nn;
        Y u = d.cross(r1).dot(nrm) * inv_nn;
        Y v = r0.cross(d).dot(nrm) * inv_nn;
        return Vec3<Y>(Y(1.0) - u - v, u, v);
    } else {
        // Zero area: the three points are collinear (or coincident), so there
        // is no plane to project onto. The longest edge spans the whole
        // degenerate triangle, which makes projecting onto it the closest
        // available answer rather than an arbitrary one.
        Vec3<Y> e1 = difference<T, Y>(t2, t1);
        Vec3<Y> e2 = difference<T, Y>(t0, t2);
        Y len0 = r0.dot(r0);
        Y len1 = e1.dot(e1);
        Y len2 = e2.dot(e2);
        if (len0 >= len1 && len0 >= len2) {
            Vec2<Y> w = point_edge_distance_coeff<T, Y>(p, t0, t1);
            return Vec3<Y>(w(0), w(1), Y(0.0));
        } else if (len1 >= len2) {
            Vec2<Y> w = point_edge_distance_coeff<T, Y>(p, t1, t2);
            return Vec3<Y>(Y(0.0), w(0), w(1));
        } else {
            Vec2<Y> w = point_edge_distance_coeff<T, Y>(p, t2, t0);
            return Vec3<Y>(w(1), Y(0.0), w(0));
        }
    }
}

// Error-free transforms are evaluated as separate operations; only TwoProduct
// uses explicit fused multiply-add. The supported range is checked below.
__device__ inline void vf_two_sum(float a, float b, SM_THREAD float &hi,
                                 SM_THREAD float &lo) {
    hi = SM_FMA(1.0f, a, b);
    const float bv = SM_FMA(-1.0f, a, hi);
    const float av = SM_FMA(-1.0f, bv, hi);
    const float br = SM_FMA(-1.0f, bv, b);
    const float ar = SM_FMA(-1.0f, av, a);
    lo = SM_FMA(1.0f, ar, br);
}
__device__ inline void vf_two_diff(float a, float b, SM_THREAD float &hi,
                                  SM_THREAD float &lo) {
    hi = SM_FMA(-1.0f, b, a);
    const float bv = SM_FMA(-1.0f, hi, a);
    const float av = SM_FMA(1.0f, hi, bv);
    const float br = SM_FMA(-1.0f, b, bv);
    const float ar = SM_FMA(-1.0f, av, a);
    lo = SM_FMA(1.0f, ar, br);
}
__device__ inline void vf_two_product(float a, float b, SM_THREAD float &hi,
                                     SM_THREAD float &lo) {
    hi = SM_FMA(a, b, 0.0f);
    lo = SM_FMA(a, b, -hi);
}
// A normal binary32 value is significand * 2^(biased_exponent - 150).
// Extract the least nonzero exponent without rounding any floating operand.
__device__ inline bool vf_normal_low_bit(unsigned bits, SM_THREAD int &low) {
    const unsigned exponent = (bits >> 23) & 255u;
    if (exponent == 0u || exponent == 255u) {
        return false;
    }
    const unsigned significand = (bits & 0x007fffffu) | 0x00800000u;
#if defined(__CUDA_ARCH__)
    low = static_cast<int>(exponent) - 151 + __ffs(static_cast<int>(significand));
#else
    // Portable spelling for the same integer test and host fixtures.
    unsigned tail = significand;
    int trailing = 0;
    if ((tail & 0xffffu) == 0u) { tail >>= 16; trailing += 16; }
    if ((tail & 0xffu) == 0u) { tail >>= 8; trailing += 8; }
    if ((tail & 0xfu) == 0u) { tail >>= 4; trailing += 4; }
    if ((tail & 0x3u) == 0u) { tail >>= 2; trailing += 2; }
    if ((tail & 0x1u) == 0u) { ++trailing; }
    low = static_cast<int>(exponent) - 150 + trailing;
#endif
    return true;
}

// Least nonzero binary exponent of a NORMAL finite float. This integer
// significance test bounds every exact product limb away from FTZ/underflow.
__device__ inline int vf_low_bit_legacy(float a) {
    a = SM_ABS(a);
    int exponent = 0;
    while (a < 1.0f) { a *= 2.0f; --exponent; }
    while (a >= 2.0f) { a *= 0.5f; ++exponent; }
    unsigned mantissa = unsigned(a * 8388608.0f);
    while ((mantissa & 1u) == 0u) { mantissa >>= 1; ++exponent; }
    return exponent - 23;
}

template <bool CudaSweep = false>
__device__ inline int vf_low_bit(float a) {
#if defined(__CUDA_ARCH__)
    int low;
    if (CudaSweep && vf_normal_low_bit(__float_as_uint(a), low)) {
        return low;
    }
#endif
    return vf_low_bit_legacy(a);
}
__device__ inline bool vf_grow(float value, SM_THREAD float *e,
                              SM_THREAD unsigned &count) {
    float q = value;
    unsigned written = 0;
    const unsigned old_count = count;
    for (unsigned i = 0; i < old_count; ++i) {
        float hi, lo;
        vf_two_sum(q, e[i], hi, lo);
        if (lo != 0.0f) {
            if (written >= 192u) { return false; }
            e[written++] = lo;
        }
        q = hi;
    }
    if (q != 0.0f) {
        if (written >= 192u) { return false; }
        e[written++] = q;
    }
    count = written;
    return true;
}
// No double, epsilon, or invented positive gap. A common exact power-of-two
// scale protects products. Near cancellation uses at most 6*8*4=192 limbs:
// six determinant terms, two-limb differences, exact three-factor products.
// An unsupported range is reported false to the existing invalid-distance gate.
template <bool CudaSweep = false>
__device__ inline bool vf_signed_height_scaled(
    SM_THREAD const Vec3f &p, SM_THREAD const Vec3f &a,
    SM_THREAD const Vec3f &b, SM_THREAD const Vec3f &c,
    float extra_scale, SM_THREAD float &height) {
    const float normal_min = 1.1754943508222875e-38f;
    float largest = 0.0f;
    Vec3f points[4] = {p, a, b, c};
    for (unsigned i = 0; i < 4; ++i) {
        for (unsigned j = 0; j < 3; ++j) {
            const float v = SM_ABS(points[i][j]);
            if (!(v <= SM_NUMERIC_MAX(float))) { return false; }
            largest = SM_MAX(largest, v);
        }
    }
    if (!(largest >= normal_min)) { return false; }
    float scale = 1.0f;
    while (largest > 1.0f) { largest *= 0.5f; scale *= 0.5f; }
    while (largest < 0.5f) { largest *= 2.0f; scale *= 2.0f; }
    // Both extra scales are exact powers of two. Check before multiplying so
    // neither an overflow nor a flushed subnormal is admitted as a scale.
    if (!(scale >= normal_min &&
          scale <= SM_NUMERIC_MAX(float) / extra_scale)) {
        return false;
    }
    scale *= extra_scale;
    if (!(scale >= normal_min && scale <= SM_NUMERIC_MAX(float))) {
        return false;
    }
    for (unsigned i = 0; i < 4; ++i) {
        for (unsigned j = 0; j < 3; ++j) {
            const float original = points[i][j];
            points[i][j] *= scale;
            // A binary32 coordinate at least 2^-102 has least bit >=2^-125,
            // so TwoDiff cannot silently lose a subnormal nonzero tail on FTZ.
            if ((original != 0.0f && SM_ABS(points[i][j]) < 1.9721522630525295e-31f) ||
                points[i][j] / scale != original) { return false; }
        }
    }
    float d[3][3][2];
    for (unsigned j = 0; j < 3; ++j) {
        vf_two_diff(points[2][j], points[1][j], d[0][j][0], d[0][j][1]);
        vf_two_diff(points[3][j], points[1][j], d[1][j][0], d[1][j][1]);
        vf_two_diff(points[0][j], points[1][j], d[2][j][0], d[2][j][1]);
    }
    const unsigned axis[6][3] = {{0,1,2},{1,2,0},{2,0,1},
                                {0,2,1},{1,0,2},{2,1,0}};
    // Explicit multiply/add calls fix the rounding/contraction model.
    // The cheap case below has neither difference tails nor determinant cancellation.
    float fast = 0.0f, permanent = 0.0f;
    bool positive = false, negative = false, has_tail = false;
    for (unsigned i = 0; i < 3; ++i) {
        for (unsigned j = 0; j < 3; ++j) {
            has_tail = has_tail || d[i][j][1] != 0.0f;
        }
    }
    for (unsigned i = 0; i < 6; ++i) {
        const float pair = SM_FMA(d[0][axis[i][0]][0], d[1][axis[i][1]][0], 0.0f);
        const float term = SM_FMA(pair, d[2][axis[i][2]][0], 0.0f);
        const float signed_term = i < 3 ? term : -term;
        positive = positive || signed_term > 0.0f;
        negative = negative || signed_term < 0.0f;
        fast = SM_FMA(1.0f, signed_term, fast);
        permanent = SM_FMA(1.0f, SM_ABS(term), permanent);
    }
    const Vec3f u(d[0][0][0], d[0][1][0], d[0][2][0]);
    const Vec3f v(d[1][0][0], d[1][1][0], d[1][2][0]);
    const float divisor = u.cross(v).norm();
    if (!(divisor > 0.0f && divisor <= SM_NUMERIC_MAX(float))) {
        return false;
    }
    // The floor 2^-100 dominates possible min-normal underflow losses in the
    // cheap noncancelling accumulation. It is an arithmetic-domain bound,
    // never a geometric offset or a replacement for a nonzero determinant.
    // A sign-only filter cannot qualify the distance magnitude near a plane.
    // Keep the cheap branch only without subtraction tails or opposed terms;
    // any determinant cancellation uses the exact expansion below.
    if (!has_tail && !(positive && negative) &&
        permanent >= 7.888609052210118e-31f) {
        height = (fast / divisor) / scale;
        return SM_ABS(height) >= normal_min && SM_ABS(height) <= SM_NUMERIC_MAX(float);
    }
    int low[3][3][2];
    for (unsigned i = 0; i < 3; ++i) {
        for (unsigned j = 0; j < 3; ++j) {
            for (unsigned k = 0; k < 2; ++k) {
                const float value = d[i][j][k];
                if (value != 0.0f && SM_ABS(value) < normal_min) { return false; }
                low[i][j][k] = value == 0.0f ? 0 : vf_low_bit<CudaSweep>(value);
            }
        }
    }
    float expansion[192];
    unsigned count = 0;
    for (unsigned i = 0; i < 6; ++i) {
        for (unsigned j = 0; j < 2; ++j) {
            for (unsigned k = 0; k < 2; ++k) {
                for (unsigned l = 0; l < 2; ++l) {
                    const float x = d[0][axis[i][0]][j];
                    const float y = d[1][axis[i][1]][k];
                    const float z = d[2][axis[i][2]][l];
                    if (x == 0.0f || y == 0.0f || z == 0.0f) { continue; }
                    const int xy = low[0][axis[i][0]][j] + low[1][axis[i][1]][k];
                    if (xy < -126 || xy + low[2][axis[i][2]][l] < -126) {
                        return false;
                    }
                    float product[2];
                    vf_two_product(x, y, product[0], product[1]);
                    for (unsigned m = 0; m < 2; ++m) {
                        float hi, lo;
                        vf_two_product(product[m], z, hi, lo);
                        const float sign = i < 3 ? 1.0f : -1.0f;
                        if (!vf_grow(sign * lo, expansion, count) ||
                            !vf_grow(sign * hi, expansion, count)) { return false; }
                    }
                }
            }
        }
    }
    float volume = 0.0f;
    for (unsigned i = 0; i < count; ++i) { volume = SM_FMA(1.0f, volume, expansion[i]); }
    if (count == 0u) { height = 0.0f; return true; }
    height = (volume / divisor) / scale;
    // Do not turn a nonzero exact determinant into a rounded/FTZ zero gap.
    return SM_ABS(height) >= normal_min && SM_ABS(height) <= SM_NUMERIC_MAX(float) &&
        ((height > 0.0f) == (expansion[count - 1u] > 0.0f));
}

template <bool CudaSweep = false>
__device__ inline bool vf_signed_height(
    SM_THREAD const Vec3f &p, SM_THREAD const Vec3f &a,
    SM_THREAD const Vec3f &b, SM_THREAD const Vec3f &c,
    SM_THREAD float &height) {
    if (vf_signed_height_scaled<CudaSweep>(p, a, b, c, 1.0f, height)) {
        return true;
    }
    // The original scale bounds coordinates by one, leaving enough exponent
    // headroom to shift an exact power of two upward when a product limb would
    // otherwise underflow. With coordinates <= 2^16, differences are <= 2^17,
    // determinant terms <= 2^51 and the squared cross-product norm <= 2^72.
    // The same exact expansion and all domain checks are applied again.
    return vf_signed_height_scaled<CudaSweep>(p, a, b, c, 65536.0f, height);
}

// A strict interior VF has a perpendicular separation, never an in-plane
// barycentric reconstruction residual. Keep the point-minus-plane orientation.
// Coefficients remain the caller's closest-point weights; this helper changes
// only the distance/normal representation. Boundary and degenerate features
// return false so their existing edge/vertex arithmetic remains in charge.
template <class T, class Y, bool CudaSweep = false>
__device__ bool point_triangle_interior_separation(
    SM_THREAD const Vec3<T> &p, SM_THREAD const Vec3<T> &t0,
    SM_THREAD const Vec3<T> &t1, SM_THREAD const Vec3<T> &t2,
    SM_THREAD const Vec3<Y> &coeff, SM_THREAD Vec3<Y> &separation) {
    if (!(coeff.minCoeff() > Y(0) && coeff.maxCoeff() < Y(1))) {
        return false;
    }
    const Vec3<Y> r0 = difference<T, Y>(t1, t0);
    const Vec3<Y> r1 = difference<T, Y>(t2, t0);
    const Vec3<Y> n = r0.cross(r1);
    const Y nn = n.dot(n);
    if (!(nn >= Y(0) && nn <= SM_NUMERIC_MAX(Y))) {
        // An invalid interior plane is not a boundary fallback. Propagate
        // the invalid norm to the existing strict distance/overlap gate.
        const Y invalid = nn - nn;
        separation = Vec3<Y>(invalid, invalid, invalid);
        return true;
    }
    if (!(nn > Y(0))) {
        return false;
    }
    const Vec3<Y> normal = n.normalized();
    Y height = difference<T, Y>(p, t0).dot(normal);
    if constexpr (sizeof(T) == sizeof(float) && sizeof(Y) == sizeof(float)) {
        float stable_height;
        if (!vf_signed_height<CudaSweep>(p, t0, t1, t2, stable_height)) {
            const Y invalid = Y(SM_INFINITY) - Y(SM_INFINITY);
            separation = Vec3<Y>(invalid, invalid, invalid);
            return true;
        }
        height = Y(stable_height);
    }
    separation = height * normal;
    return true;
}

// Clamp to the closed unit interval. The clamped values are the literals 0
// and 1, which is what lets a caller read a coefficient of exactly 0 or 1 as
// "the closest point is an endpoint". A value that is not a number leaves as
// zero, so a returned coefficient can never name a point off its segment.
template <class Y> __device__ Y clamp_unit(Y v) {
    return v > Y(0) ? (v < Y(1) ? v : Y(1)) : Y(0);
}

// Closest-point barycentric coefficients of the segments A(s) = ea0 + s r0 and
// B(t) = eb0 + t r1, returned as (1 - s, s, 1 - t, t) with s and t in [0, 1].
//
// F(s, t) = |d + s r0 - t r1|^2 is a convex quadratic on the unit square, so
// its minimum is attained either at an interior stationary point or on the
// boundary. The boundary is four segments, and on each of them F restricts to
// a one-dimensional convex quadratic whose exact minimizer is a clamped
// projection. The five candidates below therefore cover every minimizer, and
// the routine returns whichever of them realizes the smallest distance.
//
// Selecting by the distance actually realized is what leaves the routine with
// no tolerance to tune: every candidate names a point of the unit square, so
// the distance it realizes is an upper bound on the minimum, and a stationary
// point recovered from an ill-conditioned quotient can only fail to win. The
// routine is therefore never asked whether two edges are parallel enough that
// the interior solve should be distrusted, which is the question no threshold
// can answer: an interior optimum is genuinely the answer at angles far below
// any usable parallel epsilon, while for exactly parallel segments the
// minimum is always attained with at least one of the two points at an
// endpoint of its own segment, so the four boundary candidates are exact
// there.
template <class T, class Y>
__device__ Vec4<Y>
edge_edge_distance_coeff(SM_THREAD const Vec3<T> &ea0,
                         SM_THREAD const Vec3<T> &ea1,
                         SM_THREAD const Vec3<T> &eb0,
                         SM_THREAD const Vec3<T> &eb1) {
    // Every quantity below is built from pairwise endpoint differences taken
    // in T and cast to Y afterwards, never from a cast of an endpoint itself:
    // a closest-point pair is separated by far less than its distance from the
    // origin, so casting first truncates at that distance and the subtraction
    // then cancels down to the truncation, which can flip the selection below.
    Vec3<Y> r0 = difference<T, Y>(ea1, ea0);
    Vec3<Y> r1 = difference<T, Y>(eb1, eb0);
    Vec3<Y> d = difference<T, Y>(ea0, eb0);
    Y a = r0.dot(r0);
    Y e = r1.dot(r1);
    Y b = r0.dot(r1);
    Y c = r0.dot(d);
    Y f = r1.dot(d);

    // A squared length is exactly zero precisely when its segment has
    // collapsed to a point, and every parameter then names that same point, so
    // the guarded zero is the exact answer here and not a fallback.
    Y inv_a = a > Y(0) ? Y(1) / a : Y(0);
    Y inv_e = e > Y(0) ? Y(1) / e : Y(0);

    // The stationary point, s = (b f - c e) / (a e - b^2), taken through the
    // cross product rather than through those two Gram expressions. The
    // denominator a e - b^2 is |r0 x r1|^2 and the numerator b f - c e is
    // (r0 x r1) . (r1 x d), so the values are the same and the conditioning is
    // not: a e - b^2 subtracts two products that agree to within the SQUARED
    // sine of the angle between the edges, which in single precision leaves no
    // significant digits at all below about 3e-4 rad, whereas each component
    // of the cross product subtracts two products that agree only to within
    // the first power of that sine. A settled contact between two bodies
    // produces near-parallel edge pairs constantly, so this is the operating
    // regime and not a corner.
    //
    // What conditioning remains is bounded independently of the angle. Since t
    // follows s by exact projection, an error ds in s displaces the connecting
    // vector only by ds times the part of r0 perpendicular to r1, of magnitude
    // ds |r0| sin(angle); and the relative error of the quotient is itself of
    // order (unit round-off) / sin(angle), so the product is of order
    // (unit round-off) |r0| at every angle. That is already the resolution at
    // which the difference vector below can be evaluated in Y, so a
    // compensated cross product, which would remove the surviving 1/sin factor
    // from the quotient, does not lower that bound.
    Vec3<Y> n = r0.cross(r1);
    Y nn = n.dot(n);
    Y inv_nn = nn > Y(0) ? Y(1) / nn : Y(0);
    Y s_stat = clamp_unit<Y>(n.dot(r1.cross(d)) * inv_nn);
    Y t_stat = clamp_unit<Y>((b * s_stat + f) * inv_e);

    // The four boundary segments of the unit square: s = 0 and s = 1 project
    // A's endpoints onto B, and t = 0 and t = 1 project B's endpoints onto A.
    Y t_a0 = clamp_unit<Y>(f * inv_e);
    Y t_a1 = clamp_unit<Y>((f + b) * inv_e);
    Y s_b0 = clamp_unit<Y>(-c * inv_a);
    Y s_b1 = clamp_unit<Y>((b - c) * inv_a);

    const Y cand_s[5] = {s_stat, Y(0), Y(1), s_b0, s_b1};
    const Y cand_t[5] = {t_stat, t_a0, t_a1, Y(0), Y(1)};
    Y best_s = Y(0);
    Y best_t = Y(0);
    Y best_dist = SM_NUMERIC_MAX(Y);
    // The scan runs in a fixed order and keeps the first of equal distances,
    // so the choice is reproducible where the minimum is attained on a whole
    // interval. Unrolling is load-bearing: the loop indexes the two candidate
    // arrays, and a dynamic index would place them in local memory.
#pragma unroll
    for (unsigned i = 0; i < 5; ++i) {
        // This is an exact identity of the barycentric combination being
        // scored: ((1-s) ea0 + s ea1) - ((1-t) eb0 + t eb1) = d + s r0 - t r1.
        Vec3<Y> v = d + cand_s[i] * r0 - cand_t[i] * r1;
        Y dist = v.dot(v);
        if (dist < best_dist) {
            best_dist = dist;
            best_s = cand_s[i];
            best_t = cand_t[i];
        }
    }
    return Vec4<Y>(Y(1) - best_s, best_s, Y(1) - best_t, best_t);
}

// Barycentric coefficients of the point of the triangle closest to p, returned
// with all three in [0, 1] and summing to one.
//
// The squared distance from p to a point of the triangle is a convex quadratic
// on the barycentric simplex, so its minimum is attained either at the interior
// stationary point (the perpendicular projection, admissible only when it lands
// inside) or on the boundary. The boundary is three segments, and on each of
// them the exact minimizer is a clamped projection. The four candidates below
// therefore cover every minimizer, and the routine returns whichever of them
// realizes the smallest distance.
//
// Selecting by the distance actually realized is what makes the result exact
// rather than merely plausible. Every candidate names a point of the triangle,
// so the distance it realizes is an upper bound on the minimum and a candidate
// that is not the closest can only fail to win. Classifying instead by the
// SIGNS of the barycentric coordinates and then trusting the chosen edge is not
// equivalent: a sign test splits the plane by the triangle's edge lines, which
// separates the three edge regions only for an acute triangle. On an obtuse one
// the lines cross outside, and a point in the region beyond the wide vertex
// falls on the far side of an edge line whose edge is not its nearest feature,
// so the classification commits to that edge and reports a point further away
// than the true closest one.
//
// Over-reporting is the unsafe direction, because ccd_helper sizes each
// conservative advance as (distance - parking clearance) / (Lipschitz bound):
// a distance reported too large buys a step too long, and the sweep can cross a
// contact it never saw. The error is proportional to the true distance, with a
// worst case governed by the widest interior angle, so a sliver triangle turns
// a small mistake into a step that skips the collision entirely.
template <class T, class Y, bool RetainInterior>
__device__ Vec3<Y> point_triangle_distance_choice(
    SM_THREAD const Vec3<T> &p, SM_THREAD const Vec3<T> &t0,
    SM_THREAD const Vec3<T> &t1, SM_THREAD const Vec3<T> &t2,
    SM_THREAD Vec3<Y> &selected_interior, SM_THREAD bool &interior_won) {
    if constexpr (RetainInterior) {
        interior_won = false;
    }

    // Differences are taken in T and cast to Y afterwards, never a cast of an
    // endpoint itself: a closest-point pair is separated by far less than its
    // distance from the origin, so casting first truncates at that distance and
    // the subtraction then cancels down to the truncation.
    Vec3<Y> r0 = difference<T, Y>(t1, t0);
    Vec3<Y> r1 = difference<T, Y>(t2, t0);
    Vec3<Y> d = difference<T, Y>(p, t0);

    Vec3<Y> cand[4];
    // The interior candidate is the only one that can be inadmissible, so it
    // carries a flag rather than shortening the array: the scan below then runs
    // over a compile-time count with compile-time indices, which is what keeps
    // the candidates in registers. A running count would make the index dynamic
    // and place the array in local memory, in a routine every contact pair of
    // every line search evaluates.
    bool admissible[4] = {false, true, true, true};

    // The perpendicular projection, admissible only where it names a point of
    // the triangle. Elsewhere it is not a candidate at all, so a projection
    // that falls outside, or a degenerate triangle with no plane to project
    // onto, simply contributes nothing and cannot mislead the scan below. The
    // routine above owns this solve, including the cross-product conditioning
    // a sliver needs, so there is one implementation of it rather than two.
    cand[0] = point_triangle_distance_coeff<T, Y>(p, t0, t1, t2);
    admissible[0] = cand[0].minCoeff() >= Y(0);

    // The three boundary segments. clamp_unit keeps each projection on its own
    // edge, which also covers the vertices, so no separate vertex candidate is
    // needed.
    Y s01 = clamp_unit<Y>(point_edge_distance_coeff<T, Y>(p, t0, t1)(1));
    Y s12 = clamp_unit<Y>(point_edge_distance_coeff<T, Y>(p, t1, t2)(1));
    Y s20 = clamp_unit<Y>(point_edge_distance_coeff<T, Y>(p, t2, t0)(1));
    cand[1] = Vec3<Y>(Y(1) - s01, s01, Y(0));
    cand[2] = Vec3<Y>(Y(0), Y(1) - s12, s12);
    cand[3] = Vec3<Y>(s20, Y(0), Y(1) - s20);

    Vec3<Y> best = cand[1];
    Y best_dist = SM_NUMERIC_MAX(Y);
    // The scan runs in a fixed order and keeps the first of equal distances, so
    // the choice is reproducible where the minimum is attained on a whole edge.
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        if (!admissible[i]) {
            continue;
        }
        // Exact identity of the combination being scored:
        // p - (w0 t0 + w1 t1 + w2 t2) = d - w1 r0 - w2 r1, since the weights
        // sum to one.
        Vec3<Y> v = d - cand[i][1] * r0 - cand[i][2] * r1;
        bool interior = false;
        if (i == 0) {
            interior = point_triangle_interior_separation<T, Y, RetainInterior>(
                p, t0, t1, t2, cand[i], v);
        }
        Y dist = v.dot(v);
        if (dist < best_dist) {
            best_dist = dist;
            best = cand[i];
            if constexpr (RetainInterior) {
                interior_won = interior;
                if (interior) {
                    selected_interior = v;
                }
            }
        }
    }
    return best;
}

template <class T, class Y>
__device__ Vec3<Y> point_triangle_distance_coeff_unclassified(
    SM_THREAD const Vec3<T> &p, SM_THREAD const Vec3<T> &t0,
    SM_THREAD const Vec3<T> &t1, SM_THREAD const Vec3<T> &t2) {
    Vec3<Y> unused;
    bool unused_winner;
    return point_triangle_distance_choice<T, Y, false>(p, t0, t1, t2,
                                                       unused, unused_winner);
}

template <class T, class Y>
__device__ Vec2<Y> point_edge_distance_coeff_unclassified(
    SM_THREAD const Vec3<T> &p, SM_THREAD const Vec3<T> &e0,
    SM_THREAD const Vec3<T> &e1) {
    Vec2<Y> c = point_edge_distance_coeff<T, Y>(p, e0, e1);
    if (c(0) >= Y(0.0) && c(0) <= Y(1.0)) {
        return c;
    } else {
        if (c(0) > Y(1.0)) {
            return Vec2<Y>(Y(1.0), Y(0.0));
        } else {
            return Vec2<Y>(Y(0.0), Y(1.0));
        }
    }
}

} // namespace proximity

#ifdef DISTANCE_UNDEF_NUMERIC_MAX
#undef SM_NUMERIC_MAX
#undef DISTANCE_UNDEF_NUMERIC_MAX
#endif
#ifdef DISTANCE_UNDEF_THREAD
#undef SM_THREAD
#undef DISTANCE_UNDEF_THREAD
#endif

#endif
