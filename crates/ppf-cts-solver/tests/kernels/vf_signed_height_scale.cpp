// Regression for a valid interior vertex-face contact whose exact expansion
// crosses the original binary32 product-limb range.
#include "data.hpp"
#include "contact/distance.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

static float from_bits(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static double reference_height(const Vec3f &p, const Vec3f &a,
                               const Vec3f &b, const Vec3f &c) {
    double u[3], v[3], d[3];
    for (int i = 0; i < 3; ++i) {
        u[i] = double(b[i]) - double(a[i]);
        v[i] = double(c[i]) - double(a[i]);
        d[i] = double(p[i]) - double(a[i]);
    }
    const double n[3] = {
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    };
    return (d[0] * n[0] + d[1] * n[1] + d[2] * n[2]) /
           std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
}

static bool check_frame(const uint32_t bits[4][3], bool expect_original_failure,
                        int scale_exponent = 0) {
    Vec3f points[4];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            points[i][j] = std::ldexp(from_bits(bits[i][j]), scale_exponent);
        }
    }
    float original = 0.0f;
    float result = 0.0f;
    const bool original_ok = proximity::vf_signed_height_scaled(
        points[0], points[1], points[2], points[3], 1.0f, original);
    const bool ok = proximity::vf_signed_height(
        points[0], points[1], points[2], points[3], result);
    const double expected = reference_height(points[0], points[1], points[2], points[3]);
    printf("scale=2^%d original=%d result=%d height=%.10g reference=%.17g\n",
           scale_exponent, int(original_ok), int(ok), result, expected);
    if (expect_original_failure && original_ok) {
        return false;
    }
    if (!ok || !std::isfinite(result) || expected == 0.0 ||
        std::abs(double(result) - expected) > std::abs(expected) * 5e-6) {
        return false;
    }
    // Preserve the original result exactly whenever its old domain is valid.
    if (original_ok && std::memcmp(&result, &original, sizeof(result)) != 0) {
        return false;
    }
    const Vec3f coeff = proximity::point_triangle_distance_coeff<float, float>(
        points[0], points[1], points[2], points[3]);
    Vec3f separation(0.0f, 0.0f, 0.0f);
    if (!(coeff.minCoeff() > 0.0f && coeff.maxCoeff() < 1.0f) ||
        !proximity::point_triangle_interior_separation<float, float>(
            points[0], points[1], points[2], points[3], coeff, separation)) {
        return false;
    }
    const double d2 = double(separation.squaredNorm());
    const double expected_d2 = expected * expected;
    if (!(d2 > 0.0) || std::abs(d2 - expected_d2) > expected_d2 * 1e-5) {
        return false;
    }
    // The same offset boundary still separates a valid gap from an overlap.
    const float below = float(std::abs(expected) * 0.9);
    const float above = float(std::abs(expected) * 1.1);
    return d2 > double(below) * double(below) &&
           !(d2 > double(above) * double(above));
}

int main() {
    const uint32_t x0[4][3] = {
        {1016562552u,1065029347u,1038226626u},
        {2995890932u,1064767038u,827599358u},
        {2990244046u,1065415597u,1054686813u},
        {1034724925u,1065415597u,1054408212u},
    };
    const uint32_t x1[4][3] = {
        {1016760302u,1065119980u,1038242863u},
        {2995890932u,1064854906u,827599358u},
        {2990244046u,1065459531u,1054686813u},
        {1034724925u,1065459531u,1054408212u},
    };
    for (int exponent = -20; exponent <= 20; exponent += 4) {
        if (!check_frame(x0, true, exponent) ||
            !check_frame(x1, false, exponent)) {
            puts("FAIL: captured interior contact at a different world scale");
            return 1;
        }
    }
    const Vec3f a(0.0f, 0.0f, 0.0f), b(1.0f, 0.0f, 0.0f),
        c(0.0f, 1.0f, 0.0f), coplanar(0.25f, 0.25f, 0.0f);
    float zero = 1.0f;
    if (!proximity::vf_signed_height(coplanar, a, b, c, zero) || zero != 0.0f) {
        puts("FAIL: exact coplanarity");
        return 1;
    }
    // A strictly interior point exactly in its face plane, where the old
    // domain rejects an exact product limb and the wider retry reaches zero.
    const Vec3f retry_b(from_bits(0x21800004u), from_bits(0x3f000004u),
                        from_bits(0x3e800004u));
    const Vec3f retry_c(from_bits(0x21800100u), from_bits(0x3f100008u),
                        from_bits(0x3ea00008u));
    const Vec3f retry_point = (retry_b + retry_c) * 0.25f;
    const Vec3f retry_coeff =
        proximity::point_triangle_distance_coeff<float, float>(
            retry_point, a, retry_b, retry_c);
    float narrow_height = 1.0f, widened_height = 1.0f;
    Vec3f zero_separation(1.0f, 1.0f, 1.0f);
    if (!(retry_coeff.minCoeff() > 0.0f && retry_coeff.maxCoeff() < 1.0f) ||
        proximity::vf_signed_height_scaled(
            retry_point, a, retry_b, retry_c, 1.0f, narrow_height) ||
        !proximity::vf_signed_height(
            retry_point, a, retry_b, retry_c, widened_height) ||
        widened_height != 0.0f ||
        !proximity::point_triangle_interior_separation<float, float>(
            retry_point, a, retry_b, retry_c, retry_coeff, zero_separation) ||
        zero_separation.squaredNorm() != 0.0f) {
        puts("FAIL: exact coplanarity through wider retry");
        return 1;
    }
    // A real contact exactly at a positive offset must stay on the closed side.
    const Vec3f offset_point(0.25f, 0.25f, 0.125f);
    const Vec3f offset_coeff(0.5f, 0.25f, 0.25f);
    Vec3f offset_separation(0.0f, 0.0f, 0.0f);
    const float offset = 0.125f;
    if (!proximity::point_triangle_interior_separation<float, float>(
            offset_point, a, b, c, offset_coeff, offset_separation) ||
        offset_separation.squaredNorm() > offset * offset) {
        puts("FAIL: positive offset boundary");
        return 1;
    }
    const Vec3f bad(std::numeric_limits<float>::infinity(), 0.25f, 1.0f);
    float ignored = 0.0f;
    if (proximity::vf_signed_height(bad, a, b, c, ignored)) {
        puts("FAIL: nonfinite coordinate");
        return 1;
    }
    const Vec3f tiny_b(2e-38f, 0.0f, 0.0f), tiny_c(0.0f, 2e-38f, 0.0f),
        tiny_p(5e-39f, 5e-39f, 2e-38f);
    if (proximity::vf_signed_height_scaled(tiny_p, a, tiny_b, tiny_c, 65536.0f, ignored)) {
        puts("FAIL: overflowing retry scale");
        return 1;
    }
    puts("PASS: exact interior height scaling");
    return 0;
}

