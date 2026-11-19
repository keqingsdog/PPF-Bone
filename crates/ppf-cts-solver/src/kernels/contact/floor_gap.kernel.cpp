#pragma once
#include "distance.hpp"

[[seam::device_fn]] inline bool floor_scalar_finite(float x) {
    return x <= 3.402823466e38f && x >= -3.402823466e38f;
}
[[seam::device_fn]] inline bool floor_vector_finite(const Vec3f &x) {
    return floor_scalar_finite(x[0]) && floor_scalar_finite(x[1]) &&
           floor_scalar_finite(x[2]);
}

// The assembly and the read-only accepted-candidate check consume the same
// expression. No gap clamp, margin or material arithmetic is introduced.
[[seam::device_fn]] inline float floor_signed_gap(
    const Vec3f &x, const Floor &plane) {
    return (x - plane.ground).cast<float>().dot(plane.up);
}
