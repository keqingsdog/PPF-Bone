#pragma once
#include "../contact/floor_gap.kernel.cpp"

// Per-pair mask:0 ignored by the original floor barrier;1 active and positive;
// 3 active but not finite/strictly positive. Inputs are immutable actual GPU
// candidate positions; only this diagnostic mask is written.
[[seam::entry(i)]]
[[seam::device_fn]] inline void plane_candidate_domain(
    const Vec3f *candidate, const VertexProp *vertex_prop,
    const Floor *floor, unsigned floor_count,
    unsigned *out_mask, unsigned vertex_count, unsigned i) {
    const VertexProp prop = vertex_prop[i];
    const Vec3f x = candidate[i];
    for (unsigned j = 0; j < floor_count; ++j) {
        unsigned mask = 0u;
        const Floor plane = floor[j];
        // Same surface-prefix/mass/fix eligibility as vertex_constraint.
        // Kinematic colliders retain their independent clamped contract.
        if (prop.fix_index == 0u && prop.mass > 0.0f && !plane.kinematic) {
            // Nonfinite arithmetic must not impersonate an inactive contact.
            if (!floor_vector_finite(x) || !floor_vector_finite(plane.ground) ||
                !floor_vector_finite(plane.up) || !floor_scalar_finite(plane.ghat) ||
                !floor_scalar_finite(plane.thickness)) {
                out_mask[j * vertex_count + i] = 3u;
                continue;
            }
            const Vec3f up = plane.up;
            const Vec3f ground = plane.ground + plane.ghat * up;
            const Vec3f e = (x - ground).cast<float>();
            const float signed_activation = e.dot(up);
            if (!floor_vector_finite(ground) || !floor_vector_finite(e) ||
                !floor_scalar_finite(signed_activation)) {
                out_mask[j * vertex_count + i] = 3u;
                continue;
            }
            if (signed_activation < 0.0f) {
                const float depth = -signed_activation;
                if (!(plane.thickness > 0.0f && depth > plane.thickness)) {
                    const float gap = floor_signed_gap(x, plane);
                    mask = (gap > 0.0f && floor_scalar_finite(gap)) ? 1u : 3u;
                }
            }
        }
        out_mask[j * vertex_count + i] = mask;
    }
}
