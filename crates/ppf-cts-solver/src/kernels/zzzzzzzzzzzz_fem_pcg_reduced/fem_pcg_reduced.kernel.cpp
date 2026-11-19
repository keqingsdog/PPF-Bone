#pragma once
// Isolated stage2. Original reduced recurrence, NOT normal-PCG verdicts.
// Append-only generated entry registration belongs to the isolated builder.
#include "../solver/pcg.kernel.cpp"
#include "../primitives/vec_ops.kernel.cpp"

[[seam::entry(count, index)]] void fem_pcg_reduced_alpha(
    const float *rz, const float *pap, float *alpha,
    unsigned extent, unsigned count, unsigned index);

[[seam::device_fn]] inline void fem_pcg_reduced_alpha(
    const float *rz, const float *pap, float *alpha,
    unsigned extent, unsigned index) {
    if (index >= extent) return;
    const float denominator = *pap != 0.0f ? *pap : 1.0f;
    *alpha = fmath::div(*rz, denominator);
}

[[seam::entry(count, index)]] void fem_pcg_reduced_add_indirect(
    const float *source, float *destination, const float *coefficient,
    float sign, unsigned extent, unsigned count, unsigned index);

[[seam::device_fn]] inline void fem_pcg_reduced_add_indirect(
    const float *source, float *destination, const float *coefficient,
    float sign, unsigned extent, unsigned index) {
    if (index >= extent) return;
    // Delegate the original += body. Do not replace it with the explicitly
    // fused upstream vec_add_scaled_indirect implementation.
    const float scale = sign * (*coefficient);
    vec_add_scaled<float>(source, destination, scale, index);
}

[[seam::entry(count, index)]] void fem_pcg_reduced_stop_beta(
    const float *error, float initial_error, const float *rz_next,
    float *rz_previous, float *beta, float *relative, float *status,
    float tolerance, unsigned iteration, unsigned max_iterations,
    unsigned extent, unsigned count, unsigned index);

[[seam::device_fn]] inline void fem_pcg_reduced_stop_beta(
    const float *error, float initial_error, const float *rz_next,
    float *rz_previous, float *beta, float *relative, float *status,
    float tolerance, unsigned iteration, unsigned max_iterations,
    unsigned extent, unsigned index) {
    if (index >= extent) return;
    // Exact B=0 worst_relative_residual: one cloth group; nonfinite ratio
    // becomes +infinity, and the max reduction starts from zero.
    const float ratio = pcg_group_relative_residual(*error, initial_error, initial_error);
    const float residual = ratio > 0.0f ? ratio : 0.0f;
    *relative = residual;
    if (residual < tolerance) { *status = 1.0f; return; }
    if (iteration >= max_iterations || fmath::isnan(residual) || fmath::isinf(residual)) {
        *status = 2.0f;
        return;
    }
    *beta = fmath::div(*rz_next, *rz_previous);
    *rz_previous = *rz_next;
    *status = 0.0f;
}

[[seam::entry(count, index)]] void fem_pcg_reduced_direction(
    const float *z, float *p, const float *beta, const float *status,
    unsigned extent, unsigned count, unsigned index);

[[seam::device_fn]] inline void fem_pcg_reduced_direction(
    const float *z, float *p, const float *beta, const float *status,
    unsigned extent, unsigned index) {
    if (index >= extent || *status != 0.0f) return;
    vec_combine<float>(z, p, p, 1.0f, *beta, index);
}
