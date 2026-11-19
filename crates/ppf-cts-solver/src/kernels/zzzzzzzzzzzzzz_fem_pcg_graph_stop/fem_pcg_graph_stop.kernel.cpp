#pragma once
// Stage4 independent append-only counterstop. No host iteration capture.
#include "../zzzzzzzzzzzz_fem_pcg_reduced_batch/fem_pcg_reduced_batch.kernel.cpp"

[[seam::host_device_fn]] inline unsigned fem_pcg_graph_load_iteration(const unsigned *source) {
    unsigned value=0;
    unsigned char *out=reinterpret_cast<unsigned char*>(&value);
    const unsigned char *in=reinterpret_cast<const unsigned char*>(source);
    for(unsigned k=0;k<sizeof(unsigned);++k) out[k]=in[k];
    return value;
}

[[seam::entry(count, index)]] void fem_pcg_reduced_graph_stop_beta(
    const float *error, float initial_error, const float *rz_next,
    float *rz_previous, float *beta, float *relative, float *status,
    unsigned *first_iteration, float *latched_abs_error, unsigned *current_iteration,
    float tolerance, unsigned max_iterations,
    unsigned extent, unsigned count, unsigned index);

[[seam::device_fn]] inline void fem_pcg_reduced_graph_stop_beta(
    const float *error, float initial_error, const float *rz_next,
    float *rz_previous, float *beta, float *relative, float *status,
    unsigned *first_iteration, float *latched_abs_error, unsigned *current_iteration,
    float tolerance, unsigned max_iterations,
    unsigned extent, unsigned index) {
    if (index >= extent || *status != 0.0f) return;
    const unsigned iteration = fem_pcg_graph_load_iteration(current_iteration) + 1u;
    fem_pcg_batch_store_iteration(current_iteration, iteration);
    // Exact B=0 worst_relative_residual: one cloth group; nonfinite ratio
    // becomes +infinity, and the max reduction starts from zero.
    const float ratio = pcg_group_relative_residual(*error, initial_error, initial_error);
    const float residual = ratio > 0.0f ? ratio : 0.0f;
    *relative = residual;
    if (residual < tolerance) {
        fem_pcg_batch_store_iteration(first_iteration, iteration); *latched_abs_error = *error;
        *status = 1.0f; return;
    }
    if (iteration >= max_iterations || fmath::isnan(residual) || fmath::isinf(residual)) {
        fem_pcg_batch_store_iteration(first_iteration, iteration); *latched_abs_error = *error;
        *status = 2.0f;
        return;
    }
    *beta = fmath::div(*rz_next, *rz_previous);
    *rz_previous = *rz_next;
    *status = 0.0f;
}

