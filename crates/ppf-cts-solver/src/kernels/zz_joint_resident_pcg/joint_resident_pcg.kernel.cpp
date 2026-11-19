#pragma once
#include "../solver/pcg.kernel.cpp"

[[seam::entry(count, index)]] void joint_pcg_group_stop_beta(
    const float *norm, const float *initial, float initial_all,
    unsigned groups, unsigned cloth_present, const float *rz_next,
    float *rz_previous, float *beta, float *relative, float *status,
    float tolerance, unsigned iteration, unsigned max_iterations,
    unsigned extent, unsigned count, unsigned index);

[[seam::device_fn]] inline void joint_pcg_group_stop_beta(
    const float *norm, const float *initial, float initial_all,
    unsigned groups, unsigned cloth_present, const float *rz_next,
    float *rz_previous, float *beta, float *relative, float *status,
    float tolerance, unsigned iteration, unsigned max_iterations,
    unsigned extent, unsigned index) {
    if (index >= extent || *status != 0.0f) return;
    float worst = 0.0f;
    for (unsigned i = 0; i < groups; ++i) {
        const float now = (i == 0 && !cloth_present) ? 0.0f : norm[i];
        const float ratio = pcg_group_relative_residual(now, initial[i], initial_all);
        if (ratio > worst) worst = ratio;
    }
    *relative = worst;
    if (worst < tolerance) {
        *status = 1.0f;
        return;
    }
    if (iteration >= max_iterations || fmath::isnan(worst) || fmath::isinf(worst)) {
        *status = 2.0f;
        return;
    }
    *beta = fmath::div(*rz_next, *rz_previous);
    *rz_previous = *rz_next;
}
