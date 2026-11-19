#pragma once
#include "../zzzzzzzzzzzzzz_fem_pcg_graph_stop/fem_pcg_graph_stop.kernel.cpp"

[[seam::entry(count, index)]] void joint_pcg_batch8_factor_fault(
    const unsigned *fault, float *status, unsigned *first_iteration,
    unsigned *current_iteration, unsigned extent, unsigned count, unsigned index);
[[seam::device_fn]] inline void joint_pcg_batch8_factor_fault(
    const unsigned *fault, float *status, unsigned *first_iteration,
    unsigned *current_iteration, unsigned extent, unsigned index) {
    if (index != 0u || index >= extent || *status != 0.0f || *fault == 0u) return;
    const unsigned iteration = fem_pcg_graph_load_iteration(current_iteration) + 1u;
    fem_pcg_batch_store_iteration(current_iteration, iteration);
    fem_pcg_batch_store_iteration(first_iteration, iteration);
    *status = 3.0f;
}
