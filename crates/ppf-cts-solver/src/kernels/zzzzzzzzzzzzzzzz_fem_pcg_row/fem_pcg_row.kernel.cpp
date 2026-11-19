#pragma once
// Stage6 pure-FEM row fusion. Preserve Stage5 scalar/stop/fold ordering.
#include "../primitives/vec_ops.kernel.cpp"
#include "../solver/pcg.kernel.cpp"
#include "../energy/model/pdrd_rigid.kernel.cpp"
#include "../zzzzzzzzzzzz_fem_pcg_reduced_batch/fem_pcg_reduced_batch.kernel.cpp"
#include "../zzzzzzzzzzzzzz_fem_pcg_graph_stop/fem_pcg_graph_stop.kernel.cpp"

[[seam::entry(count,v)]] void fem_pcg_reduced_row_update(
    const float *direction, const float *applied, const float *alpha,
    const float *status, float *iterate, float *residual,
    const unsigned *vertex_body, const unsigned *cloth_offset,
    const Mat3x3f *inverse_diagonal, float *preconditioned,
    float *product, float *magnitude, unsigned count, unsigned v);

[[seam::device_fn]] inline void fem_pcg_reduced_row_update(
    const float *direction, const float *applied, const float *alpha,
    const float *status, float *iterate, float *residual,
    const unsigned *vertex_body, const unsigned *cloth_offset,
    const Mat3x3f *inverse_diagonal, float *preconditioned,
    float *product, float *magnitude, unsigned v) {
    // The two Stage5 add launches read status separately. They both skip for
    // stop=1/2 and NaN; preconditioning and dot terms still run afterward.
    if (*status == 0.0f) {
        const float add_scale = 1.0f * (*alpha);
        for (unsigned c = 0; c < 3u; ++c)
            vec_add_scaled<float>(direction, iterate, add_scale, 3u * v + c);
        const float residual_scale = -1.0f * (*alpha);
        for (unsigned c = 0; c < 3u; ++c)
            vec_add_scaled<float>(applied, residual, residual_scale, 3u * v + c);
    }
    pdrd_precond_cloth_row(vertex_body, cloth_offset, inverse_diagonal,
                           residual, preconditioned, v);
    pcg_dot_terms(residual, preconditioned, v, product[v], magnitude[v]);
}

// Last pAp tree level and original alpha in one writer lane.
[[seam::entry(count,block)]] [[seam::group]] void fem_pcg_reduced_fold_alpha_last(
    const float *source, unsigned length, unsigned width,
    [[seam::scratch(32)]] float *scratch, float *pap,
    const float *rz, float *alpha, const float *status,
    [[seam::lane]] unsigned lane, [[seam::width]] unsigned threads,
    unsigned block, unsigned count);
[[seam::device_fn]] inline void fem_pcg_reduced_fold_alpha_last(
    const float *source, unsigned length, unsigned width,
    float *scratch, float *pap, const float *rz, float *alpha,
    const float *status, unsigned lane, unsigned threads, unsigned block) {
    vec_block_sum_cooperative(source, length, width, scratch, pap,
                              lane, threads, block);
    if (compute::is_block_writer(lane, threads) && block == 0u)
        fem_pcg_reduced_batch_alpha(rz, pap, alpha, status, 1u, 0u);
}

// Last independent L1/rz tree level and original graph stop in one lane.
[[seam::entry(count,block)]] [[seam::group]] void fem_pcg_reduced_fold_stop_last(
    const float *first_source, unsigned first_length,
    const float *second_source, unsigned second_length, unsigned width,
    [[seam::scratch(64)]] float *scratch,
    float *error, float *rz_next, unsigned first_count, unsigned second_count,
    float initial_error, float *rz_previous, float *beta, float *relative,
    float *status, unsigned *first_iteration, float *latched_abs_error,
    unsigned *current_iteration, float tolerance, unsigned max_iterations,
    [[seam::lane]] unsigned lane, [[seam::width]] unsigned threads,
    unsigned block, unsigned count);
[[seam::device_fn]] inline void fem_pcg_reduced_fold_stop_last(
    const float *first_source, unsigned first_length,
    const float *second_source, unsigned second_length, unsigned width,
    float *scratch, float *error, float *rz_next,
    unsigned first_count, unsigned second_count, float initial_error,
    float *rz_previous, float *beta, float *relative, float *status,
    unsigned *first_iteration, float *latched_abs_error,
    unsigned *current_iteration, float tolerance, unsigned max_iterations,
    unsigned lane, unsigned threads, unsigned block) {
    vec_block_sum_dual_cooperative(first_source, first_length,
        second_source, second_length, width, scratch, error, rz_next,
        first_count, second_count, lane, threads, block);
    if (compute::is_block_writer(lane, threads) && block == 0u)
        fem_pcg_reduced_graph_stop_beta(error, initial_error, rz_next,
            rz_previous, beta, relative, status, first_iteration,
            latched_abs_error, current_iteration, tolerance,
            max_iterations, 1u, 0u);
}
