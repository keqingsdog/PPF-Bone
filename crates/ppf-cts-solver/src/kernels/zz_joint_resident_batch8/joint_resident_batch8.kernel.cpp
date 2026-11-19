#pragma once
#include "../zzzzzzzzzzzzzz_fem_pcg_graph_stop/fem_pcg_graph_stop.kernel.cpp"
#include "../zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz_restoring_cache/restoring_fem_frozen.kernel.cpp"
#include "../zzzzzzzzzzzzzzzzzzzzzzzz_factor_tet/factor_tet.kernel.cpp"

[[seam::entry(count, index)]] void joint_pcg_batch8_stop(
    const float *norm, const float *initial, float initial_all, unsigned groups,
    unsigned cloth_present, const float *rz_next, float *rz_previous, float *beta,
    float *relative, float *status, unsigned *first_iteration,
    unsigned *current_iteration, float tolerance, unsigned max_iterations,
    unsigned extent, unsigned count, unsigned index);
[[seam::device_fn]] inline void joint_pcg_batch8_stop(
    const float *norm, const float *initial, float initial_all, unsigned groups,
    unsigned cloth_present, const float *rz_next, float *rz_previous, float *beta,
    float *relative, float *status, unsigned *first_iteration,
    unsigned *current_iteration, float tolerance, unsigned max_iterations,
    unsigned extent, unsigned index) {
    if (index >= extent || *status != 0.0f) return;
    const unsigned iteration = fem_pcg_graph_load_iteration(current_iteration) + 1u;
    fem_pcg_batch_store_iteration(current_iteration, iteration);
    float worst = 0.0f;
    for (unsigned i = 0; i < groups; ++i) {
        const float now = (i == 0 && !cloth_present) ? 0.0f : norm[i];
        const float ratio = pcg_group_relative_residual(now, initial[i], initial_all);
        if (ratio > worst) worst = ratio;
    }
    *relative = worst;
    if (worst < tolerance || iteration >= max_iterations || fmath::isnan(worst) || fmath::isinf(worst)) {
        fem_pcg_batch_store_iteration(first_iteration, iteration);
        *status = worst < tolerance ? 1.0f : 2.0f;
        return;
    }
    *beta = fmath::div(*rz_next, *rz_previous);
    *rz_previous = *rz_next;
}

[[seam::entry(count, index)]] void joint_pcg_batch8_ball(
    unsigned count_rows, const float *factors, const unsigned *offsets,
    const unsigned *published, unsigned epoch, const float *input,
    compute::atomic_float_t *output, unsigned *fault, const float *status,
    unsigned count, unsigned index);
[[seam::device_fn]] inline void joint_pcg_batch8_ball(
    unsigned count_rows, const float *factors, const unsigned *offsets,
    const unsigned *published, unsigned epoch, const float *input,
    compute::atomic_float_t *output, unsigned *fault, const float *status,
    unsigned index) {
    if (*status != 0.0f) return;
    restoring_ball_frozen_apply(count_rows, factors, offsets, published, epoch, input, output, fault, index);
}

[[seam::entry(count, i)]] void resident_guard_restoring_angular_frozen_apply(
    unsigned count_rows,
    const float* factors,
    const unsigned* offsets,
    const unsigned* published,
    unsigned epoch,
    const float* input,
    compute::atomic_float_t* output,
    unsigned* fault,
    const float *status,
    unsigned count,
    unsigned i);
[[seam::device_fn]] inline void resident_guard_restoring_angular_frozen_apply(
    unsigned count_rows,
    const float* factors,
    const unsigned* offsets,
    const unsigned* published,
    unsigned epoch,
    const float* input,
    compute::atomic_float_t* output,
    unsigned* fault,
    const float *status,
    unsigned i) {
    if (*status != 0.0f) return;
    restoring_angular_frozen_apply(count_rows, factors, offsets, published, epoch, input, output, fault, i);
}

[[seam::entry(count, i)]] void resident_guard_restoring_fem_frozen_apply(
    unsigned record_count,
    const float* cached_hessian,
    const unsigned* cached_index,
    const unsigned* published_epoch,
    unsigned epoch,
    const float* input,
    compute::atomic_float_t* output,
    unsigned* fault,
    const float *status,
    unsigned count,
    unsigned i);
[[seam::device_fn]] inline void resident_guard_restoring_fem_frozen_apply(
    unsigned record_count,
    const float* cached_hessian,
    const unsigned* cached_index,
    const unsigned* published_epoch,
    unsigned epoch,
    const float* input,
    compute::atomic_float_t* output,
    unsigned* fault,
    const float *status,
    unsigned i) {
    if (*status != 0.0f) return;
    restoring_fem_frozen_apply(record_count, cached_hessian, cached_index, published_epoch, epoch, input, output, fault, i);
}

[[seam::entry(count, i)]] void resident_guard_joint_angular_extension_apply(
    const AngularControlGpu *records,
    const AngularTwistEvalContext *twist_context,
    const MaterialSpinContext *material_spin_context,
    const AngularSoftLimitGpu *limits,
    const AngularDriveGpu *drives,
    unsigned drive_count,
    const unsigned *drive_offsets,
    const AngularDriveKeyGpu *keys,
    const PdrdRigidState *state,
    const PdrdRigidState *reference,
    unsigned bodies,
    unsigned base,
    unsigned dim,
    float sample_time_s,
    float dt_s,
    unsigned action,
    const float *input,
    float *tangent_cache,
    unsigned *tangent_offsets,
    unsigned *tangent_published,
    unsigned epoch,
    compute::atomic_float_t *output,
    unsigned *fault,
    const float *status,
    unsigned count,
    unsigned i);
[[seam::device_fn]] inline void resident_guard_joint_angular_extension_apply(
    const AngularControlGpu *records,
    const AngularTwistEvalContext *twist_context,
    const MaterialSpinContext *material_spin_context,
    const AngularSoftLimitGpu *limits,
    const AngularDriveGpu *drives,
    unsigned drive_count,
    const unsigned *drive_offsets,
    const AngularDriveKeyGpu *keys,
    const PdrdRigidState *state,
    const PdrdRigidState *reference,
    unsigned bodies,
    unsigned base,
    unsigned dim,
    float sample_time_s,
    float dt_s,
    unsigned action,
    const float *input,
    float *tangent_cache,
    unsigned *tangent_offsets,
    unsigned *tangent_published,
    unsigned epoch,
    compute::atomic_float_t *output,
    unsigned *fault,
    const float *status,
    unsigned i) {
    if (*status != 0.0f) return;
    joint_angular_extension_apply(records, twist_context, material_spin_context, limits, drives, drive_count, drive_offsets, keys, state, reference, bodies, base, dim, sample_time_s, dt_s, action, input, tangent_cache, tangent_offsets, tangent_published, epoch, output, fault, i);
}

[[seam::entry(count, i)]] void resident_guard_joint_fem_extension_apply(
    const FemJointGpu* records,
    const MaterialSpinContext* material_spin_context,
    unsigned record_count,
    const FemFrameGpu* cache,
    const unsigned* vertex_to_free,
    unsigned vertex_count,
    unsigned free_count,
    unsigned rigid_base,
    unsigned dim,
    const AngularSoftLimitGpu* limits,
    const AngularDriveGpu* drives,
    unsigned drive_count,
    const AngularDriveKeyGpu* keys,
    float sample_time_s,
    float dt,
    unsigned action,
    const float* input,
    compute::atomic_float_t* output,
    compute::atomic_float_t* nodal_blocks,
    compute::atomic_float_t* rigid_blocks,
    unsigned* fault,
    const float *status,
    unsigned count,
    unsigned i);
[[seam::device_fn]] inline void resident_guard_joint_fem_extension_apply(
    const FemJointGpu* records,
    const MaterialSpinContext* material_spin_context,
    unsigned record_count,
    const FemFrameGpu* cache,
    const unsigned* vertex_to_free,
    unsigned vertex_count,
    unsigned free_count,
    unsigned rigid_base,
    unsigned dim,
    const AngularSoftLimitGpu* limits,
    const AngularDriveGpu* drives,
    unsigned drive_count,
    const AngularDriveKeyGpu* keys,
    float sample_time_s,
    float dt,
    unsigned action,
    const float* input,
    compute::atomic_float_t* output,
    compute::atomic_float_t* nodal_blocks,
    compute::atomic_float_t* rigid_blocks,
    unsigned* fault,
    const float *status,
    unsigned i) {
    if (*status != 0.0f) return;
    joint_fem_extension_apply(records, material_spin_context, record_count, cache, vertex_to_free, vertex_count, free_count, rigid_base, dim, limits, drives, drive_count, keys, sample_time_s, dt, action, input, output, nodal_blocks, rigid_blocks, fault, i);
}

[[seam::entry(count, element)]] void resident_guard_ftt_evaluate(
    const unsigned *tet,
    const float *factor,
    const float *weight,
    const float *input,
    const float *target,
    const unsigned *mask,
    float *response,
    unsigned *fault,
    unsigned lift,
    const float *status,
    unsigned extent,
    unsigned count,
    unsigned element);
[[seam::device_fn]] inline void resident_guard_ftt_evaluate(
    const unsigned *tet,
    const float *factor,
    const float *weight,
    const float *input,
    const float *target,
    const unsigned *mask,
    float *response,
    unsigned *fault,
    unsigned lift,
    const float *status,
    unsigned extent,
    unsigned element) {
    if (*status != 0.0f) return;
    ftt_evaluate(tet, factor, weight, input, target, mask, response, fault, lift, element, extent);
}

[[seam::entry(count, row)]] void resident_guard_ftt_gather(
    const unsigned *adj_offset,
    const unsigned *adj,
    const unsigned *mask,
    const unsigned *tet,
    const float *response,
    float *result,
    float *curvature,
    float *absolute,
    unsigned *fault,
    unsigned lift,
    const float *status,
    unsigned extent,
    unsigned count,
    unsigned row);
[[seam::device_fn]] inline void resident_guard_ftt_gather(
    const unsigned *adj_offset,
    const unsigned *adj,
    const unsigned *mask,
    const unsigned *tet,
    const float *response,
    float *result,
    float *curvature,
    float *absolute,
    unsigned *fault,
    unsigned lift,
    const float *status,
    unsigned extent,
    unsigned row) {
    if (*status != 0.0f) return;
    ftt_gather(adj_offset, adj, mask, tet, response, result, curvature, absolute, fault, lift, row, extent);
}
