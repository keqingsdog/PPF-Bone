#pragma once

// Inputs are immutable FP32 arrays. Reading a local float through unsigned
// char is valid C++ object-representation access and retains signed zero.
[[seam::entry(i)]]
[[seam::device_fn]] inline void b32_compare_finite_bits(
    const float *candidate_values,
    const float *proposal_values,
    compute::atomic_uint_t *mismatch,
    unsigned i) {
    const float candidate = candidate_values[i];
    const float proposal = proposal_values[i];
    const unsigned char *a = reinterpret_cast<const unsigned char *>(&candidate);
    const unsigned char *b = reinterpret_cast<const unsigned char *>(&proposal);
    bool differs = !(candidate <= 3.402823466e38f && candidate >= -3.402823466e38f);
    for (unsigned j = 0; j < 4; ++j) {
        differs = differs || a[j] != b[j];
    }
    if (differs) {
        compute::atomic_add(mismatch, 1u);
    }
}
