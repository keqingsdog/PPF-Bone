// File: pcg_warm_mask_history.kernel.cpp
// License: Apache v2.0

#pragma once

// Keep this new entry in its own final ABI source. Existing entry IDs are
// append-only, so adding it to solver/pcg.kernel.cpp would insert its backend
// registration among the older PCG kernels and shift every later CUDA ID.
//
// A warm correction is homogeneous: fixed vertex DOFs belong to the current
// affine seed and must never be carried across Newton solves. The driver
// supplies one reduced scalar mask (free=1, prescribed=0); each thread owns
// exactly one history element, so this in-place projection is disjoint.
[[seam::entry(row)]]
[[seam::device_fn]] inline void
pcg_warm_mask_history(float *history, const unsigned *free_row, unsigned row) {
    if (free_row[row] == 0u) history[row] = 0.0f;
}
