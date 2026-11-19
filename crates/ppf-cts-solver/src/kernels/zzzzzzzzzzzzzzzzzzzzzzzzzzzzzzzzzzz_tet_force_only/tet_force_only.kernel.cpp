#pragma once
[[seam::entry(row)]]
[[seam::device_fn]] inline void ftt_gather_force_only(
    const unsigned *adj_offset, const unsigned *adj, const unsigned *mask,
    const float *response, float *result, unsigned *fault,
    unsigned row, unsigned extent) {
    if (row >= extent || mask[row] != 0u) return;
    float sum[3] = {0.0f, 0.0f, 0.0f};
    for (unsigned k = adj_offset[row]; k < adj_offset[row + 1u]; ++k) {
        const unsigned element = adj[k] / 4u, slot = adj[k] % 4u;
        for (unsigned d = 0u; d < 3u; ++d) {
            const float f = slot == 0u
                ? -(response[11u * element + d] + response[11u * element + 3u + d] + response[11u * element + 6u + d])
                : response[11u * element + 3u * (slot - 1u) + d];
            sum[d] += f;
        }
    }
    for (unsigned d = 0u; d < 3u; ++d) {
        const float v = result[3u * row + d] + sum[d];
        if (!(v == v && v <= 3.402823466e38f && v >= -3.402823466e38f)) {
            fault[row] = 6u;
            return;
        }
        result[3u * row + d] = v;
    }
}
