#pragma once
#include "aabb.kernel.cpp"

// Indices form disjoint slices of independent trees at the same depth.
// Every slice calls the single-tree merge with its original node span.
[[seam::entry(element)]]
[[seam::device_fn]] inline void aabb_refit_merge_three(
    const unsigned *first_level, const unsigned *first_nodes, AABB *first_aabb,
    unsigned first_count,
    const unsigned *second_level, const unsigned *second_nodes, AABB *second_aabb,
    unsigned second_count,
    const unsigned *third_level, const unsigned *third_nodes, AABB *third_aabb,
    unsigned element) {
    if (element < first_count) {
        aabb_merge_level(first_level, first_nodes, first_aabb, element);
        return;
    }
    const unsigned after_first = element - first_count;
    if (after_first < second_count) {
        aabb_merge_level(second_level, second_nodes, second_aabb, after_first);
        return;
    }
    aabb_merge_level(third_level, third_nodes, third_aabb, after_first - second_count);
}
