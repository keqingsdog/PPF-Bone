#pragma once
#include "../contact/pair_filter.kernel.cpp"
// Startup probe exercises both sides of each lookup over the staged arrays.
[[seam::entry(element)]]
[[seam::device_fn]] inline void start_link_codec_probe(
    const unsigned *index, const unsigned *offset, unsigned mode,
    const unsigned *from_vertices, const unsigned *target_vertices,
    unsigned *output, unsigned extent, unsigned element) {
    if (element >= extent) return;
    PairSide from;
    PairSide target;
    from.vert_count = 1u;
    target.vert_count = 1u;
    from.vert[0] = from_vertices[element];
    target.vert[0] = target_vertices[element];
    output[element] = side_linked_to(from, target, index, offset, mode) ? 1u : 0u;
}

