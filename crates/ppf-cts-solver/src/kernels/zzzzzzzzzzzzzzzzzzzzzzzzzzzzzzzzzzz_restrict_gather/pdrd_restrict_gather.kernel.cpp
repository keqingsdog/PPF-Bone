#pragma once
// Candidate: each cloth lane and each body lane owns disjoint output words.
// Caller must prove vertex_list exactly matches vertex_body once at staging.
[[seam::entry(i)]]
[[seam::device_fn]] inline void pdrd_restrict_gather(
    const unsigned *vertex_body, const unsigned *cloth_offset,
    const unsigned *vert_list, const PdrdBodyProp *body_prop,
    const Vec3f *rotated_rest, const float *full, float *reduced,
    unsigned body_base, unsigned nrow, unsigned nbodies, unsigned i) {
    if (i < nrow) {
        if (vertex_body[i] == 0xffffffffu) {
            const unsigned o = cloth_offset[i];
            reduced[o] = full[3u*i]; reduced[o+1u] = full[3u*i+1u]; reduced[o+2u] = full[3u*i+2u];
        }
        return;
    }
    const unsigned b = i - nrow;
    if (b >= nbodies) return;
    const PdrdBodyProp bp = body_prop[b];
    Vec3f force(0.0f, 0.0f, 0.0f), torque(0.0f, 0.0f, 0.0f);
    for (unsigned k = 0; k < bp.vertex_count; ++k) {
        const unsigned v = vert_list[bp.vertex_start + k];
        const Vec3f y(full[3u*v], full[3u*v+1u], full[3u*v+2u]);
        const Vec3f t = rotated_rest[v].cross(y);
        for (unsigned a = 0; a < 3u; ++a) { force[a] += y[a]; torque[a] += t[a]; }
    }
    const unsigned o = body_base + 6u*b;
    for (unsigned a = 0; a < 3u; ++a) { reduced[o+a] = force[a]; reduced[o+3u+a] = torque[a]; }
}
