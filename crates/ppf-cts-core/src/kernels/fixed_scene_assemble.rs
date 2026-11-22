// Fork modification: changed from the upstream baseline.
// File: crates/ppf-cts-core/src/kernels/fixed_scene_assemble.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// FixedScene.__init__ assembly. Mirrors the body of the Python
// constructor in frontend/_scene_.py. The Python path runs
// four checks (self-intersection, rod-tri offset, contact-offset,
// invisible-collider) and a small derived-data pass (per-tri area +
// face-to-vertex weights). All of that lives here now; the Python
// __init__ collapses to a single binding call.
//
// This module is intentionally separated from `scene_build.rs` so
// the parallel `Scene.build()` port can extend `scene_build.rs`
// without conflicting on the assembly entry point.

use crate::kernels::intersection as isect;
use crate::kernels::invisible_collider as inv_coll;
use crate::kernels::proximity as prox;
use crate::kernels::scene_build as sb;
use crate::kernels::start_links::StartLinks;

const SELF_INTERSECTION_DRAW_BUDGET: usize = 100;

/// Reserve one geometry sample for each identifiable object pair, then
/// fill the remaining budget in scan order. This only affects diagnostics;
/// the total and the failure decision are taken from the full scan.
fn representative_intersection_indices<K: Eq + std::hash::Hash>(
    len: usize,
    limit: usize,
    key_at: impl Fn(usize) -> K,
) -> Vec<usize> {
    if len <= limit {
        return (0..len).collect();
    }
    if limit == 0 {
        return Vec::new();
    }
    let mut seen = std::collections::HashSet::with_capacity(limit);
    let mut selected = Vec::with_capacity(limit);
    let mut extras = Vec::with_capacity(limit);
    for i in 0..len {
        if seen.insert(key_at(i)) {
            selected.push(i);
            if selected.len() == limit {
                break;
            }
        } else if extras.len() < limit {
            extras.push(i);
        }
    }
    selected.extend(extras.into_iter().take(limit - selected.len()));
    selected.sort_unstable();
    selected
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum IntersectionOwner {
    Object(i32),
    UnknownDynamic,
    UnknownStatic,
}

fn intersection_owner(
    ids: Option<&[i32]>,
    vertices: &[i32],
    fallback: IntersectionOwner,
) -> IntersectionOwner {
    let Some(ids) = ids else { return fallback };
    let Some(&id) = vertices.first().and_then(|&v| ids.get(v as usize)) else {
        return fallback;
    };
    if id == isect::NO_OBJECT_ID
        || !vertices.iter().all(|&v| ids.get(v as usize) == Some(&id))
    {
        return fallback;
    }
    IntersectionOwner::Object(id)
}

/// One static wall (single-keyframe only). Kinematic walls are
/// filtered out *before* assembly because the solver handles their
/// violations on its own.
#[derive(Debug, Clone)]
pub struct WallEntry {
    pub pos: [f64; 3],
    pub normal: [f64; 3],
}

/// One static sphere (single-keyframe only).
#[derive(Debug, Clone)]
pub struct SphereEntry {
    pub pos: [f64; 3],
    pub radius: f64,
    pub is_inverted: bool,
    pub is_hemisphere: bool,
}

/// Inputs to `fixed_scene_assemble`. Every slice is borrowed; the
/// assembly never owns Python state.
pub struct AssembleInput<'a> {
    /// Per-vertex displacement-map index. Length N.
    pub vert_dmap: &'a [u32],
    /// Per-vertex local position, flat `(N, 3)`. Length 3N.
    pub vert_local: &'a [f64],
    /// Per-object displacement, flat `(D, 3)`. Length 3D.
    pub displacement: &'a [f64],
    /// Triangles, flat `(M, 3)` i32.
    pub tri: &'a [i32],
    /// Rod edges, flat `(K, 2)` i32.
    pub rod: &'a [i32],
    /// `(M,)` bool, collider triangles.
    pub tri_is_collider: &'a [bool],
    /// `(K,)` bool, collider rod edges.
    pub rod_is_collider: &'a [bool],
    /// `(M,)` per-tri 1-based PDRD rigid-body id (0 = not a member of any
    /// rigid body). Dynamic triangles only; static colliders are appended
    /// with id 0 in the combined self-intersection check. Two triangles of
    /// the same nonzero body are an intra-rigid-body self-intersection and
    /// are tolerated (skipped). Empty slice = treat every triangle as 0.
    pub tri_body_id: &'a [i32],
    /// `(M,)` per-tri contact-offset; empty when no offsets are set.
    pub tri_offset: &'a [f64],
    /// `(K,)` per-rod contact-offset; empty when no offsets are set.
    pub rod_offset: &'a [f64],
    /// Optional static-mesh vertex buffer (already displaced),
    /// `(S, 3)` flat. Set to `None` when no statics.
    pub static_verts: Option<&'a [f64]>,
    /// Optional static-mesh triangles, `(T, 3)` i32 flat.
    pub static_tris: Option<&'a [i32]>,
    /// Optional list of pinned vertex indices. Wall and sphere checks
    /// skip these. Empty for "none".
    pub pinned_vertices: &'a [usize],
    /// Intersection allowances (issue #138), all per DYNAMIC vertex and all
    /// optional; `None` throughout tolerates nothing, so every intersecting
    /// pair is reported. The combined self-intersection scan appends the STATIC
    /// collision vertices, which get `NO_OBJECT_ID`, an empty policy and no
    /// pin: a static is always the OTHER object, so it is the dynamic side
    /// that decides, and "either side opts in" makes that the whole rule.
    pub vert_object_id: Option<&'a [i32]>,
    /// Per DYNAMIC vertex source-group identity, read only by the inter-group
    /// allowance. The appended STATIC collision vertices get `NO_GROUP_ID`,
    /// which makes a collider another group from every object.
    pub vert_group_id: Option<&'a [i32]>,
    pub vert_policy: Option<&'a [u8]>,
    pub vert_pin_allow: Option<&'a [bool]>,
    /// Allow Existing Intersections, per DYNAMIC vertex: true where the
    /// vertex's object opted in. `None` means no object did. A pair the three
    /// scans find (a crossing, or a pair closer than its contact offsets) is
    /// LINKED rather than reported when either side opted in; see
    /// `start_links.rs`. The appended STATIC collision vertices never opt in.
    pub vert_allow_existing: Option<&'a [bool]>,
    /// Static walls only. Kinematic walls are filtered upstream.
    pub walls: &'a [WallEntry],
    /// Static spheres only.
    pub spheres: &'a [SphereEntry],
    /// True when at least one face has a non-`NONE` dyn-color entry.
    /// Drives the `face_to_vert_weights` allocation.
    pub has_dyn_color: bool,
}

/// One self-intersecting tri/rod-tri pair, with the world-space
/// triangle vertices laid out for the violation viewer. `is_rod` is
/// true for rod-triangle hits (`pair[0] == -1` in the kernel's
/// output).
#[derive(Debug, Clone)]
pub struct SelfIntersectionEntry {
    pub tri_positions: Vec<[f64; 3]>,
    pub is_rod: bool,
}

/// One contact-offset violating pair. `ei`/`ej` use the proximity
/// namespace: `0..n_tris` are triangles, `n_tris..` are rod edges.
#[derive(Debug, Clone)]
pub struct ContactOffsetEntry {
    pub ei_is_triangle: bool,
    pub ej_is_triangle: bool,
    pub ei_pos: Vec<[f64; 3]>,
    pub ej_pos: Vec<[f64; 3]>,
}

#[derive(Debug, Clone)]
pub struct WallVertexEntry {
    pub pos: [f64; 3],
    pub wall_idx: usize,
    pub signed_dist: f64,
}

#[derive(Debug, Clone)]
pub struct SphereVertexEntry {
    pub pos: [f64; 3],
    pub sphere_idx: usize,
    pub dist: f64,
}

/// Output of `fixed_scene_assemble`: validation flags, the truncated
/// preview lists feeding `ValidationError.violations`, the human-
/// readable message, and the derived data the Python `__init__`
/// stored on `self`.
#[derive(Debug, Clone, Default)]
pub struct AssembleOutput {
    pub has_self_intersection: bool,
    pub has_contact_offset_violation: bool,
    pub has_wall_violation: bool,
    pub has_sphere_violation: bool,
    /// First 100 entries (matches the Python truncation).
    pub self_intersections: Vec<SelfIntersectionEntry>,
    pub n_self_intersections_total: usize,
    pub n_tri_tri: usize,
    pub n_rod_tri: usize,
    pub contact_offset_pairs: Vec<ContactOffsetEntry>,
    pub n_contact_offset_total: usize,
    pub wall_vertices: Vec<WallVertexEntry>,
    pub n_wall_total: usize,
    pub first_wall: Option<(usize, usize, f64)>, // (vertex, wall, signed)
    pub sphere_vertices: Vec<SphereVertexEntry>,
    pub n_sphere_total: usize,
    pub first_sphere: Option<(usize, usize)>,
    pub combined_message: String,
    /// Allow Existing Intersections: the vertex links, flat `[u0, v0, ...]`
    /// with `u < v`, in the COMBINED namespace (the dynamic vertices, then the
    /// STATIC collision vertices after them). Empty when nothing was linked.
    pub start_links: Vec<u32>,
    /// How many element pairs the links came from: crossings and clearance
    /// violations the build exempted instead of refusing.
    pub n_start_link_pairs: usize,
    /// The first `start_links::MAX_RECORDED_PAIRS` of those element pairs, each
    /// as the world-space positions of its two elements' vertices (three for a
    /// triangle, two for a rod edge), for the add-on's overlay of what was
    /// exempted. A collision-mesh element reads its own static positions.
    pub start_link_pairs: Vec<(Vec<[f64; 3]>, Vec<[f64; 3]>)>,
    /// Per-tri area, computed from `vert_local` (NOT the displaced
    /// world-space positions, mirroring the Python source).
    pub area: Vec<f64>,
    /// Length-N `1/(count + epsilon)` weights, only populated when
    /// `has_dyn_color`.
    pub face_to_vert_weights: Option<Vec<f64>>,
}

/// Errors that bubble up from the rod-tri offset pre-check (the only
/// blocking error in the assembly path; everything else folds into
/// `AssembleOutput.combined_message`).
#[derive(Debug, thiserror::Error)]
pub enum AssembleError {
    /// Rod-tri offset pre-check found a fatal geometric mismatch.
    #[error(transparent)]
    RodTriOffset(#[from] sb::RodTriOffsetViolation),
}

/// Run the full FixedScene assembly.
///
/// Steps:
///   1. Compute world-space dynamic verts (`local + displacement[dmap]`).
///   2. Run the rod-tri offset pre-check (raises early on hard
///      failures: kernel keeps its current behavior).
///   3. Self-intersection scan over (dynamic + static) tris.
///   4. Contact-offset scan over the unified element namespace.
///   5. Wall + sphere scans, skipping pinned verts.
///   6. Assemble truncated violation previews + human message.
///   7. Compute per-tri area + face-to-vert weights for the
///      successful path. (These two get computed *unconditionally*
///      because the Python source did the same: `area` only gates
///      on triangle count, and the weights gate on `has_dyn_color`.)
pub fn fixed_scene_assemble(input: AssembleInput<'_>) -> Result<AssembleOutput, AssembleError> {
    fixed_scene_assemble_with_collision_filter(input, None, &[])
}

pub fn fixed_scene_assemble_with_collision_filter(input: AssembleInput<'_>, collision_filter: Option<&[u32]>, excluded_object_pairs: &[(i32,i32)]) -> Result<AssembleOutput, AssembleError> {
    fixed_scene_assemble_with_mesh_collision_filter(input,collision_filter,None,None,excluded_object_pairs)
}
pub fn fixed_scene_assemble_with_mesh_collision_filter(input: AssembleInput<'_>, collision_filter: Option<&[u32]>, static_collision_filter: Option<&[u32]>, static_tri_offset: Option<&[f64]>, excluded_object_pairs: &[(i32,i32)]) -> Result<AssembleOutput, AssembleError> {
    if let Some(filter) = collision_filter {
        assert_eq!(filter.len(), input.vert_dmap.len(), "collision_filter length");
    }
    let n_verts = input.vert_local.len() / 3;
    let n_tris = input.tri.len() / 3;
    let n_rods = input.rod.len() / 2;
    debug_assert_eq!(input.vert_dmap.len(), n_verts);
    debug_assert_eq!(input.tri_is_collider.len(), n_tris);
    debug_assert_eq!(input.rod_is_collider.len(), n_rods);

    // Step 1. Dynamic verts.
    let mut dyn_verts: Vec<f64> = vec![0.0; n_verts * 3];
    for i in 0..n_verts {
        let dm = input.vert_dmap[i] as usize;
        for k in 0..3 {
            dyn_verts[3 * i + k] = input.vert_local[3 * i + k] + input.displacement[3 * dm + k];
        }
    }

    let mut out = AssembleOutput::default();

    // Allow Existing Intersections. The opt-in array covers the COMBINED
    // namespace the self-intersection scan uses, so the STATIC collision
    // vertices appended after the dynamic ones read false, and it is `None`
    // when no object opted in, which leaves every check exactly as it was.
    let n_static_verts = match (input.static_verts, input.static_tris) {
        (Some(sv), Some(st)) if !st.is_empty() => sv.len() / 3,
        _ => 0,
    };
    if let Some(mask)=static_collision_filter {assert_eq!(mask.len(),n_static_verts,"static mask length");}
    if let Some(offset)=static_tri_offset {assert_eq!(offset.len(),input.static_tris.map(|x|x.len()/3).unwrap_or(0),"static offsets length");}
    let allow_existing: Option<Vec<bool>> = input.vert_allow_existing.map(|a| {
        assert_eq!(
            a.len(),
            n_verts,
            "vert_allow_existing has {} entries but the scene has {} dynamic vertices",
            a.len(),
            n_verts
        );
        let mut v = a.to_vec();
        v.resize(n_verts + n_static_verts, false);
        v
    });
    let mut links: Option<StartLinks<'_>> = allow_existing.as_deref().map(StartLinks::new);

    // Step 2. Rod-tri offset pre-check (fatal; bubbles up as Err).
    let has_tri_offset = !input.tri_offset.is_empty() && input.tri_offset.iter().any(|&o| o > 0.0);

    if !input.rod_offset.is_empty() && n_rods > 0 {
        let rods_u32: Vec<[u32; 2]> = (0..n_rods)
            .map(|i| [input.rod[2 * i] as u32, input.rod[2 * i + 1] as u32])
            .collect();
        let tris_u32: Vec<[u32; 3]> = (0..n_tris)
            .map(|i| {
                [
                    input.tri[3 * i] as u32,
                    input.tri[3 * i + 1] as u32,
                    input.tri[3 * i + 2] as u32,
                ]
            })
            .collect();
        let tri_off: Vec<f64> = input.tri_offset.to_vec();
        let rod_off: Vec<f64> = input.rod_offset.to_vec();
        let policy = isect::VertexIntersectPolicy::new(
            input.vert_object_id,
            input.vert_group_id,
            input.vert_policy,
            input.vert_pin_allow,
        ).with_collision_filter(collision_filter).with_excluded_objects(excluded_object_pairs);
        let allowed = |ri: usize, ti: usize| {
            !policy.is_inert()
                && policy.tolerated(
                    &policy.side_of(&input.rod[2 * ri..2 * ri + 2]),
                    &policy.side_of(&input.tri[3 * ti..3 * ti + 3]),
                )
        };
        let exempt = |ri: usize, ti: usize| match links.as_mut() {
            Some(links) => links.link_if_covered(&rods_u32[ri], &tris_u32[ti]),
            None => false,
        };
        sb::rod_tri_contact_offset_check(
            &dyn_verts, &rods_u32, &tris_u32, &tri_off, &rod_off, allowed, exempt,
        )
        .map_err(AssembleError::RodTriOffset)?;
    }

    // Step 3. Self-intersection (dynamic + static combined). The body-id
    // array mirrors the collider array: dynamic tris carry their PDRD body
    // id (0 = not rigid), static colliders are appended as 0 (never rigid),
    // so an intra-rigid-body self-intersection is tolerated while rigid-vs-
    // anything-else is still reported. An empty `tri_body_id` (no PDRD
    // bodies in the scene) collapses to all-zeros = the old behavior.
    if n_tris > 0 || ((collision_filter.is_some() || static_collision_filter.is_some()) && n_rods>0 && n_static_verts>0) {
        let dyn_body_id: Vec<i32> = if input.tri_body_id.is_empty() {
            vec![0; n_tris]
        } else {
            input.tri_body_id.to_vec()
        };
        let (combined_verts, combined_tris, combined_is_collider, combined_body_id): (
            Vec<f64>,
            Vec<i32>,
            Vec<bool>,
            Vec<i32>,
        ) = if let (Some(sv), Some(st)) = (input.static_verts, input.static_tris) {
            if !st.is_empty() {
                let n_dyn = n_verts;
                let mut cv = dyn_verts.clone();
                cv.extend_from_slice(sv);
                let mut ct = input.tri.to_vec();
                let mut sti = st.to_vec();
                for v in sti.iter_mut() {
                    *v += n_dyn as i32;
                }
                ct.extend_from_slice(&sti);
                let mut cc = input.tri_is_collider.to_vec();
                let n_static_tri = st.len() / 3;
                cc.extend(std::iter::repeat_n(true, n_static_tri));
                let mut cb = dyn_body_id.clone();
                cb.extend(std::iter::repeat_n(0, n_static_tri));
                (cv, ct, cc, cb)
            } else {
                (
                    dyn_verts.clone(),
                    input.tri.to_vec(),
                    input.tri_is_collider.to_vec(),
                    dyn_body_id.clone(),
                )
            }
        } else {
            (
                dyn_verts.clone(),
                input.tri.to_vec(),
                input.tri_is_collider.to_vec(),
                dyn_body_id.clone(),
            )
        };
        let rod_for_check: Option<Vec<i32>> = if n_rods > 0 {
            Some(input.rod.to_vec())
        } else {
            None
        };
        // Extend the three per-vertex allowance arrays over the appended
        // STATIC vertices. The extension has to happen here rather than in
        // the caller because only this function knows how many static
        // vertices were concatenated, and a length mismatch would index out
        // of the combined namespace.
        let n_combined_verts = combined_verts.len() / 3;
        let extend_object_id = input.vert_object_id.map(|a| {
            let mut v = a.to_vec();
            v.resize(n_combined_verts, isect::NO_OBJECT_ID);
            v
        });
        let extend_group_id = input.vert_group_id.map(|a| {
            let mut v = a.to_vec();
            v.resize(n_combined_verts, isect::NO_GROUP_ID);
            v
        });
        let extend_policy = input.vert_policy.map(|a| {
            let mut v = a.to_vec();
            v.resize(n_combined_verts, 0);
            v
        });
        let extend_pin_allow = input.vert_pin_allow.map(|a| {
            let mut v = a.to_vec();
            v.resize(n_combined_verts, false);
            v
        });
        let extend_collision_filter = if collision_filter.is_some() || static_collision_filter.is_some() {
            let mut values=collision_filter.map(|a|a.to_vec()).unwrap_or_else(||vec![0;n_verts]);
            values.extend(static_collision_filter.map(|a|a.to_vec()).unwrap_or_else(||vec![0;n_combined_verts-n_verts]));
            Some(values)
        } else {None};
        let found = isect::scan_self_intersection_with_collision_filter(isect::IntersectionInput {
            verts: &combined_verts,
            tris: &combined_tris,
            is_collider: Some(&combined_is_collider),
            rod_edges: rod_for_check.as_deref(),
            tri_body_id: Some(&combined_body_id),
            vert_object_id: extend_object_id.as_deref(),
            vert_group_id: extend_group_id.as_deref(),
            vert_policy: extend_policy.as_deref(),
            vert_pin_allow: extend_pin_allow.as_deref(),
        }, extend_collision_filter.as_deref(), excluded_object_pairs);
        // A pair the opt-in covers is linked and leaves the report; every
        // other pair is reported exactly as it is without the option.
        let tri_of = |ti: i32| -> [u32; 3] {
            let t = 3 * ti as usize;
            [
                combined_tris[t] as u32,
                combined_tris[t + 1] as u32,
                combined_tris[t + 2] as u32,
            ]
        };
        // Keep rod index until selecting display samples; the existing payload
        // still represents a rod hit by its intersected triangle.
        let mut pairs: Vec<(i32, i32, Option<usize>)> = Vec::new();
        for &(ti, tj) in &found.tri_tri {
            let linked = links
                .as_mut()
                .is_some_and(|l| l.link_if_covered(&tri_of(ti), &tri_of(tj)));
            if !linked {
                pairs.push((ti, tj, None));
            }
        }
        for &(ri, tj) in &found.rod_tri {
            let rod = [input.rod[2 * ri] as u32, input.rod[2 * ri + 1] as u32];
            let linked = links
                .as_mut()
                .is_some_and(|l| l.link_if_covered(&rod, &tri_of(tj)));
            if !linked {
                pairs.push((-1, tj, Some(ri)));
            }
        }
        if !pairs.is_empty() {
            out.has_self_intersection = true;
            let mut tri_data: Vec<SelfIntersectionEntry> = Vec::new();
            let ids = extend_object_id.as_deref();
            let triangle_owner = |index: i32| {
                let ti = index as usize;
                let fallback = if ti < n_tris {
                    IntersectionOwner::UnknownDynamic
                } else {
                    IntersectionOwner::UnknownStatic
                };
                intersection_owner(ids, &combined_tris[3 * ti..3 * ti + 3], fallback)
            };
            let samples = representative_intersection_indices(
                pairs.len(), SELF_INTERSECTION_DRAW_BUDGET, |i| {
                    let (a, b, rod) = pairs[i];
                    let owner_a = if let Some(ri) = rod {
                        intersection_owner(
                            ids, &input.rod[2 * ri..2 * ri + 2],
                            IntersectionOwner::UnknownDynamic,
                        )
                    } else {
                        triangle_owner(a)
                    };
                    let owner_b = triangle_owner(b);
                    (rod.is_some(), owner_a.min(owner_b), owner_a.max(owner_b))
                },
            );
            for index in samples {
                let (p0, p1, rod) = pairs[index];
                let is_rod = rod.is_some();
                let mut positions: Vec<[f64; 3]> = Vec::new();
                for ti in [p0, p1] {
                    if ti >= 0 {
                        for vi in tri_of(ti) {
                            let vi = vi as usize;
                            positions.push([
                                combined_verts[3 * vi],
                                combined_verts[3 * vi + 1],
                                combined_verts[3 * vi + 2],
                            ]);
                        }
                    }
                }
                tri_data.push(SelfIntersectionEntry {
                    tri_positions: positions,
                    is_rod,
                });
            }
            let mut n_rod_tri = 0;
            let mut n_tri_tri = 0;
            for &(_, _, rod) in &pairs {
                if rod.is_some() {
                    n_rod_tri += 1;
                } else {
                    n_tri_tri += 1;
                }
            }
            out.self_intersections = tri_data;
            out.n_self_intersections_total = pairs.len();
            out.n_tri_tri = n_tri_tri;
            out.n_rod_tri = n_rod_tri;
        }

        // Optional STATIC offset check belongs only to actual filtered scenes.
        // Existing no-mask public callers keep their prior initial policy.
        if let Some(static_offsets)=static_tri_offset {
            let n_combined_tris=combined_tris.len()/3;
            let mut offsets=input.tri_offset.to_vec();offsets.resize(n_tris,0.0);
            offsets.extend_from_slice(static_offsets);offsets.extend_from_slice(input.rod_offset);
            offsets.resize(n_combined_tris+n_rods,0.0);
            let mut colliders=combined_is_collider.clone();colliders.extend_from_slice(input.rod_is_collider);
            let pairs=prox::check_contact_offset_violation_with_collision_filter(prox::ProximityInput {
                verts:&combined_verts,tris:Some(&combined_tris),edges:rod_for_check.as_deref(),
                is_collider:Some(&colliders),contact_offset:Some(&offsets),
                vert_object_id:extend_object_id.as_deref(),vert_group_id:extend_group_id.as_deref(),
                vert_policy:extend_policy.as_deref(),vert_pin_allow:extend_pin_allow.as_deref(),
            },extend_collision_filter.as_deref(),excluded_object_pairs);
            let element=|i:usize| -> Vec<u32> {
                if i<n_combined_tris {combined_tris[3*i..3*i+3].iter().map(|v|*v as u32).collect()}
                else {let r=i-n_combined_tris;input.rod[2*r..2*r+2].iter().map(|v|*v as u32).collect()}
            };
            for (a,b) in pairs {
                let (a,b)=(a as usize,b as usize);
                let is_static=|i:usize| i>=n_tris&&i<n_combined_tris;
                if !is_static(a)&&!is_static(b) {continue;}
                let (va,vb)=(element(a),element(b));
                if links.as_mut().is_some_and(|l|l.link_if_covered(&va,&vb)) {continue;}
                out.has_contact_offset_violation=true;out.n_contact_offset_total+=1;
                if out.contact_offset_pairs.len()<100 {
                    let positions=|v:&[u32]|v.iter().map(|i|{let i=*i as usize;[combined_verts[3*i],combined_verts[3*i+1],combined_verts[3*i+2]]}).collect();
                    out.contact_offset_pairs.push(ContactOffsetEntry {ei_is_triangle:a<n_combined_tris,ej_is_triangle:b<n_combined_tris,ei_pos:positions(&va),ej_pos:positions(&vb)});
                }
            }
        }
    }

    // Step 4. Contact-offset (dynamic only). Any rods at all force the
    // scan via the `n_rods > 0` disjunct (rod offsets only exist when
    // n_rods > 0), so an explicit rod-offset term would be redundant.
    let has_elements = n_tris > 0 || n_rods > 0;
    if has_elements && (has_tri_offset || n_rods > 0) {
        let mut combined_is_collider: Vec<bool> = Vec::with_capacity(n_tris + n_rods);
        combined_is_collider.extend_from_slice(input.tri_is_collider);
        combined_is_collider.extend_from_slice(input.rod_is_collider);

        let tri_off_arr: Vec<f64> = if !input.tri_offset.is_empty() {
            input.tri_offset.to_vec()
        } else {
            vec![0.0; n_tris]
        };
        let rod_off_arr: Vec<f64> = if !input.rod_offset.is_empty() {
            input.rod_offset.to_vec()
        } else {
            vec![0.0; n_rods]
        };
        let mut combined_offset: Vec<f64> = Vec::with_capacity(n_tris + n_rods);
        combined_offset.extend_from_slice(&tri_off_arr);
        combined_offset.extend_from_slice(&rod_off_arr);

        let tris_for_prox: Option<&[i32]> = if n_tris > 0 { Some(input.tri) } else { None };
        let edges_for_prox: Option<&[i32]> = if n_rods > 0 { Some(input.rod) } else { None };

        let pairs = prox::check_contact_offset_violation_with_collision_filter(prox::ProximityInput {
            verts: &dyn_verts,
            tris: tris_for_prox,
            edges: edges_for_prox,
            is_collider: Some(&combined_is_collider),
            contact_offset: Some(&combined_offset),
            vert_object_id: input.vert_object_id,
            vert_group_id: input.vert_group_id,
            vert_policy: input.vert_policy,
            vert_pin_allow: input.vert_pin_allow,
        }, collision_filter, excluded_object_pairs);
        // The proximity namespace is the triangles, then the rod edges.
        let element_of = |e: i32| -> Vec<u32> {
            let e = e as usize;
            if e < n_tris {
                input.tri[3 * e..3 * e + 3].iter().map(|&v| v as u32).collect()
            } else {
                let r = e - n_tris;
                input.rod[2 * r..2 * r + 2].iter().map(|&v| v as u32).collect()
            }
        };
        let pairs: Vec<(i32, i32)> = pairs
            .into_iter()
            .filter(|&(ei, ej)| {
                !links
                    .as_mut()
                    .is_some_and(|l| l.link_if_covered(&element_of(ei), &element_of(ej)))
            })
            .collect();
        if !pairs.is_empty() {
            out.has_contact_offset_violation = true;
            out.n_contact_offset_total += pairs.len();
            for &(ei, ej) in pairs.iter().take(100) {
                let ei_is_tri = (ei as usize) < n_tris;
                let ej_is_tri = (ej as usize) < n_tris;
                let positions = |idx: i32, is_tri: bool| -> Vec<[f64; 3]> {
                    let i = idx as usize;
                    if is_tri {
                        let t = [
                            input.tri[3 * i] as usize,
                            input.tri[3 * i + 1] as usize,
                            input.tri[3 * i + 2] as usize,
                        ];
                        t.iter()
                            .map(|&vi| {
                                [
                                    dyn_verts[3 * vi],
                                    dyn_verts[3 * vi + 1],
                                    dyn_verts[3 * vi + 2],
                                ]
                            })
                            .collect()
                    } else if i < n_tris + n_rods {
                        let li = i - n_tris;
                        let e = [
                            input.rod[2 * li] as usize,
                            input.rod[2 * li + 1] as usize,
                        ];
                        e.iter()
                            .map(|&vi| {
                                [
                                    dyn_verts[3 * vi],
                                    dyn_verts[3 * vi + 1],
                                    dyn_verts[3 * vi + 2],
                                ]
                            })
                            .collect()
                    } else {
                        Vec::new()
                    }
                };
                out.contact_offset_pairs.push(ContactOffsetEntry {
                    ei_is_triangle: ei_is_tri,
                    ej_is_triangle: ej_is_tri,
                    ei_pos: positions(ei, ei_is_tri),
                    ej_pos: positions(ej, ej_is_tri),
                });
            }
        }
    }

    if let Some(links) = links.take() {
        // Positions in the combined namespace the links use: the dynamic
        // vertices, then the static collision vertices after them.
        let position = |v: u32| -> [f64; 3] {
            let v = v as usize;
            if v < n_verts {
                [dyn_verts[3 * v], dyn_verts[3 * v + 1], dyn_verts[3 * v + 2]]
            } else {
                let s = v - n_verts;
                let sv = input.static_verts.expect("a static link implies static vertices");
                [sv[3 * s], sv[3 * s + 1], sv[3 * s + 2]]
            }
        };
        out.start_link_pairs = links
            .recorded_pairs()
            .iter()
            .map(|(a, b)| {
                (
                    a.iter().map(|&v| position(v)).collect(),
                    b.iter().map(|&v| position(v)).collect(),
                )
            })
            .collect();
        out.n_start_link_pairs = links.element_pairs();
        out.start_links = links.into_flat();
    }

    // Step 5. Wall + sphere scans.
    if !input.walls.is_empty() || !input.spheres.is_empty() {
        let mut is_pinned = vec![false; n_verts];
        for &i in input.pinned_vertices {
            if i < n_verts {
                is_pinned[i] = true;
            }
        }
        let mut wall_violations: Vec<(usize, usize, f64)> = Vec::new();
        for (wi, wall) in input.walls.iter().enumerate() {
            // Mirror the Python wrapper's pre-normalize step.
            let nx = wall.normal[0];
            let ny = wall.normal[1];
            let nz = wall.normal[2];
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            let nrm = if len > 0.0 {
                [nx / len, ny / len, nz / len]
            } else {
                [0.0, 1.0, 0.0]
            };
            let res = inv_coll::check_wall_violations(&dyn_verts, &is_pinned, wall.pos, nrm);
            for (vi, signed) in res {
                wall_violations.push((vi, wi, signed));
            }
        }
        if !wall_violations.is_empty() {
            out.has_wall_violation = true;
            out.n_wall_total = wall_violations.len();
            out.first_wall = Some(wall_violations[0]);
            for (vi, wi, signed) in wall_violations.iter().take(100) {
                out.wall_vertices.push(WallVertexEntry {
                    pos: [
                        dyn_verts[3 * vi],
                        dyn_verts[3 * vi + 1],
                        dyn_verts[3 * vi + 2],
                    ],
                    wall_idx: *wi,
                    signed_dist: *signed,
                });
            }
        }

        let mut sphere_violations: Vec<(usize, usize, f64)> = Vec::new();
        for (si, sph) in input.spheres.iter().enumerate() {
            let res = inv_coll::check_sphere_violations(
                &dyn_verts,
                &is_pinned,
                sph.pos,
                sph.radius,
                sph.is_inverted,
                sph.is_hemisphere,
            );
            for (vi, dist) in res {
                sphere_violations.push((vi, si, dist));
            }
        }
        if !sphere_violations.is_empty() {
            out.has_sphere_violation = true;
            out.n_sphere_total = sphere_violations.len();
            out.first_sphere = Some((sphere_violations[0].0, sphere_violations[0].1));
            for (vi, si, dist) in sphere_violations.iter().take(100) {
                out.sphere_vertices.push(SphereVertexEntry {
                    pos: [
                        dyn_verts[3 * vi],
                        dyn_verts[3 * vi + 1],
                        dyn_verts[3 * vi + 2],
                    ],
                    sphere_idx: *si,
                    dist: *dist,
                });
            }
        }
    }

    // Step 6. Compose the human-readable combined message.
    let mut messages: Vec<String> = Vec::new();
    if out.has_self_intersection {
        let mut parts: Vec<String> = Vec::new();
        if out.n_tri_tri > 0 {
            parts.push(format!("{} tri-tri", out.n_tri_tri));
        }
        if out.n_rod_tri > 0 {
            parts.push(format!("{} rod-tri", out.n_rod_tri));
        }
        messages.push(format!(
            "{} self-intersections ({}).",
            out.n_self_intersections_total,
            parts.join(", ")
        ));
    }
    if out.has_contact_offset_violation {
        messages.push(format!(
            "{} element pairs too close.",
            out.n_contact_offset_total
        ));
    }
    if out.has_wall_violation {
        if let Some((vi, wi, signed)) = out.first_wall {
            messages.push(format!(
                "{} vertices violate wall constraints. First: vertex {}, {:.6} units wrong side of wall {}.",
                out.n_wall_total, vi, -signed, wi,
            ));
        }
    }
    if out.has_sphere_violation {
        if let Some((vi, si)) = out.first_sphere {
            messages.push(format!(
                "{} vertices violate sphere constraints. First: vertex {}, sphere {}.",
                out.n_sphere_total, vi, si,
            ));
        }
    }
    out.combined_message = messages.join(" | ");

    // Step 7. Derived data for the success path. The Python source
    // raises ValidationError before reaching the area/weights block,
    // so we only populate them when no flag fired.
    if !out.has_self_intersection
        && !out.has_contact_offset_violation
        && !out.has_wall_violation
        && !out.has_sphere_violation
    {
        // Build the u32 triangle table once and reuse it for both the
        // area pass and the face-to-vert weights. With n_tris == 0 the
        // range is empty, so this is an empty Vec (the has_dyn_color
        // branch relies on that to produce zero-length weights).
        let tris_u32: Vec<[u32; 3]> = (0..n_tris)
            .map(|i| {
                [
                    input.tri[3 * i] as u32,
                    input.tri[3 * i + 1] as u32,
                    input.tri[3 * i + 2] as u32,
                ]
            })
            .collect();
        if n_tris > 0 {
            out.area = sb::triangle_areas(input.vert_local, &tris_u32);
        }
        if input.has_dyn_color {
            out.face_to_vert_weights = Some(sb::face_to_vert_weights(
                n_verts,
                &tris_u32,
                sb::FACE_TO_VERT_WEIGHT_EPS,
            ));
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_input<'a>() -> AssembleInput<'a> {
        AssembleInput {
            vert_dmap: &[],
            vert_local: &[],
            displacement: &[],
            tri: &[],
            rod: &[],
            tri_is_collider: &[],
            tri_body_id: &[],
            rod_is_collider: &[],
            tri_offset: &[],
            rod_offset: &[],
            static_verts: None,
            static_tris: None,
            pinned_vertices: &[],
            walls: &[],
            spheres: &[],
            has_dyn_color: false,
            vert_object_id: None,
            vert_group_id: None,
            vert_policy: None,
            vert_pin_allow: None,
            vert_allow_existing: None,
        }
    }

    #[test]
    fn empty_inputs_yield_no_violations() {
        let out = fixed_scene_assemble(empty_input()).unwrap();
        assert!(!out.has_self_intersection);
        assert!(!out.has_contact_offset_violation);
        assert!(!out.has_wall_violation);
        assert!(!out.has_sphere_violation);
        assert!(out.area.is_empty());
        assert!(out.face_to_vert_weights.is_none());
        assert!(out.combined_message.is_empty());
    }

    #[test]
    fn single_object_no_violations_computes_area() {
        // One unit triangle, no displacement, no walls/spheres.
        let vert_local: [f64; 9] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let vert_dmap: [u32; 3] = [0, 0, 0];
        let displacement: [f64; 3] = [0.0, 0.0, 0.0];
        let tri: [i32; 3] = [0, 1, 2];
        let tri_collider: [bool; 1] = [false];
        let input = AssembleInput {
            vert_dmap: &vert_dmap,
            vert_local: &vert_local,
            displacement: &displacement,
            tri: &tri,
            rod: &[],
            tri_is_collider: &tri_collider,
            tri_body_id: &[],
            rod_is_collider: &[],
            tri_offset: &[],
            rod_offset: &[],
            static_verts: None,
            static_tris: None,
            pinned_vertices: &[],
            walls: &[],
            spheres: &[],
            has_dyn_color: false,
            vert_object_id: None,
            vert_group_id: None,
            vert_policy: None,
            vert_pin_allow: None,
            vert_allow_existing: None,
        };
        let out = fixed_scene_assemble(input).unwrap();
        assert!(!out.has_self_intersection);
        assert_eq!(out.area.len(), 1);
        assert!((out.area[0] - 0.5).abs() < 1e-10);
        assert!(out.face_to_vert_weights.is_none());
    }

    #[test]
    fn dyn_color_populates_weights() {
        // Single triangle, dyn-color on. Each vertex appears in one
        // triangle so the weight is 1 / (1 + 1e-4).
        let vert_local: [f64; 9] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let vert_dmap: [u32; 3] = [0, 0, 0];
        let displacement: [f64; 3] = [0.0, 0.0, 0.0];
        let tri: [i32; 3] = [0, 1, 2];
        let tri_collider: [bool; 1] = [false];
        let input = AssembleInput {
            vert_dmap: &vert_dmap,
            vert_local: &vert_local,
            displacement: &displacement,
            tri: &tri,
            rod: &[],
            tri_is_collider: &tri_collider,
            tri_body_id: &[],
            rod_is_collider: &[],
            tri_offset: &[],
            rod_offset: &[],
            static_verts: None,
            static_tris: None,
            pinned_vertices: &[],
            walls: &[],
            spheres: &[],
            has_dyn_color: true,
            vert_object_id: None,
            vert_group_id: None,
            vert_policy: None,
            vert_pin_allow: None,
            vert_allow_existing: None,
        };
        let out = fixed_scene_assemble(input).unwrap();
        let w = out.face_to_vert_weights.as_ref().unwrap();
        assert_eq!(w.len(), 3);
        let expected = 1.0 / (1.0 + sb::FACE_TO_VERT_WEIGHT_EPS);
        for &v in w {
            assert!((v - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn wall_violation_records_first_and_message() {
        // One vertex below a y=0 floor. Wall normal +Y; signed dist
        // = -0.5. Expected message: "1 vertices violate wall
        // constraints. First: vertex 0, 0.500000 units wrong side of
        // wall 0.".
        let vert_local: [f64; 3] = [0.0, -0.5, 0.0];
        let vert_dmap: [u32; 1] = [0];
        let displacement: [f64; 3] = [0.0, 0.0, 0.0];
        let walls = [WallEntry {
            pos: [0.0, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
        }];
        let input = AssembleInput {
            vert_dmap: &vert_dmap,
            vert_local: &vert_local,
            displacement: &displacement,
            tri: &[],
            rod: &[],
            tri_is_collider: &[],
            tri_body_id: &[],
            rod_is_collider: &[],
            tri_offset: &[],
            rod_offset: &[],
            static_verts: None,
            static_tris: None,
            pinned_vertices: &[],
            walls: &walls,
            spheres: &[],
            has_dyn_color: false,
            vert_object_id: None,
            vert_group_id: None,
            vert_policy: None,
            vert_pin_allow: None,
            vert_allow_existing: None,
        };
        let out = fixed_scene_assemble(input).unwrap();
        assert!(out.has_wall_violation);
        assert_eq!(out.n_wall_total, 1);
        assert!(out
            .combined_message
            .contains("1 vertices violate wall constraints"));
        assert!(out.combined_message.contains("0.500000 units wrong side"));
        // Area path is gated on success; should be empty here.
        assert!(out.area.is_empty());
    }

    #[test]
    fn pinned_vertex_skipped_by_wall_check() {
        // Vertex below floor but pinned: no violation.
        let vert_local: [f64; 3] = [0.0, -0.5, 0.0];
        let vert_dmap: [u32; 1] = [0];
        let displacement: [f64; 3] = [0.0, 0.0, 0.0];
        let walls = [WallEntry {
            pos: [0.0, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
        }];
        let pinned: [usize; 1] = [0];
        let input = AssembleInput {
            vert_dmap: &vert_dmap,
            vert_local: &vert_local,
            displacement: &displacement,
            tri: &[],
            rod: &[],
            tri_is_collider: &[],
            tri_body_id: &[],
            rod_is_collider: &[],
            tri_offset: &[],
            rod_offset: &[],
            static_verts: None,
            static_tris: None,
            pinned_vertices: &pinned,
            walls: &walls,
            spheres: &[],
            has_dyn_color: false,
            vert_object_id: None,
            vert_group_id: None,
            vert_policy: None,
            vert_pin_allow: None,
            vert_allow_existing: None,
        };
        let out = fixed_scene_assemble(input).unwrap();
        assert!(!out.has_wall_violation);
        assert!(out.combined_message.is_empty());
    }

    #[test]
    fn displacement_offsets_dynamic_verts() {
        // Vertex sits at local (0, 1, 0); displacement (0, -2, 0)
        // pulls it to world (0, -1, 0). Wall at y=0 normal +Y must
        // flag it.
        let vert_local: [f64; 3] = [0.0, 1.0, 0.0];
        let vert_dmap: [u32; 1] = [0];
        let displacement: [f64; 3] = [0.0, -2.0, 0.0];
        let walls = [WallEntry {
            pos: [0.0, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
        }];
        let input = AssembleInput {
            vert_dmap: &vert_dmap,
            vert_local: &vert_local,
            displacement: &displacement,
            tri: &[],
            rod: &[],
            tri_is_collider: &[],
            tri_body_id: &[],
            rod_is_collider: &[],
            tri_offset: &[],
            rod_offset: &[],
            static_verts: None,
            static_tris: None,
            pinned_vertices: &[],
            walls: &walls,
            spheres: &[],
            has_dyn_color: false,
            vert_object_id: None,
            vert_group_id: None,
            vert_policy: None,
            vert_pin_allow: None,
            vert_allow_existing: None,
        };
        let out = fixed_scene_assemble(input).unwrap();
        assert!(out.has_wall_violation);
        // World position recorded for the violation viewer should
        // reflect the displacement: y = -1.
        assert!((out.wall_vertices[0].pos[1] - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn sphere_inside_triggers_violation() {
        // Vertex at origin, sphere at origin radius 1 (default
        // outside required), so vertex inside violates.
        let vert_local: [f64; 3] = [0.0, 0.0, 0.0];
        let vert_dmap: [u32; 1] = [0];
        let displacement: [f64; 3] = [0.0, 0.0, 0.0];
        let spheres = [SphereEntry {
            pos: [0.0, 0.0, 0.0],
            radius: 1.0,
            is_inverted: false,
            is_hemisphere: false,
        }];
        let input = AssembleInput {
            vert_dmap: &vert_dmap,
            vert_local: &vert_local,
            displacement: &displacement,
            tri: &[],
            rod: &[],
            tri_is_collider: &[],
            tri_body_id: &[],
            rod_is_collider: &[],
            tri_offset: &[],
            rod_offset: &[],
            static_verts: None,
            static_tris: None,
            pinned_vertices: &[],
            walls: &[],
            spheres: &spheres,
            has_dyn_color: false,
            vert_object_id: None,
            vert_group_id: None,
            vert_policy: None,
            vert_pin_allow: None,
            vert_allow_existing: None,
        };
        let out = fixed_scene_assemble(input).unwrap();
        assert!(out.has_sphere_violation);
        assert_eq!(out.n_sphere_total, 1);
        assert!(out
            .combined_message
            .contains("1 vertices violate sphere constraints"));
    }

    #[test]
    fn self_intersection_pair_detected() {
        // Two triangles intersecting in a "T" pattern. Verts 0-2 are
        // on the y=0 plane; verts 3-5 form a second triangle that
        // pierces the first.
        let vert_local: [f64; 18] = [
            0.0, 0.0, 0.0,
            2.0, 0.0, 0.0,
            1.0, 0.0, 2.0,
            // Second triangle, vertical, crossing the first.
            1.0, -1.0, 1.0,
            1.0, 1.0, 0.5,
            1.0, 1.0, 1.5,
        ];
        let vert_dmap: [u32; 6] = [0; 6];
        let displacement: [f64; 3] = [0.0; 3];
        let tri: [i32; 6] = [0, 1, 2, 3, 4, 5];
        let tri_collider: [bool; 2] = [false, false];
        let input = AssembleInput {
            vert_dmap: &vert_dmap,
            vert_local: &vert_local,
            displacement: &displacement,
            tri: &tri,
            rod: &[],
            tri_is_collider: &tri_collider,
            tri_body_id: &[],
            rod_is_collider: &[],
            tri_offset: &[],
            rod_offset: &[],
            static_verts: None,
            static_tris: None,
            pinned_vertices: &[],
            walls: &[],
            spheres: &[],
            has_dyn_color: false,
            vert_object_id: None,
            vert_group_id: None,
            vert_policy: None,
            vert_pin_allow: None,
            vert_allow_existing: None,
        };
        let out = fixed_scene_assemble(input).unwrap();
        assert!(out.has_self_intersection);
        assert!(out.n_self_intersections_total >= 1);
        assert!(out.combined_message.contains("self-intersections"));
        // Area should NOT be populated when violations exist.
        assert!(out.area.is_empty());
    }

    #[test]
    fn area_matches_reference_for_two_triangles() {
        // Two triangles, areas 0.5 and 1.0.
        let vert_local: [f64; 12] = [
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            2.0, 0.0, 0.0,
        ];
        let vert_dmap: [u32; 4] = [0; 4];
        let displacement: [f64; 3] = [0.0; 3];
        let tri: [i32; 6] = [0, 1, 2, 1, 3, 2];
        let tri_collider: [bool; 2] = [false; 2];
        let input = AssembleInput {
            vert_dmap: &vert_dmap,
            vert_local: &vert_local,
            displacement: &displacement,
            tri: &tri,
            rod: &[],
            tri_is_collider: &tri_collider,
            tri_body_id: &[],
            rod_is_collider: &[],
            tri_offset: &[],
            rod_offset: &[],
            static_verts: None,
            static_tris: None,
            pinned_vertices: &[],
            walls: &[],
            spheres: &[],
            has_dyn_color: false,
            vert_object_id: None,
            vert_group_id: None,
            vert_policy: None,
            vert_pin_allow: None,
            vert_allow_existing: None,
        };
        let out = fixed_scene_assemble(input).unwrap();
        assert_eq!(out.area.len(), 2);
        assert!((out.area[0] - 0.5).abs() < 1e-10);
        // Triangle (1,3,2) is right-angled with legs 1 and 1 -> area 0.5.
        assert!((out.area[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn multi_violation_message_joined_with_pipe() {
        // Wall violation + sphere violation: combined message has
        // both, separated by " | ".
        let vert_local: [f64; 3] = [0.0, -0.5, 0.0];
        let vert_dmap: [u32; 1] = [0];
        let displacement: [f64; 3] = [0.0, 0.0, 0.0];
        let walls = [WallEntry {
            pos: [0.0, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
        }];
        let spheres = [SphereEntry {
            pos: [0.0, -0.5, 0.0],
            radius: 1.0,
            is_inverted: false,
            is_hemisphere: false,
        }];
        let input = AssembleInput {
            vert_dmap: &vert_dmap,
            vert_local: &vert_local,
            displacement: &displacement,
            tri: &[],
            rod: &[],
            tri_is_collider: &[],
            tri_body_id: &[],
            rod_is_collider: &[],
            tri_offset: &[],
            rod_offset: &[],
            static_verts: None,
            static_tris: None,
            pinned_vertices: &[],
            walls: &walls,
            spheres: &spheres,
            has_dyn_color: false,
            vert_object_id: None,
            vert_group_id: None,
            vert_policy: None,
            vert_pin_allow: None,
            vert_allow_existing: None,
        };
        let out = fixed_scene_assemble(input).unwrap();
        assert!(out.has_wall_violation);
        assert!(out.has_sphere_violation);
        assert!(out.combined_message.contains(" | "));
        assert!(out.combined_message.contains("wall"));
        assert!(out.combined_message.contains("sphere"));
    }

    #[test]
    fn rod_only_input_assembles() {
        // No triangles, just one rod edge. Should produce no violations
        // and no per-triangle area.
        let vert_local: [f64; 6] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let vert_dmap: [u32; 2] = [0, 0];
        let displacement: [f64; 3] = [0.0, 0.0, 0.0];
        let rod: [i32; 2] = [0, 1];
        let rod_collider: [bool; 1] = [false];
        let input = AssembleInput {
            vert_dmap: &vert_dmap,
            vert_local: &vert_local,
            displacement: &displacement,
            tri: &[],
            rod: &rod,
            tri_is_collider: &[],
            tri_body_id: &[],
            rod_is_collider: &rod_collider,
            tri_offset: &[],
            rod_offset: &[],
            static_verts: None,
            static_tris: None,
            pinned_vertices: &[],
            walls: &[],
            spheres: &[],
            has_dyn_color: false,
            vert_object_id: None,
            vert_group_id: None,
            vert_policy: None,
            vert_pin_allow: None,
            vert_allow_existing: None,
        };
        let out = fixed_scene_assemble(input).unwrap();
        assert!(!out.has_self_intersection);
        assert!(!out.has_contact_offset_violation);
        assert!(out.area.is_empty(), "no triangles ⇒ no area output");
        assert!(out.combined_message.is_empty());
    }

    #[test]
    fn collider_pair_skipped_in_self_intersection_pass() {
        // Same crossing-T setup as `self_intersection_pair_detected`,
        // but both triangles flagged as colliders so the pair must
        // be skipped (collider × collider rule).
        let vert_local: [f64; 18] = [
            0.0, 0.0, 0.0,
            2.0, 0.0, 0.0,
            1.0, 0.0, 2.0,
            1.0, -1.0, 1.0,
            1.0, 1.0, 0.5,
            1.0, 1.0, 1.5,
        ];
        let vert_dmap: [u32; 6] = [0; 6];
        let displacement: [f64; 3] = [0.0; 3];
        let tri: [i32; 6] = [0, 1, 2, 3, 4, 5];
        let tri_collider: [bool; 2] = [true, true];
        let input = AssembleInput {
            vert_dmap: &vert_dmap,
            vert_local: &vert_local,
            displacement: &displacement,
            tri: &tri,
            rod: &[],
            tri_is_collider: &tri_collider,
            tri_body_id: &[],
            rod_is_collider: &[],
            tri_offset: &[],
            rod_offset: &[],
            static_verts: None,
            static_tris: None,
            pinned_vertices: &[],
            walls: &[],
            spheres: &[],
            has_dyn_color: false,
            vert_object_id: None,
            vert_group_id: None,
            vert_policy: None,
            vert_pin_allow: None,
            vert_allow_existing: None,
        };
        let out = fixed_scene_assemble(input).unwrap();
        assert!(!out.has_self_intersection, "collider × collider must be skipped");
        // No violations ⇒ area path runs, both triangle areas reported.
        assert_eq!(out.area.len(), 2);
    }

    // THE CROSSING "T" of `self_intersection_pair_detected`, as a fixture the
    // Allow Existing Intersections cases below vary.
    const CROSSING_T: [f64; 18] = [
        0.0, 0.0, 0.0,
        2.0, 0.0, 0.0,
        1.0, 0.0, 2.0,
        1.0, -1.0, 1.0,
        1.0, 1.0, 0.5,
        1.0, 1.0, 1.5,
    ];

    fn crossing_input<'a>(allow: Option<&'a [bool]>) -> AssembleInput<'a> {
        AssembleInput {
            vert_dmap: &[0; 6],
            vert_local: &CROSSING_T,
            displacement: &[0.0; 3],
            tri: &[0, 1, 2, 3, 4, 5],
            tri_is_collider: &[false, false],
            vert_allow_existing: allow,
            ..empty_input()
        }
    }

    #[test]
    fn an_opted_in_crossing_is_linked_rather_than_reported() {
        // Only the first triangle's object opted in: either side is enough.
        let allow = [true, true, true, false, false, false];
        let out = fixed_scene_assemble(crossing_input(Some(&allow))).unwrap();
        assert!(!out.has_self_intersection, "{}", out.combined_message);
        assert!(out.combined_message.is_empty());
        assert_eq!(out.n_start_link_pairs, 1);
        // Every vertex of one triangle against every vertex of the other.
        assert_eq!(out.start_links.len(), 2 * 9);
        for u in 0..3u32 {
            for v in 3..6u32 {
                assert!(out.start_links.chunks(2).any(|p| p == [u, v]));
            }
        }
        // The success path runs, so the derived data is there.
        assert_eq!(out.area.len(), 2);
    }

    #[test]
    fn a_crossing_no_side_opted_into_is_still_reported() {
        let allow = [false; 6];
        let out = fixed_scene_assemble(crossing_input(Some(&allow))).unwrap();
        assert!(out.has_self_intersection);
        assert_eq!(out.n_start_link_pairs, 0);
        assert!(out.start_links.is_empty());
        // And with the option absent altogether, the same.
        let out = fixed_scene_assemble(crossing_input(None)).unwrap();
        assert!(out.has_self_intersection);
        assert!(out.start_links.is_empty());
    }

    #[test]
    fn a_dynamic_triangle_through_a_static_collider_links_into_the_combined_namespace() {
        // The first triangle is dynamic; the second is the STATIC collision
        // mesh, appended after the three dynamic vertices.
        let allow = [true; 3];
        let input = AssembleInput {
            vert_dmap: &[0; 3],
            vert_local: &CROSSING_T[..9],
            displacement: &[0.0; 3],
            tri: &[0, 1, 2],
            tri_is_collider: &[false],
            static_verts: Some(&CROSSING_T[9..]),
            static_tris: Some(&[0, 1, 2]),
            vert_allow_existing: Some(&allow),
            ..empty_input()
        };
        let out = fixed_scene_assemble(input).unwrap();
        assert!(!out.has_self_intersection, "{}", out.combined_message);
        assert_eq!(out.n_start_link_pairs, 1);
        // The display record reads the static element's own positions.
        assert_eq!(out.start_link_pairs.len(), 1);
        let (ref dynamic, ref collider) = out.start_link_pairs[0];
        assert_eq!(dynamic[0], [0.0, 0.0, 0.0]);
        assert_eq!(collider[0], [1.0, -1.0, 1.0]);
        // Static vertex s is combined index 3 + s.
        for pair in out.start_links.chunks(2) {
            assert!(pair[0] < 3 && (3..6).contains(&pair[1]), "{pair:?}");
        }
        assert_eq!(out.start_links.len(), 2 * 9);
    }

    // Two parallel triangles 0.01 apart: too close for offsets summing to
    // 0.1, and not crossing.
    const PARALLEL: [f64; 18] = [
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.01,
        1.0, 0.0, 0.01,
        0.0, 1.0, 0.01,
    ];

    #[test]
    fn an_opted_in_contact_offset_violation_is_linked() {
        let make = |allow: Option<&'static [bool]>| AssembleInput {
            vert_dmap: &[0; 6],
            vert_local: &PARALLEL,
            displacement: &[0.0; 3],
            tri: &[0, 1, 2, 3, 4, 5],
            tri_is_collider: &[false, false],
            tri_offset: &[0.05, 0.05],
            vert_allow_existing: allow,
            ..empty_input()
        };
        let out = fixed_scene_assemble(make(None)).unwrap();
        assert!(out.has_contact_offset_violation);
        assert!(!out.has_self_intersection);
        let out = fixed_scene_assemble(make(Some(&[false, false, false, true, true, true]))).unwrap();
        assert!(!out.has_contact_offset_violation, "{}", out.combined_message);
        assert_eq!(out.n_start_link_pairs, 1);
        assert_eq!(out.start_links.len(), 2 * 9);
    }
    #[test] fn collision_masks_control_initial_intersection_and_offsets() {
        let same = [0x00010001;6];
        let disjoint = [0x00010001,0x00010001,0x00010001,0x00020002,0x00020002,0x00020002];
        assert!(fixed_scene_assemble_with_collision_filter(crossing_input(None), Some(&same), &[]).unwrap().has_self_intersection);
        let out = fixed_scene_assemble_with_collision_filter(crossing_input(None), Some(&disjoint), &[]).unwrap();
        assert!(!out.has_self_intersection);
        assert!(out.start_links.is_empty(), "mask filter must not create huge vertex cross products");
        let mut near = crossing_input(None);
        near.tri_offset = &[0.05,0.05];
        let out = fixed_scene_assemble_with_collision_filter(near, Some(&disjoint), &[]).unwrap();
        assert!(!out.has_contact_offset_violation && !out.has_self_intersection);
        assert!(fixed_scene_assemble(crossing_input(None)).unwrap().has_self_intersection);
        let mut connected = crossing_input(None);
        connected.vert_object_id = Some(&[0,0,0,1,1,1]);
        assert!(!fixed_scene_assemble_with_collision_filter(connected, Some(&same), &[(0,1)]).unwrap().has_self_intersection);
    }

    #[test]
    fn collision_filter_static_initial_crossing_and_offset_truth_table() {
        let static_face=[0,1,2];
        for (a,b,admitted) in [(0,0,true),(0x00010001,0x00010001,true),
            (0x00010001,0x00020002,false),(0x00020001,0x00010002,true),
            (0x00030001,0x00020002,false),(1,0x00010001,false),
            (0x00010000,0x00010001,false),(0x00010001,1,false)] {
            let mut input=crossing_input(None);
            input.tri=&input.tri[..3];input.tri_is_collider=&[false];input.tri_body_id=&[];
            input.static_verts=Some(&CROSSING_T[9..]);input.static_tris=Some(&static_face);
            input.tri_offset=&[0.0];
            let output=fixed_scene_assemble_with_mesh_collision_filter(input,Some(&[a;6]),Some(&[b;3]),Some(&[0.0]),&[]).unwrap();
            assert_eq!(output.has_self_intersection,admitted,"initial {a:#x}/{b:#x}");
            let mut near=CROSSING_T[..9].to_vec();for v in near.chunks_exact_mut(3){v[2]+=0.001;}
            let mut input=crossing_input(None);
            input.tri=&input.tri[..3];input.tri_is_collider=&[false];input.tri_body_id=&[];
            input.static_verts=Some(&near);input.static_tris=Some(&static_face);input.tri_offset=&[0.01];
            let output=fixed_scene_assemble_with_mesh_collision_filter(input,Some(&[a;6]),Some(&[b;3]),Some(&[0.01]),&[]).unwrap();
            assert_eq!(output.has_contact_offset_violation,admitted,"offset {a:#x}/{b:#x}");
        }
    }

    #[test]
    fn collision_filter_static_rod_initial_crossing_and_offsets() {
        let static_face=[0,1,2];
        for (a,b,admitted) in [(0x00010001,0x00010001,true),(0x00010001,0x00020002,false)] {
            let mut input=crossing_input(None);input.tri=&[];input.tri_is_collider=&[];input.tri_body_id=&[];input.tri_offset=&[];
            input.rod=&[3,4];input.rod_is_collider=&[false];input.rod_offset=&[0.0];
            input.static_verts=Some(&CROSSING_T[..9]);input.static_tris=Some(&static_face);
            let out=fixed_scene_assemble_with_mesh_collision_filter(input,Some(&[a;6]),Some(&[b;3]),Some(&[0.0]),&[]).unwrap();
            assert_eq!(out.has_self_intersection,admitted);
            let mut near=CROSSING_T[..9].to_vec();for v in near.chunks_exact_mut(3){v[2]+=0.001;}
            let mut input=crossing_input(None);input.tri=&[];input.tri_is_collider=&[];input.tri_body_id=&[];input.tri_offset=&[];
            input.rod=&[0,1];input.rod_is_collider=&[false];input.rod_offset=&[0.01];
            input.static_verts=Some(&near);input.static_tris=Some(&static_face);
            let out=fixed_scene_assemble_with_mesh_collision_filter(input,Some(&[a;6]),Some(&[b;3]),Some(&[0.01]),&[]).unwrap();
            assert_eq!(out.has_contact_offset_violation,admitted);
        }
    }

}
