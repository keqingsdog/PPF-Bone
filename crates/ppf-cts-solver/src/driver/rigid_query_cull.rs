//! CPU-only assembly prefilter over the actual native query and tree leaf boxes.

use ppf_cts_compute::{AllocLabel, Buffer, Device, Fault, Handle};

use crate::data::{DataSet, Vec2u, Vec3u, VertexProp};

use super::contact::MeshRefs;
use super::lbvh::{Aabb, Tree};

const _: () = assert!(std::mem::size_of::<Vec2u>() == std::mem::size_of::<[u32; 2]>());
const _: () = assert!(std::mem::size_of::<Vec3u>() == std::mem::size_of::<[u32; 3]>());

pub(super) fn enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PPF_CPU_RIGID_QUERY_CULL").is_some_and(|v| v == "1"))
}

fn mixed_shell_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PPF_CPU_MIXED_QUERY_CULL")
        .is_some_and(|value| value == "1"))
}

fn trace_enabled() -> bool {
    static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TRACE.get_or_init(|| {
        std::env::var_os("PPF_CPU_RIGID_QUERY_CULL_TRACE").is_some_and(|v| v == "1")
    })
}

pub(super) fn trace_fallback() {
    if trace_enabled() {
        println!("PPF_CPU_RIGID_QUERY_CULL active=false");
    }
}

#[derive(Default)]
pub(super) struct Scratch {
    point: Buffer<Aabb>,
    edge: Buffer<Aabb>,
    prepared_pure_rigid: bool,
    host_props: Vec<VertexProp>,
    host_faces: Vec<[u32; 3]>,
    host_edges: Vec<[u32; 2]>,
    host_nodes: Vec<u32>,
    host_leaves: Vec<Aabb>,
    host_points_query: Vec<Aabb>,
    host_edges_query: Vec<Aabb>,
    host_filtered_points: Vec<Aabb>,
    host_filtered_edges: Vec<Aabb>,
    point_owners: Vec<usize>,
    face_owners: Vec<usize>,
    edge_owners: Vec<usize>,
    leaf_bounds: Vec<Aabb>,
    envelopes: Vec<Aabb>,
    order: Vec<usize>,
    neighbors: Neighbors,
}

// Includes both reusable host storage and the two private output buffers.
// The byte gate also includes the capped flat adjacency.
const MAX_SCRATCH_BYTES: usize = 64 * 1024 * 1024;
const MAX_QUERY_COMPARISONS: usize = 4_000_000;

fn directed_neighbor_cap(bodies: usize) -> Option<usize> {
    Some(bodies.checked_mul(128)?.min(1_000_000))
}

fn reserve_for<T>(values: &mut Vec<T>, len: usize) -> Option<()> {
    let additional = len.saturating_sub(values.len());
    values.try_reserve_exact(additional).ok()
}

fn resize_scratch<T: Copy>(values: &mut Vec<T>, len: usize, value: T) -> Option<()> {
    reserve_for(values, len)?;
    values.resize(len, value);
    Some(())
}

fn allocation<T>(result: Result<T, Fault>) -> Result<Option<T>, Fault> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(Fault::Alloc { .. }) => Ok(None),
        Err(other) => Err(other),
    }
}

macro_rules! eligible {
    ($value:expr) => {
        match $value {
            Some(value) => value,
            None => return Ok(None),
        }
    };
}

fn active_finite(a: &Aabb) -> bool {
    a.active && (0..3).all(|k| a.min[k].is_finite() && a.max[k].is_finite() && a.min[k] <= a.max[k])
}

fn merge(dst: &mut Aabb, b: &Aabb) {
    if !dst.active {
        *dst = *b;
        return;
    }
    for k in 0..3 {
        dst.min[k] = dst.min[k].min(b.min[k]);
        dst.max[k] = dst.max[k].max(b.max[k]);
    }
}

fn intersects(a: &Aabb, b: &Aabb) -> bool {
    a.active && b.active && (0..3).all(|k| a.min[k] <= b.max[k] && b.min[k] <= a.max[k])
}

fn element_owners<const WIDTH: usize>(
    vertices: &[usize],
    elements: &[[u32; WIDTH]],
    output: &mut Vec<usize>,
) -> Option<()> {
    output.clear();
    reserve_for(output, elements.len())?;
    for element in elements {
        // Native PairSide reads the PDRD ownership from the first vertex.
        // Limit space to elements that actually have one proven owner.
        let owner = *vertices.get(element[0] as usize)?;
        if !element
            .iter()
            .all(|i| vertices.get(*i as usize) == Some(&owner))
        {
            return None;
        }
        output.push(owner);
    }
    Some(())
}

fn include_leaf_boxes(
    bounds: &mut [Aabb],
    owners: &[usize],
    nodes: &[u32],
    boxes: &[Aabb],
) -> Option<()> {
    if nodes.len() != 2 * owners.len() || boxes.len() != owners.len() {
        return None;
    }
    for (node, aabb) in nodes.chunks_exact(2).zip(boxes) {
        // This is the same leaf bias read by aabb_leaf_{face,edge,vertex}.
        let primitive = node[0].checked_sub(1)? as usize;
        if node[1] != 0 || !active_finite(aabb) {
            return None;
        }
        merge(bounds.get_mut(*owners.get(primitive)?)?, aabb);
    }
    Some(())
}

fn include_queries(bounds: &mut [Aabb], owners: &[usize], queries: &[Aabb]) -> Option<()> {
    for (&body, aabb) in owners.iter().zip(queries) {
        if !active_finite(aabb) {
            return None;
        }
        merge(bounds.get_mut(body)?, aabb);
    }
    Some(())
}

#[derive(Default)]
struct Neighbors {
    pairs: Vec<(usize, usize)>,
    offsets: Vec<usize>,
}

impl Neighbors {
    fn of(&self, body: usize) -> &[(usize, usize)] {
        &self.pairs[self.offsets[body]..self.offsets[body + 1]]
    }
}

fn build_neighbors(
    envelopes: &[Aabb],
    order: &mut Vec<usize>,
    neighbors: &mut Neighbors,
) -> Option<()> {
    let mut extent = Aabb::default();
    for envelope in envelopes {
        merge(&mut extent, envelope);
    }
    let axis = (0..3).max_by(|&a, &b| {
        (extent.max[a] as f64 - extent.min[a] as f64)
            .total_cmp(&(extent.max[b] as f64 - extent.min[b] as f64))
    })?;
    order.clear();
    reserve_for(order, envelopes.len())?;
    order.extend(0..envelopes.len());
    order.sort_unstable_by(|&a, &b| envelopes[a].min[axis].total_cmp(&envelopes[b].min[axis]));
    neighbors.pairs.clear();
    let mut comparisons_left = envelopes.len().checked_mul(256)?.min(2_000_000);
    let directed_cap = directed_neighbor_cap(envelopes.len())?;
    for (index, &a) in order.iter().enumerate() {
        for &b in &order[index + 1..] {
            comparisons_left = comparisons_left.checked_sub(1)?;
            if envelopes[b].min[axis] > envelopes[a].max[axis] {
                break;
            }
            if intersects(&envelopes[a], &envelopes[b]) {
                let next = neighbors.pairs.len().checked_add(2)?;
                if next > directed_cap {
                    return None;
                }
                if next > neighbors.pairs.capacity() {
                    let grow_to = next
                        .max(neighbors.pairs.capacity().saturating_mul(2))
                        .min(directed_cap);
                    reserve_for(&mut neighbors.pairs, grow_to)?;
                }
                neighbors.pairs.extend([(a, b), (b, a)]);
            }
        }
    }
    neighbors.pairs.sort_unstable();
    resize_scratch(&mut neighbors.offsets, envelopes.len().checked_add(1)?, 0)?;
    let mut next = 0;
    for (body, start) in neighbors.offsets.iter_mut().enumerate() {
        *start = next;
        while neighbors.pairs.get(next).is_some_and(|pair| pair.0 == body) {
            next += 1;
        }
    }
    Some(())
}

fn query_comparisons(
    neighbors: &Neighbors,
    point_owners: &[usize],
    edge_owners: &[usize],
    limit: usize,
    nonrigid_owner: Option<usize>,
) -> Option<usize> {
    let mut work = 0usize;
    for &body in point_owners.iter().chain(edge_owners) {
        if nonrigid_owner == Some(body) { continue; }
        // Each rigid query checks its directional neighbors and, in a
        // mixed shell, the union of actual nonrigid leaf bounds.
        let cost = neighbors.of(body).len().checked_add(usize::from(nonrigid_owner.is_some()))?;
        work = work.checked_add(cost)?;
        if work > limit {
            return None;
        }
    }
    Some(work)
}

fn copy_filtered(
    bounds: &[Aabb],
    neighbors: &Neighbors,
    owners: &[usize],
    queries: &[Aabb],
    scratch: &mut [Aabb],
    nonrigid: Option<(usize, &Aabb)>,
) -> usize {
    let mut skipped = 0;
    for ((&body, input), output) in owners.iter().zip(queries).zip(scratch) {
        *output = *input;
        // Mesh queries retain their original direction. A rigid query
        // intersecting any nonrigid leaf's enclosing box also remains.
        if nonrigid.is_some_and(|(owner, union)| body == owner || intersects(union, input)) {
            continue;
        }
        if !neighbors
            .of(body)
            .iter()
            .any(|&(_, other)| intersects(&bounds[other], input))
        {
            output.active = false;
            skipped += 1;
        }
    }
    skipped
}

struct View<T> {
    ptr: std::ptr::NonNull<T>,
    count: usize,
}

impl<T> View<T> {
    unsafe fn read(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.count) }
    }

    unsafe fn read_or_copy<D: Device>(
        device: &mut D,
        handle: Handle,
        count: usize,
        scratch: &mut Vec<T>,
    ) -> Result<Option<Self>, Fault>
    where
        T: Copy + Default,
    {
        if count == 0 {
            return Ok(Some(Self {
                ptr: std::ptr::NonNull::dangling(),
                count,
            }));
        }
        let bytes = eligible!(count.checked_mul(std::mem::size_of::<T>()));
        if let Some(raw) = unsafe { device.host_view(handle, 0, bytes)? } {
            let ptr = raw.cast::<T>();
            if ptr.as_ptr() as usize % std::mem::align_of::<T>() == 0 {
                return Ok(Some(Self { ptr, count }));
            }
        }
        // The Rust CPU backend serves this exact span through Device::read.
        eligible!(resize_scratch(scratch, count, T::default()));
        let output =
            unsafe { std::slice::from_raw_parts_mut(scratch.as_mut_ptr().cast::<u8>(), bytes) };
        device.read(handle, 0, output)?;
        Ok(Some(Self {
            ptr: eligible!(std::ptr::NonNull::new(scratch.as_mut_ptr())),
            count,
        }))
    }
}

// Host-view pointers remain valid until their own buffer is grown/freed. The
// caller performs this pass between synchronous CPU kernels, volume/query
// views are read-only, and the two writable buffers are new private allocations.
impl Scratch {
    pub(super) fn prepared_pure_rigid(&self) -> bool {
        self.prepared_pure_rigid
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn prepare<D: Device>(
        &mut self,
        device: &mut D,
        data: &DataSet,
        mesh: MeshRefs,
        trees: [&mut Tree; 3],
        point: &Buffer<Aabb>,
        edge: &Buffer<Aabb>,
    ) -> Result<Option<(Handle, Handle)>, Fault> {
        self.prepared_pure_rigid = false;
        let start = trace_enabled().then(std::time::Instant::now);
        let bodies = data.prop.pdrd_body.size as usize;
        let verts = data.prop.vertex.size as usize;
        let faces = data.mesh.mesh.face.size as usize;
        let edges_count = edge.len();
        if faces == 0
            || edges_count == 0
            || bodies == 0
            || bodies > verts
            || verts != point.len()
            || data.mesh.mesh.tet.size != 0
            || data.rod_count != 0
            || edges_count != data.mesh.mesh.edge.size as usize
        {
            return Ok(None);
        }
        // Charge every geometry-dependent buffer, including the flat
        // adjacency (planned capacity or any retained capacity, whichever
        // is larger). The per-body charge includes its CSR row offset.
        let mixed_shell = mixed_shell_enabled();
        let envelope_capacity = eligible!(bodies.checked_add(usize::from(mixed_shell)));
        let neighbor_capacity =
            eligible!(directed_neighbor_cap(bodies)).max(self.neighbors.pairs.capacity());
        let neighbor_bytes =
            eligible!(neighbor_capacity.checked_mul(std::mem::size_of::<(usize, usize)>()));
        let total = verts as u64 * 220
            + edges_count as u64 * 144
            + faces as u64 * 64
            + envelope_capacity as u64 * 96
            + neighbor_bytes as u64
            + std::mem::size_of::<usize>() as u64;
        if total > MAX_SCRATCH_BYTES as u64 {
            return Ok(None);
        }
        let props = eligible!(unsafe {
            View::<VertexProp>::read_or_copy(device, mesh.vertex_prop, verts, &mut self.host_props)?
        });
        self.point_owners.clear();
        eligible!(reserve_for(&mut self.point_owners, verts));
        let mut nonrigid_owner = None;
        for p in unsafe { props.read() } {
            if p.collider || p.collision_filter != 0 || p.intersect_policy != 0 || p.pin_allow_intersection {
                return Ok(None);
            }
            let owner = match p.pdrd_body_index.checked_sub(1) {
                Some(body) if (body as usize) < bodies => body as usize,
                None if mixed_shell => {
                    nonrigid_owner = Some(bodies);
                    bodies
                }
                _ => return Ok(None),
            };
            self.point_owners.push(owner);
        }
        let face_elements = eligible!(unsafe {
            View::<[u32; 3]>::read_or_copy(device, mesh.face, faces, &mut self.host_faces)?
        });
        let edge_elements = eligible!(unsafe {
            View::<[u32; 2]>::read_or_copy(device, mesh.edge, edges_count, &mut self.host_edges)?
        });
        eligible!(element_owners(
            &self.point_owners,
            unsafe { face_elements.read() },
            &mut self.face_owners
        ));
        eligible!(element_owners(
            &self.point_owners,
            unsafe { edge_elements.read() },
            &mut self.edge_owners
        ));

        self.leaf_bounds.clear();
        eligible!(resize_scratch(
            &mut self.leaf_bounds,
            eligible!(bodies.checked_add(usize::from(nonrigid_owner.is_some()))),
            Aabb::default()
        ));
        for (tree, owners) in trees.into_iter().zip([
            self.point_owners.as_slice(),
            self.edge_owners.as_slice(),
            self.face_owners.as_slice(),
        ]) {
            let count = tree.primitive_count() as usize;
            if count != owners.len() {
                return Ok(None);
            }
            let node_count = eligible!(count.checked_mul(2));
            let nodes = eligible!(unsafe {
                View::<u32>::read_or_copy(
                    device,
                    tree.node.handle(),
                    node_count,
                    &mut self.host_nodes,
                )?
            });
            let leaf = eligible!(unsafe {
                View::<Aabb>::read_or_copy(
                    device,
                    tree.aabb.handle(),
                    count,
                    &mut self.host_leaves,
                )?
            });
            eligible!(include_leaf_boxes(
                &mut self.leaf_bounds,
                owners,
                unsafe { nodes.read() },
                unsafe { leaf.read() }
            ));
        }
        if self.leaf_bounds.iter().any(|b| !b.active) {
            return Ok(None);
        }
        let points = eligible!(unsafe {
            View::<Aabb>::read_or_copy(
                device,
                point.handle(),
                point.len(),
                &mut self.host_points_query,
            )?
        });
        let edges = eligible!(unsafe {
            View::<Aabb>::read_or_copy(
                device,
                edge.handle(),
                edges_count,
                &mut self.host_edges_query,
            )?
        });
        self.envelopes.clear();
        eligible!(reserve_for(&mut self.envelopes, self.leaf_bounds.len()));
        self.envelopes.extend_from_slice(&self.leaf_bounds);
        // Queries may exceed the body's own swept-leaf union.
        eligible!(include_queries(
            &mut self.envelopes,
            &self.point_owners,
            unsafe { points.read() }
        ));
        eligible!(include_queries(
            &mut self.envelopes,
            &self.edge_owners,
            unsafe { edges.read() }
        ));
        eligible!(build_neighbors(
            &self.envelopes[..bodies],
            &mut self.order,
            &mut self.neighbors
        ));

        let query_rows = eligible!(point.len().checked_add(edges_count));
        let query_limit = eligible!(query_rows.checked_mul(64)).min(MAX_QUERY_COMPARISONS);
        eligible!(query_comparisons(
            &self.neighbors,
            &self.point_owners,
            &self.edge_owners,
            query_limit,
            nonrigid_owner,
        ));
        let nonrigid = nonrigid_owner.map(|owner| (owner, &self.leaf_bounds[owner]));

        eligible!(allocation(self.point.size(
            device,
            point.len(),
            AllocLabel("contact.rigid_cull.point")
        ))?);
        eligible!(allocation(self.edge.size(
            device,
            edges_count,
            AllocLabel("contact.rigid_cull.edge")
        ))?);
        eligible!(resize_scratch(
            &mut self.host_filtered_points,
            point.len(),
            Aabb::default()
        ));
        eligible!(resize_scratch(
            &mut self.host_filtered_edges,
            edges_count,
            Aabb::default()
        ));
        let skip_point = copy_filtered(
            &self.leaf_bounds,
            &self.neighbors,
            &self.point_owners,
            unsafe { points.read() },
            &mut self.host_filtered_points,
            nonrigid,
        );
        let skip_edge = copy_filtered(
            &self.leaf_bounds,
            &self.neighbors,
            &self.edge_owners,
            unsafe { edges.read() },
            &mut self.host_filtered_edges,
            nonrigid,
        );
        self.point.write(device, 0, &self.host_filtered_points)?;
        self.edge.write(device, 0, &self.host_filtered_edges)?;
        if let Some(start) = start {
            println!("PPF_CPU_RIGID_QUERY_CULL active=true points={} skipped_points={skip_point} edges={} skipped_edges={skip_edge} prep_ms={:.6} mixed_shell={}",
                     point.len(), edges_count, start.elapsed().as_secs_f64() * 1000., nonrigid_owner.is_some());
        }
        // Publish only after both dedicated query buffers are ready.
        self.prepared_pure_rigid = nonrigid_owner.is_none();
        Ok(Some((self.point.handle(), self.edge.handle())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(left: f32, right: f32) -> Aabb {
        Aabb {
            min: [left, 0., 0.],
            max: [right, 1., 1.],
            active: true,
        }
    }

    #[test]
    fn preserve_touching_and_outside_own_leaf_query() {
        let bounds = vec![b(0., 1.), b(2., 3.), b(20., 21.)];
        let queries = vec![b(2., 2.25), b(2.2, 2.3), b(20.5, 20.6)];
        let owners = [0, 1, 2];
        let mut envelopes = bounds.clone();
        include_queries(&mut envelopes, &owners, &queries).unwrap();
        let mut neighbors = Neighbors::default();
        build_neighbors(&envelopes, &mut Vec::new(), &mut neighbors).unwrap();
        let mut output = vec![Aabb::default(); queries.len()];
        assert_eq!(
            copy_filtered(&bounds, &neighbors, &owners, &queries, &mut output, None),
            2
        );
        assert_eq!(output[0], queries[0]);
        assert!(!output[1].active && !output[2].active);
        assert_eq!(queries.iter().filter(|q| q.active).count(), 3);
    }

    #[test]
    fn dense_topology_exceeds_neighbor_budget_and_falls_back() {
        let dense = vec![b(0., 1.); 130];
        assert!(build_neighbors(&dense, &mut Vec::new(), &mut Neighbors::default()).is_none());
    }

    #[test]
    fn sweep_budget_counts_misses_on_secondary_axes() {
        let slabs: Vec<_> = (0..600)
            .map(|i| Aabb {
                min: [0., i as f32, 0.],
                max: [1000., i as f32 + 0.25, 1.],
                active: true,
            })
            .collect();
        assert!(build_neighbors(&slabs, &mut Vec::new(), &mut Neighbors::default()).is_none());
    }

    #[test]
    fn query_budget_counts_point_and_edge_rows_per_body() {
        let neighbors = Neighbors {
            pairs: vec![(0, 1), (0, 2), (1, 0), (2, 0)],
            offsets: vec![0, 2, 3, 4],
        };
        assert_eq!(
            query_comparisons(&neighbors, &[0, 0, 0], &[1, 1], 8, None),
            Some(8)
        );
        assert_eq!(query_comparisons(&neighbors, &[0, 0, 0], &[1, 1], 7, None), None);
        assert_eq!(query_comparisons(&neighbors, &[], &[], 0, None), Some(0));
    }

    #[test]
    fn retain_nonrigid_queries_and_boundary_touching_rigids() {
        let bounds = [b(0., 1.), b(3., 4.), b(7., 8.), b(4., 6.)];
        let queries = [b(0., 1.), b(3., 4.), b(7., 8.), b(5., 6.)];
        let owners = [0, 1, 2, 3];
        let mut neighbors = Neighbors::default();
        build_neighbors(&bounds[..3], &mut Vec::new(), &mut neighbors).unwrap();
        let mut output = [Aabb::default(); 4];
        let protected = Some((3, &bounds[3]));
        assert_eq!(copy_filtered(&bounds, &neighbors, &owners, &queries, &mut output, protected), 2);
        assert!(!output[0].active && !output[2].active);
        assert_eq!(output[1], queries[1]);
        assert_eq!(output[3], queries[3]);
        assert_eq!(query_comparisons(&neighbors, &owners, &[], 3, Some(3)), Some(3));
        assert_eq!(query_comparisons(&neighbors, &owners, &[], 2, Some(3)), None);
    }

    #[test]
    fn a_mixed_owner_element_uses_the_original_path() {
        let mut owners = Vec::new();
        assert_eq!(element_owners(&[0, 0, 2], &[[0, 1], [1, 2]], &mut owners), None);
        assert_eq!(element_owners(&[0, 2, 2], &[[0, 1, 2]], &mut owners), None);
    }

    #[test]
    fn allocation_fallback_never_swallows_platform_fault() {
        let failure: Result<(), _> = Err(Fault::Platform {
            call: "injected read/write",
            detail: "failed".into(),
        });
        assert!(matches!(allocation(failure), Err(Fault::Platform { .. })));
        assert!(allocation::<()>(Err(Fault::Alloc {
            label: AllocLabel("test"),
            bytes: 16,
            detail: "exhausted".into()
        }))
        .unwrap()
        .is_none());
    }

    #[test]
    fn refuse_mixed_primitives_nonfinite_or_inactive_bounds() {
        assert!(element_owners(&[0, 1, 0], &[[0, 1, 2]], &mut Vec::new()).is_none());
        assert!(
            include_leaf_boxes(&mut [Aabb::default()], &[0], &[1, 0], &[b(f32::NAN, 1.)]).is_none()
        );
        assert!(
            include_leaf_boxes(&mut [Aabb::default()], &[0], &[1, 0], &[Aabb::default()]).is_none()
        );
        assert!(include_leaf_boxes(&mut [Aabb::default()], &[0], &[0, 2], &[b(0., 1.)]).is_none());
    }
}
