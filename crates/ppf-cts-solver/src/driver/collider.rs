// Fork modification: changed from the upstream baseline.
// File: crates/ppf-cts-solver/src/driver/collider.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The analytic colliders: a sphere and a floor, per vertex.
//!
//! # Why this is not part of the contact subsystem
//!
//! An analytic collider is not a mesh pair. It needs no bounding hierarchy, no
//! candidate list and no dynamic sparsity, because every vertex is tested
//! against every collider directly, and the Hessian it contributes lands on that
//! vertex's own diagonal block, which the fixed pattern always carries. So it
//! runs even on a scene that set `disable-contact`, and it is owned here
//! rather than by [`super::contact::Contact`], which a `disable-contact` scene
//! does not allocate at all.
//!
//! # The split
//!
//! | shared C++ | here, in Rust |
//! |---|---|
//! | the sphere and floor shape, the barrier, the dynamic stiffness, friction (`entrypoints/shim_collider.cpp` over `analytic_contact_evaluate`) | which vertices, in what order, and where each result is scattered |
//!
//! # The scatter is serial
//!
//! Evaluation is per vertex and writes only that vertex's staging slots, so it
//! is cut into parallel ranges. The deposit is serial and in ascending vertex
//! order, which is what makes the assembled sum independent of the thread count:
//! `compute::atomic_add` on the host seam is a plain read, add and write back.
//!
//! # Two results come out of the sweep, and only one of them is a time
//!
//! [`Analytic::line_search`] returns the time of impact AND the lowest-indexed
//! fix-pinned vertex whose prescribed path crosses a collider. A prescribed
//! vertex has no freedom to yield, so clamping the time would stall the solve
//! without preventing the crossing; the step reports it and fails instead.

use crate::data::{DataSet, Floor, ParamSet, Sphere};

use ppf_cts_compute::{
    AllocLabel, Buffer, Device, EncoderExt, Fault, Pod, ReadbackBuffer, StagedBuffer,
};
use super::fixedcsr::FixedCsr;
use super::kernels::{
    VecFillArgs, VertexConstraintArgs, VertexConstraintSweepArgs,
};
use super::scene::{Fatal, FatalResult};



/// The analytic collider layer's per-run state.
pub struct Analytic {
    /// The device fold's levels; see [`super::reduce::DeviceFold`].
    fold: super::reduce::DeviceFold,
    /// How many analytic contacts each surface vertex carries.
    ///
    /// TELEMETRY, and no host code reads the array: the status channel reports
    /// its sum, which [`Self::count_total`] reduces on the device in the submit
    /// that writes the tally, so the per-vertex counts never cross the seam.
    count: Buffer<u32>,
    /// The `u64` total of `count`, carried in two words so it cannot wrap.
    count_total: super::reduce::DeviceWordFold,
    /// Non-zero is what makes a vertex a grain.
    grain_inv_inertia: StagedBuffer<f32>,
    /// The sweep's per-vertex outputs.
    toi: ReadbackBuffer<f32>,
    /// One `u32` per vertex, as `active` above and for the same reason.
    infeasible: ReadbackBuffer<u32>,
    surface_vertices: usize,
    // Lazy, retained scratch for actual accepted-coordinate plane certification.
    accept_proposal: Buffer<f32>,
    accept_candidate: ReadbackBuffer<f32>,
    accept_mask: ReadbackBuffer<u32>,
    accept_bit_difference: ReadbackBuffer<u32>,
    // Optional acceptance-failure observer; no allocation/copy when unset.
    accept_failure_unsafe: Buffer<f32>,
    accept_failure_ledger: Option<serde_json::Value>,
    /// How many analytic contacts the last assembly deposited.
    pub assembled: u64,
}

/// What the swept pin and collider test found.
pub struct Sweep {
    /// The smallest time of impact over every surface vertex, in the line
    /// search's own units.
    pub time_of_impact: f32,
    /// The lowest-indexed fix-pinned vertex driven through a collider it cannot
    /// yield to, if any.
    pub infeasible_pin: Option<u32>,
}

impl Sweep {
    /// The verdict a sweep that was not run must report: the whole span is
    /// available and no pin is infeasible.
    ///
    /// `disable-contact` gates the rigidify commit's sweep entirely
    /// (`super::step`), and a caller still has to fold a time of impact into
    /// its ratio. Returning the span rather than 1.0 keeps that fold in the line
    /// search's own units, which is what the caller divides by.
    pub fn unobstructed(span: f32) -> Self {
        Sweep {
            time_of_impact: span,
            infeasible_pin: None,
        }
    }
}

fn staged<T: Pod + Default>(
    device: &mut impl Device,
    count: usize,
    label: &'static str,
) -> Result<StagedBuffer<T>, Fault> {
    let mut buffer = StagedBuffer::default();
    buffer.size(device, count, AllocLabel(label))?;
    Ok(buffer)
}

/// The mirror sibling of [`staged`], for an array the KERNEL writes and the host
/// reads back.
fn readback<T: Pod + Default>(
    device: &mut impl Device,
    count: usize,
    label: &'static str,
) -> Result<ReadbackBuffer<T>, Fault> {
    let mut buffer = ReadbackBuffer::default();
    buffer.size(device, count, AllocLabel(label))?;
    Ok(buffer)
}

impl Analytic {
    // Diagnostic only: re-read the existing output, with no launch or state mutation.
    pub(super) fn diagnostic_toi<D: Device>(&mut self, device: &mut D) -> Result<Vec<f32>, Fault> {
        self.toi.download(device)?;
        Ok(self.toi.host()[..self.surface_vertices].to_vec())
    }

    /// Size every buffer for this scene, once, at `initialize()`.
    ///
    /// TAKES THE DEVICE because two of the arrays it sizes are device
    /// allocations: the compacted scatter inputs the deposit fills and the
    /// force embed reads. The rest are the host's own working arrays.
    ///
    /// # Safety
    /// `data` must address a live `DataSet`.
    pub unsafe fn allocate(device: &mut impl Device, data: &DataSet) -> FatalResult<Self> {
        let vertices = data.vertex.curr.size as usize;
        let surface_vertices = data.surface_vert_count as usize;
        if surface_vertices > vertices {
            return Err(Fatal::invariant(format!(
                "solver driver: the scene declares {surface_vertices} contact vertices over a \
                 vertex array of {vertices}. The contact vertices are a PREFIX of that array, \
                 so a longer prefix does not describe this mesh"
            )));
        }
        Ok(Analytic {
            fold: super::reduce::DeviceFold::default(),
            count: {
                let mut buffer = Buffer::default();
                buffer.size(device, surface_vertices, AllocLabel("collider.count"))?;
                buffer
            },
            count_total: {
                // SIZED ONCE, for the one tally every assembly reduces.
                let mut total = super::reduce::DeviceWordFold::default();
                if surface_vertices > 0 {
                    total.size_sum(device, surface_vertices as u32)?;
                }
                total
            },
            grain_inv_inertia: {
                // Non-zero is what makes a vertex a grain; a scene with no SAND
                // stages zeros and every grain branch reads false.
                let mut buffer = staged(device, surface_vertices, "grain.inv_inertia")?;
                let inertia = (data.grain_inv_inertia.size as usize).min(surface_vertices);
                if inertia > 0 {
                    buffer.at()[..inertia].copy_from_slice(
                        std::slice::from_raw_parts(data.grain_inv_inertia.data, inertia),
                    );
                }
                buffer.upload(device)?;
                buffer
            },
            toi: readback(device, surface_vertices, "collider.toi")?,
            infeasible: readback(device, surface_vertices, "collider.infeasible")?,
            surface_vertices,
            accept_proposal: Buffer::default(),
            accept_candidate: ReadbackBuffer::default(),
            accept_mask: ReadbackBuffer::default(),
            accept_bit_difference: ReadbackBuffer::default(),
            accept_failure_unsafe: Buffer::default(),
            accept_failure_ledger: None,
            assembled: 0,
        })
    }

    /// One Newton iteration's analytic constraint assembly.
    ///
    /// `reference` is the ELASTIC SNAPSHOT the dynamic stiffness reads, which is
    /// `tmp_fixed` and never the matrix being written.
    ///
    /// # Safety
    /// `data` and `param` must be live and every slice sized for the scene.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn assemble<D: Device>(
        &mut self,
        device: &mut D,
        _data: &DataSet,
        mesh: super::contact::MeshRefs,
        param: *const ParamSet,
        assembly_dt: f32,
        x0: ppf_cts_compute::Handle,
        x: ppf_cts_compute::Handle,
        pins: ppf_cts_compute::Handle,
        spheres: &StagedBuffer<Sphere>,
        floors: &StagedBuffer<Floor>,
        reference: &mut FixedCsr<'_>,
        _fixed: &mut FixedCsr<'_>,
        _staging: &mut super::state::PushStaging,
        force: ppf_cts_compute::Handle,
        // `C`, the operator's block diagonal, which is where this barrier's
        // Hessian is deposited and where `Operator` reads it from. It is NOT
        // the fixed pattern: a per-vertex 3x3 has a slot waiting for it here,
        // while placing it in the fixed pattern would need the row's slot
        // looked up on the host.
        diagonal: ppf_cts_compute::Handle,
        // THE THREE PER-VERTEX SCHUR BLOCKS A GRAIN'S SPIN IS CONDENSED OUT OF,
        // owned by `SolverState` and passed in rather than held here.
        //
        // They are per-VERTEX quantities: the condense, the recover and the
        // post-solve integrate all run over the whole vertex array, while this
        // pass covers the surface prefix. A copy owned here would be only
        // `9 * surface_vertices` long, and the integrate reads `angular[i]` for
        // every `i` up to the vertex count, so it would run off the end; it
        // would also be absent in a scene with no analytic collider, leaving
        // the call site nothing but `Handle::NONE` to pass. `SolverState` sizes
        // all three per vertex with the scene and zeroes them per iteration, so
        // they are present and correctly sized whether or not anything writes
        // them.
        grain_angular: ppf_cts_compute::Handle,
        grain_coupling: ppf_cts_compute::Handle,
        grain_rotational: ppf_cts_compute::Handle,
        // THE RESIDUAL EVERY FRICTION TERM ANCHORS ITSELF ON, a COPY of the
        // force vector taken before this pass. This entry writes only its own
        // vertex's row, but the three collision-mesh passes that follow read
        // the same array and deposit through atomics across vertices, so one
        // snapshot serves them all and none of them reads a row another has
        // already moved. It holds the momentum, elastic and strain-limit terms.
        residual: ppf_cts_compute::Handle,
        statistics: super::contact::StatisticsRefs,
    ) -> FatalResult<()> {
        self.assembled = 0;
        if self.surface_vertices == 0 {
            return Ok(());
        }

        let pattern = reference.pattern();
        // A REFERENCE THAT NAMES NOTHING is how an absent array arrives, which
        // the shim spelled as a null pointer. Every read of one is behind a
        // guard the scene decides: `fix_pair` behind `fix_index > 0`, and the
        // two collider arrays behind their own counts.
        let args = VertexConstraintArgs {
            // An analytic primitive belongs to no object, so only the dynamic
            // side is charged and no static map is named here.
            statistics_contact_count: statistics.contact_count,
            statistics_contact_count_size: statistics.contact_count_size,
            statistics_object_index: statistics.object_index,
            statistics_object_index_size: statistics.object_index_size,
            eval_x: x,
            current: x0,
            vertex_prop: mesh.vertex_prop,
            vertex_param: mesh.vertex_param,
            fix_pair: pins,
            sphere: spheres.handle(),
            sphere_count: spheres.len() as u32,
            floor: floors.handle(),
            floor_count: floors.len() as u32,
            fixed_index: mesh.fixed_index,
            fixed_offset: mesh.fixed_offset,
            fixed_value: reference.value_handle(),
            row_count: pattern.rows,
            disable_pin_dof_removal: u32::from((*param).disable_pin_dof_removal),
            constraint_tol: (*param).constraint_tol,
            friction_mode: (*param).friction_mode as u32,
            friction_eps: (*param).friction_eps,
            residual,
            dt: assembly_dt,
            force,
            diagonal,
            out_count: self.count.handle(),
            grain_inv_inertia: self.grain_inv_inertia.handle(),
            out_grain_angular: grain_angular,
            out_grain_coupling: grain_coupling,
            out_grain_rotational: grain_rotational,
            count: self.surface_vertices as u32,
            seam_arena_count: 0,
        };
        let span = self.surface_vertices;
        let counts = self.count.handle();
        let total = &self.count_total;
        // THE CUT AND THE DIAGNOSTIC RECORDS ARE THE BACKEND'S. This site states
        // the kernel and the extent; the per-chunk `ChunkDiag` records and
        // the ascending first-writer merge over them are the seam's transport,
        // and a failing check comes back as `Fault::Device` naming the file and
        // line.
        //
        // THE TALLY'S TOTAL RIDES THE SAME SUBMIT. It reads what the constraint
        // pass has just written, so it needs no boundary of its own, and only
        // its two result words are read once the region ends.
        device
            .run("collider.constraint", |encoder| {
                encoder.elements(&args, span as u32)?;
                total.encode_sum(encoder, counts, span as u32)
            })
            .map_err(|fault| match fault {
                Fault::Device { diag, .. } => Fatal::device_assert(format!(
                    "solver driver: an analytic collider contact was evaluated outside the domain \
                     the barrier is defined on. The check failed {} time(s){}. A vertex is \
                     already at or past the collider's surface, so the gap the stiffness divides \
                     by is not positive",
                    diag.failures,
                    diag.first
                        .as_ref()
                        .map_or(String::new(), |first| format!(", first at {first}"))
                )),
                other => Fatal::from(other),
            })?;

        // THE TELEMETRY TOTAL, the one thing this pass reads back: two words, the
        // low and high halves of the `u64` the status channel reports.
        self.assembled = self.count_total.read_sum(device)?;
        Ok(())
    }

    /// The swept pin and collider test over this step's candidate motion.
    ///
    /// # Safety
    /// `data` and `param` must be live and both position arrays sized for the
    /// scene.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn line_search<D: Device>(
        &mut self,
        device: &mut D,
        _data: &DataSet,
        mesh: super::contact::MeshRefs,
        param: *const ParamSet,
        x0: ppf_cts_compute::Handle,
        x1: ppf_cts_compute::Handle,
        pins: ppf_cts_compute::Handle,
        spheres: &StagedBuffer<Sphere>,
        floors: &StagedBuffer<Floor>,
    ) -> FatalResult<Sweep> {
        let max_t = (*param).line_search_max_t;
        if self.surface_vertices == 0 {
            return Ok(Sweep {
                time_of_impact: max_t,
                infeasible_pin: None,
            });
        }
        // SEEDED AT THE UNFILTERED LENGTH, the whole line-search span. The body
        // only ever lowers it.
        // THE SEED, on the device. The sweep kernel MIN-REDUCES into this
        // array rather than writing it, so every slot has to open at the
        // line-search ceiling; a host `fill` cannot reach a device allocation,
        // which is what `vec_fill` is dispatched for.
        let seed = VecFillArgs {
            array: self.toi.handle(),
            value: max_t,
            count: self.surface_vertices as u32,
            seam_arena_count: 0,
        };
        let args = VertexConstraintSweepArgs {
            x0: x0,
            x1: x1,
            vertex_prop: mesh.vertex_prop,
            fix_pair: pins,
            sphere: spheres.handle(),
            sphere_count: spheres.len() as u32,
            floor: floors.handle(),
            floor_count: floors.len() as u32,
            disable_pin_dof_removal: u32::from((*param).disable_pin_dof_removal),
            ccd_eps: (*param).ccd_eps,
            line_search_max_t: (*param).line_search_max_t,
            out_toi: self.toi.handle(),
            out_infeasible: self.infeasible.handle(),
            count: self.surface_vertices as u32,
            seam_arena_count: 0,
        };
        let span = self.surface_vertices;
        let count = span as u32;
        self.fold.prepare(device, count)?;
        device
            .run("collider.sweep", |encoder| {
                encoder.elements(&seed, count)?;
                encoder.elements(&args, count)?;
                // Only scratch is written. The invariant check below still
                // runs before the host consumes either part of the verdict.
                self.fold.encode_min(encoder, self.toi.handle(), count)
            })
            .map_err(|fault| match fault {
                Fault::Device { diag, .. } => Fatal::device_assert(format!(
                    "solver driver: the analytic collider sweep tripped an invariant {} time(s){}. \
                     The last payload entry names which check: 0 is a barrier-held pin that has \
                     left the ball its gap is measured in, and 1 and 2 are a sphere and a floor \
                     granting a non-positive time of impact, which means the vertex is already \
                     at the collider's surface. No step length can be trusted from this sweep",
                    diag.failures,
                    diag.first
                        .as_ref()
                        .map_or(String::new(), |first| format!(", first at {first}"))
                )),
                other => Fatal::from(other),
            })?;
        // THE MIRROR, refreshed before the verdict is read out of it. The
        // kernel clears every slot itself and sets only the infeasible ones, so
        // the download is the whole answer rather than a merge.
        self.infeasible.download(device)?;
        Ok(Sweep {
            time_of_impact: self.fold.read_min(device, count, max_t)?,
            // FIRST BY INDEX, not by whichever thread arrived first, so two runs
            // of one scene name the same vertex.
            infeasible_pin: self.infeasible.host()[..span]
                .iter()
                .position(|flag| *flag != 0)
                .map(|i| i as u32),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::state::PatternDevice;
    use super::super::fixedcsr::borrow_pattern;
    use crate::data::FixPair;
    use super::*;
    use crate::driver::launch::host_device;
    use crate::cvec::CVec;
    use crate::cvecvec::CVecVec;
    use crate::data::{Vec2u, Vec3f, VertexParam};
    use crate::driver::test_scene::{position, TestScene};

    /// A scene of `count` vertices whose fixed sparsity is the diagonal alone,
    /// which is what `builder.rs` registers for every vertex before any element
    /// widens it.
    fn diagonal_scene(count: usize) -> TestScene {
        let mut scene = TestScene::new(count);
        let rows: Vec<Vec<u32>> = (0..count as u32).map(|i| vec![i]).collect();
        let transpose: Vec<Vec<Vec2u>> = vec![Vec::new(); count];
        scene.data.fixed_index_table = CVecVec::from(&rows[..]);
        scene.data.transpose_table = CVecVec::from(&transpose[..]);
        scene.data.surface_vert_count = count as u32;
        scene.data.param_arrays.vertex = CVec::from(
            &[VertexParam {
                ghat: 0.01,
                offset: 0.0,
                friction: 0.0,
            }][..],
        );
        for prop in scene.data.prop.vertex.as_mut_slice() {
            prop.mass = 1.0;
            prop.param_index = 0;
        }
        scene
    }

    /// A `ParamSet` with only the fields the constraint pass reads set.
    ///
    /// Safety: `ParamSet` is a `repr(C)` aggregate of scalars with no `Drop`, no
    /// references and no niche-optimized fields, so an all-zero bit pattern is a
    /// valid inhabitant.
    fn param() -> ParamSet {
        let mut out: ParamSet = unsafe { std::mem::zeroed() };
        out.line_search_max_t = 1.0;
        out.friction_eps = 1e-5;
        out.constraint_tol = 1e-3;
        out
    }

    fn ground_plane(ghat: f32) -> Floor {
        Floor {
            ground: position(0.0, 0.0, 0.0),
            ghat,
            friction: 0.0,
            thickness: 0.0,
            up: Vec3f::new(0.0, 1.0, 0.0),
            kinematic: false,
        }
    }

    /// Run one assembly and hand back the force and the diagonal block.
    fn assemble_one(
        scene: &TestScene,
        param: &ParamSet,
        spheres: &[Sphere],
        floors: &[Floor],
    ) -> (Vec<f32>, [f32; 9], u64) {
        // Safety: the scene outlives every borrow below, and the two matrices
        // wrap its own pattern.
        unsafe {
            let data = &*scene.data;
            let vertices = data.vertex.curr.size as usize;
            // ONE DEVICE FOR THE WHOLE FIXTURE. A handle carries an arena
            // index and an offset and nothing naming the allocator that opened
            // it, so allocating on one `host_device()` and dispatching on a
            // second resolves it into a different allocation.
            let mut device = host_device();
            // The matrix owns its pattern handles, so the fixture stages the
            // scene's pattern and hands the same image to both matrices.
            let pattern = borrow_pattern(&data.fixed_index_table, &data.transpose_table)
                .expect("the fixture's pattern is valid");
            let pattern_dev = PatternDevice::of(&mut device, &pattern);
            let mut staging = super::super::state::PushStaging::default();
            let mut reference = FixedCsr::from_dataset(&mut device, pattern_dev.refs(), data).expect("the fixture builds a pattern");
            let mut fixed = FixedCsr::from_dataset(&mut device, pattern_dev.refs(), data).expect("the fixture builds a pattern");
            let mut analytic =
                Analytic::allocate(&mut device, data).expect("a scene with vertices allocates");
            // A DEVICE ALLOCATION with a mirror, because the scatter writes
            // through a handle and the assertions read the result back.
            let mut force = readback::<f32>(&mut device, 3 * vertices, "test.force")
                .expect("the test allocation succeeds");
            // `C`, the block diagonal the barrier's Hessian is deposited into.
            // The production seed is a device fill; a fresh allocation is
            // already zeroed, which is the same starting state.
            let mut diagonal = readback::<f32>(&mut device, 9 * vertices, "test.diagonal")
                .expect("the test allocation succeeds");
            // The positions are device-resident, so a test needs an
            // allocation for the handle to name.
            let positions_host = unsafe {
                crate::driver::state::slice_or_empty(
                    data.vertex.curr.data as *const f32,
                    3 * data.vertex.curr.size as usize,
                )
            };
            let positions_block = crate::driver::state::position_block(
                &mut device,
                positions_host,
                "test.positions",
            );
            let positions = positions_block.handle();
            let test_mesh = unsafe { crate::driver::state::test_mesh_of(&mut device, data) };
            let empty_pins = no_pins(&mut device);
            let sphere_set = staged_from(&mut device, spheres, "test.sphere");
            let floor_set = staged_from(&mut device, floors, "test.floor");
            let statistics =
                crate::driver::contact::StatisticsRefs::absent(&mut device).unwrap();
            let (grain_angular, grain_coupling, grain_rotational) =
                grain_blocks(&mut device, vertices);
            let mut residual = readback::<f32>(&mut device, 3 * vertices, "test.residual")
                .expect("the test allocation succeeds");
            analytic
                .assemble(
                    &mut device,
                    data,
                    test_mesh.refs(),
                    param,
                    param.dt,
                    positions,
                    positions,
                    empty_pins.handle(),
                    &sphere_set,
                    &floor_set,
                    &mut reference,
                    &mut fixed,
                    &mut staging,
                    force.handle(),
                    diagonal.handle(),
                    grain_angular.handle(),
                    grain_coupling.handle(),
                    grain_rotational.handle(),
                    // THE RESIDUAL SNAPSHOT. In production it is a copy of the
                    // force vector taken before this pass, and this fixture
                    // starts with that vector at zero, so a zeroed allocation
                    // is the snapshot it would have made. A zero residual has
                    // no tangential part, so every friction term falls back to
                    // the lagged surrogate; the anchor's own coverage is
                    // `tests/kernels/friction_branches.cpp`.
                    residual.handle(),
                    statistics,
                )
                .expect("the assembly runs");
            {
                force.download(&mut device).expect("the force mirror refreshes");
                // THE BLOCK IS ON THE DIAGONAL, not in the fixed pattern. The
                // pass deposits into `C` rather than pushing `(0, 0)` through a
                // host compaction, and `C` is what `Operator` reads.
                diagonal.download(&mut device).expect("the diagonal reads back");
                let mut block = [0.0f32; 9];
                block.copy_from_slice(&diagonal.host()[..9]);
                (force.host().to_vec(), block, analytic.assembled)
            }
        }
    }

    /// The three per-vertex grain Schur blocks a test hands the assembly.
    ///
    /// `SolverState` owns these in production, sized with the scene whenever it
    /// carries grains, so a test that dispatches the assembly directly has to
    /// supply them the way it already supplies the pins and the colliders. They
    /// are per VERTEX rather than per surface vertex, which is what the
    /// condense, the recover and the integrate all read them over.
    fn grain_blocks(
        device: &mut impl Device,
        vertices: usize,
    ) -> (
        ppf_cts_compute::Buffer<f32>,
        ppf_cts_compute::Buffer<f32>,
        ppf_cts_compute::Buffer<f32>,
    ) {
        let mut angular = ppf_cts_compute::Buffer::<f32>::none();
        let mut coupling = ppf_cts_compute::Buffer::<f32>::none();
        let mut rotational = ppf_cts_compute::Buffer::<f32>::none();
        angular
            .size(device, 9 * vertices, AllocLabel("test.grain_angular"))
            .expect("the angular blocks allocate");
        coupling
            .size(device, 9 * vertices, AllocLabel("test.grain_coupling"))
            .expect("the coupling blocks allocate");
        rotational
            .size(device, 3 * vertices, AllocLabel("test.grain_rotational"))
            .expect("the rotational gradient allocates");
        (angular, coupling, rotational)
    }

    /// An empty pin set, as a device allocation: the tests that carry no pins
    /// still have to hand the record a handle, and a zero-length one is what
    /// `&[]` became.
    fn no_pins(device: &mut impl Device) -> ppf_cts_compute::Buffer<FixPair> {
        let mut buffer = ppf_cts_compute::Buffer::<FixPair>::none();
        buffer
            .size(device, 0, AllocLabel("test.no_pins"))
            .expect("an empty pin set allocates");
        buffer
    }

    /// A collider array for a test, staged and uploaded from a host slice.
    ///
    /// Both production entry points take the allocation rather than the slice,
    /// so a test naming colliders has to stage them the way `advance` does. An
    /// EMPTY one still allocates, because a zero-length handle is what `&[]`
    /// became here, exactly as it did for the pins above.
    fn staged_from<T: Pod + Default>(
        device: &mut impl Device,
        items: &[T],
        label: &'static str,
    ) -> StagedBuffer<T> {
        let mut buffer =
            staged::<T>(device, items.len(), label).expect("the test allocation succeeds");
        buffer.at()[..items.len()].copy_from_slice(items);
        buffer.upload(device).expect("the collider array uploads");
        buffer
    }

    #[test]
    fn a_vertex_past_the_collider_surface_comes_back_through_the_diagnostic_channel() {
        // THE NEGATIVE CONTROL FOR THE SEAM'S DIAGNOSTIC TRANSPORT, which is
        // otherwise the one part of the backend surface nothing exercises. The
        // per-chunk records, the ascending first-writer merge over them and the
        // `Fault::Device` that carries the file and line all sit in
        // `super::launch`; this is the only path in the driver that reads
        // one today, so without this test the channel would be compiled,
        // wired, and never run.
        let mut scene = diagonal_scene(1);
        // Below the ground plane: the gap the dynamic stiffness divides by is
        // not positive, which is what the barrier's domain check refuses.
        scene.place(0, 0.0, -0.001, 0.0);
        // Safety: as `assemble_one`, whose body this repeats because it must
        // keep the error rather than unwrap it.
        let outcome = unsafe {
            let data = &*scene.data;
            let vertices = data.vertex.curr.size as usize;
            // ONE DEVICE FOR THE WHOLE FIXTURE. A handle carries an arena
            // index and an offset and nothing naming the allocator that opened
            // it, so allocating on one `host_device()` and dispatching on a
            // second resolves it into a different allocation.
            let mut device = host_device();
            // The matrix owns its pattern handles, so the fixture stages the
            // scene's pattern and hands the same image to both matrices.
            let pattern = borrow_pattern(&data.fixed_index_table, &data.transpose_table)
                .expect("the fixture's pattern is valid");
            let pattern_dev = PatternDevice::of(&mut device, &pattern);
            let mut staging = super::super::state::PushStaging::default();
            let mut reference = FixedCsr::from_dataset(&mut device, pattern_dev.refs(), data).expect("the fixture builds a pattern");
            let mut fixed = FixedCsr::from_dataset(&mut device, pattern_dev.refs(), data).expect("the fixture builds a pattern");
            let mut analytic =
                Analytic::allocate(&mut device, data).expect("a scene with vertices allocates");
            // A DEVICE ALLOCATION with a mirror, because the scatter writes
            // through a handle and the assertions read the result back.
            let mut force = readback::<f32>(&mut device, 3 * vertices, "test.force")
                .expect("the test allocation succeeds");
            // `C`, which this fixture never reads: it asserts that the pass
            // REFUSES, so the deposit never happens.
            let mut diagonal = readback::<f32>(&mut device, 9 * vertices, "test.diagonal")
                .expect("the test allocation succeeds");
            // The positions are device-resident, so a test needs an
            // allocation for the handle to name.
            let positions_host = unsafe {
                crate::driver::state::slice_or_empty(
                    data.vertex.curr.data as *const f32,
                    3 * data.vertex.curr.size as usize,
                )
            };
            let positions_block = crate::driver::state::position_block(
                &mut device,
                positions_host,
                "test.positions",
            );
            let positions = positions_block.handle();
            let test_mesh = unsafe { crate::driver::state::test_mesh_of(&mut device, data) };
            let empty_pins = no_pins(&mut device);
            let sphere_set = staged_from(&mut device, &[], "test.sphere");
            let floor_set = staged_from(&mut device, &[ground_plane(0.01)], "test.floor");
            let statistics =
                crate::driver::contact::StatisticsRefs::absent(&mut device).unwrap();
            let (grain_angular, grain_coupling, grain_rotational) =
                grain_blocks(&mut device, vertices);
            let mut residual = readback::<f32>(&mut device, 3 * vertices, "test.residual")
                .expect("the test allocation succeeds");
            analytic.assemble(
                &mut device,
                data,
                test_mesh.refs(),
                &param(),
                param().dt,
                positions,
                positions,
                empty_pins.handle(),
                &sphere_set,
                &floor_set,
                &mut reference,
                &mut fixed,
                &mut staging,
                force.handle(),
                diagonal.handle(),
                grain_angular.handle(),
                grain_coupling.handle(),
                grain_rotational.handle(),
                // See the sibling fixture: a zeroed residual is the snapshot
                // this pass would have made, and it selects the lagged form.
                residual.handle(),
                statistics,
            )
        };
        let fatal = outcome.expect_err(
            "a vertex past the collider surface must be refused, not assembled from a \
             non-positive gap",
        );
        assert_eq!(
            fatal.code,
            ppf_cts_formats::status::error_code::DEVICE_ASSERT,
            "a failing kernel check is a device assert, not an invariant: {fatal:?}"
        );
        // THE NEUTRAL BODY'S OWN FILE, which is what the `[[seam::diag]]` lane
        // buys: the check sits in a kernel body all three backends compile, and
        // `__FILE__` there names that body rather than whichever launcher
        // happened to call it.
        assert!(
            fatal.detail.contains("vertex_constraint.kernel.cpp"),
            "the channel must carry the file the check is in, got {}",
            fatal.detail
        );
    }

    #[test]
    fn a_vertex_inside_a_floors_barrier_is_pushed_along_its_normal() {
        // THE VALUE, NOT MERELY THE SIGN. Every number below follows from the
        // shared bodies and is written out so a change in any of them is
        // visible: the elasticity-inclusive stiffness is `n . (H n) + m / gap^2`
        // with H zero here, and the cubic push barrier's gradient is
        // `-(d^2 / ghat) n` at a signed distance `d`.
        //
        // gap  = 0.001, ghat = 0.01, mass = 1
        // d    = gap - ghat            = -0.009
        // k    = 0 + 1 / 0.001^2       = 1e6
        // f    = k * -(d^2) / ghat * n = -8100 * n
        // H    = k * -2 d / ghat * nn^T = 1.8e6 * nn^T
        let mut scene = diagonal_scene(1);
        scene.place(0, 0.0, 0.001, 0.0);
        let (force, block, count) = assemble_one(&scene, &param(), &[], &[ground_plane(0.01)]);
        assert_eq!(count, 1, "the floor contact was not counted");
        assert!(
            (force[1] + 8100.0).abs() < 1.0,
            "the floor barrier's force is {force:?}, not the -8100 along +y the \
             cubic barrier gives at a 1 mm gap under a 10 mm contact gap"
        );
        assert!(force[0].abs() < 1e-3 && force[2].abs() < 1e-3, "got {force:?}");
        // Column-major: element (1, 1) is index 3 * 1 + 1.
        assert!(
            (block[4] - 1.8e6).abs() < 1.0e3,
            "the barrier's curvature landed as {} rather than 1.8e6",
            block[4]
        );
        for (i, value) in block.iter().enumerate() {
            if i != 4 {
                assert!(value.abs() < 1.0, "block element {i} is {value}, not zero");
            }
        }
    }

    #[test]
    fn a_kinematic_floor_keeps_the_elasticity_inclusive_stiffness() {
        // `floor.kinematic` DOES EXACTLY ONE THING: it clamps the gap to
        // `constraint_tol * ghat` before the stiffness is formed, which
        // `kernels/contact/vertex_constraint.kernel.cpp` does a few lines
        // before its call. The floor's stiffness itself is
        // `up.dot(local_hess * up) + mass / gap^2` unconditionally, which is
        // why that call site passes `kinematic = false`. The `mass / ghat^2`
        // arm inside `analytic_contact_evaluate` is the SPHERE's, and handing
        // the floor to it drops the elasticity-inclusive term and softens the
        // barrier everywhere inside the band, where `gap < ghat` makes
        // `mass / gap^2` the larger of the two by `(ghat / gap)^2`.
        //
        // ABOVE THE CLAMP THRESHOLD THE TWO FLOORS MUST AGREE, which is what
        // makes this a clean comparison rather than a second copy of the
        // arithmetic: at a gap where the clamp does not bind, a kinematic floor
        // and a plain one differ in nothing at all.
        let ghat = 0.01f32;
        let param = param();
        let gap = 0.001f32;
        assert!(
            gap > param.constraint_tol * ghat,
            "the fixture must sit above the clamp threshold, or the two floors \
             differ for a reason that is not the defect"
        );

        let mut scene = diagonal_scene(1);
        scene.place(0, 0.0, gap, 0.0);
        let (plain, plain_block, _) =
            assemble_one(&scene, &param, &[], &[ground_plane(ghat)]);

        let mut driven = ground_plane(ghat);
        driven.kinematic = true;
        let (kinematic, kinematic_block, count) =
            assemble_one(&scene, &param, &[], &[driven]);

        assert_eq!(count, 1, "the kinematic floor contact was not counted");
        for k in 0..3 {
            assert!(
                (plain[k] - kinematic[k]).abs() < 1.0,
                "component {k}: a kinematic floor gave {} against the plain \
                 floor's {}. Above the clamp threshold the two must agree; the \
                 `mass / ghat^2` arm belongs to the sphere",
                kinematic[k],
                plain[k]
            );
        }
        assert!(
            (plain_block[4] - kinematic_block[4]).abs() < 1.0e3,
            "the kinematic floor's curvature is {} against the plain floor's {}",
            kinematic_block[4],
            plain_block[4]
        );
    }

    #[test]
    fn a_vertex_outside_the_barrier_contributes_nothing() {
        // The negative case the test above needs beside it: without it, an
        // assembly that pushed EVERY vertex would pass.
        let mut scene = diagonal_scene(1);
        scene.place(0, 0.0, 0.5, 0.0);
        let (force, block, count) = assemble_one(&scene, &param(), &[], &[ground_plane(0.01)]);
        assert_eq!(count, 0, "a vertex half a metre up was counted as a contact");
        assert!(force.iter().all(|v| *v == 0.0), "got {force:?}");
        assert!(block.iter().all(|v| *v == 0.0), "got {block:?}");
    }

    #[test]
    fn a_sphere_pushes_a_vertex_inside_it_back_out() {
        // The same shape for the other primitive, and it exercises the branch
        // the floor does not: the normal is the radial direction, and the gap is
        // measured from the sphere's own surface.
        //
        // radius 1, ghat 0.01, the vertex at 0.999 along +x, so it is 1 mm
        // inside the shell and the outward normal is +x.
        let mut scene = diagonal_scene(1);
        scene.place(0, 0.999, 0.0, 0.0);
        let sphere = Sphere {
            center: position(0.0, 0.0, 0.0),
            ghat: 0.01,
            friction: 0.0,
            radius: 1.0,
            thickness: 0.0,
            bowl: false,
            reverse: true,
            kinematic: false,
        };
        let (force, block, count) = assemble_one(&scene, &param(), &[sphere], &[]);
        assert_eq!(count, 1, "the sphere contact was not counted");
        assert!(
            force[0] > 0.0,
            "a vertex 1 mm inside a container sphere was not pushed inward: {force:?}"
        );
        assert!(
            block[0] > 0.0,
            "the sphere barrier contributed no curvature on the radial axis: {block:?}"
        );
    }

    /// Run one sweep and hand back its verdict.
    fn sweep_one(
        scene: &TestScene,
        param: &ParamSet,
        end: &[crate::data::Vec3f],
        pins: &[FixPair],
        floors: &[Floor],
    ) -> Sweep {
        // Safety: as `assemble_one`.
        unsafe {
            let data = &*scene.data;
            // ONE DEVICE FOR THE WHOLE FIXTURE. A handle carries an arena
            // index and an offset and nothing naming the allocator that opened
            // it, so allocating on one `host_device()` and dispatching on a
            // second resolves it into a different allocation.
            let mut device = host_device();
            let mut analytic =
                Analytic::allocate(&mut device, data).expect("a scene with vertices allocates");
            // The pin set as a device allocation, on the SAME device the sweep
            // dispatches with.
            let mut pin_set = ppf_cts_compute::Buffer::<FixPair>::none();
            pin_set
                .size(&mut device, pins.len(), AllocLabel("test.pins"))
                .expect("the pin set allocates");
            if !pins.is_empty() {
                pin_set
                    .write(&mut device, 0, pins)
                    .expect("the pin set uploads");
            }
            let sphere_set = staged_from(&mut device, &[], "test.sphere");
            let floor_set = staged_from(&mut device, floors, "test.floor");
            // The blocks are built BEFORE the call, so the device is borrowed
            // once rather than three times over.
            let n = 3 * data.vertex.curr.size as usize;
            // Safety: both arrays hold `n` position components.
            let (start_host, finish_host) = unsafe {
                (
                    crate::driver::state::slice_or_empty(data.vertex.curr.data as *const f32, n),
                    crate::driver::state::slice_or_empty(end.as_ptr() as *const f32, n),
                )
            };
            let start_block =
                crate::driver::state::position_block(&mut device, start_host, "test.sweep.start");
            let finish_block =
                crate::driver::state::position_block(&mut device, finish_host, "test.sweep.finish");
            let (start_h, finish_h) = (start_block.handle(), finish_block.handle());
            let test_mesh = unsafe { crate::driver::state::test_mesh_of(&mut device, data) };
            analytic
                .line_search(
                    &mut device,
                    data,
                    test_mesh.refs(),
                    param,
                    start_h,
                    finish_h,
                    pin_set.handle(),
                    &sphere_set,
                    &floor_set,
                )
                .expect("the sweep runs")
        }
    }

    #[test]
    fn the_floor_sweep_stops_a_vertex_before_it_crosses() {
        // A vertex commanded from 10 mm above the floor to 10 mm below it is
        // granted the fraction that lands it at the PARK, not the half that
        // would land it on the plane. A sweep that landed it on the plane
        // would leave the assembly forming `mass / gap^2` on a zero gap. The
        // number is taken from the shared rule rather than written out, so this
        // test states the invariant and not one arithmetic result.
        let ghat = 0.01f32;
        let (above, below) = (0.01f32, -0.01f32);
        let mut scene = diagonal_scene(1);
        scene.place(0, 0.0, above, 0.0);
        let end = vec![position(0.0, below, 0.0)];
        let param = param();
        let sweep = sweep_one(&scene, &param, &end, &[], &[ground_plane(ghat)]);

        let park = crate::driver::ccd::park_gap_analytic(above, ghat, param.ccd_eps);
        let expected = (above - park) / (above - below);
        assert!(
            (sweep.time_of_impact - expected).abs() < 1e-6,
            "the sweep granted {} of the step rather than the {expected} that \
             reaches the parked clearance {park}",
            sweep.time_of_impact
        );

        // The property the fraction exists for: the vertex ends STRICTLY above
        // the plane, so the gap the barrier divides by is never zero. A sweep
        // that merely stopped short of crossing would satisfy the assertion
        // above with a park of zero and still divide by zero here.
        let landed = above + sweep.time_of_impact * (below - above);
        assert!(
            landed > 0.0,
            "the vertex landed at {landed}, on or below the plane"
        );
        assert!(sweep.infeasible_pin.is_none());
    }

    #[test]
    fn a_sweep_that_stays_above_the_floor_takes_the_whole_step() {
        let mut scene = diagonal_scene(1);
        scene.place(0, 0.0, 0.5, 0.0);
        let end = vec![position(0.0, 0.4, 0.0)];
        let sweep = sweep_one(&scene, &param(), &end, &[], &[ground_plane(0.01)]);
        assert_eq!(
            sweep.time_of_impact, 1.0,
            "a vertex that never reaches the floor had its step shortened"
        );
    }

    #[test]
    fn the_analytic_sweep_covers_the_line_search_horizon_not_just_the_newton_step() {
        // THE SWEPT SEGMENT IS `x0 + line_search_max_t * (x1 - x0)`, not the
        // plain Newton segment. `vertex_constraint_sweep`
        // (`kernels/contact/vertex_constraint.kernel.cpp`) builds it that way,
        // and every fraction the body solves is a fraction
        // of that horizon, which is what makes its `line_search_max_t * t`
        // conversions and the caller's final divide by the same factor
        // consistent.
        //
        // EVERY OTHER TEST IN THIS FILE USES `line_search_max_t = 1.0`, where
        // the extrapolation is the identity and the defect is invisible. This
        // one sets the shipped default of 1.25 and places the Newton endpoint
        // ABOVE the floor while the horizon reaches below it: a body that reads
        // the endpoint verbatim finds no crossing at all and grants the whole
        // step, and the analytic colliders have no other CCD behind them, so
        // that is a tunnelling path rather than a tolerance.
        let ghat = 0.01f32;
        let mut param = param();
        param.line_search_max_t = 1.25;

        let mut scene = diagonal_scene(1);
        scene.place(0, 0.0, 0.10, 0.0);
        // The Newton endpoint clears the floor and its barrier band; the
        // horizon, 1.25 times as far, does not.
        let end = vec![position(0.0, 0.015, 0.0)];
        let sweep = sweep_one(&scene, &param, &end, &[], &[ground_plane(ghat)]);

        // THE TIME IS IN HORIZON UNITS, up to `line_search_max_t`: the body
        // converts its fraction by that factor and `step.rs` divides by it
        // again at the end, so an untouched step reads 1.25 here rather than
        // 1.0. A body that swept only the Newton segment finds no crossing at
        // all, because the endpoint at y = 0.015 clears the floor's whole
        // barrier band, and leaves the seed in place.
        let horizon_end = 0.10 + 1.25 * (0.015 - 0.10);
        assert!(
            horizon_end < 0.0,
            "the fixture must place the horizon below the plane, got {horizon_end}"
        );
        assert!(
            sweep.time_of_impact < 1.24,
            "the sweep granted essentially the whole horizon ({}) against a \
             seed of 1.25, so it tested only the Newton segment: the horizon \
             reaches y = {horizon_end}, below the floor",
            sweep.time_of_impact
        );
        assert!(
            sweep.time_of_impact > 0.0,
            "the sweep refused the whole step ({})",
            sweep.time_of_impact
        );
        assert!(sweep.infeasible_pin.is_none());
    }

    #[test]
    fn a_fix_pinned_vertex_driven_through_a_floor_is_reported_rather_than_clamped() {
        // A prescribed vertex has no freedom to yield, so a clamped time would
        // stall the solve without preventing the crossing. It is reported by
        // index instead, and the step fails naming it.
        let mut scene = diagonal_scene(1);
        scene.place(0, 0.0, 0.01, 0.0);
        scene.vertex_props_mut()[0].fix_index = 1;
        let pin = FixPair {
            position: position(0.0, -0.01, 0.0),
            step_delta: Vec3f::zeros(),
            ghat: 0.01,
            index: 0,
            kinematic: false,
            allow_intersection: false,
        };
        let end = vec![position(0.0, -0.01, 0.0)];
        let sweep = sweep_one(&scene, &param(), &end, &[pin], &[ground_plane(0.01)]);
        assert_eq!(
            sweep.infeasible_pin,
            Some(0),
            "a pin commanded from above the floor to below it was not reported"
        );
        // AND ITS TIME IS NOT CLAMPED: a prescribed vertex is excluded from the
        // collider sweep, so the step is refused rather than shortened.
        assert_eq!(sweep.time_of_impact, 1.0);
    }
    #[test]
    fn host_b32_compare_covers_later_chunks_and_propagates_bad_windows() {
        let scene = diagonal_scene(1);
        let mut device = host_device();
        let mut analytic = unsafe { Analytic::allocate(&mut device, &scene.data).unwrap() };
        // The last coordinate is in the second host read chunk.
        let vertices = 513;
        let coordinates = 3 * vertices;
        let zeros = vec![0_u8; coordinates * std::mem::size_of::<f32>()];
        analytic.accept_candidate.size(&mut device, coordinates, AllocLabel("test.b32.candidate")).unwrap();
        analytic.accept_proposal.size(&mut device, coordinates, AllocLabel("test.b32.proposal")).unwrap();
        let candidate = analytic.accept_candidate.handle();
        let proposal = analytic.accept_proposal.handle();
        for handle in [candidate, proposal] {
            device.write(handle, 0, &zeros).unwrap();
        }
        assert!(unsafe { analytic.compare_candidate_bits(&mut device, vertices).unwrap() });
        let last = zeros.len() - std::mem::size_of::<f32>();
        device.write(candidate, last, &(-0.0_f32).to_ne_bytes()).unwrap();
        assert!(!unsafe { analytic.compare_candidate_bits(&mut device, vertices).unwrap() });
        for handle in [candidate, proposal] {
            device.write(handle, last, &f32::INFINITY.to_ne_bytes()).unwrap();
        }
        assert!(!unsafe { analytic.compare_candidate_bits(&mut device, vertices).unwrap() });
        analytic.accept_proposal.free(&mut device).unwrap();
        let error = unsafe { analytic.compare_candidate_bits(&mut device, vertices).unwrap_err() };
        assert!(matches!(error, Fault::Platform { .. }));
    }

    // Exercise the production function and its actual PositionAccept and plane
    // entries. The second vertex measures alpha but is controlled, so its
    // motion cannot satisfy the original eligible-free progress predicate.
    fn bracket_fixture(mesh_ceiling:f32, plane_gate:bool)
        -> (Result<f32,String>,Vec<f32>,Vec<f32>,Vec<f32>) {
        let mut scene=diagonal_scene(2);
        scene.vertex_props_mut()[1].pdrd_body_index=1;
        let origin=[1.0_f32,1.0,0.0,0.0,1.0,0.0];
        let proposal=[f32::from_bits(1.0_f32.to_bits()+1),1.0,0.0,1.0,
            if plane_gate {0.0}else{1.0},0.0];
        unsafe {
            let data=&*scene.data;
            let mut device=host_device();
            let mut analytic=Analytic::allocate(&mut device,data).unwrap();
            let mesh=crate::driver::state::test_mesh_of(&mut device,data);
            let start=crate::driver::state::position_block(&mut device,&origin,"test.bracket.origin");
            let end=crate::driver::state::position_block(&mut device,&proposal,"test.bracket.proposal");
            let mut plane=ground_plane(0.1);plane.ground=position(0.0,0.4,0.0);
            let floors=staged_from(&mut device,if plane_gate {std::slice::from_ref(&plane)}else{&[]},"test.bracket.planes");
            let mut checked=Vec::new();
            let mut check=|device:&mut _,candidate:ppf_cts_compute::Handle,_trial:u32,_fraction:f32,_same:Option<bool>,_compare_ms:f64|->FatalResult<bool> {
                let actual=Analytic::failure_read::<f32>(device,candidate,6).unwrap();
                if plane_gate {assert!(actual[4]>0.4,"mesh callback must follow the original plane gate");}
                checked.push(actual[3]);
                Ok(actual[3]<=mesh_ceiling)
            };
            let result=analytic.certified_plane_accept(&mut device,mesh.refs(),start.handle(),end.handle(),
                2,1.0,&floors,&[0],false,Some(&mut check)).map_err(|error|format!("{error:?}"));
            let actual_origin=Analytic::failure_read::<f32>(&mut device,start.handle(),6).unwrap();
            let actual_proposal=Analytic::failure_read::<f32>(&mut device,end.handle(),6).unwrap();
            assert_eq!(actual_origin,origin,"every retry must preserve the fixed origin");
            if result.is_err() {assert_eq!(actual_proposal,proposal,"refusal must preserve the original proposal");}
            (result,actual_proposal,checked,origin.to_vec())
        }
    }

    #[test]
    fn rounded_accept_bracket_finds_safe_progress_between_halving_trials() {
        let (result,actual,checked,origin)=bracket_fixture(0.9,false);
        let alpha=result.expect("the interior rounded endpoint is safe and has free progress");
        assert_eq!(alpha,0.75);
        assert_eq!(checked,vec![1.0,0.0,0.5,0.75]);
        assert_eq!(actual[0].to_bits(),origin[0].to_bits()+1);
        assert_eq!(actual[3],alpha,"the one commit must match the returned actual fraction");
    }

    #[test]
    fn rounded_accept_bracket_refuses_when_all_progress_candidates_fail_mesh_gate() {
        let (result,_,checked,_)=bracket_fixture(0.5,false);
        assert!(result.unwrap_err().contains("bounded conservative candidate search exhausted"));
        assert!(checked.len()>3 && checked.len()<=33,"32 trial budget plus the original origin check");
        assert!(checked[3..].iter().all(|&alpha|alpha>0.5),"strictly interior refinement must reach adjacency without repeating lo");
        let (budget_result,_,budget_checked,_)=bracket_fixture(1.0/8192.0,false);
        assert!(budget_result.is_err(),"a controlled-only safe movement is not free progress");
        assert_eq!(budget_checked.len(),33,"refinement shares 32 trials with halving, plus original origin check");
    }

    #[test]
    fn rounded_accept_bracket_rechecks_plane_before_mesh_for_each_interior_candidate() {
        let (result,actual,checked,origin)=bracket_fixture(1.0,true);
        assert_eq!(result.unwrap(),0.5625);
        assert_eq!(checked,vec![0.0,0.5,0.5625],"plane-rejected .75 and .625 must never reach mesh admission");
        assert_eq!(actual[0].to_bits(),origin[0].to_bits()+1);
        assert!(actual[4]>0.4);
    }


}

// A certificate for the exact real dot of the captured finite f32 coordinates
// and plane. With binary64 unit roundoff u, each subtraction/product/three-term
// sum path has at most4 rounded operations: |error|<=gamma4*L. L's computed
// positive sum has depth<=4 too. gamma8*Lhat dominates gamma4/(1-gamma4)*Lhat;
// the final nextUp covers rounding while calculating the error bound itself.
fn plane_geometric_gap_positive(x: &[f32], plane: &Floor) -> bool {
    let mut value=0.0f64; let mut magnitude=0.0f64;
    for k in 0..3 {
        let (a,b,n)=(f64::from(x[k]),f64::from(plane.ground[k]),f64::from(plane.up[k]));
        if !a.is_finite() || !b.is_finite() || !n.is_finite() { return false; }
        value+=(a-b)*n;
        magnitude+=(a.abs()+b.abs())*n.abs();
    }
    let u=f64::EPSILON*0.5;
    let error=(8.0*u/(1.0-8.0*u))*magnitude;
    let upper=if error==0.0 {0.0} else {f64::from_bits(error.to_bits()+1)};
    value.is_finite() && upper.is_finite() && value>upper
}

impl Analytic {

    // ACCEPT FAILURE OBSERVER: BEGIN methods
    pub(super) fn take_accept_failure_ledger(&mut self)->Option<serde_json::Value> {
        self.accept_failure_ledger.take()
    }
    fn failure_read<T:Pod+Default>(device:&mut impl Device,handle:ppf_cts_compute::Handle,count:usize)->Option<Vec<T>> {
        let bytes=count.checked_mul(std::mem::size_of::<T>())?;
        if bytes>isize::MAX as usize {return None;}
        let mut values=Vec::new();values.try_reserve_exact(count).ok()?;values.resize_with(count,T::default);
        // Checked count and POD layout. Device::read validates the actual live byte span.
        let raw=unsafe{std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(),bytes)};
        device.read(handle,0,raw).ok()?;Some(values)
    }
    #[allow(clippy::too_many_arguments)]
    fn accept_failure_snapshot(&mut self,device:&mut impl Device,origin:ppf_cts_compute::Handle,
        props:ppf_cts_compute::Handle,vertices:usize,floors:&StagedBuffer<Floor>,controlled:&[u32],
        original_fraction:f32,final_fraction:f32,last_unsafe_valid:bool,last_unsafe_tag:Option<serde_json::Value>,
        trials:Vec<serde_json::Value>)->serde_json::Value {
        let Some(count)=vertices.checked_mul(3) else {return serde_json::json!({"status":"COUNT_OVERFLOW"});};
        let original=self.accept_proposal.handle();let candidate=self.accept_candidate.handle();
        let a=Self::failure_read::<f32>(device,origin,count);
        let p=Self::failure_read::<f32>(device,original,count);
        let unsafe_points=if last_unsafe_valid && self.accept_failure_unsafe.len()==count {
            Self::failure_read::<f32>(device,self.accept_failure_unsafe.handle(),count)
        }else{None};
        let safe=Self::failure_read::<f32>(device,candidate,count);
        let properties=Self::failure_read::<crate::data::VertexProp>(device,props,vertices);
        let rows=a.as_ref().zip(p.as_ref()).zip(safe.as_ref()).zip(properties.as_ref()).map(|(((a,p),s),props)| {
            (0..vertices).map(|i| {
                let prop=props[i];let controlled_point=prop.pdrd_body_index>0 && controlled.contains(&(prop.pdrd_body_index-1));
                let free=prop.mass>0.0 && prop.fix_index==0 && !controlled_point;
                let changed=|other:&[f32]|(0..3).any(|k|a[3*i+k].is_finite() && other[3*i+k].is_finite() && a[3*i+k]!=other[3*i+k]);
                serde_json::json!({"vertex":i,"pdrd_body_index_1based":prop.pdrd_body_index,"eligible_free":free,
                    "excluded_massless":!(prop.mass>0.0),"excluded_fixed":prop.fix_index!=0,"excluded_controlled":controlled_point,
                    "original_proposal_numeric_xyz_changed":changed(p),"final_safe_numeric_xyz_changed":changed(s),
                    "properties":prop,"property_float_bits":{"area":prop.area.to_bits(),"volume":prop.volume.to_bits(),
                        "mass":prop.mass.to_bits(),"rest_bend_angle":prop.rest_bend_angle.to_bits()}})
            }).collect::<Vec<_>>()
        });
        let counts=rows.as_ref().map(|rows|serde_json::json!({
            "eligible_free_vertices":rows.iter().filter(|r|r["eligible_free"]==true).count(),
            "original_proposal_eligible_free_changed_vertices":rows.iter().filter(|r|r["eligible_free"]==true && r["original_proposal_numeric_xyz_changed"]==true).count(),
            "final_safe_eligible_free_changed_vertices":rows.iter().filter(|r|r["eligible_free"]==true && r["final_safe_numeric_xyz_changed"]==true).count(),
            "original_proposal_controlled_changed_vertices":rows.iter().filter(|r|r["excluded_controlled"]==true && r["original_proposal_numeric_xyz_changed"]==true).count(),
            "final_safe_controlled_changed_vertices":rows.iter().filter(|r|r["excluded_controlled"]==true && r["final_safe_numeric_xyz_changed"]==true).count()}));
        let points=|v:Option<Vec<f32>>|v.map(|v|v.chunks_exact(3).enumerate().map(|(vertex,p)|
            serde_json::json!({"vertex":vertex,"gpu_xyz_bits":p.iter().map(|x|x.to_bits()).collect::<Vec<_>>()})).collect::<Vec<_>>());
        serde_json::json!({"schema":"actual-rounded-accept-failure-v1","status":"ORIGINAL_NO_PROGRESS_FATAL",
            "history":false,"vertices":vertices,"coordinate_f32_count":count,"coordinate_byte_length":count.checked_mul(4),
            "vertex_prop_record_byte_length":std::mem::size_of::<crate::data::VertexProp>(),
            "original_fraction_bits":original_fraction.to_bits(),"final_certified_fraction_bits":final_fraction.to_bits(),
            "controlled_bodies":controlled,"trials":trials,"last_unsafe_tag":last_unsafe_tag,
            "origin_actual_gpu":points(a),"fixed_original_proposal_actual_gpu":points(p),
            "last_rejected_actual_gpu":points(unsafe_points),"final_certified_actual_gpu":points(safe),
            "vertex_progress_classification":rows,"counts":counts,
            "runtime_floors_uploaded_host_bits":floors.host().iter().enumerate().map(|(plane,f)|serde_json::json!({
                "plane":plane,"ground_bits":std::array::from_fn::<_,3,_>(|k|f.ground[k].to_bits()),
                "up_bits":std::array::from_fn::<_,3,_>(|k|f.up[k].to_bits()),"ghat_bits":f.ghat.to_bits(),
                "friction_bits":f.friction.to_bits(),"thickness_bits":f.thickness.to_bits(),"kinematic":f.kinematic})).collect::<Vec<_>>(),
            "note":"Only last rejected scratch is retained by diagnostic D2D. No prior coordinates regenerated. Raw mesh TOI/local pair and GPU local gap are unavailable. NumericXYZ classification uses the original eligibility and finite numeric comparison, not signed-zero bits. Null means missing read."})
    }
    // ACCEPT FAILURE OBSERVER: END methods
    /// Certify the initialized retained candidate scratch on this device.
    ///
    /// # Safety
    /// `candidate` must be this object's `accept_candidate.handle()`, containing
    /// at least `3*self.surface_vertices` initialized f32 coordinates, laid out
    /// as Vec3f. `vertex_prop` must address at least `self.surface_vertices`
    /// initialized, valid VertexProp records on the same device. `floors` must
    /// have matching initialized host/device Floor records for its full length,
    /// with that length and the surface extent representable as u32.
    /// `accept_mask` must already hold `floors.len()*self.surface_vertices` u32
    /// cells; that product must fit u32 and all byte ranges must fit usize.
    /// Candidate, props and floors are read-only during the launch and must not
    /// overlap the mask. All allocations must remain live, aligned and unmoved
    /// until synchronous Device::launch/download return. No external work or
    /// alias may concurrently access the scratch/host mirrors. The caller must
    /// satisfy `Device::launch`'s unchecked buffer-domain contract; handle shape
    /// checks do not prove these pointee lengths.
    unsafe fn certify_plane_candidate<D: Device>(
        &mut self, device: &mut D, candidate: ppf_cts_compute::Handle,
        vertex_prop: ppf_cts_compute::Handle, floors: &StagedBuffer<Floor>,
    ) -> FatalResult<bool> {
        let args=super::kernels::PlaneCandidateDomainArgs {
            candidate, vertex_prop, floor:floors.handle(),floor_count:floors.len() as u32,
            out_mask:self.accept_mask.handle(),vertex_count:self.surface_vertices as u32,
            count:self.surface_vertices as u32,seam_arena_count:0,
        };
        device.launch("collider.accept.plane-domain",&args,self.surface_vertices as u32)?;
        self.accept_mask.download(device)?;
        if self.accept_mask.host().iter().any(|&m|m==3) { return Ok(false); }
        if !self.accept_mask.host().iter().any(|&m|m==1) { return Ok(true); }
        self.accept_candidate.download(device)?;
        for (j,plane) in floors.host().iter().enumerate() {
            for i in 0..self.surface_vertices {
                if self.accept_mask.host()[j*self.surface_vertices+i]==1 &&
                    !plane_geometric_gap_positive(&self.accept_candidate.host()[3*i..3*i+3],plane) {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Certify the actual rounded point before copying it over the proposal.
    /// All retries reuse immutable original endpoints. Only non-KIN planes
    /// are certified here; spheres retain their existing swept/barrier behavior.
    /// When supplied, `mesh_certificate` must certify the full actual scratch
    /// endpoint using the caller's original mesh admission/offset/CCD contract.
    /// It may refresh only contact spatial/query workspaces, must not mutate
    /// origin/proposal/candidate or physical solver state, and must reject an
    /// overlapping start before considering its returned swept time.
    ///
    /// # Safety
    /// `origin` and writable `proposed` must name distinct, nonoverlapping,
    /// same-device live allocations of at least `3*vertices` initialized f32
    /// coordinates, aligned/layout-compatible with Vec3f. `mesh.vertex_prop`
    /// must contain `vertices` initialized, valid VertexProp records.
    /// `self.surface_vertices<=vertices<=u32::MAX`; coordinate and VertexProp
    /// byte products must fit usize and isize::MAX. `floors` must expose matching
    /// current host/device Floor records for its full length (including valid
    /// Rust boolean representation), and remain immutable during the call.
    /// On the plane path, floor count and floor*surface mask extent must fit
    /// u32; the method checks that product and sizes retained candidate/proposal
    /// to `3*vertices` f32 and mask to floor*surface u32 before any dispatch.
    /// None of the caller allocations may alias those owned scratch allocations,
    /// each other, or writable host mirrors. They must remain alive and unmoved
    /// through all synchronous launches/copies/readbacks; no external work may
    /// race their access. `origin` remains immutable and `proposed` is exclusively
    /// writable here. Valid device ownership, alignment and pointee lengths are
    /// the caller's obligations under Device::launch, including the direct
    /// no-plane path; handle validation alone does not establish them.
    /// Compare immutable proposal and actual candidate using a retained one-word
    /// result. Input buffer handles describe f32 elements; the byte aliases
    /// explicitly convert lengths and are rejected if u32 indexing overflows.
    unsafe fn compare_candidate_bits<D: Device>(&mut self,device:&mut D,vertices:usize)->Result<bool,Fault> {
        if device.info().backend == "host" {
            // Host windows are copied into bounded stack storage. The optional
            // certificate needs no device allocation, dispatch, or staging buffer.
            let Some(bytes) = vertices.checked_mul(3 * std::mem::size_of::<f32>()) else {
                return Ok(false);
            };
            const CHUNK_BYTES: usize = 4096;
            let mut candidate = [0_u8; CHUNK_BYTES];
            let mut proposal = [0_u8; CHUNK_BYTES];
            for offset in (0..bytes).step_by(CHUNK_BYTES) {
                let len = (bytes - offset).min(CHUNK_BYTES);
                let candidate = &mut candidate[..len];
                let proposal = &mut proposal[..len];
                device.read(self.accept_candidate.handle(), offset, candidate)?;
                device.read(self.accept_proposal.handle(), offset, proposal)?;
                if !super::step::b32_shadow::same_finite_coordinates(candidate, proposal) {
                    return Ok(false);
                }
            }
            return Ok(true);
        }
        let count=vertices.checked_mul(3).and_then(|n|u32::try_from(n).ok())
            .filter(|n|*n<=u32::MAX/4)
            .ok_or_else(||Fault::Platform{call:"B32 bit comparison",detail:"coordinate extent overflow".into()})?;
        self.accept_bit_difference.size(device,1,AllocLabel("collider.accept.bit_difference"))?;
        let candidate=self.accept_candidate.handle();
        let args=super::kernels::B32CompareFiniteBitsArgs {
            candidate_values:candidate, proposal_values:self.accept_proposal.handle(),
            mismatch:self.accept_bit_difference.handle(), count, seam_arena_count:0,
        };
        device.run("collider.accept.compare_bits",|encoder| {
            ppf_cts_compute::encode_zero(encoder,self.accept_bit_difference.handle(),4)?;
            if count>0 { encoder.elements(&args,count)?; }
            Ok(())
        })?;
        self.accept_bit_difference.download(device)?;
        Ok(self.accept_bit_difference.host()[0]==0)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn certified_plane_accept<D: Device>(
        &mut self, device: &mut D, mesh: super::contact::MeshRefs,
        origin: ppf_cts_compute::Handle, proposed: ppf_cts_compute::Handle,
        vertices: usize, mut fraction: f32, floors: &StagedBuffer<Floor>,
        controlled_bodies: &[u32],
        b32_device_compare: bool,
        mut mesh_certificate: Option<&mut dyn FnMut(&mut D, ppf_cts_compute::Handle, u32, f32, Option<bool>, f64)->FatalResult<bool>>,
    ) -> FatalResult<f32> {
        let has_planes=self.surface_vertices>0 && floors.host().iter().any(|p|!p.kinematic);
        if !has_planes && mesh_certificate.is_none() {
            let args=super::kernels::PositionAcceptArgs {
                origin,proposed,fraction,count:vertices as u32,seam_arena_count:0};
            device.launch("step.position_accept",&args,vertices as u32)?;
            return Ok(fraction);
        }
        if !fraction.is_finite() || fraction<=0.0 || fraction>1.0 {
            return Err(Fatal::invariant("ANALYTIC_ACCEPT_UNRESOLVED: invalid CCD fraction"));
        }
        let pairs=self.surface_vertices.checked_mul(floors.len()).filter(|&n|n<=u32::MAX as usize)
            .ok_or_else(||Fatal::invariant("ANALYTIC_ACCEPT_UNRESOLVED: plane pair count overflow"))?;
        self.accept_proposal.size(device,3*vertices,AllocLabel("collider.accept.proposal"))?;
        self.accept_candidate.size(device,3*vertices,AllocLabel("collider.accept.candidate"))?;
        if has_planes {self.accept_mask.size(device,pairs,AllocLabel("collider.accept.mask"))?;}
        let bytes=3*vertices*std::mem::size_of::<f32>();
        device.copy(self.accept_proposal.handle(),0,proposed,0,bytes)?;
        // ACCEPT FAILURE OBSERVER: BEGIN initialization
        let capture_failure=std::env::var_os("PPF_ANALYTIC_STALL_TERMINAL_CAPTURE_DIR")
            .is_some_and(|p|std::path::Path::new(&p).is_absolute());
        if capture_failure {self.accept_failure_ledger=None;}
        let original_fraction=fraction;
        let mut trial_records=if capture_failure {Some(Vec::<serde_json::Value>::new())}else{None};
        let mut last_unsafe_valid=false;
        let mut last_unsafe_tag:Option<serde_json::Value>=None;
        // ACCEPT FAILURE OBSERVER: END initialization
        let mut reduced=false;
        // Only a certified no-progress plateau enables this bounded refinement.
        // Positive finite f32 bit order supplies strictly interior trial values;
        // safety is established independently by both original certificates.
        let mut last_rejected_fraction:Option<f32>=None;
        let mut no_progress_fraction:Option<f32>=None;
        let interior_fraction=|lo:f32,hi:f32|->Option<f32> {
            if !lo.is_finite() || !hi.is_finite() || lo<=0.0 || hi<=lo || hi>original_fraction {
                return None;
            }
            let lo_bits=lo.to_bits();let hi_bits=hi.to_bits();
            let mid=f32::from_bits(lo_bits+(hi_bits-lo_bits)/2);
            (lo<mid && mid<hi).then_some(mid)
        };
        // Readbacks for progress are needed only after an unsafe candidate.
        let mut origin_cpu:Option<Vec<f32>>=None;
        let mut props_cpu:Option<Vec<crate::data::VertexProp>>=None;
        for trial in 0..32 {
            device.copy(self.accept_candidate.handle(),0,self.accept_proposal.handle(),0,bytes)?;
            let candidate=self.accept_candidate.handle();
            let args=super::kernels::PositionAcceptArgs {
                origin,proposed:candidate,fraction,count:vertices as u32,seam_arena_count:0};
            device.launch("collider.accept.trial",&args,vertices as u32)?;
            let plane_ok=!has_planes || self.certify_plane_candidate(device,candidate,mesh.vertex_prop,floors)?;
            // The callback sees the actual rounded scratch, never original proposal.
            let mut same = None;
            let mut compare_ms = 0.0;
            if b32_device_compare && plane_ok && trial == 0 && fraction.to_bits() == 1.0_f32.to_bits() {
                let start=std::time::Instant::now();
                // Propagate device failures to preserve diagnostic context.
                same = Some(self.compare_candidate_bits(device,vertices)?);
                compare_ms = start.elapsed().as_secs_f64()*1000.0;
            }
            let mesh_ok=if plane_ok {match mesh_certificate.as_mut() {
                Some(check)=>check(device,candidate,trial,fraction,same,compare_ms)?,None=>true,
            }}else{false};
            // ACCEPT FAILURE OBSERVER: BEGIN trial scalars
            if capture_failure {
                let _=std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let masks=if has_planes {Some(self.accept_mask.host())}else{None};
                    let gpu_ok=masks.map(|m|!m.iter().any(|&v|v==3));
                    let geometric_ran=masks.is_some_and(|m|!m.iter().any(|&v|v==3) && m.contains(&1));
                    let tag=serde_json::json!({"trial":trial_records.as_ref().unwrap().iter().filter(|r|r.get("fraction_bits").is_some()).count(),
                        "fraction_bits":fraction.to_bits(),"plane_ran":has_planes,
                        "plane_gpu_mask_no_invalid":gpu_ok,"plane_geometric_bound_ran":geometric_ran,
                        "plane_geometric_bound_pass":if geometric_ran {Some(plane_ok)}else{None},
                        "plane_combined_pass":plane_ok,
                        "mesh_status":if !plane_ok {"NOT_CALLED"}else if mesh_certificate.is_none(){"NOT_CONFIGURED"}else if mesh_ok{"PASS"}else{"PARTIAL"},
                        "mesh_combined_pass":if plane_ok && mesh_certificate.is_some(){Some(mesh_ok)}else{None},
                        "raw_mesh_toi_bits":null,"mesh_overlap":null,
                        "missing_mesh_metadata":"existing callback exposes bool only; no extra query or contact API change"});
                    if !(plane_ok && mesh_ok) {last_unsafe_tag=Some(tag.clone());}
                    trial_records.as_mut().unwrap().push(tag);
                }));
            }
            // ACCEPT FAILURE OBSERVER: END trial scalars
            if plane_ok && mesh_ok {
                if reduced {
                    self.accept_candidate.download(device)?;
                    let a=origin_cpu.as_ref().unwrap(); let props=props_cpu.as_ref().unwrap();
                    let progress=(0..vertices).any(|i| {
                        let p=props[i];
                        let free=p.mass>0.0 && p.fix_index==0 &&
                            !(p.pdrd_body_index>0 && controlled_bodies.contains(&(p.pdrd_body_index-1)));
                        free && (0..3).any(|k| {
                            let start=a[3*i+k];let end=self.accept_candidate.host()[3*i+k];
                            start.is_finite() && end.is_finite() && start!=end
                        })
                    });
                    if !progress {
                        no_progress_fraction=Some(fraction);
                        if trial+1<32 {
                            if let Some(next)=last_rejected_fraction.and_then(|hi|interior_fraction(fraction,hi)) {
                                fraction=next;
                                continue;
                            }
                        }
                        let error=Fatal::invariant("ANALYTIC_ACCEPT_UNRESOLVED: no representable free candidate progress");
                        // ACCEPT FAILURE OBSERVER: BEGIN decided original Fatal
                        if capture_failure {
                            let capture=std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                self.accept_failure_snapshot(device,origin,mesh.vertex_prop,vertices,floors,
                                    controlled_bodies,original_fraction,fraction,last_unsafe_valid,
                                    last_unsafe_tag.take(),trial_records.take().unwrap_or_default())
                            }));
                            self.accept_failure_ledger=Some(match capture {
                                Ok(value)=>value,
                                Err(_)=>serde_json::json!({"status":"OBSERVER_PANIC","original_error_preserved":true})});
                        }
                        // ACCEPT FAILURE OBSERVER: END decided original Fatal
                        return Err(error);
                    }
                }
                device.copy(proposed,0,candidate,0,bytes)?;
                return Ok(fraction);
            }
            last_rejected_fraction=Some(fraction);
            // ACCEPT FAILURE OBSERVER: BEGIN preserve actual last rejected scratch
            if capture_failure {
                last_unsafe_valid=false;
                let copy=std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(),Fault> {
                    self.accept_failure_unsafe.size(device,3*vertices,AllocLabel("collider.accept.failure.unsafe"))?;
                    device.copy(self.accept_failure_unsafe.handle(),0,candidate,0,bytes)
                }));
                last_unsafe_valid=matches!(copy,Ok(Ok(())));
                if let Some(tag)=last_unsafe_tag.as_mut() {
                    tag.as_object_mut().unwrap().insert("actual_gpu_scratch_preserved".into(),serde_json::json!(last_unsafe_valid));
                }
            }
            // ACCEPT FAILURE OBSERVER: END preserve actual last rejected scratch
            if let Some(lo)=no_progress_fraction {
                if let Some(next)=interior_fraction(lo,fraction) {
                    fraction=next;
                    continue;
                }
                // The current scratch failed a certificate: do not label it
                // final_certified in the original no-progress observer.
                break;
            }
            if !reduced {
                device.copy(candidate,0,origin,0,bytes)?;
                let origin_plane_ok=!has_planes || self.certify_plane_candidate(device,candidate,mesh.vertex_prop,floors)?;
                let origin_mesh_ok=match mesh_certificate.as_mut() {
                    Some(check)=>check(device,candidate,u32::MAX,0.0,None,0.0)?,None=>true,
                };
                // ACCEPT FAILURE OBSERVER: BEGIN existing origin certificate
                if capture_failure {
                    let _=std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        trial_records.as_mut().unwrap().push(serde_json::json!({"label":"origin_to_origin",
                            "plane_pass":origin_plane_ok,"mesh_pass":origin_mesh_ok,
                            "raw_mesh_toi_bits":null,"note":"original origin checks, not an additional query"}));
                    }));
                }
                // ACCEPT FAILURE OBSERVER: END existing origin certificate
                if !origin_plane_ok || !origin_mesh_ok {
                    return Err(Fatal::invariant("ANALYTIC_ACCEPT_UNRESOLVED: origin is not strictly certifiable"));
                }
                self.accept_candidate.download(device)?;
                origin_cpu=Some(self.accept_candidate.host().to_vec());
                let mut props=vec![crate::data::VertexProp::default();vertices];
                let raw=unsafe{std::slice::from_raw_parts_mut(props.as_mut_ptr().cast::<u8>(),
                    vertices*std::mem::size_of::<crate::data::VertexProp>())};
                device.read(mesh.vertex_prop,0,raw)?;
                props_cpu=Some(props);reduced=true;
            }
            fraction*=0.5;
            if fraction==0.0 || !fraction.is_finite() { break; }
        }
        Err(Fatal::invariant("ANALYTIC_ACCEPT_UNRESOLVED: bounded conservative candidate search exhausted"))
    }
}

