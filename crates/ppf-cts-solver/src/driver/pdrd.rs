// Fork modification: changed from the upstream baseline.
// File: pdrd.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

//! The PDRD reduced six-DOF solve's host orchestration.
//!
//! A transcription of the launchers in
//! `src/kernels/energy/model/pdrd_rigid.hpp`, one for one. Those are the
//! neutral crate's own C++ host code around the eighteen generated rows; the
//! rows are shared and the orchestration is not, so the Rust driver needs its
//! own copy for the same reason it has its own Newton loop.
//!
//! THE REDUCED VECTOR. A PDRD body's vertices carry no per-vertex degrees of
//! freedom: the body moves by six numbers and its vertices follow. So the solve
//! runs in `3 * n_cloth + 6 * n_bodies` floats, laid out as every free vertex's
//! three, then each body's six. `super::rigid_map` is the map between that and
//! the full `3 * nrow` vector, and the four operations below move between them.

use ppf_cts_compute::{AllocLabel, Device, Encoder, EncoderExt, Fault, Handle};

use super::kernels::{
    MixedHalfPackBlockArgs, MixedHalfPdrdApplyArgs, MixedHalfPdrdArithmeticArgs,
    PdrdAssembleInertiaRowArgs, PdrdAssembleSandwichRowArgs, PdrdFitCentroidRowArgs, PdrdFitCovarianceRowArgs, PdrdFitFinishRowArgs,
    PdrdProjectBodyDofsRowArgs, PdrdProlongRowArgs, PdrdRestrictRowArgs, PdrdRestrictGatherArgs,
    PdrdComposeRunningRotationRowArgs, PdrdTranslationLockParticularRowArgs, PdrdRigidifyCentroidRowArgs, PdrdRigidifyWriteRowArgs,
    PdrdPrecondBodyRowArgs, PdrdPrecondClothRowArgs, PdrdScatterRotatedRestRowArgs,
    PdrdSeedRestrictRowArgs,
    PdrdCopyProjectedClothRowArgs,
    PdrdCopyStateRotationRowArgs,};
use super::rigid_map::{RigidMap, Staged};
use crate::args::PcgPrecision;

// Opt-in, failure-only diagnostics. This reproduces the existing acceptance
// predicates without changing the solver's fit or its transactional write.
fn diagnose_continuous_covariance(raw: [f32; 9]) -> serde_json::Value {
    use nalgebra::Matrix3;
    let bits: Vec<u32> = raw.iter().map(|x| x.to_bits()).collect();
    let mut info = serde_json::json!({"covariance_f32_bits": bits});
    macro_rules! reject { ($why:expr) => {{ info["reject"] = serde_json::json!($why); return info; }}; }
    if raw.iter().any(|v| !v.is_finite()) { reject!("nonfinite_input"); }
    let c = Matrix3::from_column_slice(&raw.map(f64::from));
    let scale = c.norm(); let det = c.determinant();
    info["norm"] = serde_json::json!(scale); info["det"] = serde_json::json!(det);
    if !scale.is_finite() || scale <= 0.0 || det <= 0.0 { reject!("norm_or_determinant"); }
    let svd = c.svd(true, true); let singular = svd.singular_values;
    info["singular_values"] = serde_json::json!(singular.as_slice());
    if singular.iter().any(|v| !v.is_finite() || *v <= scale * 1e-7) { reject!("singular_values"); }
    let (Some(u), Some(vt)) = (svd.u, svd.v_t) else { reject!("missing_svd_vectors"); };
    let recon = u * Matrix3::from_diagonal(&singular) * vt;
    let recon_residual = (recon-c).norm(); info["reconstruction_residual"] = serde_json::json!(recon_residual);
    if recon.iter().any(|v| !v.is_finite()) || recon_residual > scale * 1e-9 { reject!("svd_reconstruction"); }
    let p = u * vt; let orth = (p.transpose()*p-Matrix3::identity()).norm();
    info["rotation_det"] = serde_json::json!(p.determinant()); info["rotation_orthogonality"] = serde_json::json!(orth);
    if p.iter().any(|v| !v.is_finite()) || p.determinant() <= 0.0 || orth > 1e-9 { reject!("rotation"); }
    let h = (p.transpose()*c+c.transpose()*p)*0.5;
    if h.iter().any(|v| !v.is_finite()) { reject!("nonfinite_stretch"); }
    let eigen = h.symmetric_eigen().eigenvalues; info["stretch_eigenvalues"] = serde_json::json!(eigen.as_slice());
    if eigen.iter().any(|v| !v.is_finite() || *v <= scale*1e-7) { reject!("stretch_eigenvalues"); }
    let polar_residual = (c-p*h).norm(); info["polar_residual"] = serde_json::json!(polar_residual);
    if polar_residual > scale*1e-9 { reject!("polar_reconstruction"); }
    let out: [f32;9] = std::array::from_fn(|i| p.as_slice()[i] as f32);
    let projected = Matrix3::from_column_slice(&out.map(f64::from));
    let projected_orth = (projected.transpose()*projected-Matrix3::identity()).norm();
    let projection_error = (projected-p).norm();
    info["projected_orthogonality"] = serde_json::json!(projected_orth); info["projection_error"] = serde_json::json!(projection_error);
    if projected.iter().any(|v| !v.is_finite()) || projected.determinant() <= 0.0 || projected_orth > 2e-6 || projection_error > 2e-6 { reject!("f32_projection"); }
    reject!("diagnostic_did_not_reproduce_rejection")
}

unsafe fn fit_diagnostic_read<T: Copy + Default, D: Device>(device: &mut D, handle: Handle, offset: usize, n: usize) -> Result<Vec<T>, Fault> {
    let mut data = vec![T::default(); n];
    let bytes = std::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<u8>(), n * std::mem::size_of::<T>());
    device.read(handle, offset, bytes)?;
    Ok(data)
}

/// Everything the preconditioner's assembly reads that is not the reduction
/// itself.
///
/// A CARRIER RATHER THAN TEN PARAMETERS, for the reason the reference's own
/// `fill_row_inputs` template exists: writing the list out per dispatch is a
/// chance to disagree with itself.
#[derive(Clone, Copy)]
pub struct PrecondInputs {
    pub vert_list: Handle,
    pub vertex_prop: Handle,
    pub state: Handle,
    pub rest_centered: Handle,
    pub dyn_index: Handle,
    pub dyn_offset: Handle,
    pub dyn_value: Handle,
    pub fixed_index: Handle,
    pub fixed_offset: Handle,
    pub fixed_value: Handle,
    pub rows: u32,
    pub body_vertices: u32,
}

/// The reduction, and the buffers its rows address.
pub struct Reduction<'a> {
    pub map: &'a RigidMap,
    pub staged: &'a mut Staged,
}

impl Reduction<'_> {
    /// `x = P u`: the full vector a reduced one stands for.
    ///
    /// A free vertex takes its own three floats back; a body's vertex takes the
    /// body's translation plus its rotation acting on the vertex's
    /// body-rotated rest vector, which is what `rotated_rest` holds.
    ///
    /// # Safety
    /// `reduced` must hold `map.dim` floats and `full` must hold `3 * nrow`.
    pub unsafe fn prolong<D: Device>(
        &mut self,
        device: &mut D,
        reduced: Handle,
        full: Handle,
    ) -> Result<(), Fault> {
        let args = self.prolong_args(reduced, full);
        device.launch("pdrd.prolong", &args, args.count)?;
        Ok(())
    }

    pub(super) unsafe fn encode_prolong(
        &self,
        encoder: &mut dyn Encoder,
        reduced: Handle,
        full: Handle,
    ) -> Result<(), Fault> {
        let args = self.prolong_args(reduced, full);
        encoder.elements(&args, args.count)
    }

    fn prolong_args(&self, reduced: Handle, full: Handle) -> PdrdProlongRowArgs {
        PdrdProlongRowArgs {
            vertex_body: self.staged.vbody.handle(),
            cloth_offset: self.staged.cloth_off.handle(),
            rotated_rest: self.staged.prot.handle(),
            reduced,
            full,
            body_base: self.map.body_base as u32,
            count: self.map.nrow as u32,
            seam_arena_count: 0,
        }
    }

    /// Copy the CLOTH rows of a full-space vector back into a reduced one,
    /// leaving every body row untouched.
    ///
    /// THE HALF OF `project_tangent` THAT IS NOT A PROJECTION. A locked scene's
    /// tangent projector runs in the FULL space, because the aggregate lock's
    /// rows are built over vertices, so a reduced vector is prolonged, projected
    /// there, and its cloth rows brought back by this. The body rows are NOT
    /// brought back: they are projected in the reduced basis by
    /// `project_bodies`, on either side of this call, and copying them from the
    /// full vector would overwrite that with the prolonged image.
    ///
    /// `restrict` is not a substitute. That one ACCUMULATES body rows through
    /// the rigid Jacobian, which is the force restriction; this one STORES cloth
    /// rows and nothing else.
    ///
    /// # Safety
    /// As `prolong`.
    pub unsafe fn copy_projected_cloth<D: Device>(
        &mut self,
        device: &mut D,
        full: Handle,
        reduced: Handle,
    ) -> Result<(), Fault> {
        let args = PdrdCopyProjectedClothRowArgs {
            vertex_body: self.staged.vbody.handle(),
            cloth_offset: self.staged.cloth_off.handle(),
            full,
            reduced,
            count: self.map.nrow as u32,
            seam_arena_count: 0,
        };
        device.launch("pdrd.copy_projected_cloth", &args, self.map.nrow as u32)?;
        Ok(())
    }

    /// `u = P^T y`: the FORCE restriction.
    ///
    /// A body row accumulates `sum_v J_v^T y_v`, which is the right thing for a
    /// residual and the wrong thing for a seed; `seed_restrict` below is the
    /// other one and they must not be confused.
    ///
    /// THE WHOLE REDUCED VECTOR IS ZEROED FIRST, not only the body region: a
    /// cloth row is written by exactly one thread and a body row is
    /// accumulated, so both start from zero.
    ///
    /// AND THE SAME ALLOCATION IS PASSED TWICE, as `cloth_out` and as
    /// `reduced`. That is the row's own design, stated in its body: cloth
    /// offsets are disjoint so those words are STORED, while body rows are
    /// accumulated by atomics, and the seam has no float atomic store. Passing
    /// two different buffers here would split the reduced vector in half and
    /// leave each half missing the other's rows.
    ///
    /// # Safety
    /// As `prolong`.
    pub unsafe fn restrict<D: Device>(
        &mut self,
        device: &mut D,
        full: Handle,
        reduced: Handle,
        zero: &[f32],
    ) -> Result<(), Fault> {
        if self.staged.restrict_gather_enabled {
            return device.run("pdrd.restrict.gather", |encoder| self.encode_restrict(encoder, full, reduced, true)).map(|_| ());
        }
        self.clear(device, reduced, zero)?;
        let args = self.restrict_args(full, reduced);
        device.launch("pdrd.restrict", &args, args.count)?;
        Ok(())
    }

    pub(super) unsafe fn encode_restrict(
        &self,
        encoder: &mut dyn Encoder,
        full: Handle,
        reduced: Handle,
        allow_gather: bool,
    ) -> Result<(), Fault> {
        if allow_gather && self.staged.restrict_gather_enabled {
            let args = PdrdRestrictGatherArgs {
                vertex_body: self.staged.vbody.handle(), cloth_offset: self.staged.cloth_off.handle(),
                vert_list: self.staged.vert_list.handle(), body_prop: self.staged.body_prop.handle(),
                rotated_rest: self.staged.prot.handle(), full, reduced,
                body_base: self.map.body_base as u32, nrow: self.map.nrow as u32,
                nbodies: self.map.n_bodies as u32,
                count: (self.map.nrow + self.map.n_bodies) as u32, seam_arena_count: 0,
            };
            return encoder.elements(&args, args.count);
        }
        ppf_cts_compute::encode_zero(encoder, reduced, self.map.dim * std::mem::size_of::<f32>())?;
        let args = self.restrict_args(full, reduced);
        encoder.elements(&args, args.count)
    }

    fn restrict_args(&self, full: Handle, reduced: Handle) -> PdrdRestrictRowArgs {
        PdrdRestrictRowArgs {
            vertex_body: self.staged.vbody.handle(),
            cloth_offset: self.staged.cloth_off.handle(),
            rotated_rest: self.staged.prot.handle(),
            full,
            cloth_out: reduced,
            reduced,
            body_base: self.map.body_base as u32,
            count: self.map.nrow as u32,
            seam_arena_count: 0,
        }
    }

    /// The reduced image of an INITIAL GUESS, which is deliberately not `P^T`.
    ///
    /// It copies the cloth rows and leaves every body row at zero. The force
    /// restriction would sum a body's vertex seeds into its six rows, which is
    /// six times the body's own displacement rather than its displacement.
    ///
    /// # Safety
    /// As `prolong`.
    pub unsafe fn seed_restrict<D: Device>(
        &mut self,
        device: &mut D,
        full: Handle,
        reduced: Handle,
        zero: &[f32],
    ) -> Result<(), Fault> {
        self.clear(device, reduced, zero)?;
        let args = PdrdSeedRestrictRowArgs {
            vertex_body: self.staged.vbody.handle(),
            cloth_offset: self.staged.cloth_off.handle(),
            full,
            reduced,
            count: self.map.nrow as u32,
            seam_arena_count: 0,
        };
        device.launch("pdrd.seed_restrict", &args, self.map.nrow as u32)?;
        Ok(())
    }

    /// Remove each body's forbidden degrees of freedom from a reduced vector.
    ///
    /// A NO-OP UNLESS SOMETHING CONSTRAINS A BODY, which is the same early
    /// return the reference makes: with no joint and no lock, every body's six
    /// DOFs are free and there is nothing to project. The driver tests it
    /// rather than dispatching a pass that would do nothing.
    ///
    /// THE DEFORMABLE LOCKS ARE NOT THIS PROJECTOR'S. A locked group whose
    /// `pdrd_body_index` is zero lives in the full vector and
    /// `super::lock::Projector` removes its rows; the two have disjoint
    /// supports and `builder.rs` rejects every grouping that would break that.
    ///
    /// # Safety
    /// `reduced` must hold `map.dim` floats.
    pub unsafe fn project_bodies<D: Device>(
        &mut self,
        device: &mut D,
        reduced: Handle,
    ) -> Result<(), Fault> {
        if !self.map.needs_projection() {
            return Ok(());
        }
        let args = PdrdProjectBodyDofsRowArgs {
            joint_mode: self.staged.jmode.handle(),
            joint_axis: self.staged.jaxis.handle(),
            translation_lock: self.staged.tlock.handle(),
            translation_axis: self.staged.tlock_axis.handle(),
            translation_mode: self.staged.tlock_mode.handle(),
            rotation_lock: self.staged.rlock.handle(),
            rotation_axis: self.staged.rlock_axis.handle(),
            rotation_mode: self.staged.rlock_mode.handle(),
            reduced,
            body_base: self.map.body_base as u32,
            count: self.map.n_bodies as u32,
            seam_arena_count: 0,
        };
        device.launch("pdrd.project_bodies", &args, self.map.n_bodies as u32)?;
        Ok(())
    }

    /// Copy each body's fitted rotation out of its `RigidState` into a flat
    /// nine-float-per-body array.
    ///
    /// THE SECOND HALF OF THE RUNNING ROTATION'S SEED. `launch_seed_rprev`
    /// (`energy/model/pdrd_rigid.hpp:317`) fits the pose and then copies the
    /// rotation out with this, which is what makes the anchored rigidify start
    /// from the pose the run was HANDED rather than from the identity. On a
    /// fresh scene the two agree, because an unrotated body fits to the
    /// identity; on a `--load` resume they do not, and seeding the identity
    /// there aims the rigidify at the authored orientation instead of the saved
    /// one.
    ///
    /// # Safety
    /// As `prolong`.
    pub unsafe fn copy_state_rotation<D: Device>(
        &mut self,
        device: &mut D,
        state: Handle,
        rotation: Handle,
    ) -> Result<(), Fault> {
        if self.map.n_bodies == 0 {
            return Ok(());
        }
        let args = PdrdCopyStateRotationRowArgs {
            state,
            rotation,
            count: self.map.n_bodies as u32,
            seam_arena_count: 0,
        };
        device.launch("pdrd.copy_state_rotation", &args, self.map.n_bodies as u32)?;
        Ok(())
    }

    /// The polar fit: each body's centroid, covariance and best-fit rotation
    /// from the current pose.
    ///
    /// THREE PASSES THAT CANNOT BE FUSED, in the reference's order and for its
    /// reasons: the covariance is about the centroid, so it cannot start until
    /// the first pass finishes, and the finish reads both. The first two run
    /// over the body VERTICES and the third over the BODIES, which is why the
    /// counts differ.
    ///
    /// THE SCRATCH IS TWELVE FLOATS PER BODY and is cleared first, both passes
    /// accumulating into it by atomics. It is the caller's for the reason the
    /// zeros are: this runs every step and a per-step allocation is what the
    /// driver is held not to do.
    ///
    /// # Safety
    /// Every handle must name a live allocation sized for the scene, and
    /// `scratch` must hold `12 * n_bodies` floats.
    #[allow(clippy::too_many_arguments)]
    #[track_caller]
    pub unsafe fn fit<D: Device>(
        &mut self,
        device: &mut D,
        vert_list: Handle,
        vertex_prop: Handle,
        body_prop: Handle,
        rest_centered: Handle,
        positions: Handle,
        scratch: Handle,
        state: Handle,
        body_vertices: u32,
        zero: &[f32],
    ) -> Result<(), Fault> {
        let bodies = self.map.n_bodies;
        if bodies == 0 {
            return Ok(());
        }
        let floats = 12 * bodies;
        debug_assert!(zero.len() >= floats);

        let centroid = PdrdFitCentroidRowArgs {
            vert_list,
            prop: vertex_prop,
            body_prop,
            positions,
            scratch,
            count: body_vertices,
            seam_arena_count: 0,
        };


        let covariance = PdrdFitCovarianceRowArgs {
            vert_list,
            prop: vertex_prop,
            body_prop,
            positions,
            rest_centered,
            // THE SAME ALLOCATION TWICE, as the reference passes it: the first
            // twelve-float run holds the centroid this pass reads and the
            // covariance it accumulates, which is why one buffer serves both
            // roles and a second would leave the covariance measured about
            // nothing.
            centroid: scratch,
            scratch,
            count: body_vertices,
            seam_arena_count: 0,
        };


        let finish = PdrdFitFinishRowArgs {
            vert_list,
            body_prop,
            positions,
            scratch,
            state,
            count: bodies as u32,
            seam_arena_count: 0,
        };
        device.run("pdrd.fit", |encoder| {
            ppf_cts_compute::encode_zero(encoder, scratch, floats * std::mem::size_of::<f32>())?;
            encoder.elements(&centroid, body_vertices)?;
            encoder.elements(&covariance, body_vertices)?;
            encoder.elements(&finish, bodies as u32)
        })?;
        // Only the explicitly selected continuous bodies need an accurate
        // common fit. Legacy-only scenes return without any readback or write.
        let mask=&self.staged.continuous_fit_mask;
        if !mask.is_empty() {
            if mask.len()!=bodies {return Err(Fault::Platform{call:"continuous rigid fit",detail:"selected body count mismatch".into()});}
            if mask.iter().any(|v|*v) {
                let mut values=vec![0.0_f32;floats];
                let mut fitted=vec![crate::data::PdrdRigidState::default();bodies];
                let value_bytes=std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(),values.len()*std::mem::size_of::<f32>());
                device.read(scratch,0,value_bytes)?;
                let state_bytes=std::slice::from_raw_parts_mut(fitted.as_mut_ptr().cast::<u8>(),fitted.len()*std::mem::size_of::<crate::data::PdrdRigidState>());
                device.read(state,0,state_bytes)?;
                // Validate every replacement before the single device write;
                // no selected row is overwritten if any fit is invalid.
                for (i,&selected) in mask.iter().enumerate() {
                    if !selected {continue;}
                    let covariance=std::array::from_fn(|j|values[12*i+3+j]);
                    fitted[i].rotation=match crate::continuous_twist::accurate_fit_rotation(covariance) {
                        Some(rotation) => rotation,
                        None => {
                            if std::env::var_os("PPF_DIAGNOSTIC_FIT").is_some() {
                                let legacy = diagnose_continuous_covariance(covariance);
                                let mut diagnostic = serde_json::json!({
                                    "covariance_f32_bits":covariance.map(f32::to_bits),
                                    "reject":crate::continuous_twist::accurate_fit_rotation_details(covariance).err(),
                                    "legacy_svd_diagnostic":legacy,
                                });
                                diagnostic["body_index"] = serde_json::json!(i);
                                let caller = std::panic::Location::caller();
                                diagnostic["caller"] = serde_json::json!({"file":caller.file(),"line":caller.line()});
                                let sampled = (|| -> Result<serde_json::Value, Fault> {
                                    let props = fit_diagnostic_read::<crate::data::PdrdBodyProp, _>(device, body_prop, i * std::mem::size_of::<crate::data::PdrdBodyProp>(), 1)?;
                                    let start = props[0].vertex_start as usize;
                                    let count = props[0].vertex_count as usize;
                                    if start.checked_add(count).map_or(true, |end| end > body_vertices as usize) {
                                        return Ok(serde_json::json!({"sampling_error":"invalid_body_vertex_range"}));
                                    }
                                    let ids = fit_diagnostic_read::<u32, _>(device, vert_list, start * std::mem::size_of::<u32>(), count)?;
                                    if ids.iter().any(|v| *v as usize >= self.map.nrow) {
                                        return Ok(serde_json::json!({"sampling_error":"invalid_vertex_index","vertex_indices":ids}));
                                    }
                                    let rest = fit_diagnostic_read::<f32, _>(device, rest_centered, start * 3 * std::mem::size_of::<f32>(), count * 3)?;
                                    let xyz = fit_diagnostic_read::<f32, _>(device, positions, 0, self.map.nrow * 3)?;
                                    let position_bits: Vec<[u32;3]> = ids.iter().map(|v| {
                                        let j = *v as usize * 3;
                                        [xyz[j].to_bits(),xyz[j+1].to_bits(),xyz[j+2].to_bits()]
                                    }).collect();
                                    let rest_bits: Vec<[u32;3]> = rest.chunks_exact(3).map(|r| [r[0].to_bits(),r[1].to_bits(),r[2].to_bits()]).collect();
                                    Ok(serde_json::json!({"vertex_start":start,"vertex_indices":ids,"candidate_position_f32_bits":position_bits,"rest_centered_f32_bits":rest_bits}))
                                })();
                                diagnostic["samples"] = match sampled { Ok(value) => value, Err(err) => serde_json::json!({"sampling_error":format!("{err:?}")}) };
                                eprintln!("PDRD_FIT_FAILURE_JSON {}", diagnostic);
                            }
                            return Err(Fault::Platform{call:"continuous rigid fit",detail:format!("body {i}: proper polar fit invalid or ill-conditioned")});
                        }
                    };
                }
                let state_bytes=std::slice::from_raw_parts(fitted.as_ptr().cast::<u8>(),fitted.len()*std::mem::size_of::<crate::data::PdrdRigidState>());
                device.write(state,0,state_bytes)?;
            }
        }
        Ok(())
    }

    /// Rewrite every body's vertices as the exact rigid transform of its rest
    /// shape.
    ///
    /// WHY THE STEP NEEDS IT AT ALL. The reduced solve moves a body by six
    /// numbers, but the vertices it moves are still stored per vertex and the
    /// line search advances them along a direction that is only rigid to the
    /// solve's own tolerance. Rigidifying refits the body onto the pose the
    /// solve intended, so a body cannot drift out of rigidity across frames.
    ///
    /// TWO PASSES, and the centroid must be complete before the write reads it:
    /// the first accumulates each body's mass-weighted centroid over its
    /// vertices, the second places every vertex at `centroid + R_run * ybar`.
    ///
    /// IT TAKES THE RUNNING ROTATION, NOT THE FITTED ONE. `R_run` is the
    /// rotation the solve has actually applied, composed across frames, which
    /// is what makes the rigidify ANCHORED: refitting to a freshly polar-fitted
    /// rotation each step would let the fit's own error accumulate as drift.
    ///
    /// # Safety
    /// Every handle must name a live allocation sized for the scene, and
    /// `centroid` must hold `3 * n_bodies` floats.
    /// The lock's PARTICULAR SOLUTION, written into the reduced seed.
    ///
    /// `launch_translation_lock_particular` in the reference, called once
    /// between the seed's restriction and the first residual. A locked body's
    /// three translation DOFs are not free: the lock's accumulated drift over
    /// its total mass IS their value, so the seed carries it and the CG then
    /// solves only the homogeneous part. A HINGE is skipped, its translation
    /// being pinned by the joint rather than by the lock.
    ///
    /// # Safety
    /// Every handle must name a live allocation, and `reduced` must hold the
    /// reduced vector's `dim` floats.
    pub unsafe fn translation_lock_particular<D: Device>(
        &mut self,
        device: &mut D,
        locks: Handle,
        drift: Handle,
        reduced: Handle,
    ) -> Result<(), Fault> {
        let bodies = self.map.n_bodies;
        if bodies == 0 || !self.map.any_translation_lock {
            return Ok(());
        }
        let args = PdrdTranslationLockParticularRowArgs {
            body_lock: self.staged.tlock.handle(),
            joint_mode: self.staged.jmode.handle(),
            locks,
            drift,
            reduced,
            body_base: self.map.body_base as u32,
            count: bodies as u32,
            seam_arena_count: 0,
        };
        device.launch("pdrd.translation_lock_particular", &args, bodies as u32)?;
        Ok(())
    }

    /// Compose the rotation this Newton iteration ACTUALLY APPLIED onto each
    /// body's persistent running rotation.
    ///
    /// THE SCALE IS NEGATED, and the sign is not a convention to pick. The
    /// prolongation writes the per-vertex step as `dx_b - p x dtheta`, and the
    /// iterate is updated by SUBTRACTING it, so the body is rotated by
    /// `-(toi_recale * toi) * dtheta`: the running rotation integrates the
    /// negated scaled reduced rotation. `main.cu:1450` passes exactly
    /// `-(toi_recale * toi)`.
    ///
    /// # Safety
    /// `running` must hold `9 * n_bodies` floats and `rotation_step`
    /// `3 * n_bodies`.
    pub unsafe fn compose_running_rotation<D: Device>(
        &mut self,
        device: &mut D,
        running: Handle,
        rotation_step: Handle,
        scale: f32,
    ) -> Result<(), Fault> {
        let bodies = self.map.n_bodies;
        if bodies == 0 {
            return Ok(());
        }
        let n=bodies as usize;
        let cells=n.checked_mul(41).ok_or_else(||Fault::Platform{call:"pdrd running rotation",detail:"observation extent overflow".into()})?;
        self.staged.running_rotation_fault.size(device,n,AllocLabel("pdrd.running_rotation_fault"))?;
        self.staged.running_rotation_observed.size(device,cells,AllocLabel("pdrd.running_rotation_observed"))?;
        let args = PdrdComposeRunningRotationRowArgs {
            running,
            dtheta: rotation_step,
            fault: self.staged.running_rotation_fault.handle(),
            observed: self.staged.running_rotation_observed.handle(),
            scale,
            count: bodies as u32,
            seam_arena_count: 0,
        };
        device.launch("pdrd.compose_running_rotation", &args, bodies as u32)?;
        self.staged.running_rotation_fault.download(device)?;
        if let Some((body,&stage))=self.staged.running_rotation_fault.host().iter().enumerate().find(|(_,v)|**v!=0) {
            let mut observed=vec![0f32;cells];
            if let Err(error)=self.staged.running_rotation_observed.read(device,0,&mut observed) {
                return Err(Fault::Platform{call:"pdrd running rotation",detail:format!("body={body} stage={stage} diagnostic_read_error={error:?}")});
            }
            let row=&observed[41*body..41*(body+1)];
            let raw=|range:std::ops::Range<usize>| row[range].iter().map(|v|format!("{:08x}",v.to_bits())).collect::<Vec<_>>();
            let prev=raw(0..9); let input=raw(36..40);
            let angle2=if stage>=2 {format!("{:08x}",row[40].to_bits())} else {"unavailable".into()};
            let delta=if stage>=3 {format!("{:?}",raw(9..18))} else {"unavailable".into()};
            let x=if stage>=4 {format!("{:?}",raw(18..27))} else {"unavailable".into()};
            let y=if stage>=5 {format!("{:?}",raw(27..36))} else {"unavailable".into()};
            return Err(Fault::Platform{call:"pdrd running rotation",detail:format!("body={body} stage={stage} prev_bits={prev:?} dtheta_scale_bits={input:?} angle2_bits={angle2} delta_bits={delta} x_bits={x} y_bits={y}")});
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rigidify<D: Device>(
        &mut self,
        device: &mut D,
        vert_list: Handle,
        vertex_prop: Handle,
        body_prop: Handle,
        rest_centered: Handle,
        positions: Handle,
        running_rotation: Handle,
        centroid: Handle,
        out: Handle,
        body_vertices: u32,
        zero: &[f32],
    ) -> Result<(), Fault> {
        let bodies = self.map.n_bodies;
        if bodies == 0 {
            return Ok(());
        }
        let floats = 3 * bodies;
        debug_assert!(zero.len() >= floats);

        let sum = PdrdRigidifyCentroidRowArgs {
            vert_list,
            prop: vertex_prop,
            body_prop,
            positions,
            centroid,
            count: body_vertices,
            seam_arena_count: 0,
        };


        let write = PdrdRigidifyWriteRowArgs {
            vert_list,
            prop: vertex_prop,
            body_prop,
            positions,
            centroid,
            running_rotation,
            rest_centered,
            out,
            count: body_vertices,
            seam_arena_count: 0,
        };
        device.run("pdrd.rigidify", |encoder| {
            ppf_cts_compute::encode_zero(encoder, centroid, floats * std::mem::size_of::<f32>())?;
            encoder.elements(&sum, body_vertices)?;
            encoder.elements(&write, body_vertices)
        })?;
        Ok(())
    }

    /// Re-scatter every body vertex's rotated rest vector from the fitted
    /// state.
    ///
    /// THIS IS THE ONLY PART OF THE MAP THAT IS NOT TOPOLOGY, and it must run
    /// EVERY Newton iteration, after the fit and before any prolong or
    /// restrict. `prot[v]` is `R_b * ybar_v`, so it moves with the body's
    /// rotation, and both the prolongation and the force restriction read it as
    /// the rigid Jacobian's moment arm. Staged once with the topology it would
    /// hold the first iteration's rotation for the whole solve, and the
    /// reduction would silently work in a frame the body has left.
    ///
    /// A CLOTH VERTEX'S SLOT IS NEVER WRITTEN AND NEVER READ: the dispatch runs
    /// over `pdrd_vert_list`, the body vertices alone, and both readers touch
    /// `rotated_rest[v]` only on the body branch.
    ///
    /// # Safety
    /// Every handle must name a live allocation sized for the scene.
    pub unsafe fn scatter_rotated_rest<D: Device>(
        &mut self,
        device: &mut D,
        vert_list: Handle,
        vertex_prop: Handle,
        state: Handle,
        rest_centered: Handle,
        body_vertices: u32,
    ) -> Result<(), Fault> {
        if self.map.n_bodies == 0 {
            return Ok(());
        }
        let args = PdrdScatterRotatedRestRowArgs {
            vert_list,
            prop: vertex_prop,
            state,
            rest_centered,
            rotated_rest: self.staged.prot.handle(),
            count: body_vertices,
            seam_arena_count: 0,
        };
        device.launch("pdrd.scatter_rotated_rest", &args, body_vertices)?;
        Ok(())
    }

    /// Build each body's 6x6 preconditioner factor.
    ///
    /// TWO DISPATCHES AND A HOST FACTORIZATION. The inertia pass seeds each
    /// body's block with its analytic rigid inertia; the sandwich pass folds in
    /// `sum_k J_k^T (A(v,v) + B(v,v)) J_k`, the DIAGONAL contact and elastic
    /// blocks of the body's own vertices seen through the rigid Jacobian; and
    /// the host then factors each 6x6.
    ///
    /// THIS IS NOT `R` RESTRICTED TO A BODY, and must not be mistaken for it.
    /// It drops every body-cloth and body-body coupling and substitutes the
    /// analytic inertia for the per-vertex diagonal so the mass is not counted
    /// twice. That is what makes it a PRECONDITIONER: it is cheap, it is
    /// symmetric positive definite by construction, and it is allowed to be
    /// wrong about the coupling because being wrong costs iterations and never
    /// the answer.
    ///
    /// THE FACTORIZATION IS ON THE HOST, in double, as the reference does it:
    /// it is one 6x6 per body, it carries the guards that keep a semi-definite
    /// block from producing a NaN, and a NaN here would spread on the
    /// preconditioner's first application.
    ///
    /// # Safety
    /// Every handle must name a live allocation sized for the scene; `blocks`
    /// and `factor` must each hold `36 * n_bodies` floats.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn build_precond<D: Device>(
        &mut self,
        device: &mut D,
        inputs: &PrecondInputs,
        blocks: &mut ppf_cts_compute::ReadbackBuffer<f32>,
        factor: &mut ppf_cts_compute::StagedBuffer<f32>,
        dt: f32,
        joints: Option<super::joints::View>,
    ) -> Result<(), Fault> {
        let bodies = self.map.n_bodies;
        if bodies == 0 {
            return Ok(());
        }
        let inertia = PdrdAssembleInertiaRowArgs {
            state: inputs.state,
            blocks: blocks.handle(),
            dt,
            count: bodies as u32,
            seam_arena_count: 0,
        };
        device.launch("pdrd.precond.inertia", &inertia, bodies as u32)?;

        let sandwich = PdrdAssembleSandwichRowArgs {
            vert_list: inputs.vert_list,
            prop: inputs.vertex_prop,
            state: inputs.state,
            rest_centered: inputs.rest_centered,
            dyn_index: inputs.dyn_index,
            dyn_offset: inputs.dyn_offset,
            dyn_value: inputs.dyn_value,
            fixed_index: inputs.fixed_index,
            fixed_offset: inputs.fixed_offset,
            fixed_value: inputs.fixed_value,
            row_count: inputs.rows,
            blocks: blocks.handle(),
            count: inputs.body_vertices,
            seam_arena_count: 0,
        };
        device.launch("pdrd.precond.sandwich", &sandwich, inputs.body_vertices)?;

        // Snapshot inertia+sandwich only. Full legacy self AND cross terms
        // are exported once into the joint CSR beside material contributions.
        // The original block-Jacobi assembly/order below remains unchanged.
        if let Some(raw)=self.staged.reduced_schwarz.raw_body(){
            device.copy(raw,0,blocks.handle(),0,36*bodies*4)?;
            // All preconditioner applications select the reduced model. Its
            // body base is now captured; the unused BJ self blocks and host
            // factors need not be assembled, downloaded, factored or uploaded.
            if self.staged.reduced_schwarz.active(){return Ok(());}
        }
        if let Some(view)=joints { view.add(device,2,blocks.handle(),blocks.handle())?; }
        blocks.download(device)?;
        {
            let assembled = blocks.host();
            let staged = factor.at();
            for body in 0..bodies {
                let mut block = [0.0f32; 36];
                block.copy_from_slice(&assembled[36 * body..36 * body + 36]);
                let f = super::lock_math::factor_reduced_block(&block);
                if f.iter().any(|value| !value.is_finite()) {
                    return Err(Fault::Platform {
                        call: "pdrd reduced preconditioner",
                        detail: format!("body {body}: nonfinite assembled block, scaling or factor"),
                    });
                }
                staged[36 * body..36 * body + 36].copy_from_slice(&f);
            }
        }
        factor.upload(device)
    }

    /// Freeze the packed nodal preconditioner for the upcoming Newton solve.
    pub unsafe fn prepare_mixed_precond<D: Device>(
        &mut self,
        device: &mut D,
        inverse_diagonal: Handle,
        arithmetic_eligible: bool,
    ) -> Result<(), Fault> {
        match self.staged.pcg_precision {
            PcgPrecision::MixedFp16Nodal => {}
            PcgPrecision::MixedFp16Arithmetic if arithmetic_eligible => {}
            _ => return Ok(()),
        }
        let observe_first_pack = self.staged.mixed_precond.len() == 0
            && std::env::var_os("PPF_PCG_TIMING_RECEIPT").as_deref() == Some(std::ffi::OsStr::new("1"));
        self.staged.mixed_precond.size(
            device, 4 * self.map.nrow, AllocLabel("pdrd.mixed_precond"),
        )?;
        let count = self.map.nrow as u32;
        let args = MixedHalfPackBlockArgs {
            inverse: inverse_diagonal,
            packed: self.staged.mixed_precond.handle(),
            count,
            seam_arena_count: 0,
        };
        device.run("pdrd.mixed_precond.pack", |encoder| encoder.elements(&args, count))?;
        if observe_first_pack {
            let mut packed = vec![0u32; self.staged.mixed_precond.len()];
            self.staged.mixed_precond.read(device, 0, &mut packed)?;
            let accepted = packed.chunks_exact(4).zip(&self.map.vbody)
                .filter(|(block, body)| **body == super::rigid_map::RIGID_UNSET && block[3] != 0)
                .count();
            eprintln!("PPF_PCG_PACK_RECEIPT accepted={} fallback={}", accepted, self.map.n_cloth - accepted);
        }
        Ok(())
    }

    fn mixed_precond_args(
        &self,
        inverse: Handle,
        residual: Handle,
        out: Handle,
    ) -> MixedHalfPdrdApplyArgs {
        MixedHalfPdrdApplyArgs {
            vertex_body: self.staged.vbody.handle(),
            cloth_offset: self.staged.cloth_off.handle(),
            inverse,
            packed: self.staged.mixed_precond.handle(),
            residual,
            out,
            count: self.map.nrow as u32,
            seam_arena_count: 0,
        }
    }

    pub unsafe fn prepare_arithmetic_status<D: Device>(
        &mut self,
        device: &mut D,
        record_status: bool,
    ) -> Result<(), Fault> {
        // With observations off, the ABI receives a valid placeholder that
        // the kernel never reads. The normal inner loop writes no status.
        let rows = if record_status { self.map.nrow } else { 1 };
        let status = &mut self.staged.mixed_arithmetic_status;
        status.size(device, rows, AllocLabel("pdrd.mixed_arithmetic_status"))?;
        if record_status {
            status.write(device, 0, &vec![0; rows])?;
        }
        Ok(())
    }

    pub unsafe fn observe_arithmetic_status<D: Device>(
        &self,
        device: &mut D,
    ) -> Result<(usize, usize), Fault> {
        let mut status = vec![0u32; self.map.nrow];
        self.staged.mixed_arithmetic_status.read(device, 0, &mut status)?;
        let half_rows = status.iter().filter(|row| **row & 1 != 0).count();
        let fallback_rows = status.iter().filter(|row| **row & 2 != 0).count();
        Ok((half_rows, fallback_rows))
    }

    /// Apply the preconditioner: `z = P^-1 r`, in two arms.
    ///
    /// THE BODY ARM RUNS FIRST AND THE CLOTH ARM SECOND, which is the order the
    /// reference issues them in and which matters only because both write `z`:
    /// each touches its own rows, the bodies' six-vectors and the free
    /// vertices' triples, so the two are disjoint and the order is a statement
    /// rather than a dependency.
    ///
    /// The cloth arm uses the per-vertex block-Jacobi inverse. When mixed
    /// precision is enabled, safe blocks use packed factors; the others use
    /// the same FP32 inverse. Both paths apply the operator in FP32.
    ///
    /// # Safety
    /// `residual` and `out` must hold `map.dim` floats.
    pub unsafe fn apply_precond<D: Device>(
        &mut self,
        device: &mut D,
        factor: Handle,
        inverse_diagonal: Handle,
        residual: Handle,
        out: Handle,
    ) -> Result<(), Fault> {
        if self.staged.reduced_schwarz.active(){
            return self.staged.reduced_schwarz.apply(device,residual,out);
        }
        if self.map.n_bodies > 0 {
            let body = PdrdPrecondBodyRowArgs {
                factor,
                residual,
                out,
                body_base: self.map.body_base as u32,
                count: self.map.n_bodies as u32,
                seam_arena_count: 0,
            };
            device.launch("pdrd.precond.body", &body, self.map.n_bodies as u32)?;
        }
        if self.staged.pcg_precision == PcgPrecision::MixedFp16Nodal {
            let args = self.mixed_precond_args(inverse_diagonal, residual, out);
            return device.run("pdrd.mixed_precond.apply", |encoder| {
                encoder.elements(&args, args.count)
            }).map(|_| ());
        }
        let cloth = PdrdPrecondClothRowArgs {
            vertex_body: self.staged.vbody.handle(),
            cloth_offset: self.staged.cloth_off.handle(),
            inverse_diagonal,
            residual,
            out,
            count: self.map.nrow as u32,
            seam_arena_count: 0,
        };
        device.launch("pdrd.precond.cloth", &cloth, self.map.nrow as u32)?;
        Ok(())
    }

    /// Same shared selection encoded inside Stage10 resident/graph batches.
    /// Callers retain the original tangent projectors around this operation.
    pub unsafe fn encode_precond<E:Encoder+?Sized>(&mut self,encoder:&mut E,factor:Handle,
        inverse_diagonal:Handle,residual:Handle,out:Handle)->Result<(),Fault>{
        self.encode_precond_selected(encoder, factor, inverse_diagonal, residual, out, None)
    }

    /// Some selects the CUDA arithmetic kernel, with optional observation.
    pub unsafe fn encode_precond_selected<E: Encoder + ?Sized>(
        &mut self, encoder: &mut E, factor: Handle, inverse_diagonal: Handle,
        residual: Handle, out: Handle, arithmetic: Option<bool>,
    ) -> Result<(), Fault> {
        if self.staged.reduced_schwarz.active(){return self.staged.reduced_schwarz.encode(encoder,residual,out);}
        if self.map.n_bodies>0{encoder.elements(&PdrdPrecondBodyRowArgs{factor,residual,out,
            body_base:self.map.body_base as u32,count:self.map.n_bodies as u32,seam_arena_count:0},self.map.n_bodies as u32)?;}
        if let Some(record_status) = arithmetic {
            let args = MixedHalfPdrdArithmeticArgs {
                vertex_body: self.staged.vbody.handle(),
                cloth_offset: self.staged.cloth_off.handle(),
                inverse: inverse_diagonal,
                packed: self.staged.mixed_precond.handle(),
                residual,
                out,
                status: self.staged.mixed_arithmetic_status.handle(),
                record_status: u32::from(record_status),
                count: self.map.nrow as u32,
                seam_arena_count: 0,
            };
            return encoder.elements(&args, args.count);
        }
        if self.staged.pcg_precision == PcgPrecision::MixedFp16Nodal {
            let args = self.mixed_precond_args(inverse_diagonal, residual, out);
            return encoder.elements(&args, args.count);
        }
        self.encode_precond_cloth_fp32(encoder, inverse_diagonal, residual, out)
    }

    /// Encode only the original FP32 cloth block, without touching body rows.
    pub unsafe fn encode_precond_cloth_fp32<E: Encoder + ?Sized>(
        &mut self, encoder: &mut E, inverse_diagonal: Handle,
        residual: Handle, out: Handle,
    ) -> Result<(), Fault> {
        encoder.elements(&PdrdPrecondClothRowArgs{vertex_body:self.staged.vbody.handle(),
            cloth_offset:self.staged.cloth_off.handle(),inverse_diagonal,residual,out,
            count:self.map.nrow as u32,seam_arena_count:0},self.map.nrow as u32)
    }

    /// Zero a reduced vector.
    ///
    /// THE ZEROS COME FROM THE CALLER rather than being allocated here, because
    /// this runs several times per CG iteration and the rule the driver is held
    /// to is that a per-step routine allocates nothing: the caller owns one
    /// `vec![0.0; dim]` for the solve's lifetime.
    fn clear<D: Device>(
        &mut self,
        device: &mut D,
        reduced: Handle,
        zero: &[f32],
    ) -> Result<(), Fault> {
        debug_assert!(zero.len() >= self.map.dim);
        let bytes = unsafe {
            std::slice::from_raw_parts(
                zero.as_ptr() as *const u8,
                self.map.dim * std::mem::size_of::<f32>(),
            )
        };
        device.write(reduced, 0, bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cvec::CVec;
    use crate::driver::launch::host_device;
    use crate::driver::state::SolverState;
    use crate::driver::test_scene::TestScene;
    use ppf_cts_compute::{AllocLabel, Buffer};

    /// A scene with one rigid body over two of its five vertices.
    fn scene_with_a_body() -> TestScene {
        let mut scene = TestScene::new(5);
        for i in 0..5 {
            scene.place(i, i as f32 * 0.25, 0.5, 0.0);
            scene.vertex_props_mut()[i].mass = 1.0;
        }
        scene.vertex_props_mut()[1].pdrd_body_index = 1;
        scene.vertex_props_mut()[2].pdrd_body_index = 1;
        scene.data.prop.pdrd_body =
            CVec::from(&[crate::data::PdrdBodyProp::default()][..]);
        scene
    }

    /// A free vertex survives a restrict-then-prolong round trip unchanged.
    ///
    /// THIS IS WHAT THE MAP IS FOR, and it is the property a wrong offset
    /// breaks: a free vertex's slot is its position among the FREE vertices, so
    /// a body sitting between free vertices shifts every later one. Get it
    /// wrong and this test puts one vertex's value on another, which is a
    /// silently wrong solve rather than a crash.
    ///
    /// A BODY'S VERTICES ARE DELIBERATELY NOT ASSERTED HERE. `restrict` is the
    /// FORCE restriction, so a body row accumulates its vertices' contributions
    /// rather than storing them, and prolonging that back does not return what
    /// went in. That is the reduction working, not failing: the six numbers are
    /// the body's, not the vertices'.
    #[test]
    fn a_free_vertex_survives_the_reduction_round_trip() {
        let scene = scene_with_a_body();
        let mut state = SolverState::default();
        let mut device = host_device();
        // Safety: the scene lives in its box for the whole test.
        unsafe { state.allocate(&mut device, &scene.data) }
            .expect("a scene with a body allocates");
        assert_eq!(state.rigid.n_bodies, 1);
        assert_eq!(state.rigid.n_cloth, 3);
        assert_eq!(state.rigid.dim, 15);

        let nrow = 5usize;
        let mut full: Buffer<f32> = Default::default();
        let mut back: Buffer<f32> = Default::default();
        let mut reduced: Buffer<f32> = Default::default();
        full.size(&mut device, 3 * nrow, AllocLabel("test.pdrd.full"))
            .expect("sizes");
        back.size(&mut device, 3 * nrow, AllocLabel("test.pdrd.back"))
            .expect("sizes");
        reduced
            .size(&mut device, state.rigid.dim, AllocLabel("test.pdrd.reduced"))
            .expect("sizes");
        let source: Vec<f32> = (0..3 * nrow).map(|k| 1.0 + k as f32).collect();
        full.write(&mut device, 0, &source).expect("uploads");
        // The rotated rest vectors are zero here, which makes a body's
        // prolongation its translation alone. That is enough for the free
        // vertices this test asserts on and keeps the body's own algebra out of
        // a test that is about the MAP.
        let zero = vec![0.0f32; state.rigid.dim.max(3 * nrow)];
        state
            .rigid_staged
            .prot
            .write(&mut device, 0, &zero[..3 * nrow])
            .expect("uploads");

        {
            let SolverState { rigid, rigid_staged, .. } = &mut state;
            let mut reduction = Reduction { map: rigid, staged: rigid_staged };
            // Safety: every handle names a live allocation of the stated length.
            unsafe {
                reduction
                    .restrict(&mut device, full.handle(), reduced.handle(), &zero)
                    .expect("restricts");
                reduction
                    .prolong(&mut device, reduced.handle(), back.handle())
                    .expect("prolongs");
            }
        }

        let mut round = vec![0.0f32; 3 * nrow];
        back.read(&mut device, 0, &mut round).expect("reads back");
        for vertex in [0usize, 3, 4] {
            for k in 0..3 {
                let slot = 3 * vertex + k;
                assert!(
                    (round[slot] - source[slot]).abs() < 1.0e-5,
                    "free vertex {vertex} component {k} came back as {} rather than {}. \
                     A free vertex's reduced slot is its position among the FREE \
                     vertices, so a body between them shifts every later one",
                    round[slot],
                    source[slot]
                );
            }
        }
    }

    /// A CONTACT-FREE PDRD SCENE BUILDS ITS PRECONDITIONER.
    ///
    /// The sandwich row takes the dynamic matrix's three buffers and this
    /// driver carries that matrix as an `Option` that is `None` when the scene
    /// configures no contact. Filling the three with `Handle::NONE` reads as
    /// "there is nothing here" and is not what a generated entry can accept: it
    /// resolves every buffer it is handed BEFORE the body runs, and
    /// `Handle::NONE` carries `u32::MAX` as its arena, so the dispatch trapped
    /// on `args->dyn_index.arena < args->seam_arena_count` in exactly the
    /// scenes that have nothing for it to read. `bl_world_scaling_pdrd` aborted
    /// there, and the assert names a generated file nobody wrote, which reads
    /// like the scene rather than like the field.
    ///
    /// Zero-length handles would clear the assert and still be wrong: the body
    /// indexes `dyn_offset[v]` and `dyn_offset[v + 1]` unconditionally, so the
    /// offsets must be a real `rows + 1` run of zeros and only the index and
    /// value arrays are empty. That is the reference's own shape, where
    /// `DynCSRMat::alloc(nrow, ...)` is always allocated and a contact-free
    /// scene simply has every row empty.
    ///
    /// The assertion is that this RETURNS, since the defect was an abort. The
    /// blocks it writes are the inertia term alone, which the reduced solve's
    /// own tests already cover.
    #[test]
    fn a_contact_free_body_builds_its_preconditioner() {
        let scene = scene_with_a_body();
        let mut state = SolverState::default();
        let mut device = host_device();
        // Safety: the scene lives in its box for the whole test.
        unsafe { state.allocate(&mut device, &scene.data) }
            .expect("a scene with a body allocates");

        let nrow = 5usize;
        let pattern = state.fixed_pattern_refs();
        let mut blocks: ppf_cts_compute::ReadbackBuffer<f32> = Default::default();
        let mut factor: ppf_cts_compute::StagedBuffer<f32> = Default::default();
        blocks
            .size(&mut device, 36, AllocLabel("test.pdrd.blocks"))
            .expect("sizes");
        factor
            .size(&mut device, 36, AllocLabel("test.pdrd.factor"))
            .expect("sizes");

        let SolverState { rigid, rigid_staged, prop_vertex, .. } = &mut state;
        // The three the defect was about, taken exactly as `step.rs` takes them
        // when `operator.dynamic` is `None`.
        let inputs = PrecondInputs {
            vert_list: rigid_staged.vert_list.handle(),
            vertex_prop: prop_vertex.handle(),
            state: rigid_staged.prot.handle(),
            rest_centered: rigid_staged.rest_centered.handle(),
            dyn_index: rigid_staged.empty_dyn_index.handle(),
            dyn_offset: rigid_staged.empty_dyn_offset.handle(),
            dyn_value: rigid_staged.empty_dyn_value.handle(),
            fixed_index: pattern.index,
            fixed_offset: pattern.offset,
            fixed_value: pattern.index,
            rows: nrow as u32,
            body_vertices: 2,
        };
        let mut reduction = Reduction { map: rigid, staged: rigid_staged };
        // Safety: every handle names a live allocation for the whole call.
        let built = unsafe {
            reduction.build_precond(&mut device, &inputs, &mut blocks, &mut factor, 0.01, None)
        };
        built.expect(
            "a contact-free PDRD scene must build its preconditioner rather than trap \
             resolving an absent dynamic matrix",
        );
    }

    /// PROLONG AND RESTRICT MUST BE ADJOINT: `<P u, y> == <u, P^T y>`.
    ///
    /// THIS IS WHAT MAKES THE REDUCED OPERATOR SYMMETRIC, and CG diverges
    /// without it. `R = P^T M P` is symmetric exactly when the two halves are
    /// each other's transpose; if they are not, `R` is some other matrix and
    /// the recurrence has no reason to converge. The symptom is a residual that
    /// falls for an iteration or two and then grows geometrically, which reads
    /// like a preconditioner problem and is not.
    ///
    /// THE ROTATED REST VECTORS ARE NON-ZERO HERE ON PURPOSE. With them zero a
    /// body's rows carry only its translation, the torque half of the Jacobian
    /// never runs, and the identity holds for a reason that says nothing about
    /// the rotation rows.
    #[test]
    fn prolong_and_restrict_are_adjoint() {
        let scene = scene_with_a_body();
        let mut state = SolverState::default();
        let mut device = host_device();
        // Safety: the scene lives in its box for the whole test.
        unsafe { state.allocate(&mut device, &scene.data) }.expect("allocates");
        let nrow = 5usize;
        let dim = state.rigid.dim;

        let mut u: Buffer<f32> = Default::default();
        let mut y: Buffer<f32> = Default::default();
        let mut pu: Buffer<f32> = Default::default();
        let mut pty: Buffer<f32> = Default::default();
        u.size(&mut device, dim, AllocLabel("t.u")).expect("sizes");
        y.size(&mut device, 3 * nrow, AllocLabel("t.y")).expect("sizes");
        pu.size(&mut device, 3 * nrow, AllocLabel("t.pu")).expect("sizes");
        pty.size(&mut device, dim, AllocLabel("t.pty")).expect("sizes");

        // Deterministic, and neither vector proportional to the other.
        let u_host: Vec<f32> = (0..dim).map(|k| 1.0 + 0.37 * k as f32).collect();
        let y_host: Vec<f32> =
            (0..3 * nrow).map(|k| 0.5 - 0.21 * k as f32).collect();
        u.write(&mut device, 0, &u_host).expect("uploads");
        y.write(&mut device, 0, &y_host).expect("uploads");
        // A non-trivial moment arm per vertex.
        let prot: Vec<f32> =
            (0..3 * nrow).map(|k| 0.11 * (k as f32) - 0.4).collect();
        state
            .rigid_staged
            .prot
            .write(&mut device, 0, &prot)
            .expect("uploads");

        let zero = vec![0.0f32; dim.max(3 * nrow)];
        {
            let SolverState { rigid, rigid_staged, .. } = &mut state;
            let mut reduction = Reduction { map: rigid, staged: rigid_staged };
            // Safety: every handle names a live allocation of the stated length.
            unsafe {
                reduction
                    .prolong(&mut device, u.handle(), pu.handle())
                    .expect("prolongs");
                reduction
                    .restrict(&mut device, y.handle(), pty.handle(), &zero)
                    .expect("restricts");
            }
        }

        let mut pu_host = vec![0.0f32; 3 * nrow];
        let mut pty_host = vec![0.0f32; dim];
        pu.read(&mut device, 0, &mut pu_host).expect("reads");
        pty.read(&mut device, 0, &mut pty_host).expect("reads");

        let left: f32 = pu_host.iter().zip(y_host.iter()).map(|(a, b)| a * b).sum();
        let right: f32 = u_host.iter().zip(pty_host.iter()).map(|(a, b)| a * b).sum();
        let scale = left.abs().max(right.abs()).max(1.0);
        assert!(
            (left - right).abs() < 1.0e-4 * scale,
            "<P u, y> is {left} and <u, P^T y> is {right}. The two are not \
             adjoint, so R = P^T M P is not symmetric and CG cannot converge on \
             it: the residual falls for an iteration or two and then grows"
        );
    }

    /// The seed restriction copies the cloth rows and leaves the bodies at zero.
    ///
    /// IT IS DELIBERATELY NOT `P^T`, and the difference matters: the force
    /// restriction accumulates a body's vertices into its six rows, which for a
    /// seed would be the body's displacement multiplied by its vertex count
    /// rather than the displacement. `solver.cu` states the same in the comment
    /// above its own `launch_seed_restrict`.
    #[test]
    fn the_seed_restriction_leaves_every_body_row_at_zero() {
        let scene = scene_with_a_body();
        let mut state = SolverState::default();
        let mut device = host_device();
        // Safety: as above.
        unsafe { state.allocate(&mut device, &scene.data) }.expect("allocates");

        let nrow = 5usize;
        let mut full: Buffer<f32> = Default::default();
        let mut reduced: Buffer<f32> = Default::default();
        full.size(&mut device, 3 * nrow, AllocLabel("test.pdrd.full2"))
            .expect("sizes");
        reduced
            .size(&mut device, state.rigid.dim, AllocLabel("test.pdrd.reduced2"))
            .expect("sizes");
        let source: Vec<f32> = (0..3 * nrow).map(|k| 1.0 + k as f32).collect();
        full.write(&mut device, 0, &source).expect("uploads");
        let zero = vec![0.0f32; state.rigid.dim];

        {
            let SolverState { rigid, rigid_staged, .. } = &mut state;
            let mut reduction = Reduction { map: rigid, staged: rigid_staged };
            // Safety: as above.
            unsafe {
                reduction
                    .seed_restrict(&mut device, full.handle(), reduced.handle(), &zero)
                    .expect("seeds");
            }
        }

        let mut seeded = vec![0.0f32; state.rigid.dim];
        reduced.read(&mut device, 0, &mut seeded).expect("reads back");
        // The three free vertices' nine floats are vertices 0, 3 and 4.
        for (slot, vertex) in [(0usize, 0usize), (3, 3), (6, 4)] {
            for k in 0..3 {
                assert!(
                    (seeded[slot + k] - source[3 * vertex + k]).abs() < 1.0e-5,
                    "free vertex {vertex} did not reach reduced slot {slot}"
                );
            }
        }
        for k in 0..6 {
            assert_eq!(
                seeded[state.rigid.body_base + k], 0.0,
                "the seed restriction must leave body row {k} at zero: the force \
                 restriction would sum the body's vertex seeds instead"
            );
        }
    }
}

impl Reduction<'_> {
    /// Export each body's rotation DOFs from a reduced vector.
    ///
    /// The caller integrates these onto the persistent running rotation, which
    /// is what makes the rigidify anchored.
    ///
    /// # Safety
    /// `reduced` must hold `map.dim` floats and `rotation_out` `3 * n_bodies`.
    pub unsafe fn extract_body_rotation<D: Device>(
        &mut self,
        device: &mut D,
        reduced: Handle,
        rotation_out: Handle,
    ) -> Result<(), Fault> {
        if self.map.n_bodies == 0 {
            return Ok(());
        }
        let args = super::kernels::PdrdExtractBodyRotationRowArgs {
            reduced,
            rotation_out,
            body_base: self.map.body_base as u32,
            count: self.map.n_bodies as u32,
            seam_arena_count: 0,
        };
        device.launch("pdrd.extract_body_rotation", &args, self.map.n_bodies as u32)?;
        Ok(())
    }
}

/// The per-DOF-group L1 norms of a reduced vector.
///
/// `out[0]` is the cloth block, the rows below `body_base`; `out[1 + b]` is body
/// `b`'s six reduced wrench rows.
///
/// WHY THE GROUPS EXIST AT ALL, and this is the rule the whole reduced solve
/// turns on: the reduced vector mixes INCOMMENSURABLE rows. A body's six are a
/// wrench and a free vertex's three are a force, and a heavy or fast body owns
/// the vector's L1 norm. Its 6x6 block is exactly preconditioned, so ONE CG
/// step annihilates it, and a single global relative residual then crosses the
/// tolerance at iteration 1 while the cloth still carries its full residual.
/// The symptom is a distant PDRD body wrecking the cloth for about sixty frames
/// and then looking fine once it stops accelerating.
///
/// No PRODUCTION caller for this host copy: the reduced solve takes its group
/// norms from the device through `group_l1_device` in `driver/pcg.rs`, which
/// dispatches the shared body `pcg_rigid_group_l1` in
/// `kernels/solver/pcg.kernel.cpp`. What reads this is the test
/// `the_groups_are_the_cloth_block_and_each_bodys_six` in this file's
/// `group_tests`, which checks the cut between the cloth block and each body's
/// six rows against a reduced vector whose three group norms are known.
#[allow(dead_code)]
pub fn group_l1(reduced: &[f32], body_base: usize, bodies: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), 1 + bodies);
    let mut cloth = 0.0f32;
    for value in &reduced[..body_base] {
        cloth += value.abs();
    }
    out[0] = cloth;
    for body in 0..bodies {
        let base = body_base + 6 * body;
        let mut sum = 0.0f32;
        for k in 0..6 {
            sum += reduced[base + k].abs();
        }
        out[1 + body] = sum;
    }
}

/// The WORST group's relative residual.
///
/// A group whose seeded initial residual is exactly zero has no scale of its
/// own: it starts solved and can only pick up residual through coupling to
/// another group, so it is measured against the whole system's initial scale.
///
/// A NaN MAPS TO INFINITY RATHER THAN BEING LEFT TO THE MAXIMUM. `max` drops a
/// NaN, so a residual that has gone non-finite would read as convergence; the
/// mapping makes it survive the reduction and trip the caller's finite check.
pub fn worst_relative_residual(current: &[f32], initial: &[f32], err0_all: f32) -> f32 {
    let mut worst = 0.0f32;
    for (now, start) in current.iter().zip(initial.iter()) {
        let scale = if *start > 0.0 { *start } else { err0_all };
        let ratio = now / scale;
        let ratio = if ratio.is_finite() { ratio } else { f32::INFINITY };
        if ratio > worst {
            worst = ratio;
        }
    }
    worst
}

#[cfg(test)]
mod group_tests {
    use super::*;

    /// The per-group norms separate the cloth from each body.
    #[test]
    fn the_groups_are_the_cloth_block_and_each_bodys_six() {
        // Two free vertices (six floats) then two bodies (six each).
        let reduced: Vec<f32> = vec![
            1.0, -1.0, 1.0, -1.0, 1.0, -1.0, // cloth: L1 = 6
            2.0, 0.0, 0.0, 0.0, 0.0, 0.0, // body 0: L1 = 2
            0.0, 0.0, 0.0, 0.0, 0.0, -3.0, // body 1: L1 = 3
        ];
        let mut out = [0.0f32; 3];
        group_l1(&reduced, 6, 2, &mut out);
        assert_eq!(out, [6.0, 2.0, 3.0]);
    }

    /// A HEAVY BODY MUST NOT HIDE AN UNCONVERGED CLOTH, which is the whole
    /// reason the residual is per group.
    ///
    /// The numbers below are the failure in miniature: a body whose initial
    /// residual is a thousand times the cloth's, annihilated in one step, while
    /// the cloth has barely moved. A single global ratio reads
    /// `(0 + 1) / (1000 + 1)`, about 1e-3, and crosses any ordinary tolerance;
    /// the per-group rule reads the cloth's own 1.0 and does not.
    #[test]
    fn a_solved_heavy_body_does_not_hide_an_unconverged_cloth() {
        let initial = [1.0f32, 1000.0];
        let current = [1.0f32, 0.0];
        let err0_all: f32 = initial.iter().sum();
        let worst = worst_relative_residual(&current, &initial, err0_all);
        assert!(
            (worst - 1.0).abs() < 1.0e-6,
            "the worst group is the cloth at 1.0, got {worst}"
        );
        let global: f32 = current.iter().sum::<f32>() / err0_all;
        assert!(
            global < 1.0e-2,
            "the global ratio is {global}, which is what a collapsed tolerance \
             would accept as converged"
        );
    }

    /// A group that starts solved is measured against the whole system.
    #[test]
    fn a_group_that_starts_solved_borrows_the_systems_scale() {
        let initial = [0.0f32, 4.0];
        let current = [2.0f32, 0.0];
        // err0_all is 4, so the first group's ratio is 2/4 rather than 2/0.
        let worst = worst_relative_residual(&current, &initial, 4.0);
        assert!(
            (worst - 0.5).abs() < 1.0e-6,
            "a group with no scale of its own borrows the system's, got {worst}"
        );
    }

    /// A NON-FINITE residual survives the maximum instead of reading as
    /// convergence.
    #[test]
    fn a_non_finite_residual_becomes_infinite_rather_than_vanishing() {
        let initial = [1.0f32, 1.0];
        let current = [f32::NAN, 0.0];
        let worst = worst_relative_residual(&current, &initial, 2.0);
        assert!(
            worst.is_infinite(),
            "a NaN group must map to infinity, got {worst}: `max` drops a NaN, \
             so leaving it would read as convergence"
        );
    }
}
