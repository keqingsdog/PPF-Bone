//! Stage10 candidate preparation for full reduced Schwarz.
//!
//! Full original operator projection plus complete frozen joint blocks, with
//! explicit remaining body seed symmetry/regularization delta. Global A and
//! original outer projectors remain unchanged.
use ppf_cts_compute::{AllocLabel, Buffer, Device, Fault, Handle, ReadbackBuffer};
use super::{kernels::ReducedSchwarzJointExportArgs, soft_joints};

fn refusal(detail: impl Into<String>) -> Fault {
    Fault::Platform { call: "reduced Schwarz frozen joint export", detail: detail.into() }
}

/// Export ownership is separate from SoftJoints, so the unchanged global
/// operator's buffers and matvec cannot acquire candidate preconditioner state.
#[derive(Default)]
pub struct FrozenJointBlocks {
    scalar_ids: Buffer<u32>,
    local_hessian: Buffer<f32>,
    scalar_count: Buffer<u32>,
    fault: ReadbackBuffer<u32>,
    // Export resolves every pointer even when curves are disabled. CurveHistory
    // contains u64 and needs its own typed 8-byte aligned lifetime.
    inactive_history: Buffer<crate::restoring_curve_gpu::CurveHistory>,
    revision: u64,
    ready: bool,
    records: u32,
}

#[derive(Clone, Copy)]
pub struct FrozenJointBlockView {
    pub scalar_ids: Handle,
    pub local_hessian: Handle,
    pub scalar_count: Handle,
    pub records: u32,
    pub revision: u64,
}

impl FrozenJointBlocks {
    pub(super) fn release(&mut self,d:&mut impl Device)->Result<(),Fault>{
        self.ready=false;self.records=0;let mut first=None;
        macro_rules! release {($x:expr)=>{if let Err(e)=$x.free(d){if first.is_none(){first=Some(e);}}};}
        release!(self.scalar_ids);release!(self.local_hessian);release!(self.scalar_count);release!(self.inactive_history);release!(self.fault);
        match first {Some(e)=>Err(e),None=>Ok(())}
    }

    /// No material records still requires valid pointer fields in generated
    /// model entries. Keep a private valid zero-record export, never NONE.
    pub fn empty<D: Device>(&mut self,device:&mut D)->Result<FrozenJointBlockView,Fault>{
        self.ready=false;
        self.scalar_ids.size(device,24,AllocLabel("reduced_schwarz.empty_ids"))?;
        self.local_hessian.size(device,576,AllocLabel("reduced_schwarz.empty_hessian"))?;
        self.scalar_count.size(device,1,AllocLabel("reduced_schwarz.empty_count"))?;
        self.revision=self.revision.checked_add(1).ok_or_else(||refusal("revision overflow"))?;
        self.records=0;self.ready=true;
        self.view().ok_or_else(||refusal("empty export readiness"))
    }
    /// Refresh at every frozen Newton cache. Never key a numerical factor on
    /// counts alone: poses, controls, sample time and dt can change at fixed
    /// topology. A failed export invalidates access to the prior revision.
    ///
    /// # Safety
    /// The source view must name the current live frozen cache and records.
    pub unsafe fn refresh<D: Device>(
        &mut self, device: &mut D, source: soft_joints::View,
    ) -> Result<FrozenJointBlockView, Fault> {
        self.ready = false;
        if !source.ready || !source.dt.is_finite() || source.dt <= 0.0 {
            return Err(refusal("joint cache is not frozen or dt is invalid"));
        }
        if u64::from(source.base) != 3 * u64::from(source.free_count) ||
           u64::from(source.dim) != u64::from(source.base) + 6 * u64::from(source.bodies) {
            return Err(refusal("joint reduced layout is inconsistent"));
        }
        if source.extension_count != 0 && source.extension_count != source.count {
            return Err(refusal("extension count differs from frozen record count"));
        }
        if source.count > u32::MAX / 576 {
            return Err(refusal("joint count overflows scalar kernel indexing"));
        }
        if let Some(curves) = source.curves {
            if curves.dt.to_bits() != source.dt.to_bits() {
                return Err(refusal("restoring context dt differs from frozen operator"));
            }
        }
        let n = source.count as usize;
        let values = n.checked_mul(576).ok_or_else(|| refusal("joint block extent overflow"))?;
        let ids = n.checked_mul(24).ok_or_else(|| refusal("joint index extent overflow"))?;
        self.scalar_ids.size(device, ids, AllocLabel("reduced_schwarz.joint_scalar_ids"))?;
        self.local_hessian.size(device, values, AllocLabel("reduced_schwarz.joint_hessian"))?;
        self.scalar_count.size(device, n, AllocLabel("reduced_schwarz.joint_scalar_count"))?;
        self.fault.size(device, n, AllocLabel("reduced_schwarz.joint_export_fault"))?;
        if n != 0 {
            // All pointer fields are resolved before the entry tests counts.
            // An inactive stream uses existing valid dummy storage, never an
            // empty allocation's invalid handle.
            let cached = source.resident_cache_ready();
            let curves = source.curves;
            if curves.is_none() {
                self.inactive_history.size(device,1,AllocLabel("reduced_schwarz.inactive_curve_history"))?;
                self.inactive_history.write(device,0,&[crate::restoring_curve_gpu::CurveHistory::default()])?;
            }
            let args = ReducedSchwarzJointExportArgs {
                records: source.records, material_spin_context: source.material_spin_context, record_count: source.count, cache: source.cache,
                vertex_to_free: source.vertex_to_free, vertex_count: source.vertices,
                free_count: source.free_count, rigid_base: source.base, dim: source.dim,
                index_cache: if cached { source.index_cache } else { source.dummy },
                index_ready: (cached && source.index_ready) as u32,
                source_mode: cached as u32, token: if cached { source.hessian_epoch } else { 0 },
                hessian_dt: source.hessian_dt,
                limits: if source.extension_count != 0 { source.limits } else { source.dummy },
                drives: if source.drive_count != 0 { source.drives } else { source.dummy },
                drive_count: source.drive_count,
                keys: if source.drive_count != 0 { source.keys } else { source.dummy },
                extension_count: source.extension_count, sample_time_s: source.sample_time_s,
                dt: source.dt, curves_enabled: curves.is_some() as u32,
                descriptors: curves.map_or(source.dummy, |c| c.descriptors),
                nodes: curves.map_or(source.dummy, |c| c.nodes),
                bindings: curves.map_or(source.dummy, |c| c.bindings),
                history: curves.map_or(self.inactive_history.handle(), |c| c.history),
                descriptor_count: curves.map_or(0, |c| c.descriptor_count),
                node_count: curves.map_or(0, |c| c.node_count),
                binding_count: curves.map_or(0, |c| c.binding_count),
                history_count: curves.map_or(0, |c| c.history_count),
                scalar_ids: self.scalar_ids.handle(),
                local_hessian: self.local_hessian.handle(), scalar_count: self.scalar_count.handle(),
                fault: self.fault.handle(), count: source.count, seam_arena_count: 0,
            };
            device.launch("reduced_schwarz.joint_export", &args, source.count)?;
            self.fault.download(device)?;
            if let Some((row, code)) = self.fault.host().iter().enumerate().find(|(_, code)| **code != 0) {
                return Err(refusal(format!("row {row}: native fault {code}")));
            }
        }
        self.revision = self.revision.checked_add(1).ok_or_else(|| refusal("revision overflow"))?;
        self.records = source.count;
        self.ready = true;
        self.view().ok_or_else(|| refusal("export readiness invariant"))
    }

    pub fn view(&self) -> Option<FrozenJointBlockView> {
        self.ready.then(|| FrozenJointBlockView {
            scalar_ids: self.scalar_ids.handle(), local_hessian: self.local_hessian.handle(),
            scalar_count: self.scalar_count.handle(), records: self.records, revision: self.revision,
        })
    }
}

/// Stored in Staged so ordinary, hard and inverse Reduction callers observe
/// the exact same frozen preconditioner. No global operator source is changed.
#[derive(Default)]
pub struct Preconditioner {
    selected:bool,
    current:Option<super::reduced_model::Revision>,
    sequence:u64,
    exports:FrozenJointBlocks,
    legacy_exports:super::legacy_schwarz::LegacyJointBlocks,
    model:super::reduced_model_backend::ReducedModel,
    joint_patches:super::joint_patch::JointPatches,
    joint_patches_active:bool,
    hierarchy:super::reduced_coarse_backend::Hierarchy,
    geometry:super::reduced_coarse_backend::FrozenGeometry,
    constraints:super::constraint_space::Frozen,
    requested_levels:u32,
    effective_requested_levels:u32,
    coarse_ready:bool,
    coarse_rank_refusal:Option<String>,
    raw_body:Buffer<f32>,
    row_to_vertex:Buffer<u32>,
    vertex_to_free:Buffer<u32>,
    builds:u64,
    setup_ms:f64,
    setup_started:Option<std::time::Instant>,
    direct_apply_ms:f64,
    diagnostic_done:bool,
}

impl Preconditioner {
    pub fn active(&self)->bool{self.selected}
    pub fn raw_body(&self)->Option<Handle>{self.selected.then(||self.raw_body.handle())}

    pub fn select<D:Device>(&mut self,device:&mut D,map:&super::rigid_map::RigidMap,
        selected:bool,levels:u32)->Result<(),Fault>{
        self.selected=false;self.current=None;self.setup_started=None;
        self.joint_patches.disable();self.joint_patches_active=false;
        self.coarse_ready=false;self.coarse_rank_refusal=None;
        self.requested_levels=levels;
        self.effective_requested_levels=if levels==0 {2}else{levels};
        if !selected{return Ok(());}
        self.setup_started=Some(std::time::Instant::now());
        let map_free=soft_joints::free_indices(map)?;
        let mut inverse=vec![u32::MAX;map.n_cloth];
        for (vertex,&row) in map_free.iter().enumerate(){if row!=u32::MAX{inverse[row as usize]=vertex as u32;}}
        self.row_to_vertex.size(device,map.n_cloth.max(1),AllocLabel("reduced_schwarz.row_to_vertex"))?;
        if !inverse.is_empty(){self.row_to_vertex.write(device,0,&inverse)?;}
        self.vertex_to_free.size(device,map.nrow.max(1),AllocLabel("reduced_schwarz.vertex_to_free"))?;
        if !map_free.is_empty(){self.vertex_to_free.write(device,0,&map_free)?;}
        self.raw_body.size(device,(36*map.n_bodies).max(36),AllocLabel("reduced_schwarz.raw_body"))?;
        self.sequence=self.sequence.checked_add(1).ok_or_else(||refusal("Newton revision overflow"))?;
        self.selected=true;Ok(())
    }

    /// Rebuild each frozen Newton cache, even at unchanged topology/counts.
    pub unsafe fn rebuild<D:Device>(&mut self,device:&mut D,map:&super::rigid_map::RigidMap,
        operator:&super::operator::Operator,empty:Handle,view:super::joints::View,
        vertex_body:Handle,rotated_rest:Handle,vertices:Handle,dof_mask:Handle,
        constraint_inputs:super::constraint_space::Inputs,enabled_roots:&[u32])->Result<(),Fault>{
        if !self.selected{return Ok(());}
        self.current=None;
        let started=self.setup_started.take().unwrap_or_else(std::time::Instant::now);
        let material=match view.soft{Some(soft)=>self.exports.refresh(device,soft)?,None=>self.exports.empty(device)?};
        let joints=self.legacy_exports.refresh(device,view,material)?;
        let r=operator.rows_view(empty);
        // Upper bounds from live allocation extents; the model counts exact
        // filtered CSR entries on device before filling. Pair storage is uint2.
        if r.transpose_pair.size%2!=0 {
            return Err(refusal("source CSR transpose pair has odd element count"));
        }
        // Handle.size is a logical ELEMENT count; only Handle.off is bytes.
        // The transpose list is a flat u32 pair stream: two elements/block.
        let blocks=u64::from(r.dynamic_index.size)
            .checked_add(u64::from(r.reference_index.size))
            .and_then(|n|n.checked_add(u64::from(r.fixed_index.size)))
            .and_then(|n|n.checked_add(u64::from(r.transpose_pair.size)/2))
            .and_then(|n|n.checked_add(map.nrow as u64))
            .ok_or_else(||refusal("source CSR block capacity overflows u64"))?;
        let revision=super::reduced_model::Revision{operator:self.sequence,mapping:self.sequence,joints:joints.revision};
        let constraints=self.constraints.rebuild(device,constraint_inputs,self.row_to_vertex.handle(),dof_mask,
            map.n_cloth as u32,map.n_bodies as u32,enabled_roots,revision)?;
        if super::constraint_space::diagnostic_selected(revision)?{super::constraint_space::check_q(device,constraints,enabled_roots)?;}
        let inputs=super::reduced_model_backend::Inputs{constraints,rows:r,row_to_vertex:self.row_to_vertex.handle(),
            vertex_to_free:self.vertex_to_free.handle(),raw_body:self.raw_body.handle(),
            vertex_body,rotated_rest,body_state:view.state,dt:view.dt,
            source_rows:u32::try_from(map.nrow).map_err(|_|refusal("full source rows exceed u32"))?,
            layout:super::reduced_model::Layout{cloth:map.n_cloth as u32,bodies:map.n_bodies as u32,
                records:joints.records,source_blocks:blocks}};
        self.model.rebuild(device,inputs,joints,revision)?;
        // A failed optional build chooses the unchanged baseline before PCG;
        // the choice and all factors remain fixed for that Newton solve.
        self.joint_patches.disable();self.joint_patches_active=false;
        if std::env::var_os("PPF_JOINT_PATCH").is_some_and(|v|v=="1") {
            let attempted=(|| -> Result<(),Fault> {
                let model=self.model.model_view().ok_or_else(||refusal("missing frozen model CSR"))?;
                let owners=self.model.domain_owners(device)?;
                self.joint_patches.rebuild(device,model,joints,constraints,&owners)
            })();
            match attempted {
                Ok(())=>self.joint_patches_active=true,
                Err(fault)=>{self.joint_patches.disable();eprintln!("JOINT_PATCH_FALLBACK {fault:?}");}
            }
        }
        if !self.diagnostic_done && std::env::var_os("PPF_SCHWARZ_FROZEN_CHECK").is_some(){
            if std::env::var("PPF_SCHWARZ_FROZEN_CHECK").ok().as_deref()!=Some("1"){
                return Err(refusal("PPF_SCHWARZ_FROZEN_CHECK must be exactly 1"));
            }
            let model=self.model.diagnostic_raw_model_view().ok_or_else(||refusal("diagnostic raw CSR or same-revision coalescing proof missing"))?;
            let (seed,inertia)=self.model.diagnostic_seed_handles();
            let report=super::frozen_projection_check::run(device,map,operator,view,inputs,model,seed,inertia)?;
            eprintln!("SCHWARZ_FROZEN_PROJECTION_CHECK {}",report);
            if joints.records>material.records{
                let legacy=super::legacy_action1_equivalence::run(device,view,joints,material.records)?;
                eprintln!("SCHWARZ_FROZEN_LEGACY_CHECK {}",legacy);
            }
            self.diagnostic_done=true;
        }
        self.coarse_ready=false;self.coarse_rank_refusal=None;
        if self.effective_requested_levels>1 {
            let fine=self.model.model_view().ok_or_else(||refusal("missing current model CSR"))?;
            let (offset,column)=self.model.graph_snapshot().ok_or_else(||refusal("missing current model graph"))?;
            let weights=self.model.graph_weights().ok_or_else(||refusal("missing exact CSR multiplicities"))?;
            let geometry=self.geometry.rebuild(device,self.row_to_vertex.handle(),vertices,
                view.state,dof_mask,map.n_cloth as u32,fine.rows)?;
            match self.hierarchy.rebuild(device,fine,map.n_cloth as u32,map.n_bodies as u32,
                &offset,&column,&weights,geometry,self.effective_requested_levels-1,constraints) {
                Ok(())=>self.coarse_ready=true,
                Err(Fault::Platform{call:"reduced coarse rank ambiguity",detail})=>{
                    self.coarse_rank_refusal=Some(detail);
                }
                Err(other)=>return Err(other),
            }
        }
        self.current=Some(revision);self.builds+=1;self.setup_ms+=started.elapsed().as_secs_f64()*1000.;
        Ok(())
    }

    pub unsafe fn encode<E:ppf_cts_compute::Encoder+?Sized>(&mut self,encoder:&mut E,residual:Handle,out:Handle)->Result<(),Fault>{
        let revision=self.current.ok_or_else(||refusal("selected reduced Schwarz has no current frozen factors"))?;
        if !self.selected{return Err(refusal("reduced Schwarz encode called while deselected"));}
        self.model.encode(encoder,residual,out,revision)?;
        if self.joint_patches_active {self.joint_patches.encode(encoder,residual,out,revision)?;}
        if self.coarse_ready {self.hierarchy.encode_add(encoder,residual,out,revision)?;}
        Ok(())
    }

    pub unsafe fn apply<D:Device>(&mut self,device:&mut D,residual:Handle,out:Handle)->Result<(),Fault>{
        let started=std::time::Instant::now();
        device.run("reduced_schwarz.precondition",|encoder|self.encode(encoder,residual,out))?;
        self.model.check(device)?;
        if self.joint_patches_active {self.joint_patches.check(device)?;}
        if self.coarse_ready {self.hierarchy.check(device)?;}
        self.direct_apply_ms+=started.elapsed().as_secs_f64()*1000.;Ok(())
    }

    pub fn check<D:Device>(&mut self,device:&mut D)->Result<(),Fault>{
        if self.selected{self.model.check(device)?;if self.joint_patches_active{self.joint_patches.check(device)?;}if self.coarse_ready{self.hierarchy.check(device)?;}}Ok(())
    }

    pub fn receipt<D:Device>(&mut self,device:&mut D,label:&str)->Result<(),Fault>{
        if !self.selected{return Ok(());}
        let receipt=self.model.check(device)?;let stats=self.model.stats();
        if self.joint_patches_active {self.joint_patches.check(device)?;}
        if self.coarse_ready {self.hierarchy.check(device)?;}
        eprintln!("REDUCED_SCHWARZ_RECEIPT {}",serde_json::json!({"schema":"ppf.reduced-schwarz.v1",
            "label":label,"joint_patch_active":self.joint_patches_active,"joint_patch_count":self.joint_patches.count(),"levels":if self.coarse_ready {self.hierarchy.levels()+1}else{1},
            "requested_levels":self.requested_levels,"effective_requested_levels":self.effective_requested_levels,
            "coarse_rank_refusal":self.coarse_rank_refusal,"builds":self.builds,"newton_revision":self.sequence,
            "device_apply_dispatches":receipt.dispatches,"device_domain_successes":receipt.domain_successes,
            "device_domain_faults":receipt.domain_faults,"rows":stats.rows,"blocks":stats.blocks,"unique_blocks":stats.blocks,"raw_blocks":stats.raw_blocks,"domains":stats.domains,
            "coalesce_topology_cache_hit":stats.coalesce_cache_hit,"coalesce_topology_wall_ms":stats.coalesce_topology_ms,
            "coalesce_numeric_wall_ms":stats.coalesce_numeric_ms,"coalesce_total_wall_ms":stats.coalesce_total_ms,
            "edge_weight_mode":std::env::var("PPF_SCHWARZ_EDGE_WEIGHT").unwrap_or_else(|_|"structural".into()),
            "edge_weight_wall_ms":stats.edge_weight_ms,"partition_reused":stats.partition_reused,
            "partition_wall_ms":stats.partition_ms,
            "device_counter_scope":"current frozen Newton since rebuild, including hard/inverse and auxiliary stopped graph applies",
            "setup_wall_ms_cumulative":self.setup_ms,"direct_apply_and_check_wall_ms":self.direct_apply_ms,
            "graph_apply_gpu_ms":serde_json::Value::Null,"timing_scope":"setup and direct submissions include waits; graph cost remains in total PCG/end-to-end",
            "constraint_space":"same-Newton original FEM mask/body6 basis/enabled root Q; local QDQ+I-Q and Q-projected coarse modes",
            "model":"full Jt(A+B+C)J + full frozen material/legacy joint Hessians + physical Galerkin additive hierarchy + explicit body seed delta",
            "seed_delta":"body seed minus original analytic inertia minus seed-counted dynamic/fixed full-vertex diagonal projection; empty bodies retain original identity seed; joints symmetrized",
            "body_shifted":stats.body_shifted,"body_shift_max":stats.body_shift_max,
            "local_jittered":stats.local_jittered,"local_jitter_max":stats.local_jitter_max}));Ok(())
    }
}
