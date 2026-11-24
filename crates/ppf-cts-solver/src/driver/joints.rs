//! Native reduced-space V2 joints. Empty records dispatch nothing.
//!
//! Three parallel record families share one view: the 44-byte ball anchor, the
//! 72-byte isotropic angular record, and the 88-byte directional (per-axis)
//! angular control added in A1. Every family is dispatched only when it has
//! records, so a V2 scene never dispatches an unallocated new family.
//!
//! GEOMETRY FAULTS ARE TYPED, NEVER SILENT. A requested row whose live chart is
//! unusable writes a code into a per-joint fault lane, and the driver reads that
//! lane at each frozen Newton solve — after the CURRENT pose has been fitted and
//! before the preconditioner that consumes J^T H J, so a requested term can
//! never be silently missing from an accepted step. The reference pose is a
//! separate, SUBSTEP-START fit taken once in `begin_step`, which is why the two
//! are distinct buffers.
use ppf_cts_compute::{AllocLabel, Buffer, Device, Encoder, EncoderExt, Fault, Handle, ReadbackBuffer};
use crate::joint_gpu::BallAnchorGpu;
use crate::joint_angular_gpu::AngularJointGpuV2;
use crate::joint_angular_control_gpu::AngularControlGpu;
use crate::joint_angular_extension_gpu as ext;
use super::kernels::{JointBallApplyArgs, JointAngularApplyArgs, JointAngularControlApplyArgs, JointAngularExtensionApplyArgs};

// Neutral upload PODs only: canonical f64-time keys are never uploaded.
unsafe impl ppf_cts_compute::Pod for ext::AngularSoftLimitGpuV1 {}
unsafe impl ppf_cts_compute::Pod for ext::AngularDriveGpuV1 {}
unsafe impl ppf_cts_compute::Pod for ext::AngularDriveDeviceKeyV1 {}

/// Mirrors the kernel's `CONTROL_FAULT_*` codes so a report names the cause.
const FAULT_NAMES: [&str; 8] = [
    "none",
    "chart unusable at the current pose",
    "chart unusable at the reference pose",
    "body index out of range",
    "reduced offset out of range",
    "finite current twist outside open chart domain",
    "finite current swing outside open chart domain",
    "continuous twist history invalid or uninitialized",
];

// BEGIN PURE CHART RETRY HELPERS
/// Native detailed fault codes only; an unrelated fault anywhere blocks retry.
fn candidate_domain_lane(stage:&str,lane:&[u32])->bool {
    stage=="candidate_after_B32" && lane.iter().any(|c|matches!(c,5|6|134|135))
        && lane.iter().all(|c|matches!(c,0|5|6|134|135))
}
pub(super) fn chart_retry_dt(dt:f32,start:f64)->Option<f32> {
    let next=dt*0.5;
    if dt.is_finite() && dt>0.0 && next.is_finite() && next>0.0 && next<dt
        && start.is_finite() && (start+f64::from(next)).is_finite() && start+f64::from(next)>start {
        Some(next)
    } else {None}
}
// END PURE CHART RETRY HELPERS

#[derive(Clone,Debug,PartialEq,Eq)]
pub(super) struct CandidateFaultToken { pub trial:u64, pub start_bits:u64, pub dt_bits:u32, curves:Vec<(&'static str,usize,u32)>, control:Vec<u32>, pub reason:CandidateRetryReason }
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub(super) enum CandidateRetryReason { GeometryBank, UncertifiedPath(&'static str) }
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub(crate) struct TwistSampleFailure { pub joint_index:usize, pub code:u32, pub body_a:u32, pub body_b:u32, pub call_sequence:u64 }
/// An action-4 B31 Iterate failure, tied to the successful immediately prior
/// Target sample and the untouched, freshly downloaded full sampler bank.
#[derive(Clone,Debug,PartialEq,Eq)]
pub(super) struct SampleSwingTrial {
    pub generation:u64, pub start_bits:u64, pub dt_bits:u32,
    pub origin_sequence:u64, pub sample:TwistSampleFailure,
    lane:Vec<u32>, selected:Vec<bool>, reference:Handle,
}
#[cfg(test)]
pub(super) fn test_chart_token(h:f32)->CandidateFaultToken {
    CandidateFaultToken{trial:1,start_bits:1f64.to_bits(),dt_bits:h.to_bits(),curves:vec![],control:vec![],reason:CandidateRetryReason::UncertifiedPath("B31")}
}
#[cfg(test)]
pub(super) fn test_swing_token(h:f32)->SampleSwingTrial {
    SampleSwingTrial{generation:1,start_bits:1f64.to_bits(),dt_bits:h.to_bits(),origin_sequence:7,
        sample:TwistSampleFailure{joint_index:0,code:6,body_a:0,body_b:1,call_sequence:8},lane:vec![6],selected:vec![true],reference:Handle::NONE}
}
fn twist_sample_failure(lane:&[u32],endpoints:&[(u32,u32)],sequence:u64)->Option<TwistSampleFailure> {
    let (joint_index,code)=lane.iter().enumerate().find(|(_,v)|**v!=0).map(|(i,v)|(i,*v))?;
    let &(body_a,body_b)=endpoints.get(joint_index)?;
    Some(TwistSampleFailure{joint_index,code,body_a,body_b,call_sequence:sequence})
}
fn twist_sample_fault(index:usize,code:u32)->Fault {
    Fault::Platform{call:"continuous twist sample",detail:format!("joint {index}: chart fault {code}")}
}
fn observe_best_effort(observer:impl FnOnce()) { let _=std::panic::catch_unwind(std::panic::AssertUnwindSafe(observer)); }
fn twist_sample_fault_with_capture(index:usize,code:u32,capture:impl FnOnce()->Result<(),Fault>)->Fault {
    observe_best_effort(||if let Err(error)=capture() { eprintln!("JOINT_CHART_FAILURE_CAPTURE_ERROR {error}"); });
    twist_sample_fault(index,code)
}
#[derive(Default)]
pub struct Joints {
    anchored_flight: Option<super::anchored_flight::Components>,
    shadow_identity:Option<([u8;36],[u8;64])>,
    pub soft:super::soft_joints::SoftJoints,
    pub curves:super::restoring::Curves,
    frozen_ball_empty_history:Buffer<crate::restoring_curve_gpu::CurveHistory>,
    frozen_ball_factors:Buffer<f32>, frozen_ball_offsets:Buffer<u32>, frozen_ball_epoch:Buffer<u32>, frozen_ball_fault:ReadbackBuffer<u32>, frozen_ball_generation:u32,
    frozen_angular_factors:Buffer<f32>, frozen_angular_offsets:Buffer<u32>, frozen_angular_epoch:Buffer<u32>, frozen_angular_fault:ReadbackBuffer<u32>, frozen_angular_generation:u32,
    angular_tangent_factors:Buffer<f32>, angular_tangent_published:Buffer<u32>, angular_tangent_generation:u32,
    control_tangent_factors:Buffer<f32>, control_tangent_offsets:Buffer<u32>, control_tangent_published:Buffer<u32>, control_tangent_generation:u32,
    extension_tangent_factors:Buffer<f32>, extension_tangent_offsets:Buffer<u32>, extension_tangent_published:Buffer<u32>, extension_tangent_generation:u32,
    control_tangent_fault:ReadbackBuffer<u32>, extension_tangent_fault:ReadbackBuffer<u32>,
    drive_offsets:Buffer<u32>,
    ball: Buffer<BallAnchorGpu>, angular: Buffer<AngularJointGpuV2>,
    control: Buffer<AngularControlGpu>, fault: ReadbackBuffer<u32>,
    twist_context: Buffer<crate::continuous_twist::AngularTwistEvalContext>,
    material_spin_context: Buffer<crate::material_spin_history::MaterialSpinContext>,
    material_spin_sample: ReadbackBuffer<f32>,
    material_spin_fault: ReadbackBuffer<u32>,
    twist_sample_values: ReadbackBuffer<f32>,
    twist_sample_faults: ReadbackBuffer<u32>,
    // A failing Iterate lane must remain untouched while its accepted
    // substep-start reference is independently sampled on the real device.
    twist_reference_values: ReadbackBuffer<f32>,
    twist_reference_faults: ReadbackBuffer<u32>,
    swing_candidate: Option<SampleSwingTrial>,
    last_twist_sample_failure: Option<TwistSampleFailure>,
    twist_sample_sequence: u64,
    twist_sample_selection: Buffer<f32>,
    twist_endpoints: Vec<(u32,u32)>,
    twist_sample_fit: Buffer<crate::data::PdrdRigidState>,
    twist_sample_fit_scratch: Buffer<f32>,
    twist_virtual_start: Buffer<crate::data::PdrdRigidState>,
    twist_virtual_end: Buffer<crate::data::PdrdRigidState>,
    reference: Buffer<crate::data::PdrdRigidState>,
    limits: Buffer<ext::AngularSoftLimitGpuV1>,
    drives: Buffer<ext::AngularDriveGpuV1>,
    device_keys: Buffer<ext::AngularDriveDeviceKeyV1>,
    drive_count: u32,
    start_time_s: f64,
    end_time_s: Option<f64>,
    accepted_twist: Option<crate::continuous_twist::AcceptedTwist>,
    trial_twist: Option<crate::continuous_twist::TrialTwist>,
    twist_selected: Option<Vec<bool>>,
    accepted_material_spin: Option<crate::material_spin_history::AcceptedMaterialSpin>,
    trial_material_spin: Option<crate::material_spin_history::TrialMaterialSpin>,
    material_spin_selected: Option<Vec<bool>>,
    material_spin_retry_dt: Option<f32>,
    candidate_domain_dt: Option<f32>,
    candidate_token: Option<CandidateFaultToken>,
    candidate_trial: u64,
    candidate_fatal: bool,
}
#[derive(Clone, Copy)]
pub struct FrozenAngularView { factors:Handle, offsets:Handle, published:Handle, fault:Handle, epoch:u32 }
/// Per-row 3x3 isotropic V2 Hessians prepared for one current Newton pose.
/// The epoch is advanced on every prepare, so a later PCG solve cannot observe
/// tangent values left by an earlier pose.
#[derive(Clone,Copy)]
pub struct FrozenAngularTangentView { factors:Handle, published:Handle, epoch:u32 }
#[derive(Clone,Copy)]
pub struct FrozenControlTangentView { factors:Handle, offsets:Handle, published:Handle, epoch:u32 }
#[derive(Clone,Copy)]
pub struct FrozenExtensionTangentView { factors:Handle, offsets:Handle, published:Handle, epoch:u32 }
#[derive(Clone, Copy)]
pub struct FrozenBallView { factors:Handle, offsets:Handle, published:Handle, fault:Handle, epoch:u32 }
#[derive(Clone, Copy)]
pub struct View { pub frozen_ball:Option<FrozenBallView>, pub frozen_angular:Option<FrozenAngularView>, pub frozen_angular_tangent:Option<FrozenAngularTangentView>, pub frozen_control_tangent:Option<FrozenControlTangentView>, pub frozen_extension_tangent:Option<FrozenExtensionTangentView>, pub curves:Option<super::restoring::View>, pub soft:Option<super::soft_joints::View>, pub ball: Handle, pub angular: Handle, pub control: Handle, pub fault: Handle,
    pub(super) shadow_identity:Option<([u8;36],[u8;64])>,
    pub reference: Handle, pub state: Handle, pub twist_context: Handle, pub material_spin_context: Handle,
    pub limits: Handle, pub drives: Handle, pub drive_offsets:Handle, pub device_keys: Handle,
    pub control_tangent_factors:Handle, pub control_tangent_offsets:Handle, pub control_tangent_published:Handle,
    pub extension_tangent_factors:Handle, pub extension_tangent_offsets:Handle, pub extension_tangent_published:Handle,
    pub control_tangent_fault:Handle, pub extension_tangent_fault:Handle,
    pub extension_count: u32, pub drive_count: u32,
    pub start_time_s: f64, pub end_time_s: f64, pub sample_time_s: f32,
    /// Explicit per-family counts. A family with no records must never be
    /// dispatched, whatever the other families' counts are.
    pub count: u32, pub control_count: u32,
    pub bodies: u32, pub base: u32, pub dim: u32, pub dt: f32 }
impl View {
    pub(super) fn with_preview_buffers(
        mut self,
        map: &mut impl FnMut(Handle, bool) -> Result<Handle, Fault>,
    ) -> Result<Self, Fault> {
        self.fault = map(self.fault, true)?;
        self.control_tangent_fault = map(self.control_tangent_fault, true)?;
        self.extension_tangent_fault = map(self.extension_tangent_fault, true)?;
        if let Some(view) = self.frozen_angular.as_mut() {
            view.fault = map(view.fault, true)?;
        }
        if let Some(view) = self.frozen_ball.as_mut() {
            view.fault = map(view.fault, true)?;
        }
        self.curves = self.curves.map(|view| view.with_preview_buffers(map)).transpose()?;
        self.soft = self.soft.map(|view| view.with_preview_buffers(map)).transpose()?;
        Ok(self)
    }

    pub(super) fn reduced_capture_epochs(&self)->Vec<(&'static str,Handle,u32,usize,Handle,usize,Option<(Handle,usize)>)>{
        let mut out=Vec::new();
        if let Some(f)=self.frozen_ball {out.push(("ball",f.published,f.epoch,self.count as usize,f.factors,15,Some((f.offsets,3))));}
        if let Some(f)=self.frozen_angular {out.push(("restoring_angular",f.published,f.epoch,self.control_count as usize,f.factors,21,Some((f.offsets,2))));}
        if let Some(f)=self.frozen_angular_tangent {out.push(("angular_v2",f.published,f.epoch,self.count as usize,f.factors,9,None));}
        if let Some(f)=self.frozen_control_tangent {out.push(("control",f.published,f.epoch,self.control_count as usize,f.factors,21,Some((f.offsets,2))));}
        if let Some(f)=self.frozen_extension_tangent {out.push(("extension",f.published,f.epoch,self.extension_count as usize,f.factors,22,Some((f.offsets,2))));}
        out
    }
    pub(super) fn shadow_fault_handles(&self)->Vec<Handle>{
        let mut out=vec![self.fault,self.control_tangent_fault,self.extension_tangent_fault];
        if let Some(v)=self.frozen_ball {out.push(v.fault);}
        if let Some(v)=self.frozen_angular {out.push(v.fault);}
        if let Some(v)=self.curves {out.extend([v.fault,v.ball_fault,v.angular_fault,v.fem_fault,v.stitch_fault]);}
        if let Some(v)=self.soft {out.extend(v.shadow_fault_handles());}
        out
    }
}
impl Joints {
    pub fn install<D:Device>(&mut self, device:&mut D, owned:&crate::joint::OwnedJointScene, bodies:usize) -> Result<(),Fault> {
        self.shadow_identity=match (owned.scene_uuid.as_bytes().try_into(),owned.scene_inputs_sha256.as_bytes().try_into()) { (Ok(u),Ok(s))=>Some((u,s)),_=>None };
        self.accepted_twist=None;self.trial_twist=None;self.twist_selected=None;
        self.twist_endpoints=owned.joints.iter().map(|j|(j.parent_runtime_body_index,j.child_runtime_body_index)).collect();
        let ball=crate::joint_gpu::from_validated(owned).map_err(|e| Fault::Platform { call:"joint transport", detail:e })?;
        let angular=crate::joint_angular_gpu::from_validated(owned).map_err(|e| Fault::Platform { call:"joint transport", detail:e })?;
        let control=crate::joint_angular_control_gpu::from_validated(owned).map_err(|e| Fault::Platform { call:"joint transport", detail:e })?;
        self.ball.size(device,ball.len(),AllocLabel("joints.ball"))?;
        self.angular.size(device,angular.len(),AllocLabel("joints.angular"))?;
        self.control.size(device,control.len(),AllocLabel("joints.control"))?;
        let tangent_rows=control.len().max(1);
        self.control_tangent_factors.size(device,21usize.checked_mul(tangent_rows).ok_or_else(||Fault::Platform{call:"joint control tangent",detail:"row count overflow".into()})?,AllocLabel("joints.control.tangent"))?;
        self.control_tangent_offsets.size(device,2*tangent_rows,AllocLabel("joints.control.tangent.offsets"))?;
        self.control_tangent_published.size(device,tangent_rows,AllocLabel("joints.control.tangent.epoch"))?;
        self.control_tangent_fault.size(device,tangent_rows,AllocLabel("joints.control.tangent.fault"))?;
        self.extension_tangent_factors.size(device,22usize.checked_mul(tangent_rows).ok_or_else(||Fault::Platform{call:"joint extension tangent",detail:"row count overflow".into()})?,AllocLabel("joints.extension.tangent"))?;
        self.extension_tangent_offsets.size(device,2*tangent_rows,AllocLabel("joints.extension.tangent.offsets"))?;
        self.extension_tangent_published.size(device,tangent_rows,AllocLabel("joints.extension.tangent.epoch"))?;
        self.extension_tangent_fault.size(device,tangent_rows,AllocLabel("joints.extension.tangent.fault"))?;
        self.drive_offsets.size(device,control.len()+1,AllocLabel("joints.drive.offsets"))?;
        let empty_drive_offsets=vec![0u32;control.len()+1];
        self.drive_offsets.write(device,0,&empty_drive_offsets)?;
        self.twist_context.size(device,ball.len(),AllocLabel("joints.twist_context"))?;
        if !ball.is_empty() {self.twist_context.write(device,0,&vec![crate::continuous_twist::AngularTwistEvalContext::default();ball.len()])?;}
        self.material_spin_context.size(device,ball.len(),AllocLabel("joints.material_spin_context"))?;
        if !ball.is_empty() {self.material_spin_context.write(device,0,&vec![crate::material_spin_history::MaterialSpinContext::default();ball.len()])?;}
        if !ball.is_empty() { self.ball.write(device,0,&ball)?; self.angular.write(device,0,&angular)?; }
        if !control.is_empty() { self.control.write(device,0,&control)?; }
        // The fault lane must start defined. A FRESH sizing already carries
        // zeros; a REUSED allocation keeps the previous scene's verdicts, which
        // is harmless because the per-pose check overwrites the whole lane
        // before anything reads it, but the lane is never read before that.
        self.fault.size(device,control.len(),AllocLabel("joints.fault"))?;
        self.reference.size(device,bodies,AllocLabel("joints.step_start"))?;
        // Ingest already proved the declared feature union. Repack exactly the
        // same validated full joint order; no disabled/absent row is filtered.
        let inputs: Vec<_> = owned.joints.iter().map(|joint| { let w=&joint.wire; ext::JointExtensionInput {
            control_present:w.angular_control_v1.is_some(),
            control_model:w.angular_control_v1.as_ref().map(|c|c.coordinate_model.as_str()),
            soft:w.angular_soft_limits_v1.as_ref().map(|v| &v.0),
            drives:w.angular_drives_v1.as_ref().map(|v| &v.0),
            legacy_angular_nonzero:w.angle_stiffness_nm_per_rad!=0.0 || w.angle_damping_nms_per_rad!=0.0 || w.max_angle_rad!=0.0,
        }}).collect();
        let features: Vec<String> = ext::required_features(&inputs).into_iter().map(str::to_owned).collect();
        let packed=ext::validate_v4_and_pack(&inputs,&features).map_err(|e| Fault::Platform { call:"joint extension transport",detail:e.to_string() })?;
        self.start_time_s=0.0;
        self.end_time_s=None;
        self.drive_count=0;
        match packed {
            Some(packed) => {
                if packed.limits.len()!=control.len() { return Err(Fault::Platform { call:"joint extension transport",detail:"limit/control row count mismatch".into() }); }
                if packed.drives.windows(2).any(|pair|pair[0].joint_index>pair[1].joint_index) {
                    return Err(Fault::Platform{call:"joint extension transport",detail:"drive rows are not grouped by joint".into()});
                }
                let mut drive_offsets=vec![0u32;control.len()+1];
                for drive in &packed.drives {
                    let joint=drive.joint_index as usize;
                    if joint>=control.len() {return Err(Fault::Platform{call:"joint extension transport",detail:"drive joint index out of range".into()});}
                    drive_offsets[joint+1]=drive_offsets[joint+1].checked_add(1).ok_or_else(||Fault::Platform{call:"joint extension transport",detail:"drive span overflow".into()})?;
                }
                for joint in 0..control.len() {drive_offsets[joint+1]=drive_offsets[joint+1].checked_add(drive_offsets[joint]).ok_or_else(||Fault::Platform{call:"joint extension transport",detail:"drive span overflow".into()})?;}
                self.drive_offsets.write(device,0,&drive_offsets)?;
                let keys=packed.device_keys().map_err(|e| Fault::Platform { call:"joint extension device keys",detail:e.to_string() })?;
                let (_,drive_count,_)=packed.checked_upload_counts().map_err(|e| Fault::Platform { call:"joint extension transport",detail:e.to_string() })?;
                self.drive_count=drive_count;
                self.limits.size(device,packed.limits.len(),AllocLabel("joints.limits"))?;
                self.limits.write(device,0,&packed.limits)?;
                // Valid arena handles even for soft-only scenes: counts remain
                // zero and the unused dummy elements are never read by a kernel.
                self.drives.size(device,packed.drives.len().max(1),AllocLabel("joints.drives"))?;
                self.device_keys.size(device,keys.len().max(1),AllocLabel("joints.device_keys"))?;
                if !packed.drives.is_empty() { self.drives.write(device,0,&packed.drives)?; }
                if !keys.is_empty() { self.device_keys.write(device,0,&keys)?; }
            },
            None => {
                self.limits.size(device,0,AllocLabel("joints.limits"))?;
                self.drives.size(device,0,AllocLabel("joints.drives"))?;
                self.device_keys.size(device,0,AllocLabel("joints.device_keys"))?;
            },
        }
        self.anchored_flight = if cfg!(cuda_backend)
            && std::env::var_os("PPF_BODY_FULL_ANCHORED_FREE_FLIGHT").is_some_and(|value| value == "1")
        {
            super::anchored_flight::Components::build(bodies, &ball)
        } else {
            None
        };
        Ok(())
    }

    pub(super) fn anchored_flight_covers(&self, enabled_roots: &[u32]) -> bool {
        self.anchored_flight.as_ref().is_some_and(|graph| graph.all_prescribed(enabled_roots))
    }

    pub unsafe fn prepare_ball_cache<D:Device>(&mut self,device:&mut D,view:View)->Result<FrozenBallView,Fault> {
        // A valid device handle is supplied for every pointer even when no
        // curve pack exists. Zero counts guarantee lookup never dereferences
        // these aligned, unused dummy pointers.
        if view.curves.is_none() {
            self.frozen_ball_empty_history.size(device,1,AllocLabel("frozen.ball.empty_history"))?;
        }
        let (descriptors,nodes,bindings,history,descriptor_count,node_count,binding_count,history_count)=match view.curves {
            Some(c)=>(c.descriptors,c.nodes,c.bindings,c.history,c.descriptor_count,c.node_count,c.binding_count,c.history_count),
            None=>(view.ball,view.ball,view.ball,self.frozen_ball_empty_history.handle(),0,0,0,0),
        };
        let n=view.count as usize;
        if n==0 || n>(u32::MAX as usize)/15 {return Err(Fault::Platform{call:"frozen ball",detail:"invalid record count".into()});}
        self.frozen_ball_generation=self.frozen_ball_generation.checked_add(1).ok_or_else(||Fault::Platform{call:"frozen ball",detail:"epoch overflow".into()})?;
        self.frozen_ball_factors.size(device,15*n,AllocLabel("frozen.ball.factors"))?;
        self.frozen_ball_offsets.size(device,3*n,AllocLabel("frozen.ball.offsets"))?;
        self.frozen_ball_epoch.size(device,n,AllocLabel("frozen.ball.epoch"))?;
        self.frozen_ball_fault.size(device,n,AllocLabel("frozen.ball.fault"))?;
        self.frozen_ball_fault.seed(device,&vec![0;n])?;
        let f=FrozenBallView{factors:self.frozen_ball_factors.handle(),offsets:self.frozen_ball_offsets.handle(),published:self.frozen_ball_epoch.handle(),fault:self.frozen_ball_fault.handle(),epoch:self.frozen_ball_generation};
        let args=super::kernels::RestoringBallFrozenBuildArgs{records:view.ball,count_rows:view.count,state:view.state,bodies:view.bodies,base:view.base,dim:view.dim,dt:view.dt,descriptors,nodes,bindings,history,descriptor_count,node_count,binding_count,history_count,epoch:f.epoch,factors:f.factors,offsets:f.offsets,published:f.published,fault:f.fault,count:view.count,seam_arena_count:0};
        device.launch("restoring.ball.frozen.build",&args,view.count)?;
        Ok(f)
    }
    pub fn check_ball_cache_fault<D:Device>(&mut self,device:&mut D)->Result<(),Fault> {
        self.frozen_ball_fault.download(device)?;
        if let Some((row,code))=self.frozen_ball_fault.host().iter().copied().enumerate().find(|(_,c)|*c!=0) {
            return Err(Fault::Platform{call:"frozen ball",detail:format!("row {row}, fault {code}")});
        }
        Ok(())
    }
    pub unsafe fn prepare_angular_cache<D:Device>(&mut self,device:&mut D,view:View)->Result<FrozenAngularView,Fault> {
        let c=view.curves.ok_or_else(||Fault::Platform{call:"frozen angular",detail:"missing curve view".into()})?;
        let n=view.control_count as usize;
        if n==0 || n>(u32::MAX as usize)/21 {return Err(Fault::Platform{call:"frozen angular",detail:"invalid record count".into()});}
        self.frozen_angular_generation=self.frozen_angular_generation.checked_add(1).ok_or_else(||Fault::Platform{call:"frozen angular",detail:"epoch overflow".into()})?;
        self.frozen_angular_factors.size(device,21*n,AllocLabel("frozen.angular.factors"))?;
        self.frozen_angular_offsets.size(device,2*n,AllocLabel("frozen.angular.offsets"))?;
        self.frozen_angular_epoch.size(device,n,AllocLabel("frozen.angular.epoch"))?;
        self.frozen_angular_fault.size(device,n,AllocLabel("frozen.angular.fault"))?;
        self.frozen_angular_fault.seed(device,&vec![0;n])?;
        let f=FrozenAngularView{factors:self.frozen_angular_factors.handle(),offsets:self.frozen_angular_offsets.handle(),published:self.frozen_angular_epoch.handle(),fault:self.frozen_angular_fault.handle(),epoch:self.frozen_angular_generation};
        let args=super::kernels::RestoringAngularFrozenBuildArgs{records:view.control,twist:view.twist_context,material_context:view.material_spin_context,count_rows:view.control_count,state:view.state,reference:view.reference,bodies:view.bodies,base:view.base,dim:view.dim,dt:view.dt,descriptors:c.descriptors,nodes:c.nodes,bindings:c.bindings,history:c.history,descriptor_count:c.descriptor_count,node_count:c.node_count,binding_count:c.binding_count,history_count:c.history_count,epoch:f.epoch,factors:f.factors,offsets:f.offsets,published:f.published,fault:f.fault,count:view.control_count,seam_arena_count:0};
        device.launch("restoring.angular.frozen.build",&args,view.control_count)?;
        Ok(f)
    }
    pub fn check_angular_cache_fault<D:Device>(&mut self,device:&mut D)->Result<(),Fault> {
        self.frozen_angular_fault.download(device)?;
        if let Some((row,code))=self.frozen_angular_fault.host().iter().copied().enumerate().find(|(_,c)|*c!=0) {
            return Err(Fault::Platform{call:"frozen angular",detail:format!("row {row}, fault {code}")});
        }
        Ok(())
    }
    /// Build the regular isotropic V2 row tangents at `view.state` and return
    /// the same view tagged with this solve's cache epoch. Call once after the
    /// current Newton pose is fitted and before any PCG matvec. The ordinary
    /// gradient, true-residual, and self-block paths remain uncached.
    pub unsafe fn prepare_angular_tangents<D:Device>(&mut self,device:&mut D,mut view:View)->Result<View,Fault> {
        let n=view.count as usize;
        if n==0 { view.frozen_angular_tangent=None; return Ok(view); }
        if n>(u32::MAX as usize)/9 || self.angular.len()!=n {
            return Err(Fault::Platform{call:"angular tangent cache",detail:"invalid record count".into()});
        }
        self.angular_tangent_generation=self.angular_tangent_generation.checked_add(1)
            .ok_or_else(||Fault::Platform{call:"angular tangent cache",detail:"epoch overflow".into()})?;
        let epoch=self.angular_tangent_generation;
        self.angular_tangent_factors.size(device,9*n,AllocLabel("joints.angular.tangent"))?;
        self.angular_tangent_published.size(device,n,AllocLabel("joints.angular.tangent.epoch"))?;
        let factors=self.angular_tangent_factors.handle();
        let published=self.angular_tangent_published.handle();
        let args=JointAngularApplyArgs{records:view.angular,state:view.state,reference:view.reference,
            bodies:view.bodies,base:view.base,dim:view.dim,dt:view.dt,action:3,
            // The builder branch does not read or write the ordinary Krylov
            // vectors; reusing its own valid storage keeps this one ABI entry.
            input:factors,output:factors,tangent_cache:factors,tangent_epoch:published,epoch,
            count:view.count,seam_arena_count:0};
        device.launch("joints.angular",&args,view.count)?;
        view.frozen_angular_tangent=Some(FrozenAngularTangentView{factors,published,epoch});
        Ok(view)
    }
    /// Build the exact directional-control and limit/drive Gauss-Newton rows
    /// once at the current Newton pose. Curves-backed directional controls are
    /// already owned by `frozen_angular`, so only curves=None rows are cached
    /// here; extension limits/drives are independent and always cached.
    pub unsafe fn prepare_control_extension_tangents<D:Device>(&mut self,device:&mut D,mut view:View)->Result<View,Fault> {
        let n=view.control_count as usize;
        if view.curves.is_none() && n>0 {
            if n>(u32::MAX as usize)/21 {return Err(Fault::Platform{call:"joint control tangent",detail:"invalid record count".into()});}
            self.control_tangent_generation=self.control_tangent_generation.checked_add(1)
                .ok_or_else(||Fault::Platform{call:"joint control tangent",detail:"epoch overflow".into()})?;
            let epoch=self.control_tangent_generation;
            self.control_tangent_factors.size(device,21*n,AllocLabel("joints.control.tangent"))?;
            self.control_tangent_offsets.size(device,2*n,AllocLabel("joints.control.tangent.offsets"))?;
            self.control_tangent_published.size(device,n,AllocLabel("joints.control.tangent.epoch"))?;
            self.control_tangent_fault.size(device,n,AllocLabel("joints.control.tangent.fault"))?;
            self.control_tangent_fault.seed(device,&vec![0;n])?;
            let factors=self.control_tangent_factors.handle();
            let offsets=self.control_tangent_offsets.handle();
            let published=self.control_tangent_published.handle();
            let fault=self.control_tangent_fault.handle();
            let args=JointAngularControlApplyArgs{records:view.control,twist_context:view.twist_context,material_spin_context:view.material_spin_context,
                state:view.state,reference:view.reference,bodies:view.bodies,base:view.base,dim:view.dim,dt:view.dt,action:6,
                input:view.state,output:view.state,fault,tangent_cache:factors,tangent_offsets:offsets,tangent_published:published,epoch,
                count:view.control_count,seam_arena_count:0};
            device.launch("joints.angular_control",&args,view.control_count)?;
            self.control_tangent_fault.download(device)?;
            if let Some((row,code))=self.control_tangent_fault.host().iter().copied().enumerate().find(|(_,c)|*c!=0) {
                return Err(Fault::Platform{call:"joint control tangent",detail:format!("row {row}, fault {code}")});
            }
            view.frozen_control_tangent=Some(FrozenControlTangentView{factors,offsets,published,epoch});
        } else { view.frozen_control_tangent=None; }

        let en=view.extension_count as usize;
        if en>0 {
            if en>(u32::MAX as usize)/22 {return Err(Fault::Platform{call:"joint extension tangent",detail:"invalid record count".into()});}
            self.extension_tangent_generation=self.extension_tangent_generation.checked_add(1)
                .ok_or_else(||Fault::Platform{call:"joint extension tangent",detail:"epoch overflow".into()})?;
            let epoch=self.extension_tangent_generation;
            self.extension_tangent_factors.size(device,22*en,AllocLabel("joints.extension.tangent"))?;
            self.extension_tangent_offsets.size(device,2*en,AllocLabel("joints.extension.tangent.offsets"))?;
            self.extension_tangent_published.size(device,en,AllocLabel("joints.extension.tangent.epoch"))?;
            self.extension_tangent_fault.size(device,en,AllocLabel("joints.extension.tangent.fault"))?;
            self.extension_tangent_fault.seed(device,&vec![0;en])?;
            let factors=self.extension_tangent_factors.handle();
            let offsets=self.extension_tangent_offsets.handle();
            let published=self.extension_tangent_published.handle();
            let fault=self.extension_tangent_fault.handle();
            let args=JointAngularExtensionApplyArgs{records:view.control,twist_context:view.twist_context,material_spin_context:view.material_spin_context,
                limits:view.limits,drives:view.drives,drive_count:view.drive_count,drive_offsets:view.drive_offsets,keys:view.device_keys,
                state:view.state,reference:view.reference,bodies:view.bodies,base:view.base,dim:view.dim,sample_time_s:view.sample_time_s,dt_s:view.dt,
                action:6,input:view.state,tangent_cache:factors,tangent_offsets:offsets,tangent_published:published,epoch,
                output:view.state,fault,count:view.extension_count,seam_arena_count:0};
            device.launch("joints.angular_extension",&args,view.extension_count)?;
            self.extension_tangent_fault.download(device)?;
            if let Some((row,code))=self.extension_tangent_fault.host().iter().copied().enumerate().find(|(_,c)|*c!=0) {
                return Err(Fault::Platform{call:"joint extension tangent",detail:format!("row {row}, fault {code}")});
            }
            view.frozen_extension_tangent=Some(FrozenExtensionTangentView{factors,offsets,published,epoch});
        } else { view.frozen_extension_tangent=None; }
        Ok(view)
    }
    pub fn view(&mut self,state:Handle,map:&super::rigid_map::RigidMap,dt:f32)->View {
        View { shadow_identity:self.shadow_identity,frozen_ball:None,frozen_angular:None,frozen_angular_tangent:None,frozen_control_tangent:None,frozen_extension_tangent:None,curves:self.curves.view(),soft:self.soft.view(map,dt,self.curves.view()),ball:self.ball.handle(),angular:self.angular.handle(),control:self.control.handle(),
            fault:self.fault.handle(),reference:self.reference.handle(), state,twist_context:self.twist_context.handle(),material_spin_context:self.material_spin_context.handle(),
            count:self.ball.len() as u32, control_count:self.control.len() as u32,
            limits:self.limits.handle(),drives:self.drives.handle(),device_keys:self.device_keys.handle(),
            drive_offsets:self.drive_offsets.handle(),
            control_tangent_factors:self.control_tangent_factors.handle(),control_tangent_offsets:self.control_tangent_offsets.handle(),control_tangent_published:self.control_tangent_published.handle(),
            extension_tangent_factors:self.extension_tangent_factors.handle(),extension_tangent_offsets:self.extension_tangent_offsets.handle(),extension_tangent_published:self.extension_tangent_published.handle(),
            control_tangent_fault:self.control_tangent_fault.handle(),extension_tangent_fault:self.extension_tangent_fault.handle(),
            extension_count:self.limits.len() as u32,drive_count:self.drive_count,
            start_time_s:self.start_time_s,end_time_s:self.end_time_s.unwrap_or(self.start_time_s+f64::from(dt)),sample_time_s:self.end_time_s.unwrap_or(self.start_time_s+f64::from(dt)) as f32,
            bodies:map.n_bodies as u32,base:map.body_base as u32,dim:map.dim as u32,dt }
    }
    pub(crate) fn accepted_twist(&self)->Option<&crate::continuous_twist::AcceptedTwist> {self.accepted_twist.as_ref()}
    pub(crate) fn joint_record_count(&self)->usize {self.twist_endpoints.len()}
    pub(crate) fn accepted_material_spin(&self)->Option<&crate::material_spin_history::AcceptedMaterialSpin>{self.accepted_material_spin.as_ref()}
    pub(crate) fn material_spin_active(&self)->bool{self.material_spin_selected.is_some()}
    pub(crate) fn material_spin_retry_dt(&self)->Option<f32>{self.material_spin_retry_dt}
    pub(crate) fn mark_material_spin_retry(&mut self,dt:f32){self.material_spin_retry_dt=Some(dt);}
    pub(crate) fn material_spin_control_retry_compatible<D:Device>(&mut self,device:&mut D)->Result<bool,Fault>{
        if self.fault.len()==0{return Ok(true);}self.fault.download(device)?;Ok(self.fault.host().iter().all(|v|matches!(*v,0|8)))
    }
    pub(crate) fn consume_material_spin_faults<D:Device>(&mut self,device:&mut D)->Result<(),Fault>{
        self.curves.consume_material_faults(device)?;
        if self.fault.len()>0 {
            self.fault.download(device)?;
            if self.fault.host().iter().any(|v|!matches!(*v,0|8)){return Err(Fault::Platform{call:"material spin retry",detail:"unexpected control fault".into()});}
            self.fault.seed(device,&vec![0;self.fault.len()])?;
        }Ok(())
    }
    pub(crate) fn activate_material_spin(&mut self,history:crate::material_spin_history::AcceptedMaterialSpin,scene:&str,joints:&[String],time:f64,selected:Vec<bool>)->Result<(),Fault>{
        history.validate(scene,joints,time).map_err(|e|Fault::Platform{call:"material spin activate",detail:e.into()})?;
        if self.trial_material_spin.is_some() || self.accepted_material_spin.is_some() || selected.len()!=self.twist_endpoints.len() || selected.iter().filter(|v|**v).count()!=history.rows.len() {
            return Err(Fault::Platform{call:"material spin activate",detail:"invalid joint selection or active state".into()});
        }
        self.accepted_material_spin=Some(history);self.material_spin_selected=Some(selected);Ok(())
    }
    pub(crate) fn begin_material_spin_trial(&mut self)->Result<(),Fault>{
        if self.trial_material_spin.is_some(){return Err(Fault::Platform{call:"material spin trial",detail:"trial already active".into()});}
        self.trial_material_spin=Some(self.accepted_material_spin.as_ref().ok_or_else(||Fault::Platform{call:"material spin trial",detail:"accepted state missing".into()})?.trial());Ok(())
    }
    pub(crate) fn abort_material_spin_trial(&mut self){self.trial_material_spin=None;}
    pub(crate) fn prepare_material_spin_commit(&self,time:f64)->Result<crate::material_spin_history::AcceptedMaterialSpin,Fault>{
        let accepted=self.accepted_material_spin.as_ref().ok_or_else(||Fault::Platform{call:"material spin commit",detail:"accepted state missing".into()})?;
        let trial=self.trial_material_spin.clone().ok_or_else(||Fault::Platform{call:"material spin commit",detail:"trial missing".into()})?;
        accepted.prepare_commit(trial,time).map_err(|e|Fault::Platform{call:"material spin commit",detail:e.into()})
    }
    pub(crate) fn finish_material_spin_commit(&mut self,accepted:crate::material_spin_history::AcceptedMaterialSpin){self.accepted_material_spin=Some(accepted);self.trial_material_spin=None;}
    pub(crate) fn upload_material_spin_context<D:Device>(&mut self,device:&mut D)->Result<(),Fault>{
        let selected=self.material_spin_selected.as_ref().ok_or_else(||Fault::Platform{call:"material spin context",detail:"selection missing".into()})?;
        let accepted=self.accepted_material_spin.as_ref().ok_or_else(||Fault::Platform{call:"material spin context",detail:"accepted state missing".into()})?;
        if selected.len()!=self.material_spin_context.len() || selected.iter().filter(|v|**v).count()!=accepted.rows.len(){return Err(Fault::Platform{call:"material spin context",detail:"selection size mismatch".into()});}
        let mut rows=accepted.rows.iter();
        let mut context=Vec::with_capacity(selected.len());
        for enabled in selected {
            if *enabled {
                let row=rows.next().unwrap();let q=row.q_wxyz.map(|v|v as f32);let a=row.a.map(|v|v as f32);
                if q.iter().chain(a.iter()).any(|v|!v.is_finite()){return Err(Fault::Platform{call:"material spin context",detail:"device projection overflow".into()});}
                context.push(crate::material_spin_history::MaterialSpinContext{accepted_q_wxyz:q,accepted_a:a,model:3});
            } else {context.push(crate::material_spin_history::MaterialSpinContext::default());}
        }
        if !context.is_empty(){self.material_spin_context.write(device,0,&context)?;}Ok(())
    }
    pub(crate) unsafe fn sample_material_spin_pose<D:Device>(&mut self,device:&mut D,view:View,selected:&[bool])->Result<Vec<[f64;4]>,Fault>{
        let n=selected.len();
        if n!=view.control_count as usize {return Err(Fault::Platform{call:"material spin pose",detail:"selection count mismatch".into()});}
        self.material_spin_sample.size(device,n*4,AllocLabel("joints.material_spin_pose"))?;
        self.material_spin_fault.size(device,n,AllocLabel("joints.material_spin_pose_fault"))?;
        self.twist_sample_selection.size(device,n,AllocLabel("joints.material_spin_pose_selection"))?;
        self.twist_sample_selection.write(device,0,&selected.iter().map(|v|if *v{1.}else{0.}).collect::<Vec<_>>())?;
        self.material_spin_sample.seed(device,&vec![0.;n*4])?;self.material_spin_fault.seed(device,&vec![0;n])?;
        let args=JointAngularControlApplyArgs{records:view.control,twist_context:view.twist_context,material_spin_context:view.material_spin_context,state:view.state,reference:view.reference,
            bodies:view.bodies,base:view.base,dim:view.dim,dt:view.dt,action:5,input:self.twist_sample_selection.handle(),output:self.material_spin_sample.handle(),fault:self.material_spin_fault.handle(),
            tangent_cache:view.control_tangent_factors,tangent_offsets:view.control_tangent_offsets,tangent_published:view.control_tangent_published,epoch:0,count:view.control_count,seam_arena_count:0};
        if n>0{device.launch("joints.material_spin_pose",&args,view.control_count)?;}
        self.material_spin_fault.download(device)?;self.material_spin_sample.download(device)?;
        if let Some(code)=self.material_spin_fault.host().iter().copied().find(|v|*v!=0){return Err(Fault::Platform{call:"material spin pose",detail:format!("pose fault {code}")});}
        Ok(self.material_spin_sample.host().chunks_exact(4).map(|r|std::array::from_fn(|i|f64::from(r[i]))).collect())
    }
    /// Sample the accepted-to-current trial using the same F as device assembly.
    /// Code 8 must be turned into a physical substep retry by the caller.
    pub(crate) unsafe fn sample_material_spin_candidate<D:Device>(&mut self,device:&mut D,view:View)->Result<Option<u32>,Fault>{
        let selected=self.material_spin_selected.as_ref().ok_or_else(||Fault::Platform{call:"material spin sample",detail:"selection missing".into()})?;
        let curves=view.curves.ok_or_else(||Fault::Platform{call:"material spin sample",detail:"restoring dispatch unavailable".into()})?;
        let n=selected.len();
        if n!=view.control_count as usize {return Err(Fault::Platform{call:"material spin sample",detail:"full joint count mismatch".into()});}
        self.material_spin_sample.size(device,7*n,AllocLabel("joints.material_spin_sample"))?;
        self.material_spin_fault.size(device,n,AllocLabel("joints.material_spin_fault"))?;
        self.material_spin_sample.seed(device,&vec![0.;7*n])?;
        self.material_spin_fault.seed(device,&vec![0;n])?;
        let args=super::kernels::JointRestoringAngularApplyArgs { records:view.control,twist_context:view.twist_context,material_spin_context:view.material_spin_context,
            record_count:view.control_count,count:view.control_count,state:view.state,reference:view.reference,bodies:view.bodies,base:view.base,dim:view.dim,dt:view.dt,
            action:5,input:view.control,output:self.material_spin_sample.handle(),fault:self.material_spin_fault.handle(),descriptors:curves.descriptors,nodes:curves.nodes,
            bindings:curves.bindings,history:curves.history,descriptor_count:curves.descriptor_count,node_count:curves.node_count,binding_count:curves.binding_count,
            history_count:curves.history_count,observations:curves.observations,seam_arena_count:0};
        if n>0 {device.launch("restoring.angular",&args,view.control_count)?;}
        self.material_spin_fault.download(device)?;self.material_spin_sample.download(device)?;
        if let Some(code)=self.material_spin_fault.host().iter().copied().find(|v|*v!=0){if code==8 {self.material_spin_retry_dt=Some(view.dt);} return Ok(Some(code));}
        let trial=self.trial_material_spin.as_mut().ok_or_else(||Fault::Platform{call:"material spin sample",detail:"trial missing".into()})?;
        let mut output=Vec::with_capacity(trial.rows.len());
        for (enabled,values) in selected.iter().zip(self.material_spin_sample.host().chunks_exact(7)) {
            if !*enabled {continue;}
            let mut q:[f64;4]=std::array::from_fn(|i|f64::from(values[i]));
            let norm=q.iter().map(|x|x*x).sum::<f64>().sqrt();
            if !norm.is_finite() || norm<0.99 || norm>1.01 {return Err(Fault::Platform{call:"material spin sample",detail:"invalid candidate quaternion".into()});}
            for x in &mut q {*x/=norm;}
            let a=std::array::from_fn(|i|f64::from(values[i+4]));
            if a.iter().any(|v|!v.is_finite()){return Err(Fault::Platform{call:"material spin sample",detail:"invalid candidate state".into()});}
            output.push(crate::material_spin_history::MaterialSpinRow{q_wxyz:q,a});
        }
        if output.len()!=trial.rows.len(){return Err(Fault::Platform{call:"material spin sample",detail:"candidate count mismatch".into()});}
        trial.rows=output;Ok(None)
    }
    pub(crate) fn twist_selection(&self)->Option<&[bool]> {self.twist_selected.as_deref()}
    /// Caller must first validate selected UUIDs and current pose against the
    /// trusted sealed source and checkpoint. This does not infer selection.
    pub(crate) fn activate_twist_selection(&mut self,selected:Vec<bool>)->Result<(),Fault>{
        let history=self.accepted_twist.as_ref().ok_or_else(||Fault::Platform{call:"continuous twist activation",detail:"missing validated accepted history".into()})?;
        if self.trial_twist.is_some() || selected.len()!=self.twist_endpoints.len() || history.theta.len()!=selected.len()
            || !selected.iter().any(|v|*v) || selected.iter().zip(&history.theta).any(|(s,v)|!*s && *v!=0.0) {
            return Err(Fault::Platform{call:"continuous twist activation",detail:"invalid full-order selection/history".into()});
        }
        eprintln!("CONTINUOUS_TWIST_ACTIVATED selected={} accepted_time={} prior_turns_known={}",selected.iter().filter(|x|**x).count(),history.time,history.prior_turns_known);
        self.twist_selected=Some(selected);Ok(())
    }
    /// Translate the trusted full-joint selection to the fitted body rows.
    pub(crate) fn twist_body_selection(&self,selected:&[bool],bodies:usize)->Result<Vec<bool>,Fault>{
        if selected.len()!=self.twist_endpoints.len(){return Err(Fault::Platform{call:"continuous twist bodies",detail:"selection count mismatch".into()});}
        let mut result=vec![false;bodies];
        for (&(a,b),&enabled) in self.twist_endpoints.iter().zip(selected) {
            if !enabled {continue;}
            for endpoint in [a,b] {
                if endpoint==u32::MAX {continue;}
                let slot=result.get_mut(endpoint as usize).ok_or_else(||Fault::Platform{call:"continuous twist bodies",detail:"body index outside validated map".into()})?;
                *slot=true;
            }
        }
        Ok(result)
    }
    /// Certify a stored-vertex segment using dedicated fit snapshots. A
    /// rejected certificate is not permission to choose a nearest branch.
    pub(crate) fn certify_twist_segment(&self,selected:&[bool],a:&TwistPoseSnapshot,b:&TwistPoseSnapshot,commanded:Option<&[f64]>)->bool {
        if selected.len()!=self.twist_endpoints.len() || a.rows.len()!=selected.len() || b.rows.len()!=selected.len() {return false;}
        for i in 0..selected.len() {
            if !selected[i] {continue;}
            let (ea,eb)=self.twist_endpoints[i];
            let body_bound=|body:u32,offset:usize|->Option<f64>{
                if body==u32::MAX {return Some(0.0);}
                let (c0,e0)=a.covariances.get(body as usize)?.as_ref()?;
                let (c1,e1)=b.covariances.get(body as usize)?.as_ref()?;
                let q0=std::array::from_fn(|j|a.rows[i][offset+j]);
                let q1=std::array::from_fn(|j|b.rows[i][offset+j]);
                let fit=match crate::continuous_twist::polar_segment_bound(*c0,*c1,q0,q1,e0.max(*e1)) {
                    Some(v)=>v,
                    None=>{
                        if std::env::var_os("PPF_INTERNAL_TEST_CONTINUOUS_TWIST").as_deref()==Some(std::ffi::OsStr::new("1")) {
                            eprintln!("CONTINUOUS_TWIST_POLAR_INPUT body={body} c0={c0:?} c1={c1:?} q0={q0:?} q1={q1:?} error={}",e0.max(*e1));
                        }
                        return None;
                    }
                };
                let extra=match commanded {Some(values)=>*values.get(body as usize)?,None=>0.0};
                if !extra.is_finite() || extra<0.0 {return None;}
                Some(fit+extra)
            };
            let Some(la)=body_bound(ea,2) else{eprintln!("CONTINUOUS_TWIST_CERTIFICATE joint={i} body={ea} endpoint=A polar_uncertified");return false;};
            let Some(lb)=body_bound(eb,6) else{eprintln!("CONTINUOUS_TWIST_CERTIFICATE joint={i} body={eb} endpoint=B polar_uncertified");return false;};
            // Device f32 radius receives a conservative absolute margin.
            if !crate::continuous_twist::certified_segment(a.rows[i][1]-1e-5,la,lb) {
                eprintln!("CONTINUOUS_TWIST_CERTIFICATE joint={i} radius={} length_a={la} length_b={lb}",a.rows[i][1]);
                static PRINTED:std::sync::atomic::AtomicUsize=std::sync::atomic::AtomicUsize::new(0);
                let index=PRINTED.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
                if std::env::var_os("PPF_INTERNAL_TEST_CONTINUOUS_TWIST").as_deref()==Some(std::ffi::OsStr::new("1")) && (index==0 || index==10) {
                    for (body,offset) in [(ea,2),(eb,6)] {
                        if body==u32::MAX {continue;}
                        if let (Some(Some((c0,e0))),Some(Some((c1,e1))))=(a.covariances.get(body as usize),b.covariances.get(body as usize)) {
                            let q0:[f64;4]=std::array::from_fn(|j|a.rows[i][offset+j]);
                            let q1:[f64;4]=std::array::from_fn(|j|b.rows[i][offset+j]);
                            eprintln!("CONTINUOUS_TWIST_RADIUS_INPUT sample={index} body={body} c0={c0:?} c1={c1:?} q0={q0:?} q1={q1:?} error={}",e0.max(*e1));
                        }
                    }
                }
                return false;
            }
        }
        true
    }
    /// Certify GPU fit -> start polar -> polar path -> end GPU fit as three
    /// separate paths. No trial is published until all selected joints pass all
    /// pieces; virtual states are never used for physical force evaluation.
    pub(crate) unsafe fn advance_segment_pieces<D:Device>(&mut self,device:&mut D,view:View,selected:&[bool],a:&TwistPoseSnapshot,b:&TwistPoseSnapshot,commanded:Option<&[f64]>)->Result<bool,Fault>{
        #[cfg(all(test, feature="cpu"))]
        if PRIVATE_FORCE_UNCERTIFIED.swap(false,std::sync::atomic::Ordering::SeqCst) {return Ok(false);}
        use crate::continuous_twist::{polar_segment_parts,certified_segment,PolarSegmentParts};
        let endpoints=self.twist_endpoints.clone();
        let n=selected.len();
        let bodies=a.covariances.len();
        if endpoints.len()!=n || a.rows.len()!=n || b.rows.len()!=n || b.covariances.len()!=bodies {return Ok(false);}
        let mut parts:Vec<[Option<PolarSegmentParts>;2]>=vec![[None,None];n];
        let mut states0=vec![crate::data::PdrdRigidState::default();bodies];
        let mut states1=states0.clone();
        for i in 0..n {
            if !selected[i] {continue;}
            for (side,(body,offset)) in [(endpoints[i].0,2usize),(endpoints[i].1,6usize)].into_iter().enumerate() {
                if body==u32::MAX {continue;}
                let Some(Some((c0,e0)))=a.covariances.get(body as usize) else{return Ok(false);};
                let Some(Some((c1,e1)))=b.covariances.get(body as usize) else{return Ok(false);};
                let q0=std::array::from_fn(|j|a.rows[i][offset+j]);let q1=std::array::from_fn(|j|b.rows[i][offset+j]);
                let Some(p)=polar_segment_parts(*c0,*c1,q0,q1,e0.max(*e1)) else{
                    static PRINTED:std::sync::atomic::AtomicUsize=std::sync::atomic::AtomicUsize::new(0);
                    let count=PRINTED.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
                    if std::env::var_os("PPF_INTERNAL_TEST_CONTINUOUS_TWIST").as_deref()==Some(std::ffi::OsStr::new("1")) && (count==0 || count==10){
                        eprintln!("CONTINUOUS_TWIST_PARTS_NONE sample={count} body={body} c0={c0:?} c1={c1:?} q0={q0:?} q1={q1:?} error={}",e0.max(*e1));
                    }
                    return Ok(false);
                };
                states0[body as usize].rotation=p.polar0.map(|v|v as f32);
                states1[body as usize].rotation=p.polar1.map(|v|v as f32);
                parts[i][side]=Some(p);
            }
        }
        self.twist_virtual_start.size(device,bodies,AllocLabel("joints.twist_virtual_start"))?;
        self.twist_virtual_end.size(device,bodies,AllocLabel("joints.twist_virtual_end"))?;
        if bodies>0 {self.twist_virtual_start.write(device,0,&states0)?;self.twist_virtual_end.write(device,0,&states1)?;}
        let virtual0=self.twist_virtual_start.handle();let virtual1=self.twist_virtual_end.handle();
        let rows0=self.sample_twist_pose(device,View{state:virtual0,..view},selected)?;
        let rows1=self.sample_twist_pose(device,View{state:virtual1,..view},selected)?;
        let mut lengths=vec![[[0.0_f64;2];3];n];
        for i in 0..n {
            if !selected[i] {continue;}
            for (side,(body,offset)) in [(endpoints[i].0,2usize),(endpoints[i].1,6usize)].into_iter().enumerate() {
                if body==u32::MAX {continue;}
                let Some(p)=parts[i][side] else{return Ok(false);};
                let Some(Some((c0,e0)))=a.covariances.get(body as usize) else{return Ok(false);};
                let Some(Some((c1,e1)))=b.covariances.get(body as usize) else{return Ok(false);};
                let q0=std::array::from_fn(|j|rows0[i][offset+j]);let q1=std::array::from_fn(|j|rows1[i][offset+j]);
                let Some(v0)=polar_segment_parts(*c0,*c0,q0,q0,*e0) else{return Ok(false);};
                let Some(v1)=polar_segment_parts(*c1,*c1,q1,q1,*e1) else{return Ok(false);};
                let extra=match commanded {Some(v)=>match v.get(body as usize){Some(x) if x.is_finite() && *x>=0.0=>*x,_=>return Ok(false)},None=>0.0};
                lengths[i][0][side]=p.start+v0.start;
                lengths[i][1][side]=v0.start+p.path+v1.start+extra;
                lengths[i][2][side]=v1.start+p.end;
            }
        }
        let Some(mut trial)=self.trial_twist.clone() else{return Err(Fault::Platform{call:"continuous twist segment",detail:"missing trial".into()});};
        let anchor=trial.theta.clone();
        let stages=[(&a.rows,&rows0),(&rows0,&rows1),(&rows1,&b.rows)];
        for (stage,(from,to)) in stages.into_iter().enumerate() {
            for i in 0..n {
                if selected[i] && !certified_segment(from[i][1]-1e-5,lengths[i][stage][0],lengths[i][stage][1]) {
                    eprintln!("CONTINUOUS_TWIST_SEGMENT_REJECT joint={i} piece={stage} radius={} length_a={} length_b={}",from[i][1],lengths[i][stage][0],lengths[i][stage][1]);
                    static PRINTED:std::sync::atomic::AtomicUsize=std::sync::atomic::AtomicUsize::new(0);
                    let count=PRINTED.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
                    if std::env::var_os("PPF_INTERNAL_TEST_CONTINUOUS_TWIST").as_deref()==Some(std::ffi::OsStr::new("1")) && (count==0 || count==10){
                        for (side,body) in [endpoints[i].0,endpoints[i].1].into_iter().enumerate(){
                            if let (Some(p),Some(Some((c0,e0))),Some(Some((c1,e1))))=(parts[i][side],a.covariances.get(body as usize),b.covariances.get(body as usize)){
                                eprintln!("CONTINUOUS_TWIST_PIECE_INPUT sample={count} body={body} path={} start={} end={} c0={c0:?} c1={c1:?} error={}",p.path,p.start,p.end,e0.max(*e1));
                            }
                        }
                    }
                    return Ok(false);
                }
            }
            let principal:Vec<_>=to.iter().map(|r|r[0]).collect();
            if trial.advance_certified(&principal).is_err(){return Ok(false);}
        }
        // A single physical segment may not hide a full revolution in the
        // virtual correction detour. Accumulation happens over accepted steps.
        if !crate::continuous_twist::net_segment_within_pi(selected,&anchor,&trial.theta) {return Ok(false);}
        self.trial_twist=Some(trial);
        self.upload_certified_twist_context(device,selected)?;
        Ok(true)
    }
    /// Read principal chart twist and projected quaternion radius from the
    /// actual fitted rigid state, using the same chart and joint frames as the
    /// force kernels. Returned rows follow the validated full joint order.
    /// The caller must keep `view.state` alive through the device launch.
    pub(crate) unsafe fn sample_twist_pose<D:Device>(&mut self,device:&mut D,view:View,selected:&[bool])->Result<Vec<[f64;10]>,Fault>{
        if self.swing_candidate.is_some() || self.candidate_token.is_some() {return Err(Fault::Platform{call:"continuous twist sample",detail:"trial token pending".into()});}
        self.last_twist_sample_failure=None;
        self.twist_sample_sequence=self.twist_sample_sequence.checked_add(1).ok_or_else(||Fault::Platform{call:"continuous twist sample",detail:"sample sequence exhausted".into()})?;
        let n=view.count as usize;
        if selected.len()!=n {return Err(Fault::Platform{call:"continuous twist sample",detail:"selection count mismatch".into()});}
        if n==0 {return Ok(Vec::new());}
        if view.control_count as usize!=n {
            return Err(Fault::Platform{call:"continuous twist sample",detail:"missing full-order angular control records".into()});
        }
        // Reused with the owning Joints buffers: local device buffers have no
        // releasing Drop and would leak once for every sample or error path.
        let values=&mut self.twist_sample_values;
        let faults=&mut self.twist_sample_faults;
        let selection=&mut self.twist_sample_selection;
        values.size(device,n*10,AllocLabel("joints.twist_sample"))?;
        faults.size(device,n,AllocLabel("joints.twist_sample_fault"))?;
        selection.size(device,n,AllocLabel("joints.twist_sample_selection"))?;
        selection.write(device,0,&selected.iter().map(|v|if *v {1.0} else {0.0}).collect::<Vec<_>>())?;
        values.seed(device,&vec![0.0;n*10])?;
        faults.seed(device,&vec![0;n])?;
        let args=JointAngularControlApplyArgs { records:view.control,twist_context:view.twist_context,material_spin_context:view.material_spin_context,state:view.state,reference:view.reference,
            bodies:view.bodies,base:view.base,dim:view.dim,dt:view.dt,action:4,
            input:selection.handle(),output:values.handle(),fault:faults.handle(),tangent_cache:view.control_tangent_factors,tangent_offsets:view.control_tangent_offsets,tangent_published:view.control_tangent_published,epoch:0,
            count:view.control_count,seam_arena_count:0 };
        device.launch("joints.twist_sample",&args,view.control_count)?;
        faults.download(device)?;
        values.download(device)?;
        let failure=faults.host().iter().enumerate().find(|(_,v)|**v!=0).map(|(i,v)|(i,*v));
        let (index,code)=match failure {
            Some(found)=>found,
            None=>return Ok(values.host().chunks_exact(10).map(|v|std::array::from_fn(|i|f64::from(v[i]))).collect()),
        };
        let lane=match std::panic::catch_unwind(std::panic::AssertUnwindSafe(||faults.host().to_vec())) {
            Ok(lane)=>lane,
            Err(_)=>return Err(twist_sample_fault(index,code)),
        };
        self.last_twist_sample_failure=twist_sample_failure(&lane,&self.twist_endpoints,self.twist_sample_sequence);
        Err(twist_sample_fault_with_capture(index,code,||self.capture_failure(device,view,"continuous_twist_sample_unknown",index,code,Some(&lane),"twist_sample",None,Some(4))))
    }
    pub(crate) fn last_twist_sample_failure(&self)->Option<TwistSampleFailure> { self.last_twist_sample_failure }
    pub(super) fn twist_sample_sequence(&self)->u64 {self.twist_sample_sequence}
    /// The first host chart retry supports rootless, non-FEM continuous joints
    /// without authored angular drives. ROOT and event checks are made by the
    /// owning solver state before entering this path.
    pub(super) fn swing_retry_scope_supported(&self)->bool {
        self.twist_selected.as_ref().is_some_and(|s|s.iter().any(|v|*v)) &&
            !self.soft.active() && self.drive_count==0
    }
    pub(super) fn material_root_retry_scope_supported(&self)->bool {
        self.material_spin_active() && (!self.soft.active() || self.soft.rigid_cross_retry_supported(&self.curves.binding_metadata)) && self.drive_count==0
    }
    #[cfg(all(test,feature="cpu"))]
    pub(super) fn private_swing_scope_facts(&self)->(usize,bool,u32,usize) {
        (self.twist_selected.as_ref().map_or(0,|s|s.iter().filter(|v|**v).count()),self.soft.active(),self.drive_count,self.drives.len())
    }
    /// Stores validated accepted history only; does not enable device mode.
    pub(crate) fn restore_accepted_twist(&mut self,history:crate::continuous_twist::AcceptedTwist,scene:&str,joints:&[String],time:f64)->Result<(),Fault>{
        history.validate(scene,joints,time).map_err(|e|Fault::Platform{call:"continuous twist restore",detail:e.into()})?;
        crate::continuous_twist::project_device(&history.theta).map_err(|e|Fault::Platform{call:"continuous twist restore",detail:e.into()})?;
        self.accepted_twist=Some(history);Ok(())
    }
    pub(crate) fn begin_twist_trial(&mut self)->Result<(),Fault>{
        if self.trial_twist.is_some(){return Err(Fault::Platform{call:"continuous twist trial",detail:"trial already active".into()});}
        self.trial_twist=self.accepted_twist.as_ref().map(|h|h.trial());Ok(())
    }
    /// Upload a branch only after the caller has certified the real pose path.
    /// `enabled` follows the full validated joint order; no curve-binding
    /// indices are accepted here. This method does not itself enable a model.
    pub(crate) fn upload_certified_twist_context<D:Device>(&mut self,device:&mut D,enabled:&[bool])->Result<(),Fault>{
        let accepted=self.accepted_twist.as_ref().ok_or_else(||Fault::Platform{call:"continuous twist context",detail:"missing accepted history".into()})?;
        let trial=self.trial_twist.as_ref().ok_or_else(||Fault::Platform{call:"continuous twist context",detail:"missing certified trial".into()})?;
        let n=self.twist_context.len();
        if enabled.len()!=n || accepted.theta.len()!=n || trial.theta.len()!=n {
            return Err(Fault::Platform{call:"continuous twist context",detail:"full joint order/count mismatch".into()});
        }
        let current=crate::continuous_twist::project_device(&trial.theta).map_err(|e|Fault::Platform{call:"continuous twist context",detail:e.into()})?;
        let reference=crate::continuous_twist::project_device(&accepted.theta).map_err(|e|Fault::Platform{call:"continuous twist context",detail:e.into()})?;
        let context:Vec<_>=(0..n).map(|i|crate::continuous_twist::AngularTwistEvalContext{
            current_anchor:current[i],reference_anchor:reference[i],
            continuous:u32::from(enabled[i]),initialized:u32::from(enabled[i]),
        }).collect();
        if n>0 { self.twist_context.write(device,0,&context)?; }
        Ok(())
    }
    pub(crate) fn advance_twist_certified(&mut self,principal:&[f64])->Result<(),Fault>{
        let trial=self.trial_twist.as_mut().ok_or_else(||Fault::Platform{call:"continuous twist trial",detail:"missing trial".into()})?;
        trial.advance_certified(principal).map_err(|e|Fault::Platform{call:"continuous twist trial",detail:e.into()})
    }
    pub(crate) fn abort_twist_trial(&mut self){self.trial_twist=None;}
    #[cfg(test)]
    pub(super) fn test_twist_trial_active(&self)->bool {self.trial_twist.is_some()}
    pub(crate) fn prepare_twist_commit(&self,time:f64)->Result<crate::continuous_twist::AcceptedTwist,Fault>{
        let trial=self.trial_twist.clone().ok_or_else(||Fault::Platform{call:"continuous twist commit",detail:"missing trial".into()})?;
        let mut accepted=self.accepted_twist.clone().ok_or_else(||Fault::Platform{call:"continuous twist commit",detail:"missing accepted history".into()})?;
        accepted.commit(trial,time).map_err(|e|Fault::Platform{call:"continuous twist commit",detail:e.into()})?;
        crate::continuous_twist::project_device(&accepted.theta).map_err(|e|Fault::Platform{call:"continuous twist commit",detail:e.into()})?;
        Ok(accepted)
    }
    pub(crate) fn finish_twist_commit(&mut self,accepted:crate::continuous_twist::AcceptedTwist){
        #[cfg(all(test, feature="cpu"))]
        PRIVATE_TWIST_COMMITS.fetch_add(1,std::sync::atomic::Ordering::SeqCst);
        if std::env::var_os("PPF_INTERNAL_TEST_CONTINUOUS_TWIST").as_deref()==Some(std::ffi::OsStr::new("1")) {
            if let (Some(old),Some(selected))=(self.accepted_twist.as_ref(),self.twist_selected.as_ref()) {
                for (i,((&before,&after),enabled)) in old.theta.iter().zip(&accepted.theta).zip(selected).enumerate() {
                    if *enabled && (before/std::f64::consts::PI).floor()!=(after/std::f64::consts::PI).floor() {
                        eprintln!("CONTINUOUS_TWIST_ACCEPTED joint={i} time={} theta_rad={after}",accepted.time);
                    }
                }
            }
        }
        self.accepted_twist=Some(accepted);self.trial_twist=None;
    }
    pub fn reference_handle(&self)->Handle {self.reference.handle()}
    pub fn twist_context(&self)->(Handle,u32) {(self.twist_context.handle(),self.twist_context.len() as u32)}
    pub fn has_extensions(&self)->bool { self.limits.len()>0 || self.soft.has_extensions() }
    pub fn reset_candidate_retry(&mut self)->Result<(),Fault> {
        if self.candidate_token.is_some() || self.swing_candidate.is_some(){return Err(Fault::Platform{call:"joint retry",detail:"unconsumed candidate token".into()});}
        self.soft.clear_candidate_rejection();self.candidate_domain_dt=None;self.material_spin_retry_dt=None;self.candidate_fatal=false;self.candidate_trial=self.candidate_trial.checked_add(1).ok_or_else(||Fault::Platform{call:"joint retry",detail:"candidate trial counter overflow".into()})?;Ok(())
    }
    /// Qualify only the immediately preceding real B31 Iterate action-4 lane.
    /// The actual accepted reference is evaluated in separate scratch, so it
    /// cannot overwrite the failed lane or manufacture a fresh verdict.
    pub(super) unsafe fn mark_swing_sample_trial<D:Device>(&mut self,device:&mut D,
        map:&super::rigid_map::RigidMap,selected:&[bool],origin_sequence:u64,start:f64,dt:f32)->Result<bool,Fault> {
        if !self.swing_retry_scope_supported() || self.swing_candidate.is_some() || self.candidate_token.is_some() || self.candidate_fatal ||
            !start.is_finite() || self.start_time_s.to_bits()!=start.to_bits() || !dt.is_finite() || dt<=0.0 ||
            self.twist_selected.as_deref()!=Some(selected) || origin_sequence.checked_add(1)!=Some(self.twist_sample_sequence) {
            return Ok(false);
        }
        let Some(sample)=self.last_twist_sample_failure else{return Ok(false)};
        let n=self.control.len();
        if n==0 || self.ball.len()!=n || selected.len()!=n || self.twist_sample_faults.len()!=n ||
            sample.call_sequence!=self.twist_sample_sequence || sample.code!=6 ||
            !selected.get(sample.joint_index).copied().unwrap_or(false) || self.reference.len()!=map.n_bodies {
            return Ok(false);
        }
        self.twist_sample_faults.download(device)?;
        let lane=self.twist_sample_faults.host().to_vec();
        if lane.iter().enumerate().any(|(i,&v)|v!=if i==sample.joint_index{6}else{0}) {return Ok(false);}
        if self.fault.len()>0 {self.fault.download(device)?;if self.fault.host().iter().any(|&v|v!=0){return Ok(false);}}
        if !self.curves.snapshot_faults(device)?.is_empty(){return Ok(false);}

        let count=n.checked_mul(10).ok_or_else(||Fault::Platform{call:"continuous twist reference",detail:"sample size overflow".into()})?;
        self.twist_reference_values.size(device,count,AllocLabel("joints.twist_reference_values"))?;
        self.twist_reference_faults.size(device,n,AllocLabel("joints.twist_reference_faults"))?;
        self.twist_reference_values.seed(device,&vec![0.0;count])?;
        self.twist_reference_faults.seed(device,&vec![0;n])?;
        let reference=self.reference.handle();
        #[allow(unused_mut)] let mut sampled_reference=reference;
        // Test-only invalid reference injection affects an independent copy,
        // never the accepted fitted reference or physical state.
        #[cfg(all(test, feature="cpu"))]
        if PRIVATE_INVALID_REFERENCE.swap(false,std::sync::atomic::Ordering::SeqCst) {
            if map.n_bodies<=7 {return Err(Fault::Platform{call:"private reference",detail:"fixture lacks body 7".into()});}
            self.twist_virtual_start.size(device,map.n_bodies,AllocLabel("private.reference.copy"))?;
            device.copy(self.twist_virtual_start.handle(),0,reference,0,map.n_bodies*std::mem::size_of::<crate::data::PdrdRigidState>())?;
            let mut row=[crate::data::PdrdRigidState::default()];self.twist_virtual_start.read(device,7,&mut row)?;
            row[0].rotation[0]=f32::NAN;self.twist_virtual_start.write(device,7,&row)?;
            sampled_reference=self.twist_virtual_start.handle();
        }
        let args=JointAngularControlApplyArgs {records:self.control.handle(),twist_context:self.twist_context.handle(),material_spin_context:self.material_spin_context.handle(),state:sampled_reference,reference,
            bodies:map.n_bodies as u32,base:map.body_base as u32,dim:map.dim as u32,dt,action:4,input:self.twist_sample_selection.handle(),
            output:self.twist_reference_values.handle(),fault:self.twist_reference_faults.handle(),tangent_cache:self.control_tangent_factors.handle(),tangent_offsets:self.control_tangent_offsets.handle(),
            tangent_published:self.control_tangent_published.handle(),epoch:0,count:n as u32,seam_arena_count:0};
        device.launch("joints.twist_reference_qualification",&args,n as u32)?;
        self.twist_reference_faults.download(device)?;self.twist_reference_values.download(device)?;
        if self.twist_reference_faults.host().iter().any(|&v|v!=0) ||
            self.twist_reference_values.host().chunks_exact(10).enumerate().any(|(i,row)|
                selected[i] && (row.iter().any(|v|!v.is_finite()) || row[1]<1e-4)) {return Ok(false);}
        // Re-download: the reference dispatch must not have changed the actual
        // sampler fault bank, and no stale readback may qualify the retry.
        self.twist_sample_faults.download(device)?;
        if self.twist_sample_faults.host()!=lane.as_slice(){return Ok(false);}
        self.swing_candidate=Some(SampleSwingTrial{generation:self.candidate_trial,start_bits:start.to_bits(),dt_bits:dt.to_bits(),
            origin_sequence,sample,lane,selected:selected.to_vec(),reference});
        Ok(true)
    }
    /// Recheck the original actual lane and unrelated banks before and after
    /// rollback. No buffer is cleared to make an ineligible trial pass.
    pub(super) fn swing_candidate_retry<D:Device>(&mut self,device:&mut D)->Result<Option<SampleSwingTrial>,Fault>{
        let Some(token)=self.swing_candidate.clone() else{return Ok(None)};
        if token.generation!=self.candidate_trial || token.start_bits!=self.start_time_s.to_bits() ||
            token.sample.call_sequence!=self.twist_sample_sequence || token.origin_sequence.checked_add(1)!=Some(token.sample.call_sequence) ||
            self.last_twist_sample_failure!=Some(token.sample) || self.twist_selected.as_deref()!=Some(token.selected.as_slice()) ||
            self.reference.handle()!=token.reference || self.candidate_token.is_some() || self.candidate_fatal {return Ok(None);}
        self.twist_sample_faults.download(device)?;
        if self.twist_sample_faults.host()!=token.lane.as_slice(){return Ok(None);}
        self.twist_reference_faults.download(device)?;
        if self.twist_reference_faults.host().iter().any(|&v|v!=0){return Ok(None);}
        if self.fault.len()>0 {self.fault.download(device)?;if self.fault.host().iter().any(|&v|v!=0){return Ok(None);}}
        if !self.curves.snapshot_faults(device)?.is_empty(){return Ok(None);}
        Ok(Some(token))
    }
    pub(super) fn consume_swing_candidate<D:Device>(&mut self,device:&mut D,expected:&SampleSwingTrial)->Result<(),Fault>{
        if self.swing_candidate_retry(device)?.as_ref()!=Some(expected) {
            return Err(Fault::Platform{call:"continuous twist sample retry",detail:"candidate token or native bank changed".into()});
        }
        // The bank was certified as exactly one nonzero lane. Clear only that
        // lane after rollback, leaving all unrelated control/curve banks intact.
        device.write(self.twist_sample_faults.handle(),expected.sample.joint_index*std::mem::size_of::<u32>(),&0u32.to_ne_bytes())?;
        self.twist_sample_faults.download(device)?;
        if self.twist_sample_faults.host().iter().any(|&v|v!=0) {
            return Err(Fault::Platform{call:"continuous twist sample retry",detail:"sample lane cleanup failed".into()});
        }
        self.swing_candidate=None;self.last_twist_sample_failure=None;Ok(())
    }
    pub(super) fn material_candidate_retry<D:Device>(&mut self,device:&mut D)->Result<Option<super::soft_joints::MaterialFrameCandidateRejection>,Fault>{
        let Some(token)=self.soft.candidate_rejection().cloned() else{return Ok(None)};
        if token.generation!=self.candidate_trial || token.start_bits!=self.start_time_s.to_bits() || self.candidate_token.is_some() || self.candidate_fatal {return Ok(None);}
        if self.fault.len()>0 {self.fault.download(device)?;if self.fault.host().iter().any(|&v|v!=0){return Ok(None);}}
        if !self.curves.snapshot_faults(device)?.is_empty(){return Ok(None);}
        Ok(Some(token))
    }
    /// Request subdivision only when every native fault bank is clean. A
    /// separate reason avoids manufacturing an angular fault to enter retry.
    pub(crate) fn mark_uncertified_path_retry<D:Device>(&mut self,device:&mut D,dt:f32,stage:&'static str)->Result<(),Fault>{
        if !matches!(stage,"B31"|"B32") || !dt.is_finite() || dt<=0.0 || self.candidate_token.is_some() {
            return Err(Fault::Platform{call:"continuous twist path",detail:"invalid or duplicate retry request".into()});
        }
        let curves=self.curves.snapshot_faults(device)?;
        if self.fault.len()>0 {self.fault.download(device)?;}
        let control=self.fault.host().to_vec();
        if !curves.is_empty() || control.iter().any(|v|*v!=0) {
            self.candidate_fatal=true;
            return Err(Fault::Platform{call:"continuous twist path",detail:format!("native fault banks not clean: curves={curves:?} control={control:?}")});
        }
        eprintln!("CONTINUOUS_TWIST_PATH_RETRY stage={stage} start={} dt={dt}",self.start_time_s);
        self.candidate_domain_dt=Some(dt);
        self.candidate_token=Some(CandidateFaultToken{trial:self.candidate_trial,start_bits:self.start_time_s.to_bits(),dt_bits:dt.to_bits(),curves,control,reason:CandidateRetryReason::UncertifiedPath(stage)});
        Ok(())
    }
    /// Revalidate the existing B31/B32 chart token before exporting it to
    /// the rootless continuous host coordinator. The sampler bank must also
    /// remain clean; this token is distinct from an action-4 swing failure.
    pub(super) fn host_chart_candidate<D:Device>(&mut self,device:&mut D)->Result<Option<CandidateFaultToken>,Fault>{
        let Some(token)=self.candidate_token.clone() else{return Ok(None)};
        if self.candidate_fatal || self.swing_candidate.is_some() || token.trial!=self.candidate_trial ||
            token.start_bits!=self.start_time_s.to_bits() || Some(f32::from_bits(token.dt_bits))!=self.candidate_domain_dt {
            return Ok(None);
        }
        if self.fault.len()>0 {self.fault.download(device)?;}
        if self.fault.host()!=token.control.as_slice() || self.curves.snapshot_faults(device)?!=token.curves {return Ok(None);}
        if self.twist_sample_faults.len()>0 {self.twist_sample_faults.download(device)?;if self.twist_sample_faults.host().iter().any(|&v|v!=0){return Ok(None);}}
        Ok(Some(token))
    }
    pub fn candidate_has_fatal_fault(&self)->bool {self.candidate_fatal}
    pub fn consume_candidate_faults<D:Device>(&mut self,device:&mut D)->Result<(),Fault>{
        let token=self.candidate_token.clone().ok_or_else(||Fault::Platform{call:"joint retry",detail:"missing candidate token".into()})?;
        if let CandidateRetryReason::UncertifiedPath(stage)=token.reason {::log::info!("continuous twist: rolled back uncertified {stage} path");}
        if token.trial!=self.candidate_trial || token.start_bits!=self.start_time_s.to_bits()
            || Some(f32::from_bits(token.dt_bits))!=self.candidate_domain_dt {
            return Err(Fault::Platform{call:"joint retry",detail:"candidate token identity mismatch".into()});
        }
        if self.fault.len()>0 {self.fault.download(device)?;}
        if self.fault.host()!=token.control.as_slice() {
            return Err(Fault::Platform{call:"joint retry",detail:"control fault snapshot changed".into()});
        }
        if self.curves.active(){self.curves.consume_candidate_faults(device,&token.curves)?;}
        if self.fault.len()>0 {self.fault.seed(device,&vec![0;self.fault.len()])?;}
        self.candidate_token=None;
        Ok(())
    }    pub fn candidate_retry_dt(&self)->Option<f32> {self.candidate_domain_dt}
    pub fn set_step_start_time(&mut self,time_s:f64)->Result<(),Fault> {
        if self.has_extensions() && (!time_s.is_finite() || time_s<0.0) {
            return Err(Fault::Platform { call:"joint extension step domain",detail:format!("fault132: invalid physical step start {time_s}") });
        }
        self.start_time_s=time_s;
        self.end_time_s=None;
        self.soft.set_step_clock(time_s,None);
        Ok(())
    }
    pub fn set_step_end_time(&mut self,time_s:f64,dt_s:f32)->Result<(),Fault> {
        if !time_s.is_finite() || time_s<=self.start_time_s ||
            (time_s-self.start_time_s) as f32!=dt_s {
            return Err(Fault::Platform {call:"joint physical endpoint",detail:"invalid native event endpoint".into()});
        }
        self.end_time_s=Some(time_s);self.soft.set_step_clock(self.start_time_s,Some(time_s));Ok(())
    }
    pub fn active(&self)->bool { self.ball.len()>0 || self.soft.active() }
    /// True when a directional family exists, so the fault lane is allocated.
    pub fn has_control(&self)->bool { self.control.len()>0 || self.soft.active() }

    /// Typed geometry gate over the directional family at one frozen pose.
    ///
    /// `view.state` must be the CURRENT initialized rigid fit and
    /// `view.reference` the actual substep-start fit. Called once per frozen
    /// Newton pose, before the preconditioner that consumes the joint blocks,
    /// the reduced right-hand side and any PCG application; never synchronized
    /// inside a PCG application. `dt` must be the actual positive step size the
    /// damping term divides by.
    ///
    /// # Safety
    /// Every handle in `view` must name a live allocation of the stated length.
    pub unsafe fn check<D:Device>(&mut self, device:&mut D, view:View) -> Result<(),Fault> {
        self.check_at(device,view,"unspecified_gate")
    }
    unsafe fn check_at<D:Device>(&mut self,device:&mut D,view:View,stage:&str)->Result<(),Fault> {
        let curve_faults=if let Some(c)=view.curves.filter(|_|stage!="substep_start") {
            self.curves.clear(device)?;
            view.add(device,3,c.dummy,c.dummy)?;
            self.curves.snapshot_faults(device)?
        } else { Vec::new() };
        let control_result=if view.control_count==0 { Ok(()) } else {
            let records=self.control.handle();
            let input=self.ball.handle();
            let fault=self.fault.handle();
            let args=JointAngularControlApplyArgs { records,twist_context:view.twist_context,material_spin_context:view.material_spin_context,state:view.state,reference:view.reference,
                bodies:view.bodies,base:view.base,dim:view.dim,dt:view.dt,action:3,
                input,output:input,fault,tangent_cache:view.control_tangent_factors,tangent_offsets:view.control_tangent_offsets,tangent_published:view.control_tangent_published,epoch:0,
                count:view.control_count,seam_arena_count:0 };
            device.launch("joints.angular_control",&args,view.control_count)?;
            view.add_extension(device,3,input,input,false)?;
            self.fault.download(device)?;
            self.reap(device,view,stage)
        };
        let lane=if view.control_count==0 { &[][..] } else { self.fault.host() };
        let curve_allowed=curve_faults.iter().all(|(family,_,code)|*family=="angular" && matches!(*code,5|6|134|135));
        let lane_allowed=lane.iter().all(|code|matches!(*code,0|5|6|134|135));
        let any=!curve_faults.is_empty() || lane.iter().any(|code|*code!=0);
        if self.material_spin_active() && (curve_faults.iter().any(|(family,_,code)|*family=="angular" && *code==8) || lane.iter().any(|code|*code==8))
            && curve_faults.iter().all(|(family,_,code)|*family=="angular" && *code==8)
            && lane.iter().all(|code|matches!(*code,0|8)) {self.material_spin_retry_dt=Some(view.dt);}
        let retry=stage=="candidate_after_B32" && any && curve_allowed && lane_allowed;
        self.candidate_domain_dt=if retry {Some(view.dt)} else {None};
        self.candidate_fatal=stage=="candidate_after_B32" && any && !retry && self.material_spin_retry_dt.is_none();
        self.candidate_token=if retry {Some(CandidateFaultToken{trial:self.candidate_trial,start_bits:self.start_time_s.to_bits(),dt_bits:view.dt.to_bits(),curves:curve_faults.clone(),control:lane.to_vec(),reason:CandidateRetryReason::GeometryBank})} else {None};
        if !curve_faults.is_empty() {
            if let Some(&("angular",index,code))=curve_faults.iter().find(|(family,_,_)|*family=="angular") {
                if index < view.control_count as usize {
                    observe_best_effort(||if let Err(capture)=self.capture_failure(device,view,stage,index,code,None,"curve_angular_snapshot",Some(&curve_faults),Some(3)) {eprintln!("JOINT_CHART_FAILURE_CAPTURE_ERROR {capture}");});
                }
            }
            observe_best_effort(||eprintln!("JOINT_FAULT_BANK_SNAPSHOT stage={stage} retry_eligible={retry} curves={curve_faults:?} control={lane:?}"));
            return Err(super::restoring::Curves::fault_from_snapshot(stage,&curve_faults,retry).expect("nonempty snapshot"));
        }
        control_result
    }
    fn reap<D:Device>(&mut self, device:&mut D,view:View,stage:&str) -> Result<(),Fault> {
        if candidate_domain_lane(stage,self.fault.host()) {self.candidate_domain_dt=Some(view.dt);}
        if let Some((index,code)) = self.fault.host().iter().enumerate().find(|(_,c)| **c != 0).map(|(i,c)| (i,*c)) {
            // Capture every native refusal, including ordinary runs. Reads only
            // the failed gate's exact inputs; capture errors retain its refusal.
            observe_best_effort(||if let Err(capture)=self.capture_failure(device,view,stage,index,code,Some(self.fault.host()),"control_gate",None,Some(3)) {
                eprintln!("JOINT_CHART_FAILURE_CAPTURE_ERROR {capture}");
            });
            let name=match code {
                128=>"extension chart unusable at current pose",129=>"extension chart unusable at start reference",
                130=>"extension body index out of range",131=>"extension reduced/block offset out of range",
                132=>"extension physical step domain invalid",133=>"extension record/arithmetic nonfinite",
                134=>"finite extension current twist outside open chart domain",
                135=>"finite extension current swing outside open chart domain",
                136=>"continuous twist history invalid or uninitialized",
                _=>FAULT_NAMES.get(code as usize).copied().unwrap_or("unknown fault code"),
            };
            return Err(Fault::Platform { call:"joint angular control geometry", detail:format!("joint index {index}: {name} (code {code})") });
        }
        Ok(())
    }
    fn capture_failure<D:Device>(&self,device:&mut D,view:View,stage:&str,index:usize,code:u32,lane:Option<&[u32]>,lane_source:&str,curve_faults:Option<&[(&'static str,usize,u32)]>,action:Option<u32>)->Result<(),Fault> {
        let mut packed=[AngularControlGpu {body_a:0,body_b:0,enabled:0,coordinate_model:0,
            frame_a_wxyz:[0.;4],frame_b_wxyz:[0.;4],rest_relative_wxyz:[0.;4],
            stiffness_nm_per_rad:[0.;3],damping_nms_per_rad:[0.;3]}];
        self.control.read(device,index,&mut packed)?;
        let row=packed[0];
        let mut endpoints=Vec::new();
        for (role,body) in [("parent",row.body_a),("child",row.body_b)] {
            if body==crate::joint_angular_control_gpu::CONTROL_WORLD_ENDPOINT {
                endpoints.push(serde_json::json!({"role":role,"body_index":body,"world_endpoint":true}));continue;
            }
            if body>=view.bodies {continue;}
            let mut current=[crate::data::PdrdRigidState::default()];
            let mut reference=[crate::data::PdrdRigidState::default()];
            // SAFETY: repr(C) Pod PdrdRigidState, initialized one-element array;
            // body bounds checked above, handle and length are gate invariants.
            let dst=unsafe {std::slice::from_raw_parts_mut(current.as_mut_ptr().cast::<u8>(),std::mem::size_of_val(&current))};
            device.read(view.state,body as usize*std::mem::size_of::<crate::data::PdrdRigidState>(),dst)?;
            self.reference.read(device,body as usize,&mut reference)?;
            let describe=|v:crate::data::PdrdRigidState|serde_json::json!({
                "rotation_column_major":v.rotation,"rotation_u32_bits":v.rotation.map(f32::to_bits),
                "anchor":v.anchor,"anchor_u32_bits":v.anchor.map(f32::to_bits),
                "centroid_relative":v.x,"centroid_relative_u32_bits":v.x.map(f32::to_bits),
                "reference_inertia":v.reference_inertia,"reference_inertia_u32_bits":v.reference_inertia.map(f32::to_bits),
                "mass_total":v.mass_total,"mass_total_u32_bits":v.mass_total.to_bits(),"count":v.count});
            endpoints.push(serde_json::json!({"role":role,"body_index":body,
                "current":describe(current[0]),"reference":describe(reference[0])}));
        }
        eprintln!("JOINT_CHART_FAILURE_JSON {}",serde_json::json!({
            "schema":"ppf.joint-chart-failure.v1","joint_index":index,"fault_code":code,"fault_lane":lane,"fault_lane_source":lane_source,"curve_faults_snapshot":curve_faults,"stage":stage,"action":action,"sample_call_sequence":if action==Some(4) {Some(self.twist_sample_sequence)} else {None},
            "physical_start_s":self.start_time_s,"sample_time_s":view.sample_time_s,"dt_s":view.dt,
            "body_count":view.bodies,"reduced_base":view.base,"reduced_dim":view.dim,
            "packed_control":{"body_a":row.body_a,"body_b":row.body_b,"enabled":row.enabled,
                "coordinate_model":row.coordinate_model,"frame_a_wxyz":row.frame_a_wxyz,
                "frame_b_wxyz":row.frame_b_wxyz,"rest_relative_wxyz":row.rest_relative_wxyz,
                "frame_a_u32_bits":row.frame_a_wxyz.map(f32::to_bits),"frame_b_u32_bits":row.frame_b_wxyz.map(f32::to_bits),
                "rest_relative_u32_bits":row.rest_relative_wxyz.map(f32::to_bits),
                "stiffness_nm_per_rad":row.stiffness_nm_per_rad,"stiffness_u32_bits":row.stiffness_nm_per_rad.map(f32::to_bits),
                "damping_nms_per_rad":row.damping_nms_per_rad,"damping_u32_bits":row.damping_nms_per_rad.map(f32::to_bits)},
            "endpoint_states":endpoints,"authority":"actual unchanged GPU gate input matrices; host chart replay is diagnostic only",
            "scope":"gate refusal; source is reported by paired context when available, otherwise unknown; original native refusal retained"}));
        Ok(())
    }
}
impl View {
    /// Use the frozen operator only for PCG search directions. The ordinary
    /// add(action=1) remains the independent true-residual/reference path.
    /// Append only the prepared rigid families. Caller emits the ordinary
    /// family kernels for each corresponding None, plus V2 and extensions.
    pub unsafe fn encode_frozen_rigid(&self,encoder:&mut dyn Encoder,input:Handle,output:Handle)->Result<(),Fault> {
        if let Some(f)=self.frozen_angular {
            let args=super::kernels::RestoringAngularFrozenApplyArgs{count_rows:self.control_count,factors:f.factors,offsets:f.offsets,published:f.published,epoch:f.epoch,input,output,fault:f.fault,count:self.control_count,seam_arena_count:0};
            encoder.elements(&args,self.control_count)?;
        }
        if let Some(f)=self.frozen_ball {
            let args=super::kernels::RestoringBallFrozenApplyArgs{count_rows:self.count,factors:f.factors,offsets:f.offsets,published:f.published,epoch:f.epoch,input,output,fault:f.fault,count:self.count,seam_arena_count:0};
            encoder.elements(&args,self.count)?;
        }
        Ok(())
    }
    pub unsafe fn add_krylov<D:Device>(&self,device:&mut D,input:Handle,output:Handle)->Result<(),Fault> {
        if let Some(soft)=self.soft {soft.add_krylov(device,input,output)?;}
        let mut rigid=*self;
        rigid.soft=None;
        if self.frozen_angular.is_some() { rigid.control_count=0; }
        rigid.add_internal(device,1,input,output,self.frozen_ball.is_some(),true)?;
        if let Some(f)=self.frozen_angular {
            let args=super::kernels::RestoringAngularFrozenApplyArgs{count_rows:self.control_count,factors:f.factors,offsets:f.offsets,published:f.published,epoch:f.epoch,input,output,fault:f.fault,count:self.control_count,seam_arena_count:0};
            device.launch("restoring.angular.frozen.apply",&args,self.control_count)?;
        }
        if let Some(f)=self.frozen_ball {
            let args=super::kernels::RestoringBallFrozenApplyArgs{count_rows:self.count,factors:f.factors,offsets:f.offsets,published:f.published,epoch:f.epoch,input,output,fault:f.fault,count:self.count,seam_arena_count:0};
            device.launch("restoring.ball.frozen.apply",&args,self.count)?;
        }
        Ok(())
    }
    pub unsafe fn add_extension<D:Device>(&self,device:&mut D,action:u32,input:Handle,output:Handle,use_tangent_cache:bool)->Result<(),Fault> {
        if let Some(args) = self.extension_args(action, input, output, use_tangent_cache) {
            device.launch("joints.angular_extension", &args, args.count)?;
        }
        Ok(())
    }

    fn extension_args(&self, action: u32, input: Handle, output: Handle, use_tangent_cache: bool)
        -> Option<JointAngularExtensionApplyArgs>
    {
        (self.extension_count != 0).then(|| {
            let tangent=if use_tangent_cache && action==1 {self.frozen_extension_tangent} else {None};
            JointAngularExtensionApplyArgs {
            records:self.control,twist_context:self.twist_context,material_spin_context:self.material_spin_context,limits:self.limits,drives:self.drives,drive_count:self.drive_count,drive_offsets:self.drive_offsets,keys:self.device_keys,
            state:self.state,reference:self.reference,bodies:self.bodies,base:self.base,dim:self.dim,
            sample_time_s:self.sample_time_s,dt_s:self.dt,action:if action==1 && tangent.is_some(){7}else{action},input,
            tangent_cache:tangent.map(|f|f.factors).unwrap_or(self.extension_tangent_factors),
            tangent_offsets:tangent.map(|f|f.offsets).unwrap_or(self.extension_tangent_offsets),
            tangent_published:tangent.map(|f|f.published).unwrap_or(self.extension_tangent_published),epoch:tangent.map(|f|f.epoch).unwrap_or(0),
            output,fault:self.fault,
            count:self.extension_count,seam_arena_count:0,
        }})
    }

    // All rigid families that would be reevaluated in this path must have
    // frozen views. Unprepared scenes use the original dispatch sequence.
    pub(super) fn can_encode_krylov(&self) -> bool {
        self.count == 0 || (self.frozen_ball.is_some()
            && (self.control_count == 0 || self.frozen_angular.is_some()))
    }

    // Batch8 uses status-guarded fault-writing joint families.
    pub(super) unsafe fn encode_krylov_batch8(
        &self, encoder: &mut dyn Encoder, input: Handle, output: Handle, status: Handle, row_parallel: bool,
    ) -> Result<(), Fault> {
        if let Some(soft) = self.soft { soft.encode_krylov_batch8(encoder,input,output,status,row_parallel)?; }
        let (tangent_cache,tangent_epoch,epoch)=self.frozen_angular_tangent
            .map(|f|(f.factors,f.published,f.epoch)).unwrap_or((input,output,0));
        let isotropic = JointAngularApplyArgs {
            records: self.angular, state: self.state, reference: self.reference,
            bodies: self.bodies, base: self.base, dim: self.dim, dt: self.dt,
            action: 1, input, output, tangent_cache,tangent_epoch,epoch,
            count: self.count, seam_arena_count: 0,
        };
        encoder.elements(&isotropic, isotropic.count)?;
        if self.extension_count != 0 {
            let ext = super::kernels::ResidentGuardJointAngularExtensionApplyArgs {
                records:self.control, twist_context:self.twist_context,
                material_spin_context:self.material_spin_context, limits:self.limits,
                drives:self.drives, drive_count:self.drive_count, drive_offsets:self.drive_offsets, keys:self.device_keys,
                state:self.state, reference:self.reference, bodies:self.bodies, base:self.base,
                dim:self.dim, sample_time_s:self.sample_time_s, dt_s:self.dt, action:if self.frozen_extension_tangent.is_some(){7}else{1},
                input,tangent_cache:self.frozen_extension_tangent.map(|f|f.factors).unwrap_or(self.extension_tangent_factors),
                tangent_offsets:self.frozen_extension_tangent.map(|f|f.offsets).unwrap_or(self.extension_tangent_offsets),
                tangent_published:self.frozen_extension_tangent.map(|f|f.published).unwrap_or(self.extension_tangent_published),
                epoch:self.frozen_extension_tangent.map(|f|f.epoch).unwrap_or(0),output, fault:self.fault, status, count:self.extension_count, seam_arena_count:0,
            };
            encoder.elements(&ext, self.extension_count)?;
        }
        if let Some(f) = self.frozen_angular {
            let angular = super::kernels::ResidentGuardRestoringAngularFrozenApplyArgs {
                count_rows:self.control_count, factors:f.factors, offsets:f.offsets,
                published:f.published, epoch:f.epoch, input, output, fault:f.fault,
                status, count:self.control_count, seam_arena_count:0,
            };
            encoder.elements(&angular, self.control_count)?;
        }
        let f = self.frozen_ball.expect("batch8 requires frozen ball");
        let ball = super::kernels::JointPcgBatch8BallArgs {
            count_rows: self.count, factors: f.factors, offsets: f.offsets,
            published: f.published, epoch: f.epoch, input, output, fault: f.fault,
            status, count: self.count, seam_arena_count: 0,
        };
        encoder.elements(&ball, self.count)
    }

    pub(super) unsafe fn encode_krylov(
        &self, encoder: &mut dyn Encoder, input: Handle, output: Handle, row_parallel: bool,
    ) -> Result<(), Fault> {
        debug_assert!(self.can_encode_krylov());
        if let Some(soft) = self.soft {
            soft.encode_krylov_rows(encoder, input, output, row_parallel)?;
        }
        if self.count == 0 {
            return Ok(());
        }
        let (tangent_cache,tangent_epoch,epoch)=self.frozen_angular_tangent
            .map(|f|(f.factors,f.published,f.epoch)).unwrap_or((input,output,0));
        let isotropic = JointAngularApplyArgs {
            records: self.angular, state: self.state, reference: self.reference,
            bodies: self.bodies, base: self.base, dim: self.dim, dt: self.dt,
            action: 1, input, output, tangent_cache,tangent_epoch,epoch,
            count: self.count, seam_arena_count: 0,
        };
        encoder.elements(&isotropic, isotropic.count)?;
        if let Some(args) = self.extension_args(1, input, output, true) {
            encoder.elements(&args, args.count)?;
        }
        self.encode_frozen_rigid(encoder, input, output)
    }

    /// action 0: gradient, 1: Hessian-vector, 2: 6x6 self blocks.
    pub unsafe fn add<D:Device>(&self,device:&mut D,action:u32,input:Handle,output:Handle)->Result<(),Fault> {
        self.add_internal(device,action,input,output,false,false)
    }
    unsafe fn add_internal<D:Device>(&self,device:&mut D,action:u32,input:Handle,output:Handle,skip_ball:bool,use_tangent_cache:bool)->Result<(),Fault> {
        if let Some(soft)=self.soft {soft.add(device,action,input,output)?;}
        if self.count==0 { return Ok(()); }
        if let Some(c)=self.curves {
            let ball=super::kernels::JointRestoringBallApplyArgs{records:self.ball,record_count:self.count,count:self.count,state:self.state,bodies:self.bodies,base:self.base,dim:self.dim,dt:self.dt,action,input,output,fault:c.ball_fault,descriptors:c.descriptors,nodes:c.nodes,bindings:c.bindings,history:c.history,descriptor_count:c.descriptor_count,node_count:c.node_count,binding_count:c.binding_count,history_count:c.history_count,observations:c.observations,seam_arena_count:0};
            if !skip_ball {device.launch("restoring.ball",&ball,self.count)?;}
            // Isotropic V2 is incompatible only with active angular targets;
            // the validated packer keeps unrelated V2 rows unchanged.
            let (tangent_cache,tangent_epoch,epoch)=if use_tangent_cache&&action==1 {self.frozen_angular_tangent.map(|f|(f.factors,f.published,f.epoch)).unwrap_or((input,output,0))} else {(input,output,0)};
            let angular=JointAngularApplyArgs{records:self.angular,state:self.state,reference:self.reference,bodies:self.bodies,base:self.base,dim:self.dim,dt:self.dt,action,input,output,tangent_cache,tangent_epoch,epoch,count:self.count,seam_arena_count:0};
            if action<3 {device.launch("joints.angular",&angular,self.count)?;}
            if self.control_count!=0{let control=super::kernels::JointRestoringAngularApplyArgs{records:self.control,twist_context:self.twist_context,material_spin_context:self.material_spin_context,record_count:self.control_count,count:self.control_count,state:self.state,reference:self.reference,bodies:self.bodies,base:self.base,dim:self.dim,dt:self.dt,action,input,output,fault:c.angular_fault,descriptors:c.descriptors,nodes:c.nodes,bindings:c.bindings,history:c.history,descriptor_count:c.descriptor_count,node_count:c.node_count,binding_count:c.binding_count,history_count:c.history_count,observations:c.observations,seam_arena_count:0};device.launch("restoring.angular",&control,self.control_count)?;}
            self.add_extension(device,action,input,output,use_tangent_cache)?;return Ok(());
        }
        let ball=JointBallApplyArgs { records:self.ball,state:self.state,bodies:self.bodies,base:self.base,dim:self.dim,action,input,output,count:self.count,seam_arena_count:0 };
        if !skip_ball {device.launch("joints.ball",&ball,self.count)?;}
        let (tangent_cache,tangent_epoch,epoch)=if use_tangent_cache&&action==1 {self.frozen_angular_tangent.map(|f|(f.factors,f.published,f.epoch)).unwrap_or((input,output,0))} else {(input,output,0)};
        let angular=JointAngularApplyArgs { records:self.angular,state:self.state,reference:self.reference,bodies:self.bodies,base:self.base,dim:self.dim,dt:self.dt,action,input,output,tangent_cache,tangent_epoch,epoch,count:self.count,seam_arena_count:0 };
        device.launch("joints.angular",&angular,self.count)?;
        // The directional family is dispatched only when it owns records: a V2
        // scene must never dispatch an unallocated record family.
        if self.control_count!=0 {
            let tangent=if use_tangent_cache&&action==1{self.frozen_control_tangent}else{None};
            let control=JointAngularControlApplyArgs { records:self.control,twist_context:self.twist_context,material_spin_context:self.material_spin_context,state:self.state,reference:self.reference,bodies:self.bodies,base:self.base,dim:self.dim,dt:self.dt,
                action:if tangent.is_some(){7}else{action},input,output,fault:self.fault,
                tangent_cache:tangent.map(|f|f.factors).unwrap_or(self.control_tangent_factors),
                tangent_offsets:tangent.map(|f|f.offsets).unwrap_or(self.control_tangent_offsets),
                tangent_published:tangent.map(|f|f.published).unwrap_or(self.control_tangent_published),epoch:tangent.map(|f|f.epoch).unwrap_or(0),
                count:self.control_count,seam_arena_count:0 };
            device.launch("joints.angular_control",&control,self.control_count)?;
        }
        self.add_extension(device,action,input,output,use_tangent_cache)?;
        Ok(())
    }
}

/// Limits use physical duration and pose; drives also require distinct f32
/// endpoints for their device key sampler.
pub(super) fn valid_extension_clock(start:f64,end:f64,dt:f32,has_drives:bool)->bool {
    let sample=end as f32;
    start.is_finite() && start>=0.0 && end.is_finite() && end>start &&
        dt.is_finite() && dt>0.0 && sample.is_finite() && sample>0.0 &&
        (!has_drives || sample>start as f32)
}

/// Typed refusal for a scene whose damping term would divide by a step size that
/// is not finite and positive. Called before any dispatch, so an unusable dt is
/// never a partially assembled step.
fn validate_gate_inputs(view: View) -> Result<(),Fault> {
    if view.control_count==0 { return Ok(()); }
    if !view.dt.is_finite() || view.dt<=0.0 {
        return Err(Fault::Platform { call:"joint angular control geometry",
            detail:format!("the directional family divides its damping by the step size, which is {} here", view.dt) });
    }
    if view.extension_count!=0 {
        if !valid_extension_clock(view.start_time_s,view.end_time_s,view.dt,view.drive_count!=0) {
            return Err(Fault::Platform { call:"joint extension step domain",detail:format!("fault132: start={} dt={} end={} is not a representable physical substep",view.start_time_s,view.dt,view.sample_time_s) });
        }
        if view.extension_count!=view.control_count { return Err(Fault::Platform { call:"joint extension transport",detail:"limit/control row count mismatch".into() }); }
    }
    Ok(())
}

/// Run the typed geometry gate at one frozen pose. `view.state` must already
/// name the CURRENT fitted rigid pose and `view.reference` the substep-start one.
///
/// # Safety
/// Every handle in `view` must name a live allocation of the stated length.
pub unsafe fn gate_view<D:Device>(device:&mut D,joints:&mut Joints,view:View)->super::scene::FatalResult<()> {
    gate_view_at(device,joints,view,"unspecified_gate")
}
unsafe fn gate_view_at<D:Device>(device:&mut D,joints:&mut Joints,view:View,stage:&str)->super::scene::FatalResult<()> {
    if !joints.has_control() && !joints.curves.active() { return Ok(()); }
    validate_gate_inputs(view)?;
    joints.check_at(device,view,stage).map_err(super::scene::Fatal::from)
}

/// Fit the SUBSTEP-START rigid pose and gate it once.
///
/// This is the reference half of the pair: the pose captured here is the one
/// every action in the step measures its damping increment against, while the
/// current half is re-fitted and re-gated at each frozen Newton solve in
/// `step.rs`. Gating here too means a substep that starts outside the chart is
/// refused before any of its work happens.
pub unsafe fn begin_step<D:Device>(device:&mut D,data:&crate::data::DataSet,state:&mut super::state::SolverState,dt:f32)->super::scene::FatalResult<()> {
    if !state.joints.active() { return Ok(()); }
    if !dt.is_finite() || dt<=0.0 {
        return Err(super::scene::Fatal::invariant(format!(
            "solver driver: dt must be finite and positive for the joint step, but is {dt}")));
    }
    {
        let vert_list=state.rigid_staged.vert_list.handle();
        let body_prop=state.rigid_staged.body_prop.handle();
        let rest=state.rigid_staged.rest_centered.handle();
        let mut reduction=super::pdrd::Reduction { map:&state.rigid,staged:&mut state.rigid_staged };
        reduction.fit(device,vert_list,state.prop_vertex.handle(),body_prop,rest,state.positions.handle(),state.rigid_fit_scratch.handle(),state.joints.reference.handle(),data.pdrd_vert_list.size,&state.pcg_rigid.zero)?;
    }
    // This fit is both the current and reference pose at substep start.
    // `rigid_state` is not fitted until the Newton loop and may be uninitialized
    // on the first step or hold the previous iterate on later steps.
    state.joints.soft.begin_step(device,state.positions.handle(),&state.rigid)?;
    state.joints.soft.freeze(device,state.positions.handle(),state.joints.reference.handle(),state.joints.reference.handle(),&state.rigid,dt,None)?;
    let step_start=state.joints.reference.handle();
    let view=state.joints.view(step_start,&state.rigid,dt);
    gate_view_at(device,&mut state.joints,view,"substep_start")?;
    Ok(())
}

/// The per-frozen-pose half of the gate, called from the reduced-solve branch of
/// the Newton loop IMMEDIATELY AFTER `Reduction::fit` has rewritten
/// `rigid_state` for the current iterate and BEFORE the preconditioner that
/// consumes the joint blocks, the reduced right-hand side, or any PCG
/// application.
///
/// Gating only at `begin_step` would miss a pose that crosses the chart domain
/// later in the same substep, and on the final step -- where no next substep
/// follows -- would leave the crossing unobserved entirely, so a requested term
/// could be accepted while missing.
///
/// # Safety
/// `state` must name the CURRENT fitted rigid pose allocation for `rigid`.
pub unsafe fn gate_frozen_pose<D:Device>(device:&mut D,joints:&mut Joints,rigid:&super::rigid_map::RigidMap,state:Handle,dt:f32)->super::scene::FatalResult<()> {
    if !joints.has_control() && !joints.curves.active() { return Ok(()); }
    let view=joints.view(state,rigid,dt);
    gate_view_at(device,joints,view,"frozen_newton_pose")
}

/// Validate the accepted trial pose after contact/rigidification has changed
/// positions and before loop exit/commit. The substep-start reference is retained.
/// Supplied only after the current Newton attempt completed its B32 sweep.
pub(super) struct B32Complete;
pub unsafe fn gate_candidate<D:Device>(device:&mut D,data:&crate::data::DataSet,state:&mut super::state::SolverState,dt:f32,marker:B32Complete)->super::scene::FatalResult<()> {
    if !state.joints.has_control() { return Ok(()); }
    fit_candidate(device,data,state)?;
    check_candidate(device,state,dt,marker)
}
/// Fit is split from the check so a future certified branch can sample the
/// exact final fit and upload its trial context before any force gate uses it.
pub(crate) unsafe fn fit_candidate<D:Device>(device:&mut D,data:&crate::data::DataSet,state:&mut super::state::SolverState)->super::scene::FatalResult<()> {
    let vert_list=state.rigid_staged.vert_list.handle();
    let body_prop=state.rigid_staged.body_prop.handle();
    let rest=state.rigid_staged.rest_centered.handle();
    let mut reduction=super::pdrd::Reduction { map:&state.rigid,staged:&mut state.rigid_staged };
    reduction.fit(device,vert_list,state.prop_vertex.handle(),body_prop,rest,state.eval_x.handle(),state.rigid_fit_scratch.handle(),state.rigid_state.handle(),data.pdrd_vert_list.size,&state.pcg_rigid.zero)?;
    Ok(())
}
pub(crate) unsafe fn check_candidate<D:Device>(device:&mut D,state:&mut super::state::SolverState,dt:f32,_marker:B32Complete)->super::scene::FatalResult<()> {
    let candidate=Some((state.joints.candidate_trial,state.joints.start_time_s));
    state.joints.soft.freeze(device,state.eval_x.handle(),state.rigid_state.handle(),state.joints.reference.handle(),&state.rigid,dt,candidate)?;
    let view=state.joints.view(state.rigid_state.handle(),&state.rigid,dt);
    gate_view_at(device,&mut state.joints,view,"candidate_after_B32")
}

/// A read-only geometry snapshot of one actual stored vertex pose. Dedicated
/// fit buffers keep B32's running rigid state and scratch untouched.
pub(crate) struct TwistPoseSnapshot {
    pub rows:Vec<[f64;10]>,
    pub covariances:Vec<Option<([f64;9],f64)>>,
}
#[cfg(all(test, feature="cpu"))]
pub(crate) static PRIVATE_TWIST_COMMITS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
#[cfg(all(test, feature="cpu"))]
pub(crate) static PRIVATE_INVALID_REFERENCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[cfg(all(test, feature="cpu"))]
pub(crate) static PRIVATE_FORCE_UNCERTIFIED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[cfg(all(test, feature="cpu"))]
pub(crate) static PRIVATE_SWING_SAMPLE_ROTATIONS: std::sync::Mutex<Option<[(usize, [u32; 9]); 2]>> = std::sync::Mutex::new(None);

pub(crate) enum TwistPoseSource { Target, Iterate, Committed }
/// Fit committed positions without invoking the singular legacy swing chart.
pub(crate) unsafe fn capture_material_spin_pose<D:Device>(device:&mut D,data:&crate::data::DataSet,state:&mut super::state::SolverState,selected:&[bool])->super::scene::FatalResult<Vec<[f64;4]>>{
    let n=state.rigid.n_bodies;
    state.joints.twist_sample_fit.size(device,n,AllocLabel("joints.material_spin_fit"))?;
    state.joints.twist_sample_fit_scratch.size(device,12*n,AllocLabel("joints.material_spin_fit_scratch"))?;
    let fit=state.joints.twist_sample_fit.handle();
    let vert_list=state.rigid_staged.vert_list.handle();let body_prop=state.rigid_staged.body_prop.handle();let rest=state.rigid_staged.rest_centered.handle();
    let mut reduction=super::pdrd::Reduction{map:&state.rigid,staged:&mut state.rigid_staged};
    reduction.fit(device,vert_list,state.prop_vertex.handle(),body_prop,rest,state.positions.handle(),state.joints.twist_sample_fit_scratch.handle(),fit,data.pdrd_vert_list.size,&state.pcg_rigid.zero)?;
    let view=state.joints.view(fit,&state.rigid,1.0);
    Ok(state.joints.sample_material_spin_pose(device,view,selected)?)
}
pub(crate) unsafe fn capture_fem_material_spin_pose<D:Device>(device:&mut D,data:&crate::data::DataSet,state:&mut super::state::SolverState,selected:&[bool])->super::scene::FatalResult<Vec<crate::material_spin_history::MaterialSpinRow>>{
    let n=state.rigid.n_bodies;
    state.joints.twist_sample_fit.size(device,n,AllocLabel("joints.fem_material_spin_fit"))?;
    state.joints.twist_sample_fit_scratch.size(device,12*n,AllocLabel("joints.fem_material_spin_fit_scratch"))?;
    let fit=state.joints.twist_sample_fit.handle();
    let vert_list=state.rigid_staged.vert_list.handle();let body_prop=state.rigid_staged.body_prop.handle();let rest=state.rigid_staged.rest_centered.handle();
    let mut reduction=super::pdrd::Reduction{map:&state.rigid,staged:&mut state.rigid_staged};
    reduction.fit(device,vert_list,state.prop_vertex.handle(),body_prop,rest,state.positions.handle(),state.joints.twist_sample_fit_scratch.handle(),fit,data.pdrd_vert_list.size,&state.pcg_rigid.zero)?;
    Ok(state.joints.soft.sample_initial_material_spin(device,state.positions.handle(),fit,&state.rigid,selected)?)
}
pub(crate) unsafe fn capture_twist_pose<D:Device>(
    device:&mut D,data:&crate::data::DataSet,state:&mut super::state::SolverState,
    selected:&[bool],dt:f32,source:TwistPoseSource,existing_fit:Option<Handle>,
)->super::scene::FatalResult<TwistPoseSnapshot> {
    if state.joints.swing_candidate.is_some() || state.joints.candidate_token.is_some() {
        return Err(super::scene::Fatal::invariant("continuous twist sample forbidden while trial token is pending"));
    }
    state.joints.last_twist_sample_failure=None;
    let source_label=match source {TwistPoseSource::Target=>"Target",TwistPoseSource::Iterate=>"Iterate",TwistPoseSource::Committed=>"Committed"};
    let has_existing_fit=existing_fit.is_some();
    let n=state.rigid.n_bodies;
    let body_selected=state.joints.twist_body_selection(selected,n)?;
    let positions=match source {TwistPoseSource::Target=>&mut state.target,TwistPoseSource::Iterate=>&mut state.eval_x,TwistPoseSource::Committed=>&mut state.positions};
    positions.download(device)?;
    let covariances=crate::continuous_twist::body_covariances(
        positions.host(),state.rigid_staged.vert_list.host(),state.rigid_staged.rest_centered.host(),
        state.rigid_staged.body_prop.host(),&body_selected,
    ).ok_or_else(||super::scene::Fatal::invariant("continuous twist: cannot reconstruct fitted body covariance"))?;
    let fit=if let Some(fit)=existing_fit {fit} else {
        state.joints.twist_sample_fit.size(device,n,AllocLabel("joints.twist_sample_fit"))?;
        state.joints.twist_sample_fit_scratch.size(device,12*n,AllocLabel("joints.twist_sample_fit_scratch"))?;
        let vert_list=state.rigid_staged.vert_list.handle();
        let body_prop=state.rigid_staged.body_prop.handle();
        let rest=state.rigid_staged.rest_centered.handle();
        let position_handle=positions.handle();
        let fit=state.joints.twist_sample_fit.handle();
        let scratch=state.joints.twist_sample_fit_scratch.handle();
        let mut reduction=super::pdrd::Reduction {map:&state.rigid,staged:&mut state.rigid_staged};
        reduction.fit(device,vert_list,state.prop_vertex.handle(),body_prop,rest,position_handle,scratch,fit,data.pdrd_vert_list.size,&state.pcg_rigid.zero)?;
        fit
    };
    // Test-only: replay captured rotations in the sampler scratch, never in
    // the physical iterate or accepted reference. Action 4 still executes.
    #[cfg(all(test, feature="cpu"))]
    if source_label == "Iterate" && !has_existing_fit {
        if let Some(rows) = PRIVATE_SWING_SAMPLE_ROTATIONS.lock().unwrap().take() {
            for (body, bits) in rows {
                if body >= n { return Err(super::scene::Fatal::invariant("private swing body out of range")); }
                let mut rigid = [crate::data::PdrdRigidState::default()];
                state.joints.twist_sample_fit.read(device, body, &mut rigid)?;
                rigid[0].rotation = bits.map(f32::from_bits);
                state.joints.twist_sample_fit.write(device, body, &rigid)?;
            }
        }
    }
    let view=state.joints.view(fit,&state.rigid,dt);
    let rows=match state.joints.sample_twist_pose(device,view,selected) {
        Ok(rows)=>rows,
        Err(original)=>{
            if let Some(failure)=state.joints.last_twist_sample_failure() {
                observe_best_effort(||{let observe=(||->Result<(),Fault>{
                    let mut endpoints=Vec::new();
                    for (role,body) in [("parent",failure.body_a),("child",failure.body_b)] {
                        if body==u32::MAX {endpoints.push(serde_json::json!({"role":role,"body_index":body,"world_endpoint":true}));continue;}
                        if body as usize>=n {endpoints.push(serde_json::json!({"role":role,"body_index":body,"out_of_bounds":true}));continue;}
                        let mut running=[0.0_f32;9];
                        state.rigid_running_rotation.read(device,body as usize*9,&mut running)?;
                        let covariance=covariances.get(body as usize).and_then(|v|v.as_ref()).map(|(matrix,error)|serde_json::json!({"matrix":matrix,"matrix_f64_bits":matrix.map(f64::to_bits),"fit_error":error,"fit_error_f64_bits":error.to_bits()}));
                        endpoints.push(serde_json::json!({"role":role,"body_index":body,"world_endpoint":false,"running_rotation":running,"running_rotation_u32_bits":running.map(f32::to_bits),"covariance":covariance}));
                    }
                    eprintln!("TWIST_SAMPLE_CONTEXT_JSON {}",serde_json::json!({"source":source_label,"existing_fit":has_existing_fit,"joint_index":failure.joint_index,"fault_code":failure.code,"call_sequence":failure.call_sequence,"running_rotation_seeded":state.rigid_rotation_seeded,"endpoints":endpoints}));
                    use std::io::Write;let _=std::io::stderr().flush();
                    Ok(())
                })();
                if let Err(capture)=observe {eprintln!("TWIST_SAMPLE_CONTEXT_CAPTURE_ERROR {capture}");}});
            }
            return Err(original.into());
        }
    };
    Ok(TwistPoseSnapshot{rows,covariances})
}

#[cfg(test)]
mod tests {
    #[cfg(feature="cpu")]
    use crate::driver::kernels::{JointAngularApplyArgs,JointAngularControlApplyArgs};
    #[cfg(feature="cpu")]
    use ppf_cts_compute::Device;

    #[cfg(feature="cpu")]
    #[test]
    fn chart_pending_rejects_sample_without_changing_identity_or_bank() {
        let mut device=super::super::launch::host_device();
        let mut joints=super::Joints::default();
        joints.twist_sample_faults.size(&mut device,2,ppf_cts_compute::AllocLabel("pending.test.bank")).unwrap();
        joints.twist_sample_faults.seed(&mut device,&[0u32,6]).unwrap();
        joints.twist_sample_sequence=19;
        let failure=super::TwistSampleFailure{joint_index:1,code:6,body_a:0,body_b:1,call_sequence:19};
        joints.last_twist_sample_failure=Some(failure);
        let token=super::CandidateFaultToken{trial:3,start_bits:1f64.to_bits(),dt_bits:0.01f32.to_bits(),curves:vec![],control:vec![],reason:super::CandidateRetryReason::UncertifiedPath("B31")};
        joints.candidate_token=Some(token.clone());
        let view=joints.view(ppf_cts_compute::Handle::NONE,&super::super::rigid_map::RigidMap::default(),0.01);
        assert!(unsafe{joints.sample_twist_pose(&mut device,view,&[])}.is_err());
        joints.twist_sample_faults.download(&mut device).unwrap();
        assert_eq!(joints.twist_sample_faults.host(),&[0,6]);
        assert_eq!(joints.twist_sample_sequence,19);assert_eq!(joints.last_twist_sample_failure,Some(failure));assert_eq!(joints.candidate_token,Some(token));
    }
    #[cfg(feature="cpu")]
    #[test]
    fn chart_host_predicate_rejects_mixed_sampler_bank_and_stale_trial() {
        let mut device=super::super::launch::host_device();let mut joints=super::Joints::default();
        joints.twist_sample_faults.size(&mut device,2,ppf_cts_compute::AllocLabel("chart.predicate.bank")).unwrap();
        joints.twist_sample_faults.seed(&mut device,&[0u32,6]).unwrap();
        joints.start_time_s=1.;joints.candidate_trial=3;joints.candidate_domain_dt=Some(0.01);
        let token=super::CandidateFaultToken{trial:3,start_bits:1f64.to_bits(),dt_bits:0.01f32.to_bits(),curves:vec![],control:vec![],reason:super::CandidateRetryReason::UncertifiedPath("B31")};
        joints.candidate_token=Some(token.clone());
        assert_eq!(joints.host_chart_candidate(&mut device).unwrap(),None,"mixed sample bank must block path retry");
        joints.twist_sample_faults.seed(&mut device,&[0u32,0]).unwrap();
        assert_eq!(joints.host_chart_candidate(&mut device).unwrap(),Some(token));
        joints.candidate_trial=4;
        assert_eq!(joints.host_chart_candidate(&mut device).unwrap(),None,"stale generation must block path retry");
    }
    #[cfg(feature="cpu")]
    #[test]
    #[ignore = "requires explicit private 04 scene"]
    fn real_install_soft_limits_dummy_and_authored_drive_scope() {
        use clap::Parser;
        let path=std::env::var("PPF_PRIVATE_SWING_SCENE").expect("private 04 session");
        let output=format!("{path}/output");
        let args=crate::args::ProgramArgs::try_parse_from(["test","--path",&path,"--output",&output]).unwrap();
        let scene=crate::scene::Scene::new(&args);
        let original=scene.joint_input.as_ref().expect("04 joint input").records().clone();
        let selected:Vec<_>=original.joints.iter().map(|j|j.wire.enabled && j.wire.angular_control_v1.as_ref().is_some_and(|c|c.coordinate_model==crate::twist_overlay::CONTINUOUS_MODEL)).collect();
        assert!(selected.iter().any(|v|*v));
        let bodies=original.bodies.values().copied().max().unwrap() as usize+1;
        let ids:Vec<_>=original.joints.iter().map(|j|j.wire.joint_uuid.clone()).collect();
        let history=crate::continuous_twist::AcceptedTwist{scene_uuid:original.scene_uuid.clone(),joint_uuids:ids.clone(),time:0.,theta:vec![0.;ids.len()],prior_turns_known:true};
        let mut device=super::super::launch::host_device();let mut joints=super::Joints::default();
        let activate=|joints:&mut super::Joints|{joints.restore_accepted_twist(history.clone(),&original.scene_uuid,&ids,0.).unwrap();joints.activate_twist_selection(selected.clone()).unwrap();};
        assert!(original.joints.iter().any(|j|j.wire.angular_soft_limits_v1.is_some()));
        assert!(original.joints.iter().all(|j|j.wire.angular_drives_v1.is_none()));
        joints.install(&mut device,&original,bodies).unwrap();activate(&mut joints);
        assert_eq!(joints.drive_count,0);assert_eq!(joints.drives.len(),1);assert!(joints.swing_retry_scope_supported());
        // This copy is only a private authored-record fixture. The on-disk 04
        // project and the real scene's physical data remain untouched.
        let mut with_drive=original.clone();let index=selected.iter().position(|v|*v).unwrap();
        with_drive.joints[index].wire.angular_drives_v1=Some(crate::joint::StrictValue(serde_json::json!({"coordinate_model":"parent_local_swing_twist_x_continuous_v2","twist_x":{"position":{"enabled":true,"gain_Nm_per_rad":1.0,"max_effort_Nm":1.0,"interpolation":"linear_hold","target_keys":[{"time_s":0.0,"value_rad":0.1}]}}})));
        joints.install(&mut device,&with_drive,bodies).unwrap();activate(&mut joints);
        assert!(joints.drive_count>0);assert!(!joints.swing_retry_scope_supported(),"actual authored drive must reject");
        joints.install(&mut device,&original,bodies).unwrap();activate(&mut joints);
        assert_eq!(joints.drive_count,0);assert_eq!(joints.drives.len(),1);assert!(joints.swing_retry_scope_supported(),"reinstall must clear drive count");
    }
    #[test]
    fn twist_sample_fault_uses_downloaded_lane_and_retains_original_error() {
        let endpoints=vec![(2,3),(4,u32::MAX),(6,7)];
        assert_eq!(super::twist_sample_failure(&[0,1,0],&endpoints,9),Some(super::TwistSampleFailure{joint_index:1,code:1,body_a:4,body_b:u32::MAX,call_sequence:9}));
        assert!(super::twist_sample_failure(&[0,0,0],&endpoints,10).is_none());
        assert!(super::twist_sample_failure(&[0,0,0,1],&endpoints,11).is_none());
        let observed=super::twist_sample_fault_with_capture(1,1,||Err(ppf_cts_compute::Fault::Platform{call:"capture",detail:"injected read failure".into()}));
        let panic_observed=super::twist_sample_fault_with_capture(1,1,||panic!("injected observer panic"));
        match panic_observed {ppf_cts_compute::Fault::Platform{call,detail}=>{assert_eq!(call,"continuous twist sample");assert_eq!(detail,"joint 1: chart fault 1");},other=>panic!("unexpected fault: {other}")};
        match observed {
            ppf_cts_compute::Fault::Platform{call,detail}=>{assert_eq!(call,"continuous twist sample");assert_eq!(detail,"joint 1: chart fault 1");},
            other=>panic!("unexpected fault: {other}"),
        }
    }
    #[test]
    fn continuous_selection_refines_shared_bodies_once() {
        let mut joints=super::Joints::default();
        joints.twist_endpoints=vec![(0,1),(0,2)];
        assert_eq!(joints.twist_body_selection(&[true,false],3).unwrap(),vec![true,true,false]);
        assert_eq!(joints.twist_body_selection(&[false,true],3).unwrap(),vec![true,false,true]);
        assert!(joints.twist_body_selection(&[true],3).is_err());
    }
    /// The gate's placement is the requirement, so it is asserted against the
    /// real source rather than described: at each frozen Newton solve the
    /// CURRENT fit must already have run, and the preconditioner that consumes
    /// the joint blocks, the reduced right-hand side and the PCG solve must all
    /// still be ahead. A gate that drifted after `build_precond` would let a
    /// chart fault be discovered only after the step had been assembled with the
    /// requested term silently missing.
    #[test]
    fn frozen_pose_gate_is_between_the_current_fit_and_the_reduced_system() {
        let source = include_str!("step.rs");
        let fit = source.find("reduction.fit(").expect("the current frozen fit call");
        let gate = source.find("super::joints::gate_frozen_pose(").expect("the frozen-pose gate call");
        let precond = source.find("reduction.build_precond(").expect("the reduced preconditioner build");
        let solve = source.find("pcg::solve_rigid(").expect("the reduced PCG solve");
        assert!(fit < gate, "the gate must run after the current pose is fitted");
        assert!(gate < precond, "the gate must run before the joint blocks are assembled");
        assert!(gate < solve, "the gate must run before any PCG application");
        // Exactly one per-solve site: the substep-start call cannot be the only
        // one, because a pose may cross the chart domain mid-substep and the
        // final step has no next substep to notice it in.
        assert_eq!(source.matches("super::joints::gate_frozen_pose(").count(), 1);
    }

    /// A scene whose control family is empty must dispatch nothing at all, and
    /// the substep-start path must not fit or gate when no joint exists.
    #[test]
    fn empty_and_jointless_families_dispatch_nothing() {
        let source = include_str!("joints.rs");
        assert!(source.contains("if !state.joints.active() { return Ok(()); }"), "a jointless step fits and gates nothing");
        assert!(source.contains("if view.control_count==0 { return Ok(()); }"), "an empty control family is gated by its own count");
        assert!(source.contains("if self.control_count!=0 {"), "the directional family is dispatched only when it owns records");
        assert!(source.contains("if !joints.has_control() && !joints.curves.active() { return Ok(()); }"), "a scene without the family skips the gate");
        // The empty-family early return in the transport is what keeps a V2
        // scene from ever allocating or launching the new record family.
        assert!(include_str!("../joint_angular_control_gpu.rs")
            .contains("if !scene.joints.iter().any(|j| j.wire.angular_control_v1.is_some()) { return Ok(Vec::new()); }"));
    }
    #[test]
    fn extension_sampling_tracks_actual_retry_dt_and_start_reference() {
        let mut joints=super::Joints::default();
        let map=super::super::rigid_map::RigidMap::default();
        joints.set_step_start_time(1.25).unwrap();
        let long=joints.view(ppf_cts_compute::Handle::NONE,&map,0.125);
        let short=joints.view(ppf_cts_compute::Handle::NONE,&map,0.0625);
        assert_eq!(long.start_time_s,1.25);
        assert_eq!(long.sample_time_s,1.375);
        assert_eq!(short.sample_time_s,1.3125);
        assert_eq!(long.reference,short.reference,"dt retries must not replace the frozen start reference");
    }

    #[test]
    fn extension_event_endpoint_is_shared_and_rejects_unrelated_duration() {
        let mut joints=super::Joints::default();
        let map=super::super::rigid_map::RigidMap::default();
        joints.set_step_start_time(0.007499999832361937).unwrap();
        let dt=(0.01-joints.start_time_s) as f32;
        joints.set_step_end_time(0.01,dt).unwrap();
        assert_eq!(joints.view(ppf_cts_compute::Handle::NONE,&map,dt).sample_time_s,0.01f32);
        assert!(joints.set_step_end_time(0.02,dt).is_err());
        joints.set_step_start_time(0.01).unwrap();
        assert_eq!(joints.view(ppf_cts_compute::Handle::NONE,&map,0.0025).sample_time_s,(0.01f64+f64::from(0.0025f32)) as f32);
    }

    #[test]
    fn extension_step_domain_refuses_unrepresentable_end_clock() {
        let mut joints=super::Joints::default();
        let map=super::super::rigid_map::RigidMap::default();
        joints.set_step_start_time(1e8).unwrap();
        let mut view=joints.view(ppf_cts_compute::Handle::NONE,&map,0.001);
        view.control_count=1;view.extension_count=1;view.drive_count=1;
        assert!(super::validate_gate_inputs(view).is_err(),"drive f32 end equals start must be typed refusal");
        joints.set_step_start_time(0.0).unwrap();
        let mut valid=joints.view(ppf_cts_compute::Handle::NONE,&map,0.001);
        valid.control_count=1;valid.extension_count=1;
        assert!(super::validate_gate_inputs(valid).is_ok());
    }

    #[test]
    fn extension_clock_accepts_real_undriven_tail_but_not_drive_or_invalid_time() {
        let start=0.8333333176167428_f64;
        let end=20.0_f64/24.0;
        let dt=(end-start) as f32;
        assert_eq!(start as f32,end as f32);
        assert!(super::valid_extension_clock(start,end,dt,false));
        assert!(!super::valid_extension_clock(start,end,dt,true));
        assert!(!super::valid_extension_clock(start,start,dt,false));
        assert!(!super::valid_extension_clock(start,end,0.0,false));
        assert!(!super::valid_extension_clock(f64::NAN,end,dt,false));
    }

    #[test]
    fn candidate_gate_is_after_accepted_rigidification_and_before_commit() {
        let source=include_str!("step.rs");
        let accept=source.find("device.launch(\"step.rigidify_accept\"").unwrap();
        let gate=source.find("super::joints::gate_candidate(").unwrap();
        let check=source.find("super::joints::check_candidate(device,state,dt,super::joints::B32Complete)").unwrap();
        let loop_exit=source[gate..].find("if !result.success() {break;}").unwrap()+gate;
        let final_step=source[loop_exit..].find("if final_step {").unwrap()+loop_exit;
        assert!(accept<check && check<gate && gate<loop_exit && loop_exit<final_step);
        assert_eq!(source.matches("super::joints::gate_candidate(").count(),1);
        let set_clock=source.find("state.joints.set_step_start_time(prm.time)").unwrap();
        let begin=source.find("super::joints::begin_step(").unwrap();
        assert!(set_clock<begin);
    }

    #[cfg(feature="cpu")]
    #[test]
    fn extension_reference_is_independent_of_frozen_tangent() {
        use crate::{continuous_twist::AngularTwistEvalContext, data::PdrdRigidState,
            joint_angular_control_gpu::AngularControlGpu,
            joint_angular_extension_gpu::{AngularSoftLimitGpuV1,AngularDriveGpuV1,AngularDriveDeviceKeyV1},
            material_spin_history::MaterialSpinContext};
        use ppf_cts_compute::{AllocLabel,Buffer,Device,Pod};
        fn upload<T:Pod>(device:&mut impl Device, values:&[T])->Buffer<T> {
            let mut b=Buffer::default(); b.size(device,values.len(),AllocLabel("test.extension.reference")).unwrap();
            b.write(device,0,values).unwrap(); b
        }
        let mut device=super::super::launch::host_device();
        let row=AngularControlGpu { body_a:0,body_b:1,enabled:1,coordinate_model:1,
            frame_a_wxyz:[1.,0.,0.,0.],frame_b_wxyz:[1.,0.,0.,0.],rest_relative_wxyz:[1.,0.,0.,0.],
            stiffness_nm_per_rad:[0.;3],damping_nms_per_rad:[0.;3] };
        let records=upload(&mut device,&[row]);
        let context=upload(&mut device,&[AngularTwistEvalContext::default()]);
        let material=upload(&mut device,&[MaterialSpinContext::default()]);
        let limits=upload(&mut device,&[AngularSoftLimitGpuV1 {joint_index:0,enabled_axis_mask:1,
            lower_rad:[-0.2,-1.,-1.],upper_rad:[-0.1,1.,1.],stiffness_nm_per_rad:[20.,0.,0.],
            swing_cone_enabled:0,swing_cone_half_angle_rad:0.,swing_cone_stiffness_nm_per_rad:0.}]);
        let mut drives=Buffer::<AngularDriveGpuV1>::default(); drives.size(&mut device,1,AllocLabel("test.extension.drive")).unwrap();
        let mut keys=Buffer::<AngularDriveDeviceKeyV1>::default(); keys.size(&mut device,1,AllocLabel("test.extension.key")).unwrap();
        let drive_offsets=upload(&mut device,&[0u32,0u32]);
        let identity=[1.,0.,0.,0.,1.,0.,0.,0.,1.];
        let states=upload(&mut device,&[PdrdRigidState{rotation:identity,..Default::default()};2]);
        let input=upload(&mut device,&[0f32,0.,0.,1.,0.,0.,0.,0.,0.,-1.,0.,0.]);
        let mut output=upload(&mut device,&[0f32;12]);
        let mut fault=upload(&mut device,&[0u32]);
        let mut factors=upload(&mut device,&[0f32;22]);
        let offsets=upload(&mut device,&[u32::MAX;2]);
        let mut published=upload(&mut device,&[0u32]);
        let mut joints=super::Joints::default();
        let mut view=joints.view(states.handle(),&super::super::rigid_map::RigidMap::default(),0.01);
        view.control=records.handle(); view.twist_context=context.handle(); view.material_spin_context=material.handle();
        view.limits=limits.handle(); view.drives=drives.handle(); view.drive_offsets=drive_offsets.handle(); view.device_keys=keys.handle();
        view.reference=states.handle(); view.bodies=2; view.base=0; view.dim=12; view.sample_time_s=0.01;
        view.extension_count=1; view.fault=fault.handle();
        view.extension_tangent_factors=factors.handle(); view.extension_tangent_offsets=offsets.handle(); view.extension_tangent_published=published.handle();
        let input_h=input.handle(); let output_h=output.handle();
        unsafe {
            let mut build=view.extension_args(6,input_h,output_h,false).unwrap();
            build.epoch=7;
            device.launch("test.extension.build",&build,1).unwrap();
            let mut token=[0u32]; published.read(&mut device,0,&mut token).unwrap(); assert_eq!(token,[7]);
            view.frozen_extension_tangent=Some(super::FrozenExtensionTangentView{factors:factors.handle(),offsets:offsets.handle(),published:published.handle(),epoch:7});
            let reference=view.extension_args(1,input_h,output_h,false).unwrap();
            let cached=view.extension_args(1,input_h,output_h,true).unwrap();
            assert_eq!(reference.action,1); assert_eq!(cached.action,7);
            output.write(&mut device,0,&[0f32;12]).unwrap(); fault.write(&mut device,0,&[0u32]).unwrap();
            device.launch("test.extension.reference",&reference,1).unwrap();
            let mut expected=[0f32;12]; output.read(&mut device,0,&mut expected).unwrap();
            assert!(expected.iter().any(|v|v.abs()>1e-7));
            output.write(&mut device,0,&[0f32;12]).unwrap();
            device.launch("test.extension.cached",&cached,1).unwrap();
            let mut actual=[0f32;12]; output.read(&mut device,0,&mut actual).unwrap();
            for i in 0..12 { assert!((actual[i]-expected[i]).abs()<=2e-5,"cached HVP differs at {i}"); }
            // Keep a valid epoch but deliberately replace the factors with finite zeros.
            // This proves the reference path does not silently read the cached HVP.
            factors.write(&mut device,0,&[0f32;22]).unwrap();
            output.write(&mut device,0,&[0f32;12]).unwrap();
            device.launch("test.extension.reference",&reference,1).unwrap();
            let mut independent=[0f32;12]; output.read(&mut device,0,&mut independent).unwrap();
            assert_eq!(independent,expected);
            output.write(&mut device,0,&[0f32;12]).unwrap();
            device.launch("test.extension.cached",&cached,1).unwrap();
            let mut polluted=[0f32;12]; output.read(&mut device,0,&mut polluted).unwrap();
            assert!(polluted.iter().zip(expected.iter()).any(|(a,b)|(a-b).abs()>1e-7));
            published.write(&mut device,0,&[8u32]).unwrap();
            output.write(&mut device,0,&[0f32;12]).unwrap();
            device.launch("test.extension.cached",&cached,1).unwrap();
            let mut fallback=[0f32;12]; output.read(&mut device,0,&mut fallback).unwrap();
            assert_eq!(fallback,expected,"stale cache must use original HVP");
            let mut errors=[0u32]; fault.read(&mut device,0,&mut errors).unwrap(); assert_eq!(errors,[0]);
        }
    }

    #[cfg(feature="cpu")]
    #[test]
    fn angular_tangent_cache_matches_hvp_and_stale_epoch_falls_back() {
        use crate::{data::PdrdRigidState,joint_angular_gpu::AngularJointGpuV2};
        use ppf_cts_compute::{Buffer,AllocLabel};
        let mut device=super::super::launch::host_device();
        let identity=PdrdRigidState{rotation:[1.,0.,0.,0.,1.,0.,0.,0.,1.],..Default::default()};
        let theta=0.31f32;let (s,c)=theta.sin_cos();
        let current=[identity,PdrdRigidState{rotation:[c,s,0.,-s,c,0.,0.,0.,1.],..Default::default()}];
        let reference=[identity;2];
        let mut state={let mut b=Buffer::default();b.size(&mut device,2,AllocLabel("angular.tangent.test.state")).unwrap();b.write(&mut device,0,&current).unwrap();b};
        let reference={let mut b=Buffer::default();b.size(&mut device,2,AllocLabel("angular.tangent.test.reference")).unwrap();b.write(&mut device,0,&reference).unwrap();b};
        let row=AngularJointGpuV2{body_a:0,body_b:1,enabled:1,max_angle_rad:0.18,
            frame_a_wxyz:[1.,0.,0.,0.],frame_b_wxyz:[1.,0.,0.,0.],rest_relative_wxyz:[1.,0.,0.,0.],
            stiffness_nm_per_rad:17.,damping_nms_per_rad:0.7};
        let records={let mut b=Buffer::default();b.size(&mut device,1,AllocLabel("angular.tangent.test.row")).unwrap();b.write(&mut device,0,&[row]).unwrap();b};
        let dim=12u32;let mut input=Buffer::<f32>::default();input.size(&mut device,dim as usize,AllocLabel("angular.tangent.test.input")).unwrap();
        input.write(&mut device,0,&[0.,0.,0.,0.13,-0.08,0.04,0.,0.,0.,-0.21,0.16,0.09]).unwrap();
        let mut output=Buffer::<f32>::default();output.size(&mut device,dim as usize,AllocLabel("angular.tangent.test.output")).unwrap();
        let mut joints=super::Joints::default();
        joints.ball.size(&mut device,1,AllocLabel("angular.tangent.test.ball")).unwrap();
        joints.angular.size(&mut device,1,AllocLabel("angular.tangent.test.angular")).unwrap();
        joints.angular.write(&mut device,0,&[row]).unwrap();
        let state_h=state.handle();
        let ordinary=|input:ppf_cts_compute::Handle,output:ppf_cts_compute::Handle,tangent_cache:ppf_cts_compute::Handle,tangent_epoch:ppf_cts_compute::Handle,epoch| JointAngularApplyArgs{
            records:records.handle(),state:state_h,reference:reference.handle(),bodies:2,base:0,dim,dt:0.01,
            action:1,input,output,tangent_cache,tangent_epoch,epoch,count:1,seam_arena_count:0};
        unsafe {
            device.fill_zero(output.handle(),(dim as usize)*4).unwrap();
            let uncached=ordinary(input.handle(),output.handle(),input.handle(),output.handle(),0);
            device.launch("joints.angular",&uncached,1).unwrap();
            let mut expected=[0f32;12];output.read(&mut device,0,&mut expected).unwrap();

            let mut view=joints.view(state.handle(),&super::super::rigid_map::RigidMap::default(),0.01);
            view.count=1;view.bodies=2;view.base=0;view.dim=dim;view.reference=reference.handle();
            let view=joints.prepare_angular_tangents(&mut device,view).unwrap();
            let first=view.frozen_angular_tangent.unwrap();
            assert_eq!(first.epoch,1);
            device.fill_zero(output.handle(),(dim as usize)*4).unwrap();
            let hit=ordinary(input.handle(),output.handle(),first.factors,first.published,first.epoch);
            device.launch("joints.angular",&hit,1).unwrap();
            let mut actual=[0f32;12];output.read(&mut device,0,&mut actual).unwrap();
            for i in 0..12 {assert!((actual[i]-expected[i]).abs()<=2e-5,"cached HVP differs at {i}: {} vs {}",actual[i],expected[i]);}

            // A later Newton pose advances the token and replaces every row.
            let theta2=0.47f32;let (s2,c2)=theta2.sin_cos();
            state.write(&mut device,0,&[identity,PdrdRigidState{rotation:[c2,s2,0.,-s2,c2,0.,0.,0.,1.],..Default::default()}]).unwrap();
            device.fill_zero(output.handle(),(dim as usize)*4).unwrap();
            device.launch("joints.angular",&uncached,1).unwrap();
            let mut expected2=[0f32;12];output.read(&mut device,0,&mut expected2).unwrap();
            let mut view2=joints.view(state.handle(),&super::super::rigid_map::RigidMap::default(),0.01);
            view2.count=1;view2.bodies=2;view2.base=0;view2.dim=dim;view2.reference=reference.handle();
            let view2=joints.prepare_angular_tangents(&mut device,view2).unwrap();
            let second=view2.frozen_angular_tangent.unwrap();
            assert_eq!(second.epoch,2);
            device.fill_zero(output.handle(),(dim as usize)*4).unwrap();
            let current=ordinary(input.handle(),output.handle(),second.factors,second.published,second.epoch);
            device.launch("joints.angular",&current,1).unwrap();
            let mut actual2=[0f32;12];output.read(&mut device,0,&mut actual2).unwrap();
            for i in 0..12 {assert!((actual2[i]-expected2[i]).abs()<=2e-5,"second-pose cached HVP differs at {i}: {} vs {}",actual2[i],expected2[i]);}

            // The first Newton token cannot see the second pose's factors.
            device.fill_zero(output.handle(),(dim as usize)*4).unwrap();
            let stale=ordinary(input.handle(),output.handle(),first.factors,first.published,first.epoch);
            device.launch("joints.angular",&stale,1).unwrap();
            let mut fallback=[0f32;12];output.read(&mut device,0,&mut fallback).unwrap();
            assert_eq!(fallback,expected2,"epoch mismatch must recompute form_h instead of using stale factors");
        }
    }

    #[cfg(feature="cpu")]
    #[test]
    fn joint_control_tangent_cache_matches_hvp_and_stale_epoch_falls_back() {
        use crate::{continuous_twist::AngularTwistEvalContext,data::PdrdRigidState,joint_angular_control_gpu::AngularControlGpu};
        use ppf_cts_compute::{Buffer,AllocLabel};
        let mut device=super::super::launch::host_device();
        let identity=PdrdRigidState{rotation:[1.,0.,0.,0.,1.,0.,0.,0.,1.],..Default::default()};
        let theta=0.31f32;let (s,c)=theta.sin_cos();
        let current=[identity,PdrdRigidState{rotation:[c,s,0.,-s,c,0.,0.,0.,1.],..Default::default()}];
        let mut state={let mut b=Buffer::default();b.size(&mut device,2,AllocLabel("control.tangent.test.state")).unwrap();b.write(&mut device,0,&current).unwrap();b};
        let reference={let mut b=Buffer::default();b.size(&mut device,2,AllocLabel("control.tangent.test.reference")).unwrap();b.write(&mut device,0,&[identity;2]).unwrap();b};
        let row=AngularControlGpu{body_a:0,body_b:1,enabled:1,coordinate_model:1,frame_a_wxyz:[1.,0.,0.,0.],frame_b_wxyz:[1.,0.,0.,0.],
            rest_relative_wxyz:[1.,0.,0.,0.],stiffness_nm_per_rad:[17.,9.,5.],damping_nms_per_rad:[0.7,0.4,0.2]};
        let records={let mut b=Buffer::default();b.size(&mut device,1,AllocLabel("control.tangent.test.row")).unwrap();b.write(&mut device,0,&[row]).unwrap();b};
        let contexts={let mut b=Buffer::default();b.size(&mut device,1,AllocLabel("control.tangent.test.context")).unwrap();b.write(&mut device,0,&[AngularTwistEvalContext::default()]).unwrap();b};
        let mut input=Buffer::<f32>::default();input.size(&mut device,12,AllocLabel("control.tangent.test.input")).unwrap();
        input.write(&mut device,0,&[0.,0.,0.,0.13,-0.08,0.04,0.,0.,0.,-0.21,0.16,0.09]).unwrap();
        let mut output=Buffer::<f32>::default();output.size(&mut device,12,AllocLabel("control.tangent.test.output")).unwrap();
        let mut factors=Buffer::<f32>::default();factors.size(&mut device,21,AllocLabel("control.tangent.test.factors")).unwrap();
        let mut offsets=Buffer::<u32>::default();offsets.size(&mut device,2,AllocLabel("control.tangent.test.offsets")).unwrap();
        let mut published=Buffer::<u32>::default();published.size(&mut device,1,AllocLabel("control.tangent.test.epoch")).unwrap();
        let mut fault=Buffer::<u32>::default();fault.size(&mut device,1,AllocLabel("control.tangent.test.fault")).unwrap();
        let records_h=records.handle();let contexts_h=contexts.handle();let state_h=state.handle();let reference_h=reference.handle();
        let input_h=input.handle();let output_h=output.handle();let fault_h=fault.handle();let factors_h=factors.handle();
        let offsets_h=offsets.handle();let published_h=published.handle();
        let make_args=|action,cache,offsets_handle,published_handle,epoch|JointAngularControlApplyArgs{records:records_h,twist_context:contexts_h,
            material_spin_context:contexts_h,state:state_h,reference:reference_h,bodies:2,base:0,dim:12,dt:0.01,action,input:input_h,output:output_h,
            fault:fault_h,tangent_cache:cache,tangent_offsets:offsets_handle,tangent_published:published_handle,epoch,count:1,seam_arena_count:0};
        unsafe {
            output.write(&mut device,0,&[0f32;12]).unwrap();fault.write(&mut device,0,&[0u32]).unwrap();
            let uncached=make_args(1,input_h,offsets_h,published_h,0);
            device.launch("joints.angular_control",&uncached,1).unwrap();
            let mut expected=[0f32;12];output.read(&mut device,0,&mut expected).unwrap();
            let build=make_args(6,factors_h,offsets_h,published_h,3);
            device.launch("joints.angular_control",&build,1).unwrap();
            let mut epoch=[0u32];published.read(&mut device,0,&mut epoch).unwrap();assert_eq!(epoch,[3]);
            output.write(&mut device,0,&[0f32;12]).unwrap();
            let cached=make_args(7,factors_h,offsets_h,published_h,3);
            device.launch("joints.angular_control",&cached,1).unwrap();
            let mut actual=[0f32;12];output.read(&mut device,0,&mut actual).unwrap();
            for i in 0..12 {assert!((actual[i]-expected[i]).abs()<=2e-5,"control tangent differs at {i}");}
            // A newer pose invalidates the published epoch; action 7 must take
            // the original chart/Hessian path instead of reading old factors.
            let theta2=0.47f32;let (s2,c2)=theta2.sin_cos();
            state.write(&mut device,0,&[identity,PdrdRigidState{rotation:[c2,s2,0.,-s2,c2,0.,0.,0.,1.],..Default::default()}]).unwrap();
            output.write(&mut device,0,&[0f32;12]).unwrap();
            device.launch("joints.angular_control",&uncached,1).unwrap();
            let mut expected2=[0f32;12];output.read(&mut device,0,&mut expected2).unwrap();
            published.write(&mut device,0,&[4u32]).unwrap();output.write(&mut device,0,&[0f32;12]).unwrap();
            device.launch("joints.angular_control",&cached,1).unwrap();
            let mut fallback=[0f32;12];output.read(&mut device,0,&mut fallback).unwrap();assert_eq!(fallback,expected2);
        }
    }

}

















