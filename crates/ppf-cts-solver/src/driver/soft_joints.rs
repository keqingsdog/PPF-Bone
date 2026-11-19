//! Material-frame joints over the existing free nodal/PDRD partition.
//! No endpoint acquires mass or extra degrees of freedom. The frame cache is
//! overwritten only before a Newton solve or after its accepted trial; every
//! RHS, diagonal and Krylov application reads that same frozen cache.
use ppf_cts_compute::{AllocLabel, Buffer, Device, Encoder, EncoderExt, Fault, Handle, ReadbackBuffer};
use crate::soft_joint_gpu::{FemJointGpu,FemFrameGpu};
use super::kernels::{JointFemPrepareArgs, JointFemApplyArgs, JointFemExtensionApplyArgs};
use crate::joint_angular_extension_gpu as ext;
use super::rigid_map::{RigidMap, RIGID_UNSET};

unsafe impl ppf_cts_compute::Pod for FemJointGpu {}
unsafe impl ppf_cts_compute::Pod for FemFrameGpu {}

fn refusal(detail:impl std::fmt::Display)->Fault {
    Fault::Platform { call:"FEM material-frame joint",detail:detail.to_string() }
}

/// Strict sign certificate of a determinant represented by f32 input values.
/// The f64 arithmetic bound is deliberately conservative; an uncertain sign
/// is never eligible for retry. Matrices are column-major.
fn certified_det_sign(m:[f64;9])->Option<i8> {
    if !m.iter().all(|x|x.is_finite()) {return None;}
    let terms=[m[0]*m[4]*m[8],m[3]*m[7]*m[2],m[6]*m[1]*m[5],
               m[6]*m[4]*m[2],m[3]*m[1]*m[8],m[0]*m[7]*m[5]];
    if !terms.iter().all(|x|x.is_finite()) {return None;}
    let det=terms[0]+terms[1]+terms[2]-terms[3]-terms[4]-terms[5];
    let bound=terms.iter().map(|x|x.abs()).sum::<f64>()*(64.0*f64::EPSILON);
    if !det.is_finite() || !bound.is_finite() || det.abs()<=bound {None}
    else if det>0.0 {Some(1)}else{Some(-1)}
}
#[derive(Clone,Copy)]
struct FloatInterval{lo:f64,hi:f64}
fn eligible_candidate_side(statuses:[u32;4])->Option<usize>{
    match statuses {[259,0,0,0]=>Some(0),[0,259,0,0]=>Some(1),_=>None}
}
impl FloatInterval {
    fn exact(v:f32)->Option<Self>{let bits=v.to_bits()&0x7fff_ffff;if bits>=0x7f80_0000 || (bits!=0 && bits<0x0080_0000){None}else{Some(Self{lo:f64::from(v),hi:f64::from(v)})}}
    fn rounded(lo:f64,hi:f64)->Option<Self>{
        let size=lo.abs().max(hi.abs());
        if !lo.is_finite() || !hi.is_finite() || size>f64::from(f32::MAX) {return None;}
        // Enclose ordinary rounding and either retention or flushing of f32 subnormals.
        let e=size*f64::from(f32::EPSILON)+f64::from(f32::MIN_POSITIVE);
        let (lo,hi)=(lo-e,hi+e);if lo.is_finite()&&hi.is_finite() && lo>=-f64::from(f32::MAX) && hi<=f64::from(f32::MAX){Some(Self{lo,hi})}else{None}
    }
    fn add(self,b:Self)->Option<Self>{Self::rounded(self.lo+b.lo,self.hi+b.hi)}
    fn sub(self,b:Self)->Option<Self>{Self::rounded(self.lo-b.hi,self.hi-b.lo)}
    fn mul(self,b:Self)->Option<Self>{
        let a=[self.lo*b.lo,self.lo*b.hi,self.hi*b.lo,self.hi*b.hi];
        if a.iter().any(|v|!v.is_finite()){return None;}
        Self::rounded(a.iter().copied().fold(f64::INFINITY,f64::min),a.iter().copied().fold(f64::NEG_INFINITY,f64::max))
    }
}
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
enum F32Orientation { Negative, Positive, FiniteIntervalCrossesZero, Invalid }
fn f32_orientation_interval(x:[[f32;3];4],dm_inv:[f32;9])->Option<(i8,FloatInterval)> {
    if x.iter().flatten().any(|v|!v.is_finite()) || dm_inv.iter().any(|v|!v.is_finite()) {return None;}
    let ds=std::array::from_fn(|i|f64::from(x[1+i/3][i%3])-f64::from(x[0][i%3]));
    let a=certified_det_sign(ds)?;let b=certified_det_sign(dm_inv.map(f64::from))?;let exact_sign=a*b;
    // Independently enclose the shared kernel's f32 subtraction, matrix
    // multiplication, and determinant arithmetic. A zero-crossing interval
    // does not constitute a strict f32 negative certificate, even when the
    // exact authored geometry has a separately certified sign.
    let mut diff=[FloatInterval{lo:0.,hi:0.};9];
    for c in 0..3 {for r in 0..3 {diff[3*c+r]=FloatInterval::exact(x[c+1][r])?.sub(FloatInterval::exact(x[0][r])?)?;}}
    let mut f=[FloatInterval{lo:0.,hi:0.};9];
    for c in 0..3 {for r in 0..3 {for k in 0..3 {f[3*c+r]=f[3*c+r].add(diff[3*k+r].mul(FloatInterval::exact(dm_inv[3*c+k])?)?)?;}}}
    let d=f[0].mul(f[4].mul(f[8])?.sub(f[7].mul(f[5])?)?)?
        .sub(f[3].mul(f[1].mul(f[8])?.sub(f[7].mul(f[2])?)?)?)?
        .add(f[6].mul(f[1].mul(f[5])?.sub(f[4].mul(f[2])?)?)?)?;
    Some((exact_sign,d))
}
fn f32_orientation(x:[[f32;3];4],dm_inv:[f32;9])->F32Orientation {
    match f32_orientation_interval(x,dm_inv) {
        Some((1,d)) if d.lo>0. => F32Orientation::Positive,
        Some((-1,d)) if d.hi<0. => F32Orientation::Negative,
        Some((_,d)) if d.lo<=0. && d.hi>=0. => F32Orientation::FiniteIntervalCrossesZero,
        _ => F32Orientation::Invalid,
    }
}
fn certified_tet_orientation(x:[[f32;3];4],dm_inv:[f32;9])->Option<i8> {
    match f32_orientation(x,dm_inv) {F32Orientation::Positive=>Some(1),F32Orientation::Negative=>Some(-1),_=>None}
}

// Host-only proof over the exact authored f32 coordinates, independent of the
// kernel's separately enclosed f32 matrix arithmetic. Any untrusted f64
// intermediate (including subnormals or flushed nonzero results) fails closed.
#[derive(Clone,Copy,Debug)]
struct GeometricInterval { lo:f64, hi:f64 }
fn geo_valid(x:f64)->bool {let b=x.to_bits()&0x7fff_ffff_ffff_ffff;b==0 || (b>=0x0010_0000_0000_0000 && b<0x7ff0_0000_0000_0000)}
fn geo_zero(x:f64)->bool {x.to_bits()&0x7fff_ffff_ffff_ffff==0}
fn geo_next(x:f64,up:bool)->Option<f64> {
    if !geo_valid(x)||geo_zero(x){return None}
    let b=x.to_bits();let n=if (b>>63==0)==up {b.checked_add(1)?}else{b.checked_sub(1)?};
    let out=f64::from_bits(n);if geo_valid(out){Some(out)}else{None}
}
#[derive(Clone,Copy)] enum GeoOp { Add, Sub, Mul }
fn geo_atom(op:GeoOp,a:f64,b:f64)->Option<GeometricInterval> {
    if !geo_valid(a)||!geo_valid(b){return None}
    let v=match op {GeoOp::Add=>a+b,GeoOp::Sub=>a-b,GeoOp::Mul=>a*b};
    if !geo_valid(v){return None}
    let exact_zero=match op {GeoOp::Add=>a == -b,GeoOp::Sub=>a==b,GeoOp::Mul=>geo_zero(a)||geo_zero(b)};
    if geo_zero(v){if !exact_zero{return None}return Some(GeometricInterval{lo:0.,hi:0.})}
    Some(GeometricInterval{lo:geo_next(v,false)?,hi:geo_next(v,true)?})
}
impl GeometricInterval {
    fn exact(v:f32)->Option<Self>{let b=v.to_bits()&0x7fff_ffff;if b>=0x7f80_0000||(b!=0&&b<0x0080_0000){return None}let x=f64::from(v);Some(Self{lo:x,hi:x})}
    fn add(self,b:Self)->Option<Self>{Some(Self{lo:geo_atom(GeoOp::Add,self.lo,b.lo)?.lo,hi:geo_atom(GeoOp::Add,self.hi,b.hi)?.hi})}
    fn sub(self,b:Self)->Option<Self>{Some(Self{lo:geo_atom(GeoOp::Sub,self.lo,b.hi)?.lo,hi:geo_atom(GeoOp::Sub,self.hi,b.lo)?.hi})}
    fn mul(self,b:Self)->Option<Self>{
        let q=[geo_atom(GeoOp::Mul,self.lo,b.lo)?,geo_atom(GeoOp::Mul,self.lo,b.hi)?,geo_atom(GeoOp::Mul,self.hi,b.lo)?,geo_atom(GeoOp::Mul,self.hi,b.hi)?];
        Some(Self{lo:q.iter().map(|x|x.lo).fold(f64::INFINITY,f64::min),hi:q.iter().map(|x|x.hi).fold(f64::NEG_INFINITY,f64::max)})
    }
}
fn geo_det(f:[GeometricInterval;9])->Option<GeometricInterval>{
    f[0].mul(f[4].mul(f[8])?.sub(f[7].mul(f[5])?)?)?
        .sub(f[3].mul(f[1].mul(f[8])?.sub(f[7].mul(f[2])?)?)?)?
        .add(f[6].mul(f[1].mul(f[5])?.sub(f[4].mul(f[2])?)?)?)
}
fn geometric_orientation(x:[[f32;3];4],dm:[f32;9])->Option<GeometricInterval>{
    let zero=GeometricInterval{lo:0.,hi:0.};let mut d=[zero;9];let mut m=[zero;9];
    for c in 0..3 {for r in 0..3 {d[3*c+r]=GeometricInterval::exact(x[c+1][r])?.sub(GeometricInterval::exact(x[0][r])?)?;m[3*c+r]=GeometricInterval::exact(dm[3*c+r])?;}}
    geo_det(d)?.mul(geo_det(m)?)
}

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub(crate) enum MaterialFrameRejectReason { StrictF32Negative, CertifiedGeometricInversionF32SignUncertain }
fn candidate_orientation_reason(now:[[f32;3];4],reference:[[f32;3];4],dm:[f32;9])->Option<MaterialFrameRejectReason> {
    if certified_tet_orientation(reference,dm)!=Some(1) {return None}
    match f32_orientation(now,dm) {
        F32Orientation::Negative=>Some(MaterialFrameRejectReason::StrictF32Negative),
        F32Orientation::FiniteIntervalCrossesZero=>{
            let current_geo=geometric_orientation(now,dm)?;
            let reference_geo=geometric_orientation(reference,dm)?;
            if current_geo.hi>=0. || reference_geo.lo<=0. {return None}
            Some(MaterialFrameRejectReason::CertifiedGeometricInversionF32SignUncertain)
        },
        _=>None,
    }
}
#[derive(Clone,Debug,PartialEq,Eq)]
pub(crate) struct MaterialFrameCandidateRejection {
    pub generation:u64,pub start_bits:u64,pub dt_bits:u32,pub row:usize,
    pub side:usize,pub vertices:[u32;4],pub fault:u32,pub statuses:[u32;4],pub reason:MaterialFrameRejectReason,
}

/// The kernel uses free-vertex indices; upstream cloth_off stores FLOAT offsets.
/// A prescribed vertex retains upstream's free slot and its usual Dirichlet
/// treatment. Only actual PDRD vertices are excluded from this nodal map.
pub(super) fn free_indices(map:&RigidMap)->Result<Vec<u32>,Fault> {
    if map.vbody.len()!=map.nrow || map.cloth_off.len()!=map.nrow ||
       map.body_base!=3*map.n_cloth || map.dim!=map.body_base+6*map.n_bodies {
        return Err(refusal("inconsistent original reduced partition"));
    }
    let mut seen=vec![false;map.n_cloth];
    let mut result=vec![RIGID_UNSET;map.nrow];
    for v in 0..map.nrow {
        if map.vbody[v]!=RIGID_UNSET {continue;}
        let offset=map.cloth_off[v] as usize;
        if offset%3!=0 || offset+3>map.body_base || seen[offset/3] {
            return Err(refusal("invalid or duplicate free nodal offset"));
        }
        seen[offset/3]=true;result[v]=(offset/3) as u32;
    }
    if seen.iter().any(|s|!*s) {return Err(refusal("unmapped original free nodal slot"));}
    Ok(result)
}


/// Scheduling-only domain. Every full-space float has the exact original slot;
/// no rigid/body or aggregate-lock/root projection may be bypassed.
pub(super) fn full_space_identity(map:&RigidMap)->bool {
    map.nrow>0 && map.n_bodies==0 && map.n_cloth==map.nrow &&
    map.body_base==3*map.nrow && map.dim==3*map.nrow &&
    !map.any_joint && !map.any_translation_lock && !map.any_rotation_lock &&
    map.vbody.len()==map.nrow && map.cloth_off.len()==map.nrow &&
    map.vbody.iter().all(|v|*v==RIGID_UNSET) &&
    map.cloth_off.iter().enumerate().all(|(i,o)|*o as usize==3*i) &&
    map.jmode.is_empty() && map.jaxis.is_empty() &&
    map.tlock.is_empty() && map.tlock_axis.is_empty() && map.tlock_mode.is_empty() &&
    map.rlock.is_empty() && map.rlock_axis.is_empty() && map.rlock_mode.is_empty()
}

const CACHE_TOKEN_MAX:u32=0x1fff_ffff;
fn cache_frame_capacity(rows:usize,eligible:bool)->Result<usize,Fault> {
    let prefix=rows.checked_mul(4).ok_or_else(||refusal("frame capacity overflow"))?;
    if !eligible{return Ok(prefix);}
    if std::mem::size_of::<FemFrameGpu>()!=208 || std::mem::align_of::<FemFrameGpu>()<4 {
        return Err(refusal("frozen frame size/alignment differs from cache layout"));
    }
    let bytes=rows.checked_mul(1154).and_then(|n|n.checked_mul(4))
        .ok_or_else(||refusal("Hessian cache capacity overflow"))?;
    let tail=bytes.checked_add(207).ok_or_else(||refusal("Hessian ceil overflow"))?/208;
    prefix.checked_add(tail).ok_or_else(||refusal("Hessian frame capacity overflow"))
}

/// Accept only validated, cross-only PDRD endpoint rows, with no joint action.
pub(super) fn pure_rigid_cross_rows(owned:&crate::soft_joint::OwnedSoftScene)->bool {
    !owned.records.is_empty() && owned.wires.is_empty() && owned.bodies.is_empty()
        && owned.controlled_vertices.is_empty() && owned.control_events.is_none() && owned.extensions.is_none()
        && owned.records.iter().all(|r|r.enabled==1 && r.a.kind==crate::soft_joint_gpu::FEM_PDRD && r.b.kind==crate::soft_joint_gpu::FEM_PDRD
            && r.ball_stiffness==0. && r.v2.enabled==0 && r.v2.max_angle_rad==0.
            && r.v2.stiffness_nm_per_rad==0. && r.v2.damping_nms_per_rad==0. && r.control.enabled==0
            && r.control.stiffness_nm_per_rad==[0.;3] && r.control.damping_nms_per_rad==[0.;3])
}
fn rigid_cross_bindings_cover(count:usize,bindings:&[crate::restoring_curve_gpu::CurveBinding])->bool {
    let mut seen=vec![false;count];
    let mut histories=std::collections::HashSet::new();
    for b in bindings {
        if b.kind==1 || b.kind==3 {return false;}
        if b.kind==crate::cross_links::DISTANCE_KIND {
            let i=b.target_index as usize;
            if i>=count || seen[i] || b.channel!=3 || b.history_index==u32::MAX || !histories.insert(b.history_index) {return false;}
            seen[i]=true;
        }
    }
    count>0 && seen.into_iter().all(|v|v)
}
#[derive(Default)]
pub struct SoftJoints {
    candidate_rejection:Option<MaterialFrameCandidateRejection>,
    prescribed:bool,
    rigid_cross_source:bool,
    #[cfg(test)] private_last_freeze_dt:Option<f32>,
    records:Buffer<FemJointGpu>,
    // Four frames per row, 208 bytes per frame. Scalar storage is ABI-neutral;
    // the matching C++ POD is guarded by the kernel owner's sizeof assertions.
    cache:Buffer<FemFrameGpu>,
    // Private Stage8 mapping: 24 indices, valid, duplicate, epoch per record.
    index_cache:Buffer<u32>,index_ready:bool,
    reference_xyz:Buffer<f32>,
    vertex_to_free:Buffer<u32>,
    fault:ReadbackBuffer<u32>,
    dummy:Buffer<f32>,
    material_spin_selected:Option<Vec<bool>>,
    material_spin_eligible:Vec<bool>,
    material_spin_rigid_endpoints:Vec<[Option<u32>;2]>,
    accepted_material_spin:Option<crate::material_spin_history::AcceptedMaterialSpin>,
    trial_material_spin:Option<crate::material_spin_history::TrialMaterialSpin>,
    material_spin_context:Buffer<crate::material_spin_history::MaterialSpinContext>,
    material_spin_sample:ReadbackBuffer<f32>,
    material_spin_fault:ReadbackBuffer<u32>,
    material_spin_base_fault:bool,
    limits:Buffer<ext::AngularSoftLimitGpuV1>,drives:Buffer<ext::AngularDriveGpuV1>,
    keys:Buffer<ext::AngularDriveDeviceKeyV1>,
    start_time_s:f64,end_time_s:Option<f64>,
    reference_ready:bool,
    frozen_ready:bool,
    hessian_ready:bool,hessian_epoch:u32,hessian_dt:f32,
    restoring_h:Buffer<f32>,restoring_idx:Buffer<u32>,restoring_epoch:Buffer<u32>,restoring_fault:ReadbackBuffer<u32>,restoring_generation:u32,restoring_ext_cache:Buffer<FemFrameGpu>,restoring_ext_fault:ReadbackBuffer<u32>,restoring_ext_active:bool,
}

#[derive(Clone,Copy)]
pub struct FrozenRestoringView { h:Handle,idx:Handle,epoch_buffer:Handle,epoch:u32,fault:Handle,ext_cache:Handle,ext_fault:Handle,ext_active:bool,legacy:bool }
#[derive(Clone,Copy)]
pub struct View {
    pub frozen_restoring:Option<FrozenRestoringView>,
    pub curves:Option<super::restoring::View>,
    pub records:Handle,pub material_spin_context:Handle,pub cache:Handle,pub vertex_to_free:Handle,pub fault:Handle,
    pub index_cache:Handle,pub index_ready:bool,
    pub dummy:Handle,pub count:u32,pub vertices:u32,pub free_count:u32,
    pub bodies:u32,pub base:u32,pub dim:u32,pub dt:f32,pub ready:bool,
    pub hessian_ready:bool,pub hessian_epoch:u32,pub hessian_dt:f32,
    pub limits:Handle,pub drives:Handle,pub keys:Handle,pub extension_count:u32,pub drive_count:u32,pub sample_time_s:f32,
}
impl View {
    pub(super) fn reduced_capture_frozen(&self)->Option<(bool,Handle,u32,Handle,Handle)>{
        self.frozen_restoring.map(|f|(f.legacy,f.epoch_buffer,f.epoch,f.h,f.idx))
    }
    pub(super) fn shadow_fault_handles(&self)->Vec<Handle>{
        let mut out=vec![self.fault];
        if let Some(v)=self.frozen_restoring {out.extend([v.fault,v.ext_fault]);}
        if let Some(v)=self.curves {out.extend([v.fault,v.ball_fault,v.angular_fault,v.fem_fault,v.stitch_fault]);}
        out
    }
}

impl SoftJoints {
    pub fn active(&self)->bool {self.records.len()!=0}
    pub(super) fn set_rigid_cross_source(&mut self,qualified:bool) {self.rigid_cross_source=qualified;}
    #[cfg(test)] pub(super) fn private_retry_cache(&self)->(bool,bool,Option<f32>) {(self.reference_ready,self.frozen_ready,self.private_last_freeze_dt)}
    pub(super) fn rigid_cross_retry_supported(&self,bindings:&[crate::restoring_curve_gpu::CurveBinding])->bool {
        self.rigid_cross_source && self.active() && !self.prescribed && !self.has_extensions() && self.drives.len()==0 && self.keys.len()==0
            && self.accepted_material_spin.is_none() && self.trial_material_spin.is_none()
            && rigid_cross_bindings_cover(self.records.len(),bindings)
    }
    pub(crate) fn material_spin_body_selection(&self,selected:&[bool],bodies:usize)->Result<Vec<bool>,Fault>{
        if selected.len()!=self.material_spin_rigid_endpoints.len() || selected.len()!=self.material_spin_eligible.len() || selected.iter().zip(&self.material_spin_eligible).any(|(on,eligible)|*on&&!*eligible) || !selected.iter().any(|on|*on) {return Err(refusal("invalid FEM material fit selection"));}
        let mut mask=vec![false;bodies];
        for (on,endpoints) in selected.iter().zip(&self.material_spin_rigid_endpoints) {if *on {for id in endpoints.iter().flatten(){let slot=mask.get_mut(*id as usize).ok_or_else(||refusal("FEM material fit body out of range"))?;*slot=true;}}}
        Ok(mask)
    }
    pub(crate) fn material_spin_active(&self)->bool {self.accepted_material_spin.is_some()}
    pub(crate) fn accepted_material_spin(&self)->Option<&crate::material_spin_history::AcceptedMaterialSpin>{self.accepted_material_spin.as_ref()}
    pub(crate) fn activate_material_spin(&mut self,history:crate::material_spin_history::AcceptedMaterialSpin,scene:&str,joints:&[String],time:f64,selected:Vec<bool>)->Result<(),Fault>{
        history.validate(scene,joints,time).map_err(|e|refusal(format!("FEM material spin activation: {e}")))?;
        if self.trial_material_spin.is_some()||self.accepted_material_spin.is_some()||selected.len()!=self.records.len()||selected.len()!=self.material_spin_eligible.len()||selected.iter().zip(&self.material_spin_eligible).any(|(on,eligible)|*on&&!*eligible)||selected.iter().filter(|v|**v).count()!=history.rows.len(){return Err(refusal("FEM material spin activation selection/state mismatch"));}
        self.accepted_material_spin=Some(history);self.material_spin_selected=Some(selected);Ok(())
    }
    pub(crate) fn begin_material_spin_trial(&mut self)->Result<(),Fault>{
        self.material_spin_base_fault=false;
        if self.trial_material_spin.is_some(){return Err(refusal("FEM material spin trial already active"));}
        let accepted=self.accepted_material_spin.as_ref().ok_or_else(||refusal("FEM material spin accepted state missing"))?;
        self.trial_material_spin=Some(accepted.trial());Ok(())
    }
    pub(crate) fn abort_material_spin_trial(&mut self){self.trial_material_spin=None;}
    pub(crate) fn material_spin_base_fault(&self)->bool{self.material_spin_base_fault}
    pub(crate) fn consume_material_spin_base_fault<D:Device>(&mut self,device:&mut D)->Result<(),Fault>{
        if !self.material_spin_base_fault{return Err(refusal("missing FEM material spin fault"));}
        self.fault.download(device)?;
        if !self.fault.host().iter().any(|c|*c==8)||self.fault.host().iter().any(|c|*c!=0&&*c!=8){return Err(refusal("FEM material spin fault changed"));}
        self.fault.seed(device,&vec![0;self.records.len()])?;self.material_spin_base_fault=false;Ok(())
    }
    pub(crate) fn prepare_material_spin_commit(&self,time:f64)->Result<crate::material_spin_history::AcceptedMaterialSpin,Fault>{
        let accepted=self.accepted_material_spin.as_ref().ok_or_else(||refusal("FEM material spin accepted state missing"))?;
        let trial=self.trial_material_spin.clone().ok_or_else(||refusal("FEM material spin trial missing"))?;
        accepted.prepare_commit(trial,time).map_err(|e|refusal(format!("FEM material spin commit: {e}")))
    }
    pub(crate) fn finish_material_spin_commit(&mut self,accepted:crate::material_spin_history::AcceptedMaterialSpin){self.accepted_material_spin=Some(accepted);self.trial_material_spin=None;}
    pub(crate) fn upload_material_spin_context<D:Device>(&mut self,device:&mut D)->Result<(),Fault>{
        let selected=self.material_spin_selected.as_ref().ok_or_else(||refusal("FEM material spin selection missing"))?;
        let accepted=self.accepted_material_spin.as_ref().ok_or_else(||refusal("FEM material spin accepted state missing"))?;
        if selected.len()!=self.records.len()||selected.iter().filter(|v|**v).count()!=accepted.rows.len()||self.material_spin_context.len()<selected.len(){return Err(refusal("FEM material spin context size mismatch"));}
        let mut rows=accepted.rows.iter();let mut contexts=Vec::with_capacity(selected.len());
        for enabled in selected {if *enabled {let row=rows.next().ok_or_else(||refusal("FEM material spin row missing"))?;let q=row.q_wxyz.map(|x|x as f32);let a=row.a.map(|x|x as f32);if q.iter().chain(a.iter()).any(|v|!v.is_finite()){return Err(refusal("FEM material spin context conversion invalid"));}contexts.push(crate::material_spin_history::MaterialSpinContext{accepted_q_wxyz:q,accepted_a:a,model:3});}else{contexts.push(crate::material_spin_history::MaterialSpinContext::default());}}
        if rows.next().is_some(){return Err(refusal("FEM material spin excess rows"));}if !contexts.is_empty(){self.material_spin_context.write(device,0,&contexts)?;}Ok(())
    }
    unsafe fn material_spin_sample<D:Device>(&mut self,device:&mut D,map:&RigidMap,dt:f32,action:u32,stride:usize)->Result<Option<Vec<f32>>,Fault>{
        if !self.active() || !dt.is_finite() || dt<=0. || !matches!((action,stride),(8,7)|(9,4)){return Err(refusal("invalid FEM material spin sample request"));}
        let count=self.records.len();let values=count.checked_mul(stride).ok_or_else(||refusal("FEM material spin sample size overflow"))?;
        self.material_spin_sample.size(device,values,AllocLabel("soft_joints.material_spin_sample"))?;
        self.material_spin_fault.size(device,count,AllocLabel("soft_joints.material_spin_fault"))?;
        self.material_spin_sample.seed(device,&vec![0.;values])?;self.material_spin_fault.seed(device,&vec![0;count])?;
        let args=JointFemApplyArgs{records:self.records.handle(),material_spin_context:self.material_spin_context.handle(),record_count:count as u32,count:count as u32,cache:self.cache.handle(),vertex_to_free:self.vertex_to_free.handle(),vertex_count:map.nrow as u32,free_count:map.n_cloth as u32,rigid_base:map.body_base as u32,dim:map.dim as u32,dt,action,input:self.dummy.handle(),output:self.material_spin_sample.handle(),nodal_blocks:self.dummy.handle(),rigid_blocks:self.dummy.handle(),fault:self.material_spin_fault.handle(),seam_arena_count:0};
        device.launch("joints.fem.material_spin.sample",&args,count as u32)?;
        self.material_spin_fault.download(device)?;self.material_spin_sample.download(device)?;
        if self.material_spin_fault.host().iter().any(|&v|v!=0){return Ok(None)}
        Ok(Some(self.material_spin_sample.host().to_vec()))
    }
    /// Sample the actual initial relative quaternion without evaluating the old
    /// angular chart or publishing a history. Only for fresh initialization.
    pub(crate) unsafe fn sample_initial_material_spin<D:Device>(&mut self,device:&mut D,positions:Handle,rigid_current:Handle,map:&RigidMap,selected:&[bool])->Result<Vec<crate::material_spin_history::MaterialSpinRow>,Fault>{
        if selected.len()!=self.records.len()||selected.len()!=self.material_spin_eligible.len()||selected.iter().zip(&self.material_spin_eligible).any(|(on,eligible)|*on&&!*eligible)||!selected.iter().any(|v|*v)||self.accepted_material_spin.is_some(){return Err(refusal("invalid FEM initial material spin selection"));}
        let contexts:Vec<_>=selected.iter().map(|on|if *on{crate::material_spin_history::MaterialSpinContext{accepted_q_wxyz:[1.,0.,0.,0.],accepted_a:[0.;3],model:3}}else{crate::material_spin_history::MaterialSpinContext::default()}).collect();
        self.material_spin_context.write(device,0,&contexts)?;
        let dummy=self.dummy.handle();let rigid=if map.n_bodies==0{dummy}else{rigid_current};
        let args=JointFemPrepareArgs{records:self.records.handle(),record_count:self.records.len() as u32,count:self.records.len() as u32,current_xyz:positions,reference_xyz:positions,vertex_count:map.nrow as u32,rigid_current:rigid,rigid_reference:rigid,bodies:map.n_bodies as u32,cache:self.cache.handle(),fault:self.fault.handle(),seam_arena_count:0};
        device.launch("joints.fem.material_spin.initial_frames",&args,self.records.len() as u32)?;self.fault.download(device)?;
        if self.fault.host().iter().any(|&v|v!=0){return Err(refusal("FEM initial material spin frame prepare failed"));}
        let values=self.material_spin_sample(device,map,1.,9,4)?.ok_or_else(||refusal("FEM initial material spin quaternion sample failed"))?;
        let mut rows=Vec::new();for (enabled,q) in selected.iter().zip(values.chunks_exact(4)){if !*enabled{continue;}let mut q:[f64;4]=std::array::from_fn(|i|f64::from(q[i]));let norm=q.iter().map(|x|x*x).sum::<f64>().sqrt();if !norm.is_finite()||!(0.99..=1.01).contains(&norm){return Err(refusal("FEM initial material spin quaternion invalid"));}for v in &mut q{*v/=norm;}rows.push(crate::material_spin_history::MaterialSpinRow{q_wxyz:q,a:[0.;3]});}
        self.invalidate_trial_cache();Ok(rows)
    }
    /// Prepare and sample the final candidate before C5 publishes history.
    pub(crate) unsafe fn sample_material_spin_candidate<D:Device>(&mut self,device:&mut D,positions:Handle,rigid_current:Handle,map:&RigidMap,dt:f32)->Result<Option<u32>,Fault>{
        if !self.frozen_ready{return Err(refusal("FEM material spin candidate frame unavailable"));}
        self.reap(device,"before final FEM material spin prepare")?;
        let rigid=if map.n_bodies==0{self.dummy.handle()}else{rigid_current};
        let args=JointFemPrepareArgs{records:self.records.handle(),record_count:self.records.len() as u32,count:self.records.len() as u32,current_xyz:positions,reference_xyz:positions,vertex_count:map.nrow as u32,rigid_current:rigid,rigid_reference:rigid,bodies:map.n_bodies as u32,cache:self.cache.handle(),fault:self.fault.handle(),seam_arena_count:0};
        device.launch("joints.fem.material_spin.final_frames",&args,self.records.len() as u32)?;
        self.fault.download(device)?;
        if let Some((row,code))=self.fault.host().iter().copied().enumerate().find(|(_,c)|*c!=0){return Err(refusal(format!("final FEM material frame prepare: row {row}, native fault {code}")));}
        let selected=self.material_spin_selected.clone().ok_or_else(||refusal("FEM material spin selection missing"))?;
        if self.trial_material_spin.is_none(){return Err(refusal("FEM material spin trial missing"));}
        let Some(values)=self.material_spin_sample(device,map,dt,8,7)? else{return Ok(self.material_spin_fault.host().iter().copied().find(|v|*v!=0))};
        let trial=self.trial_material_spin.as_mut().ok_or_else(||refusal("FEM material spin trial missing"))?;let mut rows=Vec::with_capacity(trial.rows.len());
        for (enabled,values) in selected.iter().zip(values.chunks_exact(7)){if !*enabled{continue;}let mut q:[f64;4]=std::array::from_fn(|i|f64::from(values[i]));let norm=q.iter().map(|x|x*x).sum::<f64>().sqrt();if !norm.is_finite()||!(0.99..=1.01).contains(&norm){return Err(refusal("FEM material spin candidate quaternion invalid"));}for v in &mut q{*v/=norm;}let a=std::array::from_fn(|i|f64::from(values[4+i]));if a.iter().any(|v|!v.is_finite()){return Err(refusal("FEM material spin candidate invalid"));}rows.push(crate::material_spin_history::MaterialSpinRow{q_wxyz:q,a});}
        if rows.len()!=trial.rows.len(){return Err(refusal("FEM material spin candidate count mismatch"));}trial.rows=rows;Ok(None)
    }
    /// Initial retry scope excludes authored angular drives until their
    /// delayed-control interactions have independent rollback coverage.
    pub(super) fn retry_scope_supported(&self)->bool {self.active() && self.drives.len()==0}
    pub(super) fn clear_candidate_rejection(&mut self){self.candidate_rejection=None;}
    pub(super) fn candidate_rejection(&self)->Option<&MaterialFrameCandidateRejection>{self.candidate_rejection.as_ref()}
    pub(super) fn invalidate_trial_cache(&mut self){self.reference_ready=false;self.frozen_ready=false;self.hessian_ready=false;self.index_ready=false;#[cfg(test)] {self.private_last_freeze_dt=None;}}
    #[cfg(test)]
    pub(super) fn test_inject_captured_candidate<D:Device>(&self,device:&mut D,current:Handle)->Result<(),Fault>{
        let mut row=[unsafe{std::mem::zeroed::<FemJointGpu>()};1];self.records.read(device,1,&mut row)?;
        let e=row[0].b;
        let expected=[1101622376u32,1103629693,3249212243,1104196039,3251027079,3211677632,1070208225,1074320789,1102380238];
        if e.kind!=crate::soft_joint_gpu::FEM_TET || e.vertices!=[1854,1952,1954,1853] || e.dm_inv.map(f32::to_bits)!=expected {
            return Err(refusal("private material fixture endpoint mismatch"));
        }
        // Near -I: det(F)<0 but all singular values coincide within f32
        // precision, so its closest proper rotation is non-unique. The old
        // captured inverted tet is now a valid unique material frame.
        let xyz=[[0u32,0,0],[3164597628,3165560064,3164492684],[3165189418,1016231552,3164186127],[995689944,3109289989,3173532729]];
        for (i,&vertex) in e.vertices.iter().enumerate(){
            let bytes=unsafe{std::slice::from_raw_parts(xyz[i].as_ptr().cast::<u8>(),12)};
            device.write(current,12*vertex as usize,bytes)?;
        }
        Ok(())
    }
    #[cfg(all(test,feature="cpu"))]
    pub(super) fn test_inject_row2_captured_scratch<D:Device>(&self,device:&mut D,current:Handle)->Result<(),Fault>{
        let mut row=[unsafe{std::mem::zeroed::<FemJointGpu>()};1];self.records.read(device,2,&mut row)?;
        let e=row[0].b;
        let expected=[3223007922u32,1105159694,3251337653,1103137085,3250917006,3250993085,1104652651,3222679974,3212981492];
        if e.kind!=crate::soft_joint_gpu::FEM_TET || e.vertices!=[2365,2364,2469,2472] || e.dm_inv.map(f32::to_bits)!=expected {return Err(refusal("private row2 material fixture endpoint mismatch"));}
        // Near -I, for a real ambiguous-frame refusal rather than the old
        // unique reflection, which is valid under the signed-stretch frame.
        let now=[[0u32,0,0],[3129087489,992815616,3173120050],[3164477888,1017713472,3164940986],[1016416000,1018663168,3163976772]];
        let reference=[[1061679716u32,1068636318,1041135996],[1061518693,1068373968,1042880665],[1062027766,1068089700,1042467158],[1061563545,1068316089,1042994401]];
        for (i,&vertex) in e.vertices.iter().enumerate(){
            let bytes=unsafe{std::slice::from_raw_parts(now[i].as_ptr().cast::<u8>(),12)};
            device.write(current,12*vertex as usize,bytes)?;
            let ref_bytes=unsafe{std::slice::from_raw_parts(reference[i].as_ptr().cast::<u8>(),12)};
            device.write(self.reference_xyz.handle(),12*vertex as usize,ref_bytes)?;
        }
        Ok(())
    }
    #[cfg(all(test,feature="cpu"))]
    pub(super) fn test_row2_reference_bits<D:Device>(&self,device:&mut D)->Result<[[u32;3];4],Fault>{
        let mut out=[[0u32;3];4];
        for (i,vertex) in [2365usize,2364,2469,2472].into_iter().enumerate(){let mut p=[0f32;3];self.reference_xyz.read(device,3*vertex,&mut p)?;out[i]=p.map(f32::to_bits);}
        Ok(out)
    }
    pub(super) fn consume_candidate_rejection<D:Device>(&mut self,device:&mut D,expected:&MaterialFrameCandidateRejection)->Result<(),Fault>{
        let token=self.candidate_rejection.as_ref().ok_or_else(||refusal("missing FEM candidate token"))?;
        if token.generation!=expected.generation || token.start_bits!=expected.start_bits || token.dt_bits!=expected.dt_bits ||
            token.row!=expected.row || token.side!=expected.side || token.vertices!=expected.vertices || token.fault!=expected.fault || token.statuses!=expected.statuses || token.reason!=expected.reason {
            return Err(refusal("FEM candidate token identity changed"));
        }
        self.fault.download(device)?;
        if self.fault.host().iter().enumerate().any(|(i,&v)|v!=if i==expected.row{expected.fault}else{0}) {return Err(refusal("FEM candidate fault bank changed"));}
        let mut frames=[unsafe{std::mem::zeroed::<FemFrameGpu>()};4];self.cache.read(device,4*expected.row,&mut frames)?;
        if frames.map(|f|f.status)!=expected.statuses{return Err(refusal("FEM candidate frame status changed"));}
        self.fault.seed(device,&vec![0;self.fault.len()])?;
        self.candidate_rejection=None;self.invalidate_trial_cache();Ok(())
    }
    pub fn prescribed_mapping_readback<D:Device>(&self,device:&mut D)->Result<Vec<u32>,Fault>{let mut map=vec![0;self.vertex_to_free.len()];if !map.is_empty(){self.vertex_to_free.read(device,0,&mut map)?;}Ok(map)}
    /// Accepted-boundary mask refresh. The caller supplies the authenticated
    /// complete current controlled inventory, after committing coordinates.
    /// This is deliberately separate from install: material records, limits,
    /// drive keys and their author reference are not scene-reinstalled.
    pub fn update_prescribed_vertices<D:Device>(&mut self,device:&mut D,map:&RigidMap,controlled:&[u32])->Result<(),Fault> {
        let mut indices=free_indices(map)?;
        if controlled.windows(2).any(|w|w[0]>=w[1]) {return Err(refusal("controlled inventory must be sorted unique"));}
        for &v in controlled {
            let i=v as usize;
            if i>=map.nrow || map.vbody[i]!=RIGID_UNSET {return Err(refusal("controlled vertex outside ordinary native partition"));}
            indices[i]=RIGID_UNSET;
        }
        if self.active() && self.vertex_to_free.len()!=indices.len() {return Err(refusal("private map extent differs from native partition"));}
        self.reference_ready=false;self.frozen_ready=false;self.hessian_ready=false;self.index_ready=false;
        if self.active() {self.vertex_to_free.write(device,0,&indices)?;}
        self.prescribed=!controlled.is_empty();
        Ok(())
    }
    /// Only main's strict material/topology binder may supply these records.
    /// Install clears scene and frozen-pose lifetime flags even on reused buffers.
    pub fn install<D:Device>(&mut self,device:&mut D,records:&[FemJointGpu],map:&RigidMap,controlled:&[u32])->Result<(),Fault> {
        self.rigid_cross_source=false;
        self.prescribed=!controlled.is_empty();
        self.reference_ready=false;self.frozen_ready=false;self.hessian_ready=false;self.index_ready=false;
        self.material_spin_selected=None;self.material_spin_eligible=records.iter().map(|r|r.enabled==1&&r.control.enabled==1&&r.control.coordinate_model==1).collect();self.material_spin_rigid_endpoints=records.iter().map(|r|[r.a,r.b].map(|e|if e.kind==crate::soft_joint_gpu::FEM_PDRD{Some(e.id)}else{None})).collect();self.accepted_material_spin=None;self.trial_material_spin=None;
        let mut indices=free_indices(map)?;
        for &v in controlled {let i=v as usize;if i>=map.nrow || map.vbody[i]!=RIGID_UNSET {return Err(refusal("controlled vertex outside ordinary native partition"));}indices[i]=RIGID_UNSET;}
        self.limits.size(device,0,AllocLabel("soft_joints.limits"))?;
        self.drives.size(device,0,AllocLabel("soft_joints.drives"))?;
        self.keys.size(device,0,AllocLabel("soft_joints.keys"))?;
        self.records.size(device,records.len(),AllocLabel("soft_joints.records"))?;
        self.material_spin_context.size(device,records.len().max(1),AllocLabel("soft_joints.material_spin_context"))?;
        self.material_spin_context.write(device,0,&vec![crate::material_spin_history::MaterialSpinContext::default();records.len().max(1)])?;
        if records.is_empty() {return Ok(());}
        for n in [map.nrow,map.n_cloth,map.n_bodies,map.body_base,map.dim,records.len()] {
            u32::try_from(n).map_err(|_|refusal("scene extent overflows kernel ABI"))?;
        }
        self.records.write(device,0,records)?;
        let index_words=records.len().checked_mul(27).ok_or_else(||refusal("cached mapping capacity overflow"))?;
        u32::try_from(index_words).map_err(|_|refusal("cached mapping extent overflows kernel ABI"))?;
        self.index_cache.size(device,index_words,AllocLabel("soft_joints.private_cached_indices"))?;
        // Frame prefix stays exactly 4*N*208 bytes. Tail: 2 metadata words,
        // plus two independent 576-float Hessians per row; ceil never truncates.
        let frames=cache_frame_capacity(records.len(),map.n_bodies==0)?;
        self.cache.size(device,frames,AllocLabel("soft_joints.frozen_frames_and_hessians"))?;
        self.reference_xyz.size(device,3*map.nrow,AllocLabel("soft_joints.step_start_xyz"))?;
        self.vertex_to_free.size(device,map.nrow,AllocLabel("soft_joints.free_indices"))?;
        self.vertex_to_free.write(device,0,&indices)?;
        self.fault.size(device,records.len(),AllocLabel("soft_joints.fault"))?;
        // Generated entries resolve all buffers even when an action does not
        // read one of them. This is storage, never an unknown or a body mass.
        self.dummy.size(device,36,AllocLabel("soft_joints.unused_transport"))?;
        self.dummy.write(device,0,&[0.;36])?;
        Ok(())
    }
    pub fn has_extensions(&self)->bool {self.active()&&self.limits.len()!=0}
    pub fn set_step_clock(&mut self,start:f64,end:Option<f64>) {self.start_time_s=start;self.end_time_s=end;self.frozen_ready=false;self.hessian_ready=false;self.index_ready=false;}
    /// Main repacks extensions in exactly FEM-record order, including disabled
    /// rows. Canonical f64 key conversion is the existing validated function.
    pub fn install_extensions<D:Device>(&mut self,device:&mut D,packed:&ext::PackedAngularExtensionV1)->Result<(),Fault> {
        self.frozen_ready=false;self.hessian_ready=false;self.index_ready=false;
        if packed.limits.len()!=self.records.len() || packed.limits.iter().enumerate().any(|(i,l)|l.joint_index!=i as u32) ||
           packed.drives.iter().any(|d|d.joint_index as usize>=self.records.len()) {
            return Err(refusal("extension order differs from installed material-frame joint order"));
        }
        packed.checked_upload_counts().map_err(refusal)?;
        let keys=packed.device_keys().map_err(refusal)?;
        self.limits.size(device,packed.limits.len(),AllocLabel("soft_joints.limits"))?;
        if !packed.limits.is_empty(){self.limits.write(device,0,&packed.limits)?;}
        self.drives.size(device,packed.drives.len(),AllocLabel("soft_joints.drives"))?;
        self.keys.size(device,keys.len(),AllocLabel("soft_joints.keys"))?;
        if !packed.drives.is_empty(){self.drives.write(device,0,&packed.drives)?;}
        if !keys.is_empty(){self.keys.write(device,0,&keys)?;}
        self.frozen_ready=false;self.hessian_ready=false;self.index_ready=false;Ok(())
    }
    pub fn begin_step<D:Device>(&mut self,device:&mut D,positions:Handle,map:&RigidMap)->Result<(),Fault> {
        self.reference_ready=false;self.frozen_ready=false;self.hessian_ready=false;self.index_ready=false;
        if !self.active(){return Ok(());}
        device.copy(self.reference_xyz.handle(),0,positions,0,12*map.nrow)?;
        self.reference_ready=true;Ok(())
    }
    /// Prepare once at the actual CURRENT iterate, against immutable substep
    /// start positions/rigid fit. Failure prevents all assembly and PCG work.
    pub unsafe fn freeze<D:Device>(&mut self,device:&mut D,positions:Handle,
        rigid_current:Handle,rigid_reference:Handle,map:&RigidMap,dt:f32,candidate:Option<(u64,f64)>)->Result<(),Fault> {
        self.frozen_ready=false;self.hessian_ready=false;self.index_ready=false;
        if !self.active(){return Ok(());}
        if self.has_extensions() && !super::joints::valid_extension_clock(
            self.start_time_s,self.end_time_s.unwrap_or(self.start_time_s+f64::from(dt)),dt,self.drives.len()!=0) {
            return Err(refusal("extension physical step endpoint is unrepresentable"));
        }
        if !self.reference_ready || !dt.is_finite() || dt<=0. {
            return Err(refusal("missing substep reference or invalid physical dt"));
        }
        if candidate.is_some() {
            #[cfg(all(test,feature="cpu"))]
            private_pre_b32::candidate_entry(device,&mut self.fault)?;
            // Prepare overwrites this bank. Reap the preceding operator's
            // actual lane before dispatch so an unrelated error cannot be
            // mistaken for, or erased by, a candidate-domain retry.
            self.reap(device,"before candidate material frame prepare")?;
        }
        let dummy=self.dummy.handle();
        let args=JointFemPrepareArgs {records:self.records.handle(),record_count:self.records.len() as u32,count:self.records.len() as u32,
            current_xyz:positions,reference_xyz:self.reference_xyz.handle(),vertex_count:map.nrow as u32,
            rigid_current:if map.n_bodies==0{dummy}else{rigid_current},
            rigid_reference:if map.n_bodies==0{dummy}else{rigid_reference},
            bodies:map.n_bodies as u32,cache:self.cache.handle(),fault:self.fault.handle(),seam_arena_count:0};
        #[cfg(all(test,feature="cpu"))]
        if candidate.is_some() { private_pre_b32::prepare_dispatch(); }
        device.launch("joints.fem.prepare",&args,self.records.len() as u32)?;
        // A failed download has no trustworthy current lane: return it
        // directly, without inspecting the previous host mirror.
        self.fault.download(device)?;
        if let Some((row,code))=self.fault.host().iter().copied().enumerate().find(|(_,c)|*c!=0) {
            self.frozen_ready=false;self.hessian_ready=false;self.index_ready=false;
            let error=refusal(format!("material frame prepare: row {row}, native fault {code}"));
            if let Some((generation,start))=candidate {
                // Qualification is not an observer: any failed/ambiguous
                // read leaves the original fatal unchanged and creates no token.
                if let Ok(Ok(Some(token)))=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||self.qualify_candidate(device,positions,map.nrow,row,code,generation,start,dt))) {
                    self.candidate_rejection=Some(token);
                }
            }
            // Observe only the actual failing lane, before any caller can roll
            // the attempt back. Observer failures must preserve the refusal.
            let _=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||{
                if let Err(observation)=self.capture_prepare_failure(device,positions,map.nrow,row,code) {
                    eprintln!("FEM_PREPARE_OBSERVER_ERROR {observation:?}");
                }
            }));
            return Err(error);
        }
        self.frozen_ready=true;
        #[cfg(test)] {self.private_last_freeze_dt=Some(dt);}
        let view=self.view(map,dt,None).expect("installed FEM records");
        view.add_base(device,3,dummy,dummy)?;
        self.reap(device,"frozen joint geometry")?;
        view.add_extensions(device,3,dummy,dummy)?;
        self.reap(device,"frozen joint extension geometry")?;
        if !self.prescribed && map.n_bodies==0 && full_space_identity(map) {
            let next=self.hessian_epoch.checked_add(1).filter(|e|*e<=CACHE_TOKEN_MAX);
            if next.is_none(){self.frozen_ready=false;return Err(refusal("Hessian epoch overflow"));}
            self.hessian_epoch=next.unwrap();
            let action=(self.hessian_epoch<<3)|6;
            // Same frozen records/frame/dt/extension keys; only legacy base law
            // is cached. Final curves=Some view always uses original action1.
            let prepared=(|| {
                view.add_base(device,action,dummy,dummy)?;
                view.add_extensions(device,action,dummy,dummy)?;
                self.reap(device,"frozen independent base/extension Hessians")
            })();
            if let Err(error)=prepared {self.frozen_ready=false;self.hessian_ready=false;self.index_ready=false;return Err(error);}
            let index_args=super::kernels::JointFemCachedIndexPrecomputeArgs {
                records:self.records.handle(),record_count:self.records.len() as u32,
                vertex_to_free:self.vertex_to_free.handle(),vertex_count:map.nrow as u32,
                free_count:map.n_cloth as u32,rigid_base:map.body_base as u32,
                dim:map.dim as u32,token:self.hessian_epoch,index_cache:self.index_cache.handle(),
                count:self.records.len() as u32,seam_arena_count:0};
            // Device::launch waits and checks diagnostics before publishing ready.
            if let Err(error)=device.launch("joints.fem.cached.index",&index_args,self.records.len() as u32) {
                self.frozen_ready=false;self.hessian_ready=false;self.index_ready=false;return Err(error);
            }
            self.hessian_dt=dt;self.hessian_ready=true;self.index_ready=true;
        }
        Ok(())
    }
    fn reap<D:Device>(&mut self,device:&mut D,stage:&str)->Result<(),Fault> {
        #[cfg(all(test,feature="cpu"))]
        if stage=="before candidate material frame prepare" { private_pre_b32::before_download()?; }
        self.fault.download(device)?;
        if let Some((row,code))=self.fault.host().iter().copied().enumerate().find(|(_,c)|*c!=0) {
            self.material_spin_base_fault=self.trial_material_spin.is_some() && matches!(stage,"frozen joint geometry"|"frozen joint extension geometry"|"frozen independent base/extension Hessians") && code==8 && self.fault.host().iter().all(|c|*c==0||*c==8);
            self.frozen_ready=false;self.hessian_ready=false;self.index_ready=false;
            return Err(refusal(format!("{stage}: row {row}, native fault {code}")));
        }
        Ok(())
    }
    fn qualify_candidate<D:Device>(&self,device:&mut D,current:Handle,vertices:usize,row:usize,code:u32,generation:u64,start:f64,dt:f32)->Result<Option<MaterialFrameCandidateRejection>,Fault> {
        #[cfg(all(test,feature="cpu"))]
        private_pre_b32::qualify_entry();
        if code!=259 || generation==0 || !start.is_finite() || start<0. || !dt.is_finite() || dt<=0. ||
           self.fault.host().iter().enumerate().any(|(i,&c)|i!=row && c!=0) {return Ok(None);}
        let mut frames=vec![unsafe{std::mem::zeroed::<FemFrameGpu>()};4*self.records.len()];
        self.cache.read(device,0,&mut frames)?;
        if frames.iter().enumerate().any(|(i,f)|f.status!=0 && !(i==4*row || i==4*row+1)) {return Ok(None);}
        let statuses=std::array::from_fn(|i|frames[4*row+i].status);
        let Some(side)=eligible_candidate_side(statuses) else{return Ok(None)};
        let mut record=[unsafe{std::mem::zeroed::<FemJointGpu>()};1];self.records.read(device,row,&mut record)?;
        let e=if side==0{record[0].a}else{record[0].b};
        if record[0].enabled==0 || e.kind!=crate::soft_joint_gpu::FEM_TET {return Ok(None);}
        let mut now=[[0f32;3];4];let mut reference=[[0f32;3];4];
        for (i,&vertex) in e.vertices.iter().enumerate() {
            if vertex as usize>=vertices {return Ok(None);}
            let bytes=unsafe{std::slice::from_raw_parts_mut(now[i].as_mut_ptr().cast::<u8>(),12)};
            device.read(current,12*vertex as usize,bytes)?;
            self.reference_xyz.read(device,3*vertex as usize,&mut reference[i])?;
        }
        let Some(reason)=candidate_orientation_reason(now,reference,e.dm_inv) else{return Ok(None)};
        Ok(Some(MaterialFrameCandidateRejection{generation,start_bits:start.to_bits(),dt_bits:dt.to_bits(),row,side,vertices:e.vertices,fault:code,statuses,reason}))
    }
    fn capture_prepare_failure<D:Device>(&self,device:&mut D,current:Handle,vertices:usize,row:usize,code:u32)->Result<(),Fault> {
        let mut record=[unsafe{std::mem::zeroed::<FemJointGpu>()};1];
        self.records.read(device,row,&mut record)?;
        let mut frames=[unsafe{std::mem::zeroed::<FemFrameGpu>()};4];
        self.cache.read(device,4*row,&mut frames)?;
        let mut endpoints=Vec::new();
        for (side,e) in [("a",record[0].a),("b",record[0].b)] {
            let mut curr=Vec::new();let mut reference=Vec::new();
            if e.kind==crate::soft_joint_gpu::FEM_TET {
                for &vertex in &e.vertices {
                    if vertex as usize>=vertices {return Err(refusal("observer vertex outside extent"));}
                    let mut cb=[0u32;3];let mut rb=[0u32;3];
                    let cb_bytes=unsafe{std::slice::from_raw_parts_mut(cb.as_mut_ptr().cast::<u8>(),12)};
                    device.read(current,12*vertex as usize,cb_bytes)?;
                    let mut values=[0f32;3];self.reference_xyz.read(device,3*vertex as usize,&mut values)?;
                    for k in 0..3 {rb[k]=values[k].to_bits();}curr.push(cb);reference.push(rb);
                }
            }
            endpoints.push(serde_json::json!({"side":side,"kind":e.kind,"id":e.id,"vertices":e.vertices,
                "dm_inv_bits":e.dm_inv.map(f32::to_bits),"rest_rotation_bits":e.rest_rotation.map(f32::to_bits),
                "anchor_bary_bits":e.anchor_bary.map(f32::to_bits),"center_bary_bits":e.center_bary.map(f32::to_bits),
                "current_xyz_bits":curr,"reference_xyz_bits":reference}));
        }
        eprintln!("FEM_PREPARE_OBSERVATION {}",serde_json::json!({"row":row,"fault":code,
            "frame_order":["current_a","current_b","reference_a","reference_b"],
            "frame_status":frames.map(|f|f.status),"endpoints":endpoints,
            "coordinate_source":"device current handle / frozen step-start reference; raw f32 bits"}));
        Ok(())
    }
    /// Sticky fault bank belongs to the frozen operator, independent of the
    /// original restoring lanes that later evaluation may overwrite.
    pub fn check_restoring_cache_fault<D:Device>(&mut self,device:&mut D)->Result<(),Fault> {
        if self.restoring_fault.len()!=0 {
            self.restoring_fault.download(device)?;
            if let Some((row,code))=self.restoring_fault.host().iter().copied().enumerate().find(|(_,c)|*c!=0) {
                return Err(refusal(format!("frozen restoring operator: row {row}, fault {code}")));
            }
        }
        if self.restoring_ext_active {
            self.restoring_ext_fault.download(device)?;
            if let Some((row,code))=self.restoring_ext_fault.host().iter().copied().enumerate().find(|(_,c)|*c!=0) {
                return Err(refusal(format!("frozen FEM extension: row {row}, fault {code}")));
            }
        }
        Ok(())
    }
    pub unsafe fn prepare_restoring_cache<D:Device>(&mut self,device:&mut D,view:View)->Result<FrozenRestoringView,Fault> {
        self.restoring_ext_active=false;
        if view.curves.is_none() {
            self.restoring_fault.size(device,0,AllocLabel("restoring.frozen_fault"))?;
            if !view.ready || !view.dt.is_finite() || view.dt<=0. || view.count==0 {
                return Err(refusal("invalid legacy frozen operator request"));
            }
            let n=view.count as usize;
            self.restoring_generation=self.restoring_generation.checked_add(1).filter(|v|*v<=CACHE_TOKEN_MAX).ok_or_else(||refusal("frozen legacy epoch overflow"))?;
            self.restoring_ext_cache.size(device,cache_frame_capacity(n,true)?,AllocLabel("restoring.frozen_legacy_cache"))?;
            self.restoring_ext_fault.size(device,n,AllocLabel("restoring.frozen_legacy_fault"))?;
            self.restoring_ext_fault.seed(device,&vec![0;n])?;
            let frame_bytes=n.checked_mul(4).and_then(|v|v.checked_mul(std::mem::size_of::<FemFrameGpu>())).ok_or_else(||refusal("frozen legacy frame size overflow"))?;
            device.copy(self.restoring_ext_cache.handle(),0,view.cache,0,frame_bytes)?;
            let result=FrozenRestoringView{h:view.dummy,idx:view.vertex_to_free,epoch_buffer:view.vertex_to_free,epoch:self.restoring_generation,fault:self.restoring_ext_fault.handle(),ext_cache:self.restoring_ext_cache.handle(),ext_fault:self.restoring_ext_fault.handle(),ext_active:true,legacy:true};
            let mut cached=view;
            cached.cache=result.ext_cache;
            cached.fault=result.ext_fault;
            cached.add_base(device,(result.epoch<<3)|6,view.dummy,view.dummy)?;
            cached.add_extensions(device,(result.epoch<<3)|6,view.dummy,view.dummy)?;
            self.restoring_ext_active=true;
            return Ok(result);
        }
        let c=view.curves.ok_or_else(||refusal("missing restoring curve view"))?;
        if !view.ready || !view.dt.is_finite() || view.dt<=0. || view.count==0 || view.count>u32::MAX/576 {return Err(refusal("invalid frozen restoring cache request"));}
        let n=view.count as usize;
        let hn=n.checked_mul(576).ok_or_else(||refusal("frozen hessian size overflow"))?;
        let ix=n.checked_mul(24).ok_or_else(||refusal("frozen mapping size overflow"))?;
        self.restoring_generation=self.restoring_generation.checked_add(1).ok_or_else(||refusal("frozen epoch overflow"))?;
        self.restoring_h.size(device,hn,AllocLabel("restoring.frozen_h"))?;
        self.restoring_idx.size(device,ix,AllocLabel("restoring.frozen_idx"))?;
        self.restoring_epoch.size(device,n,AllocLabel("restoring.frozen_epoch"))?;
        self.restoring_fault.size(device,n,AllocLabel("restoring.frozen_fault"))?;
        self.restoring_fault.seed(device,&vec![0;n])?;
        let mut result=FrozenRestoringView{h:self.restoring_h.handle(),idx:self.restoring_idx.handle(),epoch_buffer:self.restoring_epoch.handle(),epoch:self.restoring_generation,fault:self.restoring_fault.handle(),ext_cache:view.cache,ext_fault:view.fault,ext_active:false,legacy:false};
        let args=super::kernels::RestoringFemFrozenBuildArgs{records:view.records,material_spin_context:view.material_spin_context,record_count:view.count,frames:view.cache,vertex_to_free:view.vertex_to_free,vertex_count:view.vertices,free_count:view.free_count,rigid_base:view.base,dim:view.dim,dt:view.dt,descriptors:c.descriptors,nodes:c.nodes,bindings:c.bindings,history:c.history,descriptor_count:c.descriptor_count,node_count:c.node_count,binding_count:c.binding_count,history_count:c.history_count,observations:c.observations,epoch:result.epoch,cached_hessian:result.h,cached_index:result.idx,published_epoch:result.epoch_buffer,fault:result.fault,count:view.count,seam_arena_count:0};
        device.launch("restoring.fem.frozen.build",&args,view.count)?;
        if view.extension_count!=0 {
            if result.epoch>CACHE_TOKEN_MAX {return Err(refusal("frozen extension epoch overflow"));}
            self.restoring_ext_cache.size(device,cache_frame_capacity(n,true)?,AllocLabel("restoring.frozen_extension_cache"))?;
            self.restoring_ext_fault.size(device,n,AllocLabel("restoring.frozen_extension_fault"))?;
            self.restoring_ext_fault.seed(device,&vec![0;n])?;
            let frame_bytes=n.checked_mul(4).and_then(|v|v.checked_mul(std::mem::size_of::<FemFrameGpu>())).ok_or_else(||refusal("frozen extension frame size overflow"))?;
            device.copy(self.restoring_ext_cache.handle(),0,view.cache,0,frame_bytes)?;
            result.ext_cache=self.restoring_ext_cache.handle();
            result.ext_fault=self.restoring_ext_fault.handle();
            result.ext_active=true;
            let mut ext_view=view;
            ext_view.cache=result.ext_cache;
            ext_view.fault=result.ext_fault;
            ext_view.add_extensions(device,(result.epoch<<3)|6,view.dummy,view.dummy)?;
            self.restoring_ext_active=true;
        }
        Ok(result)
    }
    pub fn view(&mut self,map:&RigidMap,dt:f32,curves:Option<super::restoring::View>)->Option<View> {
        if !self.active(){return None;}
        Some(View {frozen_restoring:None,curves,records:self.records.handle(),material_spin_context:self.material_spin_context.handle(),cache:self.cache.handle(),vertex_to_free:self.vertex_to_free.handle(),
            index_cache:self.index_cache.handle(),index_ready:self.index_ready,
            fault:self.fault.handle(),dummy:self.dummy.handle(),count:self.records.len() as u32,
            vertices:map.nrow as u32,free_count:map.n_cloth as u32,bodies:map.n_bodies as u32,
            base:map.body_base as u32,dim:map.dim as u32,dt,ready:self.frozen_ready,
            hessian_ready:self.hessian_ready,hessian_epoch:self.hessian_epoch,hessian_dt:self.hessian_dt,
            limits:self.limits.handle(),drives:if self.drives.len()==0{self.dummy.handle()}else{self.drives.handle()},
            keys:if self.keys.len()==0{self.dummy.handle()}else{self.keys.handle()},
            extension_count:self.limits.len() as u32,drive_count:self.drives.len() as u32,
            sample_time_s:self.end_time_s.unwrap_or(self.start_time_s+f64::from(dt)) as f32})
    }
}
impl View {
    /// True only when the opt-in frozen row path will enqueue real work.
    pub(super) fn row_parallel_available(&self) -> bool {
        self.count > 0 && self.count <= u32::MAX / 576
            && self.frozen_restoring.is_some_and(|f| !f.legacy)
    }

    /// Actions 0/1 write the ORIGINAL reduced vector; 2 adds only PDRD self
    /// blocks; 5 adds only free nodal self blocks to the preconditioner.
    unsafe fn add_base<D:Device>(&self,device:&mut D,action:u32,input:Handle,output:Handle)->Result<(),Fault> {
        if !self.ready || !self.dt.is_finite() || self.dt<=0. {return Err(refusal("unprepared Newton operator"));}
        if let Some(c)=self.curves {
            let args=super::kernels::JointRestoringFemApplyArgs{records:self.records,material_spin_context:self.material_spin_context,record_count:self.count,count:self.count,cache:self.cache,vertex_to_free:self.vertex_to_free,vertex_count:self.vertices,free_count:self.free_count,rigid_base:self.base,dim:self.dim,dt:self.dt,action,input,output,nodal_blocks:if action==5{output}else{self.dummy},rigid_blocks:if action==2{output}else{self.dummy},fault:c.fem_fault,descriptors:c.descriptors,nodes:c.nodes,bindings:c.bindings,history:c.history,descriptor_count:c.descriptor_count,node_count:c.node_count,binding_count:c.binding_count,history_count:c.history_count,observations:c.observations,seam_arena_count:0};
            device.launch("restoring.fem.apply",&args,self.count)?;return Ok(());
        }
        let args=JointFemApplyArgs {records:self.records,material_spin_context:self.material_spin_context,record_count:self.count,count:self.count,cache:self.cache,
            vertex_to_free:self.vertex_to_free,vertex_count:self.vertices,free_count:self.free_count,
            rigid_base:self.base,dim:self.dim,dt:self.dt,action,input,output,
            nodal_blocks:if action==5{output}else{self.dummy},
            rigid_blocks:if action==2{output}else{self.dummy},fault:self.fault,seam_arena_count:0};
        device.launch("joints.fem.apply",&args,self.count)?;Ok(())
    }
    unsafe fn add_extensions<D:Device>(&self,device:&mut D,action:u32,input:Handle,output:Handle)->Result<(),Fault> {
        if self.extension_count!=0 {
            let extension=JointFemExtensionApplyArgs {records:self.records,material_spin_context:self.material_spin_context,record_count:self.count,count:self.count,
                cache:self.cache,vertex_to_free:self.vertex_to_free,vertex_count:self.vertices,free_count:self.free_count,
                rigid_base:self.base,dim:self.dim,limits:self.limits,drives:self.drives,drive_count:self.drive_count,
                keys:self.keys,sample_time_s:self.sample_time_s,dt:self.dt,action,input,output,
                nodal_blocks:if action==5{output}else{self.dummy},rigid_blocks:if action==2{output}else{self.dummy},fault:self.fault,seam_arena_count:0};
            device.launch("joints.fem.extension",&extension,self.count)?;
        }
        Ok(())
    }
    unsafe fn encode_base(&self,encoder:&mut dyn Encoder,action:u32,input:Handle,output:Handle)->Result<(),Fault> {
        if !self.ready || !self.dt.is_finite() || self.dt<=0. {return Err(refusal("unprepared Newton operator"));}
        if let Some(c)=self.curves {
            let args=super::kernels::JointRestoringFemApplyArgs{records:self.records,material_spin_context:self.material_spin_context,record_count:self.count,count:self.count,cache:self.cache,vertex_to_free:self.vertex_to_free,vertex_count:self.vertices,free_count:self.free_count,rigid_base:self.base,dim:self.dim,dt:self.dt,action,input,output,nodal_blocks:if action==5{output}else{self.dummy},rigid_blocks:if action==2{output}else{self.dummy},fault:c.fem_fault,descriptors:c.descriptors,nodes:c.nodes,bindings:c.bindings,history:c.history,descriptor_count:c.descriptor_count,node_count:c.node_count,binding_count:c.binding_count,history_count:c.history_count,observations:c.observations,seam_arena_count:0};
            encoder.elements(&args,self.count)?;return Ok(());
        }
        let args=JointFemApplyArgs {records:self.records,material_spin_context:self.material_spin_context,record_count:self.count,count:self.count,cache:self.cache,
            vertex_to_free:self.vertex_to_free,vertex_count:self.vertices,free_count:self.free_count,
            rigid_base:self.base,dim:self.dim,dt:self.dt,action,input,output,
            nodal_blocks:if action==5{output}else{self.dummy},
            rigid_blocks:if action==2{output}else{self.dummy},fault:self.fault,seam_arena_count:0};
        encoder.elements(&args,self.count)?;Ok(())
    }
    unsafe fn encode_extensions(&self,encoder:&mut dyn Encoder,action:u32,input:Handle,output:Handle)->Result<(),Fault> {
        if self.extension_count!=0 {
            let extension=JointFemExtensionApplyArgs {records:self.records,material_spin_context:self.material_spin_context,record_count:self.count,count:self.count,
                cache:self.cache,vertex_to_free:self.vertex_to_free,vertex_count:self.vertices,free_count:self.free_count,
                rigid_base:self.base,dim:self.dim,limits:self.limits,drives:self.drives,drive_count:self.drive_count,
                keys:self.keys,sample_time_s:self.sample_time_s,dt:self.dt,action,input,output,
                nodal_blocks:if action==5{output}else{self.dummy},rigid_blocks:if action==2{output}else{self.dummy},fault:self.fault,seam_arena_count:0};
            encoder.elements(&extension,self.count)?;
        }
        Ok(())
    }
    // Isolated stage1: identical argument records and entry bodies, same ordered encoder.
    // Only action1 may be appended to the combined Krylov stream; actions0/5
    // retain their original assembly path and lifetime/fault handling.
    pub unsafe fn encode_krylov(&self,encoder:&mut dyn Encoder,input:Handle,output:Handle)->Result<(),Fault> {
        self.encode_krylov_rows(encoder,input,output,false)
    }
    pub unsafe fn encode_krylov_rows(&self,encoder:&mut dyn Encoder,input:Handle,output:Handle,row_parallel:bool)->Result<(),Fault> {
        if let Some(f)=self.frozen_restoring {
            if f.legacy {
                if row_parallel {
                    // dummy is the preallocated read-only zero word for stride1.
                    return self.encode_krylov_guarded(encoder,input,output,self.dummy);
                }
                let mut cached=*self; cached.cache=f.ext_cache; cached.fault=f.ext_fault;
                cached.encode_base(encoder,(f.epoch<<3)|7,input,output)?;
                return cached.encode_extensions(encoder,(f.epoch<<3)|7,input,output);
            }
            let _=self.curves.ok_or_else(||refusal("missing frozen curve view"))?;
            if row_parallel && self.count<=u32::MAX/576 {
                if let Some(work)=self.count.checked_mul(24) {
                    let args=super::kernels::RestoringFemFrozenRowArgs {
                        record_count:self.count,cached_hessian:f.h,cached_index:f.idx,
                        published_epoch:f.epoch_buffer,epoch:f.epoch,input,output,fault:f.fault,
                        stopped:f.fault,use_stopped:0,extent:work,count:work,seam_arena_count:0,
                    };
                    encoder.elements(&args,work)?;
                    if f.ext_active {
                        let ext=super::kernels::JointFemExtensionFrozenRowArgs {
                            records:self.records,record_count:self.count,cache:f.ext_cache,
                            vertex_to_free:self.vertex_to_free,vertex_count:self.vertices,
                            free_count:self.free_count,rigid_base:self.base,dim:self.dim,
                            limits:self.limits,drives:self.drives,drive_count:self.drive_count,
                            dt:self.dt,token:f.epoch,input,output,fault:f.ext_fault,
                            stopped:f.ext_fault,use_stopped:0,extent:work,count:work,seam_arena_count:0,
                        };
                        return encoder.elements(&ext,work);
                    }
                    return self.encode_extensions(encoder,1,input,output);
                }
            }
            let args=super::kernels::RestoringFemFrozenApplyArgs{record_count:self.count,cached_hessian:f.h,cached_index:f.idx,published_epoch:f.epoch_buffer,epoch:f.epoch,input,output,fault:f.fault,count:self.count,seam_arena_count:0};
            encoder.elements(&args,self.count)?;
            if f.ext_active {
                let mut ext=*self; ext.cache=f.ext_cache; ext.fault=f.ext_fault;
                return ext.encode_extensions(encoder,(f.epoch<<3)|7,input,output);
            }
            return self.encode_extensions(encoder,1,input,output);
        }
        let cached=self.hessian_ready && self.curves.is_none() && self.bodies==0 &&
            self.hessian_epoch!=0 && self.dt.to_bits()==self.hessian_dt.to_bits();
        let action=if cached{(self.hessian_epoch<<3)|7}else{1};
        self.encode_base(encoder,action,input,output)?;
        self.encode_extensions(encoder,action,input,output)
    }
    pub(super) fn batch8_ready(&self) -> bool {
        self.ready && self.frozen_restoring.is_some_and(|f| if f.legacy { f.epoch != 0 && self.dt.is_finite() && self.dt > 0. } else { self.curves.is_some() })
    }
    pub(super) unsafe fn encode_krylov_batch8(
        &self, encoder: &mut dyn Encoder, input: Handle, output: Handle, status: Handle, row_parallel: bool,
    ) -> Result<(), Fault> {
        if !self.batch8_ready() { return Err(refusal("batch8 requires frozen soft restoring")); }
        let f = self.frozen_restoring.expect("checked");
        if f.legacy { return self.encode_krylov_guarded(encoder,input,output,status); }
        let row_work = if row_parallel && self.row_parallel_available() { self.count.checked_mul(24) } else { None };
        if let Some(work) = row_work {
            let args = super::kernels::RestoringFemFrozenRowArgs {
                record_count:self.count, cached_hessian:f.h, cached_index:f.idx,
                published_epoch:f.epoch_buffer, epoch:f.epoch, input, output, fault:f.fault,
                stopped:status, use_stopped:1, extent:work, count:work, seam_arena_count:0,
            };
            encoder.elements(&args,work)?;
        } else {
        let base = super::kernels::ResidentGuardRestoringFemFrozenApplyArgs {
            record_count: self.count, cached_hessian: f.h, cached_index: f.idx,
            published_epoch: f.epoch_buffer, epoch: f.epoch, input, output, fault: f.fault,
            status, count: self.count, seam_arena_count: 0,
        };
        encoder.elements(&base, self.count)?;
        }
        if self.extension_count != 0 {
            if let Some(work) = row_work.filter(|_| f.ext_active) {
                let ext = super::kernels::JointFemExtensionFrozenRowArgs {
                    records:self.records,record_count:self.count,cache:f.ext_cache,
                    vertex_to_free:self.vertex_to_free,vertex_count:self.vertices,
                    free_count:self.free_count,rigid_base:self.base,dim:self.dim,
                    limits:self.limits,drives:self.drives,drive_count:self.drive_count,
                    dt:self.dt,token:f.epoch,input,output,fault:f.ext_fault,
                    stopped:status,use_stopped:1,extent:work,count:work,seam_arena_count:0,
                };
                return encoder.elements(&ext,work);
            }
            let (cache, fault, action) = if f.ext_active {
                (f.ext_cache, f.ext_fault, (f.epoch << 3) | 7)
            } else { (self.cache, self.fault, 1) };
            let ext = super::kernels::ResidentGuardJointFemExtensionApplyArgs {
                records:self.records, material_spin_context:self.material_spin_context,
                record_count:self.count, cache, vertex_to_free:self.vertex_to_free,
                vertex_count:self.vertices, free_count:self.free_count, rigid_base:self.base,
                dim:self.dim, limits:self.limits, drives:self.drives, drive_count:self.drive_count,
                keys:self.keys, sample_time_s:self.sample_time_s, dt:self.dt, action, input, output,
                nodal_blocks:self.dummy, rigid_blocks:self.dummy, fault, status,
                count:self.count, seam_arena_count:0,
            };
            encoder.elements(&ext, self.count)?;
        }
        Ok(())
    }

    pub fn resident_cache_ready(&self)->bool {
        self.ready && self.dt.is_finite() && self.dt>0. && self.hessian_ready &&
            self.curves.is_none() && self.bodies==0 && self.hessian_epoch!=0 &&
            self.dt.to_bits()==self.hessian_dt.to_bits()
    }
    pub unsafe fn encode_krylov_guarded(&self,encoder:&mut dyn Encoder,input:Handle,output:Handle,status:Handle)->Result<(),Fault> {
        let legacy=self.frozen_restoring.filter(|f| f.legacy);
        if let Some(f)=legacy {
            if !self.ready || f.epoch==0 || !self.dt.is_finite() || self.dt<=0. {return Err(refusal("batch operator requires current legacy cache"));}
        } else if !self.resident_cache_ready(){return Err(refusal("batch operator requires current resident cache"));}
        let token=legacy.map_or(self.hessian_epoch, |f| f.epoch);
        let cache=legacy.map_or(self.cache, |f| f.ext_cache);
        let fault=legacy.map_or(self.fault, |f| f.ext_fault);
        let index_ready=legacy.is_none() && self.index_ready;
        let work_count=self.count.checked_mul(24).ok_or_else(||refusal("cached row work extent overflow"))?;
        if self.extension_count==0 {
            let args=super::kernels::JointFemCachedIndexedRowsArgs {records:self.records,record_count:self.count,count:work_count,cache,
                vertex_to_free:self.vertex_to_free,vertex_count:self.vertices,free_count:self.free_count,
                rigid_base:self.base,dim:self.dim,index_cache:self.index_cache,index_ready:index_ready as u32,
                dt:self.dt,token,input,output,
                fault,stopped:status,seam_arena_count:0};
            encoder.elements(&args,work_count)?;
        } else {
            let fused=super::kernels::JointFemCachedIndexedFusedRowsArgs {records:self.records,record_count:self.count,count:work_count,
                cache,vertex_to_free:self.vertex_to_free,vertex_count:self.vertices,free_count:self.free_count,
                rigid_base:self.base,dim:self.dim,index_cache:self.index_cache,index_ready:index_ready as u32,
                limits:self.limits,drives:self.drives,drive_count:self.drive_count,
                dt:self.dt,token,input,output,
                fault,stopped:status,seam_arena_count:0};
            encoder.elements(&fused,work_count)?;
        }
        Ok(())
    }
    pub unsafe fn encode_affine(&self,encoder:&mut dyn Encoder,input:Handle,output:Handle)->Result<(),Fault> {
        self.encode_base(encoder,1,input,output)?;
        self.encode_extensions(encoder,1,input,output)
    }
    pub unsafe fn add_krylov<D:Device>(&self,device:&mut D,input:Handle,output:Handle)->Result<(),Fault> {
        if let Some(f)=self.frozen_restoring {
            if f.legacy {
                let mut cached=*self; cached.cache=f.ext_cache; cached.fault=f.ext_fault;
                cached.add_base(device,(f.epoch<<3)|7,input,output)?;
                return cached.add_extensions(device,(f.epoch<<3)|7,input,output);
            }
            let _=self.curves.ok_or_else(||refusal("missing frozen curve view"))?;
            let args=super::kernels::RestoringFemFrozenApplyArgs{record_count:self.count,cached_hessian:f.h,cached_index:f.idx,published_epoch:f.epoch_buffer,epoch:f.epoch,input,output,fault:f.fault,count:self.count,seam_arena_count:0};
            device.launch("restoring.fem.frozen.apply",&args,self.count)?;
            if f.ext_active {
                let mut ext=*self; ext.cache=f.ext_cache; ext.fault=f.ext_fault;
                return ext.add_extensions(device,(f.epoch<<3)|7,input,output);
            }
            return self.add_extensions(device,1,input,output);
        }
        self.add(device,1,input,output)
    }
    pub unsafe fn add<D:Device>(&self,device:&mut D,action:u32,input:Handle,output:Handle)->Result<(),Fault> {
        self.add_base(device,action,input,output)?;
        self.add_extensions(device,action,input,output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rigid_cross_scope_rejects_fem_angular_and_incomplete_bindings() {
        let mut row:crate::soft_joint_gpu::FemJointGpu=unsafe{std::mem::zeroed()};
        row.enabled=1;row.a.kind=crate::soft_joint_gpu::FEM_PDRD;row.b.kind=crate::soft_joint_gpu::FEM_PDRD;
        let mut owned=crate::soft_joint::OwnedSoftScene{scene_uuid:String::new(),scene_inputs_sha256:String::new(),payload_sha256:String::new(),controlled_vertices:vec![],control_receipt:serde_json::Value::Null,control_events:None,bodies:vec![],wires:vec![],records:vec![row],extensions:None,connected_collision:None};
        assert!(pure_rigid_cross_rows(&owned));
        owned.records[0].a.kind=crate::soft_joint_gpu::FEM_TET;assert!(!pure_rigid_cross_rows(&owned));
        owned.records[0].a.kind=crate::soft_joint_gpu::FEM_PDRD;owned.records[0].v2.enabled=1;assert!(!pure_rigid_cross_rows(&owned));
        owned.records[0].v2.enabled=0;owned.records[0].control.damping_nms_per_rad[0]=1.;assert!(!pure_rigid_cross_rows(&owned));
        let b=crate::restoring_curve_gpu::CurveBinding{kind:crate::cross_links::DISTANCE_KIND,channel:3,history_index:0,..Default::default()};
        assert!(rigid_cross_bindings_cover(1,&[b]));assert!(!rigid_cross_bindings_cover(2,&[b]));
        assert!(!rigid_cross_bindings_cover(1,&[b,b]));
        let angular=crate::restoring_curve_gpu::CurveBinding{kind:1,..Default::default()};
        assert!(!rigid_cross_bindings_cover(1,&[b,angular]));
    }    #[test] fn geometric_inversion_f32_uncertain_is_narrowly_qualified() {
        let dm=[3223007922u32,1105159694,3251337653,1103137085,3250917006,3250993085,1104652651,3222679974,3212981492].map(f32::from_bits);
        let current=[[1061679882u32,1068636208,1041136656],[1061518833,1068373857,1042881214],[1062027917,1068089579,1042467755],[1061563720,1068315974,1042995093]].map(|p|p.map(f32::from_bits));
        let reference=[[1061679716u32,1068636318,1041135996],[1061518693,1068373968,1042880665],[1062027766,1068089700,1042467158],[1061563545,1068316089,1042994401]].map(|p|p.map(f32::from_bits));
        assert_eq!(f32_orientation(current,dm),F32Orientation::FiniteIntervalCrossesZero);
        assert!(geometric_orientation(current,dm).unwrap().hi<0.);
        assert!(geometric_orientation(reference,dm).unwrap().lo>0.);
        assert_eq!(candidate_orientation_reason(current,reference,dm),Some(MaterialFrameRejectReason::CertifiedGeometricInversionF32SignUncertain));
        assert_eq!(candidate_orientation_reason(reference,reference,dm),None);
        assert_eq!(candidate_orientation_reason(current,current,dm),None);
        let mut invalid=current;invalid[0][0]=f32::NAN;
        assert_eq!(f32_orientation(invalid,dm),F32Orientation::Invalid);
        assert_eq!(candidate_orientation_reason(invalid,reference,dm),None);
        let mut subnormal=current;subnormal[0][0]=f32::from_bits(1);
        assert_eq!(candidate_orientation_reason(subnormal,reference,dm),None);
        let identity=[1.,0.,0.,0.,1.,0.,0.,0.,1.];
        let positive=[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]];
        let zero=[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,0.]];
        let negative=[[0.,0.,0.],[0.,1.,0.],[1.,0.,0.],[0.,0.,1.]];
        assert_eq!(candidate_orientation_reason(positive,positive,identity),None);
        assert_eq!(candidate_orientation_reason(zero,positive,identity),None);
        assert_eq!(candidate_orientation_reason(negative,positive,identity),Some(MaterialFrameRejectReason::StrictF32Negative));
        assert_eq!(eligible_candidate_side([259,259,0,0]),None);
        assert_eq!(eligible_candidate_side([0,259,259,0]),None);
    }
    #[test] fn certified_orientation_actual_capture_and_uncertainty() {
        let d=[1101622376u32,1103629693,3249212243,1104196039,3251027079,3211677632,1070208225,1074320789,1102380238].map(f32::from_bits);
        let current=[[3179423212u32,1068340661,3195503038],[3174280186,1068834876,3193941364],[3176191708,1068238198,3196067404],[3180823934,1068608462,3194349133]].map(|r|r.map(f32::from_bits));
        let reference=[[3179349321u32,1068360838,3195379237],[3174109638,1068853045,3193808202],[3176001469,1068258983,3195926146],[3180734244,1068628090,3194194489]].map(|r|r.map(f32::from_bits));
        assert_eq!(certified_tet_orientation(current,d),Some(-1));
        assert_eq!(certified_tet_orientation(reference,d),Some(1));
        let identity=[1.,0.,0.,0.,1.,0.,0.,0.,1.];
        let mut near=[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,f32::from_bits(1)]];
        assert_eq!(certified_tet_orientation(near,identity),None);
        near[3][2]=0.;assert_eq!(certified_tet_orientation(near,identity),None);
        near[3][2]=f32::NAN;assert_eq!(certified_tet_orientation(near,identity),None);
        near[3][2]=f32::INFINITY;assert_eq!(certified_tet_orientation(near,identity),None);
        let mut input_subnormal=[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]];input_subnormal[0][0]=f32::from_bits(1);
        assert_eq!(certified_tet_orientation(input_subnormal,identity),None);
        let mut inverse_subnormal=identity;inverse_subnormal[1]=f32::from_bits(1);
        assert_eq!(certified_tet_orientation([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],inverse_subnormal),None);
        assert_eq!(eligible_candidate_side([0,259,0,0]),Some(1));
        for bad in [[259,259,0,0],[0,259,0,259],[0,258,0,0],[0,0,0,0]] {assert_eq!(eligible_candidate_side(bad),None);}
    }
    fn map()->RigidMap {RigidMap{nrow:5,n_bodies:1,n_cloth:3,body_base:9,dim:15,
        vbody:vec![RIGID_UNSET,0,0,RIGID_UNSET,RIGID_UNSET],cloth_off:vec![0,0,0,3,6],..Default::default()}}
    #[test] fn uses_original_free_indices_and_excludes_real_rigid_vertices() {
        assert_eq!(free_indices(&map()).unwrap(),vec![0,RIGID_UNSET,RIGID_UNSET,1,2]);
    }
    #[test] fn refuses_aliasing_slots_and_fake_partition() {
        let mut m=map();m.cloth_off[4]=3;assert!(free_indices(&m).is_err());
        let mut m=map();m.dim+=6;assert!(free_indices(&m).is_err());
    }
    #[test] fn pure_fem_retains_every_nodal_dof() {
        let m=RigidMap {nrow:4,n_cloth:4,body_base:12,dim:12,
            vbody:vec![RIGID_UNSET;4],cloth_off:vec![0,3,6,9],..Default::default()};
        assert_eq!(free_indices(&m).unwrap(),vec![0,1,2,3]);
        assert!(!m.needs_projection());
    }
}
#[cfg(all(test, feature = "cpu"))]
pub(crate) mod private_pre_b32 {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use ppf_cts_compute::{Device, Fault, ReadbackBuffer};

    #[derive(Clone, Copy)]
    pub(crate) enum Mode { NonzeroBank, ReadbackError }
    // 0=disabled, 1=write actual bank, 2=fail at candidate download boundary.
    static MODE: AtomicUsize = AtomicUsize::new(0);
    static REAP: AtomicUsize = AtomicUsize::new(0);
    static PREPARE: AtomicUsize = AtomicUsize::new(0);
    static QUALIFY: AtomicUsize = AtomicUsize::new(0);

    pub(crate) struct Armed;
    impl Armed {
        pub(crate) fn new(mode: Mode) -> Self {
            REAP.store(0, Ordering::SeqCst); PREPARE.store(0, Ordering::SeqCst);
            QUALIFY.store(0, Ordering::SeqCst);
            MODE.store(match mode { Mode::NonzeroBank => 1, Mode::ReadbackError => 2 }, Ordering::SeqCst);
            Self
        }
    }
    impl Drop for Armed { fn drop(&mut self) { MODE.store(0, Ordering::SeqCst); } }
    pub(crate) fn hits() -> (usize, usize, usize) {
        (REAP.load(Ordering::SeqCst), PREPARE.load(Ordering::SeqCst), QUALIFY.load(Ordering::SeqCst))
    }
    pub(crate) fn candidate_entry<D: Device>(device: &mut D, fault: &mut ReadbackBuffer<u32>) -> Result<(), Fault> {
        let mode = MODE.load(Ordering::SeqCst);
        if mode == 0 { return Ok(()); }
        if mode == 1 {
            if fault.len() <= 1 { return Err(Fault::Platform { call: "private_pre_b32_fixture", detail: "03 bank has no row 1".into() }); }
            // handle() marks the CPU mirror stale. Write only the actual device
            // allocation, so production reap must download it before inspection.
            device.write(fault.handle(), 4, &259u32.to_ne_bytes())?;
        }
        Ok(())
    }
    pub(crate) fn before_download() -> Result<(), Fault> {
        if MODE.load(Ordering::SeqCst) != 0 { REAP.fetch_add(1, Ordering::SeqCst); }
        if MODE.load(Ordering::SeqCst) == 2 {
            return Err(Fault::Platform { call: "private_pre_b32_download", detail: "injected download failure".into() });
        }
        Ok(())
    }
    pub(crate) fn prepare_dispatch() {
        if MODE.load(Ordering::SeqCst) != 0 { PREPARE.fetch_add(1, Ordering::SeqCst); }
    }
    pub(crate) fn qualify_entry() {
        if MODE.load(Ordering::SeqCst) != 0 { QUALIFY.fetch_add(1, Ordering::SeqCst); }
    }
}




#[cfg(all(test,feature="cpu"))]
mod joint_row_host_test {
    use ppf_cts_compute::{AllocLabel,Buffer,Device};
    use super::super::kernels::{RestoringFemFrozenApplyArgs,RestoringFemFrozenRowArgs};
    fn seeded<T:ppf_cts_compute::Pod+Default>(d:&mut impl Device, data:&[T])->Buffer<T>{
        let mut b=Buffer::none();b.size(d,data.len(),AllocLabel("row.host.test")).unwrap();b.write(d,0,data).unwrap();b
    }
    #[test]
    fn frozen_row_real_host_device_matches_original_and_stop() {
        let mut d=super::super::launch::host_device();let n=3u32;let dim=18usize;let epoch=17u32;
        let h:Vec<f32>=(0..576*n).map(|i|((i%29) as f32-14.)*0.013).collect();
        let idx:Vec<u32>=(0..24*n).map(|i|if i%7==0{u32::MAX}else{i%dim as u32}).collect();
        let x:Vec<f32>=(0..dim).map(|i|(i as f32-7.)*0.11).collect();
        let hb=seeded(&mut d,&h);let ib=seeded(&mut d,&idx);let eb=seeded(&mut d,&[epoch;3]);let xb=seeded(&mut d,&x);
        let old=seeded(&mut d,&vec![0f32;dim]);let new=seeded(&mut d,&vec![0f32;dim]);
        let f0=seeded(&mut d,&[0u32;3]);let f1=seeded(&mut d,&[0u32;3]);let mut stop=seeded(&mut d,&[0f32]);
        let orig=RestoringFemFrozenApplyArgs{record_count:n,cached_hessian:hb.handle(),cached_index:ib.handle(),published_epoch:eb.handle(),epoch,input:xb.handle(),output:old.handle(),fault:f0.handle(),count:n,seam_arena_count:0};
        let rows=RestoringFemFrozenRowArgs{record_count:n,cached_hessian:hb.handle(),cached_index:ib.handle(),published_epoch:eb.handle(),epoch,input:xb.handle(),output:new.handle(),fault:f1.handle(),stopped:stop.handle(),use_stopped:1,extent:24*n,count:24*n,seam_arena_count:0};
        unsafe {d.launch("test.orig",&orig,n).unwrap();d.launch("test.rows",&rows,24*n).unwrap();}
        let mut a=vec![0f32;dim];let mut b=vec![0f32;dim];old.read(&mut d,0,&mut a).unwrap();new.read(&mut d,0,&mut b).unwrap();
        for (a,b) in a.iter().zip(&b){assert!((a-b).abs()<1e-5);}
        stop.write(&mut d,0,&[1f32]).unwrap();unsafe {d.launch("test.stopped",&rows,24*n).unwrap();}
        let mut after=vec![0f32;dim];new.read(&mut d,0,&mut after).unwrap();assert_eq!(b,after);
        let mut errors=[9u32;3];f1.read(&mut d,0,&mut errors).unwrap();assert_eq!(errors,[0;3]);
    }
}

impl View {
    /// Diagnostic only. The real frozen cache remains read-only and all fault/output storage is private.
    pub(super) fn legacy_rows_self_check(&self,device:&mut impl Device,input:Handle)->Result<bool,Fault> {
        let f=match self.frozen_restoring {Some(f) if f.legacy && f.ext_active && self.ready && f.epoch>0 => f,_=>return Ok(false)};
        if self.count==0 || self.dim==0 || self.dim as usize>input.size as usize {return Err(refusal("legacy rows selfcheck extent"));}
        let n=self.dim as usize; let rows=self.count as usize;
        let mut src=Buffer::<f32>::none();let mut old_out=Buffer::<f32>::none();let mut new_out=Buffer::<f32>::none();
        let mut stop=Buffer::<f32>::none();let mut faults=Buffer::<u32>::none();let mut records=Buffer::<FemJointGpu>::none();
        let result=(|| -> Result<(),Fault>{
            src.size(device,n,AllocLabel("legacy.check.input"))?;old_out.size(device,n,AllocLabel("legacy.check.old"))?;
            new_out.size(device,n,AllocLabel("legacy.check.new"))?;stop.size(device,1,AllocLabel("legacy.check.stop"))?;
            faults.size(device,rows,AllocLabel("legacy.check.fault"))?;records.size(device,rows,AllocLabel("legacy.check.records"))?;
            device.copy(records.handle(),0,self.records,0,rows*std::mem::size_of::<FemJointGpu>())?;
            let mut rec=vec![unsafe{std::mem::zeroed::<FemJointGpu>()};rows];records.read(device,0,&mut rec)?;
            let enabled=rec.iter().filter(|r|r.enabled!=0).count();if enabled==0{return Err(refusal("legacy rows selfcheck no enabled rows"));}
            let mut test_f=f;test_f.ext_fault=faults.handle();test_f.fault=faults.handle();
            let mut new=*self;new.frozen_restoring=Some(test_f);
            let mut old=*self;old.cache=f.ext_cache;old.fault=faults.handle();
            for sample in 0..2 {
                if sample==0 {device.copy(src.handle(),0,input,0,n*4)?;} else {
                    let values:Vec<f32>=(0..n).map(|i|((i%23) as f32-11.0)*0.03125).collect();src.write(device,0,&values)?;
                }
                let mut values=vec![0.0;n];src.read(device,0,&mut values)?;
                if values.iter().any(|v|!v.is_finite()){return Err(refusal("legacy rows selfcheck nonfinite input"));}
                old_out.write(device,0,&vec![0.0;n])?;new_out.write(device,0,&vec![0.0;n])?;faults.write(device,0,&vec![0u32;rows])?;stop.write(device,0,&[0.0])?;
                device.run("legacy.rows.oracle",|enc|unsafe{old.encode_base(enc,(f.epoch<<3)|7,src.handle(),old_out.handle())?;old.encode_extensions(enc,(f.epoch<<3)|7,src.handle(),old_out.handle())})?;
                let mut errors=vec![0u32;rows];faults.read(device,0,&mut errors)?;if errors.iter().any(|&x|x!=0){return Err(refusal("legacy rows oracle fault"));}
                device.run("legacy.rows.guarded",|enc|unsafe{new.encode_krylov_guarded(enc,src.handle(),new_out.handle(),stop.handle())})?;
                faults.read(device,0,&mut errors)?;if errors.iter().any(|&x|x!=0){return Err(refusal("legacy rows guarded fault"));}
                let mut a=vec![0.0;n];let mut b=vec![0.0;n];old_out.read(device,0,&mut a)?;new_out.read(device,0,&mut b)?;
                let mut maxdiff=0.0f32;let mut maxref=0.0f32;
                for (x,y) in a.iter().zip(&b){if !x.is_finite()||!y.is_finite(){return Err(refusal("legacy rows nonfinite output"));}maxdiff=maxdiff.max((x-y).abs());maxref=maxref.max(x.abs());}
                eprintln!("PPF_LEGACY_ROWS_COMPARE sample={} maxdiff={:.9e} maxref={:.9e} atol=1e-5 rtol=1e-4",sample,maxdiff,maxref);
                if maxdiff>1e-5+1e-4*maxref{return Err(refusal("legacy rows oracle mismatch"));}
            }
            for case in 0..3 {
                let mut variant=new;let mut vf=test_f;
                if case==0 {vf.epoch=if f.epoch<0x1fff_ffff {f.epoch+1}else{f.epoch-1};}
                if case==1 {variant.dt=self.dt*2.0;if !variant.dt.is_finite(){return Err(refusal("legacy rows invalid diagnostic dt"));}}
                if case==2 {vf.epoch=if f.epoch<0x1fff_ffff {f.epoch+1}else{f.epoch-1};variant.dt=self.dt*2.0;}
                variant.frozen_restoring=Some(vf);new_out.write(device,0,&vec![7.0;n])?;faults.write(device,0,&vec![0u32;rows])?;stop.write(device,0,&[if case==2{1.0}else{0.0}])?;
                device.run("legacy.rows.invalid",|enc|unsafe{variant.encode_krylov_guarded(enc,src.handle(),new_out.handle(),stop.handle())})?;
                let mut errors=vec![0u32;rows];let mut output=vec![0.0;n];faults.read(device,0,&mut errors)?;new_out.read(device,0,&mut output)?;
                if output.iter().any(|v|v.to_bits()!=7.0f32.to_bits()) || rec.iter().zip(&errors).any(|(r,&e)|e!=if case!=2 && r.enabled!=0 {261}else{0}) {return Err(refusal("legacy rows invalid guard"));}
            }
            eprintln!("PPF_LEGACY_ROWS_SELF_CHECK passed enabled={} extension_configured={} rigid_bodies={} free_vertices={} duplicate=NOT_ASSESSED extension_active=NOT_ASSESSED",enabled,self.extension_count,self.bodies,self.free_count);
            Ok(())
        })();
        let f1=src.free(device);let f2=old_out.free(device);let f3=new_out.free(device);let f4=stop.free(device);let f5=faults.free(device);let f6=records.free(device);
        result?;f1?;f2?;f3?;f4?;f5?;f6?;Ok(true)
    }
}
