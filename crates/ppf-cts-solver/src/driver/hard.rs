//! Bounded rootless hard D6 / angular hinge driver. This module never adds a
//! penalty to the smooth operator and never changes the PCG recurrence.
//!
//! Call `freeze` after the current rigid fit, `correct` after ordinary Newton
//! PCG, and `gate_candidate` on the real post-line-search/post-rigidify vertices
//! BEFORE clock or position commit. Missing any of these is an unwired feature.
use ppf_cts_compute::{AllocLabel, Buffer, Device, Fault, Handle, ReadbackBuffer};
use crate::joint_hard_bridge::{self as bridge, Correction, FrozenReducedOperator, PhysicalRow};
use crate::joint_hard_gpu::{HardGeometryGpu, HardJointGpu};
use crate::joint_hard_schur::Error;
use crate::joint_hard_coefficients::{self as precision, FrozenCoefficients, SmoothFactors};
use super::{joints, kernels::JointHardGeometryArgs, operator::Operator, pcg, pdrd,
            rigid_map::RigidMap, scene::{Fatal, FatalResult}};

// Both types are local repr(C) fixed-width POD with asserted sizes/offsets.
// Keep these impls here; do not also add them to joint_hard_gpu.rs.
unsafe impl ppf_cts_compute::Pod for HardJointGpu {}
unsafe impl ppf_cts_compute::Pod for HardGeometryGpu {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family { SingleD6, AngularHinges }

#[derive(Default)]
pub struct Hard {
    family: Option<Family>,
    author: Vec<HardJointGpu>,
    records: Buffer<HardJointGpu>,
    geometry: ReadbackBuffer<HardGeometryGpu>,
    candidate_state: Buffer<crate::data::PdrdRigidState>,
    candidate_fit: Buffer<f32>,
    rhs: Buffer<f32>,
    inverse_out: Buffer<f32>,
    apply_input: Buffer<f32>,
    apply_output: Buffer<f32>,
    apply_absolute: Buffer<f32>,
    apply_curvature: Buffer<f32>,
    corrected: Buffer<f32>,
    frozen_smooth_factors: Buffer<SmoothFactors>,
    rows: Vec<PhysicalRow>,
    zero: Vec<f32>,
    dim: usize,
    bodies: usize,
    frozen_pose: Option<Handle>,
    initial_checked: bool,
    serial: u64,
    root_source:Option<u32>,
    frozen_root:Option<super::roots::HardContext>,
    affine_force:Buffer<f32>,
}

fn fail(code: &str, detail: impl std::fmt::Display) -> Fatal {
    Fatal::invariant(format!("{code}: {detail}"))
}

fn narrow(values: &[f64]) -> Result<Vec<f32>, Error> {
    values.iter().map(|&v| {
        let f=v as f32;
        if !v.is_finite() || !f.is_finite() || (v!=0.0 && f==0.0) {
            Err(Error::Nonfinite)
        } else { Ok(f) }
    }).collect()
}

fn widen(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&v|v as f64).collect()
}

fn transported_row_gate(rows:&[PhysicalRow],actual_float:&[f64])->Result<f64,Error> {
    if actual_float.iter().any(|x|!x.is_finite()){return Err(Error::Nonfinite);}
    let mut maximum=0.0_f64;
    for r in rows {
        if r.derivative.len()!=actual_float.len(){return Err(Error::InvalidInput);}
        let coordinate=r.coordinate-r.derivative.iter().zip(actual_float).map(|(b,x)|b*x).sum::<f64>();
        if !coordinate.is_finite(){return Err(Error::Nonfinite);}
        let error=crate::joint_hard_schur::violation(&r.row,coordinate);
        maximum=maximum.max(error);
        if error>crate::joint_hard_schur::ROW_TOL{return Err(Error::FinalViolation);}
    }
    Ok(maximum)
}

fn relative_l1(actual: &[f64], rhs: &[f64]) -> Result<f64, Error> {
    if actual.len()!=rhs.len() || actual.iter().chain(rhs).any(|v|!v.is_finite()) {
        return Err(Error::Nonfinite);
    }
    let denominator=rhs.iter().map(|v|v.abs()).sum::<f64>();
    let numerator=actual.iter().zip(rhs).map(|(a,b)|(a-b).abs()).sum::<f64>();
    if denominator==0.0 { return if numerator==0.0 {Ok(0.0)} else {Err(Error::InverseFailed)}; }
    let residual=numerator/denominator;
    if !residual.is_finite() || residual>bridge::INVERSE_RELATIVE_L1_TOL {
        eprintln!("HARD_TRUE_RESIDUAL_REFUSED relative={residual:.12e} residual_l1={numerator:.12e} rhs_l1={denominator:.12e} limit={:.12e}",bridge::INVERSE_RELATIVE_L1_TOL);
        Err(Error::InverseFailed)
    } else { Ok(residual) }
}

impl Hard {
    /// No hard author rows means no allocation, fit, geometry or inverse work.
    pub fn active(&self) -> bool {
        self.author.iter().any(|j|j.enabled==1 && j.modes.iter().any(|&m|m!=0))
    }

    /// `records` must already have passed strict wire/topology/frame binding.
    /// Only the sealed single-D6 source-root union has a principal branch.
    /// Additional body projectors remain refused.
    pub fn install<D: Device>(&mut self, device:&mut D, records:&[HardJointGpu],
        family:Family, map:&RigidMap, known_root_count:usize, locked_groups:usize
    ) -> FatalResult<()> {
        if records.is_empty() { *self=Self::default(); return Ok(()); }
        if known_root_count>1 || (known_root_count!=0 && family!=Family::SingleD6) || locked_groups!=0 || map.needs_projection() {
            return Err(fail("G2_HARD_UNSUPPORTED_DOMAIN", "only single-D6/one-source-root and no additional projectors"));
        }
        if (family==Family::SingleD6 && records.len()>1) ||
           (family==Family::AngularHinges && records.len()>2) {
            return Err(fail("G2_HARD_UNSUPPORTED_DOMAIN", "single D6 or at most two hinges"));
        }
        for (index,j) in records.iter().enumerate() {
            if j.enabled>1 || j.body_a as usize>=map.n_bodies ||
               j.body_b as usize>=map.n_bodies || j.body_a==j.body_b ||
               records[..index].iter().any(|other|other.joint_index==j.joint_index) {
                return Err(fail("G2_HARD_INVALID_INPUT","invalid or duplicate hard body binding"));
            }
            for axis in 0..6 {
                if j.modes[axis]>2 || !j.lower[axis].is_finite() || !j.upper[axis].is_finite() ||
                   (j.modes[axis]!=0 && j.lower[axis]>j.upper[axis]) ||
                   (j.modes[axis]==2 && j.lower[axis]!=j.upper[axis]) {
                    return Err(fail("G2_HARD_INVALID_INPUT","invalid physical hard axis"));
                }
            }
            if j.anchor_a.iter().chain(&j.anchor_b).chain(&j.rest_translation_m).any(|v|!v.is_finite()) {
                return Err(fail("G2_HARD_INVALID_INPUT","nonfinite hard geometry"));
            }
            for q in [j.frame_a_wxyz,j.frame_b_wxyz,j.rest_relative_wxyz] {
                let norm=q.iter().map(|&x|(x as f64)*(x as f64)).sum::<f64>().sqrt();
                if !norm.is_finite() || (norm-1.0).abs()>1e-4 {
                    return Err(fail("G2_HARD_INVALID_INPUT","invalid normalized hard frame"));
                }
            }
        }
        self.root_source=if known_root_count==1{Some(records[0].body_a)}else{None};
        self.frozen_root=None;
        if known_root_count==1{self.affine_force.size(device,map.dim,AllocLabel("hard.affine_force"))?;}
        self.family=Some(family);
        self.author=records.to_vec();
        self.dim=map.dim;
        self.bodies=map.n_bodies;
        self.frozen_pose=None;
        self.initial_checked=false;
        self.rows.clear();
        self.records.size(device,records.len(),AllocLabel("hard.records"))?;
        self.records.write(device,0,records)?;
        self.geometry.size(device,records.len(),AllocLabel("hard.geometry"))?;
        if !self.active() { return Ok(()); }
        self.candidate_state.size(device,map.n_bodies,AllocLabel("hard.candidate_state"))?;
        self.candidate_fit.size(device,12*map.n_bodies,AllocLabel("hard.candidate_fit"))?;
        self.rhs.size(device,map.dim,AllocLabel("hard.inverse_rhs"))?;
        self.inverse_out.size(device,map.dim,AllocLabel("hard.inverse_out"))?;
        self.apply_input.size(device,map.dim,AllocLabel("hard.apply_input"))?;
        self.apply_output.size(device,map.dim,AllocLabel("hard.apply_output"))?;
        self.apply_absolute.size(device,map.nrow,AllocLabel("hard.apply_absolute"))?;
        self.apply_curvature.size(device,map.nrow,AllocLabel("hard.apply_curvature"))?;
        self.corrected.size(device,map.dim,AllocLabel("hard.corrected"))?;
        self.zero=vec![0.0; (12*map.n_bodies).max(map.dim)];
        Ok(())
    }

    unsafe fn measure<D: Device>(&mut self, device:&mut D, fitted:Handle, twist:(Handle,u32),
        map:&RigidMap) -> FatalResult<Vec<PhysicalRow>> {
        if map.dim!=self.dim || map.n_bodies!=self.bodies {
            return Err(fail("G2_HARD_INVALID_INPUT", "installed hard reduction changed dimensions"));
        }
        let args=JointHardGeometryArgs {records:self.records.handle(),twist_context:twist.0,twist_count:twist.1,state:fitted,
            bodies:map.n_bodies as u32,output:self.geometry.handle(),
            count:self.author.len() as u32,seam_arena_count:0};
        device.launch("hard.geometry",&args,self.author.len() as u32)?;
        self.geometry.download(device)?;
        if let Some(g)=self.geometry.host().iter().find(|g|g.fault!=0) {
            return Err(fail("G2_HARD_GEOMETRY_FAULT",
                format!("joint {} geometry fault {}",g.joint_index,g.fault)));
        }
        bridge::expand(&self.author,self.geometry.host(),map.body_base,map.dim,
            self.family==Some(Family::AngularHinges))
            .map_err(|e|fail(e.code(),"expanding measured hard rows"))
    }

    /// CURRENT rigid fit only. Does not refit or touch rotated-rest/operator
    /// buffers while frozen. The serial belongs to this exclusive mutable borrow.
    pub unsafe fn freeze<D:Device>(&mut self,device:&mut D,fitted:Handle,twist:(Handle,u32),
        map:&RigidMap) -> FatalResult<()> {
        if !self.active() {return Ok(());}
        if !self.initial_checked {
            return Err(fail("G2_HARD_INITIAL_GATE_MISSING","actual initial/resume pose was not checked"));
        }
        self.frozen_root=None;
        self.rows=self.measure(device,fitted,twist,map)?;
        self.serial=self.serial.checked_add(1)
            .ok_or_else(||fail("G2_HARD_INVALID_INPUT","frozen serial exhausted"))?;
        self.frozen_pose=Some(fitted);
        Ok(())
    }

    /// Fits the ACTUAL passed vertices into separate candidate buffers. This
    /// does not reuse the frozen Newton pose or a Schur predicted coordinate.
    pub unsafe fn gate_candidate<D:Device>(&mut self,device:&mut D,
        reduction:&mut pdrd::Reduction<'_>, vertex_prop:Handle,positions:Handle,
        body_vertices:u32, initial:bool, twist:(Handle,u32)) -> FatalResult<()> {
        if !self.active() {return Ok(());}
        let vert_list=reduction.staged.vert_list.handle();
        let body_prop=reduction.staged.body_prop.handle();
        let rest=reduction.staged.rest_centered.handle();
        reduction.fit(device,vert_list,vertex_prop,body_prop,rest,positions,
            self.candidate_fit.handle(),self.candidate_state.handle(),body_vertices,&self.zero)?;
        let rows=self.measure(device,self.candidate_state.handle(),twist,reduction.map)?;
        let result=if initial {bridge::initial_gate(&rows)} else {
            let actual:Vec<f64>=rows.iter().map(|r|r.coordinate).collect();
            bridge::final_gate(&rows,&actual)
        };
        self.frozen_pose=None;
        result.map_err(|e|fail(e.code(),"actual nonlinear candidate geometry"))?;
        if initial {self.initial_checked=true;}
        Ok(())
    }

    /// Correct the coupled Newton increment in-place, preserving/restoring the
    /// original dx and rotation export if any inverse, Schur, upload or export
    /// fails. On success both dx AND rotation are exported from the corrected
    /// FULL reduced vector. No pose, time, contact or accepted-state write occurs.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn correct<D:Device>(&mut self,device:&mut D,op:&Operator,
        reduction:&mut pdrd::Reduction<'_>,factor:Handle,inverse_diagonal:Handle,
        dx:&mut Buffer<f32>,rotation:&mut Buffer<f32>,work:&mut pcg::Workspace,
        rigid:&mut pcg::RigidWorkspace,view:joints::View,newton:&pcg::Report,
        inverse_tolerance:f32,max_iterations:u32,known_root_count:usize,
        locked_groups:usize) -> FatalResult<Option<Correction>> {
        if !self.active() {return Ok(None);}
        if known_root_count!=0 || locked_groups!=0 || reduction.map.needs_projection() {
            return Err(fail("G2_HARD_UNSUPPORTED_DOMAIN","rootless hard inverse requires no extra projector"));
        }
        if self.frozen_pose!=Some(view.state) || view.dim as usize!=self.dim ||
           view.base as usize!=reduction.map.body_base || view.bodies as usize!=self.bodies {
            return Err(fail("G2_HARD_FROZEN_OPERATOR_CHANGED","joint view differs from frozen hard geometry"));
        }
        if !matches!(newton.outcome,pcg::Outcome::Converged) {
            return Err(fail("G2_HARD_INVERSE_FAILED","ordinary Newton solve did not converge"));
        }
        if !inverse_tolerance.is_finite() || inverse_tolerance<=0.0 || max_iterations==0 {
            return Err(fail("G2_HARD_INVALID_INPUT","inverse solver configuration"));
        }
        let mut original_dx=vec![0.0;3*reduction.map.nrow];
        let mut original_rotation=vec![0.0;3*self.bodies];
        let mut original_u0=vec![0.0;self.dim];
        let mut original_force=vec![0.0;self.dim];
        dx.read(device,0,&mut original_dx)?;
        rotation.read(device,0,&mut original_rotation)?;
        rigid.xr.read(device,0,&mut original_u0)?;
        rigid.f.read(device,0,&mut original_force)?;
        let rows=self.rows.clone();
        let stamp=self.serial;
        let result={
            let coefficients=unsafe {FrozenCoefficients::capture(device,op,reduction,view,&mut self.frozen_smooth_factors)}?;
            let mut adapter=Frozen {device,op,reduction,factor,inverse_diagonal,
                work,rigid,view,rhs:&mut self.rhs,inverse_out:&mut self.inverse_out,
                apply_input:&mut self.apply_input,apply_output:&mut self.apply_output,
                apply_absolute:&mut self.apply_absolute,apply_curvature:&mut self.apply_curvature,
                coefficients:&coefficients,inverse_columns:0,stamp,tolerance:inverse_tolerance,max_iterations,last_fault:None};
            let result=(|| {
                // Ordinary PCG remains the initial approximation; refine it
                // against the same frozen coefficient mathematics in FP64.
                // Keep u0 in FP64 until final publication, preserving strict G.
                let u0=adapter.refine(&widen(&original_force),&widen(&original_u0),"ordinary_u0")?;
                // u0 is used ONLY in FP64 by Schur. Audit its coefficient64
                // residual and the real GPU cast (with its input cast bound).
                // Its never-published float cast has a separate diagnostic,
                // not an extra mathematical gate on an unused physics vector.
                adapter.certify_seed(&u0,&widen(&original_force))?;
                let mut correction=bridge::constrain_literal(&mut adapter,&rows,&u0,&vec![false;self.dim])?;
                // Check the actual f32 vector we will publish, including its
                // constrained effort. This also catches transport roundoff.
                let expected:Vec<f64>=original_force.iter().zip(&correction.hard_covector)
                    .map(|(&f,&c)|f as f64+c).collect();
                // The returned vector is the actual float vector subsequently
                // uploaded/prolonged/extracted, not an unused audit candidate.
                correction.increment=adapter.certify_transport(&correction.increment,&expected,"corrected_publication")
                    .map_err(|error| {
                        if std::env::var_os("PPF_HARD_COEFFICIENT_CAPTURE").is_some() {
                            let physical:Vec<_>=rows.iter().map(|r|serde_json::json!({"owner":r.row.owner,"axis":r.row.axis,"mode":match r.row.mode {crate::joint_hard_schur::Mode::Free=>0,crate::joint_hard_schur::Mode::Limited=>1,crate::joint_hard_schur::Mode::Locked=>2},"lower":r.row.lower,"upper":r.row.upper,"coordinate":r.coordinate,"derivative":r.derivative})).collect();
                            eprintln!("HARD_PUBLICATION_FAILURE_CONTEXT_JSON {}",serde_json::json!({"kind":"hard_publication_failure_context","stamp":stamp,"stage":"corrected_publication","error":error.code(),"high64":correction.increment,"ordinary_u0":u0,"original_force":original_force,"expected_rhs":expected,"hard_covector":correction.hard_covector,"lambda":correction.schur.lambda,"trial64":correction.schur.trial,"active_sides":format!("{:?}",correction.schur.side),"rows":physical,"row_tolerance":crate::joint_hard_schur::ROW_TOL,"scope":"same frozen capture and immediately preceding publication failure record; no vector published"}));
                        }
                        error
                    })?;
                let row_max=transported_row_gate(&rows,&correction.increment)?;
                eprintln!("HARD_LINEAR_PUBLICATION_ROWS stamp={stamp} status=PASSED actual_float_row_violation_max={row_max:.12e} row_limit={:.12e} rows={} scope=coordinate_minus_B_dot_actual_uploaded_float_distinct_from_nonlinear_geometry",crate::joint_hard_schur::ROW_TOL,rows.len());
                if adapter.stamp()!=stamp {return Err(Error::FrozenOperatorChanged);}
                Ok(correction)
            })();
            result.map_err(|e|fail(e.code(),adapter.last_fault.as_deref()
                .unwrap_or("hard inverse/true residual/Schur refused")))
        };
        let publication=(|| -> FatalResult<Correction> {
            let correction=result?;
            let corrected=narrow(&correction.increment)
                .map_err(|e|fail(e.code(),"corrected reduced vector f32 transport"))?;
            self.corrected.write(device,0,&corrected)?;
            reduction.prolong(device,self.corrected.handle(),dx.handle())?;
            reduction.extract_body_rotation(device,self.corrected.handle(),rotation.handle())?;
            // xr must contain the published solution, f the original RHS. The
            // inverse loop reuses both; its scratch is never the next iterate.
            rigid.xr.write(device,0,&corrected)?;
            rigid.f.write(device,0,&original_force)?;
            Ok(correction)
        })();
        match publication {
            Ok(correction)=>Ok(Some(correction)),
            Err(error)=>{
                // Restore externally consumed Newton solution/rotation scratch
                // even if the pure-RHS implementation gains an export later.
                // Attempt EVERY restoration even if the first upload fails,
                // so a failed dx upload cannot skip the rotation restoration.
                let restores=[dx.write(device,0,&original_dx),
                    rotation.write(device,0,&original_rotation),
                    rigid.xr.write(device,0,&original_u0),
                    rigid.f.write(device,0,&original_force)];
                if let Some(restore)=restores.into_iter().find_map(Result::err) {
                    return Err(fail("G2_HARD_RESTORE_FAILED",format!("{}; {restore}",error.detail)));
                }
                Err(error)
            }
        }
    }
}

/// Exclusive callback over exactly the operator/reduction/smooth-joint view
/// borrowed by correct(). No assembly, pose fit or preconditioner rebuild is
/// callable through this adapter. Known/aggregate projected scenes are refused.
struct Frozen<'a,'map,D:Device> {
    device:&'a mut D, op:&'a Operator<'a>, reduction:&'a mut pdrd::Reduction<'map>,
    factor:Handle,inverse_diagonal:Handle,work:&'a mut pcg::Workspace,
    rigid:&'a mut pcg::RigidWorkspace,view:joints::View,
    rhs:&'a mut Buffer<f32>,inverse_out:&'a mut Buffer<f32>,
    apply_input:&'a mut Buffer<f32>,apply_output:&'a mut Buffer<f32>,
    apply_absolute:&'a mut Buffer<f32>,apply_curvature:&'a mut Buffer<f32>,
    coefficients:&'a FrozenCoefficients,
    inverse_columns:u32,
    stamp:u64,tolerance:f32,max_iterations:u32,last_fault:Option<String>,
}

impl<D:Device> Frozen<'_,'_,D> {
    fn certify_seed(&mut self,x:&[f64],rhs:&[f64])->Result<(),Error> {
        let math=self.coefficients.evaluate(x).map_err(|e|{self.last_fault=Some(e);Error::InverseFailed})?;
        let seed_residual=math.mathematical_relative_l1(rhs).map_err(|e|{self.last_fault=Some(e);Error::InverseFailed})?;
        if !seed_residual.is_finite()||seed_residual>bridge::INVERSE_RELATIVE_L1_TOL {self.last_fault=Some(format!("ordinary_u0 actual FP64 seed residual {seed_residual:.12e}"));return Err(Error::InverseFailed);}
        let transported=widen(&narrow(x)?);let actual=self.raw_apply(&transported)?;
        let cast_math=self.coefficients.evaluate(&transported)
            .map_err(|e|{self.last_fault=Some(e);Error::InverseFailed})?;
        let unused=cast_math.mathematical_relative_l1(rhs).map_err(|e|{self.last_fault=Some(e);Error::InverseFailed})?;
        let ratio=cast_math.certify_roundoff(&actual).map_err(|e|{self.last_fault=Some(e);Error::InverseFailed})?;
        let norm=rhs.iter().map(|x|x.abs()).sum::<f64>();let raw=actual.iter().zip(rhs).map(|(a,b)|(a-b).abs()).sum::<f64>();
        let raw_relative=if norm==0.0 {if raw==0.0 {0.0}else{f64::INFINITY}}else{raw/norm};
        let status=if raw_relative<=bridge::INVERSE_RELATIVE_L1_TOL {"PASSED_RAW_AND_COEFFICIENT64"}else{"FAILED_RAW_CERTIFIED_ROUNDOFF"};
        eprintln!("HARD_PRECISION_CERTIFICATE stage=ordinary_u0 stamp={} status={status} raw_relative_l1={raw_relative:.12e} mathematical_relative_l1={seed_residual:.12e} component_bound_max_ratio={ratio:.12e} residual_limit={:.12e} unused_float_coefficient_relative_l1={unused:.12e} unused_float_mathematical_gate={} scope=actual_FP64_Schur_seed_real_GPU_cast_audit_float_never_published",self.stamp,bridge::INVERSE_RELATIVE_L1_TOL,if unused<=bridge::INVERSE_RELATIVE_L1_TOL {"PASSED"}else{"FAILED_UNUSED_CAST"});
        if std::env::var_os("PPF_HARD_COEFFICIENT_CAPTURE").is_some() {
            eprintln!("HARD_PRECISION_EVALUATION_JSON {}",serde_json::json!({"kind":"hard_precision_seed_evaluation","stage":"ordinary_u0","stamp":self.stamp,"evaluated_fp64_seed":x,"transported_never_published":transported,"rhs":rhs,"gpu_apply":actual,"seed_coefficient64_apply":math.value,"cast_coefficient64_apply":cast_math.value,"cast_float_component_error_bound":cast_math.float_error,"cast_double_component_error_bound":cast_math.double_error,"unused_float_coefficient_relative_l1":unused}));
        }
        Ok(())
    }
    fn inverse_once(&mut self,input:&[f64])->Result<(Vec<f64>,pcg::Report),Error> {
        let scale=input.iter().map(|x|x.abs()).fold(0.0,f64::max);
        if !scale.is_finite(){return Err(Error::Nonfinite);}
        if scale==0.0 {return Err(Error::InvalidInput);}
        // Normalize tiny refinement residuals before float transport; the
        // pure-RHS solve is linear and the full response is rescaled in FP64.
        let normalized:Vec<_>=input.iter().map(|x|x/scale).collect();
        let values=narrow(&normalized)?;
        let result=(|| -> FatalResult<(Vec<f64>,pcg::Report)> {unsafe {
            self.rhs.write(self.device,0,&values)?;
            let report=pcg::solve_rigid_reduced_rhs(self.device,self.op,self.reduction,
                self.factor,self.inverse_diagonal,self.rhs.handle(),self.inverse_out.handle(),
                self.work,self.rigid,None,self.view,self.tolerance,self.max_iterations)?;
            if !matches!(report.outcome,pcg::Outcome::Converged) {
                return Err(fail("G2_HARD_INVERSE_FAILED","pure RHS solve did not converge"));
            }
            let mut host=vec![0.0;self.reduction.map.dim];
            self.inverse_out.read(self.device,0,&mut host)?;
            Ok((host.iter().map(|&x|x as f64*scale).collect(),report))
        }})();
        result.map_err(|e| {self.last_fault=Some(e.detail);Error::InverseFailed})
    }
    fn raw_apply(&mut self,input:&[f64])->Result<Vec<f64>,Error> {
        let values=narrow(input)?;
        let result=(|| -> FatalResult<Vec<f64>> {unsafe {
            self.apply_input.write(self.device,0,&values)?;
            self.reduction.project_bodies(self.device,self.apply_input.handle())?;
            self.reduction.prolong(self.device,self.apply_input.handle(),self.rigid.xv.handle())?;
            self.op.apply(self.device,self.rigid.xv.handle(),self.rigid.mxv.handle(),
                self.apply_absolute.handle(),self.apply_curvature.handle())?;
            self.reduction.restrict(self.device,self.rigid.mxv.handle(),
                self.apply_output.handle(),&self.rigid.zero)?;
            self.view.add(self.device,1,self.apply_input.handle(),self.apply_output.handle())?;
            self.reduction.project_bodies(self.device,self.apply_output.handle())?;
            let mut host=vec![0.0;self.reduction.map.dim];
            self.apply_output.read(self.device,0,&mut host)?;
            Ok(widen(&host))
        }})();
        result.map_err(|e| {self.last_fault=Some(e.detail);Error::InverseFailed})
    }
    fn refine(&mut self,rhs:&[f64],initial:&[f64],context:&str)->Result<Vec<f64>,Error> {
        let coefficients=self.coefficients;
        precision::refine(coefficients,rhs,initial,bridge::INVERSE_RELATIVE_L1_TOL,context,|residual| {
            self.inverse_once(residual).map(|r|r.0).map_err(|e|e.code().to_string())
        }).map_err(|e|{self.last_fault=Some(e);Error::InverseFailed})
    }
    fn certify_transport(&mut self,x:&[f64],rhs:&[f64],stage:&str)->Result<Vec<f64>,Error> {
        let selected=self.coefficients.certify_nearest_publication(x,rhs,bridge::INVERSE_RELATIVE_L1_TOL)
            .map_err(|e|{self.last_fault=Some(format!("{stage}: {e}"));Error::InverseFailed})?;
        let math=self.coefficients.evaluate(&selected.values).map_err(|e|{self.last_fault=Some(e);Error::InverseFailed})?;
        let actual=self.raw_apply(&selected.values)?;
        let ratio=math.certify_roundoff(&actual).map_err(|e|{self.last_fault=Some(e);Error::InverseFailed})?;
        let norm=rhs.iter().map(|x|x.abs()).sum::<f64>();let raw_sum=actual.iter().zip(rhs).map(|(a,b)|(a-b).abs()).sum::<f64>();
        let raw=if norm==0. {if raw_sum==0. {0.}else{f64::INFINITY}}else{raw_sum/norm};
        let raw_upper=precision::Evaluation{value:actual.clone(),float_error:vec![0.;actual.len()],double_error:vec![0.;actual.len()]}
            .mathematical_relative_l1(rhs).map_err(|e|{self.last_fault=Some(e);Error::InverseFailed})?;
        if !raw_upper.is_finite()||raw_upper>precision::PUBLICATION_FORCE_QUALITY_CEILING {
            self.last_fault=Some(format!("actual GPU force residual upper {raw_upper:.12e} exceeds fixed publication quality ceiling"));return Err(Error::InverseFailed);
        }
        eprintln!("HARD_LITERAL_GPU_FORCE_QUALITY stage={stage} stamp={} status=PASSED actual_relative_l1={raw:.12e} conservative_relative_l1_upper={raw_upper:.12e} fixed_ceiling={:.12e}",self.stamp,precision::PUBLICATION_FORCE_QUALITY_CEILING);
        eprintln!("HARD_LITERAL_PUBLICATION_CONTRACT stage={stage} stamp={} status=PASSED_LITERAL_NEAREST_PRECISION_ADAPTER used_fp64_equilibrium={:.12e} original_math_limit={:.12e} legacy_float_math_status={} legacy_float_math_residual={:.12e} legacy_raw_gpu_status={} legacy_raw_gpu_residual={raw:.12e} actual_nearest_cast_envelope_relative={:.12e} worst_budget_diagnostic_relative={:.12e} force_quality_ceiling={:.12e} vertex_cast_bound_m={:.12e} rotation_cast_bound_rad={:.12e} physical_cast_caps=1e-8 component_gpu_bound_ratio={ratio:.12e} scope=fixed_nearest_signed_cast_action_with_evaluation_and_subtraction_bounds_distinct_legacy_gate_status",self.stamp,selected.fp64_relative_l1,bridge::INVERSE_RELATIVE_L1_TOL,if selected.float_relative_l1<=bridge::INVERSE_RELATIVE_L1_TOL{"PASSED"}else{"FAILED_LEGACY_FLOAT_MATH"},selected.float_relative_l1,if raw<=bridge::INVERSE_RELATIVE_L1_TOL{"PASSED"}else{"FAILED_RAW_CERTIFIED_ROUNDOFF"},selected.envelope_relative_l1,selected.worst_budget_envelope_relative_l1,precision::PUBLICATION_FORCE_QUALITY_CEILING,selected.vertex_cast_bound_max,selected.rotation_cast_bound_max);
        if std::env::var_os("PPF_HARD_COEFFICIENT_CAPTURE").is_some(){eprintln!("HARD_LITERAL_PUBLICATION_JSON {}",serde_json::json!({"stage":stage,"stamp":self.stamp,"high64":x,"nearest_float":selected.values,"actual_delta":selected.deltas,"predeclared_lane_rounding_budgets":selected.lane_budgets,"worst_budget_absolute_factor_propagation_component_bounds":selected.propagated_component_bounds,"actual_signed_cast_action_component_bounds":selected.cast_action_component_bounds,"delta_coefficient64_apply":selected.delta_eval_value,"delta_double_evaluation_bounds":selected.delta_eval_error,"delta_subtraction_bounds":selected.delta_subtraction_bounds,"delta_subtraction_absolute_propagation":selected.delta_subtraction_propagated,"signed_cast_identity_component_bounds":selected.cast_identity_component_bounds,"signed_cast_identity_mismatches":selected.cast_identity_mismatches,"signed_cast_identity_maximum_bound_ratio":selected.cast_identity_maximum_bound_ratio,"worst_budget_diagnostic_status":if selected.worst_budget_envelope_relative_l1<=precision::PUBLICATION_FORCE_QUALITY_CEILING{"PASSED_WORST_CASE_ALLOWANCE_ENVELOPE"}else{"FAILED_WORST_CASE_ALLOWANCE_ENVELOPE"},"worst_budget_envelope_relative_l1":selected.worst_budget_envelope_relative_l1,"rhs":rhs,"coefficient64_apply":math.value,"double_eval_bound":math.double_error,"gpu_apply":actual,"gpu_float_component_bounds":math.float_error,"used_fp64_relative_l1":selected.fp64_relative_l1,"legacy_float_relative_l1":selected.float_relative_l1,"legacy_raw_relative_l1":raw,"envelope_relative_l1":selected.envelope_relative_l1,"force_quality_ceiling":precision::PUBLICATION_FORCE_QUALITY_CEILING,"scope":"fixed nearest signed literal action with evaluation and subtraction bounds; worst-budget diagnostic only; old gates independently recorded; actual rows and nonlinear geometry follow"}));}
        eprintln!("HARD_SIGNED_CAST_IDENTITY stage={stage} stamp={} status=PASSED maximum_component_bound_ratio={:.12e} worst_budget_diagnostic_status={} worst_budget_diagnostic_relative={:.12e} fixed_quality_ceiling={:.12e}",self.stamp,selected.cast_identity_maximum_bound_ratio,if selected.worst_budget_envelope_relative_l1<=precision::PUBLICATION_FORCE_QUALITY_CEILING{"PASSED_WORST_CASE_ALLOWANCE_ENVELOPE"}else{"FAILED_WORST_CASE_ALLOWANCE_ENVELOPE"},selected.worst_budget_envelope_relative_l1,precision::PUBLICATION_FORCE_QUALITY_CEILING);
        Ok(selected.values)
    }

}

impl<D:Device> FrozenReducedOperator for Frozen<'_,'_,D> {
    fn audit_literal_gram(&mut self,b:&[Vec<f64>],y:&[Vec<f64>],g:&[Vec<f64>])->Result<(),Error>{
        self.coefficients.audit_literal_gram(b,y,g).map_err(|e|{self.last_fault=Some(e);Error::InverseFailed})
    }

    fn stamp(&self)->u64 {self.stamp}
    fn apply(&mut self,input:&[f64])->Result<Vec<f64>,Error> {
        self.coefficients.evaluate(input).map(|e|e.value)
            .map_err(|e|{self.last_fault=Some(e);Error::InverseFailed})
    }
    fn inverse(&mut self,input:&[f64])->Result<Vec<f64>,Error> {
        // No float transported basis probes and no dense matrix substitution.
        let column=self.inverse_columns;
        self.inverse_columns=self.inverse_columns.checked_add(1).ok_or(Error::InvalidInput)?;
        self.refine(input,&vec![0.0;input.len()],&format!("inverse_column_{column}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn residual_gate_is_relative_finite_and_refuses_false_convergence() {
        assert_eq!(relative_l1(&[0.0,0.0],&[0.0,0.0]),Ok(0.0));
        assert_eq!(relative_l1(&[1.0,2.0],&[1.0,2.0]),Ok(0.0));
        assert_eq!(relative_l1(&[1.01,2.0],&[1.0,2.0]),Err(Error::InverseFailed));
        assert_eq!(relative_l1(&[f64::NAN],&[1.0]),Err(Error::Nonfinite));
    }
    #[test]
    fn float_transport_refuses_overflow_underflow_and_nonfinite() {
        assert!(narrow(&[f64::MAX]).is_err());
        assert!(narrow(&[f64::MIN_POSITIVE]).is_err());
        assert!(narrow(&[f64::NAN]).is_err());
        assert_eq!(narrow(&[0.0,0.5]).unwrap(),vec![0.0,0.5]);
    }
}

include!("hard_known_root.fragment.rs");

