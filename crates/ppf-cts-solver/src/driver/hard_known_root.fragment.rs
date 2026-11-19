// Isolated known-source-root branch; rootless Hard::correct is unchanged.
impl Hard {
    pub unsafe fn freeze_known_root<D:Device>(&mut self,device:&mut D,fitted:Handle,twist:(Handle,u32),map:&RigidMap,context:super::roots::HardContext)->FatalResult<()>{
        if !self.active(){return Ok(());}
        if self.family!=Some(Family::SingleD6)||self.author.len()!=1||self.root_source!=Some(context.body)||context.body as usize>=map.n_bodies{
            return Err(fail("ROOT_HARD_SOURCE","frozen root differs from sealed single-D6 source"));
        }
        self.freeze(device,fitted,twist,map)?;self.frozen_root=Some(context);Ok(())
    }
    /// Separate read-only RHS snapshot for the root reaction. The official
    /// Newton solve subsequently builds its own f once. No inverse callback
    /// adds this gradient or snapshot again.
    pub unsafe fn capture_affine_force<D:Device>(&mut self,device:&mut D,reduction:&mut pdrd::Reduction<'_>,force:Handle,view:joints::View)->FatalResult<()>{
        if self.frozen_root.is_none(){return Err(fail("ROOT_HARD_FROZEN","root force captured without frozen root"));}
        reduction.restrict(device,force,self.affine_force.handle(),&self.zero)?;
        view.add(device,0,self.affine_force.handle(),self.affine_force.handle())?;
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn correct_known_root<D:Device>(&mut self,device:&mut D,op:&Operator,reduction:&mut pdrd::Reduction<'_>,factor:Handle,inverse_diagonal:Handle,
        dx:&mut Buffer<f32>,rotation:&mut Buffer<f32>,work:&mut pcg::Workspace,rigid:&mut pcg::RigidWorkspace,view:joints::View,root_view:super::roots::View,context:super::roots::HardContext,
        newton:&pcg::Report,tolerance:f32,max_iterations:u32,locked_groups:usize)->FatalResult<Option<Correction>>{
        use crate::joint_root_hard_affine::{self as affine,RootAffine,Epoch};
        use crate::joint_root_hard_precision::{CoefficientPrincipal,certify_affine_publication};
        if !self.active(){return Ok(None);}
        if self.family!=Some(Family::SingleD6)||self.root_source!=Some(context.body)||self.frozen_root!=Some(context)||self.frozen_pose!=Some(view.state)
            ||view.dim as usize!=self.dim||view.base as usize!=reduction.map.body_base||view.bodies as usize!=self.bodies||root_view.dim!=view.dim||root_view.base!=view.base||root_view.bodies!=view.bodies||root_view.count!=1
            ||locked_groups!=0||reduction.map.needs_projection(){return Err(fail("ROOT_HARD_FROZEN","root/context/operator or unsupported projector changed"));}
        if !matches!(newton.outcome,pcg::Outcome::Converged)||!tolerance.is_finite()||tolerance<=0.||max_iterations==0{return Err(fail("G2_HARD_INVERSE_FAILED","root Newton/configuration unconverged"));}
        let mut original_dx=vec![0.;3*reduction.map.nrow];let mut original_rotation=vec![0.;3*self.bodies];let mut original_u0=vec![0.;self.dim];let mut original_force=vec![0.;self.dim];let mut full_force=vec![0.;self.dim];
        dx.read(device,0,&mut original_dx)?;rotation.read(device,0,&mut original_rotation)?;rigid.xr.read(device,0,&mut original_u0)?;rigid.f.read(device,0,&mut original_force)?;self.affine_force.read(device,0,&mut full_force)?;
        let rows=self.rows.clone();let stamp=self.serial;
        let result=(||->FatalResult<Correction>{
            let coefficients=FrozenCoefficients::capture(device,op,reduction,view,&mut self.frozen_smooth_factors)?;
            let root=RootAffine::one_root(self.dim,reduction.map.body_base,self.bodies,context.body as usize,&widen(&original_u0),Epoch{operator:stamp,root_transaction:context.generation,control_mask:context.sequence})
                .map_err(|e|fail(e.code(),"production prescribed f32 bytes"))?;
            let force:Vec<_>=original_force.iter().zip(&full_force).zip(&root.known).map(|((&f,&full),&known)|if known{full as f64}else{f as f64}).collect();
            let inverse_known=root.known.clone();
            let approximate=|rhs:&[f64]|super::root_hard_device::inverse_once(device,op,reduction,factor,inverse_diagonal,work,rigid,view,root_view,&root,&mut self.rhs,&mut self.inverse_out,rhs,tolerance,max_iterations).map_err(|e|e.detail);
            let mut backend=CoefficientPrincipal{snapshot:&coefficients,epoch:root.epoch,known:inverse_known,approximate};
            let u0=backend.affine_seed(&root,&force,&widen(&original_u0)).map_err(|e|fail(e.code(),"known-root affine seed Q(f-Kp)"))?;
            let mut correction=affine::correct(&root,&mut backend,&rows,&u0).map_err(|e|fail(e.code(),"known-root principal Schur"))?;
            drop(backend);
            if std::env::var_os("PPF_HARD_COEFFICIENT_CAPTURE").is_some(){eprintln!("ROOT_HARD_AFFINE_INPUT_JSON {}",serde_json::json!({"stamp":stamp,"generation":context.generation,"sequence":context.sequence,"source":context.body,"known":root.known,"particular":root.particular,"ordinary_float":original_u0,"refined_seed64":u0,"corrected_high64":correction.increment,"nearest_float":correction.increment.iter().enumerate().map(|(k,&v)|if root.known[k]{root.particular[k]}else{v as f32 as f64}).collect::<Vec<_>>(),"force":force,"lambda":correction.schur.lambda,"rows":rows.iter().map(|r|serde_json::json!({"owner":r.row.owner,"axis":r.row.axis,"mode":format!("{:?}",r.row.mode),"lower":r.row.lower,"upper":r.row.upper,"coordinate":r.coordinate,"derivative":r.derivative})).collect::<Vec<_>>(),"scope":"prepublication same frozen known-source principal affine state"}));}
            let publication=certify_affine_publication(&coefficients,&root,&correction.increment,&force,&rows,&correction.schur.lambda,
                |q|super::root_hard_device::apply_actual_q(device,op,reduction,rigid,view,root_view,&root,&mut self.apply_input,&mut self.apply_output,self.apply_absolute.handle(),self.apply_curvature.handle(),q).map_err(|e|e.detail))
                .map_err(|e|fail("ROOT_HARD_PUBLICATION",e))?;
            eprintln!("ROOT_HARD_AFFINE_CONTRACT stamp={stamp} generation={} sequence={} source={} high_relative_upper={:.12e} signed_cast_quality_upper={:.12e} sameq_raw_upper={:.12e} component_ratio={:.12e} signed_identity_ratio={:.12e} h_lower={:.12e} hard_lower={:.12e} vertex_cap={:.12e} angle_cap={:.12e} prescribed_slot_exact=true root_reaction={:?} scope=V7_single_D6_one_source_root",context.generation,context.sequence,context.body,publication.high_equilibrium_relative_upper,publication.signed_cast_quality_relative_upper,publication.raw_gpu_relative_upper,publication.gpu_component_bound_ratio,publication.signed_identity_maximum_ratio,publication.normalizer_h_lower,publication.normalizer_hard_lower,publication.vertex_budget_max,publication.rotation_budget_max,publication.root_reaction);
            eprintln!("ROOT_HARD_AFFINE_LEGACY_AUDIT actual_float_math_relative_upper={:.12e} actual_float_math_1e7_status={} sameq_raw_1e7_status={} scope=independent_audit_not_inherited_rootless_PASS",publication.actual_float_equilibrium_relative_upper,if publication.actual_float_equilibrium_relative_upper<=bridge::INVERSE_RELATIVE_L1_TOL{"PASSED"}else{"FAILED_LITERAL_PRECISION_ADAPTER"},if publication.raw_gpu_relative_upper<=bridge::INVERSE_RELATIVE_L1_TOL{"PASSED"}else{"FAILED_LITERAL_PRECISION_ADAPTER"});
            if std::env::var_os("PPF_HARD_COEFFICIENT_CAPTURE").is_some(){eprintln!("ROOT_HARD_AFFINE_PUBLICATION_JSON {}",serde_json::json!({"stamp":stamp,"generation":context.generation,"sequence":context.sequence,"source":context.body,"known":root.known,"particular":root.particular,"high64":correction.increment,"actual_float":publication.values,"actual_delta":publication.actual_delta,"lane_budgets":publication.lane_budgets,"delta_subtraction_bounds":publication.delta_subtraction_bounds,"cast_action_bounds":publication.cast_action_component_bounds,"root_reaction":publication.root_reaction,"denominator_h_lower":publication.normalizer_h_lower,"denominator_hard_lower":publication.normalizer_hard_lower,"force":force,"hard_covector":correction.hard_covector,"lambda":correction.schur.lambda,"row_limit":crate::joint_hard_schur::ROW_TOL,"scope":"full same-q principal affine contract; known reactions are not free-equilibrium residual"}));}
            correction.increment=publication.values;
            transported_row_gate(&rows,&correction.increment).map_err(|e|fail(e.code(),"known root actual uploaded rows"))?;
            Ok(correction)
        })();
        let publication=(||->FatalResult<Correction>{
            let correction=result?;let selected=narrow(&correction.increment).map_err(|e|fail(e.code(),"known-root float transport"))?;
            self.corrected.write(device,0,&selected)?;reduction.prolong(device,self.corrected.handle(),dx.handle())?;reduction.extract_body_rotation(device,self.corrected.handle(),rotation.handle())?;
            rigid.xr.write(device,0,&selected)?;rigid.f.write(device,0,&original_force)?;Ok(correction)
        })();
        match publication{Ok(c)=>Ok(Some(c)),Err(e)=>{let restores=[dx.write(device,0,&original_dx),rotation.write(device,0,&original_rotation),rigid.xr.write(device,0,&original_u0),rigid.f.write(device,0,&original_force)];if let Some(restore)=restores.into_iter().find_map(Result::err){return Err(fail("G2_HARD_RESTORE_FAILED",format!("{}; {restore}",e.detail)));}Err(e)}}
    }
}

