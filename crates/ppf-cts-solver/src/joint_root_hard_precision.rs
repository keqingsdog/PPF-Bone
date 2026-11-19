//! Frozen-coefficient principal callback for the exact sealed known-source
//! root/D6 domain. No GPU basis reconstruction or material re-evaluation.
use crate::joint_hard_coefficients::{self as precision,FrozenCoefficients,Evaluation};
use crate::joint_hard_schur::Error;
use crate::joint_root_hard_affine::{Epoch,PrincipalBackend,RootAffine};
pub fn project_evaluation(mut e:Evaluation,known:&[bool])->Result<Evaluation,String>{
    if e.value.len()!=known.len(){return Err("root coefficient mask extent".into());}
    for k in 0..known.len(){if known[k]{e.value[k]=0.;e.double_error[k]=0.;e.float_error[k]=0.;}}
    Ok(e)
}
fn up(x:f64)->f64{if x>0.&&x.is_finite(){f64::from_bits(x.to_bits()+1)}else{x}}
fn sum(a:f64,b:f64)->f64{up(a+b)}
fn down(x:f64)->f64{if x>0.&&x.is_finite(){f64::from_bits(x.to_bits()-1)}else{x}}
fn sub_bound(a:f64,b:f64)->f64{if a==b{0.}else{sum(up(2f64.powi(-53)*sum(a.abs(),b.abs())),f64::MIN_POSITIVE)}}

pub struct AffinePublication {
    pub values:Vec<f64>,pub known_preserved_exactly:bool,
    pub normalizer_h_lower:f64,pub normalizer_hard_lower:f64,
    pub high_equilibrium_relative_upper:f64,pub actual_float_equilibrium_relative_upper:f64,pub signed_cast_quality_relative_upper:f64,
    pub raw_gpu_relative_upper:f64,pub gpu_component_bound_ratio:f64,
    pub signed_identity_maximum_ratio:f64,pub root_reaction:Vec<f64>,
    pub lane_budgets:Vec<f64>,pub actual_delta:Vec<f64>,pub delta_subtraction_bounds:Vec<f64>,
    pub cast_action_component_bounds:Vec<f64>,pub vertex_budget_max:f64,pub rotation_budget_max:f64,
}

/// Isolated complete publication contract for a known root. `actual_gpu_q`
/// must come from a real full K apply of the SAME q (root slots included),
/// followed by Q on its output. It is never a K(q-p) replacement probe.
/// This is the separate sealed source-root branch. No rootless receipt is
/// inherited; native acceptance remains a separate main-thread gate.
pub fn certify_affine_publication(snapshot:&FrozenCoefficients,root:&RootAffine,u:&[f64],force:&[f64],rows:&[crate::joint_hard_bridge::PhysicalRow],lambda:&[f64],mut actual_gpu_q:impl FnMut(&[f64])->Result<Vec<f64>,String>)->Result<AffinePublication,String>{
    root.enforce_prescribed(u).map_err(|e|e.code().to_string())?;
    if snapshot.dim!=u.len()||force.len()!=u.len()||rows.len()!=lambda.len()||rows.iter().any(|r|r.derivative.len()!=u.len())||force.iter().chain(lambda).any(|v|!v.is_finite()){return Err("affine publication extent/nonfinite".into());}
    let dim=u.len();let mut hard=vec![0.;dim];let mut hard_error=vec![0.;dim];
    let nu=up((2*rows.len()) as f64*2f64.powi(-53));if nu>=1.{return Err("affine covector formation overflow".into());}let gamma=up(nu/down(1.-nu));
    for k in 0..dim{let mut absolute=0.;let mut terms=0.;for (r,l) in rows.iter().zip(lambda){hard[k]+=r.derivative[k]*l;if r.derivative[k]!=0.&&*l!=0.{absolute=sum(absolute,up(r.derivative[k].abs()*l.abs()));terms+=1.;}}hard_error[k]=sum(up(gamma*absolute),up(2.*terms*f64::MIN_POSITIVE));}
    let p_eval=snapshot.evaluate(&root.particular)?;
    let mut h_lower=0.;let mut hard_lower=0.;
    for k in 0..dim{if !root.known[k]{let h=force[k]-p_eval.value[k];let e=sum(p_eval.double_error[k],sub_bound(force[k],p_eval.value[k]));h_lower=down(h_lower+down((h.abs()-e).max(0.)));hard_lower=down(hard_lower+down((hard[k].abs()-hard_error[k]).max(0.)));}}
    let norm=down(h_lower+hard_lower);
    let q:Vec<_>=u.iter().enumerate().map(|(k,&v)|if root.known[k]{root.particular[k]}else{v as f32 as f64}).collect();
    if q.iter().any(|v|!v.is_finite()){return Err("affine nearest overflow".into());}
    root.enforce_prescribed(&q).map_err(|e|e.code().to_string())?;
    // Native p must itself be EXACT published f32, including signed zero.
    if root.particular.iter().zip(&root.known).any(|(&p,&k)|k&&p!=p as f32 as f64){return Err("prescribed p is not production f32 widened".into());}
    let delta:Vec<_>=q.iter().zip(u).map(|(q,u)|q-u).collect();
    let eta:Vec<_>=q.iter().zip(u).zip(&root.known).map(|((q,u),&known)|if known{0.}else{sub_bound(*q,*u)}).collect();
    let budgets:Vec<_>=u.iter().zip(&root.known).map(|(&u,&known)|if known{0.}else{sum(up(2f64.powi(-24)*u.abs()),f32::MIN_POSITIVE as f64)}).collect();
    if (0..dim).any(|k|sum(delta[k].abs(),eta[k])>budgets[k]){return Err("affine cast lane budget".into());}
    let mut vertex_max=0f64;let mut angle_max=0f64;
    for v in 0..snapshot.rows{let body=snapshot.vertex_body[v];let upper=if body==u32::MAX{let o=snapshot.cloth_offset[v] as usize;(0..3).fold(0.,|s,a|sum(s,budgets[o+a]))}else{let o=snapshot.body_base+6*body as usize;let p=&snapshot.rotated_rest[3*v..3*v+3];(0..3).fold(0.,|s,a|{let j=(a+1)%3;let k=(a+2)%3;sum(s,sum(budgets[o+a],sum(up(p[j].abs() as f64*budgets[o+3+k]),up(p[k].abs() as f64*budgets[o+3+j]))))})};vertex_max=vertex_max.max(upper);}
    for body in 0..snapshot.bodies{let o=snapshot.body_base+6*body+3;angle_max=angle_max.max((0..3).fold(0.,|s,a|sum(s,budgets[o+a])));}
    if vertex_max>precision::PUBLICATION_VERTEX_CAST_CAP_M||angle_max>precision::PUBLICATION_ROTATION_CAST_CAP_RAD{return Err("affine physical cast cap".into());}
    let high=project_evaluation(snapshot.evaluate(u)?,&root.known)?;
    let qe=project_evaluation(snapshot.evaluate(&q)?,&root.known)?;
    let de=project_evaluation(snapshot.evaluate(&delta)?,&root.known)?;
    let eta_action=snapshot.absolute_transport_bound(&eta)?;
    let mut high_sum=0.;let mut float_sum=0.;let mut quality_sum=0.;let mut identity_ratio=0f64;let mut action=vec![0.;dim];
    for k in 0..dim{if !root.known[k]{
        let diff=qe.value[k]-high.value[k];let mismatch=(diff-de.value[k]).abs();let ib=sum(sum(sum(qe.double_error[k],high.double_error[k]),sum(de.double_error[k],eta_action[k])),sum(sub_bound(qe.value[k],high.value[k]),sub_bound(diff,de.value[k])));
        if !ib.is_finite()||!mismatch.is_finite()||mismatch>ib{return Err("affine signed identity mismatch".into());}identity_ratio=identity_ratio.max(if ib>0.{up(mismatch/ib)}else{0.});
        let r1=high.value[k]-force[k];let r2=r1-hard[k];let e=sum(sum(sum(high.double_error[k],hard_error[k]),sub_bound(high.value[k],force[k])),sub_bound(r1,hard[k]));let eh=sum(r2.abs(),e);high_sum=sum(high_sum,eh);
        action[k]=sum(sum(de.value[k].abs(),de.double_error[k]),eta_action[k]);
        let qr1=qe.value[k]-force[k];let qerr=sum(qe.double_error[k],sum(hard_error[k],sum(sub_bound(qe.value[k],force[k]),sub_bound(qr1,hard[k]))));
        float_sum=sum(float_sum,sum((qr1-hard[k]).abs(),qerr));
        quality_sum=sum(quality_sum,sum(sum(eh,action[k]),qerr));
    }}
    // No magic floor. Only structurally exact zero can pass a zero scale.
    let relative=|s:f64|if norm>0.{up(s/norm)}else if s==0.{0.}else{f64::INFINITY};
    let high_relative=relative(high_sum);let quality_relative=relative(quality_sum);
    if !high_relative.is_finite()||high_relative>crate::joint_hard_bridge::INVERSE_RELATIVE_L1_TOL||!quality_relative.is_finite()||quality_relative>precision::PUBLICATION_FORCE_QUALITY_CEILING{return Err(format!("affine publication equilibrium/quality high={high_relative:e} cast={quality_relative:e} h_lower={h_lower:e} hard_lower={hard_lower:e}"));}
    let actual=root.project(&actual_gpu_q(&q)?).map_err(|e|e.code().to_string())?;
    let ratio=qe.certify_roundoff(&actual)?;let mut raw_upper=0.;
    for k in 0..dim{if !root.known[k]{let r=actual[k]-force[k];raw_upper=sum(raw_upper,sum((r-hard[k]).abs(),sum(hard_error[k],sum(sub_bound(actual[k],force[k]),sub_bound(r,hard[k])))));}}
    let raw_relative=relative(raw_upper);if !raw_relative.is_finite()||raw_relative>precision::PUBLICATION_FORCE_QUALITY_CEILING{return Err("affine same-q GPU force quality".into());}
    for r in rows{let trial=r.coordinate-r.derivative.iter().zip(&q).map(|(b,q)|b*q).sum::<f64>();if !trial.is_finite()||crate::joint_hard_schur::violation(&r.row,trial)>crate::joint_hard_schur::ROW_TOL{return Err("affine actual uploaded row".into());}}
    let full=snapshot.evaluate(&q)?;let reaction:Vec<_>=(0..dim).map(|k|if root.known[k]{full.value[k]-force[k]-hard[k]}else{0.}).collect();
    Ok(AffinePublication{values:q,known_preserved_exactly:true,normalizer_h_lower:h_lower,normalizer_hard_lower:hard_lower,high_equilibrium_relative_upper:high_relative,actual_float_equilibrium_relative_upper:relative(float_sum),signed_cast_quality_relative_upper:quality_relative,raw_gpu_relative_upper:raw_relative,gpu_component_bound_ratio:ratio,signed_identity_maximum_ratio:identity_ratio,root_reaction:reaction,lane_budgets:budgets,actual_delta:delta,delta_subtraction_bounds:eta,cast_action_component_bounds:action,vertex_budget_max:vertex_max,rotation_budget_max:angle_max})
}

/// Evaluate QKQ using the same literal frozen coefficient object. The float
/// approximate inverse MUST already be principal: existing PCG ReducedRhs
/// receives roots view, projects both operator sides and all Krylov vectors.
pub struct CoefficientPrincipal<'a,F>{pub snapshot:&'a FrozenCoefficients,pub epoch:Epoch,pub known:Vec<bool>,pub approximate:F}
impl<F:FnMut(&[f64])->Result<Vec<f64>,String>> CoefficientPrincipal<'_,F>{
    pub fn refine_homogeneous(&mut self,rhs:&[f64],initial:&[f64])->Result<Vec<f64>,String>{
        self.refine_homogeneous_bounded(rhs,&vec![0.;rhs.len()],initial)
    }
    pub fn refine_homogeneous_bounded(&mut self,rhs:&[f64],rhs_error:&[f64],initial:&[f64])->Result<Vec<f64>,String>{
        let dim=self.snapshot.dim;
        if rhs.len()!=dim||rhs_error.len()!=dim||initial.len()!=dim||self.known.len()!=dim||rhs.iter().zip(initial).zip(&self.known).any(|((&b,&v),&k)|!b.is_finite()||!v.is_finite()||(k&&(b!=0.||v!=0.)))||rhs_error.iter().any(|e|!e.is_finite()||*e<0.){return Err("root principal refinement extent/known slots".into());}
        let norm=rhs.iter().zip(rhs_error).fold(0.,|s,(v,e)|down(s+down((v.abs()-e).max(0.))));
        if norm==0.{
            // Qf=0 cannot trigger this; only proven homogeneous h=0 may.
            if rhs.iter().chain(rhs_error).all(|v|*v==0.){return Ok(vec![0.;dim]);}
            return Err("root homogeneous RHS normalization is unproved zero".into());
        }
        let mut x=initial.to_vec();let mut last=f64::INFINITY;
        for round in 0..=6{
            let e=project_evaluation(self.snapshot.evaluate(&x)?,&self.known)?;
            let residual:Vec<_>=rhs.iter().zip(&e.value).map(|(b,a)|b-a).collect();
            let mut point=0.;let mut upper=0.;let mut unresolved=true;
            for k in 0..dim{let sub=sub_bound(rhs[k],e.value[k]);point=sum(point,sum(residual[k].abs(),sub));upper=sum(upper,sum(sum(sum(residual[k].abs(),sub),e.double_error[k]),rhs_error[k]));unresolved&=residual[k].abs()<=sum(sum(sub,e.double_error[k]),rhs_error[k]);}
            let point=up(point/norm);let upper=up(upper/norm);
            if !point.is_finite()||!upper.is_finite(){return Err("root principal nonfinite residual".into());}
            if upper<=crate::joint_hard_bridge::INVERSE_RELATIVE_L1_TOL&&(point<=1e-13||unresolved){return Ok(x);}
            if round==6||point>=last{return Err(format!("root principal refinement stagnation point={point:e} upper={upper:e}"));}last=point;
            let delta=(self.approximate)(&residual)?;
            if delta.len()!=dim||delta.iter().zip(&self.known).any(|(&v,&k)|!v.is_finite()||(k&&v!=0.)){return Err("root principal inverse leaked known slot".into());}
            for k in 0..dim{x[k]+=delta[k];}
        }unreachable!()
    }
    pub fn affine_seed(&mut self,root:&RootAffine,force:&[f64],ordinary_u0:&[f64])->Result<Vec<f64>,Error>{
        if root.epoch!=self.epoch||root.known!=self.known{return Err(Error::FrozenOperatorChanged);}
        root.enforce_prescribed(ordinary_u0)?;
        let kp=self.snapshot.evaluate(&root.particular).map_err(|_|Error::InverseFailed)?;
        if force.len()!=self.snapshot.dim{return Err(Error::InvalidInput);}
        let rhs=root.project(&force.iter().zip(&kp.value).map(|(f,k)|f-k).collect::<Vec<_>>())?;
        let rhs_error:Vec<_>=(0..force.len()).map(|k|if root.known[k]{0.}else{sum(kp.double_error[k],sub_bound(force[k],kp.value[k]))}).collect();
        let v:Vec<_>=ordinary_u0.iter().zip(&root.particular).map(|(u,p)|u-p).collect();
        let refined=self.refine_homogeneous_bounded(&rhs,&rhs_error,&v).map_err(|_|Error::InverseFailed)?;
        root.affine_from_homogeneous(&refined)
    }
    /// Full same-q apply for GPU component comparison retains known reactions.
    /// Its projection is used only for free equilibrium. Caller never checks
    /// the root reaction against zero, and must preserve actual q==p slots.
    pub fn actual_publication_evaluation(&self,root:&RootAffine,q:&[f64])->Result<Evaluation,Error>{
        if root.epoch!=self.epoch||root.known!=self.known{return Err(Error::FrozenOperatorChanged);}
        root.enforce_prescribed(q)?;
        self.snapshot.evaluate(q).and_then(|e|project_evaluation(e,&root.known)).map_err(|_|Error::InverseFailed)
    }
}
impl<F:FnMut(&[f64])->Result<Vec<f64>,String>> PrincipalBackend for CoefficientPrincipal<'_,F>{
    fn epoch(&self)->Epoch{self.epoch}
    fn apply_literal(&mut self,x:&[f64])->Result<Vec<f64>,Error>{self.snapshot.evaluate(x).map(|e|e.value).map_err(|_|Error::InverseFailed)}
    fn inverse_principal(&mut self,rhs:&[f64],known:&[bool],epoch:Epoch)->Result<Vec<f64>,Error>{
        if epoch!=self.epoch||known!=self.known{return Err(Error::FrozenOperatorChanged);}
        self.refine_homogeneous(rhs,&vec![0.;rhs.len()]).map_err(|_|Error::InverseFailed)
    }
    fn audit_literal_gram(&mut self,b:&[Vec<f64>],y:&[Vec<f64>],g:&[Vec<f64>])->Result<(),Error>{
        // Root rows of B/Y are exact zero. Full literal transpose audit has
        // the same dot identity as QKQ on that homogeneous subspace.
        if b.iter().chain(y).any(|v|v.len()!=self.known.len()||v.iter().zip(&self.known).any(|(&x,&k)|k&&x!=0.)){return Err(Error::InverseFailed);}
        self.snapshot.audit_literal_gram(b,y,g).map_err(|_|Error::InverseFailed)
    }
}
