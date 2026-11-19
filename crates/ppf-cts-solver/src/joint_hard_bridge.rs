//! Hard correction on one frozen FULL coupled reduced operator.
//! Inverse calls receive PURE reduced RHS, never the joint gradient again.
//! Translation/rotation reactions also move coupled flexible DOFs through Y.
use crate::{joint_hard_schur::{self as schur,Error,Mode,Row,Solution},joint_hard_gpu::{HardJointGpu,HardGeometryGpu}};
pub const INVERSE_RELATIVE_L1_TOL:f64=1e-7;
pub const TRANSLATION_FINAL_TOL:f64=5e-5;
pub const ANGULAR_FINAL_TOL:f64=5e-4;

/// Adapter is owned by main's step.rs integration. `apply` and `inverse` must
/// use identical frozen operator, smooth joints, masks and preconditioner.
/// Expected inverse implementation: pcg::solve_rigid_reduced_rhs(...), with
/// zero seed, NO joint gradient, NO known-root affine particular solution.
/// Unconverged / curvature-truncated inverse must return Error::InverseFailed.
pub trait FrozenReducedOperator {
    fn stamp(&self)->u64;
    fn apply(&mut self,input:&[f64])->Result<Vec<f64>,Error>;
    fn inverse(&mut self,pure_reduced_rhs:&[f64])->Result<Vec<f64>,Error>;
    /// Explicit literal adapter requires coefficient-backed skew identity.
    fn audit_literal_gram(&mut self,_b:&[Vec<f64>],_y:&[Vec<f64>],_g:&[Vec<f64>])->Result<(),Error>{Err(Error::InverseFailed)}
}
#[derive(Clone,Debug)]
pub struct PhysicalRow {pub row:Row,pub coordinate:f64,pub derivative:Vec<f64>}
#[derive(Clone,Debug)]
pub struct Correction {
    pub increment:Vec<f64>, pub schur:Solution,
    /// Full-space covector B^T lambda, including KNOWN root reaction slots.
    /// This is the force term in A*u=f+B^T*lambda. Physical restoring wrench
    /// has the opposite sign under PPF's subtractive x_trial=x-u convention.
    pub hard_covector:Vec<f64>,pub inverse_solves:usize,pub max_inverse_relative_l1:f64,
    pub frozen_stamp:u64,
}
fn dot(a:&[f64],b:&[f64])->f64 {a.iter().zip(b).map(|(x,y)|x*y).sum()}
fn finite(v:&[f64])->bool {v.iter().all(|x|x.is_finite())}

/// Interval for the physical dot(B_i,Y_j), not merely arithmetic on G's small
/// already rounded entries. gamma_(2*dim) covers products and reduction;
/// nonzero product underflow/FTZ is covered by MIN_NORMAL64. Subnormal input
/// operands are explicitly unsupported (DAZ could amplify their omission).
pub fn gram_dot_bounds(b:&[Vec<f64>],y:&[Vec<f64>])->Result<Vec<Vec<f64>>,Error>{
    let n=b.len();if y.len()!=n{return Err(Error::InvalidInput);}
    let dim=b.first().map_or(0,Vec::len);
    if b.iter().chain(y).any(|r|r.len()!=dim||r.iter().any(|v|!v.is_finite()||(*v!=0.&&v.abs()<f64::MIN_POSITIVE))){return Err(Error::Nonfinite);}
    fn up(v:f64)->f64{if v>0.&&v.is_finite(){f64::from_bits(v.to_bits()+1)}else{v}}
    let nu=up((2*dim) as f64/9007199254740992.);if nu>=1.{return Err(Error::Nonfinite);}
    let denominator=f64::from_bits((1.-nu).to_bits()-1);let gamma=up(nu/denominator);
    let mut bounds=vec![vec![0.;n];n];
    for i in 0..n{for j in 0..n{let mut absolute=0.;let mut nonzero=0usize;
        for k in 0..dim{if b[i][k]!=0.&&y[j][k]!=0.{nonzero+=1;absolute=up(absolute+up(b[i][k].abs()*y[j][k].abs()));}}
        let underflow=up((2*nonzero) as f64*f64::MIN_POSITIVE);
        bounds[i][j]=up(up(gamma*absolute)+underflow);
        if !bounds[i][j].is_finite(){return Err(Error::Nonfinite);}
    }}Ok(bounds)
}

/// Expand GPU physical rows into the FULL dim. These vectors include the
/// flexible prefix as exact zeros; inverse responses must retain that prefix.
/// Hinge uses angular XYZ rows only; a single D6 owner uses all six rows.
pub fn expand(records:&[HardJointGpu],geometry:&[HardGeometryGpu],base:usize,dim:usize,
    hinge_family:bool)->Result<Vec<PhysicalRow>,Error> {
    if records.len()!=geometry.len()||(hinge_family&&records.len()>2)||(!hinge_family&&records.len()>1){return Err(Error::InvalidInput);}
    let mut rows=Vec::new();
    for (j,g) in records.iter().zip(geometry) {
        if j.enabled==0 {continue;}
        if j.enabled!=1||g.fault!=0||j.joint_index!=g.joint_index||j.body_a!=g.body_a||j.body_b!=g.body_b||j.body_a==j.body_b{return Err(Error::InvalidInput);}
        let a=base.checked_add(6usize.checked_mul(j.body_a as usize).ok_or(Error::InvalidInput)?).ok_or(Error::InvalidInput)?;
        let b=base.checked_add(6usize.checked_mul(j.body_b as usize).ok_or(Error::InvalidInput)?).ok_or(Error::InvalidInput)?;
        if a.checked_add(6).is_none_or(|x|x>dim)||b.checked_add(6).is_none_or(|x|x>dim){return Err(Error::InvalidInput);}
        for axis in if hinge_family {3..6}else{0..6} {
            let mode=match j.modes[axis] {0=>Mode::Free,1=>Mode::Limited,2=>Mode::Locked,_=>return Err(Error::InvalidInput)};
            let mut derivative=vec![0.;dim];for k in 0..6 {derivative[a+k]=g.jacobian[12*axis+k] as f64;derivative[b+k]=g.jacobian[12*axis+6+k] as f64;}
            rows.push(PhysicalRow {row:Row {owner:j.joint_index,axis:axis as u32,mode,lower:j.lower[axis] as f64,upper:j.upper[axis] as f64,known_only:false},coordinate:g.coordinate[axis] as f64,derivative});
        }
    }
    schur::validate_rows(&rows.iter().map(|r|r.row).collect::<Vec<_>>())?;Ok(rows)
}

/// Call once before advancing. The old hard feature refuses infeasible initial
/// locks/bounds; it does not teleport an authored pose into feasibility.
pub fn initial_gate(rows:&[PhysicalRow])->Result<(),Error> {geometry_gate(rows,None,Error::InitialViolation)}
/// Actual postcast/nonlinear trial coordinates, fitted from REAL candidate
/// vertices+rigid frames after PPF's contact/strain/CCD line search. Never pass
/// Schur's predicted coordinates here. Must run before accepting the step.
pub fn final_gate(rows:&[PhysicalRow],actual_coordinates:&[f64])->Result<(),Error> {
    geometry_gate(rows,Some(actual_coordinates),Error::FinalViolation)
}
fn geometry_gate(rows:&[PhysicalRow],actual:Option<&[f64]>,code:Error)->Result<(),Error> {
    schur::validate_rows(&rows.iter().map(|r|r.row).collect::<Vec<_>>())?;
    if actual.is_some_and(|x|x.len()!=rows.len()){return Err(Error::InvalidInput);}
    for (i,r) in rows.iter().enumerate() {
        if r.row.mode==Mode::Free {continue;}
        let q=actual.map_or(r.coordinate,|x|x[i]);if !q.is_finite(){return Err(Error::Nonfinite);}
        if schur::violation(&r.row,q)>if r.row.axis<3 {TRANSLATION_FINAL_TOL}else{ANGULAR_FINAL_TOL}{return Err(code);}
    }Ok(())
}

/// `u0` comes from the ordinary coupled Newton solve, INCLUDING prescribed
/// root increments. `known[k]` excludes every fixed/root DOF from each inverse
/// RHS and inverse response; the affine u0 is preserved exactly on those slots.
/// Callback must keep the frozen stamp through ALL inverses and applications.
pub fn constrain<O:FrozenReducedOperator>(operator:&mut O,rows:&[PhysicalRow],u0:&[f64],known:&[bool])->Result<Correction,Error> {
    constrain_impl(operator,rows,u0,known,false)
}
pub fn constrain_literal<O:FrozenReducedOperator>(operator:&mut O,rows:&[PhysicalRow],u0:&[f64],known:&[bool])->Result<Correction,Error> {
    constrain_impl(operator,rows,u0,known,true)
}
fn constrain_impl<O:FrozenReducedOperator>(operator:&mut O,rows:&[PhysicalRow],u0:&[f64],known:&[bool],literal:bool)->Result<Correction,Error> {
    let n=rows.len();let dim=u0.len();if known.len()!=dim||rows.iter().any(|r|r.derivative.len()!=dim){return Err(Error::InvalidInput);}
    if !finite(u0)||rows.iter().any(|r|!r.coordinate.is_finite()||!finite(&r.derivative)){return Err(Error::Nonfinite);}
    let stamp=operator.stamp();let mut physical:Vec<Row>=rows.iter().map(|r|r.row).collect();schur::validate_rows(&physical)?;
    let free:Vec<Vec<f64>>=rows.iter().map(|r|r.derivative.iter().enumerate().map(|(k,&v)|if known[k]||r.row.mode==Mode::Free{0.}else{v}).collect()).collect();
    let mut y=vec![vec![0.;dim];n];let mut solves=0;let mut max_residual:f64=0.;
    for i in 0..n {
        physical[i].known_only=free[i].iter().all(|&x|x==0.);
        if physical[i].mode==Mode::Free {continue;}
        if physical[i].known_only {continue;}
        y[i]=operator.inverse(&free[i])?;solves+=1;
        if operator.stamp()!=stamp{return Err(Error::FrozenOperatorChanged);}
        if y[i].len()!=dim||!finite(&y[i])||y[i].iter().enumerate().any(|(k,&v)|known[k]&&v!=0.){return Err(Error::InverseFailed);}
        let applied=operator.apply(&y[i])?;
        if operator.stamp()!=stamp{return Err(Error::FrozenOperatorChanged);}
        if applied.len()!=dim||!finite(&applied){return Err(Error::InverseFailed);}
        let norm=free[i].iter().map(|v|v.abs()).sum::<f64>();
        let residual=applied.iter().zip(&free[i]).enumerate().filter(|(k,_)|!known[*k]).map(|(_,(&a,&b))|(a-b).abs()).sum::<f64>()/norm;
        if !residual.is_finite()||residual>INVERSE_RELATIVE_L1_TOL{
            eprintln!("HARD_COLUMN_TRUE_RESIDUAL_REFUSED row={i} relative={residual:.12e} limit={INVERSE_RELATIVE_L1_TOL:.12e}");
            return Err(Error::InverseFailed);
        }max_residual=max_residual.max(residual);
    }
    let mut gram=vec![vec![0.;n];n];for i in 0..n {for j in 0..n {gram[i][j]=dot(&free[i],&y[j]);}}
    let base:Vec<f64>=rows.iter().map(|r|r.coordinate-dot(&r.derivative,u0)).collect();
    let formation=if literal{gram_dot_bounds(&free,&y)?}else{vec![vec![0.;n];n]};
    if literal{
        operator.audit_literal_gram(&free,&y,&gram)?;
        if std::env::var_os("PPF_HARD_COEFFICIENT_CAPTURE").is_some(){eprintln!("HARD_LITERAL_GRAM_FORMATION_JSON {{\"stamp\":{stamp},\"literal_gram\":{:?},\"dot_product_component_bounds\":{:?},\"dimension\":{dim},\"scope\":\"gamma_2dim_sum_absolute_BiYj_plus_product_underflow; diagonals and scaled_Gs_Ga_certified_as_intervals\"}}",gram,formation);}
    }
    let solved=(if literal{schur::solve_literal(&physical,&base,&gram,&formation)}else{schur::solve(&physical,&base,&gram)}).map_err(|error| {
        if std::env::var_os("PPF_HARD_COEFFICIENT_CAPTURE").is_some() {
            // Eligible rows are NOT an invented solved active set. Schur has
            // refused before any lambda is available; literal G stays intact.
            let row_json=physical.iter().enumerate().map(|(i,r)|format!("{{\"index\":{i},\"owner\":{},\"axis\":{},\"mode\":{},\"lower\":{},\"upper\":{},\"known_only\":{},\"coordinate\":{},\"derivative\":{:?},\"projected_derivative\":{:?}}}",r.owner,r.axis,match r.mode {Mode::Free=>0,Mode::Limited=>1,Mode::Locked=>2},r.lower,r.upper,r.known_only,rows[i].coordinate,rows[i].derivative,free[i])).collect::<Vec<_>>().join(",");
            let eligible:Vec<_>=physical.iter().enumerate().filter(|(_,r)|r.mode!=Mode::Free&&!r.known_only).map(|(i,_)|i).collect();
            eprintln!("HARD_SCHUR_FAILURE_JSON {{\"kind\":\"hard_schur_failure\",\"stamp\":{stamp},\"error\":\"{}\",\"dimension\":{dim},\"rows\":[{row_json}],\"eligible_indices\":{:?},\"solved_active_set\":null,\"lambda\":null,\"literal_gram\":{:?},\"base\":{:?},\"ordinary_u0\":{:?},\"known\":{:?},\"inverse_responses\":{:?},\"inverse_solves\":{solves},\"max_inverse_relative_l1\":{max_residual},\"symmetry_tolerance\":{},\"row_tolerance\":{},\"scope\":\"literal Gram and actual full coupled inverse responses at failed Schur call; no repaired coefficients\"}}",error.code(),eligible,gram,base,u0,known,y,schur::SYMMETRY_TOL,schur::ROW_TOL);
        }
        error
    })?;
    let mut increment=u0.to_vec();let mut covector=vec![0.;dim];
    for i in 0..n {for k in 0..dim {increment[k]+=y[i][k]*solved.lambda[i];covector[k]+=rows[i].derivative[k]*solved.lambda[i];}}
    if !finite(&increment)||!finite(&covector){return Err(Error::Nonfinite);}
    if operator.stamp()!=stamp{return Err(Error::FrozenOperatorChanged);}
    // Recompute with actual inverse responses, independently of averaged G.
    for r in rows {if schur::violation(&r.row,r.coordinate-dot(&r.derivative,&increment))>schur::ROW_TOL{return Err(Error::NoFeasibleActiveSet);}}
    // Inverse response vectors stay local; none is reused at another pose.
    Ok(Correction {increment,schur:solved,hard_covector:covector,inverse_solves:solves,max_inverse_relative_l1:max_residual,frozen_stamp:stamp})
}
