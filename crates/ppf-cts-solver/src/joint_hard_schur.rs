//! Rust port of old joint_d6_schur_v1.hpp solve(). Six PHYSICAL rows max.
//! q_trial = base-G*lambda, u = u0+Y*lambda. No penalty/regularization.
pub const MAX_ROWS:usize=6;
pub const ROW_TOL:f64=1e-8;
pub const SYMMETRY_TOL:f64=1e-11;
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum Mode { Free,Limited,Locked }
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum Side { None,Lower,Upper,Equality }
#[derive(Clone,Copy,Debug)]
pub struct Row {pub owner:u32,pub axis:u32,pub mode:Mode,pub lower:f64,pub upper:f64,pub known_only:bool}
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum Error {InvalidInput,Nonfinite,NonsymmetricGram,IndefiniteGram,RankDeficient,NoFeasibleActiveSet,AmbiguousOptimum,InverseFailed,FrozenOperatorChanged,InitialViolation,FinalViolation}
impl Error {
    pub fn code(self)->&'static str {match self {
        Self::InvalidInput=>"G2_HARD_INVALID_INPUT",Self::Nonfinite=>"G2_HARD_NONFINITE",Self::NonsymmetricGram=>"G2_HARD_NONSYMMETRIC_GRAM",
        Self::IndefiniteGram=>"G2_HARD_INDEFINITE_GRAM",Self::RankDeficient=>"G2_HARD_RANK_DEFICIENT",Self::NoFeasibleActiveSet=>"G2_HARD_ACTIVE_SET_INFEASIBLE",
        Self::AmbiguousOptimum=>"G2_HARD_AMBIGUOUS_OPTIMUM",Self::InverseFailed=>"G2_HARD_INVERSE_FAILED",Self::FrozenOperatorChanged=>"G2_HARD_FROZEN_OPERATOR_CHANGED",
        Self::InitialViolation=>"G2_HARD_INITIAL_INFEASIBLE",Self::FinalViolation=>"G2_HARD_FINAL_VIOLATION"}}
}
#[derive(Clone,Debug)]
pub struct Solution {pub lambda:Vec<f64>,pub trial:Vec<f64>,pub side:Vec<Side>,pub rank:usize,pub active_inequalities:usize,pub feasible_sets:usize,pub cost:f64}
pub fn validate_rows(rows:&[Row])->Result<(),Error> {
    if rows.len()>MAX_ROWS {return Err(Error::InvalidInput);}
    for (i,r) in rows.iter().enumerate() {
        if !r.lower.is_finite()||!r.upper.is_finite() {return Err(Error::Nonfinite);}
        if r.axis>=6||rows[..i].iter().any(|p|p.owner==r.owner&&p.axis==r.axis)||
            (r.mode==Mode::Limited&&r.lower>=r.upper)||
            (r.mode==Mode::Locked&&r.lower!=r.upper) {return Err(Error::InvalidInput);}
    } Ok(())
}
pub fn violation(r:&Row,q:f64)->f64 {
    match r.mode {Mode::Free=>0.,Mode::Locked=>(q-r.lower).abs(),Mode::Limited=>(r.lower-q).max(q-r.upper).max(0.)}
}
fn factor(g:&[Vec<f64>],ids:&[usize],rhs:&[f64])->Result<Vec<f64>,Error> {
    let n=ids.len();let mut scale=vec![0.;n];let mut perm:Vec<usize>=(0..n).collect();let mut m=vec![vec![0.;n];n];let mut l=vec![vec![0.;n];n];
    for i in 0..n {let d=g[ids[i]][ids[i]];if !d.is_finite()||!rhs[i].is_finite(){return Err(Error::Nonfinite);}
        if d<=0. {return Err(if d< -1e-12 {Error::IndefiniteGram}else{Error::RankDeficient});}scale[i]=d.sqrt();}
    for i in 0..n {for j in 0..n {m[i][j]=g[ids[i]][ids[j]]/scale[i]/scale[j];if !m[i][j].is_finite(){return Err(Error::Nonfinite);}}}
    for k in 0..n {let best=(k..n).max_by(|&i,&j|m[i][i].total_cmp(&m[j][j])).unwrap();
        if m[best][best]< -1e-12{return Err(Error::IndefiniteGram);}if m[best][best]<=1e-10{return Err(Error::RankDeficient);}
        if best!=k {m.swap(k,best);for row in &mut m {row.swap(k,best);}for j in 0..k {let x=l[k][j];l[k][j]=l[best][j];l[best][j]=x;}perm.swap(k,best);}
        l[k][k]=m[k][k].sqrt();for i in k+1..n {l[i][k]=m[i][k]/l[k][k];}
        for i in k+1..n {for j in i..n {m[j][i]-=l[i][k]*l[j][k];m[i][j]=m[j][i];}}
    }
    let mut y=vec![0.;n];let mut z=vec![0.;n];let mut x=vec![0.;n];
    for i in 0..n {let mut v=rhs[perm[i]]/scale[perm[i]];for j in 0..i {v-=l[i][j]*y[j];}y[i]=v/l[i][i];}
    for i in (0..n).rev() {let mut v=y[i];for j in i+1..n {v-=l[j][i]*z[j];}z[i]=v/l[i][i];}
    for i in 0..n {x[perm[i]]=z[i]/scale[perm[i]];if !x[perm[i]].is_finite(){return Err(Error::Nonfinite);}}Ok(x)
}
pub fn solve(rows:&[Row],base:&[f64],gram:&[Vec<f64>])->Result<Solution,Error> {
    validate_rows(rows)?;let n=rows.len();
    if base.len()!=n||gram.len()!=n||gram.iter().any(|v|v.len()!=n){return Err(Error::InvalidInput);}
    if base.iter().any(|x|!x.is_finite()){return Err(Error::Nonfinite);}
    let hard:Vec<usize>=(0..n).filter(|&i|rows[i].mode!=Mode::Free&&!rows[i].known_only).collect();
    // A true all-free case need not even inspect an unused Gram matrix.
    if rows.iter().all(|r|r.mode==Mode::Free) {return Ok(Solution {lambda:vec![0.;n],trial:base.to_vec(),side:vec![Side::None;n],rank:0,active_inequalities:0,feasible_sets:1,cost:0.});}
    let mut g=vec![vec![0.;n];n];
    for i in 0..n {for j in 0..n {let a=gram[i][j];let b=gram[j][i];
        if !a.is_finite()||!b.is_finite(){return Err(Error::Nonfinite);}
        if (rows[i].known_only||rows[j].known_only)&&(a!=0.||b!=0.){return Err(Error::InvalidInput);}
        let scale=a.abs().max(b.abs()).max(gram[i][i].abs().sqrt()*gram[j][j].abs().sqrt());
        if !scale.is_finite()||!(a-b).is_finite(){return Err(Error::Nonfinite);}
        if (a-b).abs()>SYMMETRY_TOL*scale{return Err(Error::NonsymmetricGram);}g[i][j]=0.5*(a+b);
    }}
    if hard.is_empty() {
        if rows.iter().zip(base).any(|(r,&q)|violation(r,q)>ROW_TOL){return Err(Error::NoFeasibleActiveSet);}
        return Ok(Solution {lambda:vec![0.;n],trial:base.to_vec(),side:vec![Side::None;n],rank:0,active_inequalities:0,feasible_sets:1,cost:0.});
    }
    // Rank-check every potential hard row, preserving old refusal semantics.
    factor(&g,&hard,&vec![0.;hard.len()])?;
    let limited:Vec<usize>=(0..n).filter(|&i|rows[i].mode==Mode::Limited&&!rows[i].known_only).collect();
    let mut best:Option<Solution>=None;let mut feasible_sets=0;
    for state in 0..3usize.pow(limited.len() as u32) {
        let mut bits=state;let mut choice=vec![0;n];for &i in &limited {choice[i]=bits%3;bits/=3;}
        let ids:Vec<usize>=(0..n).filter(|&i|!rows[i].known_only&&(rows[i].mode==Mode::Locked||choice[i]!=0)).collect();
        let rhs:Vec<f64>=ids.iter().map(|&i|base[i]-if rows[i].mode==Mode::Locked||choice[i]==1{rows[i].lower}else{rows[i].upper}).collect();
        let sub=factor(&g,&ids,&rhs)?;let mut z=vec![0.;n];for (&i,&v) in ids.iter().zip(&sub){z[i]=v;}
        let mut trial=base.to_vec();let mut side=vec![Side::None;n];let mut feasible=true;let mut cost=0.;
        for i in 0..n {
            if (choice[i]==1&&z[i]>0.)||(choice[i]==2&&z[i]<0.){feasible=false;}
            side[i]=if rows[i].mode==Mode::Locked&&!rows[i].known_only {Side::Equality}else if choice[i]==1{Side::Lower}else if choice[i]==2{Side::Upper}else{Side::None};
            for j in 0..n {trial[i]-=g[i][j]*z[j];cost+=0.5*z[i]*g[i][j]*z[j];}
            if !trial[i].is_finite(){return Err(Error::Nonfinite);}if violation(&rows[i],trial[i])>ROW_TOL{feasible=false;}
            let target=if choice[i]==1{rows[i].lower}else{rows[i].upper};if choice[i]!=0&&(trial[i]-target).abs()>ROW_TOL{feasible=false;}
        }
        if !cost.is_finite(){return Err(Error::Nonfinite);}if !feasible{continue;}feasible_sets+=1;
        let active=choice.iter().filter(|&&c|c!=0).count();let mut tie=false;
        if let Some(ref old)=best {
            let fp=64.*f64::EPSILON*cost.abs().max(old.cost.abs());let mass=z.iter().zip(&old.lambda).map(|(a,b)|a.abs()+b.abs()).sum::<f64>();let slack=ROW_TOL*mass;
            tie=(cost-old.cost).abs()<=fp;
            if (cost-old.cost).abs()<=slack+fp {
                let mut dist=0.;let mut terms=0.;for i in 0..n {for j in 0..n {let v=(z[i]-old.lambda[i])*g[i][j]*(z[j]-old.lambda[j]);dist+=v;terms+=v.abs();}}
                if !dist.is_finite()||!terms.is_finite()||dist< -64.*f64::EPSILON*terms||dist>4.*slack+8.*fp{return Err(Error::AmbiguousOptimum);}
            }
        }
        if best.as_ref().map_or(true,|old|(cost<old.cost&&!tie)||(tie&&active<old.active_inequalities)) {
            best=Some(Solution {lambda:z,trial,side,rank:ids.len(),active_inequalities:active,feasible_sets,cost});
        }
    }
    let mut result=best.ok_or(Error::NoFeasibleActiveSet)?;result.feasible_sets=feasible_sets;Ok(result)
}





/// Explicit precision-adapter acceptance domain. This does not alter or call
/// the legacy symmetric-QP solve above. The literal G is never averaged for LU.
#[derive(Debug)]
pub struct LiteralGramCertificate {pub alpha_lower:f64,pub skew_upper:f64,pub skew_ratio_upper:f64,pub legacy_symmetry_passed:bool}
fn lit_up(v:f64)->f64 {if v>0.&&v.is_finite(){f64::from_bits(v.to_bits()+1)}else{v}}
fn lit_down(v:f64)->f64 {if v>0.&&v.is_finite(){f64::from_bits(v.to_bits()-1)}else{v}}
pub fn certify_literal_gram(rows:&[Row],g:&[Vec<f64>],formation:&[Vec<f64>])->Result<LiteralGramCertificate,Error> {
    validate_rows(rows)?;let n=rows.len();if g.len()!=n||g.iter().any(|v|v.len()!=n||v.iter().any(|x|!x.is_finite())) {return Err(Error::Nonfinite);}
    if formation.len()!=n||formation.iter().any(|r|r.len()!=n||r.iter().any(|e|!e.is_finite()||*e<0.)){return Err(Error::InvalidInput);}
    let ids:Vec<_>=(0..n).filter(|&i|rows[i].mode!=Mode::Free&&!rows[i].known_only).collect();
    let mut legacy=true;for i in 0..n {for j in 0..n {
        if (rows[i].known_only||rows[j].known_only)&&(g[i][j]!=0.||g[j][i]!=0.){return Err(Error::InvalidInput);}
        let scale=g[i][j].abs().max(g[j][i].abs()).max(g[i][i].abs().sqrt()*g[j][j].abs().sqrt());
        legacy &= (g[i][j]-g[j][i]).abs()<=SYMMETRY_TOL*scale;
    }}
    let mut alpha=1.0_f64;let mut skew=0.0_f64;
    for &i in &ids {let diagonal_i=lit_down(g[i][i]-formation[i][i]);if diagonal_i<=0. {return Err(Error::IndefiniteGram);}let mut off=0.;let mut anti=0.;
        for &j in &ids {if i==j{continue;}let diagonal_j=lit_down(g[j][j]-formation[j][j]);if diagonal_j<=0.{return Err(Error::IndefiniteGram);}
            // Exact physical response diagonals are positive throughout their
            // dot-product intervals. Scale by their exact square roots; these
            // LOWER root/product bounds enclose all possible off-diagonal ratios.
            let denominator=lit_down(lit_down(diagonal_i.sqrt())*lit_down(diagonal_j.sqrt()));
            if !denominator.is_normal(){return Err(Error::RankDeficient);}
            // 32u64 covers rounded roots, product, sum/difference and divisions
            // against the literal f64 G. It is an enclosure, not physical slack.
            let underflow=if g[i][j]!=0.||g[j][i]!=0.||formation[i][j]!=0.||formation[j][i]!=0.{16.*f64::MIN_POSITIVE}else{0.};
            let round=lit_up(32.*f64::EPSILON*(lit_up(g[i][j].abs()+g[j][i].abs())/denominator)+underflow);
            let dot_uncertainty=lit_up(lit_up(formation[i][j]+formation[j][i])*0.5/denominator);
            off=lit_up(off+lit_up(lit_up((0.5*(g[i][j]+g[j][i])/denominator).abs()+round)+dot_uncertainty));
            anti=lit_up(anti+lit_up(lit_up((0.5*(g[i][j]-g[j][i])/denominator).abs()+round)+dot_uncertainty));
        }
        alpha=alpha.min(lit_down(1.-off));skew=skew.max(anti);
    }
    if !alpha.is_finite()||alpha<=0.{return Err(Error::IndefiniteGram);}
    let ratio=lit_up(skew/alpha);if !ratio.is_finite()||ratio>1./16777216.{return Err(Error::NonsymmetricGram);}
    Ok(LiteralGramCertificate{alpha_lower:alpha,skew_upper:skew,skew_ratio_upper:ratio,legacy_symmetry_passed:legacy})
}

fn literal_lu(g:&[Vec<f64>],ids:&[usize],rhs:&[f64])->Result<Vec<f64>,Error> {
    let n=ids.len();if n==0{return Ok(Vec::new());}
    let scale:Vec<_>=ids.iter().map(|&i|g[i][i].sqrt()).collect();
    let mut a=vec![vec![0.;n];n];let mut b=vec![0.;n];
    for i in 0..n {b[i]=rhs[i]/scale[i];for j in 0..n {a[i][j]=g[ids[i]][ids[j]]/scale[i]/scale[j];}}
    for k in 0..n {let pivot=(k..n).max_by(|&i,&j|a[i][k].abs().total_cmp(&a[j][k].abs())).unwrap();
        if !a[pivot][k].is_finite(){return Err(Error::Nonfinite);}if a[pivot][k].abs()<=1e-10{return Err(Error::RankDeficient);}
        a.swap(k,pivot);b.swap(k,pivot);
        for i in k+1..n {let m=a[i][k]/a[k][k];a[i][k]=0.;for j in k+1..n {a[i][j]-=m*a[k][j];}b[i]-=m*b[k];}
    }
    let mut x=vec![0.;n];for i in (0..n).rev(){let mut value=b[i];for j in i+1..n{value-=a[i][j]*x[j];}x[i]=value/a[i][i];}
    for i in 0..n{x[i]/=scale[i];}if x.iter().any(|x|!x.is_finite()){return Err(Error::Nonfinite);}Ok(x)
}

/// Literal monotone complementarity solve; at most 3^6 active combinations.
/// Gs only certifies coercivity. Feasible candidates use literal G, multiplier
/// signs and complementarity; there is no symmetric-QP cost winner.
pub fn solve_literal(rows:&[Row],base:&[f64],g:&[Vec<f64>],formation:&[Vec<f64>])->Result<Solution,Error> {
    let certificate=certify_literal_gram(rows,g,formation)?;let n=rows.len();
    if base.len()!=n||base.iter().any(|x|!x.is_finite()){return Err(Error::Nonfinite);}
    let potential:Vec<_>=(0..n).filter(|&i|rows[i].mode!=Mode::Free&&!rows[i].known_only).collect();
    literal_lu(g,&potential,&vec![0.;potential.len()])?;
    eprintln!("HARD_LITERAL_GRAM_CONTRACT legacy_symmetry_status={} original_symmetry_limit={SYMMETRY_TOL:.12e} scaled_symmetric_coercivity_lower={:.12e} scaled_antisym_upper={:.12e} skew_over_coercivity_upper={:.12e} tiny_skew_limit={:.12e} scope=literal_operator_LU_LCP_no_Gram_average_no_QP_cost",if certificate.legacy_symmetry_passed{"PASSED"}else{"FAILED_LEGACY_SYMMETRY"},certificate.alpha_lower,certificate.skew_upper,certificate.skew_ratio_upper,1./16777216.);
    let limited:Vec<_>=(0..n).filter(|&i|rows[i].mode==Mode::Limited&&!rows[i].known_only).collect();
    let mut best:Option<Solution>=None;let mut feasible_sets=0;
    for state in 0..3usize.pow(limited.len() as u32) {
        let mut bits=state;let mut choice=vec![0;n];for &i in &limited{choice[i]=bits%3;bits/=3;}
        let ids:Vec<_>=(0..n).filter(|&i|!rows[i].known_only&&(rows[i].mode==Mode::Locked||choice[i]!=0)).collect();
        let rhs:Vec<_>=ids.iter().map(|&i|base[i]-if rows[i].mode==Mode::Locked||choice[i]==1{rows[i].lower}else{rows[i].upper}).collect();
        let sub=literal_lu(g,&ids,&rhs)?;let mut z=vec![0.;n];for (&i,&value) in ids.iter().zip(&sub){z[i]=value;}
        let mut trial=base.to_vec();let mut side=vec![Side::None;n];let mut feasible=true;
        for i in 0..n {
            if (choice[i]==1&&z[i]>0.)||(choice[i]==2&&z[i]<0.){feasible=false;}
            side[i]=if rows[i].mode==Mode::Locked&&!rows[i].known_only{Side::Equality}else if choice[i]==1{Side::Lower}else if choice[i]==2{Side::Upper}else{Side::None};
            let mut magnitude=base[i].abs()+rows[i].lower.abs()+rows[i].upper.abs();
            for j in 0..n {trial[i]-=g[i][j]*z[j];magnitude+=g[i][j].abs()*z[j].abs();}
            if !trial[i].is_finite(){return Err(Error::Nonfinite);}
            // Roundoff allowance only for numeric complementarity. ROW_TOL
            // stays the separate original physical row gate, not a cost tie.
            let round=128.*f64::EPSILON*magnitude;
            if violation(&rows[i],trial[i])>round.min(ROW_TOL){feasible=false;}
            if choice[i]!=0&&(trial[i]-if choice[i]==1{rows[i].lower}else{rows[i].upper}).abs()>round.min(ROW_TOL){feasible=false;}
        }
        if !feasible{continue;}feasible_sets+=1;
        if let Some(old)=&best {
            let scale=z.iter().chain(&old.lambda).map(|x|x.abs()).fold(0.,f64::max);
            if z.iter().zip(&old.lambda).any(|(a,b)|(a-b).abs()>128.*f64::EPSILON*scale){return Err(Error::AmbiguousOptimum);}
            continue;
        }
        best=Some(Solution{lambda:z,trial,side,rank:ids.len(),active_inequalities:choice.iter().filter(|&&v|v!=0).count(),feasible_sets,cost:0.});
    }
    let mut result=best.ok_or(Error::NoFeasibleActiveSet)?;result.feasible_sets=feasible_sets;Ok(result)
}
