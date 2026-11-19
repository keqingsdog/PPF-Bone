//! Real Device/Buffer glue for the principal pure-RHS entry and same-q apply.
//! Used only by the sealed single-D6/one-source-root runtime branch.
use ppf_cts_compute::{Device,Buffer,Handle};
use super::{operator::Operator,pdrd,pcg,joints,roots,scene::{Fatal,FatalResult}};
use crate::joint_root_hard_affine::RootAffine;
fn fatal(s:&str)->Fatal{Fatal::invariant(format!("ROOT_HARD_RUNTIME: {s}"))}
#[allow(clippy::too_many_arguments)]
pub unsafe fn inverse_once<D:Device>(device:&mut D,op:&Operator,reduction:&mut pdrd::Reduction<'_>,factor:Handle,inverse_diagonal:Handle,
    work:&mut pcg::Workspace,rigid:&mut pcg::RigidWorkspace,view:joints::View,root_view:roots::View,root:&RootAffine,
    rhs_buffer:&mut Buffer<f32>,output:&mut Buffer<f32>,rhs:&[f64],tolerance:f32,max_iterations:u32)->FatalResult<Vec<f64>>{
    if reduction.map.needs_projection()||rhs.len()!=root.known.len()||rhs.len()!=reduction.map.dim
        ||rhs.iter().zip(&root.known).any(|(&v,&known)|!v.is_finite()||(known&&v!=0.))
        ||root_view.dim as usize!=rhs.len()||root_view.base as usize!=reduction.map.body_base||root_view.bodies as usize!=reduction.map.n_bodies||root_view.count!=1{
        return Err(fatal("invalid/frozen principal mask or extra body projector"));
    }
    let scale=rhs.iter().map(|v|v.abs()).fold(0.,f64::max);if scale==0.{return Ok(vec![0.;rhs.len()]);}
    let input:Vec<_>=rhs.iter().map(|v|(v/scale) as f32).collect();
    if input.iter().zip(rhs).any(|(&v,&rhs)|!v.is_finite()||(rhs!=0.&&v==0.)){return Err(fatal("principal normalized RHS float transport"));}
    rhs_buffer.write(device,0,&input)?;
    let report=pcg::solve_rigid_reduced_rhs_with_roots(device,op,reduction,factor,inverse_diagonal,rhs_buffer.handle(),output.handle(),work,rigid,None,view,Some(root_view),tolerance,max_iterations)?;
    if !matches!(report.outcome,pcg::Outcome::Converged){return Err(fatal("principal inverse unconverged"));}
    let mut y=vec![0f32;rhs.len()];output.read(device,0,&mut y)?;
    if y.iter().zip(&root.known).any(|(&v,&known)|!v.is_finite()||(known&&v!=0.)){return Err(fatal("principal inverse leaked prescribed slot"));}
    Ok(y.iter().map(|v|*v as f64*scale).collect())
}

/// Full published q is the input, with real prescribed p slots INCLUDED.
/// Roots project only its OUTPUT for free equilibrium; this is Q*K*q, not
/// Q*K*Q*q, and therefore retains the affine K_FK*p term.
#[allow(clippy::too_many_arguments)]
pub unsafe fn apply_actual_q<D:Device>(device:&mut D,op:&Operator,reduction:&mut pdrd::Reduction<'_>,rigid:&mut pcg::RigidWorkspace,
    view:joints::View,root_view:roots::View,root:&RootAffine,input:&mut Buffer<f32>,output:&mut Buffer<f32>,absolute:Handle,curvature:Handle,q:&[f64])->FatalResult<Vec<f64>>{
    root.enforce_prescribed(q).map_err(|_|fatal("same-q prescribed slots changed"))?;
    if q.len()!=reduction.map.dim||reduction.map.needs_projection(){return Err(fatal("same-q reduction changed"));}
    let values:Vec<_>=q.iter().map(|v|*v as f32).collect();
    if q.iter().zip(&values).any(|(&a,&b)|!b.is_finite()||a!=b as f64){return Err(fatal("same-q is not exact float transport"));}
    input.write(device,0,&values)?;
    reduction.prolong(device,input.handle(),rigid.xv.handle())?;
    op.apply(device,rigid.xv.handle(),rigid.mxv.handle(),absolute,curvature)?;
    reduction.restrict(device,rigid.mxv.handle(),output.handle(),&rigid.zero)?;
    view.add(device,1,input.handle(),output.handle())?;
    root_view.project(device,output.handle())?;
    let mut host=vec![0f32;q.len()];output.read(device,0,&mut host)?;
    if host.iter().zip(&root.known).any(|(&v,&known)|!v.is_finite()||(known&&v!=0.)){return Err(fatal("same-q output mask inconsistent"));}
    Ok(host.iter().map(|v|*v as f64).collect())
}
