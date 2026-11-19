//! Opt-in bounded first-frozen-state diagnostic. No production routing changes.
//! Uses the ORIGINAL prolong/operator/restrict/joint kernels on private vectors.
//! This tests unprojected J^T A J; original outer lock/root/body Q stays unchanged.
use ppf_cts_compute::{AllocLabel,Buffer,Device,Fault,Handle};
use super::{kernels::*,joints,operator::Operator,rigid_map::RigidMap,
    reduced_model_backend::{Inputs,ModelView}};
fn refuse(s:impl Into<String>)->Fault{Fault::Platform{call:"frozen full projection check",detail:s.into()}}
unsafe fn f32s<D:Device>(d:&mut D,h:Handle,n:usize)->Result<Vec<f32>,Fault>{
    let mut x=vec![0f32;n];d.read(h,0,std::slice::from_raw_parts_mut(x.as_mut_ptr().cast(),n*4))?;Ok(x)
}
unsafe fn u32s<D:Device>(d:&mut D,h:Handle,n:usize)->Result<Vec<u32>,Fault>{
    let mut x=vec![0u32;n];d.read(h,0,std::slice::from_raw_parts_mut(x.as_mut_ptr().cast(),n*4))?;Ok(x)
}
fn floats<D:Device>(d:&mut D,n:usize,label:&'static str)->Result<Buffer<f32>,Fault>{
    let mut b=Buffer::default();b.size(d,n.max(1),AllocLabel(label))?;Ok(b)
}
// Exactly the two direct diagonal spans already included by ORIGINAL
// pdrd_assemble_sandwich_row. Reference/transpose spans are NOT seed-counted.
unsafe fn diagonal_span<D:Device>(d:&mut D,rows:usize,oh:Handle,ih:Handle,vh:Handle,
    out:&mut[f64])->Result<(),Fault>{
    let off=u32s(d,oh,rows+1)?;let used=off[rows]as usize;
    if used>1_000_000{return Err(refuse("diagnostic source span exceeds bounded 1M blocks"));}
    if used==0{return Ok(());}
    let ids=u32s(d,ih,used)?;let val=f32s(d,vh,used*9)?;
    for v in 0..rows{if off[v]>off[v+1]{return Err(refuse("source offset order"));}
        for k in off[v]as usize..off[v+1]as usize{if ids[k]as usize==v{
            for q in 0..9{out[9*v+q]+=f64::from(val[9*k+q]);}
        }}
    }Ok(())
}

pub unsafe fn run<D:Device>(d:&mut D,map:&RigidMap,op:&Operator,mut joint:joints::View,
    input:Inputs,model:ModelView,seed_h:Handle,inertia_h:Handle)->Result<serde_json::Value,Fault>{
    let dim=map.dim;let rows=map.nrow;let full=3*rows;let bodies=map.n_bodies;
    // B is the assembled roundoff approximation, not the new physical factor A.
    let assembled=Operator{fixed_pattern: None, tet_factor:None,dynamic:op.dynamic,
        fixed:super::spmv::FixedCsrView{index:op.fixed.index,offset:op.fixed.offset,value:op.fixed.value,
            transpose_pair:op.fixed.transpose_pair,transpose_offset:op.fixed.transpose_offset,rows:op.fixed.rows},
        diagonal:op.diagonal};
    let factor_afx=floats(d,full,"projection_check.physical_factor_AJx")?;
    let mut factor_difference_max=0f64;let mut factor_difference_l1_max=0f64;

    if dim==0||dim>512||rows>4096{return Err(refuse("opt-in diagnostic requires dim1..512, full rows<=4096"));}
    if model.rows as usize*3!=dim{return Err(refuse("model/layout mismatch"));}
    let off=u32s(d,model.offset,dim/3+1)?;let used=off[dim/3]as usize;
    if used!=model.blocks as usize||used>1_000_000{return Err(refuse("diagnostic model extent"));}
    let col=u32s(d,model.column,used)?;let val=f32s(d,model.value,9*used)?;
    let seed=f32s(d,seed_h,36*bodies)?;let inertia=f32s(d,inertia_h,36*bodies)?;
    let raw=f32s(d,input.raw_body,36*bodies)?;
    let state_bytes=std::mem::size_of::<crate::data::PdrdRigidState>();
    let mut states=vec![crate::data::PdrdRigidState::default();bodies];
    if bodies!=0{d.read(input.body_state,0,std::slice::from_raw_parts_mut(states.as_mut_ptr().cast(),bodies*state_bytes))?;}
    let mut diagonal=vec![0f64;9*rows];let r=input.rows;
    if op.dynamic.is_some(){diagonal_span(d,rows,r.dynamic_offset,r.dynamic_index,r.dynamic_value,&mut diagonal)?;}
    diagonal_span(d,rows,r.fixed_offset,r.fixed_index,r.fixed_value,&mut diagonal)?;
    let mut cloth=Buffer::<u32>::default();cloth.size(d,rows.max(1),AllocLabel("projection_check.cloth_offset"))?;
    cloth.write(d,0,&map.cloth_off)?;
    let mut x=floats(d,dim,"projection_check.x")?;let fx=floats(d,full,"projection_check.Jx")?;
    let afx=floats(d,full,"projection_check.AJx")?;let result=floats(d,dim,"projection_check.JtAJx")?;
    let absolute=floats(d,rows,"projection_check.row_absolute")?;
    let curvature=floats(d,rows,"projection_check.row_curvature")?;
    let nf=joint.count.max(joint.control_count).max(joint.extension_count)
        .max(joint.soft.map_or(0,|s|s.count)).max(1)as usize;
    let mut fault=Buffer::<u32>::default();fault.size(d,nf,AllocLabel("projection_check.private_fault"))?;
    joint.fault=fault.handle();
    let private_curves=|mut c:super::restoring::View|{c.fault=fault.handle();c.ball_fault=fault.handle();
        c.angular_fault=fault.handle();c.fem_fault=fault.handle();c.stitch_fault=fault.handle();c};
    joint.curves=joint.curves.map(private_curves);
    if let Some(mut s)=joint.soft{s.fault=fault.handle();s.curves=s.curves.map(private_curves);joint.soft=Some(s);}
    let prolong=|h|PdrdProlongRowArgs{vertex_body:input.vertex_body,cloth_offset:cloth.handle(),rotated_rest:input.rotated_rest,
        reduced:h,full:fx.handle(),body_base:map.body_base as u32,count:rows as u32,seam_arena_count:0};
    // Probe actual original J columns, including original sign/current rotation.
    let mut j=vec![vec![0f32;full];dim];
    for c in 0..dim{let mut e=vec![0f32;dim];e[c]=1.;x.write(d,0,&e)?;
        d.launch("projection_check.original_prolong_basis",&prolong(x.handle()),rows as u32)?;
        j[c]=f32s(d,fx.handle(),full)?;}
    // Complete explicit remaining seed delta in FP64, using exact stored
    // subtraction rounding. Includes seed symmetry/shift and sandwich rounding.
    let mut delta=vec![0f64;dim*dim];let mut regularization_max=0f64;
    for b in 0..bodies{for rr in 0..6{for cc in 0..6{
        let k=36*b+6*rr+cc;let row=map.body_base+6*b+rr;let column=map.body_base+6*b+cc;
        let q=seed[k]-if states[b].count!=0{inertia[k]}else{0.};
        delta[row*dim+column]=f64::from(q);
        regularization_max=regularization_max.max((f64::from(seed[k])-f64::from(raw[k])).abs());
    }}}
    for v in 0..rows{if map.vbody[v]==u32::MAX{continue;}let b=map.vbody[v]as usize;
        for rr in 0..6{for cc in 0..6{let row=map.body_base+6*b+rr;let column=map.body_base+6*b+cc;
            let mut q=0f64;for a in 0..3{for z in 0..3{
                q+=f64::from(j[row][3*v+a])*diagonal[9*v+3*z+a]*f64::from(j[column][3*v+z]);
            }}delta[row*dim+column]-=q;
        }}
    }
    let mut max_ratio=0f64;let mut max_error=0f64;let mut max_delta=0f64;
    for p in 0..dim+2{
        let mut v=vec![0f32;dim];if p<dim{v[p]=1.;}else{for i in 0..dim{
            v[i]=if p==dim{((7*i)%11)as f32/8.-5./8.}else{((5*i)%13)as f32/16.-6./16.};}}
        x.write(d,0,&v)?;d.launch("projection_check.original_prolong",&prolong(x.handle()),rows as u32)?;
        assembled.apply(d,fx.handle(),afx.handle(),absolute.handle(),curvature.handle())?;
        if op.has_tet_factor() {
            op.apply(d,fx.handle(),factor_afx.handle(),absolute.handle(),curvature.handle())?;
            let approx=f32s(d,afx.handle(),full)?;let physical=f32s(d,factor_afx.handle(),full)?;
            let mut l1=0f64;
            for i in 0..full {
                let difference=(f64::from(approx[i])-f64::from(physical[i])).abs();
                if !difference.is_finite(){return Err(refuse("physical factor comparison nonfinite"));}
                factor_difference_max=factor_difference_max.max(difference);l1+=difference;
            }
            factor_difference_l1_max=factor_difference_l1_max.max(l1);
        }
        d.fill_zero(result.handle(),dim*4)?;d.fill_zero(fault.handle(),nf*4)?;
        let restrict=PdrdRestrictRowArgs{vertex_body:input.vertex_body,cloth_offset:cloth.handle(),rotated_rest:input.rotated_rest,
            full:afx.handle(),cloth_out:result.handle(),reduced:result.handle(),body_base:map.body_base as u32,
            count:rows as u32,seam_arena_count:0};
        d.launch("projection_check.original_restrict",&restrict,rows as u32)?;
        joint.add(d,1,x.handle(),result.handle())?;
        let actual=f32s(d,result.handle(),dim)?;let codes=u32s(d,fault.handle(),nf)?;
        if codes.iter().any(|&q|q!=0){return Err(refuse(format!("original joint action1 refused {codes:?}")));}
        let mut bv=vec![0f64;dim];let mut terms=vec![0f64;dim];
        for row in 0..dim/3{for k in off[row]as usize..off[row+1]as usize{
            if col[k]as usize>=dim/3{return Err(refuse("model column range"));}
            for a in 0..3{for z in 0..3{let q=f64::from(val[9*k+3*z+a])*f64::from(v[3*col[k]as usize+z]);
                bv[3*row+a]+=q;terms[3*row+a]+=q.abs();}}
        }}
        for row in 0..dim{let mut correction=0f64;for c in 0..dim{
            let q=delta[row*dim+c]*f64::from(v[c]);correction+=q;terms[row]+=q.abs();}
            if !actual[row].is_finite(){return Err(refuse("original operator nonfinite"));}
            let error=(bv[row]-f64::from(actual[row])-correction).abs();
            let bound=4096.*f64::from(f32::EPSILON)*(terms[row]+f64::from(actual[row]).abs());
            let ratio=if bound==0.{if error==0.{0.}else{f64::INFINITY}}else{error/bound};
            max_ratio=max_ratio.max(ratio);max_error=max_error.max(error);max_delta=max_delta.max(correction.abs());
            if !ratio.is_finite()||ratio>1.{return Err(refuse(format!("probe{p} scalar{row}: |Bv-JtAJv-delta*v|={error:e} > {bound:e}")));}
        }
    }
    Ok(serde_json::json!({"schema":"ppf.frozen-full-projection.v1","dim":dim,"full_rows":rows,"blocks":used,
        "basis_columns":dim,"mixed_probes":2,"same_frozen_revision":model.revision.operator,
        "comparison":"CPU FP64 ORIGINAL DUPLICATE approximate CSR Bv versus assembled-approximation GPU prolong/operator/restrict/joint action1 plus seed delta",
        "csr_storage_scope":"raw duplicate assembled approximation before signed coalescing",
        "mapping_revision":model.revision.mapping,"joint_revision":model.revision.joints,
        "composed_evidence":"same-revision SCHWARZ_COALESCE_CHECK separately validates raw approximate CSR->unique approximate CSR on identical complete basis and mixed probes; raw full-projection check retains original sumabs bound",
        "triangle_scope":"unique approximate CSR to assembled approximation composes unique->raw and raw->assembled checks; physical factor difference is separately reported below, not absorbed into arithmetic tolerance",
        "physical_factor_active":op.has_tet_factor(),
        "assembled_approx_vs_physical_factor_max_component_difference":factor_difference_max,
        "assembled_approx_vs_physical_factor_max_probe_l1_difference":factor_difference_l1_max,
        "factor_difference_acceptance":"reported explicitly; this assembled projection check does not validate physical factor energy or true residual",
        "seed_delta":"stored f32(seed-private_analytic_inertia), empty body keeps seed, minus independently projected original direct dynamic/fixed diagonal spans",
        "body_seed_symmetry_regularization_max":regularization_max,"max_seed_delta_product":max_delta,
        "max_absolute_error":max_error,"max_component_error_ratio":max_ratio,
        "bound":"4096*eps32*(sumabs stored Bv terms + sumabs delta*v terms + abs original action1) componentwise; exact zero if bound zero",
        "scope":"unprojected operator algebra; lock/root/body projectors remain original outer Q, not reassessed by this diagnostic",
        "private_vectors_and_faults":true,"history_and_observations":"original read-only frozen history; original action1 does not observe"}))
}
