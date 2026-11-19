//! Hard-only synchronization/transport for the pure frozen-coefficient math.
use ppf_cts_compute::{Device, Handle, Pod, Buffer, AllocLabel};
use crate::joint_hard_coefficients::{FrozenCoefficients,Csr,SmoothFactors};
use super::{operator::Operator,pdrd,joints,kernels::HardFrozenSmoothFactorsArgs};
use super::scene::{Fatal,FatalResult};
unsafe impl Pod for SmoothFactors {}
fn fatal(s:impl std::fmt::Display)->Fatal {Fatal::invariant(format!("G2_HARD_COEFFICIENT_SNAPSHOT: {s}"))}
fn download<T:Pod+Default+Clone,D:Device>(device:&mut D,h:Handle)->FatalResult<Vec<T>> {
    let mut v=vec![T::default();h.size as usize];
    let bytes=unsafe {std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(),std::mem::size_of_val(v.as_slice()))};
    if !bytes.is_empty(){device.read(h,0,bytes)?;} Ok(v)
}
fn prefix<T:Pod+Default+Clone,D:Device>(device:&mut D,mut h:Handle,n:usize)->FatalResult<Vec<T>> {
    if n>h.size as usize || n>u32::MAX as usize {return Err(fatal("snapshot prefix exceeds allocation"));}
    h.size=n as u32;download(device,h)
}
impl FrozenCoefficients {
    /// Synchronizing downloads take place exactly once within correct()'s
    /// exclusive frozen borrow, before PCG scratch is reused. Sparse operator
    /// handles are read-only; no global matrix or P allocation is changed.
    pub unsafe fn capture<D:Device>(device:&mut D,op:&Operator,
        reduction:&pdrd::Reduction<'_>,view:joints::View,exported:&mut Buffer<SmoothFactors>)->FatalResult<Self> {
        if reduction.map.needs_projection() || view.extension_count!=0 || view.drive_count!=0 ||
            !view.dt.is_finite()||view.dt<=0.0||(view.control_count!=0&&view.control_count!=view.count) {
            return Err(fatal("root/projector/soft extension is outside the hard adapter domain"));
        }
        let rows=op.rows() as usize;
        let fixed_offset:Vec<u32>=prefix(device,op.fixed.offset,rows+1)?;
        let fixed_nnz=*fixed_offset.last().ok_or_else(||fatal("empty offsets"))? as usize;
        let fixed_transpose_offset:Vec<u32>=prefix(device,op.fixed.transpose_offset,rows+1)?;
        let fixed_refs=*fixed_transpose_offset.last().ok_or_else(||fatal("empty transpose offsets"))? as usize;
        let pair:Vec<u32>=prefix(device,op.fixed.transpose_pair,2*fixed_refs)?;
        if pair.len()%2!=0{return Err(fatal("fixed transpose pairs are not pairs"));}
        let fixed=Csr {offset:fixed_offset,index:prefix(device,op.fixed.index,fixed_nnz)?,
            value:prefix(device,op.fixed.value,9*fixed_nnz)?,transpose_offset:fixed_transpose_offset,
            transpose_index:pair.chunks_exact(2).map(|p|p[0]).collect(),
            transpose_slot:pair.chunks_exact(2).map(|p|p[1]).collect()};
        let dynamic=match op.dynamic {None=>None,Some(d)=>{
            let offset:Vec<u32>=prefix(device,d.offset,rows+1)?;
            let nnz=*offset.last().ok_or_else(||fatal("empty dynamic offsets"))? as usize;
            let transpose_offset:Vec<u32>=prefix(device,d.transpose_offset,rows+1)?;
            let refs=*transpose_offset.last().ok_or_else(||fatal("empty dynamic transpose"))? as usize;
            Some(Csr {offset,transpose_offset,index:prefix(device,d.index,nnz)?,value:prefix(device,d.value,9*nnz)?,
                transpose_index:prefix(device,d.transpose_index,refs)?,transpose_slot:prefix(device,d.transpose_value,refs)?})
        }};
        exported.size(device,view.count as usize,AllocLabel("hard.frozen_smooth_factors"))?;
        if view.count!=0 {
            let args=HardFrozenSmoothFactorsArgs {ball:view.ball,angular:view.angular,control:view.control,twist_context:view.twist_context,
                state:view.state,reference:view.reference,bodies:view.bodies,dt:view.dt,
                control_count:view.control_count,output:exported.handle(),count:view.count,seam_arena_count:0};
            device.launch("hard.frozen_smooth_factors",&args,view.count)?;
        }
        let snapshot=Self {rows:op.rows() as usize,dim:reduction.map.dim,body_base:reduction.map.body_base,
            bodies:reduction.map.n_bodies,fixed,dynamic,diagonal:download(device,op.diagonal)?,
            vertex_body:download(device,reduction.staged.vbody.handle())?,
            cloth_offset:download(device,reduction.staged.cloth_off.handle())?,
            rotated_rest:download(device,reduction.staged.prot.handle())?,
            smooth:download(device,exported.handle())?};
        snapshot.validate().map_err(fatal)?;
        let skew=snapshot.literal_skew();
        for (family,d) in ["diagonal_C","fixed_diagonal","dynamic_diagonal","smooth_isotropic_literal_h"].iter().zip(skew) {
            eprintln!("HARD_COEFFICIENT_LITERAL_SKEW family={family} maximum_absolute={:.12e} coefficient_scale={:.12e} block={} row={} column={} scope=literal_frozen_coefficients_no_average",d.maximum_absolute,d.coefficient_scale,d.block,d.row,d.column);
        }
        // Opt-in exact coefficient evidence for main's native probe. This is
        // sparse storage/factor extraction, never a production dense surrogate.
        if std::env::var_os("PPF_HARD_COEFFICIENT_CAPTURE").is_some() {
            let csr=|c:&Csr|serde_json::json!({"offset":c.offset,"index":c.index,"value":c.value,
                "transpose_offset":c.transpose_offset,"transpose_index":c.transpose_index,"transpose_slot":c.transpose_slot});
            let smooth:Vec<_>=snapshot.smooth.iter().map(|f|serde_json::json!({
                "body_a":f.body_a,"body_b":f.body_b,"ball_active":f.ball_active,"angular_active":f.angular_active,
                "control_active":f.control_active,"fault":f.fault,"p_a":f.p_a,"p_b":f.p_b,"ball_k":f.ball_k,
                "angular_h":f.angular_h,"control_j":f.control_j,"control_h":f.control_h})).collect();
            let evidence=serde_json::json!({"kind":"hard_frozen_literal_coefficients","rows":snapshot.rows,"dim":snapshot.dim,
                "body_base":snapshot.body_base,"bodies":snapshot.bodies,"fixed":csr(&snapshot.fixed),
                "dynamic":snapshot.dynamic.as_ref().map(csr),"diagonal":snapshot.diagonal,
                "vertex_body":snapshot.vertex_body,"cloth_offset":snapshot.cloth_offset,"rotated_rest":snapshot.rotated_rest,"smooth":smooth,
                "scope":"exact original float sparse coefficients and shared neutral float factors; no basis reconstruction"});
            eprintln!("HARD_FROZEN_COEFFICIENT_JSON {evidence}");
        }
        Ok(snapshot)
    }

}

