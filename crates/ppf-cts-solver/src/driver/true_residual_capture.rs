//! One opt-in frozen first-claim diagnostic. Never changes acceptance criteria.
use super::*;
use ppf_cts_compute::Handle;
use std::path::Path;

fn fault(message: impl std::fmt::Display) -> Fault {
    Fault::Platform {call:"true residual frozen capture",detail:message.to_string()}
}

fn write_bytes(dir:&Path,name:&str,bytes:&[u8])->Result<(),Fault>{
    use std::io::Write;
    let mut file=std::fs::OpenOptions::new().write(true).create_new(true)
        .open(dir.join(name)).map_err(fault)?;
    file.write_all(bytes).map_err(fault)?;
    file.sync_all().map_err(fault)
}

fn dump<D:Device>(device:&mut D,dir:&Path,files:&mut Vec<serde_json::Value>,
    name:&str,handle:Handle,count:usize,stride:usize,dtype:&str)->Result<Vec<u8>,Fault>{
    let bytes=count.checked_mul(stride).ok_or_else(||fault("byte extent overflow"))?;
    if bytes>256*1024*1024 || (handle.size as usize)<count {return Err(fault("capture extent refused"));}
    let mut data=vec![0u8;bytes];
    if bytes>0 {device.read(handle,0,&mut data)?;}
    write_bytes(dir,name,&data)?;
    let finite=if dtype=="f32" {Some(data.chunks_exact(4).all(|v|f32::from_le_bytes(v.try_into().unwrap()).is_finite()))}else{None};
    files.push(serde_json::json!({"file":name,"dtype":dtype,"elements":count,"stride":stride,
        "bytes":bytes,"handle":format!("{:?}",handle),"all_finite":finite,"sha256":crate::joint::sha256(&data)}));
    Ok(data)
}

fn dump_slice(dir:&Path,files:&mut Vec<serde_json::Value>,name:&str,values:&[f32])->Result<(),Fault>{
    let bytes=unsafe{std::slice::from_raw_parts(values.as_ptr().cast::<u8>(),4*values.len())};
    write_bytes(dir,name,bytes)?;
    files.push(serde_json::json!({"file":name,"dtype":"f32","elements":values.len(),"stride":4,
        "bytes":bytes.len(),"all_finite":values.iter().all(|v|v.is_finite()),"sha256":crate::joint::sha256(bytes)}));
    Ok(())
}

/// Independent local storage and private fault lanes. Source matrix, x, f and
/// source joint faults are never diagnostic destinations. All allocations free
/// on success and partial failure. Aggregate locks are explicitly out of scope
/// for this focused actual soft-chain capture, not silently approximated.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn capture<D:Device>(device:&mut D,op:&Operator,
    reduction:&mut super::super::pdrd::Reduction<'_>,joints:super::super::joints::View,
    roots:Option<super::super::roots::View>,no_aggregate_lock:bool,
    xr:Handle,rhs:Handle,true_r:Handle,recursive:&[f32],pre_finish_x:&[f32],
    initial:&[f32],err0:f32,iteration:u32,dir:&Path,scratch_identities:serde_json::Value)->Result<(),Fault>{
    if !cfg!(target_endian="little") || !dir.is_absolute() || !no_aggregate_lock ||
        joints.count!=0 || joints.control_count!=0 || joints.extension_count!=0 {
        return Err(fault("capture requires little-endian, absolute fresh path, material-only joints and no aggregate lock"));
    }
    let rows=op.rows() as usize;let dim=reduction.map.dim;
    if rows>20_000 || dim>100_000 {return Err(fault("focused capture row limit"));}
    std::fs::create_dir_all(dir).map_err(fault)?;
    let mut files=Vec::new();
    let mut full=Buffer::<f32>::default();let mut applied=Buffer::<f32>::default();
    let mut absolute=Buffer::<f32>::default();let mut curvature=Buffer::<f32>::default();
    let mut base=Buffer::<f32>::default();let mut first=Buffer::<f32>::default();let mut second=Buffer::<f32>::default();
    let mut soft_fault=Buffer::<u32>::default();let mut joint_ids=Buffer::<u32>::default();
    let mut joint_counts=Buffer::<u32>::default();let mut export_fault=Buffer::<u32>::default();
    let mut hessian=Buffer::<f32>::default();
    let mut history=Buffer::<crate::restoring_curve_gpu::CurveHistory>::default();
    let mut factor_response=Buffer::<f32>::default();let mut factor_fault=Buffer::<u32>::default();
    let mut non_tet_product=Buffer::<f32>::default();let mut tet_product=Buffer::<f32>::default();
    let mut joint_product=Buffer::<f32>::default();

    let result=(||->Result<(),Fault>{
        for (buffer,n) in [(&mut full,3*rows),(&mut applied,3*rows),(&mut absolute,rows),
            (&mut curvature,rows),(&mut base,dim),(&mut first,dim),(&mut second,dim)] {
            buffer.size(device,n,AllocLabel("pcg.true_capture.private"))?;
        }
        let snapshot=op.tet_factor.map(|v|v.capture_snapshot());
        let capture_factor=if let Some(view)=op.tet_factor {
            let snap=view.capture_snapshot();
            if snap.rows as usize!=rows {return Err(fault("factor/full row mismatch"));}
            factor_response.size(device,(11*snap.tets as usize).max(1),AllocLabel("pcg.v5.factor_response"))?;
            factor_fault.size(device,rows.max(snap.tets as usize).max(1),AllocLabel("pcg.v5.factor_fault"))?;
            factor_fault.write(device,0,&vec![0u32;rows.max(snap.tets as usize).max(1)])?;
            Some(view.with_capture_scratch(factor_response.handle(),factor_fault.handle()))
        }else{None};
        let fixed_view=||super::super::spmv::FixedCsrView{rows:op.fixed.rows,
            offset:op.fixed.offset,index:op.fixed.index,value:op.fixed.value,
            transpose_pair:op.fixed.transpose_pair,transpose_offset:op.fixed.transpose_offset};
        let capture_op=Operator{fixed_pattern: None, tet_factor:capture_factor,dynamic:op.dynamic,fixed:fixed_view(),diagonal:op.diagonal};
        let mut non_tet_fixed=fixed_view();
        if let Some(snap)=snapshot {non_tet_fixed.value=snap.non_tet;}
        let non_tet_op=Operator{fixed_pattern: None, tet_factor:None,dynamic:op.dynamic,fixed:non_tet_fixed,diagonal:op.diagonal};
        non_tet_product.size(device,3*rows,AllocLabel("pcg.v5.non_tet_Ax"))?;
        tet_product.size(device,3*rows,AllocLabel("pcg.v5.tet_Ax"))?;
        joint_product.size(device,dim,AllocLabel("pcg.v5.joint_Ax"))?;
        let x_before=dump(device,dir,&mut files,"xr-before.bin",xr,dim,4,"f32")?;
        let f_before=dump(device,dir,&mut files,"rhs-before.bin",rhs,dim,4,"f32")?;
        dump(device,dir,&mut files,"true-r.bin",true_r,dim,4,"f32")?;
        dump_slice(dir,&mut files,"recursive-r.bin",recursive)?;
        dump_slice(dir,&mut files,"pre-finish-x.bin",pre_finish_x)?;
        let private_joints=if let Some(soft)=joints.soft {
            soft_fault.size(device,(soft.count as usize).max(1),AllocLabel("pcg.true_capture.private_fault"))?;
            soft_fault.write(device,0,&vec![0u32;(soft.count as usize).max(1)])?;
            Some(super::super::soft_joints::View{fault:soft_fault.handle(),
                curves:soft.curves.map(|c|super::super::restoring::View{fem_fault:soft_fault.handle(),..c}),..soft})
        }else{None};
        let source_joints=super::super::joints::View{soft:private_joints,..joints};
        for (pass,destination) in [(1,first.handle()),(2,second.handle())] {
            reduction.prolong(device,xr,full.handle())?;
            capture_op.apply(device,full.handle(),applied.handle(),absolute.handle(),curvature.handle())?;
            reduction.restrict(device,applied.handle(),base.handle(),&vec![0f32;dim])?;
            if pass==1 {
                dump(device,dir,&mut files,"prolonged-x.bin",full.handle(),3*rows,4,"f32")?;
                dump(device,dir,&mut files,"global-full-Ax.bin",applied.handle(),3*rows,4,"f32")?;
                dump(device,dir,&mut files,"restricted-global-Ax.bin",base.handle(),dim,4,"f32")?;
                non_tet_op.apply(device,full.handle(),non_tet_product.handle(),absolute.handle(),curvature.handle())?;
                dump(device,dir,&mut files,"gpu-non-tet-Ax.bin",non_tet_product.handle(),3*rows,4,"f32")?;
                device.fill_zero(tet_product.handle(),3*rows*4)?;
                device.fill_zero(absolute.handle(),rows*4)?;device.fill_zero(curvature.handle(),rows*4)?;
                if let Some(factor)=capture_factor {
                    device.run("pcg.v5.private_factor_only",|e|factor.encode(e,full.handle(),tet_product.handle(),
                        curvature.handle(),absolute.handle()))?;
                }
                dump(device,dir,&mut files,"gpu-factor-only-Ax.bin",tet_product.handle(),3*rows,4,"f32")?;

            }
            combine(device,base.handle(),base.handle(),destination,1.0,0.0,dim)?;
            source_joints.add(device,1,xr,destination)?;
            if pass==1 {
                dump(device,dir,&mut files,"before-output-projection-Ax.bin",destination,dim,4,"f32")?;
            }
            reduction.project_bodies(device,destination)?;
            if let Some(root)=roots {root.project(device,destination)?;}
            dump(device,dir,&mut files,&format!("independent-Ax-{pass}.bin"),destination,dim,4,"f32")?;
        }
        device.fill_zero(joint_product.handle(),dim*4)?;
        source_joints.add(device,1,xr,joint_product.handle())?;
        dump(device,dir,&mut files,"gpu-joint-only-Ax.bin",joint_product.handle(),dim,4,"f32")?;
        if let Some(snap)=snapshot {
            dump(device,dir,&mut files,"non-tet-values.bin",snap.non_tet,snap.non_tet.size as usize,4,"f32")?;
            dump(device,dir,&mut files,"factor-tet-ids.bin",snap.tet,4*snap.tets as usize,4,"u32")?;
            dump(device,dir,&mut files,"factor-modes.bin",snap.factor,81*snap.tets as usize,4,"f32")?;
            dump(device,dir,&mut files,"factor-weights.bin",snap.weight,9*snap.tets as usize,4,"f32")?;
            dump(device,dir,&mut files,"factor-dof-mask.bin",snap.mask,rows,4,"u32")?;
            dump(device,dir,&mut files,"factor-source-fault.bin",snap.fault,
                rows.max(snap.tets as usize).max(1),4,"u32")?;
            dump(device,dir,&mut files,"factor-private-fault.bin",factor_fault.handle(),
                rows.max(snap.tets as usize).max(1),4,"u32")?;
        }
        if let Some(soft)=private_joints {
            let n=soft.count as usize;
            dump(device,dir,&mut files,"soft-records.bin",soft.records,n,std::mem::size_of::<crate::soft_joint_gpu::FemJointGpu>(),"FemJointGpu")?;
            dump(device,dir,&mut files,"soft-cache.bin",soft.cache,4*n,std::mem::size_of::<crate::soft_joint_gpu::FemFrameGpu>(),"FemFrameGpu")?;
            joint_ids.size(device,(24*n).max(1),AllocLabel("pcg.true_capture.joint_ids"))?;
            joint_counts.size(device,n.max(1),AllocLabel("pcg.true_capture.joint_counts"))?;
            hessian.size(device,(576*n).max(1),AllocLabel("pcg.true_capture.hessian"))?;
            export_fault.size(device,n.max(1),AllocLabel("pcg.true_capture.export_fault"))?;
            export_fault.write(device,0,&vec![0u32;n.max(1)])?;
            history.size(device,1,AllocLabel("pcg.true_capture.history"))?;
            history.write(device,0,&[crate::restoring_curve_gpu::CurveHistory::default()])?;
            if n>0 {
                let curves=soft.curves;let cached=soft.resident_cache_ready();
                let args=super::super::kernels::ReducedSchwarzJointExportArgs {
                    records:soft.records,material_spin_context:soft.material_spin_context,record_count:soft.count,cache:soft.cache,
                    vertex_to_free:soft.vertex_to_free,vertex_count:soft.vertices,free_count:soft.free_count,
                    rigid_base:soft.base,dim:soft.dim,index_cache:if cached{soft.index_cache}else{soft.dummy},
                    index_ready:(cached&&soft.index_ready)as u32,source_mode:cached as u32,
                    token:if cached{soft.hessian_epoch}else{0},hessian_dt:soft.hessian_dt,
                    limits:if soft.extension_count!=0{soft.limits}else{soft.dummy},
                    drives:if soft.drive_count!=0{soft.drives}else{soft.dummy},drive_count:soft.drive_count,
                    keys:if soft.drive_count!=0{soft.keys}else{soft.dummy},extension_count:soft.extension_count,
                    sample_time_s:soft.sample_time_s,dt:soft.dt,curves_enabled:curves.is_some()as u32,
                    descriptors:curves.map_or(soft.dummy,|c|c.descriptors),nodes:curves.map_or(soft.dummy,|c|c.nodes),
                    bindings:curves.map_or(soft.dummy,|c|c.bindings),history:curves.map_or(history.handle(),|c|c.history),
                    descriptor_count:curves.map_or(0,|c|c.descriptor_count),node_count:curves.map_or(0,|c|c.node_count),
                    binding_count:curves.map_or(0,|c|c.binding_count),history_count:curves.map_or(0,|c|c.history_count),
                    scalar_ids:joint_ids.handle(),local_hessian:hessian.handle(),scalar_count:joint_counts.handle(),
                    fault:export_fault.handle(),count:soft.count,seam_arena_count:0};
                device.launch("pcg.true_capture.material_hessian",&args,soft.count)?;
            }
            dump(device,dir,&mut files,"joint-scalar-ids.bin",joint_ids.handle(),24*n,4,"u32")?;
            dump(device,dir,&mut files,"joint-scalar-counts.bin",joint_counts.handle(),n,4,"u32")?;
            dump(device,dir,&mut files,"joint-hessian.bin",hessian.handle(),576*n,4,"f32")?;
            dump(device,dir,&mut files,"joint-export-fault.bin",export_fault.handle(),n,4,"u32")?;
            dump(device,dir,&mut files,"independent-soft-fault.bin",soft_fault.handle(),n,4,"u32")?;
        }
        let fixed=&op.fixed;
        for (name,h,dtype) in [("fixed-offset.bin",fixed.offset,"u32"),("fixed-index.bin",fixed.index,"u32"),
            ("fixed-values.bin",fixed.value,"f32"),("fixed-transpose-pair.bin",fixed.transpose_pair,"u32"),
            ("fixed-transpose-offset.bin",fixed.transpose_offset,"u32"),("diagonal.bin",op.diagonal,"f32")]{
            dump(device,dir,&mut files,name,h,h.size as usize,4,dtype)?;
        }
        if let Some(d)=op.dynamic {
            for (name,h,dtype) in [("dynamic-offset.bin",d.offset,"u32"),("dynamic-index.bin",d.index,"u32"),
                ("dynamic-values.bin",d.value,"f32"),("dynamic-transpose-offset.bin",d.transpose_offset,"u32"),
                ("dynamic-transpose-index.bin",d.transpose_index,"u32"),("dynamic-transpose-value.bin",d.transpose_value,"u32")]{
                dump(device,dir,&mut files,name,h,h.size as usize,4,dtype)?;
            }
        }
        dump(device,dir,&mut files,"vertex-body.bin",reduction.staged.vbody.handle(),rows,4,"u32")?;
        dump(device,dir,&mut files,"cloth-offset.bin",reduction.staged.cloth_off.handle(),rows,4,"u32")?;
        dump(device,dir,&mut files,"rotated-rest.bin",reduction.staged.prot.handle(),3*rows,4,"f32")?;
        if let Some(root)=roots {dump(device,dir,&mut files,"root-targets.bin",root.targets,root.count as usize,64,"RootTargetGpu")?;}
        let x_after=dump(device,dir,&mut files,"xr-after.bin",xr,dim,4,"f32")?;
        let f_after=dump(device,dir,&mut files,"rhs-after.bin",rhs,dim,4,"f32")?;
        let meta=serde_json::json!({"schema":"ppf.true-residual-first-claim.v5","diagnostic_only":true,
            "factor_active":snapshot.is_some(),"factor_tets":snapshot.map_or(0,|v|v.tets),
            "actual_operator":"non-tet CSR + dynamic CSR + diagonal + E^T F^T W F E + material joint operator, with original output projections",
            "fixed_values_role":"assembled FP32 roundoff approximation for preconditioner/reference; not actual factor A",
            "factor_modes_layout":"81/tet: mode*9 + (vertex_slot-1)*3 + axis, slots1..3; weight9 includes mass and damping once",
            "diagnostic_scratch":"private factor response/fault; original coefficients/mask/x/rhs unchanged",
            "helper_source_file":file!(),"compiled_pcg_sha256":option_env!("PPF_PCG_CAPTURE_COMPILED_SOURCE_SHA256"),
            "compiled_helper_sha256":option_env!("PPF_CAPTURE_HELPER_COMPILED_SOURCE_SHA256"),
            "rows":rows,"dim":dim,"body_base":reduction.map.body_base,"bodies":reduction.map.n_bodies,
            "iteration":iteration,"err0":err0,"group_initial":initial,"xr_unchanged":x_before==x_after,
            "original_scratch_handles":scratch_identities,
            "body_projection":{"joint_mode":reduction.map.jmode,"joint_axis":reduction.map.jaxis,
                "translation_lock":reduction.map.tlock,"translation_axis":reduction.map.tlock_axis,
                "translation_mode":reduction.map.tlock_mode,"rotation_lock":reduction.map.rlock,
                "rotation_axis":reduction.map.rlock_axis,"rotation_mode":reduction.map.rlock_mode},
            "rhs_unchanged":f_before==f_after,"soft_count":joints.soft.map_or(0,|s|s.count),
            "root_count":roots.map_or(0,|r|r.count),"byte_order":"little","files":files,
            "hash_sealing":"Each raw file SHA256 sealed in this manifest; parent receipt must seal capture.json and exact compiled sources/exe/DLL"});
        write_bytes(dir,"capture.json",&serde_json::to_vec_pretty(&meta).map_err(fault)?)?;
        eprintln!("PCG_TRUE_RESIDUAL_CAPTURE {}",dir.display());
        Ok(())
    })();
    let mut cleanup_error=None;
    for buffer in [&mut full,&mut applied,&mut absolute,&mut curvature,&mut base,&mut first,&mut second,&mut hessian,&mut factor_response,&mut non_tet_product,&mut tet_product,&mut joint_product] {
        if let Err(e)=buffer.free(device){if cleanup_error.is_none(){cleanup_error=Some(e);}}
    }
    for buffer in [&mut soft_fault,&mut joint_ids,&mut joint_counts,&mut export_fault,&mut factor_fault] {
        if let Err(e)=buffer.free(device){if cleanup_error.is_none(){cleanup_error=Some(e);}}
    }
    if let Err(e)=history.free(device){if cleanup_error.is_none(){cleanup_error=Some(e);}}
    match result {Err(e)=>Err(e),Ok(())=>if let Some(e)=cleanup_error{Err(e)}else{Ok(())}}
}
