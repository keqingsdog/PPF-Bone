//! Opt-in same-frozen-state legacy diagnostic; never enabled in timed production runs.
//! Call immediately after combined export, before advancing/mutating the View.
use ppf_cts_compute::{AllocLabel,Buffer,Device,Fault};
use crate::driver::{joints,reduced_schwarz::FrozenJointBlockView};

fn refusal(s:impl Into<String>)->Fault{Fault::Platform{call:"legacy export same-state equivalence",detail:s.into()}}

pub unsafe fn run<D:Device>(device:&mut D,mut source:joints::View,
    blocks:FrozenJointBlockView,material_records:u32)->Result<serde_json::Value,Fault>{
    // Bounded diagnostic admission only, never a production solver routing rule.
    let dim=source.dim as usize;let first=material_records as usize;
    let records=(blocks.records as usize).checked_sub(first).ok_or_else(||refusal("material prefix exceeds combined export"))?;
    if dim==0||dim>512||records==0||records>256{return Err(refusal("diagnostic requires dim1..512 and legacy records1..256"));}
    let mut ids=vec![0u32;24*records];let mut counts=vec![0u32;records];let mut h=vec![0f32;576*records];
    let bytes=|p:*mut u8,n|std::slice::from_raw_parts_mut(p,n);
    device.read(blocks.scalar_ids,first*24*4,bytes(ids.as_mut_ptr().cast(),ids.len()*4))?;
    device.read(blocks.scalar_count,first*4,bytes(counts.as_mut_ptr().cast(),counts.len()*4))?;
    device.read(blocks.local_hessian,first*576*4,bytes(h.as_mut_ptr().cast(),h.len()*4))?;
    for r in 0..records{if counts[r]>24{return Err(refusal("bad scalar count"));}
        for &id in &ids[24*r..24*r+counts[r]as usize]{if id as usize>=dim{return Err(refusal("bad exported index"));}}}
    let mut input=Buffer::<f32>::default();let mut output=Buffer::<f32>::default();let mut fault=Buffer::<u32>::default();
    input.size(device,dim,AllocLabel("legacy_equivalence.input"))?;
    output.size(device,dim,AllocLabel("legacy_equivalence.output"))?;
    let nf=source.count.max(source.control_count).max(source.extension_count).max(1)as usize;
    fault.size(device,nf,AllocLabel("legacy_equivalence.private_fault"))?;
    source.soft=None;source.fault=fault.handle();
    if let Some(mut c)=source.curves{c.fault=fault.handle();c.ball_fault=fault.handle();c.angular_fault=fault.handle();
        c.fem_fault=fault.handle();c.stitch_fault=fault.handle();source.curves=Some(c);}
    let mut reports=Vec::new();let mut max_ratio=0f64;
    // Full basis proves every column, then two fixed mixed vectors exercise
    // combination/cancellation on the SAME frozen state/context and original A.
    for p in 0..dim+2{
        let mut x=vec![0f32;dim];
        if p<dim{x[p]=1.;}else{for i in 0..dim{x[i]=if p==dim{((7*i)%11)as f32/8.-5./8.}else{((5*i)%13)as f32/16.-6./16.};}}
        let mut expected=vec![0f64;dim];let mut absolute=vec![0f64;dim];
        for r in 0..records{let n=counts[r]as usize;
            for i in 0..n{let row=ids[24*r+i]as usize;
                for j in 0..n{let v=f64::from(h[576*r+24*i+j])*f64::from(x[ids[24*r+j]as usize]);
                    if !v.is_finite(){return Err(refusal("nonfinite exported product"));}
                    expected[row]+=v;absolute[row]+=v.abs();}}}
        input.write(device,0,&x)?;device.fill_zero(output.handle(),dim*4)?;device.fill_zero(fault.handle(),nf*4)?;
        // Actual unmodified legacy global action1, not a mirror formula.
        source.add(device,1,input.handle(),output.handle())?;
        let mut actual=vec![0f32;dim];output.read(device,0,&mut actual)?;
        let mut codes=vec![0u32;nf];fault.read(device,0,&mut codes)?;
        if codes.iter().any(|&c|c!=0){return Err(refusal(format!("old action1 refused: {codes:?}")));}
        let mut probe_ratio=0f64;
        for i in 0..dim{
            if !actual[i].is_finite(){return Err(refusal("old action1 nonfinite"));}
            let error=(f64::from(actual[i])-expected[i]).abs();
            // Preregistered componentwise backward-error gate. No absolute
            // unit-dependent floor: an exactly zero row must remain exact zero.
            let bound=512.*f64::from(f32::EPSILON)*absolute[i];
            let ratio=if bound==0.{if error==0.{0.}else{f64::INFINITY}}else{error/bound};
            if ratio>1.{return Err(refusal(format!("probe{p} scalar{i}: error{error:e} bound{bound:e} actual{} expected{}",actual[i],expected[i])));}
            probe_ratio=probe_ratio.max(ratio);
        }
        max_ratio=max_ratio.max(probe_ratio);reports.push(serde_json::json!({"probe":p,"max_component_error_ratio":probe_ratio}));
    }
    Ok(serde_json::json!({"schema":"ppf.legacy-export-equivalence.v1","scope":"original legacy action1 versus independent FP64 sum of exported frozen blocks, same immutable View",
        "dim":dim,"material_prefix_records":material_records,"legacy_records":records,"basis_columns":dim,"mixed_probes":2,
        "dt_bits":source.dt.to_bits(),"sample_time_bits":source.sample_time_s.to_bits(),"curves":source.curves.is_some(),
        "bound":"512*eps32*sum(abs(H_ij*x_j)) per component; zero requires exact zero","max_component_error_ratio":max_ratio,
        "history_and_observations":"original immutable context; action1 observation pointers nullptr; private diagnostic fault banks",
        "probes":reports}))
}
