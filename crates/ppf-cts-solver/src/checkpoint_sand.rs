//! Accepted rolling-grain history carried inside the marked State gzip member.
//! Format identity checks are needed to avoid attaching rotations to another
//! vertex layout or rolling policy; they are not a source/package audit.
use bincode::Options;
use std::{fs,path::Path};

pub(crate) const MARKER:&[u8]=b"PPF_RUNTIME_HISTORY_REQUIRED_V2_SAND_SPIN_V1";
pub(crate) const MAGIC:&[u8;8]=b"PPFSND02";
fn invalid(reason:&str)->String{format!("CHECKPOINT_SAND_INVALID: {reason}")}

#[derive(serde::Serialize,serde::Deserialize)]
struct Sand {
    version:u32, frame:i32, time_bits:u64, prev_dt_bits:u32,
    layout:Vec<u8>, policy:[u32;3], mask:Vec<u8>, omega:Vec<f32>,
}

/// None is the old no-grain path, which must remain byte-for-byte unaffected.
pub(crate) fn mask(data:&crate::data::DataSet)->Result<Option<Vec<u8>>,String>{
    let inertia=data.grain_inv_inertia.as_slice();
    if inertia.iter().any(|v|!v.is_finite()||*v<0.0) {return Err(invalid("invalid grain inertia"));}
    if !inertia.iter().any(|x|*x>0.0) {return Ok(None);}
    let n=data.vertex.curr.size as usize;
    if n==0 || inertia.len()!=n || data.vertex.prev.size as usize!=n
        || inertia.iter().any(|v|!v.is_finite()||*v<0.0) {
        return Err(invalid("invalid grain domain or size"));
    }
    Ok(Some(inertia.iter().map(|v|u8::from(*v>0.0)).collect()))
}
fn policy()->Result<[u32;3],String>{
    let a=crate::driver::sand_checkpoint_policy();
    if !f32::from_bits(a[0]).is_finite()||!f32::from_bits(a[1]).is_finite(){return Err(invalid("nonfinite rolling policy"));}
    Ok(a)
}
fn layout(dir:&Path)->Result<Vec<u8>,String>{
    let p=dir.join(crate::checkpoint_layout::MARKER);
    let meta=fs::symlink_metadata(&p).map_err(|e|invalid(&format!("layout: {e}")))?;
    if !meta.file_type().is_file() || meta.len()>1024 {return Err(invalid("invalid layout marker"));}
    fs::read(p).map_err(|e|invalid(&format!("layout: {e}")))
}
fn validate(mask:&[u8],omega:&[f32])->Result<(),String>{
    if mask.is_empty()||mask.iter().any(|x|*x>1)||!mask.contains(&1)
        || mask.len().checked_mul(3)!=Some(omega.len())
        || omega.chunks_exact(3).zip(mask).any(|(v,g)|v.iter().any(|x|!x.is_finite()||(*g==0 && *x!=0.0))) {
        return Err(invalid("grain mask, omega shape or nonfinite value"));
    }
    Ok(())
}
pub(crate) fn encode(dir:&Path,state:&crate::backend::State,mask:&[u8],omega:&[f32])->Result<Vec<u8>,String>{
    validate(mask,omega)?;
    if state.curr_frame<0 || !state.time.is_finite() || state.time<0.0 || !state.prev_dt.is_finite() || state.prev_dt<=0.0
        || state.curr_vertex.ncols()!=mask.len() || state.prev_vertex.ncols()!=mask.len(){return Err(invalid("state boundary"));}
    let record=Sand{version:1,frame:state.curr_frame,time_bits:state.time.to_bits(),prev_dt_bits:state.prev_dt.to_bits(),
        layout:layout(dir)?,policy:policy()?,mask:mask.to_vec(),omega:omega.to_vec()};
    bincode::serialize(&record).map_err(|e|invalid(&e.to_string()))
}
pub(crate) fn decode(bytes:&[u8],dir:&Path,state:&crate::backend::State,mask:&[u8])->Result<Vec<f32>,String>{
    // Validate each serialized Vec length against the already loaded dataset
    // and layout before serde can allocate from untrusted length prefixes.
    let current_layout=layout(dir)?;
    let n=mask.len();
    let policy_at=28usize.checked_add(current_layout.len()).ok_or_else(||invalid("length overflow"))?;
    let mask_len_at=policy_at.checked_add(12).ok_or_else(||invalid("length overflow"))?;
    let mask_at=mask_len_at.checked_add(8).ok_or_else(||invalid("length overflow"))?;
    let omega_len_at=mask_at.checked_add(n).ok_or_else(||invalid("length overflow"))?;
    let omega_at=omega_len_at.checked_add(8).ok_or_else(||invalid("length overflow"))?;
    let floats=n.checked_mul(3).ok_or_else(||invalid("length overflow"))?;
    let expected=omega_at.checked_add(floats.checked_mul(4).ok_or_else(||invalid("length overflow"))?).ok_or_else(||invalid("length overflow"))?;
    if bytes.len()!=expected {return Err(invalid("grain block length"));}
    let read_len=|at:usize|->u64{u64::from_le_bytes(bytes[at..at+8].try_into().unwrap())};
    if read_len(20)!=current_layout.len() as u64 || read_len(mask_len_at)!=n as u64 || read_len(omega_len_at)!=floats as u64 {
        return Err(invalid("grain block vector lengths"));
    }
    let record:Sand=bincode::DefaultOptions::new().with_fixint_encoding().with_limit(bytes.len() as u64)
        .reject_trailing_bytes().deserialize(bytes).map_err(|e|invalid(&e.to_string()))?;
    if record.version!=1 || record.frame!=state.curr_frame || record.time_bits!=state.time.to_bits()
        || record.prev_dt_bits!=state.prev_dt.to_bits() || record.layout!=current_layout
        || record.policy!=policy()? || record.mask!=mask || state.curr_vertex.ncols()!=mask.len()
        || state.prev_vertex.ncols()!=mask.len() {return Err(invalid("frame, time, layout, grain mask or rolling policy mismatch"));}
    validate(mask,&record.omega)?;Ok(record.omega)
}
