//! Additional v2 binder over retained scene bytes and actual native inventory.
use super::*;
use crate::fem_control_events::{self as events,Mode,Body};
#[derive(Clone,Debug)]
pub struct Prepared {pub bodies:Vec<Body>,pub program_sha256:String,pub positions:Vec<[f64;3]>,pub material_bindings:Vec<crate::soft_pose_output::MaterialBinding>}
fn invalid(detail:&str)->JointError{joint::err("fem","FEM_CONTROL_EVENTS_INVALID","fem_body_controls_v2",None,None,detail)}
fn initial_velocities(bytes:&[u8],vertex_count:usize)->Result<Vec<f32>,JointError>{
    // Match Scene::load read_mat_from_file::<f32,3>, not vert.bin's f64
    // representation. Scene asserts exactly N columns; empty is not default 0.
    let expected=vertex_count.checked_mul(12).ok_or_else(||invalid("native initial velocity byte extent overflow"))?;
    if bytes.len()!=expected{return Err(invalid("native initial velocity requires exactly 3N finite f32 values"));}
    let values:Vec<_>=bytes.chunks_exact(4).map(|x|f32::from_le_bytes(x.try_into().unwrap())).collect();
    if values.iter().any(|x|!x.is_finite()){return Err(invalid("native initial velocity contains nonfinite f32"));}
    Ok(values)
}
pub(super) fn validate(input:&ValidatedJointInput,p:&Payload,pos:&[Vector3<f64>],tet:&[u32],membership:&HashMap<String,Vec<u32>>,rigid:&[u32],n_pin:i64)->Result<Option<Prepared>,JointError>{
    let present=input.contains(events::FILE);
    if present!=input.requires_feature(events::FEATURE)?||present&&(input.contains(CONTROL_FILE)||input.requires_feature(CONTROL_FEATURE)?){return Err(invalid("v2 file/feature pair required and v1/v2 mutually exclusive"));}
    if !present{return Ok(None);}
    if !cfg!(cuda_backend){return Err(invalid("FEM event program requires matching CUDA native build"));}
    let bytes=input.bytes(events::FILE)?;
    if bytes.len()>16*1024*1024{return Err(invalid("sidecar budget exceeded"));}
    let strict:joint::StrictValue=serde_json::from_slice(bytes).map_err(|_|invalid("strict JSON with duplicate fields forbidden"))?;
    let wire:events::ProgramWire=serde_json::from_value(strict.0).map_err(|_|invalid("closed typed v2 program required"))?;
    let bodies=wire.validate(&p.scene_uuid).map_err(|e|invalid(&e))?;
    let mut all=HashSet::new();let mut bindings=Vec::new();
    let positions:Vec<[f64;3]>=pos.iter().map(|v|[v.x,v.y,v.z]).collect();
    let velocities=initial_velocities(input.bytes("bin/vel.bin")?,positions.len())?;
    for body in &bodies {
        let record=p.bodies.iter().find(|b|b.body_uuid==body.body_uuid&&b.kind=="FEM_TET_MATERIAL").ok_or_else(||invalid("root must be actual material FEM body"))?;
        if p.joints.iter().any(|j|j.wire.enabled&&j.wire.child_body_uuid==body.body_uuid)||input.records().joints.iter().any(|j|j.wire.enabled&&j.wire.child_body_uuid==body.body_uuid){return Err(invalid("controlled body must be graph root"));}
        let mut actual=membership.get(&body.body_uuid).ok_or_else(||invalid("native body membership absent"))?.clone();actual.sort_unstable();actual.dedup();
        if actual!=body.vertices{return Err(invalid("exact whole native membership required"));}
        let set:HashSet<_>=actual.iter().copied().collect();let mut covered=HashSet::new();
        if membership.iter().any(|(owner,v)|owner!=&body.body_uuid&&v.iter().any(|v|set.contains(v))){return Err(invalid("controlled vertex claimed by another body"));}
        for t in tet.chunks_exact(4){if t.iter().any(|v|set.contains(v)){if !t.iter().all(|v|set.contains(v)){return Err(invalid("tet crosses controlled body boundary"));}covered.extend(t.iter().copied());}}
        if covered!=set{return Err(invalid("every full body member must be real tet node"));}
        for &v in &actual {
            if v as usize>=positions.len()||rigid.get(v as usize)!=Some(&0)||!all.insert(v){return Err(invalid("exclusive ordinary native vertices required"));}
            if body.initial_mode!=Mode::Free&&velocities[3*v as usize..3*v as usize+3].iter().any(|v|*v!=0.){return Err(invalid("initial prescribed FEM cannot carry nonzero initial velocity"));}
        }
        let binding=crate::soft_pose_output::MaterialBinding{body_uuid:body.body_uuid.clone(),tet_index:record.tet_index.ok_or_else(||invalid("material tet absent"))?,vertex_indices:record.vertex_indices.ok_or_else(||invalid("material support absent"))?,dm_inv_column_major:record.dm_inv_column_major.ok_or_else(||invalid("Dm inverse absent"))?,center_bary:record.center_bary.ok_or_else(||invalid("material center absent"))?,frame_rest_column_major:record.frame_rest_column_major.ok_or_else(||invalid("material rest frame absent"))?};
        bindings.push(binding);
    }
    for i in 0..n_pin {let ids=indices64(input.bytes(&format!("bin/pin-ind-{i}.bin"))?)?;if ids.iter().any(|v|all.contains(v)){return Err(invalid("any legacy pin overlap with event inventory is forbidden"));}}
    events::Runtime::new(bodies.clone(),&positions,"sealed-intake-validation".into()).map_err(|e|invalid(&e))?;
    Ok(Some(Prepared{bodies,program_sha256:joint::sha256(bytes),positions,material_bindings:bindings}))
}
