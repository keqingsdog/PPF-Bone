//! Read-only host command over the real retained native intake, never a solve.
//! main dispatches this before probe, parser, setup/status/output or constructors.
use crate::{connected_collision as cc, joint, soft_joint};
use serde_json::{json,Value};
use std::path::Path;
pub fn dispatch(args:&[String])->Option<i32> {
    match cc::parse_host_intake_args(args) {
        Ok(None)=>None,
        Err(e)=>{eprintln!("{}",json!({"kind":"ppf.bone_connected_collision.strict_intake","version":1,"status":"REFUSED","host_only":true,"error":e}));Some(2)},
        Ok(Some((path,load)))=>{
            match inspect(&path,load) {
                Ok(value)=>{println!("{value}");Some(0)},
                Err(e)=>{eprintln!("{}",json!({"kind":"ppf.bone_connected_collision.strict_intake","version":1,"status":"REFUSED","host_only":true,"error":e}));Some(2)}
            }
        }
    }
}
fn inspect(path:&Path,load:i32)->Result<Value,String> {
    // Same original history gate, before reading/pruning checkpoint state. No new-feature bypass.
    joint::reject_experimental_resume(load).map_err(|e|e.to_string())?;
    let filter=crate::collision_filter::prepare_fresh(path,false,load).map_err(|e|e.to_string())?;
    let input=joint::prepare_native_input(path).map_err(|e|e.to_string())?
        .ok_or_else(||"BONE_CONNECTED_HOST_INPUT: strict bone intake requires a real sealed joint carrier".to_string())?;
    // Explicit second prepare uses final validated carrier hash and executes typed FEM proof.
    // prepare_native_input already invokes this for root binding; do not report carrier-only validation.
    let material=soft_joint::prepare(&input).map_err(|e|e.to_string())?;
    let mut connected=Value::Null;
    if let Some(prepared)=material.as_ref().and_then(|s|s.connected_collision.as_ref()) {
        // Actual production ownership adapter and exact CSR constructor; no DataSet/device allocation.
        let (table,costs)=cc::build_reported(prepared.dynamic,prepared.static_count,&prepared.base,
            &prepared.proof.owners,&prepared.proof.pairs,&mut cc::Allocation::default())?;
        let marker=prepared.marker(&table)?;
        connected=json!({"marker":marker,"costs":costs,"owner_count":prepared.proof.owners.len(),
            "pair_count":prepared.proof.pairs.len(),"csr_rows":table.size,"csr_nnz":table.nnz});
    }
    input.verify_unchanged(path).map_err(|e|e.to_string())?;
    if let Some(filter)=filter {filter.verify_unchanged(path).map_err(|e|e.to_string())?;}
    Ok(json!({"kind":"ppf.bone_connected_collision.strict_intake","version":1,"status":"ACCEPTED",
        "host_only":true,"scene_uuid":input.records().scene_uuid,"scene_inputs_sha256":input.records().scene_inputs_sha256,
        "carrier_payload_sha256":input.records().payload_sha256,"legacy_joint_count":input.records().joints.len(),
        "material_joint_count":material.as_ref().map_or(0,|s|s.wires.len()),"material_body_count":material.as_ref().map_or(0,|s|s.bodies.len()),
        "required_feature":cc::FEATURE,"connected_collision":connected,
        "fem_body_controls":material.as_ref().map(|s|&s.control_receipt),
        "retained_snapshot_verified":true,"material_prepare_executed":true,"device_initialized":false,
        "gpu_execution":false,"physics_acceptance":"not_assessed","resume_supported":false}))
}
