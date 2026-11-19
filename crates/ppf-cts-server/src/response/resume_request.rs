//! Native-authored accepted horizon only. No support inferred from filenames.
use std::path::PathBuf;
use serde_json::Value;
use ppf_cts_formats::{files::{session_output_dir,session_dir},status};
pub(crate) fn validate(doc:&Value,authority:&Value,launch_id:&str,sealed:i32,active_from:Option<i32>,active_target:Option<i32>)->bool {
 let checked=||->Option<()>{
 let from=doc["restored_frame"].as_i64()?;let target=doc["requested_target_frame"].as_i64()?;
 let scene=authority["scene_uuid"].as_str()?;let digest=authority["scene_inputs_sha256"].as_str()?;
 if authority["schema"]!="ppf.build_contract.v1" || scene.is_empty() || digest.len()!=64
  || doc["scene_uuid"].as_str()?!=scene || doc["scene_inputs_sha256"].as_str()?!=digest
  || doc["kind"]!="ppf.bone_resume_request" || doc["version"]!=1 || doc["feature"]!="bone_checkpoint_resume_v1" || doc["state"]!="ACCEPTED"
  || doc["sealed_target_frame"].as_i64()?!=sealed as i64 || from<=0 || target<=from || target<=sealed as i64 || target>i32::MAX as i64
  || doc["launch_id"].as_str()?!=launch_id || !doc["restored_time_seconds"].as_f64()?.is_finite() {return None}
 match(active_from,active_target){
  (Some(f),Some(t)) if from==f as i64 && target==t as i64=>{},
  (None,None)=>{},
  _=>return None,
 }
 Some(())};checked().is_some()
}
pub(crate) fn accepted(root:&str,sealed:i32,active_from:Option<i32>,active_target:Option<i32>)->Option<Value>{
 let out=session_output_dir(&PathBuf::from(root));
 let doc:Value=serde_json::from_str(&std::fs::read_to_string(out.join("resume_request_native_receipt.json")).ok()?).ok()?;
 let record=status::read(&out).ok()??;
 let authority:Value=serde_json::from_str(&std::fs::read_to_string(session_dir(&PathBuf::from(root)).join("build_contract_receipt.json")).ok()?).ok()?;
 if !validate(&doc,&authority,&record.launch_id,sealed,active_from,active_target){return None}
 Some(doc)
}
