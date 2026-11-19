//! Derive a migration projection from raw sealed CBOR, never typed ParamPayload.
use std::collections::{BTreeMap, BTreeSet};
use ciborium::Value;
use ppf_cts_formats::{Envelope, envelope::from_cbor, kinds::KIND_PARAM};

const LEGACY: &str = "parent_local_swing_twist_x_v1";
const FIELDS: [&str; 3] = ["angular_control_v1", "angular_soft_limits_v1", "angular_drives_v1"];

fn field<'a>(v: &'a Value, key: &str) -> Result<Option<&'a Value>, &'static str> {
    let Value::Map(items) = v else { return Err("expected map") };
    let mut found = None;
    for (k, value) in items {
        if k.as_text() == Some(key) {
            if found.is_some() { return Err("duplicate key") }
            found = Some(value);
        }
    }
    Ok(found)
}
fn need<'a>(v: &'a Value, k: &str) -> Result<&'a Value, &'static str> { field(v,k)?.ok_or("missing field") }
fn text<'a>(v: &'a Value) -> Result<&'a str, &'static str> { v.as_text().ok_or("expected text") }
fn array(v: &Value) -> Result<&Vec<Value>, &'static str> { v.as_array().ok_or("expected array") }
fn id(v: &Value) -> Result<String, &'static str> {
    let s = text(v)?;
    let parsed = uuid::Uuid::parse_str(s).map_err(|_| "invalid uuid")?;
    if parsed.to_string() != s { return Err("noncanonical uuid") }
    Ok(s.to_owned())
}

/// Return only authorized old model fields for actual PDRD/PDRD joints.
/// This is not proof of solver capability or of the complete Param equality.
pub(super) fn derive_sealed_models(param: &[u8], carrier: &[u8]) -> Result<Vec<serde_json::Value>, &'static str> {
    let param: Value = from_cbor(KIND_PARAM,param).map_err(|_| "invalid param envelope")?;
    let env: Envelope<Value> = ciborium::from_reader(carrier).map_err(|_| "invalid carrier envelope")?;
    if env.kind != "ppf.pdrd_ball_anchor" || !(2..=7).contains(&env.version) { return Err("invalid carrier kind or version") }
    let mut bodies = BTreeSet::new();
    for b in array(need(&env.payload,"bodies")?)? {
        if !bodies.insert(id(need(b,"body_uuid")?)?) { return Err("duplicate body") }
    }
    let mut joints = BTreeMap::new();
    for j in array(need(&env.payload,"joints")?)? {
        let joint = id(need(j,"joint_uuid")?)?;
        let parent = id(need(j,"parent_body_uuid")?)?;
        let child = id(need(j,"child_body_uuid")?)?;
        let enabled = match need(j,"enabled")? { Value::Bool(v)=>*v, _=>return Err("invalid joint enabled") };
        let eligible = enabled && bodies.contains(&parent) && bodies.contains(&child);
        if joints.insert(joint, (eligible, parent, child, j)).is_some() { return Err("duplicate joint") }
    }
    let bindings = array(need(need(&param,"blender_bones_v1")?,"bindings")?)?;
    let mut seen = BTreeSet::new();
    let mut result = Vec::new();
    for row in bindings {
        let Some(joint) = field(row,"joint_uuid")? else { continue };
        let joint = id(joint)?;
        if !seen.insert(joint.clone()) { return Err("duplicate binding") }
        let Some((eligible,parent,child,carrier_joint)) = joints.get(&joint) else { continue };
        if !eligible { continue }
        match field(row,"enabled")? { Some(Value::Bool(true)) | None => (), _ => return Err("binding enabled mismatch") }
        if id(need(row,"parent_body_uuid")?)? != *parent || id(need(row,"body_uuid")?)? != *child {
            return Err("binding endpoint mismatch")
        }
        let physics = need(row,"physics")?;
        let Some(control) = field(physics,"angular_control_v1")? else { continue };
        if text(need(control,"coordinate_model")?)? != LEGACY { continue }
        let mut models = serde_json::Map::new();
        for key in FIELDS {
            let authored = field(physics,key)?;
            let sealed = field(carrier_joint,key)?;
            match (authored, sealed) {
                (None, None) => (),
                (Some(a), Some(b)) if text(need(a,"coordinate_model")?)? == LEGACY
                    && text(need(b,"coordinate_model")?)? == LEGACY => {
                    models.insert(key.into(), LEGACY.into());
                },
                _ => return Err("carrier model mismatch"),
            }
        }
        result.push(serde_json::json!({"joint_uuid": joint, "models": models}));
    }
    if result.is_empty() { return Err("no eligible joints") }
    Ok(result)
}

/// Explicit read-only query. A contract is usable only for the exact same
/// project generation and selected backend that was probed.
pub(crate) async fn query(engine: &crate::engine::ServerEngine) -> Result<serde_json::Value, &'static str> {
    let before = engine.state();
    if before.root.is_empty() || before.upload_id.is_empty() || before.param_hash.is_empty() {
        return Err("TWIST_RESUME_GENERATION_UNAVAILABLE")
    }
    let root = std::path::Path::new(&before.root);
    let scene = ppf_cts_formats::files::session_dir(root);
    let backend = engine.config().solver_backend.clone();
    if crate::upload::read_upload_id(root).ok().as_deref() != Some(before.upload_id.as_str()) {
        return Err("PROJECT_CONTEXT_CHANGED")
    }
    let param_path = root.join("param.pickle");
    let carrier_path = scene.join("bin/pdrd_ball_anchor.cbor");
    fn read_bounded(path: &std::path::Path) -> Result<Vec<u8>, &'static str> {
        let metadata=std::fs::symlink_metadata(path).map_err(|_| "TWIST_RESUME_INPUT_UNAVAILABLE")?;
        if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len()>64*1024*1024 {
            return Err("TWIST_RESUME_INPUT_UNAVAILABLE")
        }
        std::fs::read(path).map_err(|_| "TWIST_RESUME_INPUT_UNAVAILABLE")
    }
    let param=read_bounded(&param_path)?;
    let carrier=read_bounded(&carrier_path)?;
    let sealed_models=derive_sealed_models(&param,&carrier).map_err(|_| "TWIST_RESUME_INPUT_INVALID")?;
    let capability=super::trusted_twist_probe::probe(&scene).await?;
    if capability["selected_backend"].as_str()!=Some(backend.as_str()) {
        return Err("TWIST_RESUME_BACKEND_MISMATCH")
    }
    let after = engine.state();
    if before.name!=after.name || before.root!=after.root || before.upload_id!=after.upload_id || before.param_hash!=after.param_hash || before.data_hash!=after.data_hash
        || engine.config().solver_backend!=backend || crate::upload::read_upload_id(root).ok().as_deref()!=Some(before.upload_id.as_str()) {
        return Err("PROJECT_CONTEXT_CHANGED")
    }
    Ok(serde_json::json!({"version":1,"supported_modes":capability["supported_modes"],
        "sealed_models":sealed_models,"name":before.name,"root":before.root,"upload_id":before.upload_id,"param_hash":before.param_hash,
        "data_hash":before.data_hash,"solver_backend":backend}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ppf_cts_formats::envelope::{to_cbor,to_cbor_with_version};
    use serde_json::json;
    fn setup() -> (Vec<u8>,Vec<u8>) {
        let joint = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa";
        let parent = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb";
        let child = "cccccccc-cccc-4ccc-8ccc-cccccccccccc";
        let param=json!({"blender_bones_v1":{"bindings":[{"joint_uuid":joint,
            "parent_body_uuid":parent,"body_uuid":child,"physics":{"angular_control_v1":{"coordinate_model":LEGACY,"stiffness_Nm_per_rad":1.0}}}]},
            "unrelated":{"preserve":true}});
        let carrier=json!({"bodies":[{"body_uuid":parent},{"body_uuid":child}],
            "joints":[{"joint_uuid":joint,"parent_body_uuid":parent,"child_body_uuid":child,"enabled":true,"angular_control_v1":{"coordinate_model":LEGACY}}]});
        (to_cbor(KIND_PARAM,&param).unwrap(),to_cbor_with_version(4,"ppf.pdrd_ball_anchor",&carrier).unwrap())
    }
    #[test]
    fn sealed_twist_contract_matches_raw_sealed_joint() {
        let (p,c)=setup(); let result=derive_sealed_models(&p,&c).unwrap();
        assert_eq!(result.len(),1);
        assert_eq!(result[0]["models"]["angular_control_v1"],LEGACY);
    }
    #[test]
    fn sealed_twist_contract_rejects_wrong_carrier() {
        let (p,_)=setup();
        let bad=to_cbor_with_version(4,"other",&json!({})).unwrap();
        assert!(derive_sealed_models(&p,&bad).is_err());
    }
    #[test]
    fn sealed_twist_contract_rejects_endpoint_and_model_drift() {
        let (p,c)=setup();
        let mut param: serde_json::Value = from_cbor(KIND_PARAM,&p).unwrap();
        param["blender_bones_v1"]["bindings"][0]["body_uuid"] = json!("dddddddd-dddd-4ddd-8ddd-dddddddddddd");
        assert!(derive_sealed_models(&to_cbor(KIND_PARAM,&param).unwrap(),&c).is_err());
        let mut carrier: Envelope<serde_json::Value> = ciborium::from_reader(c.as_slice()).unwrap();
        carrier.payload["joints"][0]["angular_control_v1"]["coordinate_model"] = json!("unexpected");
        let bad=to_cbor_with_version(4,"ppf.pdrd_ball_anchor",&carrier.payload).unwrap();
        assert!(derive_sealed_models(&p,&bad).is_err());
    }
    #[test]
    fn sealed_twist_contract_preserves_unselected_and_rejects_duplicate() {
        let (p,c)=setup();
        let mut param: serde_json::Value = from_cbor(KIND_PARAM,&p).unwrap();
        let row=param["blender_bones_v1"]["bindings"][0].clone();
        let mut non_directional=row.clone();
        non_directional["joint_uuid"]=json!("eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee");
        non_directional["physics"]=json!({});
        param["blender_bones_v1"]["bindings"].as_array_mut().unwrap().push(non_directional.clone());
        let mut carrier: Envelope<serde_json::Value> = ciborium::from_reader(c.as_slice()).unwrap();
        let mut other=carrier.payload["joints"][0].clone();
        other["joint_uuid"]=non_directional["joint_uuid"].clone();
        other.as_object_mut().unwrap().remove("angular_control_v1");
        carrier.payload["joints"].as_array_mut().unwrap().push(other);
        let c2=to_cbor_with_version(4,"ppf.pdrd_ball_anchor",&carrier.payload).unwrap();
        assert_eq!(derive_sealed_models(&to_cbor(KIND_PARAM,&param).unwrap(),&c2).unwrap().len(),1);
        param["blender_bones_v1"]["bindings"].as_array_mut().unwrap().push(row);
        assert!(derive_sealed_models(&to_cbor(KIND_PARAM,&param).unwrap(),&c2).is_err());
    }

    #[test]
    fn sealed_twist_contract_excludes_synthetic_endpoint() {
        let (p,c)=setup();
        let mut param: serde_json::Value=from_cbor(KIND_PARAM,&p).unwrap();
        let mut row=param["blender_bones_v1"]["bindings"][0].clone();
        row["joint_uuid"]=json!("eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee");
        row["body_uuid"]=json!("dddddddd-dddd-4ddd-8ddd-dddddddddddd");
        param["blender_bones_v1"]["bindings"].as_array_mut().unwrap().push(row);
        let mut carrier: Envelope<serde_json::Value>=ciborium::from_reader(c.as_slice()).unwrap();
        let mut synthetic=carrier.payload["joints"][0].clone();
        synthetic["joint_uuid"]=json!("eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee");
        synthetic["child_body_uuid"]=json!("dddddddd-dddd-4ddd-8ddd-dddddddddddd");
        carrier.payload["joints"].as_array_mut().unwrap().push(synthetic);
        let c2=to_cbor_with_version(4,"ppf.pdrd_ball_anchor",&carrier.payload).unwrap();
        assert_eq!(derive_sealed_models(&to_cbor(KIND_PARAM,&param).unwrap(),&c2).unwrap().len(),1);
    }

    #[test]
    fn sealed_twist_contract_rejects_disabled_binding_and_skips_disabled_carrier() {
        let (p,c)=setup();
        let mut param: serde_json::Value=from_cbor(KIND_PARAM,&p).unwrap();
        param["blender_bones_v1"]["bindings"][0]["enabled"]=json!(false);
        assert!(derive_sealed_models(&to_cbor(KIND_PARAM,&param).unwrap(),&c).is_err());
        let mut carrier: Envelope<serde_json::Value>=ciborium::from_reader(c.as_slice()).unwrap();
        carrier.payload["joints"][0]["enabled"]=json!(false);
        let c2=to_cbor_with_version(4,"ppf.pdrd_ball_anchor",&carrier.payload).unwrap();
        assert!(derive_sealed_models(&p,&c2).is_err()); // no eligible joints, disabled is not sampled
    }

}
