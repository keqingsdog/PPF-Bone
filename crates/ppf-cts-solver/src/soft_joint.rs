//! Sealed material-frame joint input. FEM bodies remain ordinary physical nodes.
use crate::{joint::{self, JointError, JointWire, ValidatedJointInput}, soft_joint_gpu::*};
use nalgebra::{Matrix3, Vector3};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

pub const CONTROL_FEATURE:&str="fem_prescribed_roots_v1";
pub const CONTROL_FILE:&str="bin/fem_body_controls_v1.json";
#[path="fem_body_control.rs"] mod fem_body_control;
#[path="fem_control_intake.rs"] pub(crate) mod fem_control_intake;
pub const FILE: &str = "bin/fem_bone_joints.json";
pub const FEATURE: &str = "fem_material_joints_v1";
pub const KIND: &str = "ppf.fem_bone_joints";
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BodyWire {
    pub body_uuid: String, pub kind: String,
    #[serde(default, skip_serializing_if="Option::is_none")] pub runtime_body_index: Option<u32>,
    #[serde(default, skip_serializing_if="Option::is_none")] pub tet_index: Option<u32>,
    #[serde(default, skip_serializing_if="Option::is_none")] pub vertex_indices: Option<[u32;4]>,
    #[serde(default, skip_serializing_if="Option::is_none")] pub dm_inv_column_major: Option<[f64;9]>,
    #[serde(default, skip_serializing_if="Option::is_none")] pub center_bary: Option<[f64;4]>,
    #[serde(default, skip_serializing_if="Option::is_none")] pub frame_rest_column_major: Option<[f64;9]>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointRow {
    pub wire: JointWire,
    #[serde(default, skip_serializing_if="Option::is_none")] pub parent_anchor_bary: Option<[f64;4]>,
    #[serde(default, skip_serializing_if="Option::is_none")] pub child_anchor_bary: Option<[f64;4]>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Payload { pub version:u32, pub kind:String, pub scene_uuid:String, pub bodies:Vec<BodyWire>, pub joints:Vec<JointRow> }
#[derive(Debug, Clone)]
pub struct OwnedSoftScene {
    pub scene_uuid:String, pub scene_inputs_sha256:String, pub payload_sha256:String,
    pub controlled_vertices:Vec<u32>, pub control_receipt:serde_json::Value,
    pub control_events:Option<fem_control_intake::Prepared>,
    pub bodies:Vec<BodyWire>, pub wires:Vec<JointRow>, pub records:Vec<FemJointGpu>,
    pub extensions:Option<crate::joint_angular_extension_gpu::PackedAngularExtensionV1>,
    pub connected_collision:Option<crate::connected_collision_native::Prepared>,
}
pub(crate) fn bad(field:&str, detail:&str)->JointError {
    joint::err("fem", "FEM_JOINT_INVALID", field,None,None,detail)
}
pub(crate) fn float(x:f64)->Result<f32,JointError> {
    let y=x as f32;
    if !x.is_finite() || !y.is_finite() || (x!=0. && y==0.) {return Err(bad("float","outside finite representable f32 domain"));} Ok(y)
}
pub(crate) fn floats<const N:usize>(x:&[f64;N])->Result<[f32;N],JointError> {
    let mut r=[0.;N]; for i in 0..N {r[i]=float(x[i])?;} Ok(r)
}
pub(crate) fn bary(x:&[f64;4])->Result<(),JointError> {
    if !x.iter().all(|x|x.is_finite()) || (x.iter().sum::<f64>()-1.).abs()>2e-6 {
        return Err(bad("barycentric","finite affine weights summing to one required; negative weights are allowed"));
    }
    floats(x)?; Ok(())
}
pub(crate) fn u32s(bytes:&[u8])->Result<Vec<u32>,JointError> {
    if bytes.len()%4!=0 {return Err(bad("topology","u32 alignment"));}
    Ok(bytes.chunks_exact(4).map(|x|u32::from_le_bytes(x.try_into().unwrap())).collect())
}
fn f64s(bytes:&[u8])->Result<Vec<f64>,JointError> {
    if bytes.len()%8!=0 {return Err(bad("positions","f64 alignment"));}
    let v:Vec<_>=bytes.chunks_exact(8).map(|x|f64::from_le_bytes(x.try_into().unwrap())).collect();
    if !v.iter().all(|x|x.is_finite()) {return Err(bad("positions","nonfinite native rest coordinates"));} Ok(v)
}
pub(crate) fn indices64(bytes:&[u8])->Result<Vec<u32>,JointError> {
    if bytes.len()%8!=0 {return Err(bad("topology","u64 native index alignment"));}
    bytes.chunks_exact(8).map(|x|u32::try_from(u64::from_le_bytes(x.try_into().unwrap())).map_err(|_|bad("topology","native index exceeds u32 device domain"))).collect()
}
fn quat(q:&[f64;4])->Result<(),JointError> {
    if !q.iter().all(|x|x.is_finite()) || (q.iter().map(|v|v*v).sum::<f64>()-1.).abs()>2e-8 {return Err(bad("quaternion","normalized finite wxyz required"));} floats(q)?; Ok(())
}
pub(crate) fn rotation(r:&[f64;9])->Result<Matrix3<f64>,JointError> {
    if !r.iter().all(|x|x.is_finite()) {return Err(bad("rest_rotation","nonfinite"));}
    let m=Matrix3::from_column_slice(r);
    if (m.transpose()*m-Matrix3::identity()).norm()>2e-6 || (m.determinant()-1.).abs()>2e-6 {return Err(bad("rest_rotation","proper orthonormal material rest frame required"));} Ok(m)
}
fn supported_angular_model(model:&str)->bool {
    matches!(model, crate::joint_angular_control_gpu::COORDINATE_MODEL | crate::material_spin_history::MODEL)
}
fn validate_wire(w:&JointWire, connected_feature:bool)->Result<(),JointError> {
    if w.enabled && !w.collide_connected && !connected_feature {return Err(joint::err("fem","FEM_JOINT_COLLISION_FILTER_UNSUPPORTED","collide_connected",Some(&w.joint_uuid),None,"FEM connected-pair filtering must be explicitly installed; this initial native trial retains collisions"));}
    for id in [&w.joint_uuid,&w.parent_body_uuid,&w.child_body_uuid] {
        if joint::canonical_uuid(id).as_deref()!=Some(id.as_str()) {return Err(bad("uuid","canonical UUID required"));}
    }
    if w.parent_body_uuid==w.child_body_uuid || w.child_body_uuid==joint::WORLD_ENDPOINT_UUID {return Err(bad("endpoints","distinct endpoints and non-world child required"));}
    for x in [w.stiffness_N_per_m,w.angle_stiffness_nm_per_rad,w.angle_damping_nms_per_rad,w.max_angle_rad] {
        if x<0. {return Err(bad("coefficients","nonnegative physical coefficients required"));} float(x)?;
    }
    if w.max_angle_rad>std::f64::consts::PI {return Err(bad("max_angle_rad","outside [0,pi]"));}
    quat(&w.rest_relative_quaternion_wxyz)?;
    for f in [&w.parent_local_frame,&w.child_local_frame] {floats(&f.translation_m)?;quat(&f.quaternion_wxyz)?;}
    if let Some(c)=&w.angular_control_v1 {
        if !supported_angular_model(&c.coordinate_model) {return Err(bad("angular_control_v1","unsupported angular chart"));}
        for x in c.stiffness().into_iter().chain(c.damping()) {if x<0. {return Err(bad("angular_control_v1","negative coefficient"));} float(x)?;}
    }
    if w.angular_hard_hinge_v1.is_some() || w.joint_d6_hard_v1.is_some() {
        return Err(joint::err("fem","FEM_JOINT_HARD_UNSUPPORTED","hard_joint",Some(&w.joint_uuid),None,"hard Schur/material-node composition is not installed"));
    }
    Ok(())
}
pub(crate) fn rest_positions(input:&ValidatedJointInput)->Result<Vec<Vector3<f64>>,JointError> {
    let vert=f64s(input.bytes("bin/vert.bin")?)?;
    let dmap=u32s(input.bytes("bin/vert_dmap.bin")?)?;
    let displacement=f64s(input.bytes("bin/displacement.bin")?)?;
    if vert.len()!=3*dmap.len() || displacement.len()%3!=0 {return Err(bad("positions","native shape mismatch"));}
    let mut out=Vec::with_capacity(dmap.len());
    for (i,g) in dmap.into_iter().enumerate() {
        let d=3*g as usize; if d+3>displacement.len() {return Err(bad("displacement","group absent"));}
        // Match the native constructor's f32 input addition, not a f64 author image.
        out.push(Vector3::from_fn(|k,_|f64::from(vert[3*i+k] as f32+displacement[d+k] as f32)));
    }
    Ok(out)
}
/// The new marker is an exact union across the rigid carrier and material
/// sidecar, including disabled authored cones. Older angular markers retain
/// their existing carrier contract.
fn validate_swing_cone_feature_union<'a>(advertised:bool, wires:impl Iterator<Item=&'a JointWire>)->Result<(),JointError> {
    let present=wires.into_iter().any(|w| crate::joint_angular_extension_gpu::has_swing_cone(w.angular_soft_limits_v1.as_ref().map(|v| &v.0)));
    if advertised!=present {
        return Err(joint::err("fem","JOINT_G2_REQUIRED_FEATURES","required_features",None,None,
            "swing cone feature must exactly match the rigid/material authored union"));
    }
    Ok(())
}

/// Consume only retained, hashed input bytes. No second filesystem generation.
pub fn prepare(input:&ValidatedJointInput)->Result<Option<OwnedSoftScene>,JointError> {
    let controls=input.contains(CONTROL_FILE);
    let events=input.contains(crate::fem_control_events::FILE);
    if events!=input.requires_feature(crate::fem_control_events::FEATURE)? || (events&&(!input.contains(FILE)||controls)) {
        return Err(bad("required_features","v2 event file/feature/material sidecar required; v1/v2 mutually exclusive"));
    }
    if controls!=input.requires_feature(CONTROL_FEATURE)? || (controls && !input.contains(FILE)) {
        return Err(joint::err("fem","FEM_CONTROL_INVALID","required_features",None,None,"control sidecar requires its matching feature and material sidecar"));
    }
    let present=input.contains(FILE);
    let advertised=input.requires_feature(FEATURE)?;
    if present!=advertised {return Err(bad("required_features","FEM sidecar and required carrier feature must appear together"));}
    let cross_feature=input.requires_feature(crate::connected_collision_native::CROSS_FEATURE)?;
    if input.contains(crate::connected_collision_native::CROSS_FILE)!=cross_feature {
        return Err(bad("required_features","cross-link collision sidecar and feature must appear together"));
    }
    if !present {
        if !cross_feature {return Ok(None);}
        // Reuse the fully validated rigid carrier inventory. No material
        // constraint, simulated body or fake skeletal joint is introduced.
        let bodies:Vec<BodyWire>=input.records().bodies.iter().map(|(uuid,index)|BodyWire {
            body_uuid:uuid.clone(),kind:"PDRD".into(),runtime_body_index:Some(*index),
            tet_index:None,vertex_indices:None,dm_inv_column_major:None,center_bary:None,frame_rest_column_major:None,
        }).collect();
        let connected=crate::connected_collision_native::prepare(input,&bodies,&[])
            .map_err(|detail|joint::err("collision","BONE_CONNECTED_OWNERSHIP","retained_owner_partition",None,None,&detail))?;
        return Ok(Some(OwnedSoftScene {scene_uuid:input.records().scene_uuid.clone(),
            scene_inputs_sha256:input.records().scene_inputs_sha256.clone(),payload_sha256:String::new(),
            controlled_vertices:Vec::new(),control_receipt:serde_json::Value::Null,control_events:None,
            bodies:Vec::new(),wires:Vec::new(),records:Vec::new(),extensions:None,connected_collision:Some(connected)}));
    }
    let bytes=input.bytes(FILE)?;
    if bytes.len()>16*1024*1024 {return Err(bad("payload","16 MiB budget exceeded"));}
    let strict:joint::StrictValue=serde_json::from_slice(bytes).map_err(|e|bad("payload",&e.to_string()))?;
    let p:Payload=serde_json::from_value(strict.0).map_err(|e|bad("payload",&e.to_string()))?;
    let base=input.records();
    validate_swing_cone_feature_union(
        input.requires_feature(crate::joint_angular_extension_gpu::FEATURE_SWING_CONE)?,
        base.joints.iter().map(|j| &j.wire).chain(p.joints.iter().map(|j| &j.wire)))?;
    let connected_feature=input.requires_feature(crate::connected_collision::FEATURE)?;
    let connected_needed=p.joints.iter().any(|r|r.wire.enabled && !r.wire.collide_connected && r.wire.parent_body_uuid!=joint::WORLD_ENDPOINT_UUID);
    if connected_feature!=connected_needed {return Err(joint::err("fem","BONE_CONNECTED_CAPABILITY","required_features",None,None,"bone_connected_collision_v1 must appear exactly for an enabled non-WORLD material exclusion"));}

    if p.version!=1 || p.kind!=KIND || p.scene_uuid!=base.scene_uuid {return Err(bad("identity","version/kind/scene identity mismatch"));}
    let pos=rest_positions(input)?;
    let tet=indices64(input.bytes("bin/tet.bin")?)?;
    let membership=joint::collision_vertex_map(input.bytes("map.pickle")?)?;
    let rigid_vertex=u32s(input.bytes("bin/pdrd_vert_index.bin")?)?;
    if rigid_vertex.len()!=pos.len() || tet.len()%4!=0 {return Err(bad("topology","invalid tet or rigid membership array"));}
    let info:toml::Value=toml::from_str(std::str::from_utf8(input.bytes("info.toml")?).map_err(|e|bad("info.toml",&e.to_string()))?).map_err(|e|bad("info.toml",&e.to_string()))?;
    let (endpoints,fem_support)=crate::material_endpoint::validate_bodies(&p.bodies,&pos,&tet,&membership,&rigid_vertex,base)?;
    if fem_support.is_empty() {return Err(bad("bodies","at least one genuine FEM body required"));}
    let n_pin=info.get("count").and_then(|x|x.get("pin_block")).and_then(|x|x.as_integer()).unwrap_or(0);
    let control_events=fem_control_intake::validate(input,&p,&pos,&tet,&membership,&rigid_vertex,n_pin)?;
    let (mut controlled_vertices,mut control_receipt)=fem_body_control::validate(input,&p,&info,&pos,&tet,&membership,&rigid_vertex,&fem_support,n_pin)?;
    if let Some(events)=&control_events {
        controlled_vertices=events.bodies.iter().filter(|b|b.initial_mode!=crate::fem_control_events::Mode::Free).flat_map(|b|b.vertices.iter().copied()).collect();controlled_vertices.sort_unstable();
        control_receipt=serde_json::json!({"feature":crate::fem_control_events::FEATURE,"sidecar_sha256":events.program_sha256,"scope":"current_events_candidate"});
    }
    for f in ["bin/translation_lock.bin","bin/rotation_lock.bin","bin/translation_lock_mode.bin","bin/rotation_lock_mode.bin"] {
        if input.contains(f) && input.bytes(f)?.iter().any(|b|*b!=0) {return Err(joint::err("fem","FEM_JOINT_LOCK_COMPOSITION_UNSUPPORTED",f,None,None,"mixed joint work must share the aggregate projector before allowing locks"));}
    }
    let mut seen=HashSet::new(); let mut ordered=base.clone(); ordered.joints.clear();
    let mut pairs=Vec::new(); let mut parents:HashMap<&str,&str>=HashMap::new();
    for j in &base.joints {
        if parents.insert(&j.wire.child_body_uuid,&j.wire.parent_body_uuid).is_some() {
            return Err(bad("graph","legacy/mixed union has multiple parents"));
        }
    }
    let world=FemEndpointGpu {kind:FEM_WORLD,id:u32::MAX,vertices:[0;4],dm_inv:[0.;9],anchor_bary:[0.;4],center_bary:[0.;4],rest_rotation:[1.,0.,0.,0.,1.,0.,0.,0.,1.],local_anchor:[0.;3]};
    for row in &p.joints {
        let w=&row.wire; validate_wire(w,connected_feature)?;
        if !seen.insert(&w.joint_uuid) || base.joints.iter().any(|j|j.wire.joint_uuid==w.joint_uuid) {return Err(bad("joint_uuid","duplicate joint or legacy dispatch double count"));}
        if parents.insert(&w.child_body_uuid,&w.parent_body_uuid).is_some() {return Err(bad("graph","legacy/mixed union has multiple parents"));}
        let a=if w.parent_body_uuid==joint::WORLD_ENDPOINT_UUID {world} else {*endpoints.get(&w.parent_body_uuid).ok_or_else(||bad("parent","unbound endpoint"))?};
        let b=*endpoints.get(&w.child_body_uuid).ok_or_else(||bad("child","unbound endpoint"))?;
        if a.kind!=FEM_TET && b.kind!=FEM_TET {return Err(bad("joint","ordinary rigid-only records belong exclusively to old dispatch"));}
        pairs.push((crate::material_endpoint::bind_anchor(a,row.parent_anchor_bary,&w.parent_local_frame,&pos)?,crate::material_endpoint::bind_anchor(b,row.child_anchor_bary,&w.child_local_frame,&pos)?));
        // This clone is solely the shared angular packer's immutable record order.
        // It never creates rigid state, mass, a body row or solver unknowns.
        ordered.joints.push(joint::OwnedJoint {wire:w.clone(),parent_runtime_body_index:a.id,child_runtime_body_index:b.id});
    }
    for child in parents.keys() {
        let mut seen=HashSet::new(); let mut cur=*child;
        while let Some(parent)=parents.get(cur) {if !seen.insert(cur) {return Err(bad("graph","cycle"));}cur=parent;}
    }
    let v2=crate::joint_angular_gpu::from_validated(&ordered).map_err(|e|bad("angular_v2",&e))?;
    let controls=crate::joint_angular_control_gpu::from_validated(&ordered).map_err(|e|bad("angular_control",&e))?;
    let empty_control=crate::joint_angular_control_gpu::AngularControlGpu {body_a:0,body_b:0,enabled:0,coordinate_model:1,frame_a_wxyz:[1.,0.,0.,0.],frame_b_wxyz:[1.,0.,0.,0.],rest_relative_wxyz:[1.,0.,0.,0.],stiffness_nm_per_rad:[0.;3],damping_nms_per_rad:[0.;3]};
    let records=p.joints.iter().enumerate().map(|(i,r)|FemJointGpu {a:pairs[i].0,b:pairs[i].1,enabled:u32::from(r.wire.enabled),ball_stiffness:r.wire.stiffness_N_per_m as f32,v2:v2[i],control:controls.get(i).copied().unwrap_or(empty_control),pmx:Default::default()}).collect();
    use crate::joint_angular_extension_gpu as ext;
    if ordered.joints.iter().any(|j|j.wire.angular_control_v1.as_ref().is_some_and(|c|!supported_angular_model(&c.coordinate_model))) {return Err(bad("angular_control","unsupported FEM angular coordinate model"));}
    let refs:Vec<_>=ordered.joints.iter().map(|j|ext::JointExtensionInput {control_present:j.wire.angular_control_v1.is_some(),control_model:j.wire.angular_control_v1.as_ref().map(|c|c.coordinate_model.as_str()),soft:j.wire.angular_soft_limits_v1.as_ref().map(|v|&v.0),drives:j.wire.angular_drives_v1.as_ref().map(|v|&v.0),legacy_angular_nonzero:j.wire.angle_stiffness_nm_per_rad!=0. || j.wire.angle_damping_nms_per_rad!=0. || j.wire.max_angle_rad!=0.}).collect();
    let features:Vec<_>=ext::required_features(&refs).into_iter().map(str::to_owned).collect();
    let extensions=ext::validate_v4_and_pack(&refs,&features).map_err(|e|bad("extensions",&e.to_string()))?;
    let connected_collision=if connected_feature || cross_feature {Some(crate::connected_collision_native::prepare(input,&p.bodies,&p.joints)
        .map_err(|detail|joint::err("collision","BONE_CONNECTED_OWNERSHIP","retained_owner_partition",None,None,&detail))?)} else {None};
    Ok(Some(OwnedSoftScene {controlled_vertices,control_receipt,control_events,connected_collision,scene_uuid:base.scene_uuid.clone(),scene_inputs_sha256:base.scene_inputs_sha256.clone(),payload_sha256:joint::sha256(bytes),bodies:p.bodies,wires:p.joints,records,extensions}))
}

pub fn write_receipt(output:&std::path::Path, scene:&OwnedSoftScene, accepted_steps:u64,time:f64,state:&str)->Result<(),String> {
    let pmx = scene.records.iter().filter(|r| r.pmx.enabled != 0).collect::<Vec<_>>();
    if !pmx.is_empty() {
        let count = |mode| pmx.iter().flat_map(|r| r.pmx.modes).filter(|&m| m == mode).count();
        let value = serde_json::json!({
            "version": 1, "kind": "PmxJointNativeReceipt", "scene_uuid": scene.scene_uuid,
            "state": state, "accepted_steps": accepted_steps, "accepted_time_seconds": time,
            "joint_count": pmx.len(), "free_axes": count(0), "limited_axes": count(1), "locked_axes": count(2),
            "mechanism": "coupled_compliant_euler_xyz_box",
            "linear_limit_weight_N_per_m": crate::pmx_joints::LINEAR_LIMIT_WEIGHT,
            "angular_limit_weight_Nm_per_rad": crate::pmx_joints::ANGULAR_LIMIT_WEIGHT,
        });
        let temporary = output.join("pmx_joint_native_receipt.json.tmp");
        let bytes = serde_json::to_vec_pretty(&value).map_err(|e| e.to_string())?;
        std::fs::write(&temporary, bytes).map_err(|e| e.to_string())?;
        std::fs::rename(temporary, output.join("pmx_joint_native_receipt.json")).map_err(|e| e.to_string())?;
    }
    if scene.payload_sha256.is_empty(){return Ok(());}
    let value=serde_json::json!({"version":1,"kind":"FEMMaterialJointNativeReceipt","fem_body_controls":scene.control_receipt,"scene_uuid":scene.scene_uuid,"scene_inputs_sha256":scene.scene_inputs_sha256,"payload_sha256":scene.payload_sha256,"state":state,"accepted_steps":accepted_steps,"accepted_time_seconds":time,"joint_count":scene.wires.len(),"active_joint_count":scene.wires.iter().filter(|r|r.wire.enabled).count(),"fem_body_count":scene.bodies.iter().filter(|b|b.kind=="FEM_TET_MATERIAL").count(),"body_kinds":scene.bodies.iter().map(|b|(&b.body_uuid,&b.kind)).collect::<HashMap<_,_>>(),"mechanism":"material_node_force_and_matrix_free_cross_operator","pose_authority":"accepted_vertices"});
    let bytes=serde_json::to_vec_pretty(&value).map_err(|e|e.to_string())?;
    let temporary=output.join("fem_joint_native_receipt.json.tmp");
    std::fs::write(&temporary,bytes).map_err(|e|e.to_string())?;
    std::fs::rename(temporary,output.join("fem_joint_native_receipt.json")).map_err(|e|e.to_string())
}

#[cfg(test)]
mod swing_cone_union_tests {
    use super::*;
    #[test]
    fn cone_feature_is_exact_rigid_material_union_including_disabled() {
        let plain:JointWire=serde_json::from_value(serde_json::json!({"joint_uuid":"44444444-4444-4444-4444-444444444444","parent_body_uuid":"22222222-2222-2222-2222-222222222222","child_body_uuid":"33333333-3333-3333-3333-333333333333","parent_local_frame":{"translation_m":[0,0,0],"quaternion_wxyz":[1,0,0,0]},"child_local_frame":{"translation_m":[0,0,0],"quaternion_wxyz":[1,0,0,0]},"stiffness_N_per_m":0,"enabled":true,"collide_connected":true})).unwrap();
        let mut cone=plain.clone();
        cone.angular_soft_limits_v1=Some(joint::StrictValue(serde_json::json!({"swing_cone":{"enabled":false}})));
        for (rigid,material) in [(false,false),(true,false),(false,true),(true,true)] {
            let rows=[if rigid {&cone}else{&plain},if material {&cone}else{&plain}];
            let expected=rigid||material;
            assert!(validate_swing_cone_feature_union(expected,rows.into_iter()).is_ok());
            assert_eq!(validate_swing_cone_feature_union(!expected,rows.into_iter()).unwrap_err().stable_code,"JOINT_G2_REQUIRED_FEATURES");
        }
    }
}
