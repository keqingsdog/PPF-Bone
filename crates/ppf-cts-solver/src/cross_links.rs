//! Independent axial network. Tree joints and their parent proofs stay untouched.
use crate::{joint::{self, JointError, ValidatedJointInput}, soft_joint::{self, BodyWire}, soft_joint_gpu::*, restoring_curve_gpu as gpu};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const FEATURE: &str = "cross_links_v1";
pub const FILE: &str = "bin/cross_links_v1.json";
pub const KIND: &str = "ppf.cross_links";
pub const DISTANCE_KIND: u32 = 5;
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Endpoint {
    pub body_uuid: String, pub local_anchor_m: [f64;3],
    #[serde(default, skip_serializing_if="Option::is_none")] pub anchor_bary: Option<[f64;4]>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Link {
    pub link_uuid: String, pub enabled: bool, pub model: String,
    pub a: Endpoint, pub b: Endpoint,
    pub rest_length_m: f64, pub epsilon_m: f64,
    #[serde(rename="stiffness_N_per_m")] pub stiffness: f64,
    #[serde(rename="damping_Ns_per_m")] pub damping: f64,
    #[serde(default, skip_serializing_if="Option::is_none")] pub tension_enabled: Option<bool>,
    #[serde(default, skip_serializing_if="Option::is_none")] pub compression_enabled: Option<bool>,
    #[serde(default, rename="tension_stiffness_N_per_m", skip_serializing_if="Option::is_none")] pub tension_stiffness: Option<f64>,
    #[serde(default, rename="compression_stiffness_N_per_m", skip_serializing_if="Option::is_none")] pub compression_stiffness: Option<f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Payload { pub version:u32, pub kind:String, pub scene_uuid:String, pub bodies:Vec<BodyWire>, pub links:Vec<Link> }
#[derive(Clone, Debug)]
pub struct OwnedCrossLinks { pub payload:Payload, pub records:Vec<FemJointGpu> }
fn bad(field:&str, detail:&str)->JointError {joint::err("cross-links","CROSS_LINK_INVALID",field,None,None,detail)}
fn narrow(x:f64)->Result<f32,JointError>{soft_joint::float(x).map_err(|_|bad("number","finite representable f32 required"))}
fn canonical(id:&str)->bool {joint::canonical_uuid(id).as_deref()==Some(id)}
pub(crate) fn decode(bytes:&[u8], scene_uuid:&str)->Result<Payload,JointError> {
    if bytes.len()>16*1024*1024 {return Err(bad("payload","16 MiB budget exceeded"));}
    let strict:joint::StrictValue=serde_json::from_slice(bytes).map_err(|e|bad("payload",&e.to_string()))?;
    let p:Payload=serde_json::from_value(strict.0).map_err(|e|bad("payload",&e.to_string()))?;
    if p.version!=1||p.kind!=KIND||p.scene_uuid!=scene_uuid||!canonical(&p.scene_uuid) {return Err(bad("identity","version/kind/scene UUID mismatch"));}
    if p.links.is_empty()||p.links.len()>65536 {return Err(bad("links","nonempty bounded network required"));}
    let mut ids=HashSet::new();let mut pairs=HashSet::new();
    for row in &p.links {
        if !canonical(&row.link_uuid)||!ids.insert(&row.link_uuid)||!row.enabled||row.model!="AXIAL_DISTANCE" {return Err(bad("link","unique canonical enabled AXIAL_DISTANCE links required"));}
        if row.a.body_uuid==row.b.body_uuid||!canonical(&row.a.body_uuid)||!canonical(&row.b.body_uuid)||row.a.body_uuid==joint::WORLD_ENDPOINT_UUID||row.b.body_uuid==joint::WORLD_ENDPOINT_UUID {return Err(bad("endpoints","different real physical bodies required; WORLD is not supported"));}
        // Cross-network identity is an unordered physical body pair, regardless
        // of anchor positions. Tree parent pairs are deliberately not consulted.
        let pair=if row.a.body_uuid<row.b.body_uuid {(&row.a.body_uuid,&row.b.body_uuid)}else{(&row.b.body_uuid,&row.a.body_uuid)};
        if !pairs.insert(pair){return Err(bad("endpoints","duplicate unordered cross-link body pair"));}
        for x in [row.rest_length_m,row.stiffness,row.damping] {if x<0. {return Err(bad("coefficients","negative coefficient or length"));}narrow(x)?;}
        if [row.tension_stiffness.is_some(),row.compression_stiffness.is_some(),row.tension_enabled.is_some(),row.compression_enabled.is_some()].iter().any(|x| *x != row.tension_stiffness.is_some()) {return Err(bad("coefficients","directional stiffness must be supplied as a pair"));}
        for x in [row.tension_stiffness,row.compression_stiffness].into_iter().flatten() {if x<0. {return Err(bad("coefficients","negative directional stiffness"));}narrow(x)?;}
        if row.tension_enabled.is_none()&&row.stiffness==0.&&row.damping==0. {return Err(bad("coefficients","at least one nonzero stiffness or damping required"));}
        if row.epsilon_m<=0. {return Err(bad("epsilon_m","positive smoothing radius required"));}narrow(row.epsilon_m)?;
        for e in [&row.a,&row.b] {for x in e.local_anchor_m {narrow(x)?;}if let Some(b)=e.anchor_bary {soft_joint::bary(&b)?;}}
    }
    Ok(p)
}
fn inert_row(a:FemEndpointGpu,b:FemEndpointGpu)->FemJointGpu {
    FemJointGpu {a,b,enabled:1,ball_stiffness:0.,v2:crate::joint_angular_gpu::AngularJointGpuV2 {
        body_a:0,body_b:0,enabled:0,max_angle_rad:0.,frame_a_wxyz:[1.,0.,0.,0.],frame_b_wxyz:[1.,0.,0.,0.],rest_relative_wxyz:[1.,0.,0.,0.],stiffness_nm_per_rad:0.,damping_nms_per_rad:0.
    },control:crate::joint_angular_control_gpu::AngularControlGpu {body_a:0,body_b:0,enabled:0,coordinate_model:1,frame_a_wxyz:[1.,0.,0.,0.],frame_b_wxyz:[1.,0.,0.,0.],rest_relative_wxyz:[1.,0.,0.,0.],stiffness_nm_per_rad:[0.;3],damping_nms_per_rad:[0.;3]}}
}
pub fn prepare(input:&ValidatedJointInput)->Result<Option<OwnedCrossLinks>,JointError> {
    if input.contains(FILE)!=input.requires_feature(FEATURE)? {return Err(bad("required_features","cross-link sidecar and feature must appear together"));}
    if !input.contains(FILE) {return Ok(None);}
    let p=decode(input.bytes(FILE)?,&input.records().scene_uuid)?;
    let pos=soft_joint::rest_positions(input)?;
    let tet=if input.contains("bin/tet.bin"){soft_joint::indices64(input.bytes("bin/tet.bin")?)?}else{Vec::new()};
    let membership=joint::collision_vertex_map(input.bytes("map.pickle")?)?;
    let rigid=soft_joint::u32s(input.bytes("bin/pdrd_vert_index.bin")?)?;
    if rigid.len()!=pos.len()||tet.len()%4!=0 {return Err(bad("topology","native topology shape mismatch"));}
    let (bodies,_)=crate::material_endpoint::validate_bodies(&p.bodies,&pos,&tet,&membership,&rigid,input.records())?;
    let mut used=HashSet::new();let mut records=Vec::with_capacity(p.links.len());
    for link in &p.links {
        // UUID namespaces remain distinct even if runtime rows share one batch.
        if input.records().joints.iter().any(|j|j.wire.joint_uuid==link.link_uuid) {return Err(bad("link_uuid","collides with tree joint UUID"));}
        let bind=|e:&Endpoint|->Result<FemEndpointGpu,JointError>{
            let body=*bodies.get(&e.body_uuid).ok_or_else(||bad("body_uuid","endpoint not in validated body inventory"))?;
            let frame=joint::LocalFrame{translation_m:e.local_anchor_m,quaternion_wxyz:[1.,0.,0.,0.]};
            crate::material_endpoint::bind_anchor(body,e.anchor_bary,&frame,&pos)
        };
        let a=bind(&link.a)?;let b=bind(&link.b)?;
        // Rest length is checked against the actual native fitted frames in
        // verify_initial(), after the fresh builder exists, before installation.
        used.insert(link.a.body_uuid.as_str());used.insert(link.b.body_uuid.as_str());
        records.push(inert_row(a,b));
    }
    if used.len()!=p.bodies.len(){return Err(bad("bodies","unused or duplicate body inventory"));}
    if input.contains(soft_joint::FILE){
        let strict:joint::StrictValue=serde_json::from_slice(input.bytes(soft_joint::FILE)?).map_err(|e|bad("tree",&e.to_string()))?;
        let tree:soft_joint::Payload=serde_json::from_value(strict.0).map_err(|e|bad("tree",&e.to_string()))?;
        if p.links.iter().any(|l|tree.joints.iter().any(|t|t.wire.joint_uuid==l.link_uuid)){return Err(bad("link_uuid","collides with material tree joint UUID"));}
    }
    Ok(Some(OwnedCrossLinks{payload:p,records}))
}
/// Tree rows remain the prefix. Network edges never enter any parent graph.
pub fn prepare_batch(input:&ValidatedJointInput,scale:f64)->Result<Option<soft_joint::OwnedSoftScene>,JointError> {
    let mut tree=soft_joint::prepare(input)?;
    let Some(cross)=prepare(input)? else{return Ok(tree);};
    if !scale.is_finite()||scale<=0. {return Err(bad("world_scaling","positive finite scale required"));}
    let batch=tree.get_or_insert_with(||soft_joint::OwnedSoftScene {
        scene_uuid:input.records().scene_uuid.clone(),scene_inputs_sha256:input.records().scene_inputs_sha256.clone(),payload_sha256:String::new(),
        controlled_vertices:Vec::new(),control_receipt:serde_json::Value::Null,control_events:None,bodies:Vec::new(),wires:Vec::new(),records:Vec::new(),extensions:None,connected_collision:None,
    });
    for mut row in cross.records {
        for endpoint in [&mut row.a,&mut row.b] {
            for x in &mut endpoint.local_anchor {*x=narrow(f64::from(*x)*scale)?;}
            if endpoint.kind==FEM_TET {for x in &mut endpoint.dm_inv {*x=narrow(f64::from(*x)/scale)?;}}
        }
        if let Some(ext)=&mut batch.extensions {ext.limits.push(crate::joint_angular_extension_gpu::AngularSoftLimitGpuV1{joint_index:batch.records.len() as u32,enabled_axis_mask:0,lower_rad:[0.;3],upper_rad:[0.;3],stiffness_nm_per_rad:[0.;3],swing_cone_enabled:0,swing_cone_half_angle_rad:0.,swing_cone_stiffness_nm_per_rad:0.});}
        batch.records.push(row);
    }
    Ok(tree)
}
/// Merge once, preserving every existing table/binding/history/receipt index.
pub fn merge_curves(input:Option<&ValidatedJointInput>,mut old:Option<crate::restoring_native::Installed>,scene:&crate::scene::Scene,data:Option<&crate::data::DataSet>)->Result<Option<crate::restoring_native::Installed>,String> {
    let Some(input)=input else{return Ok(old);};
    let Some(cross)=prepare(input).map_err(|e|e.to_string())? else{return Ok(old);};
    let scale=scene.args().world_scaling as f64;
    let initial_rest=verify_initial(&cross,data.ok_or("CROSS_LINK_INITIAL_FRAME_MISSING")?,scale)?;
    let prefix=soft_joint::prepare(input).map_err(|e|e.to_string())?.map_or(0,|s|s.records.len());
    let installed=old.get_or_insert_with(||crate::restoring_native::Installed{tables:crate::restoring_curve::PackedTables{descriptors:Vec::new(),nodes:Vec::new()},bindings:Vec::new()});
    let mut tables=crate::restoring_curve::PackedTables{descriptors:Vec::new(),nodes:Vec::new()};let mut bindings=Vec::new();
    for (i,link) in cross.payload.links.iter().enumerate() {
        // SIGNED linear segments retain the existing kind-5 kernel ABI. The
        // direction mask below gates shared damping as well as elasticity.
        let kt=narrow(if link.tension_enabled.unwrap_or(true) {link.tension_stiffness.unwrap_or(link.stiffness)} else {0.}).map_err(|e|e.to_string())?;
        let kc=narrow(if link.compression_enabled.unwrap_or(true) {link.compression_stiffness.unwrap_or(link.stiffness)} else {0.}).map_err(|e|e.to_string())?;
        if link.tension_enabled.is_none() {
            tables.descriptors.push(gpu::CurveDescriptor{node_offset:tables.nodes.len() as u32,node_count:2,domain:1,tail:0,reserved:[0;4]});
            tables.nodes.extend([gpu::CurveNode{x:0.,effort:0.,area:0.,right_slope:kt},gpu::CurveNode{x:1.,effort:kt,area:0.5*kt,right_slope:kt}]);
        } else {
            tables.descriptors.push(gpu::CurveDescriptor{node_offset:tables.nodes.len() as u32,node_count:3,domain:0,tail:0,reserved:[0;4]});
            tables.nodes.extend([gpu::CurveNode{x:-1.,effort:-kc,area:0.5*kc,right_slope:kc},gpu::CurveNode{x:0.,effort:0.,area:0.,right_slope:kt},gpu::CurveNode{x:1.,effort:kt,area:0.5*kt,right_slope:kt}]);
        }
        bindings.push(gpu::CurveBinding{kind:DISTANCE_KIND,target_index:(prefix+i) as u32,channel:3,table_index:i as u32,history_index:i as u32,receipt_index:u32::MAX,damping:narrow(link.damping).map_err(|e|e.to_string())?,epsilon_m:narrow(link.epsilon_m).map_err(|e|e.to_string())?,rest_length_m:narrow(link.rest_length_m.hypot(link.epsilon_m)).map_err(|e|e.to_string())?,reserved:[if link.tension_enabled.is_some() {1 | ((link.tension_enabled.unwrap() as u32)<<1) | ((link.compression_enabled.unwrap() as u32)<<2)} else {0},0,0]});
    }
    let (mut tables,mut bindings)=gpu::solver_space(&tables,&bindings,scale)?;
    // These are derived from the actual native initial f32 frames, including
    // nonidentity rigid rotations. Build's SI rest is checked, never used to
    // introduce a rounding-dependent initial preload.
    for (b,r) in bindings.iter_mut().zip(initial_rest) {b.rest_length_m=r;}
    let table_offset=installed.tables.descriptors.len() as u32;let node_offset=installed.tables.nodes.len() as u32;
    let history_offset=installed.bindings.iter().filter(|b|b.history_index!=u32::MAX).map(|b|b.history_index+1).max().unwrap_or(0);
    for d in &mut tables.descriptors {d.node_offset=d.node_offset.checked_add(node_offset).ok_or("CROSS_LINK_BUDGET")?;}
    for b in &mut bindings {b.table_index=b.table_index.checked_add(table_offset).ok_or("CROSS_LINK_BUDGET")?;b.history_index=b.history_index.checked_add(history_offset).ok_or("CROSS_LINK_BUDGET")?;}
    installed.tables.descriptors.extend(tables.descriptors);installed.tables.nodes.extend(tables.nodes);installed.bindings.extend(bindings);
    Ok(old)
}

/// Exact same host-rendered rigid fit used by native mass/frame authority.
/// This never assumes initial R=I and never fits a different rotation in Rust.
fn verify_initial(cross:&OwnedCrossLinks,data:&crate::data::DataSet,scale:f64)->Result<Vec<f32>,String> {
    use std::ffi::c_void;
    extern "C" {fn restoring_mass_rest_states(vp:*const c_void,bp:*const c_void,positions:*const c_void,nv:u32,list:*const u32,rest:*const c_void,nm:u32,nb:u32,states:*mut c_void)->i32;}
    let props=data.prop.vertex.as_slice();let body_props=data.prop.pdrd_body.as_slice();let xyz=data.vertex.curr.as_slice();
    let list=data.pdrd_vert_list.as_slice();let rest=data.pdrd_rest_centered.as_slice();
    let mut states=vec![crate::data::PdrdRigidState::default();body_props.len()];
    if !states.is_empty(){let code=unsafe{restoring_mass_rest_states(props.as_ptr() as _,body_props.as_ptr() as _,xyz.as_ptr() as _,xyz.len().try_into().map_err(|_|"CROSS_LINK_BUDGET")?,list.as_ptr(),rest.as_ptr() as _,list.len().try_into().map_err(|_|"CROSS_LINK_BUDGET")?,states.len().try_into().map_err(|_|"CROSS_LINK_BUDGET")?,states.as_mut_ptr() as _)};if code!=0{return Err(format!("CROSS_LINK_INITIAL_FRAME_INVALID: native fit {code}"));}}
    let anchor=|endpoint:&FemEndpointGpu|->Result<[f32;3],String>{
        let mut out=[0f32;3];
        if endpoint.kind==FEM_TET {for v in 0..4 {let x=xyz.get(endpoint.vertices[v] as usize).ok_or("CROSS_LINK_INITIAL_FRAME_INVALID: vertex")?;for k in 0..3{out[k]+=endpoint.anchor_bary[v]*x[k];}}}
        else if endpoint.kind==FEM_PDRD {let s=states.get(endpoint.id as usize).ok_or("CROSS_LINK_INITIAL_FRAME_INVALID: body")?;let mut local=[0f32;3];for k in 0..3 {local[k]=narrow(f64::from(endpoint.local_anchor[k])*scale).map_err(|e|e.to_string())?;out[k]=s.anchor[k]+s.x[k];}for k in 0..3 {for j in 0..3 {out[k]+=s.rotation[3*j+k]*local[j];}}}
        else{return Err("CROSS_LINK_INITIAL_FRAME_INVALID: endpoint kind".into());}
        if !out.iter().all(|x|x.is_finite()){return Err("CROSS_LINK_INITIAL_FRAME_INVALID: arithmetic".into());}Ok(out)
    };
    let mut smooth=Vec::new();
    for (row,wire) in cross.records.iter().zip(&cross.payload.links) {
        let a=anchor(&row.a)?;let b=anchor(&row.b)?;let d=std::array::from_fn::<_,3,_>(|k|a[k]-b[k]);
        let length=d.iter().map(|x|f64::from(*x).powi(2)).sum::<f64>().sqrt()/scale;
        if (length-wire.rest_length_m).abs()>2e-6f64.max(2e-5*length){return Err(format!("CROSS_LINK_INITIAL_FRAME_INVALID: {} rest length differs from actual native fitted anchors",wire.link_uuid));}
        let epsilon=narrow(wire.epsilon_m*scale).map_err(|e|e.to_string())?;
        smooth.push(narrow((d.iter().map(|x|f64::from(*x).powi(2)).sum::<f64>()+f64::from(epsilon).powi(2)).sqrt()).map_err(|e|e.to_string())?);
    }
    Ok(smooth)
}

pub fn receipt_identity(input:&ValidatedJointInput)->serde_json::Value {
    match prepare(input) {
        Ok(Some(cross))=>{let offset=soft_joint::prepare(input).ok().flatten().map_or(0,|s|s.records.len());serde_json::json!({"feature":FEATURE,"kind":KIND,"rows":cross.payload.links.iter().enumerate().map(|(i,l)|serde_json::json!({"link_uuid":l.link_uuid,"runtime_row":offset+i})).collect::<Vec<_>>()})},
        _=>serde_json::Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const SCENE:&str="11111111-1111-1111-1111-111111111111";
    fn payload()->serde_json::Value {serde_json::json!({"version":1,"kind":KIND,"scene_uuid":SCENE,"bodies":[],"links":[{"link_uuid":"22222222-2222-2222-2222-222222222222","enabled":true,"model":"AXIAL_DISTANCE","a":{"body_uuid":"33333333-3333-3333-3333-333333333333","local_anchor_m":[0,0,0]},"b":{"body_uuid":"44444444-4444-4444-4444-444444444444","local_anchor_m":[0,0,0]},"rest_length_m":0.0,"epsilon_m":1e-6,"stiffness_N_per_m":100.0,"damping_Ns_per_m":1.0}]})}
    fn parse(v:&serde_json::Value)->Result<Payload,JointError>{decode(&serde_json::to_vec(v).unwrap(),SCENE)}
    #[test] fn exact_wire_and_zero_rest_are_accepted(){let p=parse(&payload()).unwrap();assert_eq!(p.links[0].rest_length_m,0.);assert_eq!(p.links[0].stiffness,100.);}
    #[test] fn wrong_or_duplicate_fields_are_refused(){
        let mut p=payload();p["links"][0]["K"]=serde_json::json!(100);assert!(parse(&p).is_err());
        let raw=serde_json::to_string(&payload()).unwrap().replacen("\"version\":1","\"version\":1,\"version\":1",1);assert!(decode(raw.as_bytes(),SCENE).is_err());
        let mut p=payload();p["links"][0]["model"]=serde_json::json!("VECTOR_OFFSET");assert!(parse(&p).is_err());
    }
    #[test] fn zero_negative_nonrepresentable_coefficients_are_refused(){
        for field in ["stiffness_N_per_m","damping_Ns_per_m","rest_length_m"]{let mut p=payload();p["links"][0][field]=serde_json::json!(-1.);assert!(parse(&p).is_err());}
        let mut p=payload();p["links"][0]["stiffness_N_per_m"]=serde_json::json!(0.);p["links"][0]["damping_Ns_per_m"]=serde_json::json!(0.);assert!(parse(&p).is_err());
        for field in ["epsilon_m","stiffness_N_per_m","damping_Ns_per_m"]{for x in [1e-100,1e100]{let mut p=payload();p["links"][0][field]=serde_json::json!(x);assert!(parse(&p).is_err());}}
        let mut p=payload();p["links"][0]["epsilon_m"]=serde_json::json!(0.);assert!(parse(&p).is_err());
    }
    #[test] fn endpoint_identity_and_affine_weights_are_strict(){
        let mut p=payload();p["links"][0]["a"]["body_uuid"]=p["links"][0]["b"]["body_uuid"].clone();assert!(parse(&p).is_err());
        let mut p=payload();p["links"][0]["a"]["body_uuid"]=serde_json::json!(joint::WORLD_ENDPOINT_UUID);assert!(parse(&p).is_err());
        let mut p=payload();p["links"][0]["a"]["anchor_bary"]=serde_json::json!([-1.0,1.0,0.5,0.5]);assert!(parse(&p).is_ok());
        p["links"][0]["a"]["anchor_bary"]=serde_json::json!([0.,0.,0.,0.]);assert!(parse(&p).is_err());
    }
    #[test] fn independent_network_cycles_are_not_tree_parent_records(){
        let mut p=payload();let first=p["links"][0].clone();
        let mut second=first.clone();second["link_uuid"]=serde_json::json!("55555555-5555-5555-5555-555555555555");second["a"]=first["b"].clone();second["b"]["body_uuid"]=serde_json::json!("66666666-6666-6666-6666-666666666666");
        let mut third=first.clone();third["link_uuid"]=serde_json::json!("77777777-7777-7777-7777-777777777777");third["a"]=second["b"].clone();third["b"]=first["a"].clone();
        p["links"].as_array_mut().unwrap().extend([second,third]);assert_eq!(parse(&p).unwrap().links.len(),3);
        let duplicate=p["links"][0].clone();p["links"].as_array_mut().unwrap().push(duplicate);assert!(parse(&p).is_err());
    }
    #[test] fn reverse_same_pair_with_new_uuid_and_anchors_is_refused(){
        let mut p=payload();let first=p["links"][0].clone();let mut reverse=first.clone();
        reverse["link_uuid"]=serde_json::json!("55555555-5555-5555-5555-555555555555");reverse["a"]=first["b"].clone();reverse["b"]=first["a"].clone();
        reverse["a"]["local_anchor_m"]=serde_json::json!([1.,2.,3.]);reverse["b"]["local_anchor_m"]=serde_json::json!([-1.,0.,1.]);
        p["links"].as_array_mut().unwrap().push(reverse);
        let error=parse(&p).unwrap_err();assert_eq!(error.stable_code,"CROSS_LINK_INVALID");assert!(error.to_string().contains("duplicate unordered cross-link body pair"));
    }
}
