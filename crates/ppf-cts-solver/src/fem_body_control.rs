//! Only initial whole-body FIX controls; original pin evaluator owns motion.
use super::*;
#[derive(Debug,Deserialize,Serialize)]
#[serde(deny_unknown_fields)]
struct Controls {version:u32,kind:String,scene_uuid:String,bodies:Vec<Control>}
#[derive(Debug,Deserialize,Serialize)]
#[serde(deny_unknown_fields)]
struct Control {body_uuid:String,pin_block_index:u32,mode:String,vertex_indices:Vec<u32>}
fn invalid(detail:&str)->JointError {joint::err("fem","FEM_CONTROL_INVALID","fem_body_controls",None,None,detail)}
fn num(v:&toml::Value,key:&str)->Result<f64,JointError>{v.get(key).and_then(|x|x.as_float().or_else(||x.as_integer().map(|x|x as f64))).filter(|x|x.is_finite()).ok_or_else(||invalid("missing finite TOML number"))}
pub(super) fn validate(input:&ValidatedJointInput,p:&Payload,info:&toml::Value,pos:&[Vector3<f64>],tet:&[u32],membership:&HashMap<String,Vec<u32>>,rigid:&[u32],support:&HashSet<u32>,n_pin:i64)->Result<(Vec<u32>,serde_json::Value),JointError>{
    let present=input.contains(CONTROL_FILE);
    if present!=input.requires_feature(CONTROL_FEATURE)? {return Err(invalid("control feature and sidecar must occur together"));}
    let mut allowed:HashMap<u32,Vec<u32>>=HashMap::new();let mut controlled=HashSet::new();
    let mut receipt=serde_json::Value::Null;
    if present {
        let bytes=input.bytes(CONTROL_FILE)?;
        if bytes.len()>16*1024*1024{return Err(invalid("sidecar budget"));}
        let strict:joint::StrictValue=serde_json::from_slice(bytes).map_err(|_|invalid("strict JSON"))?;
        let c:Controls=serde_json::from_value(strict.0).map_err(|_|invalid("typed control schema"))?;
        if c.version!=1||c.kind!="ppf.fem_body_controls"||c.scene_uuid!=p.scene_uuid||c.bodies.is_empty(){return Err(invalid("control identity"));}
        let mut prev="";
        for b in &c.bodies {
            if b.body_uuid.as_str()<=prev {return Err(invalid("body UUID rows must be sorted unique"));}prev=&b.body_uuid;
            if !p.bodies.iter().any(|e|e.body_uuid==b.body_uuid&&e.kind=="FEM_TET_MATERIAL"){return Err(invalid("controlled endpoint must be actual FEM"));}
            if p.joints.iter().any(|j|j.wire.enabled && j.wire.child_body_uuid==b.body_uuid) || input.records().joints.iter().any(|j|j.wire.enabled && j.wire.child_body_uuid==b.body_uuid) {return Err(invalid("controlled FEM must be a graph root"));}
            let actual=membership.get(&b.body_uuid).ok_or_else(||invalid("body membership missing"))?;
            let mut actual=actual.clone();actual.sort_unstable();actual.dedup();
            if actual!=b.vertex_indices||actual.is_empty()||b.vertex_indices.windows(2).any(|w|w[0]>=w[1]) {return Err(invalid("whole actual body membership equality required"));}
            let set:HashSet<_>=actual.iter().copied().collect();let mut covered=HashSet::new();
            if membership.iter().any(|(owner,vertices)|owner!=&b.body_uuid && vertices.iter().any(|v|set.contains(v))) {
                return Err(invalid("controlled vertices also claimed by another native body"));
            }
            for t in tet.chunks_exact(4){if t.iter().any(|v|set.contains(v)){if !t.iter().all(|v|set.contains(v)){return Err(invalid("tet crosses declared body"));}covered.extend(t.iter().copied());}}
            if covered!=set {return Err(invalid("every body member must be genuine tet node"));}
            for &v in &actual {if v as usize>=pos.len()||rigid[v as usize]!=0||!controlled.insert(v){return Err(invalid("ordinary disjoint controlled inventory required"));}}
            if i64::from(b.pin_block_index)>=n_pin||allowed.insert(b.pin_block_index,actual.clone()).is_some(){return Err(invalid("unique existing pin block required"));}
            let title=format!("pin-{}",b.pin_block_index);let cfg=info.get(&title).ok_or_else(||invalid("pin TOML absent"))?;
            if num(cfg,"pull")?!=0.||cfg.get("unpin_time").is_some()||input.contains(&format!("bin/pin-pullw-{}.bin",b.pin_block_index))||cfg.get("allow_intersection").and_then(|x|x.as_bool()).unwrap_or(false){return Err(invalid("only persistent exact FIX accepted"));}
            if num(cfg,"pin")?!=actual.len() as f64{return Err(invalid("pin count mismatch"));}
            match b.mode.as_str(){
                "STATIC"=>{if num(cfg,"operation_count")?!=0.{return Err(invalid("STATIC has no operations"));}}
                "KINEMATIC"=>{
                    if num(cfg,"operation_count")?!=1.{return Err(invalid("KINEMATIC requires one TRS operation"));}
                    let op=info.get(format!("{title}-op-0")).ok_or_else(||invalid("TRS metadata missing"))?;
                    if op.get("type").and_then(|x|x.as_str())!=Some("transform_keyframes"){return Err(invalid("only transform_keyframes accepted"));}
                    let base=format!("bin/{title}-op-0");let times=f64s(input.bytes(&format!("{base}-time.bin"))?)?;
                    if times.is_empty()||times[0]!=0.||times.windows(2).any(|w|w[0]>=w[1])||num(op,"keyframe_count")?!=times.len() as f64||num(op,"t_start")?!=0.||num(op,"t_end")?!=*times.last().unwrap(){return Err(invalid("finite strictly increasing zero-start timeline required"));}
                    let tr=f64s(input.bytes(&format!("{base}-translation.bin"))?)?;let q=f64s(input.bytes(&format!("{base}-quaternion.bin"))?)?;let sc=f64s(input.bytes(&format!("{base}-scale.bin"))?)?;
                    let local=f64s(input.bytes(&format!("{base}.bin"))?)?;
                    if tr.len()!=3*times.len()||q.len()!=4*times.len()||sc.len()!=3*times.len()||sc.iter().any(|x|*x!=1.)||local.len()!=3*actual.len(){return Err(invalid("finite rigid TRS array dimensions"));}
                    for &x in tr.iter().chain(sc.iter()).chain(local.iter()){float(x)?;}
                    for a in q.chunks_exact(4){quat(&a.try_into().unwrap())?;}
                    if times.len()>1 {let interp=input.bytes(&format!("{base}-interp.bin"))?;let handles=f64s(input.bytes(&format!("{base}-handles.bin"))?)?;if interp.len()!=times.len()-1||interp.iter().any(|x|*x!=0)||handles.len()!=4*(times.len()-1){return Err(invalid("only finite LINEAR segments supported"));}}
                    let rest=Vector3::new(num(op,"rest_tx")?,num(op,"rest_ty")?,num(op,"rest_tz")?);
                    let quat=nalgebra::UnitQuaternion::new_normalize(nalgebra::Quaternion::new(q[0],q[1],q[2],q[3]));
                    // Native read_mat_from_file is column-major: contiguous xyz per vertex.
                    let verts=f64s(input.bytes("bin/vert.bin")?)?;
                    for (i,&v) in actual.iter().enumerate(){let l=Vector3::new(local[3*i],local[3*i+1],local[3*i+2]);let x=quat*l+Vector3::new(tr[0],tr[1],tr[2])-rest;let raw=Vector3::new(verts[3*v as usize],verts[3*v as usize+1],verts[3*v as usize+2]);if (x-raw).norm()>2e-6*(1.+raw.norm()){return Err(invalid("initial TRS differs from native rest coordinates"));}}
                }
                _=>return Err(invalid("initial STATIC or KINEMATIC only"))
            }
        }
        receipt=serde_json::json!({"feature":CONTROL_FEATURE,"sidecar_sha256":joint::sha256(bytes),"bodies":c.bodies,"scope":"initial_only"});
    }
    for i in 0..n_pin {let ids=indices64(input.bytes(&format!("bin/pin-ind-{i}.bin"))?)?;
        if let Some(expected)=allowed.get(&(i as u32)){if &ids!=expected{return Err(invalid("pin index bytes differ from declared complete ordered membership"));}}
        else if ids.iter().any(|v|support.contains(v)||controlled.contains(v)){return Err(joint::err("fem","FEM_JOINT_PIN_COMPOSITION_UNSUPPORTED","pin",None,None,"undeclared pin overlap"));}
    }
    let mut vertices:Vec<_>=controlled.into_iter().collect();vertices.sort_unstable();Ok((vertices,receipt))
}
