//! Sealed mesh masks, including ordinary scenes with no joint payload.
use std::{collections::{HashMap,HashSet},fs,path::Path};
use serde::{Deserialize,Serialize};
use crate::joint::{ContextInput,JointError,checked_regular_file_or_absent,safe_existing_file,scan_consumed_paths,sha256,err};
pub const FILE:&str="collision_filter.cbor";
const CAP:&str="ppf_collision_filter_v1";
const RULE:&str="symmetric_membership_mask_and_v1";
#[derive(Debug,Deserialize,Serialize)]
#[serde(deny_unknown_fields)]
struct Envelope {version:u32,kind:String,payload:Payload}
#[derive(Debug,Deserialize,Serialize)]
#[serde(deny_unknown_fields)]
struct Payload {capability:String,rule:String,complete:bool,dynamic_vertices:usize,static_vertices:usize,scene_uuid:String,author_contract_sha256:String,objects:Vec<Object>,inputs:Vec<ContextInput>,scene_inputs_sha256:String}
#[derive(Debug,Deserialize,Serialize)]
#[serde(deny_unknown_fields)]
struct Object {object_uuid:String,pool:String,vertices:Vec<u32>,packed:u32}
#[derive(Debug,Clone)]
pub struct Validated {bytes:HashMap<String,Vec<u8>>,manifest:Vec<u8>}
fn bad(code:&'static str,field:&str,detail:&str)->JointError {err("collision",code,field,None,None,detail)}
impl Validated {
    pub fn bytes(&self,path:&str)->Option<&[u8]> {self.bytes.get(path).map(Vec::as_slice)}
    pub fn verify_unchanged(&self,path:&Path)->Result<(),JointError> {
        let paths=scan_consumed_paths(path)?;
        let expected:HashSet<_>=self.bytes.keys().cloned().collect();
        if paths.into_iter().collect::<HashSet<_>>()!=expected {return Err(bad("COLLISION_FILTER_SEAL_CHANGED","inventory","consumed inventory changed"));}
        for (name,bytes) in &self.bytes {
            let now=fs::read(safe_existing_file(path,name)?).map_err(|e|bad("COLLISION_FILTER_IO",name,&e.to_string()))?;
            if now!=*bytes {return Err(bad("COLLISION_FILTER_SEAL_CHANGED",name,"consumed bytes changed"));}
        }
        let now=fs::read(safe_existing_file(path,FILE)?).map_err(|e|bad("COLLISION_FILTER_IO",FILE,&e.to_string()))?;
        if now!=self.manifest {return Err(bad("COLLISION_FILTER_SEAL_CHANGED",FILE,"manifest changed"));}
        Ok(())
    }
}
fn words(bytes:&[u8])->Result<Vec<u32>,JointError> {
    if bytes.len()%4!=0 {return Err(bad("COLLISION_FILTER_INVALID","bin","not aligned u32"));}
    Ok(bytes.chunks_exact(4).map(|b|u32::from_le_bytes(b.try_into().unwrap())).collect())
}
/// Reject filtered resume before touching any saved state or output files.
pub fn prepare_fresh(path:&Path,required:bool,load:i32)->Result<Option<Validated>,JointError> {
    let input=prepare(path,required)?;
    if input.is_some() && load!=0 {return Err(bad("COLLISION_FILTER_RESUME_UNSUPPORTED","load","filtered simulations currently require a fresh run"));}
    Ok(input)
}
pub fn prepare(path:&Path,required:bool)->Result<Option<Validated>,JointError> {
    let present=checked_regular_file_or_absent(path,FILE)?;
    let bin_exists=match fs::symlink_metadata(path.join("bin")){Ok(_)=>true,Err(e) if e.kind()==std::io::ErrorKind::NotFound=>false,Err(e)=>return Err(bad("COLLISION_FILTER_IO","bin",&e.to_string()))};
    let dyn_present=bin_exists&&checked_regular_file_or_absent(path,"bin/collision_filter.bin")?;
    let static_present=bin_exists&&checked_regular_file_or_absent(path,"bin/static_collision_filter.bin")?;
    if !present && !dyn_present && !static_present && !required {return Ok(None);}
    if !present || !(dyn_present||static_present) {return Err(bad("COLLISION_FILTER_CAPABILITY",FILE,"manifest and at least one mask required"));}
    let manifest=fs::read(safe_existing_file(path,FILE)?).map_err(|e|bad("COLLISION_FILTER_IO",FILE,&e.to_string()))?;
    let mut cursor=std::io::Cursor::new(&manifest);
    let envelope:Envelope=ciborium::from_reader(&mut cursor).map_err(|e|bad("COLLISION_FILTER_CAPABILITY",FILE,&e.to_string()))?;
    if cursor.position()!=manifest.len() as u64 || envelope.version!=1 || envelope.kind!="CollisionFilterInput" || envelope.payload.capability!=CAP || envelope.payload.rule!=RULE || !envelope.payload.complete {
        return Err(bad("COLLISION_FILTER_CAPABILITY",FILE,"version/kind/rule/capability/complete mismatch"));
    }
    let p=envelope.payload;
    if crate::joint::canonical_uuid(&p.scene_uuid).as_deref()!=Some(p.scene_uuid.as_str()) || p.author_contract_sha256.len()!=64 || !p.author_contract_sha256.bytes().all(|c|c.is_ascii_digit()||(b'a'..=b'f').contains(&c)){return Err(bad("COLLISION_FILTER_INVALID","identity","canonical scene UUID and lowercase author hash required"));}
    let paths=scan_consumed_paths(path)?;
    if p.inputs.iter().map(|i|i.relative_path.clone()).collect::<Vec<_>>()!=paths {return Err(bad("COLLISION_FILTER_SEAL_CHANGED","inventory","sorted exact consumed inventory required"));}
    let mut bytes=HashMap::new();
    let mut digest=format!("ppf.collision_filter/v1\n{}\t{}\n{}\t{}\n",p.dynamic_vertices,p.static_vertices,p.scene_uuid,p.author_contract_sha256);
    let mut ids=HashSet::new();let mut covered_dynamic=HashSet::new();let mut covered_static=HashSet::new();
    if p.objects.is_empty() || p.objects.windows(2).any(|w|w[0].object_uuid>=w[1].object_uuid) {return Err(bad("COLLISION_FILTER_INVALID","objects","sorted unique UUID objects required"));}
    for row in &p.objects {
        if crate::joint::canonical_uuid(&row.object_uuid).as_ref()!=Some(&row.object_uuid) || row.object_uuid=="00000000-0000-0000-0000-000000000000" || !ids.insert(&row.object_uuid) {return Err(bad("COLLISION_FILTER_INVALID","objects","canonical UUID required"));}
        let (n,covered)=match row.pool.as_str(){"dynamic"=>(p.dynamic_vertices,&mut covered_dynamic),"static"=>(p.static_vertices,&mut covered_static),_=>return Err(bad("COLLISION_FILTER_INVALID","pool","unknown pool"))};
        if row.vertices.is_empty() || row.vertices.windows(2).any(|w|w[0]>=w[1]) || row.vertices.iter().any(|v|*v as usize>=n || !covered.insert(*v)) {return Err(bad("COLLISION_FILTER_INVALID","vertices","nonempty sorted disjoint in-range vertices required"));}
        if row.pool=="static" && row.vertices.windows(2).any(|w|w[1]!=w[0]+1) {return Err(bad("COLLISION_FILTER_INVALID","vertices","static source range must be contiguous"));}
        digest.push_str(&format!("{}\t{}\t{}\t{}\n",row.object_uuid,row.pool,row.packed,row.vertices.iter().map(u32::to_string).collect::<Vec<_>>().join(",")));
    }
    for input in &p.inputs {
        let raw=fs::read(safe_existing_file(path,&input.relative_path)?).map_err(|e|bad("COLLISION_FILTER_IO",&input.relative_path,&e.to_string()))?;
        if raw.len()!=input.byte_length || sha256(&raw)!=input.sha256 {return Err(bad("COLLISION_FILTER_SEAL_CHANGED",&input.relative_path,"length/hash mismatch"));}
        digest.push_str(&format!("{}\t{}\t{}\n",input.relative_path,input.byte_length,input.sha256));
        bytes.insert(input.relative_path.clone(),raw);
    }
    if sha256(digest.as_bytes())!=p.scene_inputs_sha256 {return Err(bad("COLLISION_FILTER_SEAL_CHANGED","scene_inputs_sha256","digest mismatch"));}
    let required_bytes=|name:&str| bytes.get(name).ok_or_else(||bad("COLLISION_FILTER_INVALID",name,"required consumed input absent"));
    let info:toml::Value=std::str::from_utf8(required_bytes("info.toml")?).ok().and_then(|s|toml::from_str(s).ok()).ok_or_else(||bad("COLLISION_FILTER_INVALID","info.toml","invalid native counts"))?;
    let count=info.get("count").and_then(|v|v.as_table()).ok_or_else(||bad("COLLISION_FILTER_INVALID","count","missing counts"))?;
    for (key,expected) in [("vert",p.dynamic_vertices),("static_vert",p.static_vertices)] {
        if count.get(key).and_then(|v|v.as_integer())!=Some(expected as i64) {return Err(bad("COLLISION_FILTER_INVALID",key,"manifest/native vertex count mismatch"));}
    }
    let dynamic=if dyn_present {words(required_bytes("bin/collision_filter.bin")?)?} else {vec![0;p.dynamic_vertices]};
    let static_mask=if static_present {words(required_bytes("bin/static_collision_filter.bin")?)?} else {vec![0;p.static_vertices]};
    if dynamic.len()!=p.dynamic_vertices || static_mask.len()!=p.static_vertices || !dynamic.iter().chain(&static_mask).any(|v|*v!=0) || (dyn_present&&!dynamic.iter().any(|v|*v!=0)) || (static_present&&!static_mask.iter().any(|v|*v!=0)) {return Err(bad("COLLISION_FILTER_INVALID","masks","one u32 per pool vertex; enabled bins must be nonzero"));}
    let map=crate::joint::collision_vertex_map(required_bytes("map.pickle")?)?;
    for row in &p.objects {
        let values=if row.pool=="dynamic" {&dynamic} else {&static_mask};
        if row.vertices.iter().any(|v|values[*v as usize]!=row.packed) {return Err(bad("COLLISION_FILTER_INVALID","objects","packed source mapping mismatch"));}
        if row.pool=="dynamic" {
            let mut expected=map.get(&row.object_uuid).cloned().ok_or_else(||bad("COLLISION_FILTER_INVALID","objects","UUID source absent"))?;
            expected.sort_unstable();
            if expected!=row.vertices {return Err(bad("COLLISION_FILTER_INVALID","objects","UUID source map mismatch"));}
        }
    }
    if dynamic.iter().enumerate().any(|(i,v)|*v!=0&&!covered_dynamic.contains(&(i as u32))) || static_mask.iter().enumerate().any(|(i,v)|*v!=0&&!covered_static.contains(&(i as u32))) {return Err(bad("COLLISION_FILTER_INVALID","masks","enabled vertices must have sealed source UUID"));}
    for (name,values,arity) in [("bin/tri.bin",&dynamic,3),("bin/rod.bin",&dynamic,2),("bin/static_tri.bin",&static_mask,3)] {
        if let Some(raw)=bytes.get(name) {
            // Official topology binaries are little-endian u64 (mask binaries are u32).
            if raw.len()%8!=0{return Err(bad("COLLISION_FILTER_INVALID",name,"topology must be aligned u64"));}
            let indices:Vec<u64>=raw.chunks_exact(8).map(|b|u64::from_le_bytes(b.try_into().unwrap())).collect();
            if indices.len()%arity!=0 {return Err(bad("COLLISION_FILTER_INVALID",name,"topology arity mismatch"));}
            for primitive in indices.chunks_exact(arity) {
                if primitive.iter().any(|v|*v as usize>=values.len()) || primitive.iter().any(|v|values[*v as usize]!=values[primitive[0] as usize]) {return Err(bad("COLLISION_FILTER_INVALID",name,"primitive must have one object-constant mask"));}
            }
        }
    }
    let input=Validated{bytes,manifest};input.verify_unchanged(path)?;Ok(Some(input))
}


#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(name:&str)->std::path::PathBuf {
        Path::new(&std::env::var("PPF_COLLISION_GENERIC_SEALED_ROOT").expect("explicit actual fixture root required")).join(name)
    }
    fn copy_tree(from:&Path,to:&Path){fs::create_dir_all(to).unwrap();for e in fs::read_dir(from).unwrap(){let e=e.unwrap();if e.file_type().unwrap().is_dir(){copy_tree(&e.path(),&to.join(e.file_name()));}else{fs::copy(e.path(),to.join(e.file_name())).unwrap();}}}
    fn copied()->std::path::PathBuf {
        let root=Path::new(&std::env::var("PPF_COLLISION_GENERIC_TEST_AREA").expect("explicit task test area")).join(format!("generic-{}-{}",std::process::id(),std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        copy_tree(&fixture("same"),&root);root
    }
    fn write_manifest(path:&Path,env:&Envelope){let mut bytes=Vec::new();ciborium::into_writer(env,&mut bytes).unwrap();fs::write(path.join(FILE),bytes).unwrap();}
    fn read_manifest(path:&Path)->Envelope{ciborium::from_reader(fs::read(path.join(FILE)).unwrap().as_slice()).unwrap()}
    fn reseal(path:&Path,env:&mut Envelope){
        let p=&mut env.payload;
        let mut digest=format!("ppf.collision_filter/v1\n{}\t{}\n{}\t{}\n",p.dynamic_vertices,p.static_vertices,p.scene_uuid,p.author_contract_sha256);
        for row in &p.objects{digest.push_str(&format!("{}\t{}\t{}\t{}\n",row.object_uuid,row.pool,row.packed,row.vertices.iter().map(u32::to_string).collect::<Vec<_>>().join(",")));}
        for input in &mut p.inputs{let raw=fs::read(path.join(&input.relative_path)).unwrap();input.byte_length=raw.len();input.sha256=sha256(&raw);digest.push_str(&format!("{}\t{}\t{}\n",input.relative_path,input.byte_length,input.sha256));}
        p.scene_inputs_sha256=sha256(digest.as_bytes());write_manifest(path,env);
    }
    #[test]
    #[ignore="requires actual Blender ordinary-only STATIC sealed fixture"]
    fn collision_filter_generic_actual_sealed_intake_and_tamper_refusals(){
        for name in ["same","disjoint","reciprocal","one_way","zero_mask","zero_member","static_zero"] {
            let path=fixture(name);let input=prepare_fresh(&path,true,0).unwrap().unwrap();for load in [-1,1,9]{assert_eq!(prepare_fresh(&path,true,load).unwrap_err().stable_code,"COLLISION_FILTER_RESUME_UNSUPPORTED");}assert!(input.bytes("bin/static_collision_filter.bin").is_some());input.verify_unchanged(&path).unwrap();
            assert!(!path.join(crate::joint::JOINT_FILE).exists());
        }
        assert!(prepare(&fixture("disabled"),false).unwrap().is_none());assert!(prepare_fresh(&fixture("disabled"),false,9).unwrap().is_none());
        assert_eq!(prepare(&fixture("disabled"),true).unwrap_err().stable_code,"COLLISION_FILTER_CAPABILITY");
        let path=copied();let input=prepare(&path,true).unwrap().unwrap();
        let target=path.join("bin/static_collision_filter.bin");let mut raw=fs::read(&target).unwrap();raw[0]^=1;fs::write(&target,&raw).unwrap();
        assert_eq!(prepare(&path,true).unwrap_err().stable_code,"COLLISION_FILTER_SEAL_CHANGED");assert!(input.verify_unchanged(&path).is_err());
        let path=copied();let target=path.join("bin/static_collision_filter.bin");let mut raw=fs::read(&target).unwrap();raw.extend(1u32.to_le_bytes());fs::write(&target,raw).unwrap();
        let mut env=read_manifest(&path);reseal(&path,&mut env);assert_eq!(prepare(&path,true).unwrap_err().stable_code,"COLLISION_FILTER_INVALID");
        let path=copied();let mut env=read_manifest(&path);env.payload.capability="unknown".into();write_manifest(&path,&env);assert_eq!(prepare(&path,true).unwrap_err().stable_code,"COLLISION_FILTER_CAPABILITY");
        let path=copied();fs::remove_file(path.join(FILE)).unwrap();assert_eq!(prepare(&path,false).unwrap_err().stable_code,"COLLISION_FILTER_CAPABILITY");
        let path=copied();let mut env=read_manifest(&path);env.payload.objects[0].vertices[0]=u32::MAX;reseal(&path,&mut env);assert_eq!(prepare(&path,true).unwrap_err().stable_code,"COLLISION_FILTER_INVALID");
    }
}
