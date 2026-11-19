//! Retained-input proof and checkpoint provenance for host connected-pair CSR.
use crate::{connected_collision as cc, joint::{self,ValidatedJointInput}, soft_joint::{BodyWire,JointRow}};
use std::path::Path;
pub const CROSS_FEATURE:&str="cross_link_collision_pairs_v1";
pub const CROSS_FILE:&str="bin/cross_link_collision_pairs_v1.json";
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CrossPairs {version:u32,kind:String,scene_uuid:String,pairs:Vec<[String;2]>}
fn decode_cross_pairs(bytes:&[u8],scene_uuid:&str)->cc::Result<Vec<[String;2]>> {
    if bytes.len()>16*1024*1024 {return Err(bad("INPUT","cross-link pair sidecar exceeds 16 MiB"));}
    let strict:joint::StrictValue=serde_json::from_slice(bytes).map_err(|e|bad("INPUT",&e.to_string()))?;
    let doc:CrossPairs=serde_json::from_value(strict.0).map_err(|e|bad("INPUT",&e.to_string()))?;
    if doc.version!=1 || doc.kind!="ppf.cross_link_collision_pairs" || doc.scene_uuid!=scene_uuid
       || doc.pairs.is_empty() || doc.pairs.len()>65536 {return Err(bad("INPUT","invalid cross-link pair identity/count"));}
    let mut prev:Option<&[String;2]>=None;
    for pair in &doc.pairs {
        if pair[0]>=pair[1] || pair.iter().any(|s|joint::canonical_uuid(s).as_ref()!=Some(s))
           || prev.is_some_and(|p|p>=pair) {return Err(bad("PAIR","cross-link pairs must be canonical, sorted and unique"));}
        prev=Some(pair);
    }
    Ok(doc.pairs)
}
fn cross_pairs(input:&ValidatedJointInput)->cc::Result<Vec<[String;2]>> {
    let present=input.contains(CROSS_FILE);
    let feature=input.requires_feature(CROSS_FEATURE).map_err(|e|e.to_string())?;
    if present!=feature {return Err(bad("FEATURE","cross-link pair sidecar and feature must appear together"));}
    if !present {return Ok(Vec::new());}
    decode_cross_pairs(required(input,CROSS_FILE)?,&input.records().scene_uuid)
}
#[cfg(test)]
mod cross_pair_tests {
    use super::*;
    #[test]
    fn sidecar_requires_identity_canonical_order_and_unique_pairs() {
        let a="11111111-1111-4111-8111-111111111111";
        let b="22222222-2222-4222-8222-222222222222";
        let scene="33333333-3333-4333-8333-333333333333";
        let make=|pairs:serde_json::Value,scene_id:&str|serde_json::to_vec(&serde_json::json!({"version":1,"kind":"ppf.cross_link_collision_pairs","scene_uuid":scene_id,"pairs":pairs})).unwrap();
        assert_eq!(decode_cross_pairs(&make(serde_json::json!([[a,b]]),scene),scene).unwrap().len(),1);
        for invalid in [serde_json::json!([[b,a]]),serde_json::json!([[a,a]]),serde_json::json!([[a,b],[a,b]]),serde_json::json!([])] {
            assert!(decode_cross_pairs(&make(invalid,scene),scene).is_err());
        }
        assert!(decode_cross_pairs(&make(serde_json::json!([[a,b]]),a),scene).is_err());
        let duplicate=br#"{"version":1,"version":1,"kind":"ppf.cross_link_collision_pairs","scene_uuid":"x","pairs":[]}"#;
        assert!(decode_cross_pairs(duplicate,scene).is_err());
    }
}
#[derive(Debug,Clone)]
pub struct Prepared { pub proof:cc::Proof, pub base:Vec<u32>, pub dynamic:usize, pub static_count:usize, pub base_sha256:String }
pub use crate::connected_collision_checkpoint::{Marker,record_checkpoint,validate_checkpoint_dir};
fn bad(code:&str,msg:&str)->String {format!("BONE_CONNECTED_{code}: {msg}")}
fn words(bytes:&[u8],width:usize)->cc::Result<Vec<u32>> {
    if bytes.len()%width!=0 {return Err(bad("INPUT","index alignment"));}
    let mut out=Vec::new();cc::Allocation::default().reserve(&mut out,bytes.len()/width)?;
    for b in bytes.chunks_exact(width) {out.push(if width==4 {u32::from_le_bytes(b.try_into().unwrap())} else {u32::try_from(u64::from_le_bytes(b.try_into().unwrap())).map_err(|_|bad("INDEX_DOMAIN","native index exceeds u32"))?});}Ok(out)
}
fn required<'a>(input:&'a ValidatedJointInput,name:&str)->cc::Result<&'a[u8]>{input.bytes(name).map_err(|e|e.to_string())}
pub fn prepare(input:&ValidatedJointInput,bodies:&[BodyWire],rows:&[JointRow])->cc::Result<Prepared> {
    let object=words(required(input,"bin/object_vert.bin")?,4)?;
    let rigid=words(required(input,"bin/pdrd_vert_index.bin")?,4)?;
    let dynamic=required(input,"bin/vert.bin")?.len()/24;
    let static_count=if input.contains("bin/static_vert.bin") {let b=required(input,"bin/static_vert.bin")?;if b.len()%24!=0{return Err(bad("INPUT","static vertex shape"));}b.len()/24} else {0};
    if object.len()!=dynamic || rigid.len()!=dynamic {return Err(bad("OWNERSHIP","complete native ownership arrays required"));}
    let map=joint::collision_vertex_map(required(input,"map.pickle")?).map_err(|e|e.to_string())?;
    let mut owners=Vec::new();let mut fem=Vec::new();let mut support=Vec::new();let mut alloc=cc::Allocation::default();
    alloc.reserve(&mut owners,bodies.len())?;alloc.reserve(&mut fem,bodies.len())?;alloc.reserve(&mut support,bodies.len())?;
    // UUID order makes pair digests and owner digests independent of transport order.
    let mut order:Vec<&BodyWire>=Vec::new();alloc.reserve(&mut order,bodies.len())?;order.extend(bodies);order.sort_unstable_by(|a,b|a.body_uuid.cmp(&b.body_uuid));
    for b in order {
        if joint::canonical_uuid(&b.body_uuid).as_ref()!=Some(&b.body_uuid) {return Err(bad("OWNERSHIP","bound body UUID must be canonical"));}
        let source=map.get(&b.body_uuid).ok_or_else(||bad("OWNERSHIP","bound body missing from retained vertex map"))?;
        let mut members=Vec::new();alloc.reserve(&mut members,source.len())?;members.extend_from_slice(source);members.sort_unstable();
        let id=*members.first().and_then(|v|object.get(*v as usize)).ok_or_else(||bad("OWNERSHIP","empty or out-of-range member"))?;
        if id as usize>=dynamic {return Err(bad("OWNERSHIP","sentinel/non-genuine object owner"));}
        let is_fem=b.kind=="FEM_TET_MATERIAL";
        if !is_fem {
            let runtime=b.runtime_body_index.ok_or_else(||bad("OWNERSHIP","missing real PDRD index"))?;
            let expected=runtime.checked_add(1).ok_or_else(||bad("INDEX_DOMAIN","PDRD index overflow"))?;
            if members.iter().any(|v|rigid.get(*v as usize)!=Some(&expected)) {return Err(bad("OWNERSHIP","PDRD map/index mismatch"));}
            if rigid.iter().filter(|&&x|x==expected).count()!=members.len() {return Err(bad("OWNERSHIP","incomplete PDRD member bijection"));}
        }
        owners.push(cc::Owner{uuid:b.body_uuid.clone(),object:id,members});fem.push(is_fem);support.push(b.vertex_indices);
    }
    cc::validate_owners(&owners,&object,&rigid,&fem,&support,&[])?;
    for (name,width) in [("bin/tri.bin",3usize),("bin/rod.bin",2usize),("bin/tet.bin",4usize)] {
        if !input.contains(name) {continue;}let raw=words(required(input,name)?,8)?;
        if raw.len()%width!=0 {return Err(bad("OWNERSHIP","simplex width mismatch"));}
        for q in raw.chunks_exact(width) {cc::validate_simplex(&owners,&object,q)?;}
    }
    let extra=cross_pairs(input)?;
    let capacity=input.records().joints.len().checked_add(rows.len()).and_then(|n|n.checked_add(extra.len())).ok_or_else(||bad("CAPACITY","joint count overflow"))?;
    let mut pairs=Vec::new();alloc.reserve(&mut pairs,capacity)?;
    let wires=input.records().joints.iter().map(|j|&j.wire).chain(rows.iter().map(|j|&j.wire));
    for w in wires {
        if !w.enabled || w.collide_connected || w.parent_body_uuid==joint::WORLD_ENDPOINT_UUID {continue;}
        let a=owners.binary_search_by(|o|o.uuid.cmp(&w.parent_body_uuid)).map_err(|_|bad("PAIR","parent absent from full ownership proof"))?;
        let b=owners.binary_search_by(|o|o.uuid.cmp(&w.child_body_uuid)).map_err(|_|bad("PAIR","child absent from full ownership proof"))?;
        if a==b {return Err(bad("PAIR","identical endpoints"));}pairs.push((a.min(b),a.max(b)));
    }
    for pair in &extra {
        let a=owners.binary_search_by(|o|o.uuid.cmp(&pair[0])).map_err(|_|bad("PAIR","cross-link endpoint absent from full ownership proof"))?;
        let b=owners.binary_search_by(|o|o.uuid.cmp(&pair[1])).map_err(|_|bad("PAIR","cross-link endpoint absent from full ownership proof"))?;
        if a==b {return Err(bad("PAIR","identical cross-link endpoints"));}
        pairs.push((a.min(b),a.max(b)));
    }
    pairs.sort_unstable();pairs.dedup();
    use sha2::{Digest,Sha256};let mut ph=Sha256::new();let mut oh=Sha256::new();
    for &(a,b) in &pairs {ph.update(owners[a].uuid.as_bytes());ph.update(b"\t");ph.update(owners[b].uuid.as_bytes());ph.update(b"\n");}
    oh.update(b"ppf.bone_connected.owner/v1\n");
    for o in &owners {oh.update(o.uuid.as_bytes());oh.update(b"\n");oh.update(o.object.to_le_bytes());for v in &o.members {oh.update(v.to_le_bytes());}oh.update(b"\n");}
    let base_bytes=if input.contains("bin/start_link.bin") {required(input,"bin/start_link.bin")?} else {&[]};
    let base=words(base_bytes,4)?;
    let material_digest=if input.contains(crate::soft_joint::FILE) {joint::sha256(required(input,crate::soft_joint::FILE)?)} else {String::new()};
    let input_sha256=joint::sha256(format!("ppf.bone_connected.input/v1\n{}\n{}\n{}\n",input.records().scene_inputs_sha256,input.records().payload_sha256,material_digest).as_bytes());
    Ok(Prepared{proof:cc::Proof{owners,pairs,input_sha256,pair_sha256:format!("{:x}",ph.finalize()),owner_sha256:format!("{:x}",oh.finalize())},base,dynamic,static_count,base_sha256:joint::sha256(base_bytes)})
}
impl Prepared {
    pub fn marker(&self,table:&crate::cvecvec::CVecVec<u32>)->cc::Result<Marker> {
        Ok(Marker{semantic_version:cc::SEMANTIC_VERSION,required_feature:cc::FEATURE.into(),scene_inputs_sha256:self.proof.input_sha256.clone(),canonical_uuid_pairs_sha256:self.proof.pair_sha256.clone(),ownership_sha256:self.proof.owner_sha256.clone(),base_links_sha256:self.base_sha256.clone(),final_links_sha256:cc::digest(table)?,pair_count:self.proof.pairs.len(),final_nnz:table.nnz,dataset_sha256:String::new(),dataset_layout_sha256:String::new()})
    }
    pub fn build(&self,allocation:&mut cc::Allocation)->cc::Result<crate::cvecvec::CVecVec<u32>> {cc::build(self.dynamic,self.static_count,&self.base,&self.proof.owners,&self.proof.pairs,allocation)}
}
/// Swap only after construction and loaded-state provenance validation succeed.
pub fn validate_resume(data:&crate::data::DataSet,prepared:&Prepared,out:&Path)->cc::Result<Marker>{
    if data.vertex.curr.size as usize!=prepared.dynamic||data.constraint.mesh.vertex.size as usize!=prepared.static_count{return Err(bad("OWNERSHIP","loaded dynamic/static namespace differs"));}
    let props=data.prop.vertex.as_slice();
    for owner in &prepared.proof.owners {for &v in &owner.members {if props.get(v as usize).map(|p|p.object_index)!=Some(owner.object){return Err(bad("OWNERSHIP","loaded owner partition differs"));}}}
    // Authenticate the already-filtered saved CSR. Do not filter/rebuild it again.
    let marker=prepared.marker(&data.start_link)?;
    validate_checkpoint_dir(&marker,&data.start_link,out)?;Ok(marker)
}
pub fn install(data:&mut crate::data::DataSet,prepared:&Prepared,load:i32,out:&Path)->cc::Result<Marker> {
    if data.vertex.curr.size as usize!=prepared.dynamic || data.prop.vertex.size as usize!=prepared.dynamic || data.constraint.mesh.vertex.size as usize!=prepared.static_count {return Err(bad("OWNERSHIP","runtime dynamic domain differs from retained proof"));}
    let props=data.prop.vertex.as_slice();
    for o in &prepared.proof.owners {for &v in &o.members {if props[v as usize].object_index!=o.object {return Err(bad("OWNERSHIP","runtime object owner differs from sealed proof"));}}}
    let (table,costs)=cc::build_reported(prepared.dynamic,prepared.static_count,&prepared.base,&prepared.proof.owners,&prepared.proof.pairs,&mut cc::Allocation::default()).map_err(|e|format!("{e}; input={} pair={}",prepared.proof.input_sha256,prepared.proof.pair_sha256))?;
    let marker=prepared.marker(&table)?;
    if load!=0 {validate_checkpoint_dir(&marker,&data.start_link,out)?;}
    let old_data=cc::checked_span::<u32>(data.start_link.nnz_allocated as usize)?;
    let old_offset=cc::checked_span::<u32>(data.start_link.offset_allocated as usize)?;
    let host_construction_bytes=old_data.checked_add(old_offset).and_then(|x|x.checked_add(costs.final_allocated_bytes)).and_then(|x|x.checked_add(costs.row_scratch_bytes)).and_then(|x|x.checked_add(costs.owner_lookup_bytes)).ok_or_else(||bad("CAPACITY","host construction byte estimate overflow"))?;
    let report=serde_json::json!({"kind":"ppf.bone_connected_collision.host_prepared","semantic_version":cc::SEMANTIC_VERSION,
        "input_sha256":prepared.proof.input_sha256,"pair_sha256":prepared.proof.pair_sha256,"owner_sha256":prepared.proof.owner_sha256,
        "final_links_sha256":marker.final_links_sha256,"costs":costs,"host_construction_bytes":host_construction_bytes,
        "host_estimate_scope":"old+new CSR, one row scratch and owner lookup; excludes retained input/proof/decode and allocator overhead",
        "gpu_uploaded":false,"physical_acceptance":"not_assessed"});
    let bytes=serde_json::to_vec_pretty(&report).map_err(|e|bad("RECEIPT_IO",&e.to_string()))?;
    std::fs::write(out.join("bone_connected_collision_host_prepared.json"),bytes).map_err(|e|bad("RECEIPT_IO",&e.to_string()))?;
    data.start_link=table;Ok(marker)
}
