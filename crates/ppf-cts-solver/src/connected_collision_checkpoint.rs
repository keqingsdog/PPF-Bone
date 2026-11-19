//! Versioned collision semantics bound to saved dataset bytes; does not bypass subsystem resume gates.
use crate::connected_collision as cc;
use std::path::Path;
use serde::{Serialize,Deserialize};
fn sha256(bytes:&[u8])->String {use sha2::{Digest,Sha256};format!("{:x}",Sha256::digest(bytes))}
fn bad(code:&str,msg:&str)->String {format!("BONE_CONNECTED_{code}: {msg}")}
#[derive(Debug,Clone,Serialize,Deserialize,PartialEq,Eq)]
#[serde(deny_unknown_fields)]
pub struct Marker {
    pub semantic_version:u32, pub required_feature:String, pub scene_inputs_sha256:String,
    pub canonical_uuid_pairs_sha256:String, pub ownership_sha256:String, pub base_links_sha256:String,
    pub final_links_sha256:String, pub pair_count:usize, pub final_nnz:u32,
    pub dataset_sha256:String, pub dataset_layout_sha256:String,
}
pub const CHECKPOINT:&str="bone_connected_collision_checkpoint_v1.json";
pub fn validate_checkpoint(expected:&Marker,loaded_csr:&crate::cvecvec::CVecVec<u32>,saved:&Marker,dataset_bytes:&[u8],layout_bytes:&[u8])->cc::Result<()> {
    let mut bound=expected.clone();bound.dataset_sha256=sha256(dataset_bytes);bound.dataset_layout_sha256=sha256(layout_bytes);
    if bound!=*saved || cc::digest(loaded_csr)?!=expected.final_links_sha256 {return Err(bad("CHECKPOINT_MISMATCH","semantic/input/pair/owner/base/final CSR or dataset/layout identity changed"));}Ok(())
}
pub fn validate_checkpoint_dir(expected:&Marker,loaded:&crate::cvecvec::CVecVec<u32>,dir:&Path)->cc::Result<()> {
    let bytes=std::fs::read(dir.join(CHECKPOINT)).map_err(|e|bad("CHECKPOINT_UNKNOWN",&e.to_string()))?;
    let saved:Marker=serde_json::from_slice(&bytes).map_err(|e|bad("CHECKPOINT_UNKNOWN",&e.to_string()))?;
    let dataset=std::fs::read(dir.join(ppf_cts_formats::files::DATASET)).map_err(|e|bad("CHECKPOINT_UNKNOWN",&e.to_string()))?;
    let layout=std::fs::read(dir.join("dataset_layout_v1.txt")).map_err(|e|bad("CHECKPOINT_UNKNOWN",&e.to_string()))?;
    validate_checkpoint(expected,loaded,&saved,&dataset,&layout)
}
pub fn record_checkpoint(marker:&Marker,dir:&Path)->cc::Result<()> {
    let dataset=std::fs::read(dir.join(ppf_cts_formats::files::DATASET)).map_err(|e|bad("CHECKPOINT_IO",&e.to_string()))?;
    let layout=std::fs::read(dir.join("dataset_layout_v1.txt")).map_err(|e|bad("CHECKPOINT_IO",&e.to_string()))?;
    let mut bound=marker.clone();bound.dataset_sha256=sha256(&dataset);bound.dataset_layout_sha256=sha256(&layout);
    let bytes=serde_json::to_vec_pretty(&bound).map_err(|e|bad("CHECKPOINT_IO",&e.to_string()))?;
    let temp=dir.join(format!("{CHECKPOINT}.tmp.{}",std::process::id()));
    {use std::io::Write;let mut f=std::fs::File::create(&temp).map_err(|e|bad("CHECKPOINT_IO",&e.to_string()))?;f.write_all(&bytes).and_then(|_|f.sync_all()).map_err(|e|bad("CHECKPOINT_IO",&e.to_string()))?;}
    std::fs::rename(temp,dir.join(CHECKPOINT)).map_err(|e|bad("CHECKPOINT_IO",&e.to_string()))
}
