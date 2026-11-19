//! Positional DataSet checkpoints require this exact vertex-property layout.
use std::{fs,path::Path};
pub const MARKER:&str="dataset_layout_v1.txt";
const LAYOUT:&str="ppf-dataset-vertex-prop-collision-filter-u32-v1";
fn note(bytes:&[u8])->String{format!("{LAYOUT}\nsha256={}\n",crate::joint::sha256(bytes))}
pub fn validate(dir:&Path)->Result<(),String>{
    let marker=fs::read_to_string(dir.join(MARKER)).map_err(|_|"CHECKPOINT_LAYOUT_UNSUPPORTED: missing dataset layout marker; run a fresh simulation".to_string())?;
    let bytes=fs::read(dir.join(ppf_cts_formats::files::DATASET)).map_err(|e|format!("CHECKPOINT_LAYOUT_IO: {e}"))?;
    if marker!=note(&bytes){return Err("CHECKPOINT_LAYOUT_UNSUPPORTED: dataset layout/hash mismatch; run a fresh simulation".into());}
    Ok(())
}
pub fn invalidate(dir:&Path){
    match fs::remove_file(dir.join(MARKER)){Ok(())=>(),Err(e) if e.kind()==std::io::ErrorKind::NotFound=>(),Err(e)=>panic!("checkpoint marker invalidate: {e}")}
}
/// Call only after the atomic serialized dataset write succeeded.
pub fn record(dir:&Path){
    let bytes=fs::read(dir.join(ppf_cts_formats::files::DATASET)).expect("complete saved dataset");
    let target=dir.join(MARKER);let tmp=dir.join(format!("{MARKER}.tmp.{}",std::process::id()));
    use std::io::Write;let mut f=fs::File::create(&tmp).expect("checkpoint marker create");
    f.write_all(note(&bytes).as_bytes()).expect("checkpoint marker write");f.sync_all().expect("checkpoint marker sync");drop(f);
    fs::rename(tmp,target).expect("checkpoint marker atomic rename");
}

#[cfg(test)] mod tests {
 use super::*;
 #[test] fn collision_filter_checkpoint_actual_dataset_roundtrip_and_layout_refusals(){
  let dir=std::env::var_os("PPF_COLLISION_TEST_AREA").map(std::path::PathBuf::from).unwrap_or_else(||std::env::temp_dir().join("ppf-collision-checkpoint-cpu"));
  let dir=dir.join(format!("layout-{}-{}",std::process::id(),std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));fs::create_dir_all(&dir).unwrap();
  // The neutral driver's existing CPU fixtures use zeroed empty CVec/DataSet aggregates.
  let mut data:Box<crate::data::DataSet>=Box::new(unsafe{std::mem::zeroed()});
  let mut p=crate::data::VertexProp::default();p.mass=2.0;p.object_index=7;p.collision_filter=0;
  data.prop.vertex=crate::cvec::CVec::from(&[p][..]);
  let path=dir.join(ppf_cts_formats::files::DATASET);super::super::save(&*data,path.to_str().unwrap());
  assert!(validate(&dir).unwrap_err().contains("missing"));record(&dir);validate(&dir).unwrap();
  let loaded:crate::data::DataSet=super::super::read(&super::super::read_gz(path.to_str().unwrap()));
  assert_eq!(loaded.prop.vertex.as_slice()[0].collision_filter,0);assert_eq!(loaded.prop.vertex.as_slice()[0].object_index,7);assert_eq!(loaded.prop.vertex.as_slice()[0].mass,2.0);
  fs::write(dir.join(MARKER),"old-layout\n").unwrap();assert!(validate(&dir).unwrap_err().contains("layout/hash mismatch"));
  record(&dir);let mut raw=fs::read(&path).unwrap();raw.push(0);fs::write(&path,raw).unwrap();assert!(validate(&dir).is_err());
  fs::remove_file(&path).unwrap();assert!(validate(&dir).is_err());assert!(!dir.join(format!("{MARKER}.tmp.{}",std::process::id())).exists());
 }
}
