//! Trusted launch-generation fence. Never a solver status or monitor input.
use std::{fs, io::{Read, Write}, path::Path};
use serde::{Deserialize, Serialize};
use ppf_cts_core::state::ServerState;

const FILE: &str = "status.resume-baseline.json";
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Baseline {
    kind: String, version: u32, name: String, root: String, upload_id: String,
    param_hash: String, data_hash: String, launch_id: String,
}
impl Baseline {
    pub(crate) fn new(s: &ServerState, launch_id: &str) -> Result<Self, &'static str> {
        if [s.name.as_str(),s.root.as_str(),s.upload_id.as_str(),s.param_hash.as_str(),s.data_hash.as_str(),launch_id].iter().any(|v|v.is_empty()) { return Err("BASELINE_CONTEXT_INVALID"); }
        Ok(Self { kind:"ppf.resume_generation_baseline".into(), version:1, name:s.name.clone(), root:s.root.clone(), upload_id:s.upload_id.clone(),param_hash:s.param_hash.clone(),data_hash:s.data_hash.clone(),launch_id:launch_id.into() })
    }
    pub(crate) fn matches(&self, s:&ServerState)->bool {
        self.kind=="ppf.resume_generation_baseline" && self.version==1 && !self.launch_id.is_empty() && self.launch_id.len()<=128 &&
        self.name==s.name && self.root==s.root && self.upload_id==s.upload_id && self.param_hash==s.param_hash && self.data_hash==s.data_hash &&
        !self.name.is_empty() && !self.root.is_empty() && !self.upload_id.is_empty() && !self.param_hash.is_empty() && !self.data_hash.is_empty()
    }
}
pub(crate) fn read(out:&Path,s:&ServerState)->Result<Baseline,&'static str> {
    let p=out.join(FILE);let m=fs::symlink_metadata(&p).map_err(|_|"BASELINE_UNAVAILABLE")?;
    if !m.is_file() || m.file_type().is_symlink() || m.len()>8192 {return Err("BASELINE_INVALID")}
    #[cfg(windows)]
    {use std::os::windows::fs::MetadataExt; if m.file_attributes() & 0x400 != 0 {return Err("BASELINE_INVALID")} }
    let mut bytes=Vec::new();fs::File::open(p).map_err(|_|"BASELINE_UNAVAILABLE")?.take(8193).read_to_end(&mut bytes).map_err(|_|"BASELINE_UNAVAILABLE")?;
    if bytes.len()>8192 {return Err("BASELINE_INVALID")}
    let b:Baseline=serde_json::from_slice(&bytes).map_err(|_|"BASELINE_INVALID")?;
    if !b.matches(s) {return Err("BASELINE_CONTEXT_MISMATCH")} Ok(b)
}
pub(crate) fn write(out:&Path,b:&Baseline)->Result<(), &'static str> {
    let bytes=serde_json::to_vec(b).map_err(|_|"BASELINE_WRITE_FAILED")?;
    if bytes.len()>8192 {return Err("BASELINE_WRITE_FAILED")}
    let p=out.join(FILE);let tmp=out.join(format!(".resume-baseline-{}.tmp",uuid::Uuid::new_v4()));
    let result=(|| {let mut f=fs::OpenOptions::new().write(true).create_new(true).open(&tmp)?;f.write_all(&bytes)?;f.sync_all()?;drop(f);fs::rename(&tmp,&p)})();
    if result.is_err(){let _=fs::remove_file(tmp);return Err("BASELINE_WRITE_FAILED")} Ok(())
}
pub(crate) fn query(engine:&crate::engine::ServerEngine)->Result<Baseline,&'static str> {
    let before=engine.state();let root=Path::new(&before.root);let out=ppf_cts_formats::files::session_output_dir(root);
    if crate::upload::read_upload_id(root).ok().as_deref()!=Some(before.upload_id.as_str()){return Err("PROJECT_CONTEXT_CHANGED")}
    match fs::symlink_metadata(out.join(ppf_cts_formats::files::STATUS_RECORD)) { Err(e) if e.kind()==std::io::ErrorKind::NotFound=>(), _=>return Err("BASELINE_STATUS_NOT_ABSENT") }
    let b=read(&out,&before)?;let after=engine.state();
    if before.project_generation!=after.project_generation || !b.matches(&after) || crate::upload::read_upload_id(root).ok().as_deref()!=Some(before.upload_id.as_str()){return Err("PROJECT_CONTEXT_CHANGED")}
    match fs::symlink_metadata(out.join(ppf_cts_formats::files::STATUS_RECORD)) { Err(e) if e.kind()==std::io::ErrorKind::NotFound=>Ok(b), _=>Err("BASELINE_STATUS_NOT_ABSENT") }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn state()->ServerState { let mut s=ServerState::default();s.name="p".into();s.root="root".into();s.upload_id="upload".into();s.param_hash="param".into();s.data_hash="data".into();s }
    #[test]
    fn bounded_baseline_rewrite_and_identity() {
        let d=tempfile::tempdir().unwrap();let s=state();
        let a=Baseline::new(&s,"launch1").unwrap();write(d.path(),&a).unwrap();assert_eq!(read(d.path(),&s).unwrap(),a);
        let b=Baseline::new(&s,"launch2").unwrap();write(d.path(),&b).unwrap();assert_eq!(read(d.path(),&s).unwrap(),b);
        let mut other=s.clone();other.upload_id="other".into();assert!(read(d.path(),&other).is_err());
        fs::write(d.path().join(FILE),b"{\"kind\":\"x\",\"kind\":\"x\"}").unwrap();assert!(read(d.path(),&s).is_err());
        fs::write(d.path().join(FILE),vec![b' ';8193]).unwrap();assert!(read(d.path(),&s).is_err());
    }
    #[test]
    fn held_status_lease_refuses_second_admission() {
        let d=tempfile::tempdir().unwrap();
        let first=ppf_cts_formats::status::lock::acquire(d.path()).unwrap();
        assert!(ppf_cts_formats::status::lock::acquire(d.path()).is_err());
        drop(first);
        assert!(ppf_cts_formats::status::lock::acquire(d.path()).is_ok());
    }
}
