//! Immutable accepted checkpoint publication. State is the final discovery marker.
//! No hashes, global transaction pointer, or changes to legacy State bytes.
use std::{fs::{self,OpenOptions},io::Write,path::Path};
fn error(s:impl std::fmt::Display)->String{format!("CHECKPOINT_COMMIT_INVALID: {s}")}
pub fn history_name(frame:i32)->String{format!("bone_history_{frame}.bin")}
const MAGIC:&[u8;8]=b"PPFBONE1";
pub fn encode_history(frame:i32,time:f64,payload:&[u8])->Result<Vec<u8>,String>{
    if frame<0||!time.is_finite()||time<0.{return Err(error("history frame/clock"));}
    let mut out=Vec::with_capacity(32+payload.len());out.extend_from_slice(MAGIC);out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&frame.to_le_bytes());out.extend_from_slice(&time.to_bits().to_le_bytes());
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());out.extend_from_slice(payload);Ok(out)
}
pub fn decode_history(frame:i32,data:&[u8])->Result<(f64,&[u8]),String>{
    if data.len()<32||&data[..8]!=MAGIC||u32::from_le_bytes(data[8..12].try_into().unwrap())!=1||
        i32::from_le_bytes(data[12..16].try_into().unwrap())!=frame{return Err(error("history magic/version/frame"));}
    let time=f64::from_bits(u64::from_le_bytes(data[16..24].try_into().unwrap()));
    let n=u64::from_le_bytes(data[24..32].try_into().unwrap());
    if frame<0||!time.is_finite()||time<0.||n!=data.len() as u64-32{return Err(error("history clock/payload length"));}
    Ok((time,&data[32..]))
}
fn bytes(path:&Path)->Result<Option<Vec<u8>>,String>{
    match fs::symlink_metadata(path){
        Err(e) if e.kind()==std::io::ErrorKind::NotFound=>Ok(None),
        Err(e)=>Err(error(e)),
        Ok(m) if !m.is_file()||m.file_type().is_symlink()=>Err(error("regular checkpoint file required")),
        Ok(_)=>fs::read(path).map(Some).map_err(error),
    }
}
fn publish_new(path:&Path,data:&[u8])->Result<(),String>{
    let temp=path.with_extension(format!("stage.{}",std::process::id()));
    let mut f=OpenOptions::new().write(true).create_new(true).open(&temp).map_err(error)?;
    f.write_all(data).and_then(|_|f.sync_all()).map_err(error)?;drop(f);
    // A same-volume hard link is atomic and refuses an existing destination.
    // rename can replace on Unix; create_new + write would expose partial bytes.
    let linked=fs::hard_link(&temp,path).map_err(error);
    let cleanup=fs::remove_file(&temp).map_err(error);
    linked?;cleanup?;Ok(())
}
/// Publish a checkpoint without a history sidecar. State remains the final
/// discovery marker; existing or orphaned files must match exactly.
pub fn publish_without_history(dir:&Path,frame:i32,state:&[u8],plastic:Option<&[u8]>)->Result<(),String>{
    if frame<0{return Err(error("negative frame"));}
    let sp=dir.join(format!("state_{frame}.bin.gz"));
    let pp=dir.join(format!("plastic_{frame}.bin.gz"));
    let old_state=bytes(&sp)?;let old_plastic=bytes(&pp)?;
    if let Some(existing)=old_state.as_deref(){
        if existing==state && old_plastic.as_deref()==plastic{return Ok(());}
        return Err("CHECKPOINT_FRAME_IMMUTABLE: existing State/plastic differ or are incomplete".into());
    }
    match (old_plastic.as_deref(),plastic){
        (Some(a),Some(b)) if a==b=>{},
        (None,Some(b))=>publish_new(&pp,b)?,
        (None,None)=>{},
        _=>return Err(error("incomplete plastic predecessor differs; preserve for inspection")),
    }
    publish_new(&sp,state)
}
pub fn publish(dir:&Path,frame:i32,state:&[u8],history:&[u8],plastic:Option<&[u8]>)->Result<(),String>{
    if frame<0{return Err(error("negative frame"));}
    let sp=dir.join(format!("state_{frame}.bin.gz"));
    let hp=dir.join(history_name(frame));let pp=dir.join(format!("plastic_{frame}.bin.gz"));
    let old_state=bytes(&sp)?;let old_history=bytes(&hp)?;let old_plastic=bytes(&pp)?;
    if old_state.is_some(){
        if old_state.as_deref()==Some(state)&&old_history.as_deref()==Some(history)&&old_plastic.as_deref()==plastic{return Ok(());}
        return Err("CHECKPOINT_FRAME_IMMUTABLE: existing State/history/plastic differ or are incomplete".into());
    }
    for (path,old,new) in [(&hp,old_history.as_deref(),Some(history)),(&pp,old_plastic.as_deref(),plastic)] {
        match(old,new){
            (Some(a),Some(b)) if a==b=>{},
            (None,Some(b))=>publish_new(path,b)?,
            (None,None)=>{},
            _=>return Err(error("incomplete orphan predecessors disagree; preserve for inspection")),
        }
    }
    publish_new(&sp,state)
}

/// Replace only a validated V1 history with its same-boundary V2 augmentation.
/// Caller must hold the output liveness lease and validate the migration semantics.
pub(crate) fn publish_migration_history(dir:&Path,frame:i32,state:&[u8],old_history:&[u8],plastic:Option<&[u8]>,new_history:&[u8])->Result<(),String>{
    let sp=dir.join(format!("state_{frame}.bin.gz"));
    let hp=dir.join(history_name(frame));
    let pp=dir.join(format!("plastic_{frame}.bin.gz"));
    if bytes(&sp)?.as_deref()!=Some(state)||bytes(&hp)?.as_deref()!=Some(old_history)||bytes(&pp)?.as_deref()!=plastic {
        return Err("CHECKPOINT_FRAME_IMMUTABLE: migration source changed".into());
    }
    let temp=hp.with_extension(format!("migration.{}",std::process::id()));
    let mut f=OpenOptions::new().write(true).create_new(true).open(&temp).map_err(error)?;
    let result=(|| {
        f.write_all(new_history).and_then(|_|f.sync_all()).map_err(error)?;drop(f);
        if bytes(&sp)?.as_deref()!=Some(state)||bytes(&hp)?.as_deref()!=Some(old_history)||bytes(&pp)?.as_deref()!=plastic {
            return Err("CHECKPOINT_FRAME_IMMUTABLE: migration source changed".into());
        }
        fs::rename(&temp,&hp).map_err(error)
    })();
    if result.is_err(){let _=fs::remove_file(&temp);}
    result
}
#[cfg(test)] mod tests{
    use super::*;
    fn dir(label:&str)->std::path::PathBuf{let p=std::env::temp_dir().join(format!("ppf-immutable-{}-{label}",std::process::id()));fs::create_dir(&p).unwrap();p}
    #[test]fn no_history_publication_preserves_existing_and_orphan(){
        let p=dir("no-history");publish_without_history(&p,1,b"state",Some(b"plastic")).unwrap();
        publish_without_history(&p,1,b"state",Some(b"plastic")).unwrap();
        assert!(publish_without_history(&p,1,b"changed",Some(b"other")).is_err());
        assert_eq!(fs::read(p.join("state_1.bin.gz")).unwrap(),b"state");
        assert_eq!(fs::read(p.join("plastic_1.bin.gz")).unwrap(),b"plastic");
        fs::write(p.join("plastic_2.bin.gz"),b"orphan").unwrap();
        assert!(publish_without_history(&p,2,b"state",Some(b"other")).is_err());
        assert!(!p.join("state_2.bin.gz").exists());
        assert_eq!(fs::read(p.join("plastic_2.bin.gz")).unwrap(),b"orphan");
        publish_without_history(&p,2,b"state",Some(b"orphan")).unwrap();
        fs::remove_dir_all(p).unwrap();
    }
    #[test]fn state_last_and_idempotent(){let p=dir("same");publish(&p,17,b"state",b"all runtime",Some(b"plastic")).unwrap();let before=fs::metadata(p.join("state_17.bin.gz")).unwrap().modified().unwrap();publish(&p,17,b"state",b"all runtime",Some(b"plastic")).unwrap();assert_eq!(before,fs::metadata(p.join("state_17.bin.gz")).unwrap().modified().unwrap());assert!(publish(&p,17,b"state",b"changed root",Some(b"plastic")).unwrap_err().starts_with("CHECKPOINT_FRAME_IMMUTABLE"));assert_eq!(fs::read(p.join(history_name(17))).unwrap(),b"all runtime");fs::remove_dir_all(p).unwrap();}
    #[test]fn orphan_and_missing_history_refused(){let p=dir("orphan");fs::write(p.join(history_name(2)),b"partial").unwrap();assert!(publish(&p,2,b"state",b"complete",None).is_err());assert!(!p.join("state_2.bin.gz").exists());fs::write(p.join("state_3.bin.gz"),b"legacy").unwrap();assert!(publish(&p,3,b"legacy",b"new history",None).is_err());assert!(!p.join(history_name(3)).exists());fs::remove_dir_all(p).unwrap();}
    #[test]fn corrupt_wrong_version_frame_truncation_cannot_touch_outputs(){
        let p=dir("corrupt");fs::write(p.join("vert_18.bin"),b"old output").unwrap();fs::write(p.join("state_17.bin.gz"),b"old State").unwrap();
        let data=encode_history(17,1.25,b"actual runtime payload").unwrap();assert_eq!(decode_history(17,&data).unwrap(),(1.25,b"actual runtime payload".as_slice()));
        assert!(decode_history(18,&data).is_err());let mut version=data.clone();version[8]=2;assert!(decode_history(17,&version).is_err());
        for n in [0,7,31,data.len()-1]{assert!(decode_history(17,&data[..n]).is_err());}
        let mut trailing=data.clone();trailing.push(0);assert!(decode_history(17,&trailing).is_err());
        let mut nan=data.clone();nan[16..24].copy_from_slice(&f64::NAN.to_bits().to_le_bytes());assert!(decode_history(17,&nan).is_err());
        assert_eq!(fs::read(p.join("vert_18.bin")).unwrap(),b"old output");assert_eq!(fs::read(p.join("state_17.bin.gz")).unwrap(),b"old State");fs::remove_dir_all(p).unwrap();
    }
    #[test] fn migration_replaces_only_history_and_rejects_wrong_source(){
        let p=dir("migration"); publish(&p,2,b"state",b"old",Some(b"plastic")).unwrap();
        assert!(publish_migration_history(&p,2,b"bad",b"old",Some(b"plastic"),b"new").is_err());
        assert_eq!(fs::read(p.join(history_name(2))).unwrap(),b"old");
        publish_migration_history(&p,2,b"state",b"old",Some(b"plastic"),b"new").unwrap();
        assert_eq!(fs::read(p.join(history_name(2))).unwrap(),b"new");
        assert_eq!(fs::read(p.join("state_2.bin.gz")).unwrap(),b"state");
        assert!(publish_migration_history(&p,2,b"state",b"old",Some(b"plastic"),b"other").is_err());
        publish(&p,2,b"state",b"new",Some(b"plastic")).unwrap(); fs::remove_dir_all(p).unwrap();
    }
}
