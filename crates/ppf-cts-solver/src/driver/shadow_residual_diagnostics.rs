//! Host-only, opt-in shadow residual evidence. No device ownership.
use std::path::Path;
use std::io::Write;

pub(super) fn modes(s:&str, available:&[usize])->Result<Vec<usize>,String>{
    let mut out=Vec::new();
    for name in s.split(',') {
        let mode=match name.trim(){"bj"=>0,"dense"=>2,"weighted_forest"=>4,_=>return Err("invalid shadow mode".into())};
        if !available.contains(&mode)||out.contains(&mode){return Err("unavailable or duplicate shadow mode".into())}
        out.push(mode);
    }
    Ok(out)
}
pub(super) fn cap_allowed(error:&str,enabled:bool)->bool{enabled && error=="shadow iteration cap"}
pub(super) fn finite(values:&[f32],max:usize)->Result<(),String>{
    if values.len()>max || values.iter().any(|v|!v.is_finite()){Err("nonfinite or excessive diagnostic extent".into())}else{Ok(())}
}
pub(super) fn fresh(path:&Path)->Result<(),String>{
    if !path.is_absolute(){return Err("diagnostic directory must be absolute".into())}
    std::fs::create_dir(path).map_err(|e|e.to_string())
}
pub(super) fn write(path:&Path,name:&str,value:&serde_json::Value)->Result<(),String>{
    let mut f=std::fs::OpenOptions::new().write(true).create_new(true).open(path.join(name)).map_err(|e|e.to_string())?;
    serde_json::to_writer(&mut f,value).map_err(|e|e.to_string())?;
    f.write_all(b"\n").map_err(|e|e.to_string())
}
#[cfg(test)] mod tests {
 use super::*;
 #[test] fn parse_and_cap(){assert_eq!(modes("bj,dense,weighted_forest",&[0,2,4]).unwrap(),vec![0,2,4]);assert!(modes("bj,bj",&[0]).is_err());assert!(modes("forest",&[0,1]).is_err());assert!(cap_allowed("shadow iteration cap",true));assert!(!cap_allowed("shadow iteration cap!",true));}
 #[test] fn actual_io_create_new(){let p=std::env::temp_dir().join(format!("ppf-shadow-{}-{:?}",std::process::id(),std::thread::current().id()));fresh(&p).unwrap();write(&p,"cold.json",&serde_json::json!({"x":[1,2]})).unwrap();assert!(write(&p,"cold.json",&serde_json::json!({})).is_err());assert!(fresh(&p).is_err());assert_eq!(std::fs::read_to_string(p.join("cold.json")).unwrap(),"{\"x\":[1,2]}\n");std::fs::remove_file(p.join("cold.json")).unwrap();std::fs::remove_dir(p).unwrap();}
 #[test] fn reject_bad_values(){assert!(finite(&[f32::NAN],4).is_err());assert!(finite(&[1.,2.],1).is_err());}
}
