/// Native CLI tail, shared by Windows/Unix actual launcher. No disk effects.
pub(super) fn arguments(load: i32, target: Option<i32>) -> Result<Vec<String>, &'static str> {
    let mut out=vec![format!("--load={load}")];
    if let Some(target)=target {
        if load<=0 || target<=load {return Err("RESUME_TARGET_INVALID")}
        out.push(format!("--target-frame={target}"));
    }
    Ok(out)
}
