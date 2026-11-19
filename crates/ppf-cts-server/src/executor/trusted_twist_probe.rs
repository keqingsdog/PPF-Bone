//! Probe only the server-selected trusted worker; never execute session scripts.
use std::{path::Path, process::Stdio, time::Duration};
use super::build::{locate_build_worker, python_executable};

pub(super) async fn probe(scene: &Path) -> Result<serde_json::Value, &'static str> {
    let worker = locate_build_worker().ok_or("TWIST_RESUME_TRUSTED_WORKER_UNAVAILABLE")?;
    let worker = worker.canonicalize().map_err(|_| "TWIST_RESUME_TRUSTED_WORKER_UNAVAILABLE")?;
    let helper = worker.parent().ok_or("TWIST_RESUME_TRUSTED_WORKER_UNAVAILABLE")?.join("resume_probe_worker.py");
    if !helper.is_file() { return Err("TWIST_RESUME_TRUSTED_WORKER_UNAVAILABLE") }
    let (python, _) = python_executable();
    let mut command = tokio::process::Command::new(python);
    command.arg("-I").arg(helper).arg(scene).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null()).kill_on_drop(true);
    let result = tokio::time::timeout(Duration::from_secs(15), command.output()).await
        .map_err(|_| "TWIST_RESUME_PROBE_TIMEOUT")?
        .map_err(|_| "TWIST_RESUME_PROBE_FAILED")?;
    if !result.status.success() || result.stdout.len()>65536 { return Err("TWIST_RESUME_CAPABILITY_UNAVAILABLE") }
    let payload: serde_json::Value = serde_json::from_slice(&result.stdout).map_err(|_| "TWIST_RESUME_PROBE_INVALID")?;
    if payload.get("supported_modes").and_then(|v|v.as_array()).map_or(true, |m|!m.iter().any(|v|v.as_str()==Some("continuous_from_current"))) {
        return Err("TWIST_RESUME_CAPABILITY_UNAVAILABLE")
    }
    Ok(payload)
}
