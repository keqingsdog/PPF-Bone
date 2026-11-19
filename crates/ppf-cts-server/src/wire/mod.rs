// Fork modification: changed from the upstream baseline.
// File: crates/ppf-cts-server/src/wire/mod.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Per-connection wire-protocol handler on top of `tokio::net::TcpStream`.
//
// Three paths after the 4-byte header:
//   * TCMD: text command line, dispatch to engine, send JSON status.
//   * JSON: JSON request line; sub-handler keyed by `request` field
//     (upload_atomic, upload_notify, data_send, data_receive,
//     notebook_send, notebook_delete).
//   * BDAT: drain bytes, ack with `BINARY_OK`.
//
// The TCMD response shape is the protocol contract: every status
// poll the addon makes goes through this code.

use std::collections::HashMap;

use ppf_cts_core::events::Event;
use ppf_cts_formats::files::{DATA_PICKLE, PARAM_PICKLE, SCENE_INFO_JSON};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::easy_parse::{self, get as get_arg};
use crate::engine::ServerEngine;
use crate::error::ServerError;
use crate::executor::{dispatch_with_executor, EffectExecutor};
use crate::protocol::{
    read_header, read_line, write_json_line, HEADER_BDAT, HEADER_JSON, HEADER_TCMD,
};
use crate::response::build_response;

mod data;
mod force_field;
mod notebook;
mod upload;

const MAX_TCMD_BYTES: usize = 64 * 1024;
const MAX_JSON_LINE_BYTES: usize = 64 * 1024;

// Unlike ordinary historical TCMD fields, an explicit migration must never be
// silently dropped by the permissive parser, including duplicate/bare flags.
fn twist_mode_well_formed(line: &str) -> bool {
    let tokens: Vec<_> = line.split_whitespace().collect();
    if !tokens.iter().any(|token| token.starts_with("--twist_resume") || token.starts_with("--twist-resume")) {
        return true;
    }
    if tokens.len() % 2 != 0 { return false; }
    let mut seen = std::collections::HashSet::new();
    let mut request = None;
    let mut mode = None;
    for pair in tokens.chunks_exact(2) {
        let (key, value) = (pair[0], pair[1]);
        if !key.starts_with("--") || value.starts_with("--") || !seen.insert(key) { return false; }
        if (key.starts_with("--twist_resume") || key.starts_with("--twist-resume")) && key != "--twist_resume_mode" { return false; }
        if key == "--request" { request = Some(value); }
        if key == "--twist_resume_mode" { mode = Some(value); }
    }
    request == Some("resume") && mode == Some("continuous_from_current")
}

/// Upper bound on a single JSON-path payload (`data_send`,
/// `upload_atomic`, `notebook_send`). The client declares the byte
/// count in the JSON header line; without a ceiling, a hostile or
/// buggy declaration of e.g. 0xFFFFFFFFFFFF would drive
/// `read_exact_n_chunked` into a multi-terabyte allocation before a
/// single payload byte is read. 4 GiB sits well above the largest
/// legitimate pickle (`data_receive` streams session artifacts of a
/// few hundred MB) while still rejecting absurd declarations fast.
/// Kept here next to MAX_TCMD_BYTES / MAX_JSON_LINE_BYTES so all wire
/// input bounds live in one place; the TCMD 64 KB bound is too small
/// to reuse for these payloads.
pub(super) const MAX_PAYLOAD_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Run the per-connection handler to completion. Errors during
/// reading/writing log and return; the caller drops the socket. Owns
/// the socket halves so we can split read/write across awaits.
pub async fn handle_connection(
    stream: tokio::net::TcpStream,
    peer: std::net::SocketAddr,
    engine: ServerEngine,
    executor: std::sync::Arc<dyn EffectExecutor>,
) {
    let (mut reader, mut writer) = stream.into_split();
    let header = match read_header(&mut reader).await {
        Ok(Some(h)) => h,
        Ok(None) => {
            log::debug!(target: "ppf::wire", "{peer}: clean disconnect before header");
            return;
        }
        Err(e) => {
            log::warn!(target: "ppf::wire", "{peer}: header read: {e}");
            return;
        }
    };

    let result = if &header == HEADER_TCMD {
        handle_tcmd(&mut reader, &mut writer, &engine, executor.as_ref(), peer).await
    } else if &header == HEADER_JSON {
        handle_json(&mut reader, &mut writer, &engine, executor.as_ref(), peer).await
    } else if &header == HEADER_BDAT {
        data::handle_bdat(&mut reader, &mut writer, peer).await
    } else {
        log::warn!(target: "ppf::wire", "{peer}: unknown header {:?}", String::from_utf8_lossy(&header));
        let payload = ServerError::UnknownHeader(hex_string(&header)).into_response();
        let _ = write_json_line(&mut writer, &payload).await;
        Ok(())
    };

    if let Err(e) = result {
        log::warn!(target: "ppf::wire", "{peer}: handler error: {e}");
    }
}

fn hex_string(buf: &[u8]) -> String {
    let mut s = String::with_capacity(buf.len() * 2);
    for b in buf {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

// ---------------------------------------------------------------------------
// TCMD path

async fn handle_tcmd<R, W>(
    reader: &mut R,
    writer: &mut W,
    engine: &ServerEngine,
    executor: &dyn EffectExecutor,
    peer: std::net::SocketAddr,
) -> std::io::Result<()>
where
    R: AsyncReadExt + Unpin,
    W: AsyncWriteExt + Unpin,
{
    // TCMD wire: the 4-byte header is followed by a big-endian u32
    // length prefix, then exactly that many payload bytes (the
    // `--key value` argument string). Earlier versions of this
    // handler read until EOF instead, relying on the client calling
    // shutdown(SHUT_WR) to signal end-of-input: that worked on POSIX
    // but on Windows tokio doesn't reliably surface the half-close
    // as Ok(0) to AsyncRead, so the read loop hung forever and
    // connections piled up in FIN_WAIT_2 state until the server
    // stopped accepting new requests entirely.
    let mut len_buf = [0u8; 4];
    if let Err(e) = reader.read_exact(&mut len_buf).await {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            log::debug!(target: "ppf::wire", "{peer}: empty TCMD (no length prefix)");
            return Ok(());
        }
        return Err(e);
    }
    let payload_len = u32::from_be_bytes(len_buf) as usize;
    if payload_len > MAX_TCMD_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "TCMD payload length {payload_len} exceeds max {MAX_TCMD_BYTES}"
            ),
        ));
    }
    let buf = if payload_len == 0 {
        Vec::new()
    } else {
        crate::protocol::read_exact_n_chunked(reader, payload_len).await?
    };

    if buf.is_empty() {
        log::debug!(target: "ppf::wire", "{peer}: empty TCMD");
        return Ok(());
    }

    let line = match std::str::from_utf8(&buf) {
        Ok(s) => s,
        Err(e) => {
            let resp = ServerError::TextDecode(e.to_string()).into_response();
            return write_response(writer, &resp).await;
        }
    };
    if !twist_mode_well_formed(line) {
        return write_response(writer, &serde_json::json!({"error":"TWIST_RESUME_MODE_INVALID"})).await;
    }
    let args = easy_parse::parse(line);

    let name = match get_arg(&args, "name") {
        Some(n) => n.to_string(),
        None => {
            let resp = ServerError::NoId.into_response();
            return write_response(writer, &resp).await;
        }
    };

    // The server (not the addon) decides where project data lives:
    // `make_root` pins every project under
    // `~/.local/share/ppf-cts/git-<branch>/<name>` (POSIX) or the
    // Windows / `PPF_CTS_DATA_ROOT` equivalent. The canonical root
    // flows back to the addon via this status response's `root`
    // field, which it stores in `state.remote_root` and uses for
    // subsequent uploads.
    let delete_request = get_arg(&args, "request") == Some("delete");
    let project_root = if delete_request {
        match crate::executor::checked_delete_root(&name, engine.config()) {
            Ok(root) => root,
            Err(error) => return write_response(writer, &serde_json::json!({
                "protocol_version": crate::PROTOCOL_VERSION,
                "error": format!("DELETE_PROJECT_INVALID: {error}"),
            })).await,
        }
    } else {
        crate::upload::make_root(&name, engine.config())
    };
    let root = project_root
        .to_string_lossy()
        .into_owned();
    // Hydrate engine state from disk before dispatching: read the
    // upload identity, hashes, and on-disk artifact presence so the
    // engine's `data` and `build` fields reflect what the previous
    // process (or an external builder like JupyterLab) left behind.
    // Without this, a server restart against an existing project
    // would report `data="NO_DATA"` even with `data.pickle` and
    // `app_state.pickle` on disk.
    let project = reconcile_project_from_disk(&name, &root, &engine.state().name);
    let delete_selection = if delete_request {
        let (selected, effects) = engine.dispatch_with_snapshot(project);
        for effect in effects {
            executor.execute(effect, engine).await;
        }
        Some(selected)
    } else {
        dispatch_with_executor(engine, executor, project).await;
        None
    };

    if get_arg(&args,"request") == Some("resume_generation_baseline") {
        let words: Vec<_> = line.split_whitespace().collect();
        let valid = words.len()==4 && words.chunks_exact(2).all(|p| (p[0]=="--name" && p[1]==name) || (p[0]=="--request" && p[1]=="resume_generation_baseline"))
            && words.iter().filter(|w| **w=="--name").count()==1 && words.iter().filter(|w| **w=="--request").count()==1;
        if !valid || engine.state().name!=name {return write_response(writer,&serde_json::json!({"protocol_version":crate::PROTOCOL_VERSION,"error":"BASELINE_QUERY_INVALID"})).await;}
        let result=crate::executor::resume_baseline::query(engine);
        let state=engine.state();
        if state.name!=name || matches!(&result,Ok(b) if !b.matches(&state)) || matches!(&result,Err("PROJECT_CONTEXT_CHANGED")) {return write_response(writer,&serde_json::json!({"protocol_version":crate::PROTOCOL_VERSION,"error":"PROJECT_CONTEXT_CHANGED"})).await;}
        let mut response=build_response(&state,engine.config());
        if let Value::Object(ref mut obj)=response {match result {Ok(b)=>{obj.insert("resume_generation_baseline".into(),serde_json::to_value(b).unwrap());},Err(e)=>{obj.insert("resume_generation_error".into(),Value::String(e.into()));}}}
        return write_response(writer,&response).await;
    }
    if get_arg(&args,"request") == Some("twist_resume_contract") {
        // A dedicated read-only request avoids executable probes on status polls.
        // Refuse extra controls rather than accepting ambiguous action flags.
        let words: Vec<_> = line.split_whitespace().collect();
        let valid = words.len() == 4 && words.chunks_exact(2).all(|pair|
            (pair[0] == "--name" && pair[1] == name) ||
            (pair[0] == "--request" && pair[1] == "twist_resume_contract"))
            && words.iter().filter(|w| **w == "--name").count() == 1
            && words.iter().filter(|w| **w == "--request").count() == 1;
        if !valid || engine.state().name != name {
            return write_response(writer, &serde_json::json!({"protocol_version":crate::PROTOCOL_VERSION,"error":"TWIST_RESUME_QUERY_INVALID"})).await;
        }
        let result = crate::executor::sealed_twist_contract::query(engine).await;
        let response_state = engine.state();
        let changed = match result.as_ref() {
            Ok(contract) => contract["name"].as_str()!=Some(response_state.name.as_str())
                || contract["root"].as_str()!=Some(response_state.root.as_str())
                || contract["upload_id"].as_str()!=Some(response_state.upload_id.as_str())
                || contract["param_hash"].as_str()!=Some(response_state.param_hash.as_str())
                || contract["data_hash"].as_str()!=Some(response_state.data_hash.as_str()),
            Err(error) => *error == "PROJECT_CONTEXT_CHANGED",
        };
        if changed || response_state.name != name {
            return write_response(writer, &serde_json::json!({"protocol_version":crate::PROTOCOL_VERSION,"error":"PROJECT_CONTEXT_CHANGED"})).await;
        }
        let mut response = build_response(&response_state, engine.config());
        if let Value::Object(ref mut obj) = response {
            match result {
                Ok(contract) => { obj.insert("twist_resume_contract".into(), contract); },
                Err(error) => { obj.insert("twist_resume_error".into(), Value::String(error.into())); },
            }
        }
        return write_response(writer, &response).await;
    }
    if delete_request {
        if get_arg(&args, "target_frame").is_some() {
            return write_response(writer, &serde_json::json!({
                "protocol_version": crate::PROTOCOL_VERSION,
                "error": "RESUME_TARGET_INVALID: explicit nonzero checkpoint and later i32 target required",
            })).await;
        }
        // Bind the command to the exact project selection observed here.
        // A concurrent client may have selected another project after the
        // reconciliation above, including the same name in a new generation.
        let selected = delete_selection.expect("delete selection captured");
        if selected.name != name || selected.root != root {
            return write_response(writer, &serde_json::json!({
                "protocol_version": crate::PROTOCOL_VERSION,
                "error": "PROJECT_CONTEXT_CHANGED",
            })).await;
        }
        let generation = selected.project_generation.checked_add(1)
            .expect("project generation exhausted");
        let effects = match engine.dispatch_matching_context(&selected, Event::DeleteRequested) {
            Some(effects) => effects,
            None => return write_response(writer, &serde_json::json!({
                "protocol_version": crate::PROTOCOL_VERSION,
                "error": "PROJECT_CONTEXT_CHANGED",
            })).await,
        };
        for effect in effects {
            executor.execute(effect, engine).await;
        }
        let deleted = engine.state();
        if deleted.name != name || deleted.project_generation != generation
            || !deleted.root.is_empty()
        {
            return write_response(writer, &serde_json::json!({
                "protocol_version": crate::PROTOCOL_VERSION,
                "error": "PROJECT_CONTEXT_CHANGED",
            })).await;
        }
        let mut response = build_response(&deleted, engine.config());
        if let Value::Object(ref mut fields) = response {
            fields.insert("delete_confirmed".into(), Value::Bool(deleted.error.is_empty()));
        }
        return write_response(writer, &response).await;
    }

    let mut request_event_dispatched = false;
    if let Some(req) = get_arg(&args, "request") {
        log::debug!(target: "ppf::wire", "{peer}: TCMD request name={name} request={req}");
        if req == "resume" && crate::executor::retained_checkpoints(std::path::Path::new(&root)) {
            let state = engine.state();
            if state.name != name || state.root != root {
                return write_response(writer, &serde_json::json!({
                    "protocol_version": crate::PROTOCOL_VERSION,
                    "error": "PROJECT_CONTEXT_CHANGED",
                })).await;
            }
            let mut response = build_response(&state, engine.config());
            response["error"] = Value::String("CHECKPOINTS_FROM_PREVIOUS_BUILD: run a fresh simulation for updated parameters".into());
            return write_response(writer, &response).await;
        }
        let resume_from = get_arg(&args, "resume_from").and_then(|s| s.parse::<i32>().ok());
        let preserve_output = get_arg(&args, "preserve_output") == Some("1");
        let event = if let Some(mode)=get_arg(&args,"twist_resume_mode") {
            let target = get_arg(&args,"target_frame");
            match (req, mode, resume_from, target.map(|v| v.parse::<i32>())) {
                ("resume", "continuous_from_current", Some(from_frame), None) if from_frame > 0 =>
                    Some(Event::ResumeTwistUpgradeRequested { from_frame, target_frame:None }),
                ("resume", "continuous_from_current", Some(from_frame), Some(Ok(target_frame))) if from_frame > 0 && target_frame > from_frame =>
                    Some(Event::ResumeTwistUpgradeRequested { from_frame, target_frame:Some(target_frame) }),
                _ => Some(Event::ErrorOccurred { error:"TWIST_RESUME_MODE_INVALID: explicit positive checkpoint and supported mode required".into() }),
            }
        } else if let Some(raw_target)=get_arg(&args,"target_frame") {
            match (req,resume_from,raw_target.parse::<i32>()) {
                ("resume",Some(from_frame),Ok(target_frame)) if from_frame>0 && target_frame>from_frame =>
                    Some(Event::ResumeHorizonRequested { from_frame,target_frame }),
                _ => Some(Event::ErrorOccurred { error:"RESUME_TARGET_INVALID: explicit nonzero checkpoint and later i32 target required".into() }),
            }
        } else { tcmd_request_to_event(req,resume_from,preserve_output) };
        if let Some(event) = event {
            request_event_dispatched = true;
            dispatch_with_executor(engine, executor, event).await;
        } else {
            log::warn!(target: "ppf::wire", "{peer}: unknown TCMD request {req}");
        }
    } else {
        log::debug!(target: "ppf::wire", "{peer}: TCMD ping name={name}");
    }

    let response_state = engine.state();
    // A probe that dispatched no request event must not expose another project,
    // including an unrecognized request that only produced a warning above.
    // Dispatched requests retain their existing response semantics: an action may
    // already have run, so a context error must not imply that retrying is safe.
    // This is a fail-closed response check, not multi-project transaction isolation.
    if !request_event_dispatched && response_state.name != name {
        return write_response(writer, &serde_json::json!({
            "protocol_version": crate::PROTOCOL_VERSION,
            "error": "PROJECT_CONTEXT_CHANGED",
        })).await;
    }
    let mut response = build_response(&response_state, engine.config());
    if let Value::Object(ref mut obj) = response {
        obj.entry("stdout".to_string()).or_insert_with(|| Value::String(String::new()));
        obj.entry("stderr".to_string()).or_insert_with(|| Value::String(String::new()));
    }
    write_response(writer, &response).await
}

/// Build a `ProjectSelected` event by reading on-disk artifacts:
/// `data.pickle` and `param.pickle` presence, the `upload_id.txt`
/// identity stamp, the data and param hash files, and
/// `app_state.pickle` (the build_worker's "make() succeeded"
/// marker). `is_resumable` falls out of `project_resumable`, which
/// scans for the solver's `state_<N>.bin.gz` checkpoints, and
/// `crash` from the solver-authored `status.cbor` so a reconnect after a
/// crash reconstructs `Solver::Failed` AND its reason (the in-memory state is
/// gone). The transition layer then advances `state.data` / `state.build` so
/// the addon's status string reflects what survived the restart.
///
/// `selected_name` is the project the engine currently holds, and it decides
/// whether the crash is reconstructed at all: see the `crash` binding.
fn reconcile_project_from_disk(name: &str, root: &str, selected_name: &str) -> Event {
    use std::path::PathBuf;

    let root_path = PathBuf::from(root);
    let has_data = root_path.join(DATA_PICKLE).exists();
    let has_param = root_path.join(PARAM_PICKLE).exists();
    // build_worker writes app_state.pickle at the tail end of make();
    // its presence is the "build is fresh" signal.
    let has_app = root_path.join("app_state.pickle").exists();
    let is_resumable = !crate::executor::retained_checkpoints(&root_path)
        && ppf_cts_core::datamodel::project_resumable(&root_path);
    // Reconstruct a crash from the on-disk status.cbor (the in-memory
    // Solver::Failed is gone after a restart): a terminal Crashed outcome, or
    // a non-terminal record whose owning process is confirmed dead (liveness
    // lock free AND pid gone) and was not an intentional terminate. A
    // still-live solver (lock held) reports None here and is re-adopted as
    // Running by the monitor's live-adoption path.
    //
    // Only when the SELECTION CHANGES. The same-project branch of
    // `Event::ProjectSelected` preserves the in-memory solver state and
    // discards this field, while a connected addon stamps the project name on
    // a status poll several times a second. Reconstructing there would read
    // all of `stdout.log` and all of `error.log` on every poll, for a value
    // nothing reads; a crashed multi-frame run's stdout is routinely tens of
    // megabytes.
    //
    // The report is rendered by the SAME `render_crash` the live path uses. A
    // crash the server classified itself is sealed into `status.cbor` when it
    // is classified, so the usual reconnect reads back a terminal Crashed and
    // reproduces the live report. The non-terminal branch below covers a
    // record nothing sealed (the server was not running when the solver
    // died), and what it can name is limited to what is on disk: the
    // launcher's exit status is an in-memory witness, so a kill that left no
    // signal record comes back as "no cause recorded".
    let crash = if name == selected_name {
        None
    } else {
        use ppf_cts_formats::status::{self, lock, signal_sidecar, Outcome};
        let out = ppf_cts_formats::files::session_output_dir(&root_path);
        match status::read(&out) {
            Ok(Some(rec)) => match rec.outcome {
                Some(Outcome::Crashed { sub_kind, detail }) => {
                    Some(ppf_cts_core::events::ReconciledCrash {
                        kind_tag: sub_kind.tag().to_string(),
                        rendered: crate::monitor::render_crash(sub_kind, &detail, &root_path),
                    })
                }
                Some(_) => None,
                None => {
                    let dead = !lock::is_held_by_other(&out)
                        && !lock::pid_alive(rec.pid)
                        && !out.join(ppf_cts_formats::files::TERMINATE_REQUEST).exists();
                    if !dead {
                        None
                    } else {
                        // The launcher's exit status belongs to a process this
                        // server may never have spawned, so the only evidence a
                        // reconnect can read is what the solver left on disk.
                        let evidence = crate::monitor::AbruptEvidence {
                            signal: signal_sidecar::read(&out, &rec.launch_id),
                            launcher_code: None,
                            launcher_signal: None,
                            pid: Some(rec.pid),
                            frame: Some(rec.frame),
                        };
                        let (kind, detail) = crate::monitor::classify_abrupt(&evidence);
                        Some(ppf_cts_core::events::ReconciledCrash {
                            kind_tag: kind.tag().to_string(),
                            rendered: crate::monitor::render_crash(kind, &detail, &root_path),
                        })
                    }
                }
            },
            _ => None,
        }
    };

    let upload_id = if has_data && has_param {
        crate::upload::read_upload_id(&root_path).unwrap_or_default()
    } else {
        String::new()
    };
    let data_hash = crate::upload::read_data_hash(&root_path);
    let param_hash = crate::upload::read_param_hash(&root_path);
    // BuildMetadata is the only event that normally sets total_frames,
    // and it only fires during a fresh build. So a re-select of a
    // previously-built project (typically the connect probe routing
    // through __probe__ and back to the real project) leaves
    // total_frames at 0 -- which makes the response shape suppress
    // the progress field entirely (shape::insert_progress requires
    // total_frames > 0), and the addon's progress bar sits at 0.
    // Read it back from scene_info.json's "Total Frames" so the
    // transition layer can rehydrate the field without forcing a
    // rebuild.
    let total_frames = read_total_frames_from_scene_info(&root_path);
    // What Allow Existing Intersections exempted in the project's last
    // successful build, so a reconnect keeps drawing it. Only when the
    // selection changes and the project is built, for the same reason the
    // crash is read only then: the same-project branch discards it, and a
    // status poll stamps the project name several times a second.
    let exemptions = if name == selected_name || !has_app {
        vec![]
    } else {
        crate::executor::read_build_exemptions(root)
    };

    Event::ProjectSelected {
        name: name.to_string(),
        root: root.to_string(),
        has_data,
        has_param,
        has_app,
        is_resumable,
        crash,
        upload_id,
        data_hash,
        param_hash,
        total_frames,
        exemptions,
    }
}

/// Parse `<root>/scene_info.json`'s `"Total Frames"` string into an
/// `i32`. Returns 0 when the file is missing, malformed, or the field
/// is absent / unparseable -- same fallback the transition uses for a
/// just-uploaded project that has never been built.
fn read_total_frames_from_scene_info(root: &std::path::Path) -> i32 {
    let path = root.join(SCENE_INFO_JSON);
    let text = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(_) => return 0,
    };
    let v: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => return 0,
    };
    v.get("Total Frames")
        .and_then(|x| x.as_str())
        .and_then(|s| s.replace(',', "").parse::<i32>().ok())
        .unwrap_or(0)
}

/// Map a TCMD `request` string to an engine `Event`. `resume_from`
/// carries the optional `--resume_from` frame for the resume request
/// (`None` resumes from the latest checkpoint). `preserve_output`
/// carries the `--preserve_output 1` flag for the build request, which
/// keeps the `session/output/` checkpoints in place across a rebuild.
fn tcmd_request_to_event(
    req: &str,
    resume_from: Option<i32>,
    preserve_output: bool,
) -> Option<Event> {
    Some(match req {
        "build" => Event::BuildRequested { preserve_output },
        "cancel_build" => Event::CancelBuildRequested,
        "start" => Event::StartRequested,
        "resume" => Event::ResumeRequested { from_frame: resume_from },
        "terminate" => Event::TerminateRequested,
        "save_and_quit" => Event::SaveAndQuitRequested,
        "delete" => Event::DeleteRequested,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// JSON path

async fn handle_json<R, W>(
    reader: &mut R,
    writer: &mut W,
    engine: &ServerEngine,
    executor: &dyn EffectExecutor,
    peer: std::net::SocketAddr,
) -> std::io::Result<()>
where
    R: AsyncReadExt + Unpin,
    W: AsyncWriteExt + Unpin,
{
    let line_bytes = match read_line(reader, MAX_JSON_LINE_BYTES).await {
        Ok(b) => b,
        Err(e) => {
            let resp = ServerError::JsonHeaderRead(e.to_string()).into_response();
            return write_response(writer, &resp).await;
        }
    };
    let trimmed: &[u8] = line_bytes
        .strip_suffix(b"\n")
        .unwrap_or(&line_bytes);
    let req: HashMap<String, Value> = match serde_json::from_slice(trimmed) {
        Ok(v) => v,
        Err(e) => {
            let resp = ServerError::JsonParse(e.to_string()).into_response();
            return write_response(writer, &resp).await;
        }
    };

    let req_type = req.get("request").and_then(Value::as_str).unwrap_or("");
    log::debug!(target: "ppf::wire", "{peer}: JSON request type={req_type}");

    match req_type {
        "upload_atomic" => upload::handle_upload_atomic(reader, writer, engine, executor, &req).await,
        "upload_notify" => upload::handle_upload_notify(writer, engine, executor, &req).await,
        "data_send" => data::handle_data_send(reader, writer, engine, &req).await,
        "data_receive" => data::handle_data_receive(writer, engine, &req).await,
        "notebook_send" => notebook::handle_notebook_send(reader, writer, &req).await,
        "notebook_delete" => notebook::handle_notebook_delete(writer, &req).await,
        "force_field_check" => force_field::handle_force_field_check(writer, &req).await,
        other => {
            let resp = ServerError::UnknownRequest(other.to_string()).into_response();
            write_response(writer, &resp).await
        }
    }
}

// ---------------------------------------------------------------------------
// shared helpers used across submodules

pub(super) fn json_u64(v: Option<&Value>) -> u64 {
    match v {
        Some(Value::Number(n)) => n.as_u64().unwrap_or(0),
        Some(Value::String(s)) => s.parse().unwrap_or(0),
        _ => 0,
    }
}

pub(super) async fn write_response<W>(writer: &mut W, response: &Value) -> std::io::Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    write_json_line(writer, response).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn twist_flag_must_have_exactly_one_value() {
        assert!(twist_mode_well_formed("--request resume --twist_resume_mode continuous_from_current"));
        assert!(twist_mode_well_formed("--request resume"));
        assert!(!twist_mode_well_formed("--request resume --twist_resume_mode"));
        assert!(!twist_mode_well_formed("--twist_resume_mode --request resume"));
        assert!(!twist_mode_well_formed("--twist_resume_mode x --twist_resume_mode y"));
        assert!(!twist_mode_well_formed("--request resume --twist_resume_mode=bad"));
        assert!(!twist_mode_well_formed("--request resume --twist_resume_mode continuous_from_current stray"));
        assert!(!twist_mode_well_formed("--name x --twist_resume_mode continuous_from_current"));
        assert!(!twist_mode_well_formed("--request resume --twist_resume_mode_typo x"));
        assert!(!twist_mode_well_formed("--request resume --request resume --twist_resume_mode continuous_from_current"));
    }

    #[test]
    fn tcmd_event_map_covers_all_known_requests() {
        for r in [
            "build", "cancel_build", "start", "resume",
            "terminate", "save_and_quit", "delete",
        ] {
            assert!(tcmd_request_to_event(r, None, false).is_some(), "missing {r}");
        }
        assert!(tcmd_request_to_event("not_a_real_request", None, false).is_none());
    }

    #[test]
    fn resume_request_carries_from_frame() {
        assert_eq!(
            tcmd_request_to_event("resume", Some(80), false),
            Some(Event::ResumeRequested { from_frame: Some(80) }),
        );
        assert_eq!(
            tcmd_request_to_event("resume", None, false),
            Some(Event::ResumeRequested { from_frame: None }),
        );
    }

    #[test]
    fn build_request_carries_preserve_output() {
        assert_eq!(
            tcmd_request_to_event("build", None, true),
            Some(Event::BuildRequested { preserve_output: true }),
        );
        assert_eq!(
            tcmd_request_to_event("build", None, false),
            Some(Event::BuildRequested { preserve_output: false }),
        );
    }
}

#[cfg(test)]
mod isolated_context_schedule_tests;
