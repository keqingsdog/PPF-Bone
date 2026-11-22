mod twist_overlay;
// Fork modification: changed from the upstream baseline.
// File: main.rs
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

mod args;
mod restoring_mass_authority;
mod restoring_curve_gpu;
mod restoring_curve;
mod joint;
mod continuous_twist;
mod material_spin_history;
mod soft_joint;
mod material_endpoint;
mod cross_links;
mod pmx_joints;
mod fem_control_events;
mod soft_joint_gpu;
mod collision_filter;
mod checkpoint_layout;
mod checkpoint_commit;
mod checkpoint_history;
mod checkpoint_state_marker;
mod checkpoint_sand;
mod checkpoint_values;
mod resume_request;
mod joint_root_motion_wire;
mod joint_root_motion_binding;
mod joint_root_motion_state;
mod joint_root_motion_gpu;
mod joint_root_motion_curve;
mod native_joint;
mod connected_collision;
mod connected_collision_native;
mod connected_collision_checkpoint;
mod connected_collision_intake;
mod pose_output;
mod soft_pose_output;
mod joint_gpu;
mod joint_angular_gpu;
mod joint_angular_control_gpu;
mod joint_angular_extension_gpu;
mod joint_angular_hard_wire;
mod joint_d6_hard_wire;
mod joint_hard_gpu;
mod joint_hard_schur;
mod joint_hard_bridge;
mod joint_hard_coefficients;
mod joint_hard_contract;
mod backend;
mod frame_output;
#[cfg(cuda_backend)]
mod gpu_clock;
mod builder;
// The neutral driver: the Newton loop, the linear solve, the assembly order and
// every dispatch decision, written once and driven through the `Device` seam.
// It defines the `extern "C"` surface the unconditional declarations in
// `backend.rs`, `main.rs` and `status_writer.rs` bind to, so not one call site
// moves when it is the thing running.
//
// The gate names no backend, which is what rule (1d) asks of every name in this
// crate. What it names is the fact that decides the question: whether this
// build links an implementation of the seam for the driver to drive. `build.rs`
// answers that in `Backend::links_neutral_driver`, in the one file that is
// allowed to know a target's name at all, and the reason a C++ backend answers
// no is written there rather than here.
#[cfg(neutral_driver)]
mod driver;
mod cvec;
mod cvecvec;
mod data;
mod force_field;
mod mesh;
mod plastic_state;
mod raw_vec;
mod scene;
#[cfg(unix)]
mod signal_sidecar;
mod statistics;
mod status_writer;
mod triutils;

use args::{ProgramArgs, SimArgs};
use backend::MeshSet;
use clap::Parser;
use data::{DataSet, EdgeParam, EdgeProp, ParamSet};
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use log::*;
use log4rs::append::console::ConsoleAppender;
use log4rs::config::{Appender, Config, Root};
use log4rs::encode::pattern::PatternEncoder;
use mesh::Mesh as SimMesh;
use rayon::prelude::*;
use scene::Scene;
use std::collections::HashMap;
use std::ffi::CString;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::os::raw::c_char;
use {builder::Props, cvec::CVec};
mod restoring_identity { include!("../../ppf-cts-compute/src/restoring_curve_identity.rs"); }
mod restoring_native;

extern crate nalgebra as na;

extern "C" {
    fn set_log_path(data_dir: *const c_char);
}

#[no_mangle]
extern "C" fn print_rust(message: *const libc::c_char) {
    let message = unsafe { std::ffi::CStr::from_ptr(message) }
        .to_str()
        .unwrap();
    info!("{message}");
}

/// The exit status of `--probe` when this build has no usable device here. Not
/// 1, which a panic or a refused argument also produces, so a crash is never
/// read as the answer "unusable".
const PROBE_UNUSABLE: i32 = 3;

/// `--probe`: whether this binary can run on this machine, answered by the
/// backend itself.
///
/// A distribution carries one directory per backend, and choosing among them
/// needs "this backend has a usable device here", which only the backend can
/// say. The answer is the preflight a session runs (`check_gpu`), which carries
/// the precise refusals, followed by opening the device through the constructor
/// a run uses, with its failure reported instead of panicked on.
///
/// The output is `key: value` lines on stdout, which the frontend's backend
/// resolution reads: `backend`, `linked` where a library is linked, `stamp`
/// (`ppf_cts_formats::SOURCE_STAMP`), then `device` with exit 0 or `unusable`
/// with [`PROBE_UNUSABLE`].
fn probe() -> i32 {
    println!("backend: {}", env!("PPF_BACKEND"));
    #[cfg(abi_backend_linked)]
    {
        println!("linked: {}", ppf_cts_compute::abi::linked_backend_name());
    }
    println!("stamp: {}", ppf_cts_formats::SOURCE_STAMP);
    if let Err(error) = ppf_cts_core::utils::check_gpu() {
        println!("unusable: {error}");
        return PROBE_UNUSABLE;
    }
    #[cfg(not(neutral_driver))]
    compile_error!("--probe opens the device through the neutral driver, which this build does not link");
    match driver::probe_device() {
        Ok(name) => {
            println!("device: {name}");
            0
        }
        Err(fault) => {
            println!("unusable: {fault}");
            PROBE_UNUSABLE
        }
    }
}

fn main() {
    // Dedicated read-only intake gate must never fall through to the normal solve.
    let host_args:Vec<String>=std::env::args().skip(1).collect();
    // Read-only build capability: never opens a device or reads a scene.
    // Reject mixed flags before other pre-parser probes can consume them.
    if host_args.iter().any(|arg|arg=="--probe-fem-retry") {
        if host_args.len()!=1 {
            eprintln!("--probe-fem-retry must be used alone");
            std::process::exit(2);
        }
        println!("{}",driver::adaptive_retry::FEM_RETRY_HOST_CAPABILITY);
        return;
    }
    for (flag, precision) in [
        ("--pcg-precision-capability-json", args::PcgPrecision::MixedFp16Nodal),
        ("--pcg-fp16-arithmetic-capability-json", args::PcgPrecision::MixedFp16Arithmetic),
    ] {
        if !host_args.iter().any(|arg| arg == flag) {
            continue;
        }
        if host_args.len() != 1 {
            eprintln!("{flag} must be used alone");
            std::process::exit(2);
        }
        let capability = mixed_pcg_device(precision);
        println!("{}", serde_json::json!({
            "version": 1,
            "mode": precision,
            "supported": capability.is_ok(),
            "backend": env!("PPF_BACKEND"),
            "reason": capability.err(),
        }));
        return;
    }
    if let Some(code)=connected_collision_intake::dispatch(&host_args) {std::process::exit(code);}

    if std::env::args().any(|arg|arg=="--fem-body-controls-capability-json") {
        println!("{}",serde_json::json!({"kind":"ppf.fem_body_controls.capability","version":1,
            "required_feature":soft_joint::CONTROL_FEATURE,"supported":cfg!(cuda_backend),"semantic_version":1,"supported_backends":["cuda"],"backend":env!("PPF_BACKEND"),
            "modes":["STATIC","KINEMATIC"],"scope":"initial_only","midrun_switching":false,
            "requires_matching_native_kernel_build":true}));
        return;
    }
    if std::env::args().any(|arg|arg=="--fem-body-control-events-capability-json") {
        println!("{}",serde_json::json!({"kind":"ppf.fem_body_control_events.capability","version":1,"semantic_version":2,
            "required_feature":fem_control_events::FEATURE,"supported":cfg!(cuda_backend),"supported_backends":["cuda"],"backend":env!("PPF_BACKEND"),"scope":"fem_roots_only",
            "modes":["FREE","STATIC","KINEMATIC"],"midrun_switching":true,"supported_event_pose_bindings":["current_hold","current_follow","absolute"],
            "resume_supported":false,"event_clock":"accepted_physical_boundary_v1","requires_matching_native_kernel_build":true}));
        return;
    }
    if std::env::args().any(|arg|arg=="--bone-connected-collision-capability-json") {
        println!("{}",serde_json::json!({"kind":"ppf.bone_connected_collision.capability","version":1,
            "required_feature":connected_collision::FEATURE,"supported":true,"semantic_version":connected_collision::SEMANTIC_VERSION}));
        return;
    }
    if std::env::args().any(|arg|arg=="--angular-swing-cone-capability-json") {
        // Opening the actual selected device validates the generated POD and
        // kernel identities against the linked backend before claiming support.
        match ensure_restoring_backend() {
            Ok(_)=>println!("{}",serde_json::json!({
                "kind":"ppf.angular_swing_cone.capability","version":1,
                "required_feature":joint_angular_extension_gpu::FEATURE_SWING_CONE,
                "supported":true,"semantic_version":1,"backend":env!("PPF_BACKEND"),
                "coordinate_model":joint_angular_extension_gpu::COORDINATE_MODEL,
                "limit_stride":joint_angular_extension_gpu::SOFT_RECORD_BYTES,
                "requires_matching_native_kernel_build":true,"solver_identity":compiled_restoring_identity()
            })),
            Err(e)=>{eprintln!("{e}");std::process::exit(2);}
        }
        return;
    }
    if std::env::args().any(|arg|arg=="--pmx-joints-capability-json") {
        let probe = probe_restoring_backend(args::PcgPrecision::Fp32);
        println!("{}",serde_json::json!({"kind":"ppf.pmx_joints.capability","version":1,"required_feature":pmx_joints::FEATURE,
            "supported":probe.is_ok(),"backend":env!("PPF_BACKEND"),"semantic_version":1,
            "model":"COMPLIANT_EULER_XYZ_BOX","detail":probe.err()}));
        return;
    }
    if std::env::args().any(|arg|arg=="--cross-links-capability-json") {
        match ensure_restoring_backend() {
            Ok(_)=>println!("{}",serde_json::json!({"kind":"ppf.cross_links.capability","version":1,"required_feature":cross_links::FEATURE,"supported":true,"backend":env!("PPF_BACKEND"),"model":"AXIAL_DISTANCE","semantic_version":1,"directional_controls_version":1,"full_force_direction_switches":true,"cross_link_collision_pairs_version":1,"binding_kind":5,"requires_matching_native_kernel_build":true,"solver_identity":compiled_restoring_identity()})),
            Err(e)=>{eprintln!("{e}");std::process::exit(2);}
        }
        return;
    }
    if std::env::args().any(|arg|arg=="--restoring-curves-capability-json") {
        match ensure_restoring_backend() {
            Ok(_)=>{let id=compiled_restoring_identity();println!("{}",serde_json::json!({
                "kind":"ppf.restoring_curves.capability","version":1,"capability":"restoring_curves_v1",
                "contract_version":restoring_identity::RESTORING_CURVE_CONTRACT_VERSION,
                "channel_mask":restoring_identity::RESTORING_CURVE_CHANNEL_MASK,
                "backend":id.selected_backend,"source_stamp":id.source_stamp,"abi_version":id.abi_version,
                "layout_sha256":id.layout_sha256,"kernel_table_sha256":id.kernel_table_sha256
            }));return;},Err(e)=>{eprintln!("{e}");std::process::exit(2);}
        }
    }
    // WHETHER THIS BINARY CAN RUN HERE; see `probe`. Handled before the
    // argument parser for the same reason `--backend` below is.
    if std::env::args().any(|argument| argument == "--probe") {
        std::process::exit(probe());
    }
    // WHICH BACKEND THIS BINARY IS, asked of the artifact itself.
    //
    // Every backend links this same executable name, because launch scripts,
    // the Windows batch files and the CI runners invoke it by that name, so
    // neither the path nor the file name says what is inside it. `build.rs`
    // compiles the answer in and this prints it. Handled before the argument
    // parser, which knows nothing of this flag and would reject it.
    if std::env::args().any(|argument| argument == "--backend") {
        // TWO ANSWERS, AND THE SECOND IS THE ONE THAT CAN DISAGREE.
        //
        // `PPF_BACKEND` is what `build.rs` SELECTED. It is not evidence about
        // what the binary can CALL: a build whose `abi_backend_linked` cfg is
        // missed on one platform still reports `cuda` from this string while
        // running the whole solve on the host renderings. A build-time label
        // cannot catch that, since it is produced by the very step that went
        // wrong.
        //
        // `be_backend_name` is answered by the LOADED library, so it reports
        // what is actually linked. When a library is linked, both are printed
        // and a disagreement is named rather than hidden. One source can build
        // for two platforms (a HIP backend compiles for AMD and, through nvcc,
        // for NVIDIA), so "selected rocm" and "linked hip-nvidia" are both true
        // and mean different things.
        println!("{}", env!("PPF_BACKEND"));
        #[cfg(abi_backend_linked)]
        {
            let linked = ppf_cts_compute::abi::linked_backend_name();
            if linked != env!("PPF_BACKEND") {
                println!("linked: {linked}");
            }
        }
        return;
    }
    let mut program_args = ProgramArgs::parse();
    // Pure host capability query: no scene reads, checkpoint scans or output.
    if program_args.resume_capability_json {
        let mut capability=serde_json::json!({"kind":"ppf.bone_resume_capability","version":1,"feature":"bone_checkpoint_resume_v1","supported":true,"target_frame_override":true,"backend":env!("PPF_BACKEND"),"scope":"same_sealed_scene_complete_runtime_checkpoint_v1","legacy_resume_fields":"superseded_only_under_complete_v1_checkpoint_contract"});
        #[cfg(neutral_driver)]
        {
            capability["supported_modes"] = serde_json::json!(["continuous_from_current"]);
            capability["accepted_receipt_source_stamps"] = serde_json::json!(accepted_receipt_source_stamps());
        }
        println!("{capability}");return;
    }
    // Retired mixed-mode runs and checkpoints must be refused before any
    // preflight can write into their existing session or output directories.
    if program_args.require_pcg_precision_v1 {
        eprintln!("PCG_PRECISION_UNSUPPORTED: FP16 PCG has been retired; rebuild and run the scene from the beginning with FP32");
        std::process::exit(2);
    }
    let param_path = std::path::Path::new(&program_args.path).join("param.toml");
    if let Ok(contents) = std::fs::read_to_string(param_path) {
        if let Ok(value) = contents.parse::<toml::Value>() {
            if let Some(precision) = value.get("param").and_then(|v| v.get("pcg_precision")).and_then(|v| v.as_str()) {
                if precision != "fp32" {
                    eprintln!("PCG_PRECISION_UNSUPPORTED: FP16 PCG has been retired; rebuild and run the scene from the beginning with FP32");
                    std::process::exit(2);
                }
            }
        }
    }
    // Resolve only the explicit migration request before any scene preflight or
    // output setup. Ordinary legacy latest-resume retains its existing path.
    if program_args.twist_resume_mode.is_some() {
        let latest=if program_args.load == -1 {
            ppf_cts_core::datamodel::list_saved_states(std::path::Path::new(&program_args.output)).into_iter().max()
        } else { None };
        program_args.load=args::resolve_twist_resume_load(program_args.load,latest)
            .unwrap_or_else(|e|{eprintln!("{e}");std::process::exit(2)});
    }
    // Build's authority command is host-only and precedes every output/setup,
    // status writer, checkpoint/resume scan and backend constructor.
    if let Some(request_path)=program_args.restoring_curves_mass_query.as_deref() {
        let result=(|| -> Result<String,String> {
            if program_args.load!=0 {return Err("RESTORING_CONFLICT: mass query requires fresh sealed input".into());}
            let base=restoring_curve::load_base_context(std::path::Path::new(&program_args.path))?;
            let request=std::fs::read(request_path).map_err(|e|format!("RESTORING_MASS_INVALID: request read: {e}"))?;
            let mut scene=Scene::new(&program_args);
            let dataset=fresh_authority_dataset(&mut scene);
            let response=restoring_mass_authority::query(&scene,&dataset,&base,&request,compiled_restoring_identity())?;
            serde_json::to_string(&response).map_err(|e|format!("RESTORING_MASS_INVALID: response serialization: {e}"))
        })();
        match result {Ok(json)=>{println!("{json}");return;},Err(e)=>{eprintln!("{e}");std::process::exit(2);}}
    }
    let mut restoring_backend_verified = program_args.require_restoring_curves_v1
        || std::path::Path::new(&program_args.path).join("bin/restoring_curves_v1.json").exists()
        || std::path::Path::new(&program_args.path).join(cross_links::FILE).exists()
        || std::path::Path::new(&program_args.path).join(pmx_joints::FILE).exists();
    if restoring_backend_verified {
        if let Err(e)=ensure_restoring_backend(){eprintln!("{e}");std::process::exit(2);}
    }
    let prepared_curves=restoring_curve::prepare_fresh(std::path::Path::new(&program_args.path),program_args.require_restoring_curves_v1,0)
        .unwrap_or_else(|e|{eprintln!("{e}");std::process::exit(2)});
    if let Err(e)=collision_filter::prepare_fresh(std::path::Path::new(&program_args.path),program_args.require_collision_filter_v1,0) {
        eprintln!("COLLISION_FILTER_PRE_SIMULATION_REFUSAL {e}");std::process::exit(2);
    }
    if std::path::Path::new(&program_args.path).join(joint::JOINT_FILE).exists() || std::path::Path::new(&program_args.path).join(joint_root_motion_binding::ROOT_MOTION_FILE).exists() || std::path::Path::new(&program_args.path).join(soft_joint::FILE).exists() || std::path::Path::new(&program_args.path).join(cross_links::FILE).exists() || std::path::Path::new(&program_args.path).join(pmx_joints::FILE).exists() || std::path::Path::new(&program_args.path).join(connected_collision_native::CROSS_FILE).exists() {
        let gate=joint::preflight_selected(std::path::Path::new(&program_args.path), joint::ExperimentalBallAnchorSelection::Requested);
        if let Err(e)=gate { joint::write_preflight_refusal(std::path::Path::new(&program_args.path),&e); eprintln!("JOINT_NATIVE_PRE_SIMULATION_REFUSAL {e}"); std::process::exit(2); }
        #[cfg(not(neutral_driver))]
        { eprintln!("JOINT_NATIVE_UNSUPPORTED_BACKEND: joints require the authoritative neutral driver"); std::process::exit(2); }
    }
    // `--load -1` is the "resume from latest checkpoint" sentinel the
    // server emits when the user clicks Resume (see
    // ppf-cts-server::executor::solver::launch_solver). The rest of
    // this binary keys off `program_args.load > 0` for the resume
    // path, so resolve the sentinel to a real frame index before
    // anything reads it. Without this, `--load -1` falls into the
    // fresh-start branch *and* trips `remove_files("vert"|"state",
    // ...)` in `setup()` starting at index 0, which destroys the
    // very checkpoint and vert frames the user wanted to resume from.
    if program_args.load < 0 {
        let output_dir = std::path::Path::new(&program_args.output);
        let saved = ppf_cts_core::datamodel::list_saved_states(output_dir);
        match saved.into_iter().max() {
            Some(n) => {
                if n == 0 {
                    eprintln!("CHECKPOINT_FRAME_ZERO_UNSUPPORTED: latest checkpoint is frame 0; it was preserved and cannot yet be resumed");
                    std::process::exit(2);
                }
                program_args.load = n as i32;
            }
            None => {
                eprintln!(
                    "--load {} (resume from latest) but no state_<N>.bin.gz found in {}",
                    program_args.load, program_args.output
                );
                std::process::exit(1);
            }
        }
    }
    if program_args.load > 0 {
        if let Err(e)=checkpoint_layout::validate(std::path::Path::new(&program_args.output)){eprintln!("{e}");std::process::exit(2);}
    }
    let info = git_info::get();
    info!(
        "git branch: {}",
        info.current_branch.unwrap_or("Unknown".to_string())
    );
    info!(
        "git hash: {}",
        info.head.last_commit_hash.unwrap_or("Unknown".to_string())
    );
    let mut scene = Scene::new(&program_args);
    if !restoring_backend_verified && scene.joint_input.as_ref().is_some_and(|input|
        input.requires_feature(joint_angular_extension_gpu::FEATURE_SWING_CONE).unwrap_or(false)) {
        if let Err(e)=ensure_restoring_backend(){eprintln!("JOINT_ANGULAR_SWING_CONE_NATIVE_CAPABILITY: {e}");std::process::exit(2);}
        restoring_backend_verified = true;
    }
    let sim_args = scene.args();
    let mut param = builder::make_param(&sim_args);
    let mixed_pcg = sim_args.pcg_precision != args::PcgPrecision::Fp32;
    if mixed_pcg {
        eprintln!("PCG_PRECISION_UNSUPPORTED: FP16 PCG has been retired; rebuild and run the scene from the beginning with FP32");
        std::process::exit(2);
    }
    resume_request::validate(program_args.load,program_args.target_frame,sim_args.frames,sim_args.fps,None)
        .unwrap_or_else(|e|{eprintln!("{e}");std::process::exit(2)});

    if program_args.load > 0 {
        info!("Loading dataset...");
        let mut dataset: DataSet = read(&read_gz(&format!(
            "{}/{}",
            program_args.output,
            ppf_cts_formats::files::DATASET
        )));
        // The dataset's rest shape (the `inv_rest*` matrices and rest angles)
        // is the build-time one, because the dataset is written once. A scene
        // whose rest shape creeps also writes it per frame, so overwrite those
        // fields with the frame being resumed (see `plastic_state`).
        let split_layout = std::path::Path::new(&format!(
            "{}/{}",
            program_args.output,
            ppf_cts_formats::files::CHECKPOINT_LAYOUT
        ))
        .exists();
        let path_plastic = format!(
            "{}/{}",
            program_args.output,
            ppf_cts_formats::files::plastic_filename(program_args.load)
        );
        let state_path=std::path::Path::new(&program_args.output).join(ppf_cts_formats::files::state_filename(program_args.load));
        let state_bytes=std::fs::read(&state_path).unwrap_or_else(|e|{eprintln!("CHECKPOINT_STATE_INVALID: {}: {e}",state_path.display());std::process::exit(2)});
        let decoded_state=checkpoint_state_marker::decode(&state_bytes).unwrap_or_else(|e|{eprintln!("{e}");std::process::exit(2)});
        let plastic_bytes=match std::fs::read(&path_plastic) {Ok(b)=>Some(b),Err(e) if e.kind()==std::io::ErrorKind::NotFound=>None,Err(e)=>{eprintln!("CHECKPOINT_PLASTIC_INVALID: {e}");std::process::exit(2)}};
        checkpoint_state_marker::validate_plastic(&decoded_state.plastic_binding,plastic_bytes.as_deref(),plastic_state::PlasticKinds::of(&dataset).any()).unwrap_or_else(|e|{eprintln!("{e}");std::process::exit(2)});
        if let Some(plastic_data)=plastic_bytes.as_ref() {
            info!("Loading plastic rest shape from {path_plastic}...");
            let mut raw=Vec::new();
            std::io::Read::read_to_end(&mut flate2::read::GzDecoder::new(plastic_data.as_slice()),&mut raw).unwrap_or_else(|e|{eprintln!("CHECKPOINT_PLASTIC_INVALID: {e}");std::process::exit(2)});
            let plastic: plastic_state::PlasticState = read(&raw);
            plastic.apply(&mut dataset);
        } else if split_layout && plastic_state::PlasticKinds::of(&dataset).any() {
            // This directory carries the layout marker, so the dataset's rest
            // shape is the build-time one, which belongs to no checkpoint, and
            // every checkpoint of a creeping scene was written with its own
            // per-frame copy. This frame's is not on disk, so its rest shape
            // exists nowhere and resuming would silently run on a shape from
            // the wrong time.
            panic!(
                "checkpoint {} has no matching {}: the rest shape for this \
                 frame is not on disk, so it cannot be resumed",
                ppf_cts_formats::files::state_filename(program_args.load),
                ppf_cts_formats::files::plastic_filename(program_args.load)
            );
        }
        // Reaching here means either the scene never creeps, so the build-time
        // rest shape in the dataset is still correct, or the directory carries
        // no layout marker, where the dataset holds the shape last written into
        // it. Both want it left as loaded. The two are told apart by that
        // marker rather than by looking for per-frame files, whose absence is
        // ambiguous: they are pruned along with their checkpoints, so "none
        // present" cannot distinguish "never written" from "deleted".
        info!("Loading backend state...");
        let state_requirement=decoded_state.requirement;
        // Consume exactly the State bytes whose gzip provenance was inspected.
        let mut backend = backend::Backend::load_state_payload(&program_args.output,&decoded_state.payload);
        let required=mixed_pcg||state_requirement!=checkpoint_state_marker::HistoryRequirement::LegacyUnmarkedAmbiguous||scene.joint_input.is_some()||prepared_curves.is_some()||
            [joint::JOINT_FILE,joint_root_motion_binding::ROOT_MOTION_FILE,soft_joint::FILE,cross_links::FILE,pmx_joints::FILE,connected_collision_native::CROSS_FILE].iter().any(|p|std::path::Path::new(&program_args.path).join(p).exists());
        let versioned_history=checkpoint_history::read_versioned(std::path::Path::new(&program_args.output),program_args.load,required)
            .unwrap_or_else(|e|{eprintln!("{e}");std::process::exit(2)});
        let history=versioned_history.as_ref().map(|h|h.legacy().clone());
        if backend.state.curr_frame!=program_args.load||backend.state.curr_vertex.ncols()!=dataset.vertex.curr.size as usize||backend.state.prev_vertex.ncols()!=dataset.vertex.prev.size as usize||!backend.state.time.is_finite()||backend.state.time<0.||
            !backend.state.prev_dt.is_finite()||backend.state.prev_dt<=0.||
            !backend.state.curr_vertex.iter().chain(backend.state.prev_vertex.iter()).all(|v|v.is_finite()){
            eprintln!("CHECKPOINT_STATE_INVALID: frame/clock/nonfinite accepted history");std::process::exit(2);
        }
        let request=resume_request::validate(program_args.load,program_args.target_frame,sim_args.frames,sim_args.fps,Some(backend.state.time))
            .unwrap_or_else(|e|{eprintln!("{e}");std::process::exit(2)});
        let soft_owned=scene.joint_input.as_ref().map(|i|cross_links::prepare_batch(i,scene.args().world_scaling as f64)).transpose()
            .unwrap_or_else(|e|{eprintln!("{e}");std::process::exit(2)}).flatten();
        let owned=scene.joint_input.as_ref().map(|i|i.records());
        if owned.is_some_and(|o|!o.hard_records.is_empty()){eprintln!("CHECKPOINT_HARD_UNSUPPORTED: HARD is outside bone resume V1");std::process::exit(2);}
        if let Some(time)=request.requested_time {
            scene.validate_resume_horizon(time).and_then(|_|resume_request::validate_control_coverage(owned.map_or(&[],|o|o.roots.as_slice()),soft_owned.as_ref().and_then(|s|s.control_events.as_ref()).map_or(&[],|p|p.bodies.as_slice()),time))
                .unwrap_or_else(|e|{eprintln!("{e}");std::process::exit(2)});
            for wire in owned.into_iter().flat_map(|o|o.joints.iter().map(|j|&j.wire)).chain(soft_owned.iter().flat_map(|s|s.wires.iter().map(|j|&j.wire))){
                if wire.enabled {resume_request::validate_angular_drive_coverage(wire.angular_drives_v1.as_ref().map(|v|&v.0),time).unwrap_or_else(|e|{eprintln!("{e}");std::process::exit(2)});}
            }
        }
        // AUTO/initial frames are verified against fresh sealed authority, never
        // checkpoint deformation or plastic rest. No device is initialized here.
        let authority=if required||history.is_some(){Some(fresh_authority_dataset(&mut scene))}else{None};
        let installed_curves=prepared_curves.as_ref().map(|p|restoring_native::verify_and_pack(p,&scene,authority.as_ref().expect("rest authority"),std::path::Path::new(&program_args.path)))
            .transpose().unwrap_or_else(|e|{eprintln!("{e}");std::process::exit(2)});
        let installed_curves=cross_links::merge_curves(scene.joint_input.as_ref(),installed_curves,&scene,authority.as_ref())
            .unwrap_or_else(|e|{eprintln!("{e}");std::process::exit(2)});
        if let Some(original)=&authority {
            checkpoint_history::validate_mass_authority(original,&dataset).unwrap_or_else(|e|{eprintln!("{e}");std::process::exit(2)});
            backend.sealed_initial=Some(original.vertex.curr.as_slice().iter().flat_map(|p|p.iter().copied()).collect());
        }
        if let Some(h)=&history {
            let mut authored=builder::make_param(&sim_args);
            if h.adaptive_enabled {scene.update_param(&sim_args,h.adaptive_start_time,&mut authored);}
            h.validate_static(std::path::Path::new(&program_args.path),&sim_args,&backend.state,scene.joint_input.as_ref().map(|i|i.records()),soft_owned.as_ref(),installed_curves.as_ref(),dataset.prop.pdrd_body.size as usize,authored.dt)
                .unwrap_or_else(|e|{eprintln!("{e}");std::process::exit(2)});
        }
        let grain_mask=checkpoint_sand::mask(&dataset).unwrap_or_else(|e|{eprintln!("{e}");std::process::exit(2)});
        match (&decoded_state.sand, grain_mask.as_ref()) {
            (Some(sand),Some(mask)) => {
                backend.resume_sand=Some(checkpoint_sand::decode(sand,std::path::Path::new(&program_args.output),&backend.state,mask)
                    .unwrap_or_else(|e|{eprintln!("{e}");std::process::exit(2)}));
            }
            (Some(_),None) => {eprintln!("CHECKPOINT_SAND_INVALID: marked State without grains");std::process::exit(2);}
            (None,Some(_)) => eprintln!("SAND_SPIN_LEGACY_COLD: checkpoint lacks accepted grain rotation; resumed spin is a cold seed and is not exact"),
            (None,None) => {}
        }
        if let Some(prepared)=soft_owned.as_ref().and_then(|s|s.connected_collision.as_ref()){
            backend.connected_collision=Some(connected_collision_native::validate_resume(&dataset,prepared,std::path::Path::new(&program_args.output)).unwrap_or_else(|e|{eprintln!("{e}");std::process::exit(2)}));
        }
        checkpoint_state_marker::require_validated_history(state_requirement,history.is_some())
            .unwrap_or_else(|e|{eprintln!("{e}");std::process::exit(2)});
        if history.is_none(){
            eprintln!("CHECKPOINT_LEGACY_HISTORY_ABSENT: unmarked checkpoint has no runtime history; using legacy reconstruction, exact continuation is not guaranteed");
        }
        let twist_restore_requested = program_args.twist_resume_mode.is_some()
            || matches!(versioned_history.as_ref(), Some(checkpoint_history::HistoryPayload::V2(_)))
            || matches!(versioned_history.as_ref(), Some(checkpoint_history::HistoryPayload::V3(v)) if v.twist.is_some() || scene.joint_input.as_ref().is_some_and(|input|input.records().joints.iter().any(|j|j.wire.enabled && j.wire.angular_control_v1.as_ref().is_some_and(|c|c.coordinate_model==twist_overlay::CONTINUOUS_MODEL))));
        if twist_restore_requested {
            let validated = scene.joint_input.as_ref().map(|input| input.records())
                .unwrap_or_else(|| {eprintln!("TWIST_RESUME_NO_ELIGIBLE_JOINTS"); std::process::exit(2)});
            let authored_continuous = validated.joints.iter().any(|j| j.wire.enabled &&
                j.wire.angular_control_v1.as_ref().is_some_and(|c| c.coordinate_model == twist_overlay::CONTINUOUS_MODEL));
            let overlay = if !program_args.twist_resume_mode.is_some() && authored_continuous {
                twist_overlay::fresh_from_validated(validated)
            } else {
                twist_overlay::from_validated(validated)
            }.unwrap_or_else(|e| {eprintln!("{e}"); std::process::exit(2)});
            backend.twist_checkpoint_context = Some(overlay.checkpoint_context(program_args.twist_resume_mode.is_some()));
        }
        if let Some(validated)=scene.joint_input.as_ref().map(|input|input.records()) {
            if validated.joints.iter().any(|j|j.wire.enabled && j.wire.angular_control_v1.as_ref().is_some_and(|c|c.coordinate_model==material_spin_history::MODEL)) {
                backend.material_spin_overlay=Some(twist_overlay::material_spin_from_validated(validated).unwrap_or_else(|e|{eprintln!("{e}");std::process::exit(2)}));
                if !matches!(versioned_history.as_ref(),Some(checkpoint_history::HistoryPayload::V3(_))){eprintln!("CHECKPOINT_MATERIAL_SPIN_INVALID: v3 requires v3 history; legacy migration is unavailable");std::process::exit(2);}
            }
        }
        if let Some(checkpoint_history::HistoryPayload::V3(v))=versioned_history.as_ref(){
            if backend.material_spin_overlay.is_some()!=v.rigid_material.is_some(){eprintln!("CHECKPOINT_MATERIAL_SPIN_INVALID: rigid history/selection mismatch");std::process::exit(2);}
        }
        backend.fem_material_spin_overlay=soft_owned.as_ref().map(twist_overlay::fem_material_spin_from_owned).transpose().unwrap_or_else(|e|{eprintln!("{e}");std::process::exit(2)}).flatten();
        match (backend.fem_material_spin_overlay.as_ref(),versioned_history.as_ref()) {
            (Some(ctx),Some(checkpoint_history::HistoryPayload::V3(v)))=>{
                let h=v.fem_material.as_ref().unwrap_or_else(||{eprintln!("CHECKPOINT_MATERIAL_SPIN_INVALID: missing FEM history");std::process::exit(2)});
                h.validate(&ctx.scene_uuid,&ctx.selected_joint_uuids,backend.state.time).unwrap_or_else(|e|{eprintln!("{e}");std::process::exit(2)});
            },
            (Some(_),_)=>{eprintln!("CHECKPOINT_MATERIAL_SPIN_INVALID: FEM v3 requires v3 history; legacy migration unavailable");std::process::exit(2);},
            (None,Some(checkpoint_history::HistoryPayload::V3(v))) if v.fem_material.is_some()=>{eprintln!("CHECKPOINT_MATERIAL_SPIN_INVALID: unexpected FEM history");std::process::exit(2);},
            _=>{},
        }
        backend.resume_history=history;
        backend.resume_twist_payload=versioned_history;
        // Every fallible resume contract above precedes output-tail trimming.
        begin_output(&program_args);
        let identity=scene.joint_input.as_ref().map(|i|i.records());
        let receipt=serde_json::json!({"kind":"ppf.bone_resume_request","version":1,"feature":"bone_checkpoint_resume_v1","state":"ACCEPTED",
            "restored_frame":program_args.load,"sealed_target_frame":sim_args.frames,"requested_target_frame":request.effective_frames,"restored_time_seconds":backend.state.time,
            "scene_uuid":identity.map(|o|o.scene_uuid.as_str()).unwrap_or(""),"scene_inputs_sha256":identity.map(|o|o.scene_inputs_sha256.as_str()).unwrap_or(""),"launch_id":status_writer::launch_id()});
        let receipt_path=std::path::Path::new(&program_args.output).join("resume_request_native_receipt.json");
        let receipt_tmp=receipt_path.with_extension("json.tmp");std::fs::write(&receipt_tmp,serde_json::to_vec_pretty(&receipt).expect("resume receipt")).expect("resume receipt stage");std::fs::rename(receipt_tmp,receipt_path).expect("resume receipt publish");
        info!("Data loaded successfully");
        param.time = backend.state.time;
        param.time_f32 = param.time as f32;
        param.prev_dt = backend.state.prev_dt;
        builder::copy_to_dataset(
            &backend.state.curr_vertex,
            &backend.state.prev_vertex,
            &mut dataset,
        );
        backend.run(&program_args, &sim_args, dataset, param, scene, installed_curves.as_ref(), prepared_curves.as_ref());
    } else {
        let fresh_continuous = scene.joint_input.as_ref().is_some_and(|input| input.records().joints.iter().any(|j|
            j.wire.enabled && j.wire.angular_control_v1.as_ref().is_some_and(|c| c.coordinate_model == twist_overlay::CONTINUOUS_MODEL)));
        let fresh_overlay = if fresh_continuous {
            Some(twist_overlay::fresh_from_validated(scene.joint_input.as_ref().expect("continuous scene").records())
                .unwrap_or_else(|e| {eprintln!("{e}"); std::process::exit(2)}))
        } else { None };
        begin_output(&program_args);
        // Bind fresh output to the mapping that produced this input. Resumes
        // retain the output identity established by the original run.
        let mapping_source = std::path::Path::new(&program_args.path).join("fetch_mapping_identity.json");
        if mapping_source.exists() {
            let output = std::path::Path::new(&program_args.output);
            let temporary = output.join("fetch_mapping_identity.json.tmp");
            let destination = output.join("fetch_mapping_identity.json");
            std::fs::copy(&mapping_source, &temporary)
                .expect("fetch mapping identity staging failed");
            std::fs::rename(&temporary, &destination)
                .expect("fetch mapping identity publication failed");
        }
        info!("Initializing mesh...");
        let mut backend = backend::Backend::new(scene.make_mesh());
        if let Some(overlay) = fresh_overlay { backend.twist_checkpoint_context = Some(overlay.checkpoint_context(false)); }
        if let Some(validated)=scene.joint_input.as_ref().map(|input|input.records()) {
            if validated.joints.iter().any(|j|j.wire.enabled && j.wire.angular_control_v1.as_ref().is_some_and(|c|c.coordinate_model==material_spin_history::MODEL)) {
                backend.material_spin_overlay=Some(twist_overlay::material_spin_from_validated(validated).unwrap_or_else(|e|{eprintln!("{e}");std::process::exit(2)}));
            }
        }
        let mesh = &backend.mesh;
        let face_area = triutils::face_areas(&mesh.vertex, &mesh.mesh.mesh.face);
        let tet_volumes = triutils::tet_volumes(&mesh.vertex, &mesh.mesh.mesh.tet);

        info!("Building material properties...");
        let mut props = scene.make_props(mesh, &face_area, &tet_volumes);

        info!("Computing constraints and parameters...");
        let time = backend.state.time;
        // Standalone snapshot for the param summary, not a step: prev == time,
        // so every fix pin's step_delta is zero.
        let temp_constraint = scene.make_constraint(time, time);
        if prepared_curves.is_none(){scene.export_param_summary(&program_args, &props, &face_area, &tet_volumes);}

        let mut total_rod_mass = 0.0;
        let mut total_area_mass = 0.0;
        let mut total_vol_mass = 0.0;

        for prop in props.face.iter() {
            total_area_mass += prop.mass;
        }

        // Process rod edges (sequential, small count)
        for i in 0..mesh.mesh.mesh.rod_count {
            let prop = props.edge[i];
            total_rod_mass += prop.mass;
        }

        append_fresh_face_edges(mesh,&mut props,&face_area);

        for prop in props.tet.iter() {
            total_vol_mass += prop.mass;
        }

        let dataset=construct_fresh_dataset(&scene,mesh,&sim_args,&mut props,temp_constraint);
        let installed_curves=prepared_curves.as_ref().map(|p|restoring_native::verify_and_pack(p,&scene,&dataset,std::path::Path::new(&program_args.path)))
            .transpose().unwrap_or_else(|e|{eprintln!("{e}");std::process::exit(2)});
        let installed_curves=cross_links::merge_curves(scene.joint_input.as_ref(),installed_curves,&scene,Some(&dataset))
            .unwrap_or_else(|e|{eprintln!("{e}");std::process::exit(2)});
        if prepared_curves.is_some(){scene.export_param_summary(&program_args, &props, &face_area, &tet_volumes);}
        let mut mass_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(format!("{}/data/total_mass.out", program_args.output).as_str())
            .unwrap();
        writeln!(
            mass_file,
            "{total_rod_mass} {total_area_mass} {total_vol_mass}"
        )
        .unwrap();
        info!("Dataset built successfully");

        backend.run(&program_args, &sim_args, dataset, param, scene, installed_curves.as_ref(), prepared_curves.as_ref());
    }
}

fn remove_old_files(dirpath: &str, prefix: &str, suffix: &str, keep_number: i32, max_frame: i32) {
    if keep_number <= 0 {
        return;
    }

    let mut n = 0;
    for i in 0..=max_frame {
        let path = format!("{dirpath}/{prefix}{i}{suffix}");
        if std::path::Path::new(&path).exists() {
            n += 1;
        }
    }

    let mut i = 0;
    while n > keep_number {
        let filename = format!("{prefix}{i}{suffix}");
        let path = format!("{dirpath}/{filename}");
        if std::path::Path::new(&path).exists() {
            info!("Removing {filename}...");
            std::fs::remove_file(path).unwrap_or(());
            n -= 1;
        }
        i += 1;
        if i > max_frame {
            break;
        }
    }
}

fn remove_files(dirpath: &str, base: &str, ext: &str, load: i32) {
    let mut count = load + 1;
    loop {
        let path = format!("{dirpath}/{base}_{count}.{ext}");
        if std::path::Path::new(&path).exists() {
            std::fs::remove_file(path).unwrap_or(());
        } else {
            break;
        }
        count += 1;
    }
}

fn remove_output_entry(path: &std::path::Path, is_dir: bool) -> std::io::Result<()> {
    let remove = || {
        if is_dir {
            std::fs::remove_dir_all(path)
        } else {
            std::fs::remove_file(path)
        }
    };
    let start = std::time::Instant::now();
    loop {
        match remove() {
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            // A frame reader may still hold the old output for a short time.
            // Retry Windows sharing and lock violations, with a bounded wait.
            Err(err)
                if cfg!(windows)
                    && matches!(err.raw_os_error(), Some(32 | 33))
                    && start.elapsed() < std::time::Duration::from_millis(750) =>
            {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            result => return result,
        }
    }
}

fn remove_files_in_dir(dir: &std::path::Path) -> Result<(), String> {
    let path_error = |path: &std::path::Path, err: std::io::Error| {
        format!("{}: {err}", path.display())
    };
    let entries = std::fs::read_dir(dir).map_err(|err| path_error(dir, err))?;
    for entry in entries {
        let entry = entry.map_err(|err| path_error(dir, err))?;
        // ``save_and_quit`` is a control sentinel the server may
        // write between launch and the solver's setup(). Wiping
        // it here would lose the save request and the solver
        // would run to completion. Preserve it. The server's
        // launch_solver already cleared stale sentinels before
        // spawning, so anything present now is fresh.
        //
        // ``hold_at_frame`` is preserved for the same reason: a
        // frontend asking to hold at an early frame writes it before
        // it launches, and a hold wiped here runs the whole scene.
        // Keep the already acquired lock's file on every platform; removing
        // it would let another process lock a newly created file.
        if entry.file_name() == ppf_cts_formats::files::SAVE_AND_QUIT
            || entry.file_name() == ppf_cts_formats::files::HOLD_AT_FRAME
            || entry.file_name() == ppf_cts_formats::files::STATUS_LOCK
        {
            continue;
        }
        let path = entry.path();
        let metadata = match std::fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
            Err(err) => return Err(path_error(&path, err)),
        };
        if metadata.is_file() || metadata.is_dir() {
            remove_output_entry(&path, metadata.is_dir())
                .map_err(|err| path_error(&path, err))?;
        }
    }
    Ok(())
}

/// Read the simulation time recorded for `load` from
/// ``<output>/data/frame_to_time.out``. That stream is written as
/// ``{frame} {time}`` once per output frame (see
/// Backend::write_frame_outputs), so the row whose first column equals
/// `load` gives the sim time at the resume checkpoint. Returns None when
/// the file is missing or has no matching row; callers then skip
/// time-keyed pruning rather than guess a threshold.
fn loaded_time_for_frame(output: &str, load: i32) -> Option<f64> {
    let contents = std::fs::read_to_string(format!("{output}/data/frame_to_time.out")).ok()?;
    for line in contents.lines() {
        let mut cols = line.split_whitespace();
        let frame: i32 = cols.next()?.parse().ok()?;
        if frame == load {
            return cols.next()?.parse().ok();
        }
    }
    None
}

/// On resume, drop rows that belong to frames the run is about to
/// resimulate from the ``<output>/data/*.out`` streams. All of these are
/// opened append-only (Rust-side time_per_frame/frame_to_time/clock and
/// the C++ SimpleLog metrics), so without this the frames after the
/// checkpoint would get duplicate keys, skewing the averaged-metric
/// readers in ppf-cts-core's session::log.
///
/// time_per_frame.out and frame_to_time.out are keyed by integer frame
/// index, so keep rows with ``frame <= load``. Every other stream is
/// keyed by sim time (the C++ SimpleLog writes ``{time} {value}``, and
/// clock.out matches), so keep rows with ``time <= loaded_time``. Lines
/// whose first column is not the expected numeric key are left in place
/// to avoid silently dropping data we don't understand.
fn prune_out_streams(output: &str, load: i32) {
    let data_dir = format!("{output}/data");
    let loaded_time = loaded_time_for_frame(output, load);
    let entries = match std::fs::read_dir(&data_dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("out") {
            continue;
        }
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(name) => name,
            None => continue,
        };
        let frame_keyed = name == "time_per_frame.out" || name == "frame_to_time.out";
        // Without a sim-time threshold there is nothing to prune the
        // time-keyed streams against, so leave them untouched.
        if !frame_keyed && loaded_time.is_none() {
            continue;
        }
        let contents = match std::fs::read_to_string(&path) {
            Ok(contents) => contents,
            Err(_) => continue,
        };
        let mut kept = String::with_capacity(contents.len());
        for line in contents.lines() {
            let keep = match line.split_whitespace().next() {
                Some(first) if frame_keyed => match first.parse::<i32>() {
                    Ok(frame) => frame <= load,
                    Err(_) => true,
                },
                Some(first) => match first.parse::<f64>() {
                    Ok(time) => time <= loaded_time.unwrap(),
                    Err(_) => true,
                },
                None => true,
            };
            if keep {
                kept.push_str(line);
                kept.push('\n');
            }
        }
        std::fs::write(&path, kept).unwrap_or(());
    }
}

fn begin_output(program_args:&ProgramArgs) {
    let output = std::path::Path::new(&program_args.output);
    std::fs::create_dir_all(output).unwrap_or_else(|e| {
        eprintln!("OUTPUT_LOCK_ERROR: cannot create output directory: {e}");
        std::process::exit(2);
    });
    let lock = ppf_cts_formats::status::lock::acquire(output).unwrap_or_else(|e| {
        eprintln!("OUTPUT_LOCK_ERROR: cannot acquire output lock: {e}");
        std::process::exit(2);
    });
    let cleanup = if program_args.load == 0 {
        remove_files_in_dir(output)
    } else {
        Ok(())
    };

    // Begin the structured status record after attempting fresh output cleanup and before any
    // backend work, so the lock covers the whole compute lifetime and a
    // fresh run never reads its own scrubbed record. launch_id only needs
    // to be unique across launches in the same output dir, so a
    // pid + wall-clock mix is sufficient (not a security token).
    let launch_id = {
        let pid = std::process::id() as u64;
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let mixed = pid.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(nanos) & 0xFFFF_FFFF_FFFF;
        format!("{mixed:012x}")
    };
    status_writer::init(&program_args.output, launch_id, program_args.load > 0, lock);
    if let Err(detail) = cleanup {
        let detail = format!("OUTPUT_CLEANUP_ERROR: cannot clear previous output at {detail}");
        eprintln!("{detail}");
        status_writer::terminal_crash(ppf_cts_formats::status::CrashKind::LaunchFailed, detail);
        std::process::exit(2);
    }
    setup(program_args);
}
fn prune_output_prefix(program_args: &ProgramArgs) {
    if program_args.load > 0 {
        remove_files(&program_args.output, "vert", "bin", program_args.load);
        remove_files(&program_args.output, "display_pin", "bin", program_args.load);
        prune_pose_tail(&program_args.output, program_args.load)
            .expect("failed to retire superseded pose generations");
        remove_files(
            &program_args.output,
            "statistics",
            "cbor",
            program_args.load,
        );
        // State checkpoints are sparse on disk: save_state writes
        // state_<N>.bin.gz only every auto_save frames, and
        // remove_old_files keeps just keep_states of them at the
        // highest indices. So the indices are scattered high values,
        // not contiguous from load+1. A contiguous break-on-gap scan
        // (like the vert one above) would stop at the first absent
        // index after the resume point and leave newer checkpoints
        // behind, which a later `--load -1` (resume from latest) would
        // then resolve to and jump past the frame the user branched
        // from. Enumerate the actual on-disk set and drop every index
        // newer than the resume frame instead of assuming contiguity.
        //
        // The per-frame rest shape is dropped in lockstep with the checkpoint it
        // belongs to. Both sets are written at the same frames and pruned to
        // the same `keep_states` count, so letting one outlive the other would
        // desync the two budgets: an orphaned `plastic_<n>` above the resume
        // point pushes the retention pass into deleting the OLDEST plastic
        // file, whose `state_<n>` is still on disk, and resuming from that
        // frame would then fail on a rest shape that no longer exists.
        let output_dir = std::path::Path::new(&program_args.output);
        for n in ppf_cts_core::datamodel::list_saved_states(output_dir) {
            if (n as i32) > program_args.load {
                for name in [
                    ppf_cts_formats::files::state_filename(n as i32),
                    ppf_cts_formats::files::plastic_filename(n as i32),
                    checkpoint_commit::history_name(n as i32),
                ] {
                    let path = format!("{}/{name}", program_args.output);
                    std::fs::remove_file(path).unwrap_or(());
                }
            }
        }
        // The .out streams in output/data are append-only, so the resume
        // run would re-append rows for frames load+1.. that the original
        // run already recorded. Prune them to frames/time <= the
        // checkpoint so resumed frames don't produce duplicate keys.
        prune_out_streams(&program_args.output, program_args.load);
    }

}

fn setup(program_args: &ProgramArgs) {
    if !std::path::Path::new(&program_args.output).exists() {
        std::fs::create_dir_all(&program_args.output).unwrap_or(());
    }
    // ``output/data/`` holds the per-frame .out streams (total_mass,
    // time_per_frame, frame_to_time, ...). Nothing else creates it,
    // so a fresh run on a clean output dir hits NotFound when the
    // first writeln! fires. Make sure the dir exists up-front.
    let data_dir = format!("{}/data", program_args.output);
    if !std::path::Path::new(&data_dir).exists() {
        std::fs::create_dir_all(&data_dir).unwrap_or(());
    }
    // The server's launch_solver removes any stale ``save_and_quit``
    // sentinel before spawning this process, so anything present here
    // was written between spawn and setup() and reflects the user's
    // intent. Don't preemptively delete it: the run() loop will pick
    // it up on the first iteration and call save_state.

    // TWO WRITERS SHARE THIS STDOUT STREAM, AND ONLY ONE OF THEM IS TIMESTAMPED.
    // This pattern prefixes every `::log::` record, which is the whole of the
    // solver's Rust-side output. The other writer is `driver::log`, which
    // reproduces `src/kernels/simplelog/SimpleLog.cpp` and prints with
    // `println!` and no prefix at all: the mark lines (`* iter: 199`), the
    // phase lines (`> linsolve...7 msec`) and the section brackets
    // (`====== advance ======`). A solve's transcript therefore carries
    // `===== advance: 28 msec =====` bare and
    // `[2026-09-08 07:26:36] GPU SM Clock: 2520 MHz` prefixed one line later.
    //
    // So the missing prefix on those lines is the specification rather than an
    // oversight, and routing them through `info!` to make the stream uniform
    // would restore the prefix and change what every reader of this log sees:
    // the frontend's `>>> Log:` tail, the addon's log panel and a person
    // watching a solve. Those lines are a host contract, not free-form output.
    //
    // Interleaving is exact and needs no coordination. log4rs's
    // `ConsoleAppender` writes through `io::stdout()` whenever stdout is not a
    // tty, which is the shipped case (the solver's stdout is redirected into
    // `session/stdout.log`), so it shares the handle and the lock `println!`
    // uses. On a tty it writes to fd 1 directly, but it flushes per record and
    // `println!` flushes per line, so neither writer leaves a partial line
    // outstanding across the other.
    // THE LEVEL IS READ FROM `RUST_LOG`, AND WITHOUT THAT A DEMOTION IS A
    // DELETION. Diagnostics that carry more than a shipped transcript needs sit
    // at `debug!`, so the default stream stays the one the addon's log panel
    // and the frontend's tail were written against. That is only a demotion if
    // something can still ask for them: with the level hardcoded at `Info` the
    // `debug!` call is unreachable in the shipped binary, and the fixtures that
    // assert on those lines fail with no way to switch them back on.
    fn level_from_environment() -> LevelFilter {
        // An unparseable value is reported rather than silently ignored: a
        // typo in `RUST_LOG` that quietly gave the default would be read as
        // the diagnostic having gone missing.
        match std::env::var("RUST_LOG") {
            Ok(text) => match text.trim().parse::<LevelFilter>() {
                Ok(level) => level,
                Err(_) => {
                    eprintln!(
                        "solver: RUST_LOG={text:?} is not a log level; \
                         using info. Valid: error, warn, info, debug, trace."
                    );
                    LevelFilter::Info
                }
            },
            Err(_) => LevelFilter::Info,
        }
    }

    let pattern = "[{d(%Y-%m-%d %H:%M:%S)}] {m}{n}";
    let stdout = ConsoleAppender::builder()
        .encoder(Box::new(PatternEncoder::new(pattern)))
        .build();
    let config = Config::builder()
        .appender(Appender::builder().build("stdout", Box::new(stdout)))
        .build(Root::builder().appender("stdout").build(level_from_environment()))
        .unwrap();

    log4rs::init_config(config).unwrap();

    info!("{}", std::env::args().collect::<Vec<_>>().join(" "));
    let data_dir = CString::new(format!("{}/data", program_args.output)).unwrap();
    unsafe {
        set_log_path(data_dir.as_ptr());
    }
}

fn decompress_from_gz(compressed_data: Vec<u8>, path: &str) -> Vec<u8> {
    let mut decoder = GzDecoder::new(compressed_data.as_slice());
    let mut decompressed_data = Vec::new();
    if let Err(e) = decoder.read_to_end(&mut decompressed_data) {
        panic!(
            "Failed to decompress '{path}': {e}. The file is likely truncated \
             or corrupt, for example an auto-save that was interrupted or an \
             incomplete file transfer. Resuming from this checkpoint is not \
             possible; re-run the simulation or resume from a valid checkpoint."
        );
    }
    decompressed_data
}

/// 64 KiB I/O write buffer.
const IO_BUF_CAPACITY: usize = 64 * 1024;

/// Streams serialization directly to a gzip-compressed file without buffering entire data in RAM.
///
/// The bytes are written to a temporary sibling file and then atomically
/// renamed into `path`. A process killed mid-write (e.g. an interrupted
/// auto-save) therefore leaves either the previous valid file or the fully
/// written new one, never a truncated gzip that fails to decompress on resume
/// with `UnexpectedEof`. The temp file is fsynced before the rename so a crash
/// right after the rename cannot expose partially flushed contents.
fn save<T: serde::Serialize>(obj: &T, path: &str) {
    let tmp_path = format!("{}.tmp.{}", path, std::process::id());
    {
        let file = std::fs::File::create(&tmp_path).unwrap();
        let buf_writer = std::io::BufWriter::with_capacity(IO_BUF_CAPACITY, file);
        let mut encoder = GzEncoder::new(buf_writer, Compression::default());
        bincode::serialize_into(&mut encoder, obj).unwrap();
        let buf_writer = encoder.finish().unwrap();
        let file = buf_writer.into_inner().unwrap();
        file.sync_all().unwrap();
    }
    std::fs::rename(&tmp_path, path).unwrap();
}

fn read_gz(path: &str) -> Vec<u8> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("Failed to read '{path}': {e}"));
    decompress_from_gz(bytes, path)
}

fn read<'de, T: serde::Deserialize<'de>>(data: &'de [u8]) -> T {
    // bincode is POSITIONAL: it records no field names and honors no
    // `serde(default)`, so a checkpoint written by a build whose structs have
    // since gained a field cannot be read back, and the failure arrives here
    // as an opaque complaint about trailing bytes or an implausible length.
    // Name the likely cause instead. The dataset is write-once per session
    // (see the checkpoint layout notes), so there is no self-healing path and
    // the answer is always to rebuild the scene rather than to retry.
    let type_name = std::any::type_name::<T>();
    bincode::deserialize(data).unwrap_or_else(|e| {
        panic!(
            "Failed to deserialize a checkpoint of type {type_name}: {e}.\n\
             This usually means it was written by a build with a different \
             in-memory layout for {type_name}, which bincode cannot migrate. \
             Rebuild the scene to produce a fresh session directory; an \
             existing dataset.bin.gz cannot be upgraded in place."
        )
    })
}

// Receipt rebind policy only: this is neither a GPU ABI guarantee nor a
// substitute for checkpoint validation and native executable/library probing.
#[cfg(neutral_driver)]
fn accepted_receipt_source_stamps() -> Vec<&'static str> {
    let mut stamps = vec![ppf_cts_formats::SOURCE_STAMP, "d45426411338a803"];
    stamps.sort_unstable();
    stamps.dedup();
    stamps
}

#[cfg(all(test, neutral_driver))]
#[test]
fn accepted_receipt_stamps_are_exact_and_unique() {
    let stamps = accepted_receipt_source_stamps();
    assert!(stamps.contains(&ppf_cts_formats::SOURCE_STAMP));
    assert!(stamps.contains(&"d45426411338a803"));
    assert!(stamps.windows(2).all(|v| v[0] < v[1]));
    assert!(stamps.iter().all(|s| s.len() == 16 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))));
    assert!(stamps.len() <= 2);
}

#[cfg(test)]
mod tests {
    use super::prune_out_streams;
    use std::collections::HashSet;
    use std::fs;

    fn write_stream(data_dir: &std::path::Path, name: &str, body: &str) {
        fs::write(data_dir.join(name), body).unwrap();
    }

    #[test]
    fn prune_out_streams_drops_resimulated_frames() {
        let output = std::env::temp_dir().join(format!(
            "ppf_cts_prune_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let data_dir = output.join("data");
        fs::create_dir_all(&data_dir).unwrap();

        let load = 3;
        // frame_to_time maps frame -> sim time; the row for `load` sets the
        // sim-time threshold the time-keyed streams are pruned against.
        write_stream(
            &data_dir,
            "frame_to_time.out",
            "0 0\n1 0.04\n2 0.08\n3 0.12\n4 0.16\n5 0.2\n",
        );
        write_stream(
            &data_dir,
            "time_per_frame.out",
            "0 10\n1 11\n2 12\n3 13\n4 14\n5 15\n",
        );
        // clock.out (and the C++ SimpleLog streams) are keyed by sim time.
        write_stream(
            &data_dir,
            "clock.out",
            "0.000000 1500\n0.040000 1500\n0.080000 1500\n0.120000 1500\n0.160000 1400\n",
        );
        write_stream(
            &data_dir,
            "advance.total_mass.out",
            "0.000000 12.5\n0.080000 12.5\n0.120000 12.5\n0.200000 12.5\n",
        );

        prune_out_streams(output.to_str().unwrap(), load);

        // Frame-keyed streams keep only frame <= load with no duplicate keys.
        for name in ["frame_to_time.out", "time_per_frame.out"] {
            let contents = fs::read_to_string(data_dir.join(name)).unwrap();
            let mut seen = HashSet::new();
            for line in contents.lines() {
                let frame: i32 = line.split_whitespace().next().unwrap().parse().unwrap();
                assert!(
                    frame <= load,
                    "{name}: frame {frame} > load {load} survived"
                );
                assert!(seen.insert(frame), "{name}: duplicate frame {frame}");
            }
            assert_eq!(seen.len(), (load + 1) as usize, "{name}: wrong row count");
        }

        // Time-keyed streams keep only rows up to the checkpoint sim time.
        for name in ["clock.out", "advance.total_mass.out"] {
            let contents = fs::read_to_string(data_dir.join(name)).unwrap();
            let mut seen: HashSet<u64> = HashSet::new();
            for line in contents.lines() {
                let time: f64 = line.split_whitespace().next().unwrap().parse().unwrap();
                assert!(
                    time <= 0.12 + 1e-9,
                    "{name}: time {time} > loaded time survived"
                );
                assert!(seen.insert(time.to_bits()), "{name}: duplicate time {time}");
            }
        }

        fs::remove_dir_all(&output).unwrap_or(());
    }

    #[test]
    fn save_round_trips_and_leaves_no_temp_file() {
        let dir = std::env::temp_dir().join(format!(
            "ppf_cts_save_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dataset.bin.gz");
        let path_str = path.to_str().unwrap();

        let original: Vec<u64> = (0..10_000u64).collect();
        super::save(&original, path_str);

        // The atomically renamed file decompresses and deserializes cleanly,
        // and no `.tmp.<pid>` sibling is left behind after a successful save.
        let loaded: Vec<u64> = super::read(&super::read_gz(path_str));
        assert_eq!(loaded, original);
        let temp_left = fs::read_dir(&dir)
            .unwrap()
            .filter_map(Result::ok)
            .any(|e| e.file_name().to_string_lossy().contains(".tmp."));
        assert!(!temp_left, "temp file was not renamed away");

        // A pre-existing truncated file must not block a fresh valid save.
        fs::write(&path, b"\x1f\x8b\x08not a complete gzip stream").unwrap();
        super::save(&original, path_str);
        let reloaded: Vec<u64> = super::read(&super::read_gz(path_str));
        assert_eq!(reloaded, original);

        fs::remove_dir_all(&dir).unwrap_or(());
    }
}

mod joint_root_hard_affine;
mod joint_root_hard_precision;

// Pure fresh construction shared by normal run and read-only Build authority.
fn append_fresh_face_edges(mesh:&MeshSet,props:&mut Props,face_area:&[f32]) {
        info!("Building edge parameter map...");
        // Build edge param map for deduplication
        let mut edge_param_map: HashMap<EdgeParam, u32> = HashMap::new();
        for (i, param) in props.edge_params.iter().enumerate() {
            edge_param_map.insert(*param, i as u32);
        }

        info!("Processing edge properties...");
        // Process non-rod edges (parallelized)
        // Step 1: Parallel computation
        let rod_count = mesh.mesh.mesh.rod_count;
        let edge_count = mesh.mesh.mesh.edge.ncols();
        let non_rod_edge_data: Vec<(f32, EdgeParam)> = (rod_count..edge_count)
            .into_par_iter()
            .map(|i| {
                let rod = mesh.mesh.mesh.edge.column(i);
                let x0 = mesh.vertex.column(rod[0]);
                let x1 = mesh.vertex.column(rod[1]);
                let length = (x1 - x0).map(f32::from).norm();
                let param = builder::averaged_edge_param(
                    &mesh.mesh.neighbor.edge.face[i],
                    &props.face,
                    &props.face_params,
                    |j, _| face_area[j],
                );
                (length, param)
            })
            .collect();

        // Step 2: Sequential deduplication
        let animated_materials = props.animated_materials;
        for (length, param) in non_rod_edge_data {
            // Identity indices for an animated scene, matching every other
            // param table: the per-frame path rewrites these values, and a
            // deduplicated table would renumber underneath the param_index
            // stored on each EdgeProp below.
            let param_idx = if animated_materials {
                props.edge_params.push(param);
                (props.edge_params.len() - 1) as u32
            } else {
                *edge_param_map.entry(param).or_insert_with(|| {
                    let new_idx = props.edge_params.len() as u32;
                    props.edge_params.push(param);
                    new_idx
                })
            };
            props.edge.push(EdgeProp {
                length,
                initial_length: length,
                mass: 0.0,
                fixed: false,
                param_index: param_idx,
                // Both `fixed` and this are derived in builder::build, which
                // runs after these face edges have been appended, so it sees
                // the full edge list.
                pin_allow_intersection: false,
            });
        }

}
fn construct_fresh_dataset(scene:&Scene,mesh:&MeshSet,sim_args:&SimArgs,props:&mut Props,temp_constraint:data::Constraint)->DataSet {
        let velocity = scene.get_initial_velocity();

        info!("Building dataset (this may take a while)...");
        let pdrd_data = builder::PdrdSceneData {
            body_rows: scene.pdrd_body_rows(),
            vert_index: scene.pdrd_vert_index(),
            vert_list: scene.pdrd_vert_list(),
            rest_centered: scene.pdrd_rest_centered(),
        };
        let lock_data = builder::LockSceneData {
            translation_axes: scene.translation_locks(),
            translation_modes: scene.translation_lock_modes(),
            rotation_axes: scene.rotation_locks(),
            rotation_modes: scene.rotation_lock_modes(),
            vert_dmap: scene.vert_dmap(),
        };
        builder::build(
            sim_args,
            mesh,
            &velocity,
            props,
            temp_constraint,
            pdrd_data,
            lock_data,
        )
}
fn fresh_authority_dataset(scene:&mut Scene)->DataSet {
let sim_args=scene.args();let mesh=scene.make_mesh();let face_area=triutils::face_areas(&mesh.vertex,&mesh.mesh.mesh.face);let tet_volume=triutils::tet_volumes(&mesh.vertex,&mesh.mesh.mesh.tet);let mut props=scene.make_props(&mesh,&face_area,&tet_volume);let constraint=scene.make_constraint(0.,0.);append_fresh_face_edges(&mesh,&mut props,&face_area);construct_fresh_dataset(scene,&mesh,&sim_args,&mut props,constraint)
}
fn compiled_restoring_identity()->restoring_mass_authority::SolverIdentity {
    let hex=|bytes:[u8;32]|bytes.iter().map(|b|format!("{b:02x}")).collect::<String>();
    restoring_mass_authority::SolverIdentity {
        source_stamp:restoring_identity::RESTORING_CURVE_SOURCE_STAMP.into(),
        selected_backend:env!("PPF_BACKEND").into(),
        abi_version:ppf_cts_compute::abi::ABI_VERSION,
        layout_sha256:hex(restoring_identity::RESTORING_CURVE_LAYOUT_SHA256),
        kernel_table_sha256:hex(restoring_identity::RESTORING_CURVE_KERNEL_TABLE_SHA256),
    }
}
fn mixed_pcg_device(_precision: args::PcgPrecision) -> Result<(), String> {
    Err("FP16 PCG has been retired; rebuild and run the scene from the beginning with FP32".into())
}

fn ensure_restoring_backend()->Result<String,String> {
    probe_restoring_backend(args::PcgPrecision::Fp32)
}

fn probe_restoring_backend(precision: args::PcgPrecision) -> Result<String,String> {
    if restoring_identity::RESTORING_CURVE_CONTRACT_VERSION!=1 || restoring_identity::RESTORING_CURVE_CHANNEL_MASK!=15 {
        return Err("RESTORING_CAPABILITY_UNAVAILABLE: this build does not advertise all V1 channels".into());
    }
    // Actual open compares the loaded probe, including ABI, PODs, channel mask
    // and both compiled fingerprints, before creating the device.
    let device=driver::probe_restoring_device(precision).map_err(|e|format!("RESTORING_CAPABILITY_UNAVAILABLE: {e}"))?;
    #[cfg(abi_backend_linked)]
    if ppf_cts_compute::abi::linked_backend_name()!=env!("PPF_BACKEND") {return Err("RESTORING_CAPABILITY_UNAVAILABLE: loaded backend differs from compiled selection".into());}
    Ok(device)
}


// Retire only previously published pose generations after a validated rewind.
// Called by the existing post-restore prune entry, never by the publisher.
fn prune_pose_tail(output: &str, load: i32) -> std::io::Result<()> {
    if load < 0 { return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput,"negative pose rewind frame")); }
    for entry in std::fs::read_dir(output)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() { continue; }
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue; };
        let index = ["rigid_pose_", "fem_pose_"].iter().find_map(|prefix| {
            name.strip_prefix(prefix).and_then(|tail| tail.strip_suffix(".cbor.tmp").or_else(||tail.strip_suffix(".cbor")))
        });
        let Some(index) = index else { continue; };
        if index.is_empty() || !index.bytes().all(|v|v.is_ascii_digit()) { continue; }
        let Ok(frame) = index.parse::<u32>() else { continue; };
        if frame > load as u32 { std::fs::remove_file(entry.path())?; }
    }
    Ok(())
}

