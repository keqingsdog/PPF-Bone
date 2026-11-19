//! Private real Backend::run save fixture; only enabled explicitly by the test.
use super::*;
use clap::Parser;
use std::cell::RefCell;

struct Captured {
    frame:i32, time:f64, prev_dt:f32, curr:Vec<f32>, prev:Vec<f32>,
    runtime:crate::driver::checkpoint::Snapshot,
    material:crate::material_spin_history::AcceptedMaterialSpin,
    adaptive:crate::checkpoint_values::Controller, accepted:u64, adaptive_start:f64,
    free_vertex:u32, rejected:bool, save_seen:bool, advances:usize, cancels:usize, finishes:usize,
}
thread_local! { static CAPTURE:RefCell<Option<Captured>>=const { RefCell::new(None) }; static ENABLED:std::cell::Cell<bool>=const { std::cell::Cell::new(false) }; }

// Invoked before the logical step and before its host seed is captured.
pub(super) fn capture_if_ready(backend:&mut Backend,data:&DataSet,param:&ParamSet) {
    if !ENABLED.with(|e|e.get()) || backend.state.curr_frame<1 || CAPTURE.with(|c| c.borrow().is_some()) { return; }
    assert!(crate::driver::host_newton_root_scope(param.playback).unwrap(),"fixture requires real root retry scope");
    let material=crate::driver::export_accepted_material_spin().unwrap().expect("selected material history");
    if !material.rows.iter().any(|r|r.a.iter().any(|v|*v!=0.)) { return; }
    let runtime=crate::driver::export_checkpoint(backend.state.time).unwrap();
    let curves=runtime.restoring.as_ref().expect("cross link history");
    assert!(curves.history.iter().any(|h|h.initialized!=0 && (h.q0!=0. || h.d0.iter().any(|v|*v!=0.))),"cross link history must be nonzero");
    backend.fetch_state(data,param);
    let free_vertex=data.mesh.mesh.tet.as_slice().iter().zip(data.prop.tet.as_slice())
        .find(|(_,p)|data.param_arrays.tet.as_slice()[p.param_index as usize].model==crate::data::Model::SNHk)
        .expect("ordinary SNHK tet").0[0];
    CAPTURE.with(|c|*c.borrow_mut()=Some(Captured{frame:backend.state.curr_frame,time:backend.state.time,prev_dt:param.prev_dt,
        curr:backend.state.curr_vertex.as_slice().to_vec(),prev:backend.state.prev_vertex.as_slice().to_vec(),runtime,material,
        adaptive:backend.checkpoint_adaptive.clone().expect("accepted controller"),accepted:backend.checkpoint_accepted,adaptive_start:backend.checkpoint_adaptive_start,free_vertex,rejected:false,save_seen:false,advances:0,cancels:0,finishes:0}));
}
pub(super) fn cap_if_armed(token:u64,param:&mut ParamSet)->Option<u32> {
    CAPTURE.with(|c| {let mut c=c.borrow_mut();let c=c.as_mut()?;assert!(!c.rejected,"unexpected additional advance after refusal");
        assert_eq!(token,0);c.advances+=1;let old=param.max_newton_steps;param.max_newton_steps=1;Some(old)})
}
pub(super) fn override_prev_if_armed(token:u64,_h:f32,_data:&DataSet)->Result<(),crate::driver::HostFatal> {
    let sentinel=CAPTURE.with(|c|c.borrow().as_ref().filter(|_|token==0).map(|x|(x.free_vertex,x.prev[x.free_vertex as usize*3])));
    if let Some((vertex,accepted))=sentinel {
        let requested=accepted+0.03125;
        assert_ne!(requested.to_bits(),accepted.to_bits(),"sentinel must differ from accepted prev");
        let actual=crate::driver::private_write_trial_prev(vertex,requested)?;
        assert_eq!(actual.to_bits(),requested.to_bits(),"device trial prev sentinel readback");
    } Ok(())
}
pub(super) fn after_advance(old:Option<u32>,param:&mut ParamSet,outcome:&Result<crate::driver::HostAdvanceOutcome,crate::driver::HostFatal>,path:&Path) {
    if let Some(old)=old {param.max_newton_steps=old;
        assert!(matches!(outcome,Ok(crate::driver::HostAdvanceOutcome::NewtonCapRejected{continuation,..}) if *continuation!=0),"forced cap must produce real Newton rejection");
        CAPTURE.with(|c|c.borrow_mut().as_mut().unwrap().rejected=true);
        std::fs::write(path,b"").expect("create save request after real refusal");
    }
}
pub(super) fn observe_cancel() { CAPTURE.with(|c|if let Some(c)=c.borrow_mut().as_mut(){if c.save_seen {c.cancels+=1;}}); }
pub(super) fn observe_finish() { CAPTURE.with(|c|if let Some(c)=c.borrow_mut().as_mut(){if c.save_seen {c.finishes+=1;}}); }
pub(super) fn observe_save_request() { if !ENABLED.with(|e|e.get()){return;} CAPTURE.with(|c|{let mut c=c.borrow_mut();let c=c.as_mut().expect("captured accepted state");assert!(c.rejected);assert!(!c.save_seen);c.save_seen=true;}); }

fn assert_bits(name:&str,a:&[f32],b:&[f32]) {assert_eq!(a.len(),b.len(),"{name} length");if let Some(i)=a.iter().zip(b).position(|(x,y)|x.to_bits()!=y.to_bits()){panic!("{name} first difference at {i}: {:08x} != {:08x}",a[i].to_bits(),b[i].to_bits());}}
#[test]
#[ignore="requires explicit private 05 input and fresh output parent"]
fn root_newton_rejection_saves_real_accepted_checkpoint() {
    assert!(adaptive_cg_retry_enabled(),"set PPF_ADAPTIVE_CG_RETRY=1");
    let path=std::env::var("PPF_PRIVATE_NEWTON_SCENE").expect("explicit 05 session");
    let parent=std::path::PathBuf::from(std::env::var("PPF_PRIVATE_NEWTON_SAVE_OUTPUT").expect("private output parent"));
    let nonce=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let output=parent.join(format!("backend_save_{}_{}",std::process::id(),nonce));
    assert!(!std::fs::canonicalize(&parent).expect("private output parent must already exist").starts_with(std::fs::canonicalize(&path).unwrap()),"output parent must be outside source session");
    std::fs::create_dir(&output).unwrap();
    let args=ProgramArgs::try_parse_from(["test","--path",&path,"--output",output.to_str().unwrap()]).unwrap();
    crate::begin_output(&args);
    let mut scene=Scene::new(&args);let sim_args=scene.args();
    let mut backend=Backend::new(scene.make_mesh());
    backend.material_spin_overlay=Some(crate::twist_overlay::material_spin_from_validated(scene.joint_input.as_ref().unwrap().records()).unwrap());
    let data=crate::fresh_authority_dataset(&mut scene);
    let prepared=crate::restoring_curve::prepare_fresh(Path::new(&path),args.require_restoring_curves_v1,0).unwrap();
    let curves=prepared.as_ref().map(|p|crate::restoring_native::verify_and_pack(p,&scene,&data,Path::new(&path)).unwrap());
    let curves=crate::cross_links::merge_curves(scene.joint_input.as_ref(),curves,&scene,Some(&data)).unwrap();
    let param=crate::builder::make_param(&sim_args);
    ENABLED.with(|e|e.set(true));
    backend.run(&args,&sim_args,data,param,scene,curves.as_ref(),prepared.as_ref());
    assert!(!output.join("save_and_quit").exists(),"save request marker must be consumed");
    let status=ppf_cts_formats::status::read(&output).unwrap().expect("saved status");
    assert!(matches!(status.outcome,Some(ppf_cts_formats::status::Outcome::SavedAndQuit)));
    CAPTURE.with(|cell|{let c=cell.borrow();let c=c.as_ref().expect("warmup must reach nonzero material history");
        assert!(c.rejected && c.save_seen);assert_eq!(c.advances,1);assert_eq!(c.cancels,1);assert_eq!(c.finishes,1);
        eprintln!("PRIVATE_SAVE_OUTPUT={} FRAME={} TIME={}",output.display(),c.frame,c.time);
        let loaded=Backend::load_state(c.frame,output.to_str().unwrap());
        assert_eq!(loaded.state.curr_frame,c.frame);assert_eq!(loaded.state.time.to_bits(),c.time.to_bits());assert_eq!(loaded.state.prev_dt.to_bits(),c.prev_dt.to_bits());
        assert_bits("curr",&c.curr,loaded.state.curr_vertex.as_slice());assert_bits("prev",&c.prev,loaded.state.prev_vertex.as_slice());
        let payload=crate::checkpoint_history::read_versioned(&output,c.frame,true).unwrap().unwrap();
        assert_eq!(payload.legacy().accepted_time.to_bits(),c.time.to_bits());
        assert_eq!(payload.legacy().joint_accepted,c.accepted);assert_eq!(payload.legacy().adaptive,c.adaptive);assert_eq!(payload.legacy().adaptive_start_time.to_bits(),c.adaptive_start.to_bits());
        assert!(bincode::serialize(&payload.legacy().runtime).unwrap()==bincode::serialize(&c.runtime).unwrap(),"root/curve and running rotation history mismatch");
        let crate::checkpoint_history::HistoryPayload::V3(v3)=payload else {panic!("expected material history V3")};
        assert!(v3.rigid_material.as_ref()==Some(&c.material),"accepted material history mismatch");
    });
}
