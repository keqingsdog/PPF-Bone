// Isolated test-only scheduling barriers. No solver or upload operations.
use super::*;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use tokio::sync::Notify;
use ppf_cts_core::effects::Effect;
use crate::executor::DefaultExecutor;

struct PauseFirstLoad {
    arrived: Notify,
    release: Notify,
    paused: AtomicBool,
}
#[async_trait::async_trait]
impl EffectExecutor for PauseFirstLoad {
    async fn execute(&self, effect: Effect, engine: &ServerEngine) {
        assert!(matches!(effect, Effect::DoLoadApp { .. } | Effect::DoLog { .. }),
                "unexpected action {effect:?} for {}", engine.state().name);
        if matches!(&effect, Effect::DoLoadApp { name, .. } if name == "schedule_a")
            && !self.paused.swap(true, Ordering::SeqCst) {
            self.arrived.notify_one();
            self.release.notified().await;
        }
        DefaultExecutor::new().execute(effect, engine).await;
    }
}
async fn status(name: &str, engine: &ServerEngine, executor: &dyn EffectExecutor) -> Value {
    tcmd(&format!("--name {name}"), engine, executor).await
}

async fn tcmd(request: &str, engine: &ServerEngine, executor: &dyn EffectExecutor) -> Value {
    let payload = request.as_bytes();
    let mut bytes = (payload.len() as u32).to_be_bytes().to_vec();
    bytes.extend(payload);
    let mut reader = bytes.as_slice();
    let mut writer = Vec::<u8>::new();
    handle_tcmd(&mut reader, &mut writer, engine, executor, "127.0.0.1:12345".parse().unwrap()).await.unwrap();
    serde_json::from_slice(&writer).unwrap()
}
fn fixture() -> (tempfile::TempDir, ServerEngine) {
    let tmp = tempfile::tempdir().unwrap();
    let engine = ServerEngine::new(crate::config::EngineConfig { data_root: Some(tmp.path().to_path_buf()), ..Default::default() });
    for (name, marker) in [("schedule_a", "a"), ("schedule_b", "b")] {
        let root = crate::upload::make_root(name, engine.config());
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("data_hash.txt"), format!("data-{marker}")).unwrap();
        std::fs::write(root.join("param_hash.txt"), format!("param-{marker}")).unwrap();
    }
    (tmp, engine)
}
#[tokio::test(flavor="multi_thread", worker_threads=2)]
async fn isolated_context_schedule_late_load() {
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        let (_tmp, engine) = fixture();
        let executor = Arc::new(PauseFirstLoad { arrived: Notify::new(), release: Notify::new(), paused: AtomicBool::new(false) });
        let e = engine.clone(); let x = executor.clone();
        let first = tokio::spawn(async move { status("schedule_a", &e, x.as_ref()).await });
        executor.arrived.notified().await;
        let second = status("schedule_b", &engine, executor.as_ref()).await;
        assert!(second["root"].as_str().unwrap().ends_with("schedule_b"));
        assert_eq!(second["data_hash"], "data-b"); assert_eq!(second["param_hash"], "param-b");
        executor.release.notify_one();
        let first = first.await.unwrap();
        println!("CONTROLLED_CONTEXT first={first} second={second}");
        assert_eq!(first["error"], "PROJECT_CONTEXT_CHANGED");
        assert_eq!(first.as_object().unwrap().len(), 2);
        let state=engine.state();assert_eq!(state.name,"schedule_b");assert_eq!(state.data_hash,"data-b");assert_eq!(state.param_hash,"param-b");
    }).await.expect("controlled test deadlocked");
}

fn make_ready_for_action(name: &str, engine: &ServerEngine) {
    let project = crate::upload::make_root(name, engine.config());
    let output = ppf_cts_formats::files::session_output_dir(&project);
    std::fs::create_dir_all(&output).unwrap();
    for filename in [DATA_PICKLE, PARAM_PICKLE, "app_state.pickle"] {
        std::fs::write(project.join(filename), []).unwrap();
    }
    std::fs::write(output.join("state_1.bin.gz"), []).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn isolated_context_schedule_stale_build_run_and_resume() {
    use ppf_cts_core::state::{Build, Data};

    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        for request in ["build", "start", "resume --resume_from 1"] {
            for reselect_first in [false, true] {
                let (_tmp, engine) = fixture();
                make_ready_for_action("schedule_b", &engine);
                let executor = Arc::new(PauseFirstLoad {
                    arrived: Notify::new(),
                    release: Notify::new(),
                    paused: AtomicBool::new(false),
                });
                let first_request = format!("--name schedule_a --request {request}");
                let first_engine = engine.clone();
                let first_executor = executor.clone();
                let first = tokio::spawn(async move {
                    tcmd(&first_request, &first_engine, first_executor.as_ref()).await
                });
                executor.arrived.notified().await;
                let original = engine.state();
                assert_eq!(original.name, "schedule_a");

                status("schedule_b", &engine, executor.as_ref()).await;
                if reselect_first {
                    make_ready_for_action("schedule_a", &engine);
                    status("schedule_a", &engine, executor.as_ref()).await;
                }
                let selected = engine.state();
                assert_eq!(selected.name, if reselect_first { "schedule_a" } else { "schedule_b" });
                assert!(selected.project_generation > original.project_generation);
                assert_eq!(selected.data, Data::Uploaded);
                assert_eq!(selected.build, Build::Built);
                assert!(selected.resumable);

                // The fake executor rejects any build or launch effect. The
                // target now selected is otherwise ready to accept each action.
                executor.release.notify_one();
                let response = first.await.unwrap();
                assert_eq!(response["error"], "PROJECT_CONTEXT_CHANGED", "{request}: {response}");
                assert_eq!(response.as_object().unwrap().len(), 2, "{request}: {response}");
                assert_eq!(engine.state(), selected, "{request} modified the selected build");
            }
        }
    })
    .await
    .expect("controlled action test deadlocked");
}

#[tokio::test]
async fn isolated_context_schedule_late_build_effect_cannot_start_for_successor() {
    for reselect_first in [false, true] {
        let (_tmp, engine) = fixture();
        let executor = DefaultExecutor::new();
        make_ready_for_action("schedule_a", &engine);
        make_ready_for_action("schedule_b", &engine);
        status("schedule_a", &engine, &executor).await;
        let stale_effects = engine.dispatch(Event::BuildRequested { preserve_output: false });
        assert!(matches!(stale_effects.as_slice(), [Effect::DoSpawnBuild { .. }]));

        status("schedule_b", &engine, &executor).await;
        if reselect_first {
            status("schedule_a", &engine, &executor).await;
        }
        let selected_effects = engine.dispatch(Event::BuildRequested { preserve_output: false });
        assert!(matches!(selected_effects.as_slice(), [Effect::DoSpawnBuild { .. }]));
        let active = engine.state();
        let cancel = engine.install_cancel_handle();

        for effect in stale_effects {
            executor.execute(effect, &engine).await;
        }
        engine.cancel_active_build();
        assert!(cancel.is_cancelled(), "stale effect replaced the successor's cancel token");
        assert_eq!(engine.state(), active);
    }
}

#[tokio::test]
async fn isolated_context_schedule_new_upload_can_retry_after_stale_build() {
    use ppf_cts_core::state::Build;

    let (_tmp, engine) = fixture();
    let executor = DefaultExecutor::new();
    make_ready_for_action("schedule_a", &engine);
    status("schedule_a", &engine, &executor).await;
    let stale_effects = engine.dispatch(Event::BuildRequested { preserve_output: false });
    assert!(matches!(stale_effects.as_slice(), [Effect::DoSpawnBuild { .. }]));

    engine.dispatch(Event::upload_landed("new-upload"));
    let uploaded = engine.state();
    let sidecar = std::path::Path::new(&uploaded.root).join("build_violations.json");
    std::fs::write(&sidecar, b"new upload marker").unwrap();
    for effect in stale_effects {
        executor.execute(effect, &engine).await;
    }

    let retired = engine.state();
    assert_eq!(retired.build, Build::Failed);
    assert_eq!(retired.upload_id, uploaded.upload_id);
    assert!(retired.error.starts_with("BUILD_INPUTS_CHANGED:"));
    assert_eq!(std::fs::read(sidecar).unwrap(), b"new upload marker");
    let retry = engine.dispatch(Event::BuildRequested { preserve_output: false });
    assert!(matches!(retry.as_slice(), [Effect::DoSpawnBuild { .. }]));
}

#[tokio::test]
async fn isolated_context_schedule_new_upload_retires_pending_run_and_resume() {
    use ppf_cts_core::state::Solver;

    for request in [Event::StartRequested, Event::ResumeRequested { from_frame: Some(1) }] {
        let (_tmp, engine) = fixture();
        let executor = DefaultExecutor::new();
        make_ready_for_action("schedule_a", &engine);
        status("schedule_a", &engine, &executor).await;
        let effects = engine.dispatch(request);
        let generation = effects.iter().find_map(|effect| match effect {
            Effect::DoLaunchSolver { generation, .. } => Some(*generation),
            _ => None,
        }).unwrap();
        assert!(engine.pending_run(generation).is_some());

        engine.dispatch(Event::upload_landed("new-upload"));
        let uploaded = engine.state();
        assert!(engine.pending_run(generation).is_none());
        assert_eq!(uploaded.solver, Solver::Idle);
        assert_eq!(uploaded.upload_id, "new-upload");
    }
}

fn select(name: &str, root: &str) -> Event {
    Event::ProjectSelected { name: name.into(), root: root.into(), has_data: false, has_param: false, has_app: false, is_resumable: false, crash: None, upload_id: String::new(), data_hash: format!("data-{name}"), param_hash: format!("param-{name}"), total_frames: 0, exemptions: vec![] }
}
#[test]
fn isolated_scope_failure_current_stale_aba_and_root_change() {
    let engine=ServerEngine::new(Default::default());
    engine.dispatch(select("a", "/scope/a"));
    let old=engine.state();
    let failure=Event::ProjectLoadFailed { name:old.name.clone(), root:old.root.clone(), generation:old.project_generation, error:"private-a-path".into() };
    engine.dispatch(select("a", "/scope/a"));
    assert_eq!(engine.state().project_generation, old.project_generation, "ordinary refresh keeps active context");
    let fx=engine.dispatch(failure.clone()); assert!(!fx.is_empty()); assert_eq!(engine.state().error,"private-a-path");
    engine.dispatch(select("b", "/scope/b")); let b=engine.state();
    assert!(engine.dispatch(failure.clone()).is_empty()); assert_eq!(engine.state(),b);
    engine.dispatch(select("a", "/scope/a")); let new_a=engine.state();
    assert!(new_a.project_generation>old.project_generation);
    assert!(engine.dispatch(failure.clone()).is_empty()); assert_eq!(engine.state(),new_a,"ABA cannot restore old error authority");
    engine.dispatch(select("a", "/scope/a-new-root"));let other_root=engine.state();
    assert!(other_root.project_generation>new_a.project_generation);
    assert!(engine.dispatch(failure).is_empty());assert_eq!(engine.state(),other_root);
    engine.set_project_context("manual", "/scope/manual");let manual=engine.state();
    assert!(manual.project_generation>other_root.project_generation);
}
#[test]
fn isolated_scope_violations_and_running_refresh() {
    use ppf_cts_core::{state::{ServerState,Build,Solver},transitions::transition};
    let (a,_)=transition(ServerState::default(),select("a","/scope/a"));
    let active=ServerState { build:Build::Building,solver:Solver::Running,frame:27,violations:vec!["private-a-violation".into()],error:"private-a-error".into(),..a };
    let (refreshed,fx)=transition(active.clone(),select("a","/scope/a"));
    assert!(fx.is_empty());assert_eq!(refreshed.build,active.build);assert_eq!(refreshed.solver,active.solver);assert_eq!(refreshed.frame,active.frame);assert_eq!(refreshed.violations,active.violations);assert_eq!(refreshed.project_generation,active.project_generation);
    let (b,_)=transition(active.clone(),select("b","/scope/b"));assert!(b.violations.is_empty());assert!(b.error.is_empty());
    let (other_root,_)=transition(active,select("a","/scope/other"));assert!(other_root.violations.is_empty());assert!(other_root.error.is_empty());
    let value=serde_json::to_value(&refreshed).unwrap();assert!(value.get("project_generation").is_none(),"generation is not wire or persisted data");
}
#[tokio::test]
async fn isolated_scope_real_mkdir_failure_and_late_effect() {
    let tmp=tempfile::tempdir().unwrap();let blocked=tmp.path().join("plain-file");std::fs::write(&blocked,b"fixture").unwrap();
    let root=blocked.join("child").to_string_lossy().into_owned();
    let engine=ServerEngine::new(Default::default());let executor=DefaultExecutor::new();
    let effects=engine.dispatch(select("bad",&root));assert_eq!(effects.len(),1);
    for effect in effects { executor.execute(effect,&engine).await; }
    assert!(engine.state().error.starts_with("project dir create:"),"current mkdir error preserves prior semantics");
    let effects=engine.dispatch(select("a",&root));
    engine.dispatch(select("b",tmp.path().to_str().unwrap()));let before=engine.state();
    for effect in effects { executor.execute(effect,&engine).await; }
    assert_eq!(engine.state(),before,"late failed effect cannot publish error or identity into B");
    let effects=engine.dispatch(select("a",&root));
    engine.dispatch(select("b",tmp.path().to_str().unwrap()));engine.dispatch(select("a",&root));let before=engine.state();
    for effect in effects { executor.execute(effect,&engine).await; }
    assert_eq!(engine.state(),before,"ABA old effect cannot publish failure into new A");
    assert_eq!(std::fs::read(&blocked).unwrap(),b"fixture");
}
#[test]
fn isolated_scope_delete_transition_never_reuses_generation() {
    use ppf_cts_core::{state::ServerState,transitions::transition};
    let (a,_)=transition(ServerState::default(),select("a","/scope/a"));let old=a.project_generation;
    let (deleted,_unexecuted_effects)=transition(a,Event::DeleteRequested);
    assert!(deleted.project_generation>old);
    let (again,_)=transition(deleted,select("a","/scope/a"));assert!(again.project_generation>old);
    // Only pure transitions were evaluated. Delete/kill effects are never executed.
}


#[cfg(windows)]
#[tokio::test]
async fn delete_ack_reports_locked_partial_directory_and_allows_retry() {
    use std::os::windows::fs::OpenOptionsExt;

    let tmp = tempfile::tempdir().unwrap();
    let engine = ServerEngine::new(crate::config::EngineConfig {
        data_root: Some(tmp.path().to_path_buf()),
        ..Default::default()
    });
    let executor = DefaultExecutor::new();
    let root = crate::upload::make_root("partial_delete", engine.config());
    let output = root.join("session").join("output");
    std::fs::create_dir_all(&output).unwrap();
    let locked_path = output.join("locked.bin");
    std::fs::write(&locked_path, b"keep").unwrap();
    // Simulates a partial delete whose scene inputs are already gone while
    // a Windows process still holds a remaining cache file without delete sharing.
    let locked = std::fs::OpenOptions::new().read(true).share_mode(0).open(&locked_path).unwrap();

    async fn command(engine: &ServerEngine, executor: &dyn EffectExecutor, request: &str) -> Value {
        let payload = format!("--name partial_delete --request {request}").into_bytes();
        let mut bytes = (payload.len() as u32).to_be_bytes().to_vec();
        bytes.extend(payload);
        let mut reader = bytes.as_slice();
        let mut writer = Vec::<u8>::new();
        handle_tcmd(&mut reader, &mut writer, engine, executor,
                    "127.0.0.1:12345".parse().unwrap()).await.unwrap();
        serde_json::from_slice(&writer).unwrap()
    }

    let selected = status("partial_delete", &engine, &executor).await;
    assert_eq!(selected["data"], "NO_DATA");
    let failed = command(&engine, &executor, "delete").await;
    assert!(failed["error"].as_str().unwrap().starts_with("DELETE_IO_FAILED:"));
    assert!(locked_path.exists());
    let reselected = status("partial_delete", &engine, &executor).await;
    assert_eq!(reselected["data"], "NO_DATA");
    assert_eq!(reselected["root"].as_str(), Some(root.to_str().unwrap()));

    drop(locked);
    let retried = command(&engine, &executor, "delete").await;
    assert_eq!(retried["error"], "");
    assert_eq!(retried["root"], "");
    assert!(!root.exists());
}

#[tokio::test]
async fn retired_delete_effect_cannot_touch_reselected_project() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = ServerEngine::new(crate::config::EngineConfig {
        data_root: Some(tmp.path().to_path_buf()),
        ..Default::default()
    });
    let executor = DefaultExecutor::new();
    let root = crate::upload::make_root("same_project", engine.config());
    std::fs::create_dir_all(&root).unwrap();
    let marker = root.join("keep.bin");
    std::fs::write(&marker, b"keep").unwrap();
    status("same_project", &engine, &executor).await;
    let effects = engine.dispatch(Event::DeleteRequested);
    // Re-select even the same project before the old effect is executed.
    status("same_project", &engine, &executor).await;
    for effect in effects {
        executor.execute(effect, &engine).await;
    }
    assert_eq!(std::fs::read(&marker).unwrap(), b"keep");
    assert!(engine.state().error.is_empty());
}


#[tokio::test]
async fn delete_with_resume_target_is_rejected_without_removing_data() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = ServerEngine::new(crate::config::EngineConfig {
        data_root: Some(tmp.path().to_path_buf()),
        ..Default::default()
    });
    let executor = DefaultExecutor::new();
    let root = crate::upload::make_root("invalid_delete", engine.config());
    std::fs::create_dir_all(&root).unwrap();
    let marker = root.join("keep.bin");
    std::fs::write(&marker, b"keep").unwrap();
    let payload = b"--name invalid_delete --request delete --target_frame 99";
    let mut bytes = (payload.len() as u32).to_be_bytes().to_vec();
    bytes.extend(payload);
    let mut reader = bytes.as_slice();
    let mut writer = Vec::<u8>::new();
    handle_tcmd(&mut reader, &mut writer, &engine, &executor,
                "127.0.0.1:12345".parse().unwrap()).await.unwrap();
    let response: Value = serde_json::from_slice(&writer).unwrap();
    assert!(response["error"].as_str().unwrap().starts_with("RESUME_TARGET_INVALID"));
    assert_eq!(std::fs::read(&marker).unwrap(), b"keep");
}
