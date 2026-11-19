//! DRAFT: copy as driver/host_plan_identity_test.rs; not compiled or run.
//! Run this ignored test alone: the native driver owns process-global state.
use clap::Parser;

#[test]
#[ignore = "requires explicit private native scene plus output/dataset.bin.gz"]
fn host_prepared_plan_rejects_wrong_identity_before_device_advance() {
    super::test_device_fault::start_private_status();
    let path = std::env::var("PPF_PRIVATE_C4_SCENE").expect("private scene path");
    let output = format!("{path}/output");
    let args = crate::args::ProgramArgs::try_parse_from(["test", "--path", &path, "--output", &output]).unwrap();
    let scene = crate::scene::Scene::new(&args);
    let raw = crate::read_gz(&format!("{output}/dataset.bin.gz"));
    let data: crate::data::DataSet = crate::read(&raw);
    let mut param = crate::builder::make_param(&scene.args());
    param.playback = 1.0;
    param.dt = 0.05;
    assert!(unsafe { super::initialize(&data, &param) });
    let owned=scene.joint_input.as_ref().expect("private roots").records().clone();
    super::install_joints_from_rest(&owned,None);
    let token = 129u64;
    {
        let mut guard = super::SOLVER.lock().unwrap();
        let state = guard.as_mut().unwrap();
        assert!(!state.roots.has_control_events());
        assert!(state.fem_roots.runtime.is_none());
        state.host_retry_pending = Some(super::HostRetryPending {
            token, dataset: (&data as *const _) as usize, start_bits: param.time.to_bits(),
            failed_h_bits: 0.1f32.to_bits(), requires_prepared: true, swing_route: false, newton_root_route: false, root_generation: None, material_generation: None,
        });
    }
    let plan = unsafe { super::prepare_host_plan(token, param.dt) }.unwrap();
    assert_eq!(plan.plan.start_s.to_bits(), param.time.to_bits());
    assert!(plan.end_s() > param.time);
    assert!(plan.integration_h() > 0.0 && plan.integration_h() < param.dt);
    assert!((plan.end_s()-1.0/24.0).abs()<1e-12);
    assert!(unsafe { super::prepare_host_plan(token + 1, param.dt) }.is_err());
    assert!(unsafe { super::prepare_host_plan(token, 0.009) }.is_err());
    assert!(unsafe { super::advance_host(token + 1, 16, true, true, Some(plan)) }.is_err());
    assert!(unsafe { super::advance_host(token, 16, true, true, None) }.is_err());
    let mut wrong = plan; wrong.dataset ^= 1;
    assert!(unsafe { super::advance_host(token, 16, true, true, Some(wrong)) }.is_err());
    let mut wrong = plan; wrong.start_bits ^= 1;
    assert!(unsafe { super::advance_host(token, 16, true, true, Some(wrong)) }.is_err());
    let mut wrong = plan; wrong.continuation += 1;
    assert!(unsafe { super::advance_host(token, 16, true, true, Some(wrong)) }.is_err());
    let mut wrong = plan; wrong.requested_bits ^= 1;
    assert!(unsafe { super::advance_host(token, 16, true, true, Some(wrong)) }.is_err());
    assert!(super::cancel_host_retry(token).is_ok());
}

