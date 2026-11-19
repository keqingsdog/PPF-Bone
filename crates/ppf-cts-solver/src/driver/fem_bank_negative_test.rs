//! DRAFT ONLY. Integration requires the reviewed, cfg(test,cpu)-only hooks
//! specified in fem_bank_negative_hooks.rs; this file is not compiled yet.
//! Run each ignored test in its own fresh process with the *full* 03 scene.
use clap::Parser;

fn run_pre_b32_case(mode: super::soft_joints::private_pre_b32::Mode) {
    use super::soft_joints::private_pre_b32::{self as hook, Mode};
    super::test_device_fault::start_private_status();
    let path = std::env::var("PPF_PRIVATE_FEM_SCENE").expect("explicit full authoring 03 scene");
    let output = format!("{path}/output");
    let args = crate::args::ProgramArgs::try_parse_from(["test", "--path", &path, "--output", &output]).unwrap();
    let mut scene = crate::scene::Scene::new(&args);
    let input = scene.joint_input.as_ref().expect("03 joint input");
    let owned = input.records().clone();
    let soft = crate::cross_links::prepare_batch(input, scene.args().world_scaling as f64)
        .unwrap().expect("03 soft joint batch");
    let data = crate::fresh_authority_dataset(&mut scene);
    let mut param = crate::builder::make_param(&scene.args());
    param.playback = 1.;
    assert!(unsafe { super::initialize(&data, &param) });
    super::install_joints_from_rest(&owned, None);
    super::install_soft_joints(&soft).unwrap();
    {
        let guard = super::SOLVER.lock().unwrap();
        let state = guard.as_ref().unwrap();
        assert!(state.joints.soft.active() && state.joints.twist_selection().is_none());
        assert!(!state.hard.active() && !state.roots.rotating()
            && !state.roots.has_control_events() && state.fem_roots.runtime.is_none());
    }
    assert!(scene.rest_shape_schedule().is_none() && !scene.has_material_animation());
    assert!(!scene.build_collision_window_table().has_windows);
    super::host_seed_capture().unwrap();
    let observed = crate::backend::private_prepare_03(&scene, param.time, param.dt, param.playback, None).unwrap();
    assert!(observed.iter().any(|x| x.0 == "constraint_fix" && x.1));

    // Same complete snapshot used by the existing passing Legacy transaction
    // test: both position buffers, fix, rigid, running R/seed, omega, roots,
    // curves, FEM counters/end and commit counters. Device data is read back.
    let snapshot = || {
        let mut guard = super::SOLVER.lock().unwrap();
        let state = guard.as_mut().unwrap();
        super::with_device(|d| {
            state.positions.download(d)?; state.positions_prev.download(d)?;
            let pos = state.positions.host().iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            let prev = state.positions_prev.host().iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            let mut fix = vec![crate::data::FixPair::default(); state.fix.len()];
            state.fix.read(d, 0, &mut fix)?;
            let mut rigid = vec![crate::data::PdrdRigidState::default(); state.rigid_state.len()];
            state.rigid_state.read(d, 0, &mut rigid)?;
            let rigid = rigid.iter().map(|r| (r.anchor.map(f32::to_bits), r.x.map(f32::to_bits),
                r.rotation.map(f32::to_bits), r.reference_inertia.map(f32::to_bits),
                r.mass_total.to_bits(), r.count)).collect::<Vec<_>>();
            let mut running = vec![0f32; state.rigid_running_rotation.len()];
            state.rigid_running_rotation.read(d, 0, &mut running)?;
            let mut omega = vec![0f32; state.grain_omega.len()]; state.grain_omega.read(d, 0, &mut omega)?;
            let root = state.roots.transaction.as_ref().map(|r| bincode::serialize(&r.export_checkpoint().unwrap()).unwrap());
            let curves = state.joints.curves.snapshot_trial(d)?;
            Ok((pos, prev, bincode::serialize(&fix).unwrap(), rigid,
                running.into_iter().map(f32::to_bits).collect::<Vec<_>>(), state.rigid_rotation_seeded,
                omega.into_iter().map(f32::to_bits).collect::<Vec<_>>(), root,
                bincode::serialize(&curves).unwrap(), state.fem_roots.accepted_steps,
                state.fem_roots.prepared_end, state.restoring_commit_count, state.restoring_commit_time.map(f64::to_bits)))
        }).unwrap()
    };
    let before = snapshot();
    let clock = (param.time.to_bits(), param.time_f32.to_bits(), param.prev_dt.to_bits(), param.dt.to_bits());
    let armed = hook::Armed::new(mode);
    let error = match unsafe { super::advance_host(0, 16, true, true, None) } {
        Err(error) => error,
        Ok(_) => panic!("pre-B32 bank fault must remain Fatal, without continuation"),
    };
    match mode {
        Mode::NonzeroBank => assert!(error.detail.contains("before candidate material frame prepare")
            && error.detail.contains("row 1") && error.detail.contains("native fault 259")),
        Mode::ReadbackError => assert!(error.detail.contains("private_pre_b32_download")),
    }
    assert_eq!(hook::hits(), (1, 0, 0), "candidate reap entered once; no prepare dispatch or qualification");
    {
        let guard = super::SOLVER.lock().unwrap();
        let state = guard.as_ref().unwrap();
        assert!(state.host_retry_pending.is_none());
        assert!(state.joints.soft.candidate_rejection().is_none());
        assert!(state.joints.candidate_retry_dt().is_none());
    }
    assert_eq!(clock, (param.time.to_bits(), param.time_f32.to_bits(), param.prev_dt.to_bits(), param.dt.to_bits()));
    let after = snapshot();
    {
        let mut guard = super::SOLVER.lock().unwrap();
        let state = guard.as_mut().unwrap();
        super::with_device(|device| {
            state.eval_x.download(device)?;
            assert_eq!(state.eval_x.host().iter().map(|v| v.to_bits()).collect::<Vec<_>>(), after.0);
            Ok(())
        }).unwrap();
    }
    assert_eq!(before.0, after.0); assert_eq!(before.1, after.1); assert_eq!(before.2, after.2);
    assert_eq!(before.3, after.3); assert_eq!(before.4, after.4); assert_eq!(before.5, after.5);
    assert_eq!(before.6, after.6); assert_eq!(before.7, after.7); assert_eq!(before.8, after.8);
    assert_eq!(before.9, after.9); assert_eq!(before.10, after.10); assert_eq!(before.11, after.11);
    assert_eq!(before.12, after.12);
    drop(armed);
    super::host_seed_discard().unwrap();
}

#[test]
#[ignore = "requires full private 03 scene; run this test alone in a fresh process"]
fn actual_pre_b32_nonzero_device_bank_stays_fatal() {
    run_pre_b32_case(super::soft_joints::private_pre_b32::Mode::NonzeroBank);
}

#[test]
#[ignore = "requires full private 03 scene; run this test alone in a fresh process"]
fn actual_pre_b32_readback_error_stays_fatal() {
    run_pre_b32_case(super::soft_joints::private_pre_b32::Mode::ReadbackError);
}

