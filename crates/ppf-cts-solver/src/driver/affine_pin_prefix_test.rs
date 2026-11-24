use clap::Parser;

#[test]
#[ignore = "requires an explicit native scene with ordinary affine fixed pins"]
fn affine_prefix_commits_actual_device_pose_and_clock() {
    super::test_device_fault::start_private_status();
    let path = std::env::var("PPF_PRIVATE_AFFINE_PIN_SCENE").expect("explicit scene path");
    let output = std::env::var("PPF_PRIVATE_AFFINE_PIN_OUTPUT").expect("isolated test output");
    std::fs::create_dir_all(&output).unwrap();
    let args = crate::args::ProgramArgs::try_parse_from([
        "test", "--path", &path, "--output", &output,
    ]).unwrap();
    let mut scene = crate::scene::Scene::new(&args);
    let data = crate::fresh_authority_dataset(&mut scene);
    let mut param = crate::builder::make_param(&scene.args());
    let start = param.time;
    assert!(unsafe { super::initialize(&data, &raw mut param) });
    assert!(super::host_kinematics_retry_scope(param.playback).unwrap());
    let plan = unsafe { super::prepare_kinematics_plan(0, param.dt, None) }.unwrap();
    let pins = scene.affine_pin_prefix(start, plan.end_s(), false).unwrap();
    let full = scene.make_constraint_with_membership(plan.end_s(), start, true);
    unsafe { super::host_update_constraint(&full) }.unwrap();

    super::step::PRIVATE_TEST_AFFINE_B3.with(|armed| armed.set(true));
    let outcome = unsafe { super::advance_host_kinematics(0, 16, false, Some(plan), Some(&pins)) }.unwrap();
    let super::HostAdvanceOutcome::Accepted(result) = outcome else {
        panic!("shortened affine trial must accept without a host replay");
    };
    assert!(result.success());
    assert!(start < result.time && result.time < plan.end_s());
    assert_eq!(param.prev_dt, ((result.time - start) * f64::from(param.playback)) as f32);
    let expected = pins.sample_fix(start, result.time);
    let mut solver = super::SOLVER.lock().unwrap();
    let state = solver.as_mut().unwrap();
    super::with_device(|device| {
        state.positions.download(device)?;
        let actual = state.positions.host();
        let mut moving = 0;
        for (pin, original) in expected.iter().zip(full.fix.as_slice()) {
            if pin.position == original.position {
                continue;
            }
            moving += 1;
            for axis in 0..3 {
                let authored = pin.position[axis];
                let tolerance = 2.0 * f32::EPSILON * authored.abs().max(1.0);
                assert!((actual[3 * pin.index as usize + axis] - authored).abs() <= tolerance);
            }
        }
        assert!(moving > 0, "the fixture must drive ordinary fixed positions");
        Ok(())
    }).unwrap();
    drop(solver);

    // A cut of the final free-vertex correction is safe when all effective
    // hard rows already land on the exact author sample. The motion-clock
    // cut is inside the same f32 integration cell and the pin positions at
    // either end agree in f32. The fixture runs the real C1/C2 and C5.
    let start = param.time;
    let plan = unsafe { super::prepare_kinematics_plan(0, param.dt, None) }.unwrap();
    let pins = scene.affine_pin_prefix(start, plan.end_s(), false).unwrap();
    let full = scene.make_constraint_with_membership(plan.end_s(), start, true);
    unsafe { super::host_update_constraint(&full) }.unwrap();
    super::step::PRIVATE_TEST_AFFINE_CLIPPED_ENABLED.with(|value| value.set(Some(true)));
    super::step::PRIVATE_TEST_AFFINE_MESH_CUT.with(|armed| armed.set(Some(1.0 - 1.0e-10)));
    let accepted = unsafe { super::advance_host_kinematics(0, 16, false, Some(plan), Some(&pins)) }.unwrap();
    let super::HostAdvanceOutcome::Accepted(result) = accepted else {
        panic!("certified clipped free correction at exact hard endpoints must commit");
    };
    assert!(result.success() && start < result.time && result.time < plan.end_s());

    let snapshot = || {
        let mut solver = super::SOLVER.lock().unwrap();
        let state = solver.as_mut().unwrap();
        super::with_device(|device| {
            state.positions.download(device)?;
            state.positions_prev.download(device)?;
            let mut staged_fix = vec![crate::data::FixPair::default(); state.fix.len()];
            state.fix.read(device, 0, &mut staged_fix)?;
            Ok((state.positions.host().to_vec(), state.positions_prev.host().to_vec(),
                bincode::serialize(&staged_fix).unwrap()))
        }).unwrap()
    };

    // The same certified cut now leaves a real moving hard row shy of its
    // new sample. Preserve accepted histories and the full prepared pin set.
    let start = param.time;
    let prior_dt = param.prev_dt;
    let plan = unsafe { super::prepare_kinematics_plan(0, param.dt, None) }.unwrap();
    let pins = scene.affine_pin_prefix(start, plan.end_s(), false).unwrap();
    let full = scene.make_constraint_with_membership(plan.end_s(), start, true);
    unsafe { super::host_update_constraint(&full) }.unwrap();
    let before = snapshot();
    // If C2 could run on this trial, this pending native test verdict would
    // refuse it for a separate intersection. Endpoint failure must win first.
    super::step::PRIVATE_TEST_AFFINE_C2_REJECT.with(|armed| armed.set(true));
    super::step::PRIVATE_TEST_AFFINE_MESH_CUT.with(|armed| armed.set(Some(0.5)));
    let refused = unsafe { super::advance_host_kinematics(0, 16, false, Some(plan), Some(&pins)) }.unwrap();
    let super::HostAdvanceOutcome::KinematicsShortened { actual_h, continuation } = refused else {
        panic!("clipped moving hard row must trigger the original rollback");
    };
    assert!(actual_h > 0.0 && actual_h < param.dt);
    assert_ne!(continuation, 0);
    assert_eq!(param.time, start);
    assert_eq!(param.prev_dt, prior_dt);
    assert_eq!(snapshot(), before);
    assert!(super::step::PRIVATE_TEST_AFFINE_C2_REJECT.with(|armed| armed.get()));
    super::cancel_host_retry(continuation).unwrap();

    // Even the exact endpoint returns through A when the opt-in is off.
    let plan = unsafe { super::prepare_kinematics_plan(0, param.dt, None) }.unwrap();
    let pins = scene.affine_pin_prefix(start, plan.end_s(), false).unwrap();
    let full = scene.make_constraint_with_membership(plan.end_s(), start, true);
    unsafe { super::host_update_constraint(&full) }.unwrap();
    let before = snapshot();
    super::step::PRIVATE_TEST_AFFINE_CLIPPED_ENABLED.with(|value| value.set(Some(false)));
    super::step::PRIVATE_TEST_AFFINE_MESH_CUT.with(|armed| armed.set(Some(1.0 - 1.0e-10)));
    let fallback = unsafe { super::advance_host_kinematics(0, 16, false, Some(plan), Some(&pins)) }.unwrap();
    let super::HostAdvanceOutcome::KinematicsShortened { actual_h, continuation } = fallback else {
        panic!("clipped option is off, A must restore and resample");
    };
    assert!(actual_h > 0.0 && actual_h < param.dt);
    assert_eq!(param.time, start);
    assert_eq!(param.prev_dt, prior_dt);
    assert_eq!(snapshot(), before);
    assert!(super::step::PRIVATE_TEST_AFFINE_C2_REJECT.with(|armed| armed.get()));
    super::cancel_host_retry(continuation).unwrap();

    // The endpoint agrees to the exact f32 row. C2 now consumes the
    // independent intersection and refuses normally before any commit.
    let plan = unsafe { super::prepare_kinematics_plan(0, param.dt, None) }.unwrap();
    let pins = scene.affine_pin_prefix(start, plan.end_s(), false).unwrap();
    let full = scene.make_constraint_with_membership(plan.end_s(), start, true);
    unsafe { super::host_update_constraint(&full) }.unwrap();
    let before = snapshot();
    super::step::PRIVATE_TEST_AFFINE_CLIPPED_ENABLED.with(|value| value.set(Some(true)));
    super::step::PRIVATE_TEST_AFFINE_MESH_CUT.with(|armed| armed.set(Some(1.0 - 1.0e-10)));
    let refused = unsafe { super::advance_host_kinematics(0, 16, false, Some(plan), Some(&pins)) }.unwrap();
    let super::HostAdvanceOutcome::Refused(result) = refused else {
        panic!("exact pin endpoint must still refuse a separate C2 intersection");
    };
    assert!(!result.intersection_free && !result.success());
    assert_eq!(param.time, start);
    assert_eq!(param.prev_dt, prior_dt);
    assert_eq!(snapshot(), before);
    assert!(!super::step::PRIVATE_TEST_AFFINE_C2_REJECT.with(|armed| armed.get()));
    super::step::PRIVATE_TEST_AFFINE_CLIPPED_ENABLED.with(|value| value.set(None));
}
