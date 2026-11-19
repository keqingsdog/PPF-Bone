//! Private actual 03 FEM B32 rejection. Never runs without an explicit fixture.
use clap::Parser;
use ppf_cts_compute::Device;

fn with_private_live_fixture(f: impl FnOnce(&crate::scene::Scene)) {
    let path=std::env::var("PPF_PRIVATE_FEM_SCENE").expect("private scene path");
    let output=format!("{path}/output");
    let args=crate::args::ProgramArgs::try_parse_from(["test","--path",&path,"--output",&output]).unwrap();
    let scene=crate::scene::Scene::new(&args);
    let data:crate::data::DataSet=crate::read(&crate::read_gz(&format!("{output}/dataset.bin.gz")));
    let param=crate::builder::make_param(&scene.args());
    assert!(unsafe{super::initialize(&data,&param)});
    f(&scene);
}

fn private_seed_len() -> usize {
    super::SOLVER.lock().unwrap().as_ref().unwrap().adaptive_seed_prev.len()
}

#[test]
#[ignore = "run alone in a fresh process with a private fixture"]
fn host_seed_capture_copy_fault_really_cleans_allocation() {
    use super::test_device_fault::{Armed,Point};
    with_private_live_fixture(|_| {
        let armed=Armed::new(&[Point::SeedCaptureCopy]);
        let err=super::host_seed_capture().unwrap_err();
        assert!(err.detail.contains("injected point 0"));assert_eq!(private_seed_len(),0);
        assert_eq!(Armed::hits(Point::SeedCaptureCopy),1);drop(armed);
    });
}

#[test]
#[ignore = "run alone in a fresh process with a private fixture"]
fn host_seed_capture_cleanup_fault_retains_owned_buffer() {
    use super::test_device_fault::{Armed,Point};
    with_private_live_fixture(|_| {
        let armed=Armed::new(&[Point::SeedCaptureCopy,Point::SeedCaptureCleanupFree]);
        let err=super::host_seed_capture().unwrap_err();
        assert!(err.detail.contains("injected point 0") && err.detail.contains("seed cleanup failed"));
        assert_eq!(Armed::hits(Point::SeedCaptureCleanupFree),1);assert!(private_seed_len()>0);
        drop(armed);super::host_seed_discard().unwrap();assert_eq!(private_seed_len(),0);
    });
}

#[test]
#[ignore = "run alone in a fresh process with a private fixture"]
fn host_seed_restore_copy_fault_retains_seed() {
    use super::test_device_fault::{Armed,Point};
    with_private_live_fixture(|_| {
        super::host_seed_capture().unwrap();let len=private_seed_len();assert!(len>0);
        let armed=Armed::new(&[Point::SeedRestoreCopy]);
        let err=super::host_seed_restore().unwrap_err();assert!(err.detail.contains("injected point 1"));
        assert_eq!(private_seed_len(),len);assert_eq!(Armed::hits(Point::SeedRestoreCopy),1);
        drop(armed);super::host_seed_discard().unwrap();
    });
}

#[test]
#[ignore = "run alone in a fresh process with a private fixture"]
fn host_constraint_fix_write_fault_returns_fatal() {
    use super::test_device_fault::{Armed,Point};
    with_private_live_fixture(|scene| {
        let constraint=scene.make_constraint(0.0,0.0);assert!(!constraint.fix.as_slice().is_empty());
        let armed=Armed::new(&[Point::ConstraintFixWrite]);
        let err=crate::backend::private_prepare_03(scene,0.0,0.005,1.0,None).unwrap_err();
        assert!(err.detail.contains("injected point 4"));assert_eq!(Armed::hits(Point::ConstraintFixWrite),1);
        assert!(!crate::backend::private_observed_constraint_upload());drop(armed);
    });
}

#[test]
#[ignore = "run alone in a fresh process with a private fixture containing pins"]
fn host_constraint_nonempty_fix_upload_matches_device() {
    with_private_live_fixture(|scene| {
        let expected=scene.make_constraint(0.005,0.0);
        assert!(!expected.fix.as_slice().is_empty());
        let observed=crate::backend::private_prepare_03(scene,0.0,0.005,1.0,None).unwrap();
        let uploaded=observed.iter().find(|row|row.0=="constraint_fix" && row.1).expect("fix upload");
        assert_eq!(uploaded.2,bincode::serialize(&expected.fix.as_slice()).unwrap());
        let mut fix=vec![crate::data::FixPair::default();expected.fix.as_slice().len()];
        {let lock=super::SOLVER.lock().unwrap();let state=lock.as_ref().unwrap();super::with_device(|d|{state.fix.read(d,0,&mut fix)?;Ok(())}).unwrap();}
        assert_eq!(uploaded.2,bincode::serialize(&fix.as_slice()).unwrap());
    });
}

#[test]
#[ignore = "run alone in a fresh process with a private fixture"]
fn host_seed_guard_failed_discard_does_not_retry_on_drop() {
    with_private_live_fixture(|_| {
        crate::backend::private_test_failed_guard_finish();
        assert!(private_seed_len()>0);
        super::host_seed_discard().unwrap();assert_eq!(private_seed_len(),0);
    });
}

#[test]
#[ignore = "requires 03 authoring fixture; run alone in a fresh process"]
fn actual_fem_retry_restore_fault_keeps_original_and_cancels() {
    super::test_device_fault::start_private_status();
    use super::test_device_fault::{Armed,Point};
    let path=std::env::var("PPF_PRIVATE_FEM_SCENE").expect("private scene path");
    let output=format!("{path}/output");
    let args=crate::args::ProgramArgs::try_parse_from(["test","--path",&path,"--output",&output]).unwrap();
    let mut scene=crate::scene::Scene::new(&args);
    let input=scene.joint_input.as_ref().expect("joint input");let owned=input.records().clone();
    let soft=crate::cross_links::prepare_batch(input,scene.args().world_scaling as f64).unwrap().expect("soft scene");
    let data=crate::fresh_authority_dataset(&mut scene);
    let mut param=crate::builder::make_param(&scene.args());param.playback=1.;
    assert!(scene.rest_shape_schedule().is_none() && !scene.has_material_animation());
    assert!(!scene.build_collision_window_table().has_windows);
    assert!(unsafe{super::initialize(&data,&param)});
    super::install_joints_from_rest(&owned,None);super::install_soft_joints(&soft).unwrap();
    let initial=param.dt;let start=param.time;
    let calls=std::cell::Cell::new(0u32);let advances=std::cell::Cell::new(0u32);let cancels=std::cell::Cell::new(0u32);
    let armed=Armed::new(&[Point::SeedRestoreCopy,Point::SeedDiscardFree]);
    let result=super::adaptive_retry::run_host_attempts(
        |h,_,token,remaining| {
            calls.set(calls.get()+1);param.dt=h;
            let plan=if token==0 {None} else {Some(unsafe{super::prepare_host_plan(token,h)}?)};
            if token==0 {super::host_seed_capture()?;} else {super::host_seed_restore()?;}
            crate::backend::private_prepare_03(&scene,param.time,h,param.playback,plan)?;
            if token==0 {super::step::set_private_test_fem_injections(1);}
            advances.set(advances.get()+1);
            unsafe{super::advance_host(token,remaining,true,true,plan)}
        },
        |token| {cancels.set(cancels.get()+1);super::cancel_host_retry(token)},
        super::host_seed_discard,initial,start,16,initial);
    let err=match result {Err(err)=>err,Ok(_)=>panic!("injected restore failure must be fatal")};
    assert!(err.detail.contains("injected point 1") && err.detail.contains("preceding FEM") && err.detail.contains("259"));
    assert!(err.detail.contains("host retry cleanup failed") && err.detail.contains("injected point 3"));
    assert_eq!(calls.get(),2);assert_eq!(advances.get(),1);assert_eq!(cancels.get(),1);
    assert_eq!(param.time.to_bits(),start.to_bits());
    assert_eq!(Armed::hits(Point::SeedRestoreCopy),1);assert_eq!(Armed::hits(Point::SeedDiscardFree),1);
    assert!(super::SOLVER.lock().unwrap().as_ref().unwrap().host_retry_pending.is_none());
    assert!(private_seed_len()>0);drop(armed);super::host_seed_discard().unwrap();
}

#[test]
#[ignore = "must run alone in a fresh process before any solver initialization"]
fn host_empty_prepare_requires_live_scene_and_state(){
    unsafe {
        assert!(super::host_override_velocity(&[],0.,0.,0.,0.01).is_err());
        assert!(super::host_override_angular_velocity(&[],0.,0.,0.,0.,0.,0.,0.01).is_err());
        assert!(super::host_gather_current_positions(&[],&mut []).is_err());
    }
    assert!(super::host_refresh_collision_active(0.).is_err());
}

#[test]
#[ignore = "poisons global solver lock; run alone in a fresh process with a private 03 fixture"]
fn host_empty_prepare_rejects_poisoned_solver_lock(){
    let path=std::env::var("PPF_PRIVATE_FEM_SCENE").expect("private scene path");let output=format!("{path}/output");
    let args=crate::args::ProgramArgs::try_parse_from(["test","--path",&path,"--output",&output]).unwrap();
    let scene=crate::scene::Scene::new(&args);let raw=crate::read_gz(&format!("{output}/dataset.bin.gz"));let data:crate::data::DataSet=crate::read(&raw);
    let param=crate::builder::make_param(&scene.args());assert!(unsafe{super::initialize(&data,&param)});
    let _=std::thread::spawn(||{let _guard=super::SOLVER.lock().unwrap();panic!("intentional private poison");}).join();
    unsafe {assert!(super::host_override_velocity(&[],0.,0.,0.,0.01).is_err());assert!(super::host_gather_current_positions(&[],&mut []).is_err());}
    assert!(super::host_refresh_collision_active(0.).is_err());
}

#[test]
#[ignore = "removes global solver state; run alone in a fresh process with a private 03 fixture"]
fn host_installed_collision_table_requires_solver_state() {
    let path = std::env::var("PPF_PRIVATE_FEM_SCENE").expect("private scene path");
    let output = format!("{path}/output");
    let args = crate::args::ProgramArgs::try_parse_from(["test", "--path", &path, "--output", &output]).unwrap();
    let scene = crate::scene::Scene::new(&args);
    let raw = crate::read_gz(&format!("{output}/dataset.bin.gz"));
    let data: crate::data::DataSet = crate::read(&raw);
    let param = crate::builder::make_param(&scene.args());
    assert!(unsafe { super::initialize(&data, &param) });
    let view = match super::require_scene_result() {
        Ok(view) => view,
        Err(_) => panic!("initialized private scene must be live"),
    };
    let groups = vec![0u32; unsafe { view.vertex_count() }];
    let windows = vec![0f32; ppf_cts_core::datamodel::object::MAX_COLLISION_WINDOWS * 2];
    super::with_device(|device| unsafe {
        super::COLLISION_WINDOWS.lock().unwrap().initialize(device, &view, &groups, &windows, &[0], 1)
    }).unwrap();
    assert!(super::COLLISION_WINDOWS.lock().unwrap().is_initialized());
    drop(view);
    *super::SOLVER.lock().unwrap() = None;
    assert!(super::host_refresh_collision_active(0.).is_err());
}

#[test]
#[ignore = "requires explicit private 03 native authoring scene"]
fn actual_legacy_fem_b32_rolls_back_and_host_retries_once(){
    super::test_device_fault::start_private_status();
    let path=std::env::var("PPF_PRIVATE_FEM_SCENE").expect("private scene path");
    let output=format!("{path}/output");
    let args=crate::args::ProgramArgs::try_parse_from(["test","--path",&path,"--output",&output]).unwrap();
    let mut scene=crate::scene::Scene::new(&args);
    let input=scene.joint_input.as_ref().expect("joint input");let owned=input.records().clone();
    let soft=crate::cross_links::prepare_batch(input,scene.args().world_scaling as f64).unwrap().expect("soft scene");
    let data=crate::fresh_authority_dataset(&mut scene);
    let mut param=crate::builder::make_param(&scene.args());param.playback=1.;
    assert!(unsafe{super::initialize(&data,&param)});
    super::install_joints_from_rest(&owned,None);super::install_soft_joints(&soft).unwrap();
    let ca=scene.make_constraint(2.17648646576950933+0.005,2.17648646576950933);
    let cb=scene.make_constraint(2.17648646576950933+0.0025,2.17648646576950933);
    eprintln!("PRIVATE_03_CONSTRAINT_SCOPE fix={} pull={} sphere={} floor={} static_vertices={} fix_changed={} sphere_changed={} floor_changed={} mesh_changed={}",
        ca.fix.size,ca.pull.size,ca.sphere.size,ca.floor.size,ca.mesh.vertex.size,
        bincode::serialize(&ca.fix.as_slice()).unwrap()!=bincode::serialize(&cb.fix.as_slice()).unwrap(),
        bincode::serialize(&ca.sphere.as_slice()).unwrap()!=bincode::serialize(&cb.sphere.as_slice()).unwrap(),
        bincode::serialize(&ca.floor.as_slice()).unwrap()!=bincode::serialize(&cb.floor.as_slice()).unwrap(),
        bincode::serialize(&ca.mesh.vertex.as_slice()).unwrap()!=bincode::serialize(&cb.mesh.vertex.as_slice()).unwrap());
    {
        let lock=super::SOLVER.lock().unwrap();let state=lock.as_ref().unwrap();
        assert!(state.joints.soft.active() && state.joints.twist_selection().is_none());
        assert!(!state.hard.active() && !state.roots.rotating() && !state.roots.has_control_events() && state.fem_roots.runtime.is_none());
    }
    assert!(scene.rest_shape_schedule().is_none() && !scene.has_material_animation());
    assert!(!scene.build_collision_window_table().has_windows);
    super::host_seed_capture().unwrap();
    let first_observed=crate::backend::private_prepare_03(&scene,param.time,param.dt,param.playback,None).unwrap();
    assert!(first_observed.iter().any(|x| x.0=="constraint_fix" && x.1));
    let snapshot=||{
        let mut lock=super::SOLVER.lock().unwrap();let state=lock.as_mut().unwrap();
        super::with_device(|d|{
            state.positions.download(d)?;state.positions_prev.download(d)?;
            let pos=state.positions.host().iter().map(|v|v.to_bits()).collect::<Vec<_>>();
            let prev=state.positions_prev.host().iter().map(|v|v.to_bits()).collect::<Vec<_>>();
            let mut fixes=vec![crate::data::FixPair::default();state.fix.len()];state.fix.read(d,0,&mut fixes)?;
            let fix=bincode::serialize(&fixes).unwrap();
            let mut rigid=vec![crate::data::PdrdRigidState::default();state.rigid_state.len()];state.rigid_state.read(d,0,&mut rigid)?;
            let rigid=rigid.iter().map(|r|(r.anchor.map(f32::to_bits),r.x.map(f32::to_bits),r.rotation.map(f32::to_bits),r.reference_inertia.map(f32::to_bits),r.mass_total.to_bits(),r.count)).collect::<Vec<_>>();
            let mut running=vec![0f32;state.rigid_running_rotation.len()];state.rigid_running_rotation.read(d,0,&mut running)?;
            let mut omega=vec![0f32;state.grain_omega.len()];state.grain_omega.read(d,0,&mut omega)?;
            let root=state.roots.transaction.as_ref().map(|r|bincode::serialize(&r.export_checkpoint().unwrap()).unwrap());
            let curves=state.joints.curves.snapshot_trial(d)?;
            Ok((pos,prev,fix,rigid,running.into_iter().map(f32::to_bits).collect::<Vec<_>>(),state.rigid_rotation_seeded,omega.into_iter().map(f32::to_bits).collect::<Vec<_>>(),root,curves,state.fem_roots.accepted_steps,state.fem_roots.prepared_end,state.restoring_commit_count,state.restoring_commit_time))
        }).unwrap()
    };
    let before=snapshot();let clock=(param.time.to_bits(),param.time_f32.to_bits(),param.prev_dt.to_bits(),param.dt.to_bits());
    super::roots::PRIVATE_TEST_BOUNDARY_COUNT.store(0,std::sync::atomic::Ordering::SeqCst);
    super::step::set_private_test_fem_injections(1);
    let (token,failed)=match unsafe{super::advance_host(0,16,true,true,None)}.unwrap(){
        super::HostAdvanceOutcome::FemMaterialRejected{refusal,continuation}=>{assert_eq!(refusal.token.row,1);assert_eq!(refusal.token.side,1);assert_eq!(refusal.token.fault,259);(continuation,f32::from_bits(refusal.token.dt_bits))},
        _=>panic!("expected actual FEM rejection"),
    };
    assert_ne!(token,0);assert_eq!(clock,(param.time.to_bits(),param.time_f32.to_bits(),param.prev_dt.to_bits(),param.dt.to_bits()));
    let after=snapshot();
    {let mut lock=super::SOLVER.lock().unwrap();let state=lock.as_mut().unwrap();super::with_device(|d|{state.eval_x.download(d)?;assert_eq!(state.eval_x.host().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),after.0);Ok(())}).unwrap();}
    assert_eq!(before.0,after.0);assert_eq!(before.1,after.1);assert_eq!(before.2,after.2);assert_eq!(before.3,after.3);assert_eq!(before.4,after.4);assert_eq!(before.5,after.5);assert_eq!(before.6,after.6);assert_eq!(before.7,after.7);
    assert_eq!(before.9,after.9);assert_eq!(before.10,after.10);assert_eq!(before.11,after.11);assert_eq!(before.12,after.12);
    assert_eq!(bincode::serialize(&before.8).unwrap(),bincode::serialize(&after.8).unwrap());
    let next=super::adaptive_retry::next_half_h(failed,param.time).unwrap();param.dt=next;super::host_seed_restore().unwrap();
    let plan=unsafe{super::prepare_host_plan(token,next)}.unwrap();
    let retried_observed=crate::backend::private_prepare_03(&scene,param.time,next,param.playback,Some(plan)).unwrap();
    let observed_clock=retried_observed.iter().find(|x|x.0=="clock").expect("actual helper clock");
    let clock:(f64,f64,f32)=bincode::deserialize(&observed_clock.2).unwrap();
    assert_eq!(clock.0.to_bits(),param.time.to_bits());assert_eq!(clock.1.to_bits(),plan.end_s().to_bits());assert_eq!(clock.2.to_bits(),plan.integration_h().to_bits());
    let observed_constraint=retried_observed.iter().find(|x|x.0=="constraint_fix" && x.1).expect("actual uploaded constraint");
    let expected=scene.make_constraint(plan.end_s(),param.time);
    assert!(!expected.fix.as_slice().is_empty(),"03 retry must exercise the authored plate pins");
    assert_eq!(observed_constraint.2,bincode::serialize(&expected.fix.as_slice()).unwrap());
    let mut device_fix=vec![crate::data::FixPair::default();expected.fix.as_slice().len()];
    {let lock=super::SOLVER.lock().unwrap();let state=lock.as_ref().unwrap();super::with_device(|d|{state.fix.read(d,0,&mut device_fix)?;Ok(())}).unwrap();}
    assert_eq!(observed_constraint.2,bincode::serialize(&device_fix.as_slice()).unwrap());
    match unsafe{super::advance_host(token,15,true,true,Some(plan))}.unwrap(){super::HostAdvanceOutcome::Accepted(r)=>assert!(r.success()),_=>panic!("shorter real attempt must accept")};
    super::host_seed_discard().unwrap();assert_eq!(super::roots::PRIVATE_TEST_BOUNDARY_COUNT.load(std::sync::atomic::Ordering::SeqCst),1);
    // A post-C5 failure must not become another material retry or rewind the
    // accepted clock. This is an additional commit gate, not the B32 proof.
    let committed_time=param.time;super::host_seed_capture().unwrap();
    let constraint=scene.make_constraint(param.time+f64::from(param.dt),param.time);unsafe{super::host_update_constraint(&constraint)}.unwrap();
    super::step::set_private_test_commit_stage(2);
    let error=match unsafe{super::advance_host(0,16,true,true,None)}{Err(e)=>e,Ok(_)=>panic!("post-C5 injection must fail")};
    assert!(error.detail.contains("PRIVATE_TEST_POST_C5"));assert!(param.time>committed_time);
    super::host_seed_discard().unwrap();assert_eq!(super::roots::PRIVATE_TEST_BOUNDARY_COUNT.load(std::sync::atomic::Ordering::SeqCst),2);
}

#[test]
#[ignore = "requires explicit private 03 native authoring scene"]
fn actual_row2_ambiguous_frame_b32_rolls_back_and_retries_once(){
    super::test_device_fault::start_private_status();
    let path=std::env::var("PPF_PRIVATE_FEM_SCENE").expect("private scene path");
    let output=format!("{path}/output");
    let args=crate::args::ProgramArgs::try_parse_from(["test","--path",&path,"--output",&output]).unwrap();
    let mut scene=crate::scene::Scene::new(&args);
    let input=scene.joint_input.as_ref().expect("joint input");let owned=input.records().clone();
    let soft=crate::cross_links::prepare_batch(input,scene.args().world_scaling as f64).unwrap().expect("soft scene");
    let data=crate::fresh_authority_dataset(&mut scene);
    let mut param=crate::builder::make_param(&scene.args());param.playback=1.;
    assert!(unsafe{super::initialize(&data,&param)});
    super::install_joints_from_rest(&owned,None);super::install_soft_joints(&soft).unwrap();
    let ca=scene.make_constraint(2.17648646576950933+0.005,2.17648646576950933);
    let cb=scene.make_constraint(2.17648646576950933+0.0025,2.17648646576950933);
    eprintln!("PRIVATE_03_CONSTRAINT_SCOPE fix={} pull={} sphere={} floor={} static_vertices={} fix_changed={} sphere_changed={} floor_changed={} mesh_changed={}",
        ca.fix.size,ca.pull.size,ca.sphere.size,ca.floor.size,ca.mesh.vertex.size,
        bincode::serialize(&ca.fix.as_slice()).unwrap()!=bincode::serialize(&cb.fix.as_slice()).unwrap(),
        bincode::serialize(&ca.sphere.as_slice()).unwrap()!=bincode::serialize(&cb.sphere.as_slice()).unwrap(),
        bincode::serialize(&ca.floor.as_slice()).unwrap()!=bincode::serialize(&cb.floor.as_slice()).unwrap(),
        bincode::serialize(&ca.mesh.vertex.as_slice()).unwrap()!=bincode::serialize(&cb.mesh.vertex.as_slice()).unwrap());
    {
        let lock=super::SOLVER.lock().unwrap();let state=lock.as_ref().unwrap();
        assert!(state.joints.soft.active() && state.joints.twist_selection().is_none());
        assert!(!state.hard.active() && !state.roots.rotating() && !state.roots.has_control_events() && state.fem_roots.runtime.is_none());
    }
    assert!(scene.rest_shape_schedule().is_none() && !scene.has_material_animation());
    assert!(!scene.build_collision_window_table().has_windows);
    super::host_seed_capture().unwrap();
    let first_observed=crate::backend::private_prepare_03(&scene,param.time,param.dt,param.playback,None).unwrap();
    assert!(first_observed.iter().any(|x| x.0=="constraint_fix" && x.1));
    let snapshot=||{
        let mut lock=super::SOLVER.lock().unwrap();let state=lock.as_mut().unwrap();
        super::with_device(|d|{
            state.positions.download(d)?;state.positions_prev.download(d)?;
            let pos=state.positions.host().iter().map(|v|v.to_bits()).collect::<Vec<_>>();
            let prev=state.positions_prev.host().iter().map(|v|v.to_bits()).collect::<Vec<_>>();
            let mut fixes=vec![crate::data::FixPair::default();state.fix.len()];state.fix.read(d,0,&mut fixes)?;
            let fix=bincode::serialize(&fixes).unwrap();
            let mut rigid=vec![crate::data::PdrdRigidState::default();state.rigid_state.len()];state.rigid_state.read(d,0,&mut rigid)?;
            let rigid=rigid.iter().map(|r|(r.anchor.map(f32::to_bits),r.x.map(f32::to_bits),r.rotation.map(f32::to_bits),r.reference_inertia.map(f32::to_bits),r.mass_total.to_bits(),r.count)).collect::<Vec<_>>();
            let mut running=vec![0f32;state.rigid_running_rotation.len()];state.rigid_running_rotation.read(d,0,&mut running)?;
            let mut omega=vec![0f32;state.grain_omega.len()];state.grain_omega.read(d,0,&mut omega)?;
            let root=state.roots.transaction.as_ref().map(|r|bincode::serialize(&r.export_checkpoint().unwrap()).unwrap());
            let curves=state.joints.curves.snapshot_trial(d)?;
            Ok((pos,prev,fix,rigid,running.into_iter().map(f32::to_bits).collect::<Vec<_>>(),state.rigid_rotation_seeded,omega.into_iter().map(f32::to_bits).collect::<Vec<_>>(),root,curves,state.fem_roots.accepted_steps,state.fem_roots.prepared_end,state.restoring_commit_count,state.restoring_commit_time))
        }).unwrap()
    };
    let before=snapshot();let clock=(param.time.to_bits(),param.time_f32.to_bits(),param.prev_dt.to_bits(),param.dt.to_bits());
    super::roots::PRIVATE_TEST_BOUNDARY_COUNT.store(0,std::sync::atomic::Ordering::SeqCst);
    super::step::set_private_test_fem_row2_injections(1);
    let (token,failed)=match unsafe{super::advance_host(0,16,true,true,None)}.unwrap(){
        super::HostAdvanceOutcome::FemMaterialRejected{refusal,continuation}=>{assert_eq!(refusal.token.row,2);assert_eq!(refusal.token.side,1);assert_eq!(refusal.token.fault,259);assert_eq!(refusal.token.reason,super::soft_joints::MaterialFrameRejectReason::StrictF32Negative);(continuation,f32::from_bits(refusal.token.dt_bits))},
        _=>panic!("expected actual FEM rejection"),
    };
    assert_ne!(token,0);assert_eq!(clock,(param.time.to_bits(),param.time_f32.to_bits(),param.prev_dt.to_bits(),param.dt.to_bits()));
    let after=snapshot();
    {let mut lock=super::SOLVER.lock().unwrap();let state=lock.as_mut().unwrap();super::with_device(|d|{state.eval_x.download(d)?;assert_eq!(state.eval_x.host().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),after.0);Ok(())}).unwrap();}
    assert_eq!(before.0,after.0);assert_eq!(before.1,after.1);assert_eq!(before.2,after.2);assert_eq!(before.3,after.3);assert_eq!(before.4,after.4);assert_eq!(before.5,after.5);assert_eq!(before.6,after.6);assert_eq!(before.7,after.7);
    assert_eq!(before.9,after.9);assert_eq!(before.10,after.10);assert_eq!(before.11,after.11);assert_eq!(before.12,after.12);
    assert_eq!(bincode::serialize(&before.8).unwrap(),bincode::serialize(&after.8).unwrap());
    let next=super::adaptive_retry::next_half_h(failed,param.time).unwrap();param.dt=next;super::host_seed_restore().unwrap();
    let plan=unsafe{super::prepare_host_plan(token,next)}.unwrap();
    let retried_observed=crate::backend::private_prepare_03(&scene,param.time,next,param.playback,Some(plan)).unwrap();
    let observed_clock=retried_observed.iter().find(|x|x.0=="clock").expect("actual helper clock");
    let clock:(f64,f64,f32)=bincode::deserialize(&observed_clock.2).unwrap();
    assert_eq!(clock.0.to_bits(),param.time.to_bits());assert_eq!(clock.1.to_bits(),plan.end_s().to_bits());assert_eq!(clock.2.to_bits(),plan.integration_h().to_bits());
    let observed_constraint=retried_observed.iter().find(|x|x.0=="constraint_fix" && x.1).expect("actual uploaded constraint");
    let expected=scene.make_constraint(plan.end_s(),param.time);
    assert!(!expected.fix.as_slice().is_empty(),"03 retry must exercise the authored plate pins");
    assert_eq!(observed_constraint.2,bincode::serialize(&expected.fix.as_slice()).unwrap());
    let mut device_fix=vec![crate::data::FixPair::default();expected.fix.as_slice().len()];
    {let lock=super::SOLVER.lock().unwrap();let state=lock.as_ref().unwrap();super::with_device(|d|{state.fix.read(d,0,&mut device_fix)?;Ok(())}).unwrap();}
    assert_eq!(observed_constraint.2,bincode::serialize(&device_fix.as_slice()).unwrap());
    match unsafe{super::advance_host(token,15,true,true,Some(plan))}.unwrap(){super::HostAdvanceOutcome::Accepted(r)=>assert!(r.success()),_=>panic!("shorter real attempt must accept")};
    // The next real begin_step must rebuild the scratch reference from the
    // restored accepted positions, not retain the captured row2 test scratch.
    let actual_reference={let lock=super::SOLVER.lock().unwrap();let state=lock.as_ref().unwrap();super::with_device(|d|Ok(state.joints.soft.test_row2_reference_bits(d)?)).unwrap()};
    let expected_reference=[2365usize,2364,2469,2472].map(|v|[before.0[3*v],before.0[3*v+1],before.0[3*v+2]]);
    assert_eq!(actual_reference,expected_reference,"retry reference must come from restored accepted positions");
    super::host_seed_discard().unwrap();assert_eq!(super::roots::PRIVATE_TEST_BOUNDARY_COUNT.load(std::sync::atomic::Ordering::SeqCst),1);
    // A post-C5 failure must not become another material retry or rewind the
    // accepted clock. This is an additional commit gate, not the B32 proof.
    let committed_time=param.time;super::host_seed_capture().unwrap();
    let constraint=scene.make_constraint(param.time+f64::from(param.dt),param.time);unsafe{super::host_update_constraint(&constraint)}.unwrap();
    super::step::set_private_test_commit_stage(2);
    let error=match unsafe{super::advance_host(0,16,true,true,None)}{Err(e)=>e,Ok(_)=>panic!("post-C5 injection must fail")};
    assert!(error.detail.contains("PRIVATE_TEST_POST_C5"));assert!(param.time>committed_time);
    super::host_seed_discard().unwrap();assert_eq!(super::roots::PRIVATE_TEST_BOUNDARY_COUNT.load(std::sync::atomic::Ordering::SeqCst),2);
}

