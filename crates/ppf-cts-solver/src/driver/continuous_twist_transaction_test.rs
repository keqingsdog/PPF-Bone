//! Explicit private-fixture test of real physical commit boundaries. This
//! module and its injected failure hooks are absent from production builds.
use clap::Parser;

#[test]
#[ignore = "requires an explicitly selected private built scene and dataset"]
fn actual_advance_commit_boundary() {
    let path=std::env::var("PPF_PRIVATE_C4_SCENE").expect("private scene path");
    let output=format!("{path}/output");
    let args=crate::args::ProgramArgs::try_parse_from(["test","--path",&path,"--output",&output]).unwrap();
    let scene=crate::scene::Scene::new(&args);
    let owned=scene.joint_input.as_ref().expect("fixture joint input").records().clone();
    let raw=crate::read_gz(&format!("{output}/dataset.bin.gz"));
    let data:crate::data::DataSet=crate::read(&raw);
    let mut param=crate::builder::make_param(&scene.args());
    let with_curves=std::env::var_os("PPF_PRIVATE_C4_WITH_CURVES").is_some();
    let curves=if with_curves {
        let prepared=crate::restoring_curve::prepare_fresh(std::path::Path::new(&path),true,0).unwrap().unwrap();
        Some(crate::restoring_native::verify_and_pack(&prepared,&scene,&data,std::path::Path::new(&path)).unwrap())
    } else {None};
    assert!(unsafe{super::initialize(&data,&param)});
    super::install_joints(&owned);
    if let Some(curves)=curves.as_ref(){super::install_restoring(curves,data.constraint.stitch.size as usize).unwrap();}
    let selected=vec![true;owned.joints.len()];
    super::prepare_twist_fit_selection(selected.clone()).unwrap();
    let sample=super::sample_committed_twist(&selected).unwrap();
    let ids:Vec<_>=owned.joints.iter().map(|j|j.wire.joint_uuid.clone()).collect();
    let history=crate::continuous_twist::AcceptedTwist{scene_uuid:owned.scene_uuid.clone(),joint_uuids:ids.clone(),time:param.time,theta:sample.iter().map(|x|x[0]).collect(),prior_turns_known:true};
    super::import_accepted_twist(history,&owned.scene_uuid,&ids,param.time).unwrap();
    super::activate_twist_selection(selected).unwrap();
    let constraint=scene.make_constraint(param.time+f64::from(param.dt),param.time);
    unsafe{super::update_constraint(&constraint);}
    // Establish a real accepted boundary and curve capture before testing the
    // next step's failure, matching production's multi-step lifecycle.
    if std::env::var_os("PPF_PRIVATE_C4_COLD").is_none() {
        let warmup={let mut lock=super::SOLVER.lock().unwrap();let state=lock.as_mut().unwrap();
            super::with_device(|device|unsafe{super::step::advance(device,&data,&mut param,state)})}.unwrap();
        assert!(warmup.success(),"fixture warmup did not accept a step");
        let constraint=scene.make_constraint(param.time+f64::from(param.dt),param.time);
        unsafe{super::update_constraint(&constraint);}
    }
    let snapshot=|| {
        let mut lock=super::SOLVER.lock().unwrap();let state=lock.as_mut().unwrap();
        super::with_device(|device| {
            state.positions.download(device)?;let positions=state.positions.host().to_vec();
            state.positions_prev.download(device)?;let previous=state.positions_prev.host().to_vec();
            state.eval_x.download(device)?;let iterate=state.eval_x.host().to_vec();
            let mut running=vec![0.;state.rigid_running_rotation.len()];
            if !running.is_empty(){state.rigid_running_rotation.read(device,0,&mut running)?;}
            let root=state.roots.transaction.as_ref().map(|r|r.export_checkpoint().map_err(|e|super::scene::Fatal::invariant(e.to_string()))).transpose()?;
            let fem=state.fem_roots.runtime.as_ref().map(|r|r.export_checkpoint().map_err(super::scene::Fatal::invariant)).transpose()?;
            let accepted=super::checkpoint::Snapshot{running_rotation:running,rotation_seeded:state.rigid_rotation_seeded,root,fem,fem_steps:state.fem_roots.accepted_steps,
                restoring:state.joints.curves.snapshot_trial(device)?,restoring_commit_count:state.restoring_commit_count,restoring_commit_time:state.restoring_commit_time};
            Ok((positions,previous,accepted,state.joints.accepted_twist().cloned(),iterate,state.joints.test_twist_trial_active()))
        }).unwrap()
    };
    let before=snapshot();
    if with_curves {assert!(before.2.restoring.is_some(),"fixture must have real curve history");}
    let clock=(param.time,param.time_f32,param.prev_dt,param.dt);
    let stage:u8=std::env::var("PPF_PRIVATE_C4_STAGE").unwrap_or_else(|_|"1".into()).parse().unwrap();
    super::step::set_private_test_commit_stage(stage);
    let result={let mut lock=super::SOLVER.lock().unwrap();let state=lock.as_mut().unwrap();
        super::with_device(|device|unsafe{super::step::advance(device,&data,&mut param,state)})};
    let error=match result {Err(error)=>error,Ok(_)=>panic!("injection must reach actual commit boundary")};
    assert!(error.detail.contains(if stage==1{"PRIVATE_TEST_C4"}else{"PRIVATE_TEST_POST_C5"}),"unexpected error: {}",error.detail);
    let after=snapshot();
    if stage==1 {
        assert_eq!((param.time,param.time_f32,param.prev_dt,param.dt),clock,"pre-C5 clock leaked");
        assert_eq!(before.0,after.0,"accepted positions changed");assert_eq!(before.1,after.1,"previous positions changed");
        assert!(before.2.semantic_eq(&after.2),"accepted root/restoring/running state changed");
        assert_eq!(before.3,after.3,"accepted twist changed");
        assert!(!after.5,"failed trial remained active");
    } else {
        assert!(param.time>clock.0,"post-C5 clock rolled back");
        assert!(after.3.as_ref().unwrap().time>before.3.as_ref().unwrap().time,"post-C5 history rolled back");
        assert_eq!(after.3.as_ref().unwrap().time,param.time,"accepted history must match the committed clock");
        assert_eq!(after.1,before.0,"C5 previous positions must be the old accepted positions");
        assert_eq!(after.0,after.4,"C5 current positions must be the accepted iterate");
        if with_curves {assert_eq!(after.2.restoring_commit_count,before.2.restoring_commit_count+1);}
    }
}
