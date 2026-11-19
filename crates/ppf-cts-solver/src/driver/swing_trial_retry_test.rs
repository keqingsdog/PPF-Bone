//! Private 04 capture replay through the actual CPU sampler and host transaction.
use clap::Parser;

#[test]
#[ignore = "requires the explicit private 04 authoring session"]
fn captured_swing_reject_rolls_back_and_shortens_real_host_step() {
    super::test_device_fault::start_private_status();
    let path=std::env::var("PPF_PRIVATE_SWING_SCENE").expect("private 04 session");
    let output=format!("{path}/output");
    let args=crate::args::ProgramArgs::try_parse_from(["test","--path",&path,"--output",&output]).unwrap();
    let require_restoring=args.require_restoring_curves_v1;
    let mut scene=crate::scene::Scene::new(&args);
    let owned=scene.joint_input.as_ref().expect("04 joints").records().clone();
    let selected:Vec<bool>=owned.joints.iter().map(|j|j.wire.enabled && j.wire.angular_control_v1.as_ref().is_some_and(|c|c.coordinate_model==crate::twist_overlay::CONTINUOUS_MODEL)).collect();
    assert!(selected.get(6)==Some(&true),"captured lane must be selected");
    let soft=crate::cross_links::prepare_batch(scene.joint_input.as_ref().expect("04 input"),scene.args().world_scaling as f64).unwrap();
    let data=crate::fresh_authority_dataset(&mut scene);
    let mut param=crate::builder::make_param(&scene.args());
    assert_eq!(param.playback,1.,"fixture must retain its authored unit playback");
    assert!(unsafe{super::initialize(&data,&param)});
    super::install_joints_from_rest(&owned,None);
    super::install_hard_joints(&owned.hard_records,owned.payload_version==5,owned.roots.len(),data.translation_lock.size as usize).unwrap();
    if let Some(soft)=soft.as_ref(){super::install_soft_joints_from_rest(soft,false).unwrap();}
    if let Some(prepared)=crate::restoring_curve::prepare_fresh(std::path::Path::new(&path),require_restoring,0).unwrap() {
        let curves=crate::restoring_native::verify_and_pack(&prepared,&scene,&data,std::path::Path::new(&path)).unwrap();
        super::install_restoring(&curves,data.constraint.stitch.size as usize).unwrap();
    }
    super::prepare_twist_fit_selection(selected.clone()).unwrap();
    let sample=super::sample_committed_twist(&selected).unwrap();
    let ids:Vec<_>=owned.joints.iter().map(|j|j.wire.joint_uuid.clone()).collect();
    let history=crate::continuous_twist::AcceptedTwist {scene_uuid:owned.scene_uuid.clone(),joint_uuids:ids.clone(),time:param.time,theta:sample.iter().map(|x|x[0]).collect(),prior_turns_known:true};
    super::import_accepted_twist(history,&owned.scene_uuid,&ids,param.time).unwrap();
    super::activate_twist_selection(selected.clone()).unwrap();
    {
        let lock=super::SOLVER.lock().unwrap();let state=lock.as_ref().unwrap();
        let (selected_count,soft_active,drive_count,drive_buffer_len)=state.joints.private_swing_scope_facts();
        eprintln!("PRIVATE_04_SCOPE playback={} selected={} hard={} roots={} root_events={} fem_runtime={} soft={} drive_count={} drive_buffer_len={} authored_roots={} authored_hard={} soft_batch={} rest_anim={} material_anim={} collision_windows={}",
            param.playback,selected_count,state.hard.active(),state.roots.transaction.is_some(),state.roots.has_control_events(),state.fem_roots.runtime.is_some(),soft_active,drive_count,drive_buffer_len,owned.roots.len(),owned.hard_records.len(),soft.as_ref().map_or(0,|v|v.records.len()),scene.rest_shape_schedule().is_some(),scene.has_material_animation(),scene.build_collision_window_table().has_windows);
        assert_eq!(selected_count,selected.len());assert!(!state.hard.active());assert!(state.roots.transaction.is_none());assert!(!state.roots.has_control_events());
        assert!(state.fem_roots.runtime.is_none());assert!(!soft_active);assert_eq!(drive_count,0);assert_eq!(drive_buffer_len,1);
    }
    assert!(super::host_swing_retry_scope(param.playback).unwrap());
    if std::env::var_os("PPF_PRIVATE_SWING_PREFLIGHT_ONLY").is_some() {
        eprintln!("PRIVATE_04_PREFLIGHT_ONLY: no advance, rollback or transaction validation");
        return;
    }
    super::host_seed_capture().unwrap();
    crate::backend::private_prepare_03(&scene,param.time,param.dt,param.playback,None).unwrap();
    let snapshot=||{
        let mut lock=super::SOLVER.lock().unwrap();let state=lock.as_mut().unwrap();
        super::with_device(|d|{
            state.positions.download(d)?;state.positions_prev.download(d)?;
            let pos=state.positions.host().iter().map(|v|v.to_bits()).collect::<Vec<_>>();
            let prev=state.positions_prev.host().iter().map(|v|v.to_bits()).collect::<Vec<_>>();
            let mut rigid=vec![crate::data::PdrdRigidState::default();state.rigid_state.len()];state.rigid_state.read(d,0,&mut rigid)?;
            let rigid=rigid.iter().map(|r|(r.anchor.map(f32::to_bits),r.x.map(f32::to_bits),r.rotation.map(f32::to_bits),r.reference_inertia.map(f32::to_bits),r.mass_total.to_bits(),r.count)).collect::<Vec<_>>();
            let mut running=vec![0f32;state.rigid_running_rotation.len()];state.rigid_running_rotation.read(d,0,&mut running)?;
            let mut omega=vec![0f32;state.grain_omega.len()];state.grain_omega.read(d,0,&mut omega)?;
            let curves=state.joints.curves.snapshot_trial(d)?;
            let mut fix=vec![crate::data::FixPair::default();state.fix.len()];state.fix.read(d,0,&mut fix)?;
            Ok((pos,prev,rigid,running.into_iter().map(f32::to_bits).collect::<Vec<_>>(),state.rigid_rotation_seeded,omega.into_iter().map(f32::to_bits).collect::<Vec<_>>(),curves,state.joints.accepted_twist().cloned(),state.restoring_commit_count,state.restoring_commit_time,bincode::serialize(&fix).unwrap()))
        }).unwrap()
    };
    let before=snapshot();let clock=(param.time.to_bits(),param.time_f32.to_bits(),param.prev_dt.to_bits(),param.dt.to_bits());
    super::roots::PRIVATE_TEST_BOUNDARY_COUNT.store(0,std::sync::atomic::Ordering::SeqCst);
    // Exact bit patterns captured from actual joint 6 / bodies 7 and 8 in 04.
    *super::joints::PRIVATE_SWING_SAMPLE_ROTATIONS.lock().unwrap()=Some([
        (7,[1044245145,3212231292,3192101747,1058855807,1049093258,3208548521,1061464699,1017915370,1059310622]),
        (8,[3191723346,1064747130,1044633747,1063594085,1034865752,1054868214,1053790124,1048653576,3210800656]),
    ]);
    let (token,failed)=match unsafe{super::advance_host_swing(0,16,true,None)}.unwrap(){
        super::HostAdvanceOutcome::SwingSampleRejected{refusal,continuation}=>{assert_eq!(refusal.token.sample.joint_index,6);(continuation,f32::from_bits(refusal.token.dt_bits))},
        _=>panic!("captured action4 must trigger its real swing refusal"),
    };
    assert_ne!(token,0);assert_eq!(clock,(param.time.to_bits(),param.time_f32.to_bits(),param.prev_dt.to_bits(),param.dt.to_bits()));
    let after=snapshot();
    assert_eq!(before.0,after.0);assert_eq!(before.1,after.1);assert_eq!(before.2,after.2);assert_eq!(before.3,after.3);assert_eq!(before.4,after.4);assert_eq!(before.5,after.5);
    assert_eq!(bincode::serialize(&before.6).unwrap(),bincode::serialize(&after.6).unwrap());
    assert_eq!(before.7,after.7);assert_eq!(before.8,after.8);assert_eq!(before.9,after.9);assert_eq!(before.10,after.10);
    let next=super::adaptive_retry::next_half_h(failed,param.time).unwrap();param.dt=next;super::host_seed_restore().unwrap();
    let plan=unsafe{super::prepare_host_plan(token,next)}.unwrap();
    let observed=crate::backend::private_prepare_03(&scene,param.time,next,param.playback,Some(plan)).unwrap();
    let clock_row=observed.iter().find(|x|x.0=="clock").expect("resampled clock");
    let actual:(f64,f64,f32)=bincode::deserialize(&clock_row.2).unwrap();
    assert_eq!(actual.1.to_bits(),plan.end_s().to_bits());assert_eq!(actual.2.to_bits(),plan.integration_h().to_bits());
    let constraint=scene.make_constraint(plan.end_s(),param.time);
    if !constraint.fix.as_slice().is_empty() {
        let pins=observed.iter().find(|x|x.0=="constraint_fix" && x.1).expect("authored pins must be uploaded");
        assert_eq!(pins.2,bincode::serialize(&constraint.fix.as_slice()).unwrap());
        let mut device_fix=vec![crate::data::FixPair::default();constraint.fix.as_slice().len()];
        {let lock=super::SOLVER.lock().unwrap();let state=lock.as_ref().unwrap();super::with_device(|d|{state.fix.read(d,0,&mut device_fix)?;Ok(())}).unwrap();}
        assert_eq!(pins.2,bincode::serialize(&device_fix.as_slice()).unwrap());
    }
    match unsafe{super::advance_host_swing(token,15,true,Some(plan))}.unwrap(){super::HostAdvanceOutcome::Accepted(r)=>assert!(r.success()),_=>panic!("shorter step did not accept")};
    super::host_seed_discard().unwrap();
    assert_eq!(snapshot().7.unwrap().time,param.time);
    assert_eq!(super::roots::PRIVATE_TEST_BOUNDARY_COUNT.load(std::sync::atomic::Ordering::SeqCst),1);
    // Separately exercise the old uncertified-path reason. The force flag only
    // replaces the certificate verdict; it does not synthesize the token.
    super::host_seed_capture().unwrap();
    crate::backend::private_prepare_03(&scene,param.time,param.dt,param.playback,None).unwrap();
    let before_path=snapshot();let start_path=param.time;
    super::joints::PRIVATE_FORCE_UNCERTIFIED.store(true,std::sync::atomic::Ordering::SeqCst);
    let (path_token,path_h)=match unsafe{super::advance_host_swing(0,16,true,None)}.unwrap(){
        super::HostAdvanceOutcome::ChartPathRejected{refusal,continuation}=>{
            assert!(matches!(refusal.token.reason,super::joints::CandidateRetryReason::UncertifiedPath("B31")));
            (continuation,f32::from_bits(refusal.token.dt_bits))
        },
        _=>panic!("real path token must leave the native loop for host resampling"),
    };
    assert_ne!(path_token,0);assert_eq!(param.time.to_bits(),start_path.to_bits());
    let after_path=snapshot();
    assert_eq!(before_path.0,after_path.0);assert_eq!(before_path.1,after_path.1);assert_eq!(before_path.2,after_path.2);
    assert_eq!(before_path.3,after_path.3);assert_eq!(before_path.4,after_path.4);assert_eq!(before_path.5,after_path.5);
    assert_eq!(bincode::serialize(&before_path.6).unwrap(),bincode::serialize(&after_path.6).unwrap());
    assert_eq!(before_path.7,after_path.7);assert_eq!(before_path.8,after_path.8);assert_eq!(before_path.9,after_path.9);assert_eq!(before_path.10,after_path.10);
    let path_next=super::adaptive_retry::next_half_h(path_h,param.time).unwrap();param.dt=path_next;super::host_seed_restore().unwrap();
    let path_plan=unsafe{super::prepare_host_plan(path_token,path_next)}.unwrap();
    let path_observed=crate::backend::private_prepare_03(&scene,param.time,path_next,param.playback,Some(path_plan)).unwrap();
    let path_clock:(f64,f64,f32)=bincode::deserialize(&path_observed.iter().find(|x|x.0=="clock").unwrap().2).unwrap();
    assert_eq!(path_clock.1.to_bits(),path_plan.end_s().to_bits());assert_eq!(path_clock.2.to_bits(),path_plan.integration_h().to_bits());
    match unsafe{super::advance_host_swing(path_token,15,true,Some(path_plan))}.unwrap(){super::HostAdvanceOutcome::Accepted(r)=>assert!(r.success()),_=>panic!("path retry must accept")};
    super::host_seed_discard().unwrap();
    assert_eq!(super::roots::PRIVATE_TEST_BOUNDARY_COUNT.load(std::sync::atomic::Ordering::SeqCst),2);
    let committed_time=param.time;
    super::host_seed_capture().unwrap();
    crate::backend::private_prepare_03(&scene,param.time,param.dt,param.playback,None).unwrap();
    super::step::set_private_test_commit_stage(2);
    let error=match unsafe{super::advance_host_swing(0,16,true,None)} {
        Err(error)=>error,
        Ok(_)=>panic!("post-C5 fault must remain fatal"),
    };
    assert!(error.detail.contains("PRIVATE_TEST_POST_C5"));
    assert!(param.time>committed_time,"post-C5 failure must not roll back the accepted clock");
    assert_eq!(snapshot().7.unwrap().time,param.time);
    assert_eq!(super::roots::PRIVATE_TEST_BOUNDARY_COUNT.load(std::sync::atomic::Ordering::SeqCst),3);
    super::host_seed_discard().unwrap();
    // A real action-4 NaN remains generic and cannot open a continuation.
    super::host_seed_capture().unwrap();
    crate::backend::private_prepare_03(&scene,param.time,param.dt,param.playback,None).unwrap();
    let before_nan=snapshot();let nan_time=param.time;
    *super::joints::PRIVATE_SWING_SAMPLE_ROTATIONS.lock().unwrap()=Some([
        (7,[f32::NAN.to_bits(),3212231292,3192101747,1058855807,1049093258,3208548521,1061464699,1017915370,1059310622]),
        (8,[3191723346,1064747130,1044633747,1063594085,1034865752,1054868214,1053790124,1048653576,3210800656]),
    ]);
    let nan=match unsafe{super::advance_host_swing(0,16,true,None)}{Err(error)=>error,Ok(_)=>panic!("NaN must not retry")};
    assert!(nan.detail.contains("chart fault 1"),"unexpected NaN error: {}",nan.detail);
    assert_eq!(param.time.to_bits(),nan_time.to_bits());
    assert!(super::SOLVER.lock().unwrap().as_ref().unwrap().host_retry_pending.is_none());
    let after_nan=snapshot();assert_eq!(before_nan.0,after_nan.0);assert_eq!(before_nan.1,after_nan.1);assert_eq!(before_nan.7,after_nan.7);
    super::host_seed_discard().unwrap();
    // Real action-4 is also used for reference qualification; corrupt only
    // its private fitted copy, not the accepted reference or physical state.
    super::host_seed_capture().unwrap();crate::backend::private_prepare_03(&scene,param.time,param.dt,param.playback,None).unwrap();
    let before_ref=snapshot();let ref_time=param.time;
    super::joints::PRIVATE_INVALID_REFERENCE.store(true,std::sync::atomic::Ordering::SeqCst);
    *super::joints::PRIVATE_SWING_SAMPLE_ROTATIONS.lock().unwrap()=Some([
        (7,[1044245145,3212231292,3192101747,1058855807,1049093258,3208548521,1061464699,1017915370,1059310622]),
        (8,[3191723346,1064747130,1044633747,1063594085,1034865752,1054868214,1053790124,1048653576,3210800656]),
    ]);
    let bad_ref=match unsafe{super::advance_host_swing(0,16,true,None)}{Err(error)=>error,Ok(_)=>panic!("invalid reference must not retry")};
    assert!(bad_ref.detail.contains("chart fault 6"),"must retain original current-swing cause: {}",bad_ref.detail);
    assert!(!bad_ref.detail.contains("swing trial qualification unavailable"),"reference gate failed before a real verdict: {}",bad_ref.detail);
    assert!(!super::joints::PRIVATE_INVALID_REFERENCE.load(std::sync::atomic::Ordering::SeqCst),"reference injection did not reach the actual qualification");
    assert!(super::SOLVER.lock().unwrap().as_ref().unwrap().host_retry_pending.is_none());
    let after_ref=snapshot();assert_eq!(ref_time.to_bits(),param.time.to_bits());
    assert_eq!(before_ref.0,after_ref.0);assert_eq!(before_ref.1,after_ref.1);assert_eq!(before_ref.2,after_ref.2);
    assert_eq!(before_ref.3,after_ref.3);assert_eq!(before_ref.4,after_ref.4);assert_eq!(before_ref.5,after_ref.5);
    assert_eq!(bincode::serialize(&before_ref.6).unwrap(),bincode::serialize(&after_ref.6).unwrap());
    assert_eq!(before_ref.7,after_ref.7);assert_eq!(before_ref.8,after_ref.8);assert_eq!(before_ref.9,after_ref.9);assert_eq!(before_ref.10,after_ref.10);
    super::host_seed_discard().unwrap();
}


#[test]
fn actual_action4_distinguishes_swing_from_invalid_input() {
    use ppf_cts_compute::{AllocLabel,Buffer,Device};
    use crate::{continuous_twist::AngularTwistEvalContext,data::PdrdRigidState,joint_angular_control_gpu::AngularControlGpu};
    use super::kernels::JointAngularControlApplyArgs;
    fn upload<T:ppf_cts_compute::Pod>(d:&mut impl Device,v:&[T],label:&'static str)->Buffer<T>{
        let mut b=Buffer::<T>::default();b.size(d,v.len(),AllocLabel(label)).unwrap();b.write(d,0,v).unwrap();b
    }
    let mut d=super::launch::host_device();
    let identity=[1.,0.,0.,0.,1.,0.,0.,0.,1.];
    let a=[1044245145u32,3212231292,3192101747,1058855807,1049093258,3208548521,1061464699,1017915370,1059310622].map(f32::from_bits);
    let b=[3191723346u32,1064747130,1044633747,1063594085,1034865752,1054868214,1053790124,1048653576,3210800656].map(f32::from_bits);
    let fa=[1060439283u32,3207922931,3013395810,3013395807].map(f32::from_bits);
    let fb=[1060439283u32,3207922931,3013659899,3013659899].map(f32::from_bits);
    let row=AngularControlGpu{body_a:0,body_b:1,enabled:1,coordinate_model:1,frame_a_wxyz:fa,frame_b_wxyz:fb,rest_relative_wxyz:[1.,0.,0.,0.],stiffness_nm_per_rad:[0.;3],damping_nms_per_rad:[0.;3]};
    let mut records=upload(&mut d,&[row],"swing.test.records");
    let context=upload(&mut d,&[AngularTwistEvalContext::default()],"swing.test.context");
    let material_context=upload(&mut d,&[crate::material_spin_history::MaterialSpinContext::default()],"swing.test.material_context");
    let mut state=upload(&mut d,&[PdrdRigidState::default();2],"swing.test.state");
    let reference=upload(&mut d,&[PdrdRigidState{rotation:identity,..Default::default()};2],"swing.test.reference");
    let input=upload(&mut d,&[1f32],"swing.test.selection");
    let mut output=upload(&mut d,&[0f32;10],"swing.test.output");
    let mut fault=upload(&mut d,&[0u32],"swing.test.fault");
    let mut run=|rotation_a:[f32;9],rotation_b:[f32;9],record:AngularControlGpu|{
        records.write(&mut d,0,&[record]).unwrap();
        state.write(&mut d,0,&[PdrdRigidState{rotation:rotation_a,..Default::default()},PdrdRigidState{rotation:rotation_b,..Default::default()}]).unwrap();
        output.write(&mut d,0,&[0.;10]).unwrap();fault.write(&mut d,0,&[0]).unwrap();
        let args=JointAngularControlApplyArgs{records:records.handle(),twist_context:context.handle(),material_spin_context:material_context.handle(),state:state.handle(),reference:reference.handle(),bodies:2,base:0,dim:12,dt:0.01,action:4,input:input.handle(),output:output.handle(),fault:fault.handle(),
            tangent_cache:input.handle(),tangent_offsets:fault.handle(),tangent_published:fault.handle(),epoch:0,count:1,seam_arena_count:0};
        unsafe{d.launch("swing.actual_action4",&args,1).unwrap();}
        let mut value=[0];fault.read(&mut d,0,&mut value).unwrap();value[0]
    };
    assert_eq!(run(a,b,row),6,"captured real swing must produce the dedicated fault");
    let mut nonfinite=a;nonfinite[0]=f32::NAN;
    assert_eq!(run(nonfinite,b,row),1,"nonfinite geometry must remain generic and fatal");
    let invalid=AngularControlGpu{frame_a_wxyz:[0.;4],..row};
    assert_eq!(run(a,b,invalid),1,"bad local quaternion must remain generic and fatal");
    let identity_frames=AngularControlGpu{frame_a_wxyz:[1.,0.,0.,0.],frame_b_wxyz:[1.,0.,0.,0.],..row};
    assert_eq!(run(identity,[-1.,0.,0.,0.,1.,0.,0.,0.,-1.],identity_frames),1,"zero twist radius must remain generic and fatal");
    assert_eq!(run(identity,identity,row),0,"valid reference must pass");
}


#[test]
#[ignore = "requires the explicit private continuous C4 scene"]
fn normal_host_matches_real_c_abi_and_commits_once() {
    let path=std::env::var("PPF_PRIVATE_C4_SCENE").expect("private continuous scene");
    let output=format!("{path}/output");
    let args=crate::args::ProgramArgs::try_parse_from(["test","--path",&path,"--output",&output]).unwrap();
    let mut scene=crate::scene::Scene::new(&args);
    let owned=scene.joint_input.as_ref().expect("fixture joints").records().clone();
    // Exercise both an explicit saved checkpoint (which can accept a partial
    // TOI) and the production fresh dataset (full-endpoint baseline).
    let checkpoint=std::env::var_os("PPF_PRIVATE_NORMAL_CHECKPOINT").is_some();
    let data=if checkpoint {
        let raw=crate::read_gz(&format!("{output}/dataset.bin.gz"));crate::read(&raw)
    } else {crate::fresh_authority_dataset(&mut scene)};
    let run=|host:bool|{
        let mut param=crate::builder::make_param(&scene.args());param.playback=1.;
        assert!(unsafe{super::initialize(&data,&param)});super::install_joints(&owned);
        let selected=vec![true;owned.joints.len()];super::prepare_twist_fit_selection(selected.clone()).unwrap();
        let sample=super::sample_committed_twist(&selected).unwrap();
        let ids:Vec<_>=owned.joints.iter().map(|j|j.wire.joint_uuid.clone()).collect();
        let history=crate::continuous_twist::AcceptedTwist{scene_uuid:owned.scene_uuid.clone(),joint_uuids:ids.clone(),time:param.time,theta:sample.iter().map(|v|v[0]).collect(),prior_turns_known:true};
        super::import_accepted_twist(history,&owned.scene_uuid,&ids,param.time).unwrap();super::activate_twist_selection(selected).unwrap();
        let before=super::export_accepted_twist().unwrap().unwrap();
        super::joints::PRIVATE_TWIST_COMMITS.store(0,std::sync::atomic::Ordering::SeqCst);
        let end=param.time+f64::from(param.dt);let constraint=scene.make_constraint(end,param.time);
        if host {super::host_seed_capture().unwrap();unsafe{super::host_update_constraint(&constraint)}.unwrap();}
        else {unsafe{super::update_constraint(&constraint);}}
        if host {match unsafe{super::advance_host_swing(0,16,false,None)}.unwrap(){super::HostAdvanceOutcome::Accepted(r)=>assert!(r.success()),_=>panic!("normal host route rejected")};super::host_seed_discard().unwrap();}
        else {let mut result=crate::data::StepResult::default();unsafe{super::advance(&mut result)};assert!(result.success());}
        let accepted=super::export_accepted_twist().unwrap().unwrap();
        assert_eq!(super::joints::PRIVATE_TWIST_COMMITS.load(std::sync::atomic::Ordering::SeqCst),1,"history must commit once");
        assert!(accepted.time>before.time);assert_eq!(accepted.time,param.time);
        assert!(param.time<=end,"accepted physical clock exceeded requested endpoint");
        if !checkpoint {assert_eq!(param.time.to_bits(),end.to_bits(),"fresh full-step fixture");}
        let positions={let mut lock=super::SOLVER.lock().unwrap();let state=lock.as_mut().unwrap();super::with_device(|d|{state.positions.download(d)?;Ok(state.positions.host().iter().map(|v|v.to_bits()).collect::<Vec<_>>())}).unwrap()};
        (param.time.to_bits(),positions,accepted)
    };
    let c=run(false);let host=run(true);
    assert_eq!(host.0,c.0);assert_eq!(host.1,c.1);assert_eq!(host.2,c.2);
}

